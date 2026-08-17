import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from config.nft import (
    sd3_iem_same_prompt_partiprompts,
    sd3_iem_same_prompt_partiprompts_ram_aligned,
    sd3_iem_same_prompt_partiprompts_smoke,
)
from flow_grpo.creativity import BalancedPromptSampler
from flow_grpo.nft_creativity_runtime import (
    DistributedPromptGroupBatchSampler,
    fixed_validation_seed,
    load_training_checkpoint,
    save_training_checkpoint,
)


class SizedDataset:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count


class FakeCreativityRewards:
    def __init__(self):
        self.state = {"iem": {"reference_prompt_mode": "same_prompt"}}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state


class FakeConfig:
    def to_dict(self):
        return {"name": "checkpoint-test"}


class PromptGroupSamplerTests(unittest.TestCase):
    def test_groups_are_complete_local_and_disjoint(self):
        dataset = SizedDataset(20)
        samplers = [
            DistributedPromptGroupBatchSampler(
                dataset,
                batch_size=2,
                group_size=6,
                num_groups=8,
                num_replicas=2,
                rank=rank,
                seed=17,
            )
            for rank in range(2)
        ]
        rank_batches = [list(sampler) for sampler in samplers]
        self.assertEqual([len(batches) for batches in rank_batches], [12, 12])
        for batches in rank_batches:
            self.assertTrue(all(len(set(batch)) == 1 for batch in batches))
            counts = {}
            for batch in batches:
                counts[batch[0]] = counts.get(batch[0], 0) + len(batch)
            self.assertEqual(set(counts.values()), {6})
        rank_groups = [{batch[0] for batch in batches} for batches in rank_batches]
        self.assertTrue(rank_groups[0].isdisjoint(rank_groups[1]))
        self.assertEqual(len(rank_groups[0] | rank_groups[1]), 8)

    def test_epoch_changes_prompt_draw_deterministically(self):
        sampler = DistributedPromptGroupBatchSampler(
            SizedDataset(20), 2, 4, 6, 2, 0, seed=9
        )
        epoch_zero = list(sampler)
        sampler.set_epoch(1)
        epoch_one = list(sampler)
        self.assertNotEqual(epoch_zero, epoch_one)
        sampler.set_epoch(0)
        self.assertEqual(epoch_zero, list(sampler))



    def test_ram_aligned_draw_matches_balanced_prompt_sampler(self):
        prompts = [f"prompt-{index}" for index in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text("\n".join(prompts) + "\n")
            expected = [
                record.prompt_id
                for record in BalancedPromptSampler([path]).sample(8, seed=17)
            ]

        rank_groups = []
        for rank in range(2):
            sampler = DistributedPromptGroupBatchSampler(
                SizedDataset(len(prompts)),
                batch_size=2,
                group_size=2,
                num_groups=8,
                num_replicas=2,
                rank=rank,
                seed=17,
                ram_aligned=True,
            )
            rank_groups.append([batch[0] for batch in sampler])
        self.assertEqual(rank_groups[0], expected[0::2])
        self.assertEqual(rank_groups[1], expected[1::2])


class ConfigAndSeedTests(unittest.TestCase):
    def test_full_config_matches_requested_run(self):
        config = sd3_iem_same_prompt_partiprompts()
        self.assertEqual(config.sample.num_steps, 25)
        self.assertEqual(config.beta, 0.1)
        self.assertEqual(config.train.beta, 0.01)
        self.assertEqual(config.save_freq, 1)
        self.assertEqual(config.eval_freq, 1)
        self.assertEqual(config.creativity.reference_prompt_mode, "same_prompt")
        self.assertEqual(config.creativity.reference_samples_per_prompt, 256)
        self.assertEqual(config.creativity.num_steps, 64)
        self.assertEqual(config.creativity.sigma_min, 0.009)
        self.assertEqual(config.creativity.sigma_max, 1000.0)
        sampler = DistributedPromptGroupBatchSampler(
            SizedDataset(226),
            config.sample.train_batch_size,
            config.sample.num_image_per_prompt,
            config.sample.num_prompt_groups,
            2,
            0,
        )
        self.assertEqual(len(sampler), config.sample.num_batches_per_epoch)

    def test_smoke_config_has_two_rank_geometry(self):
        config = sd3_iem_same_prompt_partiprompts_smoke()
        sampler = DistributedPromptGroupBatchSampler(
            SizedDataset(226),
            config.sample.train_batch_size,
            config.sample.num_image_per_prompt,
            config.sample.num_prompt_groups,
            2,
            0,
        )
        self.assertEqual(len(sampler), 1)

    def test_ram_aligned_config_uses_shared_reference_identity(self):
        config = sd3_iem_same_prompt_partiprompts_ram_aligned()
        self.assertTrue(config.sample.ram_aligned_prompt_sampling)
        self.assertEqual(config.validation.prompt_seed, 3_000_009)
        self.assertEqual(config.creativity.num_inference_steps, 20)
        self.assertEqual(
            config.creativity.reference_cache_spec_sha256,
            "d2ac2f59a948f67532f6f9dfa100992ae08fe6683bbc59768f2cccc38f566570",
        )

    def test_validation_seed_is_stable_and_prompt_specific(self):
        self.assertEqual(
            fixed_validation_seed(0, "a black cat"),
            fixed_validation_seed(0, "a black cat"),
        )
        self.assertNotEqual(
            fixed_validation_seed(0, "a black cat"),
            fixed_validation_seed(0, "a white cat"),
        )


class CheckpointTests(unittest.TestCase):
    def test_two_adapters_round_trip(self):
        base = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
        lora_config = LoraConfig(r=2, lora_alpha=2, target_modules=["0"])
        model = get_peft_model(base, lora_config)
        model.add_adapter("old", lora_config)
        model.set_adapter("default")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        creativity = FakeCreativityRewards()

        model.set_adapter("default")
        for value in get_peft_model_state_dict(model, adapter_name="default").values():
            value.fill_(1.0)
        model.set_adapter("old")
        for value in get_peft_model_state_dict(model, adapter_name="old").values():
            value.fill_(2.0)
        model.set_adapter("default")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = save_training_checkpoint(
                tmpdir,
                model,
                optimizer,
                None,
                None,
                creativity,
                FakeConfig(),
                next_epoch=3,
                global_step=7,
                rank=0,
                world_size=1,
            )
            self.assertTrue((checkpoint / "_SUCCESS").is_file())
            for name in ("default", "old"):
                for value in get_peft_model_state_dict(model, adapter_name=name).values():
                    value.zero_()
            next_epoch, global_step, loaded_path = load_training_checkpoint(
                Path(tmpdir) / "checkpoints",
                model,
                optimizer,
                None,
                None,
                creativity,
                device="cpu",
                rank=0,
            )
            self.assertEqual((next_epoch, global_step), (3, 7))
            self.assertEqual(loaded_path, checkpoint)
            for value in get_peft_model_state_dict(model, adapter_name="default").values():
                torch.testing.assert_close(value, torch.ones_like(value))
            for value in get_peft_model_state_dict(model, adapter_name="old").values():
                torch.testing.assert_close(value, torch.full_like(value, 2.0))

    def test_save_repairs_owner_write_permission(self):
        base = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
        lora_config = LoraConfig(r=2, lora_alpha=2, target_modules=["0"])
        model = get_peft_model(base, lora_config)
        model.add_adapter("old", lora_config)
        model.set_adapter("default")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        creativity = FakeCreativityRewards()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_training_checkpoint(
                tmpdir,
                model,
                optimizer,
                None,
                None,
                creativity,
                FakeConfig(),
                next_epoch=1,
                global_step=1,
                rank=0,
                world_size=1,
            )
            save_root = Path(tmpdir)
            checkpoint_root = save_root / "checkpoints"
            save_root.chmod(0o2570)
            checkpoint_root.chmod(0o2570)

            checkpoint = save_training_checkpoint(
                tmpdir,
                model,
                optimizer,
                None,
                None,
                creativity,
                FakeConfig(),
                next_epoch=2,
                global_step=2,
                rank=0,
                world_size=1,
            )

            self.assertTrue((checkpoint / "_SUCCESS").is_file())
            self.assertTrue(save_root.stat().st_mode & 0o200)
            self.assertTrue(checkpoint_root.stat().st_mode & 0o200)


if __name__ == "__main__":
    unittest.main()
