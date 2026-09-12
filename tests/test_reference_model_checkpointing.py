import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from flow_grpo.nft_creativity_runtime import (
    ReferenceModelState,
    load_training_checkpoint,
    save_training_checkpoint,
)


class FakeCreativityRewards:
    def __init__(self):
        self.state = {"iem": {"reference_prompt_mode": "same_prompt"}}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state


class FakeConfig:
    def to_dict(self):
        return {"name": "rolling-reference-checkpoint-test"}


class ReferenceModelCheckpointTests(unittest.TestCase):
    @staticmethod
    def make_model():
        base = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
        lora_config = LoraConfig(r=2, lora_alpha=2, target_modules=["0"])
        model = get_peft_model(base, lora_config)
        model.add_adapter("reference", lora_config)
        model.add_adapter("old", lora_config)
        model.set_adapter("default")
        return model

    def test_reference_adapter_and_epoch_round_trip(self):
        model = self.make_model()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        creativity = FakeCreativityRewards()

        expected_values = {
            "default": 1.0,
            "old": 2.0,
            "reference": 3.0,
        }
        for adapter_name, expected in expected_values.items():
            model.set_adapter(adapter_name)
            for value in get_peft_model_state_dict(
                model,
                adapter_name=adapter_name,
            ).values():
                value.fill_(expected)
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
                next_epoch=5,
                global_step=11,
                rank=0,
                world_size=1,
                reference_model_state=ReferenceModelState(epoch=5),
                reference_adapter_name="reference",
            )

            for adapter_name in expected_values:
                model.set_adapter(adapter_name)
                for value in get_peft_model_state_dict(
                    model,
                    adapter_name=adapter_name,
                ).values():
                    value.zero_()
            restored_state = ReferenceModelState()
            next_epoch, global_step, loaded_path = load_training_checkpoint(
                Path(tmpdir) / "checkpoints",
                model,
                optimizer,
                None,
                None,
                creativity,
                device="cpu",

                rank=0,
                reference_model_state=restored_state,
                reference_adapter_name="reference",
            )

            self.assertEqual((next_epoch, global_step), (5, 11))
            self.assertEqual(loaded_path, checkpoint)
            self.assertEqual(restored_state.epoch, 5)
            for adapter_name, expected in expected_values.items():
                for value in get_peft_model_state_dict(
                    model,
                    adapter_name=adapter_name,
                ).values():
                    torch.testing.assert_close(
                        value,
                        torch.full_like(value, expected),
                    )


if __name__ == "__main__":
    unittest.main()
