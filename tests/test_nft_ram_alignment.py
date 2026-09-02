import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from flow_grpo.creativity import BalancedPromptSampler, IEMReward
from flow_grpo.reference_cache import (
    CACHE_SUCCESS_FILENAME,
    LATENT_KEY,
    write_cache_entry,
    write_cache_spec,
)


class MockAccelerator:
    device = torch.device("cpu")
    num_processes = 1

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def reduce(tensor, reduction="sum"):
        del reduction
        return tensor


class MockModel:
    @staticmethod
    def disable_adapter():
        return nullcontext()


def creativity_config(prompt_file: Path, cache_dir: Path, spec_sha256: str):
    return SimpleNamespace(
        distance_metric="iem",
        reference_prompt_mode="same_prompt",
        reference_selection_mode="all",
        nearest_reference_fraction=0.1,
        reference_samples_per_prompt=2,
        reference_batch_size=1,
        feature_batch_size=1,
        level_batch_size=1,
        noise_table_count=1,
        sigma_min=0.009,
        sigma_max=1000.0,
        num_steps=2,
        seed=0,
        resolution=512,
        num_inference_steps=20,
        guidance_scale=1.0,
        candidate_prompt_files=[str(prompt_file)],
        clip_model_id="openai/clip-vit-large-patch14",
        dino_model_id="facebook/dinov2-base",
        reference_cache_dir=str(cache_dir),
        reference_cache_spec_sha256=spec_sha256,
        reference_cache_use_iem_statistics=True,
    )


class NftRamReferenceAlignmentTests(unittest.TestCase):
    def test_reward_reads_exact_ram_latents_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_file = root / "train.txt"
            prompt_file.write_text("a black cat\na white dog\n")
            prompt = "a black cat"
            spec = {
                "seed": 0,
                "resolution": 512,
                "num_inference_steps": 20,
                "guidance_scale": 1.0,
                "reference_samples_per_prompt": 2,
                "prompt_source_sha256": BalancedPromptSampler(
                    [prompt_file]
                ).source_sha256,
                "clip": {"model_id": "openai/clip-vit-large-patch14"},
                "dino": {"model_id": "facebook/dinov2-base"},
                "iem": {
                    "sigma_min": 0.009,
                    "sigma_max": 1000.0,
                    "num_steps": 2,
                    "noise_table_count": 1,
                    "world_size": 1,
                    "feature_weighting": "sqrt_delta_gamma_v1",
                    "noise_assignment": "rank_permutation_v1",
                },
            }
            spec_sha256 = write_cache_spec(root, spec)
            (root / CACHE_SUCCESS_FILENAME).write_text(
                json.dumps({"spec_sha256": spec_sha256}) + "\n"
            )
            latents = torch.tensor(
                [[[[1.0, 2.0]]], [[[3.0, 4.0]]]],
                dtype=torch.bfloat16,
            )
            write_cache_entry(
                root,
                spec_sha256=spec_sha256,
                epoch=3,
                prompt=prompt,
                prompt_id=0,
                latents=latents,
                clip_embeddings=torch.randn(2, 3),
                dino_embeddings=torch.randn(2, 4),
                seeds=[11, 12],
                iem_mean=torch.zeros(4),
                iem_variance=0.0,
                iem_noise_seed=123,
                iem_noise_sha256="ab" * 32,
            )

            reward = IEMReward(
                pipe=None,
                model=None,
                accelerator=MockAccelerator(),
                config=creativity_config(prompt_file, root, spec_sha256),
            )
            cached = reward._cached_references(
                epoch=3,
                prompt=prompt,
                tensor_keys=(LATENT_KEY,),
            )
            torch.testing.assert_close(cached[LATENT_KEY], latents)
            self.assertIsNone(
                reward._cached_references(
                    epoch=20,
                    prompt=prompt,
                    tensor_keys=(LATENT_KEY,),
                )
            )

            incompatible = creativity_config(prompt_file, root, spec_sha256)
            incompatible.num_inference_steps = 25
            with self.assertRaisesRegex(ValueError, "incompatible"):
                IEMReward(
                    pipe=None,
                    model=None,
                    accelerator=MockAccelerator(),
                    config=incompatible,
                )

            latent_only = creativity_config(prompt_file, root, spec_sha256)
            latent_only.sigma_max = 500.0
            latent_only.reference_cache_use_iem_statistics = False
            latent_only_reward = IEMReward(
                pipe=None,
                model=None,
                accelerator=MockAccelerator(),
                config=latent_only,
            )
            self.assertIsNone(
                latent_only_reward._cached_iem_statistics(
                    epoch=3,
                    prompt=prompt,
                    noise_seed=123,
                    noise_sha256="ab" * 32,
                    feature_dimension=4,
                )
            )



    def test_cached_image_reward_never_calls_nft_reference_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_file = root / "train.txt"
            prompt = "a black cat"
            prompt_file.write_text(prompt + "\n")
            spec = {
                "seed": 0,
                "resolution": 512,
                "num_inference_steps": 20,
                "guidance_scale": 1.0,
                "reference_samples_per_prompt": 2,
                "prompt_source_sha256": BalancedPromptSampler(
                    [prompt_file]
                ).source_sha256,
                "clip": {"model_id": "openai/clip-vit-large-patch14"},
                "dino": {"model_id": "facebook/dinov2-base"},
            }
            spec_sha256 = write_cache_spec(root, spec)
            (root / CACHE_SUCCESS_FILENAME).write_text("ok\n")
            reference_features = torch.tensor([[0.0, 0.0], [0.0, 4.0]])
            write_cache_entry(
                root,
                spec_sha256=spec_sha256,
                epoch=0,
                prompt=prompt,
                prompt_id=0,
                iem_mean=torch.zeros(2),
                iem_variance=0.0,
                iem_noise_seed=123,
                iem_noise_sha256="ab" * 32,
                latents=torch.zeros(2, 1, 1, 1),
                clip_embeddings=reference_features,
                dino_embeddings=torch.zeros(2, 3),
                seeds=[11, 12],
            )
            config = creativity_config(prompt_file, root, spec_sha256)
            config.distance_metric = "clip_l2"
            sampler = mock.Mock(side_effect=AssertionError("unexpected sampling"))
            reward = IEMReward(
                pipe=None,
                model=MockModel(),
                accelerator=MockAccelerator(),
                config=config,
                reference_latent_generator=sampler,
            )
            candidate_features = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
            reward._image_embeddings_from_latents = mock.Mock(
                return_value=candidate_features
            )

            scores, metrics = reward.score(
                torch.zeros(2, 1, 1, 1),
                torch.zeros(2, 1),
                torch.zeros(2, 1),
                [prompt, prompt],
                epoch=0,
                group_size=2,
            )

            expected = torch.cdist(
                candidate_features, reference_features
            ).mean(1)
            torch.testing.assert_close(scores, expected)
            sampler.assert_not_called()
            self.assertEqual(
                metrics["clip_l2_same_prompt_references_generated"], 0.0
            )
            self.assertEqual(
                metrics["clip_l2_same_prompt_references_loaded"], 2.0
            )


if __name__ == "__main__":
    unittest.main()
