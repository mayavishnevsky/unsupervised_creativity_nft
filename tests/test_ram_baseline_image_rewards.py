from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from flow_grpo.image_reward_runtime import StandardImageRewardSet
from flow_grpo.ram_baseline import (
    merge_ram_checkpoint_adapter,
    resolve_ram_checkpoint,
)


class _Config:
    scaling_factor = 2.0
    shift_factor = 0.25


class _Vae:
    dtype = torch.float32
    config = _Config()

    def decode(self, latents, return_dict=False):
        self.last_latents = latents
        return (latents[:, :3],)


class _ImageProcessor:
    @staticmethod
    def postprocess(images, output_type):
        assert output_type == "pt"
        return images


class _Pipe:
    def __init__(self):
        self.vae = _Vae()
        self.image_processor = _ImageProcessor()


class RamBaselineTests(unittest.TestCase):
    def test_resolve_requires_success_marker_and_weights(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            with self.assertRaises(FileNotFoundError):
                resolve_ram_checkpoint(checkpoint)
            (checkpoint / "_SUCCESS").write_text("complete\n", encoding="ascii")
            (checkpoint / "ram_adapters.safetensors").write_bytes(b"weights")
            self.assertEqual(resolve_ram_checkpoint(checkpoint), checkpoint.resolve())

    def test_merge_rejects_unknown_adapter_before_loading_model(self):
        with self.assertRaisesRegex(ValueError, "RAM adapter"):
            merge_ram_checkpoint_adapter(None, "/missing", adapter_name="unknown")

    def test_merge_rejects_invalid_scale_before_loading_checkpoint(self):
        for scale in (-0.1, float("inf"), float("nan")):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "merge_scale"):
                    merge_ram_checkpoint_adapter(
                        None,
                        "/missing",
                        merge_scale=scale,
                    )


class StandardImageRewardTests(unittest.TestCase):
    def test_decodes_and_scores_in_bounded_batches(self):
        batch_sizes = []

        def make_scorer(device, weights):
            self.assertEqual(device, torch.device("cpu"))
            self.assertEqual(dict(weights), {"pickscore": 1.0, "clipscore": 1.0})

            def score(images, prompts, metadata, only_strict=True):
                self.assertTrue(only_strict)
                self.assertEqual(len(prompts), len(metadata))
                batch_sizes.append(len(images))
                values = torch.arange(len(images), dtype=torch.float32)
                return {
                    "pickscore": values,
                    "clipscore": values + 10,
                    "avg": 2 * values + 10,
                }, {}

            return score

        with patch("flow_grpo.image_reward_runtime.multi_score", make_scorer):
            rewards = StandardImageRewardSet(
                _Pipe(),
                torch.device("cpu"),
                {"pickscore": 1.0, "clipscore": 1.0},
                decode_batch_size=2,
            )
        latents = torch.ones(5, 4, 2, 2)
        components, metrics = rewards.score(
            latents,
            None,
            None,
            [f"prompt {index}" for index in range(5)],
            epoch=0,
            group_size=1,
        )
        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertEqual(tuple(components["avg"].shape), (5,))
        self.assertIn("rewards/pickscore", metrics)
        self.assertEqual(rewards.state_dict(), {})
        rewards.load_state_dict({})


if __name__ == "__main__":
    unittest.main()
