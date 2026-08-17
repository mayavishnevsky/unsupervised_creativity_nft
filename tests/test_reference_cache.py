import json
import tempfile
import unittest
from pathlib import Path

import torch

from flow_grpo.reference_cache import (
    CACHE_SUCCESS_FILENAME,
    CLIP_KEY,
    DINO_KEY,
    LATENT_KEY,
    ReferenceCacheReader,
    cache_spec_sha256,
    entry_path,
    validate_cache_entry,
    write_cache_entry,
    write_cache_spec,
)


class ReferenceCacheTests(unittest.TestCase):
    def test_entry_round_trip_and_strict_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            spec = {
                "seed": 0,
                "resolution": 512,
                "reference_samples_per_prompt": 3,
            }
            spec_sha256 = write_cache_spec(cache_dir, spec)
            self.assertEqual(spec_sha256, cache_spec_sha256(spec))
            (cache_dir / CACHE_SUCCESS_FILENAME).write_text(
                json.dumps({"spec_sha256": spec_sha256}) + "\n"
            )

            latents = torch.arange(3 * 2 * 2).reshape(3, 1, 2, 2)
            clip = torch.randn(3, 5, generator=torch.Generator().manual_seed(1))
            dino = torch.randn(3, 7, generator=torch.Generator().manual_seed(2))
            write_cache_entry(
                cache_dir,
                spec_sha256=spec_sha256,
                epoch=4,
                prompt="a black cat",
                prompt_id=9,
                latents=latents,
                clip_embeddings=clip,
                dino_embeddings=dino,
                seeds=[11, 12, 13],
                iem_mean=torch.zeros(8),
                iem_variance=0.0,
                iem_noise_seed=123,
                iem_noise_sha256="ab" * 32,
            )

            self.assertTrue(
                validate_cache_entry(
                    cache_dir,
                    spec_sha256=spec_sha256,
                    epoch=4,
                    prompt="a black cat",
                    reference_count=3,
                )
            )
            reader = ReferenceCacheReader(
                cache_dir,
                expected_spec_sha256=spec_sha256,
            )
            self.assertTrue(reader.has_entry(epoch=4, prompt="a black cat"))
            self.assertFalse(reader.has_entry(epoch=5, prompt="a black cat"))
            loaded = reader.load(
                epoch=4,
                prompt="a black cat",
                tensor_keys=(LATENT_KEY, CLIP_KEY, DINO_KEY),
            )
            self.assertEqual(loaded[LATENT_KEY].dtype, torch.bfloat16)
            self.assertEqual(loaded[CLIP_KEY].dtype, torch.float32)
            self.assertEqual(loaded[DINO_KEY].dtype, torch.float32)
            torch.testing.assert_close(loaded[CLIP_KEY], clip)
            torch.testing.assert_close(loaded[DINO_KEY], dino)

            self.assertFalse(
                validate_cache_entry(
                    cache_dir,
                    spec_sha256=spec_sha256,
                    epoch=5,
                    prompt="a black cat",
                    reference_count=3,
                )
            )
            with self.assertRaises(ValueError):
                ReferenceCacheReader(
                    cache_dir,
                    expected_spec_sha256="0" * 64,
                )

    def test_write_spec_rejects_reusing_directory_for_new_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            write_cache_spec(directory, {"seed": 0})
            with self.assertRaises(ValueError):
                write_cache_spec(directory, {"seed": 1})

    def test_entry_filename_does_not_include_prompt_text(self):
        path = entry_path("cache", 2, "a prompt/with unsafe characters")
        self.assertEqual(path.parent.name, "epoch_002")
        self.assertEqual(len(path.stem), 64)


if __name__ == "__main__":
    unittest.main()
