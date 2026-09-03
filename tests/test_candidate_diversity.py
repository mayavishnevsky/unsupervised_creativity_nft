import unittest
from types import SimpleNamespace

import torch

from config.nft import sd3_iem_same_prompt_partiprompts
from flow_grpo.candidate_diversity import (
    CADSConditioning,
    CADSConfig,
    ContextualRepulsionConfig,
    activate_contextual_repulsion,
    cads_gamma,
    candidate_diversity_controls,
    candidate_diversity_seed,
    corrupt_condition,
    resolve_candidate_diversity,
    validate_candidate_diversity_sampling,
)


def diversity_config(method="none", **overrides):
    cads = {
        "tau1": 0.6,
        "tau2": 0.9,
        "noise_scale": 0.25,
        "psi": 0.5,
        "rescale": True,
    }
    contextual = {
        "target_tensor": "text",
        "t_stop": 4,
        "num_steps": 100,
        "eta": 6.0e5,
        "kernel_reg": 1.0e-4,
    }
    cads.update(overrides.get("cads", {}))
    contextual.update(overrides.get("contextual_repulsion", {}))
    return SimpleNamespace(
        candidate_diversity=SimpleNamespace(
            method=method,
            cads=SimpleNamespace(**cads),
            contextual_repulsion=SimpleNamespace(**contextual),
        ),
        sample=SimpleNamespace(
            guidance_scale=overrides.get("guidance_scale", 1.0),
            train_batch_size=overrides.get("train_batch_size", 24),
            num_image_per_prompt=overrides.get("num_image_per_prompt", 24),
        ),
    )


class IdentityProcessor:
    def __call__(self, hidden_states):
        return hidden_states


class FakeTransformer:
    def __init__(self):
        self.original = {
            "first": IdentityProcessor(),
            "second": IdentityProcessor(),
        }
        self.attn_processors = dict(self.original)

    def set_attn_processor(self, processors):
        self.attn_processors = dict(processors)


class CandidateDiversityTest(unittest.TestCase):
    def test_base_config_preserves_disabled_default(self):
        config = sd3_iem_same_prompt_partiprompts()
        self.assertEqual(config.candidate_diversity.method, "none")
        self.assertEqual(resolve_candidate_diversity(config), ("none", None))

    def test_resolves_both_methods(self):
        method, cads = resolve_candidate_diversity(diversity_config("cads"))
        self.assertEqual(method, "cads")
        self.assertEqual(cads, CADSConfig())

        method, contextual = resolve_candidate_diversity(
            diversity_config("contextual_repulsion")
        )
        self.assertEqual(method, "contextual_repulsion")
        self.assertEqual(contextual, ContextualRepulsionConfig())

    def test_rejects_invalid_method_parameters(self):
        with self.assertRaisesRegex(ValueError, "tau1"):
            resolve_candidate_diversity(
                diversity_config("cads", cads={"tau1": 0.95})
            )
        with self.assertRaisesRegex(ValueError, "target_tensor"):
            resolve_candidate_diversity(
                diversity_config(
                    "contextual_repulsion",
                    contextual_repulsion={"target_tensor": "unknown"},
                )
            )

    def test_contextual_requires_complete_candidate_group(self):
        with self.assertRaisesRegex(ValueError, "complete same-prompt"):
            validate_candidate_diversity_sampling(
                diversity_config(
                    "contextual_repulsion",
                    train_batch_size=6,
                    num_image_per_prompt=24,
                )
            )

        method, _ = validate_candidate_diversity_sampling(
            diversity_config("contextual_repulsion")
        )
        self.assertEqual(method, "contextual_repulsion")

    def test_enabled_methods_reject_cfg(self):
        for method in ("cads", "contextual_repulsion"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "guidance_scale=1"):
                    validate_candidate_diversity_sampling(
                        diversity_config(method, guidance_scale=4.5)
                    )

    def test_cads_schedule_endpoints(self):
        self.assertEqual(cads_gamma(1.0, 0.6, 0.9), 0.0)
        self.assertEqual(cads_gamma(0.0, 0.6, 0.9), 1.0)
        self.assertAlmostEqual(cads_gamma(0.75, 0.6, 0.9), 0.5)

    def test_cads_uses_dedicated_reproducible_rng(self):
        clean = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        config = CADSConfig()
        first = torch.Generator().manual_seed(17)
        second = torch.Generator().manual_seed(17)
        before = torch.random.get_rng_state()
        corrupted_first = corrupt_condition(clean, 0.0, config, first)
        after = torch.random.get_rng_state()
        corrupted_second = corrupt_condition(clean, 0.0, config, second)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(corrupted_first, corrupted_second))
        self.assertFalse(torch.equal(clean, corrupted_first))

    def test_cads_callback_returns_step_specific_clean_shapes(self):
        prompt = torch.randn(4, 5, 6)
        pooled = torch.randn(4, 7)
        callback = CADSConditioning(
            prompt,
            pooled,
            CADSConfig(),
            torch.Generator().manual_seed(9),
        )
        noisy_prompt, noisy_pooled = callback(1.0)
        clean_prompt, clean_pooled = callback(0.0)
        self.assertEqual(noisy_prompt.shape, prompt.shape)
        self.assertEqual(noisy_pooled.shape, pooled.shape)
        self.assertFalse(torch.equal(noisy_prompt, prompt))
        self.assertTrue(torch.equal(clean_prompt, prompt))
        self.assertTrue(torch.equal(clean_pooled, pooled))

    def test_contextual_processors_are_restored_after_scope(self):
        pipe = SimpleNamespace(transformer=FakeTransformer())
        original = dict(pipe.transformer.attn_processors)
        with activate_contextual_repulsion(
            pipe,
            ContextualRepulsionConfig(num_steps=1, eta=1.0),
        ):
            self.assertNotEqual(pipe.transformer.attn_processors, original)
        self.assertEqual(pipe.transformer.attn_processors, original)
        for name in original:
            self.assertIs(pipe.transformer.attn_processors[name], original[name])

    def test_contextual_processors_are_restored_after_error(self):
        pipe = SimpleNamespace(transformer=FakeTransformer())
        original = dict(pipe.transformer.attn_processors)
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            with activate_contextual_repulsion(
                pipe,
                ContextualRepulsionConfig(num_steps=1, eta=1.0),
            ):
                raise RuntimeError("sampling failed")
        self.assertEqual(pipe.transformer.attn_processors, original)

    def test_contextual_controls_reject_mixed_prompt_batch(self):
        pipe = SimpleNamespace(transformer=FakeTransformer())
        with self.assertRaisesRegex(ValueError, "same-prompt"):
            with candidate_diversity_controls(
                pipe,
                "contextual_repulsion",
                ContextualRepulsionConfig(),
                torch.zeros(2, 1, 1),
                torch.zeros(2, 1),
                ["first", "second"],
                None,
            ):
                pass

    def test_seed_depends_on_epoch_rank_and_batch(self):
        seed = candidate_diversity_seed(0, 1, 2, 3)
        self.assertEqual(seed, candidate_diversity_seed(0, 1, 2, 3))
        self.assertNotEqual(seed, candidate_diversity_seed(0, 2, 2, 3))
        self.assertNotEqual(seed, candidate_diversity_seed(0, 1, 3, 3))
        self.assertNotEqual(seed, candidate_diversity_seed(0, 1, 2, 4))


if __name__ == "__main__":
    unittest.main()
