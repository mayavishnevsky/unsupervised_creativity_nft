import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from flow_grpo.creativity import (
    BalancedPromptSampler,
    IEMReward,
    ReferenceReservoir,
    cosine_distance_from_reference_mean,
    iem_features,
    iem_reward_from_reference_statistics,
    mean_nearest_cosine_distance,
    mean_nearest_l2_distance,
    mean_pairwise_l2_distance,
    iem_schedule_terms,
    iem_sigma_schedule,
    sample_iem_noise_table,
    same_prompt_reference_seed,
    update_reference_feature_sum,
    update_reference_sums,
)


class IEMFormulaTests(unittest.TestCase):
    def test_sigma_schedule_descends_and_gamma_increments_are_positive(self):
        schedule = iem_sigma_schedule(1.0, 1000.0, 64)
        probe_sigmas, delta_gamma = iem_schedule_terms(schedule)

        self.assertEqual(schedule.shape, (65,))
        self.assertEqual(probe_sigmas.shape, (64,))
        self.assertTrue(torch.all(schedule[:-1] > schedule[1:]))
        self.assertTrue(torch.all(delta_gamma > 0))
        self.assertTrue(torch.isclose(schedule[0], torch.tensor(1000.0), rtol=1e-5))
        self.assertTrue(torch.isclose(schedule[-1], torch.tensor(1.0), rtol=1e-5))

    def test_equations_14_and_16_include_interval_normalization(self):
        x_0 = torch.tensor([[[1.0, -0.5]], [[-0.25, 2.0]]])
        schedule = torch.tensor([2.0, 1.0, 0.5])
        noise_table = torch.tensor(
            [
                [[[0.2, -0.3]]],
                [[[-0.4, 0.7]]],
            ]
        )

        def zero_velocity(x_t, flow_time):
            del flow_time
            return torch.zeros_like(x_t)

        features = iem_features(x_0, zero_velocity, schedule, noise_table, level_batch_size=2)
        probe_sigmas, delta_gamma = iem_schedule_terms(schedule)
        sigma = probe_sigmas.reshape(-1, 1, 1, 1)
        x_t = (x_0.unsqueeze(0) + sigma * noise_table) / (1.0 + sigma)
        e_gamma = x_0.unsqueeze(0) - x_t
        expected = (
            (delta_gamma / len(probe_sigmas)).sqrt().reshape(-1, 1, 1, 1) * e_gamma
        ).transpose(0, 1).flatten(start_dim=1)

        torch.testing.assert_close(features, expected)
        explicit_distance = (
            delta_gamma
            / len(probe_sigmas)
            * (e_gamma[:, 0] - e_gamma[:, 1]).square().flatten(start_dim=1).sum(dim=1)
        ).sum()
        feature_distance = (features[0] - features[1]).square().sum()
        torch.testing.assert_close(feature_distance, explicit_distance)

    def test_equation_21_matches_direct_pairwise_mean(self):
        generator = torch.Generator().manual_seed(7)
        references = torch.randn(9, 13, generator=generator)
        candidates = torch.randn(4, 13, generator=generator)

        count, sum_phi, sum_phi_squared_norm = 0, None, None
        count, sum_phi, sum_phi_squared_norm = update_reference_sums(
            count, sum_phi, sum_phi_squared_norm, references[:4]
        )
        count, sum_phi, sum_phi_squared_norm = update_reference_sums(
            count, sum_phi, sum_phi_squared_norm, references[4:]
        )
        mu_omega = sum_phi / count
        v_omega = sum_phi_squared_norm / count - mu_omega.square().sum()
        equation_21_scores = iem_reward_from_reference_statistics(candidates, mu_omega, v_omega)
        direct_scores = (candidates[:, None] - references[None]).square().sum(dim=2).mean(dim=1)

        torch.testing.assert_close(equation_21_scores, direct_scores.double(), rtol=1e-5, atol=1e-6)

    def test_cosine_reference_mean_matches_direct_pairwise_average(self):
        references = torch.nn.functional.normalize(
            torch.tensor(
                [
                    [1.0, 2.0, 0.5],
                    [-0.5, 1.0, 2.0],
                    [2.0, -1.0, 1.0],
                ]
            ),
            dim=1,
        )
        candidates = torch.tensor(
            [
                [0.5, 1.0, -0.5],
                [1.0, -2.0, 0.25],
            ]
        )
        count, feature_sum = update_reference_feature_sum(
            0, None, references[:1]
        )
        count, feature_sum = update_reference_feature_sum(
            count, feature_sum, references[1:]
        )
        scores = cosine_distance_from_reference_mean(
            candidates,
            feature_sum / count,
        )
        normalized_candidates = torch.nn.functional.normalize(candidates, dim=1)
        direct = (1.0 - normalized_candidates @ references.T).mean(dim=1)
        torch.testing.assert_close(scores, direct)

    def test_raw_l2_matches_direct_pairwise_average_and_nearest(self):
        references = torch.tensor([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.0]])
        candidates = torch.tensor([[0.0, 4.0], [6.0, 8.0]])
        distances = torch.linalg.vector_norm(
            candidates[:, None] - references[None],
            dim=2,
        )

        scores = mean_pairwise_l2_distance(candidates, references)
        torch.testing.assert_close(scores, distances.mean(dim=1))

        nearest_scores, selected_count = mean_nearest_l2_distance(
            candidates,
            references,
            0.5,
        )
        expected = torch.topk(
            distances,
            2,
            dim=1,
            largest=False,
        ).values.mean(dim=1)
        self.assertEqual(selected_count, 2)
        torch.testing.assert_close(nearest_scores, expected)

    def test_nearest_cosine_distance_is_candidate_specific_and_rounds_up(self):
        generator = torch.Generator().manual_seed(31)
        references = torch.randn(256, 7, generator=generator)
        candidates = torch.randn(3, 7, generator=generator)

        scores, selected_count = mean_nearest_cosine_distance(
            candidates,
            references,
            0.1,
        )
        normalized_candidates = torch.nn.functional.normalize(candidates, dim=1)
        normalized_references = torch.nn.functional.normalize(references, dim=1)
        distances = (
            1.0 - normalized_candidates @ normalized_references.T
        ).clamp(0.0, 2.0)
        expected = torch.topk(
            distances,
            26,
            dim=1,
            largest=False,
        ).values.mean(dim=1)

        self.assertEqual(selected_count, 26)
        torch.testing.assert_close(scores, expected)

        full_scores, full_count = mean_nearest_cosine_distance(
            candidates,
            references,
            1.0,
        )
        self.assertEqual(full_count, 256)
        torch.testing.assert_close(full_scores, distances.mean(dim=1))

    def test_group_centering_cancels_reference_spread_for_equal_means(self):
        candidates = torch.tensor(
            [
                [0.5, -0.25],
                [1.0, 0.5],
                [-0.5, 1.5],
            ]
        )
        compact_references = torch.tensor(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        diffuse_references = torch.tensor(
            [
                [-10.0, -6.0],
                [10.0, 6.0],
            ]
        )

        centered_scores = []
        for references in (compact_references, diffuse_references):
            count, sum_phi, sum_phi_squared_norm = update_reference_sums(
                0,
                None,
                None,
                references,
            )
            mu_omega = sum_phi / count
            v_omega = (
                sum_phi_squared_norm / count
                - mu_omega.square().sum()
            )
            scores = iem_reward_from_reference_statistics(
                candidates,
                mu_omega,
                v_omega,
            )
            centered_scores.append(scores - scores.mean())

        torch.testing.assert_close(
            centered_scores[0],
            centered_scores[1],
        )

    def test_noise_tables_are_seeded_and_independent_across_levels(self):
        schedule = iem_sigma_schedule(1.0, 10.0, 4)
        first = sample_iem_noise_table(schedule, (2, 3), device="cpu", seed=123)
        second = sample_iem_noise_table(schedule, (2, 3), device="cpu", seed=123)

        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first[0], first[1]))


class PromptAndReservoirTests(unittest.TestCase):
    def test_prompt_sampling_balances_sources_and_honors_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for source_index in range(3):
                path = Path(directory) / f"source_{source_index}.txt"
                path.write_text("".join(f"s{source_index}-{row}\n" for row in range(10)))
                paths.append(path)

            sampler = BalancedPromptSampler(paths)
            first = sampler.sample(6, seed=11, excluded_prompts={"s0-0"})
            second = sampler.sample(6, seed=11, excluded_prompts={"s0-0"})
            prefixes = [record.text.split("-")[0] for record in first]

            self.assertEqual(first, second)
            self.assertEqual({prefix: prefixes.count(prefix) for prefix in set(prefixes)}, {"s0": 2, "s1": 2, "s2": 2})
            self.assertNotIn("s0-0", [record.text for record in first])

    def test_reservoir_is_bounded_and_round_trips(self):
        reservoir = ReferenceReservoir(capacity=3, prompt_source_sha256="source-hash")
        reservoir.update(torch.arange(8).reshape(4, 2), torch.arange(4), seed=5)
        self.assertEqual(len(reservoir), 3)

        restored = ReferenceReservoir(capacity=3, prompt_source_sha256="source-hash")
        restored.load_state_dict(reservoir.state_dict())
        self.assertEqual(restored.latents.dtype, torch.bfloat16)
        torch.testing.assert_close(restored.latents, reservoir.latents)
        torch.testing.assert_close(restored.prompt_ids, reservoir.prompt_ids)
        torch.testing.assert_close(restored.priorities, reservoir.priorities)

        incompatible = ReferenceReservoir(capacity=3, prompt_source_sha256="different")
        with self.assertRaisesRegex(ValueError, "prompt sources"):
            incompatible.load_state_dict(reservoir.state_dict())


class IEMRewardFlowTests(unittest.TestCase):
    def test_image_embedding_modes_do_not_construct_iem_schedule_or_load_encoder(self):
        class MockAccelerator:
            device = torch.device("cpu")
            num_processes = 1

        model_fields = {
            "clip_cosine": "clip_model_id",
            "clip_l2": "clip_model_id",
            "dino_cosine": "dino_model_id",
            "dino_l2": "dino_model_id",
            "tpips_overall": "tpips_model_id",
        }
        for metric, model_field in model_fields.items():
            with self.subTest(metric=metric):
                values = {
                    "distance_metric": metric,
                    model_field: f"local/{metric}",
                    "reference_prompt_mode": "same_prompt",
                    "reference_samples_per_prompt": 2,
                    "reference_batch_size": 1,
                    "feature_batch_size": 1,
                }
                if metric == "tpips_overall":
                    values["tpips_batch_size"] = 1
                reward = IEMReward(
                    pipe=None,
                    model=None,
                    accelerator=MockAccelerator(),
                    config=SimpleNamespace(**values),
                )
                self.assertIsNone(reward.sigma_schedule)
                self.assertIsNone(reward._image_encoder)
                self.assertEqual(reward.distance_metric, metric)
                self.assertTrue(reward.reward_log_name.endswith("Distance"))

    def test_l2_image_embeddings_preserve_raw_encoder_scale(self):
        class MockAccelerator:
            device = torch.device("cpu")
            num_processes = 1

            @staticmethod
            def autocast():
                return nullcontext()

        class FakeVAE:
            dtype = torch.float32
            config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

            @staticmethod
            def decode(latents, return_dict=False):
                return (latents,)

        class FakeImagePostprocessor:
            @staticmethod
            def postprocess(images, output_type):
                return images

        class FakeImageProcessor:
            @staticmethod
            def __call__(images, return_tensors):
                return {"pixel_values": torch.stack(images).float()}

        class FakeEncoder:
            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, *, pixel_values):
                vectors = pixel_values.flatten(start_dim=1)
                return SimpleNamespace(
                    image_embeds=2.0 * vectors,
                    last_hidden_state=(3.0 * vectors).unsqueeze(1),
                )

        latents = torch.tensor(
            [
                [[[0.25, 0.50]], [[0.75, 1.00]], [[0.00, 0.10]]],
                [[[0.10, 0.20]], [[0.30, 0.40]], [[0.50, 0.60]]],
            ]
        )
        image_bytes = latents.mul(255).round().clamp(0, 255).float()
        pipe = SimpleNamespace(
            vae=FakeVAE(),
            image_processor=FakeImagePostprocessor(),
        )

        for metric, model_field, scale in (
            ("clip_l2", "clip_model_id", 2.0),
            ("dino_l2", "dino_model_id", 3.0),
        ):
            with self.subTest(metric=metric):
                reward = IEMReward(
                    pipe,
                    None,
                    MockAccelerator(),
                    SimpleNamespace(
                        distance_metric=metric,
                        **{model_field: f"local/{metric}"},
                        reference_prompt_mode="same_prompt",
                        reference_samples_per_prompt=2,
                        reference_batch_size=1,
                        feature_batch_size=2,
                    ),
                )
                reward._image_processor = FakeImageProcessor()
                reward._image_encoder = FakeEncoder()
                features = reward._image_embeddings_from_latents(latents)
                torch.testing.assert_close(
                    features,
                    scale * image_bytes.flatten(start_dim=1),
                )
                self.assertFalse(
                    torch.allclose(features.norm(dim=1), torch.ones(2))
                )

    def test_tpips_endpoint_embeddings_always_use_overall_factor(self):
        class MockAccelerator:
            device = torch.device("cpu")
            num_processes = 1

            @staticmethod
            def autocast():
                return nullcontext()

        class FakeVAE:
            dtype = torch.float32
            config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

            @staticmethod
            def decode(latents, return_dict=False):
                if return_dict:
                    raise AssertionError("expected tuple VAE output")
                return (latents,)

        class FakeImageProcessor:
            @staticmethod
            def postprocess(images, output_type):
                if output_type != "pt":
                    raise AssertionError("expected tensor image output")
                return images.clamp(0.0, 1.0)

        class FakeTPIPS:
            def __init__(self):
                self.calls = []

            def to(self, device):
                return self

            def eval(self):
                return self

            def embed(self, images, *, factor, normalized):
                self.calls.append((factor, normalized, images.shape[0]))
                values = torch.stack(
                    (images.flatten(start_dim=1).mean(dim=1), torch.ones(len(images))),
                    dim=1,
                )
                return torch.nn.functional.normalize(values, dim=1)

        config = SimpleNamespace(
            distance_metric="tpips_overall",
            tpips_model_id="local/tpips",
            tpips_batch_size=1,
            reference_prompt_mode="same_prompt",
            reference_samples_per_prompt=2,
            reference_batch_size=1,
            feature_batch_size=2,
        )
        pipe = SimpleNamespace(
            vae=FakeVAE(),
            image_processor=FakeImageProcessor(),
        )
        reward = IEMReward(pipe, None, MockAccelerator(), config)
        encoder = FakeTPIPS()
        reward._image_encoder = encoder

        features = reward._image_embeddings_from_latents(
            torch.stack(
                (
                    torch.zeros(3, 2, 2),
                    torch.full((3, 2, 2), 0.5),
                )
            )
        )

        self.assertEqual(
            encoder.calls,
            [("overall", True, 1), ("overall", True, 1)],
        )
        torch.testing.assert_close(features.norm(dim=1), torch.ones(2))

    def test_nearest_selection_config_and_checkpoint_compatibility(self):
        class MockAccelerator:
            device = torch.device("cpu")
            num_processes = 1

        cosine_values = {
            "distance_metric": "clip_cosine",
            "clip_model_id": "local/clip",
            "reference_prompt_mode": "same_prompt",
            "reference_samples_per_prompt": 2,
            "reference_batch_size": 1,
            "feature_batch_size": 1,
            "reference_selection_mode": "nearest",
            "nearest_reference_fraction": 0.25,
        }
        reward = IEMReward(
            pipe=None,
            model=None,
            accelerator=MockAccelerator(),
            config=SimpleNamespace(**cosine_values),
        )
        state = reward.state_dict()
        self.assertEqual(state["reference_selection_mode"], "nearest")
        self.assertEqual(state["nearest_reference_fraction"], 0.25)

        incompatible_state = dict(state, nearest_reference_fraction=0.5)
        with self.assertRaisesRegex(ValueError, "nearest_reference_fraction"):
            reward.load_state_dict(incompatible_state)

        all_reward = IEMReward(
            pipe=None,
            model=None,
            accelerator=MockAccelerator(),
            config=SimpleNamespace(
                **{
                    key: value
                    for key, value in cosine_values.items()
                    if key not in (
                        "reference_selection_mode",
                        "nearest_reference_fraction",
                    )
                }
            ),
        )
        all_reward.load_state_dict(
            {
                "distance_metric": "clip_cosine",
                "reference_prompt_mode": "same_prompt",
            }
        )
        with self.assertRaisesRegex(ValueError, "reference_selection_mode"):
            reward.load_state_dict(all_reward.state_dict())

        for invalid_fraction in (0.0, -0.1, 1.01):
            with self.subTest(invalid_fraction=invalid_fraction):
                with self.assertRaisesRegex(ValueError, "must be in"):
                    IEMReward(
                        pipe=None,
                        model=None,
                        accelerator=MockAccelerator(),
                        config=SimpleNamespace(
                            **dict(
                                cosine_values,
                                nearest_reference_fraction=invalid_fraction,
                            )
                        ),
                    )

        iem_values = {
            "distance_metric": "iem",
            "reference_prompt_mode": "same_prompt",
            "reference_samples_per_prompt": 2,
            "reference_batch_size": 1,
            "feature_batch_size": 1,
            "reference_selection_mode": "nearest",
            "nearest_reference_fraction": 0.1,
            "sigma_min": 1.0,
            "sigma_max": 4.0,
            "num_steps": 2,
            "level_batch_size": 1,
            "noise_table_count": 1,
        }
        with self.assertRaisesRegex(ValueError, "not yet supported"):
            IEMReward(
                pipe=None,
                model=None,
                accelerator=MockAccelerator(),
                config=SimpleNamespace(**iem_values),
            )

    def test_same_prompt_references_use_independent_noises_and_no_reservoir(self):
        class MockAccelerator:
            device = torch.device("cpu")
            process_index = 0
            num_processes = 1

            @staticmethod
            def autocast():
                return nullcontext()

            @staticmethod
            def reduce(tensor, reduction="sum"):
                if reduction != "sum":
                    raise AssertionError(f"unexpected reduction: {reduction}")
                return tensor

        class MockTransformer:
            def __init__(self):
                self.disabled = False
                self.forward_calls = 0

            @contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False

            def __call__(self, hidden_states, **kwargs):
                del kwargs
                if not self.disabled:
                    raise AssertionError("IEM denoising probe used an active adapter")
                self.forward_calls += 1
                return (torch.zeros_like(hidden_states),)

        class MockPipe:
            def __init__(self, transformer):
                self.transformer = transformer
                self.generated = []

            def __call__(
                self,
                *,
                prompt_embeds,
                pooled_prompt_embeds,
                generator,
                **kwargs,
            ):
                del kwargs
                if not self.transformer.disabled:
                    raise AssertionError("reference generation used an active adapter")
                if not torch.equal(
                    prompt_embeds,
                    prompt_embeds[:1].expand_as(prompt_embeds),
                ):
                    raise AssertionError(
                        "same-prompt reference batch mixed prompt embeddings"
                    )
                if not torch.equal(
                    pooled_prompt_embeds,
                    pooled_prompt_embeds[:1].expand_as(
                        pooled_prompt_embeds
                    ),
                ):
                    raise AssertionError(
                        "same-prompt reference batch mixed pooled embeddings"
                    )
                latents = torch.stack(
                    [
                        torch.randn(1, 2, 2, generator=row_generator)
                        for row_generator in generator
                    ]
                )
                condition_id = float(prompt_embeds[0].item())
                self.generated.extend(
                    (condition_id, latent.clone())
                    for latent in latents
                )
                return (latents,)

        config = SimpleNamespace(
            reference_prompt_mode="same_prompt",
            reference_samples_per_prompt=3,
            sigma_min=1.0,
            sigma_max=4.0,
            num_steps=2,
            reference_batch_size=3,
            feature_batch_size=2,
            level_batch_size=1,
            noise_table_count=2,
            seed=3,
            resolution=2,
            num_inference_steps=2,
            guidance_scale=1.0,
        )
        transformer = MockTransformer()
        pipe = MockPipe(transformer)
        reward = IEMReward(pipe, transformer, MockAccelerator(), config)
        candidate_latents = torch.zeros(4, 1, 2, 2)
        candidate_prompt_embeds = torch.tensor(
            [1.0, 1.0, 2.0, 2.0]
        ).reshape(4, 1, 1)
        candidate_pooled_prompt_embeds = torch.tensor(
            [1.0, 1.0, 2.0, 2.0]
        ).reshape(4, 1)
        scores, metrics = reward.score(
            candidate_latents,
            candidate_prompt_embeds,
            candidate_pooled_prompt_embeds,
            ["alpha", "alpha", "beta", "beta"],
            epoch=5,
            group_size=2,
        )

        self.assertEqual(scores.shape, (4,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(metrics["iem_reference_bank_size"], 0.0)
        self.assertEqual(metrics["iem_reference_subset_size"], 3.0)
        self.assertEqual(metrics["iem_same_prompt_reference_groups"], 2.0)
        self.assertEqual(metrics["iem_same_prompt_references_generated"], 6.0)
        self.assertIsNone(reward.reservoir)
        self.assertGreater(transformer.forward_calls, 0)
        self.assertFalse(transformer.disabled)

        condition_ids = {"alpha": 1.0, "beta": 2.0}
        for prompt, condition_id in condition_ids.items():
            prompt_latents = [
                latent
                for generated_condition, latent in pipe.generated
                if generated_condition == condition_id
            ]
            self.assertEqual(len(prompt_latents), 3)
            self.assertFalse(torch.equal(prompt_latents[0], prompt_latents[1]))

        assignments = reward._noise_table_assignments(4, 2, epoch=5)

        def zero_velocity(x_t, flow_time):
            del flow_time
            return torch.zeros_like(x_t)

        for group_index, prompt in enumerate(("alpha", "beta")):
            table_index = int(assignments[group_index * 2].item())
            noise_table = sample_iem_noise_table(
                reward.sigma_schedule,
                (1, 2, 2),
                device="cpu",
                seed=(
                    config.seed
                    + 90_000_000
                    + 5 * config.noise_table_count
                    + table_index
                ),
            )
            prompt_latents = torch.stack(
                [
                    latent
                    for generated_condition, latent in pipe.generated
                    if generated_condition == condition_ids[prompt]
                ]
            )
            reference_features = iem_features(
                prompt_latents,
                zero_velocity,
                reward.sigma_schedule,
                noise_table,
            )
            candidate_features = iem_features(
                candidate_latents[group_index * 2 : group_index * 2 + 2],
                zero_velocity,
                reward.sigma_schedule,
                noise_table,
            )
            direct_scores = (
                candidate_features[:, None] - reference_features[None]
            ).square().sum(dim=2).mean(dim=1)
            torch.testing.assert_close(
                scores[group_index * 2 : group_index * 2 + 2],
                direct_scores,
            )

        first_seed = same_prompt_reference_seed(3, 5, "alpha", 0)
        self.assertEqual(
            first_seed,
            same_prompt_reference_seed(3, 5, "alpha", 0),
        )
        self.assertNotEqual(
            first_seed,
            same_prompt_reference_seed(3, 5, "alpha", 1),
        )

        self.assertEqual(
            reward.state_dict(),
            {
                "distance_metric": "iem",
                "reference_prompt_mode": "same_prompt",
                "reference_selection_mode": "all",
                "nearest_reference_fraction": 0.1,
            },
        )
        reward.load_state_dict(reward.state_dict())
        with self.assertRaisesRegex(ValueError, "distance_metric"):
            reward.load_state_dict(
                {"distance_metric": "clip_cosine", "reference_prompt_mode": "same_prompt"}
            )
        with self.assertRaisesRegex(ValueError, "current config"):
            reward.load_state_dict({"reference_prompt_mode": "diverse"})

    def test_same_prompt_clip_cosine_matches_direct_reference_average(self):
        class MockAccelerator:
            device = torch.device("cpu")
            process_index = 0
            num_processes = 1

            @staticmethod
            def autocast():
                return nullcontext()

            @staticmethod
            def reduce(tensor, reduction="sum"):
                if reduction != "sum":
                    raise AssertionError(f"unexpected reduction: {reduction}")
                return tensor

        class MockTransformer:
            def __init__(self):
                self.disabled = False

            @contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False

        class MockPipe:
            def __init__(self, transformer):
                self.transformer = transformer
                self.generated = []

            def __call__(
                self,
                *,
                prompt_embeds,
                pooled_prompt_embeds,
                generator,
                **kwargs,
            ):
                del pooled_prompt_embeds, kwargs
                if not self.transformer.disabled:
                    raise AssertionError("reference generation used an active adapter")
                condition = float(prompt_embeds[0].item())
                latents = torch.stack(
                    [
                        torch.randn(1, 1, 2, generator=row_generator)
                        + condition
                        for row_generator in generator
                    ]
                )
                self.generated.extend(
                    (condition, latent.clone()) for latent in latents
                )
                return (latents,)

        config = SimpleNamespace(
            distance_metric="clip_cosine",
            clip_model_id="local/clip",
            reference_prompt_mode="same_prompt",
            reference_samples_per_prompt=3,
            reference_batch_size=2,
            feature_batch_size=2,
            seed=11,
            resolution=2,
            num_inference_steps=2,
            guidance_scale=1.0,
        )
        transformer = MockTransformer()
        pipe = MockPipe(transformer)
        reward = IEMReward(pipe, transformer, MockAccelerator(), config)
        reward._image_embeddings_from_latents = lambda values: (
            torch.nn.functional.normalize(
                torch.as_tensor(values).float().flatten(start_dim=1),
                dim=1,
            )
        )
        candidate_latents = torch.tensor(
            [
                [[[1.0, 0.0]]],
                [[[0.5, 1.0]]],
                [[[2.0, -1.0]]],
                [[[1.0, 2.0]]],
            ]
        )
        prompt_embeds = torch.tensor([1.0, 1.0, 2.0, 2.0]).reshape(4, 1, 1)
        pooled_embeds = torch.tensor([1.0, 1.0, 2.0, 2.0]).reshape(4, 1)
        scores, metrics = reward.score(
            candidate_latents,
            prompt_embeds,
            pooled_embeds,
            ["alpha", "alpha", "beta", "beta"],
            epoch=4,
            group_size=2,
        )

        for group_index, condition in enumerate((1.0, 2.0)):
            references = torch.stack(
                [
                    latent
                    for generated_condition, latent in pipe.generated
                    if generated_condition == condition
                ]
            )
            reference_features = reward._image_embeddings_from_latents(references)
            candidate_features = reward._image_embeddings_from_latents(
                candidate_latents[group_index * 2 : group_index * 2 + 2]
            )
            direct = (
                1.0 - candidate_features @ reference_features.T
            ).mean(dim=1)
            torch.testing.assert_close(
                scores[group_index * 2 : group_index * 2 + 2],
                direct,
            )
        self.assertEqual(metrics["clip_cosine_reference_subset_size"], 3.0)
        self.assertEqual(metrics["clip_cosine_same_prompt_reference_groups"], 2.0)
        self.assertFalse(transformer.disabled)

        pipe.generated.clear()
        reward.reference_selection_mode = "nearest"
        reward.nearest_reference_fraction = 0.5
        nearest_scores, nearest_metrics = reward.score(
            candidate_latents,
            prompt_embeds,
            pooled_embeds,
            ["alpha", "alpha", "beta", "beta"],
            epoch=5,
            group_size=2,
        )
        for group_index, condition in enumerate((1.0, 2.0)):
            references = torch.stack(
                [
                    latent
                    for generated_condition, latent in pipe.generated
                    if generated_condition == condition
                ]
            )
            reference_features = reward._image_embeddings_from_latents(references)
            candidate_features = reward._image_embeddings_from_latents(
                candidate_latents[group_index * 2 : group_index * 2 + 2]
            )
            distances = (
                1.0 - candidate_features @ reference_features.T
            ).clamp(0.0, 2.0)
            direct = torch.topk(
                distances,
                2,
                dim=1,
                largest=False,
            ).values.mean(dim=1)
            torch.testing.assert_close(
                nearest_scores[group_index * 2 : group_index * 2 + 2],
                direct,
            )
        self.assertEqual(
            nearest_metrics["clip_cosine_selected_reference_count"],
            2.0,
        )
        self.assertEqual(
            nearest_metrics["clip_cosine_selected_reference_fraction"],
            2 / 3,
        )
        self.assertIn(
            "timing/clip_cosine_reference_selection_seconds",
            nearest_metrics,
        )

    def test_same_prompt_mode_rejects_mixed_candidate_groups(self):
        with self.assertRaisesRegex(ValueError, "exactly one prompt"):
            IEMReward._candidate_prompt_groups(
                4,
                ["alpha", "beta", "gamma", "gamma"],
                group_size=2,
            )

    def test_distributed_reference_features_remove_uneven_padding(self):
        references = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.0, 1.0],
            ]
        )

        class MockAccelerator:
            device = torch.device("cpu")
            process_index = 0
            num_processes = 2

            @staticmethod
            def gather(tensor):
                if tensor.dtype == torch.int64:
                    return torch.tensor([3, 2], dtype=torch.int64)
                if tensor.dtype == torch.uint8:
                    return torch.tensor(
                        [1, 1, 1, 1, 1, 0],
                        dtype=torch.uint8,
                    )
                rank_zero = torch.nn.functional.normalize(
                    references[[0, 2, 4]],
                    dim=1,
                )
                rank_one = torch.nn.functional.normalize(
                    references[[1, 3]],
                    dim=1,
                )
                return torch.cat(
                    (
                        rank_zero,
                        rank_one,
                        torch.zeros(1, rank_one.shape[1]),
                    )
                )

        reward = object.__new__(IEMReward)
        reward.accelerator = MockAccelerator()
        reward.config = SimpleNamespace(feature_batch_size=2)
        reward._image_embeddings_from_latents = lambda values: (
            torch.nn.functional.normalize(torch.as_tensor(values).float(), dim=1)
        )

        count, gathered = reward._reference_cosine_features(references)
        expected = torch.nn.functional.normalize(
            references[[0, 2, 4, 1, 3]],
            dim=1,
        )
        self.assertEqual(count, 5)
        torch.testing.assert_close(gathered, expected)

    def test_diverse_nearest_cosine_matches_direct_topk(self):
        class MockAccelerator:
            device = torch.device("cpu")
            process_index = 0
            num_processes = 1

            @staticmethod
            def gather(tensor):
                return tensor

        references = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.0, 1.0],
            ]
        )
        candidates = torch.tensor(
            [
                [1.0, 0.25],
                [-0.25, -1.0],
            ]
        )
        reward = object.__new__(IEMReward)
        reward.accelerator = MockAccelerator()
        reward.config = SimpleNamespace(feature_batch_size=2)
        reward.distance_metric = "dino_cosine"
        reward.reference_selection_mode = "nearest"
        reward.nearest_reference_fraction = 0.4
        reward.reservoir = [None] * len(references)
        reward._image_embeddings_from_latents = lambda values: (
            torch.nn.functional.normalize(torch.as_tensor(values).float(), dim=1)
        )
        reward._offload_image_encoder = lambda: None

        scores, metrics = reward._score_diverse_cosine(
            candidates,
            references,
            reference_refresh_seconds=1.5,
        )
        expected, selected_count = mean_nearest_cosine_distance(
            candidates,
            references,
            0.4,
        )
        torch.testing.assert_close(scores, expected)
        self.assertEqual(selected_count, 2)
        self.assertEqual(
            metrics["dino_cosine_selected_reference_count"],
            2.0,
        )
        self.assertEqual(
            metrics["dino_cosine_selected_reference_fraction"],
            0.4,
        )

    def test_reference_generation_and_scoring_use_disabled_ram_adapter(self):
        class MockAccelerator:
            device = torch.device("cpu")
            process_index = 0
            num_processes = 1

            @staticmethod
            def autocast():
                return nullcontext()

            @staticmethod
            def gather(tensor):
                return tensor

            @staticmethod
            def reduce(tensor, reduction="sum"):
                if reduction != "sum":
                    raise AssertionError(f"unexpected reduction: {reduction}")
                return tensor

        class MockTransformer:
            def __init__(self):
                self.disabled = False
                self.forward_calls = 0

            @contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False

            def __call__(self, hidden_states, **kwargs):
                del kwargs
                if not self.disabled:
                    raise AssertionError("IEM denoising probe used an active adapter")
                self.forward_calls += 1
                return (torch.zeros_like(hidden_states),)

        class MockPipe:
            def __init__(self, transformer):
                self.transformer = transformer

            def encode_prompt(self, prompts, **kwargs):
                del kwargs
                count = len(prompts)
                return torch.zeros(count, 1, 1), None, torch.zeros(count, 1), None

            def __call__(self, prompts, generator, **kwargs):
                del kwargs
                if not self.transformer.disabled:
                    raise AssertionError("reference generation used an active adapter")
                return (torch.randn(len(prompts), 1, 2, 2, generator=generator),)

        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "references.txt"
            prompt_path.write_text("alpha\nbeta\ngamma\ndelta\n")
            config = SimpleNamespace(
                reference_prompt_files=[str(prompt_path)],
                reference_bank_size=4,
                sigma_min=1.0,
                sigma_max=4.0,
                num_steps=2,
                reference_samples_per_epoch=2,
                reference_subset_size=2,
                reference_batch_size=2,
                feature_batch_size=2,
                level_batch_size=1,
                noise_table_count=1,
                seed=3,
                resolution=2,
                num_inference_steps=2,
                guidance_scale=1.0,
            )
            transformer = MockTransformer()
            reward = IEMReward(MockPipe(transformer), transformer, MockAccelerator(), config)
            self.assertEqual(reward.reference_prompt_mode, "diverse")
            scores, metrics = reward.score(
                torch.zeros(2, 1, 2, 2),
                torch.zeros(2, 1, 1),
                torch.zeros(2, 1),
                ["candidate"],
                epoch=0,
                group_size=2,
            )

            self.assertEqual(scores.shape, (2,))
            self.assertTrue(torch.isfinite(scores).all())
            self.assertEqual(metrics["iem_reference_bank_size"], 2.0)
            self.assertGreater(transformer.forward_calls, 0)
            self.assertFalse(transformer.disabled)

            state = reward.state_dict()
            self.assertEqual(state["reference_prompt_mode"], "diverse")
            legacy_state = {
                key: value
                for key, value in state.items()
                if key
                not in (
                    "distance_metric",
                    "reference_prompt_mode",
                    "reference_selection_mode",
                    "nearest_reference_fraction",
                )
            }
            reward.load_state_dict(legacy_state)


if __name__ == "__main__":
    unittest.main()
