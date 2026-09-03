import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from flow_grpo.diffusers_patch import pipeline_with_logprob as pipeline_module
from flow_grpo.diffusers_patch import solver as solver_module
from flow_grpo.evaluation_renoising import (
    EvaluationRenoising,
    flow_forward_renoise,
)
from flow_grpo.nft_validation import render_fixed_validation


class EvaluationRenoisingTests(unittest.TestCase):
    def test_flow_forward_transition_has_expected_coefficients(self):
        lower = torch.tensor([2.0])
        noise = torch.tensor([0.25])

        result = flow_forward_renoise(
            lower,
            noise,
            sigma_current=0.8,
            sigma_next=0.4,
        )

        carry = (1.0 - 0.8) / (1.0 - 0.4)
        noise_scale = (0.8**2 - (carry * 0.4) ** 2) ** 0.5
        torch.testing.assert_close(result, lower * carry + noise * noise_scale)

    def test_noise_is_stable_across_evaluation_batch_splits(self):
        lower = torch.zeros((2, 1, 2, 2))
        together = EvaluationRenoising(
            sample_seeds=(11, 29),
            active_steps=(True,),
            repeats=1,
        ).renoise(lower, torch.tensor(0.8), torch.tensor(0.4), 0, 0)
        split = torch.cat(
            [
                EvaluationRenoising(
                    sample_seeds=(seed,),
                    active_steps=(True,),
                    repeats=1,
                ).renoise(
                    lower[index : index + 1],
                    torch.tensor(0.8),
                    torch.tensor(0.4),
                    0,
                    0,
                )
                for index, seed in enumerate((11, 29))
            ]
        )

        torch.testing.assert_close(together, split)

    def test_dpm_repeats_only_active_steps_and_restores_history(self):
        controller = EvaluationRenoising(
            sample_seeds=(17,),
            active_steps=(True, False),
            repeats=2,
        )
        denoiser_sigmas = []
        state_before_calls = []

        def denoiser(sample, sigma):
            denoiser_sigmas.append(float(sigma))
            return torch.full_like(sample, len(denoiser_sigmas))

        def fake_dpm_step(
            _order,
            *,
            model_output,
            sample,
            step_index,
            timesteps,
            sigmas,
            dpm_state,
        ):
            del timesteps, sigmas
            state_before_calls.append(
                (
                    step_index,
                    dpm_state.lower_order_nums,
                    tuple(dpm_state.model_outputs),
                )
            )
            dpm_state.update(model_output)
            dpm_state.update_lower_order()
            return sample + 1.0, model_output, None

        with mock.patch.object(
            solver_module,
            "dpm_step",
            side_effect=fake_dpm_step,
        ):
            solver_module.run_sampling(
                denoiser,
                torch.zeros((1, 1, 2, 2)),
                torch.tensor([0.8, 0.4, 0.0]),
                solver="dpm2",
                determistic=True,
                evaluation_renoising=controller,
            )

        self.assertEqual(len(denoiser_sigmas), 4)
        for actual, expected in zip(
            denoiser_sigmas,
            (0.8, 0.8, 0.8, 0.4),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            [entry[0] for entry in state_before_calls],
            [0, 0, 0, 1],
        )
        self.assertEqual(
            [entry[1] for entry in state_before_calls],
            [0, 0, 0, 1],
        )
        for _step, _lower_order, history in state_before_calls[:3]:
            self.assertEqual(history, (None, None))
        self.assertIsNotNone(state_before_calls[3][2][-1])

    def test_disabled_mode_matches_previous_dpm2_loop_bitwise(self):
        sigmas = torch.tensor([0.9, 0.65, 0.4, 0.15, 0.0])
        initial = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 10

        def denoiser(sample, sigma):
            return sample * 0.125 + sigma

        legacy_state = solver_module.DPMState(order=2)
        legacy_latents = [initial]
        legacy = initial
        for step_index, sigma in enumerate(sigmas[:-1]):
            prediction = denoiser(legacy, sigma)
            legacy, _prediction, _log_prob = solver_module.dpm_step(
                2,
                model_output=prediction.float(),
                sample=legacy.float(),
                step_index=step_index,
                timesteps=sigmas[:-1],
                sigmas=sigmas,
                dpm_state=legacy_state,
            )
            legacy = legacy.to(initial.dtype)
            legacy_latents.append(legacy)

        current, current_latents, _log_probs = solver_module.run_sampling(
            denoiser,
            initial,
            sigmas,
            solver="dpm2",
            determistic=True,
        )

        self.assertTrue(torch.equal(current, legacy))
        self.assertEqual(len(current_latents), len(legacy_latents))
        for actual, expected in zip(
            current_latents,
            legacy_latents,
            strict=True,
        ):
            self.assertTrue(torch.equal(actual, expected))

    def test_dpm2_renoising_is_finite_and_repeatable(self):
        sigmas = torch.tensor([0.9, 0.65, 0.4, 0.15, 0.0])
        initial = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 10
        outputs = []
        for _ in range(2):
            controller = EvaluationRenoising(
                sample_seeds=(41,),
                active_steps=(True, True, False, False),
                repeats=2,
            )
            output, _latents, _log_probs = solver_module.run_sampling(
                lambda sample, sigma: sample * 0.125 + sigma,
                initial.clone(),
                sigmas,
                solver="dpm2",
                determistic=True,
                evaluation_renoising=controller,
            )
            outputs.append(output)

        self.assertTrue(torch.isfinite(outputs[0]).all())
        self.assertTrue(torch.equal(outputs[0], outputs[1]))

    def test_renoising_rejects_stochastic_sampling(self):
        controller = EvaluationRenoising(
            sample_seeds=(17,),
            active_steps=(True,),
            repeats=1,
        )
        with self.assertRaisesRegex(ValueError, "deterministic"):
            solver_module.run_sampling(
                lambda sample, _sigma: torch.zeros_like(sample),
                torch.zeros((1, 1, 2, 2)),
                torch.tensor([0.8, 0.0]),
                evaluation_renoising=controller,
            )

    def test_training_and_validation_defaults_remain_disabled(self):
        self.assertIsNone(
            inspect.signature(solver_module.run_sampling)
            .parameters["evaluation_renoising"]
            .default
        )
        self.assertIsNone(
            inspect.signature(pipeline_module.pipeline_with_logprob)
            .parameters["evaluation_renoising"]
            .default
        )
        self.assertEqual(
            inspect.signature(render_fixed_validation)
            .parameters["renoising_repeats"]
            .default,
            0,
        )

    def test_baseline_validation_rejects_renoising(self):
        config = SimpleNamespace(sample=SimpleNamespace(deterministic=True))
        with self.assertRaisesRegex(ValueError, "baseline"):
            render_fixed_validation(
                None,
                None,
                config,
                None,
                0,
                1,
                0,
                None,
                [],
                label="baseline",
                baseline=True,
                renoising_repeats=1,
            )


if __name__ == "__main__":
    unittest.main()
