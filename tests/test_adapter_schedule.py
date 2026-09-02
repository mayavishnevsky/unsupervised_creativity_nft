import unittest

import torch

from flow_grpo.adapter_schedule import LoraAdapterScaler, linear_decay_strengths
from flow_grpo.diffusers_patch.solver import run_sampling


class FakeLoraLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling = {"default": 2.0, "other": 3.0}


class AdapterScheduleTests(unittest.TestCase):
    def test_linear_decay_has_exact_full_transition_and_zero_regions(self):
        strengths = linear_decay_strengths(40, 10, 10)

        self.assertEqual(len(strengths), 40)
        self.assertEqual(strengths[:10], (1.0,) * 10)
        self.assertEqual(strengths[-10:], (0.0,) * 10)
        self.assertAlmostEqual(strengths[10], 20 / 21)
        self.assertAlmostEqual(strengths[29], 1 / 21)

    def test_scaler_changes_only_requested_adapter_and_restores(self):
        model = torch.nn.Sequential(FakeLoraLayer(), FakeLoraLayer())
        scaler = LoraAdapterScaler(model, "default")

        scaler.set_strength(0.25)
        for layer in model:
            self.assertEqual(layer.scaling["default"], 0.5)
            self.assertEqual(layer.scaling["other"], 3.0)

        scaler.restore()
        for layer in model:
            self.assertEqual(layer.scaling["default"], 2.0)

    def test_sampler_calls_callback_before_each_denoiser_evaluation(self):
        callback_indices = []
        denoiser_indices = []

        def callback(step_index):
            callback_indices.append(step_index)

        def denoiser(sample, _sigma):
            denoiser_indices.append(callback_indices[-1])
            return torch.zeros_like(sample)

        run_sampling(
            denoiser,
            torch.ones(1, 1, 2, 2),
            torch.tensor([0.9, 0.5, 0.1]),
            solver="dpm1",
            determistic=True,
            denoiser_step_callback=callback,
        )

        self.assertEqual(callback_indices, [0, 1])
        self.assertEqual(denoiser_indices, [0, 1])


if __name__ == "__main__":
    unittest.main()
