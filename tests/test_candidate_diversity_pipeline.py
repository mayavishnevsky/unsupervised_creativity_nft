import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from flow_grpo.diffusers_patch import pipeline_with_logprob as pipeline_module


class RecordingDenoiser:
    def __init__(self):
        self.config = SimpleNamespace(in_channels=4)
        self.conditions = []

    def __call__(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        pooled_projections,
        **kwargs,
    ):
        del timestep, kwargs
        self.conditions.append(
            (encoder_hidden_states.clone(), pooled_projections.clone())
        )
        return (torch.zeros_like(hidden_states),)


class FakePipeline:
    default_sample_size = 2
    vae_scale_factor = 1

    def __init__(self):
        self.transformer = RecordingDenoiser()
        self.scheduler = SimpleNamespace(
            sigmas=torch.tensor([1.0, 0.5, 0.0]),
            timesteps=torch.tensor([1000.0, 500.0]),
        )
        self._execution_device = torch.device("cpu")

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    def check_inputs(self, *args, **kwargs):
        pass

    def encode_prompt(self, **kwargs):
        return (
            kwargs["prompt_embeds"],
            kwargs["negative_prompt_embeds"],
            kwargs["pooled_prompt_embeds"],
            kwargs["negative_pooled_prompt_embeds"],
        )

    def prepare_latents(self, *args):
        return args[-1]

    def maybe_free_model_hooks(self):
        pass


class CandidateConditioningPipelineTest(unittest.TestCase):
    def test_callback_changes_only_per_step_denoiser_conditioning(self):
        pipe = FakePipeline()
        prompt = torch.ones(2, 3, 4)
        pooled = torch.ones(2, 5)
        latents = torch.zeros(2, 4, 2, 2)
        callback_sigmas = []

        def conditioning_callback(sigma):
            callback_sigmas.append(sigma)
            return prompt + sigma, pooled + 2.0 * sigma

        def retrieve_timesteps(scheduler, *args, **kwargs):
            del args, kwargs
            return scheduler.timesteps, len(scheduler.timesteps)

        def run_sampling(v_pred_fn, z, sigma_schedule, *args, **kwargs):
            del args, kwargs
            for sigma in sigma_schedule[:-1]:
                v_pred_fn(z, sigma)
            return z, [z], []

        with mock.patch.object(
            pipeline_module,
            "retrieve_timesteps",
            side_effect=retrieve_timesteps,
        ), mock.patch.object(
            pipeline_module,
            "run_sampling",
            side_effect=run_sampling,
        ):
            endpoint, _, _ = pipeline_module.pipeline_with_logprob(
                pipe,
                prompt_embeds=prompt,
                pooled_prompt_embeds=pooled,
                latents=latents,
                num_inference_steps=2,
                guidance_scale=1.0,
                output_type="latent",
                model_type="sd3",
                conditioning_step_callback=conditioning_callback,
            )

        self.assertIs(endpoint, latents)
        self.assertEqual(callback_sigmas, [1.0, 0.5])
        self.assertTrue(
            torch.equal(pipe.transformer.conditions[0][0], prompt + 1.0)
        )
        self.assertTrue(
            torch.equal(pipe.transformer.conditions[1][1], pooled + 1.0)
        )
        self.assertTrue(torch.equal(prompt, torch.ones_like(prompt)))


if __name__ == "__main__":
    unittest.main()
