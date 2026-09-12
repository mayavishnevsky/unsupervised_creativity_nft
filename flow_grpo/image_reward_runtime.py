"""Standard image rewards behind the NFT creativity trainer reward interface."""

from __future__ import annotations

import torch

from flow_grpo.rewards import multi_score


class StandardImageRewardSet:
    """Decode endpoint latents in bounded batches and score generated images."""

    def __init__(self, pipe, device, reward_weights, *, decode_batch_size: int):
        self.pipe = pipe
        self.device = device
        self.decode_batch_size = int(decode_batch_size)
        if self.decode_batch_size <= 0:
            raise ValueError("reward decode batch size must be positive")
        if not reward_weights:
            raise ValueError("at least one standard image reward is required")
        self.reward_weights = dict(reward_weights)
        self.scorer = multi_score(device, self.reward_weights)

    @torch.no_grad()
    def score(
        self,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        prompts,
        *,
        epoch,
        group_size,
    ):
        del prompt_embeds, pooled_prompt_embeds, epoch, group_size
        if len(latents) != len(prompts):
            raise ValueError("latent and prompt counts differ")

        collected: dict[str, list[torch.Tensor]] = {}
        for start in range(0, len(latents), self.decode_batch_size):
            stop = min(start + self.decode_batch_size, len(latents))
            latent_batch = latents[start:stop].to(dtype=self.pipe.vae.dtype)
            normalized = (
                latent_batch / self.pipe.vae.config.scaling_factor
                + self.pipe.vae.config.shift_factor
            )
            decoded = self.pipe.vae.decode(normalized, return_dict=False)[0]
            images = self.pipe.image_processor.postprocess(decoded, output_type="pt")
            batch_prompts = list(prompts[start:stop])
            components, _ = self.scorer(
                images,
                batch_prompts,
                [{} for _ in batch_prompts],
                only_strict=True,
            )
            for name, values in components.items():
                tensor = torch.as_tensor(
                    values,
                    device=self.device,
                    dtype=torch.float32,
                )
                if tensor.shape != (stop - start,):
                    raise ValueError(
                        f"reward {name!r} returned shape {tuple(tensor.shape)}; "
                        f"expected {(stop - start,)}"
                    )
                collected.setdefault(name, []).append(tensor)

        scores = {name: torch.cat(parts) for name, parts in collected.items()}
        metrics = {
            f"rewards/{name}": float(value.detach().double().mean().item())
            for name, value in scores.items()
        }
        return scores, metrics

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError("standard image rewards have no checkpoint state")

    def on_reference_model_updated(self) -> None:
        return None
