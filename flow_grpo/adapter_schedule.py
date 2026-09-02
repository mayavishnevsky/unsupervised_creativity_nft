"""LoRA strength schedules used by creative-probe denoising."""

from __future__ import annotations


def linear_decay_strengths(
    total_steps: int,
    full_strength_steps: int,
    zero_strength_steps: int,
) -> tuple[float, ...]:
    """Return full, linearly decaying, then zero adapter strengths."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if full_strength_steps < 0 or zero_strength_steps < 0:
        raise ValueError("full_strength_steps and zero_strength_steps must be nonnegative")
    if full_strength_steps + zero_strength_steps > total_steps:
        raise ValueError(
            "full_strength_steps + zero_strength_steps must not exceed "
            f"total_steps ({total_steps})"
        )

    transition_steps = total_steps - full_strength_steps - zero_strength_steps
    transition = tuple(
        (transition_steps - offset) / (transition_steps + 1)
        for offset in range(transition_steps)
    )
    return (
        (1.0,) * full_strength_steps
        + transition
        + (0.0,) * zero_strength_steps
    )


class LoraAdapterScaler:
    """Temporarily scale one PEFT LoRA adapter without affecting merged LoRAs."""

    def __init__(self, model, adapter: str):
        self.adapter = adapter
        self._layers = []
        for module in model.modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict) and adapter in scaling:
                self._layers.append((module, scaling[adapter]))
        if not self._layers:
            raise ValueError(f"model has no LoRA layers for adapter {adapter!r}")

    def set_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"adapter strength must be in [0, 1], got {strength}")
        for module, original_scale in self._layers:
            module.scaling[self.adapter] = original_scale * strength

    def restore(self) -> None:
        for module, original_scale in self._layers:
            module.scaling[self.adapter] = original_scale
