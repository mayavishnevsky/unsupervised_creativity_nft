"""Epoch-boundary policy for optional rolling reference-model snapshots."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


REFERENCE_ADAPTER_NAME = "reference"


def _active_adapter_names(host) -> list[str]:
    active = getattr(host, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        active = getattr(host, "active_adapter", None)
    if isinstance(active, str):
        return [active]
    if active:
        return list(active)
    raise RuntimeError("cannot determine the active PEFT adapter")


@contextmanager
def reference_adapter_context(host, adapter_name: str | None) -> Iterator[None]:
    """Select the frozen rolling adapter, or the merged baseline when disabled."""

    if adapter_name is None:
        with host.disable_adapter():
            yield
        return

    previous = _active_adapter_names(host)
    host.set_adapter(adapter_name)
    try:
        yield
    finally:
        host.set_adapter(previous[0] if len(previous) == 1 else previous)


@dataclass(frozen=True)
class ReferenceModelUpdatePolicy:
    """Decide when the frozen creative-reference LoRA is replaced."""

    enabled: bool = False
    interval_epochs: int = 5
    min_remaining_epochs: int = 3

    def __post_init__(self) -> None:
        if self.interval_epochs < 1:
            raise ValueError("reference-model update interval must be positive")
        if self.min_remaining_epochs < 0:
            raise ValueError(
                "reference-model minimum remaining epochs must be nonnegative"
            )

    @classmethod
    def from_config(cls, creativity_config) -> "ReferenceModelUpdatePolicy":
        update_config = getattr(
            creativity_config,
            "reference_model_update",
            None,
        )
        if update_config is None:
            return cls()
        return cls(
            enabled=bool(getattr(update_config, "enabled", False)),
            interval_epochs=int(getattr(update_config, "interval_epochs", 5)),
            min_remaining_epochs=int(
                getattr(update_config, "min_remaining_epochs", 3)
            ),
        )

    @property
    def adapter_name(self) -> str | None:
        return REFERENCE_ADAPTER_NAME if self.enabled else None

    def cache_allowed(self, epoch: int) -> bool:
        """Allow the baseline cache only before the first model replacement."""

        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        return not self.enabled or epoch < self.interval_epochs

    def should_update(
        self,
        completed_epochs: int,
        total_epochs: int | None,
        current_reference_epoch: int,
    ) -> bool:
        """Return whether a snapshot is due after completed_epochs."""

        completed_epochs = int(completed_epochs)
        current_reference_epoch = int(current_reference_epoch)
        if completed_epochs < 0 or current_reference_epoch < 0:
            raise ValueError("epoch counters must be nonnegative")
        if not self.enabled or completed_epochs == 0:
            return False
        if completed_epochs % self.interval_epochs:
            return False
        if completed_epochs <= current_reference_epoch:
            return False
        if total_epochs is None:
            return True
        total_epochs = int(total_epochs)
        if total_epochs < completed_epochs:
            raise ValueError("total epochs cannot precede completed epochs")
        return total_epochs - completed_epochs >= self.min_remaining_epochs

    def expected_reference_epoch(
        self,
        next_epoch: int,
        total_epochs: int | None,
    ) -> int:
        """Return the snapshot epoch expected in a boundary checkpoint."""

        next_epoch = int(next_epoch)
        if next_epoch < 0:
            raise ValueError("next epoch must be nonnegative")
        reference_epoch = 0
        for completed_epochs in range(
            self.interval_epochs,
            next_epoch + 1,
            self.interval_epochs,
        ):
            if self.should_update(
                completed_epochs,
                total_epochs,
                reference_epoch,
            ):
                reference_epoch = completed_epochs
        return reference_epoch
