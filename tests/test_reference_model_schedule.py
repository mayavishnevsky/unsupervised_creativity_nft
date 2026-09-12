import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from flow_grpo.reference_model_schedule import (
    REFERENCE_ADAPTER_NAME,
    ReferenceModelUpdatePolicy,
    reference_adapter_context,
)


class FakeAdapterHost:
    def __init__(self, active=("default",)):
        self.active_adapters = list(active)
        self.disabled = False

    def set_adapter(self, adapters):
        if isinstance(adapters, str):
            adapters = [adapters]
        self.active_adapters = list(adapters)

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


class ReferenceModelUpdatePolicyTests(unittest.TestCase):
    def test_disabled_policy_preserves_baseline_and_cache_behavior(self):
        policy = ReferenceModelUpdatePolicy()

        self.assertIsNone(policy.adapter_name)
        self.assertTrue(policy.cache_allowed(100))
        self.assertFalse(policy.should_update(5, 20, 0))

    def test_enabled_policy_uses_cache_only_for_first_five_epochs(self):
        policy = ReferenceModelUpdatePolicy(enabled=True)

        self.assertEqual(policy.adapter_name, REFERENCE_ADAPTER_NAME)
        self.assertTrue(all(policy.cache_allowed(epoch) for epoch in range(5)))
        self.assertFalse(policy.cache_allowed(5))
        self.assertFalse(policy.cache_allowed(19))

    def test_twenty_epoch_schedule_updates_at_five_ten_and_fifteen(self):
        policy = ReferenceModelUpdatePolicy(enabled=True)

        self.assertEqual(policy.expected_reference_epoch(4, 20), 0)
        self.assertEqual(policy.expected_reference_epoch(5, 20), 5)
        self.assertEqual(policy.expected_reference_epoch(10, 20), 10)
        self.assertEqual(policy.expected_reference_epoch(15, 20), 15)
        self.assertEqual(policy.expected_reference_epoch(20, 20), 15)

    def test_final_update_requires_three_remaining_epochs(self):
        policy = ReferenceModelUpdatePolicy(enabled=True)

        self.assertFalse(policy.should_update(15, 17, 10))
        self.assertTrue(policy.should_update(15, 18, 10))

    def test_config_defaults_are_applied_when_block_is_missing(self):
        self.assertEqual(
            ReferenceModelUpdatePolicy.from_config(SimpleNamespace()),
            ReferenceModelUpdatePolicy(),
        )

    def test_config_values_are_read(self):
        config = SimpleNamespace(
            reference_model_update=SimpleNamespace(
                enabled=True,
                interval_epochs=3,
                min_remaining_epochs=2,
            )
        )

        self.assertEqual(
            ReferenceModelUpdatePolicy.from_config(config),
            ReferenceModelUpdatePolicy(
                enabled=True,
                interval_epochs=3,
                min_remaining_epochs=2,
            ),
        )


class ReferenceAdapterContextTests(unittest.TestCase):
    def test_disabled_mode_uses_disable_adapter_context(self):
        host = FakeAdapterHost()

        with reference_adapter_context(host, None):
            self.assertTrue(host.disabled)
            self.assertEqual(host.active_adapters, ["default"])

        self.assertFalse(host.disabled)
        self.assertEqual(host.active_adapters, ["default"])

    def test_enabled_mode_selects_and_restores_adapter_set(self):
        host = FakeAdapterHost(("default", "auxiliary"))


        with reference_adapter_context(host, REFERENCE_ADAPTER_NAME):
            self.assertFalse(host.disabled)
            self.assertEqual(host.active_adapters, [REFERENCE_ADAPTER_NAME])

        self.assertEqual(host.active_adapters, ["default", "auxiliary"])


if __name__ == "__main__":
    unittest.main()
