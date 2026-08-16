import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from flow_grpo.baseline_lora import merge_frozen_baseline_lora, resolve_baseline_lora_path, validate_no_cfg


class BaselineLoraTests(unittest.TestCase):
    def make_adapter(self, root, nested=False):
        adapter_dir = Path(root) / "lora" if nested else Path(root)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
        return adapter_dir

    def test_resolves_adapter_or_parent_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = self.make_adapter(tmp, nested=True)
            self.assertEqual(resolve_baseline_lora_path(tmp), adapter_dir.resolve())
            self.assertEqual(resolve_baseline_lora_path(adapter_dir), adapter_dir.resolve())

    def test_unset_path_preserves_plain_transformer(self):
        transformer = Mock()
        result, resolved = merge_frozen_baseline_lora(transformer, None)
        self.assertIs(result, transformer)
        self.assertIsNone(resolved)

    def test_missing_adapter_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "adapter config"):
                resolve_baseline_lora_path(tmp)

    @patch("flow_grpo.baseline_lora.PeftModel.from_pretrained")
    def test_adapter_is_merged_and_frozen(self, from_pretrained):
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = self.make_adapter(tmp)
            transformer = Mock()
            merged = Mock()
            peft_model = Mock()
            peft_model.merge_and_unload.return_value = merged
            from_pretrained.return_value = peft_model

            result, resolved = merge_frozen_baseline_lora(transformer, tmp)

            self.assertIs(result, merged)
            self.assertEqual(resolved, adapter_dir.resolve())
            from_pretrained.assert_called_once_with(transformer, str(adapter_dir.resolve()), is_trainable=False)
            peft_model.merge_and_unload.assert_called_once_with(safe_merge=True)
            merged.requires_grad_.assert_called_once_with(False)

    def test_cfg_distilled_baseline_requires_guidance_one(self):
        validate_no_cfg({"train": 1, "eval": 1.0})
        with self.assertRaisesRegex(ValueError, "expected scale 1.0"):
            validate_no_cfg({"train": 1.0, "eval": 4.5})


if __name__ == "__main__":
    unittest.main()
