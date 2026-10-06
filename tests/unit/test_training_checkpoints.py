import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.checkpoints import restore_best_accuracy, save_epoch_checkpoint


class TrainingCheckpointTests(unittest.TestCase):
    name = "calligrapher_classifier"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.best_path = self.directory / f"{self.name}.pth"

    @staticmethod
    def save(checkpoint, path):
        Path(path).write_bytes(pickle.dumps(checkpoint))

    @staticmethod
    def load(path):
        return pickle.loads(Path(path).read_bytes())

    @staticmethod
    def checkpoint(epoch, val_acc):
        return {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state_dict": {"epoch_marker": epoch},
            "backbone": "convnext_tiny",
            "num_classes": 2,
            "id_to_label": {0: "王羲之", 1: "颜真卿"},
        }

    def save_epoch(self, epoch, val_acc, best_acc):
        return save_epoch_checkpoint(
            self.checkpoint(epoch, val_acc), best_acc, self.directory,
            self.name, save=self.save,
        )

    def test_non_best_epoch_keeps_historical_best_and_current_accuracy(self):
        best_acc, improved = self.save_epoch(0, 0.9, 0.0)
        self.assertTrue(improved)
        original_best = self.best_path.read_bytes()
        best_acc, improved = self.save_epoch(1, 0.8, best_acc)
        self.assertFalse(improved)
        self.assertEqual(best_acc, 0.9)
        checkpoint = self.load(self.directory / f"{self.name}_e2.pth")
        self.assertEqual(checkpoint["best_acc"], 0.9)
        self.assertEqual(checkpoint["val_acc"], 0.8)
        self.assertEqual(self.best_path.read_bytes(), original_best)

    def test_resuming_legacy_epoch_preserves_existing_best_until_improvement(self):
        historical = {**self.checkpoint(0, 0.9), "best_acc": 0.9}
        historical.pop("val_acc")
        legacy_epoch = {**self.checkpoint(1, 0.8), "best_acc": 0.8}
        legacy_epoch.pop("val_acc")
        self.save(historical, self.best_path)
        original_best = self.best_path.read_bytes()
        best_acc = restore_best_accuracy(legacy_epoch, self.best_path, load=self.load)
        self.assertEqual(best_acc, 0.9)
        best_acc, improved = self.save_epoch(2, 0.85, best_acc)
        self.assertFalse(improved)
        self.assertEqual(self.best_path.read_bytes(), original_best)
        best_acc, improved = self.save_epoch(3, 0.95, best_acc)
        self.assertTrue(improved)
        self.assertEqual(best_acc, 0.95)
        saved = self.load(self.best_path)
        self.assertEqual(saved["model_state_dict"]["epoch_marker"], 3)
        self.assertEqual(saved["best_acc"], 0.95)
        self.assertEqual(saved["val_acc"], 0.95)

    def test_restores_recorded_best_when_no_best_file_exists(self):
        checkpoint = {**self.checkpoint(2, 0.8), "best_acc": 0.9}
        load = Mock()
        self.assertEqual(restore_best_accuracy(checkpoint, self.best_path, load=load), 0.9)
        load.assert_not_called()

    def test_different_model_or_class_mapping_does_not_change_resume_metric(self):
        checkpoint = {**self.checkpoint(2, 0.8), "best_acc": 0.9}
        differences = [
            {"backbone": "swin_tiny"},
            {"num_classes": 3},
            {"id_to_label": {0: "颜真卿", 1: "王羲之"}},
        ]
        for difference in differences:
            with self.subTest(difference=difference):
                self.save({**checkpoint, **difference, "best_acc": 0.99}, self.best_path)
                self.assertEqual(restore_best_accuracy(checkpoint, self.best_path, load=self.load), 0.9)

    def test_equal_accuracy_does_not_overwrite_best_model(self):
        best_acc, _ = self.save_epoch(0, 0.9, 0.0)
        original_best = self.best_path.read_bytes()
        _, improved = self.save_epoch(1, 0.9, best_acc)
        self.assertFalse(improved)
        self.assertEqual(self.best_path.read_bytes(), original_best)


if __name__ == "__main__":
    unittest.main()
