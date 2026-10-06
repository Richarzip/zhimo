import contextlib
import importlib.util
import io
import random
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.checkpoints import capture_rng_state, restore_rng_state, restore_training_state, save_epoch_checkpoint
from training.support import build_argument_parser, classification_metrics, create_data_loaders, normalized_confusion_matrix

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_TRAINING = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "timm", "seaborn", "sklearn"))


class TrainingArgumentTests(unittest.TestCase):
    def parser(self):
        return build_argument_parser({"max_samples": None, "batch_size": 64, "epochs": 50})

    def test_max_samples_defaults_to_unlimited_and_parses_integer(self):
        self.assertIsNone(self.parser().parse_args([]).max_samples)
        parsed = self.parser().parse_args(["--max_samples", "100"])
        self.assertEqual(parsed.max_samples, 100)
        self.assertIsInstance(parsed.max_samples, int)

    def test_max_samples_rejects_nonpositive_and_noninteger_values(self):
        for value in ("0", "-1", "1.5", "abc"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    self.parser().parse_args(["--max_samples", value])
                self.assertEqual(exc.exception.code, 2)

    def test_cli_rejects_batch_size_one(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parser().parse_args(["--batch_size", "1"])


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for training regression tests")
class TrainingLoaderTests(unittest.TestCase):
    @staticmethod
    def dataset(size):
        import torch
        return torch.utils.data.TensorDataset(torch.randn(size, 4), torch.zeros(size, dtype=torch.long))

    def loaders(self, size, batch_size=64, val_size=1):
        return create_data_loaders(self.dataset(size), self.dataset(val_size), batch_size, num_workers=0, pin_memory=False)

    def test_singleton_tail_is_dropped_and_real_batchnorm_can_train(self):
        import torch
        train_loader, val_loader = self.loaders(65)
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 2))
        sizes = []
        for images, labels in train_loader:
            torch.nn.functional.cross_entropy(model(images), labels).backward()
            sizes.append(len(images))
        self.assertEqual(sizes, [64])
        self.assertEqual([len(images) for images, _ in val_loader], [1])

    def test_non_singleton_partial_batch_and_small_dataset_are_kept(self):
        for size, expected in ((66, [64, 2]), (2, [2]), (64, [64])):
            with self.subTest(size=size):
                loader, _ = self.loaders(size)
                self.assertEqual([len(images) for images, _ in loader], expected)

    def test_training_with_fewer_than_two_samples_is_rejected(self):
        for size in (0, 1):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "at least 2 samples"):
                self.loaders(size)

    def test_programmatic_batch_size_one_and_empty_validation_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be at least 2"):
            self.loaders(4, batch_size=1)
        with self.assertRaisesRegex(ValueError, "Validation dataset"):
            self.loaders(4, val_size=0)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for checkpoint regression tests")
class TrainingRecoveryTests(unittest.TestCase):
    def components(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [5])
        return model, optimizer, scheduler

    def step(self, model, optimizer, scheduler):
        import numpy as np
        import torch
        images = torch.randn(8, 4) * (random.random() + np.random.random())
        targets = torch.randn(8, 2)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(images), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.item()

    def test_weights_only_checkpoint_restores_identical_next_update(self):
        import numpy as np
        import torch
        model, optimizer, scheduler = self.components()
        for _ in range(7):
            self.step(model, optimizer, scheduler)
        training_state = {
            "best_val_loss": 0.7, "patience_counter": 2,
            "history": {"val_losses": [0.9, 0.7, 0.8, 0.85]},
            "best_all_preds": [0, 1], "best_all_labels": [0, 1],
        }
        with tempfile.TemporaryDirectory() as directory:
            save_epoch_checkpoint({
                "epoch": 6, "val_acc": 0.8,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng_state": capture_rng_state(), "training_state": training_state,
            }, 0.9, directory, "model", save=torch.save)
            checkpoint = torch.load(Path(directory) / "model_e7.pth", map_location="cpu", weights_only=True)
        old_lr = optimizer.param_groups[0]["lr"]
        expected_random = (random.random(), np.random.random(), torch.rand(3))
        expected_loss = self.step(model, optimizer, scheduler)
        resumed_model, resumed_optimizer, resumed_scheduler = self.components()
        restored = restore_training_state(checkpoint, resumed_model, resumed_optimizer, resumed_scheduler)
        self.assertEqual(restored, training_state)
        self.assertEqual(resumed_optimizer.param_groups[0]["lr"], old_lr)
        self.assertTrue(resumed_optimizer.state)
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(np.random.random(), expected_random[1])
        torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
        actual_loss = self.step(resumed_model, resumed_optimizer, resumed_scheduler)
        self.assertEqual(actual_loss, expected_loss)
        for key, expected in model.state_dict().items():
            torch.testing.assert_close(resumed_model.state_dict()[key], expected, rtol=0, atol=0)
        self.assertEqual(resumed_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
        self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
        for expected_parameter, actual_parameter in zip(model.parameters(), resumed_model.parameters()):
            expected_state = optimizer.state[expected_parameter]
            actual_state = resumed_optimizer.state[actual_parameter]
            self.assertEqual(set(expected_state), set(actual_state))
            for name, expected in expected_state.items():
                torch.testing.assert_close(actual_state[name], expected, rtol=0, atol=0)

    def test_cpu_fallback_rng_does_not_initialize_available_but_unusable_cuda(self):
        import torch
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "is_initialized", return_value=False), patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA must remain uninitialized")), patch.object(torch.cuda, "get_rng_state_all") as get_cuda, patch.object(torch.cuda, "set_rng_state_all") as set_cuda:
            state = capture_rng_state()
            self.assertNotIn("cuda", state)
            state["cuda"] = [torch.get_rng_state()]
            restore_rng_state(state)
            get_cuda.assert_not_called()
            set_cuda.assert_not_called()

    def test_missing_optimizer_or_scheduler_is_an_explicit_error(self):
        model, optimizer, scheduler = self.components()
        checkpoint = {"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
        for missing in ("optimizer", "scheduler"):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "missing " + missing):
                restore_training_state({key: value for key, value in checkpoint.items() if key != missing}, model, optimizer, scheduler)

    def test_legacy_checkpoint_restores_optimization_but_warns_about_missing_state(self):
        model, optimizer, scheduler = self.components()
        self.step(model, optimizer, scheduler)
        checkpoint = {"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
        resumed_model, resumed_optimizer, resumed_scheduler = self.components()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored = restore_training_state(checkpoint, resumed_model, resumed_optimizer, resumed_scheduler)
        self.assertEqual(restored, {})
        self.assertTrue(resumed_optimizer.state)
        self.assertEqual(resumed_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
        self.assertTrue(any("no RNG state" in str(item.message) for item in caught))
        self.assertTrue(any("no training history" in str(item.message) for item in caught))


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required for metric regression tests")
class TrainingMetricTests(unittest.TestCase):
    def test_missing_classes_still_produce_all_54_named_metrics(self):
        labels = {index: f"author_{index}" for index in range(54)}
        result = classification_metrics([0, 0], [0, 0], labels)
        for name in labels.values():
            self.assertIn(name, result)
        self.assertEqual(result["author_0"]["recall"], 1.0)
        self.assertEqual(result["author_53"]["support"], 0.0)

    def test_confusion_matrix_keeps_label_positions_and_zero_rows(self):
        import numpy as np
        matrix = normalized_confusion_matrix([0, 0], [0, 2], range(54))
        self.assertEqual(matrix.shape, (54, 54))
        self.assertEqual(matrix[0, 0], 0.5)
        self.assertEqual(matrix[0, 2], 0.5)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue((matrix[1:] == 0).all())

    def test_empty_confusion_matrix_has_no_nan(self):
        import numpy as np
        matrix = normalized_confusion_matrix([], [], range(3))
        self.assertTrue(np.array_equal(matrix, np.zeros((3, 3))))


@unittest.skipUnless(HAS_TRAINING, "Complete training dependencies are required for entrypoint tests")
class TrainingEntrypointTests(unittest.TestCase):
    def test_parsed_max_samples_limits_a_real_temporary_image_dataset(self):
        from PIL import Image
        from training import train
        args = build_argument_parser(train.CONFIG).parse_args(["--max_samples", "2"])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            author_directory = Path(directory) / "train" / "wxz"
            author_directory.mkdir(parents=True)
            for index in range(3):
                Image.new("RGB", (8, 8)).save(author_directory / f"{index}.png")
            dataset = train.ArchiveDataset(directory, {"wxz": "王羲之"}, {}, max_samples=args.max_samples)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0][0].size, (8, 8))
            with self.assertRaisesRegex(ValueError, "max_samples"):
                train.ArchiveDataset(directory, {"wxz": "王羲之"}, {}, max_samples=0)

    def test_real_validate_and_plot_accept_missing_classes(self):
        import numpy as np
        import torch
        from training import train
        model = torch.nn.Linear(4, 54)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(2, 4), torch.zeros(2, dtype=torch.long)))
        with contextlib.redirect_stderr(io.StringIO()):
            _, accuracy, report, preds, labels = train.validate(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), {index: f"author_{index}" for index in range(54)})
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(report["author_53"]["support"], 0.0)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "matrix.png"
            with patch.object(train.sns, "heatmap", wraps=train.sns.heatmap) as heatmap:
                train.plot_confusion_matrix(labels, preds, ["one", "two", "three"], path)
            matrix = heatmap.call_args.args[0]
            self.assertEqual(matrix.shape, (3, 3))
            self.assertTrue(np.isfinite(matrix).all())
            self.assertTrue((matrix[1:] == 0).all())
            self.assertGreater(path.stat().st_size, 0)

    def test_missing_resume_path_fails_before_model_initialization(self):
        from training import train
        with tempfile.TemporaryDirectory() as directory, patch.object(train, "build_model") as build_model, patch.object(sys, "argv", ["train.py", "--resume", str(Path(directory) / "missing.pth")]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                train.main()
            build_model.assert_not_called()

    def test_main_persists_and_restores_early_stopping_and_history(self):
        import torch
        from training import train

        class Dataset(torch.utils.data.TensorDataset):
            id_to_label = {0: "one", 1: "two"}
            label_to_id = {"one": 0, "two": 1}

            def __init__(self, **kwargs):
                super().__init__(torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))

        def train_epoch(model, loader, criterion, optimizer, device, epoch):
            optimizer.step()
            return 0.5, 0.75

        def validation(loss, accuracy):
            return loss, accuracy, {"accuracy": accuracy}, [0, 1], [0, 1]

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            argv = ["train.py", "--data_root", directory, "--output_dir", directory, "--epochs", "6", "--batch_size", "2", "--early_stop_patience", "2"]
            with patch.object(train, "ArchiveDataset", Dataset), patch.object(train, "build_model", side_effect=lambda **kwargs: torch.nn.Linear(4, 2)), patch.object(train, "train_one_epoch", side_effect=train_epoch), patch.object(train, "plot_loss_acc_curve") as curves, patch.object(train, "plot_confusion_matrix") as matrix:
                with patch.object(sys, "argv", argv), patch.object(train, "validate", side_effect=[validation(1.0, 0.9), validation(0.9, 0.8), validation(0.95, 0.8), validation(0.96, 0.8)]):
                    train.main()
                checkpoint_path = Path(directory) / "calligrapher_classifier_e4.pth"
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                self.assertEqual(checkpoint["training_state"]["patience_counter"], 2)
                self.assertEqual(checkpoint["training_state"]["best_val_loss"], 0.9)
                self.assertEqual(len(checkpoint["training_state"]["history"]["train_losses"]), 4)
                best_path = Path(directory) / "calligrapher_classifier.pth"
                best_bytes = best_path.read_bytes()
                argv[-1] = "3"
                with patch.object(sys, "argv", argv + ["--resume", str(checkpoint_path)]), patch.object(train, "validate", return_value=validation(0.97, 0.85)) as validate:
                    train.main()
                validate.assert_called_once()
                resumed = torch.load(Path(directory) / "calligrapher_classifier_e5.pth", weights_only=True)
                self.assertEqual(resumed["training_state"]["patience_counter"], 3)
                self.assertEqual(len(resumed["training_state"]["history"]["train_losses"]), 5)
                self.assertEqual(len(curves.call_args.args[0]), 5)
                self.assertEqual(matrix.call_args.args[:2], ([0, 1], [0, 1]))
                self.assertEqual(best_path.read_bytes(), best_bytes)


if __name__ == "__main__":
    unittest.main()
