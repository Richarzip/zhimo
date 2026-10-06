"""Evaluate local class directories without loading trained models."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation import evaluate_ensemble as evaluation


class EnsembleEvaluationTests(unittest.TestCase):
    @staticmethod
    def write_image(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), "white").save(path)

    def test_sampling_uses_custom_class_directories_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for label in ("AuthorA-楷", "AuthorB-行"):
                for number in range(8):
                    self.write_image(root / label / f"{number}.PNG")
                (root / label / "ignore.txt").write_text("not an image")
                (root / label / "ignore.png").mkdir()
            (root / "empty").mkdir()
            before = random.getstate()
            with redirect_stdout(StringIO()):
                samples = evaluation.collect_samples(3, 17, root)
                repeated = evaluation.collect_samples(3, 17, root)
            self.assertEqual(samples, repeated)
            self.assertEqual(random.getstate(), before)
            self.assertEqual(len(samples), 6)
            self.assertEqual({label for _, label in samples}, {"AuthorA", "AuthorB"})
            self.assertTrue(all(Path(path).is_file() and root in Path(path).parents for path, _ in samples))

    def test_bad_inputs_fail_before_model_loading_or_report_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            empty = root / "empty"
            empty.mkdir()
            not_directory = root / "file.txt"
            not_directory.write_text("not a directory")
            data = root / "data"
            self.write_image(data / "AuthorA" / "sample.png")
            flat = root / "flat"
            self.write_image(flat / "sample.png")
            cases = (
                (root / "missing", "1", "数据目录不存在或不是目录"),
                (not_directory, "1", "数据目录不存在或不是目录"),
                (empty, "1", "数据目录没有可评估图片"),
                (flat, "1", "数据目录没有可评估图片"),
                (data, "0", "--n 必须为正整数"),
                (data, "-1", "--n 必须为正整数"),
            )
            for path, count, message in cases:
                with self.subTest(path=path, count=count), patch.object(evaluation, "_load_models") as load:
                    error = StringIO()
                    with redirect_stderr(error), redirect_stdout(StringIO()), self.assertRaises(SystemExit) as failure:
                        evaluation.main(["--data-root", str(path), "--n", count, "--out", str(root / "report.txt")])
                    self.assertEqual(failure.exception.code, 2)
                    self.assertIn(message, error.getvalue())
                    load.assert_not_called()
                    self.assertFalse((root / "report.txt").exists())

    def test_help_preserves_existing_options_without_loading_models(self):
        output = StringIO()
        with patch.object(evaluation, "_load_models") as load, redirect_stdout(output), self.assertRaises(SystemExit) as result:
            evaluation.main(["--help"])
        self.assertEqual(result.exception.code, 0)
        for option in ("--data-root", "--n", "--seed", "--out"):
            self.assertIn(option, output.getvalue())
        load.assert_not_called()

    def test_custom_relative_directory_and_existing_flags_run_the_full_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_image(root / "data/AuthorA-楷/sample.png")
            self.write_image(root / "data/AuthorB-行/sample.png")
            convnext = types.SimpleNamespace(id_to_label={0: "AuthorA", 1: "AuthorB"},
                                            label_to_id={"AuthorA": 0, "AuthorB": 1})
            swin = types.SimpleNamespace(id_to_label=convnext.id_to_label.copy())
            transform = Mock(return_value=Mock())
            probs = [np.array([0.9, 0.1]), np.array([0.8, 0.2]), np.array([0.1, 0.9]), np.array([0.2, 0.8])]
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(evaluation, "_load_models", return_value=("cpu", convnext, swin, transform)) as load, \
                     patch.object(evaluation, "infer_probs", side_effect=probs) as infer, redirect_stdout(StringIO()):
                    evaluation.main(["--data-root", "data", "--n", "1", "--seed", "17", "--out", "report.txt"])
                report = (root / "report.txt").read_text()
            finally:
                os.chdir(original_cwd)
            load.assert_called_once()
            self.assertEqual(infer.call_count, 4)
            self.assertEqual(transform.call_count, 2)
            self.assertIn(f"数据目录: {root / 'data'}", report)
            self.assertIn("样本: 2 张 | seed: 17", report)
            self.assertEqual(report.count("1.0000 (100.00%)"), 3)


    def test_author_ids_and_chinese_directories_use_checkpoint_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_image(root / "wxz/sample.png")
            self.write_image(root / "王羲之-行/sample.png")
            recognizer = types.SimpleNamespace(id_to_label={0: "王羲之"},
                                              label_to_id={"wxz": 0, "王羲之": 0})
            report_path = root / "report.txt"
            with patch.object(evaluation, "_load_models", return_value=("cpu", recognizer, recognizer, Mock())), \
                 patch.object(evaluation, "infer_probs", return_value=np.array([1.0])), redirect_stdout(StringIO()):
                evaluation.main(["--data-root", str(root), "--n", "1", "--out", str(report_path)])
            report = report_path.read_text()
            self.assertIn("王羲之", report)
            self.assertEqual(report.count("2/2 (1.000)"), 3)
            self.assertEqual(report.count("1.0000 (100.00%)"), 3)

    def test_unknown_labels_and_misaligned_models_cannot_produce_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for case in ("unknown", "misaligned"):
                data = root / case
                self.write_image(data / ("unknown_author" if case == "unknown" else "wxz") / "sample.png")
                convnext = types.SimpleNamespace(id_to_label={0: "王羲之", 1: "颜真卿"},
                                                label_to_id={"wxz": 0, "yzq": 1})
                swin = types.SimpleNamespace(id_to_label=(convnext.id_to_label if case == "unknown" else
                                                         {0: "颜真卿", 1: "王羲之"}))
                error = StringIO()
                report_path = root / f"{case}.txt"
                with self.subTest(case=case), patch.object(
                    evaluation, "_load_models", return_value=("cpu", convnext, swin, Mock())
                ), patch.object(evaluation, "infer_probs") as infer, redirect_stdout(StringIO()), redirect_stderr(error):
                    with self.assertRaises(SystemExit) as failure:
                        evaluation.main(["--data-root", str(data), "--out", str(report_path)])
                self.assertEqual(failure.exception.code, 2)
                self.assertIn("不支持的类别" if case == "unknown" else "类别顺序不一致", error.getvalue())
                infer.assert_not_called()
                self.assertFalse(report_path.exists())


if __name__ == "__main__":
    unittest.main()
