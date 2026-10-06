import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research import baidu_knowledge_audit as audit


def verified_record(name, text):
    return {
        "name": name, "status": "identity_verified", "verified_text": text,
        "baidu_title": name,
        "fields": {"所处时代": "唐朝" if name == "颜真卿" else "东晋", "职业": "书法家",
                   "字": "清臣" if name == "颜真卿" else "逸少"},
        "abstract": "书法家。",
    }


def card_fixture(*, title="张旭", era="唐代", birth="685年？", profession="书法家", courtesy="伯高", abstract=""):
    return {
        "title": title, "abstract": abstract, "url": "https://example.test/person",
        "card": [{"name": key, "value": [value]} for key, value in {
            "所处时代": era, "出生日期": birth, "职业": profession, "字": courtesy,
        }.items() if value],
    }


class KnowledgeAuditPersistenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "data.py"
        self.original = [
            {"name": "王羲之", "text": "已有的王羲之知识"},
            {"name": "颜真卿", "text": "已有的颜真卿知识"},
        ]
        self.path.write_text("CALLIGRAPHER_KNOWLEDGE = " + repr(self.original) + "\n", encoding="utf-8")
        path_patch = patch.object(audit, "DATA_PATH", self.path)
        path_patch.start()
        self.addCleanup(path_patch.stop)

    def test_network_failure_keeps_existing_text_and_other_records(self):
        with patch.object(audit, "card", side_effect=audit.requests.RequestException("timeout")):
            result = audit.audit_one(self.original[0])
        self.assertEqual(result["status"], "request_failed")
        audit.write_data([result])
        self.assertEqual(audit.load_data(), self.original)

    def test_missing_or_empty_verified_text_keeps_existing_text(self):
        records = [
            {"name": "王羲之", "status": "exact"},
            {"name": "王羲之", "status": "exact", "verified_text": None},
            {"name": "王羲之", "status": "exact", "verified_text": ""},
            {"name": "王羲之", "status": "exact", "verified_text": "   "},
            verified_record("王羲之", ""),
            verified_record("王羲之", None),
            {"name": "王羲之", "status": "request_failed", "verified_text": "不应写入"},
        ]
        for record in records:
            with self.subTest(record=record):
                audit.write_data([record])
                self.assertEqual(audit.load_data(), self.original)

    def test_unverified_status_keeps_existing_text(self):
        for status in ["request_failed", "redirect_or_fuzzy", "identity_mismatch", "needs_manual_review", "exact", None]:
            with self.subTest(status=status):
                record = {"name": "王羲之", "verified_text": "未经核实的新文本"}
                if status is not None:
                    record["status"] = status
                audit.write_data([record])
                self.assertEqual(audit.load_data(), self.original)

    def test_success_updates_only_verified_records(self):
        verified = "核对后的文本：'引文'\n第二行"
        audit.write_data([
            {"name": "王羲之", "status": "request_failed"},
            verified_record("颜真卿", verified),
            {"name": "未有记录", "status": "request_failed"},
        ])
        self.assertEqual(audit.load_data(), [
            self.original[0], {"name": "颜真卿", "text": verified},
        ])
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_writer_rechecks_identity_instead_of_trusting_a_verified_status(self):
        record = verified_record("王羲之", "不应写入")
        record["fields"]["出生日期"] = "1961-08-04"
        audit.write_data([record])
        self.assertEqual(audit.load_data(), self.original)

    def test_replace_failure_preserves_original_file_and_cleans_temporary_file(self):
        original_bytes = self.path.read_bytes()
        with patch.object(audit.os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                audit.write_data([verified_record("王羲之", "新知识")])
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertEqual(list(self.directory.iterdir()), [self.path])


class CalligrapherIdentityTests(unittest.TestCase):
    def audit_response(self, response, name="张旭", text="已有文本《古诗四帖》"):
        with patch.object(audit, "card", return_value=response), \
             patch.object(audit, "image_for_work") as image_lookup, \
             patch.object(audit, "save_image") as download:
            result = audit.audit_one({"name": name, "text": text})
        return result, image_lookup, download

    def assert_not_publishable(self, result, image_lookup, download):
        self.assertNotEqual(result["status"], "identity_verified")
        self.assertNotIn("verified_text", result)
        self.assertEqual(result["images"], [])
        image_lookup.assert_not_called()
        download.assert_not_called()

    def test_same_name_modern_scientist_is_not_the_tang_calligrapher(self):
        result, lookup, download = self.audit_response(card_fixture(
            era="", birth="1961年8月4日", profession="教育科研工作者", courtesy="",
            abstract="张旭，神经科学家，中国科学院院士。",
        ))
        self.assertEqual(result["status"], "identity_mismatch")
        self.assert_not_publishable(result, lookup, download)
        self.assertEqual(result["candidate_works"], [])

    def test_same_name_modern_calligrapher_is_also_rejected(self):
        result, lookup, download = self.audit_response(card_fixture(
            era="现代", birth="1914年", profession="书法家", courtesy="旭初",
        ))
        self.assertEqual(result["status"], "identity_mismatch")
        self.assert_not_publishable(result, lookup, download)

    def test_explicit_birth_conflicts_win_over_a_matching_era_and_role(self):
        for birth in ("1961", "1961-08-04", "1961年8月4日"):
            with self.subTest(birth=birth):
                result, lookup, download = self.audit_response(card_fixture(birth=birth))
                self.assertEqual(result["status"], "identity_mismatch")
                self.assert_not_publishable(result, lookup, download)

    def test_true_tang_identity_allows_disputed_birth_dates_without_copying_old_works(self):
        for birth in ("658年", "675年", "685年？", ""):
            with self.subTest(birth=birth):
                result, lookup, download = self.audit_response(card_fixture(birth=birth))
                self.assertEqual(result["status"], "identity_verified")
                self.assertTrue(result["identity_check"]["verified"])
                self.assertIn("唐代", result["verified_text"])
                self.assertNotIn("古诗四帖", result["verified_text"])
                lookup.assert_not_called()
                download.assert_not_called()

    def test_title_role_era_and_disambiguator_all_need_evidence(self):
        cases = [
            (card_fixture(title="张旭（唐代书法家）"), "redirect_or_fuzzy"),
            (card_fixture(profession="学者"), "needs_manual_review"),
            (card_fixture(era="", birth=""), "needs_manual_review"),
            (card_fixture(courtesy=""), "needs_manual_review"),
            (card_fixture(era="明朝", birth=""), "identity_mismatch"),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                result, lookup, download = self.audit_response(response)
                self.assertEqual(result["status"], expected)
                self.assert_not_publishable(result, lookup, download)

    def test_admiring_tang_calligraphers_does_not_establish_the_subjects_era(self):
        result, lookup, download = self.audit_response(card_fixture(
            era="", birth="", abstract="张旭，书法家，师法唐代书法家，常临《古诗四帖》。",
        ))
        self.assertEqual(result["status"], "needs_manual_review")
        self.assert_not_publishable(result, lookup, download)

    def test_unknown_identity_is_not_approved_from_a_name_and_profession(self):
        result, lookup, download = self.audit_response(card_fixture(title="未登记人物"), name="未登记人物")
        self.assertEqual(result["status"], "needs_manual_review")
        self.assert_not_publishable(result, lookup, download)

    def test_shen_zhou_has_an_independent_ming_identity_profile(self):
        result, _, _ = self.audit_response(card_fixture(
            title="沈周", era="明朝", birth="1427年", profession="书画家", courtesy="启南",
        ), name="沈周")
        self.assertEqual(result["status"], "identity_verified")

    def test_saved_wrong_identity_is_quarantined_and_knowledge_text_is_corrected(self):
        document = json.loads((ROOT / "research" / "baidu_knowledge_audit.json").read_text(encoding="utf-8"))
        entry = next(item for item in document["entries"] if item["name"] == "张旭")
        self.assertEqual(entry["status"], "identity_mismatch")
        self.assertNotIn("verified_text", entry)
        self.assertEqual(entry["images"], [])
        self.assertIn("1961", entry["rejected_text"])
        text = next(item["text"] for item in audit.load_data() if item["name"] == "张旭")
        self.assertIn("唐代书法家", text)
        self.assertIn("伯高", text)
        self.assertNotIn("1961", text)
        self.assertNotIn("教育科研", text)
        self.assertIn("https://www.dpm.org.cn/", text)

    def test_legacy_audit_labels_no_longer_claim_that_name_equality_verifies_identity(self):
        document = json.loads((ROOT / "research" / "baidu_knowledge_audit.json").read_text(encoding="utf-8"))
        for record in document["entries"]:
            with self.subTest(name=record["name"]):
                self.assertNotEqual(record["status"], "exact")
                if record["status"] == "identity_verified":
                    self.assertTrue(audit.verify_identity(record["name"], record["baidu_title"],
                                                         record["fields"], record["abstract"])["verified"])
                else:
                    self.assertNotIn("verified_text", record)



if __name__ == "__main__":
    unittest.main()
