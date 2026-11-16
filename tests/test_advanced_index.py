"""Offline tests for the local PDF evidence indexer."""
import json
import tempfile
import unittest
from pathlib import Path

import pymupdf as fitz

from starter.advanced_index import (_associate_caption, _bbox, _manifest_classification,
                                    _manifest_table_entry, _stable_id, build_index)


class AdvancedIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pdf = self.root / "synthetic.pdf"
        doc = fitz.open()
        page = doc.new_page(width=300, height=400)
        page.insert_text((40, 40), "Synthetic logistics page\nFigure 1. Test chart\n17")
        page.draw_rect(fitz.Rect(40, 80, 240, 220), color=(0, 0, 0), width=2)
        page.draw_line((50, 200), (220, 100), color=(1, 0, 0), width=3)
        page = doc.new_page(width=300, height=400)
        page.insert_text((40, 40), "")
        doc.save(self.pdf)
        doc.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_page_text_caption_and_empty_page_records(self):
        summary = build_index(self.pdf, self.root / "out", dpi=72)
        records = [json.loads(line) for line in (self.root / "out/records.jsonl").read_text().splitlines()]
        text_records = [r for r in records if r["evidence_type"] == "text"]
        captions = [r for r in records if r["evidence_type"] == "caption"]
        self.assertEqual(summary["page_count"], 2)
        self.assertEqual([r["source_page"] for r in text_records], [1, 2])
        self.assertIn("Synthetic logistics", text_records[0]["text"])
        self.assertEqual(text_records[0]["printed_page_label"], "17")
        self.assertEqual(captions[0]["source_page"], 1)
        self.assertEqual(captions[0]["caption_kind"], "figure")
        self.assertLess(captions[0]["bbox"][3] - captions[0]["bbox"][1], 0.1)
        self.assertEqual(captions[0]["physical_pdf_page"], 1)
        self.assertEqual(text_records[1]["text"], "")

    def test_visual_bbox_crop_and_stable_ids(self):
        build_index(self.pdf, self.root / "out", dpi=72)
        records = [json.loads(line) for line in (self.root / "out/records.jsonl").read_text().splitlines()]
        visuals = [r for r in records if r["evidence_type"] in ("figure", "chart")]
        self.assertTrue(visuals)
        for record in visuals:
            self.assertEqual(len(record["bbox"]), 4)
            x0, y0, x1, y1 = record["bbox"]
            self.assertTrue(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1)
            self.assertTrue((self.root / "out" / record["crop_path"]).is_file())
            self.assertEqual(record["evidence_id"], _stable_id(record["source_sha256"], record["source_page"], record["evidence_type"], record["bbox"], record["regeneration_locator"]["locator"]))
        self.assertEqual(_bbox(fitz.Rect(10, 20, 30, 40), fitz.Rect(0, 0, 100, 100)), [0.1, 0.2, 0.3, 0.4])

    def test_table_detection_is_best_effort_and_no_ocr_by_default(self):
        summary = build_index(self.pdf, self.root / "out", dpi=72)
        self.assertIn("table", summary["counts"])
        self.assertEqual(summary["ocr"]["attempted"], 0)
        self.assertEqual(summary["failures"], [])

    def test_caption_associates_to_nearest_region_and_reports_tie(self):
        caption = fitz.Rect(45, 45, 100, 60)
        regions = [("visual", fitz.Rect(40, 80, 120, 150), "near"),
                   ("visual", fitz.Rect(200, 80, 280, 150), "far")]
        index, distance, ambiguous = _associate_caption(caption, regions)
        self.assertEqual(index, 0)
        self.assertEqual(distance, 20)
        self.assertFalse(ambiguous)
        tied = [("visual", fitz.Rect(0, 80, 40, 100), "left"),
                ("visual", fitz.Rect(110, 80, 150, 100), "right")]
        index, distance, ambiguous = _associate_caption(fitz.Rect(55, 20, 95, 40), tied)
        self.assertIsNone(index)
        self.assertTrue(ambiguous)
        self.assertIsNotNone(distance)

    def test_manifest_classification_normalizes_caption_whitespace_and_page(self):
        manifest = {"captions": {"Hình1.2": {"physical_page": 18, "classification": "chart"},
                                 "Hình2.1": {"physical_page": 43, "classification": "figure"}}}
        self.assertEqual(_manifest_classification("Hình 1.2: doanh thu", 18, manifest), "chart")
        manifest["captions"]["Hình 3.1"] = {"physical_page": 65, "classification": "chart"}
        self.assertEqual(_manifest_classification("Hình3.1 biểu đồ", 65, manifest), "chart")
        self.assertEqual(_manifest_classification("Hình2.1 sơ đồ", 43, manifest), "figure")
        self.assertIsNone(_manifest_classification("Hình 1.2", 19, manifest))

    def test_manual_table_manifest_requires_exact_page_and_bbox(self):
        entry = {"physical_page": 32, "caption": "Bảng 1.5",
                 "bbox": [0.121981, 0.217674, 0.894737, 0.356773]}
        manifest = {"tables": {"Bảng 1.5": entry}}
        self.assertIs(_manifest_table_entry(32, entry["bbox"], manifest), entry)
        self.assertIsNone(_manifest_table_entry(33, entry["bbox"], manifest))
        self.assertIsNone(_manifest_table_entry(32, [0.121981, 0.217674, 0.894737, 0.356772], manifest))

    def test_manifest_region_bbox_overrides_nearest_drawing(self):
        manifest_path = __import__("starter.advanced_index", fromlist=["MANIFEST_PATH"]).MANIFEST_PATH
        original = manifest_path.read_text(encoding="utf-8")
        try:
            manifest_path.write_text(json.dumps({"captions": {
                "Figure 1.": {"physical_page": 1, "classification": "chart",
                               "region_bbox": [0.5, 0.5, 0.9, 0.9],
                               "visual_description": "A locally verified synthetic chart hint.",
                               "description_method": "manual_visual_inspection",
                               "description_type": "visual_inspection_hint"}
            }}), encoding="utf-8")
            build_index(self.pdf, self.root / "curated", dpi=72)
            records = [json.loads(line) for line in (self.root / "curated/records.jsonl").read_text().splitlines()]
            chart = next(r for r in records if r["evidence_type"] == "chart")
            self.assertEqual(chart["bbox"], [0.5, 0.5, 0.9, 0.9])
            self.assertEqual(chart["caption"], "Figure 1. Test chart")
            self.assertEqual(chart["source_evidence"], "candidate/manual-curated")
            self.assertEqual(chart["visual_description"], "A locally verified synthetic chart hint.")
            self.assertEqual(chart["description_method"], "manual_visual_inspection")
            self.assertEqual(chart["description_type"], "visual_inspection_hint")
        finally:
            manifest_path.write_text(original, encoding="utf-8")

    def test_manifest_caption_mapping_preserves_matching_detected_locator(self):
        manifest_path = __import__("starter.advanced_index", fromlist=["MANIFEST_PATH"]).MANIFEST_PATH
        original = manifest_path.read_text(encoding="utf-8")
        try:
            manifest_path.write_text(json.dumps({"captions": {
                "Figure 1.": {"physical_page": 1, "classification": "figure",
                               "region_bbox": [0.133333, 0.2, 0.8, 0.55],
                               "visual_description": "A reviewed candidate map hint.",
                               "description_method": "manual_visual_inspection",
                               "description_type": "visual_inspection_hint"}
            }}), encoding="utf-8")
            build_index(self.pdf, self.root / "matched", dpi=72)
            records = [json.loads(line) for line in (self.root / "matched/records.jsonl").read_text().splitlines()]
            visual = next(r for r in records if r["evidence_type"] == "figure")
            self.assertEqual(visual["bbox"], [0.133333, 0.2, 0.8, 0.55])
            self.assertEqual(visual["regeneration_locator"]["locator"], "drawing:0")
            self.assertFalse(visual["regeneration_locator"]["locator"].startswith("manifest:"))
            self.assertEqual(visual["source_evidence"], "candidate")
        finally:
            manifest_path.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
