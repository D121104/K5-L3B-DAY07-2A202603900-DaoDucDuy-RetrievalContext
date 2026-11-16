"""Build a local-only, inspectable PDF evidence inventory.

Text extraction and detected tables are heuristic; visual regions are geometric
candidates, not semantic descriptions. OCR is optional and depends on a local
Tesseract executable. All extracted/OCR content and crops require manual
validation against the PDF; this tool does not upload data or call services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pymupdf as fitz

CAPTION_RE = re.compile(r"\b(Hình|Figure|Bảng)\s*([0-9]+(?:[.\-][0-9]+)*)[^\n]*", re.IGNORECASE)
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "advanced_visual_manifest.json"


def _bbox(rect: fitz.Rect, page_rect: fitz.Rect) -> list[float]:
    """Return a clipped rectangle normalized to [0, 1] page coordinates."""
    clipped = rect & page_rect
    if clipped.is_empty or page_rect.width <= 0 or page_rect.height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round((clipped.x0 - page_rect.x0) / page_rect.width, 6),
        round((clipped.y0 - page_rect.y0) / page_rect.height, 6),
        round((clipped.x1 - page_rect.x0) / page_rect.width, 6),
        round((clipped.y1 - page_rect.y0) / page_rect.height, 6),
    ]


def _stable_id(source_sha256: str, page: int, kind: str, bbox: list[float], key: str) -> str:
    payload = json.dumps([source_sha256, page, kind, bbox, key], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _printed_label(text: str) -> str | None:
    """Detect a plausible standalone printed page number without asserting it."""
    for line in text.splitlines():
        candidate = line.strip()
        if re.fullmatch(r"(?:\d{1,4}|[ivxlcdm]{1,8})", candidate, re.IGNORECASE):
            return candidate
    return None


def _caption_matches(text: str) -> list[dict[str, str]]:
    return [{"label": match.group(0).strip(), "caption_kind": match.group(1).casefold()}
            for match in CAPTION_RE.finditer(text)]


def _caption_spans(page: fitz.Page) -> list[tuple[dict[str, str], fitz.Rect]]:
    """Return regex caption lines and their text-only PyMuPDF span rectangles."""
    result = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            for match in CAPTION_RE.finditer(line_text):
                matching_rects = []
                cursor = 0
                for span in spans:
                    span_text = span.get("text", "")
                    start, end = cursor, cursor + len(span_text)
                    if start < match.end() and end > match.start():
                        matching_rects.append(fitz.Rect(span["bbox"]))
                    cursor = end
                if matching_rects:
                    rect = matching_rects[0]
                    for other in matching_rects[1:]:
                        rect |= other
                else:
                    hits = page.search_for(match.group(0))
                    if not hits:
                        continue
                    rect = hits[0]
                result.append(({"label": match.group(0).strip(),
                                "caption_kind": match.group(1).casefold()}, rect))
    return result


def _rect_distance(a: fitz.Rect, b: fitz.Rect) -> float:
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _associate_caption(caption_rect: fitz.Rect, regions: list[tuple[str, fitz.Rect, str]]) -> tuple[int | None, float | None, bool]:
    if not regions:
        return None, None, False
    ranked = sorted(((_rect_distance(caption_rect, rect), index)
                     for index, (_, rect, _) in enumerate(regions)))
    if len(ranked) > 1 and abs(ranked[1][0] - ranked[0][0]) <= max(2.0, ranked[0][0] * 0.05):
        return None, ranked[0][0], True
    return ranked[0][1], ranked[0][0], False


def _manifest_classification(caption: str, physical_page: int, manifest: dict[str, Any]) -> str | None:
    match = CAPTION_RE.match(caption.strip())
    key = _normalize_caption_key(f"{match.group(1)}{match.group(2)}") if match else _normalize_caption_key(caption)
    entries = manifest.get("captions", {})
    entry = next((value for candidate, value in entries.items()
                  if _normalize_caption_key(candidate) == key), None)
    if not entry or entry.get("physical_page") != physical_page:
        return None
    kind = entry.get("classification")
    return kind if kind in {"chart", "figure"} else None


def _normalize_caption_key(value: str) -> str:
    match = CAPTION_RE.match(value.strip())
    if match:
        value = f"{match.group(1)}{match.group(2)}"
    return re.sub(r"\s+", "", value).casefold()


def _manifest_entry(caption: str, physical_page: int, manifest: dict[str, Any]) -> dict[str, Any] | None:
    match = CAPTION_RE.match(caption.strip())
    key = _normalize_caption_key(f"{match.group(1)}{match.group(2)}") if match else _normalize_caption_key(caption)
    for candidate, entry in manifest.get("captions", {}).items():
        if (_normalize_caption_key(candidate) == key and
                entry.get("physical_page") == physical_page):
            return entry
    return None


def _manifest_table_entry(physical_page: int, bbox: list[float],
                          manifest: dict[str, Any]) -> dict[str, Any] | None:
    return next((entry for entry in manifest.get("tables", {}).values()
                 if entry.get("physical_page") == physical_page and entry.get("bbox") == bbox), None)


def _ocr(crop: Path) -> tuple[str | None, str | None]:
    executable = shutil.which("tesseract")
    if not executable:
        return None, "tesseract_unavailable"
    try:
        result = subprocess.run([executable, str(crop), "stdout"], capture_output=True,
                                text=True, check=True, timeout=90)
        return result.stdout.strip(), None
    except (subprocess.SubprocessError, OSError) as exc:
        return None, f"ocr_error:{type(exc).__name__}"


def build_index(pdf_path: str | Path, output_dir: str | Path, dpi: int = 144,
                ocr: bool = False) -> dict[str, Any]:
    """Extract page text, best-effort tables, and locally rendered visual crops.

    Each record cites a 1-based physical page and normalized bbox. PyMuPDF's
    table and drawing detection varies by document/version, and geometric
    rectangles may include decoration or miss borderless charts. Captions and
    OCR are raw clues only; every record must be manually checked.
    """
    source = Path(pdf_path).resolve()
    out = Path(output_dir)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    records_path = out / "records.jsonl"
    counts: dict[str, int] = {"text": 0, "table": 0, "figure": 0, "chart": 0, "caption": 0}
    failures: list[dict[str, Any]] = []
    linked_caption_count = 0
    ambiguous_caption_count = 0
    ocr_inventory = {"requested": bool(ocr), "available": bool(shutil.which("tesseract")), "attempted": 0, "succeeded": 0, "failed": 0}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.is_file() else {"captions": {}}

    with fitz.open(source) as document, records_path.open("w", encoding="utf-8") as stream:
        page_count = document.page_count
        for page_index in range(document.page_count):
            page_num = page_index + 1
            try:
                page = document[page_index]
                text = page.get_text("text") or ""
                page_bbox = _bbox(page.rect, page.rect)
                text_record = {
                    "evidence_id": _stable_id(source_hash, page_num, "text", page_bbox, text),
                    "evidence_type": "text", "extraction_method": "pymupdf_page_text",
                    "text": text, "source_page": page_num, "printed_page_label": _printed_label(text),
                    "physical_pdf_page": page_num,
                    "bbox": page_bbox, "source_sha256": source_hash,
                    "crop_path": None, "regeneration_locator": {"page": page_num, "bbox": page_bbox},
                }
                stream.write(json.dumps(text_record, ensure_ascii=False) + "\n")
                counts["text"] += 1
                captions = _caption_spans(page)
                for caption, caption_rect in captions:
                    caption_bbox = _bbox(caption_rect, page.rect)
                    caption_crop = caption_rect + (-3, -3, 3, 3)
                    caption_crop_name = f"page-{page_num:03d}-caption-{_stable_id(source_hash, page_num, 'caption', caption_bbox, caption['label'])}.png"
                    caption_crop_path = crops_dir / caption_crop_name
                    try:
                        page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72),
                                        clip=caption_crop, alpha=False).save(caption_crop_path)
                    except (RuntimeError, ValueError) as exc:
                        failures.append({"source_page": page_num, "stage": "caption_crop",
                                         "bbox": caption_bbox, "error": str(exc)})
                        caption_crop_name = None
                    record = {
                        "evidence_id": _stable_id(source_hash, page_num, "caption", caption_bbox, caption["label"]),
                        "evidence_type": "caption", "extraction_method": "pymupdf_text_spans",
                        **caption, "source_page": page_num, "printed_page_label": _printed_label(text),
                        "physical_pdf_page": page_num,
                        "bbox": caption_bbox, "source_sha256": source_hash,
                        "crop_path": caption_crop_name,
                        "regeneration_locator": {"page": page_num, "bbox": caption_bbox},
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["caption"] += 1

                regions: list[tuple[str, fitz.Rect, str]] = []
                try:
                    finder = page.find_tables()
                    for table_index, table in enumerate(finder.tables):
                        table_text = "\n".join(" | ".join("" if cell is None else str(cell) for cell in row)
                                                for row in (table.extract() or []))
                        regions.append(("table", fitz.Rect(table.bbox), f"table:{table_index}:{table_text}"))
                except (AttributeError, RuntimeError, ValueError) as exc:
                    failures.append({"source_page": page_num, "stage": "table_detection", "error": str(exc)})

                for image_index, image in enumerate(page.get_image_info(xrefs=True)):
                    regions.append(("visual", fitz.Rect(image["bbox"]), f"image:{image_index}"))
                try:
                    for drawing_index, drawing in enumerate(page.get_drawings()):
                        rect = fitz.Rect(drawing["rect"])
                        if rect.width >= 24 and rect.height >= 24 and rect.width * rect.height >= 1800:
                            regions.append(("visual", rect, f"drawing:{drawing_index}"))
                except (AttributeError, RuntimeError, ValueError) as exc:
                    failures.append({"source_page": page_num, "stage": "drawing_detection", "error": str(exc)})

                caption_links = []
                curated_caption_indices: dict[str, int] = {}
                for caption_index, (caption, caption_rect) in enumerate(captions):
                    entry = _manifest_entry(caption["label"], page_num, manifest)
                    curated_bbox = entry.get("region_bbox") if entry else None
                    if curated_bbox:
                        target = fitz.Rect(*(curated_bbox[i] * (page.rect.width if i % 2 == 0 else page.rect.height)
                                             for i in range(4)))
                        key = _normalize_caption_key(caption["label"])
                        candidates = [(idx, rect) for idx, (candidate, rect) in enumerate(captions)
                                      if _manifest_entry(candidate["label"], page_num, manifest) is entry]
                        curated_caption_indices[key] = min(candidates,
                            key=lambda pair: _rect_distance(pair[1], target))[0]
                    key = _normalize_caption_key(caption["label"])
                    is_curated_caption = curated_bbox and curated_caption_indices[key] == caption_index
                    if is_curated_caption and len(curated_bbox) == 4:
                        target = fitz.Rect(*(curated_bbox[i] * (page.rect.width if i % 2 == 0 else page.rect.height)
                                             for i in range(4)))
                        region_index = next((i for i, (_, rect, _) in enumerate(regions)
                                             if _bbox(rect, page.rect) == curated_bbox), None)
                        if region_index is None:
                            regions.append(("visual", target, f"manifest:{caption['label']}"))
                            region_index = len(regions) - 1
                        distance = _rect_distance(caption_rect, target)
                        ambiguous = False
                    elif curated_bbox:
                        region_index, distance, ambiguous = None, None, False
                    else:
                        region_index, distance, ambiguous = _associate_caption(caption_rect, regions)
                    linked_caption_count += region_index is not None
                    ambiguous_caption_count += ambiguous
                    classification = (_manifest_classification(caption["label"], page_num, manifest)
                                      if not curated_bbox or is_curated_caption else None)
                    caption_links.append({"caption": caption, "region_index": region_index,
                                          "distance": distance, "ambiguous": ambiguous,
                                          "classification": classification})

                seen: set[tuple[str, tuple[float, ...]]] = set()
                for region_index, (region_type, rect, locator) in enumerate(regions):
                    norm = _bbox(rect, page.rect)
                    if norm[2] <= norm[0] or norm[3] <= norm[1]:
                        continue
                    identity = (region_type, tuple(norm))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    # Do not associate page-level captions with regions without
                    # reliable layout proximity evidence; inventory them separately.
                    linked = [link for link in caption_links if link["region_index"] == region_index]
                    manual_table = (_manifest_table_entry(page_num, norm, manifest)
                                    if region_type == "table" else None)
                    if manual_table:
                        manual_key = _normalize_caption_key(manual_table["caption"])
                        matching_caption = next(((caption, caption_rect) for caption, caption_rect in captions
                                                 if _normalize_caption_key(caption["label"]) == manual_key), None)
                        if matching_caption:
                            caption, caption_rect = matching_caption
                            linked = [{"caption": caption,
                                       "distance": _rect_distance(caption_rect, rect),
                                       "ambiguous": False}]
                    kind = "table" if region_type == "table" else next((link["classification"] for link in linked if link["classification"]), "figure")
                    crop_name = f"page-{page_num:03d}-{kind}-{_stable_id(source_hash, page_num, kind, norm, locator)}.png"
                    crop_path = crops_dir / crop_name
                    try:
                        page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=rect, alpha=False).save(crop_path)
                    except (RuntimeError, ValueError) as exc:
                        failures.append({"source_page": page_num, "stage": "crop", "bbox": norm, "error": str(exc)})
                        continue
                    extracted = locator.partition(":")[2] if region_type == "table" else None
                    rec: dict[str, Any] = {
                        "evidence_id": _stable_id(source_hash, page_num, kind, norm, locator),
                        "evidence_type": kind, "extraction_method": "pymupdf_find_tables" if kind == "table" else locator.split(":")[0],
                        "text": extracted, "caption": linked[0]["caption"]["label"] if len(linked) == 1 else None,
                        "caption_evidence_id": None, "caption_distance": linked[0]["distance"] if len(linked) == 1 else None,
                        "caption_ambiguity": any(link["ambiguous"] for link in caption_links),
                        "caption_ambiguity_reason": "equidistant_regions" if any(link["ambiguous"] for link in caption_links) else None,
                        "source_page": page_num, "printed_page_label": _printed_label(text),
                        "physical_pdf_page": page_num,
                        "bbox": norm, "source_sha256": source_hash,
                        "crop_path": str(crop_path.relative_to(out)),
                        "regeneration_locator": {"page": page_num, "bbox": norm, "dpi": dpi, "locator": locator},
                     "ocr_text": None, "ocr_error": None,
                        "source_evidence": "candidate/manual-curated" if locator.startswith("manifest:") else "candidate",
                    }
                    if len(linked) == 1:
                        cap = linked[0]["caption"]
                        cap_bbox = _bbox(next(rect for caption, rect in captions if caption == cap), page.rect)
                        rec["caption_evidence_id"] = _stable_id(source_hash, page_num, "caption", cap_bbox, cap["label"])
                        rec["caption_distance"] = round(linked[0]["distance"], 3)
                        rec["caption"] = cap["label"]
                        rec["caption_ambiguity"] = False
                        entry = _manifest_entry(cap["label"], page_num, manifest)
                        if entry and entry.get("visual_description"):
                            rec["visual_description"] = entry["visual_description"]
                            rec["description_method"] = entry.get("description_method", "manual_visual_inspection")
                            rec["description_type"] = entry.get("description_type", "visual_inspection_hint")
                    if ocr:
                        ocr_inventory["attempted"] += 1
                        rec["ocr_text"], rec["ocr_error"] = _ocr(crop_path)
                        ocr_inventory["succeeded" if rec["ocr_text"] else "failed"] += 1
                    stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts[kind] += 1
            except Exception as exc:  # Keep processing remaining pages and record the failure.
                failures.append({"source_page": page_num, "stage": "page", "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "source_pdf": str(source), "source_sha256": source_hash,
        "page_count": page_count, "dpi": dpi, "counts": counts,
        "failures": failures, "ocr": ocr_inventory, "records_path": "records.jsonl",
        "limitations": "Heuristic extraction; visual descriptions are manual inspection hints, not exhaustive interpretations. Captions and geometry are not semantic verification; manually validate evidence against the source PDF.",
        "caption_links": {"linked": linked_caption_count,
                          "unlinked": counts["caption"] - linked_caption_count,
                          "ambiguous": ambiguous_caption_count},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Input PDF; processed locally")
    parser.add_argument("--output", default="starter/results/advanced_index", help="Local output directory")
    parser.add_argument("--dpi", type=int, default=144, help="Crop render resolution")
    parser.add_argument("--ocr", action="store_true", help="Run local tesseract on visual/table crops")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    print(json.dumps(build_index(args.pdf, args.output, args.dpi, args.ocr), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
