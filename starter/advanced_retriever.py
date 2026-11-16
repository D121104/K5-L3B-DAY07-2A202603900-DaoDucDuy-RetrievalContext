"""Local BM25 retrieval over the generated advanced evidence inventory."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from baseline import BM25Index
except ModuleNotFoundError:
    from starter.baseline import BM25Index

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / 'starter' / 'results' / 'advanced_index' / 'records.jsonl'
SOURCE_ID = 'logistics-vietnam-2024'
_REFERENCE_RE = re.compile(r'(?<!\w)(Bảng|Table|Hình|Figure)\s*([0-9]+(?:[.-][0-9]+)*)(?![\w-]|\.[0-9])', re.IGNORECASE)


def _caption_reference(value: str) -> tuple[str, str] | None:
    match = _REFERENCE_RE.match(value.strip())
    if not match:
        return None
    kind = 'table' if match.group(1).casefold() in ('bảng', 'table') else 'figure'
    number = re.sub(r'\s+', '', match.group(2))
    return kind, number


def _record_text(record: dict[str, Any]) -> str:
    parts = [f"Evidence type: {record.get('evidence_type', 'unknown')}", record.get('text'),
             record.get('caption'), record.get('manual_visual_description'),
             record.get('visual_description'), record.get('ocr_text')]
    return '\n'.join(dict.fromkeys(part.strip() for part in parts
                                   if isinstance(part, str) and part.strip()))


def _is_indexable(record: dict[str, Any]) -> bool:
    kind = record.get('evidence_type')
    if kind in ('text', 'table'):
        return True
    if kind not in ('figure', 'chart'):
        return False
    # OCR enriches a selected visual record but is too noisy by itself to
    # distinguish charts from the indexer's many decorative image fragments.
    return any(isinstance(record.get(key), str) and record[key].strip()
               for key in ('caption', 'manual_visual_description', 'visual_description'))


def map_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    """Normalize one source record into the agent retriever result schema."""
    evidence_id = record.get('evidence_id')
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError('advanced index record requires a non-empty evidence_id')
    metadata = {key: record.get(key) for key in (
        'evidence_type', 'bbox', 'crop_path', 'source_sha256', 'printed_page_label',
        'caption', 'manual_visual_description', 'visual_description', 'ocr_text',
        'caption_evidence_id', 'regeneration_locator') if key in record}
    metadata['sha'] = record.get('source_sha256')
    metadata['printed_page'] = record.get('printed_page_label')
    metadata['physical_page'] = record.get('physical_pdf_page', record.get('source_page'))
    metadata['anchor'] = {
        'bbox': record.get('bbox'), 'crop_path': record.get('crop_path'),
        'physical_pdf_page': record.get('physical_pdf_page', record.get('source_page')),
    }
    metadata['region'] = metadata['anchor']
    return {'id': evidence_id, 'text': _record_text(record), 'source_id': SOURCE_ID,
            'source': source, 'page': record.get('physical_pdf_page', record.get('source_page')),
            'metadata': metadata}


class AdvancedEvidenceRetriever:
    """A Qdrant- and cloud-independent BM25 retriever for the advanced index."""

    def __init__(self, index_path: str | Path = DEFAULT_INDEX, *, source: str | None = None):
        self.index_path = Path(index_path)
        summary_path = self.index_path.with_name('summary.json')
        if source is None and summary_path.is_file():
            try:
                source = Path(json.loads(summary_path.read_text(encoding='utf-8'))['source_pdf']).name
            except (KeyError, TypeError, json.JSONDecodeError):
                source = None
        self.source = source or 'logistics_advanced.pdf'
        normalized = []
        captions = []
        seen: set[str] = set()
        with self.index_path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'invalid JSON on advanced index line {line_number}') from exc
                if record.get('evidence_type') == 'caption':
                    captions.append(record)
                if _is_indexable(record):
                    item = map_record(record, self.source)
                    if item['id'] in seen:
                        raise ValueError(f'duplicate evidence_id in advanced index: {item["id"]}')
                    seen.add(item['id'])
                    normalized.append(item)
        if not normalized:
            raise ValueError(f'no searchable evidence records found in {self.index_path}')
        self.records = normalized
        self.captions = captions
        self.index = BM25Index(normalized)

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if filters is not None:
            raise ValueError('AdvancedEvidenceRetriever does not support filters; pass filters=None')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a non-empty string')
        if top_k < 1:
            raise ValueError('top_k must be positive')
        results = self.index.search(query, top_k=top_k)
        references = list(dict.fromkeys(
            reference for match in _REFERENCE_RE.finditer(query)
            if (reference := _caption_reference(match.group(0))) is not None))
        if not references:
            return results

        anchors: list[str] = []
        for reference in references:
            for record in self.records:
                metadata = record['metadata']
                if (metadata.get('evidence_type') in ('table', 'figure', 'chart') and
                        _caption_reference(metadata.get('caption', '') or '') == reference and
                        record['id'] not in anchors):
                    anchors.append(record['id'])
            for caption in self.captions:
                if _caption_reference(caption.get('label', '') or '') != reference:
                    continue
                page = caption.get('physical_pdf_page', caption.get('source_page'))
                page_captions = [item for item in self.captions
                                 if item.get('physical_pdf_page', item.get('source_page')) == page]
                kind = reference[0]
                page_kind_captions = [item for item in page_captions
                                      if (_caption_reference(item.get('label', '') or '') or (None,))[0] == kind]
                if len(page_kind_captions) != 1:
                    continue
                page_regions = [item for item in self.records
                                if item['page'] == page and item['metadata'].get('evidence_type') in
                                (('table',) if kind == 'table' else ('figure', 'chart'))]
                if len(page_regions) == 1 and page_regions[0]['id'] not in anchors:
                    anchors.append(page_regions[0]['id'])
        if not anchors:
            return results
        bm25_by_id = {record['id']: record for record in self.index.search(query, top_k=len(self.records))}
        anchored = []
        for identifier in anchors:
            result = bm25_by_id.get(identifier)
            if result is None:
                result = {**next(record for record in self.records if record['id'] == identifier),
                          'score': 0.0, 'strategy': 'reference'}
            anchored.append(result)
        ordered = anchored + [record for record in results if record['id'] not in anchors]
        return ordered[:top_k]
