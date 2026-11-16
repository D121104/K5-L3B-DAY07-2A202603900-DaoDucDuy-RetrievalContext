"""Offline DEV-only evaluation for the advanced evidence retriever."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from advanced_retriever import AdvancedEvidenceRetriever, DEFAULT_INDEX
except ModuleNotFoundError:
    from starter.advanced_retriever import AdvancedEvidenceRetriever, DEFAULT_INDEX

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'data' / 'logistics_advanced_benchmark.json'
DEFAULT_OUTPUT = ROOT / 'starter' / 'results' / 'advanced_retrieval_dev'


def _required_ids(question: dict[str, Any]) -> list[str]:
    anchors = question.get('required_evidence') or question.get('gold') or []
    return list(dict.fromkeys(anchor['evidence_id'] for anchor in anchors))


def _context(results: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], int]:
    selected, used = [], 0
    for record in results:
        text = record.get('text', '')
        size = len(text)
        if used + size > budget:
            continue
        selected.append(record)
        used += size
    return selected, used


def _coverage(ids: set[str], required: list[str]) -> bool:
    return bool(required) and all(identifier in ids for identifier in required)


def evaluate(retriever: Any, questions: list[dict[str, Any]], budget: int = 10000,
             top_k: int = 10) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces = []
    for question in questions:
        # Retrieval receives only the question; gold is inspected afterward.
        ranked = retriever.search(question['question'], top_k=top_k)
        group = question.get('primary_group', 'unanswerable')
        if not question.get('answerable', True):
            traces.append({'id': question['id'], 'primary_group': group,
                           'answerable': False, 'retrieved': [
                               {'rank': rank, 'evidence_id': str(record['id']),
                                'record_type': record.get('metadata', {}).get('evidence_type')}
                               for rank, record in enumerate(ranked, 1)]})
            continue

        required = _required_ids(question)
        conjunctive = bool(question.get('required_evidence'))
        ranks = {identifier: next((rank for rank, record in enumerate(ranked, 1)
                                   if str(record['id']) == identifier), None)
                 for identifier in required}
        context_records, context_chars = _context(ranked, budget)
        context_ids = {str(record['id']) for record in context_records}
        metrics = {}
        for k in (1, 3, 10):
            top_ids = {str(record['id']) for record in ranked[:k]}
            metrics[f'hit@{k}'] = (_coverage(top_ids, required) if conjunctive
                                   else bool(top_ids.intersection(required)))
        metrics['context_evidence'] = (_coverage(context_ids, required) if conjunctive
                                       else bool(context_ids.intersection(required)))
        traces.append({'id': question['id'], 'primary_group': group,
                       'answerable': True, 'required_anchor_ranks': ranks,
                       'retrieved': [{'rank': rank, 'evidence_id': str(record['id']),
                                      'record_type': record.get('metadata', {}).get('evidence_type')}
                                     for rank, record in enumerate(ranked, 1)],
                       'context': [{'evidence_id': str(record['id']),
                                    'record_type': record.get('metadata', {}).get('evidence_type')}
                                   for record in context_records],
                       'context_chars': context_chars, **metrics})

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        answerable = [row for row in rows if row['answerable']]
        unanswerable = len(rows) - len(answerable)
        result: dict[str, Any] = {'questions': len(rows), 'answerable': len(answerable),
                                  'unanswerable': unanswerable}
        for metric in ('hit@1', 'hit@3', 'hit@10', 'context_evidence'):
            hits = sum(bool(row[metric]) for row in answerable)
            result[metric] = {'hits': hits, 'rate': hits / len(answerable) if answerable else 0.0}
        result['unanswerable_ids'] = [row['id'] for row in rows if not row['answerable']]
        result['misses'] = {metric: [row['id'] for row in answerable if not row[metric]]
                            for metric in ('hit@1', 'hit@3', 'hit@10', 'context_evidence')}
        return result

    groups = defaultdict(list)
    for row in traces:
        if row['answerable']:
            groups[row['primary_group']].append(row)
    summary = {'split': 'dev', 'top_k': top_k, 'context_budget_chars': budget,
               'overall': aggregate(traces),
               'unanswerable_ids': [row['id'] for row in traces if not row['answerable']],
               'by_primary_group': {group: aggregate(rows) for group, rows in sorted(groups.items())}}
    return summary, traces


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.budget < 1 or args.top_k < 1:
        parser.error('--budget and --top-k must be positive')

    # Select DEV before handing any questions to the local retriever.
    records = json.loads(args.benchmark.read_text(encoding='utf-8'))
    questions = [record for record in records if record.get('split') == 'dev']
    if not questions:
        raise SystemExit('No DEV questions found')
    summary, traces = evaluate(AdvancedEvidenceRetriever(args.index), questions, args.budget, args.top_k)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
