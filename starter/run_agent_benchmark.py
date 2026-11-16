"""Run the real agent on a benchmark split and checkpoint operational traces."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from adapters import BedrockCohereEmbedder, GeminiController, QdrantRetriever
from agentic import run_agent

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'data' / 'logistics_benchmark.json'
DEFAULT_OUTPUT = ROOT / 'starter' / 'results' / 'agent_dev'
DEFAULT_TEST_OUTPUT = ROOT / 'starter' / 'results' / 'agent_test_final'
TRACE_NAME = 'traces.jsonl'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--neighbor-window', type=int, default=1)
    parser.add_argument('--max-searches', type=int, default=3)
    parser.add_argument('--strategy', choices=('A', 'B', 'C'), default='B')
    parser.add_argument('--output', type=Path, help='output directory (defaults to a split-isolated path)')
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK)
    parser.add_argument('--retry-errors', action='store_true',
                        help='retry queries whose latest checkpoint status is error')
    parser.add_argument('--retry-ids', nargs='+', metavar='QUERY_ID',
                        help='rerun only the selected IDs from the selected split')
    return parser


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n'


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _sanitize_error(error: BaseException) -> dict[str, str]:
    message = str(error)
    message = re.sub(r'(?i)(authorization\s*[:=]\s*bearer\s+)\S+', r'\1[REDACTED]', message)
    message = re.sub(r'(?i)(api[_ -]?key|secret|password|token)(\s*[:=]\s*)[^\s,;]+',
                     r'\1\2[REDACTED]', message)
    message = re.sub(r'https?://[^\s\])}>]+', '[REDACTED_URL]', message)
    message = re.sub(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b', '[REDACTED_AWS_KEY]', message)
    return {'class': type(error).__name__, 'message': message[:1000]}


def _load_records(path: Path, split: str) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError('benchmark must contain a JSON array')
    selected = [record for record in records if record.get('split') == split]
    if split == 'dev' and any(record.get('split') != 'dev' for record in selected):
        raise ValueError('DEV-only run guard failed')
    if not selected or any(not isinstance(record.get('id'), str)
                           or not isinstance(record.get('question'), str)
                           or not record['question'].strip() for record in selected):
        raise ValueError(f'{split} split must contain records with IDs and non-empty questions')
    return selected


def _write_case_review(records: list[dict[str, Any]], path: Path, split: str) -> None:
    # This is deliberately separate from traces and is never passed to the agent.
    cases = [{key: value for key, value in record.items() if key != 'question'}
             | {'query_id': record['id'], 'question': record['question']}
             for record in records]
    _atomic_json(path / 'case_gold_for_human_review.json', {'split': split, 'cases': cases})


def run_batch(args: argparse.Namespace, *, agent: Any = None,
              embedder_factory: Any = None, retriever_factory: Any = None,
              controller_factory: Any = None) -> dict[str, Any]:
    if args.split not in ('dev', 'test'):
        raise ValueError('split must be dev or test')
    if args.top_k < 1 or args.budget < 1 or args.max_searches < 1 or args.neighbor_window < 0:
        raise ValueError('top_k, budget, max_searches must be positive and neighbor_window non-negative')
    split = args.split
    output = getattr(args, 'output', None) or (DEFAULT_TEST_OUTPUT if split == 'test' else DEFAULT_OUTPUT)
    if split == 'test' and output.resolve() == DEFAULT_OUTPUT.resolve():
        raise ValueError('test output must be isolated from the dev artifact directory')
    records = _load_records(args.benchmark, split)
    run_records = records
    retry_ids = getattr(args, 'retry_ids', None)
    if retry_ids:
        all_records = json.loads(args.benchmark.read_text(encoding='utf-8'))
        split_ids = {record['id'] for record in records}
        known_splits = {record.get('id'): record.get('split') for record in all_records
                        if isinstance(record, dict) and isinstance(record.get('id'), str)}
        invalid = [query_id for query_id in retry_ids if query_id not in split_ids]
        if invalid:
            details = [f'{query_id} ({known_splits[query_id]} split)' if query_id in known_splits
                       else f'{query_id} (unknown ID)' for query_id in invalid]
            raise ValueError(f'--retry-ids accepts {split.upper()} IDs only: ' + ', '.join(details))
        selected_ids = set(retry_ids)
        run_records = [record for record in records if record['id'] in selected_ids]
    output.mkdir(parents=True, exist_ok=True)
    _write_case_review(records, output, split)
    trace_path = output / TRACE_NAME
    completed: dict[str, dict[str, Any]] = {}
    if trace_path.exists():
        with trace_path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    item = json.loads(line)
                    if item.get('query_id'):
                        completed[item['query_id']] = item

    agent = agent or run_agent
    try:
        embedder = (embedder_factory or BedrockCohereEmbedder)()
        retriever = (retriever_factory or QdrantRetriever)(
            embedder, strategy=args.strategy, neighbor_window=args.neighbor_window)
        controller = (controller_factory or GeminiController)()
        setup_error = None
    except Exception as error:
        # Construction failures are recorded per remaining question; do not retry unsafe setup.
        setup_error = error
        retriever = controller = None

    with trace_path.open('a', encoding='utf-8') as trace_file:
        for record in run_records:
            query_id = record['id']
            previous = completed.get(query_id)
            if previous and not retry_ids and (not getattr(args, 'retry_errors', False)
                             or previous.get('status') in ('answered', 'abstain')):
                continue
            started = time.perf_counter()
            try:
                if setup_error:
                    raise setup_error
                kwargs = {'top_k': args.top_k, 'max_searches': args.max_searches,
                          'budget': args.budget}
                if isinstance(record.get('filter'), dict):
                    kwargs['filters'] = record['filter']
                result = agent(record['question'], controller, retriever, **kwargs)
                trace = {'query_id': query_id, 'status': result.get('status', 'error'),
                         'answer': result.get('answer'), 'citations': result.get('citations', []),
                         'events': result.get('events', []), 'elapsed_ms': round((time.perf_counter() - started) * 1000)}
                if trace['status'] not in ('answered', 'abstain', 'error'):
                    trace['error'] = {'class': 'InvalidAgentStatus', 'message': 'unrecognized status'}
                    trace['status'] = 'error'
            except Exception as error:
                trace = {'query_id': query_id, 'status': 'error', 'answer': None, 'citations': [],
                         'events': [], 'elapsed_ms': round((time.perf_counter() - started) * 1000),
                         'error': _sanitize_error(error)}
            trace_file.write(_json_line(trace))
            trace_file.flush()
            # Make each completed query durable before moving to the next provider call.
            import os
            os.fsync(trace_file.fileno())
            completed[query_id] = trace

    ordered = [completed[record['id']] for record in records if record['id'] in completed]
    statuses = Counter(item['status'] for item in ordered)
    search_count = sum(sum(isinstance(event, dict) and event.get('event') == 'search'
                           for event in item.get('events', [])) for item in ordered)
    summary = {
        'split': split, 'total_cases': len(records), 'completed_cases': len(ordered),
        'statuses': dict(statuses), 'search_tool_calls': search_count,
        'queries_with_search': sum(any(isinstance(event, dict) and event.get('event') == 'search'
                                       for event in item.get('events', [])) for item in ordered),
        'error_count': statuses.get('error', 0),
        'error_classes': dict(Counter(item.get('error', {}).get('class', 'unknown')
                                      for item in ordered if item['status'] == 'error')),
        'elapsed_ms_total': sum(item.get('elapsed_ms', 0) for item in ordered),
        'evaluation': 'Operational metrics only. Semantic answer correctness and citation support were not scored; manual review is required.',
        'artifacts': {'traces': TRACE_NAME, 'case_gold': 'case_gold_for_human_review.json'}
    }
    _atomic_json(output / 'summary.json', summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_batch(args)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({'error': _sanitize_error(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
