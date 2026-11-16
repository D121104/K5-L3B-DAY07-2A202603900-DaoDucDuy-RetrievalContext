"""Run the advanced logistics benchmark with local BM25 and Gemini."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from adapters import GeminiController
    from advanced_retriever import AdvancedEvidenceRetriever, DEFAULT_INDEX
    from agentic import run_agent
except ModuleNotFoundError:
    from starter.adapters import GeminiController
    from starter.advanced_retriever import AdvancedEvidenceRetriever, DEFAULT_INDEX
    from starter.agentic import run_agent

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / 'data' / 'logistics_advanced_benchmark.json'
DEFAULT_DEV_OUTPUT = ROOT / 'starter' / 'results' / 'advanced_agent_dev'
DEFAULT_TEST_OUTPUT = ROOT / 'starter' / 'results' / 'advanced_agent_test'


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n'


def _sanitize_error(error: BaseException) -> dict[str, str]:
    message = str(error)
    message = re.sub(r'(?i)(authorization\s*[:=]\s*bearer\s+)\S+', r'\1[REDACTED]', message)
    message = re.sub(r'(?i)(api[_ -]?key|secret|password|token)(\s*[:=]\s*)[^\s,;]+',
                     r'\1\2[REDACTED]', message)
    message = re.sub(r'https?://[^\s\])}>]+', '[REDACTED_URL]', message)
    message = re.sub(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b', '[REDACTED_AWS_KEY]', message)
    return {'class': type(error).__name__, 'message': message[:1000]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--max-searches', type=int, default=3)
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--output', type=Path)
    return parser


def run_batch(args: argparse.Namespace, *, agent: Any = None, retriever_factory: Any = None,
              controller_factory: Any = None) -> dict[str, Any]:
    if args.split not in ('dev', 'test'):
        raise ValueError('split must be dev or test')
    if args.top_k < 1 or args.budget < 1 or args.max_searches < 1:
        raise ValueError('top_k, budget, and max_searches must be positive')
    output = getattr(args, 'output', None) or (DEFAULT_TEST_OUTPUT if args.split == 'test' else DEFAULT_DEV_OUTPUT)
    if args.split == 'test' and output.resolve() == DEFAULT_DEV_OUTPUT.resolve():
        raise ValueError('test output must be isolated from the dev artifact directory')
    records = json.loads(args.benchmark.read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError('benchmark must contain a JSON array')
    selected = [row for row in records if row.get('split') == args.split]
    if not selected or any(not isinstance(row.get('id'), str) or not isinstance(row.get('question'), str)
                           or not row['question'].strip() for row in selected):
        raise ValueError(f'{args.split} split must contain records with IDs and non-empty questions')
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / 'traces.jsonl'
    completed: dict[str, dict[str, Any]] = {}
    if trace_path.exists():
        with trace_path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    item = json.loads(line)
                    if item.get('query_id'):
                        completed[item['query_id']] = item
    try:
        retriever = (retriever_factory or AdvancedEvidenceRetriever)(args.index)
        controller = (controller_factory or GeminiController)()
        setup_error = None
    except Exception as error:
        setup_error = error
        retriever = controller = None
    with trace_path.open('a', encoding='utf-8') as trace_file:
        for row in selected:
            if row['id'] in completed:
                continue
            started = time.perf_counter()
            try:
                if setup_error:
                    raise setup_error
                result = (agent or run_agent)(row['question'], controller, retriever,
                    top_k=args.top_k, budget=args.budget, max_searches=args.max_searches)
                trace = {'query_id': row['id'], 'status': result.get('status', 'error'),
                         'answer': result.get('answer'), 'citations': result.get('citations', []),
                         'events': result.get('events', []),
                         'elapsed_ms': round((time.perf_counter() - started) * 1000)}
            except Exception as error:
                trace = {'query_id': row['id'], 'status': 'error', 'answer': None,
                         'citations': [], 'events': [],
                         'elapsed_ms': round((time.perf_counter() - started) * 1000),
                         'error': _sanitize_error(error)}
            trace_file.write(_json_line(trace))
            trace_file.flush()
            os.fsync(trace_file.fileno())
            completed[row['id']] = trace
    ordered = [completed[row['id']] for row in selected if row['id'] in completed]
    statuses = Counter(item['status'] for item in ordered)
    summary = {'split': args.split, 'total_cases': len(selected), 'completed_cases': len(ordered),
               'statuses': dict(statuses),
               'search_tool_calls': sum(sum(isinstance(event, dict) and event.get('event') == 'search'
                                           for event in item.get('events', [])) for item in ordered),
               'error_count': statuses.get('error', 0), 'artifacts': {'traces': 'traces.jsonl'}}
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
