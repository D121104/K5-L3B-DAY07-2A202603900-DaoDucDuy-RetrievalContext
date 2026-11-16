"""Offline safety and resume tests for the benchmark runner."""
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
import run_agent_benchmark as benchmark


class AgentBenchmarkTests(unittest.TestCase):
    def args(self, directory):
        return Namespace(split='dev', top_k=10, budget=10000, neighbor_window=1,
                         max_searches=3, strategy='B', output=Path(directory),
                         benchmark=ROOT / 'data' / 'logistics_benchmark.json')

    def test_only_question_and_filter_reach_agent_and_artifacts_are_separated(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Mock(return_value={'status': 'abstain', 'answer': 'no evidence',
                                       'citations': [], 'events': [{'event': 'search'}]})
            summary = benchmark.run_batch(self.args(directory), agent=agent,
                embedder_factory=Mock(return_value=object()),
                retriever_factory=Mock(return_value=object()),
                controller_factory=Mock(return_value=object()))
            called = agent.call_args_list[0]
            self.assertEqual(called.args[0], 'Các tổ chức quốc tế dự báo tăng trưởng kinh tế thế giới năm 2024 quanh mức nào?')
            self.assertNotIn('expected_answer', called.kwargs)
            self.assertNotIn('gold', called.kwargs)
            self.assertEqual(summary['total_cases'], 17)
            trace_text = (Path(directory) / 'traces.jsonl').read_text(encoding='utf-8')
            self.assertNotIn('expected_answer', trace_text)
            self.assertIn('absence_reason', (Path(directory) / 'case_gold_for_human_review.json').read_text(encoding='utf-8'))
            self.assertIn('manual review is required', summary['evaluation'])

    def test_resume_skips_durable_query_ids_and_test_split_runs_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            trace_path = Path(directory) / 'traces.jsonl'
            trace_path.write_text(json.dumps({'query_id': 'logistics24-q01', 'status': 'abstain',
                'answer': '', 'citations': [], 'events': [], 'elapsed_ms': 1}) + '\n', encoding='utf-8')
            agent = Mock(return_value={'status': 'abstain', 'answer': '', 'citations': [], 'events': []})
            benchmark.run_batch(args, agent=agent, embedder_factory=Mock(return_value=object()),
                retriever_factory=Mock(return_value=object()), controller_factory=Mock(return_value=object()))
            self.assertEqual(agent.call_count, 16)
            args.split = 'test'
            args.output = Path(directory) / 'agent_test_final'
            test_agent = Mock(return_value={'status': 'abstain', 'answer': '', 'citations': [], 'events': []})
            summary = benchmark.run_batch(args, agent=test_agent,
                embedder_factory=Mock(return_value=object()), retriever_factory=Mock(return_value=object()),
                controller_factory=Mock(return_value=object()))
            self.assertEqual(test_agent.call_count, 7)
            self.assertEqual(summary['split'], 'test')
            self.assertEqual(summary['total_cases'], 7)
            test_ids = {record['id'] for record in benchmark._load_records(args.benchmark, 'test')}
            entries = [json.loads(line) for line in (args.output / 'traces.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual({entry['query_id'] for entry in entries}, test_ids)
            self.assertEqual(json.loads((args.output / 'case_gold_for_human_review.json').read_text())['split'], 'test')
            self.assertFalse((args.output / 'dev' / 'traces.jsonl').exists())

    def test_test_run_rejects_dev_output_and_retry_ids_from_other_split(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            args.split = 'test'
            args.output = benchmark.DEFAULT_OUTPUT
            with self.assertRaisesRegex(ValueError, 'isolated'):
                benchmark.run_batch(args, agent=Mock())
            args.output = Path(directory) / 'test'
            args.retry_ids = ['logistics24-q01']
            with self.assertRaisesRegex(ValueError, 'TEST IDs only'):
                benchmark.run_batch(args, agent=Mock())

    def test_retry_errors_is_opt_in_and_latest_trace_wins_summary(self):
        for retry_errors, expected_calls in ((False, 16), (True, 17)):
            with self.subTest(retry_errors=retry_errors), tempfile.TemporaryDirectory() as directory:
                args = self.args(directory)
                args.retry_errors = retry_errors
                trace_path = Path(directory) / 'traces.jsonl'
                trace_path.write_text(json.dumps({'query_id': 'logistics24-q01', 'status': 'error',
                    'answer': None, 'citations': [], 'events': [], 'elapsed_ms': 1}) + '\n', encoding='utf-8')
                agent = Mock(return_value={'status': 'answered', 'answer': 'recovered',
                                           'citations': [], 'events': []})
                summary = benchmark.run_batch(args, agent=agent, embedder_factory=Mock(return_value=object()),
                    retriever_factory=Mock(return_value=object()), controller_factory=Mock(return_value=object()))
                self.assertEqual(agent.call_count, expected_calls)
                entries = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(len(entries), 18 if retry_errors else 17)
                self.assertEqual(summary['statuses'].get('error', 0), 0 if retry_errors else 1)
                if retry_errors:
                    self.assertEqual(entries[0]['status'], 'error')
                    self.assertEqual(entries[1]['status'], 'answered')
                    self.assertEqual(summary['statuses']['answered'], 17)

    def test_retry_ids_runs_only_selected_completed_dev_query(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            args.retry_ids = ['logistics24-q10']
            trace_path = Path(directory) / 'traces.jsonl'
            trace_path.write_text(''.join(json.dumps({'query_id': f'logistics24-q{i:02d}',
                'status': 'answered', 'answer': 'old', 'citations': [], 'events': [], 'elapsed_ms': 1}) + '\n'
                for i in (9, 10, 11)), encoding='utf-8')
            agent = Mock(return_value={'status': 'answered', 'answer': 'new',
                                       'citations': ['chunk'], 'events': []})
            benchmark.run_batch(args, agent=agent, embedder_factory=Mock(return_value=object()),
                retriever_factory=Mock(return_value=object()), controller_factory=Mock(return_value=object()))
            agent.assert_called_once()
            self.assertEqual(agent.call_args.args[0], 'Bao nhiêu phần trăm lao động logistics được đào tạo bài bản về dịch vụ logistics?')
            entries = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([entry['query_id'] for entry in entries],
                             ['logistics24-q09', 'logistics24-q10', 'logistics24-q11', 'logistics24-q10'])

    def test_retry_ids_rejects_unknown_and_test_ids(self):
        for query_id, expected in (('not-a-query', 'unknown ID'),
                                   ('logistics24-q16', 'test split')):
            with self.subTest(query_id=query_id), tempfile.TemporaryDirectory() as directory:
                args = self.args(directory)
                args.retry_ids = [query_id]
                with self.assertRaisesRegex(ValueError, expected):
                    benchmark.run_batch(args, agent=Mock())

    def test_error_sanitizer_redacts_credentials_and_urls(self):
        sanitized = benchmark._sanitize_error(RuntimeError('api_key=abc123 https://host/path?token=secret'))
        self.assertNotIn('abc123', sanitized['message'])
        self.assertNotIn('secret', sanitized['message'])
        self.assertEqual(sanitized['class'], 'RuntimeError')


if __name__ == '__main__':
    unittest.main()
