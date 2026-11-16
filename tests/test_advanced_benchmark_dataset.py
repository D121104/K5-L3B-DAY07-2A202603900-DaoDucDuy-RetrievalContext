"""Offline integrity checks for the advanced logistics benchmark."""
import json
import sys
import tempfile
import unittest
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
_adapters = types.ModuleType('adapters')
_adapters.BedrockCohereEmbedder = object
_adapters.GeminiController = object
_adapters.QdrantRetriever = object
_agentic = types.ModuleType('agentic')
_agentic.run_agent = object()
with patch.dict(sys.modules, {'adapters': _adapters, 'agentic': _agentic}):
    import run_agent_benchmark as runner

DATA_PATH = ROOT / 'data' / 'logistics_advanced_benchmark.json'
INDEX_PATH = ROOT / 'starter' / 'results' / 'advanced_index' / 'records.jsonl'
GROUPS = {'table', 'figure', 'chart', 'multi_hop', 'text'}


def read_dataset():
    return json.loads(DATA_PATH.read_text(encoding='utf-8'))


def flatten_gold(record):
    return record.get('required_evidence', record.get('gold', []))


class AdvancedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_dataset()
        cls.index = [json.loads(line) for line in INDEX_PATH.read_text(encoding='utf-8').splitlines()]
        cls.by_id = {entry['evidence_id']: entry for entry in cls.index}

    def test_exact_split_counts_and_test_group_quotas(self):
        for split, total, answerable, unanswerable in (
                ('dev', 18, 15, 3), ('test', 30, 24, 6)):
            rows = [r for r in self.records if r['split'] == split]
            self.assertEqual(len(rows), total)
            self.assertEqual(sum(bool(r['answerable']) for r in rows), answerable)
            self.assertEqual(sum(not r['answerable'] for r in rows), unanswerable)
        test_answerable = [r for r in self.records if r['split'] == 'test' and r['answerable']]
        self.assertEqual({g: sum(r['primary_group'] == g for r in test_answerable)
                          for g in GROUPS},
                         {'table': 6, 'figure': 5, 'chart': 5, 'multi_hop': 4, 'text': 4})
        self.assertEqual({r['split'] for r in self.records}, {'dev', 'test'})

    def test_ids_questions_answers_and_splits_are_separated(self):
        self.assertEqual(len({r['id'] for r in self.records}), len(self.records))
        for split in ('dev', 'test'):
            questions = [r['question'].strip().casefold() for r in self.records if r['split'] == split]
            self.assertEqual(len(questions), len(set(questions)))
        dev = [r for r in self.records if r['split'] == 'dev' and r['answerable']]
        test = [r for r in self.records if r['split'] == 'test' and r['answerable']]
        self.assertFalse({r['question'].strip().casefold() for r in dev}
                         & {r['question'].strip().casefold() for r in test})
        dev_answers = {r['expected_answer'].strip().casefold() for r in dev}
        test_answers = {r['expected_answer'].strip().casefold() for r in test}
        self.assertFalse(dev_answers & test_answers)

    def test_answerable_gold_anchor_schema_and_index_consistency(self):
        for record in self.records:
            if not record['answerable']:
                self.assertTrue(record.get('search_scope'))
                self.assertTrue(record.get('absence_reason'))
                self.assertNotIn('expected_answer', record)
                continue
            self.assertIn(record['primary_group'], GROUPS)
            self.assertTrue(record.get('expected_answer'))
            anchors = flatten_gold(record)
            self.assertTrue(anchors, record['id'])
            if record['primary_group'] == 'multi_hop':
                self.assertGreaterEqual(len(anchors), 2)
                self.assertTrue(record.get('scoring_condition'))
                pages = {a['physical_pdf_page'] for a in anchors}
                ids = {a['evidence_id'] for a in anchors}
                self.assertTrue(len(pages) > 1 or len(ids) > 1)
            else:
                self.assertTrue(record.get('gold'))
            for anchor in anchors:
                for field in ('source_id', 'physical_pdf_page', 'evidence_type',
                              'evidence_id', 'bbox', 'crop_path', 'evidence'):
                    self.assertIn(field, anchor, (record['id'], field))
                self.assertEqual(anchor['source_id'], 'logistics-vietnam-2024')
                self.assertEqual(len(anchor['bbox']), 4)
                x0, y0, x1, y1 = anchor['bbox']
                self.assertTrue(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1,
                                (record['id'], anchor['bbox']))
                indexed = self.by_id.get(anchor['evidence_id'])
                self.assertIsNotNone(indexed, (record['id'], anchor['evidence_id']))
                for key in ('physical_pdf_page', 'evidence_type', 'bbox', 'crop_path'):
                    if key == 'crop_path' and anchor[key]:
                        self.assertEqual(Path(anchor[key]).name, Path(indexed[key]).name,
                                         (record['id'], key))
                    else:
                        self.assertEqual(anchor[key], indexed[key], (record['id'], key))
                if anchor['printed_page'] is not None:
                    self.assertEqual(str(anchor['printed_page']), str(indexed['printed_page_label']))
                if anchor['evidence_type'] in {'figure', 'chart', 'table'}:
                    self.assertTrue(anchor['crop_path'])
                    self.assertTrue((ROOT / anchor['crop_path']).is_file(), anchor['crop_path'])
                if anchor['evidence_type'] == 'table':
                    cells = anchor.get('table_cells', {})
                    self.assertTrue(cells.get('row'))
                    self.assertTrue(cells.get('column'))
                    self.assertTrue(cells.get('units'))
            if record['primary_group'] == 'table':
                self.assertTrue(all(a['evidence_type'] == 'table' for a in anchors))
            if record['primary_group'] == 'figure':
                self.assertTrue(all(a['evidence_type'] == 'figure' for a in anchors))
            if record['primary_group'] == 'chart':
                self.assertTrue(all(a['evidence_type'] == 'chart' for a in anchors))

    def test_dev_runner_never_receives_gold_or_expected_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Mock(return_value={'status': 'abstain', 'answer': '', 'citations': [], 'events': []})
            args = Namespace(split='dev', top_k=5, budget=1000, neighbor_window=1,
                             max_searches=1, strategy='B', output=Path(directory),
                             benchmark=DATA_PATH)
            runner.run_batch(args, agent=agent, embedder_factory=Mock(return_value=object()),
                             retriever_factory=Mock(return_value=object()),
                             controller_factory=Mock(return_value=object()))
            self.assertEqual(agent.call_count, 18)
            for call in agent.call_args_list:
                self.assertNotIn('expected_answer', call.kwargs)
                self.assertNotIn('gold', call.kwargs)
                self.assertNotIn('required_evidence', call.kwargs)
                self.assertNotIn('absence_reason', call.kwargs)
                self.assertNotIn('search_scope', call.kwargs)
            traces = (Path(directory) / 'traces.jsonl').read_text(encoding='utf-8')
            self.assertNotIn('expected_answer', traces)
            self.assertNotIn('required_evidence', traces)


if __name__ == '__main__':
    unittest.main()
