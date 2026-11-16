"""Smoke check fixture contract; add tests for your implementation."""
import json
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
class FixtureTest(unittest.TestCase):
    def test_fixture_gold_resolves(self):
        chunks=json.loads((ROOT/'data/example_chunks.json').read_text(encoding='utf-8'))
        questions=json.loads((ROOT/'data/example_questions.json').read_text(encoding='utf-8'))
        for q in questions:
            for g in q['gold']:
                self.assertTrue(any(c['source_id']==g['source_id'] and c['page']==g['page'] and g['evidence'] in c['text'] for c in chunks))

    def test_logistics_benchmark_contract_and_split_counts(self):
        questions = json.loads((ROOT/'data/logistics_benchmark.json').read_text(encoding='utf-8'))
        answerable = [q for q in questions if q.get('answerable', True)]
        unanswerable = [q for q in questions if q.get('answerable') is False]
        self.assertEqual(len(questions), 24)
        self.assertEqual([sum(q['split'] == split for q in answerable) for split in ('dev', 'test')], [15, 5])
        self.assertEqual([sum(q['split'] == split for q in unanswerable) for split in ('dev', 'test')], [2, 2])
        for q in answerable:
            self.assertIn(q['split'], ('dev', 'test'))
            self.assertEqual(len(q['gold']), 1)
            self.assertEqual(q['gold'][0]['source_id'], 'logistics-vietnam-2024')
            self.assertTrue(q['gold'][0]['evidence'])
        for q in unanswerable:
            self.assertNotIn('gold', q)
            self.assertEqual(q['expected_behavior'], 'search_and_abstain')
            self.assertTrue(q['absence_reason'].strip())

if __name__ == '__main__': unittest.main()
