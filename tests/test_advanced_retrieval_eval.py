import unittest
from unittest.mock import Mock

from starter.advanced_retrieval_eval import evaluate


def record(identifier, kind='text'):
    return {'id': identifier, 'text': 'indexed evidence ' + identifier,
            'metadata': {'evidence_type': kind}}


class AdvancedRetrievalEvalTests(unittest.TestCase):
    def test_simple_gold_anchors_are_alternatives(self):
        question = {'id': 'simple', 'question': 'Only this question', 'answerable': True,
                    'primary_group': 'table',
                    'gold': [{'evidence_id': 'anchor-a'}, {'evidence_id': 'anchor-b'}]}
        retriever = Mock()
        retriever.search.return_value = [record('anchor-b', 'table'), record('other')]
        summary, traces = evaluate(retriever, [question])
        retriever.search.assert_called_once_with('Only this question', top_k=10)
        self.assertTrue(traces[0]['hit@1'])
        self.assertEqual(summary['overall']['hit@1']['hits'], 1)

    def test_required_evidence_is_conjunctive_and_gold_not_in_trace_context(self):
        question = {'id': 'hop', 'question': 'Question', 'answerable': True,
                    'primary_group': 'multi_hop',
                    'required_evidence': [{'evidence_id': 'anchor-a'}, {'evidence_id': 'anchor-b'}],
                    'expected_answer': 'secret answer'}
        retriever = Mock()
        retriever.search.return_value = [record('anchor-a', 'table'), record('other')]
        summary, traces = evaluate(retriever, [question], budget=10000)
        self.assertFalse(traces[0]['hit@1'])
        self.assertFalse(traces[0]['hit@3'])
        self.assertEqual(traces[0]['required_anchor_ranks'], {'anchor-a': 1, 'anchor-b': None})
        self.assertNotIn('secret answer', str(traces))
        self.assertNotIn('required_evidence', str(traces))
        self.assertEqual(summary['overall']['hit@3']['hits'], 0)

    def test_unanswerable_is_reported_but_excluded_from_denominator(self):
        question = {'id': 'absent', 'question': 'Unknown', 'answerable': False,
                    'primary_group': 'text'}
        summary, traces = evaluate(Mock(search=Mock(return_value=[record('x')])), [question])
        self.assertEqual(summary['overall']['answerable'], 0)
        self.assertEqual(summary['overall']['unanswerable_ids'], ['absent'])
        self.assertNotIn('hit@1', traces[0])


if __name__ == '__main__':
    unittest.main()
