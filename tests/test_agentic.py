"""Offline behavioral checks only; fake adapters do not measure model quality."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from agentic import run_agent

CHUNK = {'id': 'c1', 'text': 'Hoàn tiền trong 7 ngày.', 'source_id': 'doc',
         'source': 'doc.pdf', 'page': 1, 'score': 0.8, 'metadata': {}}

class Controller:
    def __init__(self, actions): self.actions = iter(actions)
    def decide(self, question, history): return next(self.actions)

class Retriever:
    def __init__(self, results): self.results = results; self.calls = []
    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k)); return self.results

class AgentTests(unittest.TestCase):
    def test_agent_searches_and_cites_actual_context(self):
        c = Controller([{'action': 'search', 'query': 'hoàn tiền'},
                        {'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']}])
        r = Retriever([CHUNK]); result = run_agent('Hoàn tiền?', c, r, filters={'year': 2025})
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['citations'], ['c1'])
        self.assertEqual(r.calls, [('hoàn tiền', {'year': 2025}, 3)])
        self.assertEqual(result['events'][0]['context_ids'], ['c1'])

    def test_invalid_citation_and_empty_context_abstain(self):
        for rows, cid in (([CHUNK], 'not-present'), ([], 'c1')):
            c = Controller([{'action':'search','query':'q'}, {'action':'answer','answer':f'7 ngày [{cid}]','citations':[cid]}])
            self.assertEqual(run_agent('q', c, Retriever(rows))['status'], 'abstain')

    def test_empty_generated_answer_or_missing_citations_abstains(self):
        for generated in ({'action': 'answer', 'answer': '', 'citations': []},
                          {'action': 'answer', 'answer': None, 'citations': None}):
            with self.subTest(generated=generated):
                result = run_agent('q', Controller([{'action': 'search', 'query': 'q'}, generated]),
                                   Retriever([CHUNK]))
                self.assertEqual(result['status'], 'abstain')
                self.assertEqual(result['citations'], [])
                self.assertEqual(result['events'][-1]['reason'], 'missing_answer')

    def test_invalid_result_diagnostic_reasons_do_not_store_answer(self):
        cases = [
            ({'answer': 'Answer', 'citations': []}, 'missing_citations'),
            ({'answer': 'Answer [missing]', 'citations': ['missing']}, 'citation_not_in_context'),
        ]
        for response, reason in cases:
            with self.subTest(reason=reason):
                result = run_agent('q', Controller([{'action': 'search', 'query': 'q'},
                    {'action': 'answer', **response}]), Retriever([CHUNK]))
                diagnostic = result['events'][-1]
                self.assertEqual(diagnostic['reason'], reason)
                self.assertNotIn('Answer', str(diagnostic))

    def test_missing_markers_are_repaired_only_for_valid_cited_ids(self):
        result = run_agent('q', Controller([{'action': 'search', 'query': 'q'},
            {'action': 'answer', 'answer': 'Answer precisely as given.', 'citations': ['c1']}]),
            Retriever([CHUNK]))
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['answer'], 'Answer precisely as given. [c1]')

        result = run_agent('q', Controller([{'action': 'search', 'query': 'q'},
            {'action': 'answer', 'answer': 'Unsupported statement.', 'citations': ['invented']}]),
            Retriever([CHUNK]))
        self.assertEqual(result['status'], 'abstain')
        self.assertEqual(result['events'][-1]['reason'], 'citation_not_in_context')

    def test_neighbor_is_context_candidate_and_valid_citation(self):
        neighbor = {**CHUNK, 'id': 'c-neighbor', 'text': 'Supporting adjacent evidence.',
                    'metadata': {'chunk_index': 0}}
        primary = {**CHUNK, 'metadata': {'chunk_index': 1}, '_neighbors': [neighbor]}
        controller = Controller([{'action': 'search', 'query': 'q'},
                                 {'action': 'answer', 'answer': 'Supported [c-neighbor]',
                                  'citations': ['c-neighbor']}])
        result = run_agent('q', controller, Retriever([primary]))
        self.assertEqual(result['status'], 'answered')
        event = result['events'][0]
        self.assertEqual([item['id'] for item in event['results']], ['c1'])
        self.assertEqual([item['id'] for item in event['expanded_context']], ['c-neighbor', 'c1'])
        self.assertIn('c-neighbor', event['context_ids'])

    def test_each_ranked_hit_stays_with_its_neighbors(self):
        neighbor = {**CHUNK, 'id': 'lower-neighbor', 'metadata': {'chunk_index': 2}}
        high_hit = {**CHUNK, 'id': 'high', 'metadata': {'chunk_index': 3},
                    '_neighbors': [neighbor]}
        low_hit = {**CHUNK, 'id': 'low', 'metadata': {'chunk_index': 8}}
        controller = Controller([{'action': 'search', 'query': 'q'},
                                 {'action': 'answer', 'answer': 'Supported [high]', 'citations': ['high']}])
        result = run_agent('q', controller, Retriever([high_hit, low_hit]))
        self.assertEqual([item['id'] for item in result['events'][0]['expanded_context']],
                         ['lower-neighbor', 'high', 'low'])

    def test_search_limit_is_enforced(self):
        c = Controller([{'action':'search','query':'one'}, {'action':'search','query':'two'}, {'action':'search','query':'three'}])
        r = Retriever([CHUNK]); result = run_agent('q', c, r, max_searches=2)
        self.assertEqual(result['status'], 'abstain'); self.assertEqual(len(r.calls), 2)

    def test_context_budget_across_multiple_searches(self):
        other = {**CHUNK, 'id': 'c2', 'text': 'Một đoạn bằng chứng khác.'}
        class SequentialRetriever:
            def __init__(self): self.i = 0
            def search(self, query, *, filters, top_k):
                self.i += 1
                return [CHUNK] if self.i == 1 else [other]
        class InspectController:
            def __init__(self): self.i = 0
            def decide(self, question, history):
                self.i += 1
                if history:
                    assert len(history) == 1 and len(history[0]['context']) <= 90
                if self.i <= 2: return {'action': 'search', 'query': str(self.i)}
                return {'action': 'answer', 'answer': 'Evidence [c2]', 'citations': ['c2']}
        result = run_agent('q', InspectController(), SequentialRetriever(), budget=90)
        self.assertEqual(result['status'], 'answered')
        self.assertTrue(all(len(e['context']) <= 90 for e in result['events'] if e['event'] == 'search'))
        self.assertEqual(result['events'][-2]['context_ids'], ['c2'])

    def test_configurable_retrieval_allows_more_than_two_calls(self):
        c = Controller([{'action': 'search', 'query': str(i)} for i in range(3)] +
                       [{'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']}])
        r = Retriever([CHUNK])
        result = run_agent('q', c, r, max_searches=3, top_k=7, budget=1200)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(r.calls), 3)
        self.assertTrue(all(call[2] == 7 for call in r.calls))

    def test_tool_errors_are_not_hidden(self):
        class Broken:
            def search(self, query, *, filters, top_k): raise ConnectionError('database unavailable')
        with self.assertRaises(ConnectionError):
            run_agent('q', Controller([{'action':'search','query':'q'}]), Broken())

if __name__ == '__main__': unittest.main()
