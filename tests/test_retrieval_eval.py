import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
import retrieval_eval


def chunk(identifier='a', section='3.2'):
    return {'id': identifier, 'source_id': 'doc', 'source': 'doc.pdf', 'page': 2,
            'text': 'Evidence phrase appears here.', 'metadata': {'section': section}}


class RetrievalEvalTests(unittest.TestCase):
    def test_score_requires_source_page_and_evidence(self):
        q = {'answerable': True, 'gold': [{'source_id': 'doc', 'page': 2, 'evidence': 'Evidence phrase'}]}
        self.assertTrue(retrieval_eval.score_results([chunk()], q)['hit1'])
        self.assertFalse(retrieval_eval.score_results([{**chunk(), 'page': 3}], q)['hit3'])
        self.assertFalse(retrieval_eval.score_results([{**chunk(), 'source_id': 'other'}], q)['hit3'])
        self.assertFalse(retrieval_eval.score_results([{**chunk(), 'text': 'other'}], q)['hit3'])

    def test_normalize_evidence_handles_whitespace_case_and_nfkc(self):
        self.assertEqual(retrieval_eval.normalize_evidence('  ＦＯＯ\n\tＢar  '), 'foo bar')

    def test_normalized_hit_and_exact_hit_are_reported_separately(self):
        q = {'answerable': True, 'gold': [{'source_id': 'doc', 'page': 2, 'evidence': 'Evidence\nPHRASE'}]}
        result = retrieval_eval.score_results([chunk()], q)
        self.assertTrue(result['hit1'])
        self.assertFalse(result['exact_hit1'])
        self.assertTrue(result['results'][0]['normalized_hit'])
        self.assertFalse(result['results'][0]['exact_hit'])

    def test_hit3_excludes_evidence_at_rank_four_but_hitk_includes_it(self):
        q = {'answerable': True, 'gold': [{'source_id': 'doc', 'page': 2,
                                          'evidence': 'Evidence phrase'}]}
        results = [{**chunk(identifier=str(i)), 'text': 'No matching evidence'} for i in range(3)] + [chunk(identifier='rank4')]
        result = retrieval_eval.score_results(results, q)
        self.assertFalse(result['hit3'])
        self.assertTrue(result['hitk'])
        self.assertFalse(result['exact_hit3'])
        self.assertTrue(result['exact_hitk'])

    def test_stitched_hit_requires_adjacent_retrieved_chunks_and_respects_cutoffs(self):
        first = {**chunk('first'), 'text': 'The relevant evidence begins',
                 'metadata': {'chunk_index': 0}}
        second = {**chunk('second'), 'text': 'and concludes here.',
                  'metadata': {'chunk_index': 1}}
        unrelated = {**chunk('unrelated'), 'text': 'No evidence.',
                     'metadata': {'chunk_index': 8}}
        filler = {**chunk('filler'), 'text': 'Still no evidence.',
                  'metadata': {'chunk_index': 9}}
        q = {'answerable': True, 'gold': [{'source_id': 'doc', 'page': 2,
                                           'evidence': 'relevant evidence begins and concludes'}]}
        result = retrieval_eval.score_results([first, unrelated, filler, second], q)
        self.assertFalse(result['stitched_hit1'])
        self.assertFalse(result['stitched_hit3'])
        self.assertTrue(result['stitched_hitk'])
        self.assertTrue(result['stitched_context_hit'])

        result = retrieval_eval.score_results([first, second], q)
        self.assertFalse(result['stitched_hit1'])
        self.assertTrue(result['stitched_hit3'])
        self.assertTrue(result['stitched_hitk'])
        self.assertTrue(result['stitched_context_hit'])

    def test_stitched_hit_rejects_wrong_page_and_nonadjacent_indexes(self):
        first = {**chunk('first'), 'text': 'Evidence crosses',
                 'metadata': {'chunk_index': 0}}
        second = {**chunk('second'), 'text': 'these chunks.',
                  'metadata': {'chunk_index': 2}}
        q = {'answerable': True, 'gold': [{'source_id': 'doc', 'page': 2,
                                           'evidence': 'Evidence crosses these chunks.'}]}
        self.assertFalse(retrieval_eval.score_results([first, second], q)['stitched_hitk'])
        adjacent = {**second, 'metadata': {'chunk_index': 1}, 'page': 3}
        self.assertFalse(retrieval_eval.score_results([first, adjacent], q)['stitched_hitk'])

    def test_context_expansion_is_same_source_page_and_contiguous(self):
        hit = {**chunk('hit'), 'metadata': {'chunk_index': 1}}
        corpus = [hit,
                  {**chunk('previous'), 'metadata': {'chunk_index': 0}},
                  {**chunk('next'), 'metadata': {'chunk_index': 2}},
                  {**chunk('gap'), 'metadata': {'chunk_index': 3}},
                  {**chunk('other-page'), 'page': 3, 'metadata': {'chunk_index': 2}},
                  {**chunk('other-source'), 'source_id': 'other', 'metadata': {'chunk_index': 2}}]
        expanded = retrieval_eval.expand_adjacent_context([hit], corpus)
        self.assertEqual([c['id'] for c in expanded], ['previous', 'hit', 'next'])

    def test_context_expansion_deduplicates_and_preserves_rank_priority(self):
        first = {**chunk('first'), 'metadata': {'chunk_index': 1}}
        second = {**chunk('second'), 'metadata': {'chunk_index': 2}}
        neighbor = {**chunk('neighbor'), 'metadata': {'chunk_index': 0}}
        expanded = retrieval_eval.expand_adjacent_context([first, second, first],
                                                          [first, second, neighbor])
        self.assertEqual([c['id'] for c in expanded], ['neighbor', 'first', 'second'])

    def test_expansion_groups_neighbors_with_each_ranked_hit(self):
        high = {**chunk('high'), 'metadata': {'chunk_index': 3}}
        adjacent = {**chunk('adjacent'), 'metadata': {'chunk_index': 2}}
        low = {**chunk('low'), 'metadata': {'chunk_index': 8}}
        expanded = retrieval_eval.expand_adjacent_context([high, low], [high, adjacent, low])
        self.assertEqual([item['id'] for item in expanded], ['adjacent', 'high', 'low'])

    def test_neighbor_expansion_disabled_by_default(self):
        args = retrieval_eval.build_parser().parse_args([])
        self.assertEqual(args.neighbor_window, 0)
        first = {**chunk('first'), 'metadata': {'chunk_index': 0}}
        second = {**chunk('second'), 'metadata': {'chunk_index': 1}}
        scored = retrieval_eval.score_results([first], {'answerable': True,
            'gold': [{'source_id': 'doc', 'page': 2, 'evidence': 'Evidence phrase'}]},
            context_candidates=retrieval_eval.expand_adjacent_context([first], [first, second], args.neighbor_window))
        self.assertEqual(scored['context_ids'], ['first'])

    def test_unanswerable_is_outside_retrieval_denominator(self):
        result = retrieval_eval.score_results([chunk()], {'answerable': False})
        self.assertIsNone(result['hit1'])
        self.assertIsNone(result['hit3'])

    def test_metadata_filter_shape(self):
        flt = {'field': 'section', 'op': 'eq', 'value': '3.2'}
        self.assertTrue(retrieval_eval.matches_filter(chunk(), flt))
        self.assertFalse(retrieval_eval.matches_filter(chunk(section='4.2'), flt))
        with self.assertRaises(ValueError):
            retrieval_eval.matches_filter(chunk(), {'field': 'section', 'op': 'contains', 'value': '3'})

    def test_rrf_fuses_deduplicates_filters_and_orders_deterministically(self):
        a, b, c = chunk('a'), chunk('b'), chunk('c', section='other')
        results = retrieval_eval.reciprocal_rank_fusion(
            [[a, b, c], [{**b, 'text': 'dense payload'}, a]], 3, filters={
                'field': 'section', 'op': 'eq', 'value': '3.2'})
        self.assertEqual([item['id'] for item in results], ['a', 'b'])
        self.assertEqual(results[0]['score'], results[1]['score'])
        self.assertEqual(results[0]['text'], a['text'])
        self.assertTrue(all(item['strategy'] == 'hybrid' for item in results))

    def test_rrf_limits_each_ranker_to_top_k_and_validates_parameters(self):
        results = retrieval_eval.reciprocal_rank_fusion([[chunk('a'), chunk('b')]], 1)
        self.assertEqual([item['id'] for item in results], ['a'])
        with self.assertRaises(ValueError):
            retrieval_eval.reciprocal_rank_fusion([], 0)

    def test_default_methods_are_lexical_and_dense_is_not_initialized(self):
        args = retrieval_eval.build_parser().parse_args([])
        self.assertEqual(args.methods, 'tfidf,bm25')
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.budget, 5000)
        self.assertEqual(args.rrf_constant, 60)
        with tempfile.TemporaryDirectory() as directory:
            benchmark = Path(directory) / 'benchmark.json'
            benchmark.write_text(json.dumps([{'id': 'q', 'split': 'dev', 'question': 'Evidence phrase',
                                               'gold': [{'source_id': 'doc', 'page': 2,
                                                        'evidence': 'Evidence phrase'}]}]), encoding='utf-8')
            fake_client = Mock()
            fake_client.scroll.return_value = ([SimpleNamespace(id='p', payload={
                'chunk_id': 'a', 'document_id': 'doc', 'source': 'doc.pdf', 'page': 2,
                'text': 'Evidence phrase appears here.', 'metadata': {'section': '3.2'}})], None)
            fake_qdrant = ModuleType('qdrant_client')
            fake_qdrant.QdrantClient = Mock(return_value=fake_client)
            fake_adapters = ModuleType('adapters')
            fake_adapters.BedrockCohereEmbedder = Mock()
            with patch.dict(sys.modules, {'qdrant_client': fake_qdrant}), \
                    patch.dict(sys.modules, {'adapters': fake_adapters}):
                retrieval_eval.main(['--benchmark', str(benchmark), '--output', directory])
            fake_adapters.BedrockCohereEmbedder.assert_not_called()
            fake_client.scroll.assert_called_once()
            self.assertTrue((Path(directory) / 'dev_summary.json').exists())


if __name__ == '__main__':
    unittest.main()
