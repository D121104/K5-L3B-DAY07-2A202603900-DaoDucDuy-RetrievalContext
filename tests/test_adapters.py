"""Mocked adapter contract tests; no provider or database requests are made."""
import io
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from adapters import BedrockCohereEmbedder, GeminiController, QdrantRetriever


class AdapterTests(unittest.TestCase):
    def test_gemini_pacing_configuration_and_per_call_spacing(self):
        old_value = os.environ.get('GEMINI_MIN_INTERVAL_SECONDS')
        try:
            os.environ['GEMINI_MIN_INTERVAL_SECONDS'] = '4.2'
            controller = GeminiController(client=Mock())
            self.assertEqual(controller.min_interval_seconds, 4.2)
        finally:
            if old_value is None:
                os.environ.pop('GEMINI_MIN_INTERVAL_SECONDS', None)
            else:
                os.environ['GEMINI_MIN_INTERVAL_SECONDS'] = old_value

        now = [0.0]
        starts = []

        def fake_sleep(duration):
            now[0] += duration

        client = Mock()
        client.models.generate_content.side_effect = lambda **kwargs: (
            starts.append(now[0]) or SimpleNamespace(candidates=[
                SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(
                    function_call=SimpleNamespace(name='search', args={'query': 'q'}))]))]))
        controller = GeminiController(client=client, min_interval_seconds=4.15,
                                      _clock=lambda: now[0], _sleep=fake_sleep)
        controller.decide('question', [])
        controller.decide('question', [])
        controller.decide('question', [])
        self.assertEqual(starts, [0.0, 4.15, 8.3])
        self.assertEqual(client.models.generate_content.call_count, 3)

    def test_gemini_retries_only_429_with_bounded_backoff_and_pacing(self):
        now = [0.0]
        sleeps = []
        starts = []

        def sleep(duration):
            sleeps.append(duration)
            now[0] += duration

        client = Mock()
        client.models.generate_content.side_effect = [
            RuntimeError('HTTP 429 rate limit'),
            RuntimeError('HTTP 429 rate limit'),
            SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
                SimpleNamespace(function_call=SimpleNamespace(name='search', args={'query': 'q'}))]))]),
        ]
        original = client.models.generate_content.side_effect

        def generate(**kwargs):
            starts.append(now[0])
            result = original.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        client.models.generate_content.side_effect = generate
        controller = GeminiController(client=client, min_interval_seconds=3,
                                      _clock=lambda: now[0], _sleep=sleep)
        self.assertEqual(controller.decide('question', []), {'action': 'search', 'query': 'q'})
        self.assertEqual(sleeps, [1.0, 2.0, 2.0, 1.0])
        self.assertEqual(starts, [0.0, 3.0, 8.0])
        self.assertEqual(client.models.generate_content.call_count, 3)

    def test_gemini_does_not_retry_non_rate_limit_errors(self):
        client = Mock()
        client.models.generate_content.side_effect = RuntimeError('service unavailable')
        controller = GeminiController(client=client)
        with self.assertRaisesRegex(RuntimeError, 'service unavailable'):
            controller.decide('question', [])
        client.models.generate_content.assert_called_once()

    def test_answer_prompt_preserves_precision_units_and_requested_parts(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(candidates=[], text=json.dumps({
            'answer': '4,0 điểm [known]', 'citations': ['known']}))
        controller = GeminiController(client=client)
        controller.decide('Compare both requested values', [
            {'context': '[known] 4,0 điểm', 'context_ids': ['known']}])
        prompt = client.models.generate_content.call_args.kwargs['contents']
        self.assertIn('Answer every requested part', prompt)
        self.assertIn('Check each requested part and condition separately', prompt)
        self.assertIn('cite evidence for each factual claim next to that claim', prompt)
        self.assertIn('abstain rather than guess', prompt)
        self.assertIn('Preserve units, numeric values, and decimal precision exactly', prompt)
        self.assertIn('do not round or drop trailing decimal zeros', prompt)

    def test_bedrock_query_uses_search_query_and_validates_dimensions(self):
        vector = [0.1] * 1536
        runtime = Mock()
        runtime.invoke_model.return_value = {'body': io.BytesIO(json.dumps({'embeddings': [vector]}).encode())}
        embedder = BedrockCohereEmbedder(client=runtime)
        self.assertEqual(embedder.embed_query('question'), vector)
        kwargs = runtime.invoke_model.call_args.kwargs
        self.assertEqual(kwargs['modelId'], 'global.cohere.embed-v4:0')
        self.assertEqual(json.loads(kwargs['body'])['input_type'], 'search_query')
        runtime.invoke_model.return_value = {'body': io.BytesIO(json.dumps({'embeddings': [[0.1]]}).encode())}
        with self.assertRaisesRegex(ValueError, '1536'):
            embedder.embed_query('question')

    def test_qdrant_payload_maps_to_agent_contract(self):
        embedder = Mock()
        embedder.embed_query.return_value = [0.0] * 1536
        point = SimpleNamespace(id='point-id', score=0.82, payload={
            'chunk_id': 'chunk-1', 'text': 'Evidence', 'document_id': 'doc-1',
            'source': 'guide.pdf', 'page': 4, 'metadata': {'year': 2025}})
        client = Mock()
        client.query_points.return_value = SimpleNamespace(points=[point])
        retriever = QdrantRetriever(embedder, client=client, collection='AI20K', strategy='B')
        result = retriever.search('question', filters=None, top_k=2)
        self.assertEqual(retriever.collection, 'AI20K_B')
        self.assertEqual(result, [{'id': 'chunk-1', 'text': 'Evidence', 'source_id': 'doc-1',
                                  'source': 'guide.pdf', 'page': 4, 'metadata': {'year': 2025}, 'score': 0.82}])
        self.assertEqual(client.query_points.call_args.kwargs['limit'], 2)

    def test_qdrant_normalizes_benchmark_section_filter(self):
        from qdrant_client import models

        client = Mock()
        client.query_points.return_value = SimpleNamespace(points=[])
        embedder = Mock()
        embedder.embed_query.return_value = [0.0]
        retriever = QdrantRetriever(embedder, client=client, collection='custom')
        retriever.search('question', filters={
            'field': 'section', 'op': 'eq', 'value': '3.1.4'}, top_k=1)
        query_filter = client.query_points.call_args.kwargs['query_filter']
        self.assertEqual(query_filter, models.Filter(must=[models.FieldCondition(
            key='metadata.section', match=models.MatchValue(value='3.1.4'))]))

    def test_qdrant_rejects_unsupported_benchmark_filter_operator(self):
        retriever = QdrantRetriever(Mock(), client=Mock(), collection='custom')
        with self.assertRaisesRegex(ValueError, 'Benchmark filters'):
            retriever.search('question', filters={
                'field': 'section', 'op': 'contains', 'value': '3.1'}, top_k=1)

    def test_qdrant_accepts_native_filter_and_filter_dictionary(self):
        from qdrant_client import models

        retriever = QdrantRetriever(Mock(), client=Mock(), collection='custom')
        native_filter = models.Filter(must=[models.FieldCondition(
            key='metadata.section', match=models.MatchValue(value='3.1.4'))])
        for filters in (native_filter, {'must': [{'key': 'metadata.section',
                                                  'match': {'value': '3.1.4'}}]}):
            with self.subTest(filters=filters):
                retriever.client.query_points.return_value = SimpleNamespace(points=[])
                retriever.search('question', filters=filters, top_k=1)
                self.assertEqual(retriever.client.query_points.call_args.kwargs['query_filter'],
                                 native_filter)

    def test_qdrant_explicit_collection_is_not_rewritten(self):
        retriever = QdrantRetriever(Mock(), client=Mock(), collection='custom_collection', strategy='A')
        self.assertEqual(retriever.collection, 'custom_collection')

    def test_qdrant_neighbor_scroll_is_cached_and_scoped(self):
        def point(cid, index, page=4, source='doc-1', strategy='B'):
            return SimpleNamespace(id=cid, score=None, payload={
                'chunk_id': cid, 'text': cid, 'document_id': source, 'page': page,
                'metadata': {'chunk_index': index, 'chunk_strategy': strategy}})
        client = Mock()
        client.query_points.return_value = SimpleNamespace(points=[point('middle', 1)])
        client.scroll.return_value = ([point('left', 0), point('middle', 1), point('right', 2),
                                       point('wrong-page', 2, page=5), point('wrong-source', 2, source='other'),
                                       point('wrong-strategy', 2, strategy='A')], None)
        embedder = Mock()
        embedder.embed_query.return_value = [0.0]
        retriever = QdrantRetriever(embedder, client=client, collection='custom', neighbor_window=1)
        result = retriever.search('q', top_k=1)
        self.assertEqual([chunk['id'] for chunk in result[0]['_neighbors']], ['left', 'right'])
        retriever.search('q', top_k=1)
        client.scroll.assert_called_once_with(collection_name='custom', offset=None, with_payload=True,
                                             with_vectors=False, limit=256)

    def test_gemini_normalizes_function_call_and_citations(self):
        client = Mock()
        call_response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
            SimpleNamespace(function_call=SimpleNamespace(name='search', args={'query': ' refined query '}))]))])
        answer_response = SimpleNamespace(candidates=[], text=json.dumps({
            'answer': 'The policy says seven days [chunk-1].', 'citations': ['chunk-1']}))
        client.models.generate_content.side_effect = [call_response, answer_response]
        controller = GeminiController(client=client, model='gemini-3.8-flash')
        self.assertEqual(controller.model, 'gemini-3.8-flash')
        self.assertEqual(controller.decide('question', []), {'action': 'search', 'query': 'refined query'})
        initial_config = client.models.generate_content.call_args.kwargs['config']
        initial_tools = (initial_config.get('tools') if isinstance(initial_config, dict)
                         else getattr(initial_config, 'tools', None))
        self.assertIn(controller.tool, initial_tools)
        history = [{'tool': 'search', 'context': '[chunk-1] evidence', 'context_ids': ['chunk-1']}]
        self.assertEqual(controller.decide('question', history), {
            'action': 'answer', 'answer': 'The policy says seven days [chunk-1].', 'citations': ['chunk-1']})
        answer_call = client.models.generate_content.call_args.kwargs
        self.assertIn('[chunk-1] evidence', answer_call['contents'])
        answer_config = answer_call['config']
        answer_tools = (answer_config.get('tools') if isinstance(answer_config, dict)
                        else getattr(answer_config, 'tools', None))
        mime_type = (answer_config.get('response_mime_type') if isinstance(answer_config, dict)
                     else getattr(answer_config, 'response_mime_type', None))
        self.assertTrue(answer_tools)
        self.assertEqual(mime_type, 'application/json')
        self.assertIn('Return JSON', answer_call['contents'])
        self.assertIn('["chunk-1"]', answer_call['contents'])
        self.assertIn('5–7%', answer_call['contents'])

    def test_gemini_can_search_again_after_receiving_context(self):
        client = Mock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
                SimpleNamespace(function_call=SimpleNamespace(name='search', args={'query': 'missing table'}))]))]),
            SimpleNamespace(candidates=[], text=json.dumps({'answer': 'Supported [chunk-2]',
                                                             'citations': ['chunk-2']})),
        ]
        controller = GeminiController(client=client)
        history = [{'tool': 'search', 'context': '[chunk-1] partial evidence',
                    'context_ids': ['chunk-1']}]
        self.assertEqual(controller.decide('multi-part question', history),
                         {'action': 'search', 'query': 'missing table'})
        call = client.models.generate_content.call_args
        self.assertIn('focused query', call.kwargs['contents'])
        config = call.kwargs['config']
        tools = config.get('tools') if isinstance(config, dict) else getattr(config, 'tools', None)
        self.assertTrue(tools)
        updated = [{'tool': 'search', 'context': '[chunk-1] partial evidence\n[chunk-2] remaining evidence',
                    'context_ids': ['chunk-1', 'chunk-2']}]
        self.assertEqual(controller.decide('multi-part question', updated),
                         {'action': 'answer', 'answer': 'Supported [chunk-2]', 'citations': ['chunk-2']})

    def test_gemini_malformed_structured_response_raises_safe_diagnostic(self):
        secret = 'not json api_key=do-not-leak'
        for raw in (secret, json.dumps({'answer': 7, 'citations': []}),
                    json.dumps({'answer': 'Text', 'citations': 'known'})):
            with self.subTest(raw=raw):
                client = Mock()
                client.models.generate_content.return_value = SimpleNamespace(candidates=[], text=raw)
                controller = GeminiController(client=client)
                with self.assertRaisesRegex(ValueError, 'malformed structured answer') as raised:
                    controller.decide('question', [{'context': 'evidence', 'context_ids': ['known']}])
                self.assertIn('response content omitted', str(raised.exception))
                self.assertNotIn('do-not-leak', str(raised.exception))

    def test_gemini_empty_generation_normalizes_to_empty_answer(self):
        for response in (SimpleNamespace(candidates=[], text=None),
                         SimpleNamespace(candidates=None, text=''),
                         SimpleNamespace(candidates=[SimpleNamespace(content=None)], text=None)):
            with self.subTest(response=response):
                client = Mock()
                client.models.generate_content.return_value = response
                controller = GeminiController(client=client)
                result = controller.decide('question', [{'context': 'evidence', 'context_ids': ['known']}])
                self.assertEqual(result, {'action': 'answer', 'answer': '', 'citations': []})

    def test_gemini_preserves_unknown_citation_for_agent_validation(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(candidates=[], text=json.dumps({
            'answer': 'Unsupported [invented]', 'citations': ['invented']}))
        controller = GeminiController(client=client)
        self.assertEqual(controller.decide('question', [{'context': 'evidence', 'context_ids': ['known']}]), {
            'action': 'answer', 'answer': 'Unsupported [invented]', 'citations': ['invented']})

    def test_gemini_empty_or_missing_search_query_falls_back_to_question(self):
        for args in ({}, {'query': ''}, {'query': '  '}):
            with self.subTest(args=args):
                client = Mock()
                client.models.generate_content.return_value = SimpleNamespace(candidates=[
                    SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(
                        function_call=SimpleNamespace(name='search', args=args))]))])
                controller = GeminiController(client=client)
                self.assertEqual(controller.decide('original question', []),
                                 {'action': 'search', 'query': 'original question'})


if __name__ == '__main__':
    unittest.main()
