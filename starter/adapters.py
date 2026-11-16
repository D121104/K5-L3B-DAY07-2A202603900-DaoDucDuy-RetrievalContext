"""Bedrock/Cohere, Qdrant, and Gemini adapters for the Day 7 agent."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=False)


class BedrockCohereEmbedder:
    """Create Cohere Embed v4 query vectors through the Bedrock Runtime API."""

    def __init__(self, *, model: str | None = None, region_name: str | None = None,
                 client: Any = None, timeout: float = 30):
        self.model = model or os.getenv('EMBEDDING_MODEL', 'global.cohere.embed-v4:0')
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client('bedrock-runtime', region_name=region_name or os.getenv('AWS_REGION'),
                                  config=Config(connect_timeout=timeout, read_timeout=timeout,
                                                retries={'max_attempts': 2}))
        self.client = client

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a non-empty string')
        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps({'texts': [query], 'input_type': 'search_query', 'embedding_types': ['float'],
                             'output_dimension': 1536}),
            contentType='application/json', accept='application/json')
        body = response['body']
        payload = json.loads(body.read() if hasattr(body, 'read') else body)
        embeddings = payload.get('embeddings')
        if isinstance(embeddings, dict):
            embeddings = embeddings.get('float')
        vector = embeddings[0] if isinstance(embeddings, list) and embeddings else None
        if not isinstance(vector, list) or len(vector) != 1536:
            raise ValueError('Bedrock Cohere response must contain one 1536-dimensional embedding')
        return vector


class QdrantRetriever:
    """Retrieve normalized chunk records from a Qdrant collection."""

    def __init__(self, embedder: BedrockCohereEmbedder | Any, *, collection: str | None = None,
                 strategy: str | None = None, url: str | None = None, api_key: str | None = None,
                 client: Any = None, timeout: float = 30, neighbor_window: int = 0):
        if neighbor_window < 0:
            raise ValueError('neighbor_window must be non-negative')
        configured = collection or os.getenv('QDRANT_COLLECTION', 'AI20K')
        if configured == 'AI20K':
            suffix = (strategy or 'B').strip('_')
            configured = f'AI20K_{suffix}'
        self.collection = configured
        self.embedder = embedder
        if client is None:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=url or os.getenv('QDRANT_URL'),
                                  api_key=api_key or os.getenv('QDRANT_API_KEY'), timeout=timeout)
        self.client = client
        self.neighbor_window = neighbor_window
        self._chunk_cache: list[dict[str, Any]] | None = None

    @staticmethod
    def _record(point: Any) -> dict[str, Any]:
        payload = point.payload or {}
        return {'id': str(payload.get('chunk_id', point.id)), 'text': payload.get('text', ''),
                'source_id': payload.get('document_id'), 'source': payload.get('source'),
                'page': payload.get('page'), 'metadata': payload.get('metadata') or {},
                'score': getattr(point, 'score', None)}

    def _all_chunks(self) -> list[dict[str, Any]]:
        if self._chunk_cache is None:
            points, offset = [], None
            while True:
                batch, offset = self.client.scroll(collection_name=self.collection, offset=offset,
                                                   with_payload=True, with_vectors=False, limit=256)
                points.extend(batch)
                if offset is None:
                    break
            self._chunk_cache = [self._record(point) for point in points]
        return self._chunk_cache

    @staticmethod
    def _strategy(chunk: dict[str, Any]) -> Any:
        metadata = chunk.get('metadata') or {}
        return metadata.get('strategy', metadata.get('chunk_strategy'))

    def _neighbors(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        index = (chunk.get('metadata') or {}).get('chunk_index')
        if not isinstance(index, int):
            return []
        by_index = {}
        for candidate in self._all_chunks():
            metadata = candidate.get('metadata') or {}
            candidate_index = metadata.get('chunk_index')
            if (candidate.get('source_id') == chunk.get('source_id')
                    and candidate.get('page') == chunk.get('page')
                    and self._strategy(candidate) == self._strategy(chunk)
                    and isinstance(candidate_index, int)):
                by_index[candidate_index] = candidate
        return [by_index[index + delta] for distance in range(1, self.neighbor_window + 1)
                for delta in (-distance, distance) if index + delta in by_index]

    def search(self, query: str, *, filters: dict[str, Any] | Any | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError('top_k must be positive')
        from qdrant_client import models
        query_filter = filters
        if isinstance(filters, dict):
            benchmark_keys = {'field', 'op', 'value'}
            if benchmark_keys.intersection(filters):
                value = filters.get('value')
                if (set(filters) != benchmark_keys or filters.get('field') != 'section'
                        or filters.get('op') != 'eq'
                        or not isinstance(value, (str, int, float, bool))
                        or isinstance(value, str) and not value.strip()):
                    raise ValueError('Benchmark filters must be {field: "section", op: "eq", value: ...}')
                query_filter = models.Filter(must=[models.FieldCondition(
                    key='metadata.section', match=models.MatchValue(value=filters['value']))])
            else:
                try:
                    query_filter = models.Filter(**filters)
                except (TypeError, ValueError) as exc:
                    raise ValueError('filters must be a valid Qdrant Filter dictionary') from exc
        elif filters is not None and not isinstance(filters, models.Filter):
            raise ValueError('filters must be a Qdrant Filter or dictionary')
        points = self.client.query_points(collection_name=self.collection,
                                          query=self.embedder.embed_query(query), query_filter=query_filter,
                                          limit=top_k, with_payload=True).points
        results = [self._record(point) for point in points]
        if self.neighbor_window:
            for result in results:
                result['_neighbors'] = self._neighbors(result)
        return results


class GeminiController:
    """Use Gemini function calling to choose retrieval or produce a cited answer."""

    MAX_RATE_LIMIT_RETRIES = 2
    RATE_LIMIT_BACKOFF_SECONDS = 1.0

    def __init__(self, *, api_key: str | None = None, model: str | None = None,
                 client: Any = None, timeout: float = 30,
                 min_interval_seconds: float | None = None, _clock: Any = None,
                 _sleep: Any = None):
        key = api_key or os.getenv('GEMINI_API_KEY')
        if client is None:
            if not key:
                raise ValueError('GEMINI_API_KEY is required')
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        self.client = client
        self.model = model or os.getenv('GEMINI_MODEL', 'gemini-3.8-flash')
        interval = (float(os.getenv('GEMINI_MIN_INTERVAL_SECONDS', '4.15'))
                    if min_interval_seconds is None else float(min_interval_seconds))
        if interval < 0:
            raise ValueError('GEMINI_MIN_INTERVAL_SECONDS must be non-negative')
        self.min_interval_seconds = interval
        self._clock = _clock or time.monotonic
        self._sleep = _sleep or time.sleep
        self._pacing_lock = threading.Lock()
        self._last_request_at: float | None = None
        try:
            from google.genai import types
            self.tool = types.Tool(function_declarations=[types.FunctionDeclaration(
                name='search', description='Search the document collection for evidence.',
                parameters={'type': 'OBJECT', 'properties': {'query': {'type': 'STRING'}},
                            'required': ['query']})])
            self.config = types.GenerateContentConfig(tools=[self.tool], tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='AUTO')))
            self.answer_config = types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=types.Schema(type='OBJECT', properties={
                    'answer': types.Schema(type='STRING'),
                    'citations': types.Schema(type='ARRAY', items=types.Schema(type='STRING')),
                }, required=['answer', 'citations']))
            self.followup_config = types.GenerateContentConfig(
                tools=[self.tool], tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode='AUTO')),
                response_mime_type='application/json',
                response_schema=types.Schema(type='OBJECT', properties={
                    'answer': types.Schema(type='STRING'),
                    'citations': types.Schema(type='ARRAY', items=types.Schema(type='STRING')),
                }, required=['answer', 'citations']))
        except ImportError:
            self.tool = {'function_declarations': [{'name': 'search', 'description': 'Search for evidence.',
                       'parameters': {'type': 'OBJECT', 'properties': {'query': {'type': 'STRING'}},
                                      'required': ['query']}}]}
            self.config = {'tools': [self.tool]}
            self.answer_config = {'response_mime_type': 'application/json', 'response_schema': {
                'type': 'OBJECT', 'properties': {'answer': {'type': 'STRING'},
                    'citations': {'type': 'ARRAY', 'items': {'type': 'STRING'}}},
                'required': ['answer', 'citations']}}
            self.followup_config = {
                **self.config,
                **self.answer_config,
            }

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        has_context = any(isinstance(item, Mapping) and isinstance(item.get('context'), str)
                          and item['context'].strip() for item in history)
        if has_context:
            context_ids = list(dict.fromkeys(str(identifier) for item in history
                                              for identifier in item.get('context_ids', [])
                                              if isinstance(identifier, str)))
            prompt = ('Check each requested part and condition separately against the evidence. If evidence is '
                      'missing for any part, call search with a focused query for that missing evidence. After '
                      'searching, if any requested part remains unsupported, abstain rather than guess or present '
                      'an incomplete answer as complete. Otherwise Return JSON with answer (string) and citations '
                      '(array of exact chunk-ID strings). Answer every requested part and preserve its conditions. '
                      'Preserve units, numeric values, and decimal precision exactly as stated; do not round or '
                      'drop trailing decimal zeros. For visual evidence, use only details explicit in the provided '
                      'evidence and do not infer unshown map or diagram details. Give a concise direct answer; do not '
                      'abstain merely because the supported answer is brief; cite evidence for each factual claim '
                      'next to that claim, include every supporting ID in citations, and mark each cited ID in answer '
                      'as [chunk-id]. Never cite a chunk that does not support the claim it accompanies. Use only IDs from this list: '
                      f'{json.dumps(context_ids, ensure_ascii=False)}. Example: {json.dumps({"answer": "5–7% [chunk-id]", "citations": ["chunk-id"]}, ensure_ascii=False)}. '
                      'Abstain when evidence genuinely does not contain the complete answer, using an empty citations '
                      'array. Do not guess.\n\nQuestion: ' + question)
        else:
            prompt = ('You answer questions using retrieved document evidence. Call the search function when '
                      'more evidence is needed. Answer only from the provided evidence and cite each supporting '
                      'chunk using its exact ID in square brackets, such as [chunk-id]. If evidence is '
                      'insufficient, search again rather than guessing.\n\nQuestion: ' + question)
        if history:
            prompt += '\n\nRetrieval history and evidence (use only this evidence to answer):\n' + json.dumps(
                history, ensure_ascii=False)
        for attempt in range(self.MAX_RATE_LIMIT_RETRIES + 1):
            with self._pacing_lock:
                now = self._clock()
                if self._last_request_at is not None:
                    wait = self.min_interval_seconds - (now - self._last_request_at)
                    if wait > 0:
                        self._sleep(wait)
                self._last_request_at = self._clock()
                try:
                    response = self.client.models.generate_content(
                        model=self.model, contents=prompt,
                        config=self.followup_config if has_context else self.config)
                except Exception as error:
                    if not self._is_rate_limit_error(error) or attempt >= self.MAX_RATE_LIMIT_RETRIES:
                        raise
                    self._sleep(self.RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt))
                    continue
            break
        for candidate in getattr(response, 'candidates', None) or []:
            content = getattr(candidate, 'content', None)
            for part in getattr(content, 'parts', None) or []:
                call = getattr(part, 'function_call', None)
                if call and call.name == 'search':
                    args = call.args or {}
                    query = args.get('query') if isinstance(args, Mapping) else None
                    if not isinstance(query, str) or not query.strip():
                        query = question
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError('Gemini search function call requires a non-empty query')
                    return {'action': 'search', 'query': query.strip()}
        if has_context:
            raw_text = getattr(response, 'text', None)
            if not isinstance(raw_text, str) or not raw_text.strip():
                return {'action': 'answer', 'answer': '', 'citations': []}
            try:
                payload = json.loads(raw_text) if isinstance(raw_text, str) else None
            except (json.JSONDecodeError, TypeError):
                payload = None
            if (not isinstance(payload, Mapping)
                    or not isinstance(payload.get('answer'), str)
                    or not isinstance(payload.get('citations'), list)
                    or any(not isinstance(identifier, str) for identifier in payload['citations'])):
                text_length = len(raw_text) if isinstance(raw_text, str) else None
                raise ValueError(
                    'Gemini returned malformed structured answer '
                    f'(response_text_type={type(raw_text).__name__}, text_length={text_length}, '
                    f'candidate_count={len(response.candidates or [])}); response content omitted')
            answer = payload['answer'].strip()
            citations = list(dict.fromkeys(payload['citations']))
        else:
            answer = (getattr(response, 'text', None) or '').strip()
            citations = list(dict.fromkeys(re.findall(r'\[([^\[\]]+)\]', answer)))
        return {'action': 'answer', 'answer': answer, 'citations': citations}

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        status = getattr(error, 'status_code', None) or getattr(error, 'code', None)
        if status == 429:
            return True
        return bool(re.search(r'\b429\b|rate.?limit|resource.?exhausted', str(error), re.IGNORECASE))
