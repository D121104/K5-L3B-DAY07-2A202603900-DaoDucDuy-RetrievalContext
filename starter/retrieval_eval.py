"""Offline lexical and opt-in Bedrock dense retrieval evaluation for Day 7."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from baseline import BM25Index, TfidfIndex, pack_context

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'data' / 'logistics_benchmark.json'
load_dotenv(ROOT / '.env', override=False)


def matches_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    if filters.get('op') != 'eq' or not filters.get('field'):
        raise ValueError('Only field=..., op=eq, value=... filters are supported')
    return chunk.get('metadata', {}).get(filters['field']) == filters.get('value')


def normalize_evidence(text: str) -> str:
    """Normalize formatting only; evidence matching remains an exact substring check."""
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def expand_adjacent_context(results: list[dict[str, Any]], corpus: list[dict[str, Any]],
                           window: int = 1) -> list[dict[str, Any]]:
    """Group each ranked hit with its same-page contiguous neighbors."""
    if window < 0:
        raise ValueError('window must be non-negative')
    by_location = {}
    for chunk in corpus:
        index = chunk.get('metadata', {}).get('chunk_index')
        if isinstance(index, int):
            by_location[(chunk.get('source_id'), chunk.get('page'), index)] = chunk
    expanded, seen = [], set()
    for chunk in results:
        group = [chunk]
        index = chunk.get('metadata', {}).get('chunk_index')
        if window and isinstance(index, int):
            for distance in range(1, window + 1):
                for neighbor_index in (index - distance, index + distance):
                    neighbor = by_location.get((chunk.get('source_id'), chunk.get('page'), neighbor_index))
                    if neighbor is not None:
                        group.append(neighbor)
        group.sort(key=lambda item: item.get('metadata', {}).get('chunk_index', float('inf')))
        for candidate in group:
            identifier = str(candidate['id'])
            if identifier not in seen:
                expanded.append(candidate)
                seen.add(identifier)
    return expanded


def stitched_matches(results: list[dict[str, Any]], gold: dict[str, Any]) -> set[str]:
    """Return retrieved chunk IDs participating in contiguous same-page evidence."""
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for chunk in results:
        if chunk.get('source_id') == gold.get('source_id') and chunk.get('page') == gold.get('page'):
            groups.setdefault((chunk.get('source_id'), chunk.get('page')), []).append(chunk)

    evidence = normalize_evidence(gold.get('evidence', ''))
    matched_ids: set[str] = set()
    for chunks in groups.values():
        chunks.sort(key=lambda chunk: chunk.get('metadata', {}).get('chunk_index', -1))
        run: list[dict[str, Any]] = []
        previous_index = None
        for chunk in chunks:
            index = chunk.get('metadata', {}).get('chunk_index')
            if run and (not isinstance(index, int) or index != previous_index + 1):
                if evidence and evidence in normalize_evidence('\n'.join(c.get('text', '') for c in run)):
                    matched_ids.update(c['id'] for c in run if 'id' in c)
                run = []
            run.append(chunk)
            previous_index = index
        if run and evidence and evidence in normalize_evidence('\n'.join(c.get('text', '') for c in run)):
            matched_ids.update(c['id'] for c in run if 'id' in c)
    return matched_ids


def score_results(results: list[dict[str, Any]], question: dict[str, Any], budget: int = 800,
                  context_candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    context_candidates = results if context_candidates is None else context_candidates
    context, context_ids = pack_context(context_candidates, budget)
    answerable = question.get('answerable', True)
    gold = question.get('gold', [])
    stitched_ids = set().union(*(stitched_matches(results, item) for item in gold)) if gold else set()
    context_chunks = [chunk for chunk in context_candidates if chunk.get('id') in context_ids]
    context_stitched_ids = set().union(*(stitched_matches(context_chunks, item) for item in gold)) if gold else set()
    stitched_ids1 = set().union(*(stitched_matches(results[:1], item) for item in gold)) if gold else set()
    stitched_ids3 = set().union(*(stitched_matches(results[:3], item) for item in gold)) if gold else set()
    marked = []
    for chunk in results:
        eligible_gold = [item for item in gold if
            chunk.get('source_id') == item.get('source_id')
            and chunk.get('page') == item.get('page')
        ]
        text = chunk.get('text', '')
        exact_hit = answerable and any(item.get('evidence', '') in text for item in eligible_gold)
        normalized_text = normalize_evidence(text)
        normalized_hit = answerable and any(
            normalize_evidence(item.get('evidence', '')) in normalized_text for item in eligible_gold
        )
        marked.append({**chunk, 'hit': bool(normalized_hit), 'normalized_hit': bool(normalized_hit),
                       'exact_hit': bool(exact_hit), 'stitched_hit': bool(chunk.get('id') in stitched_ids)})
    hits = [chunk for chunk in marked if chunk['hit']]
    exact_hits = [chunk for chunk in marked if chunk['exact_hit']]
    context_marked = []
    for chunk in context_candidates:
        eligible_gold = [item for item in gold if chunk.get('source_id') == item.get('source_id')
                         and chunk.get('page') == item.get('page')]
        text = normalize_evidence(chunk.get('text', ''))
        context_marked.append({**chunk,
                               'hit': answerable and any(normalize_evidence(item.get('evidence', '')) in text
                                                         for item in eligible_gold),
                               'exact_hit': answerable and any(item.get('evidence', '') in chunk.get('text', '')
                                                               for item in eligible_gold)})
    hits3 = marked[:3]
    exact_hits3 = [chunk for chunk in hits3 if chunk['exact_hit']]
    return {
        'results': marked,
        'context': context,
        'context_ids': context_ids,
        'context_chars': len(context),
        'answerable': answerable,
        'hit1': bool(marked and marked[0]['hit']) if answerable else None,
        'hit3': any(chunk['hit'] for chunk in hits3) if answerable else None,
        'hitk': bool(hits) if answerable else None,
        'context_hit': any(c['id'] in context_ids and c['hit'] for c in context_marked) if answerable else None,
        'exact_hit1': bool(marked and marked[0]['exact_hit']) if answerable else None,
        'exact_hit3': bool(exact_hits3) if answerable else None,
        'exact_hitk': bool(exact_hits) if answerable else None,
        'exact_context_hit': any(c['id'] in context_ids and c['exact_hit'] for c in context_marked) if answerable else None,
        'stitched_hit': bool(stitched_ids) if answerable else None,
        'stitched_hit1': bool(stitched_ids1) if answerable else None,
        'stitched_hit3': bool(stitched_ids3) if answerable else None,
        'stitched_hitk': bool(stitched_ids) if answerable else None,
        'stitched_context_hit': bool(context_stitched_ids) if answerable else None,
    }


def point_to_chunk(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    metadata = payload.get('metadata') or {}
    # Day 6's Qdrant payload stores chunk metadata flat (including section).
    if not metadata:
        metadata = {key: value for key, value in payload.items()
                    if key not in {'chunk_id', 'document_id', 'source', 'page', 'text'}}
    return {'id': str(payload.get('chunk_id', point.id)), 'text': payload.get('text', ''),
            'source_id': payload.get('document_id'), 'source': payload.get('source'),
            'page': payload.get('page'), 'metadata': metadata}


def scroll_corpus(client: Any, collection: str = 'AI20K_B', *, with_vectors: bool = False) -> tuple[list[dict[str, Any]], list[list[float]]]:
    chunks, vectors, offset = [], [], None
    while True:
        points, next_offset = client.scroll(collection_name=collection, offset=offset,
                                            limit=256, with_payload=True, with_vectors=with_vectors)
        for point in points:
            chunks.append(point_to_chunk(point))
            if with_vectors:
                vector = point.vector
                if isinstance(vector, dict):
                    vector = vector.get('default') or next(iter(vector.values()))
                vectors.append(vector)
        if next_offset is None:
            break
        offset = next_offset
    return chunks, vectors


class DenseIndex:
    def __init__(self, chunks: list[dict[str, Any]], vectors: list[list[float]], embedder: Any):
        self.chunks, self.vectors, self.embedder = chunks, vectors, embedder

    def search(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)
        ranked = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if not matches_filter(chunk, filters):
                continue
            norm = sum(value * value for value in vector) ** 0.5
            query_norm = sum(value * value for value in query_vector) ** 0.5
            score = sum(a * b for a, b in zip(query_vector, vector)) / (norm * query_norm) if norm and query_norm else 0
            if score > 0:
                ranked.append({**chunk, 'score': score, 'strategy': 'dense'})
        return sorted(ranked, key=lambda item: (-item['score'], item['id']))[:top_k]


def reciprocal_rank_fusion(rankings: list[list[dict[str, Any]]], top_k: int,
                           rrf_constant: int = 60,
                           filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Fuse ranked chunks, retaining the first payload for each ID."""
    if top_k < 1 or rrf_constant < 0:
        raise ValueError('top_k must be positive and rrf_constant non-negative')
    fused: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for chunk in ranking:
            if not matches_filter(chunk, filters):
                continue
            identifier = str(chunk['id'])
            if identifier in seen:
                continue
            if rank >= top_k:
                break
            seen.add(identifier)
            rank += 1
            fused.setdefault(identifier, {**chunk, 'id': identifier, 'strategy': 'hybrid'})
            scores[identifier] = scores.get(identifier, 0.0) + 1 / (rrf_constant + rank)
    return sorted(({**chunk, 'score': scores[identifier]} for identifier, chunk in fused.items()),
                  key=lambda item: (-item['score'], item['id']))[:top_k]


def evaluate_method(name: str, index: Any, questions: list[dict[str, Any]], top_k: int,
                    budget: int, corpus: list[dict[str, Any]] | None = None,
                    neighbor_window: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces = []
    for question in questions:
        filters = question.get('filter')
        started = time.perf_counter()
        if name == 'dense' and question.get('answerable', True):
            results = index.search(question['question'], filters, top_k)
        elif name == 'dense':
            results = []
        elif name == 'hybrid':
            if question.get('answerable', True):
                candidates = [chunk for chunk in index.chunks if matches_filter(chunk, filters)]
                lexical = BM25Index(candidates).search(question['question'], top_k=top_k)
                dense = index.search(question['question'], filters, top_k)
                results = reciprocal_rank_fusion([lexical, dense], top_k, index.rrf_constant, filters)
            else:
                results = []
        else:
            candidates = [chunk for chunk in index.chunks if matches_filter(chunk, filters)]
            local_index = TfidfIndex(candidates) if name == 'tfidf' else BM25Index(candidates)
            results = local_index.search(question['question'], top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        expanded = expand_adjacent_context(results, corpus or [], neighbor_window)
        scored = score_results(results, question, budget, expanded)
        traces.append({'query_id': question['id'], 'split': question['split'], 'question': question['question'],
                       'method': name, 'top_k': top_k, 'filter': filters, 'gold': question.get('gold', []),
                       'neighbor_window': neighbor_window,
                       'context_candidate_ids': [str(chunk['id']) for chunk in expanded],
                       'context_expansion_ids': [str(chunk['id']) for chunk in expanded
                                                 if str(chunk['id']) not in {str(hit['id']) for hit in results}],
                       'context_expansion_count': sum(str(chunk['id']) not in {str(hit['id']) for hit in results}
                                                      for chunk in expanded),
                       **scored, 'elapsed_ms': elapsed_ms})
    answerable = [trace for trace in traces if trace['answerable']]
    summary = {'method': name, 'split': questions[0]['split'] if questions else None,
                'top_k': top_k,
               'queries': len(traces), 'answerable_queries': len(answerable),
               'hit1': sum(t['hit1'] for t in answerable), 'hit1_rate': _rate(answerable, 'hit1'),
                'hit3': sum(t['hit3'] for t in answerable), 'hit3_rate': _rate(answerable, 'hit3'),
                'hitk': sum(t['hitk'] for t in answerable), 'hitk_rate': _rate(answerable, 'hitk'),
                'context_hit': sum(t['context_hit'] for t in answerable),
                'context_hit_rate': _rate(answerable, 'context_hit'),
                'context_evidence_rate': _rate(answerable, 'context_hit'),
                'exact_hit1': sum(t['exact_hit1'] for t in answerable),
                'exact_hit1_rate': _rate(answerable, 'exact_hit1'),
                'exact_hit3': sum(t['exact_hit3'] for t in answerable),
                'exact_hit3_rate': _rate(answerable, 'exact_hit3'),
                'exact_hitk': sum(t['exact_hitk'] for t in answerable),
                'exact_hitk_rate': _rate(answerable, 'exact_hitk'),
                 'exact_context_hit': sum(t['exact_context_hit'] for t in answerable),
                'exact_context_evidence_rate': _rate(answerable, 'exact_context_hit'),
                'stitched_hit': sum(t['stitched_hit'] for t in answerable),
                'stitched_hit_rate': _rate(answerable, 'stitched_hit'),
                'stitched_hit1': sum(t['stitched_hit1'] for t in answerable),
                'stitched_hit1_rate': _rate(answerable, 'stitched_hit1'),
                'stitched_hit3': sum(t['stitched_hit3'] for t in answerable),
                'stitched_hit3_rate': _rate(answerable, 'stitched_hit3'),
                'stitched_hitk': sum(t['stitched_hitk'] for t in answerable),
                'stitched_hitk_rate': _rate(answerable, 'stitched_hitk'),
                'stitched_context_hit': sum(t['stitched_context_hit'] for t in answerable),
                'stitched_context_hit_rate': _rate(answerable, 'stitched_context_hit'),
               'avg_context_chars': statistics.mean(t['context_chars'] for t in traces) if traces else 0,
               'median_latency_ms': statistics.median(t['elapsed_ms'] for t in traces) if traces else 0,
               'service_cost_usd': 'incurred by selected APIs' if name == 'dense' else 0}
    return summary, traces


def _rate(traces: list[dict[str, Any]], metric: str) -> float:
    return sum(bool(t[metric]) for t in traces) / len(traces) if traces else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare lexical retrieval and opt-in Bedrock dense retrieval')
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--methods', default='tfidf,bm25', help='Comma-separated: tfidf,bm25,dense,hybrid (dense/hybrid call Bedrock)')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--rrf-constant', type=int, default=60)
    parser.add_argument('--budget', type=int, default=5000)
    parser.add_argument('--neighbor-window', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'starter' / 'results')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    methods = [method.strip() for method in args.methods.split(',') if method.strip()]
    if not methods or any(method not in {'tfidf', 'bm25', 'dense', 'hybrid'} for method in methods):
        raise SystemExit('--methods must contain tfidf, bm25, dense, and/or hybrid')
    if args.top_k < 1 or args.budget < 1 or args.rrf_constant < 0 or args.neighbor_window < 0:
        raise SystemExit('--top-k and --budget must be positive; --rrf-constant and --neighbor-window must be non-negative')
    # Filter first so dev runs don't validate or inspect held-out test records.
    questions = [q for q in json.loads(args.benchmark.read_text(encoding='utf-8')) if q.get('split') == args.split]
    if not questions:
        raise SystemExit(f'No questions found for split {args.split}')
    client = None
    chunks = None
    from qdrant_client import QdrantClient
    import os
    client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
    needs_dense = 'dense' in methods or 'hybrid' in methods
    chunks, vectors = scroll_corpus(client, with_vectors=needs_dense)
    output = {}
    all_traces = {}
    embedder = None
    if needs_dense:
        from adapters import BedrockCohereEmbedder
        embedder = DenseIndex(chunks, vectors, BedrockCohereEmbedder())
        embedder.rrf_constant = args.rrf_constant
    for method in methods:
        index = embedder if method in {'dense', 'hybrid'} else (TfidfIndex(chunks) if method == 'tfidf' else BM25Index(chunks))
        summary, traces = evaluate_method(method, index, questions, args.top_k, args.budget,
                                         chunks, args.neighbor_window)
        output[method], all_traces[method] = summary, traces
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'{args.split}_summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output / f'{args.split}_traces.json').write_text(json.dumps(all_traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
