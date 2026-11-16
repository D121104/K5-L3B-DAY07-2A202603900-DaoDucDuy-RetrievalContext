"""Example agent loop; connect a real model and the Day 6 retriever via adapters.

Adapters are intentionally not provided: Day 6 has no fixed database schema and
model providers differ. Never use fake adapters to report benchmark quality.
"""
from typing import Any, Protocol
from baseline import pack_context


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action.

        Search: {'action': 'search', 'query': '...'}.
        Final: {'action': 'answer', 'answer': '...', 'citations': ['chunk-id']}.
        Citations must also appear as [chunk-id] in the answer text.
        """
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the real Day 6 vector store with a compatible query embedding.

        Return each chunk's id, text, source_id, source, page, metadata and score.
        The query embedding model and dimension must match the document index.
        """
        ...


def run_agent(question: str, controller: Controller, retriever: Retriever, *,
              filters: dict[str, Any] | None = None, max_searches: int = 4,
              top_k: int = 3, budget: int = 800) -> dict[str, Any]:
    """Return answer, citation IDs and trace; no benchmark gold enters this loop.

    A final answer without valid citations, an empty context, and a search limit
    lead to abstention. This only checks citation IDs, not semantic grounding.
    Provider/retriever errors propagate; no silent mock fallback.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if sum(e['event'] == 'search' for e in events) >= max_searches:
                return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'search_limit'}]}
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            results = retriever.search(query, filters=filters, top_k=top_k)
            if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            primary_results = [{key: value for key, value in result.items() if key != '_neighbors'}
                               for result in results]
            expanded = []
            seen = set()
            for result in results:
                group = [{key: value for key, value in result.items() if key != '_neighbors'}]
                group.extend(result.get('_neighbors', []))
                group.sort(key=lambda chunk: chunk.get('metadata', {}).get('chunk_index', float('inf')))
                for chunk in group:
                    identifier = str(chunk['id'])
                    if identifier not in seen:
                        expanded.append(chunk)
                        seen.add(identifier)
            # Prefer the newest query's evidence when a fixed context budget is tight.
            candidates = expanded + candidates
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c['id']: c for c in candidates if c['id'] in ids}
            events.append({'event': 'search', 'query': query, 'filter': filters,
                           'results': primary_results, 'expanded_context': expanded,
                           'context': context, 'context_ids': ids,
                           'neighbor_window': getattr(retriever, 'neighbor_window', 0)})
            # Send only the latest cumulative context; avoid multiplying the budget.
            history = [{'tool': 'search', 'queries': [e['query'] for e in events if e['event'] == 'search'],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if answer is None or answer == '':
                answer = ''
            if citations is None:
                citations = []
            if (not isinstance(answer, str) or not isinstance(citations, list)
                    or any(not isinstance(i, str) for i in citations)):
                raise ValueError('answer/citations không hợp lệ')
            reason = None
            if not answer.strip():
                reason = 'missing_answer'
            elif not citations:
                reason = 'missing_citations'
            elif any(i not in chunks for i in citations):
                reason = 'citation_not_in_context'
            elif any(f'[{i}]' not in answer for i in citations):
                answer = answer.rstrip() + ' ' + ' '.join(
                    f'[{identifier}]' for identifier in citations if f'[{identifier}]' not in answer)
            valid = reason is None
            if valid:
                return {'status': 'answered', 'answer': answer, 'citations': citations, 'events': events + [{'event': 'answer'}]}
            return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [],
                    'events': events + [{'event': 'invalid_or_missing_citations', 'reason': reason}]}
        else:
            raise ValueError('action phải là search hoặc answer')
    return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events}
