"""Run the retrieval-augmented agent against configured live services."""
from __future__ import annotations

import argparse
import json

from adapters import BedrockCohereEmbedder, GeminiController, QdrantRetriever
from agentic import run_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='?', help='Question to answer')
    parser.add_argument('--question', dest='question_option', help='Question to answer')
    parser.add_argument('--strategy', choices=('A', 'B', 'C'), default='B')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--max-searches', type=int, default=4)
    parser.add_argument('--budget', type=int, default=800)
    parser.add_argument('--neighbor-window', type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.question and args.question_option:
        parser.error('provide the question either positionally or with --question, not both')
    question = args.question_option or args.question
    if not question or not question.strip():
        parser.error('a non-empty question is required')
    if args.top_k < 1 or args.max_searches < 1 or args.budget < 1 or args.neighbor_window < 0:
        parser.error('--top-k, --max-searches, and --budget must be positive; --neighbor-window must be non-negative')

    embedder = BedrockCohereEmbedder()
    retriever = QdrantRetriever(embedder, strategy=args.strategy, neighbor_window=args.neighbor_window)
    controller = GeminiController()
    result = run_agent(question, controller, retriever, top_k=args.top_k,
                        max_searches=args.max_searches, budget=args.budget)
    print(json.dumps({key: result[key] for key in ('answer', 'status', 'citations', 'events')},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
