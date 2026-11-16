"""CLI argument and wiring tests; all provider clients are mocked."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
import run_agent


class RunAgentCliTests(unittest.TestCase):
    @patch.object(run_agent, 'run_agent')
    @patch.object(run_agent, 'GeminiController')
    @patch.object(run_agent, 'QdrantRetriever')
    @patch.object(run_agent, 'BedrockCohereEmbedder')
    def test_positional_question_assembles_adapters_and_emits_json(
            self, embedder_cls, retriever_cls, controller_cls, agent):
        embedder, retriever, controller = Mock(), Mock(), Mock()
        embedder_cls.return_value = embedder
        retriever_cls.return_value = retriever
        controller_cls.return_value = controller
        result = {'answer': 'Answer [c1]', 'status': 'answered', 'citations': ['c1'],
                  'events': [{'event': 'answer'}], 'internal': 'not exposed'}
        agent.return_value = result
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run_agent.main(['Question?', '--strategy', 'C', '--top-k', '7',
                                             '--max-searches', '2', '--budget', '1200']), 0)

        embedder_cls.assert_called_once_with()
        retriever_cls.assert_called_once_with(embedder, strategy='C', neighbor_window=1)
        controller_cls.assert_called_once_with()
        agent.assert_called_once_with('Question?', controller, retriever,
                                      top_k=7, max_searches=2, budget=1200)
        self.assertEqual(json.loads(output.getvalue()), {key: result[key] for key in
                         ('answer', 'status', 'citations', 'events')})

    @patch.object(run_agent, 'run_agent')
    @patch.object(run_agent, 'GeminiController')
    @patch.object(run_agent, 'QdrantRetriever')
    @patch.object(run_agent, 'BedrockCohereEmbedder')
    def test_question_option_uses_default_strategy_b(self, embedder_cls, retriever_cls,
                                                    controller_cls, agent):
        agent.return_value = {'answer': '', 'status': 'abstain', 'citations': [], 'events': []}
        with contextlib.redirect_stdout(io.StringIO()):
            run_agent.main(['--question', 'Question?'])
        self.assertEqual(retriever_cls.call_args.kwargs, {'strategy': 'B', 'neighbor_window': 1})


if __name__ == '__main__':
    unittest.main()
