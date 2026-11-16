"""Offline tests for the local advanced evidence retriever and CLI."""
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from starter.advanced_retriever import AdvancedEvidenceRetriever
import run_advanced_agent as runner
import adapters


class AdvancedRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.index = self.root / 'records.jsonl'
        self.records = [
            {'evidence_id': 'text-1', 'evidence_type': 'text', 'text': 'Rail freight grew 3.61 million tonnes.',
             'physical_pdf_page': 69, 'printed_page_label': '68', 'bbox': [0, 0, 1, 1],
             'crop_path': None, 'source_sha256': 'sha-text'},
            {'evidence_id': 'table-1', 'evidence_type': 'table', 'text': 'Year | LPI\n2023 | 3.4',
             'caption': 'Bảng 1.7', 'caption_evidence_id': 'caption-id',
             'physical_pdf_page': 35, 'printed_page_label': '34', 'bbox': [.1, .2, .8, .5],
             'crop_path': 'crops/table.png', 'source_sha256': 'sha-table'},
            {'evidence_id': 'figure-1', 'evidence_type': 'figure', 'text': None,
              'caption': 'Figure 8: Port throughput chart', 'visual_description': 'Bars increase by year.',
             'ocr_text': '2023 3.4', 'physical_pdf_page': 40, 'printed_page_label': '39',
              'bbox': [.2, .2, .7, .7], 'crop_path': 'crops/figure.png', 'source_sha256': 'sha-figure'},
            {'evidence_id': 'figure-2', 'evidence_type': 'figure', 'text': 'Rail capacity map',
             'caption': 'Hình 2.4 - Rail capacity', 'physical_pdf_page': 41,
             'bbox': [.1, .1, .5, .5], 'crop_path': 'crops/map.png'},
            {'evidence_id': 'decoration', 'evidence_type': 'figure', 'text': None,
             'ocr_text': 'BAO CAO LOGISTICS VIET NAM 2024', 'physical_pdf_page': 1,
             'bbox': [0, 0, 1, 1], 'crop_path': 'crops/decor.png'},
            {'evidence_id': 'caption-only', 'evidence_type': 'caption', 'label': 'Figure 9',
             'text': 'Figure 9 decorative caption'},
        ]
        self.index.write_text(''.join(json.dumps(row) + '\n' for row in self.records), encoding='utf-8')
        (self.root / 'summary.json').write_text(json.dumps({'source_pdf': '/docs/Logistics_2024.pdf'}), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_index_maps_evidence_and_preserves_crop_anchor_metadata(self):
        retriever = AdvancedEvidenceRetriever(self.index)
        self.assertEqual({row['id'] for row in retriever.records}, {'text-1', 'table-1', 'figure-1', 'figure-2'})
        hit = next(row for row in retriever.records if row['id'] == 'table-1')
        self.assertEqual(hit['source_id'], 'logistics-vietnam-2024')
        self.assertEqual(hit['source'], 'Logistics_2024.pdf')
        self.assertEqual(hit['page'], 35)
        self.assertEqual(hit['metadata']['evidence_type'], 'table')
        self.assertIn('Evidence type: table', hit['text'])
        self.assertEqual(hit['metadata']['sha'], 'sha-table')
        self.assertEqual(hit['metadata']['printed_page'], '34')
        self.assertEqual(hit['metadata']['anchor']['crop_path'], 'crops/table.png')
        self.assertEqual(hit['metadata']['region']['bbox'], [.1, .2, .8, .5])
        self.assertIn('Bảng 1.7', hit['text'])
        self.assertEqual(retriever.search('Bảng 1.7', top_k=1)[0]['id'], 'table-1')
        visual = next(row for row in retriever.records if row['id'] == 'figure-1')
        self.assertIn('Port throughput chart', visual['text'])
        self.assertIn('Bars increase by year.', visual['text'])
        self.assertIn('2023 3.4', visual['text'])

    def test_search_ranks_bm25_and_rejects_filters(self):
        retriever = AdvancedEvidenceRetriever(self.index)
        self.assertEqual(retriever.search('LPI 2023', top_k=1)[0]['id'], 'table-1')
        with self.assertRaisesRegex(ValueError, 'does not support filters'):
            retriever.search('LPI', filters={'field': 'page', 'value': 35}, top_k=2)
        with self.assertRaisesRegex(ValueError, 'filters=None'):
            retriever.search('LPI', filters={}, top_k=2)

    def test_explicit_reference_prioritizes_exact_caption_region(self):
        retriever = AdvancedEvidenceRetriever(self.index)
        result = retriever.search('Which data is shown in Table 1.7?', top_k=1)
        self.assertEqual(result[0]['id'], 'table-1')
        self.assertEqual(retriever.search('Compare Figure8 with BẢNG 1.7', top_k=2)[0]['id'], 'figure-1')
        self.assertEqual(retriever.search('Compare Figure8 with BẢNG 1.7', top_k=2)[1]['id'], 'table-1')
        self.assertEqual(retriever.search('rail capacity Hình2.4', top_k=1)[0]['id'], 'figure-2')

    def test_no_reference_preserves_exact_bm25_results_and_unrelated_records(self):
        retriever = AdvancedEvidenceRetriever(self.index)
        query = 'rail freight capacity'
        self.assertEqual(retriever.search(query, top_k=3), retriever.index.search(query, top_k=3))
        self.assertEqual(retriever.search('Figure 99', top_k=3), retriever.index.search('Figure 99', top_k=3))

    def test_unique_physical_page_caption_links_uncaptioned_region(self):
        rows = [dict(row) for row in self.records]
        rows[1]['caption'] = None
        rows[1]['physical_pdf_page'] = 35
        rows.append({'evidence_id': 'caption-1', 'evidence_type': 'caption',
                     'label': 'BẢNG 1.7. Saudi Arabia', 'physical_pdf_page': 35})
        self.index.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        retriever = AdvancedEvidenceRetriever(self.index)
        self.assertEqual(retriever.search('Bảng1.7', top_k=1)[0]['id'], 'table-1')


class AdvancedAgentCliTests(unittest.TestCase):
    def test_cli_batch_uses_local_factory_and_writes_no_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / 'benchmark.json'
            benchmark.write_text(json.dumps([
                {'id': 'dev-1', 'split': 'dev', 'question': 'question only', 'expected_answer': 'secret',
                 'gold': [{'evidence_id': 'hidden'}]},
                {'id': 'test-1', 'split': 'test', 'question': 'other', 'expected_answer': 'private'},
            ]), encoding='utf-8')
            index = root / 'records.jsonl'
            index.write_text(json.dumps({'evidence_id': 'e-1', 'evidence_type': 'text',
                'text': 'local evidence', 'physical_pdf_page': 1, 'bbox': [0, 0, 1, 1]}) + '\n',
                encoding='utf-8')
            args = Namespace(split='dev', top_k=4, budget=900, max_searches=2,
                             benchmark=benchmark, index=index, output=root / 'dev-output')
            controller_factory = Mock(return_value=object())
            agent = Mock(return_value={'status': 'abstain', 'answer': '', 'citations': [], 'events': []})
            with patch.object(adapters, 'BedrockCohereEmbedder', side_effect=AssertionError('Bedrock used')), \
                    patch.object(adapters, 'QdrantRetriever', side_effect=AssertionError('Qdrant used')), \
                    patch.object(runner, 'GeminiController', controller_factory):
                summary = runner.run_batch(args, agent=agent, controller_factory=None)
            controller_factory.assert_called_once_with()
            self.assertEqual(agent.call_args.args[0], 'question only')
            self.assertEqual(agent.call_args.kwargs, {'top_k': 4, 'budget': 900, 'max_searches': 2})
            trace = (args.output / 'traces.jsonl').read_text(encoding='utf-8')
            self.assertNotIn('secret', trace)
            self.assertNotIn('hidden', trace)
            self.assertEqual(summary['total_cases'], 1)

    def test_test_output_defaults_to_isolated_directory(self):
        args = runner.build_parser().parse_args(['--split', 'test'])
        self.assertEqual(args.benchmark, runner.DEFAULT_BENCHMARK)
        self.assertEqual(args.index, runner.DEFAULT_INDEX)
        self.assertEqual(runner.DEFAULT_TEST_OUTPUT.name, 'advanced_agent_test')
        self.assertNotEqual(runner.DEFAULT_TEST_OUTPUT, runner.DEFAULT_DEV_OUTPUT)


if __name__ == '__main__':
    unittest.main()
