import sys
import json
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from agent_metrics import aggregate, main

class AgentMetricTests(unittest.TestCase):
    def test_counts_and_errors_are_separate(self):
        cases = [{'id':'a','answerable':True}, {'id':'b','answerable':False}, {'id':'c','answerable':True}]
        traces = [{'query_id':'a','status':'answered','events':[{'event':'search'}],'citations':['c1','c2']},
                  {'query_id':'b','status':'abstain','events':[{'event':'search'}],'citations':[]},
                  {'query_id':'c','status':'error','events':[],'citations':[]}]
        reviews = [{'query_id':'a','answer_grade':1,'citation_support':[True,False]},
                   {'query_id':'b','citation_support':[],'abstention_correct':True}]
        result = aggregate(cases, traces, reviews)
        self.assertEqual(result['provider_errors'], 1)
        self.assertEqual(result['tool_use'], {'n':2,'N':2})
        self.assertEqual(result['citation_validity'], {'n':1,'N':2})
        self.assertEqual(result['abstention'], {'n':1,'N':1})
        self.assertEqual(result['grounded_answer']['mean'], 1)

    def test_unanswerable_requires_human_confirmation(self):
        cases = [{'id':'a','answerable':False}]
        traces = [{'query_id':'a','status':'abstain','events':[],'citations':[]}]
        with self.assertRaises(ValueError): aggregate(cases, traces, [{'query_id':'a','citation_support':[]}])
        result = aggregate(cases, traces, [{'query_id':'a','citation_support':[], 'abstention_correct':False}])
        self.assertEqual(result['abstention'], {'n':0, 'N':1})

    def test_infers_answerability_from_gold_when_missing(self):
        cases = [{'id':'a','gold':'expected answer'}, {'id':'b','gold':''}]
        traces = [{'query_id':'a','status':'answered','events':[],'citations':[]},
                  {'query_id':'b','status':'abstain','events':[],'citations':[]}]
        reviews = [{'query_id':'a','answer_grade':2,'citation_support':[]},
                   {'query_id':'b','citation_support':[],'abstention_correct':True}]
        result = aggregate(cases, traces, reviews)
        self.assertEqual(result['grounded_answer']['N'], 1)
        self.assertEqual(result['abstention'], {'n':1,'N':1})

    def test_cli_loads_jsonl_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / 'cases.json'
            traces = root / 'traces.jsonl'
            reviews = root / 'reviews.json'
            output = root / 'metrics.json'
            cases.write_text(json.dumps([{'id':'a','gold':'answer','split':'dev'}]), encoding='utf-8')
            traces.write_text(json.dumps({'query_id':'a','status':'answered','events':[],'citations':[]}) + '\n', encoding='utf-8')
            reviews.write_text(json.dumps([{'query_id':'a','answer_grade':2,'citation_support':[]}]), encoding='utf-8')
            old_argv = sys.argv
            try:
                sys.argv = ['agent_metrics.py', '--cases', str(cases), '--traces', str(traces),
                            '--reviews', str(reviews), '--split', 'dev', '--output', str(output)]
                main()
            finally:
                sys.argv = old_argv
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['grounded_answer']['N'], 1)

    def test_cli_keeps_latest_jsonl_retry_and_unwraps_review_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / 'cases.json'
            traces = root / 'traces.jsonl'
            reviews = root / 'reviews.json'
            output = root / 'metrics.json'
            cases.write_text(json.dumps([{'id':'a','gold':'answer','split':'dev'}]), encoding='utf-8')
            traces.write_text('\n'.join(json.dumps(record) for record in [
                {'query_id':'a','status':'error','events':[],'citations':[]},
                {'query_id':'a','status':'answered','events':[{'event':'search'}],'citations':[]},
            ]) + '\n', encoding='utf-8')
            reviews.write_text(json.dumps({'source':'preliminary assistant labels','records':[
                {'query_id':'a','answer_grade':2,'citation_support':[]}
            ]}), encoding='utf-8')
            old_argv = sys.argv
            try:
                sys.argv = ['agent_metrics.py', '--cases', str(cases), '--traces', str(traces),
                            '--reviews', str(reviews), '--split', 'dev', '--output', str(output)]
                main()
            finally:
                sys.argv = old_argv
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['provider_errors'], 0)
            self.assertEqual(report['grounded_answer']['N'], 1)
            self.assertEqual(report['tool_use'], {'n':1,'N':1})

    def test_cli_loads_review_array(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, traces, reviews, output = (root / name for name in
                                               ('cases.json','traces.json','reviews.json','metrics.json'))
            cases.write_text(json.dumps([{'id':'a','gold':'answer','split':'dev'}]), encoding='utf-8')
            traces.write_text(json.dumps([{'query_id':'a','status':'answered','events':[],'citations':[]}]), encoding='utf-8')
            reviews.write_text(json.dumps([{'query_id':'a','answer_grade':1,'citation_support':[]}]), encoding='utf-8')
            old_argv = sys.argv
            try:
                sys.argv = ['agent_metrics.py', '--cases', str(cases), '--traces', str(traces),
                            '--reviews', str(reviews), '--split', 'dev', '--output', str(output)]
                main()
            finally:
                sys.argv = old_argv
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['grounded_answer']['1'], 1)

    def test_rejects_missing_review_and_false_abstention_grade(self):
        cases = [{'id':'a','answerable':True}]
        traces = [{'query_id':'a','status':'abstain','events':[],'citations':[]}]
        with self.assertRaises(ValueError): aggregate(cases,traces,[])
        with self.assertRaises(ValueError): aggregate(cases,traces,[{'query_id':'a','answer_grade':2,'citation_support':[]}])

if __name__ == '__main__': unittest.main()
