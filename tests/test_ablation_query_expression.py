import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ablation_query_expression import literal_scope_audit, query_packet, validate_plan


class QueryExpressionTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            'qid': 'q', 'full_question': '2028년 초등학교 교과는?', 'core_query': '초등학교 교과',
            'constraints': [
                {'kind': 'year', 'value': '2028년', 'start': 0, 'end': 5, 'placement': 'context_only'},
                {'kind': 'school', 'value': '초등학교', 'start': 6, 'end': 10, 'placement': 'search_and_context'},
            ],
        }
        self.plan = {'questions_sha256': 'hash', 'scope_filters_applied': False, 'rows': [self.row]}
        self.questions = [{'qid': 'q', 'question': self.row['full_question']}]

    def validate(self):
        return validate_plan(self.plan, self.questions, 'hash')

    def test_scope_is_preserved_without_becoming_a_filter(self):
        self.validate()
        packet = query_packet(self.row, 'manual_core')
        self.assertEqual(packet['search_query'], '초등학교 교과')
        self.assertEqual(packet['scope_constraints'], self.row['constraints'])
        self.assertFalse(packet['scope_filters_applied'])
        packet['scope_constraints'][0]['value'] = 'changed'
        self.assertEqual(self.row['constraints'][0]['value'], '2028년')

    def test_required_school_cannot_silently_disappear(self):
        self.row['core_query'] = '교과'
        with self.assertRaisesRegex(ValueError, 'condition lost'):
            self.validate()

    def test_metadata_source_hint_is_not_an_allowed_search_input(self):
        self.row['source_hint'] = 'gold document section'
        with self.assertRaisesRegex(ValueError, 'Unexpected plan fields'):
            self.validate()

    def test_scope_must_come_from_visible_query(self):
        self.row['constraints'][0]['value'] = '2030년'
        with self.assertRaisesRegex(ValueError, 'anchored'):
            self.validate()

    def test_stale_or_partial_plans_are_rejected(self):
        for change in ('hash', 'missing', 'duplicate'):
            with self.subTest(change=change):
                plan = copy.deepcopy(self.plan)
                if change == 'hash':
                    plan['questions_sha256'] = 'stale'
                elif change == 'missing':
                    plan['rows'] = []
                else:
                    plan['rows'].append(copy.deepcopy(plan['rows'][0]))
                with self.assertRaises(ValueError):
                    validate_plan(plan, self.questions, 'hash')

    def test_date_in_path_is_not_counted_as_body_or_semantic_validation(self):
        packet = query_packet(self.row, 'manual_core')
        chunks = {'c': {'body': '초등학교 교과 목록', 'path': '2028년 시행', 'doc_id': 'doc'}}
        audit = literal_scope_audit(packet, ['c'], chunks)[0]
        self.assertEqual(audit['body_literal_hits'], [])
        self.assertEqual(audit['metadata_literal_hits'], [{'rank': 1, 'chunk_id': 'c'}])
        self.assertEqual(audit['semantic_verdict'], 'unknown')


if __name__ == '__main__':
    unittest.main()
