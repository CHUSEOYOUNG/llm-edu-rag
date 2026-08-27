import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluate_evidence import evaluate_groups, validate_annotations, validate_rankings


def group(*ids):
    return {'alternatives': [{'chunk_id': cid} for cid in ids]}


class EvidenceMetricsTests(unittest.TestCase):
    def test_alternatives_do_not_require_retrieving_all_duplicates(self):
        result = evaluate_groups(['a'], [group('a', 'b', 'c')])
        self.assertEqual(result['complete@1'], 1.0)
        self.assertEqual(result['coverage@5'], 1.0)

    def test_one_school_level_is_only_partial_coverage(self):
        result = evaluate_groups(['a', 'noise', 'b'], [group('a'), group('b')])
        self.assertEqual(result['hit@1'], 1.0)
        self.assertEqual(result['coverage@1'], 0.5)
        self.assertEqual(result['complete@1'], 0.0)
        self.assertEqual(result['complete@5'], 1.0)

    def test_one_chunk_can_support_two_distinct_facts(self):
        result = evaluate_groups(['shared'], [group('shared', 'a'), group('shared', 'b')])
        self.assertEqual(result['complete@1'], 1.0)

    def test_no_hit_and_depth_cutoff(self):
        result = evaluate_groups([f'n{i}' for i in range(20)] + ['gold'], [group('gold')])
        self.assertEqual(result['hit@5'], 0.0)
        self.assertEqual(result['mrr@20'], 0.0)

    def test_invalid_inputs_are_not_silent_perfect_scores(self):
        for ranked, groups in [([], []), (['a'], [group()]), (['a', 'a'], [group('a')])]:
            with self.subTest(ranked=ranked, groups=groups), self.assertRaises(ValueError):
                evaluate_groups(ranked, groups)


class AnnotationValidationTests(unittest.TestCase):
    def setUp(self):
        self.old = [{'qid': 'q', 'type': 'fact', 'question': 'original', 'gold_chunks': ['c']}]
        self.new = copy.deepcopy(self.old)
        self.new[0].update(annotation_status='assistant_reviewed_draft', evidence_groups=[{
            'group_id': 'g', 'alternatives': [{'chunk_id': 'c', 'doc_id': 'd',
                                              'start': 0, 'end': 3, 'text': 'abc'}]}])
        self.chunks = [{'chunk_id': 'c', 'doc_id': 'd', 'body': 'abcdef'}]
        self.manifest = {'reviewed_qids': ['q']}

    def validate(self):
        validate_annotations(self.old, self.new, self.chunks, self.manifest)

    def test_valid_anchor(self):
        self.validate()

    def test_body_change_with_same_id_is_rejected(self):
        self.chunks[0]['body'] = 'XYZdef'
        with self.assertRaises(ValueError):
            self.validate()

    def test_question_change_is_rejected_before_reusing_rankings(self):
        self.new[0]['question'] = 'changed'
        with self.assertRaises(ValueError):
            self.validate()

    def approve_revision(self):
        self.new[0].update(question='revised', original_question='original',
                           question_revision_status='user_approved')
        self.manifest['approved_question_changes'] = {
            'q': {'original': 'original', 'revised': 'revised'}}

    def test_explicitly_approved_revision_is_valid(self):
        self.approve_revision()
        self.validate()

    def test_approval_does_not_allow_other_question_text(self):
        self.approve_revision()
        self.new[0]['question'] = 'another revision'
        with self.assertRaises(ValueError):
            self.validate()

    def test_question_approval_does_not_allow_gold_changes(self):
        self.approve_revision()
        self.new[0]['gold_chunks'] = []
        with self.assertRaises(ValueError):
            self.validate()

    def test_unreviewed_groups_cannot_be_scored(self):
        self.new[0]['annotation_status'] = 'pending_review'
        self.manifest['reviewed_qids'] = []
        with self.assertRaises(ValueError):
            self.validate()

    def test_scope_review_is_explicitly_excluded(self):
        self.new[0].update(annotation_status='scope_review_required', evidence_groups=[])
        self.manifest.update(reviewed_qids=[], withheld_qids={'q': 'effective date is ambiguous'})
        self.validate()
        self.manifest['withheld_qids'] = {}
        with self.assertRaisesRegex(ValueError, 'exclusions'):
            self.validate()

    def test_withheld_question_cannot_also_be_reviewed(self):
        self.manifest['withheld_qids'] = {'q': 'effective date is ambiguous'}
        with self.assertRaisesRegex(ValueError, 'must not be scored'):
            self.validate()


class RankingProvenanceTests(unittest.TestCase):
    def test_old_query_cache_rejected_even_with_same_qids(self):
        manifest = {'chunks_sha256': 'corpus', 'embedding_sha256': 'embedding', 'model': 'model'}
        ids = [f'c{i}' for i in range(20)]
        rankings = {**manifest, 'questions_sha256': 'old-query-hash', 'index_text': 'body',
                    'evaluation_depth': 20,
                    'per_question': [{'qid': 'q', 'ranked_ids': ids}]}
        targets = [{'qid': 'q'}]
        self.assertIn('q', validate_rankings(rankings, manifest, 'old-query-hash', targets, set(ids)))
        with self.assertRaisesRegex(ValueError, 'questions_sha256'):
            validate_rankings(rankings, manifest, 'revised-query-hash', targets, set(ids))


if __name__ == '__main__':
    unittest.main()
