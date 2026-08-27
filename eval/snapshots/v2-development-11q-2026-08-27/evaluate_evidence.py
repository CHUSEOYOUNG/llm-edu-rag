"""Compare original and revised queries using separately reviewed evidence groups.

This measures retrieval of annotated facts, not generated-answer correctness.
An alternative within a group is sufficient; every group is required for full coverage.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 5, 10)
DEPTH = 20


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique_by(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def evaluate_groups(ranked, groups, ks=KS, depth=DEPTH):
    if not groups or any(not g['alternatives'] for g in groups):
        raise ValueError('Every reviewed question needs nonempty evidence groups')
    if len(ranked) != len(set(ranked)):
        raise ValueError('Duplicate ranked chunk IDs')
    if depth < max(ks):
        raise ValueError('Evaluation depth is smaller than requested k')
    alternatives = [{e['chunk_id'] for e in g['alternatives']} for g in groups]
    union = set().union(*alternatives)
    metrics = {}
    for k in ks:
        top = set(ranked[:k])
        covered = sum(bool(top & options) for options in alternatives)
        metrics[f'hit@{k}'] = float(bool(top & union))
        metrics[f'coverage@{k}'] = covered / len(groups)
        metrics[f'complete@{k}'] = float(covered == len(groups))
    metrics[f'mrr@{depth}'] = next(
        (1 / rank for rank, cid in enumerate(ranked[:depth], 1) if cid in union), 0.0
    )
    return metrics


def evaluate_legacy(ranked, gold):
    if not gold:
        raise ValueError('Empty legacy gold is not an answerable retrieval target')
    gold = set(gold)
    return {
        **{f'recall@{k}': len(gold & set(ranked[:k])) / len(gold) for k in KS},
        **{f'hit@{k}': float(bool(gold & set(ranked[:k]))) for k in KS},
        'mrr@20': next((1 / i for i, cid in enumerate(ranked[:20], 1)
                       if cid in gold), 0.0),
    }


def average(rows):
    if not rows:
        raise ValueError('No evaluated questions')
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def validate_annotations(v1, v2, chunks, manifest):
    old = unique_by(v1, 'qid')
    new = unique_by(v2, 'qid')
    by_chunk = unique_by(chunks, 'chunk_id')
    if set(old) != set(new):
        raise ValueError('Draft must retain all v1 questions')
    reviewed = set(manifest['reviewed_qids'])
    actual_reviewed = {q['qid'] for q in v2
                       if q['annotation_status'] == 'assistant_reviewed_draft'}
    if reviewed != actual_reviewed:
        raise ValueError('Reviewed subset differs from manifest')
    withheld = set(manifest.get('withheld_qids', {}))
    if not withheld <= set(old) or withheld & reviewed:
        raise ValueError('Withheld questions must exist and must not be scored')
    if withheld != {q['qid'] for q in v2 if q['annotation_status'] == 'scope_review_required'}:
        raise ValueError('Scope-review exclusions differ from manifest')
    approved = manifest.get('approved_question_changes', {})
    if not set(approved) <= set(old):
        raise ValueError('Unknown approved question ID')
    for qid, q in new.items():
        expected = dict(old[qid])
        if qid in approved:
            change = approved[qid]
            if (change['original'] != old[qid]['question']
                    or q.get('original_question') != change['original']
                    or q.get('question_revision_status') != 'user_approved'):
                raise ValueError(f'Invalid approved question revision: {qid}')
            expected['question'] = change['revised']
        if any(q.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Unapproved original field change: {qid}')
        if not set(q['gold_chunks']) <= set(by_chunk):
            raise ValueError(f'Missing v1 gold: {qid}')
        groups = q['evidence_groups']
        if qid not in reviewed:
            if groups:
                raise ValueError(f'Unreviewed groups must not be scored: {qid}')
            continue
        if q['type'] == 'unans' or not groups:
            raise ValueError(f'Invalid reviewed question: {qid}')
        unique_by(groups, 'group_id')
        for group in groups:
            if not group['alternatives']:
                raise ValueError(f'Empty evidence group: {qid}')
            unique_by(group['alternatives'], 'chunk_id')
            for evidence in group['alternatives']:
                c = by_chunk[evidence['chunk_id']]
                a, b = evidence['start'], evidence['end']
                if (c['doc_id'] != evidence['doc_id'] or not 0 <= a < b <= len(c['body'])
                        or c['body'][a:b] != evidence['text']):
                    raise ValueError(f'Stale or invalid evidence anchor: {qid}')


def ranking_path(root, query_version):
    if query_version == 'v1':
        return root / 'experiments/dense_rankings_v2_audit.json'
    if query_version == 'v2':
        return root / 'experiments/dense_rankings_v2_current.json'
    raise ValueError('Unknown query version')


def validate_rankings(rankings, manifest, question_hash, targets, ids):
    for key, expected in (
        ('chunks_sha256', manifest['chunks_sha256']), ('questions_sha256', question_hash),
        ('embedding_sha256', manifest['embedding_sha256']), ('index_text', 'body'),
        ('model', manifest['model']), ('evaluation_depth', DEPTH),
    ):
        if rankings.get(key) != expected:
            raise ValueError(f'Ranking provenance mismatch: {key}')
    by_rank = unique_by(rankings['per_question'], 'qid')
    if set(by_rank) != {q['qid'] for q in targets}:
        raise ValueError('Ranking question set mismatch')
    for row in by_rank.values():
        ranked = row['ranked_ids']
        if len(ranked) != DEPTH or len(set(ranked)) != DEPTH or not set(ranked) <= ids:
            raise ValueError(f'Invalid top20 ranking: {row["qid"]}')
    return by_rank


def collect_rankings(root, query_version='v2'):
    """Explicit offline recomputation for the exact annotated corpus snapshot."""
    import os

    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    from sentence_transformers import SentenceTransformer

    manifest = json.loads((root / 'eval/questions.v2.draft.manifest.json').read_text())
    cp = root / 'data/processed/chunks.jsonl'
    v1p = root / 'eval/questions.jsonl'
    v2p = root / 'eval/questions.v2.draft.jsonl'
    qp = v1p if query_version == 'v1' else v2p
    query_hash = manifest['v1_sha256' if query_version == 'v1' else 'v2_sha256']
    ep = root / 'data/processed/embeddings.npy'
    for path, key in ((cp, 'chunks_sha256'), (v1p, 'v1_sha256'),
                      (v2p, 'v2_sha256'), (ep, 'embedding_sha256')):
        if sha256(path) != manifest[key]:
            raise ValueError(f'Cannot reuse annotated snapshot: {path}')
    chunks = read_jsonl(cp)
    validate_annotations(read_jsonl(v1p), read_jsonl(v2p), chunks, manifest)
    targets = [q for q in read_jsonl(qp) if q['type'] != 'unans']
    mat = np.load(ep)
    if mat.ndim != 2 or mat.shape[0] != len(chunks) or not np.isfinite(mat).all():
        raise ValueError('Invalid embedding matrix')
    model = SentenceTransformer(manifest['model'], local_files_only=True)
    vectors = model.encode([q['question'] for q in targets], normalize_embeddings=True)
    rows = []
    for q, vector in zip(targets, vectors):
        scores = mat @ vector
        order = np.argsort(-scores)[:DEPTH]
        rows.append({'qid': q['qid'], 'ranked_ids': [chunks[i]['chunk_id'] for i in order],
                     'scores': [float(scores[i]) for i in order]})
    result = {'model': manifest['model'], 'index_text': 'body',
              'chunks_sha256': manifest['chunks_sha256'], 'questions_sha256': query_hash,
              'embedding_sha256': manifest['embedding_sha256'], 'evaluation_depth': DEPTH,
              'per_question': rows}
    path = ranking_path(root, query_version)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


def run(root, query_version='v2'):
    v1_path = root / 'eval/questions.jsonl'
    v2_path = root / 'eval/questions.v2.draft.jsonl'
    chunks_path = root / 'data/processed/chunks.jsonl'
    selected_path = ranking_path(root, query_version)
    original_path = ranking_path(root, 'v1')
    manifest = json.loads((root / 'eval/questions.v2.draft.manifest.json').read_text())
    for path, key in ((v1_path, 'v1_sha256'), (v2_path, 'v2_sha256'),
                      (chunks_path, 'chunks_sha256')):
        if sha256(path) != manifest[key]:
            raise ValueError(f'Input fingerprint mismatch: {path}')
    v1, v2, chunks = map(read_jsonl, (v1_path, v2_path, chunks_path))
    validate_annotations(v1, v2, chunks, manifest)
    targets = [q for q in v1 if q['type'] != 'unans']
    ids = {c['chunk_id'] for c in chunks}
    original = validate_rankings(json.loads(original_path.read_text()), manifest,
                                 manifest['v1_sha256'], targets, ids)
    selected_hash = manifest['v1_sha256' if query_version == 'v1' else 'v2_sha256']
    by_rank = validate_rankings(json.loads(selected_path.read_text()), manifest,
                                selected_hash, targets, ids)
    legacy = {q['qid']: evaluate_legacy(original[q['qid']]['ranked_ids'], q['gold_chunks'])
              for q in targets}
    reviewed = []
    for q in v2:
        if q['qid'] not in manifest['reviewed_qids']:
            continue
        ranked = by_rank[q['qid']]['ranked_ids']
        groups = q['evidence_groups']
        reviewed.append({
            'qid': q['qid'], 'legacy': legacy[q['qid']],
            'draft_on_original_query': evaluate_groups(original[q['qid']]['ranked_ids'], groups),
            'draft': evaluate_groups(ranked, groups),
            'first_evidence_rank': {
                g['group_id']: next((i for i, cid in enumerate(ranked, 1)
                                    if cid in {e['chunk_id'] for e in g['alternatives']}), None)
                for g in groups
            },
        })
    return {
        'status': 'diagnostic_draft_not_held_out',
        'query_version': query_version,
        'approved_question_changes': manifest.get('approved_question_changes', {}),
        'warning': 'Label corrections and query revisions are separate; not a retriever upgrade.',
        'rankings_sha256': sha256(selected_path),
        'original_rankings_sha256': sha256(original_path),
        'v2_sha256': manifest['v2_sha256'],
        'legacy_all_answerable': {'n': len(legacy), 'overall': average(list(legacy.values())),
                                  'per_question': legacy},
        'reviewed_subset': {'n': len(reviewed),
                            'legacy_overall': average([r['legacy'] for r in reviewed]),
                            'draft_on_original_query_overall': average(
                                [r['draft_on_original_query'] for r in reviewed]),
                            'draft_overall': average([r['draft'] for r in reviewed]),
                            'per_question': reviewed},
        'excluded_from_v2': [q['qid'] for q in v2 if q['qid'] not in manifest['reviewed_qids']],
        'exclusion_reasons': {
            q['qid']: manifest.get('withheld_qids', {}).get(q['qid'], q['annotation_status'])
            for q in v2 if q['qid'] not in manifest['reviewed_qids']
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--query-version', choices=('v1', 'v2'), default='v2',
                        help='v1 replays original queries; v2 uses the approved draft queries')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute Dense query rankings with the local model cache')
    args = parser.parse_args()
    if args.recompute:
        collect_rankings(args.root, args.query_version)
    result = run(args.root, args.query_version)
    name = 'evidence_v2_audit.json' if args.query_version == 'v1' else 'evidence_v2_current.json'
    out = args.root / 'experiments' / name
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f"Query version: {args.query_version}; {result['reviewed_subset']['n']} reviewed questions in v2 draft")
    print('qid    v1 Hit@5   draft Hit@5   draft Coverage@5   draft Complete@5')
    for row in result['reviewed_subset']['per_question']:
        old, new = row['legacy'], row['draft']
        print(f"{row['qid']}    {old['hit@5']:.3f}       {new['hit@5']:.3f}"
              f"          {new['coverage@5']:.3f}            {new['complete@5']:.3f}")
    print(f'Output: {out}')


if __name__ == '__main__':
    main()
