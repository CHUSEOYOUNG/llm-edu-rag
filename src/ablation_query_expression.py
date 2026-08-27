"""Exploratory full-question vs manual-core-query comparison on frozen v2.

Scope is preserved in the output packet, not enforced as a retrieval filter.
Gold is used only for evaluation after retrieval. The manual author was not blinded.
"""

import argparse
import json
import os
from pathlib import Path
import re
import unicodedata

from evaluate_evidence import (
    DEPTH, ROOT, average, evaluate_groups, read_jsonl, sha256, unique_by,
    validate_annotations, validate_rankings,
)

SNAPSHOT_ID = 'v2-development-11q-2026-08-27'


def validate_plan(plan, questions, question_hash):
    if plan['questions_sha256'] != question_hash:
        raise ValueError('Query plan belongs to a different frozen question set')
    targets = unique_by(questions, 'qid')
    rows = unique_by(plan['rows'], 'qid')
    if set(rows) != set(targets):
        raise ValueError('Plan must cover every evaluated question exactly once')
    if plan['scope_filters_applied'] is not False:
        raise ValueError('This ablation does not implement scope filters')
    for qid, row in rows.items():
        if set(row) != {'qid', 'full_question', 'core_query', 'constraints'}:
            raise ValueError('Unexpected plan fields; gold/source hints are not search inputs')
        if row['full_question'] != targets[qid]['question']:
            raise ValueError(f'Full query changed: {qid}')
        if not isinstance(row['core_query'], str) or not row['core_query'].strip():
            raise ValueError(f'Empty core query: {qid}')
        for item in row['constraints']:
            start, end = item['start'], item['end']
            if (not 0 <= start < end <= len(row['full_question'])
                    or row['full_question'][start:end] != item['value']):
                raise ValueError(f'Scope must be anchored in the visible question: {qid}')
            if item['placement'] not in {'search_and_context', 'context_only'}:
                raise ValueError('Unknown scope placement')
            if item['placement'] == 'search_and_context' and item['value'] not in row['core_query']:
                raise ValueError(f'Required search condition lost: {qid}')
    return rows


def query_packet(row, variant):
    if variant not in {'full', 'manual_core'}:
        raise ValueError('Unknown query variant')
    return {
        'search_query': row['full_question'] if variant == 'full' else row['core_query'],
        'original_question': row['full_question'],
        'scope_constraints': [dict(item) for item in row['constraints']],
        'scope_filters_applied': False,
        'semantic_scope_validation': 'not_implemented',
    }


def compact(text):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', text)).lower()


def literal_scope_audit(packet, ranked, chunks, k=5):
    """Surface presence only: an occurrence does not prove applicability."""
    rows = []
    for item in packet['scope_constraints']:
        if item['placement'] != 'context_only':
            continue
        token = compact(item['value'])
        body_hits, metadata_hits = [], []
        for rank, cid in enumerate(ranked[:k], 1):
            c = chunks[cid]
            match = {'rank': rank, 'chunk_id': cid}
            if token in compact(c['body']):
                body_hits.append(match)
            if token in compact(c.get('path', '') + ' ' + c['doc_id']):
                metadata_hits.append(match)
        rows.append({'kind': item['kind'], 'value': item['value'],
                     'body_literal_hits': body_hits, 'metadata_literal_hits': metadata_hits,
                     'semantic_verdict': 'unknown'})
    return rows


def run(root):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    from sentence_transformers import SentenceTransformer

    snapshot = root / 'eval/snapshots' / SNAPSHOT_ID
    for name, expected in json.loads((snapshot / 'checksums.json').read_text()).items():
        if sha256(snapshot / name) != expected:
            raise ValueError(f'Frozen snapshot changed: {name}')
    manifest = json.loads((snapshot / 'questions.v2.draft.manifest.json').read_text())
    cp = root / 'data/processed/chunks.jsonl'
    ep = root / 'data/processed/embeddings.npy'
    v1p = root / 'eval/questions.jsonl'
    qp = snapshot / 'questions.v2.draft.jsonl'
    for path, key in ((cp, 'chunks_sha256'), (ep, 'embedding_sha256'),
                      (v1p, 'v1_sha256'), (qp, 'v2_sha256')):
        if sha256(path) != manifest[key]:
            raise ValueError(f'Input fingerprint mismatch: {path}')
    chunks, questions = read_jsonl(cp), read_jsonl(qp)
    validate_annotations(read_jsonl(v1p), questions, chunks, manifest)
    targets = [q for q in questions if q['qid'] in manifest['reviewed_qids']]
    by_chunk = unique_by(chunks, 'chunk_id')
    plan_path = root / 'experiments/query_expression_plan.json'
    plan = json.loads(plan_path.read_text())
    if plan['snapshot_id'] != SNAPSHOT_ID:
        raise ValueError('Wrong snapshot selected by query plan')
    # The plan validator needs only visible question text, never answer/gold/meta.
    rows = validate_plan(plan, [{'qid': q['qid'], 'question': q['question']} for q in targets], sha256(qp))
    baseline = validate_rankings(
        json.loads((snapshot / 'dense_rankings_v2_current.json').read_text()),
        manifest, manifest['v2_sha256'], targets, set(by_chunk),
    )
    matrix = np.load(ep)
    if matrix.ndim != 2 or len(matrix) != len(chunks) or not np.isfinite(matrix).all():
        raise ValueError('Invalid embedding matrix')
    model = SentenceTransformer(manifest['model'], local_files_only=True)
    by_variant = {}
    for variant in ('full', 'manual_core'):
        packets = [query_packet(rows[q['qid']], variant) for q in targets]
        vectors = model.encode([p['search_query'] for p in packets], normalize_embeddings=True)
        results = []
        for q, packet, vector in zip(targets, packets, vectors):
            scores = matrix @ vector
            order = np.argsort(-scores)[:DEPTH]
            ranked = [chunks[i]['chunk_id'] for i in order]
            if variant == 'full' and ranked != baseline[q['qid']]['ranked_ids']:
                raise ValueError(f'Frozen baseline ranking was not reproduced: {q["qid"]}')
            results.append({
                'qid': q['qid'], 'query_packet': packet,
                'metrics': evaluate_groups(ranked, q['evidence_groups']),
                'ranked_ids': ranked, 'scores': [float(scores[i]) for i in order],
                'first_evidence_rank': {
                    g['group_id']: next((i for i, cid in enumerate(ranked, 1)
                                        if cid in {e['chunk_id'] for e in g['alternatives']}), None)
                    for g in q['evidence_groups']
                },
                'top5_scope_literal_audit': literal_scope_audit(packet, ranked, by_chunk),
            })
        by_variant[variant] = {'overall': average([r['metrics'] for r in results]),
                               'per_question': results}
    comparisons = []
    for full, core in zip(by_variant['full']['per_question'], by_variant['manual_core']['per_question']):
        comparisons.append({'qid': full['qid'], 'delta': {
            metric: core['metrics'][metric] - value for metric, value in full['metrics'].items()
        }})
    return {
        'experiment': 'query_expression', 'status': 'manual_unblinded_development_ablation',
        'n_questions': len(targets), 'snapshot_id': SNAPSHOT_ID,
        'model': manifest['model'], 'index_text': 'body',
        'plan_sha256': sha256(plan_path), 'questions_sha256': sha256(qp),
        'chunks_sha256': manifest['chunks_sha256'], 'embedding_sha256': manifest['embedding_sha256'],
        'baseline_reproduced': True, 'scope_filters_applied': False,
        'limitations': [plan['limitation'],
                       'Scope text is preserved, but applicability is not automatically enforced.',
                       'Literal presence is not semantic or temporal validation.',
                       'Shortening and paraphrasing change together; not a single-token causal test.'],
        'variants': by_variant, 'comparisons': comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.root)
    out = args.root / 'experiments/ablation_query_expression.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Fixed v2: full vs manual core; scope filters NOT applied')
    print('qid   full coverage@5  core coverage@5  delta')
    for full, core in zip(result['variants']['full']['per_question'], result['variants']['manual_core']['per_question']):
        before, after = full['metrics']['coverage@5'], core['metrics']['coverage@5']
        print(f"{full['qid']}       {before:.3f}           {after:.3f}       {after-before:+.3f}")
    for name, variant in result['variants'].items():
        print(name, json.dumps(variant['overall'], ensure_ascii=False))
    print(f'Output: {out}')


if __name__ == '__main__':
    main()
