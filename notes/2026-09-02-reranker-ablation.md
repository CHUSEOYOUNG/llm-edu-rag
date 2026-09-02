# CrossEncoder reranker 비교

## 결론

`BAAI/bge-reranker-v2-m3`로 Dense top-20을 다시 정렬했지만 현재 기본 검색에는 넣지 않는다. Complete@5는 `0.818`로 Dense와 같았고, 로컬 CPU에서 질문 하나의 후보 20개를 재정렬하는 데 평균 12.8초가 걸렸다.

Reranker는 q006의 두 근거를 6·16위에서 1·3위로 올렸지만 q004의 근거를 2위에서 6위로 내렸다. 그 결과 Complete@5의 순개선은 없었다. MRR@20은 `0.636 → 0.700`, Complete@10은 `0.818 → 0.909`로 올랐으므로 후보 모델 자체는 후속 검증 대상으로 남긴다.

## 모델을 고른 이유

[FlagEmbedding 공식 문서](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/inference/reranker/README.md)는 `bge-reranker-v2-m3`를 multilingual encoder reranker로 분류하고, 질문과 passage 쌍에서 직접 관련도 점수를 계산하는 사용법을 제공한다. 현재 Dense 검색기인 BGE-M3와 같은 계열이고 한국어 입력을 다룰 수 있어 첫 학습형 후보로 골랐다.

- 모델: `BAAI/bge-reranker-v2-m3`
- Hugging Face revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- 모델 파일 SHA-256: `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`
- 라이선스: [Apache-2.0](https://huggingface.co/BAAI/bge-reranker-v2-m3)

## 실험 조건

- 평가셋: `v2-development-11q-2026-08-27`의 검토 완료 질문 11개, 근거 그룹 17개
- 1차 검색: 고정된 BGE-M3 body-only Dense top-20
- reranker 입력: 원문 질문과 청크 `body`
- 최대 입력 길이: 1,024토큰
- batch size: 4
- dtype / 장치: float32 / CPU
- 지연시간: 모델 로딩과 1회 워밍업을 제외하고 질문별 후보 20개 추론만 측정

전체 220개 질문·본문 쌍 중 3개가 1,024토큰을 넘었다. 모두 q003의 Dense 9·15·20위에 있던 비정답 장문 표였으며 정답 근거는 Dense 1위에 있었다. 가장 긴 입력은 자르기 전 2,629토큰이었다.

고정 Dense 파일의 모델명, 질문·청크·임베딩 지문, 평가 깊이를 평가 스냅샷과 대조했다. Reranker는 이 후보들의 순서만 바꾸며 top-20 밖의 문서를 새로 찾지 않는다.

## 결과

| 방식 | Complete@1 | Complete@5 | Complete@10 | Complete@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.455 | **0.818** | 0.818 | 0.909 | 0.636 |
| Dense → reranker | 0.455 | **0.818** | **0.909** | 0.909 | **0.700** |

Dense top-20 안에서 모든 근거를 찾을 수 있는 질문은 10/11이므로 이 후보군에서 가능한 Complete@5의 상한도 10/11이다. q008 근거는 후보군에 없어서 Reranker로 해결할 수 없다.

### 질문별 근거 순위

| 질문 | Dense | Dense → reranker | Complete@5 변화 |
|---|---:|---:|---:|
| q001 | 1·1 | 5·5 | 1 → 1 |
| q002 | 3·3 | 1·1 | 1 → 1 |
| q003 | 1 | 1 | 1 → 1 |
| q004 | 2·2 | 6·6 | **1 → 0** |
| q005 | 1 | 1 | 1 → 1 |
| q006 | 6·16 | 1·3 | **0 → 1** |
| q007 | 2·4 | 1·4 | 1 → 1 |
| q008 | >20 | >20 | 0 → 0 |
| q009 | 2·2 | 3·3 | 1 → 1 |
| q010 | 1 | 1 | 1 → 1 |
| q011 | 1 | 1 | 1 → 1 |

q004에서는 `수업시수`가 등장하는 출결 문서와 중학교 자료가 정답인 초등학교 시간 배당 근거보다 앞에 배치됐다. Reranker 입력에 구조 경로를 넣지 않았으므로 초등학교라는 범위를 본문만으로 구분해야 했다. 결과를 본 뒤 q004에 맞춰 경로나 점수 혼합 비율을 추가하면 개발셋 과적합이 되므로 이번 성적으로 보고하지 않는다.

q006은 Dense top-20에 초등학교와 중학교 근거가 모두 있었고 Reranker가 둘을 1·3위로 올렸다. 이것이 Complete@10과 MRR 개선의 주된 원인이다.

## 지연시간

| 측정값 | CPU 시간 |
|---|---:|
| 평균 / 질문 | 12,767 ms |
| 중앙값 / 질문 | 12,670 ms |
| p95 / 질문 | 17,614 ms |
| 11개 질문 합계 | 140,435 ms |

Dense 검색 시간과 모델 로딩은 포함하지 않았다. 한 번의 로컬 실행 결과라 다른 장치와 직접 비교할 수 없지만, 현재 검색 화면의 매 요청에 적용하기에는 충분히 느리다.

## 재현

모델 파일은 Git에 포함하지 않는다. 처음 한 번은 아래처럼 고정 revision을 Hugging Face 캐시에 받는다.

```sh
uv run python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    allow_patterns=[
        "config.json", "model.safetensors", "sentencepiece.bpe.model",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    ],
)
PY
```

그다음 실험은 네트워크 없이 실행한다.

```sh
UV_CACHE_DIR=/tmp/uv-cache \
  uv run --offline --no-sync python src/ablation_reranker.py
```

전체 점수와 질문별 순서는 `experiments/ablation_reranker.json`에 저장된다.

## 한계와 다음 단계

같은 11개 개발 질문을 앞선 설계와 평가에도 사용했으므로 독립 테스트 성능이 아니다. BGE reranker 한 종류만 비교했고 GPU, 양자화, 작은 모델, candidate depth에 따른 속도 차이는 측정하지 않았다. 생성 답변 품질도 이 점수에 포함되지 않는다.

현재 검색기는 Dense를 유지한다. 다음에는 새 질문을 추가해 개발셋과 테스트셋을 나눈 뒤 이 reranker를 다시 평가한다. 서비스 적용을 검토할 때는 더 작은 multilingual reranker나 가속 장치에서 정확도와 지연시간을 함께 비교한다.
