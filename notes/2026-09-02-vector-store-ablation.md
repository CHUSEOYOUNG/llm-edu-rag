# NumPy와 Qdrant 로컬 저장소 비교

## 결론

Qdrant 영속 로컬 색인을 구현했지만 현재 웹 검색의 기본 저장소는 NumPy로 유지한다. 1,331개 규모에서는 NumPy가 평균 0.143ms, Qdrant 로컬이 2.473ms로 NumPy가 훨씬 빨랐다. Qdrant도 평균 3ms 미만이라 절대 지연은 작고, 프로세스를 종료한 뒤 색인과 본문 metadata를 다시 불러오는 영속 저장소라는 장점이 있다.

두 방식의 검색 평가 지표는 완전히 같았다. 청크 ID의 top-20 순서가 정확히 일치한 질문은 5/11이었지만, 달라진 항목은 학교급별 PDF에 중복된 동일 경로·동일 본문이었다. 내용을 기준으로 비교하면 top-5와 top-20 순서가 11/11 모두 같았다.

## 구현 범위

[Qdrant Python Client 공식 저장소](https://github.com/qdrant/qdrant-client)는 `QdrantClient(path=...)` 형태의 디스크 영속 로컬 모드를 제공한다. 별도 서버 없이 같은 클라이언트 API를 사용할 수 있어 개발과 테스트용 백엔드로 선택했다.

- collection: `education_chunks`
- distance: Cosine
- point: 1,331개
- vector: BGE-M3 body-only Dense, 1,024차원
- payload: `chunk_id`, 문서명, 구조 경로, 본문, 글자 수, PDF 페이지
- point ID: 기존 청크 행 번호와 같은 정수 ID
- 저장 경로: `data/processed/qdrant/`
- 클라이언트: `qdrant-client 1.19.0`

원문 PDF, 청크와 마찬가지로 생성된 Qdrant 저장소는 Git에 넣지 않는다. 대신 collection 설정과 입력 지문은 `config/qdrant_index.json`에 기록한다. 현재 Dense 청크나 임베딩 파일이 달라지면 저장소를 열기 전에 지문 불일치로 중단한다.

## 실험 조건

- 평가셋: `v2-development-11q-2026-08-27`의 검토 완료 질문 11개
- 질문 벡터: `BAAI/bge-m3`로 다시 생성하고 고정 Dense top-20과 정확히 일치하는지 확인
- NumPy: 정규화 행렬과 질문 벡터의 내적 후 top-20 정렬
- Qdrant: 같은 정규화 벡터를 Cosine collection에 저장하고 `query_points`로 top-20 검색
- 반복: 질문 11개 × 30회, 방식 실행 순서를 반복마다 교대해 총 330표본씩 측정
- 측정 범위: 벡터 검색과 결과 payload 구성. 질문 임베딩 생성은 제외

Qdrant의 Python 로컬 모드는 [구현 설명](https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/local/qdrant_local.py)에 명시된 정확한 brute-force 검색이다. 이 결과를 Qdrant 서버의 HNSW 검색 성능으로 해석하면 안 된다.

## 검색 결과 동일성

| 확인 항목 | 결과 |
|---|---:|
| top-5 청크 ID 순서 완전 일치 | 10/11 |
| top-20 청크 ID 순서 완전 일치 | 5/11 |
| top-5 경로·본문 순서 일치 | **11/11** |
| top-20 경로·본문 순서 일치 | **11/11** |
| 평균 top-20 ID 중복률 | 0.995 |
| 가장 낮은 top-20 ID 중복률 | 0.950 |
| 공유 결과 점수 최대 절대 차이 | 1.43e-7 |
| 전체 검색 평가 지표 | **동일** |

q002·q005·q006·q007·q008·q009에서 일부 ID 순서가 달랐다. 순서가 교환된 청크들은 다른 학교급 파일에 존재하는 동일한 구조 경로와 본문이었다. q008에서는 20위 경계의 ID 하나가 바뀌어 top-20 ID 집합 중복률이 95%였지만 두 청크의 경로와 본문은 같고 둘 다 정답 근거가 아니었다.

따라서 이번 차이는 검색 의미의 회귀가 아니라 동점인 중복 문서의 정렬 규칙 차이다. 학교급이나 문서 판본을 반드시 구분해야 하는 질문에서는 동점 처리에 metadata 우선순위를 별도로 정해야 한다.

## 속도와 저장 공간

| 항목 | NumPy | Qdrant 로컬 |
|---|---:|---:|
| 평균 검색 | **0.143 ms** | 2.473 ms |
| 중앙값 | **0.142 ms** | 2.457 ms |
| p95 | **0.158 ms** | 2.578 ms |
| 저장 공간 | **9.09 MiB** | 15.28 MiB |

Qdrant 저장 공간은 NumPy 청크 JSONL과 임베딩 NPY를 합친 크기의 1.68배였다. Qdrant에는 검색 결과를 바로 구성할 수 있도록 본문과 출처 metadata까지 payload로 중복 저장했다.

- 새 collection 생성과 1,331개 point 저장: 1,007ms
- 검증을 포함한 재로딩: 116ms
- 재로딩 직후 첫 검색: 3.48ms

이 수치는 한 컴퓨터에서 실행한 소규모 microbenchmark다. 코퍼스가 커지거나 동시 요청이 발생할 때의 처리량을 보여주지 않는다.

## 재현

Dense 청크와 임베딩이 준비된 상태에서 색인을 만든다.

```sh
uv run python src/qdrant_store.py --recreate
```

NumPy와 Qdrant 비교 전체를 다시 실행하려면 다음 명령을 사용한다.

```sh
UV_CACHE_DIR=/tmp/uv-cache \
  uv run --offline --no-sync python src/ablation_vector_store.py
```

결과는 `experiments/ablation_vector_store.json`에 저장된다. 실험은 Qdrant 색인을 지우고 다시 만들어 build와 reload 시간까지 측정한다.

## 결정과 다음 단계

현재 규모와 단일 사용자 로컬 화면에서는 NumPy가 단순하고 빠르므로 기본값을 바꾸지 않는다. Qdrant 구현은 저장소 영속성, payload 조회, 향후 filtering과 서버 전환을 확인한 선택 백엔드로 유지한다.

실제 배포 단계에서는 Docker의 Qdrant 서버를 사용해 HNSW 설정, 동시 요청, 필터 검색, 재색인 전략을 따로 측정해야 한다. 이번 결과는 그 서버 실험을 대신하지 않는다.
