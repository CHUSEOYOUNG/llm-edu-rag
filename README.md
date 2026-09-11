# llm-edu-rag

[![tests](https://github.com/CHUSEOYOUNG/llm-edu-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/CHUSEOYOUNG/llm-edu-rag/actions/workflows/tests.yml)

학교생활기록부 기재요령과 교육과정 PDF에서 질문에 맞는 내용을 찾고, 출처가 확인된 답변을 만드는 프로젝트다.

교육 문서를 직접 찾아볼 때 가장 불편했던 점은 문서가 길고, 같은 표현이 초·중·고 자료에 반복된다는 것이었다. 단순히 PDF 검색창을 만드는 대신 자연어로 질문하고 원문까지 바로 확인할 수 있는 형태로 구현했다. 검색 품질을 감으로 고치지 않기 위해 작은 평가셋부터 만들고, 변경할 때마다 검색과 답변 생성을 따로 측정하고 있다.

현재 데이터는 교육부와 국가교육과정정보센터에서 공개한 PDF 6개이며, 문서 구조에 따라 나눈 1,331개 청크를 검색한다. 기본 실행은 모두 로컬이라 별도 API 비용이 들지 않는다.

## 현재 가능한 것

- `작년 생기부 내용을 지금 고칠 수 있나요?` 같은 구어체 질문 검색
- `출결`처럼 짧은 키워드 검색과 초·중·고 학교급 필터
- `그럼 중학교는?`처럼 직전 질문을 이어서 검색
- 초등학교와 중학교를 함께 묻는 질문의 학교급별 검색
- 관련 원문, 문서명, PDF 페이지 확인
- Ollama Qwen3 4B를 이용한 답변 생성과 출처 검사
- PostgreSQL을 이용한 최근 질문 저장과 삭제
- FastAPI 검색 서버, Spring Boot 애플리케이션 API, Docker 실행 환경

화면에는 검색 점수나 청크 ID 대신 항목명, 학교급, 페이지처럼 사용자가 원문을 확인하는 데 필요한 정보만 표시한다.

## 화면

질문을 입력하면 관련 교육 자료를 찾고, 필요한 경우 로컬 모델로 답변을 정리한다.

![학교생활 안내 검색 화면](docs/images/search-home.png)

검색 결과는 같은 항목끼리 묶어 보여주며 원문 PDF 페이지를 바로 열 수 있다.

![출결 검색 결과와 원문 확인 화면](docs/images/search-results.png)

## 동작 방식

```text
PDF
 └─ 페이지 정보와 제목 구조를 보존해 파싱
     └─ section / chunk 생성
         └─ 본문만 BGE-m3로 임베딩
             └─ Dense 검색
                 ├─ 짧은 키워드 재정렬
                 ├─ 날짜·고시명과 질문 내용 분리 검색
                 ├─ 학교급 비교 질문 분리 검색
                 └─ 관련 원문과 PDF 페이지 반환
                     └─ Ollama 답변 생성 → 출처 검사
```

문서명과 경로는 결과 설명에 사용하지만 임베딩에는 본문만 넣는다. 초기에 경로와 본문을 함께 임베딩했더니 학교급과 문서명이 과하게 반영됐다. 본문만 사용했을 때 R@5는 `0.659 → 0.750`, MRR@20은 `0.697 → 0.777`로 올랐다.

답변 생성 시에는 검색된 문단에 `[S1]`, `[S2]` 같은 번호를 붙인다. 모델은 답변과 사용한 번호만 반환하고, 실제 인용문은 프로그램이 원문에서 다시 가져온다. 존재하지 않는 출처 번호, 원문과 맞지 않는 인용, 학교급이나 기간 조건이 빠진 답변은 그대로 보여주지 않는다.

## 평가 결과

현재 회귀셋은 답변 가능한 질문 11개와 코퍼스에 답이 없는 질문 2개로 구성했다. 표본이 작고 같은 질문으로 개선 방향도 정했기 때문에 아래 수치는 **개발 중 회귀 확인용**이다. 일반 성능이나 최종 테스트 성능으로 보기는 어렵다.

| 구분 | 지표 | 결과 |
|---|---|---:|
| 검색 | Recall@5 | 1.0000 |
| 검색 | Complete@5 | 1.0000 |
| 검색 | MRR@10 | 0.8939 |
| 생성 | 답변 가능 질문 성공률 | 0.9091 |
| 생성 | 답변 불가 질문 거절률 | 1.0000 |
| 생성 | 출처 형식 검증 통과율 | 0.9091 |

`Complete@5`는 답변에 필요한 근거가 여러 개일 때 그 근거를 모두 상위 5개 안에서 찾았는지 확인하는 지표다. 초등학교와 중학교 수업 시간을 함께 묻는 q006은 학교급별 보조 검색을 적용한 뒤 두 근거를 모두 찾게 됐다.

현재 남은 실패는 q011이다. 검색은 정답 표를 1위로 찾지만, 로컬 모델이 2년 기준 값을 연간 값으로 표현해 검증 단계에서 답변을 막는다. 다음 개선 대상은 검색기가 아니라 표의 기간 정보를 구조적으로 전달하고 해석하는 부분이다.

전체 결과는 [검색·생성 회귀 결과](experiments/regression_generation_school_comparison.json), 질문 구조별 차이는 [실패 유형 분석](notes/2026-09-11-failure-type-analysis.md)에 남겼다.

## 실험하면서 내린 선택

| 실험 | 결과 | 현재 적용 여부 |
|---|---|---|
| 본문+경로 vs 본문 임베딩 | 본문만 넣었을 때 검색 성능이 더 높았음 | 본문만 사용 |
| Dense vs BM25/RRF | 현재 질문셋에서는 RRF가 Dense의 상위 순위를 개선하지 못함 | Dense 사용 |
| 구조 기반 vs overlap/고정 길이 청킹 | 구조 기반 방식의 Complete@5가 가장 높았음 | 구조 기반 사용 |
| CrossEncoder reranker | MRR은 올랐지만 Complete@5는 같고 CPU에서 질문당 평균 12.8초 소요 | 서비스에는 미적용 |
| NumPy vs Qdrant 로컬 모드 | 검색 결과는 같고 1,331개 규모에서는 NumPy가 더 단순하고 빨랐음 | NumPy 기본, Qdrant 선택 |
| 학교급별 질문 분해 | q006을 회복하고 전체 Complete@5가 `0.909 → 1.000`으로 상승 | 비교 질문에 적용 |

리랭커처럼 구현까지 했지만 사용하지 않은 기능도 실험 결과와 함께 보존했다. 각 실험의 조건과 질문별 결과는 [`notes/`](notes/)와 [`experiments/`](experiments/)에서 확인할 수 있다.

## 기술 구성

| 영역 | 사용 기술 |
|---|---|
| 문서 처리 | PyMuPDF4LLM, 구조 기반 청킹 |
| 검색 | BGE-m3, Sentence Transformers, NumPy, BM25 비교 실험 |
| 생성 | Ollama, Qwen3 4B Instruct, JSON Schema 출력 |
| Python API | FastAPI, Uvicorn |
| 애플리케이션 API | Java 17, Spring Boot, JPA, Flyway |
| 저장소 | PostgreSQL, Qdrant 로컬 모드 비교 |
| 실행·검증 | uv, Docker Compose, GitHub Actions, unittest |

Spring Boot는 검색 기록과 애플리케이션 요청을 맡고, 임베딩 검색과 Ollama 호출은 FastAPI가 처리한다. Python 모델 코드를 Java에서 다시 구현하지 않고 애플리케이션 영역과 모델 서빙 영역을 분리했다.

## 로컬 실행

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)가 필요하다. 원문 PDF, 생성된 청크, 임베딩 파일, 모델 캐시는 용량과 저작권 문제로 저장소에 포함하지 않았다. 처음 실행한다면 먼저 [데이터 준비 방법](data/README.md)에 따라 색인을 만든다.

```sh
uv sync --locked
uv run python src/search_app.py
```

브라우저에서 <http://127.0.0.1:8765/>를 연다. 포트를 바꾸려면 다음처럼 실행한다.

```sh
uv run python src/search_app.py --port 8766
```

### 로컬 답변 생성

[Ollama](https://ollama.com/download)를 설치하고 모델을 한 번 받아 둔다.

```sh
ollama pull qwen3:4b-instruct
```

Ollama가 실행 중이면 웹 화면에서 검색 직후 답변을 만들 수 있다. 기본 모델은 `OLLAMA_MODEL` 환경 변수로 바꿀 수 있다.

### Spring Boot와 PostgreSQL

애플리케이션 백엔드는 `backend/`에 분리했다. IntelliJ 실행 방법, PostgreSQL과 DBeaver 연결 정보는 [백엔드 개발 안내](backend/README.md)에 정리했다.

```sh
docker compose -f compose.dev.yaml up -d
cd backend
./gradlew bootRun
```

### Docker

```sh
docker compose up --build
```

Docker 이미지에는 원문과 모델을 넣지 않는다. 로컬의 `data/raw`, `data/processed`, Hugging Face 캐시를 읽기 전용으로 연결한다. 기본 Compose 설정은 Ollama 호출을 끈 검색 전용 구성이다.

## API 사용 예시

서버 실행 후 API 문서는 <http://127.0.0.1:8765/docs>에서 볼 수 있다.

```sh
curl -X POST http://127.0.0.1:8765/api/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"중학교 출결","top_k":3,"school_level":"middle"}'

curl -X POST http://127.0.0.1:8765/api/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"초등학교와 중학교 수업 한 시간은 각각 몇 분인가요?","top_k":5,"school_level":"all"}'
```

`/api/search`는 검색 결과만 반환하며 생성 모델을 호출하지 않는다. `/api/answer`는 검색을 다시 수행한 뒤 검증된 답변과 출처를 반환한다.

## 평가와 테스트

검색 평가만 실행:

```sh
uv run python eval/run_eval.py \
  --output experiments/regression_current.json
```

Ollama를 포함한 생성 평가 실행:

```sh
uv run python eval/run_eval.py --generate \
  --output experiments/regression_generation.json
```

전체 테스트:

```sh
uv run python -m unittest discover -s tests -v
node --test tests/test_presentation.cjs
cd backend && ./gradlew test
```

현재 Python 테스트는 146개, 화면 JavaScript 테스트는 12개다. GitHub Actions에서는 Python·JavaScript·Spring 테스트와 CPU용 Docker 이미지 빌드를 확인한다.

## 폴더 구성

```text
src/          문서 처리, 검색, RAG, 실험 코드
web/          검색 화면
backend/      Spring Boot API, JPA, Flyway
config/       Dense 색인 설정과 입력 파일 지문
data/         데이터 준비 문서와 로컬 원문/색인 위치
eval/         평가 질문, 근거 주석, 회귀 평가 코드
experiments/  실험 결과 JSON
notes/        실험 과정과 실패 분석
tests/        Python·JavaScript 테스트
```

## 아직 남은 일

- 검수된 질문을 30~50개로 늘리고 최종 평가용 질문을 따로 분리하기
- q011처럼 표의 기간과 단위를 해석해야 하는 질문 처리하기
- 답변과 인용문 사이의 의미적 일치까지 평가하기
- 문서 개정 이력과 실제 적용 시점을 더 엄격하게 확인하기
- Qdrant 서버 구성을 별도 배포 환경에서 검증하기

이 서비스의 답변은 교육 문서를 확인하기 위한 초안이다. 출처가 존재하는지는 검사하지만, 규정의 실제 적용 여부나 답변의 의미적 정확성을 모두 보장하지는 않는다. 중요한 판단에는 연결된 원문을 함께 확인해야 한다.
