# 데이터 준비

원본 PDF와 생성된 청크·임베딩은 저작권과 파일 크기 때문에 Git에 올리지 않는다. 저장소를 새로 받은 경우 아래 순서로 로컬 색인을 만든다.

## 사용한 자료

수집일은 2026-07-27이며 교육부와 국가교육과정정보센터(NCIC)에서 공개한 PDF를 사용했다.

| 자료 | 출처 | 메모 |
|---|---|---|
| 2022 개정 초·중등학교 교육과정 별책 1 총론 | NCIC | 학교급별 교육과정과 시행 시점 |
| 2026 학교생활기록부 기재요령 초·중·고 | 교육부 | 조항 구조와 표가 많은 문서 |
| 한국어 교육과정 별책 41 | NCIC | 2017년 발행 자료 |

현재 평가 결과를 그대로 재현하려면 당시 사용한 것과 같은 판본과 파일명이 필요하다. 다른 교육 PDF로도 검색 화면을 실행할 수 있지만 기존 평가셋의 청크 ID와 점수는 재현되지 않는다.

## 1. 원문 넣기

PDF를 `data/raw/` 아래에 둔다. 파일명에서 확장자를 뺀 값이 `doc_id`가 되므로 학교급과 연도가 드러나는 원래 파일명을 유지하는 편이 좋다.

```text
data/
  raw/
    2026 학교생활기록부 기재요령(초)_F_260219.pdf
    2026 학교생활기록부 기재요령(중)_F_260227.pdf
    2026 학교생활기록부 기재요령(고)_F_260219.pdf
    ...
```

`data/raw/`와 `data/processed/`는 `.gitignore`에 포함되어 있다.

## 2. PDF 파싱과 청킹

저장소 루트에서 차례대로 실행한다.

```sh
uv sync --locked
uv run python src/parse.py
uv run python src/build_sections.py
uv run python src/chunk.py
```

생성되는 주요 파일은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `data/processed/docs.jsonl` | 정규화한 문서 본문과 PDF 페이지 경계 |
| `data/processed/sections.jsonl` | 제목 구조를 반영한 섹션 |
| `data/processed/chunks.jsonl` | 검색에 사용하는 본문 청크와 metadata |
| `data/processed/sample/*.md` | 파싱 결과를 눈으로 확인하기 위한 문서 |

청크에는 `doc_id`, 구조 경로인 `path`, 검색 본문인 `body`, `page_start`, `page_end`를 보존한다. 임베딩에는 `body`만 사용한다.

## 3. Dense 색인 만들기

```sh
uv run python src/build_dense_index.py
```

처음 실행하면 `BAAI/bge-m3` 모델을 내려받을 수 있다. 이미 모델을 로컬에 저장했고 네트워크를 사용하지 않으려면 다음 옵션을 붙인다.

```sh
uv run python src/build_dense_index.py --local-files-only
```

완료되면 아래 두 파일이 만들어지거나 갱신된다.

- `data/processed/embeddings.npy`: 정규화한 body 임베딩
- `config/dense_index.json`: 모델명과 청크·임베딩 SHA-256 지문

검색기는 두 지문이 현재 파일과 다르면 오래된 임베딩을 잘못 쓰지 않도록 실행을 중단한다. 개인 자료로 색인을 다시 만들면 추적 중인 `config/dense_index.json`이 수정될 수 있으므로 공식 코퍼스를 갱신하는 작업이 아니라면 커밋하지 않는다.

## 4. 확인

```sh
uv run python src/rag.py "중학교 출결"
uv run python src/search_app.py
```

웹 화면은 <http://127.0.0.1:8765/>에서 확인한다. PDF 파일이 `data/raw/`에 남아 있으면 검색 결과의 `원문에서 확인하기` 링크도 사용할 수 있다.
