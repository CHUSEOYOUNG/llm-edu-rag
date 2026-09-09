# 애플리케이션 백엔드

검색 화면과 데이터베이스 사이를 담당하는 Spring Boot 백엔드다. 브라우저 요청을 기존 FastAPI 검색 서비스로 전달하고 검색 질문과 실제 검색문을 PostgreSQL에 저장한다.

```text
브라우저
  └─ Spring Boot :8080
       ├─ PostgreSQL :5432  검색 기록
       └─ FastAPI :8765     검색·RAG·Ollama
```

## IntelliJ에서 열기

1. IntelliJ에서 `backend/build.gradle`을 연다.
2. Gradle 프로젝트로 불러오고 Project SDK를 Java 17로 선택한다.
3. 아래 설명대로 PostgreSQL을 준비한다.
4. FastAPI를 `uv run python src/search_app.py`로 실행한다.
5. `EduRagBackendApplication`의 실행 버튼을 누른다.

전역 Gradle 설치는 필요 없다. 저장소에 포함한 `gradlew`와 Gradle Wrapper를 IntelliJ가 사용한다.

백엔드 상태는 <http://127.0.0.1:8080/actuator/health>에서 확인한다.

## macOS에서 PostgreSQL 준비

현재 개발용 맥에서는 VM 디스크를 크게 잡는 Colima보다 약 122MB인 [Postgres.app 18](https://postgresapp.com/downloads.html)을 권장한다. 앱을 `/Applications`로 옮기고 실행한 뒤 `Initialize`를 한 번 누른다. 그다음 저장소 루트에서 프로젝트용 계정과 DB를 만든다.

```sh
/Applications/Postgres.app/Contents/Versions/latest/bin/psql \
  postgres -f backend/scripts/create-local-database.sql
```

Docker Desktop을 이미 사용하고 있거나 다른 컴퓨터에서 같은 환경을 재현하려면 Postgres.app 대신 Compose를 사용할 수 있다.

```sh
docker compose -f compose.dev.yaml up -d postgres
```

## API 확인

`client_id`는 브라우저 또는 기기를 구분하기 위한 UUID다. 로그인 기능을 붙이기 전까지 개인 식별 정보 대신 임의 UUID만 저장한다.

```sh
CLIENT_ID=$(uuidgen | tr '[:upper:]' '[:lower:]')

curl -X POST http://127.0.0.1:8080/api/v1/search \
  -H 'Content-Type: application/json' \
  -d "{\"client_id\":\"$CLIENT_ID\",\"question\":\"중학교 출결\",\"top_k\":3,\"school_level\":\"middle\"}"

curl -X POST http://127.0.0.1:8080/api/v1/answer \
  -H 'Content-Type: application/json' \
  -d "{\"client_id\":\"$CLIENT_ID\",\"question\":\"중학교 수업은 몇 분인가요?\",\"top_k\":2,\"school_level\":\"middle\"}"

curl "http://127.0.0.1:8080/api/v1/search-history?clientId=$CLIENT_ID"

curl -X DELETE "http://127.0.0.1:8080/api/v1/search-history?clientId=$CLIENT_ID"
```

## DBeaver 연결

[DBeaver Community](https://dbeaver.io/download/)를 설치한 뒤 새 PostgreSQL 연결을 만들고 아래 값을 입력한다.

| 항목 | 값 |
|---|---|
| Host | `127.0.0.1` |
| Port | `5432` |
| Database | `edu_rag` |
| Username | `edu_rag` |
| Password | `edu_rag_local` |

처음 연결할 때 PostgreSQL JDBC 드라이버 다운로드 창이 뜨면 다운로드한다. `public.search_history` 테이블은 백엔드 시작 시 Flyway가 자동으로 만든다. 위 계정 정보는 로컬 개발용이며 운영 환경에서는 환경 변수 `DB_URL`, `DB_USERNAME`, `DB_PASSWORD`로 바꾼다.

## 종료

Postgres.app을 사용했다면 메뉴에서 서버를 정지한다. Compose를 사용했다면 아래 명령을 실행한다.

```sh
docker compose -f compose.dev.yaml down
```

DB 데이터까지 초기화하려는 경우에만 `down -v`를 사용한다. `-v`를 붙이면 저장한 검색 기록이 모두 삭제된다.
