# FastAPI 검색 API 전환

기존 `ThreadingHTTPServer` 기반 로컬 서버를 FastAPI와 Uvicorn으로 바꿨다. 검색 알고리즘과 화면 응답 형식은 유지하고 API 계약, 상태 확인, 자동 문서화를 추가하는 작업이다. 유료 생성 API는 연결하지 않았다.

## 구현

- `GET /health`: 서버와 검색기 준비 상태 확인
- `GET /api/info`: 검색 모델과 색인 정보
- `POST /api/search`: 질문, 결과 수, 학교급을 검증한 뒤 원문 검색
- `GET /openapi.json`, `GET /docs`: OpenAPI 스키마와 Swagger UI
- `GET /source/{doc_id}.pdf`: 색인에 등록된 원문 PDF만 제공
- `/`, `/app.css`, `/presentation.js`, `/app.js`: 기존 검색 화면 제공

검색 요청은 Pydantic 모델로 제한했다. 질문은 1~4000자, 결과 수는 1~20이며 정의하지 않은 필드는 거부한다. 기존과 같이 외부 Host와 Origin, 과도한 요청 본문, 임의 파일 경로는 차단한다. 서버는 `127.0.0.1`에만 바인딩한다.

Dense 모델은 FastAPI lifespan에서 한 번 불러와 모든 요청이 공유한다. 모델 추론에는 기존 lock을 유지해 같은 프로세스에서 동시 추론이 겹치지 않게 했다. 이 구성은 [FastAPI lifespan 문서](https://fastapi.tiangolo.com/advanced/events/)의 공유 자원 초기화 방식과 [Uvicorn 실행 문서](https://fastapi.tiangolo.com/deployment/manually/)를 참고했다.

## 확인

- Python 테스트 75개 통과
- JavaScript 테스트 8개 통과 및 두 브라우저 스크립트 구문 검사 통과
- 실제 BGE-m3 색인 1,331개 청크로 서버 실행
- `/health`, `/api/info`, `/openapi.json`, `/` 응답 200 확인
- `출결`, 중학교 필터, top-3 요청에서 `8조 출결상황` 3건 반환 확인
- 실제 검색 응답의 `generation_called`가 `false`인지 확인

CI도 `uv.lock`에 고정된 의존성을 설치한 뒤 테스트하도록 수정했다. FastAPI 자체가 배포 구성을 완성하는 것은 아니므로, 외부 서비스화 전에 Docker 이미지, 프록시, 인증, 요청 제한, Qdrant 서버 운영을 별도로 설계한다.
