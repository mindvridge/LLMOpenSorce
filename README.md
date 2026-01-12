# LLM API Server

Ollama 기반 OpenAI 호환 LLM API 서버

## 특징

- OpenAI API 호환 (Chat Completions)
- 스트리밍 지원
- 여러 오픈소스 LLM 모델 지원
- FastAPI + Uvicorn
- Mac Studio M4 Max 64GB 최적화

## 설치

### 1. 의존성 설치

```bash
cd llm-api-server
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 필요한 값 수정
```

### 3. Ollama 모델 다운로드 (선택)

```bash
# 기본 모델이 이미 설치되어 있다면 생략 가능
ollama pull qwen2.5:14b
```

## 실행

### 개발 모드

```bash
cd llm-api-server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

또는:

```bash
python app/main.py
```

### 프로덕션 모드

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API 사용 예시

### Health Check

```bash
curl http://localhost:8000/health
```

### 모델 목록

```bash
curl http://localhost:8000/v1/models
```

### 채팅 완성 (비스트리밍)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-ultra:latest",
    "messages": [
      {"role": "user", "content": "안녕하세요"}
    ]
  }'
```

### 채팅 완성 (스트리밍)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-ultra:latest",
    "messages": [
      {"role": "user", "content": "파이썬으로 피보나치 함수를 작성해줘"}
    ],
    "stream": true
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="qwen3-ultra:latest",
    messages=[
        {"role": "user", "content": "한국의 수도는?"}
    ]
)
print(response.choices[0].message.content)
```

## 디렉토리 구조

```
llm-api-server/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱 진입점
│   ├── config.py            # 설정 관리
│   ├── ollama_client.py     # Ollama 연동
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic 모델 (OpenAI 호환)
│   └── routers/
│       ├── __init__.py
│       ├── chat.py          # /v1/chat/completions
│       ├── models.py        # /v1/models
│       └── health.py        # /health
├── config.yaml
├── requirements.txt
├── .env.example
└── README.md
```

## API 엔드포인트

- `GET /` - API 정보
- `GET /health` - 서버 상태
- `GET /v1/models` - 모델 목록
- `GET /v1/models/{model_id}` - 모델 정보
- `POST /v1/chat/completions` - 채팅 완성
- `GET /docs` - API 문서 (Swagger UI)

## 설정 파일

[config.yaml](config.yaml)에서 다음 항목을 설정할 수 있습니다:

- 서버 포트, 워커 수
- Ollama 연결 정보
- 기본 모델
- 사용 가능한 모델 목록
- CORS 설정

## 다음 단계

1단계 완료:
- [x] 기본 구조
- [x] Ollama 클라이언트 구현
- [x] /v1/chat/completions 엔드포인트 (스트리밍 포함)
- [x] /v1/models 엔드포인트
- [x] /health 엔드포인트

2단계 예정:
- [ ] API Key 인증
- [ ] SQLite 데이터베이스
- [ ] 사용량 추적

3단계 예정:
- [ ] Rate Limiting
- [ ] 요청 로깅

4단계 예정:
- [ ] 관리자 API
- [ ] 통계 및 로그 조회

5단계 예정:
- [ ] Cloudflare Tunnel 설정

6단계 예정:
- [ ] Gradio 관리 UI

## 라이선스

MIT
