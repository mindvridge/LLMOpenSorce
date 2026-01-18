# LLM API Server

vLLM-MLX 및 OpenAI 기반 LLM API 서버

## 특징

- OpenAI API 호환 (Chat Completions)
- 스트리밍 지원
- vLLM-MLX (Apple Silicon 최적화, Continuous Batching)
- OpenAI GPT 모델 지원 (클라우드 백업)
- RAG 문서 검색 기능
- 자동 로드밸런싱 (로컬 우선, 초과 시 클라우드)
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
# .env 파일을 열어 OPENAI_API_KEY 등 설정
```

### 3. vLLM-MLX 서버 시작 (별도 터미널)

```bash
# vLLM-MLX 서버 (포트 8001)
mlx_vlm_server --model mlx-community/Qwen3-30B-A3B-4bit --port 8001
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
    "model": "vllm-qwen3-30b-a3b",
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
    "model": "vllm-qwen3-30b-a3b",
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
    model="vllm-qwen3-30b-a3b",
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
│   ├── load_balancer.py     # 자동 로드밸런싱
│   ├── clients/
│   │   ├── openai_client.py # OpenAI 연동
│   │   ├── mlx_client.py    # MLX 연동
│   │   └── vllm_mlx_client.py # vLLM-MLX 연동
│   ├── rag/                  # RAG 모듈
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic 모델 (OpenAI 호환)
│   └── routers/
│       ├── __init__.py
│       ├── chat.py          # /v1/chat/completions
│       ├── models.py        # /v1/models
│       ├── health.py        # /health
│       ├── rag.py           # RAG API
│       └── prompts.py       # 프롬프트 관리
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
- `GET /chat-streaming` - 스트리밍 채팅 UI
- `GET /dashboard` - 모니터링 대시보드
- `GET /docs` - API 문서 (Swagger UI)

## 설정 파일

[config.yaml](config.yaml)에서 다음 항목을 설정할 수 있습니다:

- 서버 포트, 워커 수
- vLLM-MLX 연결 정보
- OpenAI API 설정
- 기본 모델
- 사용 가능한 모델 목록
- 로드밸런싱 설정
- CORS 설정

## 라이선스

MIT
