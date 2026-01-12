# 빠른 시작 가이드

## 1. 설치

```bash
cd llm-api-server

# 가상환경 생성 (선택)
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# OpenAI SDK 설치 (테스트용)
pip install openai
```

## 2. 서버 시작

### 방법 1: 스크립트 사용 (권장)

```bash
./scripts/start_server.sh
```

### 방법 2: 직접 실행

```bash
# Ollama 실행 (별도 터미널)
ollama serve

# API 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 3. 테스트

### 방법 1: Bash 스크립트

```bash
./scripts/test_api.sh
```

### 방법 2: Python 클라이언트

```bash
python scripts/test_client.py
```

### 방법 3: cURL

```bash
# Health Check
curl http://localhost:8000/health

# 모델 목록
curl http://localhost:8000/v1/models

# 채팅 완성
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-ultra:latest",
    "messages": [
      {"role": "user", "content": "안녕하세요"}
    ]
  }'
```

## 4. OpenAI SDK 사용

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# 일반 응답
response = client.chat.completions.create(
    model="qwen3-ultra:latest",
    messages=[
        {"role": "user", "content": "안녕하세요"}
    ]
)
print(response.choices[0].message.content)

# 스트리밍
stream = client.chat.completions.create(
    model="qwen3-ultra:latest",
    messages=[
        {"role": "user", "content": "1부터 5까지 세어주세요"}
    ],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 5. API 문서

서버 실행 후 브라우저에서 접속:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 6. 설정

[config.yaml](config.yaml)에서 다음을 변경할 수 있습니다:

- 기본 모델: `default_model`
- 사용 가능한 모델: `available_models`
- 서버 포트: `server.port`
- Ollama URL: `ollama.base_url`
- CORS 설정: `cors.allowed_origins`

## 7. 다음 단계

1단계 완료:
- ✅ 기본 구조
- ✅ Ollama 클라이언트 구현
- ✅ /v1/chat/completions (스트리밍 포함)
- ✅ /v1/models
- ✅ /health

2단계: API Key 인증
3단계: Rate Limiting
4단계: 관리자 API
5단계: Cloudflare Tunnel
6단계: Gradio UI

## 문제 해결

### Ollama 연결 실패

```bash
# Ollama 실행 확인
pgrep ollama

# Ollama 시작
ollama serve
```

### 포트 충돌

config.yaml에서 포트 변경:
```yaml
server:
  port: 8001  # 다른 포트로 변경
```

또는 환경변수 사용:
```bash
SERVER_PORT=8001 uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### 모델 다운로드

```bash
# 기본 모델 다운로드
ollama pull qwen3-ultra:latest

# 다른 모델 다운로드
ollama pull qwen2.5:14b
ollama pull llama3.2:8b
```

## 유용한 명령어

```bash
# 설치된 모델 확인
ollama list

# 모델 삭제
ollama rm model-name

# 서버 로그 확인 (백그라운드 실행 시)
tail -f /path/to/log/file

# 서버 종료
pkill -f "uvicorn app.main:app"
```
