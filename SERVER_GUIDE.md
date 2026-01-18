# LLM API 서버 운영 가이드

## 서버 구성

```
┌─────────────────────────────────────────────────────────┐
│                    LLM API Server                        │
├─────────────────────────────────────────────────────────┤
│  API 서버 (FastAPI)          http://localhost:8000      │
│  vLLM-MLX 서버               http://localhost:8001      │
├─────────────────────────────────────────────────────────┤
│  로드밸런싱: 1~4명 로컬, 5명+ 클라우드                   │
└─────────────────────────────────────────────────────────┘
```

## 빠른 시작

### 서버 시작/종료

```bash
# 더블클릭 실행 (Finder에서)
서버시작.command    # 서버 시작
서버종료.command    # 서버 종료

# 터미널 실행
./start_local.sh    # 서버 시작
./stop_local.sh     # 서버 종료
```

### 접속 URL

| 용도 | URL |
|------|-----|
| API 엔드포인트 | http://localhost:8000/v1/chat/completions |
| 채팅 UI (일반) | http://localhost:8000/chat |
| 채팅 UI (스트리밍) | http://localhost:8000/chat-streaming |
| 대시보드 | http://localhost:8000/dashboard |
| API 문서 | http://localhost:8000/docs |
| 상태 확인 | http://localhost:8000/health |
| 테스트 페이지 | http://localhost:8000/test |

---

## 로컬 서버 설정

### 환경 변수 (.env)

```bash
# OpenAI API 키 (클라우드 백업용)
OPENAI_API_KEY=sk-xxx

# 관리자 키
ADMIN_API_KEY=sk-admin-xxx
```

### 로드밸런싱 설정 (config.yaml)

```yaml
load_balancing:
  enabled: true
  local_model: "vllm-qwen3-30b-a3b"   # 로컬 모델
  cloud_model: "gpt-5.2"              # 클라우드 백업
  max_queue_size: 4                    # 로컬 최대 동시 처리
```

### 동작 방식

| 동시 사용자 | 처리 방식 | 예상 응답 시간 |
|-------------|-----------|----------------|
| 1~4명 | 전체 로컬 (vLLM-MLX) | 0.6~1.5초 |
| 5~10명 | 4명 로컬 + 나머지 클라우드 | 1~3초 |
| 10명+ | 4명 로컬 + 나머지 클라우드 | 2~4초 |

---

## 외부 접속 설정

### 방법 1: Cloudflare Tunnel (권장)

무료, 보안, SSL 자동 설정

```bash
# 1. Cloudflare Tunnel 설치
brew install cloudflared

# 2. 로그인
cloudflared tunnel login

# 3. 터널 생성
cloudflared tunnel create llm-api

# 4. 설정 파일 생성 (~/.cloudflared/config.yml)
tunnel: <터널-ID>
credentials-file: ~/.cloudflared/<터널-ID>.json

ingress:
  - hostname: api.your-domain.com
    service: http://localhost:8000
  - service: http_status:404

# 5. DNS 설정
cloudflared tunnel route dns llm-api api.your-domain.com

# 6. 터널 실행
cloudflared tunnel run llm-api
```

### 방법 2: ngrok (테스트용)

빠른 설정, 임시 URL

```bash
# 1. 설치
brew install ngrok

# 2. 계정 연결 (https://ngrok.com에서 토큰 획득)
ngrok config add-authtoken <토큰>

# 3. 실행
ngrok http 8000

# URL 예시: https://abc123.ngrok.io
```

### 방법 3: 포트포워딩 (라우터)

고정 IP 필요, 직접 설정

```
1. 라우터 관리 페이지 접속 (192.168.0.1)
2. 포트포워딩 설정
   - 외부 포트: 8000
   - 내부 IP: Mac의 로컬 IP
   - 내부 포트: 8000
3. 방화벽 설정
   sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/bin/python3
```

---

## API 사용법

### 기본 요청

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "안녕하세요"}],
    "max_tokens": 100
  }'
```

### Python 예제

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="vllm-qwen3-30b-a3b",
    messages=[{"role": "user", "content": "안녕하세요"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### JavaScript 예제

```javascript
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'vllm-qwen3-30b-a3b',
    messages: [{ role: 'user', content: '안녕하세요' }],
    max_tokens: 100
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

### 스트리밍 요청

실시간으로 응답을 받아 타이핑 효과를 구현할 때 사용합니다.

#### curl 스트리밍

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "안녕하세요"}],
    "max_tokens": 100,
    "stream": true
  }'
```

#### Python 스트리밍

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="vllm-qwen3-30b-a3b",
    messages=[{"role": "user", "content": "안녕하세요"}],
    max_tokens=100,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### JavaScript 스트리밍

```javascript
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'vllm-qwen3-30b-a3b',
    messages: [{ role: 'user', content: '안녕하세요' }],
    max_tokens: 100,
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n').filter(line => line.startsWith('data: '));

  for (const line of lines) {
    const data = line.slice(6);
    if (data === '[DONE]') break;

    const parsed = JSON.parse(data);
    const content = parsed.choices[0]?.delta?.content;
    if (content) {
      process.stdout.write(content);  // 또는 DOM에 추가
    }
  }
}
```

---

## 모니터링

### 로드밸런서 상태

```bash
curl http://localhost:8000/lb/status
```

응답:
```json
{
  "enabled": true,
  "current_requests": 2,
  "max_queue_size": 4,
  "local_model": "vllm-qwen3-30b-a3b",
  "cloud_model": "gpt-5.2"
}
```

### 서버 상태

```bash
curl http://localhost:8000/health
```

### 로그 확인

```bash
# API 서버 로그
tail -f /tmp/llm-api.log

# vLLM-MLX 서버 로그
tail -f /tmp/vllm-mlx.log
```

---

## 문제 해결

### vLLM-MLX 서버 연결 실패

```bash
# 서버 상태 확인
curl http://localhost:8001/v1/models

# 서버 재시작
pkill -f "vllm-mlx serve"
./start_local.sh
```

### API 응답이 느림

1. 로드밸런서 상태 확인: `curl http://localhost:8000/lb/status`
2. 동시 요청 수 확인
3. vLLM-MLX 로그 확인: `tail -f /tmp/vllm-mlx.log`

### 클라우드 전환 안됨

1. `.env` 파일에 `OPENAI_API_KEY` 설정 확인
2. API 키 유효성 확인
3. 서버 재시작

---

## 파일 구조

```
llm-api-server/
├── 서버시작.command      # 더블클릭 시작
├── 서버종료.command      # 더블클릭 종료
├── start_local.sh        # 시작 스크립트
├── stop_local.sh         # 종료 스크립트
├── config.yaml           # 서버 설정
├── .env                  # 환경 변수 (API 키)
├── app/
│   ├── main.py           # FastAPI 앱
│   ├── config.py         # 설정 로드
│   ├── load_balancer.py  # 로드밸런서
│   ├── routers/
│   │   └── chat.py       # 채팅 API
│   └── clients/
│       └── vllm_mlx_client.py  # vLLM-MLX 클라이언트
└── venv/                 # Python 가상환경
```

---

## 기술 사양

| 항목 | 값 |
|------|-----|
| 로컬 모델 | Qwen3-30B-A3B (MoE, 4bit) |
| 추론 엔진 | vLLM-MLX (Continuous Batching) |
| API 호환성 | OpenAI API |
| 최대 동시 처리 (로컬) | 4명 |
| 평균 응답 시간 | 0.6~1.5초 |
| 처리량 | ~217 tokens/sec |
