# LLM API Server - 외부 클라이언트 API 가이드

> OpenAI 호환 API - 인증 없이 바로 사용 가능

## 서버 주소

```
https://api.mindprep.co.kr
```

**인증 불필요** - API 키 없이 바로 사용 가능

---

## 빠른 시작

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="no-key-needed",  # 아무 값이나 OK
    base_url="https://api.mindprep.co.kr/v1"
)

response = client.chat.completions.create(
    model="vllm-qwen3-30b-a3b",
    messages=[{"role": "user", "content": "안녕하세요"}]
)

print(response.choices[0].message.content)
```

### Python (requests)

```python
import requests

response = requests.post(
    "https://api.mindprep.co.kr/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [{"role": "user", "content": "안녕하세요"}]
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### curl

```bash
curl -X POST https://api.mindprep.co.kr/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "안녕하세요"}]
  }'
```

---

## 스트리밍 요청

### Python 스트리밍

```python
from openai import OpenAI

client = OpenAI(
    api_key="no-key-needed",
    base_url="https://api.mindprep.co.kr/v1"
)

stream = client.chat.completions.create(
    model="vllm-qwen3-30b-a3b",
    messages=[{"role": "user", "content": "긴 이야기를 들려주세요"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### curl 스트리밍

```bash
curl -X POST https://api.mindprep.co.kr/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "vllm-qwen3-30b-a3b", "messages": [{"role": "user", "content": "안녕"}], "stream": true}' \
  --no-buffer
```

---

## 채팅 API 상세

### 엔드포인트

```
POST /v1/chat/completions
```

### 요청 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `model` | string | ✅ | - | `vllm-qwen3-30b-a3b` |
| `messages` | array | ✅ | - | 대화 메시지 배열 |
| `temperature` | float | ❌ | 0.7 | 창의성 (0.0~2.0) |
| `max_tokens` | int | ❌ | 4096 | 최대 토큰 수 |
| `stream` | bool | ❌ | false | 스트리밍 여부 |

### 메시지 형식

```json
{"role": "system" | "user" | "assistant", "content": "메시지 내용"}
```

### 요청 예시

```json
{
  "model": "vllm-qwen3-30b-a3b",
  "messages": [
    {"role": "system", "content": "당신은 전문 면접관입니다."},
    {"role": "user", "content": "자기소개 해주세요."}
  ],
  "temperature": 0.7,
  "max_tokens": 2000,
  "stream": false
}
```

### 응답 예시

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1736870400,
  "model": "vllm-qwen3-30b-a3b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "안녕하세요..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 25, "completion_tokens": 150, "total_tokens": 175}
}
```

---

## RAG 기능 (문서 검색 기반 답변)

### 1. 질문셋 RAG

면접 질문셋에서 관련 질문을 검색하여 컨텍스트로 제공.

**추가 파라미터:**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `question_set_rag_enabled` | bool | `true`로 설정 |
| `question_set_org_type` | string | `"병원"` 또는 `"일반기업"` |
| `question_set_job_name` | string | 직무명 (예: `"간호사"`, `"마케팅영업"`) |
| `question_set_top_k` | int | 검색할 질문 수 (기본: 5) |

**요청 예시:**

```python
response = requests.post(
    "https://api.mindprep.co.kr/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [
            {"role": "system", "content": "당신은 전문 면접관입니다."},
            {"role": "user", "content": "팀워크 경험에 대해 질문해주세요."}
        ],
        "question_set_rag_enabled": True,
        "question_set_org_type": "일반기업",
        "question_set_job_name": "마케팅영업",
        "question_set_top_k": 5
    }
)
```

### 2. 이력서 RAG

업로드된 이력서에서 관련 내용을 검색.

**추가 파라미터:**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `resume_rag_enabled` | bool | `true`로 설정 |
| `resume_session_id` | string | 이력서 업로드 시 받은 세션 ID |
| `resume_top_k` | int | 검색할 청크 수 (기본: 3) |

**요청 예시:**

```python
response = requests.post(
    "https://api.mindprep.co.kr/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [
            {"role": "system", "content": "당신은 전문 면접관입니다."},
            {"role": "user", "content": "제 프로젝트 경험에 대해 질문해주세요."}
        ],
        "resume_rag_enabled": True,
        "resume_session_id": "abc12345",
        "resume_top_k": 3
    }
)
```

### 3. 문서 RAG

업로드된 PDF 문서에서 관련 내용을 검색.

**추가 파라미터:**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `rag_enabled` | bool | `true`로 설정 |
| `rag_collection` | string | 컬렉션명 (기본: `"default"`) |
| `rag_top_k` | int | 검색할 문서 수 (기본: 5) |

---

## 이력서 API

### 이력서 업로드

```
POST /resume/upload
Content-Type: multipart/form-data
```

**Python 예시:**

```python
with open("이력서.pdf", "rb") as f:
    response = requests.post(
        "https://api.mindprep.co.kr/resume/upload",
        files={"file": f}
    )

session_id = response.json()["session_id"]
print(f"세션 ID: {session_id}")  # 이 ID를 채팅에서 사용
```

**응답:**

```json
{
  "session_id": "abc12345",
  "filename": "이력서.pdf",
  "text_length": 5234,
  "summary": "## 인적사항\n- 홍길동...",
  "message": "이력서가 성공적으로 업로드되었습니다."
}
```

### 이력서 조회

```
GET /resume/{session_id}
```

### 이력서 삭제

```
DELETE /resume/{session_id}
```

---

## 전체 워크플로우 예시

```python
import requests

BASE_URL = "https://api.mindprep.co.kr"

# 1. 이력서 업로드
with open("이력서.pdf", "rb") as f:
    upload_response = requests.post(f"{BASE_URL}/resume/upload", files={"file": f})
resume_session_id = upload_response.json()["session_id"]
print(f"이력서 업로드 완료: {resume_session_id}")

# 2. 면접 질문 생성 (질문셋 RAG + 이력서 RAG)
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [
            {"role": "system", "content": "당신은 전문 면접관입니다. 지원자의 이력서를 참고하여 면접 질문을 해주세요."},
            {"role": "user", "content": "제 경력에 대해 질문해주세요."}
        ],
        "temperature": 0.7,
        # 질문셋 RAG
        "question_set_rag_enabled": True,
        "question_set_org_type": "일반기업",
        "question_set_job_name": "마케팅영업",
        "question_set_top_k": 5,
        # 이력서 RAG
        "resume_rag_enabled": True,
        "resume_session_id": resume_session_id,
        "resume_top_k": 3
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

---

## 기타 API

### 서버 상태 확인

```bash
curl https://api.mindprep.co.kr/health
```

### 모델 목록

```bash
curl https://api.mindprep.co.kr/v1/models
```

### 질문셋 목록

```bash
curl https://api.mindprep.co.kr/prompts/question-sets
```

### 특정 질문셋 조회

```bash
curl https://api.mindprep.co.kr/prompts/question-sets/병원/간호사
curl https://api.mindprep.co.kr/prompts/question-sets/일반기업/마케팅영업
```

---

## 에러 처리

```python
import requests

try:
    response = requests.post(
        "https://api.mindprep.co.kr/v1/chat/completions",
        json={
            "model": "vllm-qwen3-30b-a3b",
            "messages": [{"role": "user", "content": "안녕하세요"}]
        },
        timeout=120
    )
    response.raise_for_status()
    print(response.json()["choices"][0]["message"]["content"])
except requests.exceptions.Timeout:
    print("요청 시간 초과")
except requests.exceptions.RequestException as e:
    print(f"요청 오류: {e}")
```

---

## 요약

| 항목 | 값 |
|------|-----|
| **서버 주소** | `https://api.mindprep.co.kr` |
| **인증** | 불필요 |
| **기본 모델** | `vllm-qwen3-30b-a3b` |
| **채팅 API** | `POST /v1/chat/completions` |
| **이력서 업로드** | `POST /resume/upload` |
| **스트리밍** | `"stream": true` |

---

**마지막 업데이트**: 2025-01-14
