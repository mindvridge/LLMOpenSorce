# LLM API Server - 외부 클라이언트 API 가이드

> OpenAI 호환 API - 면접 AI 서비스용

## 서버 주소

```
https://api.mindprep.co.kr
```

**인증 불필요** - API 키 없이 바로 사용 가능

---

## 필수 파라미터

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `model` | string | ✅ | `vllm-qwen3-30b-a3b` |
| `messages` | array | ✅ | 대화 메시지 배열 |

> **권장**: `company_name` 파라미터를 추가하면 더 맞춤화된 면접 질문을 생성합니다.

---

## 빠른 시작

### Python (requests) - 기본 호출

```python
import requests

response = requests.post(
    "https://api.mindprep.co.kr/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [{"role": "user", "content": "면접 질문 해주세요"}],
        "company_name": "삼성전자"  # 권장
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="no-key-needed",
    base_url="https://api.mindprep.co.kr/v1"
)

# OpenAI SDK는 extra_body로 추가 파라미터 전달
response = client.chat.completions.create(
    model="vllm-qwen3-30b-a3b",
    messages=[{"role": "user", "content": "면접 질문 해주세요"}],
    extra_body={"company_name": "삼성전자"}  # 권장
)

print(response.choices[0].message.content)
```

### curl

```bash
curl -X POST https://api.mindprep.co.kr/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-qwen3-30b-a3b",
    "messages": [{"role": "user", "content": "면접 질문 해주세요"}],
    "company_name": "삼성전자"
  }'
```

---

## 스트리밍 요청

### Python 스트리밍

```python
import requests

response = requests.post(
    "https://api.mindprep.co.kr/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [{"role": "user", "content": "면접 질문 해주세요"}],
        "company_name": "삼성전자",
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### curl 스트리밍

```bash
curl -X POST https://api.mindprep.co.kr/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "vllm-qwen3-30b-a3b", "messages": [{"role": "user", "content": "면접 질문"}], "company_name": "삼성전자", "stream": true}' \
  --no-buffer
```

---

## 채팅 API 상세

### 엔드포인트

```
POST /v1/chat/completions
```

### 전체 요청 파라미터

#### 기본 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `model` | string | ✅ | - | `vllm-qwen3-30b-a3b` |
| `messages` | array | ✅ | - | 대화 메시지 배열 |
| `company_name` | string | ❌ | - | 지원 기업/병원명 (권장) |
| `temperature` | float | ❌ | 0.7 | 창의성 (0.0~2.0) |
| `max_tokens` | int | ❌ | 4096 | 최대 토큰 수 |
| `stream` | bool | ❌ | false | 스트리밍 여부 |

#### company_name 예시 (자유 입력 가능)

**일반기업:**
`삼성전자`, `SK하이닉스`, `현대자동차`, `LG에너지솔루션`, `삼성바이오로직스`, `기아`, `LG전자`, `포스코홀딩스`, `네이버`, `현대모비스` 등

**병원:**
`서울아산병원`, `삼성서울병원`, `서울대병원`, `세브란스병원`, `분당서울대병원`, `강남세브란스병원`, `아주대병원`, `서울성모병원`, `인하대병원`, `경희대병원` 등

> 위 목록 외에도 원하는 기업/병원명을 자유롭게 입력할 수 있습니다.

#### 면접 컨텍스트 파라미터 (선택)

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `job_posting` | string | ❌ | 채용공고 텍스트 (요약본 권장) |
| `resume_text` | string | ❌ | 지원자 이력서 텍스트 (요약본 권장) |

#### 질문셋 RAG 파라미터 (선택)

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `question_set_rag_enabled` | bool | ❌ | false | 질문셋 RAG 활성화 |
| `question_set_org_type` | string | ❌ | - | 조직 유형 (아래 enum 참조) |
| `question_set_job_name` | string | ❌ | - | 직무명 (아래 enum 참조) |
| `question_set_top_k` | int | ❌ | 5 | 검색할 질문 수 (1~10) |

### 메시지 형식

```json
{"role": "system" | "user" | "assistant", "content": "메시지 내용"}
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

## Enum 값 정의 (유형/직무)

### question_set_org_type (조직 유형)

| 값 | 설명 |
|----|------|
| `"병원"` | 병원/의료기관 |
| `"일반기업"` | 일반 기업 |

### question_set_job_name (직무명)

#### 병원 직무 (org_type: "병원")

| 값 | 설명 |
|----|------|
| `"간호사"` | 간호사 |
| `"국제의료관광코디네이터"` | 국제의료관광코디네이터 |

#### 일반기업 직무 (org_type: "일반기업")

| 값 | 설명 |
|----|------|
| `"개발엔지니어링"` | 개발/엔지니어링 |
| `"마케팅영업"` | 마케팅/영업 |
| `"고객서비스CS"` | 고객서비스/CS |
| `"인사HR"` | 인사/HR |
| `"운영관리"` | 운영/관리 |
| `"기획전략"` | 기획/전략 |
| `"재무회계"` | 재무/회계 |
| `"품질관리QA"` | 품질관리/QA |
| `"글로벌 마케팅"` | 글로벌 마케팅 |
| `"법무컴플라이언스"` | 법무/컴플라이언스 |
| `"해외영업"` | 해외 영업 |

> **참고**: 질문셋에 없는 조합으로 요청 시, RAG 없이 LLM이 자체적으로 질문을 생성합니다.

---

## 면접 컨텍스트 활용 예시

### 1. 기본 호출

```python
response = requests.post(
    "https://api.mindprep.co.kr/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [{"role": "user", "content": "면접 질문 해주세요"}],
        "company_name": "삼성전자"
    }
)
```

### 2. 채용공고 + 이력서 포함

```python
response = requests.post(
    "https://api.mindprep.co.kr/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [{"role": "user", "content": "제 경력에 맞는 질문 해주세요"}],
        "company_name": "삼성전자",
        "job_posting": "모집분야: 반도체 공정 엔지니어\n자격요건: 관련 전공 학사 이상...",
        "resume_text": "학력: 서울대 전자공학과\n경력: 반도체 장비 3년..."
    }
)
```

### 3. 질문셋 RAG 활용 (병원 간호사)

```python
response = requests.post(
    "https://api.mindprep.co.kr/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [{"role": "user", "content": "환자 케어 관련 질문 해주세요"}],
        "company_name": "서울대병원",
        "question_set_rag_enabled": True,
        "question_set_org_type": "병원",
        "question_set_job_name": "간호사",
        "question_set_top_k": 5
    }
)
```

### 4. 질문셋 RAG 활용 (일반기업 마케팅)

```python
response = requests.post(
    "https://api.mindprep.co.kr/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [{"role": "user", "content": "마케팅 역량 관련 질문 해주세요"}],
        "company_name": "LG전자",
        "question_set_rag_enabled": True,
        "question_set_org_type": "일반기업",
        "question_set_job_name": "마케팅영업",
        "question_set_top_k": 5
    }
)
```

### 5. 전체 파라미터 조합 (최대 컨텍스트)

```python
response = requests.post(
    "https://api.mindprep.co.kr/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [
            {"role": "system", "content": "당신은 전문 면접관입니다."},
            {"role": "user", "content": "제 경력에 맞는 기술 면접 질문 해주세요"}
        ],
        "company_name": "네이버",
        "job_posting": "모집: 백엔드 개발자\n요건: Python, FastAPI 경험...",
        "resume_text": "경력: 스타트업 백엔드 개발 2년\n기술: Python, Django, FastAPI...",
        "question_set_rag_enabled": True,
        "question_set_org_type": "일반기업",
        "question_set_job_name": "개발엔지니어링",
        "question_set_top_k": 5,
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }
)
```

---

## RAG 기능 (고급)

### 이력서 RAG

업로드된 이력서에서 관련 내용을 검색합니다.

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

### 문서 RAG

업로드된 PDF 문서에서 관련 내용을 검색합니다.

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
resume_summary = upload_response.json()["summary"]
print(f"이력서 업로드 완료: {resume_session_id}")

# 2. 면접 질문 생성 (전체 파라미터 활용)
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "vllm-qwen3-30b-a3b",
        "messages": [
            {"role": "system", "content": "당신은 전문 면접관입니다."},
            {"role": "user", "content": "제 경력에 대해 질문해주세요."}
        ],
        "temperature": 0.7,
        # 필수 파라미터
        "company_name": "삼성전자",
        # 선택: 면접 컨텍스트
        "job_posting": "모집: SW 개발자\n요건: Python 3년 이상...",
        "resume_text": resume_summary,  # 이력서 요약본
        # 선택: 질문셋 RAG
        "question_set_rag_enabled": True,
        "question_set_org_type": "일반기업",
        "question_set_job_name": "개발엔지니어링",
        "question_set_top_k": 5
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

### 필수 파라미터 누락 시 (422 에러)

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "model"],
      "msg": "Field required"
    }
  ]
}
```

### Python 에러 처리 예시

```python
import requests

try:
    response = requests.post(
        "https://api.mindprep.co.kr/v1/chat/completions",
        json={
            "model": "vllm-qwen3-30b-a3b",
            "messages": [{"role": "user", "content": "면접 질문 해주세요"}],
            "company_name": "삼성전자"
        },
        timeout=120
    )
    response.raise_for_status()
    print(response.json()["choices"][0]["message"]["content"])
except requests.exceptions.HTTPError as e:
    if response.status_code == 422:
        print("필수 파라미터 누락:", response.json())
    else:
        print(f"HTTP 오류: {e}")
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
| **필수 파라미터** | `model`, `messages` |
| **스트리밍** | `"stream": true` |

---

## 파라미터 요약표

| 파라미터 | 필수 | 타입 | 예시 |
|---------|------|------|------|
| `model` | ✅ | string | `"vllm-qwen3-30b-a3b"` |
| `messages` | ✅ | array | `[{"role": "user", "content": "..."}]` |
| `company_name` | ❌ | string | `"삼성전자"`, `"서울대병원"` (권장) |
| `job_posting` | ❌ | string | 채용공고 요약 텍스트 |
| `resume_text` | ❌ | string | 이력서 요약 텍스트 |
| `question_set_rag_enabled` | ❌ | bool | `true` |
| `question_set_org_type` | ❌ | string | `"병원"`, `"일반기업"` |
| `question_set_job_name` | ❌ | string | `"간호사"`, `"개발엔지니어링"` |
| `stream` | ❌ | bool | `true` / `false` |
| `temperature` | ❌ | float | `0.7` |
| `max_tokens` | ❌ | int | `500` |

---

**마지막 업데이트**: 2026-01-22
