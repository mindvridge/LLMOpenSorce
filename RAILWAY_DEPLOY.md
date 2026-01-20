# Railway 배포 가이드

> 로컬 서버 백업용 클라우드 배포 (GPU 없음, OpenAI API 전용)

## 개요

| 구분 | 로컬 서버 | Railway 서버 |
|------|-----------|--------------|
| **용도** | 메인 서버 | 백업 서버 |
| **모델** | vLLM-MLX (Qwen3) + OpenAI | OpenAI만 |
| **GPU** | Apple Silicon | 없음 |
| **비용** | 전기세만 | 사용량 기반 |

---

## Railway 배포 방법

### 1. Railway 계정 및 프로젝트 생성

1. [railway.app](https://railway.app) 접속
2. GitHub 계정으로 로그인
3. "New Project" → "Deploy from GitHub repo" 선택
4. 이 저장소 선택

### 2. 환경변수 설정

Railway 대시보드 → Variables 탭에서 설정:

```
OPENAI_API_KEY=sk-your-openai-api-key
DEPLOY_ENV=railway
```

| 변수 | 필수 | 설명 |
|------|------|------|
| `OPENAI_API_KEY` | ✅ | OpenAI API 키 |
| `DEPLOY_ENV` | ✅ | `railway` (자동 감지용) |
| `ADMIN_API_KEY` | ❌ | 관리자 키 (선택) |

### 3. 배포 확인

배포 완료 후:
- 헬스체크: `https://your-app.railway.app/health`
- API 문서: `https://your-app.railway.app/docs`

---

## 자동 환경 감지

서버가 자동으로 환경을 감지하여 적절한 설정을 로드합니다:

```
환경변수 DEPLOY_ENV=railway 또는 RAILWAY_ENVIRONMENT 존재
    ↓
config.railway.yaml 로드 (클라우드 전용)
    ↓
vLLM-MLX 웜업 건너뛰기
```

---

## 사용 가능 모델 (Railway)

Railway에서는 OpenAI 모델만 사용 가능:

| 모델 | 설명 |
|------|------|
| `gpt-5` | GPT-5 최신 모델 |
| `gpt-5-mini` | GPT-5 Mini (기본값, 빠름) |
| `gpt-4o` | GPT-4o |
| `gpt-4o-mini` | GPT-4o Mini (경제적) |

---

## 클라이언트 사용법

### Python 예시

```python
import requests

# Railway 서버 주소 (배포 후 확인)
RAILWAY_URL = "https://your-app.railway.app"

response = requests.post(
    f"{RAILWAY_URL}/v1/chat/completions",
    json={
        "model": "gpt-5-mini",  # OpenAI 모델 사용
        "messages": [{"role": "user", "content": "면접 질문 해주세요"}],
        "company_name": "삼성전자"
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### curl 예시

```bash
curl -X POST https://your-app.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5-mini",
    "messages": [{"role": "user", "content": "면접 질문"}],
    "company_name": "삼성전자"
  }'
```

---

## 로컬 ↔ Railway 전환

### 클라이언트에서 서버 전환

```python
# 서버 주소 설정
LOCAL_URL = "http://localhost:8000"
RAILWAY_URL = "https://your-app.railway.app"

# 로컬 서버 상태 확인 후 자동 전환
def get_api_url():
    try:
        response = requests.get(f"{LOCAL_URL}/health", timeout=3)
        if response.status_code == 200:
            return LOCAL_URL
    except:
        pass
    return RAILWAY_URL

API_URL = get_api_url()
```

---

## 비용 최적화

Railway 요금제:
- **Hobby**: $5/월 (500시간 무료)
- **Pro**: 사용량 기반

### 비용 절감 팁

1. **기본 모델**: `gpt-5-mini` 사용 (gpt-5보다 저렴)
2. **max_tokens 제한**: 필요한 만큼만 설정
3. **캐싱 활용**: 동일 질문 캐싱

---

## 모니터링

### 헬스체크 엔드포인트

```bash
# 서버 상태
curl https://your-app.railway.app/health

# 로드밸런서 상태
curl https://your-app.railway.app/lb/status
```

### Railway 대시보드

- 메트릭: CPU, 메모리, 네트워크
- 로그: 실시간 로그 확인
- 배포 이력: 롤백 가능

---

## 파일 구조

```
llm-api-server/
├── Dockerfile              # Railway 빌드용
├── railway.json            # Railway 설정
├── config.yaml             # 로컬 설정
├── config.railway.yaml     # Railway 설정 (자동 선택)
├── .env.railway.example    # Railway 환경변수 예시
└── app/
    └── config.py           # 환경 자동 감지
```

---

## 문제 해결

### 배포 실패 시

1. Railway 대시보드에서 빌드 로그 확인
2. 환경변수 `OPENAI_API_KEY` 설정 확인
3. Dockerfile 문법 확인

### API 오류 시

1. `/health` 엔드포인트로 서버 상태 확인
2. `/docs`에서 API 스펙 확인
3. OpenAI API 키 유효성 확인

---

**마지막 업데이트**: 2025-01-20
