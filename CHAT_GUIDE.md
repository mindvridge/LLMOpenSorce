# 💬 채팅 UI 사용 가이드

## 🎯 대화 기록 관리 방식

### ✅ 선택된 방식: **클라이언트 측 대화 기록 관리**

**이유:**
- OpenAI API 표준 방식과 동일
- 서버 무상태(Stateless) 유지로 확장성 극대화
- 추가 DB나 세션 관리 불필요
- 클라이언트가 localStorage에 대화 저장

**작동 방식:**
```javascript
// 1. 사용자 메시지를 messageHistory에 추가
messageHistory.push({ role: 'user', content: message });

// 2. API 요청 시 전체 대화 기록 전송
const messages = [
    { role: 'system', content: systemPrompt },
    ...messageHistory  // 이전 대화 모두 포함
];

// 3. 응답을 받아 messageHistory에 추가
messageHistory.push({ role: 'assistant', content: response });

// 4. localStorage에 저장
localStorage.setItem('conversations', JSON.stringify(conversations));
```

## 🚀 채팅 UI 기능

### 1️⃣ **사이드바**
- **새 대화 시작**: 새로운 면접 세션 시작
- **대화 기록**: 과거 대화 클릭으로 불러오기
- **모델 선택**: Qwen3 Ultra / Bllossom Tarot
- **Temperature 조절**: 0.0 ~ 2.0 (창의성 조절)
- **모든 대화 삭제**: localStorage 초기화

### 2️⃣ **메인 채팅 영역**
- **웰컴 스크린**: 4가지 예시 프롬프트 제공
  - 💼 프론트엔드 면접
  - 🖥️ 백엔드 면접
  - 📊 데이터 분석가 면접
  - 🎨 UI/UX 디자이너 면접
- **메시지 표시**: 시간, 역할(지원자/면접관) 표시
- **타이핑 인디케이터**: AI 응답 대기 중 애니메이션

### 3️⃣ **입력 영역**
- **시스템 프롬프트 설정**: ⚙️ 버튼으로 토글
- **메시지 입력**: Shift+Enter로 전송
- **자동 높이 조절**: 입력 내용에 따라 자동 확장

## 💡 사용 시나리오

### 시나리오 1: 프론트엔드 면접 시작
```
1. "💼 프론트엔드 면접" 예시 클릭
2. 자동으로 시스템 프롬프트와 첫 메시지 입력됨
3. "전송" 버튼 클릭 또는 Shift+Enter
4. 면접관 AI가 질문 생성
5. 답변 입력 후 대화 계속
```

### 시나리오 2: 연속 대화
```
1. 첫 메시지: "프론트엔드 개발자로 지원했습니다."
   → AI: "React 경험에 대해 설명해주세요..."

2. 두 번째 메시지: "React 3년 사용했습니다."
   → AI: "상태 관리는 어떻게 하셨나요..."

3. 세 번째 메시지: "Redux와 Context API를 사용했습니다."
   → AI: "Redux를 선택한 이유는..."

✅ 모든 이전 대화 컨텍스트가 유지됨!
```

### 시나리오 3: 대화 저장 및 불러오기
```
1. 여러 대화 진행
2. 사이드바에 자동 저장됨 (localStorage)
3. 브라우저 새로고침 후에도 대화 유지
4. 대화 클릭으로 이전 내용 불러오기
```

## 🔧 기술 구현

### 클라이언트 측 대화 관리
```javascript
// 대화 구조
const conversation = {
    id: '1673456789000',           // 타임스탬프 기반 ID
    title: '프론트엔드 개발자...',  // 첫 메시지 기반 제목
    messages: [                     // 전체 메시지 기록
        { role: 'user', content: '...' },
        { role: 'assistant', content: '...' }
    ],
    timestamp: 1673456789000       // 마지막 수정 시간
};

// localStorage 저장
localStorage.setItem('conversations', JSON.stringify([conversation]));
```

### API 요청 예시
```json
POST /v1/chat/completions
{
  "model": "qwen3-ultra:latest",
  "messages": [
    {
      "role": "system",
      "content": "당신은 전문 면접관입니다..."
    },
    {
      "role": "user",
      "content": "프론트엔드 개발자로 지원했습니다."
    },
    {
      "role": "assistant",
      "content": "React 경험에 대해 설명해주세요..."
    },
    {
      "role": "user",
      "content": "React 3년 사용했습니다."
    }
  ],
  "temperature": 0.7
}
```

## 🎨 UI 특징

### 다크 테마
- 배경: `#0f172a` (짙은 청록색)
- 카드: `#1e293b` (중간 청록색)
- 텍스트: `#e2e8f0` (밝은 회색)

### 그라데이션 버튼
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### 애니메이션
- 메시지 페이드인: `fadeIn 0.3s ease-in`
- 타이핑 도트: `typing 1.4s infinite`
- 상태 도트: `pulse 2s infinite`

### 반응형 디자인
- 데스크톱: 사이드바 + 채팅 영역
- 모바일 (768px 이하): 사이드바 숨김/토글

## 🔍 디버깅

### localStorage 확인
```javascript
// 브라우저 콘솔에서
console.log(JSON.parse(localStorage.getItem('conversations')));
```

### 대화 초기화
```javascript
// 브라우저 콘솔에서
localStorage.removeItem('conversations');
location.reload();
```

## 📊 데이터 흐름

```
┌─────────────┐
│   사용자    │
└──────┬──────┘
       │ 메시지 입력
       ▼
┌─────────────────────────┐
│  messageHistory 배열    │ ← 클라이언트 메모리
│  - user message         │
│  - assistant message    │
│  - user message         │
└──────┬──────────────────┘
       │ 전체 기록 전송
       ▼
┌─────────────────────────┐
│   LLM API Server        │
│  (Stateless)            │
└──────┬──────────────────┘
       │ 응답 생성
       ▼
┌─────────────────────────┐
│  messageHistory 추가    │
└──────┬──────────────────┘
       │ localStorage 저장
       ▼
┌─────────────────────────┐
│   브라우저 localStorage │ ← 영구 저장
└─────────────────────────┘
```

## 🆚 서버 측 세션 관리 vs 클라이언트 측 관리

| 항목 | 클라이언트 관리 (현재) | 서버 세션 관리 |
|------|---------------------|---------------|
| 확장성 | ⭐⭐⭐⭐⭐ 무한 | ⭐⭐ 메모리 제한 |
| 구현 복잡도 | ⭐⭐⭐ 중간 | ⭐⭐⭐⭐⭐ 높음 |
| 네트워크 | ⭐⭐ 대화 길수록 증가 | ⭐⭐⭐⭐⭐ 최소 |
| 보안 | ⭐⭐⭐ 클라이언트 저장 | ⭐⭐⭐⭐⭐ 서버 저장 |
| 비용 | ⭐⭐⭐⭐⭐ 무료 | ⭐⭐ DB 비용 |
| OpenAI 호환 | ⭐⭐⭐⭐⭐ 완벽 | ⭐⭐ 커스텀 |

## 🚀 향후 개선 가능 사항

### Phase 2 (선택적)
- 서버 측 대화 백업 (선택적 기능)
- 대화 검색 기능
- 대화 내보내기 (JSON/PDF)
- 다크/라이트 테마 토글

### Phase 3 (고급)
- 음성 입력/출력
- 실시간 협업 (여러 면접관)
- AI 피드백 점수
- 면접 리포트 생성

## 📝 사용 팁

1. **컨텍스트 유지**: 시스템 프롬프트를 변경하지 않으면 일관된 면접관 역할 유지
2. **Temperature 조절**:
   - 0.3: 일관된 질문
   - 0.7: 균형 (권장)
   - 1.5: 창의적 질문
3. **대화 저장**: 브라우저를 닫아도 localStorage에 저장됨
4. **긴 대화**: 30~40턴 이상 시 새 대화 시작 권장 (토큰 제한)

## ✅ 체크리스트

- [x] 클라이언트 측 대화 기록 관리
- [x] localStorage 저장
- [x] 대화 목록 표시
- [x] 대화 불러오기
- [x] 새 대화 시작
- [x] 시스템 프롬프트 설정
- [x] Temperature 조절
- [x] 타이핑 인디케이터
- [x] 반응형 디자인
- [x] 서버 상태 표시

---

**브라우저에서 테스트:** `file:///Users/mindprep/llm-api-server/chat_ui.html`
