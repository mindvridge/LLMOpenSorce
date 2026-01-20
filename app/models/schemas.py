"""OpenAI 호환 API 스키마"""
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field


# ===== Request Models =====

class ChatMessage(BaseModel):
    """채팅 메시지"""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """채팅 완성 요청"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = False
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    # RAG 파라미터 (문서 검색)
    rag_enabled: Optional[bool] = False
    rag_collection: Optional[str] = "default"
    rag_top_k: Optional[int] = Field(default=5, ge=1, le=20)
    # 질문셋 RAG 파라미터
    question_set_rag_enabled: Optional[bool] = False
    question_set_org_type: Optional[str] = None  # "병원" 또는 "일반기업"
    question_set_job_name: Optional[str] = None  # "간호사", "마케팅영업" 등
    question_set_top_k: Optional[int] = Field(default=5, ge=1, le=10)
    # 이력서 RAG 파라미터
    resume_rag_enabled: Optional[bool] = False
    resume_session_id: Optional[str] = None  # 이력서 세션 ID
    resume_top_k: Optional[int] = Field(default=3, ge=1, le=10)
    # 면접 컨텍스트 파라미터 (직접 전달)
    company_name: str  # 기업/병원명 (필수, 예: "삼성전자", "서울대병원")
    job_posting: Optional[str] = None   # 채용공고 텍스트 (선택)
    resume_text: Optional[str] = None   # 요약된 이력서 텍스트 (선택)


# ===== Response Models =====

class ChatCompletionMessageResponse(BaseModel):
    """응답 메시지"""
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """선택지"""
    index: int
    message: ChatCompletionMessageResponse
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    """토큰 사용량"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """채팅 완성 응답"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


# ===== Streaming Response Models =====

class ChatCompletionDeltaMessage(BaseModel):
    """스트리밍 델타 메시지"""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    """스트리밍 선택지"""
    index: int
    delta: ChatCompletionDeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """스트리밍 응답"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


# ===== Models List =====

class Model(BaseModel):
    """모델 정보"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vllm-mlx"


class ModelsResponse(BaseModel):
    """모델 목록 응답"""
    object: str = "list"
    data: List[Model]


# ===== Error Response =====

class ErrorDetail(BaseModel):
    """오류 세부 정보"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """오류 응답"""
    error: ErrorDetail


# ===== Health Check =====

class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    vllm_connected: bool
    timestamp: int
