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
    owned_by: str = "ollama"


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
    ollama_connected: bool
    timestamp: int
