"""채팅 완성 라우터"""
import time
import uuid
import json
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessageResponse,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatCompletionDeltaMessage,
    UsageInfo,
    ErrorResponse,
    ErrorDetail,
)
from app.ollama_client import get_ollama_client
from app.config import get_config

router = APIRouter()


def estimate_tokens(text: str) -> int:
    """토큰 수 추정 (간단한 방법: 단어 수 기반)"""
    # 한글: 글자당 약 1토큰, 영어: 단어당 약 1.3토큰
    korean_chars = sum(1 for c in text if '가' <= c <= '힣')
    other_chars = len(text) - korean_chars
    return korean_chars + int(other_chars / 4)


async def stream_chat_completion(
    request: ChatCompletionRequest,
    completion_id: str,
) -> AsyncGenerator[str, None]:
    """스트리밍 응답 생성"""
    ollama_client = get_ollama_client()

    # 메시지 변환
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    try:
        # 첫 번째 청크 전송 (role 포함)
        first_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionDeltaMessage(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # 스트리밍 응답 처리
        async for chunk in ollama_client.chat_completion_stream(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            if chunk.get("done"):
                # 마지막 청크
                final_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=ChatCompletionDeltaMessage(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                break

            # 컨텐츠가 있는 청크
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                stream_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=ChatCompletionDeltaMessage(content=content),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {stream_chunk.model_dump_json()}\n\n"

        # 스트림 종료
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "ollama_error",
                "code": "internal_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def chat_completions(request: ChatCompletionRequest):
    """
    채팅 완성 생성 (OpenAI 호환)

    - **model**: 사용할 모델 이름
    - **messages**: 대화 메시지 리스트
    - **temperature**: 샘플링 온도 (0.0 ~ 2.0)
    - **max_tokens**: 최대 생성 토큰 수
    - **stream**: 스트리밍 여부
    """
    ollama_client = get_ollama_client()
    config = get_config()

    # 모델 유효성 검사
    available_model_names = [m.name for m in config.available_models]
    if request.model not in available_model_names:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"모델 '{request.model}'을(를) 찾을 수 없습니다. 사용 가능한 모델: {', '.join(available_model_names)}",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    # 메시지가 비어있는지 확인
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "최소 1개 이상의 메시지가 필요합니다.",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "invalid_messages",
                }
            },
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # 스트리밍 응답
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, completion_id),
            media_type="text/event-stream",
        )

    # 일반 응답
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        response = await ollama_client.chat_completion(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # 프롬프트 토큰 수 계산
        prompt_text = " ".join([msg["content"] for msg in messages])
        prompt_tokens = estimate_tokens(prompt_text)

        # 응답 컨텐츠 추출
        assistant_message = response.get("message", {}).get("content", "")
        completion_tokens = estimate_tokens(assistant_message)

        # OpenAI 형식으로 변환
        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessageResponse(
                        role="assistant",
                        content=assistant_message,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"채팅 완성 생성 실패: {str(e)}",
                    "type": "internal_error",
                    "code": "internal_error",
                }
            },
        )
