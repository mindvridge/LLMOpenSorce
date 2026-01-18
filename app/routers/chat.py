"""채팅 완성 라우터"""
import time
import uuid
import json
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
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
from app.clients.openai_client import get_openai_client
from app.clients.mlx_client import get_mlx_client
from app.clients.vllm_mlx_client import get_vllm_mlx_client
from app.config import get_config
from app.database import get_db
from app.auth import verify_api_key
from app.models.database import APIKey, RequestLog
from app.load_balancer import get_load_balancer
from app.rag.retriever import get_retriever
from app.question_sets import search_relevant_questions, format_questions_for_prompt
from app.routers.resume import search_resume_context


def is_mlx_model(model: str) -> bool:
    """MLX 모델인지 확인"""
    return model.startswith("mlx-")


def is_vllm_mlx_model(model: str) -> bool:
    """vLLM-MLX 모델인지 확인"""
    return model.startswith("vllm-")

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
    # 모델 제공자 확인
    openai_client = get_openai_client()
    is_openai = openai_client.is_openai_model(request.model)

    # 메시지 변환
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # RAG 컨텍스트 주입
    if request.rag_enabled:
        try:
            retriever = get_retriever()
            # 마지막 사용자 메시지로 검색
            user_query = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_query = msg.content
                    break

            if user_query:
                # 기존 시스템 프롬프트 찾기
                base_prompt = ""
                for msg in messages:
                    if msg["role"] == "system":
                        base_prompt = msg["content"]
                        break

                # RAG 시스템 프롬프트 생성
                rag_prompt = retriever.build_rag_system_prompt(
                    query=user_query,
                    collection_name=request.rag_collection,
                    top_k=request.rag_top_k,
                    base_prompt=base_prompt
                )

                # 시스템 메시지 업데이트 또는 추가
                system_found = False
                for i, msg in enumerate(messages):
                    if msg["role"] == "system":
                        messages[i]["content"] = rag_prompt
                        system_found = True
                        break
                if not system_found:
                    messages.insert(0, {"role": "system", "content": rag_prompt})
        except Exception as e:
            print(f"RAG 컨텍스트 주입 실패 (스트리밍): {e}")

    # 질문셋 RAG 컨텍스트 주입
    if request.question_set_rag_enabled and request.question_set_org_type and request.question_set_job_name:
        try:
            # 마지막 사용자 메시지로 검색
            user_query = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_query = msg.content
                    break

            if user_query:
                # 관련 질문 검색
                relevant_questions = search_relevant_questions(
                    org_type=request.question_set_org_type,
                    job_name=request.question_set_job_name,
                    query=user_query,
                    top_k=request.question_set_top_k or 5
                )

                if relevant_questions:
                    # 질문을 프롬프트용 텍스트로 변환
                    questions_text = format_questions_for_prompt(relevant_questions)

                    # 시스템 메시지에 추가
                    system_found = False
                    for i, msg in enumerate(messages):
                        if msg["role"] == "system":
                            messages[i]["content"] = msg["content"] + "\n\n" + questions_text
                            system_found = True
                            break
                    if not system_found:
                        messages.insert(0, {"role": "system", "content": questions_text})
        except Exception as e:
            print(f"질문셋 RAG 컨텍스트 주입 실패 (스트리밍): {e}")

    # 이력서 RAG 컨텍스트 주입
    if request.resume_rag_enabled and request.resume_session_id:
        try:
            # 마지막 사용자 메시지로 검색
            user_query = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_query = msg.content
                    break

            if user_query:
                # 관련 이력서 내용 검색
                resume_context = search_resume_context(
                    session_id=request.resume_session_id,
                    query=user_query,
                    top_k=request.resume_top_k or 3
                )

                if resume_context:
                    # 시스템 메시지에 추가
                    system_found = False
                    for i, msg in enumerate(messages):
                        if msg["role"] == "system":
                            messages[i]["content"] = msg["content"] + "\n\n" + resume_context
                            system_found = True
                            break
                    if not system_found:
                        messages.insert(0, {"role": "system", "content": resume_context})
        except Exception as e:
            print(f"이력서 RAG 컨텍스트 주입 실패 (스트리밍): {e}")

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

        # vLLM-MLX 스트리밍 (Continuous Batching)
        if is_vllm_mlx_model(request.model):
            vllm_client = get_vllm_mlx_client()
            async for chunk in await vllm_client.chat_completion(
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                stream=True,
            ):
                if chunk.get("choices") and chunk["choices"][0].get("delta"):
                    delta = chunk["choices"][0]["delta"]
                    if delta.get("content"):
                        # vLLM-MLX 스트리밍 버그로 인한 replacement character 제거
                        content = delta["content"].replace('\ufffd', '')
                        if content:  # 빈 문자열이 아닌 경우에만 전송
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

        # MLX 스트리밍
        elif is_mlx_model(request.model):
            mlx_client = get_mlx_client()
            async for chunk in await mlx_client.chat_completion(
                messages=messages,
                model=request.model,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                stream=True,
            ):
                if chunk.get("choices") and chunk["choices"][0].get("delta"):
                    delta = chunk["choices"][0]["delta"]
                    if delta.get("content"):
                        stream_chunk = ChatCompletionStreamResponse(
                            id=completion_id,
                            created=int(time.time()),
                            model=request.model,
                            choices=[
                                ChatCompletionStreamChoice(
                                    index=0,
                                    delta=ChatCompletionDeltaMessage(content=delta["content"]),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {stream_chunk.model_dump_json()}\n\n"

            # MLX 마지막 청크
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

        # OpenAI 스트리밍
        elif is_openai:
            if not openai_client.is_enabled():
                raise Exception("OpenAI API가 활성화되지 않았습니다. OPENAI_API_KEY를 설정하세요.")

            async for content in openai_client.stream_chat_completion(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
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

        # 지원하지 않는 모델
        else:
            raise Exception(f"지원하지 않는 모델: {request.model}")

        # 스트림 종료
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "api_error",
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
async def chat_completions(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
    api_key: Optional[APIKey] = Depends(verify_api_key)
):
    """
    채팅 완성 생성 (OpenAI 호환)

    - **model**: 사용할 모델 이름
    - **messages**: 대화 메시지 리스트
    - **temperature**: 샘플링 온도 (0.0 ~ 2.0)
    - **max_tokens**: 최대 생성 토큰 수
    - **stream**: 스트리밍 여부
    """
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

    # 로드밸런싱 적용 (1~4명 로컬, 5명+ 클라우드)
    load_balancer = get_load_balancer()
    original_model = request.model

    # 요청 카운트 먼저 증가 (모델 선택 전)
    await load_balancer.increment_requests()

    # 모델 선택 (현재 요청 수 기반)
    lb_result = load_balancer.select_model(request.model)
    request.model = lb_result["model"]

    # 스트리밍 응답
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, completion_id),
            media_type="text/event-stream",
        )

    # 일반 응답
    start_time = time.time()

    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # RAG 컨텍스트 주입
        if request.rag_enabled:
            try:
                retriever = get_retriever()
                # 마지막 사용자 메시지로 검색
                user_query = ""
                for msg in reversed(request.messages):
                    if msg.role == "user":
                        user_query = msg.content
                        break

                if user_query:
                    # 기존 시스템 프롬프트 찾기
                    base_prompt = ""
                    for msg in messages:
                        if msg["role"] == "system":
                            base_prompt = msg["content"]
                            break

                    # RAG 시스템 프롬프트 생성
                    rag_prompt = retriever.build_rag_system_prompt(
                        query=user_query,
                        collection_name=request.rag_collection,
                        top_k=request.rag_top_k,
                        base_prompt=base_prompt
                    )

                    # 시스템 메시지 업데이트 또는 추가
                    system_found = False
                    for i, msg in enumerate(messages):
                        if msg["role"] == "system":
                            messages[i]["content"] = rag_prompt
                            system_found = True
                            break
                    if not system_found:
                        messages.insert(0, {"role": "system", "content": rag_prompt})
            except Exception as e:
                print(f"RAG 컨텍스트 주입 실패: {e}")

        # 질문셋 RAG 컨텍스트 주입
        if request.question_set_rag_enabled and request.question_set_org_type and request.question_set_job_name:
            try:
                # 마지막 사용자 메시지로 검색
                user_query = ""
                for msg in reversed(request.messages):
                    if msg.role == "user":
                        user_query = msg.content
                        break

                if user_query:
                    # 관련 질문 검색
                    relevant_questions = search_relevant_questions(
                        org_type=request.question_set_org_type,
                        job_name=request.question_set_job_name,
                        query=user_query,
                        top_k=request.question_set_top_k or 5
                    )

                    if relevant_questions:
                        # 질문을 프롬프트용 텍스트로 변환
                        questions_text = format_questions_for_prompt(relevant_questions)

                        # 시스템 메시지에 추가
                        system_found = False
                        for i, msg in enumerate(messages):
                            if msg["role"] == "system":
                                messages[i]["content"] = msg["content"] + "\n\n" + questions_text
                                system_found = True
                                break
                        if not system_found:
                            messages.insert(0, {"role": "system", "content": questions_text})
            except Exception as e:
                print(f"질문셋 RAG 컨텍스트 주입 실패: {e}")

        # 이력서 RAG 컨텍스트 주입
        if request.resume_rag_enabled and request.resume_session_id:
            try:
                # 마지막 사용자 메시지로 검색
                user_query = ""
                for msg in reversed(request.messages):
                    if msg.role == "user":
                        user_query = msg.content
                        break

                if user_query:
                    # 관련 이력서 내용 검색
                    resume_context = search_resume_context(
                        session_id=request.resume_session_id,
                        query=user_query,
                        top_k=request.resume_top_k or 3
                    )

                    if resume_context:
                        # 시스템 메시지에 추가
                        system_found = False
                        for i, msg in enumerate(messages):
                            if msg["role"] == "system":
                                messages[i]["content"] = msg["content"] + "\n\n" + resume_context
                                system_found = True
                                break
                        if not system_found:
                            messages.insert(0, {"role": "system", "content": resume_context})
            except Exception as e:
                print(f"이력서 RAG 컨텍스트 주입 실패: {e}")

        # vLLM-MLX 모델 사용 (Continuous Batching - 추천!)
        if is_vllm_mlx_model(request.model):
            vllm_client = get_vllm_mlx_client()
            response = await vllm_client.chat_completion(
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                stream=False,
            )

        # MLX 모델 사용 (단일 사용자용)
        elif is_mlx_model(request.model):
            mlx_client = get_mlx_client()
            response = await mlx_client.chat_completion(
                messages=messages,
                model=request.model,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                stream=False,
            )

        # OpenAI 모델 사용
        elif get_openai_client().is_openai_model(request.model):
            openai_client = get_openai_client()
            if not openai_client.is_enabled():
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": {
                            "message": "OpenAI API가 활성화되지 않았습니다. OPENAI_API_KEY를 설정하세요.",
                            "type": "configuration_error",
                            "code": "openai_not_configured",
                        }
                    },
                )

            response = await openai_client.chat_completion(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        # 지원하지 않는 모델
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"지원하지 않는 모델: {request.model}",
                        "type": "invalid_request_error",
                        "code": "model_not_supported",
                    }
                },
            )

        # 프롬프트 토큰 수 계산
        prompt_text = " ".join([msg["content"] for msg in messages])

        # OpenAI/MLX/vLLM 응답 처리 (usage 정보 포함)
        if "usage" in response:
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
            total_tokens = response["usage"]["total_tokens"]
            # MLX/OpenAI/vLLM 응답 처리
            if "choices" in response:
                assistant_message = response["choices"][0]["message"]["content"]
            else:
                assistant_message = response["content"]
        else:
            # 토큰 추정 (fallback)
            prompt_tokens = estimate_tokens(prompt_text)
            assistant_message = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            completion_tokens = estimate_tokens(assistant_message)
            total_tokens = prompt_tokens + completion_tokens

        # 응답 시간 계산
        response_time = time.time() - start_time

        # 요청 카운트 감소 및 응답 시간 기록
        await load_balancer.decrement_requests(response_time)

        # 사용량 로깅 (API 키가 있는 경우)
        if api_key:
            # API 키 사용량 업데이트
            api_key.total_requests += 1
            api_key.total_tokens += total_tokens

            # 요청 로그 기록
            log = RequestLog(
                api_key_id=api_key.id,
                model=request.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                response_time=response_time,
                status="success"
            )
            db.add(log)
            db.commit()

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
                total_tokens=total_tokens,
            ),
        )

    except Exception as e:
        # 요청 카운트 감소 (에러 시)
        await load_balancer.decrement_requests(time.time() - start_time)

        # 에러 로깅 (API 키가 있는 경우)
        if api_key:
            log = RequestLog(
                api_key_id=api_key.id,
                model=request.model,
                status="error",
                error_message=str(e),
                response_time=time.time() - start_time
            )
            db.add(log)
            db.commit()

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
