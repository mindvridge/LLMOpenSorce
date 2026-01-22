"""채팅 완성 라우터"""
import asyncio
import time
import uuid
import json
import hashlib
from typing import AsyncGenerator, Optional, List, Dict, Any, Tuple
from collections import OrderedDict
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
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
from app.database import get_db, SessionLocal
from app.auth import verify_api_key
from app.models.database import APIKey, RequestLog
from app.load_balancer import get_load_balancer
from app.rag.retriever import get_retriever
from app.question_sets import search_relevant_questions, format_questions_for_prompt
from app.routers.resume import search_resume_context
from app.routers.monitor import increment_request_stats


def is_mlx_model(model: str) -> bool:
    """MLX 모델인지 확인"""
    return model.startswith("mlx-")


def is_vllm_mlx_model(model: str) -> bool:
    """vLLM-MLX 모델인지 확인"""
    return model.startswith("vllm-")


# ===== RAG 컨텍스트 캐싱 (강화) =====
RAG_CONTEXT_CACHE_SIZE = 500   # 100 → 500 (5배 증가)
RAG_CONTEXT_CACHE_TTL = 1800   # 5분 → 30분 (6배 증가)
_rag_context_cache: OrderedDict = OrderedDict()
_rag_cache_hits = 0
_rag_cache_misses = 0
_rag_cache_lock = asyncio.Lock()  # 비동기 락


def _get_rag_cache_key(
    user_query: str,
    rag_enabled: bool,
    rag_collection: str,
    rag_top_k: int,
    question_set_enabled: bool,
    question_set_org: str,
    question_set_job: str,
    question_set_top_k: int,
    resume_enabled: bool,
    resume_session: str,
    resume_top_k: int
) -> str:
    """RAG 캐시 키 생성"""
    key_data = f"{user_query}|{rag_enabled}|{rag_collection}|{rag_top_k}|" \
               f"{question_set_enabled}|{question_set_org}|{question_set_job}|{question_set_top_k}|" \
               f"{resume_enabled}|{resume_session}|{resume_top_k}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _get_rag_from_cache(cache_key: str) -> Optional[Tuple[str, List[str]]]:
    """RAG 캐시에서 조회"""
    global _rag_cache_hits
    if cache_key in _rag_context_cache:
        result, timestamp = _rag_context_cache[cache_key]
        if time.time() - timestamp < RAG_CONTEXT_CACHE_TTL:
            _rag_context_cache.move_to_end(cache_key)
            _rag_cache_hits += 1
            return result
        del _rag_context_cache[cache_key]
    return None


def _add_rag_to_cache(cache_key: str, result: Tuple[str, List[str]]):
    """RAG 캐시에 추가"""
    global _rag_cache_misses
    _rag_cache_misses += 1
    while len(_rag_context_cache) >= RAG_CONTEXT_CACHE_SIZE:
        _rag_context_cache.popitem(last=False)
    _rag_context_cache[cache_key] = (result, time.time())


def get_rag_cache_stats() -> dict:
    """RAG 캐시 통계 (임베딩 캐시 포함)"""
    from app.rag.embeddings import get_embedding_client

    total = _rag_cache_hits + _rag_cache_misses
    hit_rate = (_rag_cache_hits / total * 100) if total > 0 else 0

    # 임베딩 캐시 통계도 포함
    try:
        embedding_stats = get_embedding_client().get_cache_stats()
    except:
        embedding_stats = {}

    return {
        "context_cache": {
            "cache_size": len(_rag_context_cache),
            "max_size": RAG_CONTEXT_CACHE_SIZE,
            "hits": _rag_cache_hits,
            "misses": _rag_cache_misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "ttl_seconds": RAG_CONTEXT_CACHE_TTL
        },
        "embedding_cache": embedding_stats
    }


# ===== RAG 타임아웃 설정 (GPU 가속으로 여유 확보) =====
RAG_TIMEOUT = 2.0  # 전체 RAG 처리 타임아웃 (1.5초 → 2초)
RAG_INDIVIDUAL_TIMEOUT = 1.5  # 개별 RAG 타임아웃 (1초 → 1.5초)


# ===== 스트리밍 청크 배치 설정 =====
STREAM_BATCH_SIZE = 3  # 배치할 청크 수
STREAM_BATCH_TIMEOUT = 0.02  # 배치 타임아웃 (20ms)


# ===== 메시지 히스토리 트리밍 =====
MAX_CONTEXT_TOKENS = 8000  # 최대 컨텍스트 토큰


def trim_message_history(messages: List[Dict[str, str]], max_tokens: int = MAX_CONTEXT_TOKENS) -> List[Dict[str, str]]:
    """메시지 히스토리를 토큰 제한 내로 트리밍"""
    if not messages:
        return messages

    # 시스템 메시지는 항상 유지
    system_msg = None
    other_msgs = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg
        else:
            other_msgs.append(msg)

    # 시스템 메시지 토큰 계산
    system_tokens = estimate_tokens(system_msg["content"]) if system_msg else 0
    available_tokens = max_tokens - system_tokens

    # 최신 메시지부터 역순으로 추가
    trimmed = []
    total_tokens = 0
    for msg in reversed(other_msgs):
        msg_tokens = estimate_tokens(msg["content"])
        if total_tokens + msg_tokens > available_tokens:
            break
        trimmed.insert(0, msg)
        total_tokens += msg_tokens

    # 시스템 메시지를 맨 앞에 추가
    if system_msg:
        trimmed.insert(0, system_msg)

    return trimmed


router = APIRouter()


def log_request_background(
    api_key_id: int,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    response_time: float = 0.0,
    status: str = "success",
    error_message: str = None,
    update_api_key: bool = True
):
    """
    백그라운드에서 DB 로깅 수행

    응답 반환 후 실행되므로 사용자 대기 시간에 영향 없음
    기존: 동기 커밋 (10-50ms 추가 대기)
    개선: 백그라운드 실행 (0ms 추가 대기)

    Note: 모니터링 통계는 chat_completions()에서 직접 호출됨 (중복 방지)
    """
    db = SessionLocal()
    try:
        # 요청 로그 기록
        log = RequestLog(
            api_key_id=api_key_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_time=response_time,
            status=status,
            error_message=error_message
        )
        db.add(log)

        # API 키 사용량 업데이트
        if update_api_key and status == "success":
            api_key = db.query(APIKey).filter(APIKey.id == api_key_id).first()
            if api_key:
                api_key.total_requests += 1
                api_key.total_tokens += total_tokens

        db.commit()
    except Exception as e:
        print(f"백그라운드 로깅 실패: {e}")
        db.rollback()
    finally:
        db.close()


def estimate_tokens(text: str) -> int:
    """토큰 수 추정 (간단한 방법: 단어 수 기반)"""
    # 한글: 글자당 약 1토큰, 영어: 단어당 약 1.3토큰
    korean_chars = sum(1 for c in text if '가' <= c <= '힣')
    other_chars = len(text) - korean_chars
    return korean_chars + int(other_chars / 4)


# ===== 중국어 문자 필터링 =====
# Qwen3 모델이 가끔 중국어를 출력하는 문제 해결
import re

# 중국어 문자 범위 (한자 - 한국어 한자 제외)
# CJK Unified Ideographs: U+4E00-U+9FFF
# CJK Extension A: U+3400-U+4DBF
# CJK Extension B-F: U+20000-U+2FA1F
CHINESE_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

# 일반적으로 사용되는 중국어 표현 → 한국어 대체
CHINESE_TO_KOREAN = {
    '紧张': '긴장',
    '加油': '파이팅',
    '好的': '알겠어요',
    '谢谢': '감사합니다',
    '没问题': '문제없어요',
    '不好意思': '죄송해요',
    '对不起': '미안해요',
    '你好': '안녕하세요',
    '再见': '안녕히 가세요',
    '可以': '괜찮아요',
    '当然': '물론이죠',
    '放心': '안심하세요',
    '努力': '노력',
    '成功': '성공',
    '失败': '실패',
    '开心': '기뻐요',
    '高兴': '기뻐요',
    '难过': '슬퍼요',
    '辛苦': '수고',
    '休息': '휴식',
    '加班': '야근',
    '工作': '일',
    '学习': '공부',
    '问题': '문제',
    '答案': '답변',
    '帮助': '도움',
    '了解': '이해',
    '明白': '알겠어요',
    '知道': '알아요',
    '感觉': '느낌',
    '经验': '경험',
    '准备': '준비',
    '面试': '면접',
}


def filter_chinese_characters(text: str) -> str:
    """중국어 문자를 필터링하고 한국어로 대체

    1. 알려진 중국어 표현을 한국어로 대체
    2. 남은 중국어 문자는 제거
    """
    if not text:
        return text

    # 1. 알려진 중국어 표현 대체
    for chinese, korean in CHINESE_TO_KOREAN.items():
        if chinese in text:
            text = text.replace(chinese, korean)

    # 2. 남은 중국어 문자 제거
    text = CHINESE_CHAR_PATTERN.sub('', text)

    # 3. 연속 공백 정리
    text = re.sub(r'  +', ' ', text)

    return text


def get_user_query(messages: List) -> str:
    """메시지에서 마지막 사용자 쿼리 추출"""
    for msg in reversed(messages):
        if hasattr(msg, 'role'):
            if msg.role == "user":
                return msg.content
        elif isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


async def fetch_rag_context(
    retriever,
    user_query: str,
    collection_name: str,
    top_k: int,
    base_prompt: str
) -> Optional[str]:
    """일반 RAG 컨텍스트 비동기 조회"""
    try:
        return await asyncio.to_thread(
            retriever.build_rag_system_prompt,
            query=user_query,
            collection_name=collection_name,
            top_k=top_k,
            base_prompt=base_prompt
        )
    except Exception as e:
        print(f"RAG 컨텍스트 조회 실패: {e}")
        return None


async def fetch_question_set_context(
    org_type: str,
    job_name: str,
    user_query: str,
    top_k: int
) -> Optional[str]:
    """질문셋 RAG 컨텍스트 비동기 조회"""
    try:
        relevant_questions = await asyncio.to_thread(
            search_relevant_questions,
            org_type=org_type,
            job_name=job_name,
            query=user_query,
            top_k=top_k
        )
        if relevant_questions:
            return format_questions_for_prompt(relevant_questions)
        return None
    except Exception as e:
        print(f"질문셋 RAG 컨텍스트 조회 실패: {e}")
        return None


async def fetch_resume_context(
    session_id: str,
    user_query: str,
    top_k: int
) -> Optional[str]:
    """이력서 RAG 컨텍스트 비동기 조회"""
    try:
        return await asyncio.to_thread(
            search_resume_context,
            session_id=session_id,
            query=user_query,
            top_k=top_k
        )
    except Exception as e:
        print(f"이력서 RAG 컨텍스트 조회 실패: {e}")
        return None


async def prepare_rag_contexts_parallel(
    request: ChatCompletionRequest,
    messages: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    3가지 RAG 컨텍스트를 병렬로 조회하여 메시지에 주입

    최적화:
    - 캐싱: 동일 쿼리 시 300-500ms 절약
    - 타임아웃: 느린 RAG로 인한 지연 방지 (1.5초)
    - 병렬 실행: 50-200ms
    """
    user_query = get_user_query(request.messages)
    if not user_query:
        return messages

    # ===== 캐시 확인 =====
    cache_key = _get_rag_cache_key(
        user_query=user_query,
        rag_enabled=request.rag_enabled,
        rag_collection=request.rag_collection or "default",
        rag_top_k=request.rag_top_k or 5,
        question_set_enabled=request.question_set_rag_enabled,
        question_set_org=request.question_set_org_type or "",
        question_set_job=request.question_set_job_name or "",
        question_set_top_k=request.question_set_top_k or 5,
        resume_enabled=request.resume_rag_enabled,
        resume_session=request.resume_session_id or "",
        resume_top_k=request.resume_top_k or 3
    )

    cached_result = _get_rag_from_cache(cache_key)
    if cached_result:
        rag_prompt, additional_contexts = cached_result
        # 면접 컨텍스트 빌드 (직접 전달된 기업명, 채용공고, 이력서)
        interview_context = _build_interview_context(
            company_name=request.company_name,
            job_posting=request.job_posting,
            resume_text=request.resume_text
        )
        return _apply_rag_to_messages(messages, rag_prompt, additional_contexts, interview_context)

    # 기존 시스템 프롬프트 찾기
    base_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            base_prompt = msg["content"]
            break

    # 병렬 태스크 준비
    tasks = []
    task_types = []

    # 1. 일반 RAG
    if request.rag_enabled:
        retriever = get_retriever()
        tasks.append(asyncio.wait_for(
            fetch_rag_context(
                retriever=retriever,
                user_query=user_query,
                collection_name=request.rag_collection,
                top_k=request.rag_top_k,
                base_prompt=base_prompt
            ),
            timeout=RAG_INDIVIDUAL_TIMEOUT
        ))
        task_types.append("rag")

    # 2. 질문셋 RAG
    if request.question_set_rag_enabled and request.question_set_org_type and request.question_set_job_name:
        tasks.append(asyncio.wait_for(
            fetch_question_set_context(
                org_type=request.question_set_org_type,
                job_name=request.question_set_job_name,
                user_query=user_query,
                top_k=request.question_set_top_k or 5
            ),
            timeout=RAG_INDIVIDUAL_TIMEOUT
        ))
        task_types.append("question_set")

    # 3. 이력서 RAG
    if request.resume_rag_enabled and request.resume_session_id:
        tasks.append(asyncio.wait_for(
            fetch_resume_context(
                session_id=request.resume_session_id,
                user_query=user_query,
                top_k=request.resume_top_k or 3
            ),
            timeout=RAG_INDIVIDUAL_TIMEOUT
        ))
        task_types.append("resume")

    # 면접 컨텍스트 빌드 (직접 전달된 기업명, 채용공고, 이력서)
    interview_context = _build_interview_context(
        company_name=request.company_name,
        job_posting=request.job_posting,
        resume_text=request.resume_text
    )

    if not tasks:
        # RAG 태스크가 없어도 면접 컨텍스트는 적용
        if interview_context:
            return _apply_rag_to_messages(messages, None, [], interview_context)
        return messages

    # ===== 병렬 실행 (전체 타임아웃 적용) =====
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=RAG_TIMEOUT
        )
    except asyncio.TimeoutError:
        print(f"RAG 전체 타임아웃 ({RAG_TIMEOUT}초)")
        results = [asyncio.TimeoutError()] * len(tasks)

    # 결과 처리
    rag_prompt = None
    additional_contexts = []

    for task_type, result in zip(task_types, results):
        if isinstance(result, asyncio.TimeoutError):
            print(f"{task_type} RAG 타임아웃")
            continue
        if isinstance(result, Exception):
            print(f"{task_type} RAG 오류: {result}")
            continue
        if result is None:
            continue

        if task_type == "rag":
            rag_prompt = result
        else:
            additional_contexts.append(result)

    # ===== 캐시에 저장 =====
    _add_rag_to_cache(cache_key, (rag_prompt, additional_contexts))

    return _apply_rag_to_messages(messages, rag_prompt, additional_contexts, interview_context)


# 기본 한국어 응답 지시 (중국어 출력 방지)
KOREAN_RESPONSE_INSTRUCTION = "반드시 한국어로만 답변하세요. 중국어(한자)를 절대 사용하지 마세요."


def ensure_korean_system_prompt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """시스템 프롬프트에 한국어 응답 지시 추가

    Qwen3 모델이 가끔 중국어를 출력하는 문제 방지
    """
    system_found = False
    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            system_found = True
            # 이미 한국어 지시가 있으면 스킵
            if "한국어로만" not in msg["content"] and "중국어" not in msg["content"]:
                messages[i]["content"] = f"{KOREAN_RESPONSE_INSTRUCTION}\n\n{msg['content']}"
            break

    # 시스템 메시지가 없으면 추가
    if not system_found:
        messages.insert(0, {"role": "system", "content": KOREAN_RESPONSE_INSTRUCTION})

    return messages


def _build_interview_context(
    company_name: Optional[str] = None,
    job_posting: Optional[str] = None,
    resume_text: Optional[str] = None
) -> Optional[str]:
    """면접 컨텍스트 빌드 (기업명, 채용공고, 이력서)"""
    parts = []

    if company_name:
        parts.append(f"[지원 기업/병원]\n{company_name}")

    if job_posting:
        parts.append(f"[채용공고]\n{job_posting}")

    if resume_text:
        parts.append(f"[지원자 이력서]\n{resume_text}")

    if parts:
        return "\n\n".join(parts)
    return None


def _apply_rag_to_messages(
    messages: List[Dict[str, str]],
    rag_prompt: Optional[str],
    additional_contexts: List[str],
    interview_context: Optional[str] = None
) -> List[Dict[str, str]]:
    """RAG 결과를 메시지에 적용"""
    if rag_prompt:
        system_found = False
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                messages[i]["content"] = rag_prompt
                system_found = True
                break
        if not system_found:
            messages.insert(0, {"role": "system", "content": rag_prompt})

    if additional_contexts:
        combined_context = "\n\n".join(additional_contexts)
        system_found = False
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                messages[i]["content"] = msg["content"] + "\n\n" + combined_context
                system_found = True
                break
        if not system_found:
            messages.insert(0, {"role": "system", "content": combined_context})

    # 면접 컨텍스트 추가 (기업명, 채용공고, 이력서)
    if interview_context:
        system_found = False
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                messages[i]["content"] = msg["content"] + "\n\n" + interview_context
                system_found = True
                break
        if not system_found:
            messages.insert(0, {"role": "system", "content": interview_context})

    return messages


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

    # RAG 컨텍스트 병렬 주입 (3가지 RAG 동시 처리)
    messages = await prepare_rag_contexts_parallel(request, messages)

    # 한국어 응답 지시 추가 (중국어 출력 방지)
    messages = ensure_korean_system_prompt(messages)

    # 메시지 히스토리 트리밍 (메모리 및 토큰 절약)
    messages = trim_message_history(messages)

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

        # vLLM-MLX 스트리밍 (Continuous Batching + 청크 배치)
        if is_vllm_mlx_model(request.model):
            vllm_client = get_vllm_mlx_client()
            content_buffer = []
            last_flush = time.time()

            async for chunk in await vllm_client.chat_completion(
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                stream=True,
            ):
                if chunk.get("choices") and chunk["choices"][0].get("delta"):
                    delta = chunk["choices"][0]["delta"]
                    if delta.get("content"):
                        content = delta["content"].replace('\ufffd', '')
                        if content:
                            content_buffer.append(content)

                            # 배치 조건: 크기 도달 또는 타임아웃
                            if len(content_buffer) >= STREAM_BATCH_SIZE or \
                               (time.time() - last_flush) >= STREAM_BATCH_TIMEOUT:
                                batched_content = "".join(content_buffer)
                                # 중국어 문자 필터링 적용
                                batched_content = filter_chinese_characters(batched_content)
                                if not batched_content:
                                    content_buffer = []
                                    last_flush = time.time()
                                    continue
                                stream_chunk = ChatCompletionStreamResponse(
                                    id=completion_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=[
                                        ChatCompletionStreamChoice(
                                            index=0,
                                            delta=ChatCompletionDeltaMessage(content=batched_content),
                                            finish_reason=None,
                                        )
                                    ],
                                )
                                yield f"data: {stream_chunk.model_dump_json()}\n\n"
                                content_buffer = []
                                last_flush = time.time()

            # 남은 버퍼 플러시
            if content_buffer:
                batched_content = "".join(content_buffer)
                # 중국어 문자 필터링 적용
                batched_content = filter_chinese_characters(batched_content)
                if batched_content:
                    stream_chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta=ChatCompletionDeltaMessage(content=batched_content),
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
    background_tasks: BackgroundTasks,
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

        # RAG 컨텍스트 병렬 주입 (3가지 RAG 동시 처리)
        messages = await prepare_rag_contexts_parallel(request, messages)

        # 한국어 응답 지시 추가 (중국어 출력 방지)
        messages = ensure_korean_system_prompt(messages)

        # 메시지 히스토리 트리밍 (메모리 및 토큰 절약)
        messages = trim_message_history(messages)

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

        # 중국어 문자 필터링 적용 (Qwen3 모델 대응)
        assistant_message = filter_chinese_characters(assistant_message)

        # 응답 시간 계산
        response_time = time.time() - start_time

        # 요청 카운트 감소 및 응답 시간 기록
        await load_balancer.decrement_requests(response_time)

        # 모니터링 통계 업데이트 (인증 여부와 관계없이 항상 실행)
        increment_request_stats(model=request.model, tokens=total_tokens, success=True)

        # DB 사용량 로깅 (인증 활성화 시에만)
        if api_key:
            background_tasks.add_task(
                log_request_background,
                api_key_id=api_key.id,
                model=request.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                response_time=response_time,
                status="success"
            )

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

        # 모니터링 통계 업데이트 (실패)
        increment_request_stats(model=request.model, tokens=0, success=False)

        # DB 에러 로깅 (인증 활성화 시에만)
        if api_key:
            background_tasks.add_task(
                log_request_background,
                api_key_id=api_key.id,
                model=request.model,
                response_time=time.time() - start_time,
                status="error",
                error_message=str(e),
                update_api_key=False
            )

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
