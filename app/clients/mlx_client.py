"""MLX LLM 클라이언트 - Apple Silicon 최적화 (멀티 모델 지원)"""
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List, Union
from dataclasses import dataclass
import time


# MLX 모델 매핑 (API 모델명 -> 로컬 경로 또는 HuggingFace 경로)
import os

# 로컬 캐시 경로
MLX_CACHE_DIR = os.path.expanduser("~/.cache/mlx-models")

MLX_MODEL_MAP = {
    "mlx-qwen2.5-14b": {
        "hf_path": "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "local_path": os.path.join(MLX_CACHE_DIR, "Qwen2.5-14B-Instruct-4bit"),
    },
    "mlx-qwen3-32b": {
        "hf_path": "mlx-community/Qwen3-32B-4bit",
        "local_path": os.path.join(MLX_CACHE_DIR, "Qwen3-32B-4bit"),
    },
    # Qwen3-30B-A3B: MoE 모델 (30B 파라미터, 3B 활성화) - 5배 빠름!
    "mlx-qwen3-30b-a3b": {
        "hf_path": "mlx-community/Qwen3-30B-A3B-4bit",
        "local_path": os.path.join(MLX_CACHE_DIR, "Qwen3-30B-A3B-4bit"),
    },
}


def get_model_path(model_name: str) -> str:
    """모델 경로 반환 (로컬 우선)"""
    model_info = MLX_MODEL_MAP.get(model_name)
    if not model_info:
        return None
    # 로컬 경로가 있으면 사용
    if os.path.exists(model_info["local_path"]):
        return model_info["local_path"]
    # 없으면 HuggingFace 경로 반환
    return model_info["hf_path"]


import re

def remove_think_tags(text: str) -> str:
    """Qwen3의 <think>...</think> 태그 제거"""
    # <think>...</think> 블록 전체 제거
    cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # 불완전한 <think> 태그도 제거 (응답이 중간에 잘린 경우)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


@dataclass
class MLXConfig:
    """MLX 클라이언트 설정"""
    default_model: str = "mlx-qwen2.5-14b"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class MLXClient:
    """MLX 기반 LLM 클라이언트 (멀티 모델 지원)"""

    def __init__(self, config: Optional[MLXConfig] = None):
        self.config = config or MLXConfig()
        self._models: Dict[str, Any] = {}  # 모델 캐시
        self._tokenizers: Dict[str, Any] = {}  # 토크나이저 캐시
        self._current_model: Optional[str] = None
        self._load_lock = asyncio.Lock()
        # MLX는 동시 처리 불가 - 세마포어로 순차 처리 보장
        self._generation_semaphore = asyncio.Semaphore(1)

    async def load_model(self, model_name: str) -> bool:
        """모델 로드 (비동기)"""
        async with self._load_lock:
            # 이미 로드된 모델이면 스킵
            if model_name in self._models:
                self._current_model = model_name
                return True

            # 모델 경로 확인
            model_path = get_model_path(model_name)
            if not model_path:
                print(f"❌ 알 수 없는 MLX 모델: {model_name}")
                return False

            print(f"🔄 MLX 모델 로드 시작: {model_name} ({model_path})")

            try:
                # 기존 모델 언로드 (메모리 절약)
                if self._current_model and self._current_model != model_name:
                    print(f"🔄 이전 모델 언로드: {self._current_model}")
                    if self._current_model in self._models:
                        del self._models[self._current_model]
                        del self._tokenizers[self._current_model]

                # MLX 모델 로드 (동기 작업을 별도 스레드에서 실행)
                loop = asyncio.get_event_loop()
                model, tokenizer = await loop.run_in_executor(
                    None, self._load_model_sync, model_path
                )
                self._models[model_name] = model
                self._tokenizers[model_name] = tokenizer
                self._current_model = model_name
                print(f"✅ MLX 모델 로드 완료: {model_name} ({model_path})")
                return True
            except Exception as e:
                print(f"❌ MLX 모델 로드 실패: {str(e)}")
                return False

    def _load_model_sync(self, model_path: str):
        """모델 동기 로드"""
        from mlx_lm import load
        return load(model_path)

    def is_loaded(self, model_name: Optional[str] = None) -> bool:
        """모델 로드 상태 확인"""
        if model_name:
            return model_name in self._models
        return self._current_model is not None

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> Union[AsyncGenerator[str, None], str]:
        """텍스트 생성 (세마포어로 동시 처리 제한)"""
        # 모델 로드 확인 (세마포어 외부에서)
        if model not in self._models:
            await self.load_model(model)

        if model not in self._models:
            raise Exception(f"모델 로드 실패: {model}")

        # 메시지를 프롬프트로 변환
        prompt = self._format_messages(messages, model)

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        if stream:
            # 스트리밍: 세마포어가 포함된 래퍼 사용
            return self._generate_stream_with_semaphore(model, prompt, max_tokens, temperature)
        else:
            # 비스트리밍: 세마포어 획득 후 생성
            async with self._generation_semaphore:
                return await self._generate_full(model, prompt, max_tokens, temperature)

    async def _generate_stream_with_semaphore(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> AsyncGenerator[str, None]:
        """세마포어가 포함된 스트리밍 생성"""
        async with self._generation_semaphore:
            async for token in self._generate_stream(model, prompt, max_tokens, temperature):
                yield token

    def _format_messages(self, messages: List[Dict[str, str]], model: str) -> str:
        """ChatML 형식으로 메시지 변환"""
        formatted = ""
        is_qwen3 = model.startswith("mlx-qwen3")

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Qwen3: 마지막 user 메시지에 /nothink 추가 (thinking 비활성화로 속도 향상)
            if is_qwen3 and role == "user" and i == len(messages) - 1:
                if "/think" not in content and "/nothink" not in content:
                    content = content + " /nothink"

            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        formatted += "<|im_start|>assistant\n"
        return formatted

    async def _generate_full(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """전체 응답 생성"""
        loop = asyncio.get_event_loop()
        mlx_model = self._models[model]
        tokenizer = self._tokenizers[model]

        def _generate():
            from mlx_lm import generate
            return generate(
                mlx_model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            )

        result = await loop.run_in_executor(None, _generate)
        # Qwen3 think 태그 제거
        if model.startswith("mlx-qwen3"):
            result = remove_think_tags(result)
        return result

    async def _generate_stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> AsyncGenerator[str, None]:
        """스트리밍 응답 생성"""
        from mlx_lm import stream_generate
        import threading
        from queue import Queue as ThreadQueue

        mlx_model = self._models[model]
        tokenizer = self._tokenizers[model]

        # 스레드 안전한 큐 사용
        token_queue = ThreadQueue()
        generation_done = threading.Event()

        def _stream_worker():
            try:
                for response in stream_generate(
                    mlx_model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                ):
                    # GenerationResponse 객체에서 text 추출
                    token_text = response.text if hasattr(response, 'text') else str(response)
                    token_queue.put(token_text)
            except Exception as e:
                token_queue.put(Exception(str(e)))
            finally:
                generation_done.set()

        # 워커 스레드 시작
        thread = threading.Thread(target=_stream_worker)
        thread.start()

        # 토큰 스트리밍 (Qwen3 think 태그 실시간 필터링)
        is_qwen3 = model.startswith("mlx-qwen3")
        buffer = ""  # think 태그 감지용 버퍼
        in_think = False  # think 블록 내부 여부
        passed_think = False  # think 블록 통과 여부
        first_output = True  # 첫 출력 여부 (앞쪽 공백 제거용)

        while not generation_done.is_set() or not token_queue.empty():
            # 큐에서 토큰 가져오기
            while not token_queue.empty():
                token = token_queue.get_nowait()
                if isinstance(token, Exception):
                    raise token

                if not is_qwen3:
                    # Qwen3가 아니면 바로 출력
                    yield token
                    continue

                buffer += token

                # think 블록 처리
                if not passed_think:
                    # <think> 시작 감지
                    if "<think>" in buffer:
                        in_think = True
                        buffer = buffer.split("<think>", 1)[1]

                    # </think> 종료 감지
                    if "</think>" in buffer and in_think:
                        in_think = False
                        passed_think = True
                        # </think> 이후 내용만 유지, 앞뒤 공백 제거
                        after_think = buffer.split("</think>", 1)[1]
                        buffer = after_think.lstrip("\n\r ")
                        continue

                    # think 블록 내부면 버퍼만 유지 (출력 안함)
                    if in_think:
                        continue

                    # <가 있으면 태그 시작일 수 있으니 대기
                    if "<" in buffer:
                        continue

                # think 블록 통과 후 또는 think 없는 경우: 바로 출력
                if buffer:
                    # 첫 출력일 때 앞쪽 공백/줄바꿈 제거
                    if first_output:
                        buffer = buffer.lstrip("\n\r ")
                        if buffer:
                            first_output = False
                            yield buffer
                            buffer = ""
                    else:
                        yield buffer
                        buffer = ""

            # 짧은 대기 (CPU 사용량 절약)
            if not generation_done.is_set():
                await asyncio.sleep(0.005)

        thread.join()

        # 남은 버퍼 출력
        if buffer:
            cleaned = remove_think_tags(buffer).strip()
            if cleaned:
                yield cleaned

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "mlx-qwen2.5-14b",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """OpenAI 호환 Chat Completion API"""
        start_time = time.time()

        if stream:
            return self._stream_chat_completion(
                messages, model, max_tokens, temperature, start_time
            )

        # 전체 응답 생성
        content = await self.generate(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        # 토큰 수 추정 (대략적)
        prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        completion_tokens = len(content) // 4

        return {
            "id": f"chatcmpl-mlx-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def _stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        start_time: float,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """스트리밍 Chat Completion"""
        chat_id = f"chatcmpl-mlx-{int(time.time())}"

        async for token in await self.generate(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            yield {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(start_time),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": token,
                    },
                    "finish_reason": None,
                }],
            }

        # 종료 청크
        yield {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(start_time),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }

    def get_status(self) -> Dict[str, Any]:
        """MLX 클라이언트 상태"""
        return {
            "current_model": self._current_model,
            "loaded_models": list(self._models.keys()),
            "available_models": list(MLX_MODEL_MAP.keys()),
            "backend": "MLX (Apple Silicon)",
        }


# 싱글톤 인스턴스
_mlx_client: Optional[MLXClient] = None


def get_mlx_client() -> MLXClient:
    """MLX 클라이언트 싱글톤 반환"""
    global _mlx_client
    if _mlx_client is None:
        _mlx_client = MLXClient()
    return _mlx_client


async def init_mlx_client(config: Optional[MLXConfig] = None) -> MLXClient:
    """MLX 클라이언트 초기화 및 기본 모델 로드"""
    global _mlx_client
    _mlx_client = MLXClient(config)
    # 기본 모델 프리로드
    await _mlx_client.load_model(_mlx_client.config.default_model)
    return _mlx_client
