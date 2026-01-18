"""OpenAI API 클라이언트"""
import asyncio
from typing import AsyncGenerator, Dict, Any, List
from openai import AsyncOpenAI
from app.config import get_config, get_settings
import os


class OpenAIClient:
    """OpenAI API 클라이언트"""

    def __init__(self):
        self.config = get_config()
        settings = get_settings()

        # API 키는 Settings에서 가져오기 (환경변수 자동 로드)
        api_key = settings.OPENAI_API_KEY or self.config.openai.api_key

        if not api_key:
            print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다.")
            self.client = None
            self.enabled = False
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                timeout=self.config.openai.timeout
            )
            self.enabled = self.config.openai.enabled
            print("✅ OpenAI 클라이언트 초기화 완료")

    def is_enabled(self) -> bool:
        """OpenAI 사용 가능 여부"""
        return self.enabled and self.client is not None

    def is_openai_model(self, model: str) -> bool:
        """OpenAI 모델인지 확인"""
        openai_models = ["gpt-5.2", "gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "gpt-4", "gpt-3.5-turbo"]
        return any(model.startswith(m) for m in openai_models)

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        채팅 완성 요청 (비스트리밍)

        Args:
            model: 모델 이름
            messages: 메시지 목록
            temperature: 온도
            max_tokens: 최대 토큰
            stream: 스트리밍 여부

        Returns:
            완성 결과
        """
        if not self.is_enabled():
            raise Exception("OpenAI API가 활성화되지 않았습니다.")

        # o1 시리즈는 temperature와 max_tokens를 지원하지 않음
        if model.startswith("o1"):
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages
            )
        # gpt-5 시리즈는 max_completion_tokens 사용
        elif model.startswith("gpt-5"):
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stream=False
            )
        else:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )

        return {
            "model": response.model,
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 채팅 완성

        Args:
            model: 모델 이름
            messages: 메시지 목록
            temperature: 온도
            max_tokens: 최대 토큰

        Yields:
            청크 데이터
        """
        if not self.is_enabled():
            raise Exception("OpenAI API가 활성화되지 않았습니다.")

        # o1 시리즈는 스트리밍을 지원하지 않음
        if model.startswith("o1"):
            # 일반 완성 후 한번에 반환
            result = await self.chat_completion(model, messages)
            yield result["content"]
            return

        # gpt-5 시리즈는 max_completion_tokens 사용
        if model.startswith("gpt-5"):
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stream=True
            )
        else:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# 싱글톤 인스턴스
_openai_client = None


def get_openai_client() -> OpenAIClient:
    """OpenAI 클라이언트 싱글톤 반환"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client
