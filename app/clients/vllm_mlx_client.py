"""vLLM-MLX 클라이언트 - Continuous Batching 지원"""
import os
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass

@dataclass
class VLLMMLXConfig:
    """vLLM-MLX 설정"""
    base_url: str = "http://localhost:8001"
    model: str = "mlx-community/Qwen3-30B-A3B-4bit"
    timeout: int = 120
    max_tokens: int = 4096
    temperature: float = 0.7


class VLLMMLXClient:
    """vLLM-MLX 클라이언트 (Continuous Batching)"""

    def __init__(self, config: Optional[VLLMMLXConfig] = None):
        self.config = config or VLLMMLXConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """세션 가져오기 (연결 풀링 적용)"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,           # 최대 연결 수
                limit_per_host=50,   # 호스트당 연결
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self._session

    async def close(self):
        """세션 종료"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """서버 상태 확인"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/v1/models") as resp:
                return resp.status == 200
        except:
            return False

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """채팅 완성 (OpenAI 호환)"""
        session = await self._get_session()

        # Qwen3 모델: 마지막 user 메시지에 /nothink 추가 (thinking 비활성화로 속도 향상)
        processed_messages = []
        for i, msg in enumerate(messages):
            new_msg = dict(msg)
            if msg.get("role") == "user" and i == len(messages) - 1:
                content = msg.get("content", "")
                if "/think" not in content and "/nothink" not in content:
                    new_msg["content"] = content + " /nothink"
            processed_messages.append(new_msg)

        payload = {
            "model": model or self.config.model,
            "messages": processed_messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            "stream": stream,
        }

        url = f"{self.config.base_url}/v1/chat/completions"

        if stream:
            return self._stream_response(session, url, payload)
        else:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"vLLM-MLX 오류: {error_text}")
                return await resp.json()

    async def _stream_response(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """스트리밍 응답 처리 (바이트 버퍼로 UTF-8 안전하게)"""
        import json

        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"vLLM-MLX 스트리밍 오류: {error_text}")

            # 바이트 버퍼 사용 - UTF-8 멀티바이트가 잘리지 않도록
            byte_buffer = b''

            async for chunk in resp.content.iter_any():
                byte_buffer += chunk

                # 완전한 라인만 처리 (줄바꿈으로 구분)
                while b'\n' in byte_buffer:
                    line_end = byte_buffer.index(b'\n')
                    line_bytes = byte_buffer[:line_end]
                    byte_buffer = byte_buffer[line_end + 1:]

                    # 빈 줄 건너뛰기
                    if not line_bytes.strip():
                        continue

                    # UTF-8 디코딩 시도
                    try:
                        line = line_bytes.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        # 불완전한 UTF-8 - 버퍼에 다시 추가하고 더 기다림
                        byte_buffer = line_bytes + b'\n' + byte_buffer
                        break

                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            return
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue

            # 남은 바이트 버퍼 처리
            if byte_buffer.strip():
                try:
                    line = byte_buffer.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data != '[DONE]':
                            try:
                                yield json.loads(data)
                            except json.JSONDecodeError:
                                pass
                except UnicodeDecodeError:
                    pass

    def is_available(self) -> bool:
        """vLLM-MLX 사용 가능 여부"""
        try:
            import requests
            resp = requests.get(f"{self.config.base_url}/v1/models", timeout=2)
            return resp.status_code == 200
        except:
            return False


# 싱글톤 인스턴스
_vllm_mlx_client: Optional[VLLMMLXClient] = None


def get_vllm_mlx_client() -> VLLMMLXClient:
    """vLLM-MLX 클라이언트 싱글톤"""
    global _vllm_mlx_client
    if _vllm_mlx_client is None:
        _vllm_mlx_client = VLLMMLXClient()
    return _vllm_mlx_client
