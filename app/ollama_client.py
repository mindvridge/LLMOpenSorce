"""Ollama 클라이언트 모듈"""
import json
import time
from typing import List, Dict, Any, AsyncGenerator, Optional
import httpx
from app.config import get_config


class OllamaClient:
    """Ollama API 클라이언트"""

    def __init__(self):
        config = get_config()
        self.base_url = config.ollama.base_url
        self.timeout = config.ollama.timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """채팅 완성 요청 (비스트리밍)"""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise Exception("Ollama 요청 시간이 초과되었습니다.")
        except httpx.HTTPStatusError as e:
            raise Exception(f"Ollama API 오류: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Ollama 연결 실패: {str(e)}")

    async def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """채팅 완성 요청 (스트리밍)"""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            yield chunk
                        except json.JSONDecodeError:
                            continue
        except httpx.TimeoutException:
            raise Exception("Ollama 요청 시간이 초과되었습니다.")
        except httpx.HTTPStatusError as e:
            raise Exception(f"Ollama API 오류: {e.response.status_code}")
        except Exception as e:
            raise Exception(f"Ollama 스트리밍 실패: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 조회"""
        url = f"{self.base_url}/api/tags"

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            raise Exception(f"모델 목록 조회 실패: {str(e)}")

    async def pull_model(self, model: str) -> AsyncGenerator[Dict[str, Any], None]:
        """모델 다운로드"""
        url = f"{self.base_url}/api/pull"
        payload = {"name": model, "stream": True}

        try:
            async with self.client.stream("POST", url, json=payload, timeout=600) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            yield chunk
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise Exception(f"모델 다운로드 실패: {str(e)}")

    async def delete_model(self, model: str) -> bool:
        """모델 삭제"""
        url = f"{self.base_url}/api/delete"
        payload = {"name": model}

        try:
            response = await self.client.delete(url, json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            raise Exception(f"모델 삭제 실패: {str(e)}")

    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """모델 정보 조회"""
        url = f"{self.base_url}/api/show"
        payload = {"name": model}

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"모델 정보 조회 실패: {str(e)}")

    async def health_check(self) -> bool:
        """Ollama 서버 상태 확인"""
        try:
            response = await self.client.get(self.base_url, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """클라이언트 종료"""
        await self.client.aclose()


# 전역 클라이언트 인스턴스
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Ollama 클라이언트 가져오기"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
