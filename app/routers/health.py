"""헬스 체크 라우터"""
import time
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.ollama_client import get_ollama_client

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    서버 헬스 체크

    서버 상태와 Ollama 연결 상태를 확인합니다.
    """
    ollama_client = get_ollama_client()

    # Ollama 연결 확인
    ollama_connected = await ollama_client.health_check()

    return HealthResponse(
        status="ok" if ollama_connected else "degraded",
        ollama_connected=ollama_connected,
        timestamp=int(time.time()),
    )
