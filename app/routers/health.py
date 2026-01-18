"""헬스 체크 라우터"""
import time
import httpx
from fastapi import APIRouter
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    서버 헬스 체크

    서버 상태와 vLLM-MLX 연결 상태를 확인합니다.
    """
    # vLLM-MLX 연결 확인
    vllm_connected = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8001/v1/models")
            vllm_connected = response.status_code == 200
    except Exception:
        vllm_connected = False

    return HealthResponse(
        status="ok" if vllm_connected else "degraded",
        vllm_connected=vllm_connected,
        timestamp=int(time.time()),
    )
