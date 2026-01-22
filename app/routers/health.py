"""헬스 체크 라우터 (최적화: TTL 캐싱)"""
import time
import socket
from fastapi import APIRouter
from app.models.schemas import HealthResponse

router = APIRouter()

# 헬스 체크 캐시 (TTL 기반)
_health_cache = {
    "vllm_connected": False,
    "last_check": 0,
    "ttl": 5  # 5초 캐시
}


def check_vllm_status_sync() -> bool:
    """vLLM-MLX 연결 상태 확인 (소켓 기반 - 가장 빠름)"""
    global _health_cache

    now = time.time()

    # 캐시 유효성 확인
    if now - _health_cache["last_check"] < _health_cache["ttl"]:
        return _health_cache["vllm_connected"]

    # 캐시 만료: 소켓으로 포트 열림 확인 (빠르고 안정적)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 8001))
        sock.close()
        connected = result == 0
    except Exception:
        connected = False

    # 캐시 업데이트
    _health_cache["vllm_connected"] = connected
    _health_cache["last_check"] = now

    return connected


async def check_vllm_status() -> bool:
    """vLLM-MLX 연결 상태 확인 (비동기 래퍼)"""
    return check_vllm_status_sync()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    서버 헬스 체크 (최적화)

    - 글로벌 HTTP 클라이언트 재사용
    - 5초 TTL 캐싱으로 불필요한 요청 방지
    """
    vllm_connected = await check_vllm_status()

    return HealthResponse(
        status="ok" if vllm_connected else "degraded",
        vllm_connected=vllm_connected,
        timestamp=int(time.time()),
    )


@router.get("/health/detailed")
async def health_check_detailed():
    """상세 헬스 체크 (캐시 통계 포함)"""
    vllm_connected = await check_vllm_status()

    # 임베딩 캐시 통계 (있는 경우)
    embedding_stats = None
    try:
        from app.rag.embeddings import get_embedding_client
        embedding_client = get_embedding_client()
        embedding_stats = embedding_client.get_cache_stats()
    except Exception:
        pass

    return {
        "status": "ok" if vllm_connected else "degraded",
        "vllm_connected": vllm_connected,
        "timestamp": int(time.time()),
        "cache": {
            "health_check_ttl": _health_cache["ttl"],
            "last_vllm_check": int(_health_cache["last_check"]),
            "embedding_cache": embedding_stats
        }
    }
