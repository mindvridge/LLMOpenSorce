"""헬스 체크 라우터 (최적화: 글로벌 클라이언트 + TTL 캐싱)"""
import time
import httpx
from fastapi import APIRouter
from app.models.schemas import HealthResponse

router = APIRouter()

# 글로벌 HTTP 클라이언트 (연결 재사용)
_http_client: httpx.AsyncClient = None

# 헬스 체크 캐시 (TTL 기반)
_health_cache = {
    "vllm_connected": False,
    "last_check": 0,
    "ttl": 5  # 5초 캐시
}


async def get_http_client() -> httpx.AsyncClient:
    """글로벌 HTTP 클라이언트 반환 (연결 풀링)"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=5.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    return _http_client


async def check_vllm_status() -> bool:
    """vLLM-MLX 연결 상태 확인 (캐싱 적용)"""
    global _health_cache

    now = time.time()

    # 캐시 유효성 확인
    if now - _health_cache["last_check"] < _health_cache["ttl"]:
        return _health_cache["vllm_connected"]

    # 캐시 만료: 실제 확인
    try:
        client = await get_http_client()
        response = await client.get("http://localhost:8001/v1/models")
        connected = response.status_code == 200
    except Exception:
        connected = False

    # 캐시 업데이트
    _health_cache["vllm_connected"] = connected
    _health_cache["last_check"] = now

    return connected


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
