"""모니터링 라우터"""
import time
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from fastapi import APIRouter
from app.load_balancer import get_load_balancer

router = APIRouter(prefix="/monitor", tags=["Monitoring"])

# 서버 시작 시간
SERVER_START_TIME = time.time()

# 요청 통계 저장
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_tokens": 0,
    "hourly_requests": [],  # 최근 24시간 시간별 요청
    "model_usage": {},  # 모델별 사용량
}


def increment_request_stats(model: str, tokens: int, success: bool):
    """요청 통계 업데이트"""
    request_stats["total_requests"] += 1
    request_stats["total_tokens"] += tokens

    if success:
        request_stats["successful_requests"] += 1
    else:
        request_stats["failed_requests"] += 1

    # 모델별 사용량
    if model not in request_stats["model_usage"]:
        request_stats["model_usage"][model] = {"requests": 0, "tokens": 0}
    request_stats["model_usage"][model]["requests"] += 1
    request_stats["model_usage"][model]["tokens"] += tokens


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """상세 헬스체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(time.time() - SERVER_START_TIME),
        "uptime_human": str(timedelta(seconds=int(time.time() - SERVER_START_TIME)))
    }


@router.get("/system")
async def system_status() -> Dict[str, Any]:
    """시스템 리소스 상태"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    return {
        "cpu": {
            "percent": cpu_percent,
            "cores": psutil.cpu_count()
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent": memory.percent
        },
        "uptime_seconds": int(time.time() - SERVER_START_TIME)
    }


@router.get("/stats")
async def request_statistics() -> Dict[str, Any]:
    """요청 통계"""
    lb = get_load_balancer()
    lb_status = lb.get_status()

    return {
        "requests": {
            "total": request_stats["total_requests"],
            "successful": request_stats["successful_requests"],
            "failed": request_stats["failed_requests"],
            "success_rate": round(
                request_stats["successful_requests"] / max(request_stats["total_requests"], 1) * 100, 2
            )
        },
        "tokens": {
            "total": request_stats["total_tokens"]
        },
        "model_usage": request_stats["model_usage"],
        "load_balancer": lb_status,
        "server_uptime": str(timedelta(seconds=int(time.time() - SERVER_START_TIME)))
    }


@router.get("/dashboard")
async def dashboard_data() -> Dict[str, Any]:
    """대시보드용 종합 데이터"""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    lb = get_load_balancer()
    lb_status = lb.get_status()

    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(timedelta(seconds=int(time.time() - SERVER_START_TIME))),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2)
        },
        "api": {
            "total_requests": request_stats["total_requests"],
            "success_rate": round(
                request_stats["successful_requests"] / max(request_stats["total_requests"], 1) * 100, 2
            ),
            "total_tokens": request_stats["total_tokens"]
        },
        "load_balancer": {
            "current_requests": lb_status["current_requests"],
            "max_queue": lb_status["max_queue_size"],
            "avg_response_time": lb_status["avg_response_time"],
            "auto_fallback": lb_status["auto_fallback"]
        },
        "models": request_stats["model_usage"]
    }


@router.get("/capacity")
async def capacity_status() -> Dict[str, Any]:
    """로드밸런서 처리 용량 상태"""
    lb = get_load_balancer()
    status = lb.get_status()

    return {
        "timestamp": datetime.now().isoformat(),
        "current_requests": status["current_requests"],
        "max_queue_size": status["max_queue_size"],
        "avg_response_time": status["avg_response_time"],
        "local_model": status["local_model"],
        "cloud_model": status["cloud_model"],
        "auto_fallback": status["auto_fallback"],
        "enabled": status["enabled"]
    }


@router.get("/backends")
async def backends_status() -> Dict[str, Any]:
    """백엔드별 상세 상태"""
    lb = get_load_balancer()
    status = lb.get_status()

    backends = [
        {
            "name": "vLLM-MLX",
            "type": "vllm-mlx",
            "model": status["local_model"],
            "status": "available" if status["current_requests"] < status["max_queue_size"] else "busy",
            "slots": {
                "total": status["max_queue_size"],
                "active": status["current_requests"],
                "available": max(0, status["max_queue_size"] - status["current_requests"])
            },
            "avg_response_time": status["avg_response_time"]
        },
        {
            "name": "OpenAI",
            "type": "openai",
            "model": status["cloud_model"],
            "status": "available",
            "slots": {
                "total": 100,
                "active": 0,
                "available": 100
            },
            "note": "클라우드 백업 (로컬 초과 시 사용)"
        }
    ]

    return {
        "timestamp": datetime.now().isoformat(),
        "total_local_slots": status["max_queue_size"],
        "available_local_slots": max(0, status["max_queue_size"] - status["current_requests"]),
        "backends": backends
    }


@router.get("/route-test")
async def route_test(model: str = "vllm-qwen3-30b-a3b") -> Dict[str, Any]:
    """라우팅 테스트 - 요청이 어디로 가는지 확인"""
    lb = get_load_balancer()
    route = lb.select_model(model)

    return {
        "requested_model": model,
        "routed_to": route,
        "current_status": lb.get_status()
    }


@router.get("/rag-cache")
async def rag_cache_status() -> Dict[str, Any]:
    """RAG 캐시 상태 (컨텍스트 + 임베딩)"""
    from app.routers.chat import get_rag_cache_stats

    return {
        "timestamp": datetime.now().isoformat(),
        "rag_cache": get_rag_cache_stats()
    }
