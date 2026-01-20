"""
자동 로드밸런서 모듈
- vLLM-MLX 대기열 모니터링
- 임계값 초과 시 클라우드 API로 자동 전환
"""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import httpx


@dataclass
class LoadBalancerConfig:
    """로드밸런서 설정"""
    enabled: bool = True
    local_model: str = "vllm-qwen3-30b-a3b"
    cloud_model: str = "gpt-5-mini"

    # 임계값 설정 (1~4명 로컬 보장, 5명부터 클라우드)
    max_queue_size: int = 4  # 로컬 최대 동시 처리
    max_wait_time: float = 3.0  # 최대 대기 시간 (초)

    # 자동 전환 조건
    auto_fallback: bool = True  # 자동 클라우드 전환
    prefer_local: bool = True  # 로컬 우선 사용


@dataclass
class QueueStatus:
    """대기열 상태"""
    current_requests: int = 0
    avg_response_time: float = 0.0
    last_updated: float = field(default_factory=time.time)
    recent_times: list = field(default_factory=list)


class LoadBalancer:
    """자동 로드밸런서"""

    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.queue_status = QueueStatus()
        self._lock = asyncio.Lock()
        self._active_requests = 0

    async def increment_requests(self):
        """요청 시작"""
        async with self._lock:
            self._active_requests += 1
            self.queue_status.current_requests = self._active_requests

    async def decrement_requests(self, response_time: float):
        """요청 완료"""
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self.queue_status.current_requests = self._active_requests

            # 응답 시간 기록 (최근 10개)
            self.queue_status.recent_times.append(response_time)
            if len(self.queue_status.recent_times) > 10:
                self.queue_status.recent_times.pop(0)

            # 평균 응답 시간 계산
            if self.queue_status.recent_times:
                self.queue_status.avg_response_time = sum(self.queue_status.recent_times) / len(self.queue_status.recent_times)

            self.queue_status.last_updated = time.time()

    def should_use_cloud(self, requested_model: str) -> tuple[bool, str]:
        """
        클라우드 API 사용 여부 결정

        Returns:
            (use_cloud: bool, reason: str)
        """
        # 로드밸런싱 비활성화
        if not self.config.enabled:
            return False, "로드밸런싱 비활성화"

        # 이미 클라우드 모델 요청
        if requested_model == self.config.cloud_model:
            return True, "클라우드 모델 직접 요청"

        # 자동 전환 비활성화
        if not self.config.auto_fallback:
            return False, "자동 전환 비활성화"

        # 대기열 크기 초과 (1~4명 로컬, 5명+ 클라우드)
        if self.queue_status.current_requests > self.config.max_queue_size:
            return True, f"대기열 초과 ({self.queue_status.current_requests}/{self.config.max_queue_size})"

        return False, f"로컬 사용 ({self.queue_status.current_requests}/{self.config.max_queue_size})"

    def select_model(self, requested_model: str) -> Dict[str, Any]:
        """
        최적 모델 선택

        Returns:
            {
                "model": str,
                "provider": str,  # "vllm-mlx" or "openai"
                "reason": str,
                "queue_status": dict
            }
        """
        use_cloud, reason = self.should_use_cloud(requested_model)

        if use_cloud:
            return {
                "model": self.config.cloud_model,
                "provider": "openai",
                "reason": reason,
                "original_model": requested_model,
                "queue_status": {
                    "current_requests": self.queue_status.current_requests,
                    "avg_response_time": round(self.queue_status.avg_response_time, 2)
                }
            }
        else:
            return {
                "model": self.config.local_model,
                "provider": "vllm-mlx",
                "reason": reason,
                "original_model": requested_model,
                "queue_status": {
                    "current_requests": self.queue_status.current_requests,
                    "avg_response_time": round(self.queue_status.avg_response_time, 2)
                }
            }

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        return {
            "enabled": self.config.enabled,
            "auto_fallback": self.config.auto_fallback,
            "current_requests": self.queue_status.current_requests,
            "max_queue_size": self.config.max_queue_size,
            "avg_response_time": round(self.queue_status.avg_response_time, 2),
            "max_wait_time": self.config.max_wait_time,
            "local_model": self.config.local_model,
            "cloud_model": self.config.cloud_model
        }


# 전역 로드밸런서 인스턴스
_load_balancer: Optional[LoadBalancer] = None


def get_load_balancer() -> LoadBalancer:
    """로드밸런서 인스턴스 반환"""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    return _load_balancer


def init_load_balancer(config: LoadBalancerConfig) -> LoadBalancer:
    """로드밸런서 초기화"""
    global _load_balancer
    _load_balancer = LoadBalancer(config)
    return _load_balancer
