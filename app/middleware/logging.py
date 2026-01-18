"""요청 로깅 미들웨어"""
import time
import logging
from datetime import datetime
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.routers.monitor import increment_request_stats

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """요청/응답 로깅 미들웨어"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 요청 정보
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        user_agent = request.headers.get("user-agent", "unknown")[:100]

        # 응답 처리
        try:
            response = await call_next(request)
            status_code = response.status_code
            success = status_code < 400
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise

        # 처리 시간
        process_time = time.time() - start_time

        # 로그 기록 (채팅 API만 상세 로깅)
        if "/v1/chat/completions" in path:
            logger.info(
                f"{client_ip} - {method} {path} - {status_code} - {process_time:.3f}s"
            )
            # 통계 업데이트 (토큰 수는 나중에 업데이트)
            increment_request_stats(
                model="unknown",  # 실제로는 request body에서 추출 필요
                tokens=0,
                success=success
            )

        # 응답 헤더에 처리 시간 추가
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        return response
