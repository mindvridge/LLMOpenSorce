"""API 키 인증 미들웨어"""
import secrets
from datetime import datetime
from typing import Optional
from fastapi import Header, HTTPException, Depends, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.database import APIKey
from app.config import get_config


def generate_api_key() -> str:
    """새로운 API 키 생성"""
    return f"sk-{secrets.token_urlsafe(32)}"


async def verify_api_key(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> APIKey:
    """
    API 키 검증

    Authorization 헤더에서 Bearer 토큰 또는 직접 키 추출
    """
    config = get_config()

    # 인증이 비활성화된 경우
    if not config.auth.enabled:
        return None

    # Authorization 헤더가 없는 경우
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "API 키가 필요합니다. Authorization 헤더에 'Bearer YOUR_API_KEY' 형식으로 전달하세요.",
                    "type": "invalid_request_error",
                    "code": "missing_api_key"
                }
            }
        )

    # Bearer 토큰 형식 처리
    api_key = authorization
    if authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "")

    # 데이터베이스에서 키 조회
    db_key = db.query(APIKey).filter(APIKey.key == api_key).first()

    if not db_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "유효하지 않은 API 키입니다.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )

    if not db_key.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "message": "비활성화된 API 키입니다.",
                    "type": "invalid_request_error",
                    "code": "inactive_api_key"
                }
            }
        )

    # 마지막 사용 시간 업데이트
    db_key.last_used_at = datetime.utcnow()
    db.commit()

    return db_key


async def verify_admin_key(
    api_key: APIKey = Depends(verify_api_key)
) -> APIKey:
    """관리자 키 검증"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "message": "관리자 권한이 필요합니다.",
                    "type": "invalid_request_error",
                    "code": "admin_required"
                }
            }
        )

    if not api_key.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "message": "관리자 권한이 필요합니다.",
                    "type": "invalid_request_error",
                    "code": "admin_required"
                }
            }
        )

    return api_key
