"""관리자 API 라우터"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from app.models.database import APIKey, RequestLog
from app.auth import verify_admin_key, generate_api_key

router = APIRouter(prefix="/admin", tags=["Admin"])


# ===== Pydantic 모델 =====

class APIKeyCreate(BaseModel):
    """API 키 생성 요청"""
    name: str
    is_admin: bool = False


class APIKeyResponse(BaseModel):
    """API 키 응답"""
    id: int
    key: str
    name: str
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_used_at: Optional[datetime]
    total_requests: int
    total_tokens: int

    class Config:
        from_attributes = True


class APIKeyListResponse(BaseModel):
    """API 키 목록 응답"""
    total: int
    keys: List[APIKeyResponse]


class UsageStats(BaseModel):
    """사용량 통계"""
    total_requests: int
    total_tokens: int
    total_keys: int
    active_keys: int


# ===== API 키 관리 =====

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    key_data: APIKeyCreate,
    db: Session = Depends(get_db),
    admin: APIKey = Depends(verify_admin_key)
):
    """
    새 API 키 생성 (관리자 전용)

    - **name**: 키 이름/설명
    - **is_admin**: 관리자 키 여부
    """
    # 새 키 생성
    new_key = APIKey(
        key=generate_api_key(),
        name=key_data.name,
        is_admin=key_data.is_admin
    )

    db.add(new_key)
    db.commit()
    db.refresh(new_key)

    return new_key


@router.get("/api-keys", response_model=APIKeyListResponse)
async def list_api_keys(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    admin: APIKey = Depends(verify_admin_key)
):
    """
    모든 API 키 조회 (관리자 전용)

    - **skip**: 건너뛸 개수
    - **limit**: 최대 개수
    """
    keys = db.query(APIKey).offset(skip).limit(limit).all()
    total = db.query(APIKey).count()

    return {"total": total, "keys": keys}


@router.get("/api-keys/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    admin: APIKey = Depends(verify_admin_key)
):
    """
    특정 API 키 조회 (관리자 전용)

    - **key_id**: API 키 ID
    """
    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API 키를 찾을 수 없습니다."
        )

    return key


@router.patch("/api-keys/{key_id}/toggle")
async def toggle_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    admin: APIKey = Depends(verify_admin_key)
):
    """
    API 키 활성화/비활성화 토글 (관리자 전용)

    - **key_id**: API 키 ID
    """
    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API 키를 찾을 수 없습니다."
        )

    key.is_active = not key.is_active
    db.commit()

    return {
        "id": key.id,
        "is_active": key.is_active,
        "message": f"API 키가 {'활성화' if key.is_active else '비활성화'}되었습니다."
    }


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    admin: APIKey = Depends(verify_admin_key)
):
    """
    API 키 삭제 (관리자 전용)

    - **key_id**: API 키 ID
    """
    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API 키를 찾을 수 없습니다."
        )

    db.delete(key)
    db.commit()

    return {"message": "API 키가 삭제되었습니다."}


# ===== 통계 =====

@router.get("/stats", response_model=UsageStats)
async def get_stats(
    db: Session = Depends(get_db),
    admin: APIKey = Depends(verify_admin_key)
):
    """
    전체 통계 조회 (관리자 전용)
    """
    total_keys = db.query(APIKey).count()
    active_keys = db.query(APIKey).filter(APIKey.is_active == True).count()

    # 전체 요청 및 토큰 합계
    total_requests = db.query(APIKey).with_entities(
        db.func.sum(APIKey.total_requests)
    ).scalar() or 0

    total_tokens = db.query(APIKey).with_entities(
        db.func.sum(APIKey.total_tokens)
    ).scalar() or 0

    return {
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "total_keys": total_keys,
        "active_keys": active_keys
    }


@router.get("/api-keys/{key_id}/usage")
async def get_key_usage(
    key_id: int,
    limit: int = 100,
    db: Session = Depends(get_db),
    admin: APIKey = Depends(verify_admin_key)
):
    """
    특정 API 키의 사용 로그 조회 (관리자 전용)

    - **key_id**: API 키 ID
    - **limit**: 최대 개수
    """
    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API 키를 찾을 수 없습니다."
        )

    logs = db.query(RequestLog).filter(
        RequestLog.api_key_id == key_id
    ).order_by(RequestLog.created_at.desc()).limit(limit).all()

    return {
        "key_id": key_id,
        "key_name": key.name,
        "total_requests": key.total_requests,
        "total_tokens": key.total_tokens,
        "recent_logs": [
            {
                "id": log.id,
                "model": log.model,
                "tokens": log.total_tokens,
                "response_time": log.response_time,
                "status": log.status,
                "created_at": log.created_at
            }
            for log in logs
        ]
    }
