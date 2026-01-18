"""데이터베이스 모델"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float
from app.database import Base


class APIKey(Base):
    """API 키 모델"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)  # 키 이름/설명
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)  # 관리자 키 여부
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

    # 사용량 통계
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)


class RequestLog(Base):
    """요청 로그 모델"""
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    api_key_id = Column(Integer, nullable=True)  # API 키 ID
    model = Column(String, nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    response_time = Column(Float, default=0.0)  # 응답 시간 (초)
    status = Column(String, default="success")  # success, error
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class RAGDocument(Base):
    """RAG 문서 모델"""
    __tablename__ = "rag_documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)  # 원본 파일명
    file_hash = Column(String, unique=True, index=True, nullable=False)  # 중복 방지
    collection_name = Column(String, default="default", index=True)  # ChromaDB 컬렉션
    chunk_count = Column(Integer, default=0)  # 청크 수
    total_chars = Column(Integer, default=0)  # 총 문자 수
    status = Column(String, default="processing")  # processing, completed, failed
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
