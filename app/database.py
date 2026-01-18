"""데이터베이스 설정 및 연결"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import get_settings

settings = get_settings()

# SQLite 데이터베이스 경로
DATABASE_PATH = os.path.expanduser(settings.DATABASE_PATH)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

# 데이터베이스 URL
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# 엔진 생성
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite에서 필요
)

# 세션 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스
Base = declarative_base()


def get_db():
    """데이터베이스 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """데이터베이스 초기화"""
    Base.metadata.create_all(bind=engine)
