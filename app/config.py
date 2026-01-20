"""설정 관리 모듈"""
import os
from pathlib import Path
from typing import List, Dict, Any
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """모델 설정"""
    name: str
    display_name: str
    context_length: int


class ServerConfig(BaseModel):
    """서버 설정"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"


class CloudflareConfig(BaseModel):
    """Cloudflare Tunnel 설정"""
    enabled: bool = False
    tunnel_name: str = "llm-api"


class OpenAIConfig(BaseModel):
    """OpenAI 설정"""
    api_key: str = ""
    enabled: bool = True
    timeout: int = 120


class AuthConfig(BaseModel):
    """인증 설정"""
    enabled: bool = False  # 기본값: 인증 비활성화 (외부에서도 키 없이 접근 가능)


class RateLimitConfig(BaseModel):
    """Rate Limiting 설정"""
    enabled: bool = True
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000


class CORSConfig(BaseModel):
    """CORS 설정"""
    allowed_origins: List[str] = ["*"]


class LoadBalancingConfig(BaseModel):
    """로드밸런싱 설정"""
    enabled: bool = True
    auto_fallback: bool = True
    prefer_local: bool = True
    local_model: str = "vllm-qwen3-30b-a3b"
    cloud_model: str = "gpt-5-mini"
    max_queue_size: int = 4
    max_wait_time: float = 3.0


class AppConfig(BaseModel):
    """전체 애플리케이션 설정"""
    server: ServerConfig
    cloudflare: CloudflareConfig
    openai: OpenAIConfig
    default_model: str
    available_models: List[ModelConfig]
    auth: AuthConfig
    rate_limit: RateLimitConfig
    cors: CORSConfig
    load_balancing: LoadBalancingConfig = LoadBalancingConfig()


class Settings(BaseSettings):
    """환경변수 설정"""
    ADMIN_API_KEY: str = "sk-admin-change-me"
    OPENAI_API_KEY: str = ""
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    DATABASE_PATH: str = "./data/llm_server.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """YAML 설정 파일 로드"""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return AppConfig(**config_data)


# 전역 설정 객체
settings = Settings()
app_config: AppConfig = None


def get_config() -> AppConfig:
    """앱 설정 가져오기"""
    global app_config
    if app_config is None:
        app_config = load_config()
    return app_config


def get_settings() -> Settings:
    """환경변수 설정 가져오기"""
    return settings
