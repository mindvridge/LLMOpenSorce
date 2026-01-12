"""FastAPI 메인 애플리케이션"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import get_config, get_settings
from app.routers import chat, models, health
from app.ollama_client import get_ollama_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시
    print("🚀 LLM API Server 시작 중...")

    # 설정 로드
    config = get_config()
    settings = get_settings()

    print(f"✅ 설정 로드 완료")
    print(f"   - 기본 모델: {config.default_model}")
    print(f"   - Ollama URL: {config.ollama.base_url}")

    # Ollama 연결 확인
    ollama_client = get_ollama_client()
    is_connected = await ollama_client.health_check()

    if is_connected:
        print(f"✅ Ollama 연결 성공")

        # 설치된 모델 확인
        try:
            installed_models = await ollama_client.list_models()
            print(f"✅ 설치된 모델: {len(installed_models)}개")
            for model in installed_models:
                model_name = model.get("name", "unknown")
                model_size = model.get("size", 0)
                size_gb = model_size / (1024 ** 3) if model_size else 0
                print(f"   - {model_name} ({size_gb:.1f} GB)")
        except Exception as e:
            print(f"⚠️  모델 목록 조회 실패: {str(e)}")
    else:
        print(f"⚠️  Ollama 연결 실패 - 서버가 실행 중인지 확인하세요")

    print(f"\n📡 서버 실행 중:")
    print(f"   - Local: http://localhost:{settings.SERVER_PORT}")
    print(f"   - Health: http://localhost:{settings.SERVER_PORT}/health")
    print(f"   - Docs: http://localhost:{settings.SERVER_PORT}/docs")

    yield

    # 종료 시
    print("\n🛑 LLM API Server 종료 중...")
    await ollama_client.close()
    print("✅ 클린업 완료")


# FastAPI 앱 생성
app = FastAPI(
    title="LLM API Server",
    description="Ollama 기반 OpenAI 호환 LLM API 서버",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 글로벌 예외 처리
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": f"내부 서버 오류: {str(exc)}",
                "type": "internal_error",
                "code": "internal_error",
            }
        },
    )


# 라우터 등록
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(models.router, tags=["Models"])


# 루트 엔드포인트
@app.get("/")
async def root():
    """API 루트"""
    return {
        "name": "LLM API Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=True,
        log_level="info",
    )
