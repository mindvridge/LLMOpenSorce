"""FastAPI 메인 애플리케이션"""
import sys
# MLX 패키지 경로 추가 (Apple Silicon 최적화)
sys.path.insert(0, '/Users/mindprep/Library/Python/3.9/lib/python/site-packages')

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.config import get_config, get_settings
from app.routers import chat, models, health, admin, monitor, resume, prompts, rag, tts
from app.clients.openai_client import get_openai_client
from app.clients.mlx_client import get_mlx_client
from app.database import init_db
from app.load_balancer import get_load_balancer, init_load_balancer, LoadBalancerConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시
    print("🚀 LLM API Server 시작 중...")

    # 데이터베이스 초기화
    print("📊 데이터베이스 초기화 중...")
    init_db()
    print("✅ 데이터베이스 초기화 완료")

    # 설정 로드
    config = get_config()
    settings = get_settings()

    print(f"✅ 설정 로드 완료")
    print(f"   - 기본 모델: {config.default_model}")
    print(f"   - 인증 활성화: {config.auth.enabled}")

    # OpenAI 클라이언트 초기화
    openai_client = get_openai_client()
    if openai_client.is_enabled():
        print(f"✅ OpenAI 클라이언트 활성화")
    else:
        print(f"⚠️  OpenAI 클라이언트 비활성화 (OPENAI_API_KEY 미설정)")

    # 로드밸런서 초기화 (config.yaml 설정 사용)
    lb_settings = config.load_balancing
    lb_config = LoadBalancerConfig(
        enabled=lb_settings.enabled,
        local_model=lb_settings.local_model,
        cloud_model=lb_settings.cloud_model,
        max_queue_size=lb_settings.max_queue_size,
        max_wait_time=lb_settings.max_wait_time,
        auto_fallback=lb_settings.auto_fallback,
        prefer_local=lb_settings.prefer_local
    )
    init_load_balancer(lb_config)
    print(f"✅ 로드밸런서 초기화 완료")
    print(f"   - 로컬 모델: {lb_config.local_model}")
    print(f"   - 클라우드 모델: {lb_config.cloud_model}")
    print(f"   - 최대 동시 처리: {lb_config.max_queue_size}명 (초과 시 클라우드)")

    # vLLM-MLX 웜업 (Continuous Batching 서버)
    print("\n🔥 vLLM-MLX 웜업 중...")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            # vLLM-MLX 서버 확인
            health_resp = await client.get("http://localhost:8001/v1/models")
            if health_resp.status_code == 200:
                print(f"   - vLLM-MLX 서버 연결 성공")

                # 웜업 추론 실행 (첫 추론 지연 제거)
                print(f"   - 웜업 추론 실행 중...")
                warmup_resp = await client.post(
                    "http://localhost:8001/v1/chat/completions",
                    json={
                        "model": "mlx-community/Qwen3-30B-A3B-4bit",
                        "messages": [{"role": "user", "content": "Hi /nothink"}],
                        "max_tokens": 10
                    }
                )
                if warmup_resp.status_code == 200:
                    print(f"✅ vLLM-MLX 웜업 완료")
                else:
                    print(f"⚠️  vLLM-MLX 웜업 응답 오류: {warmup_resp.status_code}")
            else:
                print(f"⚠️  vLLM-MLX 서버 연결 실패")
    except Exception as e:
        print(f"⚠️  vLLM-MLX 웜업 실패: {str(e)}")

    # RAG 임베딩 모델 및 벡터 저장소 웜업
    print("\n📚 RAG 시스템 웜업 중...")
    try:
        from app.rag.embeddings import get_embedding_client
        from app.rag.vector_store import get_vector_store

        # 1. 임베딩 모델 로딩
        print("   - 임베딩 모델 로딩 중 (jhgan/ko-sroberta-multitask)...")
        embedding_client = get_embedding_client()
        _ = embedding_client.embed_query("웜업 테스트")
        print("   ✅ 임베딩 모델 로딩 완료")

        # 2. ChromaDB 초기화
        print("   - ChromaDB 초기화 중...")
        vector_store = get_vector_store()
        _ = vector_store.list_collections()
        print("   ✅ ChromaDB 초기화 완료")

        print("✅ RAG 시스템 웜업 완료")
    except Exception as e:
        print(f"⚠️  RAG 웜업 실패 (무시됨): {str(e)}")

    # 질문셋 로드 및 인덱싱
    print("\n📋 질문셋 로드 중...")
    try:
        from app.question_sets import load_all_question_sets, index_all_question_sets
        load_all_question_sets()

        # 질문셋 ChromaDB 인덱싱
        print("\n🔍 질문셋 RAG 인덱싱 중...")
        index_all_question_sets()
    except Exception as e:
        print(f"⚠️  질문셋 로드/인덱싱 실패 (무시됨): {str(e)}")

    # TTS (CosyVoice) 모델 웜업
    print("\n🎤 TTS 모델 웜업 중...")
    try:
        from app.routers.tts import get_cosyvoice_model
        tts_model = get_cosyvoice_model()
        if tts_model is not None:
            print(f"✅ CosyVoice TTS 모델 로딩 완료 (샘플레이트: {tts_model.sample_rate})")
        else:
            print(f"⚠️  TTS 모델 로딩 중... (백그라운드에서 계속)")
    except Exception as e:
        print(f"⚠️  TTS 웜업 실패 (무시됨): {str(e)}")

    print(f"\n📡 서버 실행 중:")
    print(f"   - Local: http://localhost:{settings.SERVER_PORT}")
    print(f"   - Health: http://localhost:{settings.SERVER_PORT}/health")
    print(f"   - Docs: http://localhost:{settings.SERVER_PORT}/docs")

    yield

    # 종료 시
    print("\n🛑 LLM API Server 종료 중...")
    print("✅ 클린업 완료")


# FastAPI 앱 생성
app = FastAPI(
    title="LLM API Server",
    description="vLLM-MLX 및 OpenAI 기반 LLM API 서버",
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
app.include_router(admin.router, tags=["Admin"])
app.include_router(monitor.router, tags=["Monitoring"])
app.include_router(resume.router, tags=["Resume"])
app.include_router(prompts.router, tags=["Prompts"])
app.include_router(rag.router, tags=["RAG"])
app.include_router(tts.router, tags=["TTS"])


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
            "chat_api": "/v1/chat/completions",
            "models": "/v1/models",
            "docs": "/docs",
            "test": "/test",
            "chat": "/chat",
            "chat_streaming": "/chat-streaming",
            "dashboard": "/dashboard",
        },
    }


# 모니터링 대시보드
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """모니터링 대시보드"""
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>대시보드를 찾을 수 없습니다</h1>",
            status_code=404
        )


# 테스트 페이지 엔드포인트
@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """테스트 페이지"""
    try:
        with open("test_page.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>테스트 페이지를 찾을 수 없습니다</h1>",
            status_code=404
        )


# 채팅 UI 페이지
@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """채팅 UI 페이지"""
    try:
        with open("chat_ui.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>채팅 페이지를 찾을 수 없습니다</h1>",
            status_code=404
        )


# 스트리밍 채팅 UI 페이지
@app.get("/chat-streaming", response_class=HTMLResponse)
async def chat_streaming_page():
    """스트리밍 채팅 UI 페이지"""
    try:
        with open("chat_ui_streaming.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>스트리밍 채팅 페이지를 찾을 수 없습니다</h1>",
            status_code=404
        )


# 로드밸런서 상태 조회
@app.get("/lb/status")
async def load_balancer_status():
    """로드밸런서 상태 조회"""
    lb = get_load_balancer()
    return lb.get_status()


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
