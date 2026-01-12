#!/bin/bash

# LLM API Server 시작 스크립트

# 스크립트 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== LLM API Server 시작 ==="
echo "프로젝트 디렉토리: $PROJECT_DIR"

# 프로젝트 디렉토리로 이동
cd "$PROJECT_DIR" || exit 1

# 환경변수 로드
if [ -f .env ]; then
    echo "✓ .env 파일 로드"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠ .env 파일이 없습니다. .env.example을 참고하여 생성하세요."
fi

# Ollama 실행 확인
echo ""
echo "Ollama 상태 확인 중..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠ Ollama가 실행 중이 아닙니다."
    echo "Ollama 시작 중..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Ollama 연결 확인
if curl -s http://localhost:11434 > /dev/null; then
    echo "✓ Ollama 실행 중"
else
    echo "✗ Ollama 연결 실패. 수동으로 시작하세요: ollama serve"
    exit 1
fi

# 설치된 모델 확인
echo ""
echo "설치된 모델:"
ollama list

# 기본 모델 확인
DEFAULT_MODEL=${DEFAULT_MODEL:-"qwen3-ultra:latest"}
if ! ollama list | grep -q "$DEFAULT_MODEL"; then
    echo ""
    echo "⚠ 기본 모델 '$DEFAULT_MODEL'이(가) 설치되어 있지 않습니다."
    read -p "지금 다운로드하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "모델 다운로드 중: $DEFAULT_MODEL"
        ollama pull "$DEFAULT_MODEL"
    else
        echo "모델 다운로드를 건너뜁니다. config.yaml에서 다른 모델을 기본값으로 설정하세요."
    fi
fi

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    echo ""
    echo "✓ 가상환경 활성화"
    source venv/bin/activate
fi

# FastAPI 서버 시작
echo ""
echo "=== API 서버 시작 중 ==="
echo ""

# 환경변수에서 포트 가져오기 (기본값: 8000)
SERVER_PORT=${SERVER_PORT:-8000}
SERVER_HOST=${SERVER_HOST:-0.0.0.0}

# 서버 실행
exec uvicorn app.main:app --host "$SERVER_HOST" --port "$SERVER_PORT" --log-level info

# 또는 워커 모드로 실행하려면 아래 주석 해제
# exec uvicorn app.main:app --host "$SERVER_HOST" --port "$SERVER_PORT" --workers 4 --log-level info
