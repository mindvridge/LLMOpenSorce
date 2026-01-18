#!/bin/bash
# LLM API 로컬 서버 시작 스크립트
# 사용법: ./start_local.sh

set -e

echo "=========================================="
echo "  LLM API 로컬 서버 시작"
echo "=========================================="

# 프로젝트 디렉토리로 이동
cd "$(dirname "$0")"

# 1. vLLM-MLX 서버 시작 (Continuous Batching)
echo ""
echo "[1/2] vLLM-MLX 서버 시작 중..."

# 기존 vLLM-MLX 프로세스 종료
pkill -f "vllm-mlx serve" 2>/dev/null || true
sleep 1

# vLLM-MLX 서버 백그라운드 실행
cd /Users/mindprep/vllm-mlx
source venv/bin/activate
nohup vllm-mlx serve mlx-community/Qwen3-30B-A3B-4bit \
    --port 8001 \
    --continuous-batching \
    --max-num-seqs 10 \
    --prefill-batch-size 8 \
    --completion-batch-size 10 \
    --enable-prefix-cache \
    > /tmp/vllm-mlx.log 2>&1 &

echo "   - vLLM-MLX 서버 시작됨 (포트: 8001)"
echo "   - 로그: /tmp/vllm-mlx.log"

# vLLM-MLX 서버 준비 대기
echo "   - 서버 준비 대기 중..."
for i in {1..30}; do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "   - vLLM-MLX 서버 준비 완료"
        break
    fi
    sleep 2
done

# 2. LLM API 서버 시작
echo ""
echo "[2/2] LLM API 서버 시작 중..."

cd /Users/mindprep/llm-api-server

# 기존 API 서버 프로세스 종료
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 1

# 가상환경 활성화 및 서버 시작
source venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/llm-api.log 2>&1 &

echo "   - API 서버 시작됨 (포트: 8000)"
echo "   - 로그: /tmp/llm-api.log"

# API 서버 준비 대기
echo "   - 서버 준비 대기 중..."
sleep 5
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   - API 서버 준비 완료"
        break
    fi
    sleep 2
done

# 로컬 IP 가져오기 (여러 인터페이스 시도)
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || \
           ipconfig getifaddr en1 2>/dev/null || \
           ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' || \
           echo "알 수 없음")

# 상태 출력
echo ""
echo "=========================================="
echo "  서버 시작 완료"
echo "=========================================="
echo ""
echo "  [로컬 접속]"
echo "  - 채팅 (스트리밍): http://localhost:8000/chat-streaming"
echo "  - 채팅 (일반):     http://localhost:8000/chat"
echo "  - 대시보드:        http://localhost:8000/dashboard"
echo "  - API 문서:        http://localhost:8000/docs"
echo "  - 테스트 페이지:   http://localhost:8000/test"
echo ""
echo "  [외부 접속 (Cloudflare Tunnel)]"
echo "  - 채팅:            https://api.mindprep.co.kr/chat-streaming"
echo "  - API 엔드포인트:  https://api.mindprep.co.kr/v1/chat/completions"
echo "  - API 문서:        https://api.mindprep.co.kr/docs"
echo ""
echo "  [서버 정보]"
echo "  - vLLM-MLX:        http://localhost:8001"
echo "  - API 서버:        http://localhost:8000"
echo "  - 로드밸런싱:      1~4명 로컬, 5명+ 클라우드"
echo ""
echo "=========================================="

# 로드밸런서 상태 확인
echo ""
echo "로드밸런서 상태:"
curl -s http://localhost:8000/lb/status | python3 -m json.tool 2>/dev/null || echo "상태 확인 실패"
