#!/bin/bash
# 배포 스크립트: 로컬 서버 재시작 + Railway 자동 배포
# 사용법: ./deploy.sh "커밋 메시지"

set -e

echo "=========================================="
echo "  LLM API 배포 스크립트"
echo "=========================================="

# 프로젝트 디렉토리로 이동
cd "$(dirname "$0")"

# 커밋 메시지 확인
COMMIT_MSG="${1:-Auto deploy: $(date '+%Y-%m-%d %H:%M:%S')}"

echo ""
echo "[1/4] Git 상태 확인..."
git status --short

# 변경사항 확인
if [ -z "$(git status --porcelain)" ]; then
    echo "   - 변경사항 없음"
    read -p "   강제 재시작하시겠습니까? (y/n): " FORCE_RESTART
    if [ "$FORCE_RESTART" != "y" ]; then
        echo "   - 배포 취소"
        exit 0
    fi
    SKIP_GIT=true
else
    SKIP_GIT=false
fi

# Git 커밋 및 푸시
if [ "$SKIP_GIT" = false ]; then
    echo ""
    echo "[2/4] Git 커밋 및 푸시..."
    git add -A
    git commit -m "$COMMIT_MSG

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
    git push origin main
    echo "   - GitHub 푸시 완료 (Railway 자동 배포 시작)"
else
    echo ""
    echo "[2/4] Git 커밋 건너뜀 (변경사항 없음)"
fi

# 로컬 API 서버만 재시작 (vLLM-MLX는 유지)
echo ""
echo "[3/4] 로컬 API 서버 재시작..."

# 기존 API 서버 프로세스 종료
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 2

# 가상환경 활성화 및 서버 시작
source venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/llm-api.log 2>&1 &

echo "   - API 서버 재시작됨"
echo "   - 로그: /tmp/llm-api.log"

# API 서버 준비 대기
echo "   - 서버 준비 대기 중..."
for i in {1..15}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   - API 서버 준비 완료"
        break
    fi
    sleep 2
done

# 상태 확인
echo ""
echo "[4/4] 배포 완료"
echo ""
echo "=========================================="
echo "  배포 결과"
echo "=========================================="
echo ""
echo "  [로컬 서버]"
echo "  - 상태: $(curl -s http://localhost:8000/health | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','오류'))" 2>/dev/null || echo '시작 중...')"
echo "  - URL:  http://localhost:8000"
echo ""
echo "  [Railway 서버]"
echo "  - 상태: 자동 배포 진행 중 (2-5분 소요)"
echo "  - 대시보드: https://railway.app/dashboard"
echo ""
echo "=========================================="
