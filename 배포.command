#!/bin/bash
# 배포 스크립트 (더블클릭 실행용)
cd "$(dirname "$0")"

echo "=========================================="
echo "  LLM API 배포"
echo "=========================================="
echo ""

# 커밋 메시지 입력
read -p "커밋 메시지 (Enter: 자동 생성): " MSG

if [ -z "$MSG" ]; then
    MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
fi

./deploy.sh "$MSG"

echo ""
echo "아무 키나 누르면 종료됩니다..."
read -n 1
