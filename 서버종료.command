#!/bin/bash
# LLM API 로컬 서버 종료 (더블클릭 실행용)

cd "$(dirname "$0")"
./stop_local.sh

echo ""
echo "창을 닫으려면 아무 키나 누르세요..."
read -n 1
