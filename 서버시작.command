#!/bin/bash
# LLM API 로컬 서버 시작 (더블클릭 실행용)

cd "$(dirname "$0")"
./start_local.sh

echo ""
echo "창을 닫으려면 아무 키나 누르세요..."
read -n 1
