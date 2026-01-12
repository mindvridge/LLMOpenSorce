#!/bin/bash

# API 테스트 스크립트

BASE_URL="${1:-http://localhost:8000}"

echo "=== LLM API Server 테스트 ==="
echo "Base URL: $BASE_URL"
echo ""

# 1. Health Check
echo "1. Health Check"
echo "GET $BASE_URL/health"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""
echo ""

# 2. Root Endpoint
echo "2. Root Endpoint"
echo "GET $BASE_URL/"
curl -s "$BASE_URL/" | python3 -m json.tool
echo ""
echo ""

# 3. List Models
echo "3. List Models"
echo "GET $BASE_URL/v1/models"
curl -s "$BASE_URL/v1/models" | python3 -m json.tool
echo ""
echo ""

# 4. Chat Completion (Non-streaming)
echo "4. Chat Completion (Non-streaming)"
echo "POST $BASE_URL/v1/chat/completions"
cat <<'EOF' | curl -s -X POST "$BASE_URL/v1/chat/completions" -H "Content-Type: application/json" -d @- | python3 -m json.tool
{
  "model": "qwen3-ultra:latest",
  "messages": [
    {"role": "user", "content": "1+1은?"}
  ],
  "temperature": 0.3,
  "stream": false
}
EOF
echo ""
echo ""

# 5. Chat Completion (Streaming) - First 5 chunks
echo "5. Chat Completion (Streaming) - First 5 chunks"
echo "POST $BASE_URL/v1/chat/completions (stream=true)"
cat <<'EOF' | curl -s -N -X POST "$BASE_URL/v1/chat/completions" -H "Content-Type: application/json" -d @- | head -5
{
  "model": "qwen3-ultra:latest",
  "messages": [
    {"role": "user", "content": "1부터 3까지 세어주세요."}
  ],
  "temperature": 0.3,
  "stream": true
}
EOF
echo ""
echo "..."
echo ""

echo "=== 테스트 완료 ==="
