#!/bin/bash
# LLM API 로컬 서버 종료 스크립트
# 사용법: ./stop_local.sh

echo "=========================================="
echo "  LLM API 로컬 서버 종료"
echo "=========================================="

# vLLM-MLX 서버 종료
echo ""
echo "[1/2] vLLM-MLX 서버 종료 중..."
pkill -f "vllm-mlx serve" 2>/dev/null && echo "   - 종료 완료" || echo "   - 실행 중이 아님"

# API 서버 종료
echo ""
echo "[2/2] API 서버 종료 중..."
pkill -f "uvicorn app.main:app" 2>/dev/null && echo "   - 종료 완료" || echo "   - 실행 중이 아님"

echo ""
echo "=========================================="
echo "  모든 서버 종료 완료"
echo "=========================================="
