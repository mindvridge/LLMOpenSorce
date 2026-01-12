#!/usr/bin/env python3
"""
OpenAI SDK를 사용한 API 테스트 클라이언트

사용법:
    python test_client.py
    python test_client.py --base-url http://localhost:8000
"""

import sys
import argparse
from openai import OpenAI


def test_non_streaming(client, model):
    """비스트리밍 테스트"""
    print("=" * 60)
    print("1. 비스트리밍 테스트")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "안녕하세요! 간단히 인사해주세요."}
        ],
        temperature=0.7,
    )

    print(f"ID: {response.id}")
    print(f"Model: {response.model}")
    print(f"Role: {response.choices[0].message.role}")
    print(f"Content: {response.choices[0].message.content}")
    print(f"Finish Reason: {response.choices[0].finish_reason}")
    print(f"\nUsage:")
    print(f"  Prompt tokens: {response.usage.prompt_tokens}")
    print(f"  Completion tokens: {response.usage.completion_tokens}")
    print(f"  Total tokens: {response.usage.total_tokens}")
    print()


def test_streaming(client, model):
    """스트리밍 테스트"""
    print("=" * 60)
    print("2. 스트리밍 테스트")
    print("=" * 60)

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "1부터 5까지 세어주세요."}
        ],
        temperature=0.7,
        stream=True,
    )

    print("Response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def test_models(client):
    """모델 목록 테스트"""
    print("=" * 60)
    print("3. 모델 목록 조회")
    print("=" * 60)

    models = client.models.list()
    print(f"사용 가능한 모델 수: {len(models.data)}\n")
    for model in models.data:
        print(f"  - {model.id}")
    print()


def main():
    parser = argparse.ArgumentParser(description="LLM API 클라이언트 테스트")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="API 베이스 URL (기본값: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="qwen3-ultra:latest",
        help="사용할 모델 (기본값: qwen3-ultra:latest)",
    )
    parser.add_argument(
        "--api-key",
        default="not-needed",
        help="API 키 (기본값: not-needed)",
    )

    args = parser.parse_args()

    print("\n=== LLM API 클라이언트 테스트 ===\n")
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print()

    # OpenAI 클라이언트 생성
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    try:
        # 테스트 실행
        test_models(client)
        test_non_streaming(client, args.model)
        test_streaming(client, args.model)

        print("=" * 60)
        print("✓ 모든 테스트 통과!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
