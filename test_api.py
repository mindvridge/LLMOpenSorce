#!/usr/bin/env python3
"""API 테스트 스크립트"""
import time
from openai import OpenAI

print("🧪 LLM API 테스트 시작...")
print("=" * 60)

# 클라이언트 초기화
print("\n1️⃣ 클라이언트 초기화 중...")
client = OpenAI(
    api_key="sk-yMF-iN0klm6zt0E1D9nrYFejxozobeq-sNdNSqcU_hA",
    base_url="https://humanities-del-volunteer-manual.trycloudflare.com/v1",
    timeout=30.0
)
print("✅ 클라이언트 초기화 완료")

# 모델 목록 조회
print("\n2️⃣ 사용 가능한 모델 조회 중...")
try:
    models = client.models.list()
    print("✅ 모델 목록:")
    for model in models.data:
        print(f"   - {model.id}")
except Exception as e:
    print(f"❌ 에러: {e}")
    print("\n⏳ DNS 전파 대기 중... 30초 후 재시도합니다.")
    time.sleep(30)
    models = client.models.list()
    print("✅ 모델 목록:")
    for model in models.data:
        print(f"   - {model.id}")

# GPT-5.2 테스트
print("\n3️⃣ GPT-5.2 테스트 중...")
try:
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "user", "content": "안녕하세요! 간단하게 인사해주세요."}
        ],
        max_tokens=100
    )
    print("✅ GPT-5.2 응답:")
    print(f"   {response.choices[0].message.content}")
    print(f"\n📊 사용량:")
    print(f"   - Prompt 토큰: {response.usage.prompt_tokens}")
    print(f"   - Completion 토큰: {response.usage.completion_tokens}")
    print(f"   - 총 토큰: {response.usage.total_tokens}")
except Exception as e:
    print(f"❌ 에러: {e}")

# Qwen3 32B 테스트
print("\n4️⃣ Qwen3 32B 테스트 중...")
try:
    response = client.chat.completions.create(
        model="qwen3:32b",
        messages=[
            {"role": "user", "content": "1+1은?"}
        ],
        max_tokens=50
    )
    print("✅ Qwen3 32B 응답:")
    print(f"   {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ 에러: {e}")

# 스트리밍 테스트
print("\n5️⃣ 스트리밍 테스트 중...")
try:
    print("✅ GPT-5.2 스트리밍 응답:")
    print("   ", end="")
    stream = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "user", "content": "짧은 시를 하나 지어주세요."}
        ],
        stream=True,
        max_tokens=100
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
except Exception as e:
    print(f"❌ 에러: {e}")

print("\n" + "=" * 60)
print("🎉 테스트 완료!")
