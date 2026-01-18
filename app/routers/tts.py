"""TTS (Text-to-Speech) API 라우터 - CosyVoice 3.0 기반"""
import os
import sys
import io
import tempfile
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# CosyVoice 경로 추가
COSYVOICE_PATH = '/Users/mindprep/CosyVoice'
sys.path.insert(0, COSYVOICE_PATH)
sys.path.insert(0, f'{COSYVOICE_PATH}/third_party/Matcha-TTS')

router = APIRouter(prefix="/tts", tags=["TTS"])

# 전역 CosyVoice 모델 (싱글톤)
_cosyvoice_model = None
_is_loading = False

# 참조 음성 설정
REFERENCE_AUDIO = f'{COSYVOICE_PATH}/reference_audio.wav'
PROMPT_TEXT = "안녕하세요! 면접에 참여해 주셔서 감사합니다. 먼저, 본인에 대해 간단히 소개해 주시겠어요?"


class TTSRequest(BaseModel):
    """TTS 요청 스키마"""
    text: str
    speed: float = 1.0


class TTSStatus(BaseModel):
    """TTS 상태 응답"""
    available: bool
    model_loaded: bool
    device: str
    sample_rate: Optional[int] = None
    reference_audio: Optional[str] = None


def get_cosyvoice_model():
    """CosyVoice 모델 싱글톤 반환"""
    global _cosyvoice_model, _is_loading

    if _cosyvoice_model is not None:
        return _cosyvoice_model

    if _is_loading:
        return None

    _is_loading = True

    try:
        import torch
        from cosyvoice.cli.cosyvoice import CosyVoice3

        # 디바이스 확인
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"🎤 CosyVoice 모델 로딩 중... (디바이스: {device})")

        # 모델 경로 (models 폴더 포함)
        model_path = '/Users/mindprep/.cache/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512'

        if os.path.exists(model_path):
            _cosyvoice_model = CosyVoice3(model_path)
        else:
            print("CosyVoice 모델 다운로드 중...")
            _cosyvoice_model = CosyVoice3('FunAudioLLM/Fun-CosyVoice3-0.5B-2512')

        print(f"✅ CosyVoice 모델 로딩 완료 (샘플레이트: {_cosyvoice_model.sample_rate})")
        return _cosyvoice_model

    except Exception as e:
        print(f"❌ CosyVoice 모델 로딩 실패: {e}")
        _is_loading = False
        return None

    finally:
        _is_loading = False


@router.get("/status", response_model=TTSStatus)
async def get_tts_status():
    """TTS 서비스 상태 확인"""
    import torch

    model = get_cosyvoice_model()

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if model is not None:
        return TTSStatus(
            available=True,
            model_loaded=True,
            device=device,
            sample_rate=model.sample_rate,
            reference_audio=REFERENCE_AUDIO if os.path.exists(REFERENCE_AUDIO) else None
        )
    else:
        return TTSStatus(
            available=False,
            model_loaded=False,
            device=device,
            sample_rate=None,
            reference_audio=None
        )


@router.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """텍스트를 음성으로 변환 (Zero-shot 음성 클로닝)"""
    import soundfile as sf

    model = get_cosyvoice_model()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="TTS 모델이 로딩 중이거나 사용할 수 없습니다."
        )

    if not os.path.exists(REFERENCE_AUDIO):
        raise HTTPException(
            status_code=500,
            detail=f"참조 음성 파일을 찾을 수 없습니다: {REFERENCE_AUDIO}"
        )

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="텍스트가 비어있습니다.")

    try:
        # 이전 테스트 방식 (한국어 작동 확인됨)
        # tts_text와 prompt_text 모두 <|endofprompt|> 토큰 앞에 붙임
        tts_text = f"<|endofprompt|>{text}"
        prompt_with_token = f"<|endofprompt|>{PROMPT_TEXT}"

        # 음성 합성
        audio_data = None
        for result in model.inference_zero_shot(
            tts_text=tts_text,
            prompt_text=prompt_with_token,
            prompt_wav=REFERENCE_AUDIO,
            stream=False,
            speed=request.speed,
            text_frontend=True  # 텍스트 정규화 활성화
        ):
            audio_data = result['tts_speech']
            break  # 첫 번째 결과만 사용

        if audio_data is None:
            raise HTTPException(status_code=500, detail="음성 합성에 실패했습니다.")

        # WAV 파일로 변환
        audio_np = audio_data.squeeze().cpu().numpy()

        # 메모리에서 WAV 생성
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, model.sample_rate, format='WAV')
        buffer.seek(0)

        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"음성 합성 오류: {str(e)}"
        )


@router.post("/warmup")
async def warmup_tts():
    """TTS 모델 웜업 (사전 로딩)"""
    model = get_cosyvoice_model()

    if model is not None:
        return {
            "status": "ok",
            "message": "TTS 모델 로딩 완료",
            "sample_rate": model.sample_rate
        }
    else:
        return {
            "status": "loading",
            "message": "TTS 모델 로딩 중..."
        }
