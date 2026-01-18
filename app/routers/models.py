"""모델 관리 라우터"""
import time
from fastapi import APIRouter, HTTPException
from app.models.schemas import ModelsResponse, Model, ErrorResponse
from app.config import get_config

router = APIRouter()


@router.get(
    "/v1/models",
    response_model=ModelsResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_models():
    """
    사용 가능한 모델 목록 조회 (OpenAI 호환)

    설정 파일에 정의된 모델 목록을 반환합니다.
    """
    try:
        config = get_config()

        # 설정 파일의 모델 목록 반환
        models = []
        for model_config in config.available_models:
            # 프로바이더 결정
            provider = getattr(model_config, 'provider', 'vllm-mlx')

            # OpenAI 형식으로 변환
            model = Model(
                id=model_config.name,
                created=int(time.time()),
                owned_by=provider,
            )
            models.append(model)

        return ModelsResponse(data=models)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"모델 목록 조회 실패: {str(e)}",
                    "type": "internal_error",
                    "code": "internal_error",
                }
            },
        )


@router.get(
    "/v1/models/{model_id}",
    response_model=Model,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_model(model_id: str):
    """
    특정 모델 정보 조회 (OpenAI 호환)

    - **model_id**: 모델 ID
    """
    try:
        config = get_config()

        # 설정 파일에서 모델 검색
        for model_config in config.available_models:
            if model_config.name == model_id:
                provider = getattr(model_config, 'provider', 'vllm-mlx')
                return Model(
                    id=model_id,
                    created=int(time.time()),
                    owned_by=provider,
                )

        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"모델 '{model_id}'을(를) 찾을 수 없습니다.",
                    "type": "invalid_request_error",
                    "param": "model_id",
                    "code": "model_not_found",
                }
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"모델 조회 실패: {str(e)}",
                    "type": "internal_error",
                    "code": "internal_error",
                }
            },
        )
