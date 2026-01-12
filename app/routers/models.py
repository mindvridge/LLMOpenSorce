"""모델 관리 라우터"""
import time
from fastapi import APIRouter, HTTPException
from app.models.schemas import ModelsResponse, Model, ErrorResponse
from app.ollama_client import get_ollama_client
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
        ollama_client = get_ollama_client()

        # Ollama에서 설치된 모델 목록 가져오기
        installed_models = await ollama_client.list_models()
        installed_model_names = {m.get("name", "") for m in installed_models}

        # 설정 파일의 모델과 매칭
        models = []
        for model_config in config.available_models:
            # 설정된 모델이 실제로 설치되어 있는지 확인
            is_installed = model_config.name in installed_model_names

            # OpenAI 형식으로 변환
            model = Model(
                id=model_config.name,
                created=int(time.time()),
                owned_by="ollama",
            )
            models.append(model)

        # 추가로 설치된 모델도 포함 (설정에 없는 것)
        for installed_model in installed_models:
            model_name = installed_model.get("name", "")
            if model_name and model_name not in [m.name for m in config.available_models]:
                models.append(
                    Model(
                        id=model_name,
                        created=int(time.time()),
                        owned_by="ollama",
                    )
                )

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
        ollama_client = get_ollama_client()

        # Ollama에서 설치된 모델 확인
        installed_models = await ollama_client.list_models()
        installed_model_names = {m.get("name", "") for m in installed_models}

        if model_id not in installed_model_names:
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

        return Model(
            id=model_id,
            created=int(time.time()),
            owned_by="ollama",
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
