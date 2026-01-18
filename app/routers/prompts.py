"""프롬프트 관리 라우터 - 20년차 전문 면접관 전용"""
import os
import csv
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime


router = APIRouter(prefix="/prompts", tags=["prompts"])


# 프롬프트 파일 경로
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts")
EXPERT_INTERVIEW_FILE = os.path.join(PROMPTS_DIR, "01_면접관 프롬프트.csv")
COMPANY_FILE = os.path.join(PROMPTS_DIR, "02_(일반 기업) 회사명, 직무.csv")
HOSPITAL_FILE = os.path.join(PROMPTS_DIR, "03_(병원) 병원명, 직무의 사본.csv")

# 고정 프롬프트 ID
EXPERT_PROMPT_ID = "expert_interviewer"


class PromptResponse(BaseModel):
    """프롬프트 응답"""
    id: str
    name: str
    content: str
    category: str
    description: Optional[str]
    updated_at: str


class PromptUpdate(BaseModel):
    """프롬프트 수정 요청"""
    content: str


class PromptListResponse(BaseModel):
    """프롬프트 목록 응답"""
    prompts: list[PromptResponse]
    total: int


def load_expert_prompt() -> str:
    """전문 면접관 프롬프트 CSV 파일에서 읽기"""
    if os.path.exists(EXPERT_INTERVIEW_FILE):
        with open(EXPERT_INTERVIEW_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # 첫 행은 헤더, 두 번째 행의 두 번째 열이 프롬프트 내용
            if len(rows) > 1 and len(rows[1]) > 1:
                return rows[1][1]
    return ""


def save_expert_prompt(content: str):
    """전문 면접관 프롬프트 CSV 파일에 저장"""
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    with open(EXPERT_INTERVIEW_FILE, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Unnamed: 0", "Unnamed: 1"])  # 헤더
        writer.writerow(["", content])  # 프롬프트 내용


def get_file_updated_time() -> str:
    """파일 수정 시간 반환"""
    if os.path.exists(EXPERT_INTERVIEW_FILE):
        mtime = os.path.getmtime(EXPERT_INTERVIEW_FILE)
        return datetime.fromtimestamp(mtime).isoformat()
    return datetime.now().isoformat()


def get_expert_prompt_response() -> PromptResponse:
    """전문 면접관 프롬프트 응답 생성"""
    content = load_expert_prompt()
    return PromptResponse(
        id=EXPERT_PROMPT_ID,
        name="20년차 전문 면접관",
        content=content,
        category="면접",
        description="10개 질문 + 2회 후속질문 구조의 전문 면접 프롬프트",
        updated_at=get_file_updated_time()
    )


@router.get("", response_model=PromptListResponse)
async def list_prompts():
    """프롬프트 목록 조회 (20년차 전문 면접관만 반환)"""
    prompt = get_expert_prompt_response()
    return PromptListResponse(prompts=[prompt], total=1)


@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: str):
    """프롬프트 조회"""
    if prompt_id != EXPERT_PROMPT_ID:
        raise HTTPException(status_code=404, detail="프롬프트를 찾을 수 없습니다.")

    return get_expert_prompt_response()


@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: str, request: PromptUpdate):
    """프롬프트 수정 (파일에 직접 저장)"""
    if prompt_id != EXPERT_PROMPT_ID:
        raise HTTPException(status_code=404, detail="프롬프트를 찾을 수 없습니다.")

    # 파일에 저장
    save_expert_prompt(request.content)

    return get_expert_prompt_response()


@router.post("", response_model=dict)
async def create_prompt():
    """프롬프트 생성 (비활성화됨)"""
    raise HTTPException(
        status_code=403,
        detail="새 프롬프트를 추가할 수 없습니다. 20년차 전문 면접관 프롬프트만 수정 가능합니다."
    )


@router.delete("/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """프롬프트 삭제 (비활성화됨)"""
    raise HTTPException(
        status_code=403,
        detail="프롬프트를 삭제할 수 없습니다."
    )


# ==================== 파일 직접 접근 API ====================

@router.get("/file/content")
async def get_prompt_file_content():
    """프롬프트 파일 내용 직접 조회"""
    content = load_expert_prompt()
    if not content:
        raise HTTPException(status_code=404, detail="프롬프트 파일을 찾을 수 없습니다.")
    return {
        "filename": "01_면접관 프롬프트.csv",
        "content": content,
        "updated_at": get_file_updated_time()
    }


@router.put("/file/content")
async def update_prompt_file_content(request: PromptUpdate):
    """프롬프트 파일 내용 직접 수정"""
    save_expert_prompt(request.content)
    return {
        "message": "프롬프트가 저장되었습니다.",
        "filename": "01_면접관 프롬프트.csv",
        "updated_at": get_file_updated_time()
    }


# ==================== 기업/병원 데이터 API ====================

def load_organization_data(file_path: str) -> dict:
    """CSV 파일에서 기업/병원 데이터 로드"""
    organizations = []
    jobs = []
    org_job_map = {}  # 기업별 직무 매핑

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)

            # 헤더 스킵하고 데이터 행 처리 (2행부터 시작, 첫 행은 컬럼명)
            for i, row in enumerate(rows):
                if i < 2:  # 0: 헤더, 1: 컬럼명
                    continue
                if len(row) >= 1 and row[0] and row[0] != "직접 입력":
                    org_name = row[0].strip()
                    if org_name and org_name not in organizations:
                        organizations.append(org_name)
                    # 해당 기업의 직무 저장
                    if len(row) >= 2 and row[1]:
                        job_name = row[1].strip()
                        if job_name and job_name != "직접 입력":
                            if org_name not in org_job_map:
                                org_job_map[org_name] = []
                            if job_name not in org_job_map[org_name]:
                                org_job_map[org_name].append(job_name)

                # 직무 목록 수집 (두 번째 열)
                if len(row) >= 2 and row[1]:
                    job_name = row[1].strip()
                    if job_name and job_name != "직접 입력" and job_name not in jobs:
                        jobs.append(job_name)

    # "직접 입력" 옵션 추가
    organizations.append("직접 입력")
    jobs.append("직접 입력")

    return {
        "organizations": organizations,
        "jobs": jobs,
        "org_job_map": org_job_map
    }


@router.get("/organizations/company")
async def get_company_data():
    """일반 기업 목록 및 직무 데이터 조회"""
    data = load_organization_data(COMPANY_FILE)
    return {
        "type": "company",
        "organizations": data["organizations"],
        "jobs": data["jobs"],
        "org_job_map": data["org_job_map"]
    }


@router.get("/organizations/hospital")
async def get_hospital_data():
    """병원 목록 및 직무 데이터 조회"""
    data = load_organization_data(HOSPITAL_FILE)
    return {
        "type": "hospital",
        "organizations": data["organizations"],
        "jobs": data["jobs"],
        "org_job_map": data["org_job_map"]
    }


# ==================== 질문셋 API ====================

@router.get("/question-sets")
async def list_question_sets():
    """사용 가능한 질문셋 목록 조회"""
    from app.question_sets import list_available_question_sets, get_job_mapping
    return {
        "question_sets": list_available_question_sets(),
        "job_mapping": get_job_mapping()
    }


@router.get("/question-sets/{org_type}/{job_name}")
async def get_question_set(org_type: str, job_name: str):
    """특정 유형/직무의 질문셋 조회

    Args:
        org_type: "병원" 또는 "일반기업"
        job_name: "간호사", "마케팅영업" 등
    """
    from app.question_sets import get_question_set as get_qs, get_question_set_as_text

    questions = get_qs(org_type, job_name)
    if not questions:
        raise HTTPException(status_code=404, detail=f"질문셋을 찾을 수 없습니다: {org_type}_{job_name}")

    return {
        "org_type": org_type,
        "job_name": job_name,
        "count": len(questions),
        "questions": questions,
        "text": get_question_set_as_text(org_type, job_name)
    }
