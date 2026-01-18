"""이력서 PDF 업로드 및 분석 라우터"""
import fitz  # PyMuPDF
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/resume", tags=["resume"])


# 메모리에 이력서 저장 (세션별)
resume_storage: dict[str, dict] = {}


class ResumeResponse(BaseModel):
    """이력서 업로드 응답"""
    session_id: str
    filename: str
    text_length: int
    summary: str
    message: str


class ResumeInfo(BaseModel):
    """이력서 정보"""
    session_id: str
    filename: str
    text: str
    summary: str


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """PDF에서 텍스트 추출"""
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 파싱 실패: {str(e)}")
    return text.strip()


def summarize_resume(text: str) -> str:
    """이력서 요약 생성 (상세 추출 방식)"""
    lines = text.split('\n')

    # 주요 키워드 기반 추출 (더 상세한 키워드)
    keywords = {
        '인적사항': ['이름', '성명', 'Name', '연락처', '이메일', 'Email', '주소', '생년월일'],
        '경력사항': ['경력', '경험', 'Experience', '근무', '회사', '직장', '재직', '담당업무', '업무내용'],
        '학력사항': ['학력', '학교', '대학', 'Education', '졸업', '전공', '학위', '석사', '박사', '학사'],
        '기술스택': ['기술', '스킬', 'Skill', '언어', '프레임워크', 'Stack', 'Python', 'Java', 'React', 'Node', 'SQL', 'AWS'],
        '자격증': ['자격', '자격증', '수료', 'Certificate', '면허', '자격사항'],
        '프로젝트': ['프로젝트', 'Project', '포트폴리오', '개발', '구현', '참여'],
        '자기소개': ['자기소개', '소개', '지원동기', '포부', '목표', '강점', '약점', '성격']
    }

    current_section = None
    section_content = {}

    for line in lines:
        line = line.strip()
        if not line or len(line) < 2:
            continue

        # 섹션 감지
        for section, kws in keywords.items():
            if any(kw.lower() in line.lower() for kw in kws):
                current_section = section
                if section not in section_content:
                    section_content[section] = []
                break

        # 내용 수집 (의미있는 내용만)
        if current_section and len(line) > 3:
            # 중복 제거
            if line not in section_content[current_section]:
                section_content[current_section].append(line)

    # 요약 생성
    summary = []
    section_order = ['인적사항', '학력사항', '경력사항', '기술스택', '프로젝트', '자격증', '자기소개']

    for section in section_order:
        if section in section_content and section_content[section]:
            content = section_content[section]
            summary.append(f"\n## {section}")
            # 각 섹션에서 최대 5줄 추출
            for item in content[:5]:
                if len(item) > 150:
                    item = item[:150] + "..."
                summary.append(f"- {item}")

    if not summary:
        # 키워드가 없으면 전체 텍스트 요약
        clean_text = ' '.join(text.split())[:1000]
        return f"## 이력서 내용\n{clean_text}"

    return "\n".join(summary)


@router.post("/upload", response_model=ResumeResponse)
async def upload_resume(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """
    이력서 PDF 업로드

    - PDF 파일을 업로드하면 텍스트를 추출하고 요약합니다.
    - session_id를 반환하며, 이후 채팅에서 이 ID를 사용하여 이력서 정보를 참조합니다.
    """
    # 파일 검증
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # 파일 크기 제한 (10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다.")

    # PDF 텍스트 추출
    text = extract_text_from_pdf(contents)

    if not text:
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

    # 요약 생성
    summary = summarize_resume(text)

    # 세션 ID 생성 또는 사용
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    # 저장
    resume_storage[session_id] = {
        "filename": file.filename,
        "text": text,
        "summary": summary,
        "indexed": False,
        "chunk_count": 0
    }

    # RAG 인덱싱 (비동기로 처리하지 않고 바로 실행)
    try:
        chunk_count = index_resume_for_rag(session_id)
        print(f"  ✓ 이력서 RAG 인덱싱 완료: {session_id} ({chunk_count}개 청크)")
    except Exception as e:
        print(f"  ✗ 이력서 RAG 인덱싱 실패: {e}")

    return ResumeResponse(
        session_id=session_id,
        filename=file.filename,
        text_length=len(text),
        summary=summary,
        message="이력서가 성공적으로 업로드되었습니다. 면접 질문에 활용됩니다."
    )


@router.get("/{session_id}", response_model=ResumeInfo)
async def get_resume(session_id: str):
    """저장된 이력서 정보 조회"""
    if session_id not in resume_storage:
        raise HTTPException(status_code=404, detail="이력서를 찾을 수 없습니다.")

    data = resume_storage[session_id]
    return ResumeInfo(
        session_id=session_id,
        filename=data["filename"],
        text=data["text"],
        summary=data["summary"]
    )


@router.delete("/{session_id}")
async def delete_resume(session_id: str):
    """저장된 이력서 삭제"""
    if session_id not in resume_storage:
        raise HTTPException(status_code=404, detail="이력서를 찾을 수 없습니다.")

    # RAG 인덱스도 삭제
    delete_resume_index(session_id)

    del resume_storage[session_id]
    return {"message": "이력서가 삭제되었습니다."}


def get_resume_context(session_id: str) -> Optional[str]:
    """채팅에서 사용할 이력서 컨텍스트 반환"""
    if session_id not in resume_storage:
        return None

    data = resume_storage[session_id]
    return f"""
[첨부된 이력서 정보]
파일명: {data['filename']}

{data['summary']}

[전체 이력서 내용]
{data['text'][:3000]}
"""


# ==================== 이력서 RAG 기능 ====================

RESUME_COLLECTION_PREFIX = "resume_"


def index_resume_for_rag(session_id: str) -> int:
    """이력서를 ChromaDB에 인덱싱

    Args:
        session_id: 이력서 세션 ID

    Returns:
        인덱싱된 청크 수
    """
    if session_id not in resume_storage:
        return 0

    from app.rag.embeddings import get_embedding_client
    from app.rag.vector_store import get_vector_store

    data = resume_storage[session_id]
    text = data['text']

    # 텍스트를 청크로 분할 (500자 단위, 100자 오버랩)
    chunks = []
    chunk_size = 500
    overlap = 100

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())

    if not chunks:
        return 0

    collection_name = f"{RESUME_COLLECTION_PREFIX}{session_id}"

    embedding_client = get_embedding_client()
    vector_store = get_vector_store()

    # 기존 컬렉션 삭제
    try:
        vector_store.delete_collection(collection_name)
    except Exception:
        pass

    # 임베딩 생성
    embeddings = embedding_client.embed_documents(chunks)

    # 메타데이터 준비
    metadatas = [{"session_id": session_id, "chunk_index": i, "filename": data['filename']}
                 for i in range(len(chunks))]
    ids = [f"{session_id}_{i}" for i in range(len(chunks))]

    # ChromaDB에 저장
    vector_store.add_documents(
        collection_name=collection_name,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    # 인덱싱 완료 표시
    resume_storage[session_id]['indexed'] = True
    resume_storage[session_id]['chunk_count'] = len(chunks)

    return len(chunks)


def search_resume_context(session_id: str, query: str, top_k: int = 3) -> str:
    """사용자 질문과 관련된 이력서 내용 검색

    Args:
        session_id: 이력서 세션 ID
        query: 사용자 질문
        top_k: 반환할 청크 수

    Returns:
        관련 이력서 내용 텍스트
    """
    if session_id not in resume_storage:
        return ""

    from app.rag.embeddings import get_embedding_client
    from app.rag.vector_store import get_vector_store

    collection_name = f"{RESUME_COLLECTION_PREFIX}{session_id}"

    embedding_client = get_embedding_client()
    vector_store = get_vector_store()

    # 쿼리 임베딩
    query_embedding = embedding_client.embed_query(query)

    # 검색
    try:
        results = vector_store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k
        )
    except Exception:
        return ""

    # 결과 조합
    if not results or not results.get('documents') or not results['documents'][0]:
        return ""

    data = resume_storage[session_id]
    chunks = results['documents'][0]

    context = f"[이력서 참고 내용 - {data['filename']}]\n"
    for i, chunk in enumerate(chunks, 1):
        context += f"\n{chunk}\n"

    return context


def delete_resume_index(session_id: str) -> bool:
    """이력서 인덱스 삭제

    Args:
        session_id: 이력서 세션 ID

    Returns:
        성공 여부
    """
    from app.rag.vector_store import get_vector_store

    collection_name = f"{RESUME_COLLECTION_PREFIX}{session_id}"

    try:
        vector_store = get_vector_store()
        vector_store.delete_collection(collection_name)
        return True
    except Exception:
        return False
