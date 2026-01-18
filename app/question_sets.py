"""질문셋 로더 - 서버 시작 시 CSV 파일을 메모리에 로드"""
import os
import csv
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional


# 질문셋 저장소 (메모리)
_question_sets: Dict[str, List[dict]] = {}

# 프롬프트 디렉토리
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def load_all_question_sets():
    """모든 질문셋 CSV 파일을 메모리에 로드"""
    global _question_sets
    _question_sets = {}

    # 질문셋 파일 찾기 (pathlib + unicode 정규화 - macOS 한글 파일명 지원)
    prompts_path = Path(PROMPTS_DIR)
    suffix = '_질문셋.csv'
    files = [f for f in prompts_path.iterdir()
             if unicodedata.normalize('NFC', f.name).endswith(suffix)]

    for file_path in files:
        try:
            key, questions = _parse_question_set_file(file_path)
            if key and questions:
                _question_sets[key] = questions
                print(f"  ✓ 질문셋 로드: {key} ({len(questions)}개 질문)")
        except Exception as e:
            print(f"  ✗ 질문셋 로드 실패: {file_path} - {e}")

    print(f"총 {len(_question_sets)}개 질문셋 로드 완료")
    return _question_sets


def _parse_question_set_file(file_path) -> tuple:
    """질문셋 CSV 파일 파싱

    파일명 예시: 04_(병원)간호사_질문셋.csv -> key: "병원_간호사"
    """
    # Path 객체 또는 문자열 처리
    if isinstance(file_path, Path):
        filename = file_path.name
    else:
        filename = os.path.basename(file_path)

    # macOS 한글 파일명 정규화 (NFD -> NFC)
    filename = unicodedata.normalize('NFC', filename)

    # 파일명에서 키 추출: XX_(유형)직무_질문셋.csv
    # 예: 04_(병원)간호사_질문셋.csv -> 병원_간호사
    import re
    match = re.search(r'\(([^)]+)\)([^_]+)_질문셋\.csv$', filename)
    if not match:
        return None, []

    org_type = match.group(1)  # 병원 또는 일반기업
    job_name = match.group(2)  # 간호사, 마케팅영업 등

    key = f"{org_type}_{job_name}"

    # CSV 파싱
    questions = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question_text = row.get('질문', '').strip()
            if question_text:
                questions.append({
                    'category': row.get('질문 구성', '').strip(),
                    'number': row.get('번호', '').strip(),
                    'question': question_text,
                    'answer': row.get('모범 답변', '').strip()
                })

    return key, questions


def get_question_set(org_type: str, job_name: str) -> Optional[List[dict]]:
    """특정 유형/직무의 질문셋 조회

    Args:
        org_type: "병원" 또는 "일반기업"
        job_name: "간호사", "마케팅영업" 등
    """
    # 직접 매칭 시도
    key = f"{org_type}_{job_name}"
    if key in _question_sets:
        return _question_sets[key]

    # 부분 매칭 시도 (직무명이 포함된 키 찾기)
    for k, v in _question_sets.items():
        if job_name in k:
            return v

    return None


def get_question_set_as_text(org_type: str, job_name: str) -> str:
    """질문셋을 프롬프트용 텍스트로 변환"""
    questions = get_question_set(org_type, job_name)

    if not questions:
        return ""

    # 카테고리별로 그룹화
    categories = {}
    for q in questions:
        cat = q['category'] or '기타'
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(q['question'])

    # 텍스트 생성
    lines = []
    for cat, q_list in categories.items():
        lines.append(f"\n## {cat}")
        for i, q in enumerate(q_list, 1):
            lines.append(f"{i}. {q}")

    return "\n".join(lines)


def list_available_question_sets() -> Dict[str, int]:
    """사용 가능한 질문셋 목록 반환"""
    return {key: len(questions) for key, questions in _question_sets.items()}


def get_job_mapping() -> Dict[str, str]:
    """UI 직무명 → 질문셋 키 매핑 반환

    UI에서 사용하는 직무명과 질문셋 파일의 직무명이 다를 수 있으므로 매핑 제공
    """
    # UI 직무명 → 질문셋 키
    mapping = {
        # 병원
        "간호사": "병원_간호사",
        "국제의료관광코디네이터": "병원_국제의료관광코디네이터",
        # 일반기업
        "마케팅/영업": "일반기업_마케팅영업",
        "개발/엔지니어링": "일반기업_개발엔지니어링",
        "고객서비스/CS": "일반기업_고객서비스CS",
        "인사/HR": "일반기업_인사HR",
        "운영/관리": "일반기업_운영관리",
        "기획/전략": "일반기업_기획전략",
        "재무/회계": "일반기업_재무회계",
        "품질관리/QA": "일반기업_품질관리QA",
        "글로벌 마케팅": "일반기업_글로벌 마케팅",
        "법무/컴플라이언스": "일반기업_법무컴플라이언스",
        "해외 영업": "일반기업_해외영업",
    }
    return mapping


# ==================== RAG 기반 질문셋 검색 ====================

# 질문셋 컬렉션 접두사
QUESTION_SET_COLLECTION_PREFIX = "question_set_"


def index_all_question_sets() -> Dict[str, int]:
    """모든 질문셋을 ChromaDB에 인덱싱

    Returns:
        Dict[str, int]: 컬렉션명 → 인덱싱된 질문 수
    """
    from app.rag.embeddings import get_embedding_client
    from app.rag.vector_store import get_vector_store

    embedding_client = get_embedding_client()
    vector_store = get_vector_store()

    indexed = {}

    for key, questions in _question_sets.items():
        if not questions:
            continue

        collection_name = f"{QUESTION_SET_COLLECTION_PREFIX}{key}"

        # 기존 컬렉션 삭제 후 재생성
        try:
            vector_store.delete_collection(collection_name)
        except Exception:
            pass

        # 질문 텍스트 및 메타데이터 준비
        documents = []
        metadatas = []
        ids = []

        for i, q in enumerate(questions):
            # 검색용 텍스트: 카테고리 + 질문 + 모범답변
            doc_text = f"[{q['category']}] {q['question']}"
            if q.get('answer'):
                doc_text += f"\n모범답변: {q['answer']}"

            documents.append(doc_text)
            metadatas.append({
                "category": q['category'],
                "number": q['number'],
                "question": q['question'],
                "answer": q.get('answer', ''),
                "key": key
            })
            ids.append(f"{key}_{i}")

        # 임베딩 생성
        embeddings = embedding_client.embed_documents(documents)

        # ChromaDB에 저장
        vector_store.add_documents(
            collection_name=collection_name,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        indexed[collection_name] = len(questions)
        print(f"  ✓ 질문셋 인덱싱: {key} ({len(questions)}개 질문)")

    print(f"총 {len(indexed)}개 질문셋 인덱싱 완료")
    return indexed


def search_relevant_questions(
    org_type: str,
    job_name: str,
    query: str,
    top_k: int = 5
) -> List[dict]:
    """사용자 메시지와 관련된 질문 검색

    Args:
        org_type: "병원" 또는 "일반기업"
        job_name: 직무명
        query: 사용자 메시지 (검색 쿼리)
        top_k: 반환할 질문 수

    Returns:
        관련 질문 리스트 (category, question, answer 포함)
    """
    from app.rag.embeddings import get_embedding_client
    from app.rag.vector_store import get_vector_store

    # 질문셋 키 찾기
    key = f"{org_type}_{job_name}"
    if key not in _question_sets:
        # 부분 매칭 시도
        for k in _question_sets.keys():
            if job_name in k:
                key = k
                break
        else:
            return []

    collection_name = f"{QUESTION_SET_COLLECTION_PREFIX}{key}"

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
        return []

    # 결과 포맷팅
    questions = []
    if results and results.get('metadatas') and results['metadatas'][0]:
        for metadata in results['metadatas'][0]:
            questions.append({
                'category': metadata.get('category', ''),
                'question': metadata.get('question', ''),
                'answer': metadata.get('answer', '')
            })

    return questions


def format_questions_for_prompt(questions: List[dict]) -> str:
    """검색된 질문을 프롬프트용 텍스트로 변환

    Args:
        questions: 질문 리스트

    Returns:
        프롬프트에 삽입할 텍스트
    """
    if not questions:
        return ""

    lines = ["[참고 질문]"]
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. [{q['category']}] {q['question']}")
        if q.get('answer'):
            lines.append(f"   → 모범답변: {q['answer'][:100]}...")

    return "\n".join(lines)
