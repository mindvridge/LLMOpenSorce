"""RAG (Retrieval-Augmented Generation) API 라우터"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models.database import RAGDocument
from app.rag.document_processor import DocumentProcessor
from app.rag.vector_store import get_vector_store
from app.rag.embeddings import get_embedding_client
from app.rag.retriever import get_retriever


router = APIRouter(prefix="/rag", tags=["RAG"])


# ==================== Pydantic 스키마 ====================

class RAGDocumentResponse(BaseModel):
    """문서 응답 스키마"""
    id: int
    filename: str
    file_hash: str
    collection_name: str
    chunk_count: int
    total_chars: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class RAGDocumentListResponse(BaseModel):
    """문서 목록 응답"""
    documents: List[RAGDocumentResponse]
    total: int


class RAGSearchRequest(BaseModel):
    """검색 요청"""
    query: str
    collection_name: Optional[str] = "default"
    top_k: Optional[int] = 5


class RAGSearchResult(BaseModel):
    """검색 결과 항목"""
    content: str
    source: str
    relevance_score: float
    chunk_index: int


class RAGSearchResponse(BaseModel):
    """검색 응답"""
    query: str
    results: List[RAGSearchResult]
    context: str


class RAGCollectionInfo(BaseModel):
    """컬렉션 정보"""
    name: str
    document_count: int


# ==================== API 엔드포인트 ====================

@router.post("/documents/upload", response_model=RAGDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="default"),
    db: Session = Depends(get_db)
):
    """PDF 문서 업로드 및 인덱싱

    Args:
        file: PDF 파일
        collection_name: 저장할 컬렉션 이름

    Returns:
        업로드된 문서 정보
    """
    # 파일 형식 검증
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    try:
        # 파일 읽기
        pdf_bytes = await file.read()

        # 문서 처리
        processor = DocumentProcessor()
        result = processor.process_pdf(pdf_bytes, file.filename)

        # 중복 확인
        existing = db.query(RAGDocument).filter(
            RAGDocument.file_hash == result["file_hash"],
            RAGDocument.collection_name == collection_name
        ).first()

        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"이미 등록된 문서입니다. (ID: {existing.id})"
            )

        # DB에 문서 정보 저장
        doc = RAGDocument(
            filename=result["filename"],
            file_hash=result["file_hash"],
            collection_name=collection_name,
            chunk_count=result["chunk_count"],
            total_chars=result["total_chars"],
            status="processing"
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # 임베딩 생성 및 벡터 저장소에 저장
        try:
            embedding_client = get_embedding_client()
            vector_store = get_vector_store()

            # 청크 임베딩
            embeddings = embedding_client.embed_documents(result["chunks"])

            # 메타데이터 및 ID 생성
            metadatas = processor.create_chunk_metadatas(
                doc.id, result["filename"], result["chunk_count"]
            )
            ids = processor.create_chunk_ids(doc.id, result["chunk_count"])

            # 벡터 저장소에 추가
            vector_store.add_documents(
                collection_name=collection_name,
                documents=result["chunks"],
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            # 상태 업데이트
            doc.status = "completed"
            db.commit()

        except Exception as e:
            doc.status = "failed"
            doc.error_message = str(e)
            db.commit()
            raise HTTPException(status_code=500, detail=f"임베딩 처리 실패: {str(e)}")

        return doc

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 처리 실패: {str(e)}")


@router.get("/documents", response_model=RAGDocumentListResponse)
async def list_documents(
    collection_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """등록된 문서 목록 조회

    Args:
        collection_name: 필터링할 컬렉션 이름 (없으면 전체)

    Returns:
        문서 목록
    """
    query = db.query(RAGDocument)

    if collection_name:
        query = query.filter(RAGDocument.collection_name == collection_name)

    documents = query.order_by(RAGDocument.created_at.desc()).all()

    return RAGDocumentListResponse(
        documents=documents,
        total=len(documents)
    )


@router.get("/documents/{doc_id}", response_model=RAGDocumentResponse)
async def get_document(doc_id: int, db: Session = Depends(get_db)):
    """특정 문서 정보 조회

    Args:
        doc_id: 문서 ID

    Returns:
        문서 정보
    """
    doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    return doc


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """문서 삭제

    Args:
        doc_id: 삭제할 문서 ID

    Returns:
        삭제 결과
    """
    doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    # 벡터 저장소에서 청크 삭제
    try:
        vector_store = get_vector_store()
        vector_store.delete_documents_by_doc_id(doc.collection_name, doc_id)
    except Exception as e:
        print(f"벡터 삭제 실패: {e}")

    # DB에서 삭제
    db.delete(doc)
    db.commit()

    return {"message": f"문서 '{doc.filename}' 삭제 완료", "deleted_id": doc_id}


@router.post("/search", response_model=RAGSearchResponse)
async def search_documents(request: RAGSearchRequest):
    """문서 검색

    Args:
        request: 검색 요청 (query, collection_name, top_k)

    Returns:
        검색 결과 및 컨텍스트
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="검색어를 입력해주세요.")

    retriever = get_retriever()

    # 검색 수행
    docs = retriever.retrieve(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k
    )

    # 컨텍스트 생성
    context = retriever.build_context(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k
    )

    # 결과 포맷팅
    results = [
        RAGSearchResult(
            content=doc["content"],
            source=doc["metadata"].get("source", "unknown"),
            relevance_score=doc["relevance_score"],
            chunk_index=doc["metadata"].get("chunk_index", 0)
        )
        for doc in docs
    ]

    return RAGSearchResponse(
        query=request.query,
        results=results,
        context=context
    )


@router.get("/collections", response_model=List[RAGCollectionInfo])
async def list_collections():
    """컬렉션 목록 조회

    Returns:
        컬렉션 이름 및 문서 수 목록
    """
    vector_store = get_vector_store()
    collection_names = vector_store.list_collections()

    collections = []
    for name in collection_names:
        count = vector_store.get_collection_count(name)
        collections.append(RAGCollectionInfo(name=name, document_count=count))

    return collections


@router.delete("/collections/{name}")
async def delete_collection(name: str, db: Session = Depends(get_db)):
    """컬렉션 삭제 (해당 컬렉션의 모든 문서 포함)

    Args:
        name: 삭제할 컬렉션 이름

    Returns:
        삭제 결과
    """
    # 벡터 저장소에서 컬렉션 삭제
    vector_store = get_vector_store()
    success = vector_store.delete_collection(name)

    if not success:
        raise HTTPException(status_code=404, detail="컬렉션을 찾을 수 없습니다.")

    # DB에서 해당 컬렉션의 문서 삭제
    deleted_count = db.query(RAGDocument).filter(
        RAGDocument.collection_name == name
    ).delete()
    db.commit()

    return {
        "message": f"컬렉션 '{name}' 삭제 완료",
        "deleted_documents": deleted_count
    }


@router.get("/status")
async def rag_status(db: Session = Depends(get_db)):
    """RAG 시스템 상태 확인

    Returns:
        RAG 시스템 상태 정보
    """
    vector_store = get_vector_store()

    # 통계 수집
    total_docs = db.query(RAGDocument).count()
    completed_docs = db.query(RAGDocument).filter(
        RAGDocument.status == "completed"
    ).count()
    collections = vector_store.list_collections()

    return {
        "status": "ok",
        "total_documents": total_docs,
        "completed_documents": completed_docs,
        "collections": len(collections),
        "embedding_model": "jhgan/ko-sroberta-multitask",
        "vector_store": "ChromaDB"
    }
