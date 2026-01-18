"""PDF 문서 처리 및 청킹"""

import hashlib
from typing import List, Dict, Any


class DocumentProcessor:
    """PDF 문서 처리기"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """문서 처리기 초기화

        Args:
            chunk_size: 청크 크기 (문자 수)
            chunk_overlap: 청크 간 겹침 (문자 수)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_splitter = None

    @property
    def text_splitter(self):
        """텍스트 분할기 지연 로딩"""
        if self._text_splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
        return self._text_splitter

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """PDF에서 텍스트 추출

        Args:
            pdf_bytes: PDF 파일 바이트

        Returns:
            추출된 텍스트
        """
        import fitz  # PyMuPDF

        text = ""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()

    def compute_file_hash(self, content: bytes) -> str:
        """파일 해시 계산 (중복 방지용)

        Args:
            content: 파일 바이트

        Returns:
            SHA256 해시 (16자)
        """
        return hashlib.sha256(content).hexdigest()[:16]

    def split_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할

        Args:
            text: 분할할 텍스트

        Returns:
            청크 리스트
        """
        if not text.strip():
            return []
        return self.text_splitter.split_text(text)

    def process_pdf(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """PDF 전체 처리 파이프라인

        Args:
            pdf_bytes: PDF 파일 바이트
            filename: 원본 파일명

        Returns:
            처리 결과 (file_hash, text, chunks 등)
        """
        file_hash = self.compute_file_hash(pdf_bytes)
        text = self.extract_text_from_pdf(pdf_bytes)
        chunks = self.split_text(text)

        return {
            "file_hash": file_hash,
            "filename": filename,
            "text": text,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "total_chars": len(text)
        }

    def create_chunk_metadatas(
        self,
        doc_id: int,
        filename: str,
        chunk_count: int
    ) -> List[Dict[str, Any]]:
        """청크별 메타데이터 생성

        Args:
            doc_id: 문서 DB ID
            filename: 파일명
            chunk_count: 청크 수

        Returns:
            메타데이터 리스트
        """
        return [
            {
                "doc_id": doc_id,
                "source": filename,
                "chunk_index": i
            }
            for i in range(chunk_count)
        ]

    def create_chunk_ids(self, doc_id: int, chunk_count: int) -> List[str]:
        """청크별 ID 생성

        Args:
            doc_id: 문서 DB ID
            chunk_count: 청크 수

        Returns:
            ID 리스트
        """
        return [f"doc_{doc_id}_chunk_{i}" for i in range(chunk_count)]
