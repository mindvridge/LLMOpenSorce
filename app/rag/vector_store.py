"""ChromaDB 기반 벡터 저장소 (컬렉션 캐싱 최적화)"""

import os
import threading
from typing import List, Dict, Any, Optional


class VectorStore:
    """ChromaDB 벡터 저장소 관리 (컬렉션 캐싱)"""

    def __init__(self, persist_directory: Optional[str] = None):
        """벡터 저장소 초기화

        Args:
            persist_directory: ChromaDB 데이터 저장 경로
        """
        if persist_directory is None:
            # 프로젝트 루트의 data/chroma_db 사용
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            persist_directory = os.path.join(base_dir, "data", "chroma_db")

        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self._client = None
        self._lock = threading.Lock()
        # 컬렉션 캐시: {name: collection_object}
        self._collection_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def client(self):
        """ChromaDB 클라이언트 지연 로딩 (스레드 안전)"""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import chromadb
                    from chromadb.config import Settings

                    print(f"🔄 ChromaDB 초기화 중: {self.persist_directory}")
                    self._client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=Settings(anonymized_telemetry=False)
                    )
                    print(f"✅ ChromaDB 초기화 완료")
        return self._client

    def get_or_create_collection(self, name: str = "default"):
        """컬렉션 가져오기 또는 생성 (캐싱 적용)

        반복 조회 시 캐시에서 반환 (30-50ms 절약)

        Args:
            name: 컬렉션 이름

        Returns:
            ChromaDB Collection 객체
        """
        # 캐시 확인
        if name in self._collection_cache:
            self._cache_hits += 1
            return self._collection_cache[name]

        # 캐시 미스: 실제 조회
        self._cache_misses += 1
        collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
        )

        # 캐시에 저장
        self._collection_cache[name] = collection
        return collection

    def invalidate_collection_cache(self, name: str = None):
        """컬렉션 캐시 무효화

        Args:
            name: 무효화할 컬렉션 이름 (None이면 전체 무효화)
        """
        if name is None:
            self._collection_cache.clear()
        elif name in self._collection_cache:
            del self._collection_cache[name]

    def get_cache_stats(self) -> dict:
        """캐시 통계 반환"""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "cached_collections": len(self._collection_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """문서 및 임베딩 추가

        Args:
            collection_name: 컬렉션 이름
            documents: 문서 텍스트 리스트
            embeddings: 임베딩 벡터 리스트
            metadatas: 메타데이터 리스트
            ids: 문서 ID 리스트
        """
        collection = self.get_or_create_collection(collection_name)
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """유사 문서 검색

        Args:
            collection_name: 컬렉션 이름
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 결과 수

        Returns:
            검색 결과 (documents, metadatas, distances)
        """
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def delete_collection(self, name: str) -> bool:
        """컬렉션 삭제

        Args:
            name: 삭제할 컬렉션 이름

        Returns:
            성공 여부
        """
        try:
            self.client.delete_collection(name=name)
            # 캐시에서도 삭제
            self.invalidate_collection_cache(name)
            return True
        except Exception:
            return False

    def list_collections(self) -> List[str]:
        """모든 컬렉션 목록 반환"""
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def get_collection_count(self, name: str) -> int:
        """컬렉션 내 문서 수 반환"""
        try:
            collection = self.get_or_create_collection(name)
            return collection.count()
        except Exception:
            return 0

    def delete_documents_by_doc_id(self, collection_name: str, doc_id: int) -> bool:
        """특정 문서 ID의 청크 삭제

        Args:
            collection_name: 컬렉션 이름
            doc_id: 삭제할 문서 ID

        Returns:
            성공 여부
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            # 메타데이터에서 doc_id로 필터링하여 삭제
            collection.delete(
                where={"doc_id": doc_id}
            )
            return True
        except Exception:
            return False


# 싱글톤 인스턴스
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """벡터 저장소 싱글톤 반환"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
