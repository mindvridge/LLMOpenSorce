"""한국어 최적화 임베딩 클라이언트"""

from typing import List, Optional
from functools import lru_cache


class EmbeddingClient:
    """SentenceTransformer 기반 임베딩 클라이언트"""

    MODEL_NAME = "jhgan/ko-sroberta-multitask"
    DIMENSION = 768

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.MODEL_NAME
        self._model = None

    @property
    def model(self):
        """모델 지연 로딩"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"🔄 임베딩 모델 로딩 중: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"✅ 임베딩 모델 로딩 완료")
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트 (각 벡터는 768차원)
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """쿼리 임베딩

        Args:
            query: 임베딩할 쿼리 텍스트

        Returns:
            임베딩 벡터 (768차원)
        """
        embedding = self.model.encode(query, show_progress_bar=False)
        return embedding.tolist()

    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.DIMENSION


# 싱글톤 인스턴스
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """임베딩 클라이언트 싱글톤 반환"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
