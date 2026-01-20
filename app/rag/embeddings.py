"""한국어 최적화 임베딩 클라이언트"""

import time
from typing import List, Optional, Tuple
from functools import lru_cache
from collections import OrderedDict


class EmbeddingClient:
    """SentenceTransformer 기반 임베딩 클라이언트 (캐싱 최적화)"""

    MODEL_NAME = "jhgan/ko-sroberta-multitask"
    DIMENSION = 768
    CACHE_SIZE = 1000  # 최대 캐시 항목 수
    CACHE_TTL = 3600   # 캐시 유효 시간 (1시간)

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.MODEL_NAME
        self._model = None
        # 쿼리 캐시: {query: (embedding_tuple, timestamp)}
        self._query_cache: OrderedDict = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def model(self):
        """모델 지연 로딩"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"🔄 임베딩 모델 로딩 중: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"✅ 임베딩 모델 로딩 완료")
        return self._model

    def _get_from_cache(self, query: str) -> Optional[List[float]]:
        """캐시에서 임베딩 조회"""
        if query in self._query_cache:
            embedding_tuple, timestamp = self._query_cache[query]
            # TTL 확인
            if time.time() - timestamp < self.CACHE_TTL:
                # LRU: 최근 사용으로 이동
                self._query_cache.move_to_end(query)
                self._cache_hits += 1
                return list(embedding_tuple)
            else:
                # 만료된 캐시 삭제
                del self._query_cache[query]
        return None

    def _add_to_cache(self, query: str, embedding: List[float]):
        """캐시에 임베딩 추가"""
        # 캐시 크기 제한
        while len(self._query_cache) >= self.CACHE_SIZE:
            self._query_cache.popitem(last=False)  # 가장 오래된 항목 삭제
        # 튜플로 저장 (불변성)
        self._query_cache[query] = (tuple(embedding), time.time())

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
        """쿼리 임베딩 (캐싱 적용)

        동일 쿼리 재요청 시 캐시에서 반환 (200-300ms 절약)

        Args:
            query: 임베딩할 쿼리 텍스트

        Returns:
            임베딩 벡터 (768차원)
        """
        # 캐시 확인
        cached = self._get_from_cache(query)
        if cached is not None:
            return cached

        # 캐시 미스: 실제 임베딩 계산
        self._cache_misses += 1
        embedding = self.model.encode(query, show_progress_bar=False)
        embedding_list = embedding.tolist()

        # 캐시에 저장
        self._add_to_cache(query, embedding_list)
        return embedding_list

    def get_cache_stats(self) -> dict:
        """캐시 통계 반환"""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_size": len(self._query_cache),
            "max_size": self.CACHE_SIZE,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }

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
