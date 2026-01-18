"""RAG 검색기 및 컨텍스트 생성"""

from typing import List, Dict, Any, Optional
from .embeddings import EmbeddingClient, get_embedding_client
from .vector_store import VectorStore, get_vector_store


class RAGRetriever:
    """RAG 검색 및 컨텍스트 생성"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_client: Optional[EmbeddingClient] = None
    ):
        """검색기 초기화

        Args:
            vector_store: 벡터 저장소 (None이면 싱글톤 사용)
            embedding_client: 임베딩 클라이언트 (None이면 싱글톤 사용)
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedding_client = embedding_client or get_embedding_client()

    def retrieve(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """관련 문서 검색

        Args:
            query: 검색 쿼리
            collection_name: 컬렉션 이름
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩
        query_embedding = self.embedding_client.embed_query(query)

        # 벡터 검색
        results = self.vector_store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 결과 정리
        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                retrieved_docs.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "relevance_score": 1 - results["distances"][0][i] if results["distances"] else 1
                })

        return retrieved_docs

    def build_context(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 5,
        max_context_length: int = 4000
    ) -> str:
        """검색 결과로 LLM 컨텍스트 생성

        Args:
            query: 검색 쿼리
            collection_name: 컬렉션 이름
            top_k: 검색할 문서 수
            max_context_length: 최대 컨텍스트 길이 (문자 수)

        Returns:
            LLM에 전달할 컨텍스트 문자열
        """
        docs = self.retrieve(query, collection_name, top_k)

        if not docs:
            return ""

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs):
            content = doc["content"]
            source = doc["metadata"].get("source", "문서")
            score = doc["relevance_score"]

            # 컨텍스트 길이 제한 확인
            if current_length + len(content) > max_context_length:
                # 남은 공간만큼만 추가
                remaining = max_context_length - current_length
                if remaining > 100:  # 최소 100자 이상이면 추가
                    content = content[:remaining] + "..."
                else:
                    break

            context_parts.append(
                f"[참고 {i + 1}] (출처: {source}, 관련도: {score:.2f})\n{content}"
            )
            current_length += len(content)

        return "\n\n---\n\n".join(context_parts)

    def build_rag_system_prompt(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 5,
        base_prompt: str = ""
    ) -> str:
        """RAG 시스템 프롬프트 생성

        Args:
            query: 사용자 질문
            collection_name: 컬렉션 이름
            top_k: 검색할 문서 수
            base_prompt: 기본 시스템 프롬프트

        Returns:
            RAG 컨텍스트가 포함된 시스템 프롬프트
        """
        context = self.build_context(query, collection_name, top_k)

        if not context:
            return base_prompt

        rag_prompt = f"""다음 참고 문서 내용을 바탕으로 질문에 답변하세요.
답변할 때 관련 문서 내용을 인용하고, 문서에 없는 내용은 "참고 문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.

[참고 문서]
{context}

---

{base_prompt}""".strip()

        return rag_prompt


# 싱글톤 인스턴스
_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """검색기 싱글톤 반환"""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
