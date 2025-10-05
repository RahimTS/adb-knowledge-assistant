from loguru import logger

from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store import VectorStore
from utils.config import settings


class HybridRetriever:
    """Hybrid retrieval combining vector and keyword search"""

    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = settings.top_k_results,
        filters: dict | None = None,
        use_hybrid: bool = settings.enable_hybrid_search,
    ) -> list[dict]:
        """Retrieve relevant documents"""

        logger.info(f"Retrieving for query: {query[:100]}...")

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Vector search
        vector_results = self.vector_store.vector_search(
            query_embedding=query_embedding, top_k=top_k, filters=filters
        )

        if not use_hybrid:
            return vector_results

        # Keyword search
        try:
            keyword_results = self.vector_store.keyword_search(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            keyword_results = []

        # Combine and deduplicate results
        combined_results = self._merge_results(vector_results, keyword_results)

        logger.info(f"Retrieved {len(combined_results)} unique documents")
        return combined_results[:top_k]

    def _merge_results(self, vector_results: list[dict], keyword_results: list[dict]) -> list[dict]:
        """Merge and deduplicate results"""

        seen_ids = set()
        merged = []

        # Add vector results first (higher priority)
        for result in vector_results:
            doc_id = str(result.get("_id"))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(result)

        # Add keyword results
        for result in keyword_results:
            doc_id = str(result.get("_id"))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(result)

        return merged
