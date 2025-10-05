from loguru import logger
from sentence_transformers import SentenceTransformer

from utils.config import settings


class EmbeddingGenerator:
    """Generate embeddings for text"""

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for list of texts"""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for single text"""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
