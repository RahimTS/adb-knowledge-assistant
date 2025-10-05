import numpy as np
from loguru import logger
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

from utils.config import settings


class VectorStore:
    """MongoDB vector store operations with fallback to local similarity search"""

    def __init__(self):
        self.client = MongoClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_database]
        self.collection = self.db[settings.documents_collection]
        self.use_atlas_search = False  # Flag to track if Atlas is available
        logger.info(f"Connected to MongoDB: {settings.mongodb_database}")

    def create_vector_index(self):
        """Create vector search index (only works with Atlas)"""
        logger.info("Attempting to create vector search index...")
        logger.warning("⚠️ Vector search requires MongoDB Atlas")
        logger.info("✓ Using fallback: Local cosine similarity search")

        # Create regular text index for keyword search
        try:
            self.collection.create_index([("content", "text")])
            logger.info("✓ Created text index for keyword search")
        except Exception as e:
            logger.debug(f"Text index note: {e}")

    def insert_documents(self, documents: list[dict]) -> dict:
        """Insert documents into collection"""
        if not documents:
            return {"inserted_count": 0}

        result = self.collection.insert_many(documents)
        logger.info(f"Inserted {len(result.inserted_ids)} documents")

        return {"inserted_count": len(result.inserted_ids)}

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = settings.top_k_results,
        filters: dict | None = None,
    ) -> list[dict]:
        """Perform vector similarity search using local computation"""

        logger.info(f"Performing local vector similarity search (top_k={top_k})")

        # Build query filter
        query_filter = {}
        if filters:
            for key, value in filters.items():
                query_filter[f"metadata.{key}"] = value

        # Retrieve all documents (with filter if provided)
        all_docs = list(
            self.collection.find(query_filter, {"content": 1, "metadata": 1, "embedding": 1})
        )

        if not all_docs:
            logger.warning("No documents found in collection")
            return []

        logger.info(f"Computing similarity for {len(all_docs)} documents")

        # Extract embeddings
        doc_embeddings = []
        valid_docs = []

        for doc in all_docs:
            if "embedding" in doc and doc["embedding"]:
                doc_embeddings.append(doc["embedding"])
                valid_docs.append(doc)

        if not doc_embeddings:
            logger.warning("No documents with embeddings found")
            return []

        # Convert to numpy arrays
        query_vec = np.array(query_embedding).reshape(1, -1)
        doc_vecs = np.array(doc_embeddings)

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, doc_vecs)[0]

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results with scores
        results = []
        for idx in top_k_indices:
            doc = valid_docs[idx]
            doc["score"] = float(similarities[idx])
            results.append(doc)

        logger.info(f"Found {len(results)} similar documents")
        return results

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Perform text search"""
        try:
            results = (
                self.collection.find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}, "content": 1, "metadata": 1},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(top_k)
            )

            return list(results)
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            # Fallback to regex search
            results = self.collection.find(
                {"content": {"$regex": query, "$options": "i"}}, {"content": 1, "metadata": 1}
            ).limit(top_k)

            return list(results)

    def clear_collection(self):
        """Clear all documents"""
        result = self.collection.delete_many({})
        logger.info(f"Deleted {result.deleted_count} documents")

    def get_stats(self) -> dict:
        """Get collection statistics"""
        count = self.collection.count_documents({})

        # Get type distribution
        pipeline = [{"$group": {"_id": "$metadata.type", "count": {"$sum": 1}}}]
        type_dist = list(self.collection.aggregate(pipeline))

        return {
            "total_documents": count,
            "type_distribution": {item["_id"]: item["count"] for item in type_dist},
        }
