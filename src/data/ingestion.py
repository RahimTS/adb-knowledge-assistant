import json
from pathlib import Path

from loguru import logger

from data.chunking import TextChunker
from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store import VectorStore


class DataIngestionPipeline:
    """Process and ingest documents into vector store"""

    def __init__(self):
        self.chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()

    def ingest_json_file(self, file_path: str) -> dict:
        """Ingest JSON knowledge file"""
        logger.info(f"Ingesting file: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle both formats: wrapped with metadata or direct array
        if isinstance(data, list):
            # Direct array format
            knowledge_entries = data
            logger.info(f"Found {len(knowledge_entries)} entries (direct array format)")
        elif isinstance(data, dict) and "knowledge_entries" in data:
            # Wrapped format with metadata
            knowledge_entries = data.get("knowledge_entries", [])
            logger.info(f"Found {len(knowledge_entries)} knowledge entries")
        else:
            logger.error(f"Unknown JSON format in {file_path}")
            return {"inserted_count": 0, "error": "Unknown format"}

        # Process each entry
        all_chunks = []
        for entry in knowledge_entries:
            chunks = self.chunker.chunk_json_knowledge(entry)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks")

        # Generate embeddings
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)

        # Prepare documents for insertion
        documents = []
        for chunk, embedding in zip(all_chunks, embeddings, strict=False):
            doc = {"content": chunk["text"], "metadata": chunk["metadata"], "embedding": embedding}
            documents.append(doc)

        # Insert into vector store
        result = self.vector_store.insert_documents(documents)

        logger.success(f"Ingested {result['inserted_count']} documents from {Path(file_path).name}")
        return result

    def ingest_directory(self, directory_path: str) -> dict:
        """Ingest all JSON files from directory"""
        directory = Path(directory_path)
        json_files = list(directory.glob("**/*.json"))

        logger.info(f"Found {len(json_files)} JSON files in {directory_path}")

        total_inserted = 0
        files_succeeded = 0

        for json_file in json_files:
            try:
                result = self.ingest_json_file(str(json_file))
                if result.get("inserted_count", 0) > 0:
                    total_inserted += result["inserted_count"]
                    files_succeeded += 1
            except Exception as e:
                logger.error(f"Error ingesting {json_file}: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        return {
            "total_inserted": total_inserted,
            "files_processed": len(json_files),
            "files_succeeded": files_succeeded,
        }
