"""Setup the complete RAG system"""

from loguru import logger

from data.ingestion import DataIngestionPipeline
from retrieval.vector_store import VectorStore


def main():
    logger.info("=" * 60)
    logger.info("ADB KNOWLEDGE ASSISTANT - SETUP")
    logger.info("=" * 60)

    # Step 1: Create vector index
    logger.info("\n1. Creating vector search index...")
    vector_store = VectorStore()
    vector_store.create_vector_index()

    # Step 2: Ingest knowledge base
    logger.info("\n2. Ingesting knowledge base...")
    pipeline = DataIngestionPipeline()

    # Ingest all data sources
    result = pipeline.ingest_directory("data/raw")

    logger.info("\n✓ Setup complete!")
    logger.info(f"  Total documents: {result['total_inserted']}")
    logger.info(f"  Files processed: {result['files_processed']}")

    logger.info("\n" + "=" * 60)
    logger.info("READY TO START!")
    logger.info("Run: python src/main.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
