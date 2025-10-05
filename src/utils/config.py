import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # App Settings
    app_name: str = os.getenv("APP_NAME", "adb-knowledge-assistant")
    debug_mode: bool = os.getenv("DEBUG_MODE", "true").lower() == "true"
    env: str = os.getenv("ENV", "local")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # API Keys
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    # MongoDB Configuration
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "adb_knowledge_db")

    # Collection Names
    documents_collection: str = os.getenv("DOCUMENTS_COLLECTION", "documents")
    embeddings_collection: str = os.getenv("EMBEDDINGS_COLLECTION", "embeddings")
    conversations_collection: str = os.getenv("CONVERSATIONS_COLLECTION", "conversations")

    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "logs/adb_assistant.log")

    # Retrieval Settings
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    enable_hybrid_search: bool = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"

    # Agent Settings
    max_agent_iterations: int = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
    enable_code_generation: bool = os.getenv("ENABLE_CODE_GENERATION", "true").lower() == "true"

    # Model Settings - OpenRouter models
    llm_model: str = os.getenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4000"))

    # Chunking Settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Vector Search Settings
    vector_index_name: str = os.getenv("VECTOR_INDEX_NAME", "vector_index")
    vector_dimensions: int = int(os.getenv("VECTOR_DIMENSIONS", "384"))

    def validate(self):
        """Validate critical settings"""
        errors = []

        # Check OpenRouter API key
        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is not set")

        # Check MongoDB URI
        if not self.mongodb_uri:
            errors.append("MONGODB_URI is not set")

        # Validate port range
        if not (1 <= self.port <= 65535):
            errors.append(f"Invalid PORT: {self.port} (must be 1-65535)")

        # Validate positive integers
        if self.top_k_results <= 0:
            errors.append(f"TOP_K_RESULTS must be positive: {self.top_k_results}")

        if self.chunk_size <= 0:
            errors.append(f"CHUNK_SIZE must be positive: {self.chunk_size}")

        if self.chunk_overlap < 0:
            errors.append(f"CHUNK_OVERLAP cannot be negative: {self.chunk_overlap}")

        if self.chunk_overlap >= self.chunk_size:
            errors.append(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than CHUNK_SIZE ({self.chunk_size})"
            )

        # Validate vector dimensions
        if self.vector_dimensions not in [384, 768, 1536]:
            errors.append(
                f"VECTOR_DIMENSIONS should be 384, 768, or 1536. Got: {self.vector_dimensions}"
            )

        return errors

    def print_settings(self):
        """Print current settings (for debugging)"""
        print("=" * 60)
        print("CURRENT SETTINGS")
        print("=" * 60)
        print(f"App Name: {self.app_name}")
        print(f"Environment: {self.env}")
        print(f"Debug Mode: {self.debug_mode}")
        print(f"Host: {self.host}:{self.port}")
        print("\nMongoDB:")
        print(f"  URI: {self.mongodb_uri}")
        print(f"  Database: {self.mongodb_database}")
        print("\nLLM:")
        print(f"  Model: {self.llm_model}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max Tokens: {self.max_tokens}")
        print("\nEmbeddings:")
        print(f"  Model: {self.embedding_model}")
        print(f"  Dimensions: {self.vector_dimensions}")
        print("\nRetrieval:")
        print(f"  Top K: {self.top_k_results}")
        print(f"  Hybrid Search: {self.enable_hybrid_search}")
        print(f"  Similarity Threshold: {self.similarity_threshold}")
        print("\nChunking:")
        print(f"  Chunk Size: {self.chunk_size}")
        print(f"  Chunk Overlap: {self.chunk_overlap}")
        print("\nLogging:")
        print(f"  Level: {self.log_level}")
        print(f"  File: {self.log_file}")
        print("=" * 60)


# Global settings instance
settings = Settings()

# Validate on import
validation_errors = settings.validate()
if validation_errors:
    print("\n❌ Configuration Errors:")
    for error in validation_errors:
        print(f"  - {error}")
    print("\nPlease check your .env file and fix the errors above.")
    if not settings.openrouter_api_key:
        print("\n⚠️  Critical: OPENROUTER_API_KEY is missing!")
    sys.exit(1)
