"""
Configuration settings for AI Shine Tutor.
Centralized configuration for easy tuning and deployment.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings with environment variable overrides."""

    # ============================================================
    # RAG ENGINE SETTINGS
    # ============================================================

    # Multi-turn retrieval: When True, retrieves context considering conversation history
    # When False, only retrieves based on current query (faster, less context-aware)
    MULTI_TURN_RETRIEVAL_ENABLED: bool = os.getenv("MULTI_TURN_RETRIEVAL", "false").lower() == "true"

    # Keyword Augmented Retrieval: When True, supplements vector search with keyword matching
    # on topic/keywords fields. Set to "off" to use pure vector search only.
    KEYWORD_AUGMENTED_RETRIEVAL: bool = os.getenv("KEYWORD_AUGMENTED_RETRIEVAL", "on").lower() == "on"

    # Similarity thresholds for vector search (used for logging only - all top-K results are always returned)
    PRIMARY_SIMILARITY_THRESHOLD: float = float(os.getenv("PRIMARY_THRESHOLD", "0.50"))
    FALLBACK_SIMILARITY_THRESHOLD: float = float(os.getenv("FALLBACK_THRESHOLD", "0.35"))

    # Low confidence threshold (triggers web search fallback if enabled)
    LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.3"))

    # Number of documents to retrieve
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    # ============================================================
    # MEMORY SETTINGS
    # ============================================================

    # Maximum messages to load from conversation history
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

    # Sliding window size for long conversations (prevents context explosion)
    SLIDING_WINDOW_SIZE: int = int(os.getenv("SLIDING_WINDOW_SIZE", "10"))

    # Maximum active conversations to track in memory
    MAX_ACTIVE_CONVERSATIONS: int = int(os.getenv("MAX_ACTIVE_CONVERSATIONS", "100"))

    # Conversation TTL in seconds (1 hour default)
    CONVERSATION_TTL_SECONDS: int = int(os.getenv("CONVERSATION_TTL", "3600"))

    # ============================================================
    # RATE LIMITING
    # ============================================================

    # Requests per minute per IP
    RATE_LIMIT_PER_MINUTE: str = os.getenv("RATE_LIMIT", "60/minute")

    # ============================================================
    # INPUT VALIDATION
    # ============================================================

    # Maximum query length in characters
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "2000"))

    # Minimum query length
    MIN_QUERY_LENGTH: int = int(os.getenv("MIN_QUERY_LENGTH", "2"))

    # ============================================================
    # LLM SETTINGS
    # ============================================================

    # Gemini model configuration
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))

    # Extended thinking mode (uses more tokens but better reasoning)
    EXTENDED_THINKING_ENABLED: bool = os.getenv("EXTENDED_THINKING", "false").lower() == "true"

    # ============================================================
    # RESPONSE LENGTH SETTINGS
    # ============================================================

    # Minimum words for descriptive responses (e.g., "tell me more")
    # This ensures continuation responses are substantive (2-3 paragraphs)
    DESCRIPTIVE_RESPONSE_MIN_WORDS: int = int(os.getenv("DESCRIPTIVE_RESPONSE_MIN_WORDS", "150"))

    # Maximum words before showing "Type 'continue' for more..."
    RESPONSE_TRUNCATE_THRESHOLD: int = int(os.getenv("RESPONSE_TRUNCATE_THRESHOLD", "400"))

    # Standard response target length (for regular queries)
    STANDARD_RESPONSE_TARGET_WORDS: int = int(os.getenv("STANDARD_RESPONSE_TARGET_WORDS", "100"))

    # ============================================================
    # WEB SEARCH FALLBACK (Optional)
    # ============================================================

    # Enable web search when retrieval confidence is low
    WEB_SEARCH_FALLBACK_ENABLED: bool = os.getenv("WEB_SEARCH_FALLBACK", "false").lower() == "true"

    # Google Custom Search API (if using)
    GOOGLE_SEARCH_API_KEY: str = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    GOOGLE_SEARCH_CX: str = os.getenv("GOOGLE_SEARCH_CX", "")

    # ============================================================
    # MONGODB SETTINGS
    # ============================================================

    MONGO_DB_URI: str = os.getenv("MONGO_DB_URI", "")
    DB_NAME: str = os.getenv("DB_NAME", "aishine")
    VECTOR_COLLECTION: str = os.getenv("VECTOR_COLLECTION", "module_vectors")
    CONVERSATIONS_COLLECTION: str = os.getenv("CONVERSATIONS_COLLECTION", "conversations")

    # Connection pool settings
    MONGO_MAX_POOL_SIZE: int = int(os.getenv("MONGO_MAX_POOL_SIZE", "10"))
    MONGO_MIN_POOL_SIZE: int = int(os.getenv("MONGO_MIN_POOL_SIZE", "2"))

    # ============================================================
    # AWS BEDROCK SETTINGS
    # ============================================================

    AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
    EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

    # ============================================================
    # CHUNKING SETTINGS (for document ingestion)
    # ============================================================

    # Semantic chunking parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))  # tokens
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))  # tokens
    MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))  # tokens


# Global settings instance
settings = Settings()
