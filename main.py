"""
FastAPI Backend for AI Shine Tutor RAG Chatbot
Optimized for low latency with single-chain architecture.

Features:
- Token streaming
- Conversation history persistence
- Rate limiting
- Input validation
- Graceful degradation
- Hybrid retrieval (vector + keyword)
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from Backend.rag_engine import RAGEngine
from Backend.mongodb_client import MongoDBClient
from Backend.config import settings
from Backend.metrics import MetricsCollector
from Backend.routers import chat_router, conversations_router, admin_router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global instances (accessed by routers)
rag_engine: Optional[RAGEngine] = None
mongo_client: Optional[MongoDBClient] = None
metrics: Optional[MetricsCollector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with graceful degradation."""
    global rag_engine, mongo_client, metrics

    logger.info("[STARTUP] Initializing services...")

    # Initialize metrics collector first (never fails)
    metrics = MetricsCollector()
    logger.info("[STARTUP] Metrics collector ready")

    # Initialize MongoDB with graceful degradation
    try:
        mongo_client = MongoDBClient()
        logger.info("[STARTUP] MongoDB client ready")
    except Exception as e:
        logger.error(f"[STARTUP] MongoDB initialization failed: {e}")
        logger.warning("[STARTUP] Running in degraded mode - no persistence")
        mongo_client = None

    # Initialize RAG Engine
    try:
        rag_engine = RAGEngine(
            multi_turn_retrieval=settings.MULTI_TURN_RETRIEVAL_ENABLED
        )
        logger.info("[STARTUP] RAG Engine ready")
        logger.info(f"[STARTUP] Multi-turn retrieval: {settings.MULTI_TURN_RETRIEVAL_ENABLED}")
    except Exception as e:
        logger.error(f"[STARTUP] RAG Engine initialization failed: {e}")
        raise

    logger.info("[STARTUP] All services initialized")

    yield

    logger.info("[SHUTDOWN] Closing connections...")
    if rag_engine:
        rag_engine.cleanup()
    if mongo_client:
        mongo_client.close()
    logger.info("[SHUTDOWN] Shutdown complete")


app = FastAPI(
    title="AI Shine Tutor API",
    description="Optimized RAG-powered AI/ML tutor with low latency and hybrid retrieval",
    version="5.0.0",
    lifespan=lifespan
)

# CORS Configuration
# TODO: Replace with production origins before deployment
CORS_ORIGINS = [
    "*"  # Development - replace with specific origins for production
    # "https://yourdomain.com",
    # "https://app.yourdomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limiting middleware
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    RATE_LIMITING_ENABLED = True
    logger.info("[STARTUP] Rate limiting enabled")
except ImportError:
    RATE_LIMITING_ENABLED = False
    limiter = None
    logger.warning("[STARTUP] slowapi not installed - rate limiting disabled")


# Include routers
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(admin_router)


@app.get("/")
async def root():
    """API info endpoint."""
    return {
        "status": "online",
        "service": "AI Shine Tutor API",
        "version": "5.0.0",
        "features": [
            "token_streaming",
            "conversation_history",
            "rate_limiting",
            "graceful_degradation",
            "multi_turn_retrieval",
            "hybrid_search"  # NEW: Vector + Keyword search
        ],
        "settings": {
            "multi_turn_retrieval": settings.MULTI_TURN_RETRIEVAL_ENABLED,
            "rate_limit": settings.RATE_LIMIT_PER_MINUTE,
            "max_query_length": settings.MAX_QUERY_LENGTH
        }
    }


@app.get("/health")
async def health_check():
    """Health check with component status."""
    return {
        "api": "healthy",
        "rag_engine": "healthy" if rag_engine else "unavailable",
        "mongodb": "healthy" if mongo_client else "degraded",
        "metrics": "healthy" if metrics else "unavailable",
        "components": {
            "llm": "ready" if rag_engine else "unavailable",
            "retriever": "ready" if rag_engine else "unavailable",
            "memory": "ready" if rag_engine else "unavailable",
            "conversations": "ready" if mongo_client else "degraded"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        workers=1
    )
