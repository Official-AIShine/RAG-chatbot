"""
Admin and dashboard router.
Handles metrics, health checks, and configuration endpoints.
"""
import logging
import time
from fastapi import APIRouter, HTTPException

from Backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/metrics")
async def get_metrics():
    """Get system metrics for dashboard."""
    from main import metrics

    if not metrics:
        return {"error": "Metrics not available"}

    return metrics.get_summary()


@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check for monitoring."""
    from main import rag_engine, mongo_client, metrics

    health = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {}
    }

    # Check RAG engine
    if rag_engine:
        health["components"]["rag_engine"] = {
            "status": "healthy",
            "multi_turn_retrieval": settings.MULTI_TURN_RETRIEVAL_ENABLED
        }
    else:
        health["components"]["rag_engine"] = {"status": "unavailable"}
        health["status"] = "degraded"

    # Check MongoDB
    if mongo_client:
        try:
            mongo_client.ensure_connection()
            health["components"]["mongodb"] = {"status": "healthy"}
        except Exception as e:
            health["components"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"
    else:
        health["components"]["mongodb"] = {"status": "unavailable"}

    # Add metrics summary
    if metrics:
        health["metrics"] = metrics.get_summary()

    return health


@router.post("/config/multi-turn")
async def toggle_multi_turn(enabled: bool):
    """Toggle multi-turn retrieval at runtime."""
    from main import rag_engine

    if not rag_engine:
        raise HTTPException(status_code=503, detail="Engine unavailable")

    rag_engine.set_multi_turn_retrieval(enabled)
    settings.MULTI_TURN_RETRIEVAL_ENABLED = enabled

    return {
        "multi_turn_retrieval": enabled,
        "message": f"Multi-turn retrieval {'enabled' if enabled else 'disabled'}"
    }


@router.get("/retriever/stats")
async def get_retriever_stats():
    """Get retriever statistics."""
    from main import rag_engine

    if not rag_engine or not rag_engine.retriever:
        raise HTTPException(status_code=503, detail="Retriever unavailable")

    return rag_engine.retriever.get_retrieval_stats()


@router.post("/config/extended-thinking")
async def toggle_extended_thinking(enabled: bool):
    """Toggle extended thinking mode at runtime."""
    from main import rag_engine

    if not rag_engine:
        raise HTTPException(status_code=503, detail="Engine unavailable")

    rag_engine.set_extended_thinking(enabled)
    settings.EXTENDED_THINKING_ENABLED = enabled

    return {
        "extended_thinking": enabled,
        "message": f"Extended thinking {'enabled' if enabled else 'disabled'}"
    }


@router.get("/config/response-settings")
async def get_response_settings():
    """Get current response length settings."""
    return {
        "descriptive_response_min_words": settings.DESCRIPTIVE_RESPONSE_MIN_WORDS,
        "response_truncate_threshold": settings.RESPONSE_TRUNCATE_THRESHOLD,
        "standard_response_target_words": settings.STANDARD_RESPONSE_TARGET_WORDS,
        "extended_thinking_enabled": settings.EXTENDED_THINKING_ENABLED
    }
