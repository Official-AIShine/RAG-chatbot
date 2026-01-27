"""
User identity management for session-based tracking.
"""
from uuid import uuid4
from fastapi import Request
import logging

logger = logging.getLogger(__name__)


def get_user_key(request: Request) -> str:
    """
    Extract user identity from request headers.
    Returns session_<uuid> format.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        str: user_key in format "session_<uuid>"
    """
    session_id = request.headers.get("X-Session-ID")
    
    if not session_id:
        # Generate new session for first-time users
        session_id = str(uuid4())
        logger.info(f"[SESSION] Generated new: {session_id}")
    
    return f"session_{session_id}"


def generate_conversation_id() -> str:
    """
    Generate unique conversation ID.
    
    Returns:
        str: conversation_id in format "conv_<uuid>"
    """
    return f"conv_{uuid4()}"