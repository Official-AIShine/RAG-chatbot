"""
API Routers for AI Shine Tutor.
"""
from Backend.routers.chat import router as chat_router
from Backend.routers.conversations import router as conversations_router
from Backend.routers.admin import router as admin_router

__all__ = ["chat_router", "conversations_router", "admin_router"]
