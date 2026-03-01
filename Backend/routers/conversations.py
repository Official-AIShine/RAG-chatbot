"""
Conversation management router.
Handles CRUD operations for chat conversations.
"""
import logging
from fastapi import APIRouter, HTTPException, Request

from Backend.utils.user_identity import get_user_key, generate_conversation_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(
    raw_request: Request,
    limit: int = 20,
    skip: int = 0
):
    """List user's recent conversations."""
    from main import mongo_client

    if not mongo_client:
        return {"conversations": [], "message": "Persistence unavailable"}

    try:
        user_key = get_user_key(raw_request)
        conversations = mongo_client.list_conversations(
            user_key=user_key,
            limit=limit,
            skip=skip
        )
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"[LIST_CONVERSATIONS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str, raw_request: Request):
    """Get full conversation."""
    from main import mongo_client

    if not mongo_client:
        raise HTTPException(status_code=503, detail="Persistence unavailable")

    try:
        user_key = get_user_key(raw_request)
        conversation = mongo_client.get_conversation(
            user_key=user_key,
            conversation_id=conversation_id,
            max_messages=100
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "id": conversation["conversation_id"],
            "title": conversation.get("metadata", {}).get("title", "Untitled"),
            "created_at": conversation["created_at"].isoformat(),
            "updated_at": conversation["updated_at"].isoformat(),
            "messages": conversation.get("messages", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GET_CONVERSATION] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str, raw_request: Request):
    """Delete a conversation."""
    from main import mongo_client

    if not mongo_client:
        raise HTTPException(status_code=503, detail="Persistence unavailable")

    try:
        user_key = get_user_key(raw_request)
        deleted = mongo_client.delete_conversation(
            user_key=user_key,
            conversation_id=conversation_id
        )

        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DELETE_CONVERSATION] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/new")
async def new_conversation():
    """Start a new conversation."""
    from main import rag_engine

    if not rag_engine:
        raise HTTPException(status_code=503, detail="Engine unavailable")

    try:
        rag_engine.clear_memory()
        new_conv_id = generate_conversation_id()

        logger.info(f"[NEW_CONVERSATION] Created: {new_conv_id}")

        return {
            "conversation_id": new_conv_id,
            "message": "New conversation started"
        }
    except Exception as e:
        logger.error(f"[NEW_CONVERSATION] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
