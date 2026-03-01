"""
Chat endpoint router.
Handles the main RAG chat functionality.
"""
import logging
import asyncio
import time
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from Backend.models import ChatRequest
from Backend.config import settings
from Backend.utils.user_identity import get_user_key, generate_conversation_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def validate_query(query: str) -> tuple[bool, str]:
    """
    Validate user query for length and content.

    Returns:
        (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"

    if len(query) > settings.MAX_QUERY_LENGTH:
        return False, f"Query exceeds maximum length of {settings.MAX_QUERY_LENGTH} characters"

    # Basic sanitization check (no control characters except newlines/tabs)
    import unicodedata
    for char in query:
        if unicodedata.category(char) == 'Cc' and char not in '\n\t\r':
            return False, "Query contains invalid characters"

    return True, ""


@router.post("/chat")
async def chat(request: ChatRequest, raw_request: Request):
    """
    Optimized RAG chat endpoint with token streaming.

    Flow:
    1. Validate input
    2. Get user session
    3. Load conversation history (if resuming)
    4. Stream response tokens
    5. Save conversation turn
    """
    # Import here to avoid circular imports - these are set by main.py
    from main import rag_engine, mongo_client, metrics

    request_start = time.time()

    if not rag_engine:
        logger.error("[CHAT_ERR] RAG engine not initialized")
        raise HTTPException(status_code=503, detail="Service unavailable")

    try:
        # Get user identity
        user_key = get_user_key(raw_request)

        # Get conversation ID
        conversation_id = raw_request.headers.get("X-Conversation-ID")
        if not conversation_id:
            conversation_id = generate_conversation_id()
            logger.info(f"[CHAT] New conversation: {conversation_id}")
        else:
            logger.info(f"[CHAT] Resuming: {conversation_id}")

        # Handle empty chat history
        if not request.chat_history:
            greeting = rag_engine.get_greeting()

            async def greeting_stream():
                for i in range(0, len(greeting), 3):
                    chunk = greeting[i:i + 3]
                    yield json.dumps({"answer_chunk": chunk, "type": "greeting"}) + "\n"
                    await asyncio.sleep(0.01)
                yield json.dumps({"done": True, "type": "greeting"}) + "\n"

            return StreamingResponse(greeting_stream(), media_type="application/x-ndjson")

        # Extract current query
        current_query = None
        for msg in reversed(request.chat_history):
            if msg.role == "human":
                current_query = msg.content
                break

        if not current_query:
            raise HTTPException(status_code=400, detail="No user message found")

        # Validate query
        is_valid, error_msg = validate_query(current_query)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        logger.info(f"[CHAT] Query: {current_query[:100]}...")

        # Build conversation history for engine
        # Priority: Use request.chat_history (from frontend), fallback to MongoDB
        chat_history_for_engine = []

        # First, try to use the history sent in the request (excludes current query)
        if request.chat_history and len(request.chat_history) > 1:
            # Include all messages except the last one (current query)
            for msg in request.chat_history[:-1]:
                chat_history_for_engine.append({
                    "role": msg.role,
                    "content": msg.content
                })
            logger.info(f"[CHAT] Using {len(chat_history_for_engine)} messages from request")

        # Fallback to MongoDB if no history in request
        elif mongo_client:
            stored_conv = mongo_client.get_conversation(
                user_key=user_key,
                conversation_id=conversation_id,
                max_messages=settings.MAX_HISTORY_MESSAGES
            )

            if stored_conv and stored_conv.get("messages"):
                logger.info(f"[CHAT] Loaded {len(stored_conv['messages'])} messages from MongoDB")
                for msg in stored_conv["messages"]:
                    chat_history_for_engine.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        # Accumulate response for persistence
        full_response = []

        async def generate_stream():
            try:
                # Stream tokens from RAG engine
                async for token in rag_engine.process_query_stream(
                    query=current_query,
                    chat_history=chat_history_for_engine,
                    conversation_id=conversation_id
                ):
                    full_response.append(token)
                    yield json.dumps({"answer_chunk": token, "type": "text"}) + "\n"
                    await asyncio.sleep(0.01)

                # Send completion signal
                yield json.dumps({"done": True, "type": "text"}) + "\n"

                # Save to MongoDB (if available)
                complete_answer = "".join(full_response)

                if mongo_client:
                    save_success = mongo_client.save_conversation_turn(
                        user_key=user_key,
                        conversation_id=conversation_id,
                        user_message=current_query,
                        assistant_message=complete_answer,
                        metadata={
                            "response_time_ms": int((time.time() - request_start) * 1000)
                        }
                    )

                    if save_success:
                        logger.info(f"[CHAT] Saved to MongoDB: {conversation_id}")

                # Record metrics
                if metrics:
                    metrics.record_request(
                        latency_ms=(time.time() - request_start) * 1000,
                        tokens=len(complete_answer.split()),
                        success=True
                    )

            except Exception as e:
                logger.error(f"[STREAM_ERR] {e}", exc_info=True)
                yield json.dumps({"error": str(e), "type": "error"}) + "\n"

                if metrics:
                    metrics.record_request(
                        latency_ms=(time.time() - request_start) * 1000,
                        tokens=0,
                        success=False
                    )

        return StreamingResponse(generate_stream(), media_type="application/x-ndjson")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHAT_ERR] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
