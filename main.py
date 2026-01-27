"""
FastAPI Backend for AI Shine Tutor RAG Chatbot
Hybrid routing with Flash-Lite (fast) and Flash (detailed)
WITH conversation history and memory seeding
"""
import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
from dotenv import load_dotenv

from Backend.models import ChatRequest, ChatResponse, Message
from Backend.hybrid_rag_engine import HybridRAGEngine
from Backend.mongodb_client import MongoDBClient
from Backend.utils.user_identity import get_user_key, generate_conversation_id

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global instances
hybrid_rag_engine: Optional[HybridRAGEngine] = None
mongo_client: Optional[MongoDBClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global hybrid_rag_engine, mongo_client
    
    logger.info("[STARTUP] Initializing services...")
    
    try:
        # Initialize MongoDB client
        mongo_client = MongoDBClient()
        logger.info("[STARTUP] ✅ MongoDB client ready")
        
        # Initialize Hybrid RAG Engine
        hybrid_rag_engine = HybridRAGEngine()
        logger.info("[STARTUP] ✅ Hybrid RAG Engine ready")
        logger.info("[STARTUP] Lite: Flash-Lite (15 RPM) for fast queries")
        logger.info("[STARTUP] Full: Flash (1500 RPM) for detailed queries")
        logger.info("[STARTUP] Features: conversation history, memory seeding, bounded memory")
        
    except Exception as e:
        logger.error(f"[STARTUP] ❌ Initialization failed: {e}")
        raise
    
    yield
    
    logger.info("[SHUTDOWN] Closing connections...")
    if hybrid_rag_engine:
        hybrid_rag_engine.cleanup()
    if mongo_client:
        mongo_client.close()
    logger.info("[SHUTDOWN] ✅ Shutdown complete")


app = FastAPI(
    title="AI Shine Tutor API",
    description="Hybrid RAG-powered AI/ML tutor with conversation history",
    version="3.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AI Shine Tutor API",
        "version": "3.1.0",
        "engine": "hybrid",
        "features": [
            "conversation_history",
            "memory_seeding",
            "bounded_memory",
            "length_limits"
        ],
        "modes": {
            "lite": "Flash-Lite (15 RPM, brief responses)",
            "full": "Flash (1500 RPM, detailed responses)"
        }
    }


@app.get("/health")
async def health_check():
    health_status = {
        "api": "healthy",
        "engine": "healthy" if hybrid_rag_engine else "unavailable",
        "mongodb": "healthy" if mongo_client else "unavailable",
        "components": {
            "lite_llm": "ready" if hybrid_rag_engine else "unavailable",
            "full_llm": "ready" if hybrid_rag_engine else "unavailable",
            "memory": "ready" if hybrid_rag_engine else "unavailable",
            "conversations": "ready" if mongo_client else "unavailable"
        }
    }
    
    return health_status


@app.post("/chat")
async def chat(request: ChatRequest, raw_request: Request):
    """
    Hybrid RAG chat endpoint with conversation persistence and character streaming.
    
    Flow:
    1. Get user session ID
    2. Get/create conversation ID
    3. Load conversation from MongoDB (if resuming)
    4. Seed HybridRAGEngine memory with history
    5. Process query through HybridRAGEngine
    6. Stream response character-by-character
    7. Save turn to MongoDB
    """
    if not hybrid_rag_engine or not mongo_client:
        logger.error("[CHAT_ERR] Services not initialized")
        raise HTTPException(status_code=503, detail="Services unavailable")
    
    try:
        # Step 1: Get user identity
        user_key = get_user_key(raw_request)
        
        # Step 2: Get conversation ID from header or generate new
        conversation_id = raw_request.headers.get("X-Conversation-ID")
        if not conversation_id:
            conversation_id = generate_conversation_id()
            logger.info(f"[CHAT] New conversation: {conversation_id}")
        else:
            logger.info(f"[CHAT] Resuming conversation: {conversation_id}")
        
        # Step 3: Extract current query
        if not request.chat_history:
            logger.warning("[CHAT] Empty chat history")
            greeting = "👋 Hello! I'm **AI Shine**, your AI/ML tutor. How can I help you today?"
            
            async def greeting_stream():
                yield json.dumps({"answer_chunk": greeting, "type": "greeting"}) + "\n"
                yield json.dumps({"done": True, "type": "greeting"}) + "\n"
            
            return StreamingResponse(greeting_stream(), media_type="application/x-ndjson")
        
        current_query = None
        for msg in reversed(request.chat_history):
            if msg.role == "human":
                current_query = msg.content
                break
        
        if not current_query:
            raise HTTPException(status_code=400, detail="No user message")
        
        logger.info(f"[CHAT] Query: {current_query[:100]}...")
        
        # Step 4: Load conversation from MongoDB (for memory seeding)
        stored_conv = mongo_client.get_conversation(
            user_key=user_key,
            conversation_id=conversation_id,
            max_messages=20
        )
        
        if stored_conv and stored_conv.get("messages"):
            # MEMORY SEEDING: Load stored messages into engine's memory
            logger.info(f"[CHAT] Seeding memory with {len(stored_conv['messages'])} stored messages")
            
            # Clear existing memory
            hybrid_rag_engine.memory.clear()
            
            # Load messages into ConversationBufferMemory
            for msg in stored_conv["messages"]:
                if msg["role"] == "human":
                    hybrid_rag_engine.memory.chat_memory.add_user_message(msg["content"])
                elif msg["role"] == "ai":
                    hybrid_rag_engine.memory.chat_memory.add_ai_message(msg["content"])
            
            logger.info("[CHAT] ✅ Memory seeded from stored conversation")
        else:
            # New conversation - clear memory
            hybrid_rag_engine.memory.clear()
            logger.info("[CHAT] New conversation - memory cleared")
        
        # Step 5: Process query through HybridRAGEngine (get complete response)
        response = hybrid_rag_engine.process_query(
            query=current_query,
            chat_history=request.chat_history,  # For routing decisions
            conversation_id=conversation_id  # For memory tracking
        )
        
        logger.info(
            f"[CHAT_OK] Response: type={response['type']}, "
            f"route={response.get('route', 'N/A')}"
        )
        
        # Step 6: Stream response character-by-character + Save to MongoDB
        async def generate_stream():
            answer = response["answer"]
            chunk_size = 3  # Stream 3 characters at a time
            
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield json.dumps({
                    "answer_chunk": chunk,
                    "type": response["type"]
                }) + "\n"
                await asyncio.sleep(0.01)  # Small delay for smoother streaming
            
            # Send final metadata
            yield json.dumps({
                "done": True,
                "type": response["type"],
                "route": response.get("route", "unknown")
            }) + "\n"
            
            # Step 7: Save to MongoDB after streaming
            save_success = mongo_client.save_conversation_turn(
                user_key=user_key,
                conversation_id=conversation_id,
                user_message=current_query,
                assistant_message=answer,
                metadata={
                    "route": response.get("route", "unknown"),
                    "response_type": response["type"],
                    "request_id": response.get("request_id", "unknown")
                }
            )
            
            if save_success:
                logger.info(f"[CHAT] Saved to MongoDB: {conversation_id}")
            else:
                logger.warning(f"[CHAT] Failed to save conversation {conversation_id}")
        
        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHAT_ERR] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/conversations")
async def list_conversations(
    raw_request: Request,
    limit: int = 20,
    skip: int = 0
):
    """List user's recent conversations for sidebar."""
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")
    
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


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    raw_request: Request
):
    """Get full conversation for display."""
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")
    
    try:
        user_key = get_user_key(raw_request)
        conversation = mongo_client.get_conversation(
            user_key=user_key,
            conversation_id=conversation_id,
            max_messages=100  # Get full conversation for display
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Format for frontend
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


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    raw_request: Request
):
    """Delete a conversation."""
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")
    
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


@app.post("/api/conversations/new")
async def new_conversation():
    """Start a new conversation (clears engine memory)."""
    if not hybrid_rag_engine:
        raise HTTPException(status_code=503, detail="Engine unavailable")
    
    try:
        # Clear HybridRAGEngine's memory
        hybrid_rag_engine.memory.clear()
        
        # Generate new conversation ID
        new_conv_id = generate_conversation_id()
        
        logger.info(f"[NEW_CONVERSATION] Created: {new_conv_id}")
        
        return {
            "conversation_id": new_conv_id,
            "message": "New conversation started"
        }
        
    except Exception as e:
        logger.error(f"[NEW_CONVERSATION] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
























# # # based on the one below has everything except for live token streaming and regex greeting
# # # BEST MAIN.PY WHICH HAS EVERYTHING EXCEPT LIVE STREAMING OF TOKENS AND NO REGEX
# """
# FastAPI Backend for AI Shine Tutor RAG Chatbot
# LangChain-only implementation with ConversationalRetrievalChain
# """
# import os
# import logging
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# from Backend.models import ChatRequest, ChatResponse
# from Backend.langchain_rag_engine import LangChainRAGEngine

# load_dotenv()

# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)

# langchain_rag_engine = None


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Application lifespan manager."""
#     global langchain_rag_engine
    
#     logger.info("[STARTUP] Initializing LangChain RAG Engine...")
    
#     try:
#         langchain_rag_engine = LangChainRAGEngine()
#         logger.info("[STARTUP] ✅ LangChain RAG Engine ready")
#     except Exception as e:
#         logger.error(f"[STARTUP] ❌ Failed to initialize LangChain RAG Engine: {e}")
#         raise
    
#     yield
    
#     logger.info("[SHUTDOWN] Closing connections...")
#     if langchain_rag_engine:
#         langchain_rag_engine.cleanup()
#     logger.info("[SHUTDOWN] ✅ Shutdown complete")


# app = FastAPI(
#     title="AI Shine Tutor API",
#     description="RAG-powered AI/ML tutor with LangChain",
#     version="2.0.0",
#     lifespan=lifespan
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.get("/")
# async def root():
#     return {
#         "status": "online",
#         "service": "AI Shine Tutor API",
#         "version": "2.0.0",
#         "engine": "langchain"
#     }


# @app.get("/health")
# async def health_check():
#     health_status = {
#         "api": "healthy",
#         "engine": "healthy" if langchain_rag_engine else "unavailable",
#         "components": {
#             "llm": "ready" if langchain_rag_engine else "unavailable",
#             "memory": "ready" if langchain_rag_engine else "unavailable"
#         }
#     }
    
#     return health_status


# @app.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     """
#     LangChain-powered chat endpoint.
#     Features:
#     - Automatic conversation understanding (no regex)
#     - ConversationSummaryMemory (token efficient)
#     - Natural continuation handling
#     """
#     if not langchain_rag_engine:
#         logger.error("[CHAT_ERR] LangChain RAG engine not initialized")
#         raise HTTPException(status_code=503, detail="LangChain RAG engine unavailable")
    
#     try:
#         if not request.chat_history:
#             logger.warning("[CHAT] Empty chat history")
#             return ChatResponse(
#                 answer="👋 Hello! I'm **AI Shine**, your AI/ML tutor. How can I help you today?",
#                 type="greeting"
#             )
        
#         current_query = None
#         for msg in reversed(request.chat_history):
#             if msg.role == "human":
#                 current_query = msg.content
#                 break
        
#         if not current_query:
#             raise HTTPException(status_code=400, detail="No user message")
        
#         logger.info(f"[CHAT] Processing: {current_query[:100]}...")
        
#         response = langchain_rag_engine.process_query(
#             query=current_query,
#             chat_history=request.chat_history
#         )
        
#         logger.info(f"[CHAT_OK] Response type: {response['type']}")
        
#         return ChatResponse(
#             answer=response["answer"],
#             type=response["type"]
#         )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"[CHAT_ERR] Unexpected error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error")


# # ✅ CORRECT for Render
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 10000))
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=port,
#         log_level="info",
#         workers=1
#     )    