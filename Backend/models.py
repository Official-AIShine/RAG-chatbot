""" has everything except for live token streaming and is pydantic v1
Pydantic Models for Request/Response Validation (Pydantic v1)
Defines API contracts for the AI Shine Tutor chatbot.
"""
from typing import List, Literal, Optional, Any, Dict
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()


class Message(BaseModel):
    """
    Single chat message.
    
    Attributes:
        role: Message sender ("human" or "ai")
        content: Message text or structured content
        type: Message type for rendering ("text", "structured", "greeting", "decline")
    """
    role: Literal["human", "ai"] = Field(
        ...,
        description="Message sender role"
    )
    content: str | Dict[str, Any] = Field(
        ...,
        description="Message content (text or structured dict)"
    )
    type: Optional[str] = Field(
        default="text",
        description="Message type for frontend rendering"
    )
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ["human", "ai"]:
            raise ValueError("Role must be 'human' or 'ai'")
        return v
    
    @validator('content')
    def validate_content(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Content cannot be empty string")
        elif isinstance(v, dict):
            # Validate structured content has required keys
            if 'answer' not in v:
                raise ValueError("Structured content must have 'answer' key")
        return v


class ChatRequest(BaseModel):
    """
    Request payload for /chat endpoint.
    
    Attributes:
        chat_history: List of previous messages in conversation
    """
    chat_history: List[Message] = Field(
        ...,
        description="Conversation history with user and AI messages",
        min_items=1
    )
    
    @validator('chat_history')
    def validate_history(cls, v):
        if not v:
            raise ValueError("chat_history cannot be empty")
        
        # Ensure at least one human message exists
        has_human_message = any(msg.role == "human" for msg in v)
        if not has_human_message:
            raise ValueError("chat_history must contain at least one human message")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_history": [
                    {
                        "role": "ai",
                        "content": "👋 Hello! I'm **AI Shine**, your AI/ML tutor. Ask me anything!",
                        "type": "greeting"
                    },
                    {
                        "role": "human",
                        "content": "What is machine learning?",
                        "type": "text"
                    }
                ]
            }
        }


class ChatResponse(BaseModel):
    """
    Response payload from /chat endpoint.
    
    Attributes:
        answer: Generated response text (may contain markdown and structured format)
        type: Response type for frontend rendering logic
    """
    answer: str = Field(
        ...,
        description="AI-generated response text"
    )
    type: Literal["greeting", "text", "structured", "decline"] = Field(
        default="text",
        description="Response type for rendering strategy"
    )
    
    @validator('answer')
    def validate_answer(cls, v):
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "**Answer:**\nMachine learning is a subset of artificial intelligence...\n\n**Key Points:**\n• Learns from data without explicit programming\n• Uses algorithms to find patterns\n• Improves performance over time",
                "type": "structured"
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Attributes:
        api: Overall API status
        rag_engine: RAG engine availability
        components: Individual component statuses
    """
    api: str = Field(..., description="API status")
    rag_engine: str = Field(..., description="RAG engine status")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Component-level health"
    )


class RetrievalContext(BaseModel):
    """
    Internal model for RAG retrieval results.
    Not exposed via API - used internally by rag_engine.
    
    Attributes:
        chunks: Retrieved text chunks from vector DB
        provenance: Metadata about sources (doc_id, score, module)
        score_threshold_met: Whether results meet similarity threshold
    """
    chunks: List[str] = Field(
        default_factory=list,
        description="Retrieved context chunks"
    )
    provenance: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source metadata with scores"
    )
    score_threshold_met: bool = Field(
        default=False,
        description="Whether similarity threshold was met"
    )


class IntentResult(BaseModel):
    """
    Internal model for intent detection results.
    Not exposed via API - used internally by rag_engine.
    
    Attributes:
        intent_type: Detected intent category
        is_continuation: Whether user wants more detail
        is_greeting: Whether message is a greeting
        confidence: Intent confidence score
    """
    intent_type: Literal["greeting", "continuation", "query"] = Field(
        ...,
        description="Detected intent type"
    )
    is_continuation: bool = Field(
        default=False,
        description="Continuation cue detected"
    )
    is_greeting: bool = Field(
        default=False,
        description="Greeting detected"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Intent confidence (0-1)"
    )