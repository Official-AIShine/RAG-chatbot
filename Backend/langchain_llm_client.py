"""
LangChain Gemini Client

Factory functions for creating Gemini LLM instances.
Optimized for streaming and low latency.
"""
import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from Backend.config import settings

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_langchain_gemini_client() -> ChatGoogleGenerativeAI:
    """
    Create LangChain-compatible Gemini client with streaming.

    Returns:
        Configured ChatGoogleGenerativeAI instance

    Raises:
        ValueError: If GOOGLE_API_KEY not set
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("[GEMINI] GOOGLE_API_KEY not set in environment")

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=api_key,
        temperature=settings.GEMINI_TEMPERATURE,
        top_p=0.95,
        top_k=40,
        max_output_tokens=settings.GEMINI_MAX_TOKENS,
        streaming=True,
    )

    logger.info(f"[GEMINI] Initialized {settings.GEMINI_MODEL} with streaming")
    return llm
