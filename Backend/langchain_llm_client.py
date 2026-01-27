""" has everything except for live token streaming
LangChain Gemini Client
Wraps Google Gemini 2.5 Flash and Flash-Lite with LangChain's ChatGoogleGenerativeAI interface.
LangChain LLM Client for Gemini 2.5 Flash and Flash Lite
Updated to use google.genai (non-deprecated package)
"""
import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_langchain_gemini_client() -> ChatGoogleGenerativeAI:
    """
    Create LangChain-compatible Gemini 2.5 Flash client with STREAMING support.
    
    Configuration:
    - Model: gemini-2.5-flash
    - Temperature: 0.7 (balanced creativity)
    - Max tokens: 4096
    - Streaming: ENABLED
    
    Returns:
        Configured ChatGoogleGenerativeAI instance
        
    Raises:
        ValueError: If GOOGLE_API_KEY not set in environment
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("[LANGCHAIN_GEMINI] GOOGLE_API_KEY not set in environment")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=4096,
        streaming=True,  # ADD THIS
    )
    
    logger.info("[LANGCHAIN_GEMINI] ✅ Gemini 2.5 Flash initialized with streaming")
    return llm

def create_langchain_gemini_lite_client() -> ChatGoogleGenerativeAI:
    """
    Create LangChain-compatible Gemini 2.5 Flash Lite client with STREAMING support.
    
    Configuration:
    - Model: gemini-2.5-flash-lite
    - Temperature: 0.5 (more deterministic)
    - Max tokens: 1024
    - Streaming: ENABLED
    
    Returns:
        Configured ChatGoogleGenerativeAI instance
        
    Raises:
        ValueError: If GOOGLE_API_KEY not set in environment
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("[LANGCHAIN_GEMINI_LITE] GOOGLE_API_KEY not set in environment")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0.5,
        top_p=0.9,
        top_k=30,
        max_output_tokens=1024,
        streaming=True,  # ADD THIS
    )
    
    logger.info("[LANGCHAIN_GEMINI_LITE] ✅ Gemini 2.5 Flash Lite initialized with streaming")
    return llm
















# """ has everything except for live token streaming
# LangChain Gemini Client
# Wraps Google Gemini 2.5 Flash with LangChain's ChatGoogleGenerativeAI interface.
# """
# import os
# import logging
# from langchain_google_genai import ChatGoogleGenerativeAI
# from google.generativeai.types import HarmCategory, HarmBlockThreshold
# from dotenv import load_dotenv

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def create_langchain_gemini_client() -> ChatGoogleGenerativeAI:
#     """
#     Create LangChain-compatible Gemini 2.5 Flash client.
    
#     Returns:
#         ChatGoogleGenerativeAI instance configured for educational content
#     """
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         raise ValueError("[LANGCHAIN_GEMINI] GOOGLE_API_KEY not set")
    
#     # Configure safety settings (permissive for educational content)
#     safety_settings = {
#         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     }
    
#     # Create LangChain Gemini client
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         google_api_key=api_key,
#         temperature=0.9,  # Balanced creativity and grounding
#         top_p=0.95,
#         top_k=40,
#         max_output_tokens=4096,  # Increased to reduce truncation
#         safety_settings=safety_settings,
#         convert_system_message_to_human=True  # Gemini requires system as first user message
#     )
    
#     logger.info("[LANGCHAIN_GEMINI] ✅ Initialized Gemini 2.5 Flash via LangChain")
#     logger.info("[LANGCHAIN_GEMINI] Config: temp=0.9, max_tokens=4096")
    
#     return llm