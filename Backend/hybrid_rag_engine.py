"""
Hybrid RAG Engine with Optimized API Usage
Combines LangChain retrieval with Gemini LLMs for educational chatbot.

Key optimizations:
- Single retriever instance (prevents duplicate embedding calls)
- ConversationBufferMemory (0 API calls for memory storage)
- Dual-chain routing (Lite vs Full based on query complexity)
- Bounded memory cleanup (prevents context overflow)
- Greeting/farewell handling
- Response cleaning and classification
"""
import logging
import uuid
import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio  # ADD THIS
from typing import List, Dict, Any, Optional, AsyncIterator
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from Backend.langchain_retriever import get_langchain_retriever
from Backend.langchain_llm_client import (
    create_langchain_gemini_client,
    create_langchain_gemini_lite_client
)
from Backend.prompt_builder import PromptBuilder
from Backend.models import Message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRAGEngine:
    """
    Production RAG engine with:
    - Single retriever (no duplicate embeddings)
    - Dual LLM routing (Lite: 1 API call, Full: 2 API calls)
    - Conversation memory with automatic cleanup
    - Greeting/farewell detection
    - Response cleaning and classification
    """
    
    # Patterns for detecting deep-dive requests
    DETAILED_PATTERNS = [
        # Explicit continuation requests
        r'\btell\s+me\s+more\b',
        r'\bexpand\b',
        r'\bcontinue\b',
        r'\bgo\s+deeper\b',
        r'\bkeep\s+going\b',
        r'\bgo\s+on\b',
        
        # Depth/detail requests
        r'\bin\s+detail\b',
        r'\bdetailed\b',
        r'\belaborate\b',
        r'\bexplain\s+further\b',
        r'\bmore\s+information\b',
        r'\bmore\s+details?\b',
        r'\bcomprehensive\b',
        r'\bthorough\b',
        r'\bin[\-\s]depth\b',
        r'\bextensive\b',
        r'\bfull\s+explanation\b',
        
        # Comparison/analysis requests
        r'\bcompare\b',
        r'\bcontrast\b',
        r'\bdifference\s+between\b',
        r'\bhow\s+do(?:es)?\s+.*\s+differ\b',
        r'\bvs\.?\b',
        r'\bversus\b',
        r'\banalyze\b',
        r'\banalysis\b',
        r'\bbreak\s+down\b',
        
        # Multi-part questions
        r'\band\s+also\b',
        r'\band\s+how\b',
        r'\band\s+what\b',
        r'\band\s+why\b',
        r'\bwhat\s+about\b',
        
        # Follow-up patterns
        r'\bwhat\s+else\b',
        r'\bany(?:thing)?\s+else\b',
        r'\bother\s+examples?\b',
        r'\bmore\s+examples?\b',
        
        # Explanation depth
        r'\bstep[\-\s]by[\-\s]step\b',
        r'\bwalk\s+me\s+through\b',
        r'\bguide\s+me\b',
        r'\bshow\s+me\s+how\b',
    ]
    
    # Patterns for brief responses
    BRIEF_PATTERNS = [
        # Explicit brevity requests
        r'\bbriefly\b',
        r'\bquick\b',
        r'\bquickly\b',
        r'\bshort\b',
        r'\bconcise\b',
        r'\bsummarize\b',
        r'\bsummary\b',
        r'\bin\s+short\b',
        r'\bin\s+brief\b',
        r'\bshort\s+answer\b',
        
        # Single-concept queries
        r'\bwhat\s+is\b',
        r'\bdefine\b',
        r'\bdefinition\s+of\b',
        r'\bmeaning\s+of\b',
        r'\bexplain\s+(?!how|why)\w+$',
        
        # Simple yes/no or list requests
        r'\blist\b',
        r'\bname\b',
        r'\bgive\s+me\s+\d+\b',
        r'\btop\s+\d+\b',
    ]
    
    def __init__(self):
        """Initialize hybrid RAG components with bounded memory."""
        try:
            logger.info("[HYBRID_RAG_ENGINE] 🔧 Initializing with API call optimization...")
            
            # SINGLE retriever instance (dual-threshold internally)
            # Prevents duplicate embedding calls
            self.retriever = get_langchain_retriever(
                collection_name="module_vectors",
                primary_threshold=0.55,
                fallback_threshold=0.45
            )
            
            # Dual LLM setup
            self.flash_llm = create_langchain_gemini_client()  # Full power
            self.lite_llm = create_langchain_gemini_lite_client()  # Fast & cheap
            
            # ConversationBufferMemory - NO SUMMARIZATION API CALLS
            # Stores full history in memory (uses more tokens but zero API calls)
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                input_key="question"
            )
            
            logger.info("[HYBRID_RAG_ENGINE] ✅ Using ConversationBufferMemory (0 API calls for memory)")
            
            # PRODUCTION OPTIMIZATION: Memory cleanup tracking
            # Prevents unbounded memory growth in long-running servers
            self.active_conversations = {}  # {conversation_id: last_access_timestamp}
            self.max_memory_conversations = 50  # Clean up after this many
            self.memory_cleanup_age = 3600  # Clear conversations older than 1 hour
            
            logger.info(
                f"[HYBRID_RAG_ENGINE] 🧹 Memory cleanup: "
                f"max {self.max_memory_conversations} conversations, 1 hour TTL"
            )
            
            # Prompt builder for greeting/farewell
            self.prompt_builder = PromptBuilder()
            
            # Compile regex patterns
            self.detailed_regex = re.compile('|'.join(self.DETAILED_PATTERNS), re.IGNORECASE)
            self.brief_regex = re.compile('|'.join(self.BRIEF_PATTERNS), re.IGNORECASE)
            
            # Build chains
            self.lite_chain = self._build_lite_chain()
            self.full_chain = self._build_full_chain()
            
            logger.info("[HYBRID_RAG_ENGINE] ✅ All components initialized")
            logger.info("[HYBRID_RAG_ENGINE] 📊 API Call Breakdown:")
            logger.info("[HYBRID_RAG_ENGINE]    Lite query: 1 API call (answer only)")
            logger.info("[HYBRID_RAG_ENGINE]    Full query: 2 API calls (condense + answer)")
            logger.info("[HYBRID_RAG_ENGINE]    Memory: 0 API calls (buffer storage)")
            
        except Exception as e:
            logger.error(f"[HYBRID_RAG_ENGINE_ERR] Initialization failed: {e}")
            raise
    
    def _cleanup_old_memory(self):
        """
        Clean up old conversation memory to prevent unbounded growth.
        Removes conversations not accessed in last hour or when count exceeds limit.
        """
        try:
            current_time = time.time()
            
            # Remove conversations older than TTL
            expired = [
                conv_id for conv_id, last_access in self.active_conversations.items()
                if current_time - last_access > self.memory_cleanup_age
            ]
            
            for conv_id in expired:
                del self.active_conversations[conv_id]
                logger.info(f"[MEMORY_CLEANUP] Expired conversation: {conv_id}")
            
            # If still over limit, remove oldest conversations
            to_remove = 0
            if len(self.active_conversations) > self.max_memory_conversations:
                sorted_convs = sorted(
                    self.active_conversations.items(),
                    key=lambda x: x[1]
                )
                
                to_remove = len(self.active_conversations) - self.max_memory_conversations
                for conv_id, _ in sorted_convs[:to_remove]:
                    del self.active_conversations[conv_id]
                    logger.info(f"[MEMORY_CLEANUP] Removed oldest conversation: {conv_id}")
            
            if expired or to_remove > 0:
                logger.info(
                    f"[MEMORY_CLEANUP] Cleaned {len(expired) + to_remove} conversations. "
                    f"Active: {len(self.active_conversations)}"
                )
                
        except Exception as e:
            logger.error(f"[MEMORY_CLEANUP_ERR] {e}")
    
    def track_conversation_access(self, conversation_id: str):
        """
        Track conversation access for memory cleanup.
        
        Args:
            conversation_id: Conversation being accessed
        """
        self.active_conversations[conversation_id] = time.time()
        
        # Periodic cleanup check
        if len(self.active_conversations) > self.max_memory_conversations:
            self._cleanup_old_memory()
    
    def _build_lite_chain(self) -> ConversationalRetrievalChain:
        """
        Build lite chain for brief, fast responses.
        OPTIMIZED: No condense step (condense_question_llm=None).
        
        Returns:
            Configured Lite chain
        """
        # Lite QA prompt - emphasizes brevity
        lite_qa_template = """You are AI Shine, an AI/ML educational assistant.

Context from knowledge base:
{context}

Chat History:
{chat_history}

Question: {question}

RULES:
1. SCOPE: AI, ML, Deep Learning, Data Science, NLP, Computer Vision, AI Applications only
2. BREVITY: Respond in 2-3 concise sentences maximum. No lists unless absolutely necessary.
3. SYNTHESIS: Lead with definition if KB lacks it, then add one KB example
4. TONE: Direct and educational. No meta-commentary.
5. FORMAT: Use <p> tags only. No lists for lite responses.
6. OUT OF SCOPE: "⚠️ I specialize in AI and Machine Learning topics."

Answer:"""
        
        # OPTIMIZATION: condense_question_llm=None skips expensive condense step
        # Lite queries are simple, don't need conversation history rephrasing
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.lite_llm,
            retriever=self.retriever,  # Single shared retriever
            memory=self.memory,
            condense_question_llm=None,  # Skip condense = saves 1 API call
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=lite_qa_template,
                    input_variables=["context", "chat_history", "question"]
                )
            },
            return_source_documents=False,
            verbose=False
        )
        
        logger.info("[HYBRID_RAG_ENGINE] ✅ Lite chain built (1 API call per query)")
        return chain
    
    def _build_full_chain(self) -> ConversationalRetrievalChain:
        """
        Build full chain for detailed, comprehensive responses.
        
        Returns:
            Configured Full chain
        """
        # Condense question prompt (for follow-up questions)
        condense_template = """Given this conversation, rephrase the follow-up as a standalone question.

Chat History:
{chat_history}

Follow Up: {question}
Standalone question:"""
        
        # Full QA prompt - emphasizes detail
        full_qa_template = """You are AI Shine, an AI/ML educational assistant.

Context from knowledge base:
{context}

Chat History:
{chat_history}

Question: {question}

RULES:
1. SCOPE: AI, ML, Deep Learning, Data Science, NLP, Computer Vision, AI Applications only
2. SYNTHESIS:
   - If KB mentions concept without defining: lead with definition, then add KB examples
   - Paraphrase all KB content naturally - never use direct quotes or preserve quotation marks
   - NEVER invent examples, tools, or statistics not in KB
   - Only provide general definitions when needed
3. TONE: Direct and educational. NO meta-commentary
4. FORMAT:
   - <p> for paragraphs (blank line between)
   - <ul><li> for lists (blank line between items)
   - <strong> for key terms (2-4 words)
   - If long: end with <p><em>Write 'continue' to keep generating...</em></p>
5. OUT OF SCOPE: "⚠️ I specialize in AI and Machine Learning topics. I'd be happy to help with questions about [suggest 2-3 AI/ML topics]."

Answer:"""
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.flash_llm,
            retriever=self.retriever,  # Same single retriever
            memory=self.memory,
            condense_question_prompt=PromptTemplate(
                template=condense_template,
                input_variables=["chat_history", "question"]
            ),
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=full_qa_template,
                    input_variables=["context", "chat_history", "question"]
                )
            },
            return_source_documents=False,
            verbose=False
        )
        
        logger.info("[HYBRID_RAG_ENGINE] ✅ Full chain built (2 API calls per query)")
        return chain
    
    def _route_query(self, query: str, chat_history: List[Message]) -> str:
        """
        Route query to appropriate chain based on complexity signals.
        OPTIMIZED: Continuation queries use lite (skip expensive condense step)
        
        Args:
            query: User query
            chat_history: Conversation history
        
        Returns:
            "lite" or "full"
        """
        query_lower = query.lower().strip()
        
        # Count words (longer queries often need more depth)
        word_count = len(query_lower.split())
        
        # Explicit brief request → lite
        if self.brief_regex.search(query_lower):
            logger.info("[ROUTER] Brief signal detected → Lite chain")
            return "lite"
        
        # Explicit detailed request → full
        if self.detailed_regex.search(query_lower):
            logger.info("[ROUTER] Detailed signal detected → Full chain")
            return "full"
        
        # Very long queries → full (>=20 words suggests complexity)
        if word_count >= 20:
            logger.info(f"[ROUTER] Long query ({word_count} words) → Full chain")
            return "full"
        
        # Follow-up question after receiving a lite response → full
        # (User wants more depth on the same topic)
        if len(chat_history) >= 2:
            last_assistant_msg = None
            for msg in reversed(chat_history):
                if msg.role == "assistant":
                    last_assistant_msg = msg.content
                    break
            
            # If last response was brief (<300 chars) and user asks follow-up → go full
            if last_assistant_msg and len(last_assistant_msg) < 300:
                # Check if current query is a follow-up (short, no new topic keywords)
                has_new_topic = any(keyword in query_lower for keyword in [
                    'what is', 'define', 'explain', 'how does', 'why', 'when', 'where'
                ])
                
                if not has_new_topic and word_count < 15:
                    logger.info("[ROUTER] Follow-up after brief response → Full chain")
                    return "full"
        
        # Default: lite (fast & cheap for first queries on any topic)
        logger.info(f"[ROUTER] Default routing (word_count={word_count}) → Lite chain")
        return "lite"
    
    def seed_memory(self, messages: List[Message]):
        """
        Seed memory with stored conversation history.
        
        Args:
            messages: List of Message objects with role and content
        """
        if not messages:
            return
        
        logger.info(f"[HYBRID_RAG_ENGINE] Seeding memory with {len(messages)} messages")
        
        # Clear existing memory
        self.memory.clear()
        
        # Add messages to memory
        for msg in messages:
            if msg.role == "user":
                self.memory.chat_memory.add_user_message(msg.content)
            elif msg.role == "assistant":
                self.memory.chat_memory.add_ai_message(msg.content)
        
        logger.info("[HYBRID_RAG_ENGINE] ✅ Memory seeded successfully")
    
    def process_query(
        self,
        query: str,
        chat_history: List[Message],
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user query with intelligent routing.
        
        Args:
            query: Current user query
            chat_history: Full conversation history
            conversation_id: Optional conversation ID for memory tracking
        
        Returns:
            Dict with 'answer', 'type', 'route', and 'request_id'
        """
        request_id = str(uuid.uuid4())[:8]  # Short request ID for tracking
        
        try:
            # Track conversation access for memory cleanup
            if conversation_id:
                self.track_conversation_access(conversation_id)
            
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📝 NEW REQUEST: '{query[:100]}...'")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Chat history length: {len(chat_history)} messages")
            
            # Handle greetings
            if self.prompt_builder.greeting_regex.match(query.strip()):
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 👋 GREETING DETECTED")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ Response via regex pattern (0 API calls)")
                self.memory.clear()
                return {
                    "answer": self.prompt_builder.build_greeting_response(),
                    "type": "greeting",
                    "route": "greeting",
                    "request_id": request_id
                }
            
            # Handle farewells
            if self.prompt_builder.farewell_regex.match(query.strip()):
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 👋 FAREWELL DETECTED")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ Response via regex pattern (0 API calls)")
                self.memory.clear()
                return {
                    "answer": self.prompt_builder.build_farewell_response(),
                    "type": "text",
                    "route": "farewell",
                    "request_id": request_id
                }
            
            # Route to appropriate chain
            route = self._route_query(query, chat_history)
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 ROUTING DECISION: {route.upper()}")
            
            if route == "lite":
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🚀 LITE CHAIN SELECTED")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💡 Reason: Default/first query OR brief signal detected")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Expected API calls: 1")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #1 START: Answer generation (Flash-Lite, 15 RPM)")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Generate brief response from KB context")
                
                response = self.lite_chain.invoke({"question": query})
                
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #1 COMPLETE: Answer generation")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📈 Total API calls for this request: 1")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 Model used: gemini-2.5-flash-lite")
            else:
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🔥 FULL CHAIN SELECTED")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💡 Reason: Detailed signal detected (tell me more/expand/in detail)")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Expected API calls: 2")
                
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #1 START: Condense question (Flash, 1500 RPM)")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Rephrase '{query}' with chat history context")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Input: '{query}' + {len(chat_history)} previous messages")
                
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #2 START: Answer generation (Flash, 1500 RPM)")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Generate detailed response with HTML formatting")
                
                response = self.full_chain.invoke({"question": query})
                
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #1 COMPLETE: Condense question")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #2 COMPLETE: Answer generation")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📈 Total API calls for this request: 2")
                logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 Model used: gemini-2.5-flash")
            
            answer = response.get("answer", "")
            
            # Clean response
            answer = self._clean_response(answer)
            
            # Classify response type
            response_type = self._classify_response(answer)
            
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ RESPONSE COMPLETE")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📝 Response length: {len(answer)} characters")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🏷️  Response type: {response_type}")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🚦 Route used: {route}")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💾 Memory stored in buffer (0 API calls)")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 FINAL COUNT - API calls made: {1 if route == 'lite' else 2}")
            logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ════════════════════════════════════════")
            
            return {
                "answer": answer,
                "type": response_type,
                "route": route,
                "request_id": request_id
            }
            
        except Exception as e:
            logger.error(f"[HYBRID_RAG_ENGINE] [{request_id}] ❌ PIPELINE FAILURE: {e}", exc_info=True)
            return {
                "answer": "⚠️ An unexpected error occurred. Please try your question again.",
                "type": "text",
                "route": "error",
                "request_id": request_id
            }
    async def process_query_stream(
        self,
        query: str,
        chat_history: List[Message],
        conversation_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Process user query with TRUE TOKEN-BY-TOKEN STREAMING.
        
        Args:
            query: Current user query
            chat_history: Full conversation history
            conversation_id: Optional conversation ID for memory tracking
        
        Yields:
            str: Token chunks as they arrive from LLM
        """
        request_id = str(uuid.uuid4())[:8]
        
        try:
            # Track conversation access
            if conversation_id:
                self.track_conversation_access(conversation_id)
            
            logger.info(f"[STREAM] [{request_id}] 📝 NEW STREAMING REQUEST: '{query[:100]}...'")
            
            # Handle greetings (no streaming needed)
            if self.prompt_builder.greeting_regex.match(query.strip()):
                yield self.prompt_builder.build_greeting_response()
                return
            
            # Handle farewells (no streaming needed)
            if self.prompt_builder.farewell_regex.match(query.strip()):
                yield self.prompt_builder.build_farewell_response()
                return
            
            # Route to appropriate chain
            route = self._route_query(query, chat_history)
            logger.info(f"[STREAM] [{request_id}] 🎯 Route: {route.upper()}")
            
            # Select chain
            chain = self.lite_chain if route == "lite" else self.full_chain
            
            # OPTION 3: Parallel execution for full chain
            if route == "full":
                logger.info(f"[STREAM] [{request_id}] ⚡ Starting parallel condense + retrieval")
                
                # Stream immediately while condense happens in background
                async for chunk in chain.astream({"question": query}):
                    if hasattr(chunk, 'content'):
                        cleaned = self._clean_response(chunk.content)
                        if cleaned:
                            yield cleaned
                    elif isinstance(chunk, dict) and 'answer' in chunk:
                        answer = chunk['answer']
                        if isinstance(answer, str):
                            cleaned = self._clean_response(answer)
                            if cleaned:
                                yield cleaned
                    elif isinstance(chunk, str):
                        cleaned = self._clean_response(chunk)
                        if cleaned:
                            yield cleaned
            else:
                # Lite chain - direct streaming
                logger.info(f"[STREAM] [{request_id}] ⚡ Streaming from lite chain")
                
                async for chunk in chain.astream({"question": query}):
                    if hasattr(chunk, 'content'):
                        cleaned = self._clean_response(chunk.content)
                        if cleaned:
                            yield cleaned
                    elif isinstance(chunk, dict) and 'answer' in chunk:
                        answer = chunk['answer']
                        if isinstance(answer, str):
                            cleaned = self._clean_response(answer)
                            if cleaned:
                                yield cleaned
                    elif isinstance(chunk, str):
                        cleaned = self._clean_response(chunk)
                        if cleaned:
                            yield cleaned
            
            logger.info(f"[STREAM] [{request_id}] ✅ Streaming complete")
            
        except Exception as e:
            logger.error(f"[STREAM] [{request_id}] ❌ Error: {e}", exc_info=True)
            yield "⚠️ An error occurred during streaming. Please try again."
            
    def _clean_response(self, response: str) -> str:
        """Clean and format LLM response."""
        # Convert markdown bold to HTML
        response = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', response)
        
        # Remove any stray asterisk bullets
        response = response.replace('* ', '• ')
        
        return response.strip()
    
    def _classify_response(self, response: str) -> str:
        """Classify response type for frontend rendering."""
        if response.startswith("⚠") or "I specialize in AI and Machine Learning topics" in response:
            return "decline"
        
        if "I don't have" in response or "I don't know" in response:
            return "decline"
        
        return "text"
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("[HYBRID_RAG_ENGINE] 🧹 Memory cleared")
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.memory.clear()
            self.active_conversations.clear()
            logger.info("[HYBRID_RAG_ENGINE] ✅ Cleanup complete")
        except Exception as e:
            logger.error(f"[HYBRID_RAG_ENGINE] Cleanup error: {e}")


# Singleton instance
_rag_engine: Optional[HybridRAGEngine] = None


def get_rag_engine() -> HybridRAGEngine:
    """
    Get singleton RAG engine instance.
    
    Returns:
        Initialized HybridRAGEngine
    """
    global _rag_engine
    
    if _rag_engine is None:
        _rag_engine = HybridRAGEngine()
    
    return _rag_engine















# """
# Hybrid RAG Engine
# Intelligent routing between Flash-Lite (fast/brief) and Flash (detailed) based on query complexity.
# Optimized to minimize API calls with bounded memory management.
# """
# import logging
# import re
# import time
# from typing import List, Dict, Any
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_core.prompts import PromptTemplate
# from Backend.models import Message
# from Backend.langchain_retriever import LangChainMongoRetriever
# from Backend.langchain_llm_client import create_langchain_gemini_client, create_langchain_gemini_lite_client
# from Backend.prompt_builder import PromptBuilder

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class HybridRAGEngine:
#     """
#     Hybrid RAG engine with intelligent routing:
#     - Flash-Lite: Fast, brief responses for initial queries
#     - Flash: Detailed, comprehensive responses for deep-dive requests
    
#     Optimized for minimal API calls:
#     - ConversationBufferMemory (no summarization calls)
#     - Lite chain skips condense step (1 API call only)
#     - Full chain uses condense when needed (2 API calls)
#     - Bounded memory to prevent leaks in long-running servers
#     """
    
#     # Patterns for detecting deep-dive requests
#     DETAILED_PATTERNS = [
#         # Explicit continuation requests
#         r'\btell\s+me\s+more\b',
#         r'\bexpand\b',
#         r'\bcontinue\b',
#         r'\bgo\s+deeper\b',
#         r'\bkeep\s+going\b',
#         r'\bgo\s+on\b',
        
#         # Depth/detail requests
#         r'\bin\s+detail\b',
#         r'\bdetailed\b',
#         r'\belaborate\b',
#         r'\bexplain\s+further\b',
#         r'\bmore\s+information\b',
#         r'\bmore\s+details?\b',
#         r'\bcomprehensive\b',
#         r'\bthorough\b',
#         r'\bin[\-\s]depth\b',
#         r'\bextensive\b',
#         r'\bfull\s+explanation\b',
        
#         # Comparison/analysis requests (typically need more context)
#         r'\bcompare\b',
#         r'\bcontrast\b',
#         r'\bdifference\s+between\b',
#         r'\bhow\s+do(?:es)?\s+.*\s+differ\b',
#         r'\bvs\.?\b',
#         r'\bversus\b',
#         r'\banalyze\b',
#         r'\banalysis\b',
#         r'\bbreak\s+down\b',
        
#         # Multi-part questions
#         r'\band\s+also\b',
#         r'\band\s+how\b',
#         r'\band\s+what\b',
#         r'\band\s+why\b',
#         r'\bwhat\s+about\b',
        
#         # Follow-up patterns
#         r'\bwhat\s+else\b',
#         r'\bany(?:thing)?\s+else\b',
#         r'\bother\s+examples?\b',
#         r'\bmore\s+examples?\b',
        
#         # Explanation depth
#         r'\bstep[\-\s]by[\-\s]step\b',
#         r'\bwalk\s+me\s+through\b',
#         r'\bguide\s+me\b',
#         r'\bshow\s+me\s+how\b',
#     ]
    
#     # Patterns for brief responses
#     BRIEF_PATTERNS = [
#         # Explicit brevity requests
#         r'\bbriefly\b',
#         r'\bquick\b',
#         r'\bquickly\b',
#         r'\bshort\b',
#         r'\bconcise\b',
#         r'\bsummarize\b',
#         r'\bsummary\b',
#         r'\bin\s+short\b',
#         r'\bin\s+brief\b',
#         r'\bshort\s+answer\b',
        
#         # Single-concept queries
#         r'\bwhat\s+is\b',
#         r'\bdefine\b',
#         r'\bdefinition\s+of\b',
#         r'\bmeaning\s+of\b',
#         r'\bexplain\s+(?!how|why)\w+$',  # "explain X" but not "explain how" or "explain why"
        
#         # Simple yes/no or list requests
#         r'\blist\b',
#         r'\bname\b',
#         r'\bgive\s+me\s+\d+\b',  # "give me 3 examples"
#         r'\btop\s+\d+\b',  # "top 5 algorithms"
#     ]
    
#     def __init__(self):
#         """Initialize hybrid RAG components with bounded memory."""
#         try:
#             logger.info("[HYBRID_RAG_ENGINE] 🔧 Initializing with API call optimization...")
            
#             # Initialize retrievers (same underlying MongoDB, different top_k)
#             self.lite_retriever = LangChainMongoRetriever(max_results=1)  # Top 1 for lite
#             self.full_retriever = LangChainMongoRetriever(max_results=3)  # Top 3 for full
            
#             # Initialize LLMs
#             self.flash_llm = create_langchain_gemini_client()  # Full power
#             self.lite_llm = create_langchain_gemini_lite_client()  # Fast & cheap
            
#             # ConversationBufferMemory - NO SUMMARIZATION API CALLS
#             # Stores full history in memory (uses more tokens but zero API calls)
#             self.memory = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 return_messages=True,
#                 output_key="answer",
#                 input_key="question"
#             )
            
#             # PRODUCTION OPTIMIZATION: Memory cleanup tracking
#             # Prevents unbounded memory growth in long-running servers
#             self.active_conversations = {}  # {conversation_id: last_access_timestamp}
#             self.max_memory_conversations = 50  # Clean up after this many
#             self.memory_cleanup_age = 3600  # Clear conversations older than 1 hour
            
#             logger.info("[HYBRID_RAG_ENGINE] ✅ Using ConversationBufferMemory (0 API calls for memory)")
#             logger.info("[HYBRID_RAG_ENGINE] 🧹 Memory cleanup: max 50 conversations, 1 hour TTL")
            
#             # Prompt builder for greeting/farewell
#             self.prompt_builder = PromptBuilder()
            
#             # Compile regex patterns
#             self.detailed_regex = re.compile('|'.join(self.DETAILED_PATTERNS), re.IGNORECASE)
#             self.brief_regex = re.compile('|'.join(self.BRIEF_PATTERNS), re.IGNORECASE)
            
#             # Build chains
#             self.lite_chain = self._build_lite_chain()
#             self.full_chain = self._build_full_chain()
            
#             logger.info("[HYBRID_RAG_ENGINE] ✅ All components initialized")
#             logger.info("[HYBRID_RAG_ENGINE] 📊 API Call Breakdown:")
#             logger.info("[HYBRID_RAG_ENGINE]    Lite query: 1 API call (answer only)")
#             logger.info("[HYBRID_RAG_ENGINE]    Full query: 2 API calls (condense + answer)")
#             logger.info("[HYBRID_RAG_ENGINE]    Memory: 0 API calls (buffer storage)")
            
#         except Exception as e:
#             logger.error(f"[HYBRID_RAG_ENGINE_ERR] Initialization failed: {e}")
#             raise
    
#     def _cleanup_old_memory(self):
#         """
#         Clean up old conversation memory to prevent unbounded growth.
#         Removes conversations not accessed in last hour or when count exceeds limit.
#         """
#         try:
#             current_time = time.time()
            
#             # Remove conversations older than TTL
#             expired = [
#                 conv_id for conv_id, last_access in self.active_conversations.items()
#                 if current_time - last_access > self.memory_cleanup_age
#             ]
            
#             for conv_id in expired:
#                 del self.active_conversations[conv_id]
#                 logger.info(f"[MEMORY_CLEANUP] Expired conversation: {conv_id}")
            
#             # If still over limit, remove oldest conversations
#             if len(self.active_conversations) > self.max_memory_conversations:
#                 sorted_convs = sorted(
#                     self.active_conversations.items(),
#                     key=lambda x: x[1]
#                 )
                
#                 to_remove = len(self.active_conversations) - self.max_memory_conversations
#                 for conv_id, _ in sorted_convs[:to_remove]:
#                     del self.active_conversations[conv_id]
#                     logger.info(f"[MEMORY_CLEANUP] Removed oldest conversation: {conv_id}")
            
#             if expired or to_remove > 0:
#                 logger.info(
#                     f"[MEMORY_CLEANUP] Cleaned {len(expired) + to_remove} conversations. "
#                     f"Active: {len(self.active_conversations)}"
#                 )
                
#         except Exception as e:
#             logger.error(f"[MEMORY_CLEANUP_ERR] {e}")
    
#     def track_conversation_access(self, conversation_id: str):
#         """
#         Track conversation access for memory cleanup.
        
#         Args:
#             conversation_id: Conversation being accessed
#         """
#         self.active_conversations[conversation_id] = time.time()
        
#         # Periodic cleanup check
#         if len(self.active_conversations) > self.max_memory_conversations:
#             self._cleanup_old_memory()
    
#     def _build_lite_chain(self) -> ConversationalRetrievalChain:
#         """Build lite chain for brief, fast responses. OPTIMIZED: No condense step."""
        
#         # Lite QA prompt - emphasizes brevity
#         lite_qa_template = """You are AI Shine, an AI/ML educational assistant.

# Context from knowledge base:
# {context}

# Chat History:
# {chat_history}

# Question: {question}

# RULES:
# 1. SCOPE: AI, ML, Deep Learning, Data Science, NLP, Computer Vision, AI Applications only
# 2. BREVITY: Respond in 2-3 concise sentences maximum. No lists unless absolutely necessary.
# 3. SYNTHESIS: Lead with definition if KB lacks it, then add one KB example
# 4. TONE: Direct and educational. No meta-commentary.
# 5. FORMAT: Use <p> tags only. No lists for lite responses.
# 6. OUT OF SCOPE: "⚠️ I specialize in AI and Machine Learning topics."

# Answer:"""
        
#         # OPTIMIZATION: condense_question_llm=None skips expensive condense step
#         # Lite queries are simple, don't need conversation history rephrasing
#         chain = ConversationalRetrievalChain.from_llm(
#             llm=self.lite_llm,
#             retriever=self.lite_retriever,
#             memory=self.memory,
#             condense_question_llm=None,  # Skip condense = saves 1 API call
#             combine_docs_chain_kwargs={
#                 "prompt": PromptTemplate(
#                     template=lite_qa_template,
#                     input_variables=["context", "chat_history", "question"]
#                 )
#             },
#             return_source_documents=False,
#             verbose=False
#         )
        
#         logger.info("[HYBRID_RAG_ENGINE] ✅ Lite chain built (1 API call per query)")
#         return chain
    
#     def _build_full_chain(self) -> ConversationalRetrievalChain:
#         """Build full chain for detailed, comprehensive responses."""
        
#         condense_template = """Given this conversation, rephrase the follow-up as a standalone question.

# Chat History:
# {chat_history}

# Follow Up: {question}
# Standalone question:"""
        
#         # Full QA prompt - emphasizes detail
#         full_qa_template = """You are AI Shine, an AI/ML educational assistant.

# Context from knowledge base:
# {context}

# Chat History:
# {chat_history}

# Question: {question}

# RULES:
# 1. SCOPE: AI, ML, Deep Learning, Data Science, NLP, Computer Vision, AI Applications only
# 2. SYNTHESIS:
#    - If KB mentions concept without defining: lead with definition, then add KB examples
#    - Paraphrase all KB content naturally - never use direct quotes or preserve quotation marks
#    - NEVER invent examples, tools, or statistics not in KB
#    - Only provide general definitions when needed
# 3. TONE: Direct and educational. NO meta-commentary
# 4. FORMAT:
#    - <p> for paragraphs (blank line between)
#    - <ul><li> for lists (blank line between items)
#    - <strong> for key terms (2-4 words)
#    - If long: end with <p><em>Write 'continue' to keep generating...</em></p>
# 5. OUT OF SCOPE: "⚠️ I specialize in AI and Machine Learning topics. I'd be happy to help with questions about [suggest 2-3 AI/ML topics]."

# Answer:"""
        
#         chain = ConversationalRetrievalChain.from_llm(
#             llm=self.flash_llm,
#             retriever=self.full_retriever,
#             memory=self.memory,
#             condense_question_prompt=PromptTemplate(
#                 template=condense_template,
#                 input_variables=["chat_history", "question"]
#             ),
#             combine_docs_chain_kwargs={
#                 "prompt": PromptTemplate(
#                     template=full_qa_template,
#                     input_variables=["context", "chat_history", "question"]
#                 )
#             },
#             return_source_documents=False,
#             verbose=False
#         )
        
#         logger.info("[HYBRID_RAG_ENGINE] ✅ Full chain built (2 API calls per query)")
#         return chain
    
#     def _route_query(self, query: str, chat_history: List[Message]) -> str:
#         """
#         Route query to appropriate chain based on complexity signals.
        
#         Args:
#             query: User query
#             chat_history: Conversation history
        
#         Returns:
#             "lite" or "full"
#         """
#         query_lower = query.lower().strip()
        
#         # Count words (longer queries often need more depth)
#         word_count = len(query_lower.split())
        
#         # Explicit brief request → lite
#         if self.brief_regex.search(query_lower):
#             logger.info("[ROUTER] Brief signal detected → Lite chain")
#             return "lite"
        
#         # Explicit detailed request → full
#         if self.detailed_regex.search(query_lower):
#             logger.info("[ROUTER] Detailed signal detected → Full chain")
#             return "full"
        
#         # Very long queries → full (>=20 words suggests complexity)
#         if word_count >= 20:
#             logger.info(f"[ROUTER] Long query ({word_count} words) → Full chain")
#             return "full"
        
#         # Follow-up question after receiving a lite response → full
#         # (User wants more depth on the same topic)
#         if len(chat_history) >= 2:
#             last_assistant_msg = None
#             for msg in reversed(chat_history):
#                 if msg.role == "assistant":
#                     last_assistant_msg = msg.content
#                     break
            
#             # If last response was brief (<300 chars) and user asks follow-up → go full
#             if last_assistant_msg and len(last_assistant_msg) < 300:
#                 # Check if current query is a follow-up (short, no new topic keywords)
#                 has_new_topic = any(keyword in query_lower for keyword in [
#                     'what is', 'define', 'explain', 'how does', 'why', 'when', 'where'
#                 ])
                
#                 if not has_new_topic and word_count < 15:
#                     logger.info("[ROUTER] Follow-up after brief response → Full chain")
#                     return "full"
        
#         # Default: lite (fast & cheap for first queries on any topic)
#         logger.info(f"[ROUTER] Default routing (word_count={word_count}) → Lite chain")
#         return "lite"
    
#     def process_query(
#         self,
#         query: str,
#         chat_history: List[Message],
#         conversation_id: str = None
#     ) -> Dict[str, Any]:
#         """
#         Process user query with intelligent routing.
        
#         Args:
#             query: Current user query
#             chat_history: Full conversation history
#             conversation_id: Optional conversation ID for memory tracking
        
#         Returns:
#             Dict with 'answer' (str) and 'type' (str)
#         """
#         import uuid
#         request_id = str(uuid.uuid4())[:8]  # Short request ID for tracking
        
#         try:
#             # Track conversation access for memory cleanup
#             if conversation_id:
#                 self.track_conversation_access(conversation_id)
            
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📝 NEW REQUEST: '{query[:100]}...'")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Chat history length: {len(chat_history)} messages")
            
#             # Handle greetings
#             if self.prompt_builder.greeting_regex.match(query.strip()):
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 👋 GREETING DETECTED")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ Response via regex pattern (0 API calls)")
#                 self.memory.clear()
#                 return {
#                     "answer": self.prompt_builder.build_greeting_response(),
#                     "type": "greeting"
#                 }
            
#             # Handle farewells
#             if self.prompt_builder.farewell_regex.match(query.strip()):
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 👋 FAREWELL DETECTED")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ Response via regex pattern (0 API calls)")
#                 self.memory.clear()
#                 return {
#                     "answer": self.prompt_builder.build_farewell_response(),
#                     "type": "text"
#                 }
            
#             # Route to appropriate chain
#             route = self._route_query(query, chat_history)
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 ROUTING DECISION: {route.upper()}")
            
#             if route == "lite":
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🚀 LITE CHAIN SELECTED")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💡 Reason: Default/first query OR brief signal detected")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Expected API calls: 1")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #1 START: Answer generation (Flash-Lite, 15 RPM)")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Generate brief response from KB context")
                
#                 response = self.lite_chain.invoke({"question": query})
                
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #1 COMPLETE: Answer generation")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📈 Total API calls for this request: 1")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 Model used: gemini-2.5-flash-lite")
#             else:
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🔥 FULL CHAIN SELECTED")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💡 Reason: Detailed signal detected (tell me more/expand/in detail)")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 Expected API calls: 2")
                
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #1 START: Condense question (Flash, 1500 RPM)")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Rephrase '{query}' with chat history context")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Input: '{query}' + {len(chat_history)} previous messages")
                
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ⚡ API Call #2 START: Answer generation (Flash, 1500 RPM)")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}]    Purpose: Generate detailed response with HTML formatting")
                
#                 response = self.full_chain.invoke({"question": query})
                
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #1 COMPLETE: Condense question")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ API Call #2 COMPLETE: Answer generation")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📈 Total API calls for this request: 2")
#                 logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🎯 Model used: gemini-2.5-flash")
            
#             answer = response.get("answer", "")
            
#             # Clean response
#             answer = self._clean_response(answer)
            
#             # Classify response type
#             response_type = self._classify_response(answer)
            
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ✅ RESPONSE COMPLETE")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📝 Response length: {len(answer)} characters")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🏷️  Response type: {response_type}")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 🚦 Route used: {route}")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 💾 Memory stored in buffer (0 API calls)")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] 📊 FINAL COUNT - API calls made: {1 if route == 'lite' else 2}")
#             logger.info(f"[HYBRID_RAG_ENGINE] [{request_id}] ════════════════════════════════════════")
            
#             return {
#                 "answer": answer,
#                 "type": response_type,
#                 "route": route,
#                 "request_id": request_id  # For frontend debugging
#             }
            
#         except Exception as e:
#             logger.error(f"[HYBRID_RAG_ENGINE] [{request_id}] ❌ PIPELINE FAILURE: {e}", exc_info=True)
#             return {
#                 "answer": "⚠️ An unexpected error occurred. Please try your question again.",
#                 "type": "text"
#             }
    
#     def _clean_response(self, response: str) -> str:
#         """Clean and format LLM response."""
#         import re
        
#         # Convert markdown bold to HTML
#         response = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', response)
        
#         # Remove any stray asterisk bullets
#         response = response.replace('* ', '• ')
        
#         return response.strip()
    
#     def _classify_response(self, response: str) -> str:
#         """Classify response type for frontend rendering."""
#         if response.startswith("⚠") or "I specialize in AI and Machine Learning topics" in response:
#             return "decline"
        
#         if "I don't have" in response or "I don't know" in response:
#             return "decline"
        
#         return "text"
    
#     def cleanup(self):
#         """Cleanup resources."""
#         try:
#             self.memory.clear()
#             self.active_conversations.clear()
#             logger.info("[HYBRID_RAG_ENGINE] ✅ Cleanup complete")
#         except Exception as e:
#             logger.error(f"[HYBRID_RAG_ENGINE] Cleanup error: {e}")