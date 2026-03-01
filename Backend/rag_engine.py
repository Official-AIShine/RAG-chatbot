"""
Optimized RAG Engine for Low Latency

Key optimizations:
1. Single LLM call per query (no condense step)
2. Sliding window memory (prevents context explosion)
3. Parallel retrieval with async support
4. Optional multi-turn retrieval toggle
5. Graceful degradation on failures
6. Continuation handling (tell me more) with descriptive responses
7. Gemini fallback for low-confidence queries
8. Configurable response lengths
9. Extended thinking mode (optional)
"""
import logging
import re
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator
from concurrent.futures import ThreadPoolExecutor

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from Backend.retriever import OptimizedRetriever
from Backend.prompt_builder import PromptBuilder
from Backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Production RAG engine optimized for low latency.

    Architecture:
    - Single LLM call per query (eliminates condense step overhead)
    - Direct history injection (no memory summarization API calls)
    - Sliding window for long conversations
    - Optional multi-turn retrieval for context-aware search
    - Continuation handling for "tell me more" queries with descriptive responses
    - Gemini fallback for topics not in KB
    - Configurable response lengths
    """

    def __init__(self, multi_turn_retrieval: bool = False):
        """
        Initialize RAG engine.

        Args:
            multi_turn_retrieval: If True, considers conversation history for retrieval.
                                  If False, only uses current query (faster).
        """
        try:
            logger.info("[RAG_ENGINE] Initializing...")

            # Configuration
            self.multi_turn_retrieval = multi_turn_retrieval
            self.sliding_window_size = settings.SLIDING_WINDOW_SIZE
            self.extended_thinking = settings.EXTENDED_THINKING_ENABLED

            # Initialize retriever
            self.retriever = OptimizedRetriever(
                collection_name=settings.VECTOR_COLLECTION,
                primary_threshold=settings.PRIMARY_SIMILARITY_THRESHOLD,
                fallback_threshold=settings.FALLBACK_SIMILARITY_THRESHOLD,
                top_k=settings.RETRIEVAL_TOP_K
            )

            # Initialize LLM with streaming
            self.llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                temperature=settings.GEMINI_TEMPERATURE,
                max_output_tokens=settings.GEMINI_MAX_TOKENS,
                streaming=True
            )

            # Prompt builder for greeting/farewell detection
            self.prompt_builder = PromptBuilder()

            # Thread pool for async operations
            self.executor = ThreadPoolExecutor(max_workers=4)

            # Conversation tracking for cleanup
            self.active_conversations: Dict[str, float] = {}
            self.max_conversations = settings.MAX_ACTIVE_CONVERSATIONS

            # Build the chains
            self.chain = self._build_chain()
            self.continuation_chain = self._build_continuation_chain()
            self.fallback_chain = self._build_fallback_chain()

            logger.info("[RAG_ENGINE] Initialized successfully")
            logger.info(f"[RAG_ENGINE] Multi-turn retrieval: {multi_turn_retrieval}")
            logger.info(f"[RAG_ENGINE] Extended thinking: {self.extended_thinking}")
            logger.info(f"[RAG_ENGINE] Sliding window: {self.sliding_window_size} messages")
            logger.info(f"[RAG_ENGINE] Descriptive min words: {settings.DESCRIPTIVE_RESPONSE_MIN_WORDS}")

        except Exception as e:
            logger.error(f"[RAG_ENGINE] Initialization failed: {e}")
            raise

    def _build_chain(self):
        """
        Build optimized chain with single LLM call for regular queries.
        """
        system_template = f"""You are AI Shine, an educational AI assistant. Your PRIMARY source of truth is the knowledge base context below.

KNOWLEDGE BASE CONTEXT:
{{context}}

CONVERSATION HISTORY:
{{history}}

INSTRUCTIONS:
1. If KNOWLEDGE BASE CONTEXT contains information relevant to the question, answer using ONLY that content. Do not add outside information.
2. If the context is empty or says "(No relevant content found in knowledge base)", respond with exactly: FALLBACK_NEEDED
3. For completely unrelated questions (cooking, sports, politics), respond with: "⚠️ I specialize in AI and Machine Learning topics."
4. Target approximately {settings.STANDARD_RESPONSE_TARGET_WORDS} words.
5. If response exceeds {settings.RESPONSE_TRUNCATE_THRESHOLD} words, end with: <p><em>Type 'continue' for more...</em></p>

HTML FORMAT (required):
- Every paragraph: <p>text</p>
- Lists: <ul><li>item</li></ul> or <ol><li>item</li></ol>
- Bold terms: <strong>term</strong>
- No markdown bullets"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain

    def _build_continuation_chain(self):
        """
        Build chain specifically for continuation requests ("tell me more").
        Produces longer, more descriptive responses (2-3 paragraphs).
        """
        system_template = f"""You are AI Shine, an educational assistant. The user is asking for MORE DETAIL about a topic from the conversation.

CONVERSATION HISTORY (what was discussed before):
{{history}}

CONTEXT FROM KNOWLEDGE BASE:
{{context}}

The user said "{{question}}" - this is a CONTINUATION REQUEST meaning they want you to EXPAND on the last topic discussed.

YOUR TASK:
1. Find the LAST TOPIC in the conversation history above (look for what Assistant explained).
2. Provide a DETAILED, EXPANDED explanation with NEW information.
3. Your response MUST be at least {settings.DESCRIPTIVE_RESPONSE_MIN_WORDS} words (2-3 substantial paragraphs).
4. Add VALUE with:
   - Deeper technical explanations
   - Practical examples and use cases
   - Step-by-step breakdowns
   - Related concepts and tips
5. Use BOTH the knowledge base context AND your general knowledge.
6. Do NOT repeat what was already said - ADD NEW information.
7. If response exceeds {settings.RESPONSE_TRUNCATE_THRESHOLD} words, end with: <p><em>Type 'continue' for more...</em></p>

IMPORTANT: The conversation history contains the previous discussion. Look at what the Assistant said and expand on THAT topic.

HTML FORMATTING (STRICT):
- Wrap ALL text in <p>...</p> tags
- Lists: <ul><li>item</li></ul>
- Bold: <strong>term</strong>
- NO markdown bullets (•, -, *)"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain

    def _build_fallback_chain(self):
        """
        Build fallback chain for when KB has no relevant content.
        Uses Gemini's knowledge with clear attribution.
        """
        fallback_template = f"""You are AI Shine, an educational AI assistant. The knowledge base did not contain a direct answer to this question, so you are responding from general knowledge.

CONVERSATION HISTORY:
{{history}}

USER QUESTION: {{question}}

INSTRUCTIONS:
1. Answer using your general knowledge about AI, ML, technology, and educational tools.
2. Start your response with: "<p><em>📚 This topic isn't in my course materials, but here's what I know:</em></p>"
3. Provide at least {settings.STANDARD_RESPONSE_TARGET_WORDS} words.
4. Be accurate. Do not fabricate statistics or specific product claims.
5. For completely unrelated questions (cooking, sports, politics, medical/legal advice), respond only with: "⚠️ I specialize in AI and Machine Learning topics."
6. If response exceeds {settings.RESPONSE_TRUNCATE_THRESHOLD} words, end with: <p><em>Type 'continue' for more...</em></p>

HTML FORMAT (required):
- Every paragraph: <p>text</p>
- Lists: <ul><li>item</li></ul> or <ol><li>item</li></ol>
- Bold terms: <strong>term</strong>
- No markdown bullets"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", fallback_template),
            ("human", "{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain

    def _format_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Format chat history for context injection.
        Uses sliding window to prevent context explosion.
        """
        if not chat_history:
            return "(No previous conversation)"

        # Apply sliding window
        recent = chat_history[-self.sliding_window_size:]

        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "human" else "Assistant"
            content = msg["content"][:500]  # Truncate long messages
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _is_continuation_query(self, query: str) -> bool:
        """Check if query is a continuation request."""
        continuation_patterns = [
            r'^\s*(tell\s+me\s+more|more|continue|go\s+on|elaborate|expand|explain\s+further)\s*[.!?]?\s*$',
            r'^\s*(what\s+else|anything\s+else|more\s+details?|keep\s+going)\s*[.!?]?\s*$',
            r'^\s*(and\??|yes|ok\s+continue|go\s+ahead)\s*[.!?]?\s*$',
        ]
        query_lower = query.lower().strip()
        for pattern in continuation_patterns:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return True
        return False

    def _extract_last_topic(self, chat_history: List[Dict[str, str]]) -> Optional[str]:
        """
        Extract the main topic from the last exchange.
        """
        if not chat_history:
            return None

        # Find last user query that wasn't a continuation
        for msg in reversed(chat_history):
            if msg["role"] == "human":
                content = msg["content"].strip()
                if not self._is_continuation_query(content):
                    return content

        # Fallback: extract from last AI response
        for msg in reversed(chat_history):
            if msg["role"] == "ai":
                content = msg["content"]
                content = re.sub(r'<[^>]+>', '', content)
                match = re.match(r'^([^.!?]+[.!?])', content)
                if match:
                    return match.group(1)[:200]
                return content[:200]

        return None

    def _build_retrieval_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Build query for retrieval.
        Handles continuation queries by using last topic.
        """
        # Handle continuation queries
        if self._is_continuation_query(query) and chat_history:
            last_topic = self._extract_last_topic(chat_history)
            if last_topic:
                logger.info(f"[RAG_ENGINE] Continuation detected, using topic: {last_topic[:50]}...")
                return last_topic

        # Multi-turn retrieval
        if self.multi_turn_retrieval and chat_history:
            context_parts = [query]
            for msg in chat_history[-2:]:
                if msg["role"] == "human":
                    context_parts.append(msg["content"][:200])
            return " ".join(context_parts)

        return query

    async def _retrieve_async(self, query: str) -> List[Dict[str, Any]]:
        """Async retrieval using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.retriever.retrieve,
            query
        )

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for prompt."""
        if not documents:
            return "(No relevant content found in knowledge base)"

        parts = []
        for i, doc in enumerate(documents, 1):
            topic = doc.get("metadata", {}).get("topic", "Unknown")
            content = doc.get("content", "")[:1500]  # Allow more content for continuations
            score = doc.get("score", 0)
            parts.append(f"[{i}] {topic} (relevance: {score:.2f})\n{content}")

        return "\n\n".join(parts)

    def _clean_response(self, response: str) -> str:
        """Clean and format LLM response."""
        # Convert markdown bold to HTML
        response = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', response)
        response = response.replace('* ', '• ')
        return response.strip()

    def get_greeting(self) -> str:
        """Get greeting response."""
        return self.prompt_builder.build_greeting_response()

    def get_farewell(self) -> str:
        """Get farewell response."""
        return self.prompt_builder.build_farewell_response()

    def is_greeting(self, query: str) -> bool:
        """Check if query is a greeting."""
        return bool(self.prompt_builder.greeting_regex.match(query.strip()))

    def is_farewell(self, query: str) -> bool:
        """Check if query is a farewell."""
        return bool(self.prompt_builder.farewell_regex.match(query.strip()))

    def set_extended_thinking(self, enabled: bool):
        """Toggle extended thinking mode at runtime."""
        self.extended_thinking = enabled
        logger.info(f"[RAG_ENGINE] Extended thinking: {enabled}")

    async def process_query_stream(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None,
        conversation_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Process query with streaming response.
        """
        start_time = time.time()
        chat_history = chat_history or []

        try:
            # Track conversation
            if conversation_id:
                self.active_conversations[conversation_id] = time.time()
                self._cleanup_old_conversations()

            logger.info(f"[RAG_ENGINE] Query: {query[:100]}...")

            # Handle greetings
            if self.is_greeting(query):
                logger.info("[RAG_ENGINE] Greeting detected")
                yield self.get_greeting()
                return

            # Handle farewells
            if self.is_farewell(query):
                logger.info("[RAG_ENGINE] Farewell detected")
                yield self.get_farewell()
                return

            # Check if this is a continuation query
            is_continuation = self._is_continuation_query(query)
            if is_continuation:
                logger.info("[RAG_ENGINE] Continuation query detected - will use descriptive chain")

            # Build retrieval query (handles continuations)
            retrieval_query = self._build_retrieval_query(query, chat_history)

            # Retrieve documents (async)
            retrieval_start = time.time()
            documents = await self._retrieve_async(retrieval_query)
            retrieval_time = (time.time() - retrieval_start) * 1000

            logger.info(f"[RAG_ENGINE] Retrieved {len(documents)} docs in {retrieval_time:.0f}ms")

            # Check confidence level
            max_score = max((d.get("score", 0) for d in documents), default=0)

            # Format inputs
            context = self._format_context(documents)
            history = self._format_history(chat_history)

            # Stream from appropriate chain
            llm_start = time.time()
            response_buffer = []

            # Choose chain based on query type
            if is_continuation:
                # Use continuation chain for descriptive responses
                # Works with or without documents - uses history + general knowledge
                logger.info("[RAG_ENGINE] Using CONTINUATION chain (descriptive)")
                logger.info(f"[RAG_ENGINE] History length: {len(chat_history)}, Docs: {len(documents)}")
                async for chunk in self.continuation_chain.astream({
                    "context": context,
                    "history": history,
                    "question": query
                }):
                    if chunk:
                        response_buffer.append(chunk)
            else:
                # Use regular chain
                async for chunk in self.chain.astream({
                    "context": context,
                    "history": history,
                    "question": query
                }):
                    if chunk:
                        response_buffer.append(chunk)

            # Check if we need fallback
            full_response = "".join(response_buffer)

            if "FALLBACK_NEEDED" in full_response:
                logger.info("[RAG_ENGINE] Triggering Gemini fallback")
                async for chunk in self.fallback_chain.astream({
                    "history": history,
                    "question": query
                }):
                    if chunk:
                        cleaned = self._clean_response(chunk)
                        if cleaned:
                            yield cleaned
            else:
                # Stream the buffered response
                for chunk in response_buffer:
                    cleaned = self._clean_response(chunk)
                    if cleaned:
                        yield cleaned

            llm_time = (time.time() - llm_start) * 1000
            total_time = (time.time() - start_time) * 1000

            logger.info(f"[RAG_ENGINE] Complete - Retrieval: {retrieval_time:.0f}ms, LLM: {llm_time:.0f}ms, Total: {total_time:.0f}ms")

        except Exception as e:
            logger.error(f"[RAG_ENGINE] Error: {e}", exc_info=True)
            yield "⚠️ An error occurred. Please try again."

    def process_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process query synchronously (non-streaming)."""
        import asyncio

        async def collect_response():
            chunks = []
            async for chunk in self.process_query_stream(query, chat_history, conversation_id):
                chunks.append(chunk)
            return "".join(chunks)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(collect_response())

        response_type = "text"
        if response.startswith("⚠️") or "I specialize in AI and Machine Learning" in response:
            response_type = "decline"
        elif response.startswith("👋"):
            response_type = "greeting"
        elif "This topic isn't in my course materials" in response:
            response_type = "fallback"

        return {
            "answer": response,
            "type": response_type
        }

    def set_multi_turn_retrieval(self, enabled: bool):
        """Toggle multi-turn retrieval at runtime."""
        self.multi_turn_retrieval = enabled
        logger.info(f"[RAG_ENGINE] Multi-turn retrieval: {enabled}")

    def clear_memory(self):
        """Clear conversation tracking."""
        self.active_conversations.clear()
        logger.info("[RAG_ENGINE] Memory cleared")

    def _cleanup_old_conversations(self):
        """Remove old conversation entries."""
        cutoff = time.time() - settings.CONVERSATION_TTL_SECONDS

        expired = [
            conv_id for conv_id, last_access
            in self.active_conversations.items()
            if last_access < cutoff
        ]

        for conv_id in expired:
            del self.active_conversations[conv_id]

        if expired:
            logger.info(f"[RAG_ENGINE] Cleaned up {len(expired)} old conversations")

        if len(self.active_conversations) > self.max_conversations:
            sorted_convs = sorted(
                self.active_conversations.items(),
                key=lambda x: x[1]
            )
            to_remove = len(self.active_conversations) - self.max_conversations

            for conv_id, _ in sorted_convs[:to_remove]:
                del self.active_conversations[conv_id]

            logger.info(f"[RAG_ENGINE] Removed {to_remove} oldest conversations")

    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)
        self.active_conversations.clear()
        logger.info("[RAG_ENGINE] Cleanup complete")
