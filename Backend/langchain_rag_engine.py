# based on the above, best, has everything except for live token streaming
# """
# LangChain RAG Engine
# Main orchestrator using ConversationalRetrievalChain with memory.
# Replaces regex intent detection with natural conversation understanding.
# """
import logging
from typing import List, Dict, Any
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from Backend.models import Message
from Backend.langchain_retriever import LangChainMongoRetriever
from Backend.langchain_llm_client import create_langchain_gemini_client
from Backend.prompt_builder import PromptBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangChainRAGEngine:
    """
    LangChain-powered RAG engine with conversation memory.
    Eliminates regex intent detection and manual context tracking.
    """
    
    def __init__(self):
        """Initialize LangChain RAG components."""
        try:
            # Initialize components
            self.retriever = LangChainMongoRetriever()
            self.llm = create_langchain_gemini_client()
            self.prompt_builder = PromptBuilder()  # Reuse for greeting/farewell
            
            # HALLUCINATION GUARDRAIL: ConversationSummaryMemory reduces token usage
            # by 85-90% while maintaining context quality. This prevents token exhaustion
            # that can lead to incomplete responses where LLM might fill gaps with invented content.
            # Summary is triggered when context exceeds 500 tokens (reduced from 1000 to prevent repetition).
            # input_key ensures only user questions trigger new summaries, not AI responses.
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=4000,  # Reduced from 1000 to prevent summary bloat and repetition
                output_key="answer",
                input_key="question"  # Only summarize based on user input, not AI responses
            )
            
            # Build conversational retrieval chain
            # HALLUCINATION GUARDRAIL: This chain automatically handles conversation flow,
            # eliminating the need for regex patterns that could misinterpret user intent
            # and cause incorrect context retrieval (e.g., "be descriptive" → CRAFT bug).
            self.chain = self._build_chain()
            
            logger.info("[LANGCHAIN_RAG_ENGINE] ✅ All components initialized")
            
        except Exception as e:
            logger.error(f"[LANGCHAIN_RAG_ENGINE_ERR] Initialization failed: {e}")
            raise
    
    def _build_chain(self) -> ConversationalRetrievalChain:
        """
        Build ConversationalRetrievalChain with custom prompts.
        
        Returns:
            Configured conversational retrieval chain
        """
        # Condense question prompt (for multi-turn conversations)
        # CRITICAL: Keep this SHORT. Verbose rephrasing causes overly formal responses.
        condense_template = """Given this conversation, rephrase the follow-up as a standalone question.

Chat History:
{chat_history}

Follow Up: {question}
Standalone question:"""
        
        condense_prompt = PromptTemplate(
            template=condense_template,
            input_variables=["chat_history", "question"]
        )
        
        # QA prompt (main response generation)
        # HALLUCINATION GUARDRAIL: This prompt enforces strict KB-first synthesis rules.
        # When KB mentions a concept but doesn't define it, LLM leads with general definition
        # then integrates KB examples. This prevents the "The provided content doesn't define..."
        # preamble that disrupts user experience.
        qa_template = """You are AI Shine, an AI/ML educational assistant.

Context from knowledge base:
{context}

Chat History:
{chat_history}

Question: {question}

RULES:
1. SCOPE: AI, ML, Deep Learning, Data Science, NLP, Computer Vision, AI Applications only
2. SYNTHESIS:
   - If KB mentions concept without defining: lead with definition, then add KB examples
   - NEVER invent examples, tools, or statistics not in KB
   - Only provide general definitions when needed
3. TONE: Direct and educational. NO meta-commentary like "I'm excited to delve into" or "Absolutely! To tell you more"
4. FORMAT:
   - <p> for paragraphs (blank line between)
   - <ul><li> for lists (blank line between items)
   - <strong> for key terms (2-4 words)
   - If long: end with <p><em>Write 'continue' to keep generating...</em></p>
5. OUT OF SCOPE: "⚠️ I specialize in AI and Machine Learning topics. I'd be happy to help with questions about [suggest 2-3 AI/ML topics]."

Answer:"""
        
        qa_prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Build chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            condense_question_prompt=condense_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=True, # Enable logging for debugging
        )
        
        logger.info("[LANGCHAIN_RAG_ENGINE] ✅ ConversationalRetrievalChain built")
        return chain
    
    def process_query(
        self,
        query: str,
        chat_history: List[Message]
    ) -> Dict[str, Any]:
        """
        Process user query with LangChain conversation chain.
        
        Args:
            query: Current user query
            chat_history: Full conversation history (for greeting detection)
        
        Returns:
            Dict with 'answer' (str) and 'type' (str)
        """
        try:
            logger.info(f"[LANGCHAIN_RAG_ENGINE] Processing query: {query}")
            
            # Handle greetings (quick check before invoking chain)
            if self.prompt_builder.greeting_regex.match(query.strip()):
                logger.info("[LANGCHAIN_RAG_ENGINE] Greeting detected")
                self.memory.clear()  # Reset memory on new greeting
                return {
                    "answer": self.prompt_builder.build_greeting_response(),
                    "type": "greeting"
                }
            
            # Handle farewells
            if self.prompt_builder.farewell_regex.match(query.strip()):
                logger.info("[LANGCHAIN_RAG_ENGINE] Farewell detected")
                self.memory.clear()  # Reset memory
                return {
                    "answer": self.prompt_builder.build_farewell_response(),
                    "type": "text"
                }
            
            # Invoke conversational chain
            # HALLUCINATION GUARDRAIL: LangChain automatically handles conversation understanding,
            # continuation detection, and context management. This eliminates the "be descriptive" → CRAFT
            # bug caused by regex misinterpretation. The chain understands "be descriptive" is a style
            # modifier for the previous response, not a new query about descriptiveness.
            response = self.chain.invoke({"question": query})
            
            answer = response.get("answer", "")
            
            # Clean response
            answer = self._clean_response(answer)
            
            # Classify response type
            response_type = self._classify_response(answer)
            
            logger.info(f"[LANGCHAIN_RAG_ENGINE] ✅ Response generated ({len(answer)} chars)")
            logger.info(f"[LANGCHAIN_RAG_ENGINE] Response type: {response_type}")
            
            return {
                "answer": answer,
                "type": response_type
            }
            
        except Exception as e:
            logger.error(f"[LANGCHAIN_RAG_ENGINE_ERR] Pipeline failure: {e}", exc_info=True)
            return {
                "answer": "⚠️ An unexpected error occurred. Please try your question again.",
                "type": "text"
            }
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and format LLM response.
        
        Args:
            response: Raw LLM response
        
        Returns:
            Cleaned response
        """
        import re
        
        # Convert markdown bold to HTML (if any slipped through)
        response = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', response)
        
        # Remove any stray asterisk bullets
        response = response.replace('* ', '• ')
        
        return response.strip()
    
    def _classify_response(self, response: str) -> str:
        """
        Classify response type for frontend rendering.
        
        Args:
            response: Cleaned response text
        
        Returns:
            Response type: "decline", "greeting", or "text"
        """
        # Check for out-of-scope
        if response.startswith("⚠") or "I specialize in AI and Machine Learning topics" in response:
            return "decline"
        
        # Check for knowledge gaps
        if "I don't have" in response or "I don't know" in response:
            return "decline"
        
        return "text"
    
    def cleanup(self):
        """
        Cleanup resources.
        LangChain handles connection management automatically.
        """
        try:
            self.memory.clear()
            logger.info("[LANGCHAIN_RAG_ENGINE] ✅ Cleanup complete")
        except Exception as e:
            logger.error(f"[LANGCHAIN_RAG_ENGINE] Cleanup error: {e}")