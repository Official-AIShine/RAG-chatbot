"""
Prompt Builder for AI Shine Tutor

Handles:
- Greeting/farewell detection
- Response formatting
- Intent classification
"""
import re
import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Prompt builder with greeting/farewell detection.
    Optimized for speed with compiled regex patterns.
    """

    # Greeting patterns
    GREETING_PATTERNS = [
        r'^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|sup|yo)\s*[!.,]?\s*$'
    ]

    # Farewell patterns
    FAREWELL_PATTERNS = [
        r'^\s*(bye|goodbye|see\s+you|farewell|ttyl|later|ciao|adios|thanks|thank\s+you)\s*[!.,]?\s*$'
    ]

    # Continuation patterns (for detecting "tell me more" etc.)
    CONTINUATION_PATTERNS = [
        r'\btell\s+me\s+more\b',
        r'\belaborate\b',
        r'\bgo\s+deeper\b',
        r'\bexpand\b',
        r'\bmore\s+detail\b',
        r'\bexplain\s+further\b',
        r'\bcontinue\b',
        r'\bgo\s+on\b',
        r'^\s*more\s*\??\s*$',
    ]

    def __init__(self):
        """Initialize with compiled regex patterns."""
        self.greeting_regex = re.compile(
            '|'.join(self.GREETING_PATTERNS),
            re.IGNORECASE
        )
        self.farewell_regex = re.compile(
            '|'.join(self.FAREWELL_PATTERNS),
            re.IGNORECASE
        )
        self.continuation_regex = re.compile(
            '|'.join(self.CONTINUATION_PATTERNS),
            re.IGNORECASE
        )

    def detect_intent(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Detect user intent from message.

        Args:
            message: User message
            chat_history: Optional conversation history

        Returns:
            Dict with intent_type, is_greeting, is_farewell, is_continuation
        """
        if not message or not message.strip():
            return {
                "intent_type": "query",
                "is_greeting": False,
                "is_farewell": False,
                "is_continuation": False,
                "confidence": 0.0
            }

        text = message.strip()

        # Check greeting
        if self.greeting_regex.match(text):
            return {
                "intent_type": "greeting",
                "is_greeting": True,
                "is_farewell": False,
                "is_continuation": False,
                "confidence": 1.0
            }

        # Check farewell
        if self.farewell_regex.match(text):
            return {
                "intent_type": "farewell",
                "is_greeting": False,
                "is_farewell": True,
                "is_continuation": False,
                "confidence": 1.0
            }

        # Check continuation
        if self.continuation_regex.search(text):
            return {
                "intent_type": "continuation",
                "is_greeting": False,
                "is_farewell": False,
                "is_continuation": True,
                "confidence": 1.0
            }

        # Default: query
        return {
            "intent_type": "query",
            "is_greeting": False,
            "is_farewell": False,
            "is_continuation": False,
            "confidence": 1.0
        }

    def build_greeting_response(self) -> str:
        """
        Build greeting response.

        Returns:
            HTML-formatted greeting
        """
        return (
            "👋 Hello! I'm <strong>AI Shine</strong>, your AI/ML educational assistant. "
            "Ask me anything about <strong>Artificial Intelligence</strong>, "
            "<strong>Machine Learning</strong>, <strong>Deep Learning</strong>, "
            "<strong>Data Science</strong>, <strong>NLP</strong>, or <strong>Computer Vision</strong>!"
        )

    def build_farewell_response(self) -> str:
        """
        Build farewell response.

        Returns:
            HTML-formatted farewell
        """
        return (
            "👋 Goodbye! It was great helping you learn about <strong>AI</strong> and "
            "<strong>Machine Learning</strong>. Come back anytime!"
        )

