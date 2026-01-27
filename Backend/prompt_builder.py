"""
Prompt Builder
KB-locked, complete enumerations, deterministic continuation with <<CONTINUE>>,
proper formatting with blank lines between paragraphs and list items.
Backward-compatible signatures.
"""

import logging
import re
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    - Strict domain: AI/ML, DL, DS, NLP, CV, AI apps/ethics, AI programming.
    - Knowledge-lock: Use ONLY injected KB content; append minimal general AI ONLY if KB is insufficient.
    - HTML-only: <p>, <ul><li>, <strong>; one blank line between <p> blocks AND between <li> items.
    - Complete lists in all modes; never omit items (titles at minimum).
    - Brief mode: items 1–3 full, items 4..N titles only (no collapsing, no Learn more).
    - Detailed/Continue mode: extend without restarting; finish remaining KB items first.
    - CONTINUE marker: if token-limited, end with <<CONTINUE>> on its own line so the client resumes.
    - No keywords display, no Key Points section.
    - Backward-compatible API: accepts is_presentation param but ignores it.
    """

    BASE_SYSTEM_PROMPT = """You are AI Shine, an expert educational assistant specializing in Artificial Intelligence, Machine Learning, and related technologies.

Identity:
- Teach clearly and naturally with a warm, encouraging, professional tone. Keep small talk minimal.

Scope (strict):
- Only answer topics in: Artificial Intelligence, Machine Learning, Deep Learning, Data Science, Natural Language Processing, Computer Vision, AI Applications and Ethics, AI-powered Education, Programming for AI/ML (e.g., Python, TensorFlow, PyTorch).
- If out-of-scope, respond exactly (plain text, no HTML):
"⚠️ I specialize in AI and Machine Learning topics. I'd be happy to help with questions about [suggest 2-3 related AI/ML topics]."
- Start that out-of-scope line with the ⚠️ emoji.

Knowledge-lock (critical):
- Use ONLY the provided educational content for this turn. Do NOT add roles, items, examples, claims, steps, or statistics that are not present in the content.
- HALLUCINATION GUARDRAIL: If the KB content mentions a concept (e.g., "machine learning") but doesn't provide a complete definition, structure your answer as follows:
  [1] Lead with a clear, concise definition using your AI/ML training knowledge
  [2] Then naturally integrate KB examples and applications (e.g., "It's used in Netflix recommendations for...")
  [3] Do NOT start with phrases like "The provided content doesn't define..." or "From a general AI perspective..." - just provide the definition directly and weave in KB examples
- If the KB has NO information on a domain-specific AI/ML topic, you may provide a brief, accurate definition from your training knowledge, but acknowledge that specific applications aren't in the provided materials.
- NEVER invent examples, statistics, specific claims, roles, or applications that are not present in the KB content. Only provide general definitions when absolutely necessary.
- If details are missing, structure as: "[Clear definition]. Based on the content, [integrate KB examples]."
- Do NOT switch to prompting frameworks unless explicitly asked.

Output format (HTML only):
- Use <p> for paragraphs and insert a single blank line between consecutive <p> blocks.
- Use <strong> for key terms (2–4 words). Use <ul><li> for key points.
- Insert a single blank line between each <li> item in lists for better readability.
- CRITICAL: ALL content must be wrapped in proper HTML tags. Never output plain text lines without tags.
- For any bulleted or numbered content, ALWAYS use <ul><li> or <ol><li> tags.
- Never use plain text bullets (•, -, *) without wrapping in <li> tags.
- Never output raw text paragraphs - always wrap in <p> tags.
- No markdown; no plain text outside HTML (except the out-of-scope line).

EXAMPLE OF CORRECT FORMATTING:
<p>AI significantly aids in visualizing complex topics by transforming abstract information into intelligent, digestible visuals.</p>

<p>Here are some ways AI helps:</p>

<ul>
<li>AI can create labeled diagrams for scientific concepts (e.g., human heart, digestive system)</li>

<li>It can animate processes like mitosis or the water cycle</li>

<li>AI excels at converting tables into colorful charts (pie, bar, line)</li>
</ul>

INCORRECT (NEVER DO THIS):
AI can create labeled diagrams for scientific concepts
It can animate processes like mitosis
AI excels at converting tables into charts

Enumeration policy (complete in all modes):
- If the KB content has a numbered/bulleted list, ensure EVERY item appears.
- In brief mode for generic lists:
  - Show the first 3 items with full descriptions.
  - Show items 4..N as titles in the main <ul>.
  - Do NOT use collapsing or Learn more buttons - show all items inline with their full descriptions.
- In the special topic "Future Careers Powered by AI":
  - Show items 1–8 with full descriptions in the main <ul>.
  - Show item 9 with full description as well - no collapsing.
- Do NOT add, merge, rename, or omit items. Use ONLY KB-provided examples/details.

Detailed/Continue behavior:
- For "continue", "more detail", "elaborate", "go deeper", etc.:
  - Do NOT restate earlier content; extend using ONLY details present in the KB.
  - If a list began earlier, resume at the next item and complete all remaining KB items before adding extra depth.
  - Aim for 4–6 <p> paragraphs.

Chunking and continuation marker (for long outputs):
- Write in logically complete sections. If you must stop before completing all sections or list items,
  end the message with this exact token on its own line: <<CONTINUE>>
- On the next turn, resume exactly where you stopped without repeating earlier content.

KB-first with minimal append:
- Only after fully using the KB content may you append minimal, generic AI knowledge to fill gaps; keep such additions short and aligned with the KB.
- Never add or replace list items beyond those in the KB.
"""

    # Continuation cues
    CONTINUATION_PATTERNS = [
        r'\btell\s+me\s+more\b', r'\belaborate\b', r'\bgo\s+deeper\b', r'\bexpand\b',
        r'\bmore\s+detail\b', r'\bexplain\s+further\b', r'\bkeep\s+going\b', r'\bwhat\s+else\b',
        r'\bcontinue\b', r'\bgo\s+on\b', r'^\s*and\s*\??\s*$', r'^\s*more\s*\??\s*$',
        r'^\s*continue\s*\??\s*$',
    ]

    GREETING_PATTERNS = [r'^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|sup|yo)\s*[!.,]?\s*$']
    FAREWELL_PATTERNS = [r'^\s*(bye|goodbye|see\s+you|farewell|ttyl|later|ciao|adios)\s*[!.,]?\s*$']

    # Larger packing to preserve full lists
    MAX_CONTEXT_CHUNKS = 3
    CHUNK_TRUNCATE_CHARS = 3000

    def __init__(self):
        self.continuation_regex = re.compile('|'.join(self.CONTINUATION_PATTERNS), re.IGNORECASE)
        self.greeting_regex = re.compile('|'.join(self.GREETING_PATTERNS), re.IGNORECASE)
        self.farewell_regex = re.compile('|'.join(self.FAREWELL_PATTERNS), re.IGNORECASE)

    def detect_intent(self, message: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if not message or not message.strip():
            return {"intent_type": "query", "is_continuation": False, "is_greeting": False, "is_farewell": False, "confidence": 0.0}
        text = message.strip()
        if self.greeting_regex.match(text):
            return {"intent_type": "greeting", "is_continuation": False, "is_greeting": True, "is_farewell": False, "confidence": 1.0}
        if self.farewell_regex.match(text):
            return {"intent_type": "farewell", "is_continuation": False, "is_greeting": False, "is_farewell": True, "confidence": 1.0}
        if self.continuation_regex.search(text):
            return {"intent_type": "continuation", "is_continuation": True, "is_greeting": False, "is_farewell": False, "confidence": 1.0}
        return {"intent_type": "query", "is_continuation": False, "is_greeting": False, "is_farewell": False, "confidence": 1.0}

    def build_system_prompt(
        self,
        intent: Dict[str, Any],
        has_context: bool = True,
        is_presentation: bool = False  # ignored, kept for compatibility
    ) -> str:
        system_prompt = self.BASE_SYSTEM_PROMPT

        if intent.get("is_continuation", False):
            system_prompt += (
                "\n\n[INTERNAL NOTE - Not visible to user]\n"
                "Detailed/Continue Mode:\n"
                "- Extend prior answer without repeating.\n"
                "- Resume and complete any KB-derived lists before adding extra depth.\n"
                "- Use only KB details; append minimal general AI info only if KB is insufficient.\n"
                "- 4–6 <p> paragraphs.\n"
            )
        if not has_context:
            system_prompt += (
                "\n\n[INTERNAL NOTE - Not visible to user]\n"
                "No KB content available. Be brief: acknowledge limited details and offer related AI/ML topics.\n"
            )

        return system_prompt

    def _pack_context(self, context_chunks: List[str]) -> str:
        if not context_chunks:
            return ""
        section = "[Educational content (treat as your expertise for THIS TURN ONLY)]:\n\n"
        added = 0
        for chunk in context_chunks:
            if not chunk:
                continue
            truncated = chunk[: self.CHUNK_TRUNCATE_CHARS]
            if len(chunk) > self.CHUNK_TRUNCATE_CHARS:
                truncated += "..."
            section += truncated + "\n\n"
            added += 1
            if added >= self.MAX_CONTEXT_CHUNKS:
                break
        section += "---\n\n"
        return section

    def build_user_prompt(
        self,
        query: str,
        context_chunks: List[str],
        intent: Dict[str, Any],
        is_presentation: bool = False,  # ignored
        show_module_citation: bool = False,
        module_names: Optional[List[str]] = None,
        use_learn_more: bool = True
    ) -> str:
        if not context_chunks:
            return (
                "[No educational content available for this query]\n\n"
                f"User Question: {query}\n\n"
                "You don't have detailed information for this specific aspect. "
                "Respond briefly and suggest related AI/ML topics you can help with."
            )

        user_prompt = self._pack_context(context_chunks)
        user_prompt += f"Student Question: {query}\n\n"

        ql = (query or "").lower()
        hardlock_careers = (
            "future careers powered by ai" in ql
            or "careers in ai" in ql
            or "ai careers" in ql
        )

        # Optional footer citation
        footer_note = ""
        if show_module_citation and module_names:
            unique = [m for m in dict.fromkeys(module_names) if m]
            if unique:
                footer_note = f'\n- At the very end, append: <p><em>Source: {", ".join(unique)}</em></p>'

        # continue_rule = "\n- If you must stop before completing all sections or list items, end with a line containing exactly: <<CONTINUE>>"
        continue_rule = "\n- If you must stop before completing all sections or list items, end with: <p><em>Write 'continue' to keep generating...</em></p>"

        if intent.get("is_continuation", False):
            user_prompt += (
                "FORMATTING REQUIREMENT: Your entire response must use proper HTML tags.\n"
                "- Every paragraph must be wrapped in <p>...</p>\n"
                "- Every list item must be wrapped in <li>...</li> inside <ul>...</ul>\n"
                "- Never output plain text without HTML tags.\n\n"
                "HALLUCINATION GUARDRAIL: Use ONLY details from the KB content above. If you need to add general AI/ML knowledge to complete the explanation, clearly indicate what comes from the KB vs. your general knowledge. NEVER invent specific examples, statistics, or applications not in the KB.\n\n"
                "Provide a deeper continuation using the content above (do not restate previous content):\n"
                "- Write 4–6 paragraphs in <p> tags with NEW KB-backed details. Insert one blank line between paragraphs.\n"
                "- If a list began earlier, continue at the next item and complete all remaining KB items before adding extra depth.\n"
                "- CRITICAL: Wrap ALL list items in <ul><li> tags with a blank line between each <li>.\n"
                "- CRITICAL: Wrap ALL paragraphs in <p> tags. Never output plain text without HTML tags.\n"
                "- Use <strong> sparingly (2–4 words). HTML only; do not mention sources."
                f"{continue_rule}"
                f"{footer_note}"
            )
            return user_prompt

        if hardlock_careers:
            # Special rendering: items 1–8 full; item 9 with full description as well
            user_prompt += (
                "FORMATTING REQUIREMENT: Your entire response must use proper HTML tags.\n"
                "- Every paragraph must be wrapped in <p>...</p>\n"
                "- Every list item must be wrapped in <li>...</li> inside <ul>...</ul>\n"
                "- Never output plain text without HTML tags.\n\n"
                "HALLUCINATION GUARDRAIL: Render careers STRICTLY from the KB above. Do NOT add careers, examples, or descriptions not present in the KB. If you need to add brief general context about AI careers, keep it minimal and clearly separate from the KB-sourced list.\n\n"
                "Render the careers STRICTLY from the KB above:\n"
                "- Write 2–3 paragraphs in <p> tags (insert a blank line between paragraphs).\n"
                "- Then output a <ul> that lists ALL 9 items exactly as titled in the KB.\n"
                "- Provide full descriptions for ALL 9 items in the main <ul>.\n"
                "- Insert a single blank line between each <li> item for readability.\n"
                "- CRITICAL: ALL list items must be wrapped in <li> tags. Never use plain text bullets.\n"
                "- Do NOT add generic filler or extra careers. HTML only."
                f"{continue_rule}"
                f"{footer_note}"
            )
            return user_prompt

        # Generic brief-but-complete list rendering for all other topics
        user_prompt += (
            "FORMATTING REQUIREMENT: Your entire response must use proper HTML tags.\n"
            "- Every paragraph must be wrapped in <p>...</p>\n"
            "- Every list item must be wrapped in <li>...</li> inside <ul>...</ul>\n"
            "- Never output plain text without HTML tags.\n\n"
            "HALLUCINATION GUARDRAIL: Synthesize your answer from the KB content above. If the KB mentions a concept but doesn't fully define it (e.g., mentions 'machine learning' in examples):\n"
            "[1] Lead with a clear definition using your AI/ML training knowledge\n"
            "[2] Then naturally integrate KB examples (e.g., 'It's used in Netflix recommendations for...')\n"
            "[3] Do NOT start with 'The provided content doesn't define...' - just provide the definition and weave in KB examples\n"
            "NEVER invent specific examples, tools, statistics, or applications not in the KB. Only provide general conceptual definitions when needed.\n\n"
            "Provide a concise, KB-first answer:\n"
            "- Write 2–3 paragraphs in <p> tags; insert a blank line between them.\n"
            "- If the KB contains a numbered/bulleted list, output EVERY item with full description:\n"
            "  • Show items 1–3 with full descriptions in the main <ul>.\n"
            "  • Show items 4..N with full descriptions in the main <ul> as well.\n"
            "  • Insert a single blank line between each <li> item for readability.\n"
            "  • CRITICAL: ALL list items must be wrapped in <li> tags. Never use plain text bullets.\n"
            "- CRITICAL: ALL paragraphs must be wrapped in <p> tags. Never output plain text without HTML tags.\n"
            "- Do NOT add or omit items; use only KB-provided examples.\n"
            "- Use <strong> for key terms (2–4 words)."
            f"{continue_rule}"
            f"{footer_note}"
        )
        return user_prompt

    def build_greeting_response(self) -> str:
        return (
            "👋 Hello! I'm <strong>AI Shine</strong>, your AI/ML educational assistant. "
            "Ask anything about <strong>AI</strong>, <strong>Machine Learning</strong>, "
            "<strong>Data Science</strong>, <strong>NLP</strong>, or <strong>Computer Vision</strong>!"
        )

    def build_farewell_response(self) -> str:
        return (
            "👋 Goodbye! It was great chatting about <strong>AI</strong> and <strong>ML</strong>. "
            "Come back anytime to learn more!"
        )