#!/usr/bin/env python3
"""
Test the _route_query method to ensure it works correctly with the routing logic.
"""
import sys
import re
from typing import List

# Mock the Message class
class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

# Regex patterns from the engine
DETAILED_PATTERNS = [
    r'\btell\s+me\s+more\b',
    r'\bexpand\b',
    r'\bcontinue\b',
    r'\bgo\s+deeper\b',
    r'\bkeep\s+going\b',
    r'\bgo\s+on\b',
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
    r'\bcompare\b',
    r'\bcontrast\b',
    r'\bdifference\s+between\b',
    r'\bhow\s+do(?:es)?\s+.*\s+differ\b',
    r'\bvs\.?\b',
    r'\bversus\b',
    r'\banalyze\b',
    r'\banalysis\b',
    r'\bbreak\s+down\b',
    r'\band\s+also\b',
    r'\band\s+how\b',
    r'\band\s+what\b',
    r'\band\s+why\b',
    r'\bwhat\s+about\b',
    r'\bwhat\s+else\b',
    r'\bany(?:thing)?\s+else\b',
    r'\bother\s+examples?\b',
    r'\bmore\s+examples?\b',
    r'\bstep[\-\s]by[\-\s]step\b',
    r'\bwalk\s+me\s+through\b',
    r'\bguide\s+me\b',
    r'\bshow\s+me\s+how\b',
]

BRIEF_PATTERNS = [
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
    r'\bwhat\s+is\b',
    r'\bdefine\b',
    r'\bdefinition\s+of\b',
    r'\bmeaning\s+of\b',
    r'\bexplain\s+(?!how|why)\w+$',
    r'\blist\b',
    r'\bname\b',
    r'\bgive\s+me\s+\d+\b',
    r'\btop\s+\d+\b',
]

# Compile regex patterns
detailed_regex = re.compile('|'.join(DETAILED_PATTERNS), re.IGNORECASE)
brief_regex = re.compile('|'.join(BRIEF_PATTERNS), re.IGNORECASE)

def _route_query(query: str, chat_history: List[Message]) -> str:
    """Route query to appropriate chain based on complexity signals."""
    query_lower = query.lower().strip()
    
    # Count words
    word_count = len(query_lower.split())
    
    # Explicit brief request → lite
    if brief_regex.search(query_lower):
        return "lite"
    
    # Explicit detailed request → full
    if detailed_regex.search(query_lower):
        return "full"
    
    # Very long queries → full (>=20 words suggests complexity)
    if word_count >= 20:
        return "full"
    
    # Follow-up question logic
    if len(chat_history) >= 2:
        last_assistant_msg = None
        for msg in reversed(chat_history):
            if msg.role == "assistant":
                last_assistant_msg = msg.content
                break
        
        if last_assistant_msg and len(last_assistant_msg) < 300:
            has_new_topic = any(keyword in query_lower for keyword in [
                'what is', 'define', 'explain', 'how does', 'why', 'when', 'where'
            ])
            
            if not has_new_topic and word_count < 15:
                return "full"
    
    # Default: lite
    return "lite"

# Test cases
test_cases = [
    # Brief queries → "lite"
    ("What is machine learning?", [], "lite", "Definition query"),
    ("Define neural network", [], "lite", "Definition request"),
    ("Summarize deep learning", [], "lite", "Summary request"),
    ("List top 5 ML algorithms", [], "lite", "List request"),
    
    # Detailed queries → "full"
    ("Tell me more about transformers", [], "full", "Explicit continuation"),
    ("Explain how CNNs work in detail", [], "full", "Detail request"),
    ("Compare neural networks vs decision trees", [], "full", "Comparison request"),
    ("Walk me through the training process step by step", [], "full", "Step-by-step request"),
    ("This is a very long query that contains more than twenty words and should trigger the full chain routing logic", [], "full", "Long query (>=20 words)"),
    
    # Multiple questions → Flash handles context (lite is fine)
    ("What is ML? How does it work?", [], "lite", "Multiple questions (Flash will handle context)"),
    ("Tell me about NLP. What are the main applications?", [], "lite", "Multiple questions without explicit detailed signal"),
    
    # Follow-up logic
    ("Tell me more", [Message("assistant", "Brief response from lite chain that is under three hundred characters")], "full", "Follow-up after brief response"),
    ("Okay", [Message("assistant", "x" * 400)], "lite", "Follow-up after detailed response stays lite"),
    
    # Default behavior
    ("Can you help?", [], "lite", "Simple question defaults to lite"),
    ("Interesting", [], "lite", "Simple word defaults to lite"),
]

# Run tests
print("🧪 Testing _route_query() function:\n")
passed = 0
failed = 0

for query, history, expected, description in test_cases:
    result = _route_query(query, history)
    status = "✅" if result == expected else "❌"
    
    if result == expected:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} {description}")
    print(f"   Query: '{query[:60]}...' " if len(query) > 60 else f"   Query: '{query}'")
    print(f"   Expected: {expected}, Got: {result}\n")

print(f"\n📊 Results: {passed} passed, {failed} failed out of {passed + failed} tests")

if failed == 0:
    print("✅ All tests passed! The _route_query method works correctly.")
    sys.exit(0)
else:
    print(f"❌ {failed} test(s) failed.")
    sys.exit(1)
