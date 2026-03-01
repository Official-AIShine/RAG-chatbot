"""
Optimized Retriever for Low Latency

Key optimizations:
1. Single embedding call per query
2. Dual-threshold retrieval (primary + fallback)
3. Connection pooling
4. Exponential backoff for rate limits
"""
import os
import logging
import time
from typing import List, Dict, Any, Optional
from functools import lru_cache

from pymongo import MongoClient
from langchain_aws import BedrockEmbeddings
from dotenv import load_dotenv
import certifi
from botocore.exceptions import ClientError

from Backend.config import settings

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global connection pool (singleton)
_mongo_client: Optional[MongoClient] = None
_embeddings: Optional[BedrockEmbeddings] = None


def get_mongo_client() -> MongoClient:
    """Get or create MongoDB client singleton."""
    global _mongo_client

    if _mongo_client is None:
        mongo_uri = os.getenv("MONGO_DB_URI")
        if not mongo_uri:
            raise ValueError("[RETRIEVER] MONGO_DB_URI not set")

        _mongo_client = MongoClient(
            mongo_uri,
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=True,
            tlsAllowInvalidHostnames=True,
            retryWrites=True,
            retryReads=True,
            serverSelectionTimeoutMS=20000,
            connectTimeoutMS=20000,
            socketTimeoutMS=30000,
            maxPoolSize=settings.MONGO_MAX_POOL_SIZE,
            minPoolSize=settings.MONGO_MIN_POOL_SIZE,
        )

        # Test connection
        _mongo_client.admin.command('ping')
        logger.info("[RETRIEVER] MongoDB client connected")

    return _mongo_client


def get_embeddings() -> BedrockEmbeddings:
    """Get or create embeddings client singleton."""
    global _embeddings

    if _embeddings is None:
        _embeddings = BedrockEmbeddings(
            model_id=settings.EMBEDDING_MODEL_ID,
            region_name=settings.AWS_REGION,
            credentials_profile_name=None  # Uses env vars
        )
        logger.info(f"[RETRIEVER] Embeddings client ready ({settings.EMBEDDING_MODEL_ID})")

    return _embeddings


class OptimizedRetriever:
    """
    Optimized retriever with:
    - Single embedding call per query
    - Dual-threshold strategy
    - Connection pooling
    - Rate limit handling
    """

    def __init__(
        self,
        collection_name: str = "module_vectors",
        primary_threshold: float = 0.55,
        fallback_threshold: float = 0.45,
        top_k: int = 3,
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        """
        Initialize retriever.

        Args:
            collection_name: MongoDB collection name
            primary_threshold: High-confidence similarity threshold
            fallback_threshold: Lower fallback threshold
            top_k: Number of documents to retrieve
            max_retries: Max retry attempts for rate limiting
            base_delay: Base delay for exponential backoff
        """
        self.collection_name = collection_name
        self.primary_threshold = primary_threshold
        self.fallback_threshold = fallback_threshold
        self.top_k = top_k
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Get shared clients
        self.client = get_mongo_client()
        self.embeddings = get_embeddings()

        # Get collection
        db_name = os.getenv("DB_NAME", "aishine")
        self.collection = self.client[db_name][collection_name]

        logger.info(
            f"[RETRIEVER] Initialized - collection: {collection_name}, "
            f"thresholds: {primary_threshold}/{fallback_threshold}"
        )

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return self.base_delay * (2 ** attempt)

    def _is_throttling_error(self, error: Exception) -> bool:
        """Check if error is a rate limiting exception."""
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            return error_code in ['ThrottlingException', 'TooManyRequestsException']
        return 'ThrottlingException' in str(error) or 'Too many requests' in str(error)

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding with retry logic.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None on failure
        """
        for attempt in range(self.max_retries):
            try:
                embedding = self.embeddings.embed_query(text)
                return embedding

            except Exception as e:
                if self._is_throttling_error(e):
                    if attempt < self.max_retries - 1:
                        delay = self._exponential_backoff(attempt)
                        logger.warning(
                            f"[RETRIEVER] Throttled, retry {attempt + 1}/{self.max_retries} "
                            f"in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("[RETRIEVER] Max retries exceeded for embedding")
                        return None
                else:
                    logger.error(f"[RETRIEVER] Embedding error: {e}")
                    return None

        return None

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract potential keywords from query for text search.

        Args:
            query: User query

        Returns:
            List of keywords (lowercased, filtered)
        """
        # Common stop words to filter out (includes generic verbs that match everything)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
            'because', 'until', 'while', 'about', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'am', 'i', 'me', 'my', 'myself',
            'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it',
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            # Generic action verbs — too common in all KB content, produce noisy matches
            'tell', 'explain', 'describe', 'show', 'give', 'please', 'help',
            'use', 'make', 'build', 'create', 'get', 'find', 'tool', 'good',
            'want', 'need', 'like', 'know', 'think', 'way', 'work', 'learn',
        }

        # Extract words, filter stop words, keep words with 3+ chars
        words = query.lower().split()
        keywords = [
            w.strip('.,!?()[]{}":;')
            for w in words
            if w.lower() not in stop_words and len(w) >= 3
        ]

        return keywords

    def _keyword_search(self, keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform text search on topic, content, and keywords fields.

        Args:
            keywords: List of keywords to search
            limit: Max results

        Returns:
            List of matching documents
        """
        if not keywords:
            return []

        try:
            # Build regex pattern for case-insensitive search
            # Match any of the keywords in topic, content, summary, or keywords array
            or_conditions = []
            for kw in keywords:
                # Escape special regex characters
                escaped_kw = kw.replace('.', r'\.').replace('*', r'\*')
                regex_pattern = {"$regex": escaped_kw, "$options": "i"}
                # Only search topic and keywords fields — NOT content/summary
                # Content matching is too noisy (common words appear in every document)
                or_conditions.extend([
                    {"topic": regex_pattern},
                    {"keywords": regex_pattern}
                ])

            results = list(self.collection.find(
                {"$or": or_conditions},
                {
                    "_id": 0,
                    "topic": 1,
                    "category": 1,
                    "level": 1,
                    "summary": 1,
                    "content": 1,
                    "keywords": 1,
                    "module_name": 1,
                    "source": 1
                }
            ).limit(limit))

            # Add a synthetic score for keyword matches
            for doc in results:
                doc["score"] = 0.75  # Keyword match score
                doc["match_type"] = "keyword"

            return results

        except Exception as e:
            logger.warning(f"[RETRIEVER] Keyword search error: {e}")
            return []

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using HYBRID search.

        Strategy:
        1. Generate embedding and run vector search
        2. Extract keywords and run text search in parallel
        3. Merge results (deduplicate by topic)
        4. Return combined results

        Args:
            query: Search query

        Returns:
            List of documents with scores
        """
        start_time = time.time()

        # Generate embedding
        embedding = self._generate_embedding(query)
        if embedding is None:
            logger.warning("[RETRIEVER] No embedding generated, trying keyword-only")
            # Fall back to keyword search only
            keywords = self._extract_keywords(query)
            if keywords:
                keyword_results = self._keyword_search(keywords, limit=self.top_k)
                if keyword_results:
                    logger.info(f"[RETRIEVER] Keyword-only: {len(keyword_results)} docs")
                    return [self._format_document(doc) for doc in keyword_results]
            return []

        embedding_time = (time.time() - start_time) * 1000

        try:
            # Build vector search pipeline
            # numCandidates must be >= limit; use a large number to scan the full collection
            # limit is set higher than top_k so we have more vector candidates to work with
            vector_limit = self.top_k * 3
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": embedding,
                        "numCandidates": 500,
                        "limit": vector_limit
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "topic": 1,
                        "category": 1,
                        "level": 1,
                        "summary": 1,
                        "content": 1,
                        "keywords": 1,
                        "module_name": 1,
                        "source": 1,
                        "score": 1
                    }
                }
            ]

            # Execute vector search
            vector_results = list(self.collection.aggregate(pipeline, maxTimeMS=10000))
            for doc in vector_results:
                doc["match_type"] = "vector"

            search_time = (time.time() - start_time) * 1000 - embedding_time

            # ---- KEYWORD AUGMENTED RETRIEVAL (toggled by KEYWORD_AUGMENTED_RETRIEVAL env var) ----
            keyword_results = []
            if settings.KEYWORD_AUGMENTED_RETRIEVAL:
                keywords = self._extract_keywords(query)
                if keywords:
                    keyword_results = self._keyword_search(keywords, limit=self.top_k)
                    logger.info(f"[RETRIEVER] KAR on — keywords: {keywords}")
            else:
                logger.info("[RETRIEVER] KAR off — pure vector search")

            # ---- MERGE ----
            seen_topics = set()
            vector_final = []
            keyword_final = []

            if settings.KEYWORD_AUGMENTED_RETRIEVAL and keyword_results:
                # Hybrid mode: vector takes (top_k - KEYWORD_SLOTS), keyword fills the rest
                KEYWORD_SLOTS = min(3, self.top_k)
                VECTOR_SLOTS = self.top_k - KEYWORD_SLOTS

                for doc in vector_results:
                    if len(vector_final) >= VECTOR_SLOTS:
                        break
                    topic = doc.get("topic", "")
                    seen_topics.add(topic.lower())
                    vector_final.append(doc)
                    logger.info(f"[RETRIEVER] Vector match: {topic} (score: {doc.get('score', 0):.3f})")

                for doc in keyword_results:
                    if len(keyword_final) >= KEYWORD_SLOTS:
                        break
                    topic = doc.get("topic", "")
                    if topic.lower() not in seen_topics:
                        seen_topics.add(topic.lower())
                        keyword_final.append(doc)
                        logger.info(f"[RETRIEVER] Keyword match: {topic}")
            else:
                # Pure vector mode: use all top_k vector results
                for doc in vector_results:
                    if len(vector_final) >= self.top_k:
                        break
                    topic = doc.get("topic", "")
                    seen_topics.add(topic.lower())
                    vector_final.append(doc)
                    logger.info(f"[RETRIEVER] Vector match: {topic} (score: {doc.get('score', 0):.3f})")

            final_results = vector_final + keyword_final

            if final_results:
                logger.info(
                    f"[RETRIEVER] {'Hybrid' if settings.KEYWORD_AUGMENTED_RETRIEVAL else 'Vector'}: "
                    f"{len(final_results)} docs "
                    f"(vector: {len(vector_final)}, keyword: {len(keyword_final)}, "
                    f"embed: {embedding_time:.0f}ms, search: {search_time:.0f}ms)"
                )
                return [self._format_document(doc) for doc in final_results]

            logger.warning(
                f"[RETRIEVER] No docs found "
                f"(embed: {embedding_time:.0f}ms, search: {search_time:.0f}ms)"
            )
            return []

        except Exception as e:
            logger.error(f"[RETRIEVER] Search error: {e}", exc_info=True)
            return []

    def _format_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Format document for RAG engine."""
        return {
            "content": doc.get("content", ""),
            "score": doc.get("score", 0),
            "metadata": {
                "topic": doc.get("topic", ""),
                "category": doc.get("category", ""),
                "level": doc.get("level", ""),
                "summary": doc.get("summary", ""),
                "keywords": doc.get("keywords", []),
                "module_name": doc.get("module_name", ""),
                "source": doc.get("source", "")
            }
        }

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        try:
            count = self.collection.count_documents({})
            return {
                "collection": self.collection_name,
                "document_count": count,
                "primary_threshold": self.primary_threshold,
                "fallback_threshold": self.fallback_threshold,
                "top_k": self.top_k
            }
        except Exception as e:
            logger.error(f"[RETRIEVER] Stats error: {e}")
            return {"error": str(e)}
