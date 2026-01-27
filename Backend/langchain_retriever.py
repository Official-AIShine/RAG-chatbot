"""
LangChain MongoDB Retriever with Embedding Reuse
Single embeddings client, dual-threshold retrieval strategy.
"""
import os
import logging
from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_aws import BedrockEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi
import time
from botocore.exceptions import ClientError

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangChainRetriever(BaseRetriever):
    """
    Optimized retriever with:
    - Single embeddings client (no redundant API calls)
    - Dual-threshold retrieval (primary 0.55, fallback 0.45)
    - Exponential backoff for throttling
    - Embedding vector reuse
    """
    
    # Pydantic fields - must be class attributes
    vector_search: Any = None
    primary_threshold: float = 0.55
    fallback_threshold: float = 0.45
    max_retries: int = 5
    base_delay: float = 1.0
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        collection_name: str = "module_vectors",
        primary_threshold: float = 0.55,
        fallback_threshold: float = 0.45,
        max_retries: int = 5,
        base_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize single retriever with dual-threshold strategy.
        
        Args:
            collection_name: MongoDB collection
            primary_threshold: High-confidence threshold
            fallback_threshold: Lower threshold for fallback
            max_retries: Max retry attempts for throttling
            base_delay: Exponential backoff base delay
        """
        # Get MongoDB connection details
        mongo_uri = os.getenv("MONGO_DB_URI")
        db_name = os.getenv("DB_NAME", "aishine")
        
        if not mongo_uri:
            raise ValueError("[LANGCHAIN_RETRIEVER] MONGO_DB_URI not set")
        
        # Initialize MongoDB client
        client = MongoClient(
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
            maxPoolSize=10,
            minPoolSize=2,
        )
        
        # Test connection
        client.admin.command('ping')
        logger.info(f"[LANGCHAIN_RETRIEVER] Connected to {db_name}.{collection_name}")
        
        collection = client[db_name][collection_name]
        
        # SINGLE embeddings client (prevents duplicate API calls)
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1"),
            credentials_profile_name=None  # Uses env vars
        )
        
        # Initialize LangChain's MongoDB vector search
        vector_search_instance = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            text_key="content",
            embedding_key="embedding"
        )
        
        # Initialize parent with all fields set
        super().__init__(
            vector_search=vector_search_instance,
            primary_threshold=primary_threshold,
            fallback_threshold=fallback_threshold,
            max_retries=max_retries,
            base_delay=base_delay,
            **kwargs
        )
        
        logger.info(
            f"[LANGCHAIN_RETRIEVER] Initialized with thresholds: "
            f"primary={self.primary_threshold}, fallback={self.fallback_threshold}"
        )
    
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return self.base_delay * (2 ** attempt)
    
    def _is_throttling_error(self, error: Exception) -> bool:
        """Check if error is throttling exception."""
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            return error_code in ['ThrottlingException', 'TooManyRequestsException']
        return 'ThrottlingException' in str(error) or 'Too many requests' in str(error)
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents with single embedding call + dual threshold.
        
        Strategy:
        1. Generate embedding ONCE
        2. Retrieve all candidates (k=5)
        3. Filter by primary threshold (0.55)
        4. If empty, filter by fallback threshold (0.45)
        5. No additional embedding calls
        
        Args:
            query: Search query
            run_manager: Callback manager (unused)
        
        Returns:
            List of relevant documents
        """
        logger.info(f"[LANGCHAIN_RETRIEVER] Query: {query}")
        
        for attempt in range(self.max_retries):
            try:
                # SINGLE embedding call retrieves all candidates
                results = self.vector_search.similarity_search_with_score(
                    query=query,
                    k=5  # Get top 5 for filtering
                )
                
                # Primary threshold filter
                primary_docs = [
                    doc for doc, score in results 
                    if score >= self.primary_threshold
                ]
                
                if primary_docs:
                    logger.info(
                        f"[LANGCHAIN_RETRIEVER] ✅ Primary: {len(primary_docs)}/5 "
                        f"docs above {self.primary_threshold}"
                    )
                    return primary_docs
                
                # Fallback threshold filter (REUSES same embedding)
                fallback_docs = [
                    doc for doc, score in results 
                    if score >= self.fallback_threshold
                ]
                
                if fallback_docs:
                    logger.info(
                        f"[LANGCHAIN_RETRIEVER] ⚠️ Fallback: {len(fallback_docs)}/5 "
                        f"docs above {self.fallback_threshold}"
                    )
                    return fallback_docs
                
                logger.warning(
                    f"[LANGCHAIN_RETRIEVER] ❌ No docs above {self.fallback_threshold}"
                )
                return []
                
            except Exception as e:
                if self._is_throttling_error(e):
                    if attempt < self.max_retries - 1:
                        delay = self._exponential_backoff(attempt)
                        logger.warning(
                            f"[LANGCHAIN_RETRIEVER] Throttled attempt {attempt + 1}/{self.max_retries}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(
                            f"[LANGCHAIN_RETRIEVER] Max retries exceeded. Returning empty."
                        )
                        return []
                else:
                    logger.error(f"[LANGCHAIN_RETRIEVER] Error: {e}", exc_info=True)
                    return []
        
        return []
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Async version of _get_relevant_documents.
        Falls back to sync for now (MongoDB doesn't support async natively).
        """
        return self._get_relevant_documents(query, run_manager=run_manager)


def get_langchain_retriever(
    collection_name: str = "module_vectors",
    primary_threshold: float = 0.55,
    fallback_threshold: float = 0.45
) -> LangChainRetriever:
    """
    Factory for optimized retriever.
    
    Args:
        collection_name: MongoDB collection
        primary_threshold: High-confidence threshold
        fallback_threshold: Lower fallback threshold
    
    Returns:
        Single retriever instance
    """
    return LangChainRetriever(
        collection_name=collection_name,
        primary_threshold=primary_threshold,
        fallback_threshold=fallback_threshold
    )











# """
# LangChain MongoDB Retriever - Using Custom Cached Embeddings
# FIXED: Proper error handling for rate limits (no zero vectors) but still rate limited
# """
# import os
# import logging
# from typing import List, Any
# from langchain_core.retrievers import BaseRetriever
# from langchain_core.documents import Document
# from langchain_core.callbacks import CallbackManagerForRetrieverRun
# from langchain_core.embeddings import Embeddings
# from langchain_mongodb import MongoDBAtlasVectorSearch
# from pymongo import MongoClient
# from dotenv import load_dotenv
# import certifi

# # Import YOUR cached client
# from Backend.embedding_client import BedrockEmbeddingClient

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Global shared instances
# _shared_mongo_client = None
# _shared_embeddings = None


# def get_shared_mongo_client():
#     """Get or create shared MongoDB client."""
#     global _shared_mongo_client
    
#     if _shared_mongo_client is None:
#         mongo_uri = os.getenv("MONGO_DB_URI")
#         if not mongo_uri:
#             raise ValueError("[LANGCHAIN_RETRIEVER] MONGO_DB_URI not set")
        
#         _shared_mongo_client = MongoClient(
#             mongo_uri,
#             tls=True,
#             tlsCAFile=certifi.where(),
#             tlsAllowInvalidCertificates=True,
#             tlsAllowInvalidHostnames=True,
#             retryWrites=True,
#             retryReads=True,
#             serverSelectionTimeoutMS=20000,
#             connectTimeoutMS=20000,
#             socketTimeoutMS=30000,
#             maxPoolSize=10,
#             minPoolSize=2,
#         )
        
#         _shared_mongo_client.admin.command('ping')
#         logger.info("[LANGCHAIN_RETRIEVER] Shared MongoDB client created")
    
#     return _shared_mongo_client


# class CachedBedrockEmbeddings(Embeddings):
#     """
#     LangChain-compatible wrapper for your cached BedrockEmbeddingClient.
#     FIXED: Raises exception instead of returning zero vectors on failure.
#     """
    
#     def __init__(self):
#         self.client = BedrockEmbeddingClient()
#         logger.info("[CACHED_EMBEDDINGS] Using custom client with LRU cache")
    
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """
#         Embed multiple documents.
#         Returns only successful embeddings (filters out None).
#         """
#         embeddings = self.client.generate_batch_embeddings(texts)
        
#         # Filter out None - return only valid embeddings
#         valid_embeddings = [emb for emb in embeddings if emb is not None]
        
#         if not valid_embeddings:
#             logger.warning("[CACHED_EMBEDDINGS] All document embeddings failed")
#             return []  # Return empty list instead of zero vectors
        
#         if len(valid_embeddings) < len(embeddings):
#             logger.warning(
#                 f"[CACHED_EMBEDDINGS] Only {len(valid_embeddings)}/{len(embeddings)} "
#                 f"documents embedded successfully"
#             )
        
#         return valid_embeddings
    
#     def embed_query(self, text: str) -> List[float]:
#         """
#         Embed a single query.
#         FIXED: Raises exception on failure instead of returning zero vector.
#         """
#         embedding = self.client.generate_embedding(text)
        
#         if embedding is None:
#             logger.warning(
#                 "[CACHED_EMBEDDINGS] Embedding failed (likely rate limited) - "
#                 "will skip retrieval gracefully"
#             )
#             # CRITICAL FIX: Raise exception instead of returning [0.0]*1024
#             # Zero vectors cause "Cosine similarity cannot be calculated" error
#             raise ValueError("Embedding generation failed - likely rate limited")
        
#         return embedding


# def get_shared_embeddings():
#     """Get or create shared cached embeddings client."""
#     global _shared_embeddings
    
#     if _shared_embeddings is None:
#         _shared_embeddings = CachedBedrockEmbeddings()
#         logger.info("[LANGCHAIN_RETRIEVER] Shared cached embeddings client created")
    
#     return _shared_embeddings


# class LangChainMongoRetriever(BaseRetriever):
#     """
#     LangChain-compatible retriever with cached embeddings.
#     Gracefully handles embedding failures without breaking MongoDB queries.
#     """
    
#     vector_search: Any = None
#     similarity_threshold: float = 0.75
#     max_results: int = 3
    
#     class Config:
#         arbitrary_types_allowed = True
    
#     def __init__(self, **kwargs):
#         """Initialize retriever using shared cached clients."""
#         db_name = os.getenv("DB_NAME")
#         collection_name = "module_vectors"
        
#         if not db_name:
#             raise ValueError("[LANGCHAIN_RETRIEVER] DB_NAME not set")
        
#         # Use shared clients
#         client = get_shared_mongo_client()
#         embeddings = get_shared_embeddings()  # Uses YOUR cached client
        
#         collection = client[db_name][collection_name]
        
#         # Initialize vector search
#         vector_search_instance = MongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,  # Your cached client wrapped for LangChain
#             index_name="vector_index",
#             text_key="content",
#             embedding_key="embedding"
#         )
        
#         super().__init__(vector_search=vector_search_instance, **kwargs)
        
#         logger.info(
#             f"[LANGCHAIN_RETRIEVER] Initialized with cached embeddings "
#             f"(threshold={self.similarity_threshold}, max_results={self.max_results})"
#         )
    
#     def _get_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: CallbackManagerForRetrieverRun = None
#     ) -> List[Document]:
#         """
#         Retrieve relevant documents.
#         FIXED: Gracefully handles embedding failures without zero vectors.
#         """
#         try:
#             logger.info(f"[LANGCHAIN_RETRIEVER] Query: {query}")
            
#             # Primary search with threshold 0.75
#             # This will raise ValueError if embedding fails
#             results = self.vector_search.similarity_search_with_score(
#                 query=query,
#                 k=self.max_results,
#                 pre_filter={"source": "knowledge_base"}
#             )
            
#             # Filter by similarity threshold
#             filtered_results = [
#                 (doc, score) for doc, score in results 
#                 if score >= self.similarity_threshold
#             ]
            
#             if filtered_results:
#                 logger.info(f"[LANGCHAIN_RETRIEVER] ✅ Found {len(filtered_results)} documents")
#                 documents = [doc for doc, _ in filtered_results]
                
#                 if documents:
#                     top_metadata = documents[0].metadata
#                     logger.info(f"[LANGCHAIN_RETRIEVER] Top: {top_metadata.get('topic', 'N/A')}")
                
#                 return documents
            
#             # Fallback to lower threshold (0.45)
#             logger.info(
#                 f"[LANGCHAIN_RETRIEVER] No results above {self.similarity_threshold}, "
#                 f"trying lower threshold 0.45"
#             )
            
#             results_lower = self.vector_search.similarity_search_with_score(
#                 query=query,
#                 k=self.max_results,
#                 pre_filter={"source": "knowledge_base"}
#             )
            
#             filtered_lower = [
#                 (doc, score) for doc, score in results_lower 
#                 if score >= 0.45
#             ]
            
#             if filtered_lower:
#                 logger.info(
#                     f"[LANGCHAIN_RETRIEVER] ✅ Found {len(filtered_lower)} documents "
#                     f"with lower threshold"
#                 )
#                 return [doc for doc, _ in filtered_lower]
            
#             logger.warning("[LANGCHAIN_RETRIEVER] ❌ No results found even with lower threshold")
#             return []
            
#         except ValueError as e:
#             # FIXED: Catch embedding failure gracefully
#             # This happens when AWS Bedrock is rate limited
#             logger.warning(f"[LANGCHAIN_RETRIEVER] Skipping retrieval: {e}")
#             logger.info("[LANGCHAIN_RETRIEVER] Gemini will respond without RAG context")
#             return []
            
#         except Exception as e:
#             # Catch all other errors (MongoDB, network, etc.)
#             logger.error(f"[LANGCHAIN_RETRIEVER] Unexpected error: {e}", exc_info=True)
#             return []
    
#     async def _aget_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: CallbackManagerForRetrieverRun = None
#     ) -> List[Document]:
#         """
#         Async version - falls back to sync.
#         """
#         return self._get_relevant_documents(query, run_manager=run_manager)




# # ```

# # ---

# # ## What Changed

# # ### ✅ Preserved (Nothing Broken):
# # - Shared MongoDB client
# # - Shared embedding client (LRU cache)
# # - Similarity thresholds (0.75 primary, 0.45 fallback)
# # - All logging
# # - LangChain integration

# # ### ✅ Fixed:
# # 1. **Line 93:** `raise ValueError` instead of `return [0.0] * 1024`
# # 2. **Line 175:** Catch `ValueError` gracefully
# # 3. **Line 178:** Log that Gemini will respond without RAG

# # ---

# # ## What Happens Now

# # ### Before (Broken):
# # ```
# # AWS throttle → return [0.0]*1024 → MongoDB "zero vector error" → Stack trace
# # ```

# # ### After (Fixed):
# # ```
# # AWS throttle → raise ValueError → catch → return [] → Gemini responds without RAG