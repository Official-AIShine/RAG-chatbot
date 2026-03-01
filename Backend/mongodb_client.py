"""
MongoDB Atlas Vector Search Client
Extended with conversation storage and length limits.
"""
import os
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
import certifi

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB Atlas client with vector search + conversation storage."""

    # PRODUCTION LIMIT: Prevent unbounded conversation growth
    MAX_MESSAGES_PER_CONVERSATION = 200
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 2):
        """
        Initialize MongoDB client with retry logic.
        
        Args:
            max_retries: Number of connection retry attempts
            retry_delay: Seconds to wait between retries
        """
        load_dotenv(override=True)
        
        self.uri = os.getenv("MONGO_DB_URI")
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = "module_vectors"
        self.conversations_collection_name = "conversations"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Validate required env vars
        if not self.uri:
            raise ValueError("[MONGO_ERR] MONGO_DB_URI not set in environment")
        
        if not self.db_name:
            raise ValueError("[MONGO_ERR] DB_NAME not set in environment")
        
        logger.info(f"[MONGO_INIT] Connecting to database: {self.db_name}")
        
        self.client = None
        self.db = None
        self.collection = None
        self.conversations = None
        
        self._connect_with_retry()
    
    def _connect_with_retry(self):
        """Establish MongoDB connection with retry logic for cloud environments."""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"[MONGO] Connection attempt {attempt}/{self.max_retries}")
                
                self.client = MongoClient(
                    self.uri,
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
                self.client.admin.command('ping')
                
                self.db = self.client[self.db_name]
                self.collection = self.db[self.collection_name]
                self.conversations = self.db[self.conversations_collection_name]
                
                logger.info(f"[MONGO_OK] Connected to {self.db_name}.{self.collection_name}")
                logger.info(f"[MONGO_OK] Conversations collection: {self.conversations_collection_name}")
                
                # Create indexes for conversations (run once, safe to repeat)
                self._ensure_conversation_indexes()
                
                return
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"[MONGO] Attempt {attempt} failed: {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"[MONGO] Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"[MONGO_ERR] All {self.max_retries} connection attempts failed")
                    raise ConnectionFailure(f"Failed to connect after {self.max_retries} attempts: {e}")
    
    def _ensure_conversation_indexes(self):
        """Create indexes for conversation queries."""
        try:
            # Index for listing user's conversations (sorted by recent)
            self.conversations.create_index(
                [("user_key", 1), ("updated_at", -1)],
                name="user_conversations_idx"
            )
            
            # Index for fast conversation lookup
            self.conversations.create_index(
                [("conversation_id", 1)],
                name="conversation_id_idx",
                unique=True
            )
            
            logger.info("[MONGO_OK] Conversation indexes created")
        except Exception as e:
            logger.warning(f"[MONGO] Index creation skipped (may already exist): {e}")
    
    def ensure_connection(self):
        """Verify connection is alive, reconnect if needed."""
        try:
            self.client.admin.command('ping')
        except Exception as e:
            logger.warning(f"[MONGO] Connection lost, reconnecting: {e}")
            self._connect_with_retry()
    
    # ============================================================
    # VECTOR SEARCH METHODS (EXISTING - NO CHANGES)
    # ============================================================
    
    def vector_search(
        self,
        query_embedding: List[float],
        limit: int = 3,
        similarity_threshold: float = 0.4,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search with optional metadata filtering.
        
        Args:
            query_embedding: 1024-dim embedding vector
            limit: Max results to return
            similarity_threshold: Minimum cosine similarity score
            metadata_filters: Optional dict of metadata filters
        
        Returns:
            List of documents with score >= threshold, sorted by relevance
        """
        try:
            self.ensure_connection()
            
            # Build aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 20,
                        "limit": limit
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "score": {"$gte": similarity_threshold}
                    }
                }
            ]
            
            # Add metadata filters if provided
            if metadata_filters:
                pipeline.append({"$match": metadata_filters})
            
            # Project fields
            pipeline.append({
                "$project": {
                    "_id": 1,
                    "topic": 1,
                    "category": 1,
                    "level": 1,
                    "summary": 1,
                    "content": 1,
                    "keywords": 1,
                    "module_name": 1,
                    "source": 1,
                    "presentation_data": 1,
                    "score": 1
                }
            })
            
            results = list(self.collection.aggregate(pipeline, maxTimeMS=30000))
            
            logger.info(f"[VECTOR_SEARCH] Retrieved {len(results)} chunks above threshold {similarity_threshold}")
            if results:
                logger.info(f"[VECTOR_SEARCH_DEBUG] Top: {results[0].get('topic', 'N/A')} ({results[0].get('score', 0):.3f})")
                logger.info(f"[VECTOR_SEARCH_DEBUG] Source: {results[0].get('source', 'N/A')}")
            
            return results
            
        except OperationFailure as e:
            logger.error(f"[VECTOR_SEARCH_ERR] Operation failed: {e}")
            return []
        except Exception as e:
            logger.error(f"[VECTOR_SEARCH_ERR] Unexpected error: {e}", exc_info=True)
            return []
    
    def insert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Bulk insert documents with embeddings."""
        try:
            if not documents:
                logger.warning("[INSERT] No documents to insert")
                return False
            
            self.ensure_connection()
            
            result = self.collection.insert_many(documents, ordered=False)
            logger.info(f"[INSERT_OK] Inserted {len(result.inserted_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"[INSERT_ERR] {e}")
            return False
    
    # ============================================================
    # CONVERSATION STORAGE METHODS (NEW)
    # ============================================================
    
    def get_conversation(
        self,
        user_key: str,
        conversation_id: str,
        max_messages: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Get conversation by ID with last N messages.
        
        Args:
            user_key: User's session/auth key
            conversation_id: Unique conversation identifier
            max_messages: Maximum messages to return (default: 20)
        
        Returns:
            Conversation dict or None if not found
        """
        try:
            self.ensure_connection()
            
            # Use projection to only fetch needed fields + last N messages
            result = self.conversations.find_one(
                {
                    "conversation_id": conversation_id,
                    "user_key": user_key  # Security: ensure user owns this
                },
                {
                    "conversation_id": 1,
                    "user_key": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "metadata": 1,
                    "messages": {"$slice": -max_messages}  # Only last N messages
                }
            )
            
            if result:
                logger.info(
                    f"[CONV_GET] Found conversation {conversation_id} "
                    f"with {len(result.get('messages', []))} messages"
                )
            else:
                logger.info(f"[CONV_GET] Conversation {conversation_id} not found")
            
            return result
            
        except Exception as e:
            logger.error(f"[CONV_GET_ERR] {e}", exc_info=True)
            return None
    
    def save_conversation_turn(
    self,
    user_key: str,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a conversation turn (user message + assistant response).
        Creates new conversation if doesn't exist (upsert).
        
        PRODUCTION LIMIT: Rejects conversations with >200 messages.
        
        Args:
            user_key: User's session/auth key
            conversation_id: Unique conversation identifier
            user_message: User's input message
            assistant_message: AI's response
            metadata: Optional metadata (route, model, etc.)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.ensure_connection()
            
            # PRODUCTION GUARDRAIL: Check conversation length before saving
            existing = self.conversations.find_one(
                {"conversation_id": conversation_id},
                {"metadata.message_count": 1}
            )
            
            if existing:
                current_count = existing.get("metadata", {}).get("message_count", 0)
                if current_count >= self.MAX_MESSAGES_PER_CONVERSATION:
                    logger.warning(
                        f"[CONV_SAVE] Conversation {conversation_id} "
                        f"exceeds limit ({current_count}/{self.MAX_MESSAGES_PER_CONVERSATION})"
                    )
                    return False
            
            now = datetime.utcnow()
            
            # Generate title from first user message (truncated)
            title = user_message[:60] + ("..." if len(user_message) > 60 else "")
            
            # FIXED: Separate logic for new vs existing conversations
            if not existing:
                # NEW CONVERSATION: Insert with all fields
                document = {
                    "user_key": user_key,
                    "conversation_id": conversation_id,
                    "created_at": now,
                    "updated_at": now,
                    "last_active": now,
                    "metadata": {
                        "title": title,
                        "message_count": 2,  # Starting with first turn
                        "device_info": metadata.get("device_info", {}) if metadata else {}
                    },
                    "messages": [
                        {
                            "role": "human",
                            "content": user_message,
                            "timestamp": now
                        },
                        {
                            "role": "ai",
                            "content": assistant_message,
                            "timestamp": now,
                            "metadata": metadata or {}
                        }
                    ]
                }
                
                result = self.conversations.insert_one(document)
                logger.info(f"[CONV_SAVE] Created conversation {conversation_id}")
                return True
                
            else:
                # EXISTING CONVERSATION: Update only
                result = self.conversations.update_one(
                    {"conversation_id": conversation_id},
                    {
                        "$set": {
                            "updated_at": now,
                            "last_active": now
                        },
                        "$push": {
                            "messages": {
                                "$each": [
                                    {
                                        "role": "human",
                                        "content": user_message,
                                        "timestamp": now
                                    },
                                    {
                                        "role": "ai",
                                        "content": assistant_message,
                                        "timestamp": now,
                                        "metadata": metadata or {}
                                    }
                                ]
                            }
                        },
                        "$inc": {"metadata.message_count": 2}
                    }
                )
                
                logger.info(f"[CONV_SAVE] Updated conversation {conversation_id}")
                return True
            
        except Exception as e:
            logger.error(f"[CONV_SAVE_ERR] {e}", exc_info=True)
            return False
    
    def list_conversations(
        self,
        user_key: str,
        limit: int = 20,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get user's recent conversations for sidebar.
        
        Args:
            user_key: User's session/auth key
            limit: Max conversations to return
            skip: Pagination offset
        
        Returns:
            List of conversation summaries
        """
        try:
            self.ensure_connection()
            
            cursor = self.conversations.find(
                {"user_key": user_key},
                {
                    "conversation_id": 1,
                    "metadata.title": 1,
                    "metadata.message_count": 1,
                    "updated_at": 1,
                    "last_active": 1,
                    "messages": {"$slice": -1}  # Only last message for preview
                }
            ).sort("updated_at", -1).skip(skip).limit(limit)
            
            conversations = list(cursor)
            
            # Format for frontend
            result = []
            for conv in conversations:
                last_message = conv.get("messages", [{}])[-1]
                result.append({
                    "id": conv["conversation_id"],
                    "title": conv.get("metadata", {}).get("title", "Untitled"),
                    "last_message": last_message.get("content", "")[:100],
                    "updated_at": conv["updated_at"].isoformat(),
                    "last_active": conv.get("last_active", conv["updated_at"]).isoformat(),
                    "message_count": conv.get("metadata", {}).get("message_count", 0)
                })
            
            logger.info(f"[CONV_LIST] Found {len(result)} conversations for user")
            return result
            
        except Exception as e:
            logger.error(f"[CONV_LIST_ERR] {e}", exc_info=True)
            return []
    
    def delete_conversation(
        self,
        user_key: str,
        conversation_id: str
    ) -> bool:
        """
        Delete a conversation.
        
        Args:
            user_key: User's session/auth key
            conversation_id: Unique conversation identifier
        
        Returns:
            True if deleted, False otherwise
        """
        try:
            self.ensure_connection()
            
            result = self.conversations.delete_one({
                "conversation_id": conversation_id,
                "user_key": user_key  # Security: ensure user owns this
            })
            
            if result.deleted_count > 0:
                logger.info(f"[CONV_DELETE] Deleted conversation {conversation_id}")
                return True
            
            logger.warning(f"[CONV_DELETE] Conversation {conversation_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"[CONV_DELETE_ERR] {e}", exc_info=True)
            return False
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("[MONGO_CLOSE] Connection closed")
            except Exception as e:
                logger.warning(f"[MONGO_CLOSE] Error during close: {e}")