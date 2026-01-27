"""
UNTESTED AWS Bedrock Embedding Client with LRU Caching
Uses Amazon Titan Embed Text v2 for generating 1024-dimensional embeddings.
Optimized with in-memory cache for 200ms latency reduction on repeated queries.
"""
import os
import logging
import json
from typing import List, Optional
from functools import lru_cache
import unicodedata
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BedrockEmbeddingClient:
    """AWS Bedrock client for Titan v2 embeddings with caching."""
    
    def __init__(self):
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
        self.model_id = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
        
        if not all([self.aws_access_key, self.aws_secret_key]):
            raise ValueError("[BEDROCK_ERR] AWS credentials not set")
        
        try:
            self.client = boto3.client(
                service_name='bedrock-runtime',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            logger.info(f"[BEDROCK_OK] Connected to {self.model_id}")
        except Exception as e:
            logger.error(f"[BEDROCK_ERR] Initialization failed: {e}")
            raise
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for consistent embeddings.
        - NFKC normalization
        - Lowercase
        - Strip extra whitespace
        """
        text = unicodedata.normalize('NFKC', text)
        text = text.lower().strip()
        text = ' '.join(text.split())  # Collapse multiple spaces
        return text
    
    @lru_cache(maxsize=100)
    def _cached_embedding(self, normalized_text: str) -> Optional[tuple]:
        """
        Generate embedding with LRU cache.
        
        CRITICAL: Uses tuple return for hashability (required by lru_cache).
        Cache stores 100 most recent normalized queries.
        Industry standard practice for embedding APIs with high latency (~200ms).
        
        Args:
            normalized_text: Pre-normalized text (immutable for caching)
        
        Returns:
            Tuple of embedding values, or None on failure
        """
        try:
            body = json.dumps({
                "inputText": normalized_text,
                "dimensions": 1024,
                "normalize": True
            })
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding')
            
            if not embedding or len(embedding) != 1024:
                logger.error(f"[EMBEDDING_ERR] Invalid embedding dimension: {len(embedding) if embedding else 0}")
                return None
            
            logger.info(f"[EMBEDDING_OK] Generated 1024-dim vector")
            return tuple(embedding)  # Convert to tuple for caching
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"[EMBEDDING_ERR] AWS error: {e}")
            return None
        except Exception as e:
            logger.error(f"[EMBEDDING_ERR] Unexpected error: {e}")
            return None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate a 1024-dim embedding for a single text with caching.
        
        Args:
            text: Input text to embed
        
        Returns:
            List of 1024 floats, or None on failure
        """
        # Normalize first (consistent cache keys)
        normalized = self.normalize_text(text)
        
        # Check cache stats periodically
        cache_info = self._cached_embedding.cache_info()
        if cache_info.hits + cache_info.misses > 0:
            hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses) * 100
            if (cache_info.hits + cache_info.misses) % 10 == 0:  # Log every 10 calls
                logger.info(f"[CACHE_STATS] Hits: {cache_info.hits}, Misses: {cache_info.misses}, Hit Rate: {hit_rate:.1f}%")
        
        # Get cached or generate new
        embedding_tuple = self._cached_embedding(normalized)
        
        if embedding_tuple is None:
            return None
        
        return list(embedding_tuple)  # Convert back to list
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of embeddings (same order as input)
        """
        embeddings = []
        for idx, text in enumerate(texts):
            logger.info(f"[BATCH] Processing {idx+1}/{len(texts)}")
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings


















