"""
AWS Bedrock Embedding Client with LRU Caching

Uses Amazon Titan Embed Text v2 for generating 1024-dimensional embeddings.
Optimized with in-memory cache for latency reduction on repeated queries.
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

from Backend.config import settings

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BedrockEmbeddingClient:
    """AWS Bedrock client for Titan v2 embeddings with caching."""

    def __init__(self):
        """Initialize Bedrock client."""
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = settings.AWS_REGION
        self.model_id = settings.EMBEDDING_MODEL_ID

        if not all([self.aws_access_key, self.aws_secret_key]):
            raise ValueError("[BEDROCK] AWS credentials not set")

        try:
            self.client = boto3.client(
                service_name='bedrock-runtime',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            logger.info(f"[BEDROCK] Connected to {self.model_id}")
        except Exception as e:
            logger.error(f"[BEDROCK] Initialization failed: {e}")
            raise

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for consistent embeddings.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        text = unicodedata.normalize('NFKC', text)
        text = text.lower().strip()
        text = ' '.join(text.split())  # Collapse whitespace
        return text

    @lru_cache(maxsize=100)
    def _cached_embedding(self, normalized_text: str) -> Optional[tuple]:
        """
        Generate embedding with LRU cache.

        Uses tuple return for hashability (required by lru_cache).

        Args:
            normalized_text: Pre-normalized text

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
                logger.error(f"[EMBEDDING] Invalid dimension: {len(embedding) if embedding else 0}")
                return None

            return tuple(embedding)

        except (BotoCoreError, ClientError) as e:
            logger.error(f"[EMBEDDING] AWS error: {e}")
            return None
        except Exception as e:
            logger.error(f"[EMBEDDING] Unexpected error: {e}")
            return None

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate a 1024-dim embedding with caching.

        Args:
            text: Input text to embed

        Returns:
            List of 1024 floats, or None on failure
        """
        normalized = self.normalize_text(text)
        embedding_tuple = self._cached_embedding(normalized)

        if embedding_tuple is None:
            return None

        return list(embedding_tuple)

    def generate_batch_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embeddings (same order as input)
        """
        return [self.generate_embedding(text) for text in texts]

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        info = self._cached_embedding.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize,
            "hit_rate": info.hits / (info.hits + info.misses) * 100 if (info.hits + info.misses) > 0 else 0
        }
