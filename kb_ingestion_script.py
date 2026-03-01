"""
Multi-module ingestion script with comprehensive error handling
"""
import json
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from Backend.embedding_client import BedrockEmbeddingClient
from Backend.mongodb_client import MongoDBClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Retry decorator for transient failures
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def generate_embedding_with_retry(embedder, content):
    """Generate embedding with automatic retry on timeout/connection errors."""
    return embedder.generate_embedding(content)

def validate_document(doc, idx):
    """Validate document has required fields."""
    required_fields = ["topic", "content", "module_name"]
    missing = [f for f in required_fields if f not in doc or not doc[f]]
    
    if missing:
        logger.error(f"❌ Doc {idx}: Missing required fields: {missing}")
        return False
    
    if not isinstance(doc.get("keywords", []), list):
        logger.warning(f"⚠️  Doc {idx}: 'keywords' should be a list, got {type(doc['keywords'])}")
    
    return True

def main():
    # Initialize clients with error handling
    try:
        embedder = BedrockEmbeddingClient()
        logger.info("✅ Bedrock embeddings client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Bedrock client: {e}")
        return
    
    try:
        mongo_client = MongoDBClient()
        collection = mongo_client.collection
        logger.info("✅ MongoDB client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize MongoDB client: {e}")
        return
    
    # Test MongoDB connection
    try:
        mongo_client.client.admin.command('ping')
        logger.info("✅ MongoDB connection verified")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        return
    
    # Files to ingest
    files = [
            # r"C:\Users\newbr\OneDrive\Desktop\AISHINEBE_CLAUDE\faqs.json",
        # r"C:\Users\newbr\OneDrive\Desktop\AISHINEBE_CLAUDE_unifiedrag\Parsed_Module6_KB.json",
        # r"C:\Users\newbr\OneDrive\Desktop\AISHINEBE_CLAUDE_unifiedrag\Parsed_Module7_KB.json",
        # r"C:\Users\newbr\OneDrive\Desktop\AISHINEBE_CLAUDE_unifiedrag\Parsed_Module8_KB.json",
        # r"C:\Users\newbr\OneDrive\Desktop\AISHINEBE_CLAUDE_unifiedrag\Parsed_Module9_KB.json"
    ]
    
    total_inserted = 0
    total_failed = 0
    total_skipped = 0
    
    for file_path in files:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {file_path}")
        logger.info(f"{'='*70}")
        
        # Load JSON file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            if not isinstance(documents, list):
                logger.error(f"❌ Expected list, got {type(documents)}")
                continue
            
            logger.info(f"✅ Loaded {len(documents)} documents from file")
            
        except FileNotFoundError:
            logger.error(f"❌ File not found: {file_path}")
            logger.error(f"   Check path and ensure file exists")
            continue
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in {file_path}: {e}")
            continue
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            continue
        
        # Process each document
        for idx, doc in enumerate(documents, 1):
            try:
                # Validate document structure
                if not validate_document(doc, idx):
                    total_skipped += 1
                    continue
                
                # Check for duplicate topic
                existing = collection.find_one({
                    "topic": doc["topic"],
                    "source": "knowledge_base"
                })
                
                if existing:
                    logger.warning(f"⚠️  [{idx}/{len(documents)}] Topic already exists: {doc['topic'][:50]}... (SKIPPING)")
                    total_skipped += 1
                    continue
                
                # Generate embedding with retry
                logger.info(f"[{idx}/{len(documents)}] Generating embedding: {doc['topic'][:50]}...")
                
                try:
                    embedding = generate_embedding_with_retry(embedder, doc["content"])
                except Exception as e:
                    logger.error(f"❌ Embedding generation failed after retries: {e}")
                    total_failed += 1
                    continue
                
                # Validate embedding dimensions
                if len(embedding) != 1024:
                    logger.error(f"❌ Invalid embedding dimensions: expected 1024, got {len(embedding)}")
                    total_failed += 1
                    continue
                
                # Add metadata
                doc["embedding"] = embedding
                doc["source"] = "knowledge_base"  # CRITICAL for retriever filter
                
                # Remove _id if present (let MongoDB generate)
                if "_id" in doc:
                    del doc["_id"]
                
                # Insert to MongoDB
                try:
                    result = collection.insert_one(doc)
                    logger.info(f"✅ [{idx}/{len(documents)}] Inserted: {doc['topic']} (ID: {result.inserted_id})")
                    total_inserted += 1
                    
                    # Rate limiting (be nice to APIs)
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ MongoDB insert failed: {e}")
                    total_failed += 1
                    continue
                
            except KeyError as e:
                logger.error(f"❌ Doc {idx}: Missing key {e}")
                total_failed += 1
            except Exception as e:
                logger.error(f"❌ Doc {idx}: Unexpected error: {e}")
                total_failed += 1
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info(f"INGESTION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"✅ Successfully inserted: {total_inserted}")
    logger.info(f"⚠️  Skipped (duplicates): {total_skipped}")
    logger.info(f"❌ Failed: {total_failed}")
    
    # Verify final counts
    try:
        total_kb_docs = collection.count_documents({"source": "knowledge_base"})
        logger.info(f"\n📊 Total knowledge_base documents in DB: {total_kb_docs}")
        
        # Count by module
        modules = collection.distinct("module_name", {"source": "knowledge_base"})
        logger.info(f"📦 Modules in DB: {sorted(modules)}")
        
        for module in sorted(modules):
            count = collection.count_documents({
                "module_name": module,
                "source": "knowledge_base"
            })
            logger.info(f"   - {module}: {count} documents")
            
    except Exception as e:
        logger.error(f"❌ Failed to verify counts: {e}")
    
    logger.info(f"{'='*70}\n")

if __name__ == "__main__":
    main()