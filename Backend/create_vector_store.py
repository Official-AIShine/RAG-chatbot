"""
Vector Store Creation Script
Ingests KB JSON and presentation.json, generates embeddings, and populates MongoDB.
Handles both knowledge base and presentation prompts in unified collection.
"""
import os
import json
import logging
from typing import List, Dict, Any
from Backend.embedding_client import BedrockEmbeddingClient
from Backend.mongodb_client import MongoDBClient
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_file(file_path: str) -> Any:
    """Load any JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"[LOAD] Loaded {file_path}")
        return data
    except Exception as e:
        logger.error(f"[LOAD_ERR] Failed to load {file_path}: {e}")
        return None


def extract_presentation_keywords(prompt: Dict[str, Any]) -> List[str]:
    """
    Extract keywords from presentation prompt for better matching.
    Combines title, aliases, and key terms from response.
    """
    keywords = []
    
    # Add title words
    keywords.extend(prompt.get('title', '').lower().split())
    
    # Add aliases
    keywords.extend(prompt.get('aliases', []))
    
    # Extract key terms from response
    response = prompt.get('response', {})
    
    # From intro
    if 'intro' in response:
        keywords.append(response['intro'].lower())
    
    # From features
    if 'features' in response:
        for feature in response['features']:
            keywords.append(feature.get('title', '').lower())
    
    # From activities
    if 'activities' in response:
        for activity in response['activities']:
            keywords.append(activity.get('title', '').lower())
    
    # From careers
    if 'careers' in response:
        for career in response['careers']:
            keywords.append(career.get('title', '').lower())
    
    # Deduplicate and clean
    keywords = list(set([k.strip() for k in keywords if k.strip()]))
    return keywords


def create_presentation_documents(
    presentation_json: Dict[str, Any],
    embedding_client: BedrockEmbeddingClient
) -> List[Dict[str, Any]]:
    """
    Convert presentation.json prompts into MongoDB documents with embeddings.
    """
    documents = []
    prompts = presentation_json.get('prompts', [])
    
    for idx, prompt in enumerate(prompts):
        logger.info(f"[PRESENTATION] Processing {idx+1}/{len(prompts)}: {prompt.get('title')}")
        
        # Extract response content for embedding
        response = prompt.get('response', {})
        
        # Build embedding text
        embedding_parts = [
            f"Topic: {prompt.get('title', '')}",
        ]
        
        if 'intro' in response:
            embedding_parts.append(f"Introduction: {response['intro']}")
        
        if 'description' in response:
            embedding_parts.append(f"Description: {response['description']}")
        
        if 'features' in response:
            for feature in response['features']:
                embedding_parts.append(f"{feature['title']}: {feature['description']}")
        
        if 'activities' in response:
            for activity in response['activities']:
                embedding_parts.append(f"{activity['title']}: {activity['description']}")
        
        embedding_text = "\n".join(embedding_parts)
        
        # Generate embedding
        embedding = embedding_client.generate_embedding(embedding_text)
        
        if not embedding:
            logger.warning(f"[SKIP] Failed to generate embedding for {prompt.get('title')}")
            continue
        
        # Create summary from response
        summary_parts = []
        if 'intro' in response:
            summary_parts.append(response['intro'])
        if 'features' in response:
            for feature in response['features']:
                summary_parts.append(f"{feature['title']}: {feature['description']}")
        
        summary = " ".join(summary_parts[:3])  # First 3 parts only
        
        # Extract keywords
        keywords = extract_presentation_keywords(prompt)
        
        # Create document
        document = {
            "topic": prompt.get('title', ''),
            "category": "Presentation",
            "level": "Introductory",
            "type": "Workshop Prompt",
            "summary": summary,
            "content": "",  # Presentation has no detailed content
            "keywords": keywords,
            "module_name": "presentation",
            "source": "presentation",
            "presentation_data": response,  # Store full response for formatting
            "embedding": embedding
        }
        
        documents.append(document)
    
    return documents


def create_kb_documents(
    kb_json: List[Dict[str, Any]],
    embedding_client: BedrockEmbeddingClient,
    module_name: str = "module1_kb"
) -> List[Dict[str, Any]]:
    """
    Convert KB JSON entries into MongoDB documents with embeddings.
    """
    documents = []
    
    for idx, entry in enumerate(kb_json):
        logger.info(f"[KB] Processing {idx+1}/{len(kb_json)}: {entry.get('topic')}")
        
        # Create embedding text (topic + summary + content + keywords)
        topic = entry.get('topic', '')
        summary = entry.get('summary', '')
        content = entry.get('content', '')
        keywords = ' '.join(entry.get('keywords', []))
        
        embedding_text = f"Topic: {topic}\n\nSummary: {summary}\n\nContent: {content}\n\nKeywords: {keywords}"
        
        # Generate embedding
        embedding = embedding_client.generate_embedding(embedding_text)
        
        if not embedding:
            logger.warning(f"[SKIP] Failed to generate embedding for {topic}")
            continue
        
        # Create document
        document = {
            "topic": entry.get('topic', ''),
            "category": entry.get('category', ''),
            "level": entry.get('level', ''),
            "type": entry.get('type', ''),
            "summary": entry.get('summary', ''),
            "content": content,
            "keywords": entry.get('keywords', []),
            "module_name": module_name,
            "source": "knowledge_base",
            "embedding": embedding
        }
        
        documents.append(document)
    
    return documents


def main():
    """Main execution: ingest both KB and presentation into unified collection."""
    
    # Initialize clients
    embedding_client = BedrockEmbeddingClient()
    mongo_client = MongoDBClient()
    
    all_documents = []
    
    # ===== 1. Process Presentation.json =====
    logger.info("\n" + "="*60)
    logger.info("PROCESSING PRESENTATION.JSON")
    logger.info("="*60 + "\n")
    
    presentation_path = r"C:\Users\newbr\OneDrive\Desktop\AISHINEBE_CLAUDE\presentation.json"
    presentation_json = load_json_file(presentation_path)
    
    if presentation_json:
        pres_docs = create_presentation_documents(presentation_json, embedding_client)
        all_documents.extend(pres_docs)
        logger.info(f"✅ Created {len(pres_docs)} presentation documents")
    else:
        logger.error("❌ Failed to load presentation.json")
    
    # ===== 2. Process KB JSON =====
    logger.info("\n" + "="*60)
    logger.info("PROCESSING KNOWLEDGE BASE")
    logger.info("="*60 + "\n")
    
    kb_path = r"C:\Users\newbr\OneDrive\Desktop\AISHINEBE_CLAUDE\Parsed_Module1_KB.json"
    kb_json = load_json_file(kb_path)
    
    if kb_json:
        kb_docs = create_kb_documents(kb_json, embedding_client, module_name="module1_kb")
        all_documents.extend(kb_docs)
        logger.info(f"✅ Created {len(kb_docs)} KB documents")
    else:
        logger.error("❌ Failed to load KB JSON")
    
    # ===== 3. Bulk Insert =====
    logger.info("\n" + "="*60)
    logger.info(f"INSERTING {len(all_documents)} TOTAL DOCUMENTS")
    logger.info("="*60 + "\n")
    
    if all_documents:
        success = mongo_client.insert_documents(all_documents)
        if success:
            logger.info(f"✅ Successfully inserted all documents into module_vectors collection")
        else:
            logger.error("❌ Failed to insert documents")
    else:
        logger.error("❌ No documents to insert")
    
    # Close connections
    mongo_client.close()
    logger.info("\n[COMPLETE] Vector store creation finished")


if __name__ == "__main__":
    main()