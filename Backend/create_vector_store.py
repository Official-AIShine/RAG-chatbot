"""
Vector Store Creation Script with Semantic Chunking

Creates MongoDB vector store from knowledge base JSON files.
Supports semantic chunking for better retrieval quality.

Usage:
    python -m Backend.create_vector_store
    python -m Backend.create_vector_store --kb-dir ./data
    python -m Backend.create_vector_store --chunk
    python.exe -m Backend.create_vector_store --module 8 --clear-module --chunk to target only module 8 with chunking and clear existing docs for module 8.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from Backend.embedding_client import BedrockEmbeddingClient
from Backend.mongodb_client import MongoDBClient
from Backend.config import settings
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def find_kb_files(kb_dir: Optional[Path] = None) -> List[Path]:
    """
    Find all Parsed_Module*_KB.json files.

    Args:
        kb_dir: Directory to search, defaults to project root

    Returns:
        List of KB file paths
    """
    if kb_dir is None:
        kb_dir = get_project_root()

    kb_files = list(kb_dir.glob("Parsed_Module*_KB.json"))
    kb_files.sort()  # Sort by module number

    logger.info(f"Found {len(kb_files)} KB files in {kb_dir}")
    for f in kb_files:
        logger.info(f"  - {f.name}")

    return kb_files


def load_json_file(file_path: Path) -> Optional[Any]:
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {file_path.name}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def chunk_content(content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split content into overlapping chunks.

    Semantic chunking strategy:
    1. Split by paragraphs first
    2. Combine paragraphs until reaching chunk_size
    3. Include overlap between chunks for context continuity

    Args:
        content: Text to chunk
        chunk_size: Target chunk size in words
        overlap: Number of words to overlap between chunks

    Returns:
        List of text chunks
    """
    if not content:
        return []

    # Split by paragraphs
    paragraphs = content.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        paragraphs = [content]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If single paragraph exceeds chunk size, split it
        if para_words > chunk_size:
            # Save current chunk first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0

            # Split long paragraph by sentences
            sentences = para.replace('. ', '.|').split('|')
            for sentence in sentences:
                sentence_words = len(sentence.split())
                if current_word_count + sentence_words > chunk_size:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        # Keep overlap
                        overlap_start = max(0, len(current_chunk) - overlap // 10)
                        current_chunk = current_chunk[overlap_start:]
                        current_word_count = sum(len(w.split()) for w in current_chunk)
                current_chunk.append(sentence)
                current_word_count += sentence_words

        elif current_word_count + para_words > chunk_size:
            # Save current chunk and start new one with overlap
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last few elements for overlap
                overlap_items = max(1, len(current_chunk) // 4)
                current_chunk = current_chunk[-overlap_items:]
                current_word_count = sum(len(w.split()) for w in current_chunk)

            current_chunk.append(para)
            current_word_count += para_words
        else:
            current_chunk.append(para)
            current_word_count += para_words

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def create_kb_documents(
    kb_json: List[Dict[str, Any]],
    embedding_client: BedrockEmbeddingClient,
    module_name: str,
    use_chunking: bool = False
) -> List[Dict[str, Any]]:
    """
    Convert KB JSON entries into MongoDB documents with embeddings.

    Args:
        kb_json: Parsed KB JSON data
        embedding_client: Bedrock embedding client
        module_name: Module identifier (e.g., "module1_kb")
        use_chunking: Whether to split content into chunks

    Returns:
        List of documents ready for MongoDB
    """
    documents = []

    for idx, entry in enumerate(kb_json):
        topic = entry.get('topic', '')
        logger.info(f"[{module_name}] Processing {idx + 1}/{len(kb_json)}: {topic}")

        # Build embedding text
        summary = entry.get('summary', '')
        content = entry.get('content', '')
        keywords = ' '.join(entry.get('keywords', []))

        if use_chunking and len(content.split()) > settings.CHUNK_SIZE:
            # Create multiple documents from chunks
            chunks = chunk_content(
                content,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP
            )

            logger.info(f"  Chunked into {len(chunks)} parts")

            for chunk_idx, chunk in enumerate(chunks):
                embedding_text = f"Topic: {topic}\n\nSummary: {summary}\n\nContent: {chunk}\n\nKeywords: {keywords}"

                embedding = embedding_client.generate_embedding(embedding_text)
                if not embedding:
                    logger.warning(f"  Skipped chunk {chunk_idx + 1} - embedding failed")
                    continue

                document = {
                    "topic": topic,
                    "category": entry.get('category', ''),
                    "level": entry.get('level', ''),
                    "type": entry.get('type', ''),
                    "summary": summary,
                    "content": chunk,
                    "keywords": entry.get('keywords', []),
                    "module_name": module_name,
                    "source": "knowledge_base",
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "embedding": embedding
                }
                documents.append(document)
        else:
            # Single document for entry
            embedding_text = f"Topic: {topic}\n\nSummary: {summary}\n\nContent: {content}\n\nKeywords: {keywords}"

            embedding = embedding_client.generate_embedding(embedding_text)
            if not embedding:
                logger.warning(f"  Skipped - embedding failed")
                continue

            document = {
                "topic": topic,
                "category": entry.get('category', ''),
                "level": entry.get('level', ''),
                "type": entry.get('type', ''),
                "summary": summary,
                "content": content,
                "keywords": entry.get('keywords', []),
                "module_name": module_name,
                "source": "knowledge_base",
                "embedding": embedding
            }
            documents.append(document)

    return documents


def create_presentation_documents(
    presentation_json: Dict[str, Any],
    embedding_client: BedrockEmbeddingClient
) -> List[Dict[str, Any]]:
    """
    Convert presentation.json prompts into MongoDB documents.

    Args:
        presentation_json: Parsed presentation JSON
        embedding_client: Bedrock embedding client

    Returns:
        List of documents ready for MongoDB
    """
    documents = []
    prompts = presentation_json.get('prompts', [])

    for idx, prompt in enumerate(prompts):
        title = prompt.get('title', '')
        logger.info(f"[presentation] Processing {idx + 1}/{len(prompts)}: {title}")

        response = prompt.get('response', {})

        # Build embedding text
        embedding_parts = [f"Topic: {title}"]

        if 'intro' in response:
            embedding_parts.append(f"Introduction: {response['intro']}")
        if 'description' in response:
            embedding_parts.append(f"Description: {response['description']}")
        if 'features' in response:
            for feature in response['features']:
                embedding_parts.append(f"{feature['title']}: {feature['description']}")

        embedding_text = "\n".join(embedding_parts)
        embedding = embedding_client.generate_embedding(embedding_text)

        if not embedding:
            logger.warning(f"  Skipped - embedding failed")
            continue

        # Extract keywords
        keywords = []
        keywords.extend(title.lower().split())
        keywords.extend(prompt.get('aliases', []))

        document = {
            "topic": title,
            "category": "Presentation",
            "level": "Introductory",
            "type": "Workshop Prompt",
            "summary": response.get('intro', '')[:500],
            "content": "",
            "keywords": list(set(keywords)),
            "module_name": "presentation",
            "source": "presentation",
            "presentation_data": response,
            "embedding": embedding
        }
        documents.append(document)

    return documents


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Create vector store from KB files')
    parser.add_argument(
        '--kb-dir',
        type=Path,
        default=None,
        help='Directory containing KB JSON files (default: project root)'
    )
    parser.add_argument(
        '--chunk',
        action='store_true',
        help='Enable semantic chunking for long documents'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear ALL existing documents before inserting'
    )
    parser.add_argument(
        '--module',
        type=int,
        nargs='+',
        help='Only process specific module number(s), e.g., --module 8 or --module 5 6 7 8'
    )
    parser.add_argument(
        '--clear-module',
        action='store_true',
        help='Clear existing docs for specified modules before inserting (use with --module)'
    )

    args = parser.parse_args()

    # Initialize clients
    logger.info("=" * 60)
    logger.info("VECTOR STORE CREATION")
    logger.info("=" * 60)

    embedding_client = BedrockEmbeddingClient()
    mongo_client = MongoDBClient()

    all_documents = []

    # Find and process KB files
    kb_dir = args.kb_dir or get_project_root()
    kb_files = find_kb_files(kb_dir)

    # Filter to specific modules if requested
    if args.module:
        module_numbers = set(args.module)
        kb_files = [
            f for f in kb_files
            if any(f"module{n}" in f.name.lower() for n in module_numbers)
        ]
        logger.info(f"Filtered to modules: {sorted(module_numbers)}")
        logger.info(f"Processing {len(kb_files)} files: {[f.name for f in kb_files]}")

        # Clear specific modules if requested
        if args.clear_module:
            for n in module_numbers:
                module_name = f"module{n}_kb"
                result = mongo_client.collection.delete_many({"module_name": module_name})
                logger.warning(f"Cleared {result.deleted_count} docs from {module_name}")

    for kb_file in kb_files:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {kb_file.name}")
        logger.info("=" * 60)

        kb_json = load_json_file(kb_file)
        if not kb_json:
            continue

        # Extract module name from filename (e.g., "Parsed_Module5_KB.json" -> "module5_kb")
        module_name = kb_file.stem.lower().replace('parsed_', '').replace('_kb', '_kb')

        docs = create_kb_documents(
            kb_json,
            embedding_client,
            module_name=module_name,
            use_chunking=args.chunk
        )
        all_documents.extend(docs)
        logger.info(f"Created {len(docs)} documents from {kb_file.name}")

    # Check for presentation.json
    presentation_path = kb_dir / "presentation.json"
    if presentation_path.exists():
        logger.info(f"\n{'=' * 60}")
        logger.info("Processing: presentation.json")
        logger.info("=" * 60)

        presentation_json = load_json_file(presentation_path)
        if presentation_json:
            pres_docs = create_presentation_documents(presentation_json, embedding_client)
            all_documents.extend(pres_docs)
            logger.info(f"Created {len(pres_docs)} presentation documents")

    # Insert documents
    logger.info(f"\n{'=' * 60}")
    logger.info(f"INSERTING {len(all_documents)} TOTAL DOCUMENTS")
    logger.info("=" * 60)

    if args.clear:
        logger.warning("Clearing existing documents...")
        mongo_client.collection.delete_many({})

    if all_documents:
        success = mongo_client.insert_documents(all_documents)
        if success:
            logger.info(f"Successfully inserted {len(all_documents)} documents")
        else:
            logger.error("Failed to insert documents")
    else:
        logger.warning("No documents to insert")

    mongo_client.close()
    logger.info("\n[COMPLETE] Vector store creation finished")


if __name__ == "__main__":
    main()
