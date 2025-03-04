import os
import glob
from pathlib import Path
from typing import List, Dict, Any
import logging

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from chonkie import RecursiveChunker, RecursiveLevel, RecursiveRules
from sentence_transformers import SentenceTransformer

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "data/md"
ES_INDEX_NAME = "documents"
CHUNK_SIZE = 1024  # Approximate size of each chunk in characters https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5
MODEL_NAME = "multi-qa-mpnet-base-cos-v1"  # Model for generating embeddings

def read_markdown_files(directory: str) -> List[Dict[str, Any]]:
    """
    Reads all markdown files in the specified directory.
    
    Args:
        directory: Path to the directory with markdown files
        
    Returns:
        List of dictionaries with content and metadata of each file
    """
    documents = []
    markdown_files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)
    
    logger.info(f"Found {len(markdown_files)} markdown files")
    
    for file_path in markdown_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            relative_path = os.path.relpath(file_path, directory)
            doc_id = Path(relative_path).stem
            
            documents.append({
                "id": doc_id,
                "path": relative_path,
                "content": content,
                "filename": os.path.basename(file_path)
            })
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    return documents

def create_chunks(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Divides documents into chunks using chonkie.
    
    Args:
        documents: List of documents to process
        
    Returns:
        List of chunks with their metadata
    """
    rules = RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=["######", "#####", "####", "###", "##", "#"], include_delim="next"),
            RecursiveLevel(delimiters=["\n\n", "\n", "\r\n", "\r"]),
            RecursiveLevel(delimiters=".?!;:"),
            RecursiveLevel(),
        ]
    )
    chunker = RecursiveChunker(rules=rules, chunk_size=CHUNK_SIZE)    

    all_chunks = []
    for doc in documents:
        try:
            doc_chunks = chunker(doc["content"])
            logger.info(f"Total number of chunks: {len(doc_chunks)}")

            for i, chunk_text in enumerate(doc_chunks):
                logger.info("--------------------------------")
                logger.info(f"Chunk {i}: {chunk_text}")
                logger.info("--------------------------------")

                all_chunks.append({
                    "id": f"{doc['id']}_chunk_{i}",
                    "doc_id": doc["id"],
                    "path": doc["path"],
                    "filename": doc["filename"],
                    "chunk_index": i,
                    "content": chunk_text,
                    "total_chunks": len(doc_chunks)
                })
                
        except Exception as e:
            logger.error(f"Error creating chunks for document {doc['id']}: {e}")
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks

def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates embeddings for each chunk using a sentence-transformers model.
    
    Args:
        chunks: List of chunks to generate embeddings for
        
    Returns:
        List of chunks with their embeddings added
    """
    try:
        logger.info(f"Loading embeddings model '{MODEL_NAME}'...")
        model = SentenceTransformer(MODEL_NAME)
        
        # Extract texts to generate embeddings in batch
        texts = [chunk["content"] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["vector_embedding"] = embeddings[i].tolist()
            logger.info("--------------------------------")
            logger.info(f"Chunk {i}: {chunk}")
            logger.info("--------------------------------")
        
        logger.info(f"Embeddings generated successfully")
        return chunks
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return chunks

def index_chunks_to_elasticsearch(chunks: List[Dict[str, Any]], es_host: str = "localhost:9200"):
    """
    Indexes chunks into Elasticsearch.
    
    Args:
        chunks: List of chunks to index
        es_host: Elasticsearch host and port
    """
    try:
        # Connect to Elasticsearch
        es = Elasticsearch([es_host])
        
        # Check if index exists, if not, create it
        if not es.indices.exists(index=ES_INDEX_NAME):
            # Get embedding dimension from the first chunk
            embedding_dims = len(chunks[0].get("vector_embedding", [])) if chunks else 768
            
            es.indices.create(
                index=ES_INDEX_NAME,
                body={
                    "mappings": {
                        "properties": {
                            "id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                            "path": {"type": "keyword"},
                            "filename": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "total_chunks": {"type": "integer"},
                            "content": {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            "vector_embedding": {
                                "type": "dense_vector",
                                "dims": embedding_dims,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
            )
            logger.info(f"Created index '{ES_INDEX_NAME}' with embedding dimension: {embedding_dims}")
        
        # Prepare actions for bulk indexing
        actions = []
        for chunk in chunks:
            action = {
                "_index": ES_INDEX_NAME,
                "_id": chunk["id"],
                "_source": chunk
            }
            actions.append(action)
        
        # Perform bulk indexing
        success, failed = bulk(es, actions, stats_only=True)
        logger.info(f"Indexed {success} chunks in Elasticsearch. Failures: {failed}")
        
    except Exception as e:
        logger.error(f"Error indexing to Elasticsearch: {e}")

def main():
    """Main function that executes the complete process"""
    logger.info("Starting document processing")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        logger.error(f"Directory {DATA_DIR} does not exist")
        return
    
    # Read documents
    documents = read_markdown_files(DATA_DIR)
    if not documents:
        logger.warning("No documents found to process")
        return
    
    # Create chunks
    chunks = create_chunks(documents)
    if not chunks:
        logger.warning("Could not create chunks")
        return
    
    # Generate embeddings for chunks
    chunks_with_embeddings = generate_embeddings(chunks)
    
    # Index to Elasticsearch
    #index_chunks_to_elasticsearch(chunks_with_embeddings)
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()
