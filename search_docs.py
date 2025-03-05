import argparse
import logging
from typing import List, Dict, Any
import json
import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables
load_dotenv()
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "documents")
MODEL_NAME = os.getenv("MODEL_NAME", "multi-qa-mpnet-base-cos-v1")
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
DEFAULT_TOP_K = 5

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Search documents using vector embeddings")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, 
                        help=f"Number of results to return (default: {DEFAULT_TOP_K})")
    return parser.parse_args()

def generate_query_embedding(query: str) -> List[float]:
    """
    Generate embedding vector for the search query.
    
    Args:
        query: The search query text
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        logger.info(f"Loading embeddings model '{MODEL_NAME}'...")
        model = SentenceTransformer(MODEL_NAME)
        
        logger.info(f"Generating embedding for query: '{query}'")
        embedding = model.encode(query)
        
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        raise

def search_documents(query_embedding: List[float], es_host: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Search for documents using vector similarity.
    
    Args:
        query_embedding: The embedding vector of the query
        es_host: Elasticsearch host URL
        top_k: Number of results to return
        
    Returns:
        List of matching documents
    """
    try:
        es = Elasticsearch([es_host])
        
        if not es.indices.exists(index=ES_INDEX_NAME):
            logger.error(f"Index '{ES_INDEX_NAME}' does not exist")
            return []
        
        search_query = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector_embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "_source": ["id", "doc_id", "path", "filename", "content", "chunk_index", "total_chunks"]
        }
        
        logger.info(f"Searching in index '{ES_INDEX_NAME}'...")
        response = es.search(index=ES_INDEX_NAME, body=search_query)
        
        results = []
        for hit in response["hits"]["hits"]:
            result = hit["_source"]
            result["score"] = hit["_score"]
            results.append(result)
        
        logger.info(f"Found {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error searching in Elasticsearch: {e}")
        return []

def format_results(results: List[Dict[str, Any]]) -> None:
    """
    Format and print search results.
    
    Args:
        results: List of search results
    """
    if not results:
        print("No results found.")
        return
    
    print(f"\n{'=' * 80}\n")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results):
        print(f"Result {i+1} [Score: {result['score']:.4f}]")
        print(f"Document: {result['filename']} (Chunk {result['chunk_index']+1}/{result['total_chunks']})")
        print(f"Path: {result['path']}")
        print(f"Content: {result['content']}")
        print(f"\n{'=' * 80}\n")

def main():
    """Main function that executes the search process."""
    args = parse_arguments()
    
    try:
        query_embedding = generate_query_embedding(args.query)
        results = search_documents(query_embedding, ES_HOST, args.top_k)
        format_results(results)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")

if __name__ == "__main__":
    main() 