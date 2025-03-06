import argparse
import logging
from typing import List, Dict, Any
import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import lmstudio as lms

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
ES_HOST = os.getenv("ES_HOST")
DEFAULT_TOP_K = 5
LLM_MODEL = os.getenv("LLM_MODEL")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Search documents using vector embeddings")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, 
                        help=f"Number of results to return (default: {DEFAULT_TOP_K})")
    parser.add_argument("--rag", action="store_true", 
                        help="Use RAG to generate an answer using retrieved documents")
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
    https://www.elastic.co/search-labs/blog/introduction-to-vector-search
    https://www.elastic.co/search-labs/blog/vector-search-set-up-elasticsearch
    https://www.elastic.co/search-labs/blog/hybrid-search-elasticsearch

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
                "knn": {
                    "field": "vector_embedding",
                    "query_vector": query_embedding,
                    "num_candidates": 100
                }
            },
            "_source": ["id", "doc_id", "path", "filename", "content", "chunk_index", "total_chunks"]
        }
        
        logger.info(f"Searching in index '{ES_INDEX_NAME}' using KNN...")
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

def generate_rag_response(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Generate a response using RAG (Retrieval Augmented Generation).
    
    Args:
        query: The user's query
        results: The retrieved documents to use as context
        
    Returns:
        Generated response from the LLM
    """
    try:
        logger.info(f"Generating RAG response using LLM model '{LLM_MODEL}'...")
        context = ""
        for i, result in enumerate(results):
            context += f"Document {i+1}: {result['content']}\n\n"
        
        prompt = f"""You are a helpful assistant. Use the following retrieved documents to answer the user's question.
If you don't know the answer based on these documents, just say so.

RETRIEVED DOCUMENTS:
{context}

USER QUESTION: {query}

ANSWER:"""

        model = lms.llm(LLM_MODEL)
        response = model.respond(prompt)
        
        logger.info("RAG response generated successfully")

        return response
        
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return f"Error generating response: {str(e)}"

def process_search_query_using_rag(query: str, top_k: int) -> tuple:
    """
    Process a search query, retrieve documents, and optionally generate a RAG response.
    
    Args:
        query: The search query
        top_k: Number of results to return
        
    Returns:
        tuple: A tuple containing (response, results) where response is the generated answer
              and results is the list of retrieved documents
    """
    try:
        query_embedding = generate_query_embedding(query)
        results = search_documents(query_embedding, ES_HOST, top_k)
        response = generate_rag_response(query, results)

        return response, results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return None, []
