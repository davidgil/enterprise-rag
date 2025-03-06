import argparse
import logging
from typing import List, Dict, Any
import json
from decomposed_search import DecomposedSearchProcessor
import lmstudio as lms
import os
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")

def summarize_responses(original_question: str, responses: List[Dict[str, Any]]) -> str:
    """
    Summarize multiple responses using an LLM.
    
    Args:
        original_question: The original user question
        responses: List of response objects containing questions and answers
        
    Returns:
        A summarized response from the LLM
    """
    logger.info(f"Summarizing responses for question: {original_question}")
    
    # Prepare the context from all responses
    context = ""
    for i, response_obj in enumerate(responses):
        sub_question = response_obj["question"]
        response = response_obj["response"]
        context += f"Sub-question {i+1}: {sub_question}\nAnswer {i+1}: {response}\n\n"
    
    # Create the prompt for the LLM
    prompt = f"""You are a helpful assistant. The user's original question has been decomposed into multiple sub-questions, 
and each sub-question has been answered separately. Your task is to synthesize these answers into a single, 
coherent response that fully addresses the original question.

Original Question: {original_question}

SUB-QUESTIONS AND ANSWERS:
{context}

Please provide a comprehensive summary that integrates all the information from the sub-answers to address the original question.
Your summary should be well-structured, clear, and directly answer the original question.

SUMMARY:"""

    try:
        model = lms.llm(LLM_MODEL)
        summary = model.respond(prompt)
        logger.info("Successfully generated summary response")
        
        return summary
    except Exception as e:
        logger.error(f"Error generating summary response: {e}")
        return "Error: Could not generate a summary of the responses."
