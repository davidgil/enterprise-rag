import logging
from typing import List
from question_decomposer import QuestionDecomposer
from search_docs import process_search_query_using_rag

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DecomposedSearchProcessor:
    """
    A class that processes questions by decomposing them into simpler questions
    and then processes each decomposed question using RAG.
    """
    
    def __init__(self, top_k: int = 3):
        """
        Initialize the DecomposedSearchProcessor.
        
        Args:
            top_k: Number of results to return for each search
            use_rag: Whether to use RAG to generate answers
        """
        self.decomposer = QuestionDecomposer()
        self.top_k = top_k
    
    def process_question(self, question: str) -> dict:
        """
        Process a single question by decomposing it and processing each decomposed question.
        
        Args:
            question: The question to process
            
        Returns:
            dict: A dictionary containing the original question, decomposed questions,
                  and the responses for each decomposed question
        """
        logger.info(f"Processing question: {question}")

        result = {
            "original_question": question,
            "decomposed_questions": [],
            "responses": []
        }
        
        # Decompose the question
        decomposed_questions = self.decomposer.decompose_to_list(question)
        result["decomposed_questions"] = decomposed_questions
        
        if not decomposed_questions:
            logger.warning(f"Could not decompose question: {question}")
            response, search_results = process_search_query_using_rag(question, self.top_k)
            result["decomposed_questions"] = [question]
            result["responses"] = [{"question": question, "response": response, "search_results": search_results}]
            return result
        
        # Process each decomposed question
        logger.info(f"Decomposed questions: {decomposed_questions}")    
        for i, sub_question in enumerate(decomposed_questions):
            response, search_results = process_search_query_using_rag(sub_question, self.top_k)
            result["responses"].append({
                "question": sub_question,
                "response": response,
                "search_results": search_results
            })
        
        return result