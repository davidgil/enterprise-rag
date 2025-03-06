import os
from pprint import pprint
from pydantic import BaseModel
from typing import Optional, List
import lmstudio as lms
from dotenv import load_dotenv

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")

class Question(BaseModel):
    """Model to represent an individual question."""
    question: str

class Questions(BaseModel):
    """Model to represent a list of questions."""
    questions: List[Question]

class QuestionDecomposer:
    """
    Class for decomposing complex questions into simpler ones.
    
    This class uses a language model to break down complex questions
    into multiple simpler questions that can be answered independently.
    """
    
    def __init__(self):
        self.model = lms.llm(LLM_MODEL)
        self.splitter_prompt = """
You are a helpful assistant that prepares queries that will be sent to a search component.
Sometimes, these queries are very complex.
Your job is to simplify complex queries into multiple queries that can be answered
in isolation to eachother.

If the query is simple, then keep it as it is.
Examples
1. Query: Did Microsoft or Google make more money last year?
   Decomposed Questions: How much profit did Microsoft make last year?, How much profit did Google make last year?
2. Query: What is the capital of France?
   Decomposed Questions: What is the capital of France?
3. Query: {question}
   Decomposed Questions:
"""
    
    def decompose(self, question: str) -> dict:
        """
        Decompose a complex question into simpler questions.
        
        Args:
            question (str): The complex question to decompose.
            
        Returns:
            dict: A dictionary containing the list of decomposed questions.
        """
        prompt = self.splitter_prompt.format(question=question)
        result = self.model.respond(prompt, response_format=Questions)
        
        return result.parsed
    
    def decompose_to_list(self, question: str) -> List[str]:
        """
        Decompose a complex question and return a list of strings with the questions.
        
        Args:
            question (str): The complex question to decompose.
            
        Returns:
            List[str]: List of decomposed questions as strings.
        """
        questions_dict = self.decompose(question)
        if isinstance(questions_dict, dict) and 'questions' in questions_dict:
            return [q['question'] for q in questions_dict['questions']]

        return []