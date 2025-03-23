"""
Generative Question Answering module using Ollama.
This module uses large language models to generate natural language answers from context.
"""

import logging
import requests
from typing import List, Dict, Any
from loguru import logger

class GenerativeQA:
    """QA extractor using Ollama for generating natural language answers from context."""
    
    def __init__(self, model_name: str = "llama2", api_url: str = "http://localhost:11434"):
        """
        Initialize the QA extractor.
        
        Args:
            model_name: Name of the Ollama model to use
            api_url: URL of the Ollama API server
        """
        self.model_name = model_name
        self.api_url = api_url.rstrip('/')
        logger.info(f"Initialized generative QA with model: {model_name}")

    def extract_answers_from_multiple_contexts(self, question: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract answers for a question from multiple contexts.
        
        Args:
            question: The question to answer
            contexts: List of context dictionaries containing text and metadata
            
        Returns:
            List of answer dictionaries containing:
            - answer: The generated answer
            - source: Source document information
            - confidence: Confidence score
        """
        try:
            # Extract text from contexts
            context_texts = [ctx.get('text', '') for ctx in contexts]
            
            # Generate answer using the combined contexts
            answer = self._extract_answer(question, context_texts)
            
            # Create answer dictionary with source information
            answer_dict = {
                'answer': answer,
                'sources': [{
                    'title': ctx.get('title', 'Unknown'),
                    'url': ctx.get('url', ''),
                    'text': ctx.get('text', '')[:200] + '...'  # Truncate for display
                } for ctx in contexts]
            }
            
            return [answer_dict]
            
        except Exception as e:
            logger.error(f"Error extracting answers: {str(e)}", exc_info=True)
            return [{
                'answer': f"Error generating answer: {str(e)}",
                'sources': []
            }]
    
    def _extract_answer(self, question: str, contexts: List[str]) -> str:
        """
        Extract an answer for a question from the given contexts.
        
        Args:
            question: The question to answer
            contexts: List of text contexts to use for answering
            
        Returns:
            Generated answer
        """
        try:
            # Combine contexts with the question into a prompt
            prompt = self._create_prompt(question, contexts)
            
            # Call Ollama API
            logger.info(f"Calling Ollama API with model: {self.model_name}")
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            
            # Log the response
            response_data = response.json()
            logger.info("=== API Response ===")
            logger.info(f"Model: {response_data.get('model', 'unknown')}")
            logger.info(f"Raw Response text: {response_data.get('response', '')}")
            logger.info("===================")
            
            # Extract answer from response
            answer = response_data.get('response', '').strip()
            
            if not answer:
                logger.warning("No answer received from Ollama API")
                return "I could not find a relevant answer in the provided context."
            
            # Clean up the answer
            # Remove thinking process if present
            if "<think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            
            # Remove any instances of "Answer:" or "answer:"
            answer = answer.replace("Answer:", "").replace("answer:", "").strip()
            
            # Remove any leading/trailing quotes
            answer = answer.strip('"\'')
            
            # Remove any markdown formatting
            answer = answer.replace('**', '').replace('*', '')
            
            logger.info("=== Final Answer ===")
            logger.info(answer)
            logger.info("===================")
            
            return answer
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            return f"Error calling Ollama API: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in extract_answer: {str(e)}", exc_info=True)
            return f"Unexpected error: {str(e)}"
    
    def _create_prompt(self, question: str, contexts: List[str]) -> str:
        """Create a prompt for the model."""
        context_text = "\n\n".join(contexts)
        
        prompt = f"""You are a helpful AI assistant that provides accurate answers based on the provided context. Your task is to answer the question using ONLY the information from the given context.

Instructions:
1. Read the context carefully and identify relevant information
2. Provide a clear, concise answer that directly addresses the question
3. If the context contains multiple relevant pieces of information, combine them logically
4. If you cannot find a relevant answer in the context, explicitly state that
5. Do not make assumptions or add information beyond what is in the context
6. If the context is ambiguous or incomplete, acknowledge this in your response
7. Provide your answer in a single, coherent response without repeating yourself
8. Do not include the word "Answer:" anywhere in your response

Context:
{context_text}

Question: {question}

Provide your response:"""

        # Log the prompt details
        logger.info("=== Prompt Details ===")
        logger.info(f"Number of contexts: {len(contexts)}")
        logger.info(f"Total context length: {len(context_text)} characters")
        logger.info(f"Question: {question}")
        logger.info("=== Full Prompt ===")
        logger.info(prompt)
        logger.info("=====================")
        
        return prompt 