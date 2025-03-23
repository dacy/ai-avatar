"""
Extractive Question Answering module using pre-trained transformer models.
This module uses models specifically trained for extractive QA to find exact answer spans in text.
"""

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger

class ExtractiveQA:
    """
    Extractive Question Answering model for finding exact answer spans in text.
    Uses a pre-trained transformer model specifically trained for extractive QA.
    """
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        """
        Initialize the QA model.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        logger.info(f"Loading extractive QA model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Extractive QA model loaded and moved to {self.device}")

    def predict(self, question: str, context: str) -> Dict[str, Any]:
        """
        Predict the answer to a question given a context.
        
        Args:
            question: The question to answer
            context: The context in which to find the answer
        
        Returns:
            A dictionary containing the predicted answer and other information
        """
        # Tokenize the input
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the predicted answer
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

        return {
            "answer": answer,
            "start": answer_start.item(),
            "end": answer_end.item()
        } 