"""
Local Generative Question Answering module using Hugging Face Transformers.
This module uses local language models to generate natural language answers from context.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os
import yaml

class LocalGenerativeQA:
    """QA extractor using local Transformers models for generating natural language answers from context."""
    
    def __init__(self, model_path: Optional[str] = None, model_name: Optional[str] = None, offline_only: bool = False):
        """
        Initialize the QA extractor.
        
        Args:
            model_path: Path to local model directory (for offline mode)
            model_name: HuggingFace model name (for online mode)
            offline_only: Whether to force offline mode regardless of model_name
        """
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Configure model loading based on device
        if self.device == "cuda":
            try:
                # Try to use 4-bit quantization if CUDA is available
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization for better performance")
            except Exception as e:
                logger.warning(f"Could not use 4-bit quantization: {e}. Falling back to FP16.")
                quantization_config = None
        else:
            logger.info("CUDA not available, using CPU with FP32")
            quantization_config = None
        
        # Set offline mode based on configuration
        self.offline_only = offline_only
        
        if self.offline_only:
            if model_path is None:
                raise ValueError("model_path must be provided when offline_only is True")
            # Offline mode - use local files
            self.model_path = model_path
            logger.info(f"Loading model from local path: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise ValueError(f"Model directory not found: {self.model_path}")
                
            # Load model and tokenizer from local directory
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",  # Automatically moves the model to GPU
                quantization_config=quantization_config,
                local_files_only=True
            )
            
        else:
            if model_name is None:
                raise ValueError("model_name must be provided when offline_only is False")
            # Online mode - download from Hugging Face
            self.model_name = model_name
            self.model_path = os.path.join("models", model_name.split("/")[-1])
            logger.info(f"Loading model from Hugging Face: {self.model_name}")
            logger.info(f"Model will be saved to: {self.model_path}")
            
            # Load model and tokenizer from Hugging Face
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" and not quantization_config else torch.float32
            )
            
        # Set default generation parameters
        self.max_new_tokens = 512
        self.temperature = 0.7
            
        logger.info(f"Initialized local generative QA with model from: {self.model_path if self.offline_only else self.model_name}")

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
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            logger.info(f"Input shape: {inputs['input_ids'].shape}")
            
            # Generate response
            logger.info(f"Generating response with model: {self.model_path if self.offline_only else self.model_name}")
            
            # Try different generation parameters if the first attempt fails
            generation_params = {
                "max_new_tokens": 256,  # Reduced for shorter responses
                "temperature": 0.3,      # Lower temperature for more focused responses
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "length_penalty": 0.8,   # Reduced to discourage long responses
                "no_repeat_ngram_size": 3,
                "num_return_sequences": 1,
                "early_stopping": True,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            logger.info(f"Generating with parameters: {generation_params}")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_params
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Log the response
            logger.info("=== Model Response ===")
            logger.info(f"Raw Response text: {response}")
            logger.info("===================")
            
            # Handle thinking process if present
            if "</think>" in response:
                answer = response.split("</think>")[-1].strip()
            else:
                answer = response.strip()

            # Try to extract answer from XML tags if present
            try:
                answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1).strip()            
            except Exception as e:
                logger.warning(f"Could not extract answer from XML tags: {e}")
                # If XML extraction fails, try to get text after the last <answer> tag
                if "<answer>" in response:
                    answer = response.split("<answer>")[-1].strip()
            
            if not answer:
                logger.warning("No answer generated")
                return "I could not find a relevant answer in the provided context."
            
            
            return answer
            
        except Exception as e:
            logger.error(f"Unexpected error in extract_answer: {str(e)}", exc_info=True)
            return f"Unexpected error: {str(e)}"
    
    def _create_prompt(self, question: str, contexts: List[str]) -> str:
        """Create a prompt for the model."""
        context_text = "\n\n".join(contexts)
        
        prompt = f"""
        
            You are a helpful AI assistant. Answer using ONLY the information inside <context>, and place your answer within answer tags.

            <rules>
            - Do NOT repeat the prompt or question
            - Do NOT include any reasoning
            - If the context does not contain the answer, respond with: "The context does not specify."
            </rules>

            <context>
            {context_text}
            </context>

            <question> 
            {question}
            </question>

            <answer>"""

        # Log the prompt details
        logger.info("=== Prompt Details ===")
        logger.info(f"Number of contexts: {len(contexts)}")
        logger.info(f"Total context length: {len(context_text)} characters")
        logger.info(f"Question: {question}")
        logger.info("=== Full Prompt ===")
        logger.info(prompt)
        logger.info("=====================")
        
        return prompt 