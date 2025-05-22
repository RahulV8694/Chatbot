import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name="microsoft/DialoGPT-medium"):

    logger.info(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        raise

def load_qa_dataset(dataset_name="squad", split="train"):

    logger.info(f"Loading dataset: {dataset_name}, split: {split}")
    try:
        dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Dataset loaded successfully with {len(dataset)} examples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def preprocess_input(text, tokenizer, max_length=128):
    """
    Preprocess input text for the model.
    
    Args:
        text (str): Input text to preprocess
        tokenizer: Tokenizer to use for preprocessing
        max_length (int): Maximum length of the input
        
    Returns:
        dict: Preprocessed input
    """
    return tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

def generate_response(model, tokenizer, input_ids, attention_mask=None, max_length=100):
    try:
        # Set the pad token ID to the EOS token ID if not set
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
            
        # Generate response
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask if attention_mask is not None else None,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=model.config.pad_token_id
        )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm sorry, I couldn't generate a response."
