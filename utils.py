import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name="microsoft/DialoGPT-medium"):
    try:
        logger.info(f"Loading model and tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        raise

def load_qa_dataset(dataset_name="squad", split="train"):
    try:
        logger.info(f"Loading dataset: {dataset_name} ({split})")
        dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def preprocess_input(text, tokenizer, max_length=128):
    return tokenizer.encode_plus(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

def generate_response(model, tokenizer, input_ids, attention_mask=None, max_length=100):
    try:
        model.config.pad_token_id = model.config.pad_token_id or model.config.eos_token_id
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=model.config.pad_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return "Sorry, I couldn't generate a response."
