import os
import argparse
import logging
from utils import (
    load_model_and_tokenizer,
    load_qa_dataset,
    preprocess_input,
    generate_response
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium", dataset_name=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.conversation_history = []

        self._load_resources()
    
    def _load_resources(self):
        try:
            self.model, self.tokenizer = load_model_and_tokenizer(self.model_name)
            if self.dataset_name:
                self.dataset = load_qa_dataset(self.dataset_name)
                logger.info(f"Loaded dataset: {self.dataset_name}")
            logger.info("Resources loaded successfully")
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
            raise
    
    def process_input(self, user_input):
        self.conversation_history.append(user_input)
        context = " ".join(self.conversation_history[-5:])
        inputs = preprocess_input(context, self.tokenizer)
        response = generate_response(
            self.model,
            self.tokenizer,
            inputs["input_ids"],
            inputs["attention_mask"]
        )
        self.conversation_history.append(response)
        return response
    
    def reset_conversation(self):
        self.conversation_history = []
        logger.info("Conversation history reset")

def main():
    parser = argparse.ArgumentParser(description="Run a chatbot with a pretrained model")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-medium",
                        help="Pretrained model to use")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to use (optional)")
    args = parser.parse_args()

    try:
        chatbot = Chatbot(model_name=args.model, dataset_name=args.dataset)
        logger.info(f"Chatbot initialized with model: {args.model}")

        print("\n🤖 Welcome to the Chatbot!")
        print("Type 'exit' to quit or 'reset' to clear conversation.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            elif user_input.lower() == "reset":
                chatbot.reset_conversation()
                print("Conversation reset.")
                continue

            response = chatbot.process_input(user_input)
            print(f"Chatbot: {response}")

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Error running chatbot: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
