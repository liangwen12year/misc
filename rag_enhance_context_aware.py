import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Disable unnecessary logs
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_SAVE_DIR = "/local/speech/users/wl2904"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class RAGConversationSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", llm_model_name="meta-llama/Llama-3.1-8B-Instruct", max_history=20):
        """
        Initialize the RAG conversation system with embedding model, LLM, and FAISS index.
        """
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name, cache_folder=MODEL_SAVE_DIR)
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()

        # Load LLM and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=MODEL_SAVE_DIR)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir=MODEL_SAVE_DIR).to(device)

        # Initialize conversation history and embeddings
        self.conversation_history = []
        self.embeddings = np.empty((0, self.embedding_size))  # Dynamic embedding storage

        # Initialize FAISS index (GPU if available)
        self.faiss_index = faiss.IndexFlatL2(self.embedding_size)
        if device == "cuda":
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        # Maximum number of conversation turns to keep
        self.max_history = max_history

        # Track key facts (e.g., user preferences)
        self.key_facts = {}

    def save_conversation_history(self, file_path="conversation_history.json"):
        """
        Save the conversation history to a JSON file.
        """
        with open(file_path, "w") as f:
            json.dump(self.conversation_history, f)

    def load_conversation_history(self, file_path="conversation_history.json"):
        """
        Load conversation history from a JSON file and rebuild the FAISS index.
        """
        try:
            with open(file_path, "r") as f:
                self.conversation_history = json.load(f)
                # Rebuild FAISS index with the last `max_history` turns
                self.embeddings = np.array([self.embedding_model.encode(f"User: {turn['user']}\nModel: {turn['model']}") for turn in self.conversation_history[-self.max_history:]])
                if len(self.embeddings) > 0:
                    self.faiss_index.reset()
                    self.faiss_index.add(self.embeddings)
        except FileNotFoundError:
            # Suppress the message about no existing history
            self.conversation_history = []

    def update_rag_database(self, user_query, model_response):
        """
        Update the conversation history and FAISS index with a new turn.
        """
        # Add the new turn to the conversation history
        self.conversation_history.append({"user": user_query, "model": model_response})

        # Remove the oldest turn if history exceeds max_history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
            # Rebuild embeddings and FAISS index
            self.embeddings = np.array([self.embedding_model.encode(f"User: {turn['user']}\nModel: {turn['model']}") for turn in self.conversation_history])
            self.faiss_index.reset()
            if len(self.embeddings) > 0:
                self.faiss_index.add(self.embeddings)
        else:
            # Embed the new turn and update FAISS index
            new_embedding = self.embedding_model.encode(f"User: {user_query}\nModel: {model_response}").reshape(1, -1)
            self.embeddings = np.vstack([self.embeddings, new_embedding])
            self.faiss_index.add(new_embedding)

    def retrieve_relevant_context(self, query, top_k=2):
        """
        Retrieve the most relevant context from the conversation history using FAISS.
        """
        if len(self.conversation_history) == 0:
            return []

        # Embed the query
        query_embedding = self.embedding_model.encode([query])

        # Search FAISS for the most relevant context
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        # Retrieve the relevant context
        relevant_context = [self.conversation_history[i] for i in indices[0] if i < len(self.conversation_history)]
        return relevant_context

    def generate_response(self, query, relevant_context):
        """
        Generate a concise and contextually relevant response using the LLM.
        """
        # Prepare input text for the LLM
        if not relevant_context:
            input_text = f"User: {query}\nModel:"
        else:
            # Include only the most relevant context
            context_str = "\n".join([f"User: {turn['user']}\nModel: {turn['model']}" for turn in relevant_context])
            input_text = f"Context:\n{context_str}\nUser: {query}\nModel:"

        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        # Dynamically adjust max_new_tokens to avoid exceeding the model's token limit
        max_model_length = self.llm_model.config.max_position_embeddings  # Maximum token limit for the model
        max_new_tokens = min(100, max_model_length - input_length - 1)  # Limit response length

        if max_new_tokens <= 0:
            raise ValueError("Input length exceeds the model's maximum token limit. Please reduce the context size.")

        # Generate response
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Control the number of new tokens generated
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.7,  # Adjust for more focused responses
            top_p=0.9,  # Use nucleus sampling to avoid overly verbose outputs
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the model's response
        response = response.split("Model:")[-1].strip()
        return response

    def run_conversation(self):
        """
        Run the conversation loop.
        """
        print("Starting conversation. Type 'exit' to end.")
        while True:
            user_query = input("User: ")
            if user_query.lower() == "exit":
                break

            # Retrieve relevant context (limit to last 3 turns)
            relevant_context = self.retrieve_relevant_context(user_query, top_k=3)

            # Generate and display response
            try:
                model_response = self.generate_response(user_query, relevant_context)
                print(f"Model: {model_response}")
            except ValueError as e:
                print(f"Model: Error - {e}. Please reduce the context size or start a new conversation.")

            # Update the RAG database with the new turn
            self.update_rag_database(user_query, model_response)

        # Save conversation history at the end
        self.save_conversation_history()
        print("Conversation history saved.")

# Main execution
if __name__ == "__main__":
    # Initialize the RAG conversation system
    rag_system = RAGConversationSystem(max_history=100)

    # Load existing conversation history (if any)
    rag_system.load_conversation_history()

    # Start the conversation
    rag_system.run_conversation()