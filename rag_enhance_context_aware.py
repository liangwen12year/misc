import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_SAVE_DIR = "/local/speech/users/wl2904"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Step 1: Initialize components
class RAGConversationSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", llm_model_name="meta-llama/Llama-3.1-8B-Instruct", max_history=20):
        # Load embedding model and save to the custom directory
        self.embedding_model = SentenceTransformer(embedding_model_name, cache_folder=MODEL_SAVE_DIR)

        # Load LLM model and tokenizer and save to the custom directory
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=MODEL_SAVE_DIR)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir=MODEL_SAVE_DIR).to(device)  # Move model to GPU

        # Initialize conversation history and FAISS index
        self.conversation_history = []
        self.embeddings = np.array([]).reshape(0, 384)  # Adjust based on embedding size

        # Use FAISS GPU index if available
        if device == "cuda":
            self.faiss_index = faiss.IndexFlatL2(384)  # Use CPU index as fallback
            res = faiss.StandardGpuResources()  # Create GPU resources
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)  # Move index to GPU
        else:
            self.faiss_index = faiss.IndexFlatL2(384)  # Use CPU index if GPU is not available

        # Maximum number of conversation turns to keep
        self.max_history = max_history

    # Step 2: Save and load conversation history
    def save_conversation_history(self, file_path="conversation_history.json"):
        with open(file_path, "w") as f:
            json.dump(self.conversation_history, f)

    def load_conversation_history(self, file_path="conversation_history.json"):
        try:
            with open(file_path, "r") as f:
                self.conversation_history = json.load(f)
                # Rebuild FAISS index from loaded history (only last `max_history` turns)
                self.embeddings = np.array([self.embedding_model.encode(f"User: {turn['user']}\nModel: {turn['model']}") for turn in self.conversation_history[-self.max_history:]])
                if len(self.embeddings) > 0:
                    self.faiss_index.add(self.embeddings)
        except FileNotFoundError:
            self.conversation_history = []

    # Step 3: Update RAG database with new conversation turn
    def update_rag_database(self, user_query, model_response):
        # Add the new turn to the conversation history
        self.conversation_history.append({"user": user_query, "model": model_response})

        # If history exceeds max_history, remove the oldest turn
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

    # Step 4: Retrieve relevant context using FAISS
    def retrieve_relevant_context(self, query, top_k=2):
        # If there is no conversation history, return an empty list
        if len(self.conversation_history) == 0:
            return []

        # Embed the query
        query_embedding = self.embedding_model.encode([query])

        # Search FAISS for the most relevant context
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        # Retrieve the relevant context
        relevant_context = [self.conversation_history[i] for i in indices[0] if i < len(self.conversation_history)]
        return relevant_context

    # Step 5: Generate a response using the LLM
    def generate_response(self, query, relevant_context):
        # If there is no relevant context, use only the query
        if not relevant_context:
            input_text = f"User: {query}\nModel:"
        else:
            # Combine the relevant context into a single string
            context_str = "\n".join([f"User: {turn['user']}\nModel: {turn['model']}" for turn in relevant_context])
            # Prepare the input for the LLM
            input_text = f"Context:\n{context_str}\nUser: {query}\nModel:"

        # Generate a response
        inputs = self.tokenizer(input_text, return_tensors="pt").to(device)  # Move inputs to GPU
        outputs = self.llm_model.generate(
            **inputs,
            max_length=200,
            pad_token_id=self.tokenizer.eos_token_id,  # Explicitly set pad_token_id to suppress the warning
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the model's response
        response = response.split("Model:")[-1].strip()
        return response

    # Step 6: Run the conversation
    def run_conversation(self):
        print("Starting conversation. Type 'exit' to end.")
        while True:
            # Get user input
            user_query = input("User: ")
            if user_query.lower() == "exit":
                break

            # Retrieve relevant context
            relevant_context = self.retrieve_relevant_context(user_query)

            # Generate a response
            model_response = self.generate_response(user_query, relevant_context)
            print(f"Model: {model_response}")

            # Update the RAG database with the new turn
            self.update_rag_database(user_query, model_response)

        # Save the conversation history at the end
        self.save_conversation_history()
        print("Conversation history saved.")

# Step 7: Run the system
if __name__ == "__main__":
    # Initialize the RAG conversation system
    rag_system = RAGConversationSystem(max_history=20)  # Keep only the last 20 turns

    # Load existing conversation history (if any)
    rag_system.load_conversation_history()

    # Start the conversation
    rag_system.run_conversation()
