import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Disable unnecessary logs
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_SAVE_DIR = "/local/speech/users/wl2904"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class RAGConversationSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2",
                 llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
                 max_history=20):
        """
        Initialize the enhanced RAG system with attention-based context fusion.
        """
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name, cache_folder=MODEL_SAVE_DIR)
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()

        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=MODEL_SAVE_DIR)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir=MODEL_SAVE_DIR).to(device)

        # Initialize conversation history
        self.conversation_history = []
        self.embeddings = np.empty((0, self.embedding_size))

        # Initialize FAISS index for dynamic retrieval
        self.faiss_index = faiss.IndexFlatL2(self.embedding_size)
        if device == "cuda":
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        # Context management
        self.max_history = max_history

        # NEW: Attention layer for context fusion (learnable weights)
        self.attention = nn.Linear(self.embedding_size, 1).to(device)

    def _compute_attention_weights(self, query_embedding, context_embeddings):
        """
        Compute attention weights by combining query and context features.
        Args:
            query_embedding: Current query embedding (dim,)
            context_embeddings: Stack of context embeddings (N, dim)
        Returns:
            weights: Normalized relevance scores (N,)
        """
        # Broadcast multiplication to combine query and each context
        combined = query_embedding.unsqueeze(0) * context_embeddings  # shape (N, dim)
        # Compute raw scores using the learnable attention layer
        scores = self.attention(combined).squeeze(-1)  # shape (N,)
        # Normalize with softmax
        return F.softmax(scores, dim=0)

    def _retrieve_static_docs(self, query, top_k=2):
        """
        Placeholder for static knowledge retrieval (e.g., from Wikipedia).
        Replace with your actual retrieval method.
        """
        # Mock implementation - replace with real retrieval if needed
        return [{"content": f"Static doc about '{query[:20]}...'", "source": "Wikipedia"}]

    def retrieve_relevant_context(self, query, top_k=3):
        """
        Enhanced retrieval with attention-weighted fusion of:
        - Static knowledge (from RAG)
        - Dynamic history (from conversation)
        """
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(device)

        # Retrieve static docs
        static_docs = self._retrieve_static_docs(query)

        # Retrieve dynamic history using FAISS
        dynamic_context = []
        if len(self.conversation_history) > 0:
            _, indices = self.faiss_index.search(
                query_embedding.cpu().numpy().reshape(1, -1),
                top_k
            )
            dynamic_context = [
                {"type": "history", "content": self.conversation_history[i]}
                for i in indices[0] if i < len(self.conversation_history)
            ]

        # Combine both contexts
        all_context = (
            [{"type": "static", "content": doc} for doc in static_docs] +
            dynamic_context
        )

        if not all_context:
            return []

        # Compute embeddings for each context entry
        context_embeddings = torch.stack([
            self.embedding_model.encode(str(ctx["content"]), convert_to_tensor=True)
            for ctx in all_context
        ]).to(device)

        # Calculate attention weights with the new mechanism
        weights = self._compute_attention_weights(query_embedding, context_embeddings)

        # Detach weights before converting to NumPy to avoid gradient issues
        sorted_contexts = sorted(
            zip(all_context, weights.detach().cpu().numpy()),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return sorted_contexts

    def _build_conversation_history_block(self):
        """
        Build a chronological block of the recent conversation history.
        """
        history_lines = []
        for turn in self.conversation_history[-self.max_history:]:
            history_lines.append(f"User: {turn['user']}\nModel: {turn['model']}")
        return "\n\n".join(history_lines)

    def generate_response(self, query, weighted_contexts):
        """
        Generate response using attention-weighted context and full conversation history.
        """
        # Build the conversation history block
        conversation_history_text = self._build_conversation_history_block()

        # Build the relevant context block from weighted contexts
        relevant_context_lines = []
        for ctx, weight in weighted_contexts:
            if ctx["type"] == "static":
                # For static docs, include the source information
                relevant_context_lines.append(
                    f"[Relevance: {weight:.2f}] Knowledge: {ctx['content']['content']} (Source: {ctx['content']['source']})"
                )
            else:
                hist = ctx["content"]
                relevant_context_lines.append(
                    f"[Relevance: {weight:.2f}] Past Conversation:\nUser: {hist['user']}\nModel: {hist['model']}"
                )
        relevant_context_text = "\n".join(relevant_context_lines)

        # Build the full prompt with clear instructions
        prompt_parts = [
            "You are a context-aware assistant that uses both conversation history and relevant background knowledge to generate accurate and personalized responses.",
            "Recent Conversation History:\n" + (conversation_history_text if conversation_history_text else "None"),
            "Relevant Context Retrieved:\n" + (relevant_context_text if relevant_context_text else "None"),
            f"User: {query}",
            "Model:"
        ]
        full_prompt = "\n\n".join(prompt_parts)

        # Tokenize prompt and check length
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        max_new_tokens = min(150, self.llm_model.config.max_position_embeddings - input_length - 1)
        if max_new_tokens <= 0:
            raise ValueError("Input too long. Reduce context size.")

        # Generate response from the LLM
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response.strip()

    def update_rag_database(self, user_query, model_response):
        """Update conversation history and FAISS index."""
        self.conversation_history.append({"user": user_query, "model": model_response})

        # Update FAISS index either incrementally or rebuild if history exceeds max_history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
            self.embeddings = np.array([
                self.embedding_model.encode(f"User: {turn['user']}\nModel: {turn['model']}")
                for turn in self.conversation_history
            ])
            self.faiss_index.reset()
            if len(self.embeddings) > 0:
                self.faiss_index.add(self.embeddings)
        else:
            new_embedding = self.embedding_model.encode(
                f"User: {user_query}\nModel: {model_response}"
            ).reshape(1, -1)
            self.embeddings = np.vstack([self.embeddings, new_embedding])
            self.faiss_index.add(new_embedding)

    def run_conversation(self):
        """Main conversation loop."""
        print("RAG System Ready. Type 'exit' to quit.")
        while True:
            user_query = input("User: ")
            if user_query.lower() == "exit":
                break

            # Retrieve context with attention weighting
            context = self.retrieve_relevant_context(user_query)

            try:
                # Generate and print response
                response = self.generate_response(user_query, context)
                print(f"Model: {response}")

                # Update conversation database
                self.update_rag_database(user_query, response)
            except ValueError as e:
                print(f"Error: {e}")

        # Save conversation history upon exit
        self.save_conversation_history()

    def save_conversation_history(self, path="conversation_history.json"):
        with open(path, "w") as f:
            json.dump(self.conversation_history, f)

    def load_conversation_history(self, path="conversation_history.json"):
        try:
            with open(path, "r") as f:
                self.conversation_history = json.load(f)
                self.embeddings = np.array([
                    self.embedding_model.encode(f"User: {turn['user']}\nModel: {turn['model']}")
                    for turn in self.conversation_history[-self.max_history:]
                ])
                if len(self.embeddings) > 0:
                    self.faiss_index.reset()
                    self.faiss_index.add(self.embeddings)
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    rag_system = RAGConversationSystem(max_history=15)
    rag_system.load_conversation_history()
    rag_system.run_conversation()