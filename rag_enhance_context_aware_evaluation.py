import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import argparse
import os

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

        # Initialize conversation history and corresponding embeddings
        self.conversation_history = []
        self.embeddings = np.empty((0, self.embedding_size))

        # Initialize FAISS index for dynamic retrieval
        self.faiss_index = faiss.IndexFlatL2(self.embedding_size)
        if device == "cuda":
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        # Context management
        self.max_history = max_history

        # Attention layer for context fusion (learnable weights)
        self.attention = nn.Linear(self.embedding_size, 1).to(device)

    def _compute_attention_weights(self, query_embedding, context_embeddings):
        """
        Compute attention weights by combining query and context features.
        """
        # Element-wise multiplication between query and each context
        combined = query_embedding.unsqueeze(0) * context_embeddings  # shape: (N, dim)
        scores = self.attention(combined).squeeze(-1)  # shape: (N,)
        return F.softmax(scores, dim=0)

    def _retrieve_static_docs(self, query, top_k=2):
        """
        Placeholder for static knowledge retrieval (e.g., from Wikipedia).
        Replace with an actual retrieval method as needed.
        """
        return [{"content": f"Static doc about '{query[:20]}...'", "source": "Wikipedia"}]

    def retrieve_relevant_context(self, query, top_k=3):
        """
        Enhanced retrieval with attention-weighted fusion of:
        - Static knowledge
        - Dynamic conversation history
        """
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(device)

        # Retrieve static documents
        static_docs = self._retrieve_static_docs(query)

        # Retrieve dynamic context using FAISS search on conversation history
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

        # Combine both static and dynamic contexts
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

        # Calculate attention weights
        weights = self._compute_attention_weights(query_embedding, context_embeddings)

        # Sort contexts by relevance weight and select the top_k items
        sorted_contexts = sorted(
            zip(all_context, weights.detach().cpu().numpy()),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return sorted_contexts

    def generate_response(self, query, weighted_contexts):
        """
        Generate a response using the query and attention-weighted context.
        """
        # Build the conversation history block (if any)
        history_lines = []
        for turn in self.conversation_history[-self.max_history:]:
            history_lines.append(f"User: {turn['user']}\nModel: {turn['model']}")
        history_text = "\n\n".join(history_lines) if history_lines else "None"

        # Build the relevant context block
        context_lines = []
        for ctx, weight in weighted_contexts:
            if ctx["type"] == "static":
                context_lines.append(
                    f"[Relevance: {weight:.2f}] Knowledge: {ctx['content']['content']} (Source: {ctx['content']['source']})"
                )
            else:
                hist = ctx["content"]
                context_lines.append(
                    f"[Relevance: {weight:.2f}] Past Conversation:\nUser: {hist['user']}\nModel: {hist['model']}"
                )
        context_text = "\n".join(context_lines) if context_lines else "None"

        # Build the full prompt
        prompt_parts = [
            "You are a context-aware assistant that uses both conversation history and relevant background knowledge to generate accurate responses.",
            "Recent Conversation History:\n" + history_text,
            "Relevant Context Retrieved:\n" + context_text,
            f"User: {query}",
            "Model:"
        ]
        full_prompt = "\n\n".join(prompt_parts)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        max_new_tokens = min(150, self.llm_model.config.max_position_embeddings - input_length - 1)
        if max_new_tokens <= 0:
            raise ValueError("Input too long. Reduce context size.")

        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            do_sample=True
        )

        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response.strip()

    def update_rag_database(self, user_query, model_response):
        """
        Update conversation history and FAISS index with the latest turn.
        """
        self.conversation_history.append({"user": user_query, "model": model_response})

        # If history exceeds max_history, trim the oldest turn and rebuild the FAISS index
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
            if self.embeddings.size == 0:
                self.embeddings = new_embedding
            else:
                self.embeddings = np.vstack([self.embeddings, new_embedding])
            self.faiss_index.add(new_embedding)

def evaluate_on_topiocqa(dataset_path, output_path):
    """
    Load the TopiOCQA dataset, evaluate the RAG system on each conversation,
    and save the evaluation results to an output file.
    """
    # Load the dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset file '{dataset_path}' not found.")
        return

    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading dataset file: {e}")
        return

    # Group conversation turns by Conversation_no and sort by Turn_no
    conversations = {}
    for turn in data:
        conv_no = turn["Conversation_no"]
        conversations.setdefault(conv_no, []).append(turn)
    for conv_no in conversations:
        conversations[conv_no].sort(key=lambda x: x["Turn_no"])

    # Initialize the RAG system (using a smaller max_history if desired)
    rag_system = RAGConversationSystem(max_history=15)
    results = []

    # Evaluate each conversation
    for conv_no, turns in conversations.items():
        print(f"\n--- Conversation {conv_no} ---")
        # Reset conversation history and FAISS index for each conversation
        rag_system.conversation_history = []
        rag_system.embeddings = np.empty((0, rag_system.embedding_size))
        rag_system.faiss_index.reset()

        for turn in turns:
            query = turn["Question"]
            ground_truth = turn["Answer"]

            # Retrieve relevant context using attention weighting
            weighted_context = rag_system.retrieve_relevant_context(query)

            try:
                generated_response = rag_system.generate_response(query, weighted_context)
            except ValueError as e:
                generated_response = f"Error: {e}"

            print(f"User: {query}")
            print(f"Generated: {generated_response}")
            print(f"Ground Truth: {ground_truth}\n")

            results.append({
                "Conversation_no": conv_no,
                "Turn_no": turn["Turn_no"],
                "Question": query,
                "Generated": generated_response,
                "Ground_Truth": ground_truth
            })

            # For evaluation, update conversation history with the ground-truth answer
            rag_system.update_rag_database(query, ground_truth)

    # Save evaluation results
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation complete. Results saved to {output_path}")
    except IOError as e:
        print(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the RAGConversationSystem on a TopiOCQA dataset."
    )
    parser.add_argument("--dataset_path", type=str, default="topiocqa_dev.json",
                        help="Path to the TopiOCQA JSON dataset")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json",
                        help="Path where the evaluation results will be saved")
    args = parser.parse_args()

    evaluate_on_topiocqa(args.dataset_path, args.output_path)

if __name__ == "__main__":
    main()

