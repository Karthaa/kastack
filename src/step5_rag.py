"""
STEP 5: Basic RAG Retrieval
----------------------------
RAG = Retrieval Augmented Generation.

When a question comes in:
  1. Convert the question into a TF-IDF vector
  2. Compare it against all our stored summaries using cosine similarity
  3. Return the top-K most similar summaries as "context"

We search TWO sources:
  - topic_summaries.csv    (9 rows, one per topic)
  - checkpoint_summaries.csv (1916 rows, one per 100 messages)

The chatbot (Step 7) will use this retrieved context to answer questions.

Run from D:\chat_analyzer to test:
    python src/step5_rag.py
"""

import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOPIC_PATH      = os.path.join("outputs", "topic_summaries.csv")
CHECKPOINT_PATH = os.path.join("outputs", "checkpoint_summaries.csv")


class RAGRetriever:
    """
    Loads all summaries once, builds a TF-IDF index,
    and answers queries by returning the most similar summaries.
    """

    def __init__(self):
        self.topic_df      = None
        self.checkpoint_df = None
        self.all_texts     = []   # all summary texts combined
        self.all_meta      = []   # metadata for each summary
        self.vectorizer    = None
        self.tfidf_matrix  = None
        self.is_loaded     = False

    def load(self):
        """Load summaries and build TF-IDF index."""
        if not os.path.exists(TOPIC_PATH):
            print(f"ERROR: {TOPIC_PATH} not found. Run step3_summaries.py first.")
            return False
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"ERROR: {CHECKPOINT_PATH} not found. Run step4_checkpoints.py first.")
            return False

        print("Loading RAG index...")

        # ── Load topic summaries ─────────────────────────────────────────────
        self.topic_df = pd.read_csv(TOPIC_PATH)
        for _, row in self.topic_df.iterrows():
            self.all_texts.append(str(row["summary"]))
            self.all_meta.append({
                "source" : "topic",
                "label"  : f"Topic: {row['topic']} ({row['message_count']} messages)",
                "content": str(row["summary"])
            })

        # ── Load checkpoint summaries ────────────────────────────────────────
        self.checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
        for _, row in self.checkpoint_df.iterrows():
            self.all_texts.append(str(row["summary"]))
            self.all_meta.append({
                "source" : "checkpoint",
                "label"  : (f"Checkpoint #{row['checkpoint']} "
                            f"(msgs {row['msg_start']}-{row['msg_end']}, "
                            f"topics: {row['top_topics']})"),
                "content": str(row["summary"])
            })

        # ── Build TF-IDF matrix over all summaries ───────────────────────────
        print(f"  -> Building TF-IDF index over {len(self.all_texts)} summaries...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=3000,
            ngram_range=(1, 2)   # use single words AND two-word phrases
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.all_texts)
        self.is_loaded = True
        print(f"  -> RAG index ready!")
        return True

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Find the top-K summaries most similar to the query.
        Returns a list of dicts with 'label', 'source', 'content', 'score'.
        """
        if not self.is_loaded:
            print("ERROR: Call .load() first.")
            return []

        # Convert query to TF-IDF vector
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity between query and all summaries
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-K indices sorted by similarity score (highest first)
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0:   # only include results with some similarity
                result = dict(self.all_meta[idx])
                result["score"] = round(float(score), 4)
                results.append(result)

        return results

    def retrieve_as_text(self, query: str, top_k: int = 5) -> str:
        """
        Convenience method: returns retrieved context as a single string.
        Used by the chatbot to build its prompt.
        """
        results = self.retrieve(query, top_k)
        if not results:
            return "No relevant context found."

        parts = []
        for r in results:
            parts.append(f"[{r['label']}]\n{r['content'][:300]}")

        return "\n\n".join(parts)


# ── Singleton: load once, reuse everywhere ───────────────────────────────────
_retriever = None

def get_retriever() -> RAGRetriever:
    """Return a loaded RAGRetriever (loads only once per session)."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
        _retriever.load()
    return _retriever


# ── Test the RAG system ──────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = get_retriever()

    test_queries = [
        "What kind of food do people like?",
        "Do people talk about their pets?",
        "What hobbies come up most often?",
        "Do people discuss work or jobs?",
        "What travel destinations are mentioned?",
    ]

    for query in test_queries:
        print(f"\nQUERY: {query}")
        print("-" * 60)
        results = retriever.retrieve(query, top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['label']}")
            # Show first 120 chars of matched content
            preview = r["content"][:120].replace("\n", " ")
            print(f"           {preview}...")