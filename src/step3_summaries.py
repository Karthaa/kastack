"""
STEP 3: Topic-Based Summaries
------------------------------
For each topic, we collect all messages in that topic and pick the
top N most "important" sentences using TF-IDF scoring.

TF-IDF means: words that appear often in THIS topic but rarely in
other topics get a high score. Sentences with high-scoring words
are chosen as the summary.

Run from D:\chat_analyzer:
    python src/step3_summaries.py
"""

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

INPUT_PATH  = os.path.join("outputs", "topics.csv")
OUTPUT_PATH = os.path.join("outputs", "topic_summaries.csv")

# How many sentences to include in each topic summary
SUMMARY_SENTENCES = 8


def summarize_topic(messages: list, n: int = SUMMARY_SENTENCES) -> str:
    """
    Pick the top-N most representative messages from a list using TF-IDF.
    Steps:
      1. Build a TF-IDF matrix from all messages in this topic
      2. Score each message by summing its TF-IDF values
      3. Return the top-N highest-scoring messages joined together
    """
    # Need at least 2 messages to do TF-IDF
    if len(messages) <= 2:
        return " | ".join(messages)

    # Limit to 2000 messages max to keep it fast
    sample = messages[:2000] if len(messages) > 2000 else messages

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",   # ignore common words like "the", "is"
            max_features=500,       # only consider top 500 words
            min_df=2                # word must appear in at least 2 messages
        )
        tfidf_matrix = vectorizer.fit_transform(sample)

        # Score each message = sum of its TF-IDF values across all words
        scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        # Get indices of top-N scoring messages
        top_indices = scores.argsort()[-n:][::-1]

        # Sort by original order (chronological)
        top_indices = sorted(top_indices)

        top_messages = [sample[i] for i in top_indices]
        return " | ".join(top_messages)

    except Exception as e:
        # Fallback: just take first N messages
        return " | ".join(messages[:n])


def run_summarization():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Run step2_topics.py first.")
        return

    print("Loading topics...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  -> {len(df)} messages loaded")

    topics = df["topic"].unique()
    print(f"  -> {len(topics)} topics found: {list(topics)}")

    results = []

    for topic in topics:
        # Get all messages for this topic
        topic_df = df[df["topic"] == topic]
        messages = topic_df["message"].dropna().tolist()

        print(f"  Summarizing '{topic}' ({len(messages)} messages)...")

        summary = summarize_topic(messages)

        results.append({
            "topic"        : topic,
            "message_count": len(messages),
            "summary"      : summary
        })

    df_summaries = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    df_summaries.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  -> Saved to: {OUTPUT_PATH}")
    print("\n--- SUMMARIES ---")
    for _, row in df_summaries.iterrows():
        print(f"\nTOPIC: {row['topic'].upper()} ({row['message_count']} messages)")
        # Print each sentence on its own line for readability
        sentences = row["summary"].split(" | ")
        for s in sentences:
            print(f"  • {s[:100]}")

    return df_summaries


if __name__ == "__main__":
    run_summarization()