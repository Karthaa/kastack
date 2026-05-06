"""
STEP 4: 100-Message Checkpoint Summaries
-----------------------------------------
We slice ALL messages into chunks of 100 (chunk 0 = msgs 0-99,
chunk 1 = msgs 100-199, etc.) and create a summary for each chunk.

Each summary includes:
  - Which messages are in this chunk (start/end index)
  - Which topics were discussed
  - A short text summary of the key messages

This lets the RAG system later say:
  "The user discussed food around messages 400-499"

Run from D:\chat_analyzer:
    python src/step4_checkpoints.py
"""

import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_PATH  = os.path.join("outputs", "topics.csv")
OUTPUT_PATH = os.path.join("outputs", "checkpoint_summaries.csv")

CHUNK_SIZE        = 100   # messages per checkpoint
SUMMARY_SENTENCES = 5     # sentences to pick per checkpoint


def summarize_chunk(messages: list, n: int = SUMMARY_SENTENCES) -> str:
    """Pick top-N most important messages from a chunk using TF-IDF."""
    if len(messages) == 0:
        return ""
    if len(messages) <= n:
        return " | ".join(messages)

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=200,
            min_df=1          # min_df=1 because chunks are small (100 msgs)
        )
        tfidf_matrix = vectorizer.fit_transform(messages)
        scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        # Get top-N indices, sorted chronologically
        top_indices = sorted(scores.argsort()[-n:][::-1])
        return " | ".join([messages[i] for i in top_indices])

    except Exception:
        return " | ".join(messages[:n])


def run_checkpoints():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Run step2_topics.py first.")
        return

    print("Loading messages...")
    df = pd.read_csv(INPUT_PATH)
    total = len(df)
    print(f"  -> {total} messages loaded")

    # Sort by msg_index to ensure chronological order
    df = df.sort_values("msg_index").reset_index(drop=True)

    # Calculate how many chunks we'll have
    num_chunks = (total // CHUNK_SIZE) + (1 if total % CHUNK_SIZE else 0)
    print(f"  -> Creating {num_chunks} checkpoints (every {CHUNK_SIZE} messages)")

    records = []

    for chunk_num in range(num_chunks):
        start_idx = chunk_num * CHUNK_SIZE
        end_idx   = min(start_idx + CHUNK_SIZE, total)

        # Slice the chunk
        chunk_df  = df.iloc[start_idx:end_idx]
        messages  = chunk_df["message"].dropna().tolist()

        # Which topics appeared in this chunk?
        topic_counts = chunk_df["topic"].value_counts()
        top_topics   = topic_counts.head(3).index.tolist()  # top 3 topics
        topics_str   = ", ".join(top_topics)

        # Which conversations are in this chunk?
        conv_ids = chunk_df["conv_id"].unique().tolist()

        # Summarize
        summary = summarize_chunk(messages)

        records.append({
            "checkpoint"   : chunk_num,
            "msg_start"    : start_idx,
            "msg_end"      : end_idx - 1,
            "message_count": len(messages),
            "top_topics"   : topics_str,
            "conv_ids"     : str(conv_ids[:5]),  # first 5 conv IDs for reference
            "summary"      : summary
        })

        # Print progress every 200 checkpoints
        if chunk_num % 200 == 0:
            print(f"  ... checkpoint {chunk_num}/{num_chunks} done")

    df_checkpoints = pd.DataFrame(records)
    os.makedirs("outputs", exist_ok=True)
    df_checkpoints.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  -> {len(df_checkpoints)} checkpoints saved to: {OUTPUT_PATH}")

    # Show a few examples
    print("\n--- SAMPLE CHECKPOINTS ---")
    for _, row in df_checkpoints.head(3).iterrows():
        print(f"\nCheckpoint #{row['checkpoint']}  "
              f"(msgs {row['msg_start']}–{row['msg_end']})")
        print(f"  Topics  : {row['top_topics']}")
        print(f"  Summary :")
        for s in row["summary"].split(" | "):
            print(f"    • {s[:90]}")

    return df_checkpoints


if __name__ == "__main__":
    run_checkpoints()