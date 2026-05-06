"""
STEP 1: Load and Parse CSV
--------------------------
Run from D:\chat_analyzer:
    python src/step1_load.py
"""

import pandas as pd
import re
import os

DATA_PATH   = os.path.join("data", "conversations.csv")
OUTPUT_PATH = os.path.join("outputs", "messages.csv")

def load_and_parse():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Cannot find {DATA_PATH}")
        print("Make sure conversations.csv is inside the 'data' folder.")
        return None

    print("Loading CSV...")
    df_raw = pd.read_csv(DATA_PATH, header=None, names=["conversation"])
    print(f"  -> {len(df_raw)} conversations loaded")

    records = []
    global_msg_index = 0

    for conv_id, row in df_raw.iterrows():
        raw_text = str(row["conversation"])
        parts = re.split(r'(User \d+:\s)', raw_text)

        i = 1
        while i < len(parts) - 1:
            sender  = parts[i].strip().rstrip(":")
            message = parts[i + 1].strip()
            if message:
                records.append({
                    "conv_id"  : conv_id,
                    "sender"   : sender,
                    "message"  : message,
                    "msg_index": global_msg_index
                })
                global_msg_index += 1
            i += 2

    df_messages = pd.DataFrame(records)
    os.makedirs("outputs", exist_ok=True)
    df_messages.to_csv(OUTPUT_PATH, index=False)

    print(f"  -> {len(df_messages)} individual messages parsed")
    print(f"  -> Saved to: {OUTPUT_PATH}")
    print("\nSample output:")
    print(df_messages.head(6).to_string(index=False))
    return df_messages

if __name__ == "__main__":
    load_and_parse()