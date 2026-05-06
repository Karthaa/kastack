"""
STEP 2: Topic Detection
-----------------------
We assign a topic label to each message using keyword matching.
Each message is checked against topic keyword lists.
The topic with the most keyword hits wins. Ties go to "general".

Run from D:\chat_analyzer:
    python src/step2_topics.py
"""

import pandas as pd
import os
import re

INPUT_PATH  = os.path.join("outputs", "messages.csv")
OUTPUT_PATH = os.path.join("outputs", "topics.csv")

# ── Topic keyword dictionary ──────────────────────────────────────────────────
# Add or remove topics/keywords to suit your data
TOPIC_KEYWORDS = {
    "food": [
        "eat", "food", "cook", "recipe", "restaurant", "meal", "dinner",
        "lunch", "breakfast", "bake", "grill", "kitchen", "dish", "cuisine",
        "coffee", "drink", "hungry", "taste", "flavor", "chef", "burger",
        "pizza", "salad", "chicken", "steak", "vegetarian", "vegan"
    ],
    "travel": [
        "travel", "trip", "city", "country", "visit", "moving", "live",
        "place", "town", "state", "vacation", "flight", "hotel", "tour",
        "explore", "abroad", "destination", "portland", "midwest", "beach",
        "mountain", "road", "map", "passport", "abroad"
    ],
    "family": [
        "family", "mom", "dad", "parent", "sister", "brother", "child",
        "kids", "son", "daughter", "husband", "wife", "married", "wedding",
        "baby", "grandma", "grandpa", "relative", "aunt", "uncle", "nephew",
        "niece", "sibling", "home", "house"
    ],
    "work": [
        "work", "job", "career", "office", "boss", "employee", "company",
        "business", "meeting", "project", "salary", "hire", "profession",
        "manager", "colleague", "team", "client", "interview", "resume",
        "degree", "college", "study", "student", "school", "class", "course"
    ],
    "hobbies": [
        "hobby", "read", "book", "music", "sport", "game", "play", "art",
        "draw", "paint", "write", "sing", "dance", "swim", "run", "hike",
        "craft", "sew", "knit", "garden", "photography", "film", "movie",
        "tv", "show", "series", "festival", "concert", "exercise", "yoga",
        "gym", "soccer", "basketball", "tennis", "golf", "chess"
    ],
    "pets": [
        "pet", "dog", "cat", "animal", "puppy", "kitten", "bird", "fish",
        "hamster", "rabbit", "vet", "paw", "fur", "breed", "walk the dog",
        "companionship", "adopt", "shelter"
    ],
    "health": [
        "health", "sick", "doctor", "hospital", "medicine", "diet", "sleep",
        "mental", "anxiety", "stress", "therapy", "exercise", "fit", "weight",
        "vitamin", "pain", "symptom", "wellness", "recover", "injury"
    ],
    "technology": [
        "tech", "computer", "phone", "app", "software", "internet", "online",
        "website", "coding", "program", "ai", "robot", "device", "gadget",
        "laptop", "social media", "instagram", "facebook", "twitter", "youtube"
    ],
}

def detect_topic(message: str) -> str:
    """Return the best-matching topic for a message."""
    text = message.lower()
    # Remove punctuation for cleaner matching
    text = re.sub(r"[^\w\s]", " ", text)
    words = set(text.split())

    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        # Count how many keywords appear in this message
        score = sum(1 for kw in keywords if kw in words or kw in text)
        if score > 0:
            scores[topic] = score

    if not scores:
        return "general"

    # Return the topic with the highest score
    return max(scores, key=scores.get)


def run_topic_detection():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Run step1_load.py first.")
        return

    print("Loading messages...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  -> {len(df)} messages loaded")

    print("Detecting topics (this may take 30-60 seconds)...")
    df["topic"] = df["message"].apply(detect_topic)

    os.makedirs("outputs", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"  -> Saved to: {OUTPUT_PATH}")
    print("\nTopic distribution:")
    print(df["topic"].value_counts().to_string())

    print("\nSample results:")
    sample = df[["sender", "message", "topic"]].head(12)
    for _, row in sample.iterrows():
        print(f"  [{row['topic']:12s}] {row['sender']}: {row['message'][:60]}")

    return df


if __name__ == "__main__":
    run_topic_detection()