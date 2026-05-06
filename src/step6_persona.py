"""
STEP 6: User Persona Extraction
---------------------------------
We analyze all User 1 messages and extract:
  - Hobbies & interests
  - Personality traits
  - Communication style
  - Habits & lifestyle
  - Pet ownership
  - Work / study situation

Method: rule-based regex patterns + keyword counters.
No AI or paid API needed.

Run from D:\chat_analyzer:
    python src/step6_persona.py
"""

import pandas as pd
import os
import re
from collections import Counter

INPUT_PATH  = os.path.join("outputs", "topics.csv")
OUTPUT_PATH = os.path.join("outputs", "persona.txt")


# ── Keyword lists for trait detection ────────────────────────────────────────

HOBBY_KEYWORDS = {
    "reading"     : ["read", "book", "novel", "fiction", "library", "author"],
    "music"       : ["music", "sing", "song", "band", "concert", "guitar", "piano", "playlist"],
    "cooking"     : ["cook", "bake", "recipe", "kitchen", "chef", "grill", "meal"],
    "gaming"      : ["game", "gaming", "video game", "play games", "console", "xbox", "playstation"],
    "sports"      : ["sport", "soccer", "basketball", "tennis", "football", "swim", "run", "gym", "workout"],
    "hiking"      : ["hike", "hiking", "trail", "outdoors", "nature", "camping"],
    "travel"      : ["travel", "trip", "visit", "explore", "vacation", "abroad", "city", "country"],
    "arts & crafts": ["draw", "paint", "art", "craft", "sketch", "design", "creative"],
    "movies & tv" : ["movie", "film", "watch", "series", "show", "netflix", "cinema", "tv"],
    "gardening"   : ["garden", "plant", "grow", "flower", "vegetable", "herb"],
}

PERSONALITY_KEYWORDS = {
    "friendly"    : ["love", "great", "awesome", "amazing", "wonderful", "happy", "excited", "fun"],
    "empathetic"  : ["understand", "feel", "sorry", "support", "help", "care", "hope you", "that's tough"],
    "curious"     : ["interesting", "wonder", "curious", "learn", "question", "tell me", "how does", "why"],
    "humorous"    : ["haha", "lol", "funny", "joke", "laugh", "hilarious", "hehe"],
    "adventurous" : ["adventure", "try new", "explore", "challenge", "risk", "brave", "dare"],
    "family-oriented": ["family", "mom", "dad", "kids", "children", "parent", "sibling", "home"],
    "responsible" : ["work hard", "responsible", "serious", "dedicated", "careful", "organize"],
}

COMMUNICATION_MARKERS = {
    "uses questions"     : r"\?",
    "uses exclamations"  : r"!",
    "expresses agreement": r"\b(agree|exactly|absolutely|totally|yes|yeah|right)\b",
    "shares opinions"    : r"\bI think\b|\bI believe\b|\bI feel\b|\bin my opinion\b",
    "tells stories"      : r"\bonce\b|\bwhen I was\b|\bI remember\b|\bone time\b",
}

PET_KEYWORDS  = ["dog", "cat", "puppy", "kitten", "pet", "fish", "bird", "hamster", "rabbit"]
WORK_KEYWORDS = ["work", "job", "career", "office", "boss", "company", "employee", "profession"]
STUDY_KEYWORDS= ["study", "student", "college", "school", "university", "class", "degree", "major"]


def count_keyword_hits(messages: list, keywords: list) -> int:
    """Count how many messages contain at least one keyword from the list."""
    text = " ".join(messages).lower()
    return sum(1 for kw in keywords if kw in text)


def extract_hobbies(messages: list) -> list:
    """Return list of hobbies ordered by how often they appear."""
    text = " ".join(messages).lower()
    scores = {}
    for hobby, keywords in HOBBY_KEYWORDS.items():
        score = sum(text.count(kw) for kw in keywords)
        if score > 0:
            scores[hobby] = score
    # Return top hobbies sorted by score
    return [h for h, _ in sorted(scores.items(), key=lambda x: -x[1])]


def extract_personality(messages: list) -> list:
    """Return personality traits present in messages."""
    text = " ".join(messages).lower()
    traits = []
    for trait, keywords in PERSONALITY_KEYWORDS.items():
        score = sum(text.count(kw) for kw in keywords)
        if score >= 5:   # must appear at least 5 times to count
            traits.append(trait)
    return traits


def extract_communication_style(messages: list) -> dict:
    """Analyze how the user communicates."""
    text = " ".join(messages)
    total = len(messages)
    style = {}

    for label, pattern in COMMUNICATION_MARKERS.items():
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        # Express as % of messages
        pct = round((matches / total) * 100, 1)
        style[label] = pct

    # Average message length
    avg_len = round(sum(len(m.split()) for m in messages) / total, 1)
    style["avg words per message"] = avg_len

    return style


def extract_self_descriptions(messages: list) -> list:
    """Find sentences where user describes themselves (I am a ..., I work as ...)."""
    patterns = [
        r"I(?:'m| am) (?:a |an )([a-zA-Z\s]{3,30}?)(?:\.|,|!|\band\b)",
        r"I (?:work as|work in|work at) ([a-zA-Z\s]{3,25}?)(?:\.|,|!)",
        r"I (?:study|studied|majored in) ([a-zA-Z\s]{3,25}?)(?:\.|,|!)",
    ]
    found = []
    for msg in messages:
        for p in patterns:
            matches = re.findall(p, msg, re.IGNORECASE)
            for m in matches:
                clean = m.strip().lower()
                if 2 < len(clean) < 40:
                    found.append(clean)
    # Return most common self-descriptions
    return [item for item, _ in Counter(found).most_common(10)]


def build_persona_report(messages: list) -> str:
    """Build a human-readable persona report."""
    total = len(messages)

    hobbies      = extract_hobbies(messages)
    personality  = extract_personality(messages)
    comm_style   = extract_communication_style(messages)
    self_desc    = extract_self_descriptions(messages)
    has_pet      = count_keyword_hits(messages, PET_KEYWORDS) > 20
    works        = count_keyword_hits(messages, WORK_KEYWORDS) > 30
    studies      = count_keyword_hits(messages, STUDY_KEYWORDS) > 10

    # ── Build report text ────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("       USER PERSONA REPORT")
    lines.append("=" * 60)
    lines.append(f"\nBased on analysis of {total:,} User 1 messages.\n")

    lines.append("── IDENTITY ─────────────────────────────────────────────")
    if self_desc:
        lines.append("Self-descriptions found:")
        for d in self_desc[:6]:
            lines.append(f"  • {d}")
    if works:
        lines.append("  • Mentions work/job regularly")
    if studies:
        lines.append("  • Mentions studying/school regularly")
    if has_pet:
        lines.append("  • Likely a pet owner (mentions pets frequently)")

    lines.append("\n── HOBBIES & INTERESTS ──────────────────────────────────")
    if hobbies:
        for h in hobbies[:6]:
            lines.append(f"  • {h}")
    else:
        lines.append("  • No strong hobby signals detected")

    lines.append("\n── PERSONALITY TRAITS ───────────────────────────────────")
    if personality:
        for t in personality:
            lines.append(f"  • {t}")
    else:
        lines.append("  • Traits unclear from data")

    lines.append("\n── COMMUNICATION STYLE ──────────────────────────────────")
    for label, value in comm_style.items():
        if label == "avg words per message":
            lines.append(f"  • Average message length : {value} words")
        else:
            lines.append(f"  • {label:28s}: {value}% of messages")

    lines.append("\n── HABITS & LIFESTYLE ───────────────────────────────────")
    avg_len = comm_style.get("avg words per message", 0)
    if avg_len < 10:
        lines.append("  • Tends to write short, concise messages")
    elif avg_len < 20:
        lines.append("  • Writes medium-length, conversational messages")
    else:
        lines.append("  • Writes long, detailed messages")

    q_pct = comm_style.get("uses questions", 0)
    if q_pct > 30:
        lines.append("  • Very inquisitive — asks a lot of questions")
    elif q_pct > 15:
        lines.append("  • Moderately curious — asks questions regularly")

    agree_pct = comm_style.get("expresses agreement", 0)
    if agree_pct > 10:
        lines.append("  • Agreeable and easy-going in conversations")

    story_pct = comm_style.get("tells stories", 0)
    if story_pct > 5:
        lines.append("  • Often shares personal stories and memories")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def run_persona_extraction():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Run step2_topics.py first.")
        return

    print("Loading messages...")
    df = pd.read_csv(INPUT_PATH)

    # Focus on User 1 (the "primary user" we're profiling)
    user1_msgs = df[df["sender"] == "User 1"]["message"].dropna().tolist()
    print(f"  -> {len(user1_msgs):,} User 1 messages found")

    print("Extracting persona...")
    report = build_persona_report(user1_msgs)

    # Save report
    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  -> Saved to: {OUTPUT_PATH}")
    print("\n" + report)

    return report


if __name__ == "__main__":
    run_persona_extraction()