# 🤖 Conversation Analyzer — AI/ML Engineer Intern Task

A beginner-friendly NLP pipeline that performs **topic detection**, **RAG retrieval**,
**persona extraction**, and serves a **Streamlit chatbot** — all without paid APIs.

---

## 🚀 Live Demo

> **Streamlit Cloud URL:** *https://kastack.streamlit.app/*  
> **Demo Video:** *https://www.loom.com/share/2b33cf29b38a4f91853ea4668e50ace2*

---

## 📁 Project Structure

```
chat_analyzer/
├── data/
│   └── conversations.csv          # Raw input (11,001 conversations)
├── outputs/
│   ├── messages.csv               # Parsed individual messages (191,592 rows)
│   ├── topics.csv                 # Each message with topic label
│   ├── topic_summaries.csv        # TF-IDF summary per topic (9 topics)
│   ├── checkpoint_summaries.csv   # Summary every 100 messages (1,916 checkpoints)
│   └── persona.txt                # Extracted user persona report
├── src/
│   ├── step1_load.py              # CSV parsing
│   ├── step2_topics.py            # Topic detection
│   ├── step3_summaries.py         # Topic-based summaries
│   ├── step4_checkpoints.py       # 100-message checkpoints
│   ├── step5_rag.py               # RAG retrieval engine
│   ├── step6_persona.py           # Persona extraction
│   └── step7_chatbot.py           # Streamlit chatbot UI
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run (Step by Step)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your CSV
Put `conversations.csv` inside the `data/` folder.

### 3. Run the pipeline (in order)
```bash
python src/step1_load.py        # Parse CSV → messages.csv
python src/step2_topics.py      # Detect topics → topics.csv
python src/step3_summaries.py   # Summarize topics → topic_summaries.csv
python src/step4_checkpoints.py # 100-msg summaries → checkpoint_summaries.csv
python src/step6_persona.py     # Extract persona → persona.txt
```

### 4. Launch the chatbot
```bash
streamlit run src/step7_chatbot.py
```
Opens at `http://localhost:8501`

---

## 🧠 How Topic Detection Works

**Method:** Keyword-based scoring (no ML model needed)

Each message is compared against 9 topic dictionaries:

```
food     → ["eat", "cook", "recipe", "restaurant", "meal", ...]
travel   → ["trip", "city", "visit", "moving", "vacation", ...]
hobbies  → ["read", "music", "sport", "game", "art", ...]
family   → ["mom", "dad", "kids", "married", "baby", ...]
work     → ["job", "career", "office", "boss", "salary", ...]
pets     → ["dog", "cat", "puppy", "kitten", "vet", ...]
health   → ["doctor", "sick", "diet", "sleep", "therapy", ...]
technology → ["phone", "app", "coding", "internet", ...]
general  → (default, no strong keyword match)
```

**Scoring formula:**
```
score(topic) = number of topic keywords found in message
best_topic   = argmax(score)
```

**Topic Segmentation** — consecutive messages with the same topic
form a "segment":

```
Conversation #1:
  Segment 1 | GENERAL  | Msgs 0–1    → "Hi! How are you?"
  Segment 2 | TRAVEL   | Msgs 2–8    → "I'm moving to Portland..."
  Segment 3 | HOBBIES  | Msgs 9–10   → "I love Powell's Books..."
  Segment 4 | TRAVEL   | Msgs 11–11  → "You definitely should visit!"
  Segment 5 | GENERAL  | Msgs 12–13  → "Thanks for the recommendation!"
```

This is visible in the **📚 Topic Segments** tab of the chatbot.

---

## 🔁 How 100-Message Checkpoints Work

Every 100 messages across the full dataset gets a summary checkpoint:

```
Checkpoint #0   | Msgs 0–99     | Topics: general, hobbies, travel
Checkpoint #1   | Msgs 100–199  | Topics: general, food, hobbies
Checkpoint #2   | Msgs 200–299  | Topics: general, food, work
...
Checkpoint #1915| Msgs 191500–191591 | Topics: general, hobbies, food
```

**Summary method:** TF-IDF scoring
- Each message in the chunk is vectorized
- Words that are important in THIS chunk but rare elsewhere score high
- Top 5 highest-scoring sentences are selected as the summary

Total: **1,916 checkpoints** covering all 191,592 messages.

---

## 🔍 How RAG Retrieval Works

**RAG = Retrieve, then Answer**

```
User Question
      │
      ▼
TF-IDF vectorize the question
      │
      ▼
Cosine similarity vs. 1,925 stored summaries
  ├── 9   topic summaries      (broad topic coverage)
  └── 1,916 checkpoint summaries (specific time windows)
      │
      ▼
Top-K most similar summaries returned as "context"
      │
      ▼
Answer displayed with source labels + similarity scores
```

**Example:**
```
Query: "What do people say about food?"

Retrieved:
  [score=0.412] Topic: food (30,462 messages)
                → "I love to cook Italian dishes. My favorite recipe..."
  [score=0.234] Checkpoint #1764 (Msgs 176400–176499, topics: food, hobbies)
                → "I make pasta from scratch every Sunday..."
```

No vector database needed — pure sklearn TF-IDF + cosine similarity.

---

## 👤 How Persona Is Built

**Method:** Rule-based regex + keyword frequency counting

### Layer 1 — Self-description extraction (regex)
Patterns like:
```python
r"I(?:'m| am) (?:a |an )([a-zA-Z\s]{3,30})"
# Matches: "I'm a teacher", "I am a software engineer"
```
Finds: teacher, writer, student, musician, artist, firefighter, EMT...

### Layer 2 — Hobby detection (keyword frequency)
```
reading   → book, novel, fiction, library, author       → 8,200+ hits
music     → music, sing, song, band, concert, guitar    → 7,100+ hits
movies/tv → movie, film, watch, series, netflix         → 6,800+ hits
```

### Layer 3 — Personality traits (evidence-based)

| Trait | Evidence Keywords | Detection |
|-------|------------------|-----------|
| friendly | love, great, awesome, amazing | high positive word usage |
| curious | ? (questions) | 27% of messages have `?` |
| humorous | haha, lol, funny, hilarious | frequent laughter words |
| empathetic | sorry, hope, feel, support | response-to-feelings pattern |
| adventurous | travel, explore, try new | travel/exploration frequency |
| family-oriented | family, mom, dad, kids | family word frequency |

### Layer 4 — Communication style metrics
```
Avg words per message : 11.1  → conversational, medium-length
Messages with "!"     : 55%   → very expressive
Messages with "?"     : 27%   → moderately curious
Uses "I think/feel"   : frequent → opinionated, self-aware
```

These are **measurable signals**, not AI-generated guesses.

---

## 💬 Chatbot Capabilities

The chatbot (Tab: 💬 Chat) answers using a routing system:

| Question Type | Strategy |
|---------------|----------|
| "What kind of person is this user?" | Persona report |
| "What are their habits?" | Habit/communication metrics |
| "What hobbies do they have?" | Hobby frequency table |
| "What topics do people discuss?" | Topic distribution |
| "How many messages in dataset?" | Direct stats |
| Any other question | TF-IDF RAG retrieval |

**RAG debug panel** shows exactly which summaries were retrieved and their scores.

---

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total conversations | 11,001 |
| Total messages | 191,592 |
| User 1 messages | 98,079 |
| User 2 messages | 93,513 |
| Topics detected | 9 |
| Checkpoints (100-msg) | 1,916 |
| RAG index size | 1,925 documents |

---

## 🛠️ Tech Stack

| Component | Library | Why |
|-----------|---------|-----|
| Data loading | pandas | Simple CSV handling |
| Topic detection | regex + keywords | No model needed |
| Summarization | scikit-learn TF-IDF | Lightweight, fast |
| RAG retrieval | sklearn cosine_similarity | No vector DB needed |
| Persona extraction | re, collections.Counter | Rule-based, explainable |
| UI | Streamlit | Beginner-friendly web UI |

**No paid APIs. No GPU. No heavy models.**

---

## ☁️ Deploying to Streamlit Cloud (Free)

1. Push your project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → connect your GitHub repo
4. Set **Main file path:** `src/step7_chatbot.py`
5. Click **Deploy**

> ⚠️ The `outputs/` CSV files must be committed to GitHub for the app to work.
> Since `checkpoint_summaries.csv` is large (~1.4MB), make sure Git LFS is not
> blocking it, or pre-generate and commit all output files.

---

## 🎥 Demo Video Script (Loom)

**Suggested flow (3–5 minutes):**

1. **(0:00)** "Hi, I built a conversation analyzer for the AI/ML intern task."
2. **(0:15)** Show `outputs/messages.csv` — "Step 1 parsed 191,592 messages."
3. **(0:30)** Open **📚 Topic Segments** tab → select Conversation #1 → show segments
   - "This is topic segmentation — you can see each topic change with exact message ranges."
4. **(1:00)** Open **🔁 Checkpoints** tab → show Checkpoint #0, #1, #2
   - "Every 100 messages gets a TF-IDF summary. 1,916 total checkpoints."
5. **(1:30)** Open **🧠 Persona** tab → show trait evidence table
   - "Traits are backed by measurable signals — not AI guessing."
6. **(2:00)** Open **💬 Chat** tab → ask: *"What kind of person is this user?"*
7. **(2:20)** Ask: *"What do people say about food?"* → show RAG debug panel
   - "You can see exactly which documents were retrieved and their similarity scores."
8. **(2:45)** Ask: *"What are their habits?"*
9. **(3:00)** Show `src/step5_rag.py` briefly — "Pure TF-IDF cosine similarity, no vector DB."
10. **(3:20)** "Everything runs locally, no paid APIs, beginner-readable code."

---

## 👨‍💻 Author

Built for AI/ML Engineer Intern screening task.
Pipeline: CSV parsing → keyword topic detection → TF-IDF summarization →
100-message checkpoints → cosine similarity RAG → rule-based persona → Streamlit UI.
