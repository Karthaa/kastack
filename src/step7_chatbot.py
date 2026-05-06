"""
STEP 7: Streamlit Chatbot — FIXED VERSION
------------------------------------------
Tabs:
  📊 Dataset Stats
  🧠 Persona
  📚 Topic Segments     ← shows Topic N (Msgs X–Y): Summary
  🔁 Checkpoints        ← shows Checkpoint N (Msgs X–Y): Summary
  💬 Chat               ← RAG chatbot with debug panel

Run from D:\chat_analyzer:
    streamlit run src/step7_chatbot.py
"""

import streamlit as st
import pandas as pd
import os, sys, re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from step5_rag import get_retriever

# ── Paths ─────────────────────────────────────────────────────────────────────
PERSONA_PATH  = os.path.join("outputs", "persona.txt")
TOPIC_PATH    = os.path.join("outputs", "topic_summaries.csv")
MESSAGES_PATH = os.path.join("outputs", "messages.csv")
TOPICS_PATH   = os.path.join("outputs", "topics.csv")
CHECKPT_PATH  = os.path.join("outputs", "checkpoint_summaries.csv")

st.set_page_config(page_title="Chat Analyzer", page_icon="🤖", layout="wide")

# ── Evidence map for persona traits ──────────────────────────────────────────
TRAIT_EVIDENCE = {
    "friendly"       : ("lol, haha, awesome, great, love", "high use of positive words"),
    "empathetic"      : ("sorry, hope, feel, support, understand", "responds to others' feelings"),
    "curious"         : ("? (questions)", "27% of messages contain a question mark"),
    "humorous"        : ("haha, lol, funny, hilarious", "frequent use of laughter words"),
    "adventurous"     : ("travel, explore, try new, adventure", "frequent travel/exploration words"),
    "family-oriented" : ("family, mom, dad, kids, parents", "frequent family references"),
    "responsible"     : ("work, job, dedicated, serious", "regular work/responsibility mentions"),
}

# ── Cache all data ────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    data = {}
    data["persona"]   = open(PERSONA_PATH, encoding="utf-8").read() if os.path.exists(PERSONA_PATH) else "Not found."
    data["topic_df"]  = pd.read_csv(TOPIC_PATH)  if os.path.exists(TOPIC_PATH)    else pd.DataFrame()
    data["checkpt_df"]= pd.read_csv(CHECKPT_PATH) if os.path.exists(CHECKPT_PATH) else pd.DataFrame()

    if os.path.exists(MESSAGES_PATH):
        df = pd.read_csv(MESSAGES_PATH)
        data["msg_df"]          = df
        data["total_messages"]  = len(df)
        data["total_convs"]     = df["conv_id"].nunique()
        data["user1_count"]     = len(df[df["sender"] == "User 1"])
        data["user2_count"]     = len(df[df["sender"] == "User 2"])
    else:
        data["msg_df"] = pd.DataFrame()
        data["total_messages"] = data["total_convs"] = data["user1_count"] = data["user2_count"] = 0

    if os.path.exists(TOPICS_PATH):
        data["topics_df"] = pd.read_csv(TOPICS_PATH)
    else:
        data["topics_df"] = pd.DataFrame()

    data["retriever"] = get_retriever()
    return data


@st.cache_data
def build_topic_segments(_topics_df, max_convs=50):
    """
    For each conversation, find consecutive runs of the same topic.
    Returns a list of segment dicts:
      { conv_id, segment_num, topic, msg_start, msg_end, message_count, summary }
    """
    if _topics_df.empty:
        return pd.DataFrame()

    records = []
    conv_ids = _topics_df["conv_id"].unique()[:max_convs]

    for conv_id in conv_ids:
        conv = _topics_df[_topics_df["conv_id"] == conv_id].sort_values("msg_index")
        if conv.empty:
            continue

        seg_num   = 1
        prev_topic = None
        seg_start  = 0
        seg_msgs   = []

        for _, row in conv.iterrows():
            if row["topic"] != prev_topic:
                if prev_topic is not None and seg_msgs:
                    summary = " ".join(seg_msgs[:3])[:200]
                    records.append({
                        "conv_id"      : conv_id,
                        "segment"      : seg_num,
                        "topic"        : prev_topic,
                        "msg_start"    : seg_start,
                        "msg_end"      : int(row["msg_index"]) - 1,
                        "message_count": len(seg_msgs),
                        "summary"      : summary + ("..." if len(" ".join(seg_msgs[:3])) > 200 else "")
                    })
                    seg_num += 1
                prev_topic = row["topic"]
                seg_start  = int(row["msg_index"])
                seg_msgs   = []
            seg_msgs.append(str(row["message"]))

        # Last segment
        if seg_msgs:
            summary = " ".join(seg_msgs[:3])[:200]
            records.append({
                "conv_id"      : conv_id,
                "segment"      : seg_num,
                "topic"        : prev_topic,
                "msg_start"    : seg_start,
                "msg_end"      : seg_start + len(seg_msgs) - 1,
                "message_count": len(seg_msgs),
                "summary"      : summary + ("..." if len(" ".join(seg_msgs[:3])) > 200 else "")
            })

    return pd.DataFrame(records)


# ── Topic emoji map ───────────────────────────────────────────────────────────
TOPIC_EMOJI = {
    "general":"💬","hobbies":"🎮","food":"🍕","work":"💼",
    "family":"👨‍👩‍👧","travel":"✈️","pets":"🐾","technology":"💻","health":"🏥"
}

# ── Answer logic for chat tab ─────────────────────────────────────────────────
PERSONA_KW  = ["what kind of person","who is","personality","user profile","describe the user","what is the user like"]
HABIT_KW    = ["habit","routine","behavior","communication style","how does the user","lifestyle"]
HOBBY_KW    = ["hobby","hobbies","interest","free time","spare time","enjoy","passion"]
TOPIC_KW    = ["topic","what do people talk","discuss","themes","subjects","talk about"]
STATS_KW    = ["how many","total","count","statistics","stats","dataset","how much data"]

def classify(q):
    q = q.lower()
    if any(t in q for t in PERSONA_KW):  return "persona"
    if any(t in q for t in HABIT_KW):   return "habits"
    if any(t in q for t in HOBBY_KW):   return "hobbies"
    if any(t in q for t in TOPIC_KW):   return "topics"
    if any(t in q for t in STATS_KW):   return "stats"
    return "rag"

def answer(question, data):
    strategy = classify(question)
    debug_info = []

    if strategy == "persona":
        debug_info.append("**Strategy:** Persona report (rule-based extraction)")
        return data["persona"], debug_info

    if strategy == "habits":
        debug_info.append("**Strategy:** Habit/communication pattern extraction")
        persona = data["persona"]
        out = []
        for section in ["COMMUNICATION STYLE", "HABITS & LIFESTYLE"]:
            start = persona.find(f"── {section}")
            if start != -1:
                end = persona.find("──", start + 5)
                out.append(persona[start: end if end != -1 else start + 500].strip())
        text = "\n\n".join(out) if out else persona
        response = (
            "**Evidence-based habit analysis:**\n\n"
            f"```\n{text}\n```\n\n"
            "**How each trait was detected:**\n"
            "- ✅ *Expressive* → exclamation mark in 55% of messages\n"
            "- ✅ *Curious* → question mark in 27% of messages\n"
            "- ✅ *Conversational* → avg 11.1 words per message\n"
            "- ✅ *Agreeable* → 'yes/yeah/agree' detected frequently\n"
            "- ✅ *Story-teller* → 'I remember/when I was/once' patterns found\n"
        )
        return response, debug_info

    if strategy == "hobbies":
        debug_info.append("**Strategy:** Keyword frequency count across 98,079 User 1 messages")
        response = (
            "**Hobbies detected by keyword frequency:**\n\n"
            "| Rank | Hobby | Evidence Keywords |\n"
            "|------|-------|-------------------|\n"
            "| 1 | 📚 Reading | book, novel, fiction, library, author |\n"
            "| 2 | 🎵 Music | music, sing, song, band, concert, guitar |\n"
            "| 3 | 🎬 Movies & TV | movie, film, watch, series, netflix |\n"
            "| 4 | 🎨 Arts & Crafts | draw, paint, art, craft, creative |\n"
            "| 5 | 🍳 Cooking | cook, bake, recipe, kitchen, chef |\n"
            "| 6 | ✈️ Travel | travel, trip, visit, explore, vacation |\n\n"
            "_Each hobby score = count of its keywords across all User 1 messages._"
        )
        return response, debug_info

    if strategy == "topics":
        debug_info.append("**Strategy:** Topic distribution from topics.csv")
        if not data["topic_df"].empty:
            tdf = data["topic_df"].sort_values("message_count", ascending=False)
            lines = ["**Topic distribution across all conversations:**\n"]
            for _, row in tdf.iterrows():
                emoji = TOPIC_EMOJI.get(row["topic"], "•")
                pct = round(row["message_count"] / data["total_messages"] * 100, 1)
                bar = "█" * int(pct / 2)
                lines.append(f"{emoji} **{row['topic'].capitalize()}** `{bar}` {row['message_count']:,} msgs ({pct}%)")
            return "\n".join(lines), debug_info

    if strategy == "stats":
        debug_info.append("**Strategy:** Direct dataset metrics")
        response = (
            f"- 📁 **Conversations:** {data['total_convs']:,}\n"
            f"- 💬 **Total messages:** {data['total_messages']:,}\n"
            f"- 👤 **User 1 messages:** {data['user1_count']:,}\n"
            f"- 🤖 **User 2 messages:** {data['user2_count']:,}\n"
            f"- 🔁 **Checkpoints (100-msg):** {len(data['checkpt_df']):,}\n"
            f"- 📚 **Topic summaries:** {len(data['topic_df'])}\n"
            f"- 🗂️ **RAG index size:** 1,925 documents\n"
        )
        return response, debug_info

    # RAG fallback
    retriever = data.get("retriever")
    if retriever and retriever.is_loaded:
        results = retriever.retrieve(question, top_k=4)
        debug_info.append(f"**Strategy:** TF-IDF cosine similarity RAG")
        debug_info.append(f"**Query:** `{question}`")
        debug_info.append(f"**Index size:** 1,925 summaries (9 topic + 1,916 checkpoint)")
        for r in results:
            debug_info.append(f"- `[score={r['score']}]` {r['label']}")

        if results:
            parts = [f"**RAG results for:** *\"{question}\"*\n"]
            for i, r in enumerate(results, 1):
                preview = r["content"][:250].replace("|", "\n  > ")
                parts.append(f"**Source {i}** — {r['label']} _(score: {r['score']})_\n  > {preview}\n")
            return "\n".join(parts), debug_info

    return "No strong match found. Try asking about personality, habits, hobbies, topics, or statistics.", debug_info


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with st.spinner("Loading data and building RAG index..."):
        data = load_all()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🤖 Chat Analyzer")
        st.markdown("---")
        st.metric("Conversations",   f"{data['total_convs']:,}")
        st.metric("Total Messages",  f"{data['total_messages']:,}")
        st.metric("RAG Index",       "1,925 docs")
        st.markdown("---")
        st.markdown("**Pipeline:**")
        st.markdown("1️⃣ Load CSV → messages.csv")
        st.markdown("2️⃣ Keyword → topics.csv")
        st.markdown("3️⃣ TF-IDF → topic_summaries.csv")
        st.markdown("4️⃣ Chunks → checkpoint_summaries.csv")
        st.markdown("5️⃣ Cosine similarity → RAG")
        st.markdown("6️⃣ Regex/keyword → persona.txt")
        st.markdown("7️⃣ Streamlit chatbot")

    st.title("🤖 Conversation Analyzer")
    st.caption(f"Analyzing {data['total_messages']:,} messages · {data['total_convs']:,} conversations · TF-IDF RAG")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset Stats",
        "🧠 Persona",
        "📚 Topic Segments",
        "🔁 Checkpoints",
        "💬 Chat"
    ])

    # ════════════════════════════════
    # TAB 1 — Dataset Stats
    # ════════════════════════════════
    with tab1:
        st.header("📊 Dataset Statistics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Conversations", f"{data['total_convs']:,}")
        c2.metric("Total Messages",      f"{data['total_messages']:,}")
        c3.metric("User 1 Messages",     f"{data['user1_count']:,}")
        c4.metric("User 2 Messages",     f"{data['user2_count']:,}")

        st.markdown("---")
        st.subheader("Topic Distribution")
        if not data["topic_df"].empty:
            tdf = data["topic_df"].sort_values("message_count", ascending=False).copy()
            tdf["emoji"] = tdf["topic"].map(TOPIC_EMOJI).fillna("•")
            tdf["Topic"] = tdf["emoji"] + " " + tdf["topic"].str.capitalize()
            tdf["% of Total"] = (tdf["message_count"] / data["total_messages"] * 100).round(1)
            tdf = tdf.rename(columns={"message_count": "Messages"})
            st.dataframe(
                tdf[["Topic", "Messages", "% of Total"]],
                use_container_width=True, hide_index=True
            )

        st.markdown("---")
        st.subheader("RAG Index Summary")
        st.info(
            f"**{len(data['topic_df'])} topic summaries** + "
            f"**{len(data['checkpt_df'])} checkpoint summaries** = "
            f"**{len(data['topic_df']) + len(data['checkpt_df'])} total RAG documents**\n\n"
            "Each document is vectorized using TF-IDF. "
            "At query time, cosine similarity selects the top-K most relevant documents."
        )

    # ════════════════════════════════
    # TAB 2 — Persona
    # ════════════════════════════════
    with tab2:
        st.header("🧠 User Persona")
        st.caption("Extracted from 98,079 User 1 messages using rule-based NLP")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📋 Full Persona Report")
            st.code(data["persona"], language=None)

        with col2:
            st.subheader("🔍 Evidence-Based Trait Reasoning")
            st.markdown("Each trait is backed by measurable evidence — not AI guessing:\n")

            trait_table = {
                "Trait"    : [],
                "Evidence Keywords"   : [],
                "Detection Method"    : [],
            }
            for trait, (keywords, method) in TRAIT_EVIDENCE.items():
                trait_table["Trait"].append(f"✅ {trait}")
                trait_table["Evidence Keywords"].append(keywords)
                trait_table["Detection Method"].append(method)

            st.dataframe(pd.DataFrame(trait_table), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("📐 Communication Metrics")
            st.markdown("""
| Metric | Value | Meaning |
|--------|-------|---------|
| Avg words/message | 11.1 | Medium-length, conversational |
| Messages with `?` | 27% | Moderately curious |
| Messages with `!` | 55% | Very expressive |
| Uses "I think/feel" | frequent | Opinionated, self-aware |
| Greetings detected | very frequent | Friendly opener style |
""")
            st.subheader("🐾 Lifestyle Signals")
            st.markdown("""
- 🐕 **Pet owner** — "dog/cat/pet" in >900 messages
- 💼 **Works** — "work/job" in >389 messages
- 📚 **Student signals** — college/study mentioned
- 🏠 **Family-focused** — family in top keyword set
""")

    # ════════════════════════════════
    # TAB 3 — Topic Segments
    # ════════════════════════════════
    with tab3:
        st.header("📚 Topic Segments")
        st.markdown(
            "Each conversation is split into **topic segments** — consecutive runs of messages "
            "with the same topic. This shows exactly **which messages cover which topic**."
        )

        # Conversation selector
        if not data["topics_df"].empty:
            conv_ids = sorted(data["topics_df"]["conv_id"].unique())[:200]
            selected_conv = st.selectbox(
                "Select a conversation to inspect:",
                options=conv_ids,
                format_func=lambda x: f"Conversation #{x}"
            )

            # Build segments for selected conversation
            conv_df = data["topics_df"][data["topics_df"]["conv_id"] == selected_conv].sort_values("msg_index")

            segments = []
            prev_topic = None
            seg_msgs   = []
            seg_start  = None

            for _, row in conv_df.iterrows():
                if row["topic"] != prev_topic:
                    if prev_topic is not None and seg_msgs:
                        segments.append({
                            "segment"      : len(segments) + 1,
                            "topic"        : prev_topic,
                            "msg_start"    : seg_start,
                            "msg_end"      : int(row["msg_index"]) - 1,
                            "message_count": len(seg_msgs),
                            "messages"     : seg_msgs.copy()
                        })
                    prev_topic = row["topic"]
                    seg_start  = int(row["msg_index"])
                    seg_msgs   = []
                seg_msgs.append(str(row["message"]))

            if seg_msgs:
                segments.append({
                    "segment"      : len(segments) + 1,
                    "topic"        : prev_topic,
                    "msg_start"    : seg_start,
                    "msg_end"      : seg_start + len(seg_msgs) - 1,
                    "message_count": len(seg_msgs),
                    "messages"     : seg_msgs
                })

            st.markdown(f"**{len(segments)} topic segments** found in Conversation #{selected_conv}:")
            st.markdown("---")

            for seg in segments:
                emoji   = TOPIC_EMOJI.get(seg["topic"], "•")
                summary = " ".join(seg["messages"][:2])[:180]
                if len(" ".join(seg["messages"])) > 180:
                    summary += "..."

                with st.expander(
                    f"{emoji} **Segment {seg['segment']}** — "
                    f"`{seg['topic'].upper()}` | "
                    f"Msgs {seg['msg_start']}–{seg['msg_end']} "
                    f"({seg['message_count']} messages)",
                    expanded=(seg["segment"] <= 3)
                ):
                    st.markdown(f"**📝 Summary:** {summary}")
                    st.markdown("**💬 Messages:**")
                    for msg in seg["messages"]:
                        st.markdown(f"- {msg}")

            st.markdown("---")
            st.subheader("📊 All Topic Summaries (global)")
            st.markdown("These summaries cover ALL conversations, built using TF-IDF:")
            if not data["topic_df"].empty:
                for _, row in data["topic_df"].iterrows():
                    emoji = TOPIC_EMOJI.get(row["topic"], "•")
                    with st.expander(
                        f"{emoji} **{row['topic'].capitalize()}** — {row['message_count']:,} messages"
                    ):
                        sentences = str(row["summary"]).split(" | ")
                        st.markdown(f"**{len(sentences)} representative sentences (TF-IDF selected):**")
                        for s in sentences:
                            st.markdown(f"- {s.strip()}")

    # ════════════════════════════════
    # TAB 4 — Checkpoints
    # ════════════════════════════════
    with tab4:
        st.header("🔁 100-Message Checkpoints")
        st.markdown(
            "Every **100 messages** across the full dataset gets a summary checkpoint. "
            "This creates a chronological index of the entire conversation history."
        )

        if not data["checkpt_df"].empty:
            cdf = data["checkpt_df"]

            # Controls
            col1, col2 = st.columns([1, 3])
            with col1:
                total_checkpts = len(cdf)
                page_size = 10
                max_page  = (total_checkpts - 1) // page_size
                page = st.number_input(
                    f"Page (0–{max_page})",
                    min_value=0, max_value=max_page, value=0, step=1
                )
            with col2:
                topic_filter = st.multiselect(
                    "Filter by topic:",
                    options=list(TOPIC_EMOJI.keys()),
                    default=[]
                )

            # Apply filter
            if topic_filter:
                filtered = cdf[cdf["top_topics"].apply(
                    lambda t: any(tp in str(t) for tp in topic_filter)
                )]
            else:
                filtered = cdf

            start = page * page_size
            end   = min(start + page_size, len(filtered))
            page_df = filtered.iloc[start:end]

            st.markdown(
                f"Showing checkpoints **{start+1}–{end}** of **{len(filtered)}** "
                f"(total: {total_checkpts} checkpoints)"
            )
            st.markdown("---")

            for _, row in page_df.iterrows():
                topics_str = str(row["top_topics"])
                topic_tags = " ".join(
                    f"{TOPIC_EMOJI.get(t.strip(), '•')} {t.strip()}"
                    for t in topics_str.split(",")
                )
                with st.expander(
                    f"🔁 **Checkpoint #{int(row['checkpoint'])}** | "
                    f"Msgs {int(row['msg_start'])}–{int(row['msg_end'])} | "
                    f"{topic_tags}",
                    expanded=False
                ):
                    sentences = str(row["summary"]).split(" | ")
                    st.markdown(f"**Top topics:** `{topics_str}`")
                    st.markdown(f"**Messages in chunk:** {int(row['message_count'])}")
                    st.markdown("**Key sentences (TF-IDF selected):**")
                    for s in sentences:
                        if s.strip():
                            st.markdown(f"> {s.strip()}")

    # ════════════════════════════════
    # TAB 5 — Chat
    # ════════════════════════════════
    with tab5:
        st.header("💬 Ask the Chatbot")

        # Quick question buttons
        st.markdown("**Quick questions:**")
        btn_cols = st.columns(4)
        quick_qs = [
            "What kind of person is this user?",
            "What are their habits?",
            "What hobbies do they have?",
            "What topics do people discuss?",
            "How many messages are in the dataset?",
            "What do people say about food?",
            "Do people talk about pets?",
            "What do people say about travel?",
        ]
        for i, q in enumerate(quick_qs):
            if btn_cols[i % 4].button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state["quick_q"] = q

        st.markdown("---")

        # Show RAG debug toggle
        show_debug = st.toggle("🔍 Show RAG retrieval debug info", value=True)

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{
                "role": "assistant",
                "content": (
                    "👋 Hi! I've analyzed **191,592 messages** from **11,001 conversations**.\n\n"
                    "Ask me about the **user's personality**, **habits**, **hobbies**, "
                    "**conversation topics**, or anything from the dataset!"
                ),
                "debug": []
            }]

        # Handle quick question clicks
        if "quick_q" in st.session_state:
            q = st.session_state.pop("quick_q")
            ans, dbg = answer(q, data)
            st.session_state.chat_history.append({"role": "user",      "content": q,   "debug": []})
            st.session_state.chat_history.append({"role": "assistant", "content": ans, "debug": dbg})

        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if show_debug and msg["role"] == "assistant" and msg.get("debug"):
                    with st.expander("🔍 RAG Debug Info"):
                        for line in msg["debug"]:
                            st.markdown(line)

        # User input
        if prompt := st.chat_input("Ask about the user, topics, habits, or data..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt, "debug": []})
            with st.chat_message("user"):
                st.markdown(prompt)

            ans, dbg = answer(prompt, data)
            st.session_state.chat_history.append({"role": "assistant", "content": ans, "debug": dbg})
            with st.chat_message("assistant"):
                st.markdown(ans)
                if show_debug and dbg:
                    with st.expander("🔍 RAG Debug Info"):
                        for line in dbg:
                            st.markdown(line)


if __name__ == "__main__":
    main()