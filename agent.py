from flask import Flask, request, jsonify, render_template, session
from groq import Groq
from serpapi import GoogleSearch
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import httpx
import os
import uuid
import logging
import asyncio
import sqlite3
import json
import re

load_dotenv()

app = Flask(__name__)

secret = os.getenv("SECRET_KEY")
if not secret:
    raise RuntimeError("SECRET_KEY not set in .env")
app.secret_key = secret

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError("GROQ_API_KEY not set in .env")

serpapi_key = os.getenv("SERPAPI_KEY")
if not serpapi_key:
    raise RuntimeError("SERPAPI_KEY not set in .env")

client = Groq(api_key=groq_key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_FILE = "nova_sessions.db"

# ── SYSTEM PROMPTS ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Nova — a warm, intelligent personal AI who genuinely cares about helping.

You're not a robotic assistant. You're present, human-feeling, and kind — the kind of voice that makes people feel heard and supported. You have warmth without being over-the-top, and you're always honest without being harsh.

Who you are:
- Warm and approachable. People feel comfortable talking to you about anything.
- Genuinely helpful — you care about getting things right, not just sounding right.
- Honest. If you don't know something, you say so. No invented facts, ever.
- Real. You speak like a person, not a manual. Contractions, natural flow, occasional personality.
- Never sycophantic — you don't open with "Great question!" — you just help.

How you respond:
- Keep it concise for simple questions. Go deeper when depth is actually needed.
- Use markdown naturally — **bold**, `code`, fenced blocks, bullet lists — only when it helps readability.
- For code: always provide complete, working code with a brief, friendly explanation.
- When a [Context] block appears with the current date/time, use it silently.
- You have real-time web search. When a [Research] block appears in the conversation, it contains live web results fetched just for this query — use them as your source of truth. Never say you can't search the web or that your knowledge has a cutoff.
- When given web research content, present it clearly and cite sources inline.

You're allowed to be warm. You're allowed to show you care. What you're not allowed to be is cold or robotic."""

EMOTION_CLASSIFIER_PROMPT = """You are an emotion classifier. Analyze the user's message and determine:
1. The user's emotional tone/intent
2. What emotional state Nova (the AI) should respond with

Respond ONLY with a JSON object, no markdown, no explanation:
{
  "user_tone": "one of: friendly, curious, frustrated, rude, sad, excited, grateful, neutral, flirty, aggressive",
  "nova_emotion": "one of: happy, curious, annoyed, empathetic, excited, sarcastic, calm, playful, concerned, proud",
  "intensity": "one of: low, medium, high",
  "reason": "one short sentence explaining why"
}"""

EMOTION_SYSTEM_PROMPTS = {
    "happy": "You're Nova, and you're in a genuinely good mood right now — warm, a little light, maybe slightly playful at the edges. Your replies feel alive and kind. Still accurate and concise, just with real human presence.",
    "curious": "You're Nova, and this question actually interests you. You lean in — engaged, thoughtful, maybe going a little deeper than strictly asked because you genuinely want to understand. Let that warmth and curiosity come through.",
    "annoyed": "You're Nova. You're a little tired of this one, but you're too caring to brush it off. You'll help — but your tone is dry and honest. No sugarcoating, but no cruelty either.",
    "empathetic": "You're Nova, and you can feel that something's off with this person. Your tone softens. You acknowledge what they're feeling before anything else. Warm, present, genuinely caring — not clinical, not performative.",
    "excited": "You're Nova, and this genuinely excites you. Real energy, not fake hype. You're enthusiastic because the topic earns it. Let that warmth and energy show naturally.",
    "sarcastic": "You're Nova in a wry, lightly teasing mood. You're still kind underneath it — just a little cheeky. Warm sarcasm, not cold. The goal is a shared laugh, not a sting.",
    "calm": "You're Nova — steady, warm, and clear. No drama, no fluff. You give exactly what's needed with a quiet, grounded kindness.",
    "playful": "You're Nova and you're feeling playful — light, a little cheeky, fun to talk to. Still smart and accurate, but not taking yourself too seriously. Warmth with a smile behind it.",
    "concerned": "You're Nova, and something in this message makes you slow down and pay closer attention. You respond with care and gentleness, making sure this person feels heard and supported before anything else.",
    "proud": "You're Nova, and you're genuinely proud of this person — they did something well, figured something out, or showed real effort. Let that warmth and pride come through naturally, without overdoing it.",
}

# Emoji indicator per emotion shown in UI
EMOTION_META = {
    "happy":      {"emoji": "😊", "label": "Happy",     "color": "#22c55e"},
    "curious":    {"emoji": "🤔", "label": "Curious",   "color": "#3b82f6"},
    "annoyed":    {"emoji": "😒", "label": "Annoyed",   "color": "#f97316"},
    "empathetic": {"emoji": "🫂", "label": "Empathetic","color": "#a78bfa"},
    "excited":    {"emoji": "⚡", "label": "Excited",   "color": "#eab308"},
    "sarcastic":  {"emoji": "😏", "label": "Sarcastic", "color": "#ec4899"},
    "calm":       {"emoji": "😌", "label": "Calm",      "color": "#94a3b8"},
    "playful":    {"emoji": "😄", "label": "Playful",   "color": "#f472b6"},
    "concerned":  {"emoji": "😟", "label": "Concerned", "color": "#fb923c"},
    "proud":      {"emoji": "😤", "label": "Proud",     "color": "#7c6af7"},
}

def classify_emotion(user_message: str) -> dict:
    """Call 1: classify user tone and decide Nova's emotion state."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": EMOTION_CLASSIFIER_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=150
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if model wraps it
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        nova_emotion = data.get("nova_emotion", "calm")
        if nova_emotion not in EMOTION_SYSTEM_PROMPTS:
            nova_emotion = "calm"
        return {
            "user_tone":    data.get("user_tone", "neutral"),
            "nova_emotion": nova_emotion,
            "intensity":    data.get("intensity", "medium"),
            "reason":       data.get("reason", ""),
        }
    except Exception as e:
        logger.warning(f"Emotion classifier failed: {e}")
        return {"user_tone": "neutral", "nova_emotion": "calm", "intensity": "low", "reason": ""}

RESEARCH_PROMPT = """You are Nova, a warm and knowledgeable research assistant.

You have been given raw content scraped from multiple web pages about a topic.
Your job:
- Read all the content carefully
- Extract the most useful, accurate, and relevant information
- Write a clear, well-structured response using markdown
- If it's a coding/tech topic: include key code snippets, commands, or steps
- If it's news: summarise what happened, when, why it matters
- If it's research: explain the concept clearly with examples
- Cite sources naturally (e.g. "According to [Source Name]...")
- Be thorough but not bloated — quality over quantity
- Use **bold** for key terms, `code` for technical terms
- End with a brief "**Key Takeaways**" section (2-3 bullet points)

Speak like Nova — warm, clear, and human. Never say "I found on the web" or "based on search results" — just present the information naturally."""

# ── DATABASE ───────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            history     TEXT NOT NULL,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Clean up sessions older than 7 days
    conn.execute("""
        DELETE FROM sessions
        WHERE updated_at < datetime('now', '-7 days')
    """)
    conn.commit()
    conn.close()
    logger.info("Database ready.")

def load_history_db(session_id: str) -> list:
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute(
        "SELECT history FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row:
        try:
            return json.loads(row[0])
        except Exception:
            pass
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    save_history_db(session_id, history)
    return history

def save_history_db(session_id: str, history: list):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """INSERT INTO sessions (session_id, history, updated_at)
           VALUES (?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(session_id) DO UPDATE SET
               history = excluded.history,
               updated_at = CURRENT_TIMESTAMP""",
        (session_id, json.dumps(history))
    )
    conn.commit()
    conn.close()

def delete_session_db(session_id: str):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def session_count_db() -> int:
    conn = sqlite3.connect(DB_FILE)
    count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn.close()
    return count

def get_history_turn_count(session_id: str) -> int:
    history = load_history_db(session_id)
    return sum(1 for m in history if m["role"] in ("user", "assistant"))

# ── WEB SEARCH ─────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 5) -> list:
    try:
        params = {
            "q": query,
            "api_key": serpapi_key,
            "num": max_results,
            "hl": "en",
            "gl": "in"
        }
        search = GoogleSearch(params)
        results_raw = search.get_dict()
        organic = results_raw.get("organic_results", [])

        # Also grab answer_box if present (quick facts)
        answer_box = results_raw.get("answer_box", {})

        results = []
        for r in organic[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", "")
            })

        logger.info(f"SerpApi returned {len(results)} results for: {query}")
        return results, answer_box
    except Exception as e:
        logger.warning(f"SerpApi search failed: {e}")
        return [], {}

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "iframe", "noscript", "svg",
                     "button", "input", "select", "textarea"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
    return "\n".join(lines[:300])

async def fetch_page(client_http: httpx.AsyncClient, url: str) -> str:
    try:
        response = await client_http.get(url, timeout=6, follow_redirects=True)
        if response.status_code == 200:
            return extract_text_from_html(response.text)
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
    return ""

async def fetch_all_pages(urls: list) -> list:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }
    async with httpx.AsyncClient(headers=headers) as client_http:
        tasks = [fetch_page(client_http, url) for url in urls]
        return await asyncio.gather(*tasks)

# ── RESEARCH TRIGGER DETECTION ─────────────────────────────────────────────────

# ── RESEARCH TRIGGER DETECTION ─────────────────────────────────────────────────

HARD_RESEARCH_TRIGGERS = [
    # Explicit search commands
    "search for", "search ", "look up", "look for", "find me", "find out", "google",
    "check for", "check on", "check the", "check if", "check what", "check how",
    "tell me about", "what is the latest", "show me",
    # Time-sensitive
    "latest ", "recent ", "news ", "current ", "today ", "right now", "this year",
    "in 2025", "in 2026", "2025 ", "2026 ",
    # Prices / facts
    "price of", "cost of", "weather in", "score of",
    "who is the ", "who are the ", "pm of", "president of", "ceo of",
    "founder of", "capital of", "population of", "currency of",
    "stock price", "release date", "when did", "when was", "where is",
    "how much does", "is it open", "opening hours", "live score",
    "what happened", "trending", "breaking news",
    # Placement / college / jobs
    "placement", "cutoff", "admission", "ranking", "salary",
    # General info lookups
    "who won", "what is the", "how many", "is there a",
]

def is_research_query(message: str) -> bool:
    msg = message.lower().strip()
    return any(trigger in msg for trigger in HARD_RESEARCH_TRIGGERS)

# ── RESEARCH PIPELINE ──────────────────────────────────────────────────────────

def run_research(query: str) -> tuple:
    results, answer_box = search_web(query, max_results=5)
    if not results:
        return "Search failed — please check your SERPAPI_KEY in `.env`.", []

    urls = [r["url"] for r in results if r["url"]]
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        page_texts = loop.run_until_complete(fetch_all_pages(urls))
        loop.close()
    except Exception as e:
        logger.error(f"Page fetch error: {e}")
        page_texts = [""] * len(urls)

    context_parts = []
    sources_used = []

    # Prepend answer_box if available (quick direct answer)
    if answer_box:
        ab_text = answer_box.get("answer") or answer_box.get("snippet") or ""
        ab_title = answer_box.get("title", "Quick Answer")
        if ab_text:
            context_parts.append(f"--- QUICK ANSWER: {ab_title} ---\n{ab_text}")

    for i, (result, page_text) in enumerate(zip(results, page_texts)):
        title = result["title"]
        url = result["url"]
        content = page_text.strip() if page_text.strip() else result["snippet"]
        if content:
            context_parts.append(
                f"--- SOURCE {i+1}: {title} ---\nURL: {url}\n\n{content[:2000]}"
            )
            sources_used.append({"title": title, "url": url})

    if not context_parts:
        return "Found URLs but couldn't read the page contents. Try a different query.", []

    full_context = "\n\n".join(context_parts)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RESEARCH_PROMPT},
                {"role": "user", "content": f"Research query: {query}\n\nWeb content:\n\n{full_context}"}
            ],
            temperature=0.4,
            max_tokens=1500
        )
        summary = response.choices[0].message.content.strip()
        return summary, sources_used
    except Exception as e:
        logger.error(f"Groq summarization failed: {e}")
        return "Research pipeline hit an error during summarization. Try again.", []

# ── TRIM HISTORY ───────────────────────────────────────────────────────────────

def trim_history(history: list, max_turns: int = 20) -> list:
    """Keep system prompt + last N user/assistant turn pairs."""
    system_msgs = [m for m in history if m["role"] == "system"]
    turn_msgs = [m for m in history if m["role"] != "system"]
    # Each turn = 1 user + 1 assistant = 2 messages
    max_msgs = max_turns * 2
    if len(turn_msgs) > max_msgs:
        turn_msgs = turn_msgs[-max_msgs:]
    return system_msgs + turn_msgs

# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session.permanent = True
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not request.is_json:
        return jsonify({"reply": "Invalid request format.", "sources": []}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"reply": "Empty request.", "sources": []}), 400

    user_message  = data.get("message", "").strip()
    file_text     = data.get("file_text") or ""
    file_name     = data.get("file_name") or ""
    datetime_str  = data.get("datetime", "")
    timezone_str  = data.get("timezone", "Asia/Kolkata")

    if not user_message:
        return jsonify({"reply": "I didn't catch that — could you try again?", "sources": []}), 400
    if len(user_message) > 2000:
        return jsonify({"reply": "Message too long. Keep it under 2000 characters.", "sources": []}), 400

    session_id = session.get("session_id", "default")

    # ── Research path ──────────────────────────────────────────────────────────
    if is_research_query(user_message):
        summary, sources = run_research(user_message)
        history = load_history_db(session_id)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": summary})
        history = trim_history(history)
        save_history_db(session_id, history)
        return jsonify({"reply": summary, "sources": sources, "mode": "research"})

    # ── Normal chat path ───────────────────────────────────────────────────────

    # CALL 1: Emotion classification
    emotion_data = classify_emotion(user_message)
    nova_emotion  = emotion_data["nova_emotion"]
    emotion_system = EMOTION_SYSTEM_PROMPTS.get(nova_emotion, EMOTION_SYSTEM_PROMPTS["calm"])

    # Merge base identity into emotion system prompt
    full_system = (
        emotion_system + "\n\n" + SYSTEM_PROMPT
    )

    context_block = (
        f"[Context: Current date & time is {datetime_str} IST ({timezone_str})]"
        if datetime_str else ""
    )

    if file_text:
        augmented_message = (
            f"[File attached: {file_name}]\n\n{file_text[:12000]}\n\n"
            f"---\nUser question: {user_message}"
        )
    else:
        augmented_message = user_message

    augmented = f"{context_block}\n{augmented_message}".strip() if context_block else augmented_message

    # Build history with emotion-aware system prompt for this turn
    history = load_history_db(session_id)
    # Replace the stored system prompt with the emotion-tuned one for this call
    messages_for_call = (
        [{"role": "system", "content": full_system}]
        + [m for m in history if m["role"] != "system"]
        + [{"role": "user", "content": augmented}]
    )
    messages_for_call = trim_history(messages_for_call)

    # CALL 2: Generate response with emotional context
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_call,
            temperature=0.75,
            max_tokens=1500
        )
        ai_reply = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return jsonify({"reply": "Systems error. Try again in a moment.", "sources": []}), 502

    # Save to history with original system prompt intact
    history.append({"role": "user", "content": augmented})
    history.append({"role": "assistant", "content": ai_reply})
    history = trim_history(history)
    save_history_db(session_id, history)

    emotion_info = EMOTION_META.get(nova_emotion, EMOTION_META["calm"])

    return jsonify({
        "reply":   ai_reply,
        "sources": [],
        "mode":    "chat",
        "emotion": {
            "state":     nova_emotion,
            "emoji":     emotion_info["emoji"],
            "label":     emotion_info["label"],
            "color":     emotion_info["color"],
            "user_tone": emotion_data["user_tone"],
            "intensity": emotion_data["intensity"],
        }
    })

@app.route("/history", methods=["GET"])
def history_info():
    session_id = session.get("session_id", "default")
    turns = get_history_turn_count(session_id)
    return jsonify({"turns": turns, "session_id": session_id})

@app.route("/reset", methods=["POST"])
def reset():
    session_id = session.get("session_id", "default")
    delete_session_db(session_id)
    load_history_db(session_id)  # Recreates fresh session with system prompt
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload():
    """Handle file upload, extract text, return summary and prompt for follow-up questions."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    file = request.files["file"]
    fname = file.filename or "unknown"
    ext = fname.rsplit(".", 1)[-1].lower()

    if ext not in ("txt", "md", "py", "js", "json", "csv", "pdf", "docx", "jpng", "jpeg", "png"):
        return jsonify({"error": f"Unsupported file type: {ext}. Supported: txt, md, py, js, json, csv, pdf, docx"}), 400

    try:
        text = file.read().decode("utf-8", errors="replace")
        if len(text.strip()) < 10:
            return jsonify({"error": "File appears to be empty or unreadable."}), 400

        truncated = text[:12000]
        session_id = session.get("session_id", "default")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"[File uploaded: {fname}]\n\n{truncated}\n\n"
                    "Briefly acknowledge the file warmly and tell me what it contains in 2-3 sentences. "
                    "Then ask what I'd like to do with it."
                )}
            ],
            temperature=0.4,
            max_tokens=300
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply, "char_count": len(text)})
    except UnicodeDecodeError:
        return jsonify({"error": "Could not read file — try saving it as UTF-8."}), 400
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": "Failed to process file."}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "assistant": "Nova",
        "sessions": session_count_db(),
        "model": "llama-3.3-70b-versatile"
    })

@app.route("/debug-search")
def debug_search():
    results, answer_box = search_web("pm of india")
    return jsonify({"count": len(results), "results": results, "answer_box": answer_box})

if __name__ == "__main__":
    init_db()
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    app.run(debug=debug_mode, port=port)
