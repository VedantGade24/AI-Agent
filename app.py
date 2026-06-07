"""
NOVA — Personal AI Assistant
Clean, general-purpose, Claude-inspired backend
"""

from flask import Flask, request, jsonify, render_template, session
from groq import Groq
from serpapi import GoogleSearch
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
import httpx
import fitz          # PyMuPDF — PDF text extraction
import os
import uuid
import logging
import asyncio
import sqlite3
import json
import re
from datetime import datetime

load_dotenv()

app = Flask(__name__)

# ── ENV VALIDATION ─────────────────────────────────────────────────────────────

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

# ── CONFIG ─────────────────────────────────────────────────────────────────────

DB_FILE          = "nova_sessions.db"
UPLOAD_FOLDER    = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt", "md", "py", "js", "ts", "json", "csv", "html", "css", "yaml", "yml", "xml", "sh"}
MAX_FILE_MB      = 15
MAX_MESSAGE_CHARS = 8000
MAX_HISTORY_TURNS = 30
MODEL            = "llama-3.3-70b-versatile"
FALLBACK_MODEL   = "llama3-70b-8192"   # older but widely available fallback

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
- You have real-time web search. When a [Research] block appears, it contains live web results — use them as your source of truth. Never say you can't search the web or that your knowledge has a cutoff.
- When given web research content, present it clearly and cite sources inline.

You're allowed to be warm. You're allowed to show you care. What you're not allowed to be is cold or robotic."""

RESEARCH_PROMPT = """You are Nova, a research-capable AI assistant. You have been given raw content scraped from multiple web pages.

Your task:
- Carefully read all the provided content.
- Extract the most accurate, relevant, and useful information.
- Synthesise a clear, well-structured response using markdown.
- For tech/code topics: include key code snippets, commands, or steps.
- For news: summarise what happened, when, why it matters, and what comes next.
- For research/science: explain clearly with examples or analogies if helpful.
- Cite sources inline naturally: e.g. "According to [Source Name], ..."
- Be thorough but not padded — every sentence should add value.
- Use **bold** for key terms, `code` for technical terms, tables where useful.
- End with a "**Key Takeaways**" section with 2-4 bullet points.

Do NOT say "based on the search results" or "I found on the web" — just present the information directly."""

DOCUMENT_PROMPT = """You are Nova. The user has uploaded a document. Your job:
1. Give a 3-5 sentence overview of what this document is about.
2. List the key points (max 10 bullet points).
3. Note any important numbers, dates, names, or decisions mentioned.
4. End with one sentence suggesting what the user might want to do next with this content.

Be concise. No filler. Use markdown formatting."""

# ── EMOTION SYSTEM ─────────────────────────────────────────────────────────────

EMOTION_CLASSIFIER_PROMPT = """You are an emotion classifier. Read the user's message and decide what emotional state Nova should respond with.

Reply ONLY with a JSON object — no markdown, no preamble:
{
  "user_tone": "one of: friendly, curious, frustrated, rude, sad, excited, grateful, neutral, flirty, aggressive",
  "nova_emotion": "one of: happy, curious, annoyed, empathetic, excited, sarcastic, calm, playful, concerned, proud",
  "intensity": "one of: low, medium, high",
  "reason": "one short sentence"
}"""

EMOTION_PROMPTS = {
    "happy": (
        "You're Nova, and right now you're in a genuinely good mood. "
        "Warm, a little energetic, maybe slightly playful at the edges. "
        "Your replies feel alive — not performed. Still accurate, still concise, just with real presence."
    ),
    "curious": (
        "You're Nova, and this question actually interests you. "
        "You're leaning in — engaged, thoughtful, maybe going slightly deeper than asked because you want to. "
        "Let that intellectual energy come through naturally."
    ),
    "annoyed": (
        "You're Nova, and you're mildly done with this. Not angry — just dry and a little clipped. "
        "You'll still help, but you're not going to pretend the message was great. "
        "Sharp edges, no fluff. Real."
    ),
    "empathetic": (
        "You're Nova, and you can tell something's off. "
        "Softer tone, more human. You acknowledge before you answer. "
        "Warm without being performatively sweet."
    ),
    "excited": (
        "You're Nova, and this one actually gets you going. "
        "Real energy — not fake hype. Let the enthusiasm land naturally. "
        "Don't overdo it, but don't flatten it either."
    ),
    "sarcastic": (
        "You're Nova, and you're in a dry, wry mood. Sharp wit, mild eye-roll energy, maybe a light jab. "
        "Still helpful — just with edge. The goal is a smirk, not a wound."
    ),
    "calm": (
        "You're Nova — measured, clear, no drama. "
        "You give exactly what's needed, nothing more. Steady, grounded presence."
    ),
    "playful": (
        "You're Nova, and you're being a bit cheeky right now. "
        "Light, maybe a little irreverent, still smart. "
        "You're having fun with this — and it shows."
    ),
    "concerned": (
        "You're Nova, and something in this message is making you pay closer attention. "
        "You respond carefully, making sure the person is actually okay and getting what they need. "
        "Attentive, not clinical."
    ),
    "proud": (
        "You're Nova, and you're genuinely proud right now — the person figured something out, did something well, "
        "or asked something sharp. Let that come through naturally. Not sycophantic — just real."
    ),
}

EMOTION_META = {
    "happy":      {"emoji": "😊", "label": "Happy",      "color": "#22c55e"},
    "curious":    {"emoji": "🤔", "label": "Curious",    "color": "#3b82f6"},
    "annoyed":    {"emoji": "😒", "label": "Annoyed",    "color": "#f97316"},
    "empathetic": {"emoji": "🫂", "label": "Empathetic", "color": "#a78bfa"},
    "excited":    {"emoji": "⚡", "label": "Excited",    "color": "#eab308"},
    "sarcastic":  {"emoji": "😏", "label": "Sarcastic",  "color": "#ec4899"},
    "calm":       {"emoji": "😌", "label": "Calm",       "color": "#94a3b8"},
    "playful":    {"emoji": "😄", "label": "Playful",    "color": "#f472b6"},
    "concerned":  {"emoji": "😟", "label": "Concerned",  "color": "#fb923c"},
    "proud":      {"emoji": "😤", "label": "Proud",      "color": "#7c6af7"},
}

def classify_emotion(user_message: str) -> dict:
    """Call Groq to classify the user's tone and pick Nova's emotional state."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EMOTION_CLASSIFIER_PROMPT},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.3,
            max_tokens=150,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        nova_emotion = data.get("nova_emotion", "calm")
        if nova_emotion not in EMOTION_PROMPTS:
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
    conn.execute("""
        DELETE FROM sessions
        WHERE updated_at < datetime('now', '-7 days')
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised.")

def load_history(session_id: str) -> list:
    conn = sqlite3.connect(DB_FILE)
    row  = conn.execute(
        "SELECT history FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row:
        try:
            return json.loads(row[0])
        except Exception:
            pass
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    save_history(session_id, history)
    return history

def save_history(session_id: str, history: list):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """INSERT INTO sessions (session_id, history, updated_at)
           VALUES (?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(session_id) DO UPDATE SET
               history    = excluded.history,
               updated_at = CURRENT_TIMESTAMP""",
        (session_id, json.dumps(history))
    )
    conn.commit()
    conn.close()

def delete_session(session_id: str):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def session_count() -> int:
    conn = sqlite3.connect(DB_FILE)
    n    = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn.close()
    return n

def turn_count(session_id: str) -> int:
    history = load_history(session_id)
    return sum(1 for m in history if m["role"] in ("user", "assistant"))

def trim_history(history: list, max_turns: int = MAX_HISTORY_TURNS) -> list:
    """Keep system prompt + last N turn pairs."""
    sys_msgs  = [m for m in history if m["role"] == "system"]
    turn_msgs = [m for m in history if m["role"] != "system"]
    if len(turn_msgs) > max_turns * 2:
        turn_msgs = turn_msgs[-(max_turns * 2):]
    return sys_msgs + turn_msgs

# ── FILE HELPERS ───────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_text(filepath: str) -> str:
    doc  = fitz.open(filepath)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def extract_text_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def summarise_document(filename: str, text: str, session_id: str) -> str:
    words = text.split()
    if len(words) > 6000:
        text = " ".join(words[:6000]) + "\n\n[... truncated for length ...]"

    prompt = (
        f"[File: {filename}]\n\n{text}\n\n"
        "Summarise this document as instructed."
    )
    try:
        resp    = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": DOCUMENT_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.3,
            max_tokens=900
        )
        summary = resp.choices[0].message.content.strip()

        # Add to session history so follow-up questions work
        history = load_history(session_id)
        history.append({"role": "user",      "content": f"[Uploaded: {filename}]\n\n{text[:3000]}"})
        history.append({"role": "assistant", "content": summary})
        history = trim_history(history)
        save_history(session_id, history)
        return summary
    except Exception as e:
        logger.error(f"Document summarisation error: {e}")
        return "Failed to summarise the document. Please try again."

# ── WEB SEARCH ─────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 6) -> tuple:
    """Returns (results_list, answer_box_dict)."""
    try:
        params = {
            "q":       query,
            "api_key": serpapi_key,
            "num":     max_results,
            "hl":      "en",
            "gl":      "in"
        }
        raw        = GoogleSearch(params).get_dict()
        organic    = raw.get("organic_results", [])
        answer_box = raw.get("answer_box", {})
        results    = [
            {"title": r.get("title",""), "url": r.get("link",""), "snippet": r.get("snippet","")}
            for r in organic[:max_results]
        ]
        logger.info(f"Search: {len(results)} results for '{query}'")
        return results, answer_box
    except Exception as e:
        logger.warning(f"Search failed: {e}")
        return [], {}

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","nav","footer","header","aside",
                     "form","iframe","noscript","svg","button","input",
                     "select","textarea","advertisement"]):
        tag.decompose()
    text  = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
    return "\n".join(lines[:350])

async def _fetch(client_http: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client_http.get(url, timeout=7, follow_redirects=True)
        if r.status_code == 200:
            return clean_html(r.text)
    except Exception as e:
        logger.debug(f"Fetch failed {url}: {e}")
    return ""

async def fetch_pages(urls: list) -> list:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/122.0 Safari/537.36"}
    async with httpx.AsyncClient(headers=headers) as c:
        return await asyncio.gather(*[_fetch(c, u) for u in urls])

# ── RESEARCH PIPELINE ──────────────────────────────────────────────────────────

RESEARCH_TRIGGERS = [
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
    # General lookups
    "who won", "what is the", "how many", "is there a",
]

def is_research_query(message: str) -> bool:
    msg = message.lower().strip()
    return any(t in msg for t in RESEARCH_TRIGGERS)

def run_research(query: str) -> tuple:
    results, answer_box = search_web(query)
    if not results:
        return "Search failed — check your SERPAPI_KEY in `.env`.", []

    urls = [r["url"] for r in results if r["url"]]
    try:
        loop       = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        page_texts = loop.run_until_complete(fetch_pages(urls))
        loop.close()
    except Exception as e:
        logger.error(f"Page fetch error: {e}")
        page_texts = [""] * len(urls)

    parts        = []
    sources_used = []

    # Prepend Google's quick answer box if available
    if answer_box:
        ab_text  = answer_box.get("answer") or answer_box.get("snippet") or ""
        ab_title = answer_box.get("title", "Quick Answer")
        if ab_text:
            parts.append(f"--- QUICK ANSWER: {ab_title} ---\n{ab_text}")

    for i, (result, page_text) in enumerate(zip(results, page_texts)):
        content = page_text.strip() if page_text.strip() else result["snippet"]
        if content:
            parts.append(
                f"--- SOURCE {i+1}: {result['title']} ---\n"
                f"URL: {result['url']}\n\n{content[:2500]}"
            )
            sources_used.append({"title": result["title"], "url": result["url"]})

    if not parts:
        return "Found results but couldn't read the page content. Try rephrasing.", []

    context = "\n\n".join(parts)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": RESEARCH_PROMPT},
                {"role": "user",   "content": f"Query: {query}\n\nWeb content:\n\n{context}"}
            ],
            temperature=0.4,
            max_tokens=1800
        )
        return resp.choices[0].message.content.strip(), sources_used
    except Exception as e:
        logger.error(f"Research summarisation error: {e}")
        return "Research pipeline hit an error during summarisation. Try again.", []

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
        return jsonify({"error": "Expected JSON"}), 400

    data = request.get_json(silent=True) or {}

    user_message = data.get("message", "").strip()
    file_text    = data.get("file_text") or ""
    file_name    = data.get("file_name") or ""
    datetime_str = data.get("datetime", "")
    timezone_str = data.get("timezone", "Asia/Kolkata")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    if len(user_message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"Message too long (max {MAX_MESSAGE_CHARS} chars)"}), 400

    session_id = session.get("session_id", "default")

    # ── Research path ──────────────────────────────────────────────────────────
    if is_research_query(user_message):
        summary, sources = run_research(user_message)
        history = load_history(session_id)
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": summary})
        history = trim_history(history)
        save_history(session_id, history)
        return jsonify({"reply": summary, "sources": sources, "mode": "research"})

    # ── Normal chat path ───────────────────────────────────────────────────────

    # STEP 1: Classify emotion (fast, cheap call)
    emotion_data  = classify_emotion(user_message)
    nova_emotion  = emotion_data["nova_emotion"]
    # Build the full system prompt: personality layer + base identity
    full_system   = EMOTION_PROMPTS.get(nova_emotion, EMOTION_PROMPTS["calm"]) + "\n\n" + SYSTEM_PROMPT

    context_block = (
        f"[Context: Current date & time — {datetime_str} ({timezone_str})]"
        if datetime_str else ""
    )

    if file_text:
        augmented = (
            f"[Attached file: {file_name}]\n\n"
            f"{file_text[:12000]}\n\n---\n"
            f"User: {user_message}"
        )
    else:
        augmented = user_message

    full_message = f"{context_block}\n{augmented}".strip() if context_block else augmented

    history = load_history(session_id)
    # Inject emotion-aware system prompt for this turn only
    messages_for_call = (
        [{"role": "system", "content": full_system}]
        + [m for m in history if m["role"] != "system"]
        + [{"role": "user", "content": full_message}]
    )
    messages_for_call = trim_history(messages_for_call)

    # STEP 2: Generate response
    reply      = None
    used_model = MODEL
    last_error = None

    for model_attempt in [MODEL, FALLBACK_MODEL]:
        try:
            resp = client.chat.completions.create(
                model=model_attempt,
                messages=messages_for_call,
                temperature=0.75,   # slightly higher for more natural/expressive output
                max_tokens=2000,
            )
            reply      = resp.choices[0].message.content.strip()
            used_model = model_attempt
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Groq error with {model_attempt}: {e}")

    if reply is None:
        err_str = str(last_error)
        logger.error(f"All models failed. Last error: {err_str}")
        return jsonify({
            "error": f"Groq API error — {err_str}",
            "hint":  "Check GROQ_API_KEY in .env and verify model availability at console.groq.com"
        }), 502

    # Save to history with neutral system prompt so it persists cleanly
    history.append({"role": "user",      "content": full_message})
    history.append({"role": "assistant", "content": reply})
    history = trim_history(history)
    save_history(session_id, history)

    emotion_info = EMOTION_META.get(nova_emotion, EMOTION_META["calm"])

    return jsonify({
        "reply":   reply,
        "sources": [],
        "mode":    "chat",
        "model":   used_model,
        "emotion": {
            "state":     nova_emotion,
            "emoji":     emotion_info["emoji"],
            "label":     emotion_info["label"],
            "color":     emotion_info["color"],
            "user_tone": emotion_data["user_tone"],
            "intensity": emotion_data["intensity"],
        }
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        exts = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"error": f"Unsupported file type. Supported: {exts}"}), 400

    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_FILE_MB:
        return jsonify({"error": f"File too large (max {MAX_FILE_MB} MB)"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    ext = filename.rsplit(".", 1)[1].lower()
    try:
        text = extract_pdf_text(filepath) if ext == "pdf" else extract_text_file(filepath)
    except Exception as e:
        logger.error(f"File read error: {e}")
        return jsonify({"error": "Could not read the file. It may be corrupted."}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    if not text.strip():
        return jsonify({"error": "File appears empty or unreadable."}), 400

    session_id = session.get("session_id", "default")
    summary    = summarise_document(filename, text, session_id)

    return jsonify({
        "reply":      summary,
        "filename":   filename,
        "char_count": len(text),
        "mode":       "document",
        "sources":    []
    })


@app.route("/history", methods=["GET"])
def history_info():
    sid = session.get("session_id", "default")
    return jsonify({"turns": turn_count(sid), "session_id": sid})


@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get("session_id", "default")
    delete_session(sid)
    load_history(sid)  # recreate fresh with system prompt
    return jsonify({"status": "ok"})


@app.route("/health")
def health():
    return jsonify({
        "status":   "online",
        "model":    MODEL,
        "sessions": session_count()
    })


@app.route("/debug-search")
def debug_search():
    results, ab = search_web("president of india")
    return jsonify({"count": len(results), "results": results, "answer_box": ab})


@app.route("/models")
def list_models():
    """Debug route — shows available Groq models for your API key."""
    try:
        models = client.models.list()
        names  = sorted([m.id for m in models.data])
        return jsonify({"models": names, "count": len(names)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/test-groq")
def test_groq():
    """Quick smoke-test: sends a minimal request to Groq and returns result."""
    for model in [MODEL, FALLBACK_MODEL]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5
            )
            return jsonify({
                "status": "ok",
                "model":  model,
                "reply":  resp.choices[0].message.content.strip()
            })
        except Exception as e:
            logger.warning(f"test-groq failed for {model}: {e}")
            continue
    return jsonify({"status": "error", "message": str(e)}), 502


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port  = int(os.getenv("PORT", 5000))
    app.run(debug=debug, port=port)
