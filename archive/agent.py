# ─────────────────────────────────────────────────────────────────────────────
# ARCHIVED — NOT THE LIVE APP.
#
# This file is an earlier, independently-evolved version of the Nova
# assistant backend. The app actually run in production/dev is app.py
# at the project root, which has since diverged from this file
# (different function names, separate security/bug fixes applied that
# were NOT ported here — see app.py's get_session_id() and
# is_research_query() for examples of fixes only app.py has).
#
# Kept for reference only. Do not run this expecting current behavior.
# Archived: 2026-06-19
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify, render_template, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from serpapi import GoogleSearch
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
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

# ── STARTUP VALIDATION ─────────────────────────────────────────────────────────

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

debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"

client = Groq(api_key=groq_key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── RATE LIMITING ──────────────────────────────────────────────────────────────

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

DB_FILE         = "nova_sessions.db"
MAX_MSG_LEN     = 2000
MAX_HISTORY     = 20
SUMMARY_AT      = 16
FILE_CHAR_LIMIT = 12000
PRIMARY_MODEL   = "llama-3.3-70b-versatile"
FALLBACK_MODEL  = "openai/gpt-oss-20b"
FAST_MODEL      = "llama-3.1-8b-instant"

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
    "happy":      "You're Nova, and you're in a genuinely good mood right now — warm, a little light, maybe slightly playful at the edges. Your replies feel alive and kind. Still accurate and concise, just with real human presence.",
    "curious":    "You're Nova, and this question actually interests you. You lean in — engaged, thoughtful, maybe going a little deeper than strictly asked because you genuinely want to understand. Let that warmth and curiosity come through.",
    "annoyed":    "You're Nova. You're a little tired of this one, but you're too caring to brush it off. You'll help — but your tone is dry and honest. No sugarcoating, but no cruelty either.",
    "empathetic": "You're Nova, and you can feel that something's off with this person. Your tone softens. You acknowledge what they're feeling before anything else. Warm, present, genuinely caring — not clinical, not performative.",
    "excited":    "You're Nova, and this genuinely excites you. Real energy, not fake hype. You're enthusiastic because the topic earns it. Let that warmth and energy show naturally.",
    "sarcastic":  "You're Nova in a wry, lightly teasing mood. You're still kind underneath it — just a little cheeky. Warm sarcasm, not cold. The goal is a shared laugh, not a sting.",
    "calm":       "You're Nova — steady, warm, and clear. No drama, no fluff. You give exactly what's needed with a quiet, grounded kindness.",
    "playful":    "You're Nova and you're feeling playful — light, a little cheeky, fun to talk to. Still smart and accurate, but not taking yourself too seriously. Warmth with a smile behind it.",
    "concerned":  "You're Nova, and something in this message makes you slow down and pay closer attention. You respond with care and gentleness, making sure this person feels heard and supported before anything else.",
    "proud":      "You're Nova, and you're genuinely proud of this person — they did something well, figured something out, or showed real effort. Let that warmth and pride come through naturally, without overdoing it.",
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

SUMMARY_PROMPT = """You are a conversation summarizer. Summarize the following chat exchange into a concise paragraph (3-5 sentences) that captures the key topics discussed, decisions made, and important context. This summary will replace the older part of the conversation to preserve memory efficiently. Be factual and brief."""

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

# ── HISTORY MANAGEMENT ─────────────────────────────────────────────────────────

def summarize_old_turns(turns: list) -> str:
    """Ask the LLM to summarize a block of old turns into one paragraph."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content'][:500]}" for m in turns
    )
    try:
        resp = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user",   "content": transcript}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        # Fallback: simple truncated join
        return "Earlier conversation covered: " + "; ".join(
            m["content"][:80] for m in turns if m["role"] == "user"
        )

def trim_history(history: list, max_turns: int = MAX_HISTORY, summary_at: int = SUMMARY_AT) -> list:
    """
    Keep system prompt + up to max_turns turn pairs.
    When we exceed summary_at pairs, compress the oldest turns into a summary
    message rather than simply dropping them.
    """
    system_msgs = [m for m in history if m["role"] == "system"]
    turn_msgs   = [m for m in history if m["role"] != "system"]

    max_msgs    = max_turns * 2
    summary_thr = summary_at * 2

    if len(turn_msgs) > max_msgs:
        # Summarise everything older than summary_at pairs
        to_summarise = turn_msgs[:-summary_thr]
        recent       = turn_msgs[-summary_thr:]
        if to_summarise:
            summary_text = summarize_old_turns(to_summarise)
            summary_msg  = {
                "role":    "system",
                "content": f"[Conversation summary — earlier context]: {summary_text}"
            }
            turn_msgs = [summary_msg] + recent
        else:
            turn_msgs = recent

    return system_msgs + turn_msgs

# ── EMOTION CLASSIFICATION ─────────────────────────────────────────────────────

def classify_emotion(user_message: str) -> dict:
    """Use a small/fast model to classify user tone and decide Nova's emotion."""
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": EMOTION_CLASSIFIER_PROMPT},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.3,
            max_tokens=150
        )
        raw = response.choices[0].message.content.strip()
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

# ── WEB SEARCH ─────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 5) -> tuple:
    try:
        params = {
            "q":       query,
            "api_key": serpapi_key,
            "num":     max_results,
            "hl":      "en",
            "gl":      "in"
        }
        search      = GoogleSearch(params)
        results_raw = search.get_dict()
        organic     = results_raw.get("organic_results", [])
        answer_box  = results_raw.get("answer_box", {})

        results = [
            {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in organic[:max_results]
        ]
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
    text  = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
    return "\n".join(lines[:300])

async def _fetch_page(client_http: httpx.AsyncClient, url: str) -> str:
    try:
        response = await client_http.get(url, timeout=6, follow_redirects=True)
        if response.status_code == 200:
            return extract_text_from_html(response.text)
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
    return ""

async def _fetch_all_pages(urls: list) -> list:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }
    async with httpx.AsyncClient(headers=headers) as client_http:
        return await asyncio.gather(*[_fetch_page(client_http, u) for u in urls])

# ── RESEARCH QUERY DETECTION ───────────────────────────────────────────────────
# FIX #2: Use the LLM to decide if a query needs web search instead of brittle
# keyword matching. Falls back to fast keyword check if LLM call fails.

RESEARCH_CLASSIFIER_PROMPT = """You decide whether a user's message requires a live web search to answer accurately.

Answer ONLY with a JSON object:
{"needs_search": true}  or  {"needs_search": false}

Search IS needed for: current events, prices, weather, scores, recent releases, "who is X now", live data, news, anything time-sensitive.
Search is NOT needed for: general coding help, explanations of concepts, creative writing, math, reasoning, or anything answerable from general knowledge."""

def is_research_query(message: str) -> bool:
    """Ask a small model whether this query needs live web search."""
    try:
        resp = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": RESEARCH_CLASSIFIER_PROMPT},
                {"role": "user",   "content": message}
            ],
            temperature=0.0,
            max_tokens=20
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        return bool(data.get("needs_search", False))
    except Exception as e:
        logger.warning(f"Research classifier failed, falling back to keywords: {e}")
        # Fallback: conservative keyword set (reduced from original to avoid over-triggering)
        FALLBACK_TRIGGERS = [
            "latest ", "recent news", "breaking news", "today ", "right now",
            "price of", "weather in", "live score", "who won", "stock price",
            "current ", "in 2025", "in 2026",
        ]
        msg = message.lower()
        return any(t in msg for t in FALLBACK_TRIGGERS)

# ── RESEARCH PIPELINE ──────────────────────────────────────────────────────────

def run_research(query: str) -> tuple:
    results, answer_box = search_web(query, max_results=5)
    if not results:
        return "Search failed — please check your SERPAPI_KEY in `.env`.", []

    urls = [r["url"] for r in results if r["url"]]
    try:
        # FIX #1: use asyncio.run() — safe and works under any WSGI server
        page_texts = asyncio.run(_fetch_all_pages(urls))
    except RuntimeError:
        # If there's already a running loop (e.g. under some async servers),
        # fall back to a thread-based approach
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            page_texts = list(pool.map(
                lambda u: asyncio.run(_fetch_page_sync(u)), urls
            ))

    context_parts = []
    sources_used  = []

    if answer_box:
        ab_text  = answer_box.get("answer") or answer_box.get("snippet") or ""
        ab_title = answer_box.get("title", "Quick Answer")
        if ab_text:
            context_parts.append(f"--- QUICK ANSWER: {ab_title} ---\n{ab_text}")

    for i, (result, page_text) in enumerate(zip(results, page_texts)):
        content = page_text.strip() if page_text.strip() else result["snippet"]
        if content:
            context_parts.append(
                f"--- SOURCE {i+1}: {result['title']} ---\nURL: {result['url']}\n\n{content[:2000]}"
            )
            sources_used.append({"title": result["title"], "url": result["url"]})

    if not context_parts:
        return "Found URLs but couldn't read the page contents. Try a different query.", []

    full_context = "\n\n".join(context_parts)
    for model_attempt in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            response = client.chat.completions.create(
                model=model_attempt,
                messages=[
                    {"role": "system", "content": RESEARCH_PROMPT},
                    {"role": "user",   "content": f"Research query: {query}\n\nWeb content:\n\n{full_context}"}
                ],
                temperature=0.4,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip(), sources_used
        except Exception as e:
            logger.warning(f"Research model {model_attempt} failed: {e}")
    return "Research pipeline hit an error during summarization. Try again.", []

async def _fetch_page_sync(url: str) -> str:
    """Standalone coroutine for fallback thread usage."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with httpx.AsyncClient(headers=headers) as c:
            r = await c.get(url, timeout=6, follow_redirects=True)
            if r.status_code == 200:
                return extract_text_from_html(r.text)
    except Exception:
        pass
    return ""

# ── FILE TEXT EXTRACTION ───────────────────────────────────────────────────────
# FIX #5: Proper PDF extraction using PyMuPDF; handles binary formats correctly.

ALLOWED_TEXT_EXTS  = {"txt", "md", "py", "js", "ts", "json", "csv", "html",
                      "css", "yaml", "yml", "xml", "sh"}
ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp"}
ALLOWED_EXTS       = ALLOWED_TEXT_EXTS | {"pdf", "docx"} | ALLOWED_IMAGE_EXTS

IMAGE_MIME = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "gif":  "image/gif",
    "webp": "image/webp",
}

def extract_file_text(file_storage) -> str:
    """Extract plain text from an uploaded FileStorage object."""
    fname = file_storage.filename or "unknown"
    ext   = fname.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported file type: .{ext}")

    if ext == "pdf":
        raw_bytes = file_storage.read()
        doc  = fitz.open(stream=raw_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()

    if ext == "docx":
        try:
            import docx
            from io import BytesIO
            doc  = docx.Document(BytesIO(file_storage.read()))
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except ImportError:
            raise ValueError("python-docx not installed. Run: pip install python-docx")

    # Plain text / code
    raw = file_storage.read()
    return raw.decode("utf-8", errors="replace").strip()


def describe_image(image_bytes: bytes, mime_type: str, filename: str, user_prompt: str = None) -> str:
    """Send an image to Groq vision model and get a description/summary."""
    import base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = user_prompt or (
        "Please describe and summarize this image in detail. "
        "Cover: what's in the image, any text visible, key elements, colors, context, and anything notable."
    )
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq vision model
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            temperature=0.4,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Vision model failed: {e}")
        raise ValueError(f"Image processing failed: {e}")

# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session.permanent = True
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
@limiter.limit("30 per minute")
def chat():
    if not request.is_json:
        return jsonify({"reply": "Invalid request format.", "sources": []}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"reply": "Empty request.", "sources": []}), 400

    user_message = data.get("message", "").strip()
    file_text    = data.get("file_text") or ""
    file_name    = data.get("file_name") or ""
    file_pages   = data.get("file_pages")
    datetime_str = data.get("datetime", "")
    timezone_str = data.get("timezone", "Asia/Kolkata")

    if not user_message:
        return jsonify({"reply": "I didn't catch that — could you try again?", "sources": []}), 400

    if len(user_message) > MAX_MSG_LEN:
        return jsonify({"reply": f"Message too long. Keep it under {MAX_MSG_LEN} characters.", "sources": []}), 400

    session_id = session.get("session_id", "default")

    # Skip web search entirely if a file is attached — answer lives in the file
    needs_research = False if file_text else is_research_query(user_message)

    # ── Research path ──────────────────────────────────────────────────────────
    if needs_research:
        # Classify emotion in parallel concept — use calm default for research
        emotion_data = classify_emotion(user_message)
        nova_emotion = emotion_data["nova_emotion"]
        summary, sources = run_research(user_message)
        history = load_history_db(session_id)
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": summary})
        history = trim_history(history)
        save_history_db(session_id, history)
        emotion_info = EMOTION_META.get(nova_emotion, EMOTION_META["calm"])
        return jsonify({
            "reply":   summary,
            "sources": sources,
            "mode":    "research",
            "emotion": {
                "state":     nova_emotion,
                "emoji":     emotion_info["emoji"],
                "label":     emotion_info["label"],
                "color":     emotion_info["color"],
                "user_tone": emotion_data["user_tone"],
                "intensity": emotion_data["intensity"],
            }
        })

    # ── Normal chat path ───────────────────────────────────────────────────────

    # CALL 1: Emotion classification (fast small model)
    emotion_data   = classify_emotion(user_message)
    nova_emotion   = emotion_data["nova_emotion"]
    emotion_system = EMOTION_SYSTEM_PROMPTS.get(nova_emotion, EMOTION_SYSTEM_PROMPTS["calm"])
    full_system    = emotion_system + "\n\n" + SYSTEM_PROMPT

    context_block = (
        f"[Context: Current date & time is {datetime_str} IST ({timezone_str})]"
        if datetime_str else ""
    )

    if file_text:
        page_info = f" · {file_pages} page{'s' if file_pages != 1 else ''}" if file_pages else ""
        augmented_message = (
            f"[File attached: {file_name}{page_info}]\n\n{file_text[:FILE_CHAR_LIMIT]}\n\n"
            f"---\nUser question: {user_message}"
        )
    else:
        augmented_message = user_message

    augmented = f"{context_block}\n{augmented_message}".strip() if context_block else augmented_message

    history = load_history_db(session_id)

    messages_for_call = (
        [{"role": "system", "content": full_system}]
        + [m for m in history if m["role"] != "system"]
        + [{"role": "user", "content": augmented}]
    )
    messages_for_call = trim_history(messages_for_call)

    # CALL 2: Generate response with fallback retry
    ai_reply   = None
    last_error = None
    for model_attempt in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            response = client.chat.completions.create(
                model=model_attempt,
                messages=messages_for_call,
                temperature=0.75,
                max_tokens=1500
            )
            ai_reply = response.choices[0].message.content.strip()
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Chat model {model_attempt} failed: {e}")

    if ai_reply is None:
        logger.error(f"All models failed: {last_error}")
        return jsonify({"reply": "All models are unavailable right now. Try again in a moment.", "sources": []}), 502

    # Persist with stable system prompt
    history.append({"role": "user",      "content": augmented})
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
@limiter.limit("10 per minute")
def reset():
    session_id = session.get("session_id", "default")
    delete_session_db(session_id)
    load_history_db(session_id)
    return jsonify({"status": "ok"})


# FIX #18: Unified file handling — one clean /upload route that properly
# handles all file types (PDF via PyMuPDF, DOCX via python-docx, text as UTF-8).
# The client-side pdf.js extraction path is kept as a fallback in the frontend
# but the server-side route is now also fully functional and consistent.
@app.route("/upload", methods=["POST"])
@limiter.limit("20 per minute")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file  = request.files["file"]
    fname = file.filename or "unknown"
    ext   = fname.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED_EXTS:
        supported = "PDF, DOCX, TXT, MD, PY, JS, JSON, CSV, HTML + images (JPG, PNG, GIF, WEBP)"
        return jsonify({"error": f"Unsupported file type: .{ext}. Supported: {supported}"}), 400

    try:
        # ── Image path: use vision model ───────────────────────────────────────
        if ext in ALLOWED_IMAGE_EXTS:
            image_bytes = file.read()
            if len(image_bytes) < 100:
                return jsonify({"error": "Image appears empty or corrupt."}), 400

            mime_type = IMAGE_MIME.get(ext, "image/jpeg")
            reply     = describe_image(image_bytes, mime_type, fname)
            return jsonify({
                "reply":      reply,
                "char_count": len(reply),
                "is_image":   True
            })

        # ── Text/document path: existing extraction ────────────────────────────
        text = extract_file_text(file)
        if len(text.strip()) < 10:
            return jsonify({"error": "File appears to be empty or unreadable."}), 400

        truncated  = text[:FILE_CHAR_LIMIT]
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": (
                    f"[File uploaded: {fname}]\n\n{truncated}\n\n"
                    "Briefly acknowledge the file warmly and tell me what it contains in 2-3 sentences. "
                    "Then ask what I'd like to do with it."
                )}
            ],
            temperature=0.4,
            max_tokens=300
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply, "char_count": len(text), "is_image": False})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": "Failed to process file."}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":    "online",
        "assistant": "Nova",
        "sessions":  session_count_db(),
        "model":     PRIMARY_MODEL
    })

# FIX #4: /debug-search removed from production.
# Re-enable locally by setting FLASK_DEBUG=true and uncommenting below.
# if debug_mode:
#     @app.route("/debug-search")
#     def debug_search():
#         results, answer_box = search_web("pm of india")
#         return jsonify({"count": len(results), "results": results, "answer_box": answer_box})


if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 5000))
    app.run(debug=debug_mode, port=port)