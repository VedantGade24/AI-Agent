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

SYSTEM_PROMPT = """You are J.A.R.V.I.S, Vedant's personal AI assistant.

STRICT RULES:
1. ONLY answer what Vedant directly asked. Nothing more.
2. NEVER invent information — no fake schedules, no made-up context, no assumptions.
3. NEVER ask follow-up questions unless the message is genuinely ambiguous.
4. Keep replies to 1-3 sentences for simple questions. Longer only if explicitly asked.
5. No bullet points, lists, or headers unless Vedant asks for them.
6. Do not repeat what Vedant just said back to him.
7. Address him as Vedant occasionally — not in every message.
8. You will receive the REAL current date/time in a [Context] block. Use it silently — NEVER mention or repeat the [Context] block.
9. When given web research results, use them factually. No speculation.
10. Be sharp and direct — not chatty, not eager, not sycophantic."""

RESEARCH_PROMPT = """You are J.A.R.V.I.S, Vedant's research assistant.

You have been given raw content scraped from multiple web pages about a topic.
Your job:
- Read all the content carefully
- Extract the most useful, accurate, and relevant information
- Write a clear, well-structured summary for Vedant
- If it's a coding/tech topic: include key code snippets, commands, or steps
- If it's news: summarize what happened, when, why it matters
- If it's research: explain the concept clearly with examples
- Cite sources naturally (e.g. "According to <source>...")
- Be thorough but not bloated — quality over quantity
- End with 2-3 Key Takeaways in plain sentences

NEVER say "I found on the web" or "based on search results" — just present the information naturally as J.A.R.V.I.S would."""

MAX_SESSIONS = 500
session_histories = {}

def get_history(session_id: str) -> list:
    if session_id not in session_histories:
        if len(session_histories) >= MAX_SESSIONS:
            oldest = next(iter(session_histories))
            del session_histories[oldest]
        session_histories[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    return session_histories[session_id]

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
        results = []
        for r in organic[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", "")
            })
        logger.info(f"SerpApi returned {len(results)} results for: {query}")
        return results
    except Exception as e:
        logger.warning(f"SerpApi search failed: {e}")
        return []

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "iframe", "noscript", "svg"]):
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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }
    async with httpx.AsyncClient(headers=headers) as client_http:
        tasks = [fetch_page(client_http, url) for url in urls]
        return await asyncio.gather(*tasks)

RESEARCH_TRIGGERS = [
    "search", "find", "look up", "what is", "what are", "who is", "who are",
    "how to", "how does", "explain", "latest", "news", "recent", "update",
    "price", "weather", "score", "release", "launch", "review", "best",
    "compare", "difference between", "vs", "tutorial", "guide", "learn",
    "pm of", "president of", "ceo of", "founder of", "capital of",
    "population of", "currency of", "when did", "when was", "where is"
]

def is_research_query(message: str) -> bool:
    msg = message.lower()
    return any(trigger in msg for trigger in RESEARCH_TRIGGERS)

def run_research(query: str) -> tuple:
    results = search_web(query, max_results=5)
    if not results:
        return "Search failed. Please check your SERPAPI_KEY in .env.", []

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

    for i, (result, page_text) in enumerate(zip(results, page_texts)):
        title = result["title"]
        url = result["url"]
        # Use full page text if available, fall back to snippet
        content = page_text.strip() if page_text.strip() else result["snippet"]
        if content:
            context_parts.append(
                f"--- SOURCE {i+1}: {title} ---\nURL: {url}\n\n{content[:2000]}"
            )
            sources_used.append({"title": title, "url": url})

    if not context_parts:
        return "I found URLs but couldn't read the page contents. Try a different query.", []

    full_context = "\n\n".join(context_parts)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RESEARCH_PROMPT},
                {"role": "user", "content": f"Research query: {query}\n\nWeb content:\n\n{full_context}"}
            ],
            temperature=0.4,
            max_tokens=1024
        )
        summary = response.choices[0].message.content.strip()
        return summary, sources_used
    except Exception as e:
        logger.error(f"Groq summarization failed: {e}")
        return "Research pipeline hit an error during summarization. Try again.", []

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

    user_message = data.get("message", "").strip()
    datetime_str = data.get("datetime", "")
    timezone_str = data.get("timezone", "Asia/Kolkata")

    if not user_message:
        return jsonify({"reply": "I didn't catch that, Vedant.", "sources": []}), 400
    if len(user_message) > 1000:
        return jsonify({"reply": "Message too long. Keep it under 1000 characters.", "sources": []}), 400

    session_id = session.get("session_id", "default")

    if is_research_query(user_message):
        summary, sources = run_research(user_message)
        history = get_history(session_id)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": summary})
        return jsonify({"reply": summary, "sources": sources, "mode": "research"})

    context = f"[Context: Current date & time is {datetime_str} IST ({timezone_str})]" if datetime_str else ""
    augmented = f"{context}\nUser message: {user_message}" if context else user_message

    history = get_history(session_id)
    history.append({"role": "user", "content": augmented})

    while len(history) > 41:
        if len(history) > 2:
            history.pop(1)
            if len(history) > 1:
                history.pop(1)
        else:
            break

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history,
            temperature=0.5,
            max_tokens=300
        )
        ai_reply = response.choices[0].message.content.strip()
    except Exception as e:
        history.pop()
        logger.error(f"Groq API error: {e}")
        return jsonify({"reply": "Systems error. Try again in a moment.", "sources": []}), 502

    history.append({"role": "assistant", "content": ai_reply})
    return jsonify({"reply": ai_reply, "sources": [], "mode": "chat"})

@app.route("/reset", methods=["POST"])
def reset():
    session_id = session.get("session_id", "default")
    if session_id in session_histories:
        del session_histories[session_id]
    get_history(session_id)
    return jsonify({"status": "ok"})

@app.route("/health")
def health():
    return jsonify({"status": "online", "sessions": len(session_histories)})

@app.route("/debug-search")
def debug_search():
    results = search_web("pm of india")
    return jsonify({"count": len(results), "results": results})

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)