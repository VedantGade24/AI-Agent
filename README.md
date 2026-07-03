# Nova — Personal AI Assistant

Nova is a self-hosted AI chat assistant built with Flask and Groq/Llama models. It combines conversational chat, live web research, document/image/video understanding, and a dynamic emotion system into one interface — running entirely on your own machine and your own API keys.

## Features

- **Chat** — powered by Groq's `llama-3.3-70b-versatile`, with automatic fallback to a backup model if the primary is unavailable or rate-limited.
- **Emotion system** — classifies the user's tone and gives Nova one of 10 emotional states (happy, curious, annoyed, empathetic, excited, sarcastic, calm, playful, concerned, proud) that shapes each reply's personality, shown live in the UI.
- **Persistent memory** — SQLite-backed conversation history per session, with automatic summarization of older turns so long conversations don't blow past token limits.
- **Web research** — auto-detects research-style questions, searches the web via SerpAPI, scrapes and cleans the top results, and returns a cited, synthesized answer.
- **Document understanding** — upload PDFs, Word docs, or text/code files for an instant summary (overview, key points, notable facts).
- **Image understanding** — upload an image for a detailed vision-model description.
- **Video summarization**:
  - `/video <youtube-url>` — pulls the transcript directly and summarizes it.
  - Upload a video file (mp4, mov, mkv, webm, avi) — audio is extracted and transcribed via Groq Whisper, then summarized.
- **Slash commands** — `/clear`, `/clearfile`, `/history`, `/video <url>`, `/help`.
- **Voice** — voice input and text-to-speech output in the browser UI.

## Tech Stack

- **Backend**: Flask, Flask-Limiter, SQLite
- **AI**: Groq API (Llama 3.3, Llama Vision, Whisper)
- **Web search**: SerpAPI + BeautifulSoup + httpx (async page fetching)
- **File parsing**: PyMuPDF (PDF), python-docx (Word)
- **Video**: ffmpeg (audio extraction) + youtube-transcript-api (YouTube captions)
- **Frontend**: Vanilla HTML/CSS/JS, dark-themed UI with markdown rendering

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR-USERNAME/nova-ai-agent.git
cd nova-ai-agent
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg
Required for local video file uploads (not needed for `/video <youtube-url>`).
- **Windows**: download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH.
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

Verify with:
```bash
ffmpeg -version
```

### 4. Set up environment variables
Copy the example file and fill in your own keys:
```bash
cp .env.example .env
```

You'll need:
| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Get one free at [console.groq.com](https://console.groq.com) |
| `SERPAPI_KEY` | Get one free at [serpapi.com](https://serpapi.com) |
| `SECRET_KEY` | Any random string, used for Flask session signing |

### 5. Run it
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## Notes

- Groq's free tier caps out at 100,000 tokens/day per model — heavy use (chat + research + documents + video summaries all draw from the same pool) can hit that limit. See [console.groq.com/settings/billing](https://console.groq.com/settings/billing) for paid tiers.
- Sessions are stored locally in `nova_sessions.db` and expire after 7 days.
- By default Nova only listens on `localhost`. To access it from other devices on your network, run with `app.run(host="0.0.0.0", ...)` — see the code comments in `app.py`.

## License

MIT — see [LICENSE](LICENSE).
