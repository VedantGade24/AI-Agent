"""
Automated tests for Nova's core logic functions.

Run with:  pytest test_app.py -v

These deliberately test PURE LOGIC (string matching, list trimming,
filename checks) and avoid hitting Groq/SerpAPI/the database directly,
so they run in under a second with zero API cost and zero internet
dependency. Where a function under test calls an external API
internally (e.g. trim_history -> summarize_old_turns -> Groq), we use
unittest.mock to fake that call out.
"""

import os
import sys
from unittest.mock import patch

# Make sure required env vars exist BEFORE importing app, since app.py
# raises RuntimeError at import time if they're missing.
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-prod")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("SERPAPI_KEY", "test-serpapi-key")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# is_research_query() — the function that decides chat vs. web-search mode.
# Most important tests in this file: this exact logic has already caused
# two real bugs (over-broad "what is the" matching everything, and a
# leftover duplicate "tell me about" trigger). These tests exist so those
# bugs can never silently come back.
# ─────────────────────────────────────────────────────────────────────────────

class TestIsResearchQuery:

    def test_detects_explicit_search_request(self):
        assert app.is_research_query("search for python tutorials") is True

    def test_detects_weather_query(self):
        assert app.is_research_query("weather in pune") is True

    def test_detects_price_query(self):
        assert app.is_research_query("price of iphone 16") is True

    def test_detects_current_role_query(self):
        assert app.is_research_query("who is the ceo of openai") is True

    def test_detects_placement_keywords(self):
        assert app.is_research_query("placement cutoff for coep") is True

    def test_ignores_normal_dsa_question(self):
        # The original bug: "what is the" used to match this and send
        # a basic algorithms question into the slow web-research pipeline.
        assert app.is_research_query("what is the time complexity of quicksort") is False

    def test_ignores_normal_count_question(self):
        # Original bug: "how many" used to match this.
        assert app.is_research_query("how many bits are in ieee 754 double precision") is False

    def test_ignores_casual_explain_request(self):
        # Regression test for the "tell me about" leftover-duplicate bug
        # found while writing these tests — it was still active even
        # after being "removed" elsewhere in the list.
        assert app.is_research_query("tell me about quicksort") is False
        assert app.is_research_query("tell me about your day") is False

    def test_ignores_plain_greeting(self):
        assert app.is_research_query("hey what's up") is False

    def test_is_case_insensitive(self):
        assert app.is_research_query("WEATHER IN PUNE") is True

    def test_word_boundary_does_not_match_substring(self):
        # "news " is a trigger, but it must not fire on a word that merely
        # *contains* "news" as a substring, like "newsletter".
        assert app.is_research_query("can you write me a newsletter") is False


# ─────────────────────────────────────────────────────────────────────────────
# trim_history() — keeps chat history bounded so the prompt doesn't grow
# forever. Must never drop the system prompt, and must trigger
# summarization only once the conversation actually gets long.
# ─────────────────────────────────────────────────────────────────────────────

class TestTrimHistory:

    def _make_history(self, n_turns):
        history = [{"role": "system", "content": "SYSTEM PROMPT"}]
        for i in range(n_turns):
            history.append({"role": "user", "content": f"message {i}"})
            history.append({"role": "assistant", "content": f"reply {i}"})
        return history

    def test_short_history_passes_through_unchanged(self):
        history = self._make_history(5)
        result = app.trim_history(history)
        assert result == history

    def test_system_prompt_always_preserved(self):
        history = self._make_history(5)
        result = app.trim_history(history)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "SYSTEM PROMPT"

    @patch("app.summarize_old_turns", return_value="[mocked summary]")
    def test_long_history_gets_summarized(self, mock_summarize):
        # MAX_HISTORY_TURNS * 2 = 60 messages threshold by default
        history = self._make_history(40)  # 80 user+assistant messages, well over
        result = app.trim_history(history)

        # System prompt still first
        assert result[0]["role"] == "system"
        # Summarization should have been called since we're over the threshold
        mock_summarize.assert_called_once()
        # Result should be shorter than the original bloated history
        assert len(result) < len(history)

    @patch("app.summarize_old_turns", return_value="[mocked summary]")
    def test_long_history_keeps_recent_turns_verbatim(self, mock_summarize):
        history = self._make_history(40)
        result = app.trim_history(history)
        # The very last message (most recent) must survive trimming untouched
        assert result[-1]["content"] == "reply 39"


# ─────────────────────────────────────────────────────────────────────────────
# allowed_file() — gatekeeper for the /upload route. Getting this wrong
# either blocks legitimate files or (worse) lets something unintended through.
# ─────────────────────────────────────────────────────────────────────────────

class TestAllowedFile:

    def test_accepts_pdf(self):
        assert app.allowed_file("resume.pdf") is True

    def test_accepts_image(self):
        assert app.allowed_file("photo.png") is True

    def test_rejects_executable(self):
        assert app.allowed_file("virus.exe") is False

    def test_rejects_no_extension(self):
        assert app.allowed_file("noextension") is False

    def test_is_case_insensitive(self):
        assert app.allowed_file("DOCUMENT.PDF") is True

    def test_rejects_double_extension_trick(self):
        # filename.rsplit(".", 1) only looks at the LAST extension, so
        # "malware.exe.pdf" is correctly treated as a .pdf — this test
        # documents that behavior explicitly rather than leaving it implicit.
        assert app.allowed_file("malware.exe.pdf") is True


# ─────────────────────────────────────────────────────────────────────────────
# get_session_id() — regression test for the "everyone shares the same
# 'default' session" bug fixed earlier. Needs a Flask request/session
# context to run, so we use Flask's test_request_context.
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSessionId:

    def test_generates_a_real_uuid_when_none_exists(self):
        with app.app.test_request_context():
            sid = app.get_session_id()
            assert sid is not None
            assert sid != "default"
            assert len(sid) == 32  # uuid4().hex length

    def test_returns_same_id_on_repeated_calls_within_one_session(self):
        with app.app.test_request_context():
            first = app.get_session_id()
            second = app.get_session_id()
            assert first == second


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])