import os
import re
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from ddgs import DDGS

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    BrowserConfig,
    CacheMode,
)
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


# ---------------------------
# Configuration & Defaults
# ---------------------------

DEFAULT_MODEL = "openai/gpt-oss-120b"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Reasoning Model Configuration (defaults)
REASONING_EFFORT = "low"  # "low", "medium", "high"
REASONING_EXCLUDE = False
MAX_RETRIES = 3
BASE_TIMEOUT = 180
RETRY_DELAY = 10
MAX_TOKENS_SEARCH_QUERIES = 1000
MAX_TOKENS_FINAL_ANSWER = 2000


log = logging.getLogger("seexsum_ai")


# ---------------------------
# Exceptions
# ---------------------------

class SeExSumAIError(RuntimeError):
    pass


class ConfigError(SeExSumAIError):
    pass


class SearchError(SeExSumAIError):
    pass


class CrawlError(SeExSumAIError):
    pass


class LLMError(SeExSumAIError):
    pass


# ---------------------------
# Helpers (module-level, reusable)
# ---------------------------

def load_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY in environment or .env."
        )
    return key


def normalize_url(u: str) -> str:
    try:
        p = urlparse(u)
        host = p.netloc.lower()
        path = re.sub(r"/+\Z", "", p.path or "/")
        return f"{host}{path}"
    except Exception:
        return u


def is_low_value(url: str) -> bool:
    low = [
        "pinterest.",
        "link.springer.com/epdf",
        "facebook.",
        "twitter.",
        "x.com/",
        "instagram.",
        "tiktok.",
        "reddit.com/r/",
        "github.com/issues",
        "medium.com/p/",
        "youtube.com/shorts",
    ]
    u = url.lower()
    return any(s in u for s in low)


def safe_json_extract(text: str) -> Optional[List[str]]:
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        pass
    return None


def call_openrouter(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 600,
    response_format: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    reasoning_config: Optional[Dict[str, Any]] = None,
    max_retries: int = MAX_RETRIES,
    base_timeout: int = BASE_TIMEOUT,
    retry_delay: int = RETRY_DELAY,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local-script",
        "X-Title": "SeExSum_AI",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    if reasoning_config:
        payload["reasoning"] = reasoning_config

    last_error: Optional[str] = None
    current_delay = retry_delay
    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (attempt + 1)
            log.info(f"  Attempt {attempt + 1}/{max_retries} with {timeout}s timeout...")

            resp = requests.post(
                OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=timeout
            )
            if resp.status_code != 200:
                raise LLMError(f"OpenRouter error {resp.status_code}: {resp.text}")

            data = resp.json()
            try:
                content = data["choices"][0]["message"]["content"]
                reasoning = data["choices"][0]["message"].get("reasoning", "")
                if not content:
                    raise LLMError(
                        "Model returned empty content. The model may have failed to generate a response."
                    )
                return content
            except Exception as e:
                raise LLMError(f"Unexpected OpenRouter response: {data}") from e

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {timeout}s (attempt {attempt + 1}/{max_retries})"
            log.warning(f"  {last_error}")
            if attempt < max_retries - 1:
                log.info(f"  Waiting {current_delay}s before retry...")
                time.sleep(current_delay)
                current_delay *= 2
            continue
        except Exception as e:
            last_error = str(e)
            log.warning(f"  Error on attempt {attempt + 1}/{max_retries}: {last_error}")
            if attempt < max_retries - 1:
                log.info(f"  Waiting {current_delay}s before retry...")
                time.sleep(current_delay)
                current_delay *= 2
            continue

    raise LLMError(f"All {max_retries} attempts failed. Last error: {last_error}")


def llm_make_search_queries(api_key: str, model: str, question: str) -> List[str]:
    system = {
        "role": "system",
        "content": (
            "You generate excellent DuckDuckGo queries for web Q&A. "
            "Be specific, avoid brand bias, and include keywords that disambiguate. "
            "You MUST return ONLY a JSON array of 3-6 short string queries, no commentary, no reasoning text. "
            "Example format: [\"query 1\", \"query 2\", \"query 3\"]"
        ),
    }
    user = {
        "role": "user",
        "content": f"User question:\n{question}\n\nOutput: JSON array of queries.",
    }

    reply = call_openrouter(
        api_key,
        model,
        [system, user],
        temperature=0.1,
        max_tokens=MAX_TOKENS_SEARCH_QUERIES,
        reasoning_config={"effort": REASONING_EFFORT, "exclude": REASONING_EXCLUDE},
    )

    queries: Optional[List[str]] = None
    try:
        obj = json.loads(reply)
        if isinstance(obj, list):
            queries = obj
        elif isinstance(obj, dict) and "queries" in obj and isinstance(obj["queries"], list):
            queries = obj["queries"]
    except Exception:
        pass

    if not queries:
        queries = safe_json_extract(reply)

    if not queries:
        queries = [line.strip("-• ").strip() for line in reply.splitlines() if line.strip()][0:5]

    if not queries:
        quoted_queries = re.findall(r'"([^"]+)"', reply)
        if quoted_queries:
            queries = quoted_queries[:6]
        if not queries:
            pattern_queries = re.findall(
                r'(?:query|queries?)\s+(?:like\s+)?["\']([^"\']+)["\']', reply, re.IGNORECASE
            )
            if pattern_queries:
                queries = pattern_queries[:6]
        if not queries:
            phrases = re.split(r'[.!?]', reply)
            meaningful_phrases = []
            for phrase in phrases:
                phrase = phrase.strip()
                if 10 < len(phrase) < 100 and not phrase.startswith('**'):
                    clean_phrase = re.sub(r'\*\*[^*]+\*\*', '', phrase)
                    clean_phrase = re.sub(
                        r"\b(?:I need to|Let's|I will|I can|I might|I should)\b",
                        '',
                        clean_phrase,
                        flags=re.IGNORECASE,
                    ).strip()
                    if clean_phrase and len(clean_phrase) > 10:
                        meaningful_phrases.append(clean_phrase)
            if meaningful_phrases:
                queries = meaningful_phrases[:6]

    cleaned: List[str] = []
    seen = set()
    for q in queries or []:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            cleaned.append(q)
    return cleaned[:6]


def ddg_web_search(queries: List[str], k: int = 8) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    seen_urls = set()

    with DDGS() as ddgs:
        for q in queries:
            try:
                hits = ddgs.text(q, max_results=k)
            except Exception as e:
                log.warning(f"[search] DDG failed for '{q}': {e}")
                continue

            for h in hits or []:
                href = h.get("href") or h.get("link") or ""
                if not href:
                    continue
                if is_low_value(href):
                    continue
                key = normalize_url(href)
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                results.append(
                    {
                        "title": (h.get("title") or "").strip(),
                        "href": href.strip(),
                        "body": (h.get("body") or h.get("snippet") or "").strip(),
                    }
                )

    return results


async def crawl_pages(
    urls: List[str],
    per_page_char_limit: int = 5000,
    timeout_ms: int = 60000,
) -> Dict[str, str]:
    md_gen = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.35, threshold_type="fixed")
    )
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_gen,
        page_timeout=timeout_ms,
    )
    browser_conf = BrowserConfig(headless=True, java_script_enabled=True)

    extracted: Dict[str, str] = {}
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        results = await crawler.arun_many(urls, config=run_conf)

        for res in results:
            if not getattr(res, "success", False):
                log.warning(
                    f"[crawl] Failed: {res.url} -> {getattr(res, 'error_message', 'unknown error')}"
                )
                continue

            md = None
            try:
                md = getattr(res.markdown, "fit_markdown", None) or getattr(
                    res.markdown, "raw_markdown", None
                )
            except Exception:
                md = str(getattr(res, "markdown", ""))

            if not md:
                continue

            snippet = md[:per_page_char_limit]
            extracted[res.url] = snippet

    return extracted


def build_answer_prompt(question: str, url_to_md: Dict[str, str]) -> List[Dict[str, str]]:
    sources_list = sorted(url_to_md.keys())
    numbered = "\n".join(f"[{i+1}] {u}" for i, u in enumerate(sources_list))

    context_blocks = []
    for i, u in enumerate(sources_list, start=1):
        context_blocks.append(f"SOURCE [{i}] {u}\n---\n{url_to_md[u]}")
    context_text = "\n\n".join(context_blocks)

    system = {
        "role": "system",
        "content": (
            "You are a careful research assistant. Read the provided source snippets "
            "and answer the user's question concisely and accurately. If facts conflict, say so. "
            "If information is insufficient, say what is missing. "
            "Cite sources inline using [#] that match the numbered list."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"QUESTION:\n{question}\n\n"
            f"NUMBERED SOURCES:\n{numbered}\n\n"
            f"SNIPPETS (Markdown, lightly cleaned):\n{context_text}\n\n"
            "Now write a concise answer (3–8 sentences when possible), include 1–4 inline citations like [2], [3]."
        ),
    }
    return [system, user]


def clamp_text(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[:max_chars]


# ---------------------------
# Public API
# ---------------------------

class SeExSumAI:
    """High-level interface to generate a source-backed answer for a question.

    Optionally provides progress callbacks for embedding into other apps.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.api_key = api_key or load_api_key()
        self.model = model
        self.progress_callback = progress_callback
        self.log = logger or logging.getLogger("seexsum_ai")

    def _emit(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(event, details or {})
            except Exception:
                pass

    async def get_answer(
        self,
        question: str,
        *,
        k: int = 6,
        max_pages: int = 6,
        per_page_chars: int = 5000,
        total_context_chars: int = 16000,
        timeout_ms: int = 60000,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        if not question or not isinstance(question, str):
            raise ValueError("Parameter 'question' must be a non-empty string")

        self._emit("reformulate:start", {"question": question})
        self.log.info("→ Reformulating your question into search queries...")
        try:
            queries = llm_make_search_queries(self.api_key, self.model, question)
        except Exception as e:
            raise LLMError(f"Failed to generate search queries: {e}") from e
        if not queries:
            raise LLMError("The LLM returned no queries.")
        self._emit("reformulate:done", {"queries": queries})
        self.log.info("  Queries: " + " | ".join(queries))

        self._emit("search:start", {"k": k})
        self.log.info("→ Searching DuckDuckGo...")
        try:
            hits = ddg_web_search(queries, k=k)
        except Exception as e:
            raise SearchError(f"DuckDuckGo search failed: {e}") from e
        if not hits:
            raise SearchError("No search results found. Try rephrasing the question.")

        unique: List[Dict[str, str]] = []
        seen = set()
        for h in hits:
            key = normalize_url(h["href"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(h)
            if len(unique) >= max_pages:
                break
        urls = [h["href"] for h in unique]
        self._emit("search:done", {"selected_urls": urls})
        self.log.info(f"  Selected {len(urls)} URLs to crawl.")

        self._emit("crawl:start", {"count": len(urls)})
        self.log.info("→ Crawling & cleaning pages (Crawl4AI)...")
        try:
            url_to_md = await crawl_pages(
                urls, per_page_char_limit=per_page_chars, timeout_ms=timeout_ms
            )
        except Exception as e:
            raise CrawlError(f"Crawling failed: {e}") from e
        if not url_to_md:
            raise CrawlError("Crawling failed for all pages.")

        joined = "\n".join(url_to_md.values())
        if len(joined) > total_context_chars:
            ratio = total_context_chars / max(1, len(joined))
            new_map: Dict[str, str] = {}
            for u, md in url_to_md.items():
                keep = max(1000, int(len(md) * ratio))
                new_map[u] = clamp_text(md, keep)
            url_to_md = new_map
        self._emit("crawl:done", {"effective_pages": len(url_to_md)})

        self._emit("synthesize:start", {})
        self.log.info("→ Asking the LLM to synthesize the answer...")
        messages = build_answer_prompt(question, url_to_md)
        try:
            answer = call_openrouter(
                self.api_key,
                self.model,
                messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS_FINAL_ANSWER,
                reasoning_config={
                    "effort": REASONING_EFFORT,
                    "exclude": REASONING_EXCLUDE,
                },
            )
        except Exception as e:
            raise LLMError(f"Failed to synthesize final answer: {e}") from e

        used_urls = list(url_to_md.keys())
        self._emit("synthesize:done", {"sources": used_urls})
        self._emit("done", {"answer_length": len(answer), "sources": used_urls})
        return {"answer": answer, "sources": used_urls}

    def get_answer_sync(
        self,
        question: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "An event loop is already running. Use 'await SeExSumAI.get_answer(...)' in async contexts."
            )
        return asyncio.run(self.get_answer(question, **kwargs))


def print_result(answer: str, used_urls: List[str]) -> None:
    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer.strip())
    print("\nSOURCES:")
    for u in used_urls:
        print(" -", u)
    print("=" * 80 + "\n")


