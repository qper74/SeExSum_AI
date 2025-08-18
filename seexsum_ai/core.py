import os
import re
import json
import time
import asyncio
import logging
import warnings
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlparse

# Suppress deprecation warnings from dependencies
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*text.*argument.*deprecated.*')

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


def startpage_web_search(queries: List[str], k: int = 8) -> List[Dict[str, str]]:
    """Scrape Startpage search results.

    Implementation notes:
    - Uses simple HTML parsing via regex to avoid extra deps
    - Attempts multiple known Startpage paths for resilience
    - Dedupes by normalized URL and filters low-value domains
    """
    results: List[Dict[str, str]] = []
    seen_urls = set()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.startpage.com/",
        }
    )

    search_urls = [
        "https://www.startpage.com/sp/search",
        "https://www.startpage.com/do/search",
    ]

    # Regex patterns for result anchors and snippets (multiple variants for resilience)
    anchor_patterns = [
        re.compile(r'<a[^>]+class="[^"]*(?:w-gl__result-title|result-link|result-title)[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<a[^>]+data-testid="result-title-a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL),
    ]
    snippet_patterns = [
        re.compile(r'<p[^>]+class="[^"]*(?:w-gl__description|result-summary|result-snippet)[^"]*"[^>]*>(.*?)</p>', re.IGNORECASE | re.DOTALL),
    ]

    def _clean_html(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    for q in queries:
        per_query_collected = 0
        for base_url in search_urls:
            if per_query_collected >= k:
                break
            try:
                # Startpage expects the query parameter key 'q'
                resp = session.get(base_url, params={"q": q}, timeout=20)
                if resp.status_code != 200:
                    continue
                html = resp.text

                # Extract anchors
                anchors: List[tuple] = []
                for pat in anchor_patterns:
                    anchors.extend(pat.findall(html))
                # Extract snippets
                snippets: List[str] = []
                for spat in snippet_patterns:
                    snippets.extend([_clean_html(m) for m in spat.findall(html)])

                snippet_idx = 0
                for href, inner_html in anchors:
                    if per_query_collected >= k:
                        break
                    if not href:
                        continue
                    # Normalize and filter
                    if is_low_value(href):
                        continue
                    key = normalize_url(href)
                    if key in seen_urls:
                        continue

                    title = _clean_html(inner_html)
                    body = snippets[snippet_idx] if snippet_idx < len(snippets) else ""
                    snippet_idx += 1

                    seen_urls.add(key)
                    results.append({"title": title, "href": href.strip(), "body": body})
                    per_query_collected += 1

            except Exception as e:
                log.warning(f"[search] Startpage failed for '{q}': {e}")
                continue

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
            "Cite sources inline using [#] that match the numbered list. "
            "Your answer must be self-contained, begin as a proper sentence (not mid-sentence), and end with proper punctuation."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"QUESTION:\n{question}\n\n"
            f"NUMBERED SOURCES:\n{numbered}\n\n"
            f"SNIPPETS (Markdown, lightly cleaned):\n{context_text}\n\n"
            "Now write a concise answer (3–8 sentences when possible), include 1–4 inline citations like [2], [3]. "
            "Do not start mid-sentence; begin with a clear subject."
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
        engines: Optional[List[str]] = None,
        max_pages: int = 6,
        per_page_chars: int = 5000,
        total_context_chars: int = 16000,
        timeout_ms: int = 60000,
        temperature: float = 0.2,
        max_answer_tokens: int = MAX_TOKENS_FINAL_ANSWER,
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
        engines_list = [e.strip().lower() for e in (engines or ["ddg"]) if e and e.strip()]
        if not engines_list:
            engines_list = ["ddg"]

        all_hits: List[Dict[str, str]] = []
        for engine in engines_list:
            if engine == "ddg":
                self.log.info("→ Searching DuckDuckGo...")
                try:
                    all_hits.extend(ddg_web_search(queries, k=k))
                except Exception as e:
                    self.log.warning(f"DuckDuckGo search failed: {e}")
            elif engine == "startpage":
                self.log.info("→ Searching Startpage...")
                try:
                    all_hits.extend(startpage_web_search(queries, k=k))
                except Exception as e:
                    self.log.warning(f"Startpage search failed: {e}")
            else:
                self.log.warning(f"Unknown search engine: {engine}")

        if not all_hits:
            raise SearchError("No search results found across engines. Try rephrasing the question.")

        unique: List[Dict[str, str]] = []
        seen = set()
        for h in all_hits:
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
                max_tokens=max_answer_tokens,
                reasoning_config={
                    "effort": REASONING_EFFORT,
                    "exclude": REASONING_EXCLUDE,
                },
            )
        except Exception as e:
            raise LLMError(f"Failed to synthesize final answer: {e}") from e

        # If the model response appears truncated (no sentence-ending punctuation),
        # request a short continuation to finish the thought cleanly.
        def _looks_truncated(text: str) -> bool:
            t = (text or "").strip()
            if not t:
                return False
            # Consider it complete if it ends with sentence punctuation or a citation bracket
            if re.search(r"[\.\!\?]$", t):
                return False
            if re.search(r"\]\s*$", t):
                return False
            return True

        if _looks_truncated(answer):
            try:
                # Reuse the same prompt context, add the partial assistant answer and ask to continue.
                continuation_messages: List[Dict[str, str]] = [
                    messages[0],  # system
                    messages[1],  # user with question + sources + snippets
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            "Continue exactly where you left off to finish the current sentence/paragraph. "
                            "Do not repeat prior text. Keep it concise and end naturally."
                        ),
                    },
                ]
                continuation = call_openrouter(
                    self.api_key,
                    self.model,
                    continuation_messages,
                    temperature=max(0.1, min(temperature, 0.7)),
                    max_tokens=max(128, min(max_answer_tokens // 4, 512)),
                    reasoning_config={
                        "effort": REASONING_EFFORT,
                        "exclude": REASONING_EXCLUDE,
                    },
                )
                if continuation:
                    answer = (answer + " " + continuation).strip()
            except Exception:
                # Best-effort; keep the original answer if continuation fails
                pass

        # If the model started mid-sentence (e.g., begins with a closing quote, comma, or conjunction),
        # ask it to rewrite the answer into a clean, self-contained paragraph while preserving citations.
        def _looks_mid_sentence_start(text: str) -> bool:
            t = (text or "").lstrip()
            if not t:
                return False
            # Suspicious starts: closing punctuation/quotes, comma, dash, ellipsis, or conjunctions
            if re.match(r'^[\)\]\"\'\u201D\,\-\u2026]', t):
                return True
            if re.match(r'^(and|but|so|or|because|however|therefore)\b', t, re.IGNORECASE):
                return True
            # If first character is lowercase ASCII letter and not a proper noun, likely mid-sentence
            if re.match(r'^[a-z]', t):
                return True
            return False

        if _looks_mid_sentence_start(answer):
            try:
                rewrite_messages: List[Dict[str, str]] = [
                    messages[0],
                    messages[1],
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the assistant's answer into a self-contained, well-formed paragraph that starts with a full sentence "
                            "and ends with proper punctuation. Preserve the factual content and keep the existing inline citations [#]; "
                            "do not introduce new claims."
                        ),
                    },
                ]
                rewritten = call_openrouter(
                    self.api_key,
                    self.model,
                    rewrite_messages,
                    temperature=max(0.1, min(temperature, 0.7)),
                    max_tokens=max(256, min(max_answer_tokens // 2, 600)),
                    reasoning_config={
                        "effort": REASONING_EFFORT,
                        "exclude": REASONING_EXCLUDE,
                    },
                )
                if rewritten:
                    answer = rewritten.strip()
            except Exception:
                pass

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


