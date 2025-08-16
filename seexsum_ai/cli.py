import argparse
import asyncio
import logging
import warnings
from typing import Any

# Suppress deprecation warnings from dependencies
warnings.filterwarnings('ignore', category=DeprecationWarning)

from .core import SeExSumAI, print_result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SeExSum_AI-based Quick Answer (DuckDuckGo + Crawl4AI + OpenRouter)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("question", type=str, help="Your question in natural language")
    p.add_argument("--model", type=str, default=None, help="OpenRouter model id")
    p.add_argument("--k", type=int, default=6, help="Max DDG results per query")
    p.add_argument("--max-pages", type=int, default=6, help="Max pages to crawl total (after dedupe)")
    p.add_argument("--per-page-chars", type=int, default=5000, help="Chars to keep per crawled page")
    p.add_argument("--total-context-chars", type=int, default=16000, help="Total chars sent to LLM (safety clamp)")
    p.add_argument("--timeout-ms", type=int, default=60000, help="Per-page crawl timeout (ms)")
    p.add_argument("--temperature", type=float, default=0.2, help="LLM temperature for final answer")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    ai = SeExSumAI(model=args.model or "openai/gpt-oss-120b")

    async def _run() -> None:
        result = await ai.get_answer(
            args.question,
            k=args.k,
            max_pages=args.max_pages,
            per_page_chars=args.per_page_chars,
            total_context_chars=args.total_context_chars,
            timeout_ms=args.timeout_ms,
            temperature=args.temperature,
        )
        print_result(result["answer"], result["sources"])

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        logging.getLogger("seexsum_ai").error(f"\nERROR: {e}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()


