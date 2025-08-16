# SeExSum_AI

**SeExSum_AI** — an intelligent web-based question-answering system that uses LLM to generate accurate, source-backed answers.

This repository provides:
- a reusable Python library package `seexsum_ai` (importable in your apps), and
- a CLI tool (`seexsum-ai`).

## What it does (in plain English)

1. **Question reformulation**: Uses LLM (via OpenRouter) to transform your question into strong web search queries
2. **Web search**: Searches the web with DuckDuckGo and collects top result URLs
3. **Page cleaning**: Uses Crawl4AI to convert each page into LLM-friendly Markdown format
4. **Answer synthesis**: Asks the LLM to synthesize the answer based on sources
5. **Result display**: Prints the answer and the list of sources

## Installation (one time)

```bash
# 1. Create and activate virtual environment (recommended)
python -m venv venv

# Activate virtual environment:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install the package (choose one option)
# Option A: install from this repo (editable)
pip install -e .

# Or pip install directly from GitHub (replace with your repo URL)
pip install git+https://github.com/your-username/SeExSum_AI.git

# 3. Install Crawl4AI browser (one-time)
crawl4ai-setup

# 4. Set up API key
echo 'OPENROUTER_API_KEY=sk-or-...' >> .env
```

## Usage

### Basic usage (CLI)

```bash
seexsum-ai "Who directed Dune (2021) and when did it release?"
```

### With specific parameters (CLI)

```bash
seexsum-ai "What is the capital of France?" \
  --model "google/gemini-2.5-flash-lite" \
  --k 6 \
  --max-pages 5
```

### All parameters (CLI)

```bash
seexsum-ai --help
```

## Library usage

```python
from seexsum_ai import SeExSumAI

# Async usage (recommended)
ai = SeExSumAI()
result = await ai.get_answer("Who directed Dune (2021) and when did it release?", max_pages=5)
print(result["answer"])  # str
print(result["sources"]) # list[str]

# Sync (convenience) usage
result = ai.get_answer_sync("What is the capital of France?", k=6)
```

**Parameters:**
- `--model`: OpenRouter model identifier (default: `openai/gpt-oss-120b`)
- `--k`: Max DDG results per query (default: 6)
- `--max-pages`: Max pages to crawl (default: 6)
- `--per-page-chars`: Characters per page (default: 5000)
- `--total-context-chars`: Total characters sent to LLM (default: 16000)
- `--timeout-ms`: Per-page timeout (ms) (default: 60000)
- `--temperature`: LLM temperature for final answer (default: 0.2)
- `--max-answer-tokens`: Max tokens for final answer (default: 600)

## Configuration

### Environment variables

Create a `.env` file in the project root directory:

```env
OPENROUTER_API_KEY=sk-or-your-api-key-here
```

### Progress reporting (embedding into apps)

`SeExSumAI` accepts an optional `progress_callback(event: str, details: dict)` to receive structured progress events:
- reformulate:start/done
- search:start/done
- crawl:start/done
- synthesize:start/done
- done

Example:
```python
def on_progress(event, details):
    print(event, details)

ai = SeExSumAI(progress_callback=on_progress)
result = ai.get_answer_sync("Latest on James Webb discoveries")
```

### Default model

The script uses `openai/gpt-oss-120b` by default, but you can switch to any model supported by OpenRouter.

## Example output

```
→ Reformulating your question into search queries...
  Queries: capital of France | France capital city | what is the capital of France
→ Searching DuckDuckGo...
  Selected 6 URLs to crawl.
→ Crawling & cleaning pages (Crawl4AI)...
→ Asking the LLM to synthesize the answer...

================================================================================
ANSWER
================================================================================
The capital of France is Paris. Paris has served as the French capital since its 
liberation in 1944, and it remains both the political and largest city of the 
country today.

SOURCES:
 - https://en.wikipedia.org/wiki/Capital_of_France
 - https://worldpopulationreview.com/countries/france
 - https://en.wikipedia.org/wiki/Paris
================================================================================
```

## Troubleshooting

### Common issues

1. **Missing API key**: Check that `OPENROUTER_API_KEY` is set
2. **Crawl4AI browser**: Run the `crawl4ai-setup` command
3. **Network issues**: Some sites may block bots
4. **Timeout errors**: Increase the `--timeout-ms` value

### Error reporting

If you encounter problems, read the error message carefully. Most issues stem from missing API keys, uninstalled Crawl4AI browser, or bot-blocking sites.

## Technical details

### Technologies used

- **OpenRouter**: LLM API service
- **DuckDuckGo**: Web search
- **Crawl4AI**: Web page crawling and cleaning
- **Python**: Programming language

### Architecture

1. **Question processing**: Transforms to search queries using LLM
2. **Web search**: Uses DuckDuckGo API
3. **Content extraction**: Asynchronous crawling with Crawl4AI
4. **Answer generation**: LLM synthesizes answer from sources

## License

This project is open source. Use at your own risk.

## Contributing

For suggestions and bug reports, please open an issue in the GitHub repository.
