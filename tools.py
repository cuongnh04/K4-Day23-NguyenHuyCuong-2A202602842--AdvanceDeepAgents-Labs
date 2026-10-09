"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random
import re
import threading
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = max(0.0, float(exc.retry_after))
            else:
                delay = min(cap, base * (2 ** attempt))
                delay += random.uniform(0.0, min(base, cap - delay if cap > delay else 0.0))
            time.sleep(min(cap, delay))


def _response_or_retry(response):
    if response.status_code in {429, 500, 502, 503, 504}:
        retry_after = response.headers.get("Retry-After")
        try:
            retry_after = float(retry_after) if retry_after is not None else None
        except ValueError:
            retry_after = None
        raise RetryableError(f"HTTP {response.status_code}", retry_after=retry_after)
    response.raise_for_status()
    return response


def _json_get(url, params=None):
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url, params=params)
        return _response_or_retry(response).json()


def _compact(value, limit=600):
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _error(exc, secret=""):
    message = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
    return f"ERROR: {type(exc).__name__}: {message}"


def _paper_record(item):
    paper = item.get("paper") or item
    paper_id = paper.get("id")
    if not paper_id:
        return None
    return {
        "id": paper_id,
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
        "title": _compact(paper.get("title") or item.get("title")),
        "summary": _compact(paper.get("ai_summary") or paper.get("summary") or item.get("summary")),
        "upvotes": int(paper.get("upvotes") or 0),
        "github": paper.get("githubRepo") or "",
        "stars": int(paper.get("githubStars") or 0),
    }


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    global _ARXIV_LAST_CALL
    try:
        terms = re.findall(r"[A-Za-z0-9]+", query or "")
        if not terms:
            return "NO RESULTS"
        clean_query = " AND ".join(f"all:{term}" for term in terms)
        with _ARXIV_LOCK:
            wait = 3.0 - (time.monotonic() - _ARXIV_LAST_CALL)
            if wait > 0:
                time.sleep(wait)
            _ARXIV_LAST_CALL = time.monotonic()
        params = {
            "search_query": clean_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max(1, min(int(max_results), 30)),
        }
        response = with_retry(lambda: _get_arxiv(params), attempts=6, base=2.0, cap=60.0)
        root = xml.etree.ElementTree.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", ns):
            raw_id = (entry.findtext("atom:id", default="", namespaces=ns).strip().rsplit("/", 1)[-1])
            paper_id = re.sub(r"v\d+$", "", raw_id)
            if not paper_id:
                continue
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": entry.findtext("atom:published", default="", namespaces=ns)[:10],
                "title": _compact(entry.findtext("atom:title", default="", namespaces=ns)),
                "summary": _compact(entry.findtext("atom:summary", default="", namespaces=ns)),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


def _get_arxiv(params):
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        return _response_or_retry(client.get(ARXIV_URL, params=params))


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        params = {"limit": max(1, min(int(limit), 100))}
        if date:
            params["date"] = date
        data = with_retry(lambda: _json_get(HF_DAILY_URL, params), cap=30.0)
        records = [record for item in data if (record := _paper_record(item))]
        if keyword:
            needle = keyword.casefold()
            records = [r for r in records if needle in f"{r['title']} {r['summary']}".casefold()]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    try:
        if not (query or "").strip():
            return "NO RESULTS"
        data = with_retry(lambda: _json_get(HF_SEARCH_URL, {"q": query.strip(), "limit": max(1, min(int(limit), 50))}))
        records = [record for item in data if (record := _paper_record(item))]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    try:
        return _exa_call(
            "web_search_exa",
            {"query": query, "objective": objective or f"Find reliable sources about {query}", "numResults": max(1, min(int(num_results), 10))},
        )
    except Exception as exc:
        return _error(exc, os.getenv("EXA_API_KEY", ""))


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    try:
        return _exa_call("web_fetch_exa", {"urls": [url]})[:12000]
    except Exception as exc:
        return _error(exc, os.getenv("EXA_API_KEY", ""))


def _exa_call(name, arguments):
    key = (os.getenv("EXA_API_KEY") or "").strip()
    endpoint = EXA_URL + (f"?exaApiKey={key}" if key else "")

    def call():
        with httpx.Client(timeout=45.0) as client:
            response = client.post(
                endpoint,
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
            )
            response = _response_or_retry(response)
            payloads = []
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    try:
                        payloads.append(json.loads(line[5:].strip()))
                    except json.JSONDecodeError:
                        continue
            payload = payloads[-1] if payloads else response.json()
            if "error" in payload:
                raise RetryableError(str(payload["error"])) if _is_rate_limited(payload) else RuntimeError(str(payload["error"]))
            if _is_rate_limited(payload):
                raise RetryableError("Exa rate limit")
            content = (payload.get("result") or {}).get("content") or []
            text = "\n".join(str(part.get("text", "")) for part in content if part.get("type") == "text").strip()
            return text or "NO RESULTS"

    return with_retry(call, attempts=6, base=3.0, cap=60.0)


def _is_rate_limited(payload):
    text = json.dumps(payload, ensure_ascii=False).casefold()
    return any(marker in text for marker in ("rate limit", "rate_limit", "too many requests", "throttl"))


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
