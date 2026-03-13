import html as _html
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import date
from typing import Any, Dict, List, Optional

from agent_scaling.env.base import AgentEnvironmentTools
from agent_scaling.env.registry import register_env
from agent_scaling.env.tools import cls_tool
from agent_scaling.logger import logger


def _strip_html(text: str) -> str:
    # Minimal HTML to text conversion without external deps.
    text = _html.unescape(text)
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"[ \\t\\r\\f\\v]+", " ", text)
    text = re.sub(r"\\n\\s*\\n+", "\\n", text)
    return text.strip()


@register_env("finance-agent")
class FinanceAgentEnvironment(AgentEnvironmentTools):
    """Finance-Agent tool environment (web + EDGAR + lightweight retrieval)."""

    def __init__(self, *args, **kwargs):
        self.is_done = False
        self._store: Dict[str, str] = {}
        super().__init__(*args, **kwargs)

    def get_instance_prompt_info(self) -> Dict[str, str]:
        base = super().get_instance_prompt_info()
        base["current_date"] = date.today().isoformat()
        return base

    def env_done(self) -> bool:
        return self.is_done

    @cls_tool
    def web_search(self, search_query: str) -> str:
        """Search the web for relevant information (Tavily)."""
        try:
            from tavily import TavilyClient  # type: ignore
        except Exception as exc:
            return f"ERROR: tavily dependency is missing: {exc}"

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "ERROR: TAVILY_API_KEY is not set."

        client = TavilyClient(api_key=api_key)
        try:
            response = client.search(search_query)
        except Exception as exc:
            return f"ERROR: Tavily search failed: {exc}"

        results = []
        for result in (response or {}).get("results", [])[:10]:
            results.append(
                {
                    "title": result.get("title", ""),
                    "snippet": result.get("content", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0),
                }
            )
        payload = {
            "query": search_query,
            "results": results,
            "answer": (response or {}).get("answer", ""),
        }
        text = json.dumps(payload, indent=2)
        self._store[f"web_search:{search_query}"] = text
        self._store["web_search:last"] = text
        return text

    @cls_tool
    def edgar_search(
        self,
        query: str,
        form_types: Optional[List[str]] = None,
        ciks: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: str = "1",
        top_n_results: int = 10,
    ) -> str:
        """Search SEC EDGAR filings using sec-api.io Full-Text Search API."""
        api_key = os.environ.get("SEC_EDGAR_API_KEY")
        if not api_key:
            return "ERROR: SEC_EDGAR_API_KEY is not set."

        try:
            page_int = int(page)
        except Exception:
            page_int = 1

        body: Dict[str, Any] = {"query": query, "page": page_int}
        if form_types:
            body["formTypes"] = form_types
        if ciks:
            body["ciks"] = ciks
        if start_date:
            body["startDate"] = start_date
        if end_date:
            body["endDate"] = end_date

        url = (
            "https://api.sec-api.io/full-text-search?token="
            + urllib.parse.quote(api_key)
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "agent-scaling/finance-agent",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            return f"ERROR: EDGAR search failed: {exc}"

        try:
            data = json.loads(raw)
        except Exception:
            return raw

        filings = data.get("filings", []) or []
        simplified = []
        for filing in filings[: max(1, int(top_n_results))]:
            simplified.append(
                {
                    "companyNameLong": filing.get("companyNameLong"),
                    "cik": filing.get("cik"),
                    "formType": filing.get("formType"),
                    "filedAt": filing.get("filedAt"),
                    "filingUrl": filing.get("filingUrl"),
                    "accessionNo": filing.get("accessionNo"),
                }
            )

        payload = {
            "query": query,
            "total": data.get("total", len(filings)),
            "page": data.get("page", page_int),
            "filings": simplified,
        }
        text = json.dumps(payload, indent=2)
        self._store[f"edgar_search:{query}"] = text
        self._store["edgar_search:last"] = text
        return text

    @cls_tool
    def parse_html_page(self, url: str) -> str:
        """Fetch a webpage and return a cleaned text rendering."""
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "agent-scaling/finance-agent",
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "*/*;q=0.8"
                ),
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw_bytes = resp.read()
        except Exception as exc:
            return f"ERROR: Failed to fetch url={url}: {exc}"

        raw = raw_bytes.decode("utf-8", errors="replace")
        text = _strip_html(raw)
        if not text:
            return (
                f"ERROR: No text extracted from url={url} "
                f"(content-type={content_type})"
            )

        max_store_chars = 200_000
        stored_text = text[:max_store_chars]
        self._store[url] = stored_text
        self._store["parse_html_page:last_url"] = url
        self._store["parse_html_page:last"] = stored_text

        max_return_chars = 10_000
        snippet = stored_text[:max_return_chars]
        if len(stored_text) > max_return_chars:
            snippet += "\n\n[TRUNCATED: stored full page text for retrieval]"
        return snippet

    @cls_tool
    def retrieve_information(
        self,
        prompt: str,
        input_character_ranges: Optional[Dict[str, List[int]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Fill {{key}} placeholders from stored tool outputs.

        This is a lightweight retrieval step; it does not call an LLM.
        """

        def replace(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            value = self._store.get(key)
            if value is None:
                return f"<MISSING:{key}>"
            if input_character_ranges and key in input_character_ranges:
                try:
                    start, end = input_character_ranges[key]
                    value = value[start:end]
                except Exception:
                    pass
            return value[:8000]

        del system_prompt  # Included for API compatibility.
        filled = re.sub(r"\\{\\{([^}]+)\\}\\}", replace, prompt)
        self._store["retrieve_information:last"] = filled
        logger.debug(
            "retrieve_information produced %d chars (store keys=%d)",
            len(filled),
            len(self._store),
        )
        return filled

    @cls_tool
    def done(self, answer: str, confidence_score: int = 100) -> str:
        """Finish the task and return the final answer with a confidence score."""
        self.is_done = True
        self.success = True
        return f"{answer}\\t{confidence_score}"
