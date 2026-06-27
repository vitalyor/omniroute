"""OmniRoute web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`.
All search/extract requests are proxied through a local OmniRoute gateway.

Two capabilities advertised:

- ``supports_search()``  -> True (OmniRoute ``/v1/search``)
- ``supports_extract()`` -> True (OmniRoute ``/v1/web/fetch``)

Config keys this provider responds to::

    web:
      search_backend: "omniroute"
      extract_backend: "omniroute"
      backend: "omniroute"

Env vars::

    OMNIROUTE_URL=http://192.168.10.210:20129   # required
    OMNIROUTE_SEARCH_API_KEY=sk-...              # required

    # Optional — comma-separated provider chains in priority order.
    # Providers not listed here are appended in default order.
    OMNIROUTE_SEARCH_CHAIN=brave-search,exa-search,tavily-search
    OMNIROUTE_FETCH_CHAIN=firecrawl,jina-reader,tavily-search
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default chains
# ---------------------------------------------------------------------------

_DEFAULT_SEARCH_CHAIN = [
    "brave-search",
    "exa-search",
    "tavily-search",
    "serper-search",
    "youcom-search",
    "searxng-search",
    "ollama-search",
    "duckduckgo-free",
    "google-pse-search",
    "searchapi-search",
    "linkup-search",
    "zai-search",
    "perplexity-search",
]

_DEFAULT_FETCH_CHAIN = [
    "firecrawl",
    "jina-reader",
    "tavily-search",
]

_ALL_SEARCH_PROVIDERS = ", ".join(_DEFAULT_SEARCH_CHAIN)
_ALL_FETCH_PROVIDERS = ", ".join(_DEFAULT_FETCH_CHAIN)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(name: str) -> str:
    """Read env var from Hermes config-aware env, falling back to process env."""
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value(name)
    except Exception:
        val = None
    if val is None:
        val = os.getenv(name, "")
    return (val or "").strip()


def _omniroute_url() -> str:
    return _env("OMNIROUTE_URL").rstrip("/") or "http://192.168.10.210:20128"


def _api_key() -> str:
    return _env("OMNIROUTE_SEARCH_API_KEY")


def _headers() -> Dict[str, str]:
    key = _api_key()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _build_chain(env_var: str, defaults: List[str]) -> List[str]:
    """Merge user-specified chain from *env_var* with *defaults*.

    User's providers keep their order; any default provider not mentioned
    by the user is appended at the end in its original default order.
    """
    raw = _env(env_var)
    if not raw:
        return list(defaults)
    user = [p.strip() for p in raw.split(",") if p.strip()]
    seen = set(user)
    rest = [p for p in defaults if p not in seen]
    return user + rest


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OmniRouteWebSearchProvider(WebSearchProvider):
    """Search + extract via a local OmniRoute gateway."""

    def __init__(self) -> None:
        self._search_providers: List[str] | None = None

    # -- identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "omniroute"

    @property
    def display_name(self) -> str:
        return "OmniRoute"

    # -- availability -------------------------------------------------------

    def is_available(self) -> bool:
        """Return True when both OMNIROUTE_URL and OMNIROUTE_SEARCH_API_KEY are set."""
        return bool(_omniroute_url() and _api_key())

    # -- capabilities -------------------------------------------------------

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    # -- provider discovery -------------------------------------------------

    def get_search_providers(self) -> List[str]:
        """Return the list of search providers known to OmniRoute's catalog."""
        if self._search_providers is not None:
            return self._search_providers

        base = _omniroute_url()
        headers = _headers()
        if not headers:
            self._search_providers = []
            return self._search_providers

        try:
            resp = httpx.get(f"{base}/v1/search", headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            providers = [
                p["id"]
                for p in data.get("data", [])
                if isinstance(p, dict) and p.get("id")
            ]
            self._search_providers = providers
            logger.info(
                "Discovered %d search providers from OmniRoute", len(providers)
            )
        except Exception as exc:
            logger.warning("Could not fetch search providers from OmniRoute: %s", exc)
            self._search_providers = []

        return self._search_providers

    def get_fetch_providers(self) -> List[str]:
        """Return the list of fetch (extract) providers known to OmniRoute."""
        return ["firecrawl", "jina-reader", "tavily-search"]

    # -- chain helpers ------------------------------------------------------

    @staticmethod
    def get_search_chain() -> List[str]:
        """Return the search provider chain (user override ∪ defaults)."""
        return _build_chain("OMNIROUTE_SEARCH_CHAIN", _DEFAULT_SEARCH_CHAIN)

    @staticmethod
    def get_fetch_chain() -> List[str]:
        """Return the fetch provider chain (user override ∪ defaults)."""
        return _build_chain("OMNIROUTE_FETCH_CHAIN", _DEFAULT_FETCH_CHAIN)

    # -- search -------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search via OmniRoute, falling back through the chain."""
        base = _omniroute_url()
        headers = _headers()
        if not headers:
            return {"success": False, "error": "OMNIROUTE_SEARCH_API_KEY is not set"}

        chain = self.get_search_chain()
        last_error = "No search providers configured"

        for provider in chain:
            try:
                logger.info(
                    "OmniRoute search via %s: '%s' (limit=%d)",
                    provider, query, limit,
                )
                resp = httpx.post(
                    f"{base}/v1/search",
                    json={
                        "query": query,
                        "max_results": min(limit, 20),
                        "provider": provider,
                    },
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()
                raw = resp.json()

                # OmniRoute returns errors as a top-level error key
                if "error" in raw:
                    last_error = f"{provider}: {raw['error']}"
                    logger.warning("OmniRoute search %s failed: %s", provider, last_error)
                    continue

                results = raw.get("results", [])
                if not results:
                    logger.info("OmniRoute search %s returned no results, trying next", provider)
                    continue

                web_results = []
                for i, r in enumerate(results):
                    web_results.append(
                        {
                            "title": str(r.get("title", "")),
                            "url": str(r.get("url", "")),
                            "description": str(
                                r.get("snippet") or r.get("content") or ""
                            ),
                            "position": i + 1,
                        }
                    )

                return {"success": True, "data": {"web": web_results}}

            except httpx.HTTPStatusError as exc:
                last_error = f"{provider}: HTTP {exc.response.status_code}"
                logger.warning("OmniRoute search %s error: %s", provider, last_error)
                continue
            except httpx.RequestError as exc:
                last_error = f"{provider}: connection error — {exc}"
                logger.warning("OmniRoute search %s error: %s", provider, last_error)
                continue
            except Exception as exc:
                last_error = f"{provider}: {exc}"
                logger.warning("OmniRoute search %s error: %s", provider, last_error)
                continue

        return {"success": False, "error": f"All search providers failed: {last_error}"}

    # -- extract ------------------------------------------------------------

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from URLs via OmniRoute, falling back through the chain."""
        base = _omniroute_url()
        headers = _headers()
        if not headers:
            return [
                {"url": u, "title": "", "content": "", "error": "OMNIROUTE_SEARCH_API_KEY is not set"}
                for u in urls
            ]

        chain = self.get_fetch_chain()
        documents: List[Dict[str, Any]] = []

        for url in urls:
            doc = None
            for provider in chain:
                try:
                    logger.info("OmniRoute extract via %s: %s", provider, url)
                    resp = httpx.post(
                        f"{base}/v1/web/fetch",
                        json={"url": url, "format": "markdown", "provider": provider},
                        headers=headers,
                        timeout=60,
                    )
                    resp.raise_for_status()
                    raw = resp.json()

                    if "error" in raw:
                        logger.warning(
                            "OmniRoute extract %s error: %s", provider, raw["error"]
                        )
                        continue

                    content = str(raw.get("content", "") or raw.get("markdown", "") or "")
                    title = str(raw.get("title") or "")
                    resolved_url = str(raw.get("url", url))

                    doc = {
                        "url": resolved_url,
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": {
                            "sourceURL": resolved_url,
                            "title": title,
                            "provider": str(raw.get("provider", provider)),
                        },
                    }
                    break  # got a result from this provider

                except (httpx.HTTPStatusError, httpx.RequestError, Exception) as exc:
                    logger.warning(
                        "OmniRoute extract %s failed for %s: %s", provider, url, exc
                    )
                    continue

            if doc is None:
                doc = {
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": f"All fetch providers failed for {url}",
                }
            documents.append(doc)

        return documents

    # -- setup UI -----------------------------------------------------------

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OmniRoute",
            "badge": "self-hosted · gateway",
            "tag": (
                "Local AI gateway. Set OMNIROUTE_URL and OMNIROUTE_SEARCH_API_KEY. "
                "Optionally customise provider chains via OMNIROUTE_SEARCH_CHAIN "
                "and OMNIROUTE_FETCH_CHAIN."
            ),
            "env_vars": [
                {
                    "key": "OMNIROUTE_URL",
                    "prompt": "OmniRoute base URL (e.g. http://192.168.10.210:20128)",
                    "url": "",
                },
                {
                    "key": "OMNIROUTE_SEARCH_API_KEY",
                    "prompt": "OmniRoute search API key",
                    "url": "",
                },
                {
                    "key": "OMNIROUTE_SEARCH_CHAIN",
                    "prompt": (
                        f"Search provider priority (comma-separated). "
                        f"Known: {_ALL_SEARCH_PROVIDERS}"
                    ),
                    "url": "",
                },
                {
                    "key": "OMNIROUTE_FETCH_CHAIN",
                    "prompt": (
                        f"Fetch provider priority (comma-separated). "
                        f"Known: {_ALL_FETCH_PROVIDERS}"
                    ),
                    "url": "",
                },
            ],
        }

    # -- tests ---------------------------------------------------------------

    def test_connection(self) -> Dict[str, Any]:
        """Check basic connectivity to OmniRoute.

        Pings the search catalog endpoint.  Returns OK when the API is
        reachable and the API key is valid.
        """
        base = _omniroute_url()
        headers = _headers()
        if not headers:
            return {"status": "error", "error": "OMNIROUTE_SEARCH_API_KEY is not set"}

        try:
            resp = httpx.get(f"{base}/v1/search", headers=headers, timeout=15)
            if resp.is_success:
                return {"status": "ok", "url": base, "api_key_valid": True}
            if resp.status_code == 401 or resp.status_code == 403:
                return {
                    "status": "error",
                    "url": base,
                    "api_key_valid": False,
                    "detail": f"HTTP {resp.status_code} — invalid API key",
                }
            return {
                "status": "error",
                "url": base,
                "api_key_valid": True,
                "detail": f"HTTP {resp.status_code}",
            }
        except httpx.RequestError as exc:
            return {
                "status": "error",
                "url": base,
                "api_key_valid": False,
                "detail": f"Cannot reach {base}: {exc}",
            }
        except Exception as exc:
            return {
                "status": "error",
                "url": base,
                "api_key_valid": False,
                "detail": str(exc),
            }

    def test_providers(self, chain_type: str = "search") -> Dict[str, Any]:
        """Test each provider in the chain — returns who works and who doesn't.

        *chain_type* is ``"search"`` or ``"fetch"``.
        This makes one real request per provider, so be mindful of rate limits.
        """
        base = _omniroute_url()
        headers = _headers()
        if not headers:
            return {"status": "error", "error": "API key not set"}

        chain = (
            self.get_search_chain() if chain_type == "search"
            else self.get_fetch_chain()
        )
        results: List[Dict[str, Any]] = []

        for provider in chain:
            try:
                if chain_type == "search":
                    resp = httpx.post(
                        f"{base}/v1/search",
                        json={"query": "test", "max_results": 1, "provider": provider},
                        headers=headers,
                        timeout=30,
                    )
                else:
                    resp = httpx.post(
                        f"{base}/v1/web/fetch",
                        json={"url": "https://example.com", "provider": provider},
                        headers=headers,
                        timeout=30,
                    )

                if resp.is_success:
                    data = resp.json()
                    if "error" in data:
                        results.append({
                            "provider": provider,
                            "status": "error",
                            "detail": data["error"],
                        })
                    else:
                        results.append({"provider": provider, "status": "ok"})
                else:
                    results.append({
                        "provider": provider,
                        "status": "error",
                        "detail": f"HTTP {resp.status_code}",
                    })
            except Exception as exc:
                results.append({
                    "provider": provider,
                    "status": "error",
                    "detail": str(exc),
                })

        return {"status": "done", "chain_type": chain_type, "results": results}

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _get_configured_backend(capability: str) -> str | None:
        """Read the configured backend for *capability* (``search`` | ``extract``)."""
        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            web = cfg.get("web") if isinstance(cfg.get("web"), dict) else {}
            key = web.get(f"{capability}_backend") or web.get("backend") or ""
            return key.strip().lower() or None
        except Exception:
            return None
