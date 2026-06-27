"""OmniRoute web search + extraction plugin — user-installable.

Proxies ``web_search`` and ``web_extract`` through a local OmniRoute gateway,
giving Hermes access to OmniRoute's multi-provider search and fetch backends.

Configuration (in .env)::

    OMNIROUTE_URL=http://192.168.10.210:20129
    OMNIROUTE_SEARCH_API_KEY=***

To use, set in config.yaml::

    web:
      backend: omniroute

The plugin discovers available search providers at startup via
``GET /v1/search`` and caches them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _patch_web_tools() -> None:
    """Monkey-patch Hermes' web tool backend resolution to accept 'omniroute'.

    ``tools.web_tools._get_backend()`` only recognizes a hardcoded set of
    provider names.  Without this patch, ``web.backend: omniroute`` is ignored
    and falls through to the env-var cascade.  Also patches
    ``_is_backend_available`` so the per-capability override path
    (``web.search_backend`` / ``web.extract_backend``) works.

    The patch runs every time the plugin loads (i.e. every Hermes restart)
    and does not modify any source files — it survives upgrades.
    """
    try:
        import tools.web_tools as wt

        # ---- _get_backend -------------------------------------------------
        _orig_get_backend = wt._get_backend

        def _patched_get_backend() -> str:
            configured = (
                (wt._load_web_config().get("backend") or "").lower().strip()
            )
            if configured == "omniroute":
                return configured
            return _orig_get_backend()

        wt._get_backend = _patched_get_backend

        # ---- _is_backend_available ----------------------------------------
        _orig_is_available = wt._is_backend_available

        def _patched_is_available(backend: str) -> bool:
            if backend == "omniroute":
                try:
                    from hermes_cli.config import get_env_value

                    url = get_env_value("OMNIROUTE_URL")
                    key = get_env_value("OMNIROUTE_SEARCH_API_KEY")
                except Exception:
                    import os
                    url = os.getenv("OMNIROUTE_URL", "")
                    key = os.getenv("OMNIROUTE_SEARCH_API_KEY", "")
                return bool(url) and bool(key)
            return _orig_is_available(backend)

        wt._is_backend_available = _patched_is_available

        # ---- check_web_api_key --------------------------------------------
        _orig_check_key = wt.check_web_api_key

        def _patched_check_key() -> bool:
            configured = (
                (wt._load_web_config().get("backend") or "").lower().strip()
            )
            if configured == "omniroute":
                return _patched_is_available("omniroute")
            return _orig_check_key()

        wt.check_web_api_key = _patched_check_key

        logger.info("OmniRoute: patched web tool backend resolution")
    except Exception as exc:
        logger.warning("OmniRoute: failed to patch web tools (%s)", exc)


def register(ctx) -> None:
    """Register the OmniRoute web search/extract provider."""
    from .provider import OmniRouteWebSearchProvider

    _patch_web_tools()
    ctx.register_web_search_provider(OmniRouteWebSearchProvider())
