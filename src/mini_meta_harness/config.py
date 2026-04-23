"""Environment, client construction, and model identifiers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

MOONSHOT_BASE_URL_DEFAULT = "https://api.moonshot.ai/anthropic"

# Default model ids — update these when Moonshot refreshes their catalog.
# See README §22 (open questions).
DEFAULT_PROPOSER_MODEL = "kimi-k2-0711-preview"
DEFAULT_TARGET_MODEL = "kimi-k2-0711-preview"


@dataclass(frozen=True)
class Endpoints:
    moonshot_api_key: str | None
    moonshot_base_url: str
    anthropic_api_key: str | None


def load_endpoints() -> Endpoints:
    load_dotenv()
    return Endpoints(
        moonshot_api_key=os.environ.get("MOONSHOT_API_KEY"),
        moonshot_base_url=os.environ.get("MOONSHOT_BASE_URL", MOONSHOT_BASE_URL_DEFAULT),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


def make_moonshot_client(endpoints: Endpoints | None = None):
    """Return an Anthropic SDK client pointed at Moonshot's compatible endpoint."""
    from anthropic import Anthropic

    endpoints = endpoints or load_endpoints()
    if not endpoints.moonshot_api_key:
        raise RuntimeError(
            "MOONSHOT_API_KEY not set. Copy .env.example to .env and fill it in, "
            "or run with --mock for a no-API-call dry run."
        )
    return Anthropic(
        api_key=endpoints.moonshot_api_key,
        base_url=endpoints.moonshot_base_url,
    )


def make_target_client(target_model: str, endpoints: Endpoints | None = None):
    """Pick the right client for the target model.

    Kimi models go through Moonshot; Claude models go through Anthropic proper.
    """
    from anthropic import Anthropic

    endpoints = endpoints or load_endpoints()
    if target_model.startswith("claude"):
        if not endpoints.anthropic_api_key:
            raise RuntimeError(
                f"Target model {target_model!r} needs ANTHROPIC_API_KEY set."
            )
        return Anthropic(api_key=endpoints.anthropic_api_key)
    return make_moonshot_client(endpoints)
