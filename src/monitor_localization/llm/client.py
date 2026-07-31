"""OpenAI-compatible client for the monitor LLM.

Uses strict structured outputs so a verdict is schema-enforced rather than
regex-scraped out of prose. Wraps every call in the response cache, and treats
output truncation as an explicit error instead of a parse failure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from monitor_localization.config import ModelConfig
from monitor_localization.llm.cache import ResponseCache, cache_key
from monitor_localization.utils.logging import get_logger

logger = get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Long transcripts make for large prompts; the default SDK timeout is far too
# tight for a 300k-token request. Scaled with input size rather than fixed.
BASE_TIMEOUT_SECONDS = 120.0
SECONDS_PER_1K_TOKENS = 0.6
MAX_TIMEOUT_SECONDS = 900.0

# Rough chars-per-token for MALT transcripts, measured with o200k_base on 36 real
# runs (aggregate 4.40). Used only for the pre-flight size guard, so an estimate
# is sufficient and avoids a tokenizer dependency on the hot path.
CHARS_PER_TOKEN = 4.4


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def _timeout_for(prompt_chars: int) -> float:
    tokens = estimate_tokens_from_chars(prompt_chars)
    return min(
        BASE_TIMEOUT_SECONDS + (tokens / 1000.0) * SECONDS_PER_1K_TOKENS,
        MAX_TIMEOUT_SECONDS,
    )


def estimate_tokens_from_chars(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN)


class MonitorLLMError(RuntimeError):
    """Base class for monitor inference failures."""


class MissingAPIKeyError(MonitorLLMError):
    pass


class TruncatedResponseError(MonitorLLMError):
    """The model hit the output cap before completing its JSON.

    Raised rather than swallowed: with strict structured outputs a truncated
    response is unparseable, so silently treating it as a low score would put
    fabricated data into the confusion matrix.
    """


class RefusalError(MonitorLLMError):
    """The model declined to answer. Distinct from a low suspiciousness score."""


class TranscriptTooLargeError(MonitorLLMError):
    """The request exceeds what the account can send in a single call.

    MALT's largest transcripts run to ~480k tokens, above a 400k tokens-per-minute
    account limit, so no amount of retrying will get them through as one request.
    Detected before sending rather than as a 429, so the run is not billed and the
    reason is unambiguous.

    This is a property of the *global* architecture, not of the transcript: the
    chunk monitor sends the same content in pieces that each fit comfortably. Runs
    that trip this are worth reporting rather than quietly dropping.
    """


class UpstreamError(MonitorLLMError):
    """An SDK or transport failure (connection dropped, 429, 5xx, timeout).

    Wrapped so a single bad transcript is recorded as a failed row rather than
    aborting a 562-run stage partway through.
    """


@dataclass(slots=True)
class Usage:
    """Token accounting, aggregated across a stage to project cost."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_calls: int = 0
    api_calls: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_calls += other.cached_calls
        self.api_calls += other.api_calls

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_calls": self.cached_calls,
            "api_calls": self.api_calls,
        }


@dataclass
class Completion:
    """One structured response plus the metadata worth persisting."""

    parsed: BaseModel
    usage: Usage
    from_cache: bool
    model: str
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class MonitorClient:
    """Thin wrapper over the Chat Completions API with caching and validation."""

    def __init__(
        self,
        cfg: ModelConfig | None = None,
        cache: ResponseCache | None = None,
        client: Any | None = None,
    ) -> None:
        self.cfg = cfg or ModelConfig()
        self.cache = cache if cache is not None else ResponseCache()
        self._client = client
        self.usage = Usage()
        # Per-request input budget. Defaults below the account's 400k TPM cap so
        # a single request can actually be admitted; 0 disables the guard.
        self.max_input_tokens = getattr(self.cfg, "max_input_tokens", 350_000)

    # -- client construction -------------------------------------------------

    @property
    def client(self) -> Any:
        """Lazily construct the SDK client so tests and --dry-run need no key."""
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise MissingAPIKeyError(
                    "OPENAI_API_KEY is not set. Add it to .env (see .env.example) "
                    "or export it in your shell."
                )
            self._client = OpenAI(
                api_key=api_key,
                max_retries=self.cfg.max_retries,
            )
        return self._client

    # -- requests ------------------------------------------------------------

    def _request_payload(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> dict[str, Any]:
        """Everything that can affect the response, for both the call and the key."""
        return {
            "model": self.cfg.name,
            "temperature": self.cfg.temperature,
            "max_completion_tokens": self.cfg.max_tokens,
            "schema": schema.model_json_schema(),
            "system": system,
            "user": user,
        }

    def complete(
        self,
        system: str,
        user: str,
        schema: type[SchemaT],
    ) -> Completion:
        """Return a schema-validated completion, from cache when available."""
        payload = self._request_payload(system, user, schema)
        key = cache_key(payload)

        # Pre-flight size guard. Checked before the cache so an oversized request
        # is never silently satisfied by a stale entry either.
        estimated = estimate_tokens(system) + estimate_tokens(user)
        limit = self.max_input_tokens
        if limit and estimated > limit:
            raise TranscriptTooLargeError(
                f"Estimated {estimated:,} input tokens exceeds the {limit:,}-token "
                "per-request budget, so this cannot be sent as a single call. "
                "The chunk monitor can still score this transcript."
            )

        cached = self.cache.get(key)
        if cached is not None:
            usage = Usage(
                prompt_tokens=cached.get("prompt_tokens", 0),
                completion_tokens=cached.get("completion_tokens", 0),
                cached_calls=1,
            )
            self.usage.add(usage)
            return Completion(
                parsed=schema.model_validate(cached["parsed"]),
                usage=usage,
                from_cache=True,
                model=cached.get("model", self.cfg.name),
                finish_reason=cached.get("finish_reason"),
            )

        try:
            response = self.client.chat.completions.parse(
                model=self.cfg.name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=schema,
                temperature=self.cfg.temperature,
                max_completion_tokens=self.cfg.max_tokens,
                timeout=_timeout_for(len(system) + len(user)),
            )
        except MonitorLLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - any SDK/transport failure
            # Recorded as a failed row rather than aborting the stage. The SDK has
            # already exhausted its own retries by this point.
            raise UpstreamError(f"{type(exc).__name__}: {exc}") from exc

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)

        if finish_reason == "length":
            raise TruncatedResponseError(
                f"Model hit the {self.cfg.max_tokens}-token output cap before finishing "
                "its JSON, so the verdict cannot be parsed. Raise model.max_tokens or "
                "tighten the reasoning instruction in the prompt."
            )
        if getattr(choice.message, "refusal", None):
            raise RefusalError(f"Model refused: {choice.message.refusal}")

        parsed = choice.message.parsed
        if parsed is None:
            raise MonitorLLMError(
                f"No parsed content returned (finish_reason={finish_reason!r})."
            )

        raw_usage = getattr(response, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            api_calls=1,
        )
        self.usage.add(usage)

        self.cache.put(
            key,
            {
                "parsed": parsed.model_dump(mode="json"),
                "model": getattr(response, "model", self.cfg.name),
                "finish_reason": finish_reason,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
        )

        return Completion(
            parsed=parsed,
            usage=usage,
            from_cache=False,
            model=getattr(response, "model", self.cfg.name),
            finish_reason=finish_reason,
        )
