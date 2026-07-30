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

# Long transcripts make for large prompts; the default SDK timeout is tight for
# them. Not a config knob until there's a reason to vary it per experiment.
REQUEST_TIMEOUT_SECONDS = 120.0


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

        response = self.client.chat.completions.parse(
            model=self.cfg.name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
            temperature=self.cfg.temperature,
            max_completion_tokens=self.cfg.max_tokens,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

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
