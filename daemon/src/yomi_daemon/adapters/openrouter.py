"""OpenRouter chat-completions adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Protocol, cast

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from yomi_daemon.adapters.base import (
    AdapterConstructionError,
    BasePolicyAdapter,
    PolicyAdapterMetadata,
    PolicyDecisionResult,
    PromptTrace,
    metadata_from_policy_config,
)
from yomi_daemon.orchestrator import resolve_policy_decision
from yomi_daemon.prompt import (
    DEFAULT_PROMPT_VERSION,
    decision_output_json_schema,
    render_prompt,
)
from yomi_daemon.protocol import ActionDecision, DecisionRequest, FallbackMode, JsonObject
from yomi_daemon.response_parser import ResponseParsingConfig

if TYPE_CHECKING:
    from yomi_daemon.config import PolicyConfig


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterTransport(Protocol):
    async def create_completion(
        self,
        *,
        api_key: str,
        payload: JsonObject,
        timeout_ms: int,
        http_referer: str | None,
        title: str | None,
        categories: str | None = None,
    ) -> JsonObject: ...


class OpenRouterProviderError(RuntimeError):
    """Raised when the upstream OpenRouter request fails before parsing."""


class OpenRouterProviderTraceError(OpenRouterProviderError):
    """Provider error that preserves request/response payloads for prompt traces."""

    def __init__(
        self,
        message: str,
        *,
        request_payload: JsonObject,
        response_payload: JsonObject | None = None,
    ) -> None:
        super().__init__(message)
        self.request_payload = request_payload
        self.response_payload = response_payload


@dataclass(frozen=True, slots=True)
class _ProviderCallResult:
    output: object
    request_payload: JsonObject
    response_payload: JsonObject


class DefaultOpenRouterTransport:
    def __init__(self) -> None:
        self._client_by_key_and_headers: dict[tuple[str, ...], AsyncOpenAI] = {}

    async def create_completion(
        self,
        *,
        api_key: str,
        payload: JsonObject,
        timeout_ms: int,
        http_referer: str | None,
        title: str | None,
        categories: str | None = None,
        base_url: str | None = None,
    ) -> JsonObject:
        try:
            response = await self._client_for_config(
                api_key=api_key,
                http_referer=http_referer,
                title=title,
                categories=categories,
                base_url=base_url,
            ).chat.completions.create(
                **cast(Any, payload),
                timeout=timeout_ms / 1000,
            )
        except APIConnectionError as exc:
            raise OSError(f"OpenRouter transport failed: {exc}") from exc
        except APIStatusError as exc:
            from yomi_daemon.redact import sanitize_provider_error

            import logging

            logging.getLogger("yomi_daemon.adapters.openrouter").warning(
                "OpenRouter API error %d: %s", exc.status_code, sanitize_provider_error(exc)
            )
            raise OpenRouterProviderError(sanitize_provider_error(exc)) from exc

        dumped = response.model_dump(mode="json")
        if not isinstance(dumped, dict):
            raise OpenRouterProviderError("OpenRouter SDK response did not serialize to an object")
        return cast(JsonObject, dumped)

    def _client_for_config(
        self,
        *,
        api_key: str,
        http_referer: str | None,
        title: str | None,
        categories: str | None = None,
        base_url: str | None = None,
    ) -> AsyncOpenAI:
        resolved_base_url = base_url or _OPENROUTER_BASE_URL
        cache_key = (api_key, http_referer, title, categories, resolved_base_url)
        client = self._client_by_key_and_headers.get(cache_key)
        if client is None:
            default_headers: dict[str, str] = {}
            if http_referer is not None:
                default_headers["HTTP-Referer"] = http_referer
            if title is not None:
                default_headers["X-Title"] = title
            if categories is not None:
                default_headers["X-OpenRouter-Categories"] = categories
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=resolved_base_url,
                default_headers=default_headers or None,
                max_retries=0,
            )
            self._client_by_key_and_headers[cache_key] = client
        return client


class OpenRouterAdapter(BasePolicyAdapter):
    def __init__(
        self,
        *,
        metadata: PolicyAdapterMetadata,
        api_key: str | None,
        decision_timeout_ms: int,
        fallback_mode: FallbackMode,
        parsing_config: ResponseParsingConfig,
        temperature: float | None,
        max_tokens: int | None,
        http_referer: str | None,
        title: str | None,
        categories: str | None = None,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
        transport: OpenRouterTransport | None = None,
        default_trace_seed: int = 0,
    ) -> None:
        super().__init__(metadata=metadata, default_trace_seed=default_trace_seed)
        self._api_key = api_key
        self._decision_timeout_ms = decision_timeout_ms
        self._fallback_mode = fallback_mode
        self._parsing_config = parsing_config
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._http_referer = http_referer
        self._title = title
        self._categories = categories
        self._reasoning_effort = reasoning_effort
        self._base_url = base_url
        self._transport = transport or DefaultOpenRouterTransport()

    async def decide(self, request: DecisionRequest) -> ActionDecision:
        return (await self.decide_with_trace(request)).decision

    async def decide_with_trace(self, request: DecisionRequest) -> PolicyDecisionResult:
        rendered_prompt = render_prompt(
            request,
            configured_prompt_version=self.metadata.prompt_version,
            policy_id=self.id,
        )
        request_attempts: list[JsonObject] = []
        response_attempts: list[JsonObject] = []

        async def decision_provider(_: DecisionRequest) -> object:
            try:
                call = await self._call_provider(
                    prompt_text=rendered_prompt.prompt_text,
                    request=request,
                    attempt_kind="initial",
                )
            except OpenRouterProviderTraceError as exc:
                request_attempts.append(deepcopy(exc.request_payload))
                if exc.response_payload is not None:
                    response_attempts.append(deepcopy(exc.response_payload))
                raise

            request_attempts.append(call.request_payload)
            response_attempts.append(call.response_payload)
            return call.output

        async def correction_callback(correction_prompt: str) -> object:
            try:
                call = await self._call_provider(
                    prompt_text=correction_prompt,
                    request=request,
                    attempt_kind="correction",
                )
            except OpenRouterProviderTraceError as exc:
                request_attempts.append(deepcopy(exc.request_payload))
                if exc.response_payload is not None:
                    response_attempts.append(deepcopy(exc.response_payload))
                raise

            request_attempts.append(call.request_payload)
            response_attempts.append(call.response_payload)
            return call.output

        result = await resolve_policy_decision(
            request,
            decision_provider=decision_provider,
            fallback_mode=self._fallback_mode,
            correction_callback=correction_callback,
            parsing_config=self._parsing_config,
            timeout_ms=self._decision_timeout_ms,
        )

        tokens_in_total, tokens_out_total = _sum_usage_tokens(response_attempts)
        decision = _with_provider_metadata(
            result.decision,
            policy_id=self.id,
            tokens_in=tokens_in_total,
            tokens_out=tokens_out_total,
        )
        return PolicyDecisionResult(
            decision=decision,
            prompt_trace=PromptTrace(
                prompt_text=rendered_prompt.prompt_text,
                prompt_version=rendered_prompt.prompt_version,
                provider_request={"attempts": deepcopy(request_attempts)},
                provider_response={"attempts": deepcopy(response_attempts)},
            ),
        )

    async def _call_provider(
        self,
        *,
        prompt_text: str,
        request: DecisionRequest,
        attempt_kind: str,
    ) -> _ProviderCallResult:
        if not self._api_key:
            raise OpenRouterProviderError("OPENROUTER_API_KEY is not configured for this policy")

        provider_request = self._build_request_payload(
            prompt_text=prompt_text,
            request=request,
            attempt_kind=attempt_kind,
        )
        provider_response = await self._transport.create_completion(
            api_key=self._api_key,
            payload=provider_request,
            timeout_ms=self._decision_timeout_ms,
            http_referer=self._http_referer,
            title=self._title,
            categories=self._categories,
            base_url=self._base_url,
        )
        try:
            output = _extract_response_output(provider_response)
        except OpenRouterProviderError as exc:
            raise OpenRouterProviderTraceError(
                str(exc),
                request_payload=provider_request,
                response_payload=provider_response,
            ) from exc
        return _ProviderCallResult(
            output=output,
            request_payload=provider_request,
            response_payload=provider_response,
        )

    def _build_request_payload(
        self,
        *,
        prompt_text: str,
        request: DecisionRequest,
        attempt_kind: str,
    ) -> JsonObject:
        prompt_version = (
            request.prompt_version or self.metadata.prompt_version or DEFAULT_PROMPT_VERSION
        )
        payload: JsonObject = {
            "model": cast(str, self.metadata.model),
            "messages": [{"role": "user", "content": prompt_text}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "yomi_action_decision",
                    "strict": False,
                    "schema": decision_output_json_schema(),
                },
            },
            "user": f"{request.match_id}:{request.turn_id}:{attempt_kind}",
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if self._reasoning_effort is not None:
            payload["extra_body"] = {"reasoning": {"effort": self._reasoning_effort}}
        if request.trace_seed is not None:
            payload["seed"] = request.trace_seed
        payload["metadata"] = {"prompt_version": prompt_version, "policy_id": self.id}
        return payload


def build_openrouter_adapter(
    policy_id: str,
    policy: "PolicyConfig",
    *,
    decision_timeout_ms: int,
    fallback_mode: FallbackMode,
    transport: OpenRouterTransport | None = None,
    default_trace_seed: int = 0,
) -> OpenRouterAdapter:
    if policy.model is None:
        raise AdapterConstructionError(f"openrouter policy {policy_id!r} must define a model")
    metadata = metadata_from_policy_config(policy_id, policy)
    parsing_config = ResponseParsingConfig.from_policy_options(policy.options)
    options = policy.options
    return OpenRouterAdapter(
        metadata=metadata,
        api_key=policy.credential.value,
        decision_timeout_ms=decision_timeout_ms,
        fallback_mode=fallback_mode,
        parsing_config=parsing_config,
        temperature=policy.temperature,
        max_tokens=policy.max_tokens,
        http_referer=_optional_string_option(options, "http_referer"),
        title=_optional_string_option(options, "title"),
        categories=_optional_string_option(options, "categories"),
        reasoning_effort=_optional_string_option(options, "reasoning_effort"),
        base_url=_optional_string_option(options, "base_url"),
        transport=transport,
        default_trace_seed=default_trace_seed,
    )


def _optional_string_option(options: Mapping[str, object], key: str) -> str | None:
    value = options.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AdapterConstructionError(
            f"openrouter policy option {key!r} must be a non-empty string"
        )
    return value


def _with_provider_metadata(
    decision: ActionDecision,
    *,
    policy_id: str,
    tokens_in: int,
    tokens_out: int,
) -> ActionDecision:
    return ActionDecision(
        match_id=decision.match_id,
        turn_id=decision.turn_id,
        action=decision.action,
        data=decision.data,
        extra=decision.extra,
        policy_id=decision.policy_id if decision.fallback_reason is not None else policy_id,
        latency_ms=decision.latency_ms,
        tokens_in=decision.tokens_in if decision.tokens_in is not None else tokens_in or None,
        tokens_out=decision.tokens_out if decision.tokens_out is not None else tokens_out or None,
        reasoning=decision.reasoning,
        notes=decision.notes,
        fallback_reason=decision.fallback_reason,
    )


def _sum_usage_tokens(response_attempts: list[JsonObject]) -> tuple[int, int]:
    tokens_in_total = 0
    tokens_out_total = 0
    for response in response_attempts:
        usage_raw = response.get("usage")
        if not isinstance(usage_raw, Mapping):
            continue
        usage = cast(Mapping[str, object], usage_raw)
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            tokens_in_total += prompt_tokens
        if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
            tokens_out_total += completion_tokens
    return tokens_in_total, tokens_out_total


def _extract_response_output(response_payload: JsonObject) -> object:
    choices_raw = response_payload.get("choices")
    if not isinstance(choices_raw, list) or not choices_raw:
        raise OpenRouterProviderError("OpenRouter response did not include choices")

    first_choice = choices_raw[0]
    if not isinstance(first_choice, Mapping):
        raise OpenRouterProviderError("OpenRouter choice payload was not an object")

    message_raw = first_choice.get("message")
    if not isinstance(message_raw, Mapping):
        raise OpenRouterProviderError("OpenRouter response did not include a message")

    parsed = message_raw.get("parsed")
    if isinstance(parsed, Mapping):
        return cast(JsonObject, deepcopy(parsed))

    tool_calls_raw = message_raw.get("tool_calls")
    if isinstance(tool_calls_raw, list):
        for tool_call in tool_calls_raw:
            if not isinstance(tool_call, Mapping):
                continue
            function_raw = tool_call.get("function")
            if not isinstance(function_raw, Mapping):
                continue
            arguments = function_raw.get("arguments")
            if isinstance(arguments, str):
                try:
                    decoded = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, Mapping):
                    return cast(JsonObject, decoded)

    content = message_raw.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text_fragments: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            if isinstance(text, str):
                text_fragments.append(text)
        if text_fragments:
            return "".join(text_fragments)

    reasoning = message_raw.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        json_candidate = _extract_json_object(reasoning)
        if json_candidate is not None:
            return cast(JsonObject, json_candidate)

    raise OpenRouterProviderError("OpenRouter response did not include structured content")


def _extract_json_object(text: str) -> Mapping[str, object] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return cast(Mapping[str, object], parsed)
    return None
