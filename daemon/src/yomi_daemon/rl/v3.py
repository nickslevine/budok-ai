"""RL v3 helpers for policy specs, leagues, benchmark suites, and promotion.

These utilities are used by RL scripts to:
- describe candidate/opponent policies in a config-compatible way,
- build weighted leagues for rollout sampling,
- evaluate checkpoints against benchmark suites,
- and maintain a simple champion registry.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

JsonDict = dict[str, Any]
SideMode = Literal["p1", "p2", "both"]
PlayerSide = Literal["p1", "p2"]
LeagueEntryKind = Literal["self", "policy"]

TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
DEFAULT_RL_PROMPT_VERSION = "strategic_rl_v1"
DEFAULT_SFT_SAMPLER = (
    "tinker://d3d1239e-cb3f-5cf5-b675-3cbb2cd928c2:train:0/sampler_weights/yomi-sft-cowboy-v1"
)
DEFAULT_RESPONSE_PARSER: JsonDict = {
    "enable_correction_retry": True,
    "max_correction_retries": 1,
}


@dataclass(frozen=True, slots=True)
class PolicySpec:
    policy_id: str
    provider: str
    model: str
    prompt_version: str = DEFAULT_RL_PROMPT_VERSION
    credential_env_var: str | None = None
    temperature: float | None = 0.7
    max_tokens: int | None = 256
    options: JsonDict = field(default_factory=dict)
    label: str | None = None

    def to_config_entry(self) -> JsonDict:
        entry: JsonDict = {
            "provider": self.provider,
            "model": self.model,
        }
        if self.prompt_version is not None:
            entry["prompt_version"] = self.prompt_version
        if self.credential_env_var is not None:
            entry["credential_env_var"] = self.credential_env_var
        if self.temperature is not None:
            entry["temperature"] = self.temperature
        if self.max_tokens is not None:
            entry["max_tokens"] = self.max_tokens
        if self.options:
            entry["options"] = deepcopy(self.options)
        return entry

    def to_dict(self) -> JsonDict:
        payload = self.to_config_entry()
        payload["policy_id"] = self.policy_id
        if self.label is not None:
            payload["label"] = self.label
        return payload

    @classmethod
    def from_dict(cls, raw: JsonDict, *, context: str = "policy") -> "PolicySpec":
        policy_id = raw.get("policy_id")
        provider = raw.get("provider")
        model = raw.get("model")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError(f"{context}.policy_id must be a non-empty string")
        if not isinstance(provider, str) or not provider:
            raise ValueError(f"{context}.provider must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{context}.model must be a non-empty string")
        prompt_version = raw.get("prompt_version", DEFAULT_RL_PROMPT_VERSION)
        if not isinstance(prompt_version, str) or not prompt_version:
            raise ValueError(f"{context}.prompt_version must be a non-empty string")
        credential_env_var = raw.get("credential_env_var")
        if credential_env_var is not None and not isinstance(credential_env_var, str):
            raise ValueError(f"{context}.credential_env_var must be a string or null")
        temperature = raw.get("temperature", 0.7)
        if temperature is not None and not isinstance(temperature, int | float):
            raise ValueError(f"{context}.temperature must be numeric or null")
        max_tokens = raw.get("max_tokens", 256)
        if max_tokens is not None and not isinstance(max_tokens, int):
            raise ValueError(f"{context}.max_tokens must be an int or null")
        options_raw = raw.get("options", {})
        if not isinstance(options_raw, dict):
            raise ValueError(f"{context}.options must be an object")
        label = raw.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError(f"{context}.label must be a string or null")
        return cls(
            policy_id=policy_id,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            credential_env_var=credential_env_var,
            temperature=float(temperature) if isinstance(temperature, int | float) else None,
            max_tokens=max_tokens,
            options=deepcopy(options_raw),
            label=label,
        )


@dataclass(frozen=True, slots=True)
class LeagueEntry:
    entry_id: str
    weight: float
    kind: LeagueEntryKind
    policy: PolicySpec | None = None
    label: str | None = None

    @property
    def trains_both_sides(self) -> bool:
        return self.kind == "self"

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "entry_id": self.entry_id,
            "weight": self.weight,
            "kind": self.kind,
        }
        if self.policy is not None:
            payload["policy"] = self.policy.to_dict()
        if self.label is not None:
            payload["label"] = self.label
        return payload

    @classmethod
    def from_dict(cls, raw: JsonDict, *, context: str = "league_entry") -> "LeagueEntry":
        entry_id = raw.get("entry_id")
        weight = raw.get("weight")
        kind = raw.get("kind")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError(f"{context}.entry_id must be a non-empty string")
        if not isinstance(weight, int | float) or weight <= 0:
            raise ValueError(f"{context}.weight must be > 0")
        if kind not in {"self", "policy"}:
            raise ValueError(f"{context}.kind must be 'self' or 'policy'")
        label = raw.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError(f"{context}.label must be a string or null")
        policy_raw = raw.get("policy")
        policy = None
        if kind == "policy":
            if not isinstance(policy_raw, dict):
                raise ValueError(f"{context}.policy must be present for policy league entries")
            policy = PolicySpec.from_dict(policy_raw, context=f"{context}.policy")
        return cls(
            entry_id=entry_id,
            weight=float(weight),
            kind=cast(LeagueEntryKind, kind),
            policy=policy,
            label=label,
        )


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    entries: tuple[LeagueEntry, ...]
    version: str = "v1"

    def to_dict(self) -> JsonDict:
        return {
            "version": self.version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, raw: JsonDict) -> "LeagueConfig":
        entries_raw = raw.get("entries")
        if not isinstance(entries_raw, list):
            raise ValueError("league.entries must be a list")
        entries = tuple(
            LeagueEntry.from_dict(cast(JsonDict, item), context=f"league.entries[{index}]")
            for index, item in enumerate(entries_raw)
            if isinstance(item, dict)
        )
        if not entries:
            raise ValueError("league.entries must not be empty")
        version = raw.get("version", "v1")
        if not isinstance(version, str):
            raise ValueError("league.version must be a string")
        return cls(entries=entries, version=version)


@dataclass(frozen=True, slots=True)
class SuiteEntry:
    entry_id: str
    opponent: PolicySpec
    num_matches: int = 10
    sides: SideMode = "both"
    weight: float = 1.0
    label: str | None = None

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "entry_id": self.entry_id,
            "opponent": self.opponent.to_dict(),
            "num_matches": self.num_matches,
            "sides": self.sides,
            "weight": self.weight,
        }
        if self.label is not None:
            payload["label"] = self.label
        return payload

    @classmethod
    def from_dict(cls, raw: JsonDict, *, context: str = "suite_entry") -> "SuiteEntry":
        entry_id = raw.get("entry_id")
        num_matches = raw.get("num_matches", 10)
        sides = raw.get("sides", "both")
        weight = raw.get("weight", 1.0)
        label = raw.get("label")
        opponent_raw = raw.get("opponent")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError(f"{context}.entry_id must be a non-empty string")
        if not isinstance(num_matches, int) or num_matches <= 0:
            raise ValueError(f"{context}.num_matches must be > 0")
        if sides not in {"p1", "p2", "both"}:
            raise ValueError(f"{context}.sides must be 'p1', 'p2', or 'both'")
        if not isinstance(weight, int | float) or weight <= 0:
            raise ValueError(f"{context}.weight must be > 0")
        if label is not None and not isinstance(label, str):
            raise ValueError(f"{context}.label must be a string or null")
        if not isinstance(opponent_raw, dict):
            raise ValueError(f"{context}.opponent must be an object")
        opponent = PolicySpec.from_dict(opponent_raw, context=f"{context}.opponent")
        return cls(
            entry_id=entry_id,
            opponent=opponent,
            num_matches=num_matches,
            sides=cast(SideMode, sides),
            weight=float(weight),
            label=label,
        )


@dataclass(frozen=True, slots=True)
class SuiteConfig:
    entries: tuple[SuiteEntry, ...]
    version: str = "v1"

    def to_dict(self) -> JsonDict:
        return {
            "version": self.version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, raw: JsonDict) -> "SuiteConfig":
        entries_raw = raw.get("entries")
        if not isinstance(entries_raw, list):
            raise ValueError("suite.entries must be a list")
        entries = tuple(
            SuiteEntry.from_dict(cast(JsonDict, item), context=f"suite.entries[{index}]")
            for index, item in enumerate(entries_raw)
            if isinstance(item, dict)
        )
        if not entries:
            raise ValueError("suite.entries must not be empty")
        version = raw.get("version", "v1")
        if not isinstance(version, str):
            raise ValueError("suite.version must be a string")
        return cls(entries=entries, version=version)


@dataclass(frozen=True, slots=True)
class SuiteSeedBankEntry:
    entry_id: str
    seeds_by_side: dict[PlayerSide, tuple[int, ...]]

    def to_dict(self) -> JsonDict:
        return {
            "entry_id": self.entry_id,
            "seeds_by_side": {side: list(seeds) for side, seeds in self.seeds_by_side.items()},
        }

    @classmethod
    def from_dict(
        cls, raw: JsonDict, *, context: str = "suite_seed_bank_entry"
    ) -> "SuiteSeedBankEntry":
        entry_id = raw.get("entry_id")
        seeds_by_side_raw = raw.get("seeds_by_side")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError(f"{context}.entry_id must be a non-empty string")
        if not isinstance(seeds_by_side_raw, dict):
            raise ValueError(f"{context}.seeds_by_side must be an object")

        seeds_by_side: dict[PlayerSide, tuple[int, ...]] = {}
        for side, seeds_raw in seeds_by_side_raw.items():
            if side not in {"p1", "p2"}:
                raise ValueError(f"{context}.seeds_by_side keys must be 'p1' or 'p2'")
            if not isinstance(seeds_raw, list) or not all(
                isinstance(seed, int) for seed in seeds_raw
            ):
                raise ValueError(f"{context}.seeds_by_side[{side!r}] must be list[int]")
            seeds_by_side[cast(PlayerSide, side)] = tuple(seeds_raw)
        if not seeds_by_side:
            raise ValueError(f"{context}.seeds_by_side must not be empty")
        return cls(entry_id=entry_id, seeds_by_side=seeds_by_side)


@dataclass(frozen=True, slots=True)
class SuiteSeedBank:
    entries: tuple[SuiteSeedBankEntry, ...]
    version: str = "v1"

    def to_dict(self) -> JsonDict:
        return {
            "version": self.version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, raw: JsonDict) -> "SuiteSeedBank":
        entries_raw = raw.get("entries")
        if not isinstance(entries_raw, list):
            raise ValueError("suite_seed_bank.entries must be a list")
        entries = tuple(
            SuiteSeedBankEntry.from_dict(
                cast(JsonDict, item), context=f"suite_seed_bank.entries[{index}]"
            )
            for index, item in enumerate(entries_raw)
            if isinstance(item, dict)
        )
        if not entries:
            raise ValueError("suite_seed_bank.entries must not be empty")
        entry_ids = [entry.entry_id for entry in entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("suite_seed_bank entry_ids must be unique")
        version = raw.get("version", "v1")
        if not isinstance(version, str):
            raise ValueError("suite_seed_bank.version must be a string")
        return cls(entries=entries, version=version)

    def entry_map(self) -> dict[str, SuiteSeedBankEntry]:
        return {entry.entry_id: entry for entry in self.entries}


@dataclass(frozen=True, slots=True)
class ChampionRecord:
    champion_id: str
    policy: PolicySpec
    promoted_at: str
    benchmark_summary: JsonDict | None = None

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "champion_id": self.champion_id,
            "policy": self.policy.to_dict(),
            "promoted_at": self.promoted_at,
        }
        if self.benchmark_summary is not None:
            payload["benchmark_summary"] = deepcopy(self.benchmark_summary)
        return payload

    @classmethod
    def from_dict(cls, raw: JsonDict, *, context: str = "champion_record") -> "ChampionRecord":
        champion_id = raw.get("champion_id")
        promoted_at = raw.get("promoted_at")
        policy_raw = raw.get("policy")
        benchmark_summary = raw.get("benchmark_summary")
        if not isinstance(champion_id, str) or not champion_id:
            raise ValueError(f"{context}.champion_id must be a non-empty string")
        if not isinstance(promoted_at, str) or not promoted_at:
            raise ValueError(f"{context}.promoted_at must be a non-empty string")
        if not isinstance(policy_raw, dict):
            raise ValueError(f"{context}.policy must be an object")
        if benchmark_summary is not None and not isinstance(benchmark_summary, dict):
            raise ValueError(f"{context}.benchmark_summary must be an object or null")
        return cls(
            champion_id=champion_id,
            policy=PolicySpec.from_dict(policy_raw, context=f"{context}.policy"),
            promoted_at=promoted_at,
            benchmark_summary=deepcopy(benchmark_summary)
            if isinstance(benchmark_summary, dict)
            else None,
        )


@dataclass(frozen=True, slots=True)
class ChampionRegistry:
    current_champion: ChampionRecord | None = None
    history: tuple[ChampionRecord, ...] = ()
    promotions: tuple[JsonDict, ...] = ()
    version: str = "v1"

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "version": self.version,
            "history": [record.to_dict() for record in self.history],
            "promotions": [deepcopy(record) for record in self.promotions],
        }
        if self.current_champion is not None:
            payload["current_champion"] = self.current_champion.to_dict()
        else:
            payload["current_champion"] = None
        return payload

    @classmethod
    def from_dict(cls, raw: JsonDict) -> "ChampionRegistry":
        current_raw = raw.get("current_champion")
        history_raw = raw.get("history", [])
        promotions_raw = raw.get("promotions", [])
        version = raw.get("version", "v1")
        if current_raw is not None and not isinstance(current_raw, dict):
            raise ValueError("current_champion must be an object or null")
        if not isinstance(history_raw, list):
            raise ValueError("history must be a list")
        if not isinstance(promotions_raw, list):
            raise ValueError("promotions must be a list")
        if not isinstance(version, str):
            raise ValueError("version must be a string")
        return cls(
            current_champion=(
                ChampionRecord.from_dict(current_raw, context="current_champion")
                if isinstance(current_raw, dict)
                else None
            ),
            history=tuple(
                ChampionRecord.from_dict(cast(JsonDict, item), context=f"history[{index}]")
                for index, item in enumerate(history_raw)
                if isinstance(item, dict)
            ),
            promotions=tuple(
                deepcopy(cast(JsonDict, item)) for item in promotions_raw if isinstance(item, dict)
            ),
            version=version,
        )


@dataclass(frozen=True, slots=True)
class ScheduledMatch:
    match_index: int
    trace_seed: int
    opponent_entry_id: str
    learner_player_id: str
    train_player_ids: tuple[str, ...]
    p1_policy: PolicySpec
    p2_policy: PolicySpec

    def to_dict(self) -> JsonDict:
        return {
            "match_index": self.match_index,
            "trace_seed": self.trace_seed,
            "opponent_entry_id": self.opponent_entry_id,
            "learner_player_id": self.learner_player_id,
            "train_player_ids": list(self.train_player_ids),
            "p1_policy": self.p1_policy.to_dict(),
            "p2_policy": self.p2_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: JsonDict) -> "ScheduledMatch":
        train_player_ids_raw = raw.get("train_player_ids")
        if not isinstance(train_player_ids_raw, list) or not all(
            isinstance(item, str) for item in train_player_ids_raw
        ):
            raise ValueError("scheduled_match.train_player_ids must be a list[str]")
        match_index = raw.get("match_index")
        trace_seed = raw.get("trace_seed")
        opponent_entry_id = raw.get("opponent_entry_id")
        learner_player_id = raw.get("learner_player_id")
        p1_policy_raw = raw.get("p1_policy")
        p2_policy_raw = raw.get("p2_policy")
        if not isinstance(match_index, int):
            raise ValueError("scheduled_match.match_index must be an int")
        if not isinstance(trace_seed, int):
            raise ValueError("scheduled_match.trace_seed must be an int")
        if not isinstance(opponent_entry_id, str) or not opponent_entry_id:
            raise ValueError("scheduled_match.opponent_entry_id must be a non-empty string")
        if learner_player_id not in {"p1", "p2"}:
            raise ValueError("scheduled_match.learner_player_id must be 'p1' or 'p2'")
        if not isinstance(p1_policy_raw, dict) or not isinstance(p2_policy_raw, dict):
            raise ValueError("scheduled_match policies must be objects")
        return cls(
            match_index=match_index,
            trace_seed=trace_seed,
            opponent_entry_id=opponent_entry_id,
            learner_player_id=learner_player_id,
            train_player_ids=tuple(train_player_ids_raw),
            p1_policy=PolicySpec.from_dict(p1_policy_raw, context="scheduled_match.p1_policy"),
            p2_policy=PolicySpec.from_dict(p2_policy_raw, context="scheduled_match.p2_policy"),
        )


def _default_openrouter_options(*, base_url: str | None = None) -> JsonDict:
    options: JsonDict = {
        "response_parser": deepcopy(DEFAULT_RESPONSE_PARSER),
    }
    if base_url is not None:
        options["base_url"] = base_url
    return options


def default_tinker_policy(
    *,
    policy_id: str,
    model: str,
    prompt_version: str = DEFAULT_RL_PROMPT_VERSION,
    temperature: float = 0.7,
    max_tokens: int = 256,
    label: str | None = None,
) -> PolicySpec:
    return PolicySpec(
        policy_id=policy_id,
        provider="openrouter",
        model=model,
        prompt_version=prompt_version,
        credential_env_var="TINKER_API_KEY",
        temperature=temperature,
        max_tokens=max_tokens,
        options=_default_openrouter_options(base_url=TINKER_OAI_BASE_URL),
        label=label,
    )


def default_policy_for_model(
    *,
    policy_id: str,
    model: str,
    prompt_version: str = DEFAULT_RL_PROMPT_VERSION,
    temperature: float = 0.7,
    max_tokens: int = 256,
    label: str | None = None,
) -> PolicySpec:
    if model.startswith("tinker://"):
        return default_tinker_policy(
            policy_id=policy_id,
            model=model,
            prompt_version=prompt_version,
            temperature=temperature,
            max_tokens=max_tokens,
            label=label,
        )
    return PolicySpec(
        policy_id=policy_id,
        provider="openrouter",
        model=model,
        prompt_version=prompt_version,
        credential_env_var="OPENROUTER_API_KEY",
        temperature=temperature,
        max_tokens=max_tokens,
        options=_default_openrouter_options(),
        label=label,
    )


def default_sft_policy(
    *,
    policy_id: str = "tinker/sft-v1",
    prompt_version: str = DEFAULT_RL_PROMPT_VERSION,
    label: str | None = "SFT v1",
) -> PolicySpec:
    return default_tinker_policy(
        policy_id=policy_id,
        model=DEFAULT_SFT_SAMPLER,
        prompt_version=prompt_version,
        label=label,
    )


def load_json_document(path: Path) -> JsonDict:
    return cast(JsonDict, json.loads(path.read_text()))


def write_json_document(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_league_config(path: Path) -> LeagueConfig:
    return LeagueConfig.from_dict(load_json_document(path))


def write_league_config(path: Path, league: LeagueConfig) -> None:
    write_json_document(path, league.to_dict())


def load_suite_config(path: Path) -> SuiteConfig:
    return SuiteConfig.from_dict(load_json_document(path))


def write_suite_config(path: Path, suite: SuiteConfig) -> None:
    write_json_document(path, suite.to_dict())


def load_suite_seed_bank(path: Path) -> SuiteSeedBank:
    return SuiteSeedBank.from_dict(load_json_document(path))


def write_suite_seed_bank(path: Path, seed_bank: SuiteSeedBank) -> None:
    write_json_document(path, seed_bank.to_dict())


def load_champion_registry(path: Path) -> ChampionRegistry:
    if not path.exists():
        return ChampionRegistry()
    return ChampionRegistry.from_dict(load_json_document(path))


def write_champion_registry(path: Path, registry: ChampionRegistry) -> None:
    write_json_document(path, registry.to_dict())


def build_head_to_head_config(
    base_template: JsonDict,
    *,
    p1_policy: PolicySpec,
    p2_policy: PolicySpec,
) -> JsonDict:
    template = deepcopy(base_template)
    template["policy_mapping"] = {"p1": p1_policy.policy_id, "p2": p2_policy.policy_id}
    template["policies"] = {
        p1_policy.policy_id: p1_policy.to_config_entry(),
        p2_policy.policy_id: p2_policy.to_config_entry(),
    }
    return template


def suite_entry_sides(entry: SuiteEntry) -> tuple[PlayerSide, ...]:
    return ("p1", "p2") if entry.sides == "both" else (entry.sides,)


def build_suite_seed_bank(
    suite: SuiteConfig,
    *,
    seed_base: int,
) -> SuiteSeedBank:
    entries: list[SuiteSeedBankEntry] = []
    for entry_index, entry in enumerate(suite.entries):
        seeds_by_side: dict[PlayerSide, tuple[int, ...]] = {}
        for side_index, side in enumerate(suite_entry_sides(entry)):
            start_seed = seed_base + entry_index * 1000 + side_index * 500
            seeds_by_side[side] = tuple(
                start_seed + match_index for match_index in range(entry.num_matches)
            )
        entries.append(SuiteSeedBankEntry(entry_id=entry.entry_id, seeds_by_side=seeds_by_side))
    return SuiteSeedBank(entries=tuple(entries))


def resolve_suite_seed_bank(
    suite: SuiteConfig,
    *,
    seed_base: int,
    seed_bank: SuiteSeedBank | None = None,
) -> SuiteSeedBank:
    resolved = seed_bank or build_suite_seed_bank(suite, seed_base=seed_base)
    entry_map = resolved.entry_map()
    for entry in suite.entries:
        bank_entry = entry_map.get(entry.entry_id)
        if bank_entry is None:
            raise ValueError(
                f"suite seed bank is missing entry {entry.entry_id!r} required by the suite"
            )
        for side in suite_entry_sides(entry):
            seeds = bank_entry.seeds_by_side.get(side)
            if seeds is None:
                raise ValueError(
                    f"suite seed bank entry {entry.entry_id!r} is missing side {side!r}"
                )
            if len(seeds) != entry.num_matches:
                raise ValueError(
                    f"suite seed bank entry {entry.entry_id!r} side {side!r} has "
                    f"{len(seeds)} seeds but suite requires {entry.num_matches}"
                )
    return resolved


def choose_weighted_league_entry(entries: tuple[LeagueEntry, ...], rng) -> LeagueEntry:
    weights = np.array([entry.weight for entry in entries], dtype=np.float64)
    weights = weights / weights.sum()
    index = int(rng.choice(len(entries), p=weights))
    return entries[index]


# NumPy is only used here; defer import to avoid mandatory dependency at import time in docs.
import numpy as np  # noqa: E402


def schedule_matches(
    *,
    learner_model: str,
    matches_per_iter: int,
    iteration: int,
    rng,
    league: LeagueConfig | None = None,
    learner_prompt_version: str = DEFAULT_RL_PROMPT_VERSION,
    learner_temperature: float = 0.9,
    learner_max_tokens: int = 256,
) -> list[ScheduledMatch]:
    schedules: list[ScheduledMatch] = []
    learner_entries = {
        "p1": default_tinker_policy(
            policy_id="rl/learner-p1",
            model=learner_model,
            prompt_version=learner_prompt_version,
            temperature=learner_temperature,
            max_tokens=learner_max_tokens,
            label="learner",
        ),
        "p2": default_tinker_policy(
            policy_id="rl/learner-p2",
            model=learner_model,
            prompt_version=learner_prompt_version,
            temperature=learner_temperature,
            max_tokens=learner_max_tokens,
            label="learner",
        ),
    }

    entries = (
        league.entries
        if league is not None
        else (LeagueEntry(entry_id="mirror", weight=1.0, kind="self", label="mirror"),)
    )

    for match_index in range(matches_per_iter):
        trace_seed = 20000 + iteration * 1000 + match_index
        selected = choose_weighted_league_entry(entries, rng)
        if selected.kind == "self":
            schedules.append(
                ScheduledMatch(
                    match_index=match_index,
                    trace_seed=trace_seed,
                    opponent_entry_id=selected.entry_id,
                    learner_player_id="p1",
                    train_player_ids=("p1", "p2"),
                    p1_policy=learner_entries["p1"],
                    p2_policy=learner_entries["p2"],
                )
            )
            continue

        assert selected.policy is not None
        learner_player_id = "p1" if float(rng.random()) < 0.5 else "p2"
        if learner_player_id == "p1":
            p1_policy = learner_entries["p1"]
            p2_policy = selected.policy
        else:
            p1_policy = selected.policy
            p2_policy = learner_entries["p2"]
        schedules.append(
            ScheduledMatch(
                match_index=match_index,
                trace_seed=trace_seed,
                opponent_entry_id=selected.entry_id,
                learner_player_id=learner_player_id,
                train_player_ids=(learner_player_id,),
                p1_policy=p1_policy,
                p2_policy=p2_policy,
            )
        )
    return schedules


def build_default_league(
    *,
    registry: ChampionRegistry,
    include_sft: bool = True,
    include_mirror: bool = True,
    history_limit: int = 3,
    mirror_weight: float = 0.2,
    champion_weight: float = 0.35,
    sft_weight: float = 0.25,
    history_weight: float = 0.2,
    prompt_version: str = DEFAULT_RL_PROMPT_VERSION,
) -> LeagueConfig:
    entries: list[LeagueEntry] = []
    seen_models: set[str] = set()
    if include_mirror:
        entries.append(
            LeagueEntry(
                entry_id="mirror", weight=mirror_weight, kind="self", label="Learner mirror"
            )
        )
    if registry.current_champion is not None:
        champion_policy = replace_policy_prompt_version(
            registry.current_champion.policy, prompt_version
        )
        entries.append(
            LeagueEntry(
                entry_id="champion",
                weight=champion_weight,
                kind="policy",
                policy=champion_policy,
                label=registry.current_champion.champion_id,
            )
        )
        seen_models.add(champion_policy.model)
    sft_policy = default_sft_policy(prompt_version=prompt_version)
    if include_sft and sft_policy.model not in seen_models:
        entries.append(
            LeagueEntry(
                entry_id="sft-v1",
                weight=sft_weight,
                kind="policy",
                policy=sft_policy,
                label="SFT v1",
            )
        )
        seen_models.add(sft_policy.model)
    historical = [record for record in registry.history]
    if registry.current_champion is not None:
        historical = [
            record
            for record in historical
            if record.champion_id != registry.current_champion.champion_id
        ]
    history_index = 0
    for record in historical:
        if history_index >= history_limit:
            break
        history_policy = replace_policy_prompt_version(record.policy, prompt_version)
        if history_policy.model in seen_models:
            continue
        history_index += 1
        seen_models.add(history_policy.model)
        entries.append(
            LeagueEntry(
                entry_id=f"history-{history_index}",
                weight=(history_weight / max(history_limit, 1)),
                kind="policy",
                policy=history_policy,
                label=record.champion_id,
            )
        )
    if not entries:
        entries.append(
            LeagueEntry(entry_id="mirror", weight=1.0, kind="self", label="Learner mirror")
        )
    return LeagueConfig(entries=tuple(entries))


def build_default_suite(
    *,
    registry: ChampionRegistry,
    num_matches: int = 10,
    prompt_version: str = DEFAULT_RL_PROMPT_VERSION,
    include_sft: bool = True,
    include_champion: bool = True,
    history_limit: int = 3,
    exclude_models: set[str] | None = None,
) -> SuiteConfig:
    excluded = exclude_models or set()
    entries: list[SuiteEntry] = []
    seen_models: set[str] = set()
    if include_champion and registry.current_champion is not None:
        champion_policy = replace_policy_prompt_version(
            registry.current_champion.policy, prompt_version
        )
        if champion_policy.model not in excluded:
            entries.append(
                SuiteEntry(
                    entry_id="champion",
                    opponent=champion_policy,
                    num_matches=num_matches,
                    sides="both",
                    weight=1.0,
                    label=registry.current_champion.champion_id,
                )
            )
            seen_models.add(champion_policy.model)
    sft_policy = default_sft_policy(prompt_version=prompt_version)
    if include_sft and sft_policy.model not in excluded and sft_policy.model not in seen_models:
        entries.append(
            SuiteEntry(
                entry_id="sft-v1",
                opponent=sft_policy,
                num_matches=num_matches,
                sides="both",
                weight=1.0,
                label="SFT v1",
            )
        )
        seen_models.add(sft_policy.model)
    historical = list(registry.history)
    if registry.current_champion is not None:
        historical = [
            record
            for record in historical
            if record.champion_id != registry.current_champion.champion_id
        ]
    history_index = 0
    for record in historical:
        if history_index >= history_limit:
            break
        history_policy = replace_policy_prompt_version(record.policy, prompt_version)
        if history_policy.model in excluded or history_policy.model in seen_models:
            continue
        history_index += 1
        seen_models.add(history_policy.model)
        entries.append(
            SuiteEntry(
                entry_id=f"history-{history_index}",
                opponent=history_policy,
                num_matches=num_matches,
                sides="both",
                weight=1.0,
                label=record.champion_id,
            )
        )
    if not entries:
        raise ValueError("default suite would be empty after excluded-model filtering")
    return SuiteConfig(entries=tuple(entries))


def clone_policy(
    policy: PolicySpec,
    *,
    policy_id: str | None = None,
    prompt_version: str | None = None,
    label: str | None = None,
) -> PolicySpec:
    return PolicySpec(
        policy_id=policy_id or policy.policy_id,
        provider=policy.provider,
        model=policy.model,
        prompt_version=prompt_version or policy.prompt_version,
        credential_env_var=policy.credential_env_var,
        temperature=policy.temperature,
        max_tokens=policy.max_tokens,
        options=deepcopy(policy.options),
        label=label if label is not None else policy.label,
    )


def replace_policy_prompt_version(policy: PolicySpec, prompt_version: str) -> PolicySpec:
    return clone_policy(policy, prompt_version=prompt_version)


def is_completed_run_dir(run_dir: Path) -> bool:
    result_file = run_dir / "result.json"
    if not result_file.exists():
        return False
    result = cast(JsonDict, json.loads(result_file.read_text()))
    return result.get("status", "completed") == "completed"


def count_completed_runs(runs_root: Path) -> int:
    return sum(1 for path in runs_root.iterdir() if path.is_dir() and is_completed_run_dir(path))


def summarize_runs_root(runs_root: Path, *, candidate_player_id: str) -> JsonDict:
    games: list[JsonDict] = []
    action_histogram: Counter[str] = Counter()
    opening_histogram: Counter[str] = Counter()
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        result_file = run_dir / "result.json"
        decisions_file = run_dir / "decisions.jsonl"
        manifest_file = run_dir / "manifest.json"
        if not result_file.exists():
            continue
        result = cast(JsonDict, json.loads(result_file.read_text()))
        if result.get("status", "completed") != "completed":
            continue
        winner = result.get("winner")
        p1_hp = 0
        p2_hp = 0
        candidate_fallbacks = 0
        candidate_decisions = 0
        candidate_actions: list[str] = []
        if decisions_file.exists():
            lines = [line for line in decisions_file.read_text().splitlines() if line.strip()]
            for line in reversed(lines):
                record = cast(JsonDict, json.loads(line))
                obs = cast(JsonDict, record.get("request_payload", {})).get("observation")
                if isinstance(obs, dict):
                    fighters = obs.get("fighters")
                    if (
                        isinstance(fighters, list)
                        and len(fighters) == 2
                        and isinstance(fighters[0], dict)
                        and isinstance(fighters[1], dict)
                    ):
                        p1_hp = int(cast(JsonDict, fighters[0]).get("hp", 0))
                        p2_hp = int(cast(JsonDict, fighters[1]).get("hp", 0))
                        break
            for line in lines:
                record = cast(JsonDict, json.loads(line))
                request_payload = cast(JsonDict, record.get("request_payload", {}))
                decision_payload = cast(JsonDict, record.get("decision_payload", {}))
                if request_payload.get("player_id") == candidate_player_id:
                    candidate_decisions += 1
                    if decision_payload.get("fallback_reason") is not None:
                        candidate_fallbacks += 1
                    action = decision_payload.get("action")
                    if isinstance(action, str) and action:
                        candidate_actions.append(action)
                        action_histogram[action] += 1
        opening_action = candidate_actions[0] if candidate_actions else None
        if opening_action is not None:
            opening_histogram[opening_action] += 1
        opponent_id = "?"
        if manifest_file.exists():
            manifest = cast(JsonDict, json.loads(manifest_file.read_text()))
            policy_mapping = cast(JsonDict, manifest.get("policy_mapping", {}))
            opponent_slot = "p2" if candidate_player_id == "p1" else "p1"
            opponent = policy_mapping.get(opponent_slot)
            if isinstance(opponent, str):
                opponent_id = opponent
        if candidate_player_id == "p1":
            won = winner == "p1"
            losses = winner == "p2"
            hp_diff = p1_hp - p2_hp
        else:
            won = winner == "p2"
            losses = winner == "p1"
            hp_diff = p2_hp - p1_hp
        games.append(
            {
                "run_dir": run_dir.name,
                "winner": winner,
                "won": won,
                "lost": losses,
                "hp_diff": hp_diff,
                "turns": int(result.get("total_turns", 0)),
                "end_reason": result.get("end_reason", "?"),
                "candidate_player_id": candidate_player_id,
                "candidate_fallbacks": candidate_fallbacks,
                "candidate_decisions": candidate_decisions,
                "opening_action": opening_action,
                "opponent_policy_id": opponent_id,
            }
        )

    total_games = len(games)
    wins = sum(1 for game in games if game["won"])
    losses = sum(1 for game in games if game["lost"])
    draws = total_games - wins - losses
    avg_hp_diff = sum(int(game["hp_diff"]) for game in games) / total_games if total_games else 0.0
    avg_turns = sum(int(game["turns"]) for game in games) / total_games if total_games else 0.0
    total_fallbacks = sum(int(game["candidate_fallbacks"]) for game in games)
    total_decisions = sum(int(game["candidate_decisions"]) for game in games)
    unique_actions = len(action_histogram)
    unique_openings = len(opening_histogram)
    return {
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": (wins / total_games) if total_games else 0.0,
        "avg_hp_diff": round(avg_hp_diff, 1),
        "avg_turns": round(avg_turns, 1),
        "candidate_fallbacks": total_fallbacks,
        "candidate_decisions": total_decisions,
        "candidate_fallback_rate": (total_fallbacks / total_decisions) if total_decisions else 0.0,
        "candidate_action_histogram": dict(action_histogram),
        "candidate_opening_histogram": dict(opening_histogram),
        "candidate_unique_actions": unique_actions,
        "candidate_unique_openings": unique_openings,
        "candidate_action_diversity": (unique_actions / total_decisions)
        if total_decisions
        else 0.0,
        "candidate_opening_diversity": (unique_openings / total_games) if total_games else 0.0,
        "games_detail": games,
    }


def aggregate_suite_score(suite_result: JsonDict) -> float:
    entries_raw = suite_result.get("entries", [])
    if not isinstance(entries_raw, list) or not entries_raw:
        return 0.0
    weighted = 0.0
    total_weight = 0.0
    for entry in entries_raw:
        if not isinstance(entry, dict):
            continue
        weight = entry.get("weight", 1.0)
        aggregate = entry.get("aggregate", {})
        if not isinstance(weight, int | float) or not isinstance(aggregate, dict):
            continue
        win_rate = aggregate.get("win_rate", 0.0)
        if not isinstance(win_rate, int | float):
            continue
        weighted += float(weight) * float(win_rate)
        total_weight += float(weight)
    return weighted / total_weight if total_weight > 0 else 0.0


def timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
