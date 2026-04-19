from __future__ import annotations

import json

import numpy as np

from yomi_daemon.rl.v3 import (
    ChampionRecord,
    ChampionRegistry,
    LeagueConfig,
    LeagueEntry,
    PolicySpec,
    SuiteConfig,
    SuiteEntry,
    aggregate_suite_score,
    build_default_league,
    build_default_suite,
    build_head_to_head_config,
    build_suite_seed_bank,
    clone_policy,
    default_sft_policy,
    resolve_suite_seed_bank,
    schedule_matches,
    count_completed_runs,
    summarize_runs_root,
)


def make_policy(policy_id: str, model: str) -> PolicySpec:
    return PolicySpec(
        policy_id=policy_id,
        provider="openrouter",
        model=model,
        prompt_version="strategic_rl_v1",
        credential_env_var="TINKER_API_KEY",
        temperature=0.7,
        max_tokens=256,
        options={"base_url": "https://example.test"},
    )


def test_build_head_to_head_config_replaces_policy_mapping_and_policies() -> None:
    template = {
        "policy_mapping": {"p1": "old/a", "p2": "old/b"},
        "policies": {"old/a": {"model": "a"}, "old/b": {"model": "b"}},
        "character_selection": {
            "mode": "assigned",
            "assignments": {"p1": "Cowboy", "p2": "Cowboy"},
        },
    }
    p1 = make_policy("rl/p1", "tinker://model-a")
    p2 = make_policy("rl/p2", "tinker://model-b")

    config = build_head_to_head_config(template, p1_policy=p1, p2_policy=p2)

    assert config["policy_mapping"] == {"p1": "rl/p1", "p2": "rl/p2"}
    assert set(config["policies"].keys()) == {"rl/p1", "rl/p2"}
    assert config["policies"]["rl/p1"]["model"] == "tinker://model-a"
    assert config["policies"]["rl/p2"]["model"] == "tinker://model-b"


def test_schedule_matches_with_policy_league_trains_only_learner_side() -> None:
    league = LeagueConfig(
        entries=(
            LeagueEntry(
                entry_id="sft",
                weight=1.0,
                kind="policy",
                policy=default_sft_policy(),
            ),
        )
    )
    rng = np.random.default_rng(123)

    matches = schedule_matches(
        learner_model="tinker://learner",
        matches_per_iter=4,
        iteration=0,
        rng=rng,
        league=league,
    )

    assert len(matches) == 4
    assert all(len(match.train_player_ids) == 1 for match in matches)
    assert {match.learner_player_id for match in matches} <= {"p1", "p2"}
    assert all(match.opponent_entry_id == "sft" for match in matches)


def test_build_default_suite_excludes_candidate_model() -> None:
    champion = ChampionRecord(
        champion_id="champ-1",
        policy=make_policy("rl/champ", "tinker://champ-model"),
        promoted_at="2026-04-15T00:00:00Z",
    )
    registry = ChampionRegistry(current_champion=champion, history=(champion,))

    suite = build_default_suite(
        registry=registry,
        exclude_models={"tinker://champ-model"},
    )

    assert all(
        entry.opponent.model != "tinker://champ-model" for entry in suite.entries
    )
    assert any(entry.entry_id == "sft-v1" for entry in suite.entries)


def test_build_default_league_includes_current_champion_and_sft() -> None:
    champion = ChampionRecord(
        champion_id="champ-1",
        policy=make_policy("rl/champ", "tinker://champ-model"),
        promoted_at="2026-04-15T00:00:00Z",
    )
    older = ChampionRecord(
        champion_id="champ-0",
        policy=make_policy("rl/old", "tinker://old-model"),
        promoted_at="2026-04-14T00:00:00Z",
    )
    registry = ChampionRegistry(current_champion=champion, history=(older,))

    league = build_default_league(registry=registry, history_limit=1)
    entry_ids = {entry.entry_id for entry in league.entries}

    assert "mirror" in entry_ids
    assert "champion" in entry_ids
    assert "sft-v1" in entry_ids
    assert "history-1" in entry_ids


def test_build_default_league_skips_duplicate_sft_model_when_champion_is_sft() -> None:
    champion = ChampionRecord(
        champion_id="champ-sft",
        policy=default_sft_policy(policy_id="rl/champ-sft"),
        promoted_at="2026-04-15T00:00:00Z",
    )
    registry = ChampionRegistry(current_champion=champion)

    league = build_default_league(registry=registry)
    entry_ids = [entry.entry_id for entry in league.entries]

    assert entry_ids == ["mirror", "champion"]


def test_build_default_suite_skips_duplicate_sft_model_when_champion_is_sft() -> None:
    champion = ChampionRecord(
        champion_id="champ-sft",
        policy=default_sft_policy(policy_id="rl/champ-sft"),
        promoted_at="2026-04-15T00:00:00Z",
    )
    registry = ChampionRegistry(current_champion=champion)

    suite = build_default_suite(registry=registry)

    assert [entry.entry_id for entry in suite.entries] == ["champion"]


def test_aggregate_suite_score_uses_weighted_mean() -> None:
    result = {
        "entries": [
            {"weight": 1.0, "aggregate": {"win_rate": 0.25}},
            {"weight": 3.0, "aggregate": {"win_rate": 0.75}},
        ]
    }

    assert aggregate_suite_score(result) == 0.625


def test_build_suite_seed_bank_matches_suite_shape() -> None:
    suite = SuiteConfig(
        entries=(
            SuiteEntry(
                entry_id="champion",
                opponent=make_policy("opp-a", "tinker://opp-a"),
                num_matches=2,
                sides="both",
            ),
            SuiteEntry(
                entry_id="sft",
                opponent=make_policy("opp-b", "tinker://opp-b"),
                num_matches=3,
                sides="p1",
            ),
        )
    )

    seed_bank = build_suite_seed_bank(suite, seed_base=100)

    assert seed_bank.entries[0].seeds_by_side["p1"] == (100, 101)
    assert seed_bank.entries[0].seeds_by_side["p2"] == (600, 601)
    assert seed_bank.entries[1].seeds_by_side["p1"] == (1100, 1101, 1102)


def test_resolve_suite_seed_bank_validates_required_counts() -> None:
    suite = SuiteConfig(
        entries=(
            SuiteEntry(
                entry_id="champion",
                opponent=make_policy("opp-a", "tinker://opp-a"),
                num_matches=2,
                sides="both",
            ),
        )
    )
    invalid_bank = build_suite_seed_bank(suite, seed_base=10)
    invalid_bank = invalid_bank.__class__(
        entries=(
            invalid_bank.entries[0].__class__(
                entry_id="champion",
                seeds_by_side={"p1": (10,), "p2": (510, 511)},
            ),
        )
    )

    try:
        resolve_suite_seed_bank(suite, seed_base=10, seed_bank=invalid_bank)
    except ValueError as exc:
        assert "has 1 seeds but suite requires 2" in str(exc)
    else:
        raise AssertionError(
            "expected resolve_suite_seed_bank() to reject wrong seed counts"
        )


def test_summarize_runs_root_uses_candidate_side_for_win_and_hp_diff(tmp_path) -> None:
    run_dir = tmp_path / "match-001"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "winner": "p2",
                "total_turns": 20,
                "end_reason": "ko",
            }
        )
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"policy_mapping": {"p1": "opp", "p2": "candidate"}})
    )
    decisions = [
        {
            "request_payload": {
                "player_id": "p1",
                "observation": {"fighters": [{"hp": 30}, {"hp": 80}]},
            },
            "decision_payload": {"fallback_reason": None},
        },
        {
            "request_payload": {
                "player_id": "p2",
                "observation": {"fighters": [{"hp": 30}, {"hp": 80}]},
            },
            "decision_payload": {"fallback_reason": None},
        },
    ]
    (run_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(item) for item in decisions) + "\n"
    )

    summary = summarize_runs_root(tmp_path, candidate_player_id="p2")

    assert summary["games"] == 1
    assert summary["wins"] == 1
    assert summary["avg_hp_diff"] == 50.0
    assert summary["candidate_fallback_rate"] == 0.0


def test_count_completed_runs_and_summary_ignore_failed_runs(tmp_path) -> None:
    failed = tmp_path / "failed-match"
    failed.mkdir()
    (failed / "result.json").write_text(json.dumps({"status": "failed"}))
    completed = tmp_path / "completed-match"
    completed.mkdir()
    (completed / "result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "winner": "p1",
                "total_turns": 5,
                "end_reason": "ko",
            }
        )
    )
    (completed / "manifest.json").write_text(
        json.dumps({"policy_mapping": {"p1": "candidate", "p2": "opp"}})
    )
    (completed / "decisions.jsonl").write_text(
        json.dumps(
            {
                "request_payload": {
                    "player_id": "p1",
                    "observation": {"fighters": [{"hp": 100}, {"hp": 0}]},
                },
                "decision_payload": {"action": "Grab", "fallback_reason": None},
            }
        )
        + "\n"
    )

    assert count_completed_runs(tmp_path) == 1
    summary = summarize_runs_root(tmp_path, candidate_player_id="p1")
    assert summary["games"] == 1
    assert summary["wins"] == 1


def test_summarize_runs_root_tracks_action_and_opening_diversity(tmp_path) -> None:
    run_a = tmp_path / "match-001"
    run_b = tmp_path / "match-002"
    run_a.mkdir()
    run_b.mkdir()
    for run_dir, winner, hp_a, hp_b, opening, followup in [
        (run_a, "p1", 60, 20, "Grab", "Wait"),
        (run_b, "p1", 70, 10, "Wait", "Grab"),
    ]:
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "winner": winner,
                    "total_turns": 10,
                    "end_reason": "ko",
                }
            )
        )
        (run_dir / "manifest.json").write_text(
            json.dumps({"policy_mapping": {"p1": "candidate", "p2": "opp"}})
        )
        decisions = [
            {
                "request_payload": {
                    "player_id": "p1",
                    "observation": {"fighters": [{"hp": hp_a}, {"hp": hp_b}]},
                },
                "decision_payload": {"action": opening, "fallback_reason": None},
            },
            {
                "request_payload": {
                    "player_id": "p1",
                    "observation": {"fighters": [{"hp": hp_a}, {"hp": hp_b}]},
                },
                "decision_payload": {"action": followup, "fallback_reason": None},
            },
        ]
        (run_dir / "decisions.jsonl").write_text(
            "\n".join(json.dumps(item) for item in decisions) + "\n"
        )

    summary = summarize_runs_root(tmp_path, candidate_player_id="p1")

    assert summary["candidate_action_histogram"] == {"Grab": 2, "Wait": 2}
    assert summary["candidate_opening_histogram"] == {"Grab": 1, "Wait": 1}
    assert summary["candidate_unique_actions"] == 2
    assert summary["candidate_unique_openings"] == 2
    assert summary["candidate_action_diversity"] == 0.5
    assert summary["candidate_opening_diversity"] == 1.0


def test_clone_policy_allows_side_specific_ids() -> None:
    policy = make_policy("base", "tinker://base")

    clone = clone_policy(policy, policy_id="base-p2", label="clone")

    assert clone.policy_id == "base-p2"
    assert clone.label == "clone"
    assert clone.model == policy.model
