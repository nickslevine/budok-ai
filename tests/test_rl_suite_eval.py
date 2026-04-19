from __future__ import annotations

from pathlib import Path

from yomi_daemon.rl.suite_eval import (
    combine_side_summaries,
    evaluate_suite,
    resolve_model_for_evaluation,
)
from yomi_daemon.rl.v3 import PolicySpec, SuiteConfig, SuiteEntry, build_suite_seed_bank


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


def test_combine_side_summaries_merges_histograms_and_diversity() -> None:
    aggregate = combine_side_summaries(
        {
            "p1": {
                "games": 2,
                "wins": 1,
                "losses": 1,
                "draws": 0,
                "avg_hp_diff": 5.0,
                "avg_turns": 10.0,
                "candidate_fallbacks": 1,
                "candidate_decisions": 4,
                "candidate_action_histogram": {"Grab": 3, "Wait": 1},
                "candidate_opening_histogram": {"Grab": 2},
            },
            "p2": {
                "games": 2,
                "wins": 2,
                "losses": 0,
                "draws": 0,
                "avg_hp_diff": 15.0,
                "avg_turns": 20.0,
                "candidate_fallbacks": 0,
                "candidate_decisions": 2,
                "candidate_action_histogram": {"Wait": 2},
                "candidate_opening_histogram": {"Wait": 2},
            },
        }
    )

    assert aggregate["games"] == 4
    assert aggregate["wins"] == 3
    assert aggregate["candidate_action_histogram"] == {"Grab": 3, "Wait": 3}
    assert aggregate["candidate_opening_histogram"] == {"Grab": 2, "Wait": 2}
    assert aggregate["candidate_unique_actions"] == 2
    assert aggregate["candidate_unique_openings"] == 2
    assert aggregate["candidate_action_diversity"] == 2 / 6
    assert aggregate["candidate_opening_diversity"] == 0.5


def test_resolve_model_for_evaluation_keeps_sampler_paths() -> None:
    model = "tinker://train/sampler_weights/iter1"
    assert resolve_model_for_evaluation(model, sampler_name="unused") == model


def test_resolve_model_for_evaluation_exports_sampler_from_weights(
    monkeypatch,
) -> None:
    exported_calls: list[tuple[str, str]] = []

    async def fake_export_sampler_from_state_async(*, state_path: str, sampler_name: str) -> str:
        exported_calls.append((state_path, sampler_name))
        return "tinker://train/sampler_weights/exported"

    monkeypatch.setattr(
        "yomi_daemon.rl.suite_eval._export_sampler_from_state_async",
        fake_export_sampler_from_state_async,
    )

    resolved = resolve_model_for_evaluation(
        "tinker://train/weights/iter1",
        sampler_name="candidate-eval",
    )

    assert resolved == "tinker://train/sampler_weights/exported"
    assert exported_calls == [("tinker://train/weights/iter1", "candidate-eval")]


def test_evaluate_suite_uses_provided_seed_bank(monkeypatch, tmp_path: Path) -> None:
    suite = SuiteConfig(
        entries=(
            SuiteEntry(
                entry_id="champion",
                opponent=make_policy("opp", "tinker://opp"),
                num_matches=2,
                sides="both",
            ),
        )
    )
    candidate = make_policy("candidate", "tinker://candidate")
    base_template = tmp_path / "base_template.json"
    base_template.write_text("{}")
    seen_seeds: list[int] = []

    def fake_write_temp_config(base_template_doc, *, p1_policy, p2_policy):
        path = tmp_path / f"config_{len(seen_seeds)}.json"
        path.write_text(
            f'{{"policy_mapping":{{"p1":"{p1_policy.policy_id}","p2":"{p2_policy.policy_id}"}}}}'
        )
        return path

    def fake_run_match_subprocess(
        *, repo_root, config_path, runs_dir, trace_seed, timeout_s, skip_mod_push=True
    ):
        seen_seeds.append(trace_seed)
        config = config_path.read_text()
        candidate_side = "p1" if "candidate-p1" in config else "p2"
        run_dir = runs_dir / f"run_{trace_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if candidate_side == "p1":
            winner = "p1"
            manifest = '{"policy_mapping":{"p1":"candidate","p2":"opp"}}'
            decisions = (
                '{"request_payload":{"player_id":"p1","observation":{"fighters":[{"hp":100},{"hp":0}]}},'
                '"decision_payload":{"action":"Grab","fallback_reason":null}}\n'
            )
        else:
            winner = "p2"
            manifest = '{"policy_mapping":{"p1":"opp","p2":"candidate"}}'
            decisions = (
                '{"request_payload":{"player_id":"p2","observation":{"fighters":[{"hp":0},{"hp":100}]}},'
                '"decision_payload":{"action":"Wait","fallback_reason":null}}\n'
            )
        (run_dir / "result.json").write_text(
            f'{{"status":"completed","winner":"{winner}","total_turns":1,"end_reason":"ko"}}'
        )
        (run_dir / "manifest.json").write_text(manifest)
        (run_dir / "decisions.jsonl").write_text(decisions)
        return True

    monkeypatch.setattr(
        "yomi_daemon.rl.suite_eval.write_temp_config", fake_write_temp_config
    )
    monkeypatch.setattr(
        "yomi_daemon.rl.suite_eval.run_match_subprocess", fake_run_match_subprocess
    )

    seed_bank = build_suite_seed_bank(suite, seed_base=123)
    result = evaluate_suite(
        repo_root=tmp_path,
        candidate_policy=candidate,
        suite=suite,
        output_root=tmp_path / "runs",
        base_template_path=base_template,
        match_timeout_s=1,
        seed_base=999,
        seed_bank=seed_bank,
    )

    assert seen_seeds == [123, 124, 623, 624]
    assert result["seed_bank"] == seed_bank.to_dict()
    assert result["entries"][0]["seeds_by_side"] == {
        "p1": [123, 124],
        "p2": [623, 624],
    }
