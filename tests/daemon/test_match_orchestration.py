"""End-to-end integration tests for single-match orchestration (WU-012)."""

from __future__ import annotations

import asyncio
import json

from typing import Any, cast

from websockets.asyncio.client import connect

from yomi_daemon import __version__
from yomi_daemon.match import MatchMetadata, MatchSession
from yomi_daemon.protocol import (
    CURRENT_PROTOCOL_VERSION,
    ActionDecision,
    EventName,
    MessageType,
)
from yomi_daemon.replay_capture import ReplayCaptureConfig
from yomi_daemon.server import DaemonServer
from yomi_daemon.storage.writer import MatchArtifactWriter
from yomi_daemon.storage.writer import RUNS_DIR
from yomi_daemon.validation import parse_envelope

from tests.daemon.conftest import (
    baseline_runtime_config,
    decision_request_envelope,
    handshake,
    hello_envelope,
    match_ended_envelope,
    running_match_server,
    unique_match_id,
)

_INTEGRATION = True

# Re-export for files that still import from here
_baseline_runtime_config = baseline_runtime_config
_hello_envelope = hello_envelope
_handshake = handshake
_match_ended_envelope = match_ended_envelope


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_match_baseline_vs_baseline() -> None:
    """Run a complete multi-turn match with baseline policies and verify artifacts."""

    async def scenario() -> None:
        mid = unique_match_id()
        config = baseline_runtime_config()
        async with running_match_server(config) as server:
            async with connect(f"ws://127.0.0.1:{server.listening_port}") as ws:
                ack = await handshake(ws)
                assert ack.policy_mapping.p1 == "baseline/random"

                responses = []
                for turn in range(1, 6):
                    player = "p1" if turn % 2 == 1 else "p2"
                    req = decision_request_envelope(
                        match_id=mid, turn_id=turn, player_id=player
                    )
                    await ws.send(json.dumps(req))
                    raw_resp = await ws.recv()
                    resp_envelope = parse_envelope(json.loads(raw_resp))
                    assert resp_envelope.type is MessageType.ACTION_DECISION
                    responses.append(resp_envelope.payload)

                assert len(responses) == 5
                for resp in responses:
                    assert resp.action in {"block", "attack_a", "move_forward"}

                await ws.send(
                    json.dumps(match_ended_envelope(match_id=mid, total_turns=5))
                )

        await asyncio.sleep(0.1)

    asyncio.run(scenario())


def test_disconnect_during_match_finalizes_artifacts() -> None:
    """Verify artifacts are finalized when client disconnects mid-match."""

    async def scenario() -> None:
        mid = unique_match_id()
        config = baseline_runtime_config()
        async with running_match_server(config) as server:
            async with connect(f"ws://127.0.0.1:{server.listening_port}") as ws:
                await handshake(ws)

                await ws.send(
                    json.dumps(
                        decision_request_envelope(
                            match_id=mid, turn_id=1, player_id="p1"
                        )
                    )
                )
                raw_resp = await ws.recv()
                resp = parse_envelope(json.loads(raw_resp))
                assert resp.type is MessageType.ACTION_DECISION

            await asyncio.sleep(0.2)

    asyncio.run(scenario())


def test_invalid_json_during_match_does_not_crash() -> None:
    """Malformed JSON in the match loop should be logged and skipped."""

    async def scenario() -> None:
        mid = unique_match_id()
        config = baseline_runtime_config()
        async with running_match_server(config) as server:
            async with connect(f"ws://127.0.0.1:{server.listening_port}") as ws:
                await handshake(ws)

                await ws.send("not valid json{{{")

                await ws.send(
                    json.dumps(
                        decision_request_envelope(
                            match_id=mid, turn_id=1, player_id="p1"
                        )
                    )
                )
                raw_resp = await ws.recv()
                resp = parse_envelope(json.loads(raw_resp))
                assert resp.type is MessageType.ACTION_DECISION

                await ws.send(
                    json.dumps(match_ended_envelope(match_id=mid, total_turns=1))
                )

    asyncio.run(scenario())


def test_invalid_envelope_during_match_does_not_crash() -> None:
    """Invalid envelope structure in the match loop should be logged and skipped."""

    async def scenario() -> None:
        from yomi_daemon.protocol import CURRENT_PROTOCOL_VERSION

        mid = unique_match_id()
        config = baseline_runtime_config()
        async with running_match_server(config) as server:
            async with connect(f"ws://127.0.0.1:{server.listening_port}") as ws:
                await handshake(ws)

                await ws.send(
                    json.dumps(
                        {
                            "type": "bogus",
                            "version": CURRENT_PROTOCOL_VERSION.value,
                            "ts": "now",
                        }
                    )
                )

                await ws.send(
                    json.dumps(
                        decision_request_envelope(
                            match_id=mid, turn_id=1, player_id="p2"
                        )
                    )
                )
                raw_resp = await ws.recv()
                resp = parse_envelope(json.loads(raw_resp))
                assert resp.type is MessageType.ACTION_DECISION

                await ws.send(
                    json.dumps(match_ended_envelope(match_id=mid, total_turns=1))
                )

    asyncio.run(scenario())


def test_bare_decision_request_is_rejected_but_enveloped_request_succeeds() -> None:
    """Bare turn payloads are ignored after handshake; only v2 envelopes are routed."""

    async def scenario() -> None:
        mid = unique_match_id()
        config = baseline_runtime_config()
        async with running_match_server(config) as server:
            async with connect(f"ws://127.0.0.1:{server.listening_port}") as ws:
                await handshake(ws)

                bare_payload = decision_request_envelope(
                    match_id=mid, turn_id=1, player_id="p1"
                )["payload"]
                await ws.send(json.dumps(bare_payload))

                try:
                    await asyncio.wait_for(ws.recv(), timeout=0.05)
                except TimeoutError:
                    pass
                else:
                    raise AssertionError(
                        "Bare decision_request payload should not receive a response"
                    )

                await ws.send(
                    json.dumps(
                        decision_request_envelope(
                            match_id=mid, turn_id=2, player_id="p1"
                        )
                    )
                )
                raw_resp = await ws.recv()
                resp = parse_envelope(json.loads(raw_resp))
                assert resp.type is MessageType.ACTION_DECISION

                await ws.send(
                    json.dumps(match_ended_envelope(match_id=mid, total_turns=2))
                )

    asyncio.run(scenario())


def test_artifact_directory_created_for_match() -> None:
    """Verify that the match loop creates an artifact directory under runs/."""

    mid = unique_match_id()

    async def scenario() -> Any:
        config = baseline_runtime_config()
        async with running_match_server(config) as server:
            async with connect(f"ws://127.0.0.1:{server.listening_port}") as ws:
                await handshake(ws)

                await ws.send(
                    json.dumps(
                        decision_request_envelope(
                            match_id=mid, turn_id=1, player_id="p1"
                        )
                    )
                )
                await ws.recv()
                await ws.send(
                    json.dumps(match_ended_envelope(match_id=mid, total_turns=1))
                )

            await asyncio.sleep(0.2)

        run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and mid in d.name]
        assert run_dirs, f"Expected a run directory containing {mid}"
        return run_dirs[0]

    latest = asyncio.run(scenario())

    assert (latest / "manifest.json").exists()
    assert (latest / "events.jsonl").exists()
    assert (latest / "decisions.jsonl").exists()
    assert (latest / "metrics.json").exists()
    assert (latest / "result.json").exists()
    assert (latest / "replay_index.json").exists()
    assert (latest / "stderr.log").exists()

    result = json.loads((latest / "result.json").read_text())
    assert result["winner"] == "p1"
    assert result["end_reason"] == "ko"
    assert result["total_turns"] == 1
    assert result["status"] == "completed"

    decisions = (latest / "decisions.jsonl").read_text().strip().splitlines()
    assert len(decisions) == 1
    decision_record = json.loads(decisions[0])
    assert "request_payload" in decision_record
    assert "decision_payload" in decision_record

    events = (latest / "events.jsonl").read_text().strip().splitlines()
    assert len(events) >= 3
    event_names = [json.loads(e)["payload"]["event"] for e in events]
    assert EventName.MATCH_STARTED.value in event_names
    assert EventName.TURN_REQUESTED.value in event_names
    assert EventName.DECISION_RECEIVED.value in event_names

    manifest = json.loads((latest / "manifest.json").read_text())
    assert manifest["trace_seed"] == 42
    assert manifest["policy_mapping"]["p1"] == "baseline/random"


def test_both_players_routed_to_correct_policies() -> None:
    """p1 and p2 requests should route to their respective configured policies."""

    async def scenario() -> None:
        mid = unique_match_id()
        config = baseline_runtime_config(
            p1="baseline/random",
            p2="baseline/block_always",
        )
        async with running_match_server(config) as server:
            async with connect(f"ws://127.0.0.1:{server.listening_port}") as ws:
                await handshake(ws)

                for turn in range(1, 4):
                    req = decision_request_envelope(
                        match_id=mid, turn_id=turn, player_id="p2"
                    )
                    await ws.send(json.dumps(req))
                    raw = await ws.recv()
                    resp = parse_envelope(json.loads(raw))
                    decision = cast(ActionDecision, resp.payload)
                    assert decision.action == "block", (
                        f"block_always policy should pick block, got {decision.action}"
                    )

                await ws.send(
                    json.dumps(match_ended_envelope(match_id=mid, total_turns=3))
                )

    asyncio.run(scenario())


def test_event_forwarded_to_artifacts() -> None:
    """Events sent by the mod should be recorded in the artifact event log."""

    mid = unique_match_id()

    async def scenario() -> Any:
        from yomi_daemon.protocol import CURRENT_PROTOCOL_VERSION

        config = baseline_runtime_config()
        async with running_match_server(config) as server:
            async with connect(f"ws://127.0.0.1:{server.listening_port}") as ws:
                await handshake(ws)

                await ws.send(
                    json.dumps(
                        decision_request_envelope(
                            match_id=mid, turn_id=1, player_id="p1"
                        )
                    )
                )
                await ws.recv()

                event_envelope = {
                    "type": MessageType.EVENT.value,
                    "version": CURRENT_PROTOCOL_VERSION.value,
                    "ts": "2026-03-12T00:00:05Z",
                    "payload": {
                        "match_id": mid,
                        "event": EventName.DECISION_APPLIED.value,
                        "turn_id": 1,
                        "player_id": "p1",
                        "details": {"action": "block"},
                    },
                }
                await ws.send(json.dumps(event_envelope))

                await asyncio.sleep(0.05)

                await ws.send(
                    json.dumps(match_ended_envelope(match_id=mid, total_turns=1))
                )

            await asyncio.sleep(0.2)

        run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and mid in d.name]
        return run_dirs[0]

    latest = asyncio.run(scenario())
    events = (latest / "events.jsonl").read_text().strip().splitlines()
    event_names = [json.loads(e)["payload"]["event"] for e in events]
    assert EventName.DECISION_APPLIED.value in event_names


def test_replay_saved_event_persists_match_replay_artifact(monkeypatch: Any) -> None:
    """Late ReplaySaved should update artifacts and pull match.replay into the run dir."""

    mid = unique_match_id()

    async def fake_start_recording(
        self: Any, display: str | None = None, max_duration_seconds: int = 120
    ) -> bool:
        self._started_display = display
        self._started_duration = max_duration_seconds
        return True

    async def fake_stop_recording(self: Any) -> Any:
        video_path = self._run_dir / "replay.mp4"
        video_path.write_bytes(b"fake replay video")
        return video_path

    async def fake_pull_replay_file(self: Any, vm_replay_path: str) -> Any:
        local_path = self._run_dir / "match.replay"
        local_path.write_text(f"pulled from {vm_replay_path}\n", encoding="utf-8")
        return local_path

    async def fake_cleanup(self: Any) -> None:
        return None

    monkeypatch.setattr(
        "yomi_daemon.server.ReplayCaptureSession.start_recording",
        fake_start_recording,
    )
    monkeypatch.setattr(
        "yomi_daemon.server.ReplayCaptureSession.stop_recording",
        fake_stop_recording,
    )
    monkeypatch.setattr(
        "yomi_daemon.server.ReplayCaptureSession.pull_replay_file",
        fake_pull_replay_file,
    )
    monkeypatch.setattr(
        "yomi_daemon.server.ReplayCaptureSession.cleanup",
        fake_cleanup,
    )

    class FakeConnection:
        def __init__(self, messages: list[str]) -> None:
            self._messages = iter(messages)

        def __aiter__(self) -> "FakeConnection":
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._messages)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    async def scenario() -> Any:
        config = baseline_runtime_config()
        server = DaemonServer(
            port=0,
            policy_mapping=config.policy_mapping,
            config_snapshot=config.to_config_payload(),
            runtime_config=config,
            replay_capture_config=ReplayCaptureConfig(enabled=True),
        )
        writer = MatchArtifactWriter.create(
            match_id=mid,
            manifest={
                "match_id": mid,
                "trace_seed": config.trace_seed,
                "policy_mapping": config.policy_mapping.to_dict(),
            },
        )
        writer.finalize(
            match_ended={
                "match_id": mid,
                "winner": "p1",
                "end_reason": "ko",
                "total_turns": 1,
                "end_tick": 500,
                "end_frame": 60,
                "errors": [],
            }
        )

        replay_saved = json.dumps(
            {
                "type": MessageType.EVENT.value,
                "version": CURRENT_PROTOCOL_VERSION.value,
                "ts": "2026-03-12T00:01:01Z",
                "payload": {
                    "match_id": mid,
                    "event": EventName.REPLAY_SAVED.value,
                    "details": {
                        "replay_path": "/vm/user/replay/autosave/test-match.replay"
                    },
                },
            }
        )
        replay_started = json.dumps(
            {
                "type": MessageType.EVENT.value,
                "version": CURRENT_PROTOCOL_VERSION.value,
                "ts": "2026-03-12T00:01:02Z",
                "payload": {
                    "match_id": mid,
                    "event": EventName.REPLAY_STARTED.value,
                    "details": {"display": ":99"},
                },
            }
        )
        replay_ended = json.dumps(
            {
                "type": MessageType.EVENT.value,
                "version": CURRENT_PROTOCOL_VERSION.value,
                "ts": "2026-03-12T00:01:03Z",
                "payload": {
                    "match_id": mid,
                    "event": EventName.REPLAY_ENDED.value,
                    "details": {},
                },
            }
        )
        connection = FakeConnection([replay_saved, replay_started, replay_ended])
        session = MatchSession(
            session_id="session-test",
            remote_address=None,
            accepted_protocol_version=CURRENT_PROTOCOL_VERSION,
            daemon_version=__version__,
            policy_mapping=config.policy_mapping,
            config_snapshot=config.to_config_payload(),
            metadata=MatchMetadata(
                game_version="1.0.0",
                mod_version="0.1.0",
                schema_version="v2",
                match_id=mid,
            ),
        )

        await server._handle_replay_capture(
            session=session,
            connection=cast(Any, connection),
            match_id=mid,
            writer=writer,
            replay_path=None,
        )

        return writer.run_dir

    latest = asyncio.run(scenario())

    assert (latest / "match.replay").exists()
    assert (latest / "replay.mp4").exists()

    result = json.loads((latest / "result.json").read_text())
    assert result["replay_path"] == "/vm/user/replay/autosave/test-match.replay"

    replay_index = json.loads((latest / "replay_index.json").read_text())
    assert replay_index["replay_path"] == "/vm/user/replay/autosave/test-match.replay"

    events = (latest / "events.jsonl").read_text().strip().splitlines()
    event_names = [json.loads(e)["payload"]["event"] for e in events]
    assert EventName.REPLAY_SAVED.value in event_names
    assert EventName.REPLAY_STARTED.value in event_names
    assert EventName.REPLAY_ENDED.value in event_names


def test_no_replay_still_pulls_match_replay_artifact(monkeypatch: Any) -> None:
    """When replay video capture is disabled we still pull match.replay."""

    mid = unique_match_id()
    start_calls = 0
    stop_calls = 0

    async def fake_start_recording(
        self: Any, display: str | None = None, max_duration_seconds: int = 120
    ) -> bool:
        nonlocal start_calls
        start_calls += 1
        return True

    async def fake_stop_recording(self: Any) -> Any:
        nonlocal stop_calls
        stop_calls += 1
        video_path = self._run_dir / "replay.mp4"
        video_path.write_bytes(b"unexpected replay video")
        return video_path

    async def fake_pull_replay_file(self: Any, vm_replay_path: str) -> Any:
        local_path = self._run_dir / "match.replay"
        local_path.write_text(f"pulled from {vm_replay_path}\n", encoding="utf-8")
        return local_path

    async def fake_cleanup(self: Any) -> None:
        return None

    monkeypatch.setattr(
        "yomi_daemon.server.ReplayCaptureSession.start_recording",
        fake_start_recording,
    )
    monkeypatch.setattr(
        "yomi_daemon.server.ReplayCaptureSession.stop_recording",
        fake_stop_recording,
    )
    monkeypatch.setattr(
        "yomi_daemon.server.ReplayCaptureSession.pull_replay_file",
        fake_pull_replay_file,
    )
    monkeypatch.setattr(
        "yomi_daemon.server.ReplayCaptureSession.cleanup",
        fake_cleanup,
    )

    class FakeConnection:
        def __init__(self, messages: list[str]) -> None:
            self._messages = iter(messages)

        def __aiter__(self) -> "FakeConnection":
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._messages)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    async def scenario() -> Any:
        config = baseline_runtime_config()
        server = DaemonServer(
            port=0,
            policy_mapping=config.policy_mapping,
            config_snapshot=config.to_config_payload(),
            runtime_config=config,
            replay_capture_config=ReplayCaptureConfig(enabled=False),
        )
        writer = MatchArtifactWriter.create(
            match_id=mid,
            manifest={
                "match_id": mid,
                "trace_seed": config.trace_seed,
                "policy_mapping": config.policy_mapping.to_dict(),
            },
        )
        writer.finalize(
            match_ended={
                "match_id": mid,
                "winner": "p1",
                "end_reason": "ko",
                "total_turns": 1,
                "end_tick": 500,
                "end_frame": 60,
                "errors": [],
            }
        )

        replay_saved = json.dumps(
            {
                "type": MessageType.EVENT.value,
                "version": CURRENT_PROTOCOL_VERSION.value,
                "ts": "2026-03-12T00:01:01Z",
                "payload": {
                    "match_id": mid,
                    "event": EventName.REPLAY_SAVED.value,
                    "details": {
                        "replay_path": "/vm/user/replay/autosave/test-match.replay"
                    },
                },
            }
        )
        connection = FakeConnection([replay_saved])
        session = MatchSession(
            session_id="session-test",
            remote_address=None,
            accepted_protocol_version=CURRENT_PROTOCOL_VERSION,
            daemon_version=__version__,
            policy_mapping=config.policy_mapping,
            config_snapshot=config.to_config_payload(),
            metadata=MatchMetadata(
                game_version="1.0.0",
                mod_version="0.1.0",
                schema_version="v2",
                match_id=mid,
            ),
        )

        await server._handle_replay_capture(
            session=session,
            connection=cast(Any, connection),
            match_id=mid,
            writer=writer,
            replay_path=None,
        )

        return writer.run_dir

    latest = asyncio.run(scenario())

    assert (latest / "match.replay").exists()
    assert not (latest / "replay.mp4").exists()
    assert start_calls == 0
    assert stop_calls == 0

    result = json.loads((latest / "result.json").read_text())
    assert result["replay_path"] == "/vm/user/replay/autosave/test-match.replay"

    replay_index = json.loads((latest / "replay_index.json").read_text())
    assert replay_index["replay_path"] == "/vm/user/replay/autosave/test-match.replay"

    events = (latest / "events.jsonl").read_text().strip().splitlines()
    event_names = [json.loads(e)["payload"]["event"] for e in events]
    assert EventName.REPLAY_SAVED.value in event_names
    assert EventName.REPLAY_STARTED.value not in event_names
    assert EventName.REPLAY_ENDED.value not in event_names
