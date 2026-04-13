"""WebSocket daemon server, handshake, and match-turn dispatch loop."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from yomi_daemon import __version__
from yomi_daemon.adapters.base import PolicyAdapter, build_player_policy_adapters
from yomi_daemon.character_selection import resolve_character_assignments
from yomi_daemon.ids import SessionIdGenerator
from yomi_daemon.manifest import build_match_manifest
from yomi_daemon.match import MatchSession
from yomi_daemon.orchestrator import resolve_adapter_decision
from yomi_daemon.protocol import (
    ActionDecision,
    CharacterSelectionConfig,
    CharacterSelectionMode,
    ConfigPayload,
    CURRENT_SCHEMA_VERSION,
    DecisionRequest,
    Envelope,
    Event,
    EventName,
    Hello,
    HelloAck,
    MatchEnded,
    MessageType,
    PlayerPolicyMapping,
    PlayerSlot,
    ProtocolVersion,
    SUPPORTED_PROTOCOL_VERSIONS,
)
from yomi_daemon.redact import redact_secrets
from yomi_daemon.replay_capture import ReplayCaptureConfig, ReplayCaptureSession
from yomi_daemon.storage.writer import MatchArtifactWriter
from yomi_daemon.validation import ProtocolValidationError, parse_envelope

if TYPE_CHECKING:
    from yomi_daemon.config import DaemonRuntimeConfig


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
SUPPORTED_SCHEMA_VERSION = CURRENT_SCHEMA_VERSION


class HandshakeRejectedError(ValueError):
    """Raised when the first client message cannot complete a handshake."""


@dataclass(frozen=True, slots=True)
class ServerRuntimeConfig:
    policy_mapping: PlayerPolicyMapping
    config_snapshot: ConfigPayload | None = None


class DaemonServer:
    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        policy_mapping: PlayerPolicyMapping | None = None,
        config_snapshot: ConfigPayload | None = None,
        runtime_config: "DaemonRuntimeConfig | None" = None,
        auth_secret: str | None = None,
        replay_capture_config: ReplayCaptureConfig | None = None,
        match_history: dict[str, Any] | None = None,
        runs_root: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._runtime_config = runtime_config
        self._auth_secret = auth_secret
        self._runs_root = runs_root
        self._replay_capture_config = replay_capture_config or ReplayCaptureConfig()
        self._match_history = match_history
        self.runtime_config = ServerRuntimeConfig(
            policy_mapping=policy_mapping
            if policy_mapping is not None
            else PlayerPolicyMapping(
                p1="baseline/random",
                p2="baseline/random",
            ),
            config_snapshot=config_snapshot,
        )
        self.logger = logger or logging.getLogger("yomi_daemon.server")
        self._server: Server | None = None
        self._session_ids = SessionIdGenerator()
        self._active_sessions: dict[str, MatchSession] = {}
        self._stopped = asyncio.Event()
        self._stopped.set()

        # Build policy adapters if runtime config is available
        self._player_adapters: Mapping[PlayerSlot, PolicyAdapter] | None = None
        if self._runtime_config is not None:
            self._player_adapters = build_player_policy_adapters(self._runtime_config)

    @property
    def listening_port(self) -> int:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("Server is not listening")
        return int(self._server.sockets[0].getsockname()[1])

    @property
    def active_sessions(self) -> dict[str, MatchSession]:
        return dict(self._active_sessions)

    async def start(self) -> None:
        if self._server is not None:
            return

        if self.host not in ("127.0.0.1", "localhost", "::1"):
            self.logger.warning(
                "Server binding to non-local address %s — the daemon is designed for "
                "local use only. Exposing it to the network is not recommended.",
                self.host,
            )

        self._server = await serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=30,
        )
        self._stopped.clear()
        self.logger.info(
            "Daemon server listening on ws://%s:%d%s",
            self.host,
            self.listening_port,
            " (auth required)" if self._auth_secret else "",
        )

    async def stop(self) -> None:
        if self._server is None:
            return

        self._server.close()
        await self._server.wait_closed()
        self._server = None
        self._stopped.set()
        self.logger.info("Daemon server stopped")

    async def serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("Server must be started before serving forever")
        await self._stopped.wait()

    async def _handle_connection(self, connection: ServerConnection) -> None:
        session_id = self._session_ids.next()
        remote_address = self._format_remote_address(connection.remote_address)
        self.logger.info("Accepted connection %s from %s", session_id, remote_address)

        try:
            session = await self._perform_handshake(session_id, remote_address, connection)
        except HandshakeRejectedError as exc:
            self.logger.warning(
                "Rejected handshake for %s from %s: %s",
                session_id,
                remote_address,
                exc,
            )
            return
        except ConnectionClosed as exc:
            self.logger.info(
                "Connection %s closed during handshake from %s: %s",
                session_id,
                remote_address,
                exc,
            )
            return

        self._active_sessions[session.session_id] = session
        self.logger.info(
            "Handshake complete for %s using protocol=%s schema=%s",
            session.session_id,
            session.accepted_protocol_version.value,
            session.metadata.schema_version,
        )

        try:
            await self._run_match_loop(session, connection)
        finally:
            self._active_sessions.pop(session.session_id, None)
            self.logger.info("Session %s closed", session.session_id)

        # Stop the server after the first completed match to prevent
        # the mod from reconnecting and auto-starting another match.
        if self._runtime_config is not None:
            self.logger.info("Shutting down after completed match")
            await self.stop()

    async def _run_match_loop(
        self,
        session: MatchSession,
        connection: ServerConnection,
    ) -> None:
        """Post-handshake message loop: route decisions, write artifacts, handle match end."""

        if self._player_adapters is None or self._runtime_config is None:
            # No runtime config: legacy handshake-only mode (tests, basic server)
            try:
                await connection.wait_closed()
            except ConnectionClosed:
                pass
            return

        runtime_config = self._runtime_config

        # Per-player last valid decision tracking for LAST_VALID_REPLAYABLE fallback
        last_valid: dict[str, ActionDecision] = {}
        # Dedup: the game may fire player_actionable multiple times per tick,
        # causing the mod to send duplicate decision_requests for the same
        # (player_id, state_hash) pair.  Skip duplicates to avoid wasted LLM calls.
        seen_requests: set[tuple[str, str]] = set()
        pending_decisions: set[asyncio.Task[None]] = set()
        errors: list[str] = []
        match_ended_payload: MatchEnded | None = None
        match_id: str | None = None
        writer: MatchArtifactWriter | None = None

        try:
            async for raw_message in connection:
                if not isinstance(raw_message, str):
                    self.logger.warning(
                        "Session %s: ignoring binary frame",
                        session.session_id,
                    )
                    continue

                try:
                    raw_data = json.loads(raw_message)
                except json.JSONDecodeError as exc:
                    self.logger.warning(
                        "Session %s: invalid JSON: %s",
                        session.session_id,
                        exc.msg,
                    )
                    continue

                try:
                    envelope = parse_envelope(raw_data)
                except ProtocolValidationError as exc:
                    self.logger.warning(
                        "Session %s: invalid envelope: %s",
                        session.session_id,
                        exc,
                    )
                    continue

                if envelope.type is MessageType.DECISION_REQUEST:
                    request = cast(DecisionRequest, envelope.payload)

                    # Lazy initialization of match artifacts from first request
                    if writer is None:
                        match_id = request.match_id
                        session.metadata.match_id = match_id
                        manifest = build_match_manifest(
                            match_id=match_id,
                            runtime_config=runtime_config,
                            metadata=session.metadata,
                        )
                        writer = MatchArtifactWriter.create(
                            match_id=match_id,
                            manifest=manifest.to_dict(),
                            runs_root=self._runs_root,
                        )
                        self.logger.info(
                            "Match %s started for session %s, artifacts at %s",
                            match_id,
                            session.session_id,
                            writer.run_dir,
                        )
                        writer.append_event(
                            Event(
                                match_id=match_id,
                                event=EventName.MATCH_STARTED,
                                details={
                                    "session_id": session.session_id,
                                    "p1_policy": runtime_config.policy_mapping.p1,
                                    "p2_policy": runtime_config.policy_mapping.p2,
                                },
                            ).to_dict()
                        )

                        # Persist character selection traces
                        if session.character_selection_traces:
                            traces_path = writer.run_dir / "character_selection.json"
                            import json as _json

                            traces_data = [t.to_dict() for t in session.character_selection_traces]
                            traces_path.write_text(
                                _json.dumps(traces_data, indent=2) + "\n", encoding="utf-8"
                            )

                    # Dedup: skip if we've already dispatched for this
                    # (player_id, state_hash).  The mod may fire the same
                    # request twice when the game emits player_actionable
                    # multiple times per tick.
                    dedup_key = (request.player_id, request.state_hash)
                    if dedup_key in seen_requests:
                        self.logger.debug(
                            "Session %s: skipping duplicate request for %s state=%s",
                            session.session_id,
                            request.player_id,
                            request.state_hash[:12],
                        )
                        continue
                    seen_requests.add(dedup_key)

                    # Fire off decision requests concurrently so both
                    # players' LLM calls run in parallel.
                    task = asyncio.create_task(
                        self._handle_decision_request(
                            session=session,
                            connection=connection,
                            request=request,
                            writer=writer,
                            last_valid=last_valid,
                        )
                    )
                    pending_decisions.add(task)
                    task.add_done_callback(pending_decisions.discard)

                elif envelope.type is MessageType.EVENT:
                    event = cast(Event, envelope.payload)
                    if writer is not None:
                        writer.append_event(event.to_dict())
                    self.logger.debug(
                        "Session %s: recorded event %s turn=%s",
                        session.session_id,
                        event.event.value,
                        event.turn_id,
                    )

                elif envelope.type is MessageType.MATCH_ENDED:
                    match_ended_payload = cast(MatchEnded, envelope.payload)
                    self.logger.info(
                        "Session %s: match ended, winner=%s reason=%s turns=%d",
                        session.session_id,
                        match_ended_payload.winner,
                        match_ended_payload.end_reason,
                        match_ended_payload.total_turns,
                    )
                    # Wait for any in-flight decision tasks to complete
                    if pending_decisions:
                        await asyncio.gather(*pending_decisions, return_exceptions=True)
                    break

                else:
                    self.logger.debug(
                        "Session %s: ignoring message type %s in match loop",
                        session.session_id,
                        envelope.type.value,
                    )

        except ConnectionClosed as exc:
            msg = f"Connection lost during match: {exc}"
            self.logger.warning("Session %s: %s", session.session_id, msg)
            errors.append(msg)
        except Exception as exc:
            msg = f"Unexpected error in match loop: {redact_secrets(str(exc))}"
            self.logger.error("Session %s: %s", session.session_id, msg, exc_info=True)
            errors.append(msg)
            if writer is not None:
                writer.append_stderr(f"{msg}\n")
        finally:
            # Cancel and await any in-flight decision tasks
            for task in pending_decisions:
                task.cancel()
            if pending_decisions:
                await asyncio.gather(*pending_decisions, return_exceptions=True)
            pending_decisions.clear()

        # Finalize match artifacts (decisions, prompts, result)
        if writer is not None:
            writer.finalize(
                match_ended=(
                    match_ended_payload.to_dict() if match_ended_payload is not None else None
                ),
                end_reason=(
                    match_ended_payload.end_reason
                    if match_ended_payload is not None
                    else "disconnect"
                ),
                errors=errors,
            )
            self.logger.info(
                "Match %s finalized, result at %s",
                match_id,
                writer.result_path,
            )

            # Always try to pull the saved .replay file. When replay capture is
            # enabled we also wait for ReplayStarted/ReplayEnded and record an
            # MP4; when disabled we only wait briefly for ReplaySaved.
            if match_ended_payload is not None:
                await self._handle_replay_capture(
                    session=session,
                    connection=connection,
                    match_id=match_id or "",
                    writer=writer,
                    replay_path=match_ended_payload.replay_path,
                )
        else:
            self.logger.info(
                "Session %s closed without any decision requests",
                session.session_id,
            )

    async def _handle_decision_request(
        self,
        *,
        session: MatchSession,
        connection: ServerConnection,
        request: DecisionRequest,
        writer: MatchArtifactWriter,
        last_valid: dict[str, ActionDecision],
    ) -> None:
        """Route a single decision request to the correct policy adapter and respond."""

        assert self._player_adapters is not None
        assert self._runtime_config is not None
        runtime_config = self._runtime_config

        player_id = request.player_id
        slot = _player_id_to_slot(player_id)
        adapter = self._player_adapters.get(slot)

        if adapter is None:
            self.logger.error(
                "Session %s: no adapter for player %s (slot %s)",
                session.session_id,
                player_id,
                slot,
            )
            return

        # Log TurnRequested event
        writer.append_event(
            Event(
                match_id=request.match_id,
                event=EventName.TURN_REQUESTED,
                turn_id=request.turn_id,
                player_id=player_id,
                details={
                    "state_hash": request.state_hash,
                    "legal_actions_hash": request.legal_actions_hash,
                    "legal_action_count": len(request.legal_actions),
                    "policy_id": adapter.id,
                },
            ).to_dict()
        )

        # Resolve decision via traced adapter path (persists prompt + decision artifacts)
        result = await resolve_adapter_decision(
            request,
            adapter=adapter,
            artifact_writer=writer,
            logging_config=runtime_config.logging,
        )

        decision = result.decision
        used_fallback = decision.fallback_reason is not None

        # Track last valid for fallback
        if not used_fallback:
            last_valid[player_id] = decision

        # Emit telemetry event
        if used_fallback:
            self.logger.warning(
                "FALLBACK turn %d: %s (%s) -> %s",
                request.turn_id,
                adapter.id,
                player_id,
                decision.fallback_reason,
            )
            writer.append_event(
                Event(
                    match_id=request.match_id,
                    event=EventName.DECISION_FALLBACK,
                    turn_id=request.turn_id,
                    player_id=player_id,
                    fallback_reason=decision.fallback_reason,
                    latency_ms=decision.latency_ms,
                    details={
                        "policy_id": adapter.id,
                    },
                ).to_dict()
            )
        else:
            writer.append_event(
                Event(
                    match_id=request.match_id,
                    event=EventName.DECISION_RECEIVED,
                    turn_id=request.turn_id,
                    player_id=player_id,
                    latency_ms=decision.latency_ms,
                    details={
                        "action": decision.action,
                        "policy_id": adapter.id,
                    },
                ).to_dict()
            )

        # Send action_decision envelope back to mod
        response_envelope = Envelope(
            type=MessageType.ACTION_DECISION,
            version=session.accepted_protocol_version,
            ts=_utc_now(),
            payload=decision,
        )
        try:
            await connection.send(json.dumps(response_envelope.to_dict()))
        except ConnectionClosed:
            self.logger.warning(
                "Session %s: connection closed while sending turn %d for %s",
                session.session_id,
                request.turn_id,
                player_id,
            )
            return

        self.logger.debug(
            "Session %s: turn %d %s (%s) -> action=%s fallback=%s",
            session.session_id,
            request.turn_id,
            adapter.id,
            player_id,
            decision.action,
            used_fallback,
        )

        # Periodic HP status (every 10 turns, on either player's turn)
        if request.turn_id % 10 == 0:
            try:
                fighters = request.observation.fighters
                if len(fighters) >= 2:
                    f1, f2 = fighters[0], fighters[1]
                    self.logger.info(
                        "HP_STATUS turn %d: %s %s HP vs %s %s HP",
                        request.turn_id,
                        f1.character,
                        f1.hp,
                        f2.character,
                        f2.hp,
                    )
            except Exception:
                pass

    async def _handle_replay_capture(
        self,
        *,
        session: MatchSession,
        connection: ServerConnection,
        match_id: str,
        writer: MatchArtifactWriter,
        replay_path: str | None,
    ) -> None:
        """Pull match.replay for every match and optionally record replay video."""

        capture = ReplayCaptureSession(
            config=self._replay_capture_config,
            match_id=match_id,
            run_dir=writer.run_dir,
            logger=self.logger,
        )
        record_video = self._replay_capture_config.enabled
        timeout_seconds = 120 if record_video else 15

        effective_replay_path = replay_path
        pulled_replay_path: str | None = None

        # Pull the .replay file immediately when the path is already known.
        if replay_path:
            writer.update_replay_path(replay_path)
            await capture.pull_replay_file(replay_path)
            pulled_replay_path = replay_path
            if not record_video:
                await capture.cleanup()
                return

        wait_target = "replay events" if record_video else "ReplaySaved"
        self.logger.info(
            "Session %s: waiting for %s (timeout %ss)...",
            session.session_id,
            wait_target,
            timeout_seconds,
        )

        try:
            async with asyncio.timeout(timeout_seconds):
                async for raw_message in connection:
                    if not isinstance(raw_message, str):
                        continue

                    try:
                        raw_data = json.loads(raw_message)
                    except json.JSONDecodeError:
                        continue

                    try:
                        envelope = parse_envelope(raw_data)
                    except ProtocolValidationError:
                        continue

                    if envelope.type is not MessageType.EVENT:
                        continue

                    event = cast(Event, envelope.payload)
                    writer.append_event(event.to_dict())

                    if event.event is EventName.REPLAY_SAVED:
                        replay_path_raw = event.details.get("replay_path")
                        if isinstance(replay_path_raw, str) and replay_path_raw:
                            effective_replay_path = replay_path_raw
                            writer.update_replay_path(replay_path_raw)
                            if pulled_replay_path != replay_path_raw:
                                await capture.pull_replay_file(replay_path_raw)
                                pulled_replay_path = replay_path_raw
                            if not record_video:
                                break
                        else:
                            self.logger.warning(
                                "Session %s: ReplaySaved missing replay_path",
                                session.session_id,
                            )

                    elif event.event is EventName.REPLAY_STARTED:
                        if not record_video:
                            continue
                        display = str(
                            event.details.get("display", self._replay_capture_config.display)
                        )
                        self.logger.info(
                            "Session %s: replay started on display %s, beginning recording",
                            session.session_id,
                            display,
                        )
                        # Calculate recording duration from match HP.
                        # Timer ticks = starting_hp * 3 (our scaling).
                        # Replay runs at half speed (30 ticks/s, playback_speed_mod=2).
                        # Add 15s padding for match-end screen + post-KO animation.
                        starting_hp = 1500  # game default
                        if (
                            self._runtime_config
                            and self._runtime_config.match_options
                            and self._runtime_config.match_options.starting_hp
                        ):
                            starting_hp = self._runtime_config.match_options.starting_hp
                        timer_ticks = starting_hp * 3
                        replay_seconds = timer_ticks / 30  # half-speed = 30 ticks/s
                        record_duration = int(replay_seconds + 15)
                        await capture.start_recording(
                            display=display, max_duration_seconds=record_duration
                        )

                    elif event.event is EventName.REPLAY_ENDED:
                        if not record_video:
                            continue
                        self.logger.info(
                            "Session %s: replay ended, stopping recording",
                            session.session_id,
                        )
                        # ffmpeg was started with -t and will stop on its own.
                        # Wait for it to finish, then pull the video.
                        video_path = await capture.stop_recording()
                        if video_path:
                            self.logger.info("Replay video saved to %s", video_path)
                        break

        except (TimeoutError, ConnectionClosed):
            phase = "replay capture" if record_video else "replay file wait"
            self.logger.info(
                "Session %s: %s phase ended (timeout or disconnect)",
                session.session_id,
                phase,
            )
            # Pull the video even if ffmpeg already exited (was_started covers
            # the case where -t duration expired before ReplayEnded arrived).
            if record_video and (capture.is_recording or capture.was_started):
                video_path = await capture.stop_recording()
                if video_path:
                    self.logger.info("Replay video saved to %s (after timeout)", video_path)
            if effective_replay_path and pulled_replay_path != effective_replay_path:
                writer.update_replay_path(effective_replay_path)
                await capture.pull_replay_file(effective_replay_path)
            await capture.cleanup()
            return

        if record_video and (capture.is_recording or capture.was_started):
            video_path = await capture.stop_recording()
            if video_path:
                self.logger.info("Replay video saved to %s", video_path)
        if effective_replay_path and pulled_replay_path != effective_replay_path:
            writer.update_replay_path(effective_replay_path)
            await capture.pull_replay_file(effective_replay_path)
        await capture.cleanup()

    async def _perform_handshake(
        self,
        session_id: str,
        remote_address: str | None,
        connection: ServerConnection,
    ) -> MatchSession:
        raw_message = await connection.recv()
        if not isinstance(raw_message, str):
            await self._close_for_handshake_error(
                connection,
                code=1003,
                reason="expected text handshake frame",
            )
            raise HandshakeRejectedError("expected a text frame")

        try:
            raw_envelope = self._load_json_object(raw_message)
            self._preflight_handshake(raw_envelope)
        except HandshakeRejectedError as exc:
            await self._close_for_handshake_error(connection, code=1002, reason=str(exc))
            raise

        try:
            envelope = parse_envelope(raw_envelope)
        except ProtocolValidationError as exc:
            await self._close_for_handshake_error(connection, code=1007, reason=str(exc))
            raise HandshakeRejectedError(str(exc)) from exc

        if envelope.type is not MessageType.HELLO:
            await self._close_for_handshake_error(
                connection,
                code=1002,
                reason="expected hello envelope",
            )
            raise HandshakeRejectedError("expected hello envelope")

        hello = envelope.payload
        if not isinstance(hello, Hello):
            await self._close_for_handshake_error(
                connection,
                code=1002,
                reason="hello payload did not decode correctly",
            )
            raise HandshakeRejectedError("hello payload did not decode correctly")

        # Shared-secret authentication check
        if self._auth_secret is not None:
            client_token = hello.auth_token or ""
            if not hmac.compare_digest(client_token, self._auth_secret):
                await self._close_for_handshake_error(
                    connection,
                    code=1008,
                    reason="authentication failed",
                )
                raise HandshakeRejectedError("authentication failed: invalid or missing auth_token")

        accepted_version = negotiate_protocol_version(hello.supported_protocol_versions)
        if accepted_version is None:
            await self._close_for_handshake_error(
                connection,
                code=1002,
                reason="unsupported protocol version",
            )
            raise HandshakeRejectedError("unsupported protocol version")
        if hello.schema_version != SUPPORTED_SCHEMA_VERSION:
            await self._close_for_handshake_error(
                connection,
                code=1002,
                reason="unsupported schema version",
            )
            raise HandshakeRejectedError("unsupported schema version")

        # Resolve LLM character selection before building session
        config_snapshot = self.runtime_config.config_snapshot
        char_select_traces: list[Any] = []
        if (
            self._runtime_config is not None
            and self._runtime_config.character_selection.mode is CharacterSelectionMode.LLM_CHOICE
        ):
            self.logger.info("Resolving LLM character choices for session %s...", session_id)
            char_result = await resolve_character_assignments(
                self._runtime_config, match_history=self._match_history
            )
            assignments = char_result.assignments
            char_select_traces = char_result.traces
            # Replace character_selection in config snapshot with concrete assignments
            if config_snapshot is not None:
                config_snapshot = ConfigPayload(
                    decision_timeout_ms=config_snapshot.decision_timeout_ms,
                    fallback_mode=config_snapshot.fallback_mode,
                    logging=config_snapshot.logging,
                    policy_mapping=config_snapshot.policy_mapping,
                    character_selection=CharacterSelectionConfig(
                        mode=CharacterSelectionMode.ASSIGNED,
                        assignments=assignments,
                    ),
                    stage_id=config_snapshot.stage_id,
                    match_options=config_snapshot.match_options,
                )
            self.logger.info(
                "Character selection resolved: P1=%s, P2=%s",
                assignments.p1,
                assignments.p2,
            )

        session = MatchSession.from_hello(
            session_id=session_id,
            remote_address=remote_address,
            hello=hello,
            accepted_protocol_version=accepted_version,
            daemon_version=__version__,
            policy_mapping=self.runtime_config.policy_mapping,
            config_snapshot=config_snapshot,
        )
        if char_select_traces:
            session.character_selection_traces = char_select_traces
        await connection.send(json.dumps(self._build_hello_ack_envelope(session).to_dict()))
        return session

    async def _close_for_handshake_error(
        self,
        connection: ServerConnection,
        *,
        code: int,
        reason: str,
    ) -> None:
        await connection.close(code=code, reason=reason[:123])

    def _load_json_object(self, raw_message: str) -> Mapping[str, object]:
        try:
            decoded = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            raise HandshakeRejectedError(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(decoded, Mapping):
            raise HandshakeRejectedError("handshake frame must decode to an object")
        return decoded

    def _preflight_handshake(self, raw_envelope: Mapping[str, object]) -> None:
        message_type = raw_envelope.get("type")
        if message_type != MessageType.HELLO.value:
            raise HandshakeRejectedError("expected hello envelope")

        raw_version = raw_envelope.get("version")
        if raw_version not in {item.value for item in SUPPORTED_PROTOCOL_VERSIONS}:
            raise HandshakeRejectedError(f"unsupported envelope version: {raw_version!r}")

        payload = raw_envelope.get("payload")
        if not isinstance(payload, Mapping):
            raise HandshakeRejectedError("hello payload must be an object")
        payload_mapping = cast(Mapping[str, object], payload)

        raw_schema_version = payload_mapping.get("schema_version")
        if not isinstance(raw_schema_version, str):
            raise HandshakeRejectedError("hello.schema_version must be a string")
        if raw_schema_version != SUPPORTED_SCHEMA_VERSION:
            raise HandshakeRejectedError(f"unsupported schema version: {raw_schema_version!r}")

        raw_supported_versions = payload_mapping.get("supported_protocol_versions")
        if not isinstance(raw_supported_versions, Sequence) or isinstance(
            raw_supported_versions, str | bytes | bytearray
        ):
            raise HandshakeRejectedError("hello.supported_protocol_versions must be an array")
        if not any(
            version in {item.value for item in SUPPORTED_PROTOCOL_VERSIONS}
            for version in raw_supported_versions
            if isinstance(version, str)
        ):
            raise HandshakeRejectedError(
                f"unsupported protocol versions: {list(raw_supported_versions)!r}"
            )

    def _build_hello_ack_envelope(self, session: MatchSession) -> Envelope:
        return Envelope(
            type=MessageType.HELLO_ACK,
            version=session.accepted_protocol_version,
            ts=_utc_now(),
            payload=HelloAck(
                accepted_protocol_version=session.accepted_protocol_version,
                accepted_schema_version=SUPPORTED_SCHEMA_VERSION,
                daemon_version=session.daemon_version,
                policy_mapping=session.policy_mapping,
                config=session.config_snapshot,
            ),
        )

    @staticmethod
    def _format_remote_address(remote_address: Any) -> str | None:
        if remote_address is None:
            return None
        if isinstance(remote_address, tuple):
            return ":".join(str(part) for part in remote_address[:2])
        return str(remote_address)


def _player_id_to_slot(player_id: str) -> PlayerSlot:
    """Map a player_id string to a PlayerSlot enum value."""
    try:
        return PlayerSlot(player_id)
    except ValueError:
        # Fallback: treat "p1"/"p2" case-insensitively
        normalized = player_id.strip().lower()
        return PlayerSlot(normalized)


def negotiate_protocol_version(
    supported_versions: Sequence[ProtocolVersion],
) -> ProtocolVersion | None:
    client_supported = set(supported_versions)
    for version in SUPPORTED_PROTOCOL_VERSIONS:
        if version in client_supported:
            return version
    return None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
