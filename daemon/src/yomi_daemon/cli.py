"""CLI entry points for the daemon package."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from yomi_daemon import __version__
from yomi_daemon.config import ConfigError, RuntimeConfigOverrides, load_runtime_config
from yomi_daemon.replay_capture import ReplayCaptureConfig
from yomi_daemon.server import DaemonServer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yomi-daemon",
        description="Run the local YOMI Hustle arena daemon.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a daemon runtime config JSON file.",
    )
    parser.add_argument("--host", default=None, help="Override the WebSocket listen host.")
    parser.add_argument(
        "--port", type=int, default=None, help="Override the WebSocket listen port."
    )
    parser.add_argument(
        "--p1-policy",
        default=None,
        help="Override the policy ID assigned to player slot p1.",
    )
    parser.add_argument(
        "--p2-policy",
        default=None,
        help="Override the policy ID assigned to player slot p2.",
    )
    parser.add_argument(
        "--trace-seed",
        type=int,
        default=None,
        help="Override the manifest trace seed used for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Daemon log verbosity.",
    )
    parser.add_argument(
        "--record-replay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record replay video via ffmpeg in the OrbStack VM after match ends (default: from config, or on).",
    )
    parser.add_argument(
        "--replay-vm",
        default=None,
        help="OrbStack VM machine name for replay recording (default: from config, or ubuntu).",
    )
    parser.add_argument(
        "--replay-display",
        default=None,
        help="X display for replay recording (default: from config, or :99).",
    )
    parser.add_argument(
        "--match-history",
        type=Path,
        default=None,
        help="JSON file with per-player match history for character selection context.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Override the directory where match artifacts are written (default: runs/).",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> int:
    runtime_config = load_runtime_config(
        args.config,
        overrides=RuntimeConfigOverrides(
            host=args.host,
            port=args.port,
            p1_policy=args.p1_policy,
            p2_policy=args.p2_policy,
            trace_seed=args.trace_seed,
        ),
    )
    # CLI flags override config file values for replay capture
    replay_capture_config = ReplayCaptureConfig(
        enabled=args.record_replay
        if args.record_replay is not None
        else runtime_config.replay_capture.enabled,
        vm_machine=args.replay_vm or runtime_config.replay_capture.vm_machine,
        display=args.replay_display or runtime_config.replay_capture.display,
        resolution=runtime_config.replay_capture.resolution,
        framerate=runtime_config.replay_capture.framerate,
        video_codec=runtime_config.replay_capture.video_codec,
        preset=runtime_config.replay_capture.preset,
    )
    # Load match history for character selection context (tournament series)
    match_history = None
    if args.match_history and args.match_history.is_file():
        import json

        from yomi_daemon.prompt import MatchHistoryEntry

        raw = json.loads(args.match_history.read_text(encoding="utf-8"))
        match_history = {}
        for player_slot, entries in raw.items():
            match_history[player_slot] = [
                MatchHistoryEntry(
                    your_character=e["your_character"],
                    opponent_character=e["opponent_character"],
                    result=e["result"],
                    your_final_hp=e["your_final_hp"],
                    opponent_final_hp=e["opponent_final_hp"],
                )
                for e in entries
            ]

    server = DaemonServer(
        host=runtime_config.transport.host,
        port=runtime_config.transport.port,
        policy_mapping=runtime_config.policy_mapping,
        config_snapshot=runtime_config.to_config_payload(),
        runtime_config=runtime_config,
        auth_secret=runtime_config.transport.auth_secret,
        replay_capture_config=replay_capture_config,
        match_history=match_history,
        runs_root=args.runs_root,
    )
    await server.start()
    try:
        await server.serve_forever()
    finally:
        await server.stop()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Suppress noisy websockets handshake errors from port-check probes
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

    try:
        return asyncio.run(_run_async(args))
    except ConfigError as exc:
        logging.getLogger("yomi_daemon.cli").error(str(exc))
        return 2
    except KeyboardInterrupt:
        logging.getLogger("yomi_daemon.cli").info(
            "Stopped daemon %s from CLI interrupt",
            __version__,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
