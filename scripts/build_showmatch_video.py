#!/usr/bin/env python3
"""Build a single ffmpeg-only combined video for a replay-recorded showmatch.

This script reads `series_summary.json` from a showmatch directory, creates:
- a series intro card that explicitly says "Best of 7"
- per-game title cards with the running series score
- an ending card with the final result

Then it concatenates everything into one output mp4. It never modifies the
source replay videos.

Example:
    uv run --project daemon python scripts/build_showmatch_video.py \
      runs/showmatches/gemini_bo7_20260418T190500Z
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_STEM = "showmatch_cut_v1"
DEFAULT_INTRO_SECONDS = 3.0
DEFAULT_GAME_CARD_SECONDS = 1.75
DEFAULT_OUTRO_SECONDS = 2.5
DEFAULT_START_TRIM_SECONDS = 2.1
DEFAULT_END_OVERLAY_KEEP_SECONDS = 0.2
FPS = 30
WIDTH = 64
HEIGHT = 36
FRAME_SIZE = WIDTH * HEIGHT
OVERLAY_XS = 20
OVERLAY_XE = 44
OVERLAY_YS = 13
OVERLAY_YE = 23
OVERLAY_PIXEL_THRESHOLD = 180
OVERLAY_BRIGHT_COUNT_THRESHOLD = 8
OVERLAY_WINDOW_FRAMES = 8
OVERLAY_MIN_BRIGHT_FRAMES = 4
OVERLAY_SEARCH_BACK_SECONDS = 8.0
TAIL_MIN_SECONDS = 2.0
OVERLAY_CROP_IDXS = [
    y * WIDTH + x
    for y in range(OVERLAY_YS, OVERLAY_YE)
    for x in range(OVERLAY_XS, OVERLAY_XE)
]
FONT_CANDIDATES = [
    REPO_ROOT / "docs/decompile-output/project/ui/PixeloidSans.ttf",
    REPO_ROOT / "docs/decompile-output/project/ui/monobit.ttf",
    Path("/System/Library/Fonts/SFNS.ttf"),
    Path("/System/Library/Fonts/Supplemental/Helvetica.ttc"),
    Path("/System/Library/Fonts/Supplemental/Menlo.ttc"),
]


@dataclass(frozen=True, slots=True)
class ShowmatchGameCard:
    game_index: int
    source_clip: str
    candidate_side: str
    score_entering: str
    score_after: str
    output_card: str
    start_trim_seconds: float
    end_trim_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ShowmatchBuildManifest:
    generated_at_utc: str
    showmatch_dir: str
    output_video: str
    candidate_label: str
    opponent_label: str
    intro_seconds: float
    game_card_seconds: float
    outro_seconds: float
    game_cards: list[dict[str, Any]]
    source_videos: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined showmatch video with ffmpeg title cards.",
    )
    parser.add_argument("showmatch_dir", type=Path)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional explicit series_summary.json path. Defaults to <showmatch_dir>/series_summary.json.",
    )
    parser.add_argument(
        "--candidate-label",
        default="Qwen/Qwen3-8B-rl",
        help="Display label for the champion side (default: Qwen/Qwen3-8B-rl).",
    )
    parser.add_argument(
        "--opponent-label",
        default="Gemini 3.1 Pro",
        help="Display label for the opponent side (default: Gemini 3.1 Pro).",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help=f"Output filename stem, without extension (default: {DEFAULT_OUTPUT_STEM}).",
    )
    parser.add_argument(
        "--intro-seconds",
        type=float,
        default=DEFAULT_INTRO_SECONDS,
        help=f"Intro card duration in seconds (default: {DEFAULT_INTRO_SECONDS}).",
    )
    parser.add_argument(
        "--game-card-seconds",
        type=float,
        default=DEFAULT_GAME_CARD_SECONDS,
        help=f"Per-game card duration in seconds (default: {DEFAULT_GAME_CARD_SECONDS}).",
    )
    parser.add_argument(
        "--outro-seconds",
        type=float,
        default=DEFAULT_OUTRO_SECONDS,
        help=f"Outro card duration in seconds (default: {DEFAULT_OUTRO_SECONDS}).",
    )
    parser.add_argument(
        "--start-trim-seconds",
        type=float,
        default=DEFAULT_START_TRIM_SECONDS,
        help=f"Seconds to trim from the start of each replay (default: {DEFAULT_START_TRIM_SECONDS}).",
    )
    parser.add_argument(
        "--end-overlay-keep-seconds",
        type=float,
        default=DEFAULT_END_OVERLAY_KEEP_SECONDS,
        help=(
            "Seconds of the final win overlay to keep before trimming away the static tail "
            f"(default: {DEFAULT_END_OVERLAY_KEEP_SECONDS})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest but do not encode the combined video.",
    )
    return parser.parse_args()


def ensure_tooling() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"Required tool(s) not found on PATH: {', '.join(missing)}")


def choose_font_file() -> Path | None:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def choose_output_path(showmatch_dir: Path, output_stem: str) -> Path:
    base = showmatch_dir / f"{output_stem}.mp4"
    if not base.exists():
        return base
    for index in range(2, 100):
        candidate = showmatch_dir / f"{output_stem}_v{index}.mp4"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not choose output path under {showmatch_dir}")


def ffprobe_video_info(path: Path) -> tuple[int, int, Fraction]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    stream = payload["streams"][0]
    return int(stream["width"]), int(stream["height"]), Fraction(stream["avg_frame_rate"])


def decode_frames(path: Path) -> list[bytes]:
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={FPS},scale={WIDTH}:{HEIGHT},format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    frames: list[bytes] = []
    while True:
        buf = proc.stdout.read(FRAME_SIZE)
        if len(buf) < FRAME_SIZE:
            break
        frames.append(buf)
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}")
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def analyze_frames(frames: list[bytes]) -> list[int]:
    bright_counts: list[int] = []
    for frame in frames:
        bright_counts.append(
            sum(1 for idx in OVERLAY_CROP_IDXS if frame[idx] > OVERLAY_PIXEL_THRESHOLD)
        )
    return bright_counts


def detect_overlay_start_seconds(bright_counts: list[int], *, duration_seconds: float) -> float | None:
    search_start = max(duration_seconds - OVERLAY_SEARCH_BACK_SECONDS, 0.0)
    start_frame = max(int(search_start * FPS), 0)
    end_frame = max(start_frame, len(bright_counts) - OVERLAY_WINDOW_FRAMES)
    for frame_idx in range(start_frame, end_frame + 1):
        window = bright_counts[frame_idx : frame_idx + OVERLAY_WINDOW_FRAMES]
        if sum(1 for count in window if count >= OVERLAY_BRIGHT_COUNT_THRESHOLD) >= OVERLAY_MIN_BRIGHT_FRAMES:
            return frame_idx / FPS
    return None


def detect_end_trim_seconds(
    source_clip: Path,
    *,
    start_trim_seconds: float,
    end_overlay_keep_seconds: float,
) -> float | None:
    frames = decode_frames(source_clip)
    duration_seconds = len(frames) / FPS
    if duration_seconds <= start_trim_seconds + 0.1:
        return None
    overlay_start_seconds = detect_overlay_start_seconds(
        analyze_frames(frames), duration_seconds=duration_seconds
    )
    if overlay_start_seconds is None:
        return None
    trimmed_end = min(
        max(overlay_start_seconds + end_overlay_keep_seconds, start_trim_seconds + 0.1),
        duration_seconds,
    )
    tail_removed = duration_seconds - trimmed_end
    if tail_removed < TAIL_MIN_SECONDS:
        return None
    return round(trimmed_end, 3)


def escape_drawtext(value: str) -> str:
    return (
        value.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("%", r"\%")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


def drawtext_filter(
    *,
    text: str,
    x: str,
    y: str,
    font_size: int,
    font_color: str,
    font_file: Path | None,
    shadow: bool = True,
    alpha: str = "1",
) -> str:
    parts = [
        f"drawtext=text='{escape_drawtext(text)}'",
        f"x={x}",
        f"y={y}",
        f"fontsize={font_size}",
        f"fontcolor={font_color}",
        f"alpha={alpha}",
    ]
    if font_file is not None:
        parts.append(f"fontfile={font_file}")
    if shadow:
        parts.append("shadowcolor=0x000000")
        parts.append("shadowx=2")
        parts.append("shadowy=2")
    return ":".join(parts)


def build_intro_filter(
    *,
    width: int,
    height: int,
    duration: float,
    candidate_label: str,
    opponent_label: str,
    font_file: Path | None,
) -> str:
    filters = [
        f"color=c=black:s={width}x{height}:r=30:d={duration}",
        f"drawbox=x=18:y=18:w={width - 36}:h={height - 36}:color=white:t=2",
        f"drawbox=x=64:y=76:w={width - 128}:h=2:color=white:t=fill",
        drawtext_filter(
            text="BEST OF 7",
            x="(w-text_w)/2",
            y="96",
            font_size=56,
            font_color="white",
            font_file=font_file,
        ),
        drawtext_filter(
            text=f"{candidate_label} vs {opponent_label}",
            x="(w-text_w)/2",
            y="190",
            font_size=38,
            font_color="white",
            font_file=font_file,
        ),
        drawtext_filter(
            text="Cowboy mirror",
            x="(w-text_w)/2",
            y="278",
            font_size=22,
            font_color="0xbbbbbb",
            font_file=font_file,
        ),
        "format=yuv420p",
    ]
    return ",".join(filters)


def build_game_filter(
    *,
    width: int,
    height: int,
    duration: float,
    game_index: int,
    candidate_label: str,
    opponent_label: str,
    candidate_side: str,
    score_entering: str,
    font_file: Path | None,
) -> str:
    candidate_role = f"{candidate_label} ({candidate_side.upper()})"
    opponent_side = "P2" if candidate_side == "p1" else "P1"
    opponent_role = f"{opponent_label} ({opponent_side})"
    filters = [
        f"color=c=black:s={width}x{height}:r=30:d={duration}",
        f"drawbox=x=18:y=18:w={width - 36}:h={height - 36}:color=white:t=2",
        drawtext_filter(
            text=f"GAME {game_index}",
            x="(w-text_w)/2",
            y="96",
            font_size=48,
            font_color="white",
            font_file=font_file,
        ),
        drawtext_filter(
            text=f"Series score entering: {score_entering}",
            x="(w-text_w)/2",
            y="165",
            font_size=28,
            font_color="0xdddddd",
            font_file=font_file,
        ),
        drawtext_filter(
            text=candidate_role,
            x="(w-text_w)/2",
            y="245",
            font_size=34,
            font_color="white",
            font_file=font_file,
        ),
        drawtext_filter(
            text="VS",
            x="(w-text_w)/2",
            y="300",
            font_size=28,
            font_color="white",
            font_file=font_file,
        ),
        drawtext_filter(
            text=opponent_role,
            x="(w-text_w)/2",
            y="345",
            font_size=34,
            font_color="white",
            font_file=font_file,
        ),
        "format=yuv420p",
    ]
    return ",".join(filters)


def build_outro_filter(
    *,
    width: int,
    height: int,
    duration: float,
    candidate_label: str,
    opponent_label: str,
    candidate_wins: int,
    opponent_wins: int,
    font_file: Path | None,
) -> str:
    filters = [
        f"color=c=black:s={width}x{height}:r=30:d={duration}",
        f"drawbox=x=18:y=18:w={width - 36}:h={height - 36}:color=white:t=2",
        drawtext_filter(
            text="FINAL SCORE",
            x="(w-text_w)/2",
            y="110",
            font_size=48,
            font_color="white",
            font_file=font_file,
        ),
        drawtext_filter(
            text=f"{candidate_label} {candidate_wins} - {opponent_wins} {opponent_label}",
            x="(w-text_w)/2",
            y="220",
            font_size=38,
            font_color="white",
            font_file=font_file,
        ),
        drawtext_filter(
            text=(candidate_label + " wins the showmatch") if candidate_wins > opponent_wins else (opponent_label + " wins the showmatch"),
            x="(w-text_w)/2",
            y="295",
            font_size=28,
            font_color="0xdddddd",
            font_file=font_file,
        ),
        "format=yuv420p",
    ]
    return ",".join(filters)


def encode_lavfi_card(*, filtergraph: str, duration: float, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            filtergraph,
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def encode_trimmed_clip(
    *,
    source_clip: Path,
    output_path: Path,
    start_trim_seconds: float,
    end_trim_seconds: float | None,
) -> None:
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_trim_seconds:.3f}",
    ]
    if end_trim_seconds is not None:
        cmd.extend(["-to", f"{end_trim_seconds:.3f}"])
    cmd.extend(
        [
            "-i",
            str(source_clip),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    subprocess.run(cmd, check=True)


def load_summary(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text())
    if payload.get("series_type") != "best_of_7":
        raise RuntimeError(f"Unsupported showmatch summary type: {payload.get('series_type')}")
    return payload


def build_game_cards(
    summary: dict[str, Any],
    *,
    showmatch_dir: Path,
    start_trim_seconds: float,
    end_overlay_keep_seconds: float,
) -> list[ShowmatchGameCard]:
    candidate_wins = 0
    opponent_wins = 0
    cards: list[ShowmatchGameCard] = []
    for game in summary["games"]:
        game_index = int(game["game_index"])
        source_clip = (REPO_ROOT / game["replay_video"]).resolve()
        score_entering = f"{candidate_wins}-{opponent_wins}"
        outcome = game["outcome"]
        if outcome == "win":
            candidate_wins += 1
        elif outcome == "loss":
            opponent_wins += 1
        score_after = f"{candidate_wins}-{opponent_wins}"
        cards.append(
            ShowmatchGameCard(
                game_index=game_index,
                source_clip=str(source_clip),
                candidate_side=str(game["candidate_side"]),
                score_entering=score_entering,
                score_after=score_after,
                output_card=str(showmatch_dir / f"game_{game_index:02d}_card.mp4"),
                start_trim_seconds=round(start_trim_seconds, 3),
                end_trim_seconds=detect_end_trim_seconds(
                    source_clip,
                    start_trim_seconds=start_trim_seconds,
                    end_overlay_keep_seconds=end_overlay_keep_seconds,
                ),
            )
        )
    return cards


def write_manifest(path: Path, manifest: ShowmatchBuildManifest) -> None:
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")


def main() -> int:
    args = parse_args()
    ensure_tooling()
    for name, value in (
        ("--intro-seconds", args.intro_seconds),
        ("--game-card-seconds", args.game_card_seconds),
        ("--outro-seconds", args.outro_seconds),
    ):
        if value <= 0:
            raise SystemExit(f"{name} must be positive")

    showmatch_dir = args.showmatch_dir.resolve()
    if not showmatch_dir.is_dir():
        raise SystemExit(f"Showmatch directory not found: {showmatch_dir}")

    summary_path = args.summary_path.resolve() if args.summary_path else showmatch_dir / "series_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Series summary not found: {summary_path}")

    summary = load_summary(summary_path)
    output_path = choose_output_path(showmatch_dir, args.output_stem)
    manifest_path = showmatch_dir / f"{output_path.stem}_manifest.json"
    game_cards = build_game_cards(
        summary,
        showmatch_dir=showmatch_dir,
        start_trim_seconds=args.start_trim_seconds,
        end_overlay_keep_seconds=args.end_overlay_keep_seconds,
    )
    source_videos = [card.source_clip for card in game_cards]
    if not source_videos:
        raise SystemExit("Showmatch summary contains no games")

    width, height, _ = ffprobe_video_info(Path(source_videos[0]))
    font_file = choose_font_file()
    intro_filter = build_intro_filter(
        width=width,
        height=height,
        duration=args.intro_seconds,
        candidate_label=args.candidate_label,
        opponent_label=args.opponent_label,
        font_file=font_file,
    )
    outro_filter = build_outro_filter(
        width=width,
        height=height,
        duration=args.outro_seconds,
        candidate_label=args.candidate_label,
        opponent_label=args.opponent_label,
        candidate_wins=int(summary["candidate_wins"]),
        opponent_wins=int(summary["opponent_wins"]),
        font_file=font_file,
    )

    manifest = ShowmatchBuildManifest(
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        showmatch_dir=str(showmatch_dir),
        output_video=str(output_path),
        candidate_label=args.candidate_label,
        opponent_label=args.opponent_label,
        intro_seconds=round(args.intro_seconds, 3),
        game_card_seconds=round(args.game_card_seconds, 3),
        outro_seconds=round(args.outro_seconds, 3),
        game_cards=[asdict(card) for card in game_cards],
        source_videos=source_videos,
    )
    write_manifest(manifest_path, manifest)

    if args.dry_run:
        print(f"Manifest: {manifest_path}")
        print(f"Output (not encoded): {output_path}")
        return 0

    with tempfile.TemporaryDirectory(prefix="yomi_showmatch_video_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        intro_path = temp_dir / "intro.mp4"
        outro_path = temp_dir / "outro.mp4"
        encode_lavfi_card(filtergraph=intro_filter, duration=args.intro_seconds, output_path=intro_path)
        encode_lavfi_card(filtergraph=outro_filter, duration=args.outro_seconds, output_path=outro_path)

        concat_entries = [intro_path]
        for card in game_cards:
            filtergraph = build_game_filter(
                width=width,
                height=height,
                duration=args.game_card_seconds,
                game_index=card.game_index,
                candidate_label=args.candidate_label,
                opponent_label=args.opponent_label,
                candidate_side=card.candidate_side,
                score_entering=card.score_entering,
                font_file=font_file,
            )
            card_path = temp_dir / f"game_{card.game_index:02d}_card.mp4"
            clip_path = temp_dir / f"game_{card.game_index:02d}_trimmed.mp4"
            encode_lavfi_card(
                filtergraph=filtergraph,
                duration=args.game_card_seconds,
                output_path=card_path,
            )
            encode_trimmed_clip(
                source_clip=Path(card.source_clip),
                output_path=clip_path,
                start_trim_seconds=card.start_trim_seconds,
                end_trim_seconds=card.end_trim_seconds,
            )
            concat_entries.extend([card_path, clip_path])
        concat_entries.append(outro_path)

        list_path = temp_dir / "concat.txt"
        list_path.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in concat_entries) + "\n"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )

    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
