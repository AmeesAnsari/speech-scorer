"""Command line: ``speech-scorer TRANSCRIPT.json SCRIPT.txt [--json]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .scoring import Settings, score
from .transcript import load_words, make_normalizer


def _fmt(seconds) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def render(r) -> str:
    lines = [
        f"Score: {r.score:.1f} / 100",
        "",
        f"  Accuracy   {r.accuracy:6.1%}   (WER {r.wer:.1%}: {r.substitutions} substituted, "
        f"{r.deletions} missed, {r.insertions} extra, of {r.script_words} script words)",
        f"  Coverage   {r.coverage:6.1%}   of the script spoken correctly",
        f"  Pace       {r.pace_wpm or 0:6.0f} wpm  ({r.pace_in_range:.0%} of time in range; "
        f"windows {r.slowest_window_wpm or 0:.0f}-{r.fastest_window_wpm or 0:.0f} wpm)",
        f"  Reading    {_fmt(r.reading_start)} to {_fmt(r.reading_end)}",
    ]
    if r.feedback:
        lines += ["", "Feedback:"] + [f"  - {tip}" for tip in r.feedback]
    if r.low_confidence_words:
        lines += ["", f"Low-confidence words from the recogniser: {', '.join(r.low_confidence_words[:10])}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="speech-scorer",
                                     description="Score a timed speech transcript against its script.")
    parser.add_argument("transcript", type=Path, help="Whisper-style JSON with word timestamps")
    parser.add_argument("script", type=Path, help="the text the speaker was meant to read")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--pace", nargs=2, type=float, metavar=("MIN", "MAX"), default=(100, 160),
                        help="target words per minute (default 100 160)")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    script = args.script.read_text(encoding="utf-8")
    words = load_words(args.transcript, make_normalizer(script))
    report = score(script, words, Settings(pace_range_wpm=tuple(args.pace)))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
