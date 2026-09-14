#!/usr/bin/env python3
"""
Usage:
    python main.py "The Pythagorean Theorem"
    python main.py "Why 0.999... equals 1" --quality qm
    VOICE_PROVIDER=omnivoice python main.py "Intro to Derivatives"

Takes a video title, generates a narration script + Manim storyboard with
Gemini, generates + renders a Manim scene per beat (with narration synced
in automatically via manim-voiceover), and stitches everything into one
final .mp4 with audio already embedded.
"""

from __future__ import annotations

import argparse
import sys

from pipeline.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a narrated math explainer video from a title.")
    parser.add_argument("title", help="The video's title / topic, e.g. \"The Pythagorean Theorem\"")
    parser.add_argument(
        "--quality", choices=["ql", "qm", "qh", "qk"], default=None,
        help="Render quality (default: from RENDER_QUALITY env var, or qh).",
    )
    args = parser.parse_args()

    try:
        final_path = run_pipeline(args.title, quality=args.quality)
    except Exception as e:  # noqa: BLE001 - top-level CLI error boundary
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(final_path)


if __name__ == "__main__":
    main()
