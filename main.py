#!/usr/bin/env python3
"""
Usage:
    python main.py "The Pythagorean Theorem"
    python main.py "Why 0.999... equals 1" --quality qm
    python main.py "Introduction to Sets" --description "Use a friendly classroom example."
    python main.py --olx /path/to/course --list
    python main.py --olx /path/to/course --unit chapter_u01 --print-brief
    python main.py --olx /path/to/course --unit chapter_u01
    python main.py --olx /path/to/course --section vert_phy_u01_s05

Takes a video title, or a MyMarian OLX unit/section, generates a narration
script + Manim storyboard with OpenAI, generates + renders a Manim scene per
beat (with narration synced in automatically via manim-voiceover), and
stitches everything into one final .mp4 with audio already embedded.
"""

from __future__ import annotations

import argparse
import sys

from pipeline.olx import brief_for_section, brief_for_unit, format_outline, load_olx
from pipeline.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a narrated explainer video from a title or OLX lesson.")
    parser.add_argument("title", nargs="?", help="The video's title / topic, e.g. \"The Pythagorean Theorem\"")
    parser.add_argument(
        "--quality", choices=["ql", "qm", "qh", "qk"], default=None,
        help="Render quality (default: from RENDER_QUALITY env var, or ql).",
    )
    parser.add_argument(
        "--description",
        "--prompt",
        dest="description",
        default=None,
        help="Optional creative direction for the narration and visuals.",
    )
    parser.add_argument("--olx", help="Path to an Open edX OLX directory, course.xml, or .tar.gz export.")
    parser.add_argument("--list", action="store_true", help="Print the OLX course outline and exit.")
    parser.add_argument("--unit", help="OLX unit/chapter id or name to generate, e.g. chapter_u01.")
    parser.add_argument("--section", help="OLX vertical/section id or name to generate, e.g. vert_phy_u01_s05.")
    parser.add_argument("--print-brief", action="store_true", help="Print the extracted lesson brief and do not render.")
    args = parser.parse_args()

    title = args.title
    description = args.description

    try:
        if args.olx:
            course = load_olx(args.olx)
            if args.list:
                print(format_outline(course))
                return
            if args.section:
                brief = brief_for_section(course, args.section)
            elif args.unit:
                brief = brief_for_unit(course, args.unit)
            else:
                parser.error("With --olx, pass --list, --unit, or --section.")
            title = brief.title
            description = brief.extra_instructions()
            if args.description:
                description = description + "\n\nAdditional direction:\n" + args.description.strip()
            print(f"OLX {brief.course_id} / {brief.unit_id}: {len(brief.section_ids)} lesson page(s)", flush=True)
            if args.print_brief:
                print(description)
                return
        elif args.list or args.unit or args.section or args.print_brief:
            parser.error("--list, --unit, --section, and --print-brief require --olx.")
        if not title:
            parser.error("Provide a title, or use --olx with --unit/--section.")

        final_path = run_pipeline(
            title,
            quality=args.quality,
            extra_instructions=description,
        )
    except Exception as e:  # noqa: BLE001 - top-level CLI error boundary
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(final_path)


if __name__ == "__main__":
    main()
