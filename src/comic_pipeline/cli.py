from __future__ import annotations

import argparse
from pathlib import Path

from comic_pipeline.project import create_project, scan_pages
from comic_pipeline.step1 import detect_page, format_report, validate_step1
from comic_pipeline.types import RuntimeDependencyError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comic-pipeline",
        description="NaNoBananacomic local pipeline scaffold for Step 1 testing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-project", help="Create the project structure.")
    init_parser.add_argument("project_root", help="Path to the selected project folder.")
    init_parser.add_argument(
        "--profile-mode",
        default="auto",
        choices=["auto", "bw_manga", "color_comic", "three_d_comic"],
        help="Initial profile mode stored in project.json.",
    )

    scan_parser = subparsers.add_parser("scan-pages", help="Discover image files in the project root.")
    scan_parser.add_argument("project_root", help="Path to the selected project folder.")

    detect_parser = subparsers.add_parser("detect", help="Run Step 1 speech-bubble detection.")
    detect_parser.add_argument("project_root", help="Path to the selected project folder.")
    detect_parser.add_argument("--page-id", required=True, help="Page id to detect, e.g. 0001.")

    validate_parser = subparsers.add_parser(
        "validate-step1",
        help="Run the Step 1 validation checks against a detected page.",
    )
    validate_parser.add_argument("project_root", help="Path to the selected project folder.")
    validate_parser.add_argument("--page-id", required=True, help="Page id to validate, e.g. 0001.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()

    try:
        if args.command == "init-project":
            manifest = create_project(project_root, profile_mode=args.profile_mode)
            print(f"[OK] project initialized: {manifest.project_root}")
            return 0

        if args.command == "scan-pages":
            pages = scan_pages(project_root)
            print(f"[OK] discovered {len(pages)} page(s)")
            for page in pages:
                print(f" - {page.page_id}: {page.original_path}")
            return 0

        if args.command == "detect":
            result = detect_page(project_root, args.page_id)
            print("[OK] step1 detection finished")
            for key, value in result.items():
                print(f"{key}: {value}")
            return 0

        if args.command == "validate-step1":
            report = validate_step1(project_root, args.page_id)
            print(format_report(report))
            return 0 if report.passed else 1

    except RuntimeDependencyError as exc:
        print(f"[ERROR] {exc}")
        return 2
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
