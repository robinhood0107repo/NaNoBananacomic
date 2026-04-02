from __future__ import annotations

import argparse
import json
from pathlib import Path

from comic_pipeline.benchmark_detection import benchmark_detectors
from comic_pipeline.detectors.registry import list_detector_names
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
    detect_parser.add_argument(
        "--detector",
        default="auto",
        choices=["auto", *list_detector_names()],
        help="Detector backend. Use auto to resolve from the page profile.",
    )

    validate_parser = subparsers.add_parser(
        "validate-step1",
        help="Run the Step 1 validation checks against a detected page.",
    )
    validate_parser.add_argument("project_root", help="Path to the selected project folder.")
    validate_parser.add_argument("--page-id", required=True, help="Page id to validate, e.g. 0001.")

    benchmark_parser = subparsers.add_parser(
        "benchmark-detectors",
        help="Run a proxy benchmark across open-source detector candidates.",
    )
    benchmark_parser.add_argument("project_root", help="Path to the selected project folder.")
    benchmark_parser.add_argument(
        "--detectors",
        nargs="+",
        default=list_detector_names(),
        choices=list_detector_names(),
        help="Detector candidates to compare.",
    )
    benchmark_parser.add_argument(
        "--page-ids",
        nargs="+",
        default=None,
        help="Optional subset of page ids to benchmark.",
    )

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
            result = detect_page(project_root, args.page_id, detector_name=args.detector)
            print("[OK] step1 detection finished")
            for key, value in result.items():
                print(f"{key}: {value}")
            return 0

        if args.command == "validate-step1":
            report = validate_step1(project_root, args.page_id)
            print(format_report(report))
            return 0 if report.passed else 1

        if args.command == "benchmark-detectors":
            page_ids = args.page_ids
            if not page_ids:
                page_ids = [
                    path.name.replace(".page.json", "")
                    for path in sorted((project_root / "pages").glob("*.page.json"))
                ]
            summary = benchmark_detectors(
                project_root,
                detector_names=args.detectors,
                page_ids=page_ids,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0

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
