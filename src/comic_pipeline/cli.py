from __future__ import annotations

import argparse
import json
from pathlib import Path

from comic_pipeline.benchmark_detection import benchmark_detectors
from comic_pipeline.make_balloons_only import (
    format_step2_report,
    make_balloons_only,
    validate_step2,
)
from comic_pipeline.step3 import (
    format_step3_report,
    import_external_result,
    make_handoff,
    run_external_edit,
    validate_step3,
)
from comic_pipeline.detectors.registry import list_detector_names
from comic_pipeline.project import create_project, scan_pages
from comic_pipeline.step1 import detect_page, format_report, validate_step1
from comic_pipeline.types import RuntimeDependencyError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comic-pipeline",
        description="NaNoBananacomic local pipeline scaffold for the bubble masking workflow.",
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
    init_parser.add_argument(
        "--target-language",
        default="한국어",
        help="Target language for Step 3 translation/typesetting. Default: 한국어.",
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

    make_layer_parser = subparsers.add_parser(
        "make-layer",
        help="Run Step 2 balloon-only RGBA generation.",
    )
    make_layer_parser.add_argument("project_root", help="Path to the selected project folder.")
    make_layer_parser.add_argument("--page-id", required=True, help="Page id to process, e.g. 0001.")

    validate_step2_parser = subparsers.add_parser(
        "validate-step2",
        help="Run the Step 2 validation checks against a balloon-only RGBA layer.",
    )
    validate_step2_parser.add_argument("project_root", help="Path to the selected project folder.")
    validate_step2_parser.add_argument("--page-id", required=True, help="Page id to validate, e.g. 0001.")

    make_handoff_parser = subparsers.add_parser(
        "make-handoff",
        help="Prepare the Step 3 handoff package for manual web or API execution.",
    )
    make_handoff_parser.add_argument("project_root", help="Path to the selected project folder.")
    make_handoff_parser.add_argument("--page-id", required=True, help="Page id to process, e.g. 0001.")

    import_result_parser = subparsers.add_parser(
        "import-external-result",
        help="Import an externally edited image and normalize it into the Step 3 raw contract.",
    )
    import_result_parser.add_argument("project_root", help="Path to the selected project folder.")
    import_result_parser.add_argument("--page-id", required=True, help="Page id to import, e.g. 0001.")
    import_result_parser.add_argument(
        "--input",
        required=True,
        help="Path to the user-supplied external edit result.",
    )

    validate_step3_parser = subparsers.add_parser(
        "validate-step3",
        help="Run the Step 3 validation checks against the normalized raw external result.",
    )
    validate_step3_parser.add_argument("project_root", help="Path to the selected project folder.")
    validate_step3_parser.add_argument("--page-id", required=True, help="Page id to validate, e.g. 0001.")

    run_external_edit_parser = subparsers.add_parser(
        "run-external-edit",
        help="Run the Step 3 API automatic external edit flow.",
    )
    run_external_edit_parser.add_argument("project_root", help="Path to the selected project folder.")
    run_external_edit_parser.add_argument("--page-id", required=True, help="Page id to process, e.g. 0001.")

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
            manifest = create_project(
                project_root,
                profile_mode=args.profile_mode,
                target_language=args.target_language,
            )
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

        if args.command == "make-layer":
            result = make_balloons_only(project_root, args.page_id)
            print("[OK] step2 layer generation finished")
            for key, value in result.items():
                print(f"{key}: {value}")
            return 0 if result["passed"] else 1

        if args.command == "validate-step2":
            report = validate_step2(project_root, args.page_id)
            print(format_step2_report(report))
            return 0 if report.passed else 1

        if args.command == "make-handoff":
            result = make_handoff(project_root, args.page_id)
            print("[OK] step3 handoff package created")
            for key, value in result.items():
                print(f"{key}: {value}")
            return 0

        if args.command == "import-external-result":
            result = import_external_result(project_root, args.page_id, Path(args.input).resolve())
            print("[OK] step3 external result import finished")
            for key, value in result.items():
                print(f"{key}: {value}")
            return 0 if result["passed"] else 1

        if args.command == "validate-step3":
            report = validate_step3(project_root, args.page_id)
            print(format_step3_report(report))
            return 0 if report.passed else 1

        if args.command == "run-external-edit":
            result = run_external_edit(project_root, args.page_id)
            print("[OK] step3 API execution finished")
            for key, value in result.items():
                print(f"{key}: {value}")
            return 0 if result["passed"] else 1

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
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
