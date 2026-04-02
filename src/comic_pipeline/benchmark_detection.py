from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from comic_pipeline.detectors.registry import build_detector
from comic_pipeline.project import load_page_manifest, write_json
from comic_pipeline.step1 import (
    _require_cv_runtime,
    build_overlay_preview,
    build_union_mask,
    summarize_step1_validation,
)


def benchmark_detectors(
    project_root: Path,
    *,
    detector_names: list[str],
    page_ids: list[str],
) -> dict:
    cv2, _ = _require_cv_runtime()
    run_id = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    benchmark_root = project_root / "artifacts" / "benchmarks" / run_id
    results: list[dict] = []
    summary_rows: list[dict] = []

    for detector_name in detector_names:
        detector = build_detector(detector_name)
        detector_result_root = benchmark_root / detector.name
        detector_result_root.mkdir(parents=True, exist_ok=True)
        detector_rows: list[dict] = []
        detector_error: str | None = None

        try:
            for page_id in page_ids:
                page_manifest = load_page_manifest(project_root, page_id)
                image_path = project_root / page_manifest.original_path
                image = cv2.imread(str(image_path))
                if image is None:
                    raise FileNotFoundError(f"Unable to read image: {image_path}")

                started = perf_counter()
                predictions = detector.predict(image)
                runtime_ms = (perf_counter() - started) * 1000.0
                union_mask = build_union_mask(image.shape[:2], predictions)
                overlay_preview = build_overlay_preview(image, union_mask)

                mask_path = detector_result_root / f"{page_id}_union.png"
                overlay_path = detector_result_root / f"{page_id}_overlay.png"
                cv2.imwrite(str(mask_path), union_mask)
                cv2.imwrite(str(overlay_path), overlay_preview)

                report = summarize_step1_validation(
                    page_id=page_id,
                    width=image.shape[1],
                    height=image.shape[0],
                    predictions=predictions,
                    mask_nonzero_pixels=int((union_mask > 0).sum()),
                )
                row = {
                    "page_id": page_id,
                    "detector_name": detector.name,
                    "prediction_count": len(predictions),
                    "runtime_ms": runtime_ms,
                    "area_ratio": report.area_ratio,
                    "largest_prediction_area_ratio": report.largest_prediction_area_ratio,
                    "passed": report.passed,
                    "notes": report.notes,
                    "mask_path": str(mask_path.relative_to(project_root)),
                    "overlay_path": str(overlay_path.relative_to(project_root)),
                }
                detector_rows.append(row)
                results.append(row)
        except Exception as exc:
            detector_error = f"{type(exc).__name__}: {exc}"

        pass_count = sum(1 for row in detector_rows if row["passed"])
        suspicious_count = sum(
            1
            for row in detector_rows
            if any("largest prediction area ratio" in note for note in row["notes"])
        )
        mean_prediction_count = mean(row["prediction_count"] for row in detector_rows) if detector_rows else 0.0
        mean_runtime_ms = mean(row["runtime_ms"] for row in detector_rows) if detector_rows else 0.0
        mean_area_ratio = mean(row["area_ratio"] for row in detector_rows) if detector_rows else 0.0
        summary_rows.append(
            {
                "detector_name": detector.name,
                "page_count": len(detector_rows),
                "pass_count": pass_count,
                "pass_rate": pass_count / max(len(detector_rows), 1),
                "suspicious_count": suspicious_count,
                "mean_prediction_count": mean_prediction_count,
                "mean_runtime_ms": mean_runtime_ms,
                "mean_area_ratio": mean_area_ratio,
                "error": detector_error or "",
            }
        )

    summary_rows.sort(
        key=lambda row: (
            row["pass_count"],
            -row["suspicious_count"],
            -row["mean_runtime_ms"],
        ),
        reverse=True,
    )
    winner = summary_rows[0]["detector_name"] if summary_rows else ""
    summary = {
        "run_id": run_id,
        "page_ids": page_ids,
        "detectors": detector_names,
        "winner_detector": winner,
        "method": "proxy_benchmark_without_ground_truth",
        "rows": summary_rows,
    }
    write_json(benchmark_root / "benchmark_results.json", {"summary": summary, "rows": results})
    markdown_lines = [
        "# Detector Benchmark",
        "",
        f"- run_id: `{run_id}`",
        f"- winner_detector: `{winner}`",
        "- method: proxy benchmark without GT labels",
        "",
        "| detector | pass_count | pass_rate | suspicious_count | mean_prediction_count | mean_area_ratio | mean_runtime_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        markdown_lines.append(
            "| "
            f"{row['detector_name']} | {row['pass_count']} | {row['pass_rate']:.3f} | "
            f"{row['suspicious_count']} | {row['mean_prediction_count']:.2f} | "
            f"{row['mean_area_ratio']:.4f} | {row['mean_runtime_ms']:.2f} |"
        )
        if row["error"]:
            markdown_lines.append(f"> error: `{row['error']}`")
    (benchmark_root / "benchmark_results.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    return summary
