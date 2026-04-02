from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from comic_pipeline.make_balloons_only import build_soft_alpha_from_union_mask
from comic_pipeline.project import (
    load_page_manifest,
    load_project_manifest,
    save_page_manifest,
    save_project_manifest,
    write_json,
)
from comic_pipeline.types import RuntimeDependencyError, Step3ValidationReport

SUPPORTED_INTEGRATION_MODES = frozenset({"api_auto", "manual_web"})
GEMINI_MODEL_ALIASES = {
    "nano-banana-2": "gemini-3.1-flash-image-preview",
    "nano-banana-pro": "gemini-3-pro-image-preview",
    "nano-banana": "gemini-2.5-flash-image",
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image": "gemini-2.5-flash-image",
}

_SESSION_PROVIDER_CREDENTIALS: dict[str, str] = {}


class Step3NormalizationError(RuntimeError):
    """Raised when an external edit result cannot be normalized into the raw contract."""

    def __init__(
        self,
        message: str,
        *,
        readable_image: bool,
        size_matches: bool,
        has_alpha_channel: bool = False,
    ) -> None:
        super().__init__(message)
        self.readable_image = readable_image
        self.size_matches = size_matches
        self.has_alpha_channel = has_alpha_channel


def _require_step3_runtime() -> tuple[object, object, object]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
        from PIL import UnidentifiedImageError  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "Step 3 runtime requires numpy and Pillow. "
            "Install the dependencies from pyproject.toml before running Step 3 commands."
        ) from exc
    return np, Image, UnidentifiedImageError


def set_session_api_key(provider: str, api_key: str) -> None:
    _SESSION_PROVIDER_CREDENTIALS[provider.strip().lower()] = api_key.strip()


def clear_session_api_key(provider: str | None = None) -> None:
    if provider is None:
        _SESSION_PROVIDER_CREDENTIALS.clear()
        return
    _SESSION_PROVIDER_CREDENTIALS.pop(provider.strip().lower(), None)


def _resolve_provider_api_key(provider: str) -> str | None:
    normalized_provider = provider.strip().lower()
    if normalized_provider in _SESSION_PROVIDER_CREDENTIALS:
        return _SESSION_PROVIDER_CREDENTIALS[normalized_provider]
    if normalized_provider == "gemini":
        return os.environ.get("GEMINI_API_KEY")
    return None


def _resolve_gemini_model_id(model_name: str) -> str:
    normalized_name = model_name.strip()
    return GEMINI_MODEL_ALIASES.get(normalized_name, normalized_name)


def _project_timestamp(project_root: Path) -> None:
    project_manifest = load_project_manifest(project_root)
    project_manifest.last_run_id = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    save_project_manifest(project_root, project_manifest)


def _as_relative(project_root: Path, path: Path) -> str:
    return str(path.relative_to(project_root))


def _load_page_context(project_root: Path, page_id: str) -> dict[str, Any]:
    np, Image, _ = _require_step3_runtime()
    project_manifest = load_project_manifest(project_root)
    page_manifest = load_page_manifest(project_root, page_id)

    if project_manifest.nano_integration_mode not in SUPPORTED_INTEGRATION_MODES:
        raise ValueError(
            "Unsupported nano_integration_mode in project.json: "
            f"{project_manifest.nano_integration_mode}"
        )

    if not page_manifest.balloons_only_rgba_path:
        raise FileNotFoundError(
            f"Page {page_id} has no Step 2 layer yet. Run make-layer first."
        )
    if not page_manifest.balloon_union_mask_path:
        raise FileNotFoundError(
            f"Page {page_id} has no saved union mask. Run detect first."
        )

    original_path = project_root / page_manifest.original_path
    layer_path = project_root / page_manifest.balloons_only_rgba_path
    union_mask_path = project_root / page_manifest.balloon_union_mask_path

    if not original_path.exists():
        raise FileNotFoundError(f"Original page does not exist: {original_path}")
    if not layer_path.exists():
        raise FileNotFoundError(f"Step 2 layer does not exist: {layer_path}")
    if not union_mask_path.exists():
        raise FileNotFoundError(f"Union mask does not exist: {union_mask_path}")

    with Image.open(original_path) as image:
        expected_size = image.size
    if page_manifest.width <= 0 or page_manifest.height <= 0:
        page_manifest.width, page_manifest.height = expected_size
        save_page_manifest(project_root, page_manifest)
    expected_size = (page_manifest.width, page_manifest.height)

    return {
        "np": np,
        "image_cls": Image,
        "project_manifest": project_manifest,
        "page_manifest": page_manifest,
        "original_path": original_path,
        "layer_path": layer_path,
        "union_mask_path": union_mask_path,
        "expected_size": expected_size,
    }


def build_step3_prompt(
    *,
    page_id: str,
    width: int,
    height: int,
    include_original_page: bool,
) -> str:
    context_line = (
        "You will also receive the original full page as context only. "
        "Use it to understand tone, panel flow, and scene context, but do not redraw the page."
        if include_original_page
        else "You will only receive the balloon-only transparent layer."
    )
    return (
        f"Edit the provided page-sized transparent speech-balloon layer for page {page_id}.\n\n"
        "Requirements:\n"
        f"- Keep the exact canvas size at {width}x{height}.\n"
        "- Only edit inside the visible speech balloons.\n"
        "- Translate the Japanese source text into natural Korean.\n"
        "- Typeset the Korean text cleanly inside each balloon.\n"
        "- Clean up the balloon interior as needed for typesetting.\n"
        "- Keep everything outside the balloons transparent.\n"
        "- Do not redraw the full manga page or create new non-balloon artwork.\n"
        "- Return exactly one edited image.\n\n"
        f"{context_line}\n"
    )


def _write_png_copy(image_path: Path, output_path: Path) -> None:
    _, Image, UnidentifiedImageError = _require_step3_runtime()
    try:
        with Image.open(image_path) as image:
            converted = image.convert("RGBA" if ("A" in image.getbands() or "transparency" in image.info) else "RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            converted.save(output_path, format="PNG")
    except (OSError, UnidentifiedImageError) as exc:
        raise Step3NormalizationError(
            f"Unable to read image for PNG conversion: {image_path}",
            readable_image=False,
            size_matches=False,
        ) from exc


def make_handoff(project_root: Path, page_id: str) -> dict[str, Any]:
    context = _load_page_context(project_root, page_id)
    project_manifest = context["project_manifest"]
    page_manifest = context["page_manifest"]
    request_dir = project_root / "artifacts" / "handoff" / page_id
    request_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = request_dir / "prompt.md"
    request_manifest_path = request_dir / "request.json"
    layer_copy_path = request_dir / "balloons_only_rgba.png"
    shutil.copy2(context["layer_path"], layer_copy_path)

    original_copy_path: Path | None = None
    if project_manifest.nano_include_original_page:
        original_copy_path = request_dir / "original_page.png"
        _write_png_copy(context["original_path"], original_copy_path)

    prompt = build_step3_prompt(
        page_id=page_id,
        width=page_manifest.width,
        height=page_manifest.height,
        include_original_page=project_manifest.nano_include_original_page,
    )
    prompt_path.write_text(prompt, encoding="utf-8")

    request_payload = {
        "page_id": page_id,
        "integration_mode": project_manifest.nano_integration_mode,
        "provider": project_manifest.nano_provider,
        "model_label": project_manifest.nano_model,
        "resolved_model_id": _resolve_gemini_model_id(project_manifest.nano_model)
        if project_manifest.nano_provider == "gemini"
        else project_manifest.nano_model,
        "expected_output": {
            "filename": f"{page_id}_raw.png",
            "canvas_width": page_manifest.width,
            "canvas_height": page_manifest.height,
            "format": "png",
        },
        "inputs": {
            "balloons_only_rgba": "balloons_only_rgba.png",
            "original_page": "original_page.png" if original_copy_path else None,
        },
        "prompt_path": "prompt.md",
    }
    write_json(request_manifest_path, request_payload)

    page_manifest.nano_request_dir = _as_relative(project_root, request_dir)
    page_manifest.nano_request_prompt_path = _as_relative(project_root, prompt_path)
    page_manifest.nano_request_manifest_path = _as_relative(project_root, request_manifest_path)
    save_page_manifest(project_root, page_manifest)
    _project_timestamp(project_root)

    return {
        "page_id": page_id,
        "nano_request_dir": page_manifest.nano_request_dir,
        "nano_request_prompt_path": page_manifest.nano_request_prompt_path,
        "nano_request_manifest_path": page_manifest.nano_request_manifest_path,
        "status": page_manifest.status,
        "passed": True,
    }


def _save_step3_failure_report(
    project_root: Path,
    page_id: str,
    *,
    source_kind: str | None,
    readable_image: bool,
    size_matches: bool,
    has_alpha_channel: bool,
    notes: list[str],
) -> Step3ValidationReport:
    report = Step3ValidationReport(
        page_id=page_id,
        source_kind=source_kind,
        readable_image=readable_image,
        size_matches=size_matches,
        has_alpha_channel=has_alpha_channel,
        opaque_output=readable_image and not has_alpha_channel,
        outside_support_rgb_nonzero_pixels=0,
        outside_support_alpha_nonzero_pixels=0,
        passed=False,
        notes=notes,
    )
    report_path = project_root / "artifacts" / "debug" / f"{page_id}_step3_validation.json"
    write_json(report_path, report.to_dict())
    page_manifest = load_page_manifest(project_root, page_id)
    page_manifest.step3_validation_report_path = _as_relative(project_root, report_path)
    page_manifest.status = "check"
    save_page_manifest(project_root, page_manifest)
    return report


def _preserve_manual_import(project_root: Path, page_id: str, input_path: Path) -> Path:
    project_manifest = load_project_manifest(project_root)
    imports_dir = project_root / project_manifest.imports_dir / "nano"
    imports_dir.mkdir(parents=True, exist_ok=True)
    suffix = input_path.suffix.lower() or ".png"
    destination = imports_dir / f"{page_id}_submitted{suffix}"
    shutil.copy2(input_path, destination)
    page_manifest = load_page_manifest(project_root, page_id)
    page_manifest.nano_manual_import_path = _as_relative(project_root, destination)
    save_page_manifest(project_root, page_manifest)
    return destination


def _normalize_external_result(
    *,
    input_path: Path,
    output_path: Path,
    expected_size: tuple[int, int],
) -> bool:
    _, Image, UnidentifiedImageError = _require_step3_runtime()
    try:
        with Image.open(input_path) as image:
            source_size = image.size
            if source_size != expected_size:
                raise Step3NormalizationError(
                    "External result size does not match the original page size: "
                    f"{source_size} != {expected_size}",
                    readable_image=True,
                    size_matches=False,
                    has_alpha_channel="A" in image.getbands() or "transparency" in image.info,
                )
            has_alpha_channel = "A" in image.getbands() or "transparency" in image.info
            converted = image.convert("RGBA" if has_alpha_channel else "RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            converted.save(output_path, format="PNG")
            return has_alpha_channel
    except Step3NormalizationError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise Step3NormalizationError(
            f"Unable to read external result image: {input_path}",
            readable_image=False,
            size_matches=False,
            has_alpha_channel=False,
        ) from exc


def summarize_step3_validation(
    *,
    page_id: str,
    source_kind: str | None,
    readable_image: bool,
    size_matches: bool,
    has_alpha_channel: bool,
    outside_support_rgb_nonzero_pixels: int,
    outside_support_alpha_nonzero_pixels: int,
) -> Step3ValidationReport:
    notes: list[str] = []
    opaque_output = readable_image and not has_alpha_channel
    if not readable_image:
        notes.append("raw external result is unreadable")
    if readable_image and not size_matches:
        notes.append("raw external result size does not match the original page size")
    if opaque_output:
        notes.append("raw external result is opaque and will require Phase 4 alpha restore")
    if outside_support_rgb_nonzero_pixels > 0:
        notes.append(
            "outside-support RGB pixels were detected in the raw external result: "
            f"{outside_support_rgb_nonzero_pixels}"
        )
    if outside_support_alpha_nonzero_pixels > 0:
        notes.append(
            "outside-support alpha pixels were detected in the raw external result: "
            f"{outside_support_alpha_nonzero_pixels}"
        )

    passed = readable_image and size_matches
    if passed:
        notes.append("step3 validation passed and the raw result is ready for Phase 4")

    return Step3ValidationReport(
        page_id=page_id,
        source_kind=source_kind,
        readable_image=readable_image,
        size_matches=size_matches,
        has_alpha_channel=has_alpha_channel,
        opaque_output=opaque_output,
        outside_support_rgb_nonzero_pixels=outside_support_rgb_nonzero_pixels,
        outside_support_alpha_nonzero_pixels=outside_support_alpha_nonzero_pixels,
        passed=passed,
        notes=notes,
    )


def validate_step3(project_root: Path, page_id: str) -> Step3ValidationReport:
    np, Image, UnidentifiedImageError = _require_step3_runtime()
    context = _load_page_context(project_root, page_id)
    page_manifest = context["page_manifest"]
    if not page_manifest.nano_banana_raw_path:
        raise FileNotFoundError(
            f"Page {page_id} has no Step 3 raw result yet. Run make-handoff/import or API execution first."
        )

    raw_path = project_root / page_manifest.nano_banana_raw_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Unable to read Step 3 raw result: {raw_path}")

    try:
        with Image.open(raw_path) as image:
            size_matches = image.size == context["expected_size"]
            has_alpha_channel = "A" in image.getbands() or "transparency" in image.info
            converted = image.convert("RGBA" if has_alpha_channel else "RGB")
            raw_array = np.array(converted)
    except (OSError, UnidentifiedImageError):
        report = _save_step3_failure_report(
            project_root,
            page_id,
            source_kind=page_manifest.nano_source_kind,
            readable_image=False,
            size_matches=False,
            has_alpha_channel=False,
            notes=["raw external result is unreadable"],
        )
        return report

    outside_support_rgb_nonzero_pixels = 0
    outside_support_alpha_nonzero_pixels = 0
    if size_matches:
        cv2, _ = _require_cv_runtime()
        union_mask = cv2.imread(str(context["union_mask_path"]), cv2.IMREAD_GRAYSCALE)
        if union_mask is None:
            raise FileNotFoundError(f"Unable to read union mask: {context['union_mask_path']}")
        _, support = build_soft_alpha_from_union_mask(union_mask)
        outside_support = support == 0
        if raw_array.ndim == 2:
            rgb = np.repeat(raw_array[:, :, None], 3, axis=2)
            alpha = None
        elif raw_array.shape[2] == 4:
            rgb = raw_array[:, :, :3]
            alpha = raw_array[:, :, 3]
        else:
            rgb = raw_array[:, :, :3]
            alpha = None
        outside_rgb = rgb[outside_support]
        if len(outside_rgb) > 0:
            outside_support_rgb_nonzero_pixels = int(
                np.count_nonzero(np.any(outside_rgb > 0, axis=1))
            )
        if alpha is not None:
            outside_support_alpha_nonzero_pixels = int(
                np.count_nonzero(alpha[outside_support] > 0)
            )

    report = summarize_step3_validation(
        page_id=page_id,
        source_kind=page_manifest.nano_source_kind,
        readable_image=True,
        size_matches=size_matches,
        has_alpha_channel=has_alpha_channel,
        outside_support_rgb_nonzero_pixels=outside_support_rgb_nonzero_pixels,
        outside_support_alpha_nonzero_pixels=outside_support_alpha_nonzero_pixels,
    )
    report_path = project_root / "artifacts" / "debug" / f"{page_id}_step3_validation.json"
    write_json(report_path, report.to_dict())
    page_manifest.step3_validation_report_path = _as_relative(project_root, report_path)
    page_manifest.status = "nano_pending" if report.passed else "check"
    save_page_manifest(project_root, page_manifest)
    return report


def import_external_result(project_root: Path, page_id: str, input_path: Path) -> dict[str, Any]:
    context = _load_page_context(project_root, page_id)
    if not input_path.exists():
        raise FileNotFoundError(f"External result does not exist: {input_path}")

    preserved_path = _preserve_manual_import(project_root, page_id, input_path)
    raw_output_path = project_root / "artifacts" / "nano" / f"{page_id}_raw.png"
    page_manifest = load_page_manifest(project_root, page_id)
    page_manifest.nano_source_kind = "manual_web"

    try:
        _normalize_external_result(
            input_path=preserved_path,
            output_path=raw_output_path,
            expected_size=context["expected_size"],
        )
        page_manifest.nano_banana_raw_path = _as_relative(project_root, raw_output_path)
        page_manifest.status = "nano_pending"
        save_page_manifest(project_root, page_manifest)
        report = validate_step3(project_root, page_id)
    except Step3NormalizationError as exc:
        page_manifest.nano_banana_raw_path = ""
        page_manifest.status = "check"
        save_page_manifest(project_root, page_manifest)
        report = _save_step3_failure_report(
            project_root,
            page_id,
            source_kind="manual_web",
            readable_image=exc.readable_image,
            size_matches=exc.size_matches,
            has_alpha_channel=exc.has_alpha_channel,
            notes=[str(exc)],
        )

    _project_timestamp(project_root)
    final_page_manifest = load_page_manifest(project_root, page_id)
    return {
        "page_id": page_id,
        "nano_manual_import_path": final_page_manifest.nano_manual_import_path,
        "nano_banana_raw_path": final_page_manifest.nano_banana_raw_path,
        "step3_validation_report_path": final_page_manifest.step3_validation_report_path,
        "status": final_page_manifest.status,
        "passed": report.passed,
    }


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def _image_path_to_inline_part(path: Path) -> dict[str, Any]:
    return {
        "inlineData": {
            "mimeType": _guess_mime_type(path),
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    }


class GeminiRestImageAdapter:
    name = "gemini"

    def generate(
        self,
        *,
        prompt: str,
        layer_path: Path,
        original_page_path: Path | None,
        output_dir: Path,
        api_key: str,
        model_name: str,
    ) -> Path:
        model_id = _resolve_gemini_model_id(model_name)
        parts: list[dict[str, Any]] = [{"text": prompt}, _image_path_to_inline_part(layer_path)]
        if original_page_path is not None:
            parts.append(_image_path_to_inline_part(original_page_path))
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }
        request = urllib.request.Request(
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw_response = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Gemini image request failed with HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini image request failed: {exc.reason}") from exc

        response_payload = json.loads(raw_response)
        image_part: dict[str, Any] | None = None
        for candidate in response_payload.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                inline_data = part.get("inlineData")
                if inline_data and inline_data.get("data"):
                    image_part = inline_data
                    break
            if image_part is not None:
                break
        if image_part is None:
            raise RuntimeError(
                "Gemini response did not contain an inline image part. "
                f"Response: {json.dumps(response_payload, ensure_ascii=False)[:800]}"
            )

        mime_type = image_part.get("mimeType", "image/png")
        extension = mimetypes.guess_extension(mime_type) or ".png"
        response_path = output_dir / f"provider_response{extension}"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_bytes(base64.b64decode(image_part["data"]))
        return response_path


def build_provider_adapter(provider_name: str) -> GeminiRestImageAdapter:
    normalized_provider = provider_name.strip().lower()
    if normalized_provider != "gemini":
        raise ValueError(f"Unsupported external edit provider: {provider_name}")
    return GeminiRestImageAdapter()


def run_external_edit(project_root: Path, page_id: str) -> dict[str, Any]:
    handoff_result = make_handoff(project_root, page_id)
    context = _load_page_context(project_root, page_id)
    project_manifest = context["project_manifest"]
    page_manifest = context["page_manifest"]
    page_manifest.nano_source_kind = "api_auto"
    page_manifest.nano_manual_import_path = ""
    save_page_manifest(project_root, page_manifest)

    if project_manifest.nano_integration_mode != "api_auto":
        report = _save_step3_failure_report(
            project_root,
            page_id,
            source_kind="api_auto",
            readable_image=False,
            size_matches=False,
            has_alpha_channel=False,
            notes=[
                "project.json is not configured for api_auto integration mode",
            ],
        )
        final_page_manifest = load_page_manifest(project_root, page_id)
        return {
            "page_id": page_id,
            "nano_request_dir": handoff_result["nano_request_dir"],
            "nano_banana_raw_path": final_page_manifest.nano_banana_raw_path,
            "step3_validation_report_path": final_page_manifest.step3_validation_report_path,
            "status": final_page_manifest.status,
            "passed": report.passed,
        }

    api_key = _resolve_provider_api_key(project_manifest.nano_provider)
    if not api_key:
        report = _save_step3_failure_report(
            project_root,
            page_id,
            source_kind="api_auto",
            readable_image=False,
            size_matches=False,
            has_alpha_channel=False,
            notes=[
                "API credential is missing. Set a GUI session key or GEMINI_API_KEY first.",
            ],
        )
        final_page_manifest = load_page_manifest(project_root, page_id)
        return {
            "page_id": page_id,
            "nano_request_dir": handoff_result["nano_request_dir"],
            "nano_banana_raw_path": final_page_manifest.nano_banana_raw_path,
            "step3_validation_report_path": final_page_manifest.step3_validation_report_path,
            "status": final_page_manifest.status,
            "passed": report.passed,
        }

    request_dir = project_root / page_manifest.nano_request_dir
    layer_path = request_dir / "balloons_only_rgba.png"
    original_page_path = request_dir / "original_page.png"
    adapter = build_provider_adapter(project_manifest.nano_provider)

    try:
        response_path = adapter.generate(
            prompt=build_step3_prompt(
                page_id=page_id,
                width=page_manifest.width,
                height=page_manifest.height,
                include_original_page=project_manifest.nano_include_original_page,
            ),
            layer_path=layer_path,
            original_page_path=original_page_path if original_page_path.exists() else None,
            output_dir=request_dir,
            api_key=api_key,
            model_name=project_manifest.nano_model,
        )
        raw_output_path = project_root / "artifacts" / "nano" / f"{page_id}_raw.png"
        _normalize_external_result(
            input_path=response_path,
            output_path=raw_output_path,
            expected_size=context["expected_size"],
        )
        page_manifest = load_page_manifest(project_root, page_id)
        page_manifest.nano_source_kind = "api_auto"
        page_manifest.nano_banana_raw_path = _as_relative(project_root, raw_output_path)
        page_manifest.status = "nano_pending"
        save_page_manifest(project_root, page_manifest)
        report = validate_step3(project_root, page_id)
    except (RuntimeError, Step3NormalizationError) as exc:
        page_manifest = load_page_manifest(project_root, page_id)
        page_manifest.nano_banana_raw_path = ""
        page_manifest.status = "check"
        save_page_manifest(project_root, page_manifest)
        readable_image = False
        size_matches = False
        has_alpha_channel = False
        if isinstance(exc, Step3NormalizationError):
            readable_image = exc.readable_image
            size_matches = exc.size_matches
            has_alpha_channel = exc.has_alpha_channel
        report = _save_step3_failure_report(
            project_root,
            page_id,
            source_kind="api_auto",
            readable_image=readable_image,
            size_matches=size_matches,
            has_alpha_channel=has_alpha_channel,
            notes=[str(exc)],
        )

    _project_timestamp(project_root)
    final_page_manifest = load_page_manifest(project_root, page_id)
    return {
        "page_id": page_id,
        "nano_request_dir": final_page_manifest.nano_request_dir,
        "nano_banana_raw_path": final_page_manifest.nano_banana_raw_path,
        "step3_validation_report_path": final_page_manifest.step3_validation_report_path,
        "status": final_page_manifest.status,
        "passed": report.passed,
    }


def format_step3_report(report: Step3ValidationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def _require_cv_runtime() -> tuple[object, object]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "Step 3 validation requires numpy and opencv-python. "
            "Install the dependencies from pyproject.toml before running validate-step3."
        ) from exc
    return cv2, np
