from __future__ import annotations

import json
from pathlib import Path

from comic_pipeline.profile_router import route_page_profile
from comic_pipeline.types import PageManifest, ProjectManifest

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PROJECT_SUBDIRS = [
    "imports/nano",
    "artifacts/handoff",
    "artifacts/masks",
    "artifacts/layers",
    "artifacts/nano",
    "artifacts/composite",
    "artifacts/debug",
    "artifacts/previews",
    "logs",
    "pages",
    "result",
]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify_project_id(project_root: Path) -> str:
    slug = project_root.name.strip().lower().replace(" ", "_")
    return slug or "comic_project"


def ensure_project_dirs(project_root: Path) -> None:
    for rel in PROJECT_SUBDIRS:
        (project_root / rel).mkdir(parents=True, exist_ok=True)


def project_manifest_path(project_root: Path) -> Path:
    return project_root / "project.json"


def page_manifest_path(project_root: Path, page_id: str) -> Path:
    return project_root / "pages" / f"{page_id}.page.json"


def load_project_manifest(project_root: Path) -> ProjectManifest:
    data = read_json(project_manifest_path(project_root))
    return ProjectManifest(**data)


def save_project_manifest(project_root: Path, manifest: ProjectManifest) -> None:
    write_json(project_manifest_path(project_root), manifest.to_dict())


def load_page_manifest(project_root: Path, page_id: str) -> PageManifest:
    data = read_json(page_manifest_path(project_root, page_id))
    return PageManifest(**data)


def save_page_manifest(project_root: Path, manifest: PageManifest) -> None:
    write_json(page_manifest_path(project_root, manifest.page_id), manifest.to_dict())


def create_project(project_root: Path, profile_mode: str = "auto") -> ProjectManifest:
    project_root.mkdir(parents=True, exist_ok=True)
    ensure_project_dirs(project_root)
    manifest = ProjectManifest(
        project_id=slugify_project_id(project_root),
        project_root=str(project_root),
        profile_mode=profile_mode,
    )
    save_project_manifest(project_root, manifest)
    return manifest


def discover_project_images(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for child in sorted(project_root.iterdir()):
        if child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if child.suffix.lower() in IMAGE_EXTENSIONS:
            candidates.append(child)
    return candidates


def probe_image_size(image_path: Path) -> tuple[int, int]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return (0, 0)

    image = cv2.imread(str(image_path))
    if image is None:
        return (0, 0)
    height, width = image.shape[:2]
    return (width, height)


def build_page_manifest(project_root: Path, image_path: Path, resolved_profile: str) -> PageManifest:
    width, height = probe_image_size(image_path)
    page_id = image_path.stem
    return PageManifest(
        page_id=page_id,
        original_path=image_path.name,
        width=width,
        height=height,
        profile=resolved_profile,
        result_path=f"result/{page_id}.png",
    )


def scan_pages(project_root: Path) -> list[PageManifest]:
    project_manifest = load_project_manifest(project_root)
    pages: list[PageManifest] = []
    for image_path in discover_project_images(project_root):
        resolved_profile = (
            project_manifest.profile_mode
            if project_manifest.profile_mode != "auto"
            else route_page_profile(image_path)
        )
        fresh_manifest = build_page_manifest(project_root, image_path, resolved_profile)
        manifest_path = page_manifest_path(project_root, fresh_manifest.page_id)
        if manifest_path.exists():
            page_manifest = load_page_manifest(project_root, fresh_manifest.page_id)
            page_manifest.original_path = fresh_manifest.original_path
            page_manifest.width = fresh_manifest.width
            page_manifest.height = fresh_manifest.height
            page_manifest.profile = fresh_manifest.profile
            page_manifest.result_path = fresh_manifest.result_path
        else:
            page_manifest = fresh_manifest
        save_page_manifest(project_root, page_manifest)
        pages.append(page_manifest)
    project_manifest.page_count = len(pages)
    save_project_manifest(project_root, project_manifest)
    return pages
