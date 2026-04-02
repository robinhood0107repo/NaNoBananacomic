from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_pipeline.project import create_project, load_page_manifest, read_json, scan_pages, save_page_manifest


class ProjectTests(unittest.TestCase):
    def test_create_project_writes_project_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            manifest = create_project(project_root, profile_mode="auto")
            self.assertEqual(manifest.project_root, str(project_root))
            saved = read_json(project_root / "project.json")
            self.assertEqual(saved["profile_mode"], "auto")
            self.assertEqual(saved["result_dir"], "result")
            self.assertEqual(saved["nano_integration_mode"], "manual_web")
            self.assertEqual(saved["nano_provider"], "gemini")
            self.assertEqual(saved["nano_model"], "nano-banana-2")
            self.assertEqual(saved["imports_dir"], "imports")
            self.assertTrue((project_root / "imports" / "nano").exists())
            self.assertTrue((project_root / "artifacts" / "handoff").exists())

    def test_scan_pages_registers_top_level_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            create_project(project_root)
            (project_root / "0001.png").write_bytes(b"")
            (project_root / "0002.jpg").write_bytes(b"")
            (project_root / "notes.txt").write_text("ignore", encoding="utf-8")

            pages = scan_pages(project_root)

            self.assertEqual([page.page_id for page in pages], ["0001", "0002"])
            self.assertTrue((project_root / "pages" / "0001.page.json").exists())
            self.assertTrue((project_root / "pages" / "0002.page.json").exists())

    def test_scan_pages_preserves_existing_page_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            create_project(project_root)
            (project_root / "0001.png").write_bytes(b"")
            scan_pages(project_root)

            page = load_page_manifest(project_root, "0001")
            page.status = "nano_pending"
            page.balloon_union_mask_path = "artifacts/masks/0001_union.png"
            page.balloons_only_rgba_path = "artifacts/layers/0001_balloons_only.png"
            page.step2_validation_report_path = "artifacts/debug/0001_step2_validation.json"
            page.step1_detector_name = "manga109_seg_v1"
            save_page_manifest(project_root, page)

            pages = scan_pages(project_root)
            saved = load_page_manifest(project_root, "0001")

            self.assertEqual(len(pages), 1)
            self.assertEqual(saved.status, "nano_pending")
            self.assertEqual(saved.balloon_union_mask_path, "artifacts/masks/0001_union.png")
            self.assertEqual(saved.balloons_only_rgba_path, "artifacts/layers/0001_balloons_only.png")
            self.assertEqual(
                saved.step2_validation_report_path,
                "artifacts/debug/0001_step2_validation.json",
            )
            self.assertEqual(saved.step1_detector_name, "manga109_seg_v1")


if __name__ == "__main__":
    unittest.main()
