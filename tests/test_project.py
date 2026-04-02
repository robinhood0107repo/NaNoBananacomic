from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_pipeline.project import create_project, read_json, scan_pages


class ProjectTests(unittest.TestCase):
    def test_create_project_writes_project_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            manifest = create_project(project_root, profile_mode="auto")
            self.assertEqual(manifest.project_root, str(project_root))
            saved = read_json(project_root / "project.json")
            self.assertEqual(saved["profile_mode"], "auto")
            self.assertEqual(saved["result_dir"], "result")

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


if __name__ == "__main__":
    unittest.main()

