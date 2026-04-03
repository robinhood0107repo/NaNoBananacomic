from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import comic_pipeline  # noqa: F401
import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
from PySide6.QtWidgets import QApplication

from comic_pipeline.gui.app import create_application
from comic_pipeline.gui.launcher import LauncherDialog
from comic_pipeline.gui.main_window import MainWindow
from comic_pipeline.gui.settings_dialog import ProjectSettingsDialog
from comic_pipeline.project import load_project_manifest


def _app() -> QApplication:
    return create_application([])


class GuiSmokeTests(unittest.TestCase):
    def _build_project_dir(self) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        project_root = Path(tmpdir.name)
        image = np.full((640, 480, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (440, 600), (10, 10, 10), 4)
        cv2.ellipse(image, (240, 180), (110, 72), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (240, 180), (110, 72), 0, 0, 360, (15, 15, 15), 4)
        cv2.imwrite(str(project_root / "0001.png"), image)
        return project_root

    def test_launcher_opens_project_and_initializes_manifest(self) -> None:
        _app()
        project_root = self._build_project_dir()
        dialog = LauncherDialog()
        dialog.open_project_path(project_root)

        self.assertEqual(dialog.selected_project, project_root.resolve())
        self.assertTrue((project_root / "project.json").exists())
        self.assertTrue((project_root / "pages" / "0001.page.json").exists())

    def test_main_window_loads_pages_and_preview_tabs(self) -> None:
        _app()
        project_root = self._build_project_dir()
        launcher = LauncherDialog()
        launcher.open_project_path(project_root)

        window = MainWindow(project_root)
        self.addCleanup(window.close)

        self.assertEqual(window.page_list.count(), 1)
        self.assertEqual(window.preview_tabs.count(), 8)
        self.assertEqual(window.current_page_id, "0001")
        self.assertIn("mode=", window.step3_mode_label.text())

    def test_settings_dialog_saves_non_secret_project_fields(self) -> None:
        _app()
        project_root = self._build_project_dir()
        launcher = LauncherDialog()
        launcher.open_project_path(project_root)

        dialog = ProjectSettingsDialog(project_root)
        dialog.integration_mode_combo.setCurrentIndex(dialog.integration_mode_combo.findData("api_auto"))
        dialog.model_edit.setText("gemini-3.1-flash-image-preview")
        dialog.target_language_edit.setText("영어")
        dialog.api_key_edit.setText("secret-session-key")
        dialog.include_original_checkbox.setChecked(False)
        dialog._save()

        manifest = load_project_manifest(project_root)
        self.assertEqual(manifest.nano_integration_mode, "api_auto")
        self.assertEqual(manifest.nano_model, "gemini-3.1-flash-image-preview")
        self.assertEqual(manifest.nano_target_language, "영어")
        self.assertFalse(manifest.nano_include_original_page)
        self.assertNotIn("secret-session-key", (project_root / "project.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
