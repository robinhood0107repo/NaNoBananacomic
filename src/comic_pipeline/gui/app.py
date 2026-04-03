from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox, QStyleFactory

from comic_pipeline.gui.launcher import LauncherDialog
from comic_pipeline.gui.main_window import MainWindow


LOGGER = logging.getLogger("comic_pipeline.gui")


def _configure_logging() -> Path | None:
    raw_path = os.environ.get("COMIC_PIPELINE_GUI_LOGFILE", "").strip()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    log_path: Path | None = None
    if raw_path:
        log_path = Path(raw_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_path


def _install_excepthook(log_path: Path | None) -> None:
    def _hook(exc_type, exc_value, exc_traceback) -> None:
        message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        LOGGER.error("Unhandled GUI exception\n%s", message)
        if QApplication.instance() is not None:
            QMessageBox.critical(
                None,
                "NaNoBananacomic 오류",
                "예상치 못한 오류가 발생했습니다.\n\n"
                f"{exc_value}\n\n"
                f"로그 파일: {log_path if log_path else 'stderr'}",
            )
        else:
            print(message, file=sys.stderr)

    sys.excepthook = _hook


def create_application(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance()
    if app is not None:
        return app
    app = QApplication(argv or sys.argv)
    QApplication.setOrganizationName("NaNoBananacomic")
    QApplication.setOrganizationDomain("local")
    QApplication.setApplicationName("NaNoBananacomic")
    available_styles = {style.lower(): style for style in QStyleFactory.keys()}
    if "windowsvista" in available_styles:
        app.setStyle(available_styles["windowsvista"])
    elif "windows" in available_styles:
        app.setStyle(available_styles["windows"])
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)
    return app


def main(argv: list[str] | None = None) -> int:
    log_path = _configure_logging()
    _install_excepthook(log_path)
    LOGGER.info("Starting NaNoBananacomic GUI")
    try:
        app = create_application(argv)
        launcher = LauncherDialog()
        if launcher.exec() != launcher.DialogCode.Accepted or launcher.selected_project is None:
            LOGGER.info("Launcher closed without selecting a project")
            return 0
        LOGGER.info("Opening project: %s", launcher.selected_project)
        window = MainWindow(launcher.selected_project)
        window.show()
        return app.exec()
    except Exception:
        LOGGER.exception("Fatal GUI startup failure")
        if QApplication.instance() is not None:
            QMessageBox.critical(
                None,
                "NaNoBananacomic 시작 실패",
                "GUI를 시작하는 중 오류가 발생했습니다.\n\n"
                f"로그 파일: {log_path if log_path else 'stderr'}",
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
