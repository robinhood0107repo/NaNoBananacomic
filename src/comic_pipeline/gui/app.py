from __future__ import annotations

import sys

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QStyleFactory

from comic_pipeline.gui.launcher import LauncherDialog
from comic_pipeline.gui.main_window import MainWindow


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
    app = create_application(argv)
    launcher = LauncherDialog()
    if launcher.exec() != launcher.DialogCode.Accepted or launcher.selected_project is None:
        return 0
    window = MainWindow(launcher.selected_project)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
