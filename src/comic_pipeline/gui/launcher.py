from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QDialog,
)

from comic_pipeline.gui.state import ensure_project_ready


class LauncherDialog(QDialog):
    SETTINGS_KEY = "recentProjects"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._settings = QSettings()
        self.selected_project: Path | None = None
        self.setWindowTitle("NaNoBananacomic 프로젝트 열기")
        self.resize(760, 460)

        root = QVBoxLayout(self)
        title = QLabel("NaNoBananacomic")
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        description = QLabel(
            "Windows 데스크톱 워크벤치입니다. 프로젝트 폴더를 선택하면 "
            "필요 시 project.json을 생성하고 이미지를 스캔한 뒤 메인 작업창으로 들어갑니다."
        )
        description.setWordWrap(True)
        root.addWidget(description)

        actions = QHBoxLayout()
        self.browse_button = QPushButton("프로젝트 폴더 선택")
        self.browse_button.clicked.connect(self._browse_project)
        actions.addWidget(self.browse_button)
        self.open_selected_button = QPushButton("최근 프로젝트 열기")
        self.open_selected_button.clicked.connect(self._open_selected_recent)
        actions.addWidget(self.open_selected_button)
        root.addLayout(actions)

        root.addWidget(QLabel("최근 프로젝트"))
        self.recent_list = QListWidget(self)
        self.recent_list.itemDoubleClicked.connect(lambda _: self._open_selected_recent())
        root.addWidget(self.recent_list, stretch=1)

        structure = QLabel(
            "프로젝트 폴더 안에는 `project.json`, `pages/`, `imports/`, `artifacts/`, `logs/`, `result/`가 생성됩니다."
        )
        structure.setWordWrap(True)
        structure.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(structure)

        self._load_recent_projects()

    def _load_recent_projects(self) -> None:
        self.recent_list.clear()
        recent_paths = self._settings.value(self.SETTINGS_KEY, [], list)
        for raw_path in recent_paths:
            path = Path(str(raw_path))
            item = QListWidgetItem(str(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            if not path.exists():
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setText(f"{path} (없음)")
            self.recent_list.addItem(item)

    def _save_recent_project(self, project_path: Path) -> None:
        recent_paths = [str(project_path)]
        for raw_path in self._settings.value(self.SETTINGS_KEY, [], list):
            normalized = str(Path(str(raw_path)))
            if normalized != str(project_path):
                recent_paths.append(normalized)
        self._settings.setValue(self.SETTINGS_KEY, recent_paths[:10])
        self._load_recent_projects()

    def _browse_project(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "프로젝트 폴더 선택", str(Path.home()))
        if not selected:
            return
        self.open_project_path(Path(selected))

    def _open_selected_recent(self) -> None:
        item = self.recent_list.currentItem()
        if item is None:
            QMessageBox.information(self, "최근 프로젝트", "열 프로젝트를 먼저 선택하세요.")
            return
        path = Path(str(item.data(Qt.ItemDataRole.UserRole)))
        self.open_project_path(path)

    def open_project_path(self, project_path: Path) -> None:
        project_path = project_path.resolve()
        try:
            ensure_project_ready(project_path)
        except Exception as exc:
            QMessageBox.critical(self, "프로젝트 열기 실패", str(exc))
            return
        self.selected_project = project_path
        self._save_recent_project(project_path)
        self.accept()
