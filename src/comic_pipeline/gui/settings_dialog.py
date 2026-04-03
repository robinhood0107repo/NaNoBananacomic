from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from comic_pipeline.gui.state import save_project_settings
from comic_pipeline.project import load_project_manifest
from comic_pipeline.step3 import clear_session_api_key, set_session_api_key
from comic_pipeline.types import ProjectManifest


class ProjectSettingsDialog(QDialog):
    def __init__(self, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self.project_root = project_root
        self.setWindowTitle("프로젝트 설정")
        self.setModal(True)
        self.resize(520, 320)
        self._manifest = load_project_manifest(project_root)

        root_layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        root_layout.addLayout(form)

        self.project_path_label = QLabel(str(project_root))
        self.project_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("프로젝트 폴더", self.project_path_label)

        self.integration_mode_combo = QComboBox()
        self.integration_mode_combo.addItem("무료 웹 수동", "manual_web")
        self.integration_mode_combo.addItem("유료 API 자동", "api_auto")
        form.addRow("연동 모드", self.integration_mode_combo)

        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Gemini", "gemini")
        form.addRow("Provider", self.provider_combo)

        self.model_edit = QLineEdit()
        form.addRow("Model", self.model_edit)

        self.target_language_edit = QLineEdit()
        form.addRow("도착 언어", self.target_language_edit)

        self.include_original_checkbox = QCheckBox("원본 페이지를 context로 함께 보냄")
        form.addRow("원본 포함", self.include_original_checkbox)

        api_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("세션 동안만 사용됩니다")
        api_row.addWidget(self.api_key_edit, stretch=1)
        self.clear_key_button = QPushButton("세션 키 삭제")
        self.clear_key_button.clicked.connect(self._clear_session_key)
        api_row.addWidget(self.clear_key_button)
        form.addRow("API Credential", api_row)

        hint = QLabel(
            "API 키는 project.json에 저장하지 않습니다. "
            "이번 실행 세션에만 유지되고, 없으면 GEMINI_API_KEY 환경변수를 사용합니다."
        )
        hint.setWordWrap(True)
        root_layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root_layout.addWidget(buttons)

        self._load_manifest()

    @property
    def manifest(self) -> ProjectManifest:
        return self._manifest

    def _load_manifest(self) -> None:
        self.integration_mode_combo.setCurrentIndex(
            max(0, self.integration_mode_combo.findData(self._manifest.nano_integration_mode))
        )
        self.provider_combo.setCurrentIndex(
            max(0, self.provider_combo.findData(self._manifest.nano_provider))
        )
        self.model_edit.setText(self._manifest.nano_model)
        self.target_language_edit.setText(self._manifest.nano_target_language)
        self.include_original_checkbox.setChecked(self._manifest.nano_include_original_page)

    def _clear_session_key(self) -> None:
        provider = self.provider_combo.currentData()
        clear_session_api_key(str(provider))
        self.api_key_edit.clear()
        QMessageBox.information(self, "세션 키 삭제", "현재 GUI 세션의 API 키를 제거했습니다.")

    def _save(self) -> None:
        integration_mode = str(self.integration_mode_combo.currentData())
        provider = str(self.provider_combo.currentData())
        model = self.model_edit.text().strip() or "nano-banana-2"
        target_language = self.target_language_edit.text().strip() or "한국어"
        include_original_page = self.include_original_checkbox.isChecked()

        self._manifest = save_project_settings(
            self.project_root,
            integration_mode=integration_mode,
            provider=provider,
            model=model,
            target_language=target_language,
            include_original_page=include_original_page,
        )

        api_key = self.api_key_edit.text().strip()
        if api_key:
            set_session_api_key(provider, api_key)

        self.accept()
