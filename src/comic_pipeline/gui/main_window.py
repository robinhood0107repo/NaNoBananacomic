from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt, QThreadPool, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QDialog,
    QDockWidget,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from comic_pipeline.gui.history_dialog import HistoryDialog
from comic_pipeline.gui.settings_dialog import ProjectSettingsDialog
from comic_pipeline.gui.state import (
    PREVIEW_FIELDS,
    STATUS_LABELS,
    latest_step6_summary,
    latest_step6_payload,
    load_page_detail,
    load_project_state,
    page_preview_paths,
)
from comic_pipeline.gui.widgets import PreviewWidget, StatusBadge
from comic_pipeline.gui.workers import FunctionWorker, WorkerResult
from comic_pipeline.make_balloons_only import make_balloons_only
from comic_pipeline.project import load_project_manifest
from comic_pipeline.step1 import detect_page
from comic_pipeline.step3 import (
    import_external_result,
    make_handoff,
    run_external_edit,
    validate_step3,
)
from comic_pipeline.step4 import _discover_manual_import_candidate, restore_alpha, validate_step4
from comic_pipeline.step5 import compose_final, validate_step5
from comic_pipeline.step6 import validate_step6
from comic_pipeline.types import PageManifest, ProjectManifest


def _payload_to_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if is_dataclass(payload):
        return asdict(payload)
    if hasattr(payload, "to_dict"):
        return payload.to_dict()
    if isinstance(payload, dict):
        return payload
    return {"value": payload}


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self.project_root = project_root.resolve()
        self.project_manifest: ProjectManifest = load_project_manifest(self.project_root)
        self.page_manifests: list[PageManifest] = []
        self.current_page_id: str | None = None
        self.current_page_detail = None
        self.thread_pool = QThreadPool.globalInstance()
        self._history_dialog: HistoryDialog | None = None

        self.setWindowTitle(f"NaNoBananacomic - {self.project_root.name}")
        self.resize(1440, 900)
        self.setMinimumSize(1280, 800)

        self._build_ui()
        self._restore_window_settings()
        self.refresh_project_state()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_window_settings()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_status_bar()
        self._build_central()
        self._build_page_dock()
        self._build_inspector_dock()
        self._build_log_dock()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("작업", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        def add_action(text: str, slot) -> QAction:
            action = QAction(text, self)
            action.triggered.connect(slot)
            toolbar.addAction(action)
            return action

        add_action("프로젝트 열기", self._open_project_folder)
        add_action("설정", self._open_settings)
        add_action("결과 폴더", self._open_result_folder)
        add_action("검증 인스펙터", self._open_history_dialog)
        toolbar.addSeparator()
        add_action("Step1 검출", self._run_detect)
        add_action("Step2 레이어", self._run_make_layer)
        add_action("Step3 Handoff", self._run_make_handoff)
        add_action("Step3 가져오기", self._run_import_result)
        add_action("Step3 API", self._run_external_api)
        add_action("Step3 검증", self._run_validate_step3)
        add_action("Step4 복구", self._run_restore_alpha)
        add_action("Step4 검증", self._run_validate_step4)
        add_action("Step5 합성", self._run_compose_final)
        add_action("Step5 검증", self._run_validate_step5)
        add_action("Step6 현재", self._run_validate_current_page)
        add_action("Step6 전체", self._run_validate_all_pages)

    def _build_status_bar(self) -> None:
        self.status_label = QLabel("준비")
        self.project_label = QLabel(str(self.project_root))
        self.project_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.page_status_badge = StatusBadge()
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.page_status_badge)
        self.statusBar().addPermanentWidget(self.project_label)

    def _build_central(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.project_summary_label = QLabel()
        self.project_summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.project_summary_label)

        self.preview_tabs = QTabWidget(self)
        self.preview_widgets: dict[str, PreviewWidget] = {}
        for label in PREVIEW_FIELDS:
            preview = PreviewWidget(label, self)
            self.preview_widgets[label] = preview
            self.preview_tabs.addTab(preview, label)
        layout.addWidget(self.preview_tabs, stretch=1)
        self.setCentralWidget(central)

    def _build_page_dock(self) -> None:
        dock = QDockWidget("페이지", self)
        dock.setObjectName("pageDock")
        container = QWidget(dock)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("상태 필터"))
        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItem("전체", "")
        for status_key, label in STATUS_LABELS.items():
            self.status_filter_combo.addItem(label, status_key)
        self.status_filter_combo.currentIndexChanged.connect(self._populate_page_list)
        filter_row.addWidget(self.status_filter_combo, stretch=1)
        layout.addLayout(filter_row)

        self.page_list = QListWidget()
        self.page_list.currentItemChanged.connect(self._on_page_selection_changed)
        layout.addWidget(self.page_list, stretch=1)

        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_inspector_dock(self) -> None:
        dock = QDockWidget("Inspector", self)
        dock.setObjectName("inspectorDock")
        container = QWidget(dock)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self.page_title_label = QLabel("페이지를 선택하세요")
        title_font = self.page_title_label.font()
        title_font.setBold(True)
        self.page_title_label.setFont(title_font)
        layout.addWidget(self.page_title_label)

        self.inspector_tree = QTreeWidget()
        self.inspector_tree.setColumnCount(2)
        self.inspector_tree.setHeaderLabels(["항목", "값"])
        self.inspector_tree.header().setStretchLastSection(True)
        layout.addWidget(self.inspector_tree, stretch=1)

        step3_group = QGroupBox("Step 3")
        step3_layout = QVBoxLayout(step3_group)
        self.step3_mode_label = QLabel()
        self.step3_paths_label = QLabel()
        self.step3_paths_label.setWordWrap(True)
        self.step3_paths_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        step3_layout.addWidget(self.step3_mode_label)
        step3_layout.addWidget(self.step3_paths_label)
        step3_buttons = QHBoxLayout()
        self.make_handoff_button = QPushButton("웹용 prompt 생성")
        self.make_handoff_button.clicked.connect(self._run_make_handoff)
        step3_buttons.addWidget(self.make_handoff_button)
        self.open_handoff_button = QPushButton("handoff 폴더 열기")
        self.open_handoff_button.clicked.connect(self._open_handoff_folder)
        step3_buttons.addWidget(self.open_handoff_button)
        self.import_result_button = QPushButton("결과 파일 가져오기")
        self.import_result_button.clicked.connect(self._run_import_result)
        step3_buttons.addWidget(self.import_result_button)
        step3_layout.addLayout(step3_buttons)
        step3_buttons_2 = QHBoxLayout()
        self.refresh_imports_button = QPushButton("imports/nano 새로고침")
        self.refresh_imports_button.clicked.connect(self._refresh_import_candidate)
        step3_buttons_2.addWidget(self.refresh_imports_button)
        self.api_run_button = QPushButton("API 실행")
        self.api_run_button.clicked.connect(self._run_external_api)
        step3_buttons_2.addWidget(self.api_run_button)
        self.validate_step3_button = QPushButton("Step 3 검증")
        self.validate_step3_button.clicked.connect(self._run_validate_step3)
        step3_buttons_2.addWidget(self.validate_step3_button)
        step3_layout.addLayout(step3_buttons_2)
        layout.addWidget(step3_group)

        step4_group = QGroupBox("Step 4~6")
        step4_layout = QVBoxLayout(step4_group)
        buttons1 = QHBoxLayout()
        self.restore_button = QPushButton("투명도 복구")
        self.restore_button.clicked.connect(self._run_restore_alpha)
        buttons1.addWidget(self.restore_button)
        self.compose_button = QPushButton("최종 합성")
        self.compose_button.clicked.connect(self._run_compose_final)
        buttons1.addWidget(self.compose_button)
        self.validate_step4_button = QPushButton("Step 4 검증")
        self.validate_step4_button.clicked.connect(self._run_validate_step4)
        buttons1.addWidget(self.validate_step4_button)
        step4_layout.addLayout(buttons1)
        buttons2 = QHBoxLayout()
        self.validate_step5_button = QPushButton("Step 5 검증")
        self.validate_step5_button.clicked.connect(self._run_validate_step5)
        buttons2.addWidget(self.validate_step5_button)
        self.validate_page_button = QPushButton("현재 페이지 검증")
        self.validate_page_button.clicked.connect(self._run_validate_current_page)
        buttons2.addWidget(self.validate_page_button)
        self.validate_all_button = QPushButton("전체 프로젝트 검증")
        self.validate_all_button.clicked.connect(self._run_validate_all_pages)
        buttons2.addWidget(self.validate_all_button)
        step4_layout.addLayout(buttons2)
        layout.addWidget(step4_group)

        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_log_dock(self) -> None:
        dock = QDockWidget("로그", self)
        dock.setObjectName("logDock")
        self.log_view = QPlainTextEdit(self)
        self.log_view.setReadOnly(True)
        dock.setWidget(self.log_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _restore_window_settings(self) -> None:
        settings = QSettings()
        geometry = settings.value("mainWindow/geometry")
        state = settings.value("mainWindow/state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)

    def _save_window_settings(self) -> None:
        settings = QSettings()
        settings.setValue("mainWindow/geometry", self.saveGeometry())
        settings.setValue("mainWindow/state", self.saveState())

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _open_project_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "프로젝트 폴더 선택", str(self.project_root))
        if not selected:
            return
        self.project_root = Path(selected).resolve()
        from comic_pipeline.gui.state import ensure_project_ready

        ensure_project_ready(self.project_root)
        self.project_manifest = load_project_manifest(self.project_root)
        self.project_label.setText(str(self.project_root))
        self.setWindowTitle(f"NaNoBananacomic - {self.project_root.name}")
        self.refresh_project_state()

    def _open_result_folder(self) -> None:
        path = self.project_root / self.project_manifest.result_dir
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_handoff_folder(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None or not page_manifest.nano_request_dir:
            QMessageBox.information(self, "handoff", "현재 페이지에 handoff 폴더가 아직 없습니다.")
            return
        path = self.project_root / page_manifest.nano_request_dir
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_history_dialog(self) -> None:
        if self._history_dialog is None:
            self._history_dialog = HistoryDialog(self.project_root, self)
        else:
            self._history_dialog.project_root = self.project_root
            self._history_dialog.refresh_summary()
        self._history_dialog.show()
        self._history_dialog.raise_()
        self._history_dialog.activateWindow()

    def _open_settings(self) -> None:
        dialog = ProjectSettingsDialog(self.project_root, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.project_manifest = dialog.manifest
        self.refresh_project_state()
        self._append_log("프로젝트 설정을 저장했습니다.")

    def refresh_project_state(self) -> None:
        self.project_manifest, self.page_manifests = load_project_state(self.project_root)
        self.project_summary_label.setText(
            "프로젝트: "
            f"{self.project_root} | 페이지 {self.project_manifest.page_count}장 | "
            f"mode={self.project_manifest.nano_integration_mode}, "
            f"provider={self.project_manifest.nano_provider}, "
            f"model={self.project_manifest.nano_model}, "
            f"target={self.project_manifest.nano_target_language}"
        )
        self._populate_page_list()
        if self.current_page_id and any(item.page_id == self.current_page_id for item in self.page_manifests):
            self._select_page_by_id(self.current_page_id)
        elif self.page_list.count() > 0:
            self.page_list.setCurrentRow(0)
        else:
            self._clear_page_detail()

    def _populate_page_list(self) -> None:
        selected_page_id = self.current_page_id
        filter_status = str(self.status_filter_combo.currentData())
        self.page_list.clear()
        for page_manifest in self.page_manifests:
            if filter_status and page_manifest.status != filter_status:
                continue
            text = f"{page_manifest.page_id}  [{STATUS_LABELS.get(page_manifest.status, page_manifest.status)}]"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, page_manifest.page_id)
            self.page_list.addItem(item)
            if page_manifest.page_id == selected_page_id:
                self.page_list.setCurrentItem(item)

    def _select_page_by_id(self, page_id: str) -> None:
        for row in range(self.page_list.count()):
            item = self.page_list.item(row)
            if str(item.data(Qt.ItemDataRole.UserRole)) == page_id:
                self.page_list.setCurrentItem(item)
                return

    def _on_page_selection_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            self._clear_page_detail()
            return
        page_id = str(current.data(Qt.ItemDataRole.UserRole))
        self.current_page_id = page_id
        self.refresh_page_detail()

    def _clear_page_detail(self) -> None:
        self.current_page_id = None
        self.current_page_detail = None
        self.page_title_label.setText("페이지를 선택하세요")
        self.page_status_badge.set_status("pending")
        self.inspector_tree.clear()
        self.step3_mode_label.setText("Step 3 정보 없음")
        self.step3_paths_label.setText("경로 정보 없음")
        for preview in self.preview_widgets.values():
            preview.set_image_path(None)

    def refresh_page_detail(self) -> None:
        if self.current_page_id is None:
            self._clear_page_detail()
            return
        detail = load_page_detail(self.project_root, self.current_page_id)
        self.current_page_detail = detail
        manifest = detail.page_manifest
        self.page_title_label.setText(f"{manifest.page_id} ({manifest.width}x{manifest.height})")
        self.page_status_badge.set_status(manifest.status)
        self.status_label.setText(f"현재 페이지: {manifest.page_id}")
        self._refresh_previews(manifest)
        self._refresh_inspector(detail)
        self._refresh_step3_panel(detail)

    def _refresh_previews(self, page_manifest: PageManifest) -> None:
        previews = page_preview_paths(self.project_root, page_manifest)
        for label, preview_widget in self.preview_widgets.items():
            preview_widget.set_image_path(previews.get(label))

    def _set_tree_section(self, parent: QTreeWidgetItem, title: str, rows: list[tuple[str, Any]]) -> None:
        section = QTreeWidgetItem([title, ""])
        parent.addChild(section)
        for key, value in rows:
            section.addChild(QTreeWidgetItem([key, "" if value is None else str(value)]))

    def _refresh_inspector(self, detail) -> None:
        manifest = detail.page_manifest
        self.inspector_tree.clear()
        root = self.inspector_tree.invisibleRootItem()
        self._set_tree_section(
            root,
            "Page",
            [
                ("page_id", manifest.page_id),
                ("size", f"{manifest.width}x{manifest.height}"),
                ("profile", manifest.profile),
                ("status", STATUS_LABELS.get(manifest.status, manifest.status)),
                ("balloon_count", manifest.balloon_count),
            ],
        )
        self._set_tree_section(
            root,
            "Step 3",
            [
                ("integration_mode", self.project_manifest.nano_integration_mode),
                ("provider", self.project_manifest.nano_provider),
                ("model", self.project_manifest.nano_model),
                ("target_language", self.project_manifest.nano_target_language),
                ("request_dir", manifest.nano_request_dir),
                ("raw_path", manifest.nano_banana_raw_path),
                ("manual_import", manifest.nano_manual_import_path),
                ("passed", detail.step3_report.get("passed") if detail.step3_report else None),
            ],
        )
        registration_payload = detail.registration_report or {}
        self._set_tree_section(
            root,
            "Step 4",
            [
                ("source_kind", manifest.nano_source_kind),
                ("rgba_path", manifest.nano_banana_rgba_path),
                ("passed", detail.step4_report.get("passed") if detail.step4_report else None),
                ("registration_score", registration_payload.get("final_score")),
                ("registration_level", registration_payload.get("warning_level")),
                ("alignment_applied", detail.step4_report.get("alignment_applied") if detail.step4_report else None),
            ],
        )
        self._set_tree_section(
            root,
            "Step 5",
            [
                ("final_composite", manifest.final_composite_path),
                ("result_path", manifest.result_path),
                ("diff_preview", manifest.diff_preview_path),
                ("passed", detail.step5_report.get("passed") if detail.step5_report else None),
                ("outside_diff_ratio", detail.step5_report.get("outside_mask_diff_ratio") if detail.step5_report else None),
            ],
        )
        step6_payload = detail.step6_report or latest_step6_payload(self.project_root) or {}
        summary = None
        for item in step6_payload.get("page_summaries", []):
            if item.get("page_id") == manifest.page_id:
                summary = item
                break
        self._set_tree_section(
            root,
            "Step 6",
            [
                ("step6_report", manifest.step6_validation_report_path),
                ("summary_status", summary.get("status") if summary else None),
                ("missing_artifacts", ", ".join(summary.get("missing_artifacts", [])) if summary else None),
                ("severe_registration", summary.get("registration_warning_level") if summary else None),
            ],
        )
        self.inspector_tree.expandAll()

    def _refresh_step3_panel(self, detail) -> None:
        manifest = detail.page_manifest
        manual_candidate = _discover_manual_import_candidate(self.project_root, manifest.page_id)
        self.step3_mode_label.setText(
            f"mode={self.project_manifest.nano_integration_mode} | provider={self.project_manifest.nano_provider} | "
            f"model={self.project_manifest.nano_model} | target={self.project_manifest.nano_target_language}"
        )
        lines = [
            f"handoff: {manifest.nano_request_dir or '-'}",
            f"raw: {manifest.nano_banana_raw_path or '-'}",
            f"manual import: {manifest.nano_manual_import_path or '-'}",
            f"imports/nano 후보: {manual_candidate or '-'}",
            f"step3 validation: {manifest.step3_validation_report_path or '-'}",
        ]
        self.step3_paths_label.setText("\n".join(lines))
        is_api_mode = self.project_manifest.nano_integration_mode == "api_auto"
        self.api_run_button.setEnabled(is_api_mode)

    def _require_selected_page_manifest(self) -> PageManifest | None:
        if self.current_page_id is None:
            QMessageBox.information(self, "페이지 선택", "먼저 페이지를 선택하세요.")
            return None
        return load_page_detail(self.project_root, self.current_page_id).page_manifest

    def _run_task(self, label: str, fn, *args, success_message: str | None = None, **kwargs) -> None:
        worker = FunctionWorker(label, fn, *args, **kwargs)
        worker.signals.started.connect(lambda text: self._append_log(f"[START] {text}"))
        worker.signals.failed.connect(self._on_worker_failed)

        def _on_success(result: WorkerResult) -> None:
            payload = _payload_to_dict(result.payload)
            self._append_log(f"[OK] {result.label}\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
            if success_message:
                self.statusBar().showMessage(success_message, 6000)
            self.refresh_project_state()
            if self.current_page_id:
                self.refresh_page_detail()
            if self._history_dialog is not None and self._history_dialog.isVisible():
                self._history_dialog.refresh_summary()

        worker.signals.succeeded.connect(_on_success)
        self.thread_pool.start(worker)

    def _on_worker_failed(self, label: str, traceback_text: str) -> None:
        self._append_log(f"[ERROR] {label}\n{traceback_text}")
        self.statusBar().showMessage(f"{label} 실패", 8000)
        QMessageBox.critical(self, f"{label} 실패", traceback_text.splitlines()[-1] if traceback_text else label)

    def _run_detect(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        self._run_task("Step 1 detect", detect_page, self.project_root, page_manifest.page_id, success_message="Step 1 완료")

    def _run_make_layer(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        self._run_task("Step 2 make-layer", make_balloons_only, self.project_root, page_manifest.page_id, success_message="Step 2 완료")

    def _run_make_handoff(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        self._run_task("Step 3 make-handoff", make_handoff, self.project_root, page_manifest.page_id, success_message="Step 3 handoff 생성 완료")

    def _run_import_result(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "외부 편집 결과 가져오기",
            str(self.project_root / self.project_manifest.imports_dir / "nano"),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if not selected:
            return
        self._run_task(
            "Step 3 import-external-result",
            import_external_result,
            self.project_root,
            page_manifest.page_id,
            Path(selected).resolve(),
            success_message="외부 결과를 가져왔습니다.",
        )

    def _run_external_api(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        if self.project_manifest.nano_integration_mode != "api_auto":
            QMessageBox.information(self, "API 실행", "프로젝트 설정에서 연동 모드를 api_auto로 바꾸세요.")
            return
        self._run_task("Step 3 run-external-edit", run_external_edit, self.project_root, page_manifest.page_id, success_message="API 실행 완료")

    def _run_validate_step3(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        self._run_task("Step 3 validate-step3", validate_step3, self.project_root, page_manifest.page_id, success_message="Step 3 검증 완료")

    def _run_restore_alpha(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        self._run_task("Step 4 restore-alpha", restore_alpha, self.project_root, page_manifest.page_id, success_message="Step 4 완료")

    def _run_validate_step4(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        self._run_task("Step 4 validate-step4", validate_step4, self.project_root, page_manifest.page_id, success_message="Step 4 검증 완료")

    def _run_compose_final(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        self._run_task("Step 5 compose-final", compose_final, self.project_root, page_manifest.page_id, success_message="Step 5 완료")

    def _run_validate_step5(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        self._run_task("Step 5 validate-step5", validate_step5, self.project_root, page_manifest.page_id, success_message="Step 5 검증 완료")

    def _run_validate_current_page(self) -> None:
        page_manifest = self._require_selected_page_manifest()
        if page_manifest is None:
            return
        self._run_task("Step 6 validate-step6(page)", validate_step6, self.project_root, page_id=page_manifest.page_id, success_message="현재 페이지 Step 6 검증 완료")

    def _run_validate_all_pages(self) -> None:
        self._run_task("Step 6 validate-step6(all)", validate_step6, self.project_root, validate_all=True, success_message="전체 프로젝트 Step 6 검증 완료")

    def _refresh_import_candidate(self) -> None:
        self.refresh_project_state()
        if self.current_page_id:
            self.refresh_page_detail()
        self.statusBar().showMessage("imports/nano 후보를 다시 읽었습니다.", 4000)
