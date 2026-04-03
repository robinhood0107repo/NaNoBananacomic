from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from comic_pipeline.gui.state import latest_step6_summary, serialize_payload
from comic_pipeline.step6 import validate_step6


class HistoryDialog(QDialog):
    def __init__(self, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self.project_root = project_root
        self._payload: dict | None = None
        self.setWindowTitle("Run History / Validation Inspector")
        self.resize(980, 640)

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.summary_label = QLabel("요약 없음")
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.summary_label, stretch=1)
        self.check_only = QCheckBox("CHECK 페이지만 보기")
        self.check_only.toggled.connect(self._populate_table)
        top.addWidget(self.check_only)
        self.refresh_button = QPushButton("요약 새로고침")
        self.refresh_button.clicked.connect(self.refresh_summary)
        top.addWidget(self.refresh_button)
        self.rebuild_button = QPushButton("전체 Step6 다시 실행")
        self.rebuild_button.clicked.connect(self._rebuild_summary)
        top.addWidget(self.rebuild_button)
        self.open_button = QPushButton("요약 파일 열기")
        self.open_button.clicked.connect(self._open_summary_file)
        top.addWidget(self.open_button)
        root.addLayout(top)

        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(
            ["페이지", "상태", "Step 3", "Step 4", "Step 5", "Registration"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self.table, stretch=2)

        self.json_view = QPlainTextEdit(self)
        self.json_view.setReadOnly(True)
        root.addWidget(self.json_view, stretch=1)

        self.refresh_summary()

    def _open_summary_file(self) -> None:
        path = latest_step6_summary(self.project_root)
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _rebuild_summary(self) -> None:
        validate_step6(self.project_root, validate_all=True)
        self.refresh_summary()

    def refresh_summary(self) -> None:
        summary_path = latest_step6_summary(self.project_root)
        if summary_path is None:
            self._payload = None
            self.summary_label.setText("아직 Step 6 요약이 없습니다.")
            self.json_view.setPlainText("아직 생성되지 않음")
            self.table.setRowCount(0)
            return
        self._payload = validate_step6(self.project_root, validate_all=True).to_dict()
        self.summary_label.setText(str(summary_path))
        self.json_view.setPlainText(serialize_payload(self._payload))
        self._populate_table()

    def _populate_table(self) -> None:
        payload = self._payload or {}
        summaries = list(payload.get("page_summaries", []))
        if self.check_only.isChecked():
            summaries = [item for item in summaries if item.get("status") == "check"]

        self.table.setRowCount(len(summaries))
        for row, item in enumerate(summaries):
            values = [
                item.get("page_id", ""),
                item.get("status", ""),
                "OK" if item.get("step3_passed") else "CHECK",
                "OK" if item.get("step4_passed") else "CHECK",
                "OK" if item.get("step5_passed") else "CHECK",
                item.get("registration_warning_level", "missing"),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
