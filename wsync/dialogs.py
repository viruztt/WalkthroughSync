"""The 'New game' and 'Where are you?' dialogs."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout,
)

from . import config as C

HELP_TIME = "e.g. 1:30:00, 1h30m, 45m — or paste a YouTube link with a timestamp"


class NewGameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New game — Walkthrough Sync")
        self.setMinimumWidth(480)
        lay = QVBoxLayout(self)
        intro = QLabel("Paste a full walkthrough of the game you're playing. It is downloaded "
                       "once in low resolution and analysed on your GPU (a few minutes for "
                       "a long video).")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #969caa;")
        lay.addWidget(intro)

        form = QFormLayout()
        self.url = QLineEdit()
        self.url.setPlaceholderText("https://www.youtube.com/watch?v=…")
        self.name = QLineEdit()
        self.name.setPlaceholderText("Taken from the video title")
        self.start = QLineEdit("0:00")
        self.start.setToolTip(HELP_TIME)
        form.addRow("Walkthrough video", self.url)
        form.addRow("Game name", self.name)
        form.addRow("You're at about", self.start)
        hint = QLabel(HELP_TIME + ". A rough guess is fine.")
        hint.setStyleSheet("color: #7c8394; font-size: 9pt;")
        form.addRow("", hint)
        lay.addLayout(form)

        self.err = QLabel("")
        self.err.setStyleSheet("color: #e06c6c;")
        lay.addWidget(self.err)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Analyse walkthrough")
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.url.textChanged.connect(self._url_changed)

    def _url_changed(self, text: str):
        t = C.parse_time(text) if "t=" in text else None
        if t and self.start.text().strip() in ("", "0:00"):
            self.start.setText(C.fmt_time(t))

    def _ok(self):
        if not C.video_id(self.url.text()):
            self.err.setText("That doesn't look like a YouTube video link.")
            return
        if C.parse_time(self.start.text()) is None:
            self.err.setText("Couldn't read the time. " + HELP_TIME)
            return
        self.accept()

    def values(self) -> tuple[str, str, float]:
        url = self.url.text().strip()
        return url, self.name.text().strip(), float(C.parse_time(self.start.text()) or 0)


def ask_position(parent, current: float, duration: float) -> float | None:
    dlg = QDialog(parent)
    dlg.setWindowTitle("Where are you?")
    lay = QVBoxLayout(dlg)
    lab = QLabel("Your spot in the walkthrough video:")
    lay.addWidget(lab)
    edit = QLineEdit(C.fmt_time(current))
    edit.selectAll()
    lay.addWidget(edit)
    hint = QLabel(HELP_TIME + f".\nThe video is {C.fmt_time(duration)} long.")
    hint.setStyleSheet("color: #7c8394;")
    lay.addWidget(hint)
    err = QLabel("")
    err.setStyleSheet("color: #e06c6c;")
    lay.addWidget(err)
    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    result: list[float] = []

    def ok():
        t = C.parse_time(edit.text())
        if t is None:
            err.setText("Couldn't read that time.")
            return
        result.append(max(0.0, min(t, duration or t)))
        dlg.accept()
    bb.accepted.connect(ok)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)
    dlg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    if dlg.exec() and result:
        return result[0]
    return None
