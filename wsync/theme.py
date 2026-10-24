"""One dark Fusion theme for menus and dialogs (same as Progress Pies)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    surface, panel, ink = QColor(20, 21, 26), QColor(26, 28, 36), QColor(238, 240, 245)
    muted = QColor(124, 131, 148)
    for role, c in (
        (QPalette.Window, surface), (QPalette.WindowText, ink),
        (QPalette.Base, panel), (QPalette.AlternateBase, QColor(32, 34, 44)),
        (QPalette.Text, ink), (QPalette.Button, panel), (QPalette.ButtonText, ink),
        (QPalette.ToolTipBase, panel), (QPalette.ToolTipText, ink),
        (QPalette.Highlight, QColor(63, 193, 224)), (QPalette.HighlightedText, QColor(12, 13, 17)),
        (QPalette.PlaceholderText, muted), (QPalette.Link, QColor(63, 193, 224)),
    ):
        pal.setColor(role, c)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, muted)
    app.setPalette(pal)
    app.setStyleSheet(
        "QToolTip { background: #1a1c24; color: #eef0f5;"
        " border: 1px solid rgba(255,255,255,0.18); padding: 5px; }"
        "QMenu::item:disabled { color: #7c8394; }")
