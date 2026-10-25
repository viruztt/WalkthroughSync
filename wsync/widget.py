"""The floating card: progress donut, game name, time, status, matched frame."""
from __future__ import annotations

import os
import subprocess
import sys
import time

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction, QActionGroup, QColor, QDesktopServices, QFont, QFontMetrics,
    QGuiApplication, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QInputDialog, QMenu, QMessageBox, QSystemTrayIcon, QWidget,
)

from . import capture
from . import config as C
from . import games as G
from .dialogs import NewGameDialog, ask_position
from .workers import IndexWorker, LiveWorker

BG = QColor(14, 15, 20, 225)
TEXT = QColor(238, 240, 245)
MUTED = QColor(150, 156, 170)
TRACK = QColor(255, 255, 255, 30)
ACCENT = QColor("#3FC1E0")
AMBER = QColor("#E0A03F")
GREEN = QColor("#5BC97A")
RED = QColor("#E06C6C")
GREY = QColor(124, 131, 148)

SNAP_PX = 40
PAD = 10
DONUT = 58
GAP = 11


class SyncWindow(QWidget):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.game: G.Game | None = G.get(cfg.get("current_game", ""))
        self.thumbs: G.Thumbs | None = None
        self.live: dict = {}
        self.index_info: tuple[str, float, str] | None = None
        self.index_worker: IndexWorker | None = None
        self.thumb_pix: QPixmap | None = None
        self.thumb_at: float | None = None
        self._thumb_rect = QRect()
        self._last_save = 0.0
        self._drag_offset: QPoint | None = None
        self._dragged = False

        self.setWindowTitle("Walkthrough Sync")
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setWindowFlag(Qt.Tool, True)
        self._apply_flags()
        self.setWindowOpacity(float(cfg.get("opacity", 0.93)))
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

        self.worker = LiveWorker(float(cfg.get("interval", 2.0)),
                                 bool(cfg.get("record_sessions", True)))
        self.worker.state.connect(self._on_state)
        self.worker.exe_detected.connect(self._on_exe)
        self.worker.start()

        self._tray()
        self._start_current()
        self._resize()
        # An analysis that was interrupted or failed last time picks up again
        # on launch (a finished download is reused).
        g = self.game
        if g is not None and not g.ready and g.meta.get("url") \
                and g.meta.get("status") in ("new", "indexing", "failed"):
            QTimer.singleShot(1500, lambda: self.index_worker is None and self._index(g))

        screens = QGuiApplication.instance()
        screens.screenAdded.connect(lambda *_: QTimer.singleShot(500, self.place))
        screens.screenRemoved.connect(lambda *_: QTimer.singleShot(500, self.place))

    # ------------------------------------------------------------------ #
    # game lifecycle
    # ------------------------------------------------------------------ #
    def _start_current(self):
        g = self.game
        self.thumbs = None
        self.thumb_pix, self.thumb_at = None, None
        self.live = {}
        if g is None:
            self.worker.set_game(None)
        elif g.ready:
            self.thumbs = g.thumbs()
            pos = g.position or 0.0
            self.live = {"status": "starting", "position": pos, "confidence": 0.0}
            self.worker.set_game(g, start=pos, sigma=300)
        else:
            self.worker.set_game(None)
        self._resize()
        self.update()

    def _switch(self, slug: str):
        self._save_position(force=True)
        self.game = G.get(slug)
        self.cfg["current_game"] = slug
        C.save(self.cfg)
        self._start_current()

    def new_game(self):
        if self.index_worker is not None:
            QMessageBox.information(self, "Walkthrough Sync",
                                    "Another walkthrough is still being analysed.")
            return
        dlg = NewGameDialog(self)
        if not dlg.exec():
            return
        url, name, start = dlg.values()
        self._save_position(force=True)
        g = G.create(name, url, start)
        self.game = g
        self.cfg["current_game"] = g.slug
        C.save(self.cfg)
        self._index(g)

    def _index(self, g: G.Game):
        self.worker.set_game(None)
        self.live = {}
        self.thumbs = None
        self.thumb_pix = None
        self.index_info = ("info", 0.0, "Starting…")
        self.index_worker = IndexWorker(g, bool(self.cfg.get("keep_video")))
        self.index_worker.progress.connect(self._on_index_progress)
        self.index_worker.done.connect(self._on_index_done)
        self.index_worker.start()
        self._resize()
        self.update()

    def reindex(self):
        if self.game is not None and self.index_worker is None:
            self.game.load_meta()
            self._index(self.game)

    def cancel_index(self):
        if self.index_worker is not None:
            self.index_worker.cancel.set()

    def _on_index_progress(self, stage, frac, text):
        prev = self.index_info[0] if self.index_info else None
        self.index_info = (stage, frac, text)
        if stage != prev and self.game is not None:
            self.game.load_meta()   # picks up the name taken from the video title
        self._resize()
        self.update()

    def _on_index_done(self, slug, ok, msg):
        w = self.index_worker
        self.index_worker = None
        self.index_info = None
        if w is not None:
            w.wait(2000)
        if self.game is not None and self.game.slug == slug:
            self.game.load_meta()
            self._start_current()
        if ok:
            self.tray.showMessage("Walkthrough Sync",
                                  f"{self.game.name if self.game else slug} is ready. "
                                  "Start playing — tracking begins when the game is in focus.",
                                  QSystemTrayIcon.Information, 6000)
        elif msg != "Cancelled":
            self.tray.showMessage("Walkthrough Sync", f"Analysis failed: {msg}",
                                  QSystemTrayIcon.Warning, 8000)

    def _on_state(self, d: dict):
        if self.game is None or d.get("slug") != self.game.slug:
            return
        self.live = d
        pos = d.get("position")
        if pos is not None and d.get("status") in ("tracking", "updated", "paused"):
            self.game.set_position(pos)
            self._save_position()
        self._update_thumb()
        self._resize()
        self.update()

    def _on_exe(self, slug, exe):
        if self.game is not None and self.game.slug == slug:
            self.game.meta["exe"] = exe
            self.game.save_meta()

    def _save_position(self, force=False):
        if self.game is None or not self.game.ready:
            return
        now = time.monotonic()
        if force or now - self._last_save > 20:
            self._last_save = now
            self.game.save_meta()

    def _update_thumb(self):
        if not self.cfg.get("show_thumb", True) or self.thumbs is None:
            self.thumb_pix = None
            return
        pos = self.live.get("position")
        if pos is None:
            return
        if self.thumb_at is not None and abs(self.thumb_at - pos) < G.THUMB_EVERY:
            return
        data = self.thumbs.get(pos)
        if not data:
            return
        img = QImage.fromData(data, "JPEG")
        if img.isNull():
            return
        self.thumb_pix = QPixmap.fromImage(img)
        self.thumb_at = pos

    # ------------------------------------------------------------------ #
    # what to show
    # ------------------------------------------------------------------ #
    def _lines(self) -> tuple[str, str, str, QColor, float, QColor, str]:
        """title, line2, status, status colour, fraction, ring colour, centre text"""
        g = self.game
        if g is None:
            return ("Walkthrough Sync", "No game yet", "Right-click → New game…",
                    GREY, 0.0, ACCENT, "")
        if self.index_info is not None:
            stage, frac, text = self.index_info
            label = {"info": "Preparing", "download": "Downloading", "analyse": "Analysing",
                     "check": "Self-check"}.get(stage, "Working")
            return (g.name, f"{label} walkthrough", text, AMBER, frac, AMBER,
                    f"{frac * 100:.0f}%")
        if not g.ready:
            st = g.meta.get("status")
            msg = g.meta.get("error", "") if st == "failed" else \
                "Analysis was interrupted" if st in ("indexing", "new") else "Not analysed"
            return (g.name, "Walkthrough not ready", (msg or "")[:90] + " · right-click → Retry",
                    RED, 0.0, ACCENT, "")

        dur = g.duration or 1
        pos = self.live.get("position", g.position or 0.0) or 0.0
        frac = max(0.0, min(1.0, pos / dur))
        line2 = self._progress_text(pos, dur, frac)
        st = self.live.get("status", "")
        conf = float(self.live.get("confidence", 0.0) or 0.0)
        if st == "error":
            s, col = "Error: " + self.live.get("message", ""), RED
        elif st == "loading":
            s, col = "Loading the image model…", AMBER
        elif st == "user_paused":
            s, col = "Tracking paused", GREY
        elif st == "away":
            exe = g.meta.get("exe", "")
            s = f"Waiting for {exe}" if exe else "Waiting for a fullscreen game"
            col = GREY
        elif st in ("still", "paused"):
            s, col = "Paused (menu or pause screen)", GREY
        elif st in ("tracking", "updated"):
            if conf >= 0.5:
                s, col = "Tracking · confident", GREEN
            elif conf >= 0.2:
                s, col = "Tracking · fairly sure", AMBER
            else:
                s, col = "Searching…", AMBER
        else:
            s, col = "Starting…", GREY
        return (g.name, line2, s, col, frac, ACCENT, f"{frac * 100:.0f}%")

    def _progress_text(self, pos: float, dur: float, frac: float) -> str:
        """The line under the name; empty when time and percentage are both off."""
        mode = self.cfg.get("time_text", "position")
        pct = f"{frac * 100:.1f}%"
        if mode == "position":
            t = f"{C.fmt_time(pos)} / {C.fmt_time(dur)}"
        elif mode == "remaining":
            t = f"{C.fmt_time(max(0.0, dur - pos))} left"
        else:
            t = ""
        if not self.cfg.get("percent_text"):
            return t
        return f"{pct} · {t}" if t else f"{pct} complete"

    # ------------------------------------------------------------------ #
    # geometry & painting
    # ------------------------------------------------------------------ #
    def _fonts(self):
        base = QFont(self.font())
        base.setPointSizeF(float(self.cfg.get("font_size", 10)))
        title = QFont(base)
        title.setBold(True)
        title.setPointSizeF(base.pointSizeF() + 1)
        small = QFont(base)
        small.setPointSizeF(max(7.0, base.pointSizeF() - 1))
        return title, base, small

    def _text_width(self) -> int:
        title, base, small = self._fonts()
        k = float(self.cfg.get("font_size", 10)) / 10
        return int(max(220, min(340, 250 * k)))

    def _minimal(self) -> bool:
        return self.cfg.get("layout") == "minimal"

    def _minimal_font(self) -> QFont:
        f = QFont(self.font())
        f.setPointSizeF(float(self.cfg.get("font_size", 10)) + 1)
        f.setBold(True)
        return f

    def _minimal_text(self) -> tuple[str, QColor, QColor]:
        """'12.34%', its colour, and the status dot's colour — the same dot as
        on the full card: green confident, amber unsure / analysing, grey
        waiting for the game or paused, red when something's wrong."""
        g = self.game
        _, _, _, dot, frac, _, _ = self._lines()
        if g is None:
            return "–", MUTED, dot
        if self.index_info is not None:
            return f"{self.index_info[1] * 100:.0f}%", AMBER, dot
        if not g.ready:
            return "!", RED, dot
        if self.cfg.get("minimal_decimals", True):
            return f"{frac * 100:.2f}%", TEXT, dot
        # whole numbers: floor, so it only reads 100% at the very end
        return f"{min(100, int(frac * 100 + 1e-9))}%", TEXT, dot

    def _minimal_dot(self, fm: QFontMetrics) -> int:
        return max(7, round(fm.height() * 0.38))

    def _resize(self):
        if self._minimal():
            fm = QFontMetrics(self._minimal_font())
            # sized for the widest value so the pill doesn't twitch as it counts
            widest = "100.00%" if self.cfg.get("minimal_decimals", True) else "100%"
            w = 9 + self._minimal_dot(fm) + 7 + fm.horizontalAdvance(widest) + 9
            h = fm.height() + 2 * 4
            if self.size().width() != w or self.size().height() != h:
                self.setFixedSize(w, h)
                self.place()
            return
        title, base, small = self._fonts()
        tw = self._text_width()
        w = PAD * 2 + DONUT + GAP + tw
        has2 = bool(self._lines()[1])
        lh = (QFontMetrics(title).height() + QFontMetrics(small).height() + 3
              + ((QFontMetrics(base).height() + 3) if has2 else 0))
        h = PAD * 2 + max(DONUT, lh)
        if self.thumb_pix is not None and self.cfg.get("show_thumb", True) \
                and self.index_info is None:
            h += int((w - 2 * PAD) * 9 / 16) + 8
        if self.size().width() != w or self.size().height() != h:
            self.setFixedSize(w, h)
            self.place()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        rad = 7 if self._minimal() else 10
        path.addRoundedRect(r, rad, rad)
        p.fillPath(path, BG)
        p.setPen(QPen(QColor(255, 255, 255, 26), 1.0))
        p.drawPath(path)

        if self._minimal():
            self._thumb_rect = QRect()
            text, col, dcol = self._minimal_text()
            f = self._minimal_font()
            fm = QFontMetrics(f)
            d = self._minimal_dot(fm)
            p.setPen(Qt.NoPen)
            p.setBrush(dcol)
            p.drawEllipse(QRectF(9, (self.height() - d) / 2, d, d))
            p.setFont(f)
            p.setPen(col)
            p.drawText(QRect(9 + d + 7, 0, self.width() - (9 + d + 7) - 9, self.height()),
                       Qt.AlignRight | Qt.AlignVCenter, text)
            p.end()
            return

        title, base, small = self._fonts()
        name, line2, status, scol, frac, ring, centre = self._lines()

        # donut
        dr = QRectF(PAD, PAD, DONUT, DONUT)
        wpen = DONUT * 0.15
        ar = dr.adjusted(wpen / 2, wpen / 2, -wpen / 2, -wpen / 2)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(TRACK, wpen, Qt.SolidLine, Qt.FlatCap))
        p.drawEllipse(ar)
        if frac > 0:
            p.setPen(QPen(ring, wpen, Qt.SolidLine, Qt.FlatCap if frac >= 0.999 else Qt.RoundCap))
            if frac >= 0.999:
                p.drawEllipse(ar)
            else:
                p.drawArc(ar, 90 * 16, -max(1, int(round(frac * 5760))))
        if centre:
            f = QFont(base)
            f.setBold(True)
            f.setPixelSize(max(9, int(DONUT * 0.24)))
            p.setFont(f)
            p.setPen(TEXT)
            p.drawText(dr, Qt.AlignCenter, centre)

        # text
        x = PAD + DONUT + GAP
        tw = self.width() - x - PAD
        fm_t, fm_b, fm_s = QFontMetrics(title), QFontMetrics(base), QFontMetrics(small)
        total = fm_t.height() + fm_s.height() + 3 + ((fm_b.height() + 3) if line2 else 0)
        y = PAD + max(0, (DONUT - total) // 2)
        p.setFont(title)
        p.setPen(TEXT)
        p.drawText(QRect(x, y, tw, fm_t.height()), Qt.AlignLeft | Qt.AlignVCenter,
                   fm_t.elidedText(name, Qt.ElideRight, tw))
        y += fm_t.height() + 3
        if line2:
            p.setFont(base)
            p.setPen(TEXT)
            p.drawText(QRect(x, y, tw, fm_b.height()), Qt.AlignLeft | Qt.AlignVCenter,
                       fm_b.elidedText(line2, Qt.ElideRight, tw))
            y += fm_b.height() + 3
        dot = 7
        p.setPen(Qt.NoPen)
        p.setBrush(scol)
        p.drawEllipse(QRectF(x, y + (fm_s.height() - dot) / 2, dot, dot))
        p.setFont(small)
        p.setPen(MUTED)
        p.drawText(QRect(x + dot + 6, y, tw - dot - 6, fm_s.height()),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   fm_s.elidedText(status, Qt.ElideRight, tw - dot - 6))

        # matched frame
        self._thumb_rect = QRect()
        if self.thumb_pix is not None and self.cfg.get("show_thumb", True) \
                and self.index_info is None:
            tw_ = self.width() - 2 * PAD
            th = int(tw_ * 9 / 16)
            top = self.height() - PAD - th
            rect = QRect(PAD, top, tw_, th)
            self._thumb_rect = rect
            clip = QPainterPath()
            clip.addRoundedRect(QRectF(rect), 6, 6)
            p.save()
            p.setClipPath(clip)
            p.drawPixmap(rect, self.thumb_pix)
            p.restore()
            # timestamp chip
            label = "▶ " + C.fmt_time(self.thumb_at)
            fm = QFontMetrics(small)
            cw, chh = fm.horizontalAdvance(label) + 12, fm.height() + 4
            chip = QRectF(rect.right() - cw - 6, rect.bottom() - chh - 6, cw, chh)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRoundedRect(chip, 4, 4)
            p.setFont(small)
            p.setPen(TEXT)
            p.drawText(chip, Qt.AlignCenter, label)
        p.end()

    # ------------------------------------------------------------------ #
    # placement & dragging
    # ------------------------------------------------------------------ #
    def _screen(self):
        pos = self.cfg.get("position") or {}
        if pos.get("x") is not None and pos.get("y") is not None:
            s = QGuiApplication.screenAt(QPoint(int(pos["x"]) + 5, int(pos["y"]) + 5))
            if s is not None:
                return s
        return QGuiApplication.primaryScreen()

    def place(self):
        g = self._screen().availableGeometry()
        margin = int(self.cfg.get("margin", 12))
        pos = self.cfg.get("position") or {}
        dock = self.cfg.get("dock", "right")
        w, h = self.width(), self.height()
        if pos.get("y") is not None:
            y = int(pos["y"])
        else:
            va = self.cfg.get("valign", "top")
            y = (g.top() + margin if va == "top" else
                 g.bottom() - h - margin + 1 if va == "bottom" else
                 g.top() + (g.height() - h) // 2)
        if dock == "right":
            x = g.right() - w - margin + 1
        elif dock == "left":
            x = g.left() + margin
        elif pos.get("x") is not None:
            x = int(pos["x"])
        else:
            x = g.right() - w - margin + 1
        x = max(g.left(), min(x, g.right() - w + 1))
        y = max(g.top(), min(y, g.bottom() - h + 1))
        self.move(x, y)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._dragged = False
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_offset is not None and e.buttons() & Qt.LeftButton:
            target = e.globalPosition().toPoint() - self._drag_offset
            if not self._dragged and (target - self.pos()).manhattanLength() < 4:
                return
            self._dragged = True
            self.move(target)
            e.accept()
        else:
            on_thumb = self._thumb_rect.contains(e.position().toPoint())
            self.setCursor(Qt.PointingHandCursor if on_thumb else Qt.ArrowCursor)
            self.setToolTip("Open the walkthrough at this point" if on_thumb else self._tooltip())

    def mouseReleaseEvent(self, e):
        if self._drag_offset is None:
            return
        self._drag_offset = None
        if not self._dragged:
            if self._thumb_rect.contains(e.position().toPoint()):
                self.open_walkthrough()
            return
        scr = QGuiApplication.screenAt(self.frameGeometry().center()) or self._screen()
        g = scr.availableGeometry()
        fg = self.frameGeometry()
        if abs(fg.right() - g.right()) <= SNAP_PX:
            self.cfg["dock"] = "right"
        elif abs(fg.left() - g.left()) <= SNAP_PX:
            self.cfg["dock"] = "left"
        else:
            self.cfg["dock"] = "none"
        self.cfg["position"] = {"x": self.x(), "y": self.y()}
        self.place()
        self.cfg["position"] = {"x": self.x(), "y": self.y()}
        C.save(self.cfg)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton and not self._thumb_rect.contains(e.position().toPoint()):
            self.set_position()

    def _tooltip(self) -> str:
        g = self.game
        if g is None:
            return ""
        parts = [g.meta.get("title", g.name), self._lines()[2]]
        if self.live.get("position") is not None:
            parts.append(f"Best guess {C.fmt_time(self.live['position'])} "
                         f"(±{self.live.get('spread', 0):.0f}s), "
                         f"confidence {self.live.get('confidence', 0) * 100:.0f}%")
            parts.append(f"Last screenshot looked most like {C.fmt_time(self.live.get('best'))} "
                         f"(distinctiveness {self.live.get('zmax', 0):.1f})")
        if g.meta.get("exe"):
            parts.append(f"Game window: {g.meta['exe']}")
        if g.meta.get("selfcheck"):
            parts.append(f"Self-check: {g.meta['selfcheck']}")
        parts.append("Double-click to set your position")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #
    def set_position(self):
        g = self.game
        if g is None or not g.ready:
            return
        cur = self.live.get("position", g.position or 0.0) or 0.0
        t = ask_position(self, cur, g.duration)
        if t is not None:
            self.worker.set_position(t, 90)
            g.set_position(t)
            self.live["position"] = t
            self._save_position(force=True)
            self._update_thumb()
            self.update()

    def nudge(self, seconds: float):
        if self.game is not None and self.game.ready:
            self.worker.nudge(seconds)

    def open_walkthrough(self):
        if self.game is None:
            return
        pos = self.live.get("position", self.game.position)
        QDesktopServices.openUrl(QUrl(self.game.youtube_url(pos)))

    def toggle_pause(self, on: bool):
        self.worker.set_paused(on)

    def link_exe(self, exe: str):
        if self.game is None:
            return
        self.game.meta["exe"] = exe
        self.game.save_meta()
        self.worker.set_exe(exe)

    def rename_game(self):
        g = self.game
        if g is None:
            return
        name, ok = QInputDialog.getText(self, "Rename", "Game name:", text=g.name)
        if ok and name.strip():
            g.meta["name"] = name.strip()
            g.meta["name_from_title"] = False
            g.save_meta()
            self.update()

    def remove_game(self):
        g = self.game
        if g is None:
            return
        if self.index_worker is not None:
            return
        if QMessageBox.question(
                self, "Remove game",
                f"Remove “{g.name}” and its analysed walkthrough? This cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.worker.set_game(None)
        g.delete()
        rest = G.list_games()
        self._switch(rest[-1].slug if rest else "")

    # ------------------------------------------------------------------ #
    # menu
    # ------------------------------------------------------------------ #
    def _menu(self, pos):
        m = QMenu(self)
        g = self.game
        if g is not None:
            m.addSection(g.name)
            if g.ready:
                m.addAction("Set my position…", self.set_position)
                nm = m.addMenu("Nudge")
                for label, s in (("Back 5 min", -300), ("Back 1 min", -60),
                                 ("Forward 1 min", 60), ("Forward 5 min", 300)):
                    nm.addAction(label, lambda s=s: self.nudge(s))
                m.addAction("Open walkthrough here", self.open_walkthrough)
                a = m.addAction("Pause tracking")
                a.setCheckable(True)
                a.setChecked(self.live.get("status") == "user_paused")
                a.toggled.connect(self.toggle_pause)

                wm = m.addMenu("Game window")
                grp = QActionGroup(wm)
                cur = (g.meta.get("exe") or "").lower()
                act = QAction("Detect automatically (fullscreen game)", wm)
                act.setCheckable(True)
                act.setChecked(not cur)
                act.triggered.connect(lambda: self.link_exe(""))
                grp.addAction(act)
                wm.addAction(act)
                wm.addSeparator()
                seen = set()
                if cur:
                    act = QAction(cur, wm)
                    act.setCheckable(True)
                    act.setChecked(True)
                    grp.addAction(act)
                    wm.addAction(act)
                    seen.add(cur)
                for w in capture.list_windows():
                    if w.exe in seen:
                        continue
                    seen.add(w.exe)
                    title = w.title if len(w.title) < 40 else w.title[:38] + "…"
                    act = QAction(f"{w.exe} — {title}", wm)
                    act.setCheckable(True)
                    act.triggered.connect(lambda _c=False, e=w.exe: self.link_exe(e))
                    grp.addAction(act)
                    wm.addAction(act)
            if self.index_worker is not None:
                m.addAction("Cancel analysis", self.cancel_index)
            else:
                m.addAction("Retry analysis" if not g.ready else "Re-analyse walkthrough",
                            self.reindex)
            m.addSeparator()

        gm = m.addMenu("Games")
        grp = QActionGroup(gm)
        for game in G.list_games():
            pct = ""
            if game.ready and game.duration and game.position is not None:
                pct = f"  ·  {game.position / game.duration * 100:.0f}%"
            act = QAction(game.name + pct, gm)
            act.setCheckable(True)
            act.setChecked(g is not None and game.slug == g.slug)
            act.triggered.connect(lambda _c=False, s=game.slug: self._switch(s))
            act.setEnabled(self.index_worker is None)
            grp.addAction(act)
            gm.addAction(act)
        gm.addSeparator()
        gm.addAction("New game…", self.new_game)
        if g is not None:
            gm.addAction("Rename current…", self.rename_game)
            gm.addAction("Remove current…", self.remove_game)
        m.addAction("New game…", self.new_game)
        m.addSeparator()

        a = m.addAction("Minimal (percentage only)")
        a.setCheckable(True)
        a.setChecked(self._minimal())
        a.toggled.connect(lambda on: self._set_key("layout", "minimal" if on else "full"))
        a = m.addAction("    Decimals  (12.34% instead of 12%)")
        a.setCheckable(True)
        a.setChecked(bool(self.cfg.get("minimal_decimals", True)))
        a.setEnabled(self._minimal())
        a.toggled.connect(lambda on: self._set_key("minimal_decimals", bool(on)))

        pm = m.addMenu("Progress text")
        pm.setEnabled(not self._minimal())
        self._radio(pm, "time_text",
                    (("position", "Position  (2:15:34 / 7:53:40)"),
                     ("remaining", "Time remaining  (5:38:06 left)"),
                     ("off", "No time")), self._set_key)
        pm.addSeparator()
        a = pm.addAction("Percentage  (29.3%)")
        a.setCheckable(True)
        a.setChecked(bool(self.cfg.get("percent_text")))
        a.toggled.connect(lambda on: self._set_key("percent_text", bool(on)))

        a = m.addAction("Show matched frame")
        a.setEnabled(not self._minimal())
        a.setCheckable(True)
        a.setChecked(bool(self.cfg.get("show_thumb", True)))
        a.toggled.connect(self._set_thumb)

        self._radio(m.addMenu("Dock"), "dock",
                    (("right", "Right edge"), ("left", "Left edge"),
                     ("none", "Floating (drag anywhere)")), self._set_dock)
        self._radio(m.addMenu("Vertical position"), "valign",
                    (("top", "Top"), ("center", "Middle"), ("bottom", "Bottom")),
                    self._set_valign)
        self._radio(m.addMenu("Window layer"), "window_layer",
                    (("top", "Always on top"), ("normal", "Normal (can be covered)"),
                     ("bottom", "Always at the bottom")), self._set_layer)
        om = m.addMenu("Opacity")
        grp = QActionGroup(om)
        for v in (1.0, 0.93, 0.8, 0.65, 0.5):
            act = QAction(f"{int(v * 100)}%", om)
            act.setCheckable(True)
            act.setChecked(abs(float(self.cfg.get("opacity", 0.93)) - v) < 0.01)
            act.triggered.connect(lambda _c=False, val=v: self._set_opacity(val))
            grp.addAction(act)
            om.addAction(act)
        sm = m.addMenu("Text size")
        grp = QActionGroup(sm)
        for v in (8, 9, 10, 11, 12, 14):
            act = QAction(f"{v} pt", sm)
            act.setCheckable(True)
            act.setChecked(int(self.cfg.get("font_size", 10)) == v)
            act.triggered.connect(lambda _c=False, val=v: self._set_key("font_size", val))
            grp.addAction(act)
            sm.addAction(act)
        m.addSeparator()
        m.addAction("Open data folder", self._open_data)
        m.addAction("Reset position", self._reset_position)
        m.addAction("Quit", self.quit)
        m.exec(self.mapToGlobal(pos))

    def _radio(self, menu, key, options, handler):
        grp = QActionGroup(menu)
        for value, text in options:
            act = QAction(text, menu)
            act.setCheckable(True)
            act.setChecked(self.cfg.get(key) == value)
            act.triggered.connect(lambda _c=False, v=value, k=key: handler(k, v))
            grp.addAction(act)
            menu.addAction(act)

    # ------------------------------------------------------------------ #
    # settings
    # ------------------------------------------------------------------ #
    def _set_key(self, key, value):
        self.cfg[key] = value
        C.save(self.cfg)
        self._resize()
        self.update()

    def _set_thumb(self, on):
        self.cfg["show_thumb"] = bool(on)
        C.save(self.cfg)
        self.thumb_at = None
        self._update_thumb()
        self._resize()
        self.update()

    def _set_dock(self, _k, value):
        self.cfg["dock"] = value
        self.cfg["position"] = {"x": self.x(), "y": self.y()}
        C.save(self.cfg)
        self.place()

    def _set_valign(self, _k, value):
        self.cfg["valign"] = value
        self.cfg["position"] = {"x": self.x(), "y": None}
        if self.cfg.get("dock") == "none":
            self.cfg["dock"] = "right"
        C.save(self.cfg)
        self.place()

    def _apply_flags(self):
        layer = self.cfg.get("window_layer", "top")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, layer == "top")
        self.setWindowFlag(Qt.WindowStaysOnBottomHint, layer == "bottom")

    def showEvent(self, e):
        super().showEvent(e)
        # Keep the widget out of screen captures: it sits on top of the game,
        # and it shouldn't end up in the screenshots matched against the video.
        if sys.platform.startswith("win") and self.cfg.get("hide_from_capture", True):
            try:
                import ctypes
                ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x11)
            except Exception:
                pass

    def _set_layer(self, _k, value):
        self.cfg["window_layer"] = value
        C.save(self.cfg)
        self._apply_flags()
        self.show()

    def _set_opacity(self, v):
        self.cfg["opacity"] = v
        self.setWindowOpacity(v)
        C.save(self.cfg)

    def _reset_position(self):
        self.cfg["dock"] = "right"
        self.cfg["valign"] = "top"
        self.cfg["position"] = {"x": None, "y": None}
        C.save(self.cfg)
        self.place()
        self.show()

    def _open_data(self):
        path = str(self.game.dir if self.game else C.DATA_DIR)
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # noqa: S606
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # tray
    # ------------------------------------------------------------------ #
    def _tray(self):
        pm = QPixmap(64, 64)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(TRACK, 9))
        p.drawEllipse(QRectF(8, 8, 48, 48))
        p.setPen(QPen(ACCENT, 9, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(8, 8, 48, 48), 90 * 16, -int(0.62 * 5760))
        p.setPen(Qt.NoPen)
        p.setBrush(TEXT)
        tri = QPainterPath()
        tri.moveTo(27, 22)
        tri.lineTo(44, 32)
        tri.lineTo(27, 42)
        tri.closeSubpath()
        p.drawPath(tri)
        p.end()
        icon = QIcon(pm)
        self.setWindowIcon(icon)
        self.tray = QSystemTrayIcon(icon, self)
        self.tray.setToolTip("Walkthrough Sync")
        menu = QMenu()
        menu.addAction("Bring to front", self._bring_to_front)
        menu.addAction("Show / hide", lambda: self.setVisible(not self.isVisible()))
        menu.addAction("New game…", self.new_game)
        menu.addAction("Reset position", self._reset_position)
        menu.addSeparator()
        menu.addAction("Quit", self.quit)
        self._tray_menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda r: self.setVisible(not self.isVisible())
            if r == QSystemTrayIcon.Trigger else None)
        self.tray.show()

    def _bring_to_front(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def quit(self):
        self._save_position(force=True)
        if self.index_worker is not None:
            self.index_worker.cancel.set()
            self.index_worker.wait(5000)
        self.worker.stop()
        self.worker.wait(5000)
        self.tray.hide()
        QApplication.quit()

    def closeEvent(self, e):
        self.quit()
        e.accept()
