"""Background threads: the live tracking loop and the indexer."""
from __future__ import annotations

import queue
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from . import capture
from . import config as C
from . import embedder as E
from . import games as G
from . import indexer
from . import sessions as S
from .tracker import PauseDetector, Tracker, prepare

STATIC_DIFF = 1.2      # mean abs change (0-255) of the tiny frame below which it's "still"
STATIC_TICKS = 3       # this many still screenshots in a row = paused
DARK_LEVEL = 12        # mean brightness below this = black / loading screen


class LiveWorker(QThread):
    """Screenshots -> embeddings -> tracker, every `interval` seconds.

    The UI talks to it through commands (thread-safe queue); it reports back
    through `state` with a plain dict.
    """
    state = Signal(dict)
    exe_detected = Signal(str, str)   # slug, exe

    def __init__(self, interval: float, record: bool = True):
        super().__init__()
        self.interval = interval
        self.record = record
        self.cmds: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.game: G.Game | None = None
        self.tracker: Tracker | None = None
        self.mean: np.ndarray | None = None
        self.user_paused = False
        self.rec: S.Recorder | None = None
        self.pause = PauseDetector()

    # --- commands (called from the UI thread) -------------------------- #
    def set_game(self, game: G.Game | None, start: float | None = None, sigma: float = 600):
        self.cmds.put(("game", game, start, sigma))

    def set_position(self, seconds: float, sigma: float = 120):
        self.cmds.put(("pos", seconds, sigma))

    def nudge(self, seconds: float):
        self.cmds.put(("nudge", seconds))

    def set_paused(self, on: bool):
        self.cmds.put(("pause", on))

    def set_exe(self, exe: str):
        self.cmds.put(("exe", exe))

    def stop(self):
        self._stop.set()
        self.cmds.put(("noop",))

    # --- loop ----------------------------------------------------------- #
    def _emit(self, status: str, **kw):
        d = {"status": status, "slug": self.game.slug if self.game else ""}
        if self.tracker is not None:
            est = self.tracker.estimate()
            d.update(position=est.position, confidence=est.confidence, spread=est.spread,
                     best=est.best_match, zmax=est.zmax)
        d.update(kw)
        self.state.emit(d)

    def _load_game(self, game, start, sigma):
        if self.rec is not None:
            self.rec.flush()
            self.rec = None
        self.game, self.tracker = game, None
        if game is None or not game.ready:
            return
        raw, mean = game.load_index()
        self.mean = mean
        emb = prepare(raw, mean)
        self.tracker = Tracker(emb, weight=game.load_weight(emb))
        if start is None:
            start = game.position or 0.0
        self.tracker.reset(start, sigma)
        if self.record:
            self.rec = S.Recorder(game.dir, raw.shape[1])
            self.rec.add(S.START, val=start, val2=sigma)
        C.log(f"live {game.slug}: start at {C.fmt_time(start)} +-{sigma:.0f}s")

    def _handle(self, cmd) -> None:
        kind = cmd[0]
        if kind == "game":
            self._load_game(*cmd[1:])
        elif kind == "pos" and self.tracker is not None:
            self.tracker.reset(cmd[1], cmd[2])
            if self.rec is not None:
                self.rec.add(S.SET, val=cmd[1], val2=cmd[2])
            C.log(f"live: user set position {C.fmt_time(cmd[1])}")
        elif kind == "nudge" and self.tracker is not None:
            self.tracker.shift(cmd[1])
            if self.rec is not None:
                self.rec.add(S.NUDGE, val=cmd[1])
            C.log(f"live: user nudged {cmd[1]:+.0f}s")
        elif kind == "pause":
            self.user_paused = bool(cmd[1])
        elif kind == "exe" and self.game is not None:
            self.game.meta["exe"] = cmd[1]

    def run(self):
        grabber = None
        last_t = None
        prev_tiny = None
        still = 0
        n = 0
        try:
            while not self._stop.is_set():
                # commands, then sleep until the next tick (woken by commands)
                deadline = time.monotonic() + self.interval
                handled = False
                while True:
                    timeout = deadline - time.monotonic()
                    try:
                        cmd = self.cmds.get(timeout=max(0.0, timeout)) if timeout > 0 \
                            else self.cmds.get_nowait()
                    except queue.Empty:
                        break
                    self._handle(cmd)
                    handled = True
                    if self._stop.is_set():
                        return
                if handled and self.tracker is not None:
                    self._emit("updated")

                if self.game is None or self.tracker is None:
                    last_t = None
                    continue
                if self.user_paused:
                    last_t = None
                    self._emit("user_paused")
                    continue
                if not E.loaded():
                    self._emit("loading")
                    E.get()
                    last_t = None
                    continue

                win, fresh = capture.find_game(self.game.meta.get("exe", ""))
                if win is None:
                    self.pause.reset()
                    if last_t is not None and self.rec is not None:
                        self.rec.add(S.AWAY)
                    last_t, prev_tiny, still = None, None, 0
                    self._emit("away")
                    continue
                if fresh:
                    self.game.meta["exe"] = win.exe
                    self.exe_detected.emit(self.game.slug, win.exe)
                    C.log(f"live {self.game.slug}: linked to {win.exe} ('{win.title}')")

                if grabber is None:
                    grabber = capture.Grabber()
                frame = grabber.grab(win.rect)
                now = time.monotonic()
                dt = self.interval if last_t is None else min(now - last_t, 3 * self.interval)
                last_t = now

                small = capture.tiny(frame)
                if prev_tiny is not None and float(np.abs(small - prev_tiny).mean()) < STATIC_DIFF:
                    still += 1
                else:
                    still = 0
                prev_tiny = small
                if still >= STATIC_TICKS:
                    self.pause.reset()
                    if still == STATIC_TICKS and self.rec is not None:
                        self.rec.add(S.STILL)
                    self._emit("still")
                    continue

                dark = float(small.mean()) < DARK_LEVEL
                q_raw = None if dark else E.get().embed(frame)[0]
                # Paused (a menu, the map, a pause screen): time spent here isn't
                # progress, so the tracker doesn't move forward.
                paused = self.pause.update(q_raw)
                if not paused:
                    self.tracker.predict(dt)
                if q_raw is not None:
                    self.tracker.observe(prepare(q_raw[None], self.mean)[0])
                self.tracker.settle(dt)
                est = self.tracker.estimate()
                if self.rec is not None:
                    self.rec.add(S.DARK if dark else S.OBS, dt, q_raw, est=est.position,
                                 conf=est.confidence, frame=frame)
                n += 1
                if n % 5 == 0:
                    C.log(f"tick pos {C.fmt_time(est.position)} conf {est.confidence:.2f} "
                          f"spread {est.spread:.0f}s best {C.fmt_time(est.best_match)} "
                          f"z {est.zmax:.1f}{' dark' if dark else ''}{' paused' if paused else ''}")
                self._emit("paused" if paused else "tracking", dark=dark)
        except Exception as e:  # keep the UI alive; show the error on the card
            C.log("live worker crashed:\n" + traceback.format_exc())
            self.state.emit({"status": "error", "message": str(e)[:200],
                             "slug": self.game.slug if self.game else ""})
        finally:
            if grabber is not None:
                grabber.close()
            if self.rec is not None:
                self.rec.flush()


class IndexWorker(QThread):
    progress = Signal(str, float, str)       # stage, fraction, text
    done = Signal(str, bool, str)            # slug, ok, message

    def __init__(self, game: G.Game, keep_video: bool):
        super().__init__()
        self.game = game
        self.keep_video = keep_video
        self.cancel = threading.Event()

    def run(self):
        try:
            indexer.build(self.game, lambda s, f, t: self.progress.emit(s, f, t),
                          self.cancel, keep_video=self.keep_video)
            self.done.emit(self.game.slug, True, self.game.meta.get("selfcheck", ""))
        except indexer.Cancelled:
            self.game.meta["status"] = "cancelled"
            self.game.save_meta()
            self.done.emit(self.game.slug, False, "Cancelled")
        except Exception as e:
            C.log("index failed:\n" + traceback.format_exc())
            self.game.meta["status"] = "failed"
            self.game.meta["error"] = str(e)[:300]
            self.game.save_meta()
            self.done.emit(self.game.slug, False, str(e)[:300])
