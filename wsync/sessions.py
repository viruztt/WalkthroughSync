"""Record what the tracker saw during a play session, and replay it.

Each session is written to data/games/<game>/sessions/<start>.npz:
one row per screenshot tick (time, dt, kind, the screenshot's embedding,
the estimate shown) plus your manual corrections. A small JPEG of every
15th screenshot goes into a folder next to it. ~3 MB per hour of play.

With that, tracker changes can be tested against your real playthrough
(`main.py --replay-session FILE`) instead of simulations.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np

from . import config as C

OBS, DARK, STILL, AWAY, SET, NUDGE, START = range(7)
FRAME_EVERY = 15        # ticks between saved JPEGs
MAX_FRAMES = 1500       # per session


class Recorder:
    def __init__(self, game_dir: Path, dim: int):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(game_dir) / "sessions"
        self.path = self.dir / f"{stamp}.npz"
        self.frames_dir = self.dir / stamp
        self.dim = dim
        self.rows: list[tuple] = []
        self.qs: list[np.ndarray] = []
        self._last_flush = time.monotonic()
        self._ticks = 0
        self._frames = 0

    def add(self, kind: int, dt: float = 0.0, q: np.ndarray | None = None,
            val: float = 0.0, val2: float = 0.0, est: float = -1.0, conf: float = 0.0,
            frame: np.ndarray | None = None) -> None:
        self.rows.append((time.time(), dt, kind, val, val2, est, conf))
        self.qs.append(np.zeros(self.dim, np.float16) if q is None
                       else np.asarray(q, np.float16).reshape(-1))
        if kind in (OBS, DARK):
            self._ticks += 1
            if frame is not None and self._ticks % FRAME_EVERY == 1 and self._frames < MAX_FRAMES:
                self._save_frame(frame, est)
        if time.monotonic() - self._last_flush > 60:
            self.flush()

    def _save_frame(self, frame: np.ndarray, est: float) -> None:
        try:
            from PIL import Image
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            name = f"{time.strftime('%H%M%S')}_at_{C.fmt_time(est).replace(':', '-')}.jpg"
            bio = io.BytesIO()
            Image.fromarray(frame).save(bio, "JPEG", quality=80)
            (self.frames_dir / name).write_bytes(bio.getvalue())
            self._frames += 1
        except Exception:
            pass

    def flush(self) -> None:
        self._last_flush = time.monotonic()
        if not self.rows:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            r = np.array(self.rows, dtype=np.float64)
            tmp = self.path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, t=r[:, 0], dt=r[:, 1].astype(np.float32),
                                kind=r[:, 2].astype(np.int8), val=r[:, 3].astype(np.float32),
                                val2=r[:, 4].astype(np.float32), est=r[:, 5].astype(np.float32),
                                conf=r[:, 6].astype(np.float32), q=np.stack(self.qs))
            tmp.replace(self.path)
        except Exception as e:
            C.log(f"session save failed: {e!r}")


def replay(path: Path, tracker, mean: np.ndarray, prepare) -> dict:
    """Run `tracker` over a recorded session. Returns arrays: t, recorded
    estimate, replayed estimate, replayed confidence (per OBS/DARK tick)."""
    z = np.load(path)
    out_t, rec, new, conf = [], [], [], []
    for i in range(len(z["kind"])):
        k = int(z["kind"][i])
        if k == START:
            tracker.reset(float(z["val"][i]), float(z["val2"][i]))
        elif k == SET:
            tracker.reset(float(z["val"][i]), float(z["val2"][i]))
        elif k == NUDGE:
            tracker.shift(float(z["val"][i]))
        elif k in (OBS, DARK):
            dt = float(z["dt"][i])
            tracker.predict(dt)
            if k == OBS:
                q = prepare(z["q"][i].astype(np.float32)[None], mean)[0]
                tracker.observe(q)
            if hasattr(tracker, "settle"):
                tracker.settle(dt)
            e = tracker.estimate()
            out_t.append(float(z["t"][i]))
            rec.append(float(z["est"][i]))
            new.append(e.position)
            conf.append(e.confidence)
    return {"t": np.array(out_t), "recorded": np.array(rec), "replayed": np.array(new),
            "confidence": np.array(conf)}
