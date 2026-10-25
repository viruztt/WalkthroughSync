"""One folder per game under data/games/<slug>/:

    meta.json     name, walkthrough URL, duration, last position, game exe …
    emb.npy       (N, D) float16 frame embeddings, one per second of video
    mean.npy      (D,) the walkthrough's average look (subtracted before matching)
    thumbs.bin    160x90 JPEGs, one every THUMB_EVERY seconds, back to back
    thumbs.idx    int64 offsets into thumbs.bin (len = count + 1)
    replay.txt    result of the self-check run after indexing
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import config as C

THUMB_EVERY = 2  # seconds


class Game:
    def __init__(self, folder: Path):
        self.dir = Path(folder)
        self.slug = self.dir.name
        self.meta: dict = {}
        self.load_meta()

    # ------------------------------------------------------------------ #
    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    def load_meta(self) -> None:
        try:
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            self.meta = {}

    def save_meta(self) -> None:
        C.write_json(self.meta_path, self.meta)

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.slug

    @property
    def model_key(self) -> str:
        """The image model this game's walkthrough was analysed with."""
        return self.meta.get("model_key") or "clip-b16"   # games from before the choice

    @property
    def ready(self) -> bool:
        return self.meta.get("status") == "ready" and (self.dir / "emb.npy").exists()

    @property
    def duration(self) -> float:
        return float(self.meta.get("duration") or 0)

    @property
    def position(self) -> float | None:
        p = self.meta.get("position")
        return float(p) if p is not None else None

    def set_position(self, seconds: float) -> None:
        self.meta["position"] = round(float(seconds), 1)
        self.meta["position_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    def youtube_url(self, at: float | None = None) -> str:
        vid = self.meta.get("video_id", "")
        url = f"https://www.youtube.com/watch?v={vid}" if vid else self.meta.get("url", "")
        if at is not None and vid:
            url += f"&t={int(at)}s"
        return url

    # ------------------------------------------------------------------ #
    def load_index(self) -> tuple[np.ndarray, np.ndarray]:
        raw = np.load(self.dir / "emb.npy").astype(np.float32)
        mean = np.load(self.dir / "mean.npy").astype(np.float32)
        return raw, mean

    def load_weight(self, emb: np.ndarray) -> np.ndarray:
        """How unique each frame is (cached in weight.npy; computed once,
        on the GPU when PyTorch has one — a few seconds either way)."""
        from .tracker import WEIGHT_VERSION, distinctiveness
        path = self.dir / "weight.npy"
        if path.exists() and self.meta.get("weight_version") == WEIGHT_VERSION:
            w = np.load(path)
            if len(w) == len(emb):
                return w.astype(np.float32)
        matmul = None
        try:
            import torch
            if torch.cuda.is_available():
                et = torch.from_numpy(emb).cuda().half()
                matmul = lambda a, _b: (torch.from_numpy(a).cuda().half() @ et.T).float().cpu().numpy()  # noqa: E731
        except Exception:
            matmul = None
        w = distinctiveness(emb, matmul=matmul)
        np.save(path, w)
        self.meta["weight_version"] = WEIGHT_VERSION
        self.save_meta()
        return w

    def thumbs(self) -> "Thumbs | None":
        if (self.dir / "thumbs.bin").exists() and (self.dir / "thumbs.idx").exists():
            return Thumbs(self.dir)
        return None

    def delete(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class Thumbs:
    def __init__(self, folder: Path):
        self.offsets = np.fromfile(folder / "thumbs.idx", dtype=np.int64)
        self.path = folder / "thumbs.bin"

    def get(self, seconds: float) -> bytes:
        i = int(round(seconds / THUMB_EVERY))
        n = len(self.offsets) - 1
        if n <= 0:
            return b""
        # nearest non-empty thumb
        for d in range(0, 6):
            for j in (i - d, i + d):
                if 0 <= j < n and self.offsets[j + 1] > self.offsets[j]:
                    with self.path.open("rb") as f:
                        f.seek(int(self.offsets[j]))
                        return f.read(int(self.offsets[j + 1] - self.offsets[j]))
        return b""


def write_thumbs(folder: Path, thumbs: dict[int, bytes], count: int) -> None:
    offsets = np.zeros(count + 1, dtype=np.int64)
    with (folder / "thumbs.bin").open("wb") as f:
        pos = 0
        for i in range(count):
            data = thumbs.get(i, b"")
            f.write(data)
            pos += len(data)
            offsets[i + 1] = pos
    offsets.tofile(folder / "thumbs.idx")


def list_games() -> list[Game]:
    if not C.GAMES_DIR.exists():
        return []
    out = [Game(p) for p in sorted(C.GAMES_DIR.iterdir()) if (p / "meta.json").exists()]
    out.sort(key=lambda g: g.meta.get("created", ""))
    return out


def get(slug: str) -> Game | None:
    if not slug:
        return None
    p = C.GAMES_DIR / slug
    return Game(p) if (p / "meta.json").exists() else None


def create(name: str, url: str, start: float) -> Game:
    base = C.slugify(name or C.video_id(url) or "game")
    slug, n = base, 2
    while (C.GAMES_DIR / slug).exists():
        slug, n = f"{base}-{n}", n + 1
    folder = C.GAMES_DIR / slug
    folder.mkdir(parents=True, exist_ok=True)
    g = Game(folder)
    g.meta = {
        "name": name, "url": url.strip(), "video_id": C.video_id(url),
        "status": "new", "duration": 0, "position": round(float(start), 1),
        "exe": "", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "name_from_title": not name,
    }
    g.save_meta()
    return g
