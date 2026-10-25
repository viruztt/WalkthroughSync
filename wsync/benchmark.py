"""Compare image models on real play: which one finds where you actually were?

Triggered by data/benchmark_request.json (written by hand or by Claude):

    {"game": "<slug>", "models": ["clip-b16", "dinov2-b14", ...],
     "queries": [{"file": "<session>/<capture>.jpg", "truth": 14983.0}, ...]}

`file` is relative to the game's sessions/ folder (the JPEGs a recorded
session saves); `truth` is where you were, in walkthrough seconds. The widget
runs it in the background at start-up: the 360p walkthrough is downloaded
again if needed, decoded every 2 s, and every model embeds the frames and your
screenshots the same way the tracker does. Results go to data/benchmark.txt.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config as C
from . import embedder as E
from . import games as G
from . import indexer

REQUEST = C.DATA_DIR / "benchmark_request.json"
RESULT = C.DATA_DIR / "benchmark.txt"
STEP = 2            # seconds between gallery frames
W, H = 320, 180


def pending() -> dict | None:
    try:
        req = json.loads(REQUEST.read_text(encoding="utf-8"))
        return req if isinstance(req, dict) and req.get("queries") else None
    except Exception:
        return None


def _decode_gallery(video: Path, duration: float, progress, cancel) -> np.ndarray:
    n = int(math.ceil(duration / STEP))
    frames = np.zeros((n, H, W, 3), dtype=np.uint8)
    workers = max(2, min(8, (os.cpu_count() or 4) // 2))
    chunk = int(math.ceil(n / workers)) * STEP
    done = [0]
    lock = threading.Lock()

    def run(k: int):
        start = k * chunk
        if start >= duration:
            return
        cmd = [indexer.ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
               "-ss", str(start), "-t", str(chunk), "-i", str(video), "-an", "-sn",
               "-threads", "2", "-vf", f"fps=1/{STEP},scale={W}:{H}:flags=area,format=rgb24",
               "-f", "rawvideo", "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                **indexer._no_window())
        i = 0
        try:
            while not cancel.is_set():
                buf = proc.stdout.read(W * H * 3)
                if len(buf) < W * H * 3:
                    break
                g = start // STEP + i
                i += 1
                if g < n:
                    frames[g] = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
                with lock:
                    done[0] += 1
                    if done[0] % 200 == 0:
                        progress("bench", 0.1 * done[0] / n, f"Decoding · {done[0]:,} / {n:,} frames")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(workers)))
    if cancel.is_set():
        raise indexer.Cancelled()
    return frames


def _metrics(q: np.ndarray, g: np.ndarray, truth: np.ndarray) -> dict:
    mean = g.mean(axis=0)
    gc = g - mean
    gc /= np.linalg.norm(gc, axis=1, keepdims=True) + 1e-8
    qc = q - mean
    qc /= np.linalg.norm(qc, axis=1, keepdims=True) + 1e-8
    s = qc @ gc.T
    z = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-6)
    t_idx = np.round(truth / STEP).astype(int)
    order = np.argsort(-z, axis=1)
    top1 = order[:, 0] * STEP
    within = np.abs(top1 - truth) <= 60
    top5 = np.array([np.any(np.abs(order[i, :5] * STEP - truth[i]) <= 60) for i in range(len(q))])
    # rank of the best frame within +-10 s of the truth
    ranks, ztrue = [], []
    for i in range(len(q)):
        lo, hi = max(0, t_idx[i] - 5), min(g.shape[0], t_idx[i] + 6)
        best_true = z[i, lo:hi].max()
        ztrue.append(best_true)
        ranks.append(int((z[i] > best_true).sum()) + 1)
    ranks = np.array(ranks)
    return {"top1": within.mean() * 100, "top5": top5.mean() * 100,
            "rank_med": float(np.median(ranks)), "rank10": (ranks <= 10).mean() * 100,
            "z_true": float(np.median(ztrue))}


def run(progress, cancel: threading.Event) -> str:
    req = pending()
    if req is None:
        return "no benchmark request"
    game = G.get(req.get("game", ""))
    if game is None:
        raise RuntimeError(f"benchmark: game {req.get('game')!r} not found")
    models = [m for m in req.get("models") or list(E.MODELS) if m in E.MODELS]
    qs = req["queries"]
    from PIL import Image
    imgs, truth = [], []
    for item in qs:
        p = game.dir / "sessions" / item["file"]
        if p.exists():
            imgs.append(np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BOX)))
            truth.append(float(item["truth"]))
    if not imgs:
        raise RuntimeError("benchmark: none of the screenshots were found")
    imgs = np.stack(imgs)
    truth = np.array(truth)

    progress("bench", 0.0, "Fetching the walkthrough…")
    had_video = any(game.dir.glob("video.*"))
    video = indexer.download(game.meta["url"], game.dir, progress, cancel, game.duration)
    duration = indexer.probe_duration(video) or game.duration
    gallery = _decode_gallery(video, duration, progress, cancel)

    lines = [f"Image model benchmark — {game.name}",
             f"{len(imgs)} of your screenshots with known positions vs {len(gallery):,} "
             f"walkthrough frames (every {STEP} s)", "",
             f"{'model':32} {'top-1':>7} {'top-5':>7} {'rank≤10':>8} {'med rank':>9} "
             f"{'z@truth':>8} {'frames/s':>9}"]
    results = {}
    for mi, key in enumerate(models):
        if cancel.is_set():
            raise indexer.Cancelled()
        base = 0.1 + 0.9 * mi / len(models)
        progress("bench", base, f"Model {mi + 1}/{len(models)}: loading {key}…")
        try:
            emb = E.get(key) if E.loaded(key) else E.Embedder(key)
        except Exception as e:
            lines.append(f"{E.MODELS[key]['label']:32} failed to load: {str(e)[:80]}")
            C.log(f"benchmark {key} failed: {e!r}")
            continue
        t0 = time.time()
        feats = []
        for i in range(0, len(gallery), 256):
            if cancel.is_set():
                raise indexer.Cancelled()
            feats.append(emb.embed(gallery[i:i + 256]))
            progress("bench", base + 0.9 / len(models) * i / len(gallery),
                     f"Model {mi + 1}/{len(models)} ({key}) · {i:,} / {len(gallery):,}")
        g = np.concatenate(feats)
        fps = len(gallery) / max(1e-6, time.time() - t0)
        q = np.concatenate([emb.embed(imgs[i:i + 64]) for i in range(0, len(imgs), 64)])
        m = _metrics(q, g, truth)
        results[key] = m
        lines.append(f"{E.MODELS[key]['label']:32} {m['top1']:6.1f}% {m['top5']:6.1f}% "
                     f"{m['rank10']:7.1f}% {m['rank_med']:9.0f} {m['z_true']:8.2f} {fps:9.0f}")
        C.log(f"benchmark {key}: {m}")
        if not E.loaded(key):
            emb.close()
        del g, feats

    lines += ["", "top-1: the single best match is within 60 s of where you were",
              "top-5: one of the 5 best is; rank≤10: the right moment is among the 10 best",
              "z@truth: how strongly the right moment stands out (higher is better)"]
    RESULT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.replace(REQUEST, REQUEST.with_suffix(".done.json"))
    except OSError:
        pass
    if not had_video and not C.load().get("keep_video"):
        try:
            video.unlink()
        except OSError:
            pass
    best = max(results, key=lambda k: results[k]["top1"]) if results else None
    return f"best: {E.MODELS[best]['label']} ({results[best]['top1']:.0f}% top-1)" if best else "no results"
