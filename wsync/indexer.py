"""Turn a YouTube walkthrough into a searchable index.

1. yt-dlp reads the title and length, then downloads a small (360p,
   video-only) copy — plenty for recognising scenes and quick to decode.
2. ffmpeg decodes it in parallel chunks at one frame per second, 320x180.
3. Every frame is embedded on the GPU; every 2nd second also becomes a tiny
   JPEG for the widget's "matched frame" preview.
4. A replay self-check simulates a slow, noisy playthrough from those thumbs
   and writes how close the tracker stayed (replay.txt).
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from . import config as C
from . import embedder as E
from . import games as G
from .tracker import Tracker, calibration, prepare

W, H = 320, 180
FRAME_BYTES = W * H * 3
FORMAT = ("bv*[height<=360][vcodec^=avc1]/bv*[height<=360]/bv*[height<=480]/"
          "b[height<=480]/worst")

Progress = Callable[[str, float, str], None]


class Cancelled(Exception):
    pass


def _no_window() -> dict:
    if sys.platform.startswith("win"):
        return {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}


def _python() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def _js_runtime_args() -> list[str]:
    """yt-dlp needs a JavaScript runtime for YouTube; point it at Deno if it
    lives in the usual per-user place rather than on PATH."""
    for p in (Path.home() / ".deno" / "bin" / "deno.exe", Path.home() / ".deno" / "bin" / "deno"):
        if p.exists():
            return ["--js-runtimes", f"deno:{p}"]
    return []


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _ytdlp(args: list[str]) -> list[str]:
    return [_python(), "-m", "yt_dlp", "--no-warnings", "--no-playlist",
            *_js_runtime_args(), *args]


# ---------------------------------------------------------------------- #
def fetch_info(url: str) -> dict:
    r = subprocess.run(_ytdlp(["-J", url]), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180, **_no_window())
    if r.returncode != 0:
        raise RuntimeError("Could not read the video: " + _last_error(r.stderr))
    return json.loads(r.stdout)


def _last_error(text: str) -> str:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    errs = [l for l in lines if "ERROR" in l]
    return (errs or lines or ["unknown error"])[-1][:300]


def _complete(video: Path, expected: float) -> bool:
    """Is an earlier download whole? Its length must match and the last
    seconds must decode (a cut-off file can still report the full length)."""
    try:
        dur = probe_duration(video)
        if not dur or (expected and abs(dur - expected) > max(5.0, expected * 0.01)):
            return False
        r = subprocess.run([ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
                            "-ss", str(max(0.0, dur - 20)), "-i", str(video),
                            "-an", "-frames:v", "5", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=60, **_no_window())
        return r.returncode == 0 and not r.stderr.strip()
    except Exception:
        return False


def download(url: str, folder: Path, progress: Progress, cancel: threading.Event,
             expected: float = 0.0) -> Path:
    for old in folder.glob("video.*"):
        if old.suffix in (".mp4", ".webm", ".mkv") and _complete(old, expected):
            C.log(f"reusing earlier download {old.name}")
            return old
    for old in folder.glob("video.*"):
        try:
            old.unlink()
        except OSError:
            pass
    cmd = _ytdlp(["-f", FORMAT, "-o", str(folder / "video.%(ext)s"), "--newline",
                  "--concurrent-fragments", "4", "--no-part",
                  "--progress-template",
                  "download:PROG %(progress.downloaded_bytes)s %(progress.total_bytes)s "
                  "%(progress.total_bytes_estimate)s", url])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", **_no_window())
    tail: list[str] = []
    try:
        for line in proc.stdout:
            if cancel.is_set():
                proc.kill()
                raise Cancelled()
            line = line.strip()
            if line.startswith("PROG"):
                parts = line.split()
                try:
                    done = float(parts[1])
                    total = next((float(p) for p in parts[2:] if p not in ("NA", "None")), 0)
                except (ValueError, IndexError):
                    continue
                frac = done / total if total else 0
                progress("download", min(frac, 1.0),
                         f"Downloading walkthrough · {done / 1e6:,.0f} of {total / 1e6:,.0f} MB")
            elif line:
                tail = (tail + [line])[-20:]
    finally:
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Download failed: " + _last_error("\n".join(tail)))
    vids = [p for p in folder.glob("video.*") if p.suffix not in (".part", ".ytdl", ".json")]
    if not vids:
        raise RuntimeError("Download finished but no video file was found.")
    return vids[0]


def probe_duration(video: Path) -> float:
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", **_no_window())
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


# ---------------------------------------------------------------------- #
def decode_and_embed(video: Path, duration: float, progress: Progress,
                     cancel: threading.Event, model_key: str | None = None
                     ) -> tuple[np.ndarray, dict[int, bytes]]:
    from PIL import Image

    emb_model = E.get(model_key)
    n = int(math.ceil(duration))
    workers = max(2, min(8, (os.cpu_count() or 4) // 2))
    chunk = int(math.ceil(n / workers))
    dim = None
    emb: np.ndarray | None = None
    filled = np.zeros(n, dtype=bool)
    thumbs: dict[int, bytes] = {}
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def store(indices: list[int], frames: list[np.ndarray]) -> None:
        nonlocal emb, dim
        vecs = emb_model.embed(np.stack(frames))
        with lock:
            if emb is None:
                dim = vecs.shape[1]
                emb = np.zeros((n, dim), dtype=np.float16)
            emb[indices] = vecs.astype(np.float16)
            filled[indices] = True
            done[0] += len(indices)
            d = done[0]
        el = time.time() - t0
        eta = el / max(d, 1) * (n - d)
        progress("analyse", d / n,
                 f"{d:,} / {n:,} frames" + (f" · ~{max(1, round(eta / 60))} min left" if d > 200 else ""))

    def run_chunk(k: int) -> None:
        start = k * chunk
        if start >= n:
            return
        length = min(chunk, n - start)
        cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
               "-ss", str(start), "-t", str(length), "-i", str(video),
               "-an", "-sn", "-threads", "2",
               "-vf", f"fps=1,scale={W}:{H}:flags=area,format=rgb24",
               "-f", "rawvideo", "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                bufsize=FRAME_BYTES * 8, **_no_window())
        idx, batch = [], []
        i = 0
        try:
            while True:
                if cancel.is_set():
                    raise Cancelled()
                buf = proc.stdout.read(FRAME_BYTES)
                if not buf or len(buf) < FRAME_BYTES:
                    break
                g = start + i
                i += 1
                if g >= n or i > length + 1:
                    continue
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
                if g % G.THUMB_EVERY == 0:
                    bio = io.BytesIO()
                    Image.fromarray(frame).resize((160, 90), Image.BILINEAR).save(
                        bio, "JPEG", quality=72)
                    with lock:
                        thumbs[g // G.THUMB_EVERY] = bio.getvalue()
                idx.append(g)
                batch.append(frame)
                if len(batch) >= 48:
                    store(idx, batch)
                    idx, batch = [], []
            if batch:
                store(idx, batch)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    progress("analyse", 0.0, "Analysing frames…")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_chunk, k) for k in range(workers)]
        for f in futs:
            f.result()

    if emb is None or filled.sum() < max(10, n * 0.5):
        raise RuntimeError("Could not decode the walkthrough video.")
    # Fill gaps (a missing frame at a chunk edge) from the nearest neighbour.
    if not filled.all():
        good = np.flatnonzero(filled)
        for i in np.flatnonzero(~filled):
            j = good[np.argmin(np.abs(good - i))]
            emb[i] = emb[j]
    return emb, thumbs


# ---------------------------------------------------------------------- #
def build(game: G.Game, progress: Progress, cancel: threading.Event,
          keep_video: bool = False) -> None:
    url = game.meta["url"]
    key = C.load().get("image_model") or E.DEFAULT_KEY
    if key not in E.MODELS:
        key = E.DEFAULT_KEY
    game.meta["status"] = "indexing"
    game.save_meta()

    progress("info", 0.0, "Reading the video's details…")
    info = fetch_info(url)
    game.meta["title"] = info.get("title", "")
    game.meta["channel"] = info.get("channel") or info.get("uploader") or ""
    game.meta["video_id"] = info.get("id") or game.meta.get("video_id", "")
    game.meta["duration"] = float(info.get("duration") or 0)
    game.meta["chapters"] = [{"t": c.get("start_time", 0), "title": c.get("title", "")}
                             for c in (info.get("chapters") or [])]
    if game.meta.get("name_from_title") or not game.meta.get("name"):
        game.meta["name"] = C.clean_title(game.meta["title"]) or game.slug
    game.save_meta()
    C.log(f"index {game.slug}: '{game.meta['title']}' {game.meta['duration']:.0f}s")

    # Load the model while the download runs.
    model_err: list[BaseException] = []

    def _load():
        try:
            E.get(key)
        except BaseException as e:  # noqa: BLE001
            model_err.append(e)
    loader = threading.Thread(target=_load, daemon=True)
    loader.start()

    video = download(url, game.dir, progress, cancel, game.meta["duration"])
    progress("analyse", 0.0, "Loading the image model…")
    loader.join()
    if model_err:
        raise RuntimeError(f"Could not load the image model: {model_err[0]}")

    duration = probe_duration(video) or game.meta["duration"]
    if not duration:
        raise RuntimeError("Could not tell how long the video is.")
    game.meta["duration"] = round(duration, 2)

    t0 = time.time()
    emb, thumbs = decode_and_embed(video, duration, progress, cancel, key)
    np.save(game.dir / "emb.npy", emb)
    np.save(game.dir / "mean.npy", emb.astype(np.float32).mean(axis=0))
    G.write_thumbs(game.dir, thumbs, int(math.ceil(duration / G.THUMB_EVERY)) + 1)
    C.log(f"index {game.slug}: {len(emb)} frames in {time.time() - t0:.0f}s on {E.get(key).gpu_name}")

    if not keep_video:
        try:
            video.unlink()
        except OSError:
            pass

    game.meta["status"] = "ready"
    game.meta["indexed"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    game.meta["model_key"] = key
    game.meta["model"] = E.MODELS[key]["label"]
    game.save_meta()

    progress("check", 0.0, "Self-check…")
    try:
        summary = replay(game, progress=progress, cancel=cancel)
        C.log(f"replay {game.slug}: {summary}")
    except Cancelled:
        raise
    except Exception as e:  # the index is fine even if the check fails
        C.log(f"replay {game.slug} failed: {e!r}")


# ---------------------------------------------------------------------- #
def _augment(img, rng: random.Random):
    """Make a walkthrough thumb look like a different capture of the same shot."""
    from PIL import Image, ImageEnhance, ImageFilter
    w, h = img.size
    s = rng.uniform(0.86, 1.0)
    cw, ch = int(w * s), int(h * s)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    img = img.crop((x, y, x + cw, y + ch)).resize((W, H), Image.BICUBIC)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    bio = io.BytesIO()
    img.save(bio, "JPEG", quality=rng.randint(40, 85))
    return Image.open(io.BytesIO(bio.getvalue())).convert("RGB")


def replay(game: G.Game, segments: int = 4, minutes: float = 15, seed: int = 7,
           progress: Progress | None = None, cancel: threading.Event | None = None) -> str:
    """Simulate playthroughs: the 'player' moves through the video at a
    changing pace (0-100% of real time, with pauses), screenshots are
    augmented thumbs, and the tracker starts off by up to 5 minutes."""
    from PIL import Image

    raw, mean = game.load_index()
    emb = prepare(raw, mean)
    weight = game.load_weight(emb)
    cal = calibration(raw, emb)
    thumbs = game.thumbs()
    if thumbs is None:
        return "no thumbnails"
    dur = game.duration
    model = E.get(game.model_key)
    rng = random.Random(seed)
    seg_len = minutes * 60
    lines, errs_all, within_all = [], [], []
    for s in range(segments):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        start = rng.uniform(60, max(61, dur - seg_len * 1.1))
        # build the simulated trajectory in 2-second ticks
        t, pos, rate = 0.0, start, rng.uniform(0.4, 1.0)
        truth, imgs = [], []
        while t < seg_len:
            if rng.random() < 1 / 30:
                rate = rng.choice([0.0, rng.uniform(0.3, 1.05)])
            pos = min(dur - 1, pos + 2 * rate)
            t += 2
            truth.append(pos)
            jpg = thumbs.get(pos)
            img = Image.open(io.BytesIO(jpg)).convert("RGB").resize((W, H), Image.BICUBIC)
            imgs.append(np.asarray(_augment(img, rng)))
        qs = []
        for i in range(0, len(imgs), 64):
            qs.append(model.embed(np.stack(imgs[i:i + 64])))
        q = prepare(np.concatenate(qs), mean)

        tr = Tracker(emb, weight=weight, new_shot=cal["new_shot"], same_shot=cal["same_shot"])
        guess = start + rng.uniform(-300, 300)
        tr.reset(guess, 600)
        errs = []
        for i, true in enumerate(truth):
            tr.predict(2.0)
            tr.observe(q[i])
            tr.settle(2.0)
            est = tr.estimate()
            if i >= len(truth) // 3:
                errs.append(abs(est.position - true))
        errs = np.array(errs)
        within = float((errs <= 60).mean())
        errs_all.extend(errs.tolist())
        within_all.append(within)
        lines.append(f"segment {s + 1}: start {C.fmt_time(start)}, guess {C.fmt_time(guess)}, "
                     f"median error {np.median(errs):.0f}s, within 60s {within * 100:.0f}%")
        if progress:
            progress("check", (s + 1) / segments, f"Self-check · {s + 1} / {segments}")
    summary = (f"median error {np.median(errs_all):.0f}s, within 60s "
               f"{np.mean(within_all) * 100:.0f}% of the time")
    (game.dir / "replay.txt").write_text(
        "Replay self-check (simulated slow, noisy playthrough from the walkthrough's own frames)\n"
        + "\n".join(lines) + "\n\noverall: " + summary + "\n", encoding="utf-8")
    game.meta["selfcheck"] = summary
    game.save_meta()
    return summary
