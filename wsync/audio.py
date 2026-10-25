"""Audio fingerprints: where in the walkthrough is this sound from?

The same idea as Shazam. A spectrogram's loudest local peaks are paired up,
and each pair (frequency, frequency difference, time gap) becomes a hash.
Hashes survive noise, volume changes and compression, and they come with
exact times — so matching the last few seconds of game audio against the
walkthrough's hashes gives a precise position whenever the sound is the same
in both: cutscene dialogue and music, scripted lines, set pieces.

walkthrough  ffmpeg -> 8 kHz mono -> peaks -> hashes   (audio_fp.npz)
your PC      WASAPI loopback (what the speakers play) -> same -> lookup
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from . import config as C

SR = 8000                 # Hz — speech and music detail up to ~3 kHz is plenty
N_FFT = 1024              # 128 ms windows …
HOP = 256                 # … every 32 ms
FPS = SR / HOP            # 31.25 spectrogram frames per second
F_LO, F_HI = 4, 400       # bins used: ~30 Hz to ~3.1 kHz
NB_F, NB_T = 5, 3         # a peak is the loudest within +-5 bins (~40 Hz), +-3 frames (~0.1 s)
PEAKS_PER_S = 40          # keep the strongest this many per second
FLOOR_DB = 0.0            # a peak must stand this far (log units) above its frame's median
FAN = 6                   # pair each peak with the next FAN in time …
MAX_DT = 63               # … up to ~2 s later …
MAX_DF = 63               # … and within +-63 bins
MAX_OCC = 300             # hashes this common say nothing
CONFIDENT = 12            # aligned hashes for a confident match (unrelated audio: <= ~8) …
CONFIDENT_RATIO = 2.0     # … and at least this many times the runner-up elsewhere

_WINDOW = np.hanning(N_FFT).astype(np.float32)


# ---------------------------------------------------------------------- #
# fingerprinting
# ---------------------------------------------------------------------- #
def _spectrogram(x: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(x) - N_FFT) // HOP)
    if len(x) < N_FFT:
        return np.zeros((0, F_HI - F_LO), np.float32)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * _WINDOW
    mag = np.abs(np.fft.rfft(frames, axis=1))[:, F_LO:F_HI]
    return np.log(mag + 1e-6).astype(np.float32)


def _sliding_max(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="constant", constant_values=-np.inf)
    v = np.lib.stride_tricks.sliding_window_view(p, 2 * r + 1, axis=axis)
    return v.max(axis=-1)


def peaks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(frame index, bin index) of spectral peaks, strongest PEAKS_PER_S per second."""
    s = _spectrogram(x)
    if s.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    m = _sliding_max(_sliding_max(s, NB_F, 1), NB_T, 0)
    floor = np.median(s, axis=1, keepdims=True) + FLOOR_DB   # above each frame's typical level
    t, f = np.nonzero((s == m) & (s > floor))
    if len(t) == 0:
        return t, f
    val = s[t, f]
    sec = (t / FPS).astype(np.int64)
    # strongest per second: sort by (second, -value), keep the first PEAKS_PER_S of each
    order = np.lexsort((-val, sec))
    t, f, sec = t[order], f[order], sec[order]
    first = np.r_[0, np.flatnonzero(np.diff(sec)) + 1]
    rank = np.arange(len(sec)) - np.repeat(first, np.diff(np.r_[first, len(sec)]))
    keep = rank < PEAKS_PER_S
    t, f = t[keep], f[keep]
    order = np.lexsort((f, t))
    return t[order], f[order]


def hashes(t: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair each peak with the next FAN peaks: hash = f1 | (df+63) << 9 | dt << 16."""
    hs, ts = [], []
    for k in range(1, FAN + 1):
        if len(t) <= k:
            break
        dt = t[k:] - t[:-k]
        df = f[k:] - f[:-k]
        ok = (dt >= 1) & (dt <= MAX_DT) & (np.abs(df) <= MAX_DF)
        h = f[:-k][ok] | ((df[ok] + MAX_DF) << 9) | (dt[ok] << 16)
        hs.append(h.astype(np.uint32))
        ts.append(t[:-k][ok].astype(np.uint32))
    if not hs:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
    return np.concatenate(hs), np.concatenate(ts)


# ---------------------------------------------------------------------- #
# the walkthrough's index
# ---------------------------------------------------------------------- #
class Index:
    def __init__(self, h: np.ndarray, t: np.ndarray):
        order = np.argsort(h, kind="stable")
        self.h = h[order]
        self.t = t[order]

    @classmethod
    def load(cls, path: Path) -> "Index":
        z = np.load(path)
        idx = cls.__new__(cls)
        idx.h, idx.t = z["h"], z["t"]
        return idx

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, h=self.h, t=self.t, sr=SR, hop=HOP)
        tmp.replace(path)

    def match(self, x: np.ndarray) -> tuple[float, int, int]:
        """x: the last few seconds of game audio (8 kHz mono float).
        Returns (walkthrough seconds at the END of x, aligned hashes, runner-up)."""
        t, f = peaks(x)
        hq, tq = hashes(t, f)
        if len(hq) == 0:
            return -1.0, 0, 0
        lo = np.searchsorted(self.h, hq, "left")
        hi = np.searchsorted(self.h, hq, "right")
        cnt = hi - lo
        ok = (cnt > 0) & (cnt <= MAX_OCC)
        if not ok.any():
            return -1.0, 0, 0
        lo, cnt, tq = lo[ok], cnt[ok], tq[ok].astype(np.int64)
        # every (index entry - query time) offset; the true one piles up
        rep = np.repeat(np.arange(len(lo)), cnt)
        pos = np.repeat(lo - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
        off = self.t[pos].astype(np.int64) - tq[rep]
        off = off[off >= 0]
        if len(off) == 0:
            return -1.0, 0, 0
        bins = off // 2
        counts = np.bincount(bins)
        b = int(np.argmax(counts))
        c1 = int(counts[b])
        counts[max(0, b - 5):b + 6] = 0          # runner-up: elsewhere
        c2 = int(counts.max()) if len(counts) else 0
        end = (b * 2 + 1) / FPS + len(x) / SR
        return end, c1, c2


def confident(c1: int, c2: int) -> bool:
    return c1 >= CONFIDENT and c1 >= CONFIDENT_RATIO * max(1, c2)


def decode(path: Path, ffmpeg: str, no_window: dict):
    """Yield the file's audio as 8 kHz mono float32, about a minute at a time."""
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **no_window)
    try:
        while True:
            buf = proc.stdout.read(SR * 60 * 2)
            if not buf:
                break
            yield np.frombuffer(buf[:len(buf) // 2 * 2], np.int16).astype(np.float32) / 32768.0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def build_index(path: Path, duration: float, ffmpeg: str, no_window: dict,
                progress=None, cancel: threading.Event | None = None) -> Index:
    """Fingerprint a whole walkthrough. Peaks are found block by block (with
    overlap so the neighbourhood test is exact at the seams), then paired."""
    overlap = N_FFT + 2 * NB_T * HOP
    carry = np.zeros(0, np.float32)
    carry_start = 0                 # sample index of carry[0]
    all_t, all_f = [], []
    done_until = 0                  # frames before this are final
    seen = 0
    for chunk in decode(path, ffmpeg, no_window):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled")
        x = np.concatenate([carry, chunk])
        t, f = peaks(x)
        t = t + carry_start // HOP
        last_final = (carry_start + len(x) - overlap) // HOP
        keep = (t >= done_until) & (t < last_final)
        all_t.append(t[keep])
        all_f.append(f[keep])
        done_until = last_final
        cut = max(0, len(x) - overlap)
        cut -= cut % HOP
        carry = x[cut:]
        carry_start += cut
        seen += len(chunk)
        if progress and duration:
            progress(min(1.0, seen / SR / duration))
    if len(carry):
        t, f = peaks(carry)
        t = t + carry_start // HOP
        keep = t >= done_until
        all_t.append(t[keep])
        all_f.append(f[keep])
    t = np.concatenate(all_t) if all_t else np.zeros(0, np.int64)
    f = np.concatenate(all_f) if all_f else np.zeros(0, np.int64)
    order = np.lexsort((f, t))
    h, ht = hashes(t[order], f[order])
    return Index(h, ht)


# ---------------------------------------------------------------------- #
# what your PC is playing
# ---------------------------------------------------------------------- #
def _lowpass(rate: int) -> np.ndarray:
    """Windowed-sinc low-pass at 3.6 kHz for resampling `rate` -> 8 kHz."""
    n = 63
    fc = 3600 / rate
    k = np.arange(n) - (n - 1) / 2
    h = np.sinc(2 * fc * k) * np.hamming(n)
    return (h / h.sum()).astype(np.float32)


class Resampler:
    """Streaming `rate` -> 8 kHz: low-pass, then keep every (rate/8000)th sample."""

    def __init__(self, rate: int):
        self.h = _lowpass(rate)
        self.step = rate // SR
        self.tail = np.zeros(len(self.h) - 1, np.float32)
        self.phase = 0

    def __call__(self, mono: np.ndarray) -> np.ndarray:
        x = np.r_[self.tail, mono.astype(np.float32)]
        y = np.convolve(x, self.h, mode="valid")
        self.tail = x[-(len(self.h) - 1):]
        out = y[self.phase::self.step]
        self.phase = (self.phase - len(y)) % self.step
        return out


class Capture:
    """Records the default speaker's output (WASAPI loopback) into a rolling
    buffer of 8 kHz mono audio. Runs on its own thread; restarts itself if
    the output device changes. Optionally appends everything to a .pcm file
    (int16) with time marks, so sessions can be replayed offline."""

    RATE = 48000
    KEEP_S = 20

    def __init__(self):
        self.buf: deque = deque()        # (wall time at end, samples)
        self.lock = threading.Lock()
        self.ok = False
        self.error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.pcm_path: Path | None = None
        self.marks: list[tuple[float, int]] = []      # (wall time, samples written so far)
        self._written = 0
        # Only saved while the game itself is in focus (set by the tracker),
        # so voice chat or a video you tab out to never ends up on disk.
        self.writing = False

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="audio-capture")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def record_to(self, path: Path | None) -> None:
        with self.lock:
            self.pcm_path = path
            self.marks = []
            self._written = 0

    def _run(self) -> None:
        if sys.platform.startswith("win"):
            try:  # WASAPI is COM: every thread that uses it must initialise COM itself
                import ctypes
                ctypes.windll.ole32.CoInitializeEx(None, 0)     # COINIT_MULTITHREADED
            except Exception:
                pass
        while not self._stop.is_set():
            try:
                import soundcard as sc
                spk = sc.default_speaker()
                mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
                resample = Resampler(self.RATE)
                with mic.recorder(samplerate=self.RATE, blocksize=self.RATE // 10) as rec:
                    self.ok, self.error = True, ""
                    C.log(f"audio: capturing '{spk.name}'")
                    while not self._stop.is_set():
                        data = rec.record(numframes=self.RATE // 5)       # 0.2 s
                        mono = data.mean(axis=1).astype(np.float32) if data.ndim == 2 \
                            else data.astype(np.float32)
                        self._push(resample(mono))
            except ImportError:
                self.ok, self.error = False, "soundcard not installed"
                return
            except Exception as e:  # device changed, unplugged, …
                self.ok, self.error = False, str(e)[:120]
                C.log(f"audio capture error: {e!r}")
                self._stop.wait(5)

    def _push(self, x: np.ndarray) -> None:
        now = time.time()
        with self.lock:
            self.buf.append((now, x))
            total = sum(len(b) for _, b in self.buf)
            while self.buf and total - len(self.buf[0][1]) >= self.KEEP_S * SR:
                total -= len(self.buf.popleft()[1])
            if self.pcm_path is not None and self.writing:
                try:
                    with self.pcm_path.open("ab") as fh:
                        fh.write(np.clip(x * 32767, -32768, 32767).astype(np.int16).tobytes())
                    self._written += len(x)
                    self.marks.append((now, self._written))
                except OSError:
                    pass

    def last(self, seconds: float) -> np.ndarray | None:
        """The most recent `seconds` of audio, or None if it isn't current."""
        with self.lock:
            if not self.buf or time.time() - self.buf[-1][0] > 1.0:
                return None
            x = np.concatenate([b for _, b in self.buf])
        n = int(seconds * SR)
        return x[-n:] if len(x) >= n else None


_capture: Capture | None = None


def capture() -> Capture:
    global _capture
    if _capture is None:
        _capture = Capture()
        _capture.start()
    return _capture


def ensure_installed() -> bool:
    """Install the loopback-capture library into this environment if it's
    missing (an install made before audio support existed won't have it)."""
    import importlib.util
    # look, don't import: importing it here would set up COM on the wrong thread
    if importlib.util.find_spec("soundcard") is not None:
        return True
    exe = Path(sys.executable)
    py = exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" else exe
    flags = {"creationflags": 0x08000000} if sys.platform.startswith("win") else {}
    C.log("audio: installing soundcard…")
    r = subprocess.run([str(py), "-m", "pip", "install", "--disable-pip-version-check",
                        "soundcard"], capture_output=True, text=True, **flags)
    C.log("audio: soundcard " + ("installed" if r.returncode == 0
                                 else "install failed: " + r.stderr.strip()[-300:]))
    return r.returncode == 0
