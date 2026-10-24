"""Where am I in the walkthrough? — a histogram filter over video time.

No Qt and no torch here, only numpy, so it can be tested on its own.

The video is cut into states, one per second. The belief is a probability for
each state. Every few seconds of play two things happen:

predict(dt)   time passed: shift the belief forward by somewhere between 0
              and ~1.3x the elapsed time (players are usually slower than
              the walkthrough, sometimes a bit faster), with a sliver of
              probability spread everywhere so a wrong lock can recover.
observe(q)    a new screenshot embedding: compare it with every video frame
              and multiply the belief by how well each one matches. Matches
              are judged by z-score, so a frame that looks like half the game
              (a dark corridor, a menu) barely moves anything, while a
              distinctive cutscene shot pulls hard.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

WEIGHT_VERSION = 1   # bump when distinctiveness() changes, to recompute caches


@dataclass
class Estimate:
    position: float      # seconds into the video
    confidence: float    # probability mass within +-CONF_WINDOW of position
    spread: float        # rough +- seconds of the main peak
    best_match: float    # seconds of the single best-matching frame last time
    zmax: float          # how distinctive the last screenshot was


class Tracker:
    RATE_MAX = 1.3        # fastest believable pace vs. the video
    BACK_S = 3            # allow this much backward slop per step
    TELEPORT = 3e-4       # per-step chance you're not where we think (skipped ahead, …)
    FAR_SHARE = 0.03      # of that, the part spread over the whole video — only once lost
    LOST_CONF = 0.3       # below this confidence we count as lost
    NEAR_FWD_S = 600      # nearby jumps: forward on a ~10 min scale, at most 45 min …
    NEAR_FWD_MAX = 2700
    NEAR_BACK_S = 300     # … and back on a ~5 min scale, at most 30 min, so an
    NEAR_BACK_MAX = 1800  #   estimate that ran ahead can always come back
    SNAP_S = 90           # the shown position follows moves this small at once …
    HOLD_NEAR_S = 8       # … a jump of up to 5 min must hold for this long first …
    HOLD_FAR_S = 24       # … and a bigger one this long (no 20-second flickers)
    PACE_Z = 3.8          # matches at least this distinctive …
    PACE_NEAR = 600       # … and within 10 min of the estimate …
    PACE_WINDOW = 300     # … over the last 5 min of play tell us your pace …
    PACE_MIN_PTS = 8      # … once there are enough of them
    PACE_SHARE = 0.65     # how much the learned pace shapes each step
    Z_FLOOR = 2.0         # z below this tells us nothing
    Z_FULL = 5.0          # z at or above this is trusted fully
    ALPHA = 1.0           # sharpness of the match likelihood
    OBS_FLOOR = 0.1       # no single observation can veto a state
    CONF_WINDOW = 45      # seconds
    SAME_SHOT = 0.92      # a screenshot this similar to the last one counted adds nothing new …
    NEW_SHOT = 0.77       # … and one below this counts in full

    def __init__(self, emb: np.ndarray, step: float = 1.0, weight: np.ndarray | None = None):
        """emb: (N, D) float32, centred and L2-normalised per row.
        weight: (N,) 0..1, how unique each frame is (see distinctiveness());
        frames that recur all over the video (fades, loading, menus) count less."""
        self.emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.n = int(self.emb.shape[0])
        self.step = float(step)
        self.weight = None if weight is None else np.asarray(weight, dtype=np.float32)
        self.pos_axis = np.arange(self.n, dtype=np.float64) * self.step
        self.belief = np.full(self.n, 1.0 / self.n)
        self.last_best = 0.0
        self.last_z = 0.0
        self._last_q: np.ndarray | None = None
        self._shown: int | None = None       # index on display (see settle())
        self._pending: int | None = None
        self._pending_age = 0.0
        self._clock = 0.0                    # seconds of active play
        self._marks: deque = deque()         # (clock, matched video time)
        self.pace: float | None = None       # video seconds per second of play

    # ------------------------------------------------------------------ #
    def reset(self, seconds: float, sigma: float = 600.0) -> None:
        """Start believing we're around `seconds`, give or take `sigma`."""
        idx = np.arange(self.n) * self.step
        sigma = max(5.0, float(sigma))
        b = np.exp(-0.5 * ((idx - float(seconds)) / sigma) ** 2)
        b += 1e-6 * b.max()
        self.belief = b / b.sum()
        self._shown = self._pending = None
        self._marks.clear()
        self.pace = None

    def shift(self, seconds: float) -> None:
        """Move the whole belief by `seconds` (a manual nudge)."""
        k = int(round(seconds / self.step))
        if k == 0:
            return
        b = np.zeros_like(self.belief)
        if k > 0:
            b[k:] = self.belief[:-k] if k < self.n else 0
            b[-1] += self.belief[max(0, self.n - k):].sum()
        else:
            k = -k
            b[:-k] = self.belief[k:] if k < self.n else 0
            b[0] += self.belief[:min(k, self.n)].sum()
        self.belief = b / b.sum()
        self._shown = self._pending = None
        self._marks.clear()
        self.pace = None

    def predict(self, dt: float) -> None:
        dt = max(0.0, min(float(dt), 30.0))
        if dt <= 0:
            return
        steps = dt / self.step
        fwd = int(math.ceil(steps * self.RATE_MAX)) + 1
        back = int(math.ceil(self.BACK_S / self.step))
        shifts = np.arange(-back, fwd + 1)
        w = np.where(shifts < 0, 0.04, 1.0).astype(float)
        over = shifts > steps
        if over.any():
            # taper from 1 at the real-time pace to 0.15 at RATE_MAX
            span = max(1e-6, steps * (self.RATE_MAX - 1.0) + 1)
            w[over] = np.clip(1.0 - 0.85 * (shifts[over] - steps) / span, 0.15, 1.0)
        w /= w.sum()
        self._clock += dt
        if self.pace is not None:
            # Lean on the pace the recent matches show — ~0 while you're stuck
            # on a puzzle the walkthrough player solved in a minute.
            sd = 0.25 * steps + 1.0
            wp = np.exp(-0.5 * ((shifts - self.pace * steps) / sd) ** 2)
            if wp.sum() > 0:
                w = (1 - self.PACE_SHARE) * w + self.PACE_SHARE * wp / wp.sum()

        full = np.convolve(self.belief, w)          # index i+j  <-  shift j-back
        b = full[back:back + self.n].copy()
        b[0] += full[:back].sum()                    # fell off the start
        b[-1] += full[back + self.n:].sum()          # ran past the end
        b = (1.0 - self.TELEPORT) * b + self.TELEPORT * self._jump_prior(b)
        self.belief = b / b.sum()

    def _jump_prior(self, b: np.ndarray) -> np.ndarray:
        """Where you'd be if you're not where we think: usually a bit further on
        (skipped a cutscene or an optional area), rarely anywhere at all."""
        i = int(np.argmax(b))
        cw = int(round(self.CONF_WINDOW / self.step))
        conf = float(b[max(0, i - cw):i + cw + 1].sum() / b.sum())
        d = self.pos_axis - i * self.step
        near = np.where(d >= 0, np.exp(-d / self.NEAR_FWD_S), np.exp(d / self.NEAR_BACK_S))
        near[(d > self.NEAR_FWD_MAX) | (d < -self.NEAR_BACK_MAX)] = 0.0
        near /= near.sum()
        # A confident tracker never jumps across the video (30% -> 50% because
        # a menu looked like something hours ahead); a lost one may.
        far = self.FAR_SHARE * max(0.0, min(1.0, (self.LOST_CONF - conf) / self.LOST_CONF))
        return (1.0 - far) * near + far / self.n

    def similarities(self, q: np.ndarray) -> np.ndarray:
        return self.emb @ np.asarray(q, dtype=np.float32).reshape(-1)

    def observe(self, q: np.ndarray) -> float:
        """Fold in one screenshot embedding (centred + normalised). Returns zmax."""
        sims = self.similarities(q)
        mu, sd = float(sims.mean()), float(sims.std()) + 1e-6
        z = (sims - mu) / sd
        if self.weight is not None:
            z = z * self.weight
        best = int(np.argmax(z))
        zmax = float(z[best])
        self.last_best = best * self.step
        self.last_z = zmax
        self._learn_pace(best * self.step, zmax)
        trust = (zmax - self.Z_FLOOR) / (self.Z_FULL - self.Z_FLOOR)
        trust = max(0.0, min(1.0, trust))
        # A screen that isn't changing (a menu, the map, standing still) is one
        # piece of evidence, not one every two seconds — otherwise half a
        # minute in a menu adds up to "certain" about wherever the menu matched.
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        if self._last_q is not None:
            same = float(q @ self._last_q)
            trust *= max(0.0, min(1.0, (self.SAME_SHOT - same) / (self.SAME_SHOT - self.NEW_SHOT)))
        if trust <= 0:
            return zmax
        if trust >= 0.5 or self._last_q is None:
            self._last_q = q
        like = np.exp(self.ALPHA * trust * (z - zmax)) + self.OBS_FLOOR
        b = self.belief * like
        s = b.sum()
        if s > 0 and np.isfinite(s):
            self.belief = b / s
        return zmax

    # ------------------------------------------------------------------ #
    def _learn_pace(self, matched: float, z: float) -> None:
        """Fit how fast the confident nearby matches move through the video."""
        here = self._shown if self._shown is not None else self._peak()
        if z >= self.PACE_Z and abs(matched - here * self.step) <= self.PACE_NEAR:
            self._marks.append((self._clock, matched))
        while self._marks and self._marks[0][0] < self._clock - self.PACE_WINDOW:
            self._marks.popleft()
        if len(self._marks) < self.PACE_MIN_PTS:
            self.pace = None
            return
        c = np.array(self._marks)
        if c[-1, 0] - c[0, 0] < self.PACE_WINDOW * 0.4:
            self.pace = None
            return
        # Theil-Sen: median of pairwise slopes, unbothered by stray matches
        i, j = np.triu_indices(len(c), k=1)
        dtm = c[j, 0] - c[i, 0]
        ok = dtm > 20
        if ok.sum() < 10:
            self.pace = None
            return
        slope = float(np.median((c[j, 1] - c[i, 1])[ok] / dtm[ok]))
        self.pace = max(0.0, min(self.RATE_MAX, slope))

    def _peak(self) -> int:
        win = max(1, int(round(10 / self.step)))
        smooth = np.convolve(self.belief, np.ones(2 * win + 1), mode="same")
        return int(np.argmax(smooth))

    def settle(self, dt: float) -> None:
        """Call once per screenshot, after predict/observe: decides whether the
        shown position follows the belief's peak. Small moves follow at once;
        a jump has to stay the peak for a while before it's shown."""
        i = self._peak()
        if self._shown is None or abs(i - self._shown) * self.step <= self.SNAP_S:
            self._shown, self._pending, self._pending_age = i, None, 0.0
            return
        if self._pending is not None and abs(i - self._pending) * self.step <= self.SNAP_S:
            self._pending_age += max(0.0, float(dt))
        else:
            self._pending_age = 0.0
        self._pending = i
        hold = self.HOLD_NEAR_S if abs(i - self._shown) * self.step <= 300 else self.HOLD_FAR_S
        if self._pending_age >= hold:
            self._shown, self._pending, self._pending_age = i, None, 0.0

    def estimate(self) -> Estimate:
        i = self._shown if self._shown is not None else self._peak()
        cw = int(round(self.CONF_WINDOW / self.step))
        lo, hi = max(0, i - cw), min(self.n, i + cw + 1)
        conf = float(self.belief[lo:hi].sum())
        # spread: std of the belief within +-10 min of the position
        sw = int(round(600 / self.step))
        lo2, hi2 = max(0, i - sw), min(self.n, i + sw + 1)
        seg = self.belief[lo2:hi2]
        t = np.arange(lo2, hi2) * self.step
        m = seg.sum()
        spread = float(np.sqrt(((t - i * self.step) ** 2 * seg).sum() / m)) if m > 0 else 0.0
        return Estimate(i * self.step, conf, spread, self.last_best, self.last_z)


def distinctiveness(emb: np.ndarray, gap_s: int = 600, block: int = 2048,
                    matmul=None) -> np.ndarray:
    """For every frame: how unique it is across the video, 0..1.

    A frame whose near-twin turns up more than `gap_s` away (a fade to white,
    a loading screen, a menu, the same corridor revisited) is a poor landmark,
    so a screenshot matching it shouldn't drag the tracker there."""
    n = len(emb)
    far = np.empty(n, dtype=np.float32)
    mm = matmul or (lambda a, b: a @ b.T)
    for s in range(0, n, block):
        sim = np.asarray(mm(emb[s:s + block], emb), dtype=np.float32)
        rows = np.arange(sim.shape[0])
        for k in rows:
            i = s + k
            sim[k, max(0, i - gap_s):min(n, i + gap_s + 1)] = -1.0
        far[s:s + block] = sim.max(axis=1)
    lo = max(0.65, float(np.percentile(far, 85)))
    hi = lo + 0.15
    return (1.0 - np.clip((far - lo) / (hi - lo), 0.0, 1.0)).astype(np.float32)


def prepare(raw: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Centre embeddings on the video's mean look and re-normalise."""
    x = raw.astype(np.float32) - mean.astype(np.float32)
    x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x
