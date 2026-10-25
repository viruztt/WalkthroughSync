"""Settings (config.json next to the app) plus small shared helpers."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_DIR / "config.json"
DATA_DIR = APP_DIR / "data"
GAMES_DIR = DATA_DIR / "games"
MODELS_DIR = APP_DIR / "models"
LOG_PATH = DATA_DIR / "log.txt"

DEFAULTS = {
    "dock": "right",           # "right" | "left" | "none"
    "valign": "top",
    "margin": 12,
    "font_size": 10,
    "opacity": 0.93,
    "window_layer": "top",     # "top" | "normal" | "bottom"
    "layout": "full",          # "full" card | "minimal" (just "12.34%")
    "minimal_decimals": True,  # minimal mode: 12.34% (True) or 12% (False)
    "show_thumb": True,        # show the matched walkthrough frame
    "time_text": "position",   # line under the name: "position" | "remaining" | "off"
    "percent_text": False,     # … plus the percentage on that line
    "interval": 2.0,           # seconds between screenshots
    "current_game": "",
    "keep_video": False,       # keep the downloaded walkthrough after indexing
    "image_model": "",         # for newly analysed walkthroughs; "" = recommended (embedder.MODELS)
    "hide_from_capture": True, # widget invisible to screenshots / recordings
    "record_sessions": True,   # keep what the tracker saw, for tuning (~3 MB/hour)
    "position": {"x": None, "y": None},
}


def load() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(user, dict):
                cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
                old = user.get("progress_text")   # setting from an earlier version
                if old in ("time", "percent", "both") and "time_text" not in user:
                    cfg["time_text"] = "off" if old == "percent" else "position"
                    cfg["percent_text"] = old in ("percent", "both")
        except Exception:
            pass
    if cfg["dock"] not in ("right", "left", "none"):
        cfg["dock"] = "right"
    if cfg["valign"] not in ("top", "center", "bottom"):
        cfg["valign"] = "top"
    if cfg["window_layer"] not in ("top", "normal", "bottom"):
        cfg["window_layer"] = "top"
    if cfg["layout"] not in ("full", "minimal"):
        cfg["layout"] = "full"
    if cfg["time_text"] not in ("position", "remaining", "off"):
        cfg["time_text"] = "position"
    cfg["percent_text"] = bool(cfg["percent_text"])
    cfg["minimal_decimals"] = bool(cfg["minimal_decimals"])
    if not isinstance(cfg.get("position"), dict):
        cfg["position"] = {"x": None, "y": None}
    try:
        cfg["interval"] = min(10.0, max(0.5, float(cfg["interval"])))
    except (TypeError, ValueError):
        cfg["interval"] = 2.0
    return cfg


def save(cfg: dict) -> None:
    write_json(CONFIG_PATH, cfg)


def write_json(path: Path, obj) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------- #
# time formatting / parsing
# ---------------------------------------------------------------------- #
def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "–:––"
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_time(text: str) -> float | None:
    """Accepts 1:30:00, 12:34 (m:ss), 1h30m, 1h 30min, 90m, 5400s, 5400,
    or a YouTube link carrying t=…  Returns seconds or None."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if "youtu" in t or t.startswith("http"):
        q = parse_qs(urlparse(text.strip()).query)
        if "t" in q:
            return parse_time(q["t"][0])
        return None
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return float(t)
    if re.fullmatch(r"\d+(:\d{1,2}){1,2}", t):
        parts = [int(p) for p in t.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    total, found = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?|m|min|mins|minutes?|s|sec|secs|seconds?)\b", t):
        found = True
        n = float(num)
        total += n * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1)
    return total if found else None


def video_id(url: str) -> str:
    u = urlparse(url.strip())
    if u.hostname and "youtu.be" in u.hostname:
        return u.path.strip("/").split("/")[0]
    q = parse_qs(u.query)
    if "v" in q:
        return q["v"][0]
    m = re.search(r"/(?:shorts|live|embed)/([\w-]{6,})", u.path)
    return m.group(1) if m else ""


def clean_title(title: str) -> str:
    """'RESONANCE A PLAGUE TALE LEGACY Gameplay Walkthrough FULL GAME [4K]' ->
    'Resonance A Plague Tale Legacy'."""
    t = re.split(r"\b(gameplay|walkthrough|full game|longplay|playthrough|let'?s play|"
                 r"no commentary|part \d+)\b", title, maxsplit=1, flags=re.I)[0]
    t = re.sub(r"[\[\(].*?[\]\)]", "", t)
    t = re.sub(r"\s+[-|–:]+\s*$", "", t.strip()).strip(" -|–:")
    if not t:
        t = title.strip()
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        small = {"the", "of", "and", "or", "in", "on", "to", "at", "for"}
        words = t.lower().split()
        t = " ".join(w if (i and w in small) else w[:1].upper() + w[1:]
                     for i, w in enumerate(words))
        t = re.sub(r"\b(i{1,3}|iv|vi{0,3}|ix|x)\b", lambda m: m.group(1).upper(), t,
                   flags=re.I)
    return t[:80]


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:48] or "game"


# ---------------------------------------------------------------------- #
# log
# ---------------------------------------------------------------------- #
def log(msg: str) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > 2_000_000:
            old = LOG_PATH.with_suffix(".old.txt")
            try:
                os.replace(LOG_PATH, old)
            except OSError:
                pass
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except Exception:
        pass
