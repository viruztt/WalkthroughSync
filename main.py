"""Entry point: python main.py  (or run.bat)

Flags:
  --list               print every game and how far you are, then exit
  --index URL [TIME]   analyse a walkthrough from the console (TIME = where you are)
  --replay [NAME]      re-run the tracking self-check for a game
  --replay-session [FILE]  re-run the tracker over a recorded play session
  --selftest           check the tracker maths and helpers (no GPU needed), then exit
"""
from __future__ import annotations

import sys


def list_games() -> int:
    from wsync import config as C
    from wsync import games as G
    print(f"Walkthrough Sync  —  {C.GAMES_DIR}\n" + "-" * 64)
    games = G.list_games()
    if not games:
        print("(no games)")
    for g in games:
        pos = g.position or 0.0
        frac = pos / g.duration if g.duration else 0
        bar = "#" * round(frac * 20)
        st = "" if g.ready else f"  [{g.meta.get('status')}]"
        print(f"{g.name[:28]:28} [{bar:20}] {frac * 100:5.1f}%  "
              f"{C.fmt_time(pos)} / {C.fmt_time(g.duration)}{st}")
    return 0


def find_game(name: str):
    from wsync import config as C
    from wsync import games as G
    games = G.list_games()
    if not name:
        cfg = C.load()
        return G.get(cfg.get("current_game", "")) or (games[-1] if games else None)
    key = name.lower()
    for g in games:
        if g.slug == key or g.name.lower() == key:
            return g
    pref = [g for g in games if g.name.lower().startswith(key) or g.slug.startswith(key)]
    return pref[0] if len(pref) == 1 else None


def index_cli(args: list[str]) -> int:
    import threading
    from wsync import config as C
    from wsync import games as G
    from wsync import indexer
    if not args:
        print("usage: main.py --index URL [TIME]")
        return 2
    url = args[0]
    start = C.parse_time(args[1]) if len(args) > 1 else 0.0
    g = G.create("", url, start or 0.0)
    last = [""]

    def prog(stage, frac, text):
        line = f"{text}"
        if line != last[0]:
            print("\r" + line[:100].ljust(100), end="", flush=True)
            last[0] = line
    try:
        indexer.build(g, prog, threading.Event(), keep_video=C.load().get("keep_video", False))
    except Exception as e:
        print(f"\nFailed: {e}")
        return 1
    cfg = C.load()
    cfg["current_game"] = g.slug
    C.save(cfg)
    print(f"\nReady: {g.name} ({C.fmt_time(g.duration)}). Self-check: {g.meta.get('selfcheck')}")
    return 0


def replay_cli(args: list[str]) -> int:
    from wsync import indexer
    g = find_game(" ".join(args))
    if g is None or not g.ready:
        print("No analysed game matches. Try --list.")
        return 1
    print(f"Self-check for {g.name}…")
    print(indexer.replay(g, progress=lambda s, f, t: print("  " + t)))
    print((g.dir / "replay.txt").read_text(encoding="utf-8"))
    return 0


def replay_session_cli(args: list[str]) -> int:
    """Re-run the current tracker over a recorded session and compare."""
    from pathlib import Path

    from wsync import config as C
    from wsync import sessions as S
    from wsync.tracker import PauseDetector, Tracker, prepare
    g = find_game("")
    if args and Path(args[0]).exists():
        path = Path(args[0])
        g = find_game(path.parent.parent.name) or g
    else:
        files = sorted((g.dir / "sessions").glob("*.npz")) if g else []
        if not files:
            print("No recorded sessions yet.")
            return 1
        path = files[-1]
    raw, mean = g.load_index()
    emb = prepare(raw, mean)
    r = S.replay(path, Tracker(emb, weight=g.load_weight(emb)), mean, prepare,
                 pause=PauseDetector())
    t0 = r["t"][0] if len(r["t"]) else 0
    print(f"{path.name}: {len(r['t'])} screenshots")
    print(" minute   recorded   replayed   conf")
    last = -1
    for t, a, b, c in zip(r["t"], r["recorded"], r["replayed"], r["confidence"]):
        m = int((t - t0) // 60)
        if m != last:
            last = m
            flag = "  <-- differs" if abs(a - b) > 120 else ""
            print(f"{m:6d}   {C.fmt_time(a):>8}   {C.fmt_time(b):>8}   {c:4.2f}{flag}")
    return 0


def selftest() -> int:
    import tempfile
    from pathlib import Path

    import numpy as np

    from wsync import config as C
    from wsync.tracker import PauseDetector, Tracker, distinctiveness, prepare

    ok = True

    def check(label, cond, got=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"   -> {got}" if got != "" else ""))

    print("Walkthrough Sync — self-test\n" + "-" * 50)
    check("parse 1:30:00", C.parse_time("1:30:00") == 5400)
    check("parse 1h 30min", C.parse_time("1h 30min") == 5400)
    check("parse 90m", C.parse_time("90m") == 5400)
    check("parse 12:34 (m:ss)", C.parse_time("12:34") == 754)
    check("parse youtube link", C.parse_time(
        "https://www.youtube.com/watch?v=DCqSCazDv64&t=5400s") == 5400)
    check("fmt", C.fmt_time(5405) == "1:30:05" and C.fmt_time(65) == "1:05")
    check("video id", C.video_id("https://youtu.be/DCqSCazDv64?t=3") == "DCqSCazDv64"
          and C.video_id("https://www.youtube.com/watch?v=DCqSCazDv64") == "DCqSCazDv64")
    t = C.clean_title("RESONANCE A PLAGUE TALE LEGACY Gameplay Walkthrough FULL GAME "
                      "[4K 60FPS PC ULTRA] - No Commentary")
    check("clean title", t == "Resonance A Plague Tale Legacy", t)

    # Synthetic walkthrough: scenes of 15 s - 4 min, each a slowly drifting
    # shot; scenes in the same game area share a look (like real levels).
    rng = np.random.default_rng(3)
    n, d = 8000, 64
    areas = rng.normal(size=(12, d))
    parts, total = [], 0
    while total < n:
        ln = int(rng.integers(15, 240))
        base = areas[rng.integers(12)] + rng.normal(size=d) * 0.8
        drift = np.cumsum(rng.normal(size=(ln, d)) * 0.08, axis=0)
        parts.append(base + drift + rng.normal(size=(ln, d)) * 0.25)
        total += ln
    raw = np.concatenate(parts)[:n].astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    mean = raw.mean(axis=0)
    emb = prepare(raw, mean)

    def play(start, guess, rate, ticks, noise):
        tr = Tracker(emb)
        tr.reset(guess, 600)
        pos = start
        errs = []
        for i in range(ticks):
            pos = min(n - 1, pos + 2 * rate)
            tr.predict(2.0)
            q = raw[int(pos)] + rng.normal(size=d).astype(np.float32) * noise
            q = prepare(q[None], mean)[0]
            tr.observe(q)
            errs.append(abs(tr.estimate().position - pos))
        return np.array(errs), tr

    errs, tr = play(2000, 2300, 0.7, 300, 0.02)
    check("locks on from a 5-minute-off guess", np.median(errs[100:]) < 30,
          f"median error {np.median(errs[100:]):.0f}s")
    check("confident once locked", tr.estimate().confidence > 0.5,
          f"{tr.estimate().confidence:.2f}")
    errs, _ = play(4000, 3500, 0.4, 300, 0.12)
    check("slow player, noisy frames", np.median(errs[150:]) < 45,
          f"median error {np.median(errs[150:]):.0f}s")

    tr = Tracker(emb)
    tr.reset(1000, 30)
    before = tr.estimate().position
    for _ in range(30):
        tr.predict(2.0)
    after = tr.estimate().position
    check("no evidence: drifts forward slower than real time", 0 <= after - before <= 60,
          f"{after - before:.0f}s in 60s")
    tr.shift(300)
    check("nudge +5 min", abs(tr.estimate().position - (after + 300)) <= 15,
          f"{tr.estimate().position:.0f}")
    tr.reset(10, 5)
    tr.shift(-300)
    check("nudge clamps at start", tr.estimate().position <= 15)

    # A menu: the same screen for 40 s that happens to look like a scene far
    # ahead. A confident tracker must stay put.
    errs, tr = play(2000, 2000, 0.7, 120, 0.05)
    p0 = tr.estimate().position
    menu = prepare((raw[6500] + rng.normal(size=d).astype(np.float32) * 0.02)[None], mean)[0]
    for _ in range(20):
        tr.predict(2.0)
        tr.observe(menu + rng.normal(size=d).astype(np.float32) * 0.003)
    check("40 s in a menu that looks like 1 h ahead: no jump",
          abs(tr.estimate().position - p0) < 120, f"moved {tr.estimate().position - p0:+.0f}s")
    w = distinctiveness(emb)
    check("distinctiveness in 0..1", w.shape == (n,) and 0 <= w.min() and w.max() <= 1,
          f"min {w.min():.2f}")

    # Pause detection: three practically identical screenshots in a row.
    pd = PauseDetector()
    menu_q = rng.normal(size=d).astype(np.float32)
    seq = [pd.update(menu_q + rng.normal(size=d).astype(np.float32) * 0.01) for _ in range(4)]
    check("paused after three alike screenshots", seq == [False, False, True, True], str(seq))
    check("unpauses as soon as the screen changes",
          pd.update(rng.normal(size=d).astype(np.float32)) is False)
    check("black screen isn't a pause", pd.update(None) is False)

    # Uninformative screenshot (looks like everything) must not move the belief.
    tr = Tracker(emb)
    tr.reset(3000, 20)
    p0 = tr.estimate().position
    flat = prepare(mean[None] + 1e-3, mean)[0]
    tr.observe(flat)
    check("bland frame ignored", abs(tr.estimate().position - p0) < 5)

    with tempfile.TemporaryDirectory() as tmp:
        from wsync import games as G
        C.GAMES_DIR = Path(tmp)
        g = G.create("", "https://www.youtube.com/watch?v=DCqSCazDv64", 5400)
        thumbs = {0: b"A" * 10, 2: b"B" * 5}
        G.write_thumbs(g.dir, thumbs, 4)
        th = g.thumbs()
        check("thumb store", th.get(0) == b"A" * 10 and th.get(4) == b"B" * 5
              and th.get(2) in (b"A" * 10, b"B" * 5))
        check("game meta round-trip", G.get(g.slug).position == 5400)
        check("youtube url at time", g.youtube_url(5400).endswith("&t=5400s"))

    print("\nresult:", "ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    argv = sys.argv[1:]
    if "--selftest" in argv:
        return selftest()
    if "--list" in argv:
        return list_games()
    if "--index" in argv:
        return index_cli(argv[argv.index("--index") + 1:])
    if "--replay-session" in argv:
        return replay_session_cli(argv[argv.index("--replay-session") + 1:])
    if "--replay" in argv:
        return replay_cli(argv[argv.index("--replay") + 1:])

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        raise RuntimeError("PySide6 is not installed - delete the .venv folder "
                           "and run run.bat again.") from e
    from PySide6.QtNetwork import QLocalServer, QLocalSocket

    from wsync import config as C
    from wsync import theme
    from wsync.widget import SyncWindow

    cfg = C.load()
    app = QApplication(sys.argv)
    app.setApplicationName("Walkthrough Sync")
    theme.apply(app)
    app.setQuitOnLastWindowClosed(False)

    key = "WalkthroughSync-" + str(C.APP_DIR).lower().replace("\\", "-").replace("/", "-")
    probe = QLocalSocket()
    probe.connectToServer(key)
    if probe.waitForConnected(400):
        probe.write(b"show")
        probe.flush()
        probe.waitForBytesWritten(400)
        probe.disconnectFromServer()
        print("Walkthrough Sync is already running — showing the existing one.")
        return 0
    QLocalServer.removeServer(key)
    server = QLocalServer()
    server.listen(key)

    win = SyncWindow(cfg)
    win.show()
    if not cfg.get("current_game"):
        from PySide6.QtCore import QTimer
        QTimer.singleShot(400, win.new_game)

    def _second_launch():
        conn = server.nextPendingConnection()
        if conn is not None:
            conn.disconnectFromServer()
        win.show()
        win.raise_()
        win.activateWindow()

    server.newConnection.connect(_second_launch)
    app.aboutToQuit.connect(server.close)
    return app.exec()


def _run() -> int:
    """pythonw has no console, so a crash would vanish without a trace:
    write it to data/log.txt and show it in a message box."""
    import os
    # Under pythonw sys.stdout / sys.stderr are None, and libraries that print
    # progress bars (the model download, for one) crash writing to them.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        return main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        tb = traceback.format_exc()
        try:
            from wsync import config as C
            C.log("startup crash:\n" + tb)
        except Exception:
            pass
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, tb[-1500:] + "\n\nDetails are in data\\log.txt.",
                "Walkthrough Sync couldn't start", 0x10)
        except Exception:
            print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
