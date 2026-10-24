"""Find the game window and grab what it shows (Windows only).

The game is whichever window is in the foreground, if either
  * its process matches the exe this game has been linked to, or
  * nothing is linked yet and the window fills its monitor (borderless or
    fullscreen) and isn't a browser, video player, Explorer and so on.
The first window found automatically gets linked, so alt-tabbing to a
fullscreen YouTube tab later never counts as playing.

Frames are cropped to 16:9 (ultrawide monitors show extra sides the
walkthrough never had) and box-downscaled to 320x180, like the index.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np

W, H = 320, 180

NOT_GAMES = {
    "explorer.exe", "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe",
    "opera_gx.exe", "vivaldi.exe", "arc.exe", "claude.exe", "vlc.exe", "mpc-hc64.exe",
    "mpc-be64.exe", "mpv.exe", "potplayermini64.exe", "plex.exe", "plex htpc.exe",
    "code.exe", "python.exe", "pythonw.exe", "windowsterminal.exe", "cmd.exe",
    "powershell.exe", "pwsh.exe", "steamwebhelper.exe", "steam.exe", "xboxpcapp.exe",
    "applicationframehost.exe", "searchhost.exe", "startmenuexperiencehost.exe",
    "shellexperiencehost.exe", "lockapp.exe", "taskmgr.exe", "discord.exe", "obs64.exe",
    "spotify.exe", "teams.exe", "ms-teams.exe", "slack.exe", "lmstudio.exe", "lm studio.exe",
    "gamebar.exe", "gamebarftserver.exe", "textinputhost.exe", "systemsettings.exe",
}

IS_WIN = sys.platform.startswith("win")


@dataclass
class Window:
    hwnd: int
    exe: str           # lower-case file name, e.g. "resonance.exe"
    title: str
    rect: tuple[int, int, int, int]   # client area in screen pixels: l, t, r, b
    fills_monitor: bool


if IS_WIN:
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.MonitorFromWindow.restype = wintypes.HANDLE
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]

    def _exe_of(hwnd) -> str:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return os.path.basename(buf.value).lower()
            return ""
        finally:
            kernel32.CloseHandle(h)

    def _title_of(hwnd) -> str:
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    def _info(hwnd) -> Window | None:
        if not hwnd or user32.IsIconic(hwnd) or not user32.IsWindowVisible(hwnd):
            return None
        rc = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
            return None
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(pt))
        l, t = pt.x, pt.y
        r, b = l + rc.right, t + rc.bottom
        if rc.right < 320 or rc.bottom < 180:
            return None
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        fills = False
        mon = user32.MonitorFromWindow(hwnd, 2)
        if mon and user32.GetMonitorInfoW(mon, ctypes.byref(mi)):
            m = mi.rcMonitor
            mw, mh = m.right - m.left, m.bottom - m.top
            fills = rc.right * rc.bottom >= 0.9 * mw * mh
        return Window(int(hwnd), _exe_of(hwnd), _title_of(hwnd), (l, t, r, b), fills)

    def foreground() -> Window | None:
        return _info(user32.GetForegroundWindow())

    def list_windows() -> list[Window]:
        out: dict[str, Window] = {}
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def cb(hwnd, _):
            if user32.GetWindow(hwnd, 4):  # GW_OWNER: skip owned popups
                return True
            if user32.GetWindowTextLengthW(hwnd) == 0:
                return True
            w = _info(hwnd)
            if w and w.exe and w.exe not in out and w.exe not in NOT_GAMES:
                out[w.exe] = w
            return True
        user32.EnumWindows(WNDENUMPROC(cb), 0)
        return sorted(out.values(), key=lambda w: (not w.fills_monitor, w.exe))
else:
    def foreground() -> Window | None:
        return None

    def list_windows() -> list[Window]:
        return []


def find_game(linked_exe: str) -> tuple[Window | None, bool]:
    """(window, newly_detected). Only ever returns the foreground window."""
    w = foreground()
    if w is None or not w.exe:
        return None, False
    if linked_exe:
        return (w, False) if w.exe == linked_exe.lower() else (None, False)
    if w.fills_monitor and w.exe not in NOT_GAMES:
        return w, True
    return None, False


class Grabber:
    """Screen grabs via mss. Create it on the thread that uses it."""

    def __init__(self):
        import mss
        self.sct = mss.mss()

    def grab(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        from PIL import Image
        l, t, r, b = rect
        w, h = r - l, b - t
        # crop to 16:9 around the centre before grabbing
        if w * 9 > h * 16:
            nw = h * 16 // 9
            l += (w - nw) // 2
            w = nw
        elif w * 9 < h * 16:
            nh = w * 9 // 16
            t += (h - nh) // 2
            h = nh
        shot = self.sct.grab({"left": l, "top": t, "width": w, "height": h})
        img = Image.frombuffer("RGB", shot.size, shot.bgra, "raw", "BGRX")
        img = img.resize((W, H), Image.BOX)
        return np.asarray(img)

    def close(self) -> None:
        try:
            self.sct.close()
        except Exception:
            pass


def tiny(frame: np.ndarray) -> np.ndarray:
    """32x18 grey thumbnail for change / darkness checks."""
    g = frame.astype(np.float32).mean(axis=2)
    return g.reshape(18, 10, 32, 10).mean(axis=(1, 3))
