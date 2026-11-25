"""찾은 순서를 마우스 드래그로 실행한다 (Windows). 화면/마우스는 교체 가능한 객체다."""
from __future__ import annotations

import ctypes
import time

import numpy as np

from .vision import Grid, has_apple


class Aborted(Exception):
    pass


class WinDevice:
    """실제 화면 캡처와 마우스."""

    def __init__(self):
        self.u = ctypes.windll.user32
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # 캡처 좌표 = 마우스 좌표
        except (AttributeError, OSError):
            self.u.SetProcessDPIAware()
        self.last = None
        self.down = False
        self.u.GetAsyncKeyState(0x1B)          # 실행 전에 눌렀던 Esc 기록을 비운다

    def grab(self) -> np.ndarray:
        from PIL import ImageGrab
        return np.asarray(ImageGrab.grab().convert("RGB"))

    def now(self):
        return time.perf_counter()

    def sleep(self, s):
        end = time.perf_counter() + s
        while True:
            self.check()
            left = end - time.perf_counter()
            if left <= 0:
                return
            time.sleep(min(left, 0.02))

    def pos(self):
        p = (ctypes.c_long * 2)()
        self.u.GetCursorPos(ctypes.byref(p))
        return p[0], p[1]

    def check(self):
        if self.u.GetAsyncKeyState(0x1B) & 0x8001:
            raise Aborted("Esc 를 눌러 중단했습니다.")
        if self.last is not None:
            x, y = self.pos()
            if abs(x - self.last[0]) > 12 or abs(y - self.last[1]) > 12:
                raise Aborted("마우스가 움직여 중단했습니다.")

    def move(self, x, y):
        self.check()
        self.last = (int(round(x)), int(round(y)))
        self.u.SetCursorPos(*self.last)

    def press(self):
        self.u.mouse_event(0x0002, 0, 0, 0, 0)
        self.down = True

    def release(self):
        if self.down:
            self.u.mouse_event(0x0004, 0, 0, 0, 0)
            self.down = False


def drag(dev, grid: Grid, rect, steps: int, step_s: float):
    """칸 중심에서 0.45칸 바깥 모서리끼리 드래그 (이웃 사과의 중심은 포함하지 않는다)."""
    r1, c1, r2, c2 = rect
    ax, ay = grid.center(r1, c1)
    bx, by = grid.center(r2, c2)
    ax, ay = ax - 0.45 * grid.dx, ay - 0.45 * grid.dy
    bx, by = bx + 0.45 * grid.dx, by + 0.45 * grid.dy
    dev.move(ax, ay)
    dev.sleep(step_s)
    dev.press()
    try:
        dev.sleep(step_s)
        for i in range(1, steps + 1):
            dev.move(ax + (bx - ax) * i / steps, ay + (by - ay) * i / steps)
            dev.sleep(step_s)
        dev.sleep(step_s)
    finally:
        dev.release()


def play(dev, grid: Grid, board, path, steps=5, step_s=0.02, settle=0.8, log=print):
    """경로를 실행한다. 반환: (남은 보드, 실패한 수 번호 또는 None)"""
    from .rules import rect_action
    for n, rect in enumerate(path, 1):
        a = rect_action(board, tuple(rect))
        if sum(d * k for d, k in enumerate(a.combo)) != 10:
            raise ValueError(f"{n}번째 수의 합이 10이 아닙니다: {rect}")
        cells = [divmod(i, board.cols) for i in a.cells]
        for attempt in range(3):
            slow = 1 + attempt
            drag(dev, grid, rect, steps * slow, step_s * slow)
            end = dev.now() + settle
            while True:                        # 떨어지는 사과가 잠시 겹칠 수 있어 기다리며 확인
                img = dev.grab()
                gone = [not has_apple(img, grid, r, c) for r, c in cells]
                if all(gone) or dev.now() >= end:
                    break
                dev.sleep(0.03)
            if all(gone):
                board = board.apply(a)
                break
            log(f"  {n}번째 수 {rect}: 사라지지 않음 (시도 {attempt + 1})")
        else:
            return board, n
    return board, None
