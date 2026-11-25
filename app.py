"""사과게임 자동 풀이 패널 (항상 위). build_exe.ps1 로 exe 를 만든다."""
from __future__ import annotations

import ctypes
import os
import queue
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))                 # exe 안에서는 압축이 풀린 폴더
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else HERE
WORKERS = max(1, (os.cpu_count() or 2) - 1)


def make_solver():
    """exe 옆에 policy.json 이 있으면 그것을, 없으면 내장 정책을 쓴다."""
    from apple_ai.solver import Solver
    for p in (os.path.join(APP_DIR, "policy.json"), os.path.join(HERE, "data", "policy.json")):
        if os.path.exists(p):
            return Solver.from_file(p, WORKERS)
    raise FileNotFoundError("policy.json")


class Panel:
    def __init__(self):
        from apple_ai.player import Aborted, WinDevice
        self.stop = threading.Event()
        stop = self.stop

        class Device(WinDevice):
            def check(self):
                if stop.is_set():
                    raise Aborted("중지했습니다.")
                super().check()

        self.Device = Device
        Device()                                    # Tk 보다 먼저 DPI 설정
        import tkinter as tk
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self.solver = None
        self.deadline = None

        r = self.root = tk.Tk()
        r.title("사과게임 AI")
        r.attributes("-topmost", True)
        r.resizable(False, False)
        bg, fg = "#1f2a24", "#f2fff6"
        r.configure(bg=bg)
        font = ("맑은 고딕", 10)
        row = tk.Frame(r, bg=bg)
        row.pack(padx=10, pady=(10, 4))
        self.btn = tk.Button(row, text="준비 중...", width=12, font=("맑은 고딕", 12, "bold"), bg="#14b35a",
                             fg="white", relief="flat", state="disabled", command=self.start)
        self.btn.pack(side="left")
        self.btn_stop = tk.Button(row, text="중지 (Esc)", width=9, font=font, bg="#c0392b", fg="white",
                                  relief="flat", state="disabled", command=self.do_stop)
        self.btn_stop.pack(side="left", padx=(6, 0), fill="y")
        opt = tk.Frame(r, bg=bg)
        opt.pack(padx=10, anchor="w")
        tk.Label(opt, text="계산(초)", bg=bg, fg=fg, font=font).pack(side="left")
        self.secs = tk.Spinbox(opt, from_=3, to=100, width=4, font=font)
        self.secs.delete(0, "end")
        self.secs.insert(0, "40")
        self.secs.pack(side="left", padx=(4, 10))
        tk.Label(opt, text="드래그(ms)", bg=bg, fg=fg, font=font).pack(side="left")
        self.step = tk.Spinbox(opt, from_=10, to=100, increment=5, width=4, font=font)
        self.step.delete(0, "end")
        self.step.insert(0, "20")
        self.step.pack(side="left", padx=4)
        self.status = tk.Label(r, text="준비 중...", bg=bg, fg=fg, font=font, width=36, height=3,
                               wraplength=290, justify="left", anchor="nw")
        self.status.pack(padx=10, pady=(6, 10), anchor="w")
        r.update_idletasks()
        r.geometry(f"+{r.winfo_screenwidth() - r.winfo_reqwidth() - 30}+30")
        r.update()
        u = ctypes.windll.user32
        self.hwnd = u.GetParent(r.winfo_id())
        # 화면 캡처에서 이 창을 제외 (Windows 10 2004+)
        self.excluded = bool(u.SetWindowDisplayAffinity(self.hwnd, 0x11))
        r.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self.prepare, daemon=True).start()
        r.after(50, self.tick)

    # ---------- UI 스레드 ----------
    def tick(self):
        try:
            while True:
                self.q.get_nowait()()
        except queue.Empty:
            pass
        if self.busy:
            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                self.do_stop()
            if self.deadline is not None and not self.stop.is_set():
                left = max(0, self.deadline - time.time())
                self.status.config(text=f"계산 중... {left:.0f}초 남음\n지금까지 최고 {self.solver.best}점")
        self.root.after(50, self.tick)

    def ui(self, fn):
        self.q.put(fn)

    def say(self, text):
        self.ui(lambda: self.status.config(text=text))

    def start(self):
        try:
            secs, step = float(self.secs.get()), float(self.step.get()) / 1000
            if not (1 <= secs <= 600 and 0.005 <= step <= 0.5):
                raise ValueError
        except ValueError:
            self.status.config(text="계산 시간과 드래그 값을 숫자로 확인하세요.")
            return
        self.stop.clear()
        self.solver.cancel.clear()
        self.busy = True
        self.btn.config(state="disabled", text="실행 중")
        self.btn_stop.config(state="normal")
        if not self.excluded:
            self.root.withdraw()
            self.root.update()
        threading.Thread(target=self.run, args=(secs, step), daemon=True).start()

    def do_stop(self):
        self.stop.set()
        if self.solver:
            self.solver.cancel.set()

    def finished(self, text):
        self.busy, self.deadline = False, None
        self.root.deiconify()
        self.btn.config(state="normal", text="시작")
        self.btn_stop.config(state="disabled")
        self.status.config(text=text)

    def avoid(self, box):
        """패널이 판과 겹치면 판 밖으로 옮긴다. 자리가 없으면 숨긴다 (Esc 로 중지)."""
        r = self.root
        r.deiconify()
        r.update()
        rect = (ctypes.c_long * 4)()
        ctypes.windll.user32.GetWindowRect(self.hwnd, rect)
        x, y, w, h = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
        X1, Y1, X2, Y2 = box

        def clear(px, py):
            return px + w <= X1 or px >= X2 or py + h <= Y1 or py >= Y2

        if clear(x, y):
            return
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        for px, py in ((X2 + 5, Y1), (X1 - w - 5, Y1), (X1, Y2 + 5), (X1, Y1 - h - 5)):
            if 0 <= px and px + w <= sw and 0 <= py and py + h <= sh - 40:
                r.geometry(f"+{int(px)}+{int(py)}")
                r.update()
                return
        r.withdraw()

    # ---------- 작업 스레드 ----------
    def prepare(self):
        try:
            s = make_solver()
            self.solver = s

            def ready():
                self.btn.config(state="normal", text="시작")
                self.status.config(text="게임을 시작하고 판 전체가 보이면\n'시작'을 누르세요.")
            self.ui(ready)
        except Exception as e:
            self.say(f"준비 실패: {e}")

    def call_ui(self, fn):
        done = threading.Event()

        def wrap():
            try:
                fn()
            finally:
                done.set()
        self.ui(wrap)
        done.wait(3)

    def run(self, secs, step):
        from apple_ai.player import Aborted, play
        from apple_ai.solver import Cancelled
        from apple_ai.vision import VisionError, find_grid, load_templates, read_board
        dev = None
        removed = 0
        try:
            dev = self.Device()
            dev.sleep(0.2)
            t0 = time.time()
            T = load_templates()
            img = dev.grab()
            grid = find_grid(img)
            board, _ = read_board(img, grid, T)
            x1, y1 = grid.center(0, 0)
            x2, y2 = grid.center(9, 16)
            box = (x1 - grid.dx, y1 - grid.dy, x2 + grid.dx, y2 + grid.dy)
            self.call_ui(lambda: self.avoid(box))
            self.deadline = time.time() + secs
            res = self.solver.solve(board, secs)
            self.deadline = None
            dev.check()
            if read_board(dev.grab(), grid, T)[0] != board:
                raise VisionError("계산하는 동안 화면이 바뀌었습니다.")
            path, start = res.path, board.remaining()
            for _ in range(6):
                self.call_ui(lambda: self.avoid(box))
                self.say(f"드래그 중... 예상 {removed + res.score}점\n마우스를 만지지 마세요 (중지: Esc)")
                before = board.remaining()
                board, failed = play(dev, grid, board, path, step_s=step, log=lambda *_: None)
                removed = start - board.remaining()
                if failed is None:
                    break
                dev.sleep(1.0)
                board, _ = read_board(dev.grab(), grid, T, allow_empty=True)
                removed = start - board.remaining()
                self.say(f"{failed}번째 드래그 실패 → 다시 계산 중")
                res = self.solver.solve(board, 2.0)
                path = res.path
            text = f"완료: {removed}개 제거 (총 {time.time() - t0:.0f}초)"
            if failed is not None:
                text = f"드래그가 계속 실패했습니다 ({removed}개 제거).\n드래그(ms)를 40으로 늘려 보세요."
        except (Aborted, Cancelled, VisionError) as e:
            text = f"중단: {e}"
        except Exception as e:
            text = f"오류: {e}"
            try:
                with open(os.path.join(APP_DIR, "app_error.log"), "a", encoding="utf-8") as f:
                    f.write(time.strftime("%F %T\n") + traceback.format_exc() + "\n")
            except OSError:
                pass
        finally:
            if dev is not None:
                dev.release()
        self.ui(lambda: self.finished(text))

    def close(self):
        self.do_stop()
        self.root.destroy()


def selftest(image, out):
    """마우스 없이 인식·계산·취소를 확인하고 결과를 파일에 쓴다."""
    import numpy as np
    from PIL import Image
    from apple_ai.solver import Cancelled
    from apple_ai.vision import find_grid, load_templates, read_board
    lines = []
    try:
        img = np.asarray(Image.open(image).convert("RGB"))
        board, conf = read_board(img, find_grid(img), load_templates())
        lines.append(f"read ok conf={conf:.2f} sum={sum(board.cells)}")
        s = make_solver()
        lines.append(f"threads={WORKERS}")
        r = s.solve(board, 5)
        lines.append(f"solve score={r.score} moves={len(r.path)} sec={r.elapsed:.1f}")
        threading.Timer(1.0, s.cancel.set).start()
        t = time.time()
        try:
            s.solve(board, 30)
            lines.append("cancel FAILED")
        except Cancelled:
            lines.append(f"cancel ok {time.time() - t:.1f}s")
    except Exception:
        lines.append(traceback.format_exc())
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--selftest":
        selftest(sys.argv[2], sys.argv[3])
    else:
        Panel().root.mainloop()
