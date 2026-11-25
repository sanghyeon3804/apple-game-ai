"""C++ NRPA + 학습된 정책(조합 가중치 + 박스 특징)으로 한 판을 푼다."""
from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass

from . import native as nv
from .board import Board, Rect
from .rules import verify


class Cancelled(Exception):
    pass


@dataclass
class SolveResult:
    score: int
    path: list[Rect]
    elapsed: float


class _Cancel:
    """threading.Event 처럼 쓰지만 C++ 이 직접 읽는 정수를 갱신한다."""

    def __init__(self):
        self.flag = ctypes.c_int(0)

    def set(self):
        self.flag.value = 1

    def clear(self):
        self.flag.value = 0

    def is_set(self):
        return self.flag.value != 0


class Solver:
    def __init__(self, vec, threads: int):
        self.policy, self.threads = nv.policy(vec), threads
        self.cancel = _Cancel()
        self._progress = ctypes.c_int(0)

    @classmethod
    def from_file(cls, path, threads):
        with open(path, encoding="utf-8") as f:
            vec = json.load(f)["vec"]
        if len(vec) != nv.NP:
            raise ValueError(f"정책 길이가 {nv.NP}이 아닙니다: {os.path.basename(path)}")
        return cls(vec, threads)

    @property
    def best(self) -> int:
        return self._progress.value

    def solve(self, board: Board, time_limit: float = 60.0) -> SolveResult:
        self._progress.value = 0
        t0 = time.time()
        score, path, _ = nv.nrpa(board, self.policy, time_limit,
                                 threads=self.threads, cancel=self.cancel.flag, progress=self._progress)
        if self.cancel.is_set():
            raise Cancelled("계산을 중지했습니다.")
        verify(board, path, score)
        return SolveResult(score, path, time.time() - t0)
