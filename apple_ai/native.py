"""C++ 엔진(native/engine.dll) 연결."""
from __future__ import annotations

import ctypes as ct
import math
import os

import numpy as np

from .board import combo_str
from .rules import all_combos

NC, NF = 41, 7
NP = NC + NF
FEATURES = ["empty", "edge", "span", "kill", "orphan", "own_alt", "t_kill"]
COMBOS = all_combos()
NAMES = [combo_str(c) for c in COMBOS]


class CPolicy(ct.Structure):
    _fields_ = [("logw", ct.c_double * NC), ("phi", ct.c_double * NF), ("eps", ct.c_double)]


def policy(vec, eps=0.0, temp=1.0) -> CPolicy:
    """vec: 길이 NC(조합 로그 가중치) 또는 NP(+박스 특징). temp 로 나눈다."""
    v = np.zeros(NP)
    v[:len(vec)] = np.asarray(vec, float)
    v /= temp
    p = CPolicy()
    p.logw[:] = v[:NC].tolist()
    p.phi[:] = v[NC:].tolist()
    p.eps = eps
    return p


def vec_from_weights(weights: dict) -> np.ndarray:
    return np.array([math.log(weights[c]) for c in COMBOS])


_dll = ct.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "native", "engine.dll"))
_pp = ct.POINTER(CPolicy)
_dll.params.restype = ct.c_int
for f in ("nrpa", "enum_moves"):
    getattr(_dll, f).restype = ct.c_int
assert _dll.params() == NP
_dll.init(np.array(COMBOS, np.int32).ctypes.data_as(ct.c_void_p))
THREADS = max(1, (os.cpu_count() or 2) - 1)


def _a(x):
    return x.ctypes.data_as(ct.c_void_p)


def pack(boards) -> tuple[np.ndarray, int, int]:
    """Board 목록 → (n x RC, R, C)"""
    return (np.frombuffer(b"".join(b.cells for b in boards), np.uint8).reshape(len(boards), -1).copy(),
            boards[0].rows, boards[0].cols)


def enum_moves(b, pol):
    o, p, f = np.zeros(5 * 4000, np.int32), np.zeros(4000), np.zeros(4000 * NF)
    c = np.frombuffer(b.cells, np.uint8).copy()
    n = _dll.enum_moves(_a(c), b.rows, b.cols, _a(o), ct.byref(pol), _a(p), _a(f))
    return o[:5 * n].reshape(n, 5), p[:n], f[:n * NF].reshape(n, NF)


def games(X, R, C, pol, k=1, seed=0, threads=THREADS):
    m, b = np.zeros(len(X)), np.zeros(len(X), np.int32)
    _dll.games(_a(X), len(X), R, C, ct.byref(pol), k, ct.c_uint64(seed), threads, _a(m), _a(b))
    return m, b


def pg(X, R, C, pol, k, seed, threads=THREADS):
    g, h = np.zeros((len(X), NP)), np.zeros((len(X), NP))
    _dll.pg(_a(X), len(X), R, C, ct.byref(pol), k, ct.c_uint64(seed), threads, _a(g), _a(h))
    return g, h


def nrpa(b, pol, seconds, seed=0, level=2, n=(100, 400, 100), alpha=1.0, threads=THREADS,
         cancel=None, progress=None):
    """반환: (점수, 경로, 진행 횟수). cancel 을 1로 바꾸면 멈추고, progress 에 최고 점수가 들어간다."""
    c = np.frombuffer(b.cells, np.uint8).copy()
    path, k, st = np.zeros(400, np.int32), ct.c_int(0), np.zeros(4)
    cancel = cancel if cancel is not None else ct.c_int(0)
    progress = progress if progress is not None else ct.c_int(0)
    s = _dll.nrpa(_a(c), b.rows, b.cols, ct.byref(pol), ct.c_double(seconds), threads, level, *n,
                  ct.c_double(alpha), ct.c_uint64(seed), ct.byref(cancel), ct.byref(progress),
                  _a(path), ct.byref(k), _a(st))
    return s, [tuple(int(v) for v in path[4 * i:4 * i + 4]) for i in range(k.value)], int(st[0])
