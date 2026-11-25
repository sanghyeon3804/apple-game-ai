"""빔 서치용 보드 가치 함수 학습 (약 26분).  python train_value.py  → value_new.json

정책 진행에서 뽑은 상태로 회귀한 뒤, 빔 서치가 실제로 만난 상태를 더해 다시 학습하기를 반복한다.
"""
from __future__ import annotations

import json
import time

import numpy as np

from apple_ai import native as nv
from apple_ai.board_gen import balanced_board

SEED = {"value": 5_300_000_000, "value_beam": 5_400_000_000, "value_test": 5_600_000_000}


def load(path):
    with open(path, encoding="utf-8") as f:
        return np.array(json.load(f)["vec"])


def boards(kind, start, n):
    return [balanced_board(SEED[kind] + start + i) for i in range(n)]


def fit_value(F, y, lam=1e-3):
    s = F.std(0) + 1e-9
    A = F / s
    return np.linalg.solve(A.T @ A / len(y) + lam * np.eye(F.shape[1]), A.T @ y / len(y)) / s


def within_corr(F, y, X, w):
    """남은 수가 같은 상태끼리의 순위 상관 근사: 남은 수별 평균을 뺀 뒤 상관."""
    rem = (X > 0).sum(1)
    p = F @ w
    a, b = y.copy(), p.copy()
    for r in np.unique(rem):
        i = rem == r
        a[i] -= a[i].mean()
        b[i] -= b[i].mean()
    return np.corrcoef(a, b)[0, 1]


def train_value(n0=40000, reps=32, iters=3, nb=60, secs=3.0):
    t0, pol = time.time(), nv.policy(load("data/policy.json"))
    X, R, C = nv.pack(boards("value", 0, n0))
    S = nv.sample_states(X, R, C, pol, 3)
    m, b = nv.games(S, R, C, pol, reps, 4)
    F = nv.value_features(S, R, C)
    test = boards("value_test", 0, 30)
    res = []
    for it in range(iters + 1):
        row = {}
        for name, y in (("mean", m), ("best", b.astype(float))):
            w = fit_value(F, y)
            s = [nv.value_beam(q, w, secs)[0] for q in test]
            row[name] = (float(np.mean(s)), w)
            print(f"[value {it}] 목표 {name}: 상태 {len(y)} 상관 {within_corr(F, y, S, w):.3f} "
                f"{secs:g}초 빔 {np.mean(s):.2f} | {(time.time() - t0) / 60:.1f}분", flush=True)
        k = max(row, key=lambda q: row[q][0])
        res.append((row[k][0], row[k][1], k, it))
        if it == iters:
            break
        new = [nv.value_beam(q, row[k][1], secs, seed=j, max_samples=1500)[3]
               for j, q in enumerate(boards("value_beam", it * nb, nb))]
        N = np.vstack(new)
        N = N[np.random.default_rng(it).permutation(len(N))[:40000]]
        m2, b2 = nv.games(N, R, C, pol, reps, 100 + it)
        S, m, b = np.vstack([S, N]), np.concatenate([m, m2]), np.concatenate([b, b2])
        F = np.vstack([F, nv.value_features(N, R, C)])
    best = max(res, key=lambda q: q[0])
    with open("value_new.json", "w", encoding="utf-8") as f:
        json.dump({"vec": best[1].tolist()}, f, indent=1)


if __name__ == "__main__":
    train_value()
