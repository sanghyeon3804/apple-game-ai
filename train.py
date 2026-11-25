"""정책(조합 가중치 41 + 박스 특징 7) 학습.  python train.py [라운드=16]

data/start_weights.json 에서 출발해 반사실 기여도 정책 경사로 학습하고 policy_new.json 에 저장한다.
매 라운드 저장하므로 다시 실행하면 이어진다.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from apple_ai import native as nv
from apple_ai.board_gen import balanced_board

OUT = "policy_new.json"
SEED = 5_000_000_000


def packed(start, n):
    return nv.pack([balanced_board(SEED + start + i) for i in range(n)])


def center(v):
    v = v.copy()
    v[:nv.NC] -= v[:nv.NC].mean()
    return v


def start_vec():
    with open("data/start_weights.json", encoding="utf-8") as f:
        w = {tuple(map(int, k.split("+"))): x for k, x in json.load(f).items()}
    v = np.zeros(nv.NP)
    v[:nv.NC] = [math.log(w[tuple(map(int, n.split("+")))]) for n in nv.NAMES]
    return center(v)


def train(rounds=16, states=1200, k=4):
    v, first = start_vec(), 0
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            d = json.load(f)
        v, first = np.array(d["vec"]), d["round"]
    for r in range(first, rounds):
        X, R, C = packed(r * states, states)
        g, h = nv.pg(X, R, C, nv.policy(v), k, 1000 + r)
        G, H, se = g.mean(0), h.mean(0), g.std(0) / math.sqrt(states)
        d = np.clip((0.6 if r < 15 else 0.3) * G / (H + 0.05), -0.3, 0.3)
        d[np.abs(G) < se] = 0                      # 오차보다 작은 기울기는 무시
        v = center(np.clip(v + d, -8, 8))
        with open(OUT + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"round": r + 1, "names": nv.NAMES + nv.FEATURES, "vec": v.tolist()}, f, indent=1)
        os.replace(OUT + ".tmp", OUT)
        if r % 4 == 3:
            X, R, C = packed(10_000_000, 1000)
            m, _ = nv.games(X, R, C, nv.policy(v), 4, 9)
            print(f"[{r + 1}/{rounds}] 한 번 둘 때 평균 {m.mean():.2f}", flush=True)


if __name__ == "__main__":
    train(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
