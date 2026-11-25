"""같은 새 판에서 탐색 방식을 짝지어 비교한다.  python bench.py [초=60] [판 수=10]"""
import math
import sys

import numpy as np

from apple_ai import native as nv
from apple_ai.board_gen import balanced_board
from apple_ai.rules import verify
from train_value import load


def main(secs, n):
    pol, vw = nv.policy(load("data/policy.json")), load("data/value.json")
    fns = {"rollout beam": lambda b: nv.rollout_beam(b, pol, secs),
           "value beam": lambda b: nv.value_beam(b, vw, secs),
           "NRPA": lambda b: nv.nrpa(b, pol, secs)}
    res = {k: [] for k in fns}
    for i in range(n):
        b = balanced_board(7_000_000_000 + i)
        for k, fn in fns.items():
            s, p, _ = fn(b)
            verify(b, p, s)
            res[k].append(s)
        print(f"[{i + 1}/{n}]", {k: v[-1] for k, v in res.items()}, flush=True)
    base = np.array(res["rollout beam"])
    for k, v in res.items():
        d = np.array(v) - base
        ci = 1.96 * d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        print(f"{k:<13} 평균 {np.mean(v):.2f} | rollout beam 대비 {d.mean():+.2f} ±{ci:.2f}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 60.0, int(sys.argv[2]) if len(sys.argv) > 2 else 10)
