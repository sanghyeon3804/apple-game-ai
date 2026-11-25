"""숫자별 비율을 어느 정도 맞춘 무작위 판 생성."""
from __future__ import annotations

import random

from .board import COLS, ROWS, Board


def balanced_board(seed: int, rows: int = ROWS, cols: int = COLS, jitter: float = 0.15) -> Board:
    """1~9 를 각각 약 1/9 씩 배치한다. 숫자별 비중을 (1 ± jitter) 범위에서 흔든 뒤
    개수를 정하고, 총합이 10의 배수가 될 때까지 다시 뽑은 다음 위치를 섞는다."""
    rng = random.Random(seed)
    n = rows * cols
    while True:
        w = [1 + rng.uniform(-jitter, jitter) for _ in range(9)]
        exact = [n * x / sum(w) for x in w]
        cnt = [int(x) for x in exact]
        order = sorted(range(9), key=lambda i: exact[i] - cnt[i], reverse=True)
        for i in order[:n - sum(cnt)]:
            cnt[i] += 1
        if sum((i + 1) * c for i, c in enumerate(cnt)) % 10 == 0:
            break
    cells = [i + 1 for i, c in enumerate(cnt) for _ in range(c)]
    rng.shuffle(cells)
    return Board(cells, rows, cols)
