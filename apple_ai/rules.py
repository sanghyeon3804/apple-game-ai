"""A. 게임 규칙 — 유효 직사각형 판정 및 후보 생성.

규칙:
  - 직사각형 내부 숫자 합(0은 무시)이 정확히 10이면 제거 가능.
  - 같은 칸 집합을 제거하는 직사각형은 하나로 취급한다.
    대표 직사각형으로 "꼭 맞는(tight)" 직사각형, 즉 네 변 각각에
    0이 아닌 칸이 최소 하나 있는 직사각형만 생성한다.
    (tight 직사각형 = 제거되는 칸들의 bounding box 이므로 칸 집합당 정확히 1개)
"""
from __future__ import annotations

from .board import TARGET_SUM, Action, Board, Combo, Rect


def _prefix_sums(board: Board):
    R, C = board.rows, board.cols
    S = [[0] * (C + 1) for _ in range(R + 1)]   # 값의 합
    N = [[0] * (C + 1) for _ in range(R + 1)]   # 0이 아닌 칸의 개수
    cells = board.cells
    for r in range(R):
        rs = rn = 0
        Sr, Sp, Nr, Np = S[r + 1], S[r], N[r + 1], N[r]
        base = r * C
        for c in range(C):
            v = cells[base + c]
            rs += v
            rn += 1 if v else 0
            Sr[c + 1] = Sp[c + 1] + rs
            Nr[c + 1] = Np[c + 1] + rn
    return S, N


def _iter_rects(board: Board, target: int):
    """합이 target 인 tight 직사각형 (r1,c1,r2,c2) 를 생성한다."""
    S, N = _prefix_sums(board)
    R, C = board.rows, board.cols
    for r1 in range(R):
        S1, N1 = S[r1], N[r1]
        for c1 in range(C):
            for r2 in range(r1, R):
                S2, N2 = S[r2 + 1], N[r2 + 1]
                # c1 열 한 줄의 합이 이미 target 초과면 r2를 더 늘려도 초과
                if S2[c1 + 1] - S1[c1 + 1] - S2[c1] + S1[c1] > target:
                    break
                for c2 in range(c1, C):
                    s = S2[c2 + 1] - S1[c2 + 1] - S2[c1] + S1[c1]
                    if s > target:
                        break
                    if s != target:
                        continue
                    # tight 판정: 위/아래 행, 왼/오른 열에 사과가 있어야 함
                    Na, Nb = N[r1 + 1], N[r2]
                    if N1[c2 + 1] - N1[c1] == Na[c2 + 1] - Na[c1]:      # 위쪽 행 비어 있음
                        continue
                    if N2[c2 + 1] - N2[c1] == Nb[c2 + 1] - Nb[c1]:      # 아래쪽 행 비어 있음
                        continue
                    if (N2[c1 + 1] - N1[c1 + 1] - N2[c1] + N1[c1]) == 0:  # 왼쪽 열
                        continue
                    if (N2[c2 + 1] - N1[c2 + 1] - N2[c2] + N1[c2]) == 0:  # 오른쪽 열
                        continue
                    yield r1, c1, r2, c2


def enumerate_actions(board: Board, target: int = TARGET_SUM) -> list[Action]:
    """현재 보드에서 가능한 모든 행동. 순서는 결정적(재현성 보장)."""
    cells = board.cells
    C = board.cols
    actions = []
    for r1, c1, r2, c2 in _iter_rects(board, target):
        idx = []
        combo = [0] * 10
        for r in range(r1, r2 + 1):
            base = r * C
            for c in range(c1, c2 + 1):
                v = cells[base + c]
                if v:
                    idx.append(base + c)
                    combo[v] += 1
        actions.append(Action((r1, c1, r2, c2), tuple(combo), tuple(idx)))
    return actions


def all_combos(target: int = 10) -> list[Combo]:
    """합이 target 인 1~9 숫자 멀티셋 전체 (10이면 41개)."""
    out = []

    def rec(remaining, max_d, cur):
        if remaining == 0:
            c = [0] * 10
            for d in cur:
                c[d] += 1
            out.append(tuple(c))
            return
        for d in range(min(max_d, remaining, 9), 0, -1):
            rec(remaining - d, d, cur + [d])

    rec(target, 9, [])
    return out


def rect_action(board: Board, rect: Rect) -> Action:
    r1, c1, r2, c2 = rect
    cells, combo = [], [0] * 10
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            v = board.cells[r * board.cols + c]
            if v:
                cells.append(r * board.cols + c)
                combo[v] += 1
    return Action(rect, tuple(combo), tuple(cells))


def verify(board: Board, path: list[Rect], score: int) -> Board:
    """경로를 규칙대로 재생해 모든 수의 유효성과 점수를 확인한다."""
    b = board
    for rect in path:
        a = next((x for x in enumerate_actions(b) if x.rect == tuple(rect)), None)
        if a is None:
            raise AssertionError(f"유효하지 않은 수: {rect}")
        b = b.apply(a)
    if board.remaining() - b.remaining() != score:
        raise AssertionError("재생 점수 불일치")
    return b
