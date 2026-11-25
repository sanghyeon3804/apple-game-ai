"""A. 보드 및 행동 관리 — 보드 상태, 행동 표현, 숫자별 개수 계산, 행동 적용.

보드는 불변(immutable) 객체로 다룬다. apply()는 새 Board를 반환하므로
"현재 보드 복사"가 따로 필요 없고, 탐색 중 상태 공유가 안전하다.

좌표 규약: Rect = (r1, c1, r2, c2), 양 끝 포함(inclusive).
  (r1, c1) = 왼쪽 위 꼭짓점, (r2, c2) = 오른쪽 아래 꼭짓점.
기본 크기: 가로 17칸 x 세로 10칸 (rows=10, cols=17).
"""
from __future__ import annotations

from dataclasses import dataclass

ROWS = 10
COLS = 17
TARGET_SUM = 10

Rect = tuple[int, int, int, int]
# Combo: 길이 10 튜플. combo[i] = 해당 조합에서 숫자 i의 등장 횟수 (i=1..9), combo[0]은 항상 0.
Combo = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Action:
    rect: Rect               # 선택한 직사각형
    combo: Combo             # 제거되는 숫자들의 멀티셋
    cells: tuple[int, ...]   # 실제로 제거되는 칸(0이 아닌 칸)의 flat index

    @property
    def gain(self) -> int:
        """이 행동으로 제거되는 사과 개수."""
        return len(self.cells)


def combo_str(combo: Combo) -> str:
    """(0,1,1,1,1,0,...) -> '1+2+3+4'"""
    return "+".join(str(d) for d in range(1, 10) for _ in range(combo[d]))


class Board:
    __slots__ = ("rows", "cols", "cells")

    def __init__(self, cells, rows: int = ROWS, cols: int = COLS):
        cells = bytes(cells)
        if len(cells) != rows * cols:
            raise ValueError(f"보드 크기 불일치: {len(cells)} != {rows}x{cols}")
        if any(v > 9 for v in cells):
            raise ValueError("칸의 값은 0~9 여야 합니다.")
        self.rows = rows
        self.cols = cols
        self.cells = cells

    # ---------- 생성 ----------
    @classmethod
    def from_rows(cls, grid) -> "Board":
        rows = len(grid)
        cols = len(grid[0])
        flat = []
        for row in grid:
            if len(row) != cols:
                raise ValueError("모든 행의 길이가 같아야 합니다.")
            flat.extend(int(v) for v in row)
        return cls(flat, rows, cols)

    # ---------- 조회 ----------
    def get(self, r: int, c: int) -> int:
        return self.cells[r * self.cols + c]

    def counts(self) -> list[int]:
        """count[i] = 숫자 i가 남아 있는 개수 (i=1..9). count[0] = 제거된 칸 수."""
        cnt = [0] * 10
        for v in self.cells:
            cnt[v] += 1
        return cnt

    def remaining(self) -> int:
        """남아 있는 사과 수."""
        return len(self.cells) - self.cells.count(0)

    # ---------- 변경 ----------
    def apply(self, action: Action) -> "Board":
        """행동을 적용한 새 보드를 반환한다 (원본은 변하지 않음)."""
        buf = bytearray(self.cells)
        for i in action.cells:
            buf[i] = 0
        return Board(buf, self.rows, self.cols)

    def __eq__(self, other) -> bool:
        return isinstance(other, Board) and self.cells == other.cells and self.cols == other.cols

    def __hash__(self) -> int:
        return hash(self.cells)

    def __str__(self) -> str:
        lines = []
        for r in range(self.rows):
            row = self.cells[r * self.cols:(r + 1) * self.cols]
            lines.append(" ".join("." if v == 0 else str(v) for v in row))
        return "\n".join(lines)
