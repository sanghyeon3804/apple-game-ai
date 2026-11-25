"""화면 이미지에서 사과 격자와 숫자를 읽는다 (Pillow + numpy 만 사용).

기본 색(빨간 사과, 흰 숫자)을 전제로 한다. 게임의 '薄い色'(연한 색)은 꺼 두어야 한다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .board import COLS, ROWS, Board

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "digit_templates.json")
TW, TH = 16, 24


class VisionError(Exception):
    pass


@dataclass
class Grid:
    x0: float          # (0,0) 칸 중심
    y0: float
    dx: float          # 칸 간격
    dy: float

    def center(self, r, c):
        return self.x0 + c * self.dx, self.y0 + r * self.dy


def red_mask(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 190) & (g < 110) & (b < 110)


def _components(mask: np.ndarray):
    """가로 구간 + union-find 연결 요소 → (x1, y1, x2, y2) 목록."""
    parent, box, prev = [], [], []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for y in range(mask.shape[0]):
        row = mask[y]
        if not row.any():
            prev = []
            continue
        d = np.diff(np.concatenate([[0], row.view(np.int8), [0]]))
        runs, j = [], 0
        for s, e in zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()):
            idx = len(parent)
            parent.append(idx)
            box.append([s, y, e, y + 1])
            while j < len(prev) and prev[j][1] <= s:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < e:
                a, b = find(idx), find(prev[k][2])
                if a != b:
                    parent[a] = b
                    A, B = box[a], box[b]
                    box[b] = [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])]
                k += 1
            runs.append((s, e, idx))
        prev = runs
    return [box[i] for i in range(len(parent)) if parent[i] == i]


def _cluster(vals, gap):
    out = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(g)) for g in out]


def find_grid(img: np.ndarray) -> Grid:
    """사과 170개가 모두 있는 화면에서 격자를 찾는다."""
    boxes = []
    for x1, y1, x2, y2 in _components(red_mask(img)):
        w, h = x2 - x1, y2 - y1
        if 12 <= w <= 160 and 12 <= h <= 160 and 0.7 <= w / h <= 1.45:
            boxes.append((x1, y1, x2, y2))
    # 크기가 비슷한 덩어리끼리 모아 170개인 무리를 찾는다
    for ref in sorted({b[2] - b[0] for b in boxes}):
        grp = [b for b in boxes if abs((b[2] - b[0]) - ref) <= max(2, 0.12 * ref)]
        if len(grp) < ROWS * COLS:
            continue
        xs = _cluster([(b[0] + b[2]) / 2 for b in grp], 0.4 * ref)
        ys = _cluster([(b[1] + b[3]) / 2 for b in grp], 0.4 * ref)
        if len(grp) == ROWS * COLS and len(xs) == COLS and len(ys) == ROWS:
            dx, dy = (xs[-1] - xs[0]) / (COLS - 1), (ys[-1] - ys[0]) / (ROWS - 1)
            ok = (all(abs(x - xs[0] - i * dx) < 0.15 * dx for i, x in enumerate(xs))
                  and all(abs(y - ys[0] - i * dy) < 0.15 * dy for i, y in enumerate(ys)))
            if ok:
                return Grid(xs[0], ys[0], dx, dy)
    raise VisionError(f"17x10 사과 격자를 찾지 못했습니다 (사과 모양 덩어리 {len(boxes)}개). "
                      "판 전체가 보이는지, 연한 색 옵션이 꺼져 있는지 확인하세요.")


def _crop(img, grid, r, c, k=0.42):
    x, y = grid.center(r, c)
    x1, x2 = int(round(x - k * grid.dx)), int(round(x + k * grid.dx))
    y1, y2 = int(round(y - k * grid.dy)), int(round(y + k * grid.dy))
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        raise VisionError("격자가 화면 밖에 있습니다.")
    return img[y1:y2, x1:x2]


def has_apple(img, grid, r, c) -> bool:
    return bool(red_mask(_crop(img, grid, r, c)).mean() > 0.2)


def digit_glyph(img, grid, r, c):
    """숫자의 흰 정도(0~1)를 외곽에 맞춰 잘라 TH x TW 로 정규화. 끝에 가로/세로 비를 붙인다."""
    a = _crop(img, grid, r, c, 0.30).astype(float)
    white = np.clip((np.minimum(a[..., 1], a[..., 2]) - 60) / 180, 0, 1)
    m = white > 0.5
    if m.sum() < 8:
        return None
    ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    g = white[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    asp = g.shape[1] / g.shape[0]
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((TW, TH), Image.Resampling.BILINEAR)
    return np.concatenate([np.asarray(im, float).ravel() / 255, [asp * 6]])


def build_templates(img, grid, rows, path=TEMPLATE_PATH):
    acc = {}
    for r in range(ROWS):
        for c in range(COLS):
            acc.setdefault(int(rows[r][c]), []).append(digit_glyph(img, grid, r, c))
    t = {str(d): np.mean(v, axis=0).round(4).tolist() for d, v in sorted(acc.items())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(t, f)


def load_templates(path=TEMPLATE_PATH):
    with open(path, encoding="utf-8") as f:
        return {int(k): np.array(v) for k, v in json.load(f).items()}


def read_board(img, grid, templates, allow_empty=False) -> tuple[Board, float]:
    """(보드, 최소 신뢰도). 신뢰도 = 1 - 1등 거리/2등 거리."""
    cells, worst = [], 1.0
    for r in range(ROWS):
        for c in range(COLS):
            if not has_apple(img, grid, r, c):
                if not allow_empty:
                    raise VisionError(f"({r},{c}) 칸에 사과가 없습니다.")
                cells.append(0)
                continue
            g = digit_glyph(img, grid, r, c)
            if g is None:
                raise VisionError(f"({r},{c}) 칸의 숫자를 찾지 못했습니다.")
            d = sorted((float(np.linalg.norm(g - t)), k) for k, t in templates.items())
            conf = 1 - d[0][0] / d[1][0]
            if conf < 0.15:
                raise VisionError(f"({r},{c}) 칸의 숫자가 불확실합니다 ({d[0][1]} 또는 {d[1][1]}).")
            worst = min(worst, conf)
            cells.append(d[0][1])
    return Board(cells, ROWS, COLS), worst
