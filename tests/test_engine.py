"""python -m tests.test_engine"""
import random
import time

from apple_ai import native as nv
from apple_ai.board import Board
from apple_ai.board_gen import balanced_board
from apple_ai.rules import enumerate_actions
from apple_ai.solver import Cancelled, Solver


def main():
    solver = Solver.from_file("data/policy.json", nv.THREADS)
    rng, n = random.Random(0), 0
    for s in range(40):                        # C++ 후보가 Python 규칙과 같은지
        b = balanced_board(s, 10, 7 if s % 2 else 17)
        while True:
            acts = enumerate_actions(b)
            mv, p, f = nv.enum_moves(b, solver.policy)
            assert [a.rect for a in acts] == [tuple(int(x) for x in r[:4]) for r in mv]
            if not acts:
                break
            assert [nv.COMBOS[r[4]] for r in mv] == [a.combo for a in acts]
            assert abs(p.sum() - 1) < 1e-9
            cov = {}                           # 박스 특징도 직접 계산과 비교
            for a in acts:
                for c in a.cells:
                    cov.setdefault(c, set()).add(a.rect)
            for a, x in zip(acts, f):
                hit = {o.rect for o in acts if o is not a and set(o.cells) & set(a.cells)}
                orphan = sum(1 for c, q in cov.items() if c not in a.cells and q <= hit)
                own = sum(len(cov[c]) - 1 for c in a.cells)
                assert abs(x[3] - len(hit) / 5) + abs(x[4] - orphan / 3) + abs(x[5] - own / 5) < 1e-9
            b = b.apply(rng.choice(acts))
            n += 1
    print(f"후보/조합/특징 OK ({n}개 상태)")
    r = solver.solve(balanced_board(77), 5)     # solve 가 경로를 규칙대로 재생해 검증한다
    print(f"탐색 OK: {r.score}점, {len(r.path)}수, {r.elapsed:.1f}초")
    assert solver.solve(Board(bytes(170)), 1).score == 0
    solver.cancel.set()
    t = time.time()
    try:
        solver.solve(balanced_board(77), 20)
        raise AssertionError("취소되지 않음")
    except Cancelled:
        assert time.time() - t < 3
    print("취소 OK")


if __name__ == "__main__":
    main()
