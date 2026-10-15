"""Task 321 - Sudoku uniqueness check optimization. (Tier: Intermediate, CTCI Ch 8)

Count the completions of a 9x9 sudoku board but stop as soon as two are found: a
count of two is already enough to know the puzzle is not unique.  The search is
kept fast by constraint propagation - bitmask candidate sets plus naked-single
assignment - and a most-constrained-cell branching rule, which is what makes the
required cases (proper puzzle, empty board, unsatisfiable board) quick.
"""

FULL = 0x1FF  # nine candidate bits


def _box_of(cell):
    return (cell // 27) * 3 + (cell % 9) // 3


def _make_peers():
    peers = []
    for i in range(81):
        peers.append([
            j for j in range(81)
            if j != i and (j // 9 == i // 9 or j % 9 == i % 9 or _box_of(j) == _box_of(i))
        ])
    return peers


PEERS = _make_peers()


def _search(board, limit):
    """Returns (solutions found capped at `limit`, first solution found or None)."""
    grid = list(board)
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    trail = []
    found = []

    def place(cell, bit):
        rows[cell // 9] |= bit
        cols[cell % 9] |= bit
        boxes[_box_of(cell)] |= bit
        grid[cell] = bit.bit_length()
        trail.append((cell, bit))

    def undo_to(mark):
        while len(trail) > mark:
            cell, bit = trail.pop()
            grid[cell] = 0
            rows[cell // 9] ^= bit
            cols[cell % 9] ^= bit
            boxes[_box_of(cell)] ^= bit

    def candidates(cell):
        return FULL & ~(rows[cell // 9] | cols[cell % 9] | boxes[_box_of(cell)])

    def propagate(seeds):
        """Naked singles: a peer left with a single candidate is forced; False on contradiction."""
        work = list(seeds)
        while work:
            cell = work.pop()
            for peer in PEERS[cell]:
                if grid[peer]:
                    continue
                cand = candidates(peer)
                if cand == 0:
                    return False
                if cand & (cand - 1) == 0:       # exactly one bit set -> forced
                    place(peer, cand)
                    work.append(peer)
        return True

    for cell, value in enumerate(grid):
        if value:
            bit = 1 << (value - 1)
            if candidates(cell) & bit == 0:
                return 0, None                   # the given digits already conflict
            place(cell, bit)
    if not propagate(range(81)):
        return 0, None

    def search():
        best_cell, best_cand, best_size = -1, 0, 10
        for cell in range(81):
            if grid[cell] == 0:
                cand = candidates(cell)
                if cand == 0:
                    return                       # dead end
                size = bin(cand).count("1")
                if size < best_size:
                    best_cell, best_cand, best_size = cell, cand, size
                    if size == 2:
                        break                    # cannot do better
        if best_cell == -1:
            found.append(grid[:])
            return
        while best_cand:
            bit = best_cand & -best_cand
            best_cand ^= bit
            mark = len(trail)
            place(best_cell, bit)
            if propagate([best_cell]):
                search()
            undo_to(mark)
            if len(found) >= limit:
                return

    search()
    return len(found), (found[0] if found else None)


def count_solutions(board, limit=2):
    """Count completions of `board` (81 values, 0 = empty), capped at `limit`."""
    return _search(board, limit)[0]


def uniqueness(board):
    """0 = unsatisfiable, 1 = exactly one solution, 2 = "many" (at least two)."""
    return _search(board, 2)[0]


def first_solution(board):
    """One completion of `board`, or None when the board has no solution."""
    return _search(board, 1)[1]


def units(board):
    """The 27 units (9 rows, 9 columns, 9 boxes) as lists of nine values."""
    result = [[board[r * 9 + c] for c in range(9)] for r in range(9)]
    result += [[board[r * 9 + c] for r in range(9)] for c in range(9)]
    for box_row in range(3):
        for box_col in range(3):
            result.append([
                board[(box_row * 3 + r) * 9 + box_col * 3 + c]
                for r in range(3) for c in range(3)
            ])
    return result


PUZZLE = [
    5, 3, 0, 0, 7, 0, 0, 0, 0,
    6, 0, 0, 1, 9, 5, 0, 0, 0,
    0, 9, 8, 0, 0, 0, 0, 6, 0,
    8, 0, 0, 0, 6, 0, 0, 0, 3,
    4, 0, 0, 8, 0, 3, 0, 0, 1,
    7, 0, 0, 0, 2, 0, 0, 0, 6,
    0, 6, 0, 0, 0, 0, 2, 8, 0,
    0, 0, 0, 4, 1, 9, 0, 0, 5,
    0, 0, 0, 0, 8, 0, 0, 7, 9,
]

# Published completion of the puzzle above.
KNOWN_SOLUTION = [
    5, 3, 4, 6, 7, 8, 9, 1, 2,
    6, 7, 2, 1, 9, 5, 3, 4, 8,
    1, 9, 8, 3, 4, 2, 5, 6, 7,
    8, 5, 9, 7, 6, 1, 4, 2, 3,
    4, 2, 6, 8, 5, 3, 7, 9, 1,
    7, 1, 3, 9, 2, 4, 8, 5, 6,
    9, 6, 1, 5, 3, 7, 2, 8, 4,
    2, 8, 7, 4, 1, 9, 6, 3, 5,
    3, 4, 5, 2, 8, 6, 1, 7, 9,
]

# Each unit stays duplicate-free, yet the board has no completion at all: only the
# global search finds the contradiction (one clue of the unique puzzle was altered).
UNSATISFIABLE = list(PUZZLE)
UNSATISFIABLE[1] = 1

EMPTY = [0] * 81


def main():
    # empty board: plenty of solutions, so the capped counter reports "many" (2)
    assert count_solutions(EMPTY) == 2
    assert uniqueness(EMPTY) == 2
    assert count_solutions(EMPTY, limit=4) == 4

    # proper puzzle: exactly one completion; with limit 2 the whole tree is explored
    assert count_solutions(PUZZLE) == 1
    assert uniqueness(PUZZLE) == 1
    solution = first_solution(PUZZLE)
    assert solution == KNOWN_SOLUTION
    for unit in units(solution):
        assert sorted(unit) == list(range(1, 10))

    # unsatisfiable board: no completion, and it is not a trivially conflicting one
    for unit in units(UNSATISFIABLE):
        givens = [value for value in unit if value]
        assert len(set(givens)) == len(givens)
    assert count_solutions(UNSATISFIABLE) == 0
    assert uniqueness(UNSATISFIABLE) == 0
    assert first_solution(UNSATISFIABLE) is None

    # a plainly conflicting board is rejected while loading the givens
    conflicting = list(PUZZLE)
    conflicting[0] = 6                          # two 6s in the top-left box
    assert count_solutions(conflicting) == 0

    # sparse valid board (a few clues of the known solution) has multiple completions
    sparse = [0] * 81
    for index in range(0, 81, 10):
        sparse[index] = KNOWN_SOLUTION[index]
    assert uniqueness(sparse) == 2


if __name__ == "__main__":
    main()
    print("PASS")
