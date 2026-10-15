"""Task 286 - Sudoku solver with backtracking and constraint propagation. (Tier: Intermediate, CTCI Ch 8)

Candidates come from row/column/box bitmasks and the search always branches on the
most-constrained empty cell (MRV). Verified on the classic 9x9 puzzle against its known
solution, on 25 symmetry-transformed variants (digit relabeling plus row/column
permutations within the legal band/stack structure), and on unsolvable boards.
"""

FULL = 0x1FF                                  # bits 0..8 for digits 1..9

PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def solve_sudoku(board):
    """Returns the solved 9x9 board, or None when the puzzle is unsolvable."""
    grid = [row[:] for row in board]
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    for r in range(9):
        for c in range(9):
            value = grid[r][c]
            if not 0 <= value <= 9:
                return None
            if value == 0:
                continue
            bit = 1 << (value - 1)
            box = (r // 3) * 3 + c // 3
            if rows[r] & bit or cols[c] & bit or boxes[box] & bit:
                return None                   # duplicate clue
            rows[r] |= bit
            cols[c] |= bit
            boxes[box] |= bit
    return grid if search(grid, rows, cols, boxes) else None


def search(grid, rows, cols, boxes):
    best_r = best_c = -1
    best_mask = 0
    best_count = 10
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                continue
            box = (r // 3) * 3 + c // 3
            mask = ~(rows[r] | cols[c] | boxes[box]) & FULL
            count = bin(mask).count("1")
            if count == 0:
                return False                  # dead end found by propagation
            if count < best_count:            # most-constrained cell
                best_count = count
                best_r, best_c, best_mask = r, c, mask
    if best_r == -1:
        return True                           # no empty cell left
    box = (best_r // 3) * 3 + best_c // 3
    mask = best_mask
    while mask:
        bit = mask & -mask
        mask ^= bit
        grid[best_r][best_c] = bit.bit_length()
        rows[best_r] |= bit
        cols[best_c] |= bit
        boxes[box] |= bit
        if search(grid, rows, cols, boxes):
            return True
        grid[best_r][best_c] = 0
        rows[best_r] &= ~bit
        cols[best_c] &= ~bit
        boxes[box] &= ~bit
    return False


def is_valid_solution(board):
    """Every row, column and 3x3 box must contain the digits 1..9 exactly once."""
    target = list(range(1, 10))
    for r in range(9):
        if sorted(board[r]) != target:
            return False
    for c in range(9):
        if sorted(board[r][c] for r in range(9)) != target:
            return False
    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            digits = [board[box_r + dr][box_c + dc] for dr in range(3) for dc in range(3)]
            if sorted(digits) != target:
                return False
    return True


class Lcg:
    """Deterministic generator: next = (next * 75 + 74) % 65537."""

    def __init__(self, seed=13579):
        self.state = seed

    def next(self):
        self.state = (self.state * 75 + 74) % 65537
        return self.state


def random_permutation(rng, size):
    values = list(range(size))
    for i in range(size - 1, 0, -1):
        j = rng.next() % (i + 1)
        values[i], values[j] = values[j], values[i]
    return values


def transform(grid, digit_map, row_order, col_order):
    return [[digit_map[grid[row_order[r]][col_order[c]]] for c in range(9)] for r in range(9)]


def random_symmetry(rng):
    """Digit relabeling plus band/stack and in-band/in-stack row and column permutations."""
    permutation = random_permutation(rng, 9)
    digit_map = [0] + [value + 1 for value in permutation]   # digit_map[0] stays 0
    bands = random_permutation(rng, 3)
    stacks = random_permutation(rng, 3)
    row_order = []
    col_order = []
    for group in range(3):
        inner = random_permutation(rng, 3)
        row_order.extend(bands[group] * 3 + inner[i] for i in range(3))
        inner = random_permutation(rng, 3)
        col_order.extend(stacks[group] * 3 + inner[i] for i in range(3))
    return digit_map, row_order, col_order


def main():
    solved = solve_sudoku(PUZZLE)
    assert solved == SOLUTION
    assert is_valid_solution(solved)
    for r in range(9):
        for c in range(9):
            assert PUZZLE[r][c] == 0 or solved[r][c] == PUZZLE[r][c], "clue changed"

    assert solve_sudoku(SOLUTION) == SOLUTION      # already solved board is returned as is

    duplicate_in_column = [row[:] for row in SOLUTION]
    duplicate_in_column[0][0] = 6                  # column 0 already has a 6
    assert solve_sudoku(duplicate_in_column) is None

    duplicate_in_row = [row[:] for row in SOLUTION]
    duplicate_in_row[0][1] = 5                     # row 0 already has a 5
    assert solve_sudoku(duplicate_in_row) is None
    duplicate_in_box = [row[:] for row in SOLUTION]
    duplicate_in_box[1][1] = 5                     # box 0 already has a 5
    assert solve_sudoku(duplicate_in_box) is None

    forced_conflict = [[0] * 9 for _ in range(9)]
    forced_conflict[0] = [1, 2, 3, 4, 5, 6, 0, 8, 9]   # (0,6) can only be 7
    forced_conflict[3][6] = 7                          # but column 6 already has a 7
    assert solve_sudoku(forced_conflict) is None       # empty candidate set, no guessing

    empty = [[0] * 9 for _ in range(9)]
    filled = solve_sudoku(empty)
    assert filled is not None and is_valid_solution(filled)

    rng = Lcg()
    for trial in range(25):
        digit_map, row_order, col_order = random_symmetry(rng)
        puzzle_variant = transform(PUZZLE, digit_map, row_order, col_order)
        solution_variant = transform(SOLUTION, digit_map, row_order, col_order)
        assert is_valid_solution(solution_variant), trial
        assert solve_sudoku(puzzle_variant) == solution_variant, trial


if __name__ == "__main__":
    main()
    print("PASS")
