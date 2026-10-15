"""Task 284 - Word search in a board with DFS backtracking. (Tier: Intermediate, CTCI Ch 8)

Find whether a word can be spelled by a 4-directionally adjacent path with no cell
reuse. The DFS must unmark a cell when a branch fails, otherwise a dead-end branch can
poison later branches (see the "ABCESEEEFS" trap case). Cross-checked against an
independent state-space BFS oracle over (cell, visited-mask) pairs.
"""

from collections import deque

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def exist(board, word):
    """Backtracking DFS; cells cannot be reused. Returns True iff such a path exists."""
    if not word:
        return False
    rows, cols = len(board), len(board[0])
    visited = [[False] * cols for _ in range(rows)]

    def search(r, c, i):
        if i == len(word):
            return True
        if not (0 <= r < rows and 0 <= c < cols) or visited[r][c] or board[r][c] != word[i]:
            return False
        visited[r][c] = True
        found = any(search(r + dr, c + dc, i + 1) for dr, dc in DIRECTIONS)
        visited[r][c] = False                  # unmark: later branches may need this cell
        return found

    return any(search(r, c, 0) for r in range(rows) for c in range(cols))


def oracle_exist(board, word):
    """Independent BFS over states (row, col, visited-cell mask); popcount = prefix length."""
    rows, cols = len(board), len(board[0])
    shift = rows * cols
    queue = deque()
    seen = set()
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == word[0]:
                state = (r, c, 1 << (r * cols + c))
                seen.add(state)
                queue.append(state)
    while queue:
        r, c, mask = queue.popleft()
        i = bin(mask).count("1")
        if i == len(word):
            return True
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if mask >> (nr * cols + nc) & 1 or board[nr][nc] != word[i]:
                continue
            state = (nr, nc, mask | (1 << (nr * cols + nc)))
            if state not in seen:
                seen.add(state)
                queue.append(state)
    return False


def buggy_exist_without_unmarking(board, word):
    """Deliberately never unmarks cells; only used to prove the trap case needs backtracking."""
    rows, cols = len(board), len(board[0])
    visited = [[False] * cols for _ in range(rows)]

    def search(r, c, i):
        if i == len(word):
            return True
        if not (0 <= r < rows and 0 <= c < cols) or visited[r][c] or board[r][c] != word[i]:
            return False
        visited[r][c] = True
        return any(search(r + dr, c + dc, i + 1) for dr, dc in DIRECTIONS)

    return any(search(r, c, 0) for r in range(rows) for c in range(cols))


class Lcg:
    """Deterministic generator: next = (next * 75 + 74) % 65537."""

    def __init__(self, seed=13579):
        self.state = seed

    def next(self):
        self.state = (self.state * 75 + 74) % 65537
        return self.state


def random_board(rng, rows, cols):
    return ["".join("ABC"[rng.next() % 3] for _ in range(cols)) for _ in range(rows)]


def random_case(rng, rows, cols, max_len):
    board = random_board(rng, rows, cols)
    length = 1 + rng.next() % max_len
    word = "".join("ABC"[rng.next() % 3] for _ in range(length))
    return board, word


def planted_case(rng, rows, cols, max_len):
    """Reads a random simple path, so the word is guaranteed to exist."""
    board = random_board(rng, rows, cols)
    r, c = rng.next() % rows, rng.next() % cols
    used = {(r, c)}
    letters = [board[r][c]]
    length = 1 + rng.next() % max_len
    while len(letters) < length:
        candidates = [(r + dr, c + dc) for dr, dc in DIRECTIONS
                      if 0 <= r + dr < rows and 0 <= c + dc < cols and (r + dr, c + dc) not in used]
        if not candidates:
            break
        r, c = candidates[rng.next() % len(candidates)]
        used.add((r, c))
        letters.append(board[r][c])
    return board, "".join(letters)


def main():
    classic = ["ABCE", "SFCS", "ADEE"]
    assert exist(classic, "ABCCED") is True
    assert exist(classic, "SEE") is True
    assert exist(classic, "ABCB") is False

    trap = ["ABCE", "SFES", "ADEE"]            # needs the visited-unmarking
    assert exist(trap, "ABCESEEEFS") is True
    assert buggy_exist_without_unmarking(trap, "ABCESEEEFS") is False

    assert exist(["A"], "A") is True           # single cell
    assert exist(["A"], "B") is False
    assert exist(["A"], "AA") is False         # a cell cannot be reused

    assert exist(["AB", "CD"], "ABDC") is True
    assert exist(["AB", "CD"], "ABCD") is False

    rng = Lcg()
    for _ in range(150):
        board, word = random_case(rng, 3, 3, 4)
        assert exist(board, word) == oracle_exist(board, word), (board, word)

    found_planted = 0
    for _ in range(50):
        board, word = planted_case(rng, 3, 3, 5)
        assert exist(board, word) is True, (board, word)
        assert oracle_exist(board, word) is True, (board, word)
        found_planted += 1
    assert found_planted == 50


if __name__ == "__main__":
    main()
    print("PASS")
