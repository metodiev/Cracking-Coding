"""Task 285 - N-Queens via backtracking. (Tier: Intermediate, CTCI Ch 8)

Place queens row by row (one per row), trying columns in increasing order, skipping
columns and diagonals already attacked. Solutions are therefore emitted in
lexicographic order, so the first solution is deterministic. Counts are asserted
against the known sequence and re-derived with a permutation oracle.
"""

def solve_n_queens(n):
    """All solutions as lists of column indices per row, in lexicographic order."""
    solutions = []
    columns = []

    def place(row):
        if row == n:
            solutions.append(columns[:])
            return
        for col in range(n):
            ok = True
            for pr, pc in enumerate(columns):
                if col == pc or abs(col - pc) == row - pr:
                    ok = False
                    break
            if ok:
                columns.append(col)
                place(row + 1)
                columns.pop()

    place(0)
    return solutions


def count_solutions(n):
    return len(solve_n_queens(n))


def first_solution(n):
    """Lexicographically first solution, or None when no placement exists."""
    solutions = solve_n_queens(n)
    return solutions[0] if solutions else None


def is_valid_solution(solution):
    """One queen per row is structural; check columns are a permutation and no shared diagonal."""
    n = len(solution)
    if sorted(solution) != list(range(n)):
        return False
    for r in range(n):
        for s in range(r + 1, n):
            if abs(solution[r] - solution[s]) == s - r:
                return False
    return True


def brute_force_count(n):
    """Permutation oracle: count column permutations with no shared diagonal."""
    from itertools import permutations

    count = 0
    for permutation in permutations(range(n)):
        if all(abs(permutation[r] - permutation[s]) != s - r
               for r in range(n) for s in range(r + 1, n)):
            count += 1
    return count


def main():
    assert count_solutions(1) == 1
    assert count_solutions(2) == 0
    assert count_solutions(3) == 0
    assert count_solutions(4) == 2
    assert count_solutions(5) == 10
    assert count_solutions(6) == 4
    assert count_solutions(7) == 40
    assert count_solutions(8) == 92

    assert first_solution(1) == [0]
    assert first_solution(4) == [1, 3, 0, 2]        # hand-verified lexicographic first
    assert first_solution(2) is None
    assert first_solution(3) is None

    for n in range(1, 9):
        solutions = solve_n_queens(n)
        assert len(solutions) == count_solutions(n)
        assert solutions == sorted(solutions)        # deterministic lexicographic order
        for solution in solutions:
            assert is_valid_solution(solution), (n, solution)
        if solutions:
            assert first_solution(n) == solutions[0]
        assert brute_force_count(n) == len(solutions), n


if __name__ == "__main__":
    main()
    print("PASS")
