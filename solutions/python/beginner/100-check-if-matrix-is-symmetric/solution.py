"""Task 100 - Check if matrix is symmetric. (Tier: Beginner, CTCI Ch 1)

Symmetric means m[i][j] == m[j][i] for all i, j across the main diagonal.
A non-square (or ragged) matrix is immediately false; the empty matrix is true.
"""


def is_symmetric(m):
    n = len(m)
    for row in m:
        if len(row) != n:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            if m[i][j] != m[j][i]:
                return False
    return True


def main():
    assert is_symmetric([[1, 2], [2, 1]]) is True
    assert is_symmetric([[1, 2], [3, 4]]) is False
    assert is_symmetric([[5]]) is True
    assert is_symmetric([]) is True
    assert is_symmetric([[]]) is False
    assert is_symmetric([[1, 2, 3], [2, 4, 5], [3, 5, 6]]) is True
    assert is_symmetric([[1, 2, 3], [2, 4, 5], [3, 6, 6]]) is False
    assert is_symmetric([[1, 2, 3], [4, 5, 6]]) is False
    assert is_symmetric([[1, 2], [2]]) is False
    assert is_symmetric([[0, -1], [-1, 2]]) is True
    assert is_symmetric([[0, -1], [1, 2]]) is False


if __name__ == "__main__":
    main()
    print("PASS")
