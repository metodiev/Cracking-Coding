"""Task 099 - Rotate matrix 90 degrees clockwise (in-place for square). (Tier: Beginner, CTCI Ch 1)

In-place: transpose across the main diagonal, then reverse every row.
Mutates the given matrix and also returns it for convenience.
"""


def rotate_matrix_90_clockwise(m):
    n = len(m)
    for row in m:
        if len(row) != n:
            raise ValueError("matrix must be square")
    for i in range(n):
        for j in range(i + 1, n):
            m[i][j], m[j][i] = m[j][i], m[i][j]
    for row in m:
        row.reverse()
    return m


def main():
    m1 = [[1]]
    assert rotate_matrix_90_clockwise(m1) == [[1]]

    m2 = [[1, 2], [3, 4]]
    assert rotate_matrix_90_clockwise(m2) == [[3, 1], [4, 2]]

    m3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert rotate_matrix_90_clockwise(m3) == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

    m4 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    assert rotate_matrix_90_clockwise(m4) == [
        [13, 9, 5, 1],
        [14, 10, 6, 2],
        [15, 11, 7, 3],
        [16, 12, 8, 4],
    ]

    m5 = [[1, 2], [3, 4]]
    assert rotate_matrix_90_clockwise(m5) is m5  # same object: in place
    assert m5 == [[3, 1], [4, 2]]

    try:
        rotate_matrix_90_clockwise([[1, 2, 3], [4, 5, 6]])
        raise AssertionError("expected ValueError for non-square")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
