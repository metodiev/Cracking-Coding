"""Task 098 - Transpose a 2D matrix. (Tier: Beginner, CTCI Ch 1)

Returns a new matrix with rows and columns swapped; works for non-square
rectangular matrices. [] and [[]] transpose to [].
"""


def transpose(matrix):
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    result = []
    for c in range(cols):
        new_row = []
        for r in range(rows):
            new_row.append(matrix[r][c])
        result.append(new_row)
    return result


def main():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert transpose([[1, 2, 3]]) == [[1], [2], [3]]
    assert transpose([[1], [2], [3]]) == [[1, 2, 3]]
    assert transpose([[1]]) == [[1]]
    assert transpose([]) == []
    assert transpose([[]]) == []
    assert transpose([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]

    m = [[1, 2], [3, 4]]
    t = transpose(m)
    assert t is not m
    assert m == [[1, 2], [3, 4]]


if __name__ == "__main__":
    main()
    print("PASS")
