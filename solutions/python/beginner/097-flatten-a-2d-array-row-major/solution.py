"""Task 097 - Flatten a 2D array (row-major). (Tier: Beginner, CTCI Ch 1)

Hand-rolled nested loop: rows in order, elements of each row in order.
Empty rows are fine; ragged input works.
"""


def flatten_2d(matrix):
    result = []
    for row in matrix:
        for value in row:
            result.append(value)
    return result


def main():
    assert flatten_2d([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]
    assert flatten_2d([[1], [2, 3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6]
    assert flatten_2d([[1, 2, 3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6]
    assert flatten_2d([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flatten_2d([[], [1, 2], []]) == [1, 2]
    assert flatten_2d([[1]]) == [1]
    assert flatten_2d([]) == []
    assert flatten_2d([[]]) == []
    assert flatten_2d([[], []]) == []
    assert flatten_2d([[-1, 0], [0, -1]]) == [-1, 0, 0, -1]


if __name__ == "__main__":
    main()
    print("PASS")
