
"""Task 177 - Implement basic sparse matrix storage (coordinate list). (Tier: Beginner, CTCI Ch 7)

Coordinate (COO) format: a dict keyed by (row, col) holding non-zero values.
Great for very sparse data; dense conversion and row iteration included.
"""


class SparseMatrix:
    def __init__(self, rows, cols):
        if rows < 0 or cols < 0:
            raise ValueError("dimensions must be non-negative")
        self.rows = rows
        self.cols = cols
        self._data = {}              # (row, col) -> value (non-zero only)

    def set(self, row, col, value):
        self._check(row, col)
        if value == 0:
            self._data.pop((row, col), None)      # zero means "remove"
        else:
            self._data[(row, col)] = value

    def get(self, row, col):
        self._check(row, col)
        return self._data.get((row, col), 0)

    def _check(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"index out of range: ({row}, {col})")

    def nnz(self):
        """Number of stored (non-zero) entries."""
        return len(self._data)

    def coordinates(self):
        return sorted(self._data.items())

    def to_dense(self):
        return [[self._data.get((r, c), 0) for c in range(self.cols)] for r in range(self.rows)]

    def row(self, r):
        return [self._data.get((r, c), 0) for c in range(self.cols)]


def main():
    m = SparseMatrix(3, 4)
    m.set(0, 0, 5)
    m.set(1, 2, 7)
    m.set(2, 3, -1)
    assert m.nnz() == 3
    assert m.get(0, 0) == 5 and m.get(0, 1) == 0 and m.get(2, 3) == -1
    assert m.coordinates() == [((0, 0), 5), ((1, 2), 7), ((2, 3), -1)]
    assert m.to_dense() == [[5, 0, 0, 0], [0, 0, 7, 0], [0, 0, 0, -1]]

    m.set(1, 2, 0)                       # overwrite with zero removes the entry
    assert m.nnz() == 2 and m.get(1, 2) == 0

    empty = SparseMatrix(2, 2)
    assert empty.nnz() == 0 and empty.to_dense() == [[0, 0], [0, 0]]

    try:
        m.get(5, 0)
        raise AssertionError("expected IndexError")
    except IndexError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
