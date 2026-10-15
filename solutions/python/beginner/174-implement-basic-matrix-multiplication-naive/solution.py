
"""Task 174 - Implement basic matrix multiplication (naive). (Tier: Beginner, CTCI Ch 6)

C = A x B with the triple loop, O(n^3) for square n x n. Dimensions must match:
A is (n x m) and B is (m x p) -> C is (n x p).
"""


def matmul(a, b):
    if not a or not b:
        return []
    n, m = len(a), len(a[0])
    m2, p = len(b), len(b[0])
    if m != m2:
        raise ValueError(f"dimension mismatch: {n}x{m} cannot multiply {m2}x{p}")
    c = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            total = 0
            for k in range(m):
                total += a[i][k] * b[k][j]
            c[i][j] = total
    return c


def main():
    assert matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]) == [[58, 64], [139, 154]]
    identity = [[1, 0], [0, 1]]
    m = [[5, 6], [7, 8]]
    assert matmul(m, identity) == m and matmul(identity, m) == m
    assert matmul([], []) == []
    assert matmul([[1, 2]], [[3], [4]]) == [[11]]          # 1x2 * 2x1 = 1x1
    try:
        matmul([[1, 2]], [[1, 2]])                          # 1x2 * 1x2 -> mismatch
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
