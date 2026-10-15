"""Task 276 - Implement the Hungarian algorithm for assignment (basic). (Tier: Intermediate, CTCI Ch 4)

JV / e-maxx formulation: potentials u and v are adjusted by the shortest delta while a
matching grows one row at a time, giving O(n^3). Cross-checked against brute-force
permutations, which is feasible up to n = 5.
"""

from itertools import permutations


def hungarian(cost):
    """Returns (minimum total cost, assignment) where assignment[row] == column."""
    n = len(cost)
    if n == 0:
        return 0, []
    infinity = float("inf")
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [infinity] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = infinity
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    total = sum(cost[i][assignment[i]] for i in range(n))
    return total, assignment


def brute_force(cost):
    n = len(cost)
    best = min(sum(cost[i][perm[i]] for i in range(n)) for perm in permutations(range(n)))
    return best


def lcg_matrix(n, seed):
    matrix = []
    for _ in range(n):
        row = []
        for _ in range(n):
            seed = (seed * 75 + 74) % 65537
            row.append(seed % 20)
        matrix.append(row)
    return matrix, seed


def main():
    assert hungarian([[1, 2], [3, 4]]) == (5, [0, 1])
    assert hungarian([[10, 19], [20, 11]])[0] == 21
    assert hungarian([[1, 2, 3], [4, 5, 6], [7, 8, 9]])[0] == 15
    assert hungarian([[7]]) == (7, [0])
    assert hungarian([]) == (0, [])
    # anti-diagonal forces the expensive-looking assignment
    assert hungarian([[9, 1, 1], [1, 9, 1], [1, 1, 9]])[0] == 3

    seed = 4242
    for size in (2, 3, 4, 5):
        for _ in range(15):
            matrix, seed = lcg_matrix(size, seed)
            total, assignment = hungarian(matrix)
            assert sorted(assignment) == list(range(size)), (matrix, assignment)
            assert total == sum(matrix[i][assignment[i]] for i in range(size))
            assert total == brute_force(matrix), (matrix, total, brute_force(matrix))


if __name__ == "__main__":
    main()
    print("PASS")
