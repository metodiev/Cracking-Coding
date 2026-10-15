"""Task 423 - Implement offline queries with divide & conquer on value (CDQ). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Counts pairs i < j with x[i] > x[j] AND y[i] > y[j] (2D dominance / two-coordinate
inversion). cdq(lo, hi) splits the index range in half, counts the cross pairs with a
Fenwick tree over compressed y values while sweeping x in descending order, then merges the
halves sorted by x. Every pair is counted exactly once, at the split where it crosses.
Cross-checked against an O(n^2) double loop on LCG points.
"""


def count_dominating_pairs(points):
    n = len(points)
    if n < 2:
        return 0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ranks = {value: i for i, value in enumerate(sorted(set(ys)))}
    m = len(ranks)
    tree = [0] * (m + 1)
    total = 0

    def bit_add(rank, delta):
        i = rank + 1
        while i <= m:
            tree[i] += delta
            i += i & -i

    def bit_prefix(count):                 # active values with rank < count
        result = 0
        i = count
        while i > 0:
            result += tree[i]
            i -= i & -i
        return result

    def cdq(lo, hi):
        # returns indices [lo, hi) sorted by x descending; adds cross pairs to total
        nonlocal total
        if hi - lo <= 1:
            return [lo]
        mid = (lo + hi) // 2
        left = cdq(lo, mid)
        right = cdq(mid, hi)
        touched = []
        pointer = 0
        for j in right:                    # right is sorted by x descending too
            while pointer < len(left) and xs[left[pointer]] > xs[j]:
                rank = ranks[ys[left[pointer]]]
                bit_add(rank, 1)
                touched.append(rank)
                pointer += 1
            total += pointer - bit_prefix(ranks[ys[j]] + 1)
        for rank in touched:
            bit_add(rank, -1)
        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            if xs[left[i]] >= xs[right[j]]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    cdq(0, n)
    return total


def brute_count(points):
    total = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i][0] > points[j][0] and points[i][1] > points[j][1]:
                total += 1
    return total


def main():
    assert count_dominating_pairs([]) == 0
    assert count_dominating_pairs([(1, 1)]) == 0
    assert count_dominating_pairs([(3, 1), (2, 2)]) == 0
    assert count_dominating_pairs([(2, 2), (3, 1)]) == 0
    assert count_dominating_pairs([(5, 5), (3, 4), (1, 2)]) == 3
    assert count_dominating_pairs([(1, 1), (2, 2), (3, 3)]) == 0
    assert count_dominating_pairs([(3, 3), (3, 2)]) == 0        # equal x is not ">"
    assert count_dominating_pairs([(4, 4), (3, 4)]) == 0        # equal y is not ">"
    decreasing = [(7 - i, 9 - i) for i in range(5)]
    assert count_dominating_pairs(decreasing) == 10             # all C(5,2) pairs count
    assert count_dominating_pairs([(5, 1), (4, 4), (6, 2), (3, 3)]) == 1

    local = 424242
    for trial in range(40):
        local = (local * 75 + 74) % 65537
        n = 2 + local % 59
        points = []
        for _ in range(n):
            local = (local * 75 + 74) % 65537
            x = local % 11 - 5
            local = (local * 75 + 74) % 65537
            y = local % 11 - 5
            points.append((x, y))
        got = count_dominating_pairs(points)
        expected = brute_count(points)
        assert got == expected, (trial, n, got, expected)

    # all-equal points produce no strict dominance at all
    assert count_dominating_pairs([(7, 7)] * 12) == 0


if __name__ == "__main__":
    main()
    print("PASS")
