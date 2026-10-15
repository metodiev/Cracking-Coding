"""Task 445 - Implement fractional cascading for multiple binary searches. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Given k sorted lists, answer "successor of x in every list" with one binary search plus O(k)
O(1) hops. Bridge construction: B[k-1] = A[k-1]; for i = k-2..0, B[i] = merge(A[i], every
second element of B[i+1]) (foreign copies keep the total size linear), and bridge[i][p] is the
first position q in B[i+1] with B[i+1][q] >= B[i][p]. After the single binary search in B[0],
the successor position in B[i] is bridge[i-1][p] or bridge[i-1][p] - 1 (one comparison): the
copied "every second" elements can never leave two survivors between bridge targets. Each
augmented list also stores next_native[p], the first native element at or after p, so the
successor value in A[i] is read in O(1). Total: O(N) build, O(log N + k) per query.
Documents the standard distinct-value presentation; every result is checked against an
independent bisect_left on each original list.
"""

from bisect import bisect_left


class FractionalCascade:
    def __init__(self, lists):
        k = len(lists)
        self.augmented = [None] * k
        self.native = [None] * k
        self.augmented[k - 1] = list(lists[k - 1])
        self.native[k - 1] = [True] * len(self.augmented[k - 1])
        for i in range(k - 2, -1, -1):
            copied = self.augmented[i + 1][0::2]
            self.augmented[i], self.native[i] = self._merge(lists[i], copied)
        self.bridge = [None] * k
        for i in range(k - 1):
            upper = self.augmented[i + 1]
            bridge = [0] * len(self.augmented[i])
            j = 0
            for p, value in enumerate(self.augmented[i]):
                while j < len(upper) and upper[j] < value:
                    j += 1
                bridge[p] = j
            self.bridge[i] = bridge
        self.next_native = []
        for i in range(k):
            flags = self.native[i]
            size = len(flags)
            nxt = [size] * (size + 1)
            for p in range(size - 1, -1, -1):
                nxt[p] = p if flags[p] else nxt[p + 1]
            self.next_native.append(nxt)

    @staticmethod
    def _merge(native_list, copies):
        values = []
        flags = []
        p = 0
        q = 0
        while p < len(native_list) or q < len(copies):
            if q == len(copies) or (p < len(native_list) and native_list[p] <= copies[q]):
                values.append(native_list[p])
                flags.append(True)
                p += 1
            else:
                values.append(copies[q])
                flags.append(False)
                q += 1
        return values, flags

    def successors(self, x):
        """Successor of x in every original list (None when x exceeds the whole list)."""
        results = [None] * len(self.augmented)
        pos = bisect_left(self.augmented[0], x)
        for i in range(len(self.augmented)):
            if i > 0:
                if pos == len(self.augmented[i - 1]):
                    # x is above all of B[i-1]; since its even positions are exactly the
                    # copies living in B[i-1] they are all < x too, so the only candidate left
                    # in B[i] is its final element (the survivors below are all < x otherwise).
                    m = len(self.augmented[i])
                    pos = m - 1 if m > 0 and self.augmented[i][m - 1] >= x else m
                else:
                    q = self.bridge[i - 1][pos]
                    if q > 0 and self.augmented[i][q - 1] >= x:
                        q -= 1
                    pos = q
            nxt = self.next_native[i][pos]
            results[i] = self.augmented[i][nxt] if nxt < len(self.augmented[i]) else None
        return results


def lcg_lists(count, size, seed, step_max=9):
    """Each list is a strictly increasing sample, so values are distinct across all lists."""
    lists = []
    for _ in range(count):
        values = []
        current = -40
        for _ in range(size):
            seed = (seed * 75 + 74) % 65537
            current += 1 + seed % step_max
            values.append(current)
        lists.append(values)
    return lists


def independent_successors(lists, x):
    results = []
    for values in lists:
        pos = bisect_left(values, x)
        results.append(values[pos] if pos < len(values) else None)
    return results


def main():
    lists = [[1, 10], [5, 20], [3, 7, 30]]
    cascade = FractionalCascade(lists)
    assert cascade.augmented[0] == [1, 3, 10, 20]
    assert cascade.augmented[1] == [3, 5, 20, 30]
    assert cascade.augmented[2] == [3, 7, 30]
    assert cascade.successors(6) == [10, 20, 7]
    assert cascade.successors(3) == [10, 5, 3]
    assert cascade.successors(0) == [1, 5, 3]
    assert cascade.successors(10) == [10, 20, 30]
    assert cascade.successors(31) == [None, None, None]
    assert cascade.successors(20) == [None, 20, 30]

    lists = lcg_lists(5, 12, 445)
    cascade = FractionalCascade(lists)
    for i in range(4):
        expected = len(lists[i]) + -(-len(cascade.augmented[i + 1]) // 2)
        assert len(cascade.augmented[i]) == expected

    seed = 555
    for _ in range(200):
        seed = (seed * 75 + 74) % 65537
        x = seed % 200 - 90
        assert cascade.successors(x) == independent_successors(lists, x), x

    for values in lcg_lists(3, 5, 616):
        cascade = FractionalCascade([values])
        seed = 77
        for _ in range(40):
            seed = (seed * 75 + 74) % 65537
            x = seed % 120 - 60
            assert cascade.successors(x) == independent_successors([values], x)


if __name__ == "__main__":
    main()
    print("PASS")
