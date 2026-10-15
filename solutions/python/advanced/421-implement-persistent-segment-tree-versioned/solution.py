"""Task 421 - Implement persistent segment tree (versioned). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Every point update returns a NEW version that reuses all untouched nodes, so old versions
stay intact and queryable. Nodes live in flat arrays (left, right, total); roots[i] is the
root of version i. Range sums per version are cross-checked against snapshot copies of a
plain array over LCG-generated update/query sequences, including branch-offs from old
versions.
"""


class PersistentSegTree:
    """Persistent point-update segment tree over the fixed index domain [0, n)."""

    def __init__(self, values):
        self.n = len(values)
        self.left = [0]                    # node 0 acts as the null node
        self.right = [0]
        self.total = [0]
        self.roots = [self._build(values, 0, self.n - 1)]

    def version_count(self):
        return len(self.roots)

    def _node(self, left, right, total):
        self.left.append(left)
        self.right.append(right)
        self.total.append(total)
        return len(self.total) - 1

    def _build(self, values, lo, hi):
        if lo == hi:
            return self._node(0, 0, values[lo])
        mid = (lo + hi) // 2
        left = self._build(values, lo, mid)
        right = self._build(values, mid + 1, hi)
        return self._node(left, right, self.total[left] + self.total[right])

    def update(self, version, pos, value):
        root = self._update(self.roots[version], 0, self.n - 1, pos, value)
        self.roots.append(root)
        return len(self.roots) - 1

    def _update(self, node, lo, hi, pos, value):
        if lo == hi:
            return self._node(0, 0, value)
        mid = (lo + hi) // 2
        if pos <= mid:
            left = self._update(self.left[node], lo, mid, pos, value)
            right = self.right[node]
        else:
            left = self.left[node]
            right = self._update(self.right[node], mid + 1, hi, pos, value)
        return self._node(left, right, self.total[left] + self.total[right])

    def range_sum(self, version, lo, hi):
        return self._range_sum(self.roots[version], 0, self.n - 1, lo, hi)

    def _range_sum(self, node, lo, hi, ql, qr):
        if node == 0 or qr < lo or hi < ql:
            return 0
        if ql <= lo and hi <= qr:
            return self.total[node]
        mid = (lo + hi) // 2
        return (self._range_sum(self.left[node], lo, mid, ql, qr)
                + self._range_sum(self.right[node], mid + 1, hi, ql, qr))


def lcg_array(n, seed):
    values = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % 41 - 20)
    return values


def main():
    st = PersistentSegTree([1, 2, 3, 4, 5])
    assert st.version_count() == 1
    assert st.range_sum(0, 0, 4) == 15
    assert st.range_sum(0, 1, 3) == 9
    assert st.range_sum(0, 2, 2) == 3
    assert st.range_sum(0, 0, 0) == 1

    v1 = st.update(0, 2, 10)                        # [1, 2, 10, 4, 5]
    assert v1 == 1
    assert st.range_sum(1, 0, 4) == 22
    assert st.range_sum(1, 2, 2) == 10
    assert st.range_sum(1, 1, 3) == 16
    assert st.range_sum(0, 0, 4) == 15              # version 0 is untouched
    assert st.range_sum(0, 2, 2) == 3

    v2 = st.update(1, 0, -1)                        # [-1, 2, 10, 4, 5]
    assert v2 == 2
    assert st.range_sum(2, 0, 4) == 20
    assert st.range_sum(2, 0, 2) == 11
    assert st.range_sum(1, 0, 2) == 13

    v3 = st.update(0, 4, 7)                         # branch off OLD version: [1, 2, 3, 4, 7]
    assert v3 == 3
    assert st.range_sum(3, 0, 4) == 17
    assert st.range_sum(3, 3, 4) == 11
    assert st.range_sum(3, 2, 2) == 3
    assert st.range_sum(2, 0, 4) == 20              # the other branch is unchanged
    assert st.range_sum(1, 0, 4) == 22
    assert st.range_sum(0, 0, 4) == 15

    for seed in (11, 421, 9001):
        n = 1 + seed % 29
        values = lcg_array(n, seed)
        st = PersistentSegTree(values)
        snapshots = [values[:]]
        local = seed
        for _ in range(60):
            local = (local * 75 + 74) % 65537
            if local % 3 == 0:
                version = local % len(snapshots)
                left = local % n
                right = (local // n) % n
                if left > right:
                    left, right = right, left
                expected = sum(snapshots[version][left:right + 1])
                got = st.range_sum(version, left, right)
                assert got == expected, (seed, version, left, right, got, expected)
            else:
                pos = local % n
                local = (local * 75 + 74) % 65537
                value = local % 201 - 100
                version = st.update(st.version_count() - 1, pos, value)
                state = snapshots[-1][:]
                state[pos] = value
                snapshots.append(state)
                assert version == len(snapshots) - 1
                assert st.range_sum(version, 0, n - 1) == sum(state)

    # a fresh tree over a one-element array
    small = PersistentSegTree([42])
    assert small.range_sum(0, 0, 0) == 42
    small.update(0, 0, -8)
    assert small.range_sum(1, 0, 0) == -8
    assert small.range_sum(0, 0, 0) == 42


if __name__ == "__main__":
    main()
    print("PASS")
