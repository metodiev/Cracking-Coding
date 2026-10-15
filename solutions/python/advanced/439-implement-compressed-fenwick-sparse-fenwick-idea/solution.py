"""Task 439 - Implement compressed Fenwick / sparse Fenwick idea. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A Fenwick tree that is touched at only a few coordinates does not need a dense array: the sparse
version keeps a dict of just the cells an update really visits (i, i + lowbit(i), ...) and treats
every untouched cell as 0 while walking a prefix query (i, i - lowbit(i), ...). Documented setup:
SIZE = 2^20 = 1048576, coordinates 1..SIZE, updates propagate while i <= SIZE. A dense Fenwick of
the same size is the oracle: LCG updates are applied to both and every prefix query must agree,
with a brute-force sweep over the recorded updates as a third opinion. The touched-cell count is
asserted to stay within 21 * (#updates) - at most log2(SIZE) = 20 cells are visited per update -
and far below SIZE, which is the evidence that the representation really is compressed.
"""

SIZE = 1 << 20


class SparseFenwick:
    def __init__(self, size):
        self.size = size
        self.tree = {}

    def add(self, index, delta):
        i = index
        while i <= self.size:
            self.tree[i] = self.tree.get(i, 0) + delta
            i += i & -i

    def prefix(self, index):
        total = 0
        i = index
        while i > 0:
            total += self.tree.get(i, 0)
            i -= i & -i
        return total

    def touched(self):
        return len(self.tree)


class DenseFenwick:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def add(self, index, delta):
        i = index
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, index):
        total = 0
        i = index
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total


def main():
    sparse = SparseFenwick(16)
    dense = DenseFenwick(16)
    sparse.add(1, 5)
    sparse.add(3, 2)
    sparse.add(16, 4)
    dense.add(1, 5)
    dense.add(3, 2)
    dense.add(16, 4)
    assert sparse.prefix(0) == 0
    assert sparse.prefix(1) == 5
    assert sparse.prefix(2) == 5
    assert sparse.prefix(3) == 7
    assert sparse.prefix(15) == 7
    assert sparse.prefix(16) == 11
    assert sparse.touched() == 6, sparse.touched()
    for index in range(0, 17):
        assert sparse.prefix(index) == dense.prefix(index), index

    sparse = SparseFenwick(SIZE)
    dense = DenseFenwick(SIZE)
    assert sparse.prefix(0) == 0 and sparse.prefix(SIZE) == 0
    assert sparse.touched() == 0

    seed = 4390
    recorded = []
    queries = 0
    updates = 0
    for step in range(800):
        seed = (seed * 75 + 74) % 65537
        seed = (seed * 75 + 74) % 65537
        coord = 1 + (seed * 1024 + (seed >> 4)) % (SIZE - 1)
        seed = (seed * 75 + 74) % 65537
        if step % 2 == 0:
            delta = seed % 21 - 10
            sparse.add(coord, delta)
            dense.add(coord, delta)
            recorded.append((coord, delta))
            updates += 1
        else:
            assert sparse.prefix(coord) == dense.prefix(coord), (step, coord)
            queries += 1
    assert updates == 400 and queries == 400

    for coord in (1, 12345, SIZE // 2, SIZE - 1, SIZE):
        expected = sum(delta for position, delta in recorded if position <= coord)
        assert sparse.prefix(coord) == dense.prefix(coord) == expected, coord
    assert sparse.touched() >= 1
    assert sparse.touched() <= 21 * updates, sparse.touched()
    assert sparse.touched() < SIZE // 100, sparse.touched()
    assert sparse.prefix(SIZE) == sum(delta for _, delta in recorded)


if __name__ == "__main__":
    main()
    print("PASS")
