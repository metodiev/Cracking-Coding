"""Task 345 - RandomizedSet: insert/delete/getRandom in O(1). (Tier: Intermediate, CTCI Ch 7)

Array + value -> index map.  Removal swaps the victim with the last element, so
insert/remove are O(1) expected and the array has no holes.  getRandom must be
deterministic for the test suite, so instead of an unseeded RNG the class draws
from the standard repo LCG (seed = (seed*75+74) % 65537); the exact draw sequence
after the documented operation script is therefore identical in all four
languages and is asserted literally below (oracle run, cross-checked).
"""


class RandomizedSet:
    def __init__(self, seed=2024):
        self._items = []
        self._index = {}
        self._rng = seed

    def insert(self, value):
        if value in self._index:
            return False
        self._index[value] = len(self._items)
        self._items.append(value)
        return True

    def remove(self, value):
        if value not in self._index:
            return False
        i = self._index[value]
        last = self._items[-1]
        self._items[i] = last                 # move the last element into the hole
        self._index[last] = i
        self._items.pop()
        del self._index[value]
        return True

    def get_random(self):
        if not self._items:
            raise ValueError("getRandom on an empty set")
        self._rng = (self._rng * 75 + 74) % 65537
        return self._items[self._rng % len(self._items)]

    def size(self):
        return len(self._items)

    def to_list(self):
        """Internal layout, for assertions only."""
        return list(self._items)

    def validate(self):
        """Array and map must agree (O(1) invariant kept intact)."""
        if len(self._items) != len(self._index):
            return False
        for value, i in self._index.items():
            if self._items[i] != value:
                return False
        return True


def main():
    random_set = RandomizedSet(2024)
    assert random_set.size() == 0

    # fixed operation script; duplicates are rejected, missing removes return False
    for value, expected in [(5, True), (3, True), (9, True), (1, True), (7, True), (5, False)]:
        assert random_set.insert(value) is expected, value
    assert random_set.remove(3) is True
    assert random_set.remove(42) is False
    assert random_set.insert(3) is True
    assert random_set.to_list() == [5, 7, 9, 1, 3]      # swap-remove layout
    assert random_set.size() == 5
    assert random_set.validate()

    # deterministic LCG draws - this exact sequence must hold in every language
    assert [random_set.get_random() for _ in range(8)] == [5, 1, 3, 3, 3, 1, 1, 1]

    # an empty set documents a raising getRandom
    empty = RandomizedSet(1)
    try:
        empty.get_random()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError from getRandom on an empty set")

    # bulk: 1..50, duplicates rejected, evens removed, only odds remain
    bulk = RandomizedSet(98765)
    for value in range(1, 51):
        assert bulk.insert(value) is True
    for value in range(1, 51):
        assert bulk.insert(value) is False
    for value in range(2, 51, 2):
        assert bulk.remove(value) is True
    for value in range(2, 51, 2):
        assert bulk.remove(value) is False
    assert bulk.size() == 25
    assert sorted(bulk.to_list()) == list(range(1, 51, 2))
    assert bulk.validate()
    odds = set(range(1, 51, 2))
    draws = [bulk.get_random() for _ in range(500)]
    assert all(draw in odds for draw in draws)
    assert set(draws) == odds                          # 500 deterministic draws cover all odds

    # exercise removing the swap victim and the last element
    small = RandomizedSet(7)
    for value in (10, 20, 30):
        small.insert(value)
    assert small.remove(10) is True                    # 30 moves into the first slot
    assert small.to_list() == [30, 20]
    assert small.remove(20) is True                    # last element removed
    assert small.to_list() == [30]
    assert small.remove(30) is True
    assert small.to_list() == []
    assert small.validate()


if __name__ == "__main__":
    main()
    print("PASS")
