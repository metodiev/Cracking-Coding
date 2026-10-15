"""Task 188 - Implement stable sort wrapper around existing unstable sort. (Tier: Beginner, CTCI Beyond — Software Engineering & Tooling)

Python's built-in sort is already stable, so the *unstable* black box here is a
hand-rolled Lomuto quicksort: its strict "<" comparison lets records with equal
keys swap places. The general fix — usable with any unstable sort — is the
decorate-sort-undecorate trick: sort (key, original_index) pairs, which makes
every comparison key unique, then drop the decoration.
"""


def unstable_sort(items, key):
    """Quicksort `items` by `key`; equal keys may end up in any order."""
    result = list(items)

    def quick(lo, hi):
        while lo < hi:
            pivot_key = key(result[hi])
            store = lo
            for j in range(lo, hi):
                if key(result[j]) < pivot_key:
                    result[store], result[j] = result[j], result[store]
                    store += 1
            result[store], result[hi] = result[hi], result[store]
            quick(lo, store - 1)
            lo = store + 1

    if result:
        quick(0, len(result) - 1)
    return result


def stable_sort(items, key):
    """Stable sort by `key`, built on the unstable black box above."""
    decorated = [(key(item), index, item) for index, item in enumerate(items)]
    ordered = unstable_sort(decorated, key=lambda entry: (entry[0], entry[1]))
    return [item for _, _, item in ordered]


def main():
    records = [(3, "a"), (1, "x"), (3, "b"), (2, "y"), (3, "c"), (1, "w"), (2, "z")]
    key = lambda record: record[0]

    stable = stable_sort(records, key)
    assert stable == [(1, "x"), (1, "w"), (2, "y"), (2, "z"), (3, "a"), (3, "b"), (3, "c")]
    assert records == [(3, "a"), (1, "x"), (3, "b"), (2, "y"), (3, "c"), (1, "w"), (2, "z")]

    # the black box only promises key order, not tie order; it must not lose records
    raw = unstable_sort(records, key)
    assert [record[0] for record in raw] == [record[0] for record in stable]
    assert sorted(raw) == sorted(records)

    # one distinct key: the stable wrapper keeps the original order exactly
    same_keys = [(1, "p"), (1, "q"), (1, "r"), (1, "s")]
    assert stable_sort(same_keys, key) == same_keys
    assert stable_sort([], key) == []
    assert stable_sort([(9, "only")], key) == [(9, "only")]

    # negative keys work the same way
    mixed = [(0, "p"), (-1, "q"), (0, "r"), (-1, "s")]
    assert stable_sort(mixed, key) == [(-1, "q"), (-1, "s"), (0, "p"), (0, "r")]

    # larger pseudo-random batch: equal keys must stay in their original order
    import random
    rng = random.Random(41)
    batch = [(rng.randrange(5), "tag-%03d" % i) for i in range(120)]
    ordered = stable_sort(batch, key)
    assert [record[0] for record in ordered] == sorted(record[0] for record in batch)
    assert sorted(ordered) == sorted(batch)
    for value in range(5):
        group = [record for record in ordered if record[0] == value]
        original = [record for record in batch if record[0] == value]
        assert group == original, value


if __name__ == "__main__":
    main()
    print("PASS")
