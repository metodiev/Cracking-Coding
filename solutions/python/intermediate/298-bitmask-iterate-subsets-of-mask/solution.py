"""Task 298 - Bitmask: iterate subsets of a mask. (Tier: Intermediate, CTCI Ch 5)

The classic `for (sub = mask; sub; sub = (sub - 1) & mask)` enumeration visits the
submasks of mask in strictly descending numeric order, mask first; the loop emitted
here additionally yields the final 0 (the empty submask), so there are exactly
2^popcount(mask) results. Each step is O(1): the trick `(sub - 1) & mask` clears the
lowest set bit and refills the lower bits with the mask's own bits, skipping over every
number that is not a submask. Hence O(2^k) for one mask with k bits set.
"""


def enumerate_submasks(mask):
    """All submasks of mask in descending order, including 0 (mask == 0 -> [0])."""
    result = []
    sub = mask
    while True:
        result.append(sub)
        if sub == 0:
            break
        sub = (sub - 1) & mask
    return result


def popcount(value):
    return bin(value).count("1")


def main():
    assert enumerate_submasks(0) == [0]
    assert enumerate_submasks(1) == [1, 0]
    assert enumerate_submasks(0b1011) == [11, 10, 9, 8, 3, 2, 1, 0]
    assert enumerate_submasks(0b11111) == list(range(31, -1, -1))

    masks = [0, 1, 0b1011, 0b11111, 0b1000, 0b101010]
    seed = 9090
    for _ in range(40):
        seed = (seed * 75 + 74) % 65537
        masks.append(seed % 4096)
    for mask in masks:
        submasks = enumerate_submasks(mask)
        assert len(submasks) == 2 ** popcount(mask), mask
        assert len(set(submasks)) == len(submasks), mask
        assert all(sub & ~mask == 0 for sub in submasks), mask
        assert submasks[0] == mask and submasks[-1] == 0, mask
        assert all(submasks[i] > submasks[i + 1] for i in range(len(submasks) - 1)), mask
        # independent oracle: descending sweep over the integers below mask
        assert submasks == [value for value in range(mask, -1, -1) if value & ~mask == 0], mask


if __name__ == "__main__":
    main()
    print("PASS")
