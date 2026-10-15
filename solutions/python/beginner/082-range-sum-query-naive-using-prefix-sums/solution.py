"""Task 082 - Range sum query naive using prefix sums. (Tier: Beginner, CTCI Ch 1)

Build the prefix sums once in O(n); then every inclusive range_sum(lo, hi) query
is O(1) via prefix[hi + 1] - prefix[lo]. Invalid ranges (lo < 0, hi >= n, lo > hi)
raise ValueError.
"""


class RangeSumQuery:
    def __init__(self, nums):
        self.prefix = [0] * (len(nums) + 1)
        for i, value in enumerate(nums):
            self.prefix[i + 1] = self.prefix[i] + value

    def range_sum(self, lo, hi):
        n = len(self.prefix) - 1
        if lo < 0 or hi >= n or lo > hi:
            raise ValueError(f"invalid range [{lo}, {hi}] for n={n}")
        return self.prefix[hi + 1] - self.prefix[lo]


def expect_value_error(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError: " + label)


def main():
    rs = RangeSumQuery([3, 1, 4, 1, 5])
    assert rs.range_sum(0, 4) == 14
    assert rs.range_sum(1, 3) == 6
    assert rs.range_sum(2, 2) == 4
    assert rs.range_sum(0, 0) == 3
    assert rs.range_sum(4, 4) == 5
    assert RangeSumQuery([7]).range_sum(0, 0) == 7

    expect_value_error(lambda: rs.range_sum(-1, 2), "lo < 0")
    expect_value_error(lambda: rs.range_sum(2, 1), "lo > hi")
    expect_value_error(lambda: rs.range_sum(1, 5), "hi >= n")
    expect_value_error(lambda: RangeSumQuery([]).range_sum(0, 0), "empty array")


if __name__ == "__main__":
    main()
    print("PASS")
