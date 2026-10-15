"""Task 222 - Stock buy/sell single transaction max profit. (Tier: Intermediate, CTCI Ch 8)

One buy and one sell with the buy before the sell. Track the cheapest price seen
so far and the best profit from selling today at every step (this is Kadane's
idea applied to daily changes). No transaction is possible in fewer than two
days, so [] and single-day inputs return 0. O(n) time, O(1) space.
"""


def max_profit_single(prices):
    best = 0
    cheapest = float("inf")
    for p in prices:
        cheapest = min(cheapest, p)
        best = max(best, p - cheapest)
    return best


def main():
    assert max_profit_single([7, 1, 5, 3, 6, 4]) == 5     # buy 1, sell 6
    assert max_profit_single([7, 6, 4, 3, 1]) == 0        # only falls
    assert max_profit_single([1]) == 0                     # single day
    assert max_profit_single([]) == 0                      # no days
    assert max_profit_single([1, 2]) == 1
    assert max_profit_single([2, 1]) == 0
    assert max_profit_single([3, 3]) == 0                  # flat
    assert max_profit_single([5, 4, 6, 2, 7]) == 5        # buy 2, sell 7


if __name__ == "__main__":
    main()
    print("PASS")
