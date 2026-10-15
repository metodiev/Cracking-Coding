"""Task 223 - Stock buy/sell multiple transactions (greedy). (Tier: Intermediate, CTCI Ch 8)

Unlimited transactions with the constraint that you must sell before buying
again. Every rise between consecutive days can be captured by buying the day
before and selling the day after (or merging adjacent trades), so the optimum is
simply the sum of all positive consecutive gains. O(n) time, O(1) space.
Empty and single-day inputs return 0.
"""


def max_profit_multiple(prices):
    total = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            total += prices[i] - prices[i - 1]
    return total


def main():
    assert max_profit_multiple([7, 1, 5, 3, 6, 4]) == 7   # (5-1) + (6-3)
    assert max_profit_multiple([7, 6, 4, 3, 1]) == 0      # monotone decreasing
    assert max_profit_multiple([1, 2, 3, 4, 5]) == 4      # one long rise
    assert max_profit_multiple([1, 3, 2, 8]) == 8         # +2 then +6
    assert max_profit_multiple([]) == 0
    assert max_profit_multiple([5]) == 0
    assert max_profit_multiple([3, 3]) == 0


if __name__ == "__main__":
    main()
    print("PASS")
