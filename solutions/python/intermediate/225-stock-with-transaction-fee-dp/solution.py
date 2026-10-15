"""Task 225 - Stock with transaction fee (DP). (Tier: Intermediate, CTCI Ch 8)

Unlimited transactions, each sale pays a fixed fee. Two states: cash (not
holding) and hold (holding). Transitions:
cash' = max(cash, hold + price - fee), hold' = max(hold, cash - price).
Answer is cash after the last day. With fee 0 this equals the greedy answer of
task 223 (sum of positive consecutive gains). O(n) time, O(1) space.
A negative fee raises; empty and single-day inputs return 0.
"""


def max_profit_fee(prices, fee):
    if fee < 0:
        raise ValueError("fee must be non-negative")
    if not prices:
        return 0
    cash, hold = 0, -prices[0]
    for p in prices[1:]:
        hold, cash = max(hold, cash - p), max(cash, hold + p - fee)
    return cash


def main():
    assert max_profit_fee([1, 3, 2, 8, 4, 9], 2) == 8     # (8-1-2) + (9-4-2)
    assert max_profit_fee([1, 3, 2, 8, 4, 9], 0) == 13    # matches greedy 223
    assert max_profit_fee([1, 2], 5) == 0                 # fee eats the gain
    assert max_profit_fee([1, 2], 0) == 1
    assert max_profit_fee([1, 4], 1) == 2
    assert max_profit_fee([], 2) == 0
    assert max_profit_fee([5], 1) == 0
    assert max_profit_fee([5], 0) == 0
    try:
        max_profit_fee([1, 2], -1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
