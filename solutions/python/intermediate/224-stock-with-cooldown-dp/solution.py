"""Task 224 - Stock with cooldown (DP). (Tier: Intermediate, CTCI Ch 8)

Unlimited transactions, but after selling you must skip the next day. Three
states per day: held (owning a share), sold (sold today, tomorrow is forced
rest), rest (idle, including a finished cooldown). Transitions:
held' = max(held, rest - price), sold' = held + price, rest' = max(rest, sold).
Answer max(sold, rest) after the last day. O(n) time, O(1) space.
"""


def max_profit_cooldown(prices):
    if not prices:
        return 0
    held = -prices[0]                 # bought on day 0
    sold = float("-inf")              # no sale yet
    rest = 0                          # idle on day 0
    for p in prices[1:]:
        held, sold, rest = max(held, rest - p), held + p, max(rest, sold)
    return max(sold, rest)


def main():
    assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3      # buy 1/sell 2, rest, buy 0/sell 2
    assert max_profit_cooldown([1]) == 0
    assert max_profit_cooldown([]) == 0
    assert max_profit_cooldown([2, 1, 4]) == 3            # buy 1, sell 4
    assert max_profit_cooldown([6, 1, 3, 2, 4, 7]) == 6   # hold through, one sale
    assert max_profit_cooldown([1, 2]) == 1
    assert max_profit_cooldown([2, 2, 2]) == 0


if __name__ == "__main__":
    main()
    print("PASS")
