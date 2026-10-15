
"""Task 126 - House robber problem (1D DP). (Tier: Beginner, CTCI Ch 8)

Cannot rob two adjacent houses. take = best ending here if robbed,
skip = best so far not using the previous house.
"""


def rob(houses):
    take = skip = 0
    for value in houses:
        take, skip = skip + value, max(skip, take)
    return max(take, skip)


def rob_with_plan(houses):
    """Return (best value, chosen indices) using f[i] = max(f[i-1], f[i-2] + v[i])."""
    n = len(houses)
    if n == 0:
        return 0, []
    f = [0] * n
    for i in range(n):
        take = houses[i] + (f[i - 2] if i >= 2 else 0)
        f[i] = max(f[i - 1], take) if i > 0 else take
    plan = []
    i = n - 1
    while i >= 0:
        if i == 0 or f[i] != f[i - 1]:
            plan.append(i)          # house i is robbed
            i -= 2
        else:
            i -= 1
    return f[-1], sorted(plan)


def main():
    assert rob([2, 7, 9, 3, 1]) == 12              # 2 + 9 + 1
    assert rob([]) == 0
    assert rob([5]) == 5
    assert rob([2, 1, 1, 2]) == 4                 # ends
    best, plan = rob_with_plan([2, 7, 9, 3, 1])
    assert best == 12
    values = sum([2, 7, 9, 3, 1][i] for i in plan)
    assert values == 12
    assert all(plan[i + 1] - plan[i] > 1 for i in range(len(plan) - 1)), plan


if __name__ == "__main__":
    main()
    print("PASS")
