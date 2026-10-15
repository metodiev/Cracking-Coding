"""Task 155 - Josephus problem simulation. (Tier: Beginner, CTCI Ch 2)

People numbered 0..n-1 stand in a circle; every k-th person is eliminated
(counting from the current position) until one survives.

* josephus_simulate  - direct array simulation, O(n^2) time, O(n) space.
* josephus_recurrence - bottom-up recurrence J(n,k) = (J(n-1,k) + k) % n,
  starting from J(1,k) = 0; O(n) time, O(1) space.

Both must agree; known checkpoints: J(5,2)=2, J(7,3)=3, J(41,3)=30.
"""


def josephus_simulate(n, k):
    """Eliminate every k-th person in a circle of n; returns the 0-based survivor."""
    if n < 1 or k < 1:
        raise ValueError("n and k must be >= 1")
    people = list(range(n))
    idx = 0
    while len(people) > 1:
        idx = (idx + k - 1) % len(people)    # the k-th person counted from idx
        people.pop(idx)
    return people[0]


def josephus_recurrence(n, k):
    """Bottom-up form of J(n,k) = (J(n-1,k) + k) % n; O(n) time, O(1) space."""
    if n < 1 or k < 1:
        raise ValueError("n and k must be >= 1")
    survivor = 0                             # J(1, k) = 0
    for i in range(2, n + 1):
        survivor = (survivor + k) % i
    return survivor


def main():
    # known checkpoints
    assert josephus_simulate(5, 2) == 2
    assert josephus_simulate(7, 3) == 3
    assert josephus_simulate(41, 3) == 30

    # both variants agree, including the n = 1 base case
    for n in (1, 2, 3, 5, 7, 10, 20, 41):
        for k in (1, 2, 3, 4, 7, 11):
            assert josephus_recurrence(n, k) == josephus_simulate(n, k), (n, k)
    assert josephus_simulate(1, 7) == 0 and josephus_recurrence(1, 999) == 0

    # k = 1 eliminates in order 0, 1, 2, ... so the last index survives
    assert josephus_simulate(10, 1) == 9 == josephus_recurrence(10, 1)

    # direct recurrence values used as independent checkpoints
    assert josephus_recurrence(5, 2) == 2
    assert josephus_recurrence(7, 3) == 3
    assert josephus_recurrence(41, 3) == 30

    for bad in (0, -1):
        try:
            josephus_simulate(bad, 2)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
        try:
            josephus_recurrence(3, bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


if __name__ == "__main__":
    main()
    print("PASS")
