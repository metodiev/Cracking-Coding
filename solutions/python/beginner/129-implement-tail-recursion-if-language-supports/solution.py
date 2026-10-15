
"""Task 129 - Implement tail recursion (if language supports). (Tier: Beginner, CTCI Ch 8)

None of Python / Java / JavaScript / C# guarantee tail-call optimization. This file
writes factorial in tail-recursive *shape* (accumulator, recursive call last) and
adds a trampoline for stack safety on deep inputs.
"""


def factorial_tail(n, acc=1):
    """Tail-recursive shape; recursion depth = n (fine only for small n)."""
    return acc if n <= 1 else factorial_tail(n - 1, n * acc)


def factorial_iterative(n):
    acc = 1
    for i in range(2, n + 1):
        acc *= i
    return acc


def trampoline(fn, *args):
    """Drive a thunk-returning function until it yields a plain value."""
    result = fn(*args)
    while callable(result):
        result = result()
    return result


def factorial_trampolined(n):
    """Stack-safe: each step returns a zero-arg thunk; the loop unwinds it."""

    def step(remaining, acc):
        return acc if remaining <= 1 else (lambda: step(remaining - 1, acc * remaining))

    return trampoline(step, n, 1)


def main():
    assert factorial_tail(5) == 120
    assert factorial_tail(20) == 2432902008176640000

    # deep "tail" recursion exceeds Python's stack; the trampoline stays at O(1) frames
    assert factorial_trampolined(5000) == factorial_iterative(5000)
    assert factorial_trampolined(20000) == factorial_iterative(20000)

    raised = False
    try:
        factorial_tail(20000)                 # > default recursion limit (1000)
    except RecursionError:
        raised = True
    assert raised, "plain recursion is not stack safe at 20000 deep"


if __name__ == "__main__":
    main()
    print("PASS")
