"""Task 011 - Generate first N Fibonacci numbers (iterative).  (Tier: Beginner, CTCI Ch 8)"""


def fib_sequence(n):
    seq, a, b = [], 0, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


def main():
    assert fib_sequence(0) == []
    assert fib_sequence(1) == [0]
    assert fib_sequence(8) == [0, 1, 1, 2, 3, 5, 8, 13]
    print("PASS")


if __name__ == "__main__":
    main()
