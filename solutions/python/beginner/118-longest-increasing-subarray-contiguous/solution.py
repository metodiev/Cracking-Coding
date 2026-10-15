
"""Task 118 - Longest increasing subarray (contiguous). (Tier: Beginner, CTCI Ch 8)

O(n) single pass: extend the current run while strictly increasing.
"""


def longest_increasing_subarray(a):
    """Returns (length, start_index); start is -1 for an empty array."""
    if not a:
        return 0, -1
    best_len, best_start = 1, 0
    cur_len, cur_start = 1, 0
    for i in range(1, len(a)):
        if a[i] > a[i - 1]:
            cur_len += 1
        else:
            cur_len, cur_start = 1, i
        if cur_len > best_len:
            best_len, best_start = cur_len, cur_start
    return best_len, best_start


def main():
    assert longest_increasing_subarray([1, 2, 3, 1, 2]) == (3, 0)
    assert longest_increasing_subarray([5, 4, 3]) == (1, 0)
    assert longest_increasing_subarray([1, 3, 2, 4, 5, 6]) == (4, 2)
    assert longest_increasing_subarray([]) == (0, -1)
    assert longest_increasing_subarray([7]) == (1, 0)


if __name__ == "__main__":
    main()
    print("PASS")
