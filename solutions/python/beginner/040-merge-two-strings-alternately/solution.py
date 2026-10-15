"""Task 040 - Merge two strings alternately. (Tier: Beginner, CTCI Ch 1)"""

def merge_alternately(a, b):
    parts = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            parts.append(a[i])
        if i < len(b):
            parts.append(b[i])
    return "".join(parts)


def main():
    assert merge_alternately("abc", "123") == "a1b2c3"
    assert merge_alternately("ab", "1234") == "a1b2" + "34"
    assert merge_alternately("", "xyz") == "xyz"


if __name__ == "__main__":
    main()
    print("PASS")
