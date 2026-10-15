"""Task 041 - Check if brackets are balanced (single type). (Tier: Beginner, CTCI Ch 3)"""

def is_balanced(s):
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def main():
    assert is_balanced("((()))")
    assert is_balanced("()()()")
    assert is_balanced("")
    assert not is_balanced("(()")
    assert not is_balanced(")(")


if __name__ == "__main__":
    main()
    print("PASS")
