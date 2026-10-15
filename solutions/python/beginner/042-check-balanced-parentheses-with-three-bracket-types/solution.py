"""Task 042 - Check balanced parentheses with three bracket types. (Tier: Beginner, CTCI Ch 3)"""

def is_balanced(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def main():
    assert is_balanced("{[()]}")
    assert is_balanced("()[]{}")
    assert not is_balanced("{[(])}")
    assert not is_balanced("(")
    assert not is_balanced(")")


if __name__ == "__main__":
    main()
    print("PASS")
