"""Task 288 - Generate all balanced parentheses strings. (Tier: Intermediate, CTCI Ch 8)

Open-first DFS: at each step add "(" if some pair is still unopened, then ")" if the
closing count is below the opening count. This emits the strings in lexicographic order
("(" < ")"), so results are deterministic. Counts must equal the Catalan numbers and
every string is validated with an independent balance checker.
"""

def generate_parentheses(n):
    """All balanced strings with n pairs, in lexicographic order."""
    if n < 0:
        return []
    results = []

    def build(current, opens, closes):
        if len(current) == 2 * n:
            results.append(current)
            return
        if opens < n:
            build(current + "(", opens + 1, closes)
        if closes < opens:
            build(current + ")", opens, closes + 1)

    build("", 0, 0)
    return results


def is_balanced(text):
    """Independent checker: only parentheses, no negative prefix, ends at depth 0."""
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
        else:
            return False
    return depth == 0


def catalan(n):
    """C(0)=1, C(n+1) = C(n) * 2 * (2n + 1) / (n + 2)."""
    value = 1
    for i in range(n):
        value = value * 2 * (2 * i + 1) // (i + 2)
    return value


def brute_force_balanced(n):
    """Enumerates all 2^(2n) parenthesis strings and keeps the balanced ones."""
    results = []
    for mask in range(1 << (2 * n)):
        text = "".join("(" if (mask >> bit) & 1 else ")" for bit in range(2 * n))
        if is_balanced(text):
            results.append(text)
    return sorted(results)


def main():
    assert generate_parentheses(0) == [""]
    assert generate_parentheses(1) == ["()"]
    assert generate_parentheses(2) == ["(())", "()()"]
    assert generate_parentheses(3) == ["((()))", "(()())", "(())()", "()(())", "()()()"]
    assert generate_parentheses(-1) == []

    for n in range(0, 11):
        strings = generate_parentheses(n)
        assert len(strings) == catalan(n), n
        assert strings == sorted(strings), n              # documented lexicographic order
        for text in strings:
            assert len(text) == 2 * n, (n, text)
            assert is_balanced(text), (n, text)
            assert text.count("(") == n and text.count(")") == n, (n, text)
        assert generate_parentheses(n) == strings         # deterministic

    for n in range(0, 8):
        assert generate_parentheses(n) == brute_force_balanced(n), n

    assert catalan(0) == 1 and catalan(1) == 1 and catalan(2) == 2 and catalan(3) == 5
    assert catalan(4) == 14 and catalan(5) == 42 and catalan(8) == 1430 and catalan(10) == 16796


if __name__ == "__main__":
    main()
    print("PASS")
