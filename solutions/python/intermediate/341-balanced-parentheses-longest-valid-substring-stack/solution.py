"""Task 341 - Longest valid parentheses substring (stack of indices). (Tier: Intermediate, CTCI Ch 8)

Two independent implementations are cross-checked on every input:
  * stack of indices - push '(' positions; on ')' pop and measure against the new
    top, which is the position just before the current valid run;
  * DP - dp[i] = length of the longest valid substring ending exactly at i.
A brute-force oracle tries every substring for the random cross-checks.
"""


def longest_valid_substring_stack(s):
    stack = [-1]                       # base: position of the last unmatched ')'
    best = 0
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)        # this ')' becomes the new base
            else:
                best = max(best, i - stack[-1])
    return best


def longest_valid_substring_dp(s):
    dp = [0] * len(s)
    best = 0
    for i, ch in enumerate(s):
        if ch != ")":
            continue
        if i >= 1 and s[i - 1] == "(":
            dp[i] = (dp[i - 2] if i >= 2 else 0) + 2
        elif i >= 1 and dp[i - 1] > 0:
            j = i - dp[i - 1] - 1      # character in front of the previous run
            if j >= 0 and s[j] == "(":
                dp[i] = dp[i - 1] + 2 + (dp[j - 1] if j >= 1 else 0)
        best = max(best, dp[i])
    return best


def longest_valid_substring_brute(s):
    """Exhaustive oracle: every substring that stays balanced and returns to 0."""
    best = 0
    for i in range(len(s)):
        depth = 0
        for j in range(i, len(s)):
            depth += 1 if s[j] == "(" else -1
            if depth < 0:
                break
            if depth == 0:
                best = max(best, j - i + 1)
    return best


def lcg_strings(count, min_len, max_len, seed=34142):
    strings = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        length = min_len + seed % (max_len - min_len + 1)
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append("()"[seed % 2])
        strings.append("".join(chars))
    return strings


def main():
    # examples from the brief
    assert longest_valid_substring_stack("(()") == 2
    assert longest_valid_substring_stack(")()())") == 4
    assert longest_valid_substring_stack("") == 0
    assert longest_valid_substring_stack("()(())") == 6

    # hand-checked extras
    assert longest_valid_substring_stack("()(()") == 2
    assert longest_valid_substring_stack("((())") == 4
    assert longest_valid_substring_stack("())") == 2
    assert longest_valid_substring_stack(")(") == 0
    assert longest_valid_substring_stack("(((") == 0
    assert longest_valid_substring_stack("()") == 2
    assert longest_valid_substring_stack("()()") == 4
    assert longest_valid_substring_stack(")()(") == 2

    # the DP variant must agree with the stack implementation
    for s in ["(()", ")()())", "", "()(())", "()(()", "((())", "())", ")(", "(((", "()", "()()", ")()("]:
        assert longest_valid_substring_dp(s) == longest_valid_substring_stack(s), s
        assert longest_valid_substring_brute(s) == longest_valid_substring_stack(s), s

    # random LCG strings: all three implementations agree
    for s in lcg_strings(150, 0, 24):
        expected = longest_valid_substring_brute(s)
        assert longest_valid_substring_stack(s) == expected, s
        assert longest_valid_substring_dp(s) == expected, s


if __name__ == "__main__":
    main()
    print("PASS")
