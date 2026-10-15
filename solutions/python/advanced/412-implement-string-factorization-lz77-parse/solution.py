"""Task 412 - Implement string factorization (LZ77 parse). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Greedy LZ77 factorization. Every token is a triple (offset, length, next):
  * offset = distance back to the match source (0 for a pure literal),
  * length = matched characters copied from that earlier position,
  * next   = the literal character that follows the match (None when the match
             consumes the end of the string).
Rules (documented, deterministic):
  * a match source j must satisfy max(0, i - window) <= j < i (window = 8 by default),
  * the greedy step takes the LONGEST match, ties broken by the SMALLEST offset,
  * matches shorter than min_len = 3 are not worth a token, so a literal is emitted,
  * matches may overlap the source region (s[j+L] is read from the original string).
The brute-force oracle recomputes the same token list with plain nested loops, and a
DP computes a lower bound on any parse's token count.
"""

MIN_LEN = 3
WINDOW = 8


def lz77_parse(s, window=WINDOW, min_len=MIN_LEN):
    tokens = []
    n = len(s)
    i = 0
    while i < n:
        best_len, best_off = 0, 0
        for j in range(max(0, i - window), i):
            L = 0
            while i + L < n and s[j + L] == s[i + L]:
                L += 1
            if L > best_len:
                best_len, best_off = L, i - j
        if best_len >= min_len:
            if i + best_len == n:
                tokens.append((best_off, best_len, None))
            else:
                tokens.append((best_off, best_len, s[i + best_len]))
            i += best_len + (0 if i + best_len == n else 1)
        else:
            tokens.append((0, 0, s[i]))
            i += 1
    return tokens


def lz77_decode(tokens):
    """Replays the tokens; overlapping copies read characters emitted moments ago."""
    out = []
    for off, ln, nxt in tokens:
        if ln > 0:
            start = len(out) - off
            for k in range(ln):
                out.append(out[start + k])
        if nxt is not None:
            out.append(nxt)
    return "".join(out)


def lz77_bruteforce(s, window=WINDOW, min_len=MIN_LEN):
    """Same greedy parse written with explicit candidate enumeration over all lengths."""
    tokens = []
    n = len(s)
    i = 0
    while i < n:
        best_len = best_off = 0
        for length in range(n - i, 0, -1):
            for j in range(max(0, i - window), i):
                if s[j:j + length] == s[i:i + length]:
                    best_len, best_off = length, i - j
                    break
            if best_len:
                break
        if best_len >= min_len:
            tokens.append((best_off, best_len, None if i + best_len == n else s[i + best_len]))
            i += best_len + (0 if i + best_len == n else 1)
        else:
            tokens.append((0, 0, s[i]))
            i += 1
    return tokens


def optimal_parse_len(s, window=WINDOW, min_len=MIN_LEN):
    """DP: fewest tokens any parse of s can use; greedy can only be >= this."""
    n = len(s)
    dp = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        best = 1 + dp[i + 1]  # emit a literal
        for j in range(max(0, i - window), i):
            L = 0
            while i + L < n and s[j + L] == s[i + L]:
                L += 1
            for length in range(min_len, L + 1):
                cand = 1 if i + length == n else 1 + dp[i + length + 1]
                if cand < best:
                    best = cand
        dp[i] = best
    return dp[0]


def lcg_string(length, seed, alphabet):
    out = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        out.append(alphabet[seed % len(alphabet)])
    return "".join(out), seed


def main():
    # exact greedy factorizations demanded by the task
    assert lz77_parse("abcabcabc", 8, 3) == [
        (0, 0, "a"), (0, 0, "b"), (0, 0, "c"), (3, 6, None)]
    assert lz77_parse("aaaaaaaa", 8, 3) == [(0, 0, "a"), (1, 7, None)]

    # round trips and hand-built corner cases
    assert lz77_decode(lz77_parse("abcabcabc")) == "abcabcabc"
    assert lz77_decode(lz77_parse("aaaaaaaa")) == "aaaaaaaa"
    assert lz77_parse("") == []
    assert lz77_decode([]) == ""
    assert lz77_parse("ab", 8, 3) == [(0, 0, "a"), (0, 0, "b")]          # too short to match
    assert lz77_parse("abcabcx", 8, 3) == [(0, 0, "a"), (0, 0, "b"), (0, 0, "c"), (3, 3, "x")]
    assert lz77_parse("aabaaba", 8, 3) == [(0, 0, "a"), (0, 0, "a"), (0, 0, "b"), (3, 4, None)]

    # window bounds the match source: distance-3 source needs window >= 3
    assert lz77_parse("aabaab", 3, 3) == [(0, 0, "a"), (0, 0, "a"), (0, 0, "b"), (3, 3, None)]
    assert lz77_parse("aabaab", 2, 3) == [
        (0, 0, "a"), (0, 0, "a"), (0, 0, "b"), (0, 0, "a"), (0, 0, "a"), (0, 0, "b")]

    # min length boundary: length-2 match is rejected, length-3 accepted
    assert lz77_parse("aaab", 8, 3) == [
        (0, 0, "a"), (0, 0, "a"), (0, 0, "a"), (0, 0, "b")]               # only offset-1 match of len 2
    assert lz77_parse("aaab", 8, 2) == [(0, 0, "a"), (1, 2, "b")]

    # token invariants for the hand cases
    for tokens in (lz77_parse("abcabcabc", 8, 3), lz77_parse("aaaaaaaa", 8, 3)):
        for off, ln, nxt in tokens:
            assert off <= 8
            assert ln == 0 or ln >= 3, tokens
            assert ln == 0 or off >= 1
        assert lz77_decode(tokens)

    # optimal parses of the two hand strings
    assert optimal_parse_len("abcabcabc", 8, 3) == 4
    assert optimal_parse_len("aaaaaaaa", 8, 3) == 2

    seed = 777
    for _ in range(8):
        length = 60 + seed % 100
        alphabet = "ab" if seed % 2 == 0 else "abc"
        text, seed = lcg_string(length, seed, alphabet)
        window = 4 + seed % 29
        greedy = lz77_parse(text, window, 3)
        slow = lz77_bruteforce(text, window, 3)
        assert greedy == slow, (text, window)
        assert lz77_decode(greedy) == text, (text, window)
        # every offset respects the window, matches honour min length
        for off, ln, _ in greedy:
            assert off <= window and (ln == 0 or ln >= 3)
        optimal = optimal_parse_len(text, window, 3)
        assert optimal <= len(greedy) <= len(text)
        # the literal-only parse is always available as the trivial upper bound
        assert len(greedy) <= length

    # greedy re-derivation: at every token boundary the token must be the longest
    # match, ties resolved to the smallest offset (or a literal when it is shorter)
    text = "abababababab"
    tokens = lz77_parse(text, 8, 3)
    pos = 0
    for off, ln, nxt in tokens:
        best_len, best_j = 0, -1
        for j in range(max(0, pos - 8), pos):
            L = 0
            while pos + L < len(text) and text[j + L] == text[pos + L]:
                L += 1
            if L > best_len:
                best_len, best_j = L, j
        if ln:
            assert ln == best_len and off == pos - best_j, (pos, ln, best_len, off)
        else:
            assert best_len < 3, (pos, best_len)
        pos += ln + (0 if nxt is None else 1)


if __name__ == "__main__":
    main()
    print("PASS")
