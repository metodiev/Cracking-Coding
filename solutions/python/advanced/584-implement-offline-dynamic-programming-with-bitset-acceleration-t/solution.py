"""Task 584 - Implement offline dynamic programming with bitset acceleration tricks. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)

Five offline DP problems solved by packing a whole DP layer into one Python arbitrary-precision
integer and letting the big-integer ALU do a word-parallel update:

  * subset-sum reachability -- `bits |= (bits << v) & cap`; bit s of `bits` means "some subset of
    the values processed so far sums to s".  One shift-and-or replaces a loop over all sums.
  * subset sum per cardinality -- one integer per item count k, updated in decreasing k so each
    item is used at most once; bit s of level k answers "is there a k-subset with sum s".
  * word break -- precompute, for every dictionary word length L, the bitmask of text positions
    where a word of that length matches; then iterate `reach |= (reach & match[L]) << L` to a
    fixpoint (monotone, so it terminates).
  * bit-parallel LCS (Hyyrö / Crochemore-Iliopoulos-Pinzon-Reid) -- a match mask per character
    and the carry recurrence `S = ((S + U) | (S - U)) & mask` with `U = S & match[c]` for each
    character of the second string; the length is n minus the popcount of S.  The file asserts
    this exact recurrence against the O(nm) table on 400 random pairs and two long strings; the
    `S - U` borrow propagation is exact for arbitrary-precision ints (a 64-bit-word port needs
    the usual per-word borrow bookkeeping, which is the real "trick" in C).
  * transitive closure of a digraph -- one integer per vertex, iterated to a fixpoint, i.e. a
    vector of bitsets instead of an n x n byte matrix; and maximum independent set -- branch and
    bound whose candidate set is a bitmask, with `cand & ~adj[v] & ~(1 << v)` for the include
    branch and the prune `chosen + popcount(cand) <= best`.

Randomness (graph edges, random strings) comes from splitmix64, not from the repo's 16-bit LCG
`(seed * 75 + 74) % 65537`: that LCG cycles within a few hundred values and its low bits are
strongly correlated, so the hundreds of "independent" draws used here (60 subset-sum instances,
400 string pairs, graph edges) would repeat patterns.  Every structure is checked against an
independent scalar or brute-force implementation, and edge cases are covered explicitly: empty
input, target 0, values above the cap, zero-valued items, isolated vertices, empty strings and a
64-value subset-sum instance that is far beyond the 2^n enumeration.
"""

MASK64 = (1 << 64) - 1


def splitmix64(state):
    """Documented 64-bit mixer used for all sample data (see the module docstring)."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def uniform(state, bound):
    state, value = splitmix64(state)
    return state, value % bound


# ---------------------------------------------------------------------------------------
# 1. subset-sum reachability
# ---------------------------------------------------------------------------------------
def subset_sum_bitset(values, limit):
    """Bit s of the result is 1 iff some subset of `values` sums to s with s <= limit."""
    cap = (1 << (limit + 1)) - 1
    bits = 1
    for value in values:
        if 0 < value <= limit:
            bits |= (bits << value) & cap
    return bits & cap


def subset_sum_scalar(values, limit):
    """Independent boolean table implementation (the obvious O(n * limit) DP)."""
    reachable = [False] * (limit + 1)
    reachable[0] = True
    for value in values:
        for total in range(limit, value - 1, -1):
            if reachable[total - value]:
                reachable[total] = True
    return reachable


def subset_sum_bruteforce(values):
    """Every subset sum, by enumeration (only usable for a handful of values)."""
    sums = {0}
    for value in values:
        sums |= {s + value for s in sums}
    return sums


# ---------------------------------------------------------------------------------------
# 2. subset sum per cardinality
# ---------------------------------------------------------------------------------------
def reach_by_cardinality_bitset(values, limit):
    """levels[k] is a bitset of the sums reachable using exactly k of the values."""
    cap = (1 << (limit + 1)) - 1
    levels = [0] * (len(values) + 1)
    levels[0] = 1
    used = 0
    for value in values:
        used += 1
        for k in range(used, 0, -1):
            if 0 < value <= limit:
                levels[k] |= (levels[k - 1] << value) & cap
    return levels


def reach_by_cardinality_bruteforce(values, limit):
    """Set of (subset size, subset sum) pairs with sum <= limit, by enumeration."""
    from itertools import combinations
    pairs = {(0, 0)}
    for size in range(1, len(values) + 1):
        for combo in combinations(values, size):
            total = sum(combo)
            if total <= limit:
                pairs.add((size, total))
    return pairs


# ---------------------------------------------------------------------------------------
# 3. word break
# ---------------------------------------------------------------------------------------
def word_break_bitset(text, words):
    """Can `text` be segmented into dictionary words?  Bitset fixpoint over positions."""
    n = len(text)
    match = {}
    for word in words:
        length = len(word)
        if length == 0 or length > n:
            continue
        bits = 0
        for i in range(n - length + 1):
            if text[i:i + length] == word:
                bits |= 1 << i
        match[length] = match.get(length, 0) | bits
    reach = 1
    changed = True
    while changed:                       # monotone: positions are only ever added
        changed = False
        for length, bits in match.items():
            grown = (reach & bits) << length
            if grown & ~reach:
                changed = True
                reach |= grown
    return bool(reach >> n & 1)


def word_break_scalar(text, words):
    """Independent position DP: dp[i] = text[:i] is segmentable."""
    n = len(text)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(n):
        if not dp[i]:
            continue
        for word in words:
            if word and text.startswith(word, i):
                dp[i + len(word)] = True
    return dp[n]


# ---------------------------------------------------------------------------------------
# 4. bit-parallel LCS
# ---------------------------------------------------------------------------------------
def lcs_bitparallel(a, b):
    """Hyyrö's bit-vector LCS length; U = S & match[c] drives the carry recurrence."""
    n = len(a)
    if n == 0:
        return 0
    matches = {}
    for i, ch in enumerate(a):
        matches[ch] = matches.get(ch, 0) | (1 << i)
    mask = (1 << n) - 1
    s = mask
    for ch in b:
        u = s & matches.get(ch, 0)
        s = ((s + u) | (s - u)) & mask
    return n - bin(s).count("1")


def lcs_dp(a, b):
    """Textbook O(len(a) * len(b)) table."""
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                table[i + 1][j + 1] = table[i][j] + 1
            else:
                table[i + 1][j + 1] = max(table[i][j + 1], table[i + 1][j])
    return table[n][m]


# ---------------------------------------------------------------------------------------
# 5. transitive closure
# ---------------------------------------------------------------------------------------
def closure_bitset(n, edges):
    """reach[u] = bitmask of the vertices reachable from u (including u itself)."""
    reach = [1 << u for u in range(n)]
    for (u, v) in edges:
        reach[u] |= 1 << v
    changed = True
    while changed:
        changed = False
        for u in range(n):
            new = reach[u]
            pending = new
            while pending:
                low = pending & -pending
                v = low.bit_length() - 1
                pending ^= low
                new |= reach[v]
            if new != reach[u]:
                reach[u] = new
                changed = True
    return reach


def closure_bfs(n, edges):
    """Independent per-vertex depth-first reachability."""
    adjacency = [[] for _ in range(n)]
    for (u, v) in edges:
        adjacency[u].append(v)
    reach = []
    for start in range(n):
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        mask = 0
        for v in seen:
            mask |= 1 << v
        reach.append(mask)
    return reach


# ---------------------------------------------------------------------------------------
# 6. maximum independent set (bitmask branch and bound)
# ---------------------------------------------------------------------------------------
def max_independent_set(n, edges):
    adjacency = [0] * n
    for (u, v) in edges:
        if u != v:
            adjacency[u] |= 1 << v
            adjacency[v] |= 1 << u
    best = 0

    def branch(candidates, chosen):
        nonlocal best
        if chosen + bin(candidates).count("1") <= best:
            return
        if candidates == 0:
            best = chosen
            return
        low = candidates & -candidates
        v = low.bit_length() - 1
        branch(candidates & ~adjacency[v] & ~low, chosen + 1)   # take v
        branch(candidates & ~low, chosen)                       # skip v

    branch((1 << n) - 1, 0)
    return best


def mis_bruteforce(n, edges):
    """Exhaustive maximum independent set over all 2^n vertex subsets."""
    adjacency = [0] * n
    for (u, v) in edges:
        if u != v:
            adjacency[u] |= 1 << v
            adjacency[v] |= 1 << u
    best = 0
    for subset in range(1 << n):
        size = bin(subset).count("1")
        if size <= best:
            continue
        members = subset
        ok = True
        while members:
            low = members & -members
            v = low.bit_length() - 1
            members ^= low
            if adjacency[v] & subset:
                ok = False
                break
        if ok:
            best = size
    return best


# ---------------------------------------------------------------------------------------
# sample data
# ---------------------------------------------------------------------------------------
def random_values(state, count, high):
    values = []
    for _ in range(count):
        state, value = uniform(state, high)
        values.append(value + 1)
    return state, values


def random_graph(state, n, extra_edges):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            state, roll = uniform(state, 100)
            if roll < 30:
                edges.append((u, v))
    for _ in range(extra_edges):
        state, u = uniform(state, n)
        state, v = uniform(state, n)
        edges.append((u, v))
    return state, edges


def random_text(state, length, alphabet):
    chars = []
    for _ in range(length):
        state, c = uniform(state, len(alphabet))
        chars.append(alphabet[c])
    return state, "".join(chars)


def main():
    # --- 1. subset sum ---------------------------------------------------------------
    key = [3, 34, 4, 12, 5, 2]
    bits = subset_sum_bitset(key, 30)
    assert bits >> 9 & 1 == 1, "3 + 4 + 2 = 9"
    assert bits >> 30 & 1 == 0, "nothing sums to 30"
    exact = subset_sum_bruteforce(key)
    for total in range(31):
        assert bool(bits >> total & 1) == (total in exact), total
    assert subset_sum_bitset([], 10) == 1, "only the empty subset"
    assert subset_sum_bitset([5], 10) == 0b100001
    assert subset_sum_bitset([4, 4, 4], 12) == 0b1000100010001, "0, 4, 8, 12"
    assert subset_sum_bitset([7, 7], 3) == 1, "values above the cap contribute nothing"
    assert subset_sum_bitset([0, 0, 3], 4) == 0b1001, "zero-valued items are harmless"
    assert subset_sum_bitset([2, 2], 0) == 1, "only the empty subset fits under a cap of 0"

    state = 20240584
    for trial in range(60):
        state, values = random_values(state, 1 + trial % 9, 14)
        limit = 20 + trial % 40
        scalar = subset_sum_scalar(values, limit)
        packed = subset_sum_bitset(values, limit)
        assert packed == sum((1 << t) for t in range(limit + 1) if scalar[t]), trial
        for total in subset_sum_bruteforce(values):
            if total <= limit:
                assert packed >> total & 1 == 1
    # 64 values are hopeless for the 2^n enumeration but trivial for the bitset; the scalar
    # table still agrees entry by entry.
    state, big = random_values(state, 64, 97)
    big_bits = subset_sum_bitset(big, 4000)
    big_scalar = subset_sum_scalar(big, 4000)
    assert big_bits == sum((1 << t) for t in range(4001) if big_scalar[t])
    assert big_bits.bit_length() <= 4001, "the bitset never grows past the cap"
    assert sum(big) <= 4000 and big_bits >> sum(big) & 1 == 1, "the whole set is reachable"

    # --- 2. cardinality-aware subset sum ---------------------------------------------
    levels = reach_by_cardinality_bitset(key, 60)
    pairs = reach_by_cardinality_bruteforce(key, 60)
    for size in range(len(key) + 1):
        for total in range(61):
            assert bool(levels[size] >> total & 1) == ((size, total) in pairs), (size, total)
    state, values = random_values(state, 11, 9)
    levels = reach_by_cardinality_bitset(values, 40)
    pairs = reach_by_cardinality_bruteforce(values, 40)
    for size in range(12):
        for total in range(41):
            assert bool(levels[size] >> total & 1) == ((size, total) in pairs), (size, total)
    assert levels[0] == 1, "the empty subset has size 0 and sum 0"
    assert len(levels) == len(values) + 1, "one level per value, plus the empty subset"
    assert reach_by_cardinality_bitset([1, 2, 3], 6)[3] >> 6 & 1 == 1, "all values at once"
    assert reach_by_cardinality_bitset([1, 2, 3], 5)[3] == 0, "no subset fits under the cap"
    assert reach_by_cardinality_bitset([2, 2, 2], 6)[2] >> 4 & 1 == 1
    assert reach_by_cardinality_bitset([5], 10)[1] == 0b100000

    # --- 3. word break ---------------------------------------------------------------
    assert word_break_bitset("leetcode", ["leet", "code"])
    assert not word_break_bitset("catsandog", ["cats", "dog", "sand", "and", "cat"])
    assert word_break_bitset("", ["a"]), "the empty string segments vacuously"
    assert not word_break_bitset("a", [])
    assert word_break_bitset("aaaaaa", ["aa", "aaa"])
    assert not word_break_bitset("aaaaaab", ["aa", "aaa"])
    assert word_break_bitset("abc", ["abc"]), "one word covering the whole string"
    alphabet = "abc"
    for trial in range(200):
        state, text = random_text(state, trial % 11, alphabet)
        words = set()
        for _ in range(1 + trial % 4):
            state, word = random_text(state, 1 + (trial + len(words)) % 3, alphabet)
            words.add(word)
        words = sorted(words)
        assert word_break_bitset(text, words) == word_break_scalar(text, words), (text, words)

    # --- 4. bit-parallel LCS ---------------------------------------------------------
    assert lcs_bitparallel("AGGTAB", "GXTXAYB") == 4
    assert lcs_bitparallel("banana", "atana") == 4
    assert lcs_bitparallel("dynamic", "dynamic") == 7
    assert lcs_bitparallel("", "abc") == 0 and lcs_bitparallel("abc", "") == 0
    assert lcs_bitparallel("abc", "xyz") == 0
    assert lcs_bitparallel("aaaaaaaaaa", "aaaa") == 4
    for trial in range(400):
        state, a = random_text(state, trial % 13, "abcd")
        state, b = random_text(state, (trial * 5) % 13, "abcd")
        assert lcs_bitparallel(a, b) == lcs_dp(a, b), (a, b)
    state, long_a = random_text(state, 150, "abcdefgh")
    state, long_b = random_text(state, 170, "abcdefgh")
    assert lcs_bitparallel(long_a, long_b) == lcs_dp(long_a, long_b)

    # --- 5. transitive closure -------------------------------------------------------
    state, edges = random_graph(state, 12, 6)
    assert closure_bitset(12, edges) == closure_bfs(12, edges)
    assert closure_bitset(4, [(0, 1), (1, 2), (2, 3)])[0] == 0b1111, "path closure"
    assert closure_bitset(3, [(0, 1), (1, 0)])[0] == 0b011, "2-cycle; vertex 2 unreachable"
    assert closure_bitset(3, [])[2] == 0b100, "isolated vertex reaches only itself"
    for trial in range(40):
        n = 1 + trial % 11
        state, edges = random_graph(state, n, trial % 5)
        assert closure_bitset(n, edges) == closure_bfs(n, edges), trial

    # --- 6. maximum independent set --------------------------------------------------
    assert max_independent_set(5, []) == 5
    assert max_independent_set(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]) == 1
    assert max_independent_set(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]) == 2, "C5"
    assert max_independent_set(1, []) == 1
    for trial in range(60):
        n = 4 + trial % 9
        state, edges = random_graph(state, n, trial % 4)
        assert max_independent_set(n, edges) == mis_bruteforce(n, edges), (n, edges)
    for trial in range(20):
        state, edges = random_graph(state, 12, 2)
        assert max_independent_set(12, edges) == mis_bruteforce(12, edges)

    # --- determinism -----------------------------------------------------------------
    state_a, edges_a = random_graph(31337, 10, 3)
    state_b, edges_b = random_graph(31337, 10, 3)
    assert edges_a == edges_b and state_a == state_b, "sample data is reproducible"


if __name__ == "__main__":
    main()
    print("PASS")
