"""Task 601 - Implement a full suffix tree (Ukkonen) and test with large text. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

Ukkonen's online construction: phase i appends text[i] and walks the *active point* (active node,
active edge, active length); every suffix still needing an extension applies one of three rules -
rule 1 creates a leaf when the active node has no transition for the incoming character, rule 2
splits an edge when the existing transition diverges inside it, rule 3 stops the phase because the
extension is already implicit. After each extension the active node becomes the suffix link of the
node created last, keeping a whole phase amortised constant. Leaves store no end position: their
edge label always ends at the current position (the implicit-end trick) - that is what makes the
construction O(n) instead of O(n^2).

A unique smallest sentinel is appended internally, turning the implicit tree into a *full* suffix
tree: every suffix ends at its own leaf, so a pattern's occurrence count is exactly the leaf count of
the subtree below the node where the pattern ends. A leaf's path label is a whole suffix, so its
suffix start is |text| + 1 - depth; the stored edge start is only the position of the edge *label*
and may be shifted, because the split rule creates a leaf whose label is text[i:]. Queries answer on
the original text: node_for/contains descend the edges (a pattern may end inside an edge);
occurrences/positions count and list the leaves below that node, and the sentinel leaf can never be
below a non-empty pattern; distinct_substrings sums all edge-label lengths minus the leaf count,
since each leaf contributes exactly one locus containing the sentinel; longest_repeated_substring is
the deepest internal node, which has at least two leaves below it.

Checks: hand values (banana, mississippi, abracadabra, unary, empty and single-character texts);
suffix links verified as path(v)[1:] for every internal node; internal nodes verified to be exactly
the right-maximal repeated substrings; every substring and every generated non-substring of small
texts; and a large-text run whose distinct substring count and longest repeated substring are
cross-checked against a suffix automaton and a binary search over duplicate substring lengths.
"""
from itertools import product

SENTINEL = "\x00"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


class SuffixTree:
    """Full suffix tree of `text`, built by Ukkonen over text + SENTINEL; nodes are integers."""

    def __init__(self, text):
        self.s = text + SENTINEL            # the sentinel makes every suffix end at a leaf
        n = len(self.s)
        self.children = [{}]                # node -> {first character of the edge label: child}
        self.edge_start = [0]
        self.edge_end = [0]                 # leaves end at n, i.e. their label is open to the right
        self.link = [0]                     # suffix links; the root links to itself
        self.leaf_start = [-1]              # leaves: the suffix start, else -1
        self.depth = [0]                    # length of the node's path label
        self.leaf_count = [0]               # leaves below the node = occurrences of the path label
        self._build(n)
        self._finalize()

    def _new_node(self, start, end):
        self.children.append({})
        for series, value in ((self.edge_start, start), (self.edge_end, end), (self.link, 0),
                              (self.leaf_start, -1), (self.depth, 0), (self.leaf_count, 0)):
            series.append(value)
        return len(self.depth) - 1

    def _build(self, n):
        s = self.s
        active_node, active_edge, active_length, remaining = 0, -1, 0, 0
        for i in range(n):
            remaining += 1
            last_new = -1                       # internal node created last: needs a suffix link
            while remaining > 0:
                if active_length == 0:
                    active_edge = i
                ch = s[active_edge]
                if ch not in self.children[active_node]:
                    leaf = self._new_node(i, n)                       # rule 1: a new leaf
                    self.children[active_node][ch] = leaf
                    if last_new != -1:
                        self.link[last_new] = active_node
                        last_new = -1
                else:
                    nxt = self.children[active_node][ch]
                    edge_len = self.edge_end[nxt] - self.edge_start[nxt]
                    if active_length >= edge_len:
                        active_node = nxt                             # walk down and retry
                        active_edge += edge_len
                        active_length -= edge_len
                        continue
                    if s[self.edge_start[nxt] + active_length] == s[i]:
                        if last_new != -1:                            # rule 3: already implicit
                            self.link[last_new] = active_node
                            last_new = -1
                        active_length += 1
                        break
                    mid = self._new_node(self.edge_start[nxt],                # rule 2: split
                                         self.edge_start[nxt] + active_length)
                    self.children[active_node][ch] = mid
                    self.children[mid][s[self.edge_start[nxt] + active_length]] = nxt
                    self.edge_start[nxt] += active_length
                    self.children[mid][s[i]] = self._new_node(i, n)
                    if last_new != -1:
                        self.link[last_new] = mid
                    last_new = mid
                remaining -= 1
                if active_node == 0 and active_length > 0:
                    active_length -= 1                                # the root links nowhere
                    active_edge = i - remaining + 1
                elif active_node != 0:
                    active_node = self.link[active_node]

    def _finalize(self):
        total = len(self.s)
        self.kids = [list(c.values()) for c in self.children]
        order = [0]
        for node in order:                      # breadth-first: parents before their children
            order.extend(self.kids[node])
        for node in order:
            for child in self.kids[node]:
                self.depth[child] = self.depth[node] + self.edge_end[child] - self.edge_start[child]
        for node in reversed(order):            # bottom-up: leaf counts and suffix starts
            if node and not self.kids[node]:
                self.leaf_start[node] = total - self.depth[node]
                self.leaf_count[node] = 1
            else:
                self.leaf_count[node] = sum(self.leaf_count[c] for c in self.kids[node])
        self.leaves = [v for v in range(1, len(self.depth)) if not self.kids[v]]

    def node_for(self, pattern):
        """Node (or leaf) where `pattern` ends, or -1; an end inside an edge returns that child."""
        node, idx = 0, 0
        while idx < len(pattern):
            child = self.children[node].get(pattern[idx])
            if child is None:
                return -1
            take = min(self.edge_end[child] - self.edge_start[child], len(pattern) - idx)
            if self.s[self.edge_start[child]:self.edge_start[child] + take] != pattern[idx:idx + take]:
                return -1
            idx += take
            node = child
        return node

    def contains(self, pattern):
        return self.node_for(pattern) != -1

    def occurrences(self, pattern):
        if not pattern:                         # documented: the empty pattern is not searched
            return 0
        node = self.node_for(pattern)
        return 0 if node == -1 else self.leaf_count[node]

    def positions(self, pattern):
        node = self.node_for(pattern) if pattern else -1
        if node == -1:
            return []
        stack, starts = [node], []
        while stack:
            cur = stack.pop()
            if self.kids[cur]:
                stack.extend(self.kids[cur])
            elif cur:
                starts.append(self.leaf_start[cur])
        return sorted(starts)

    def path_label(self, node):
        """Text of the path from the root to `node`, recovered from any leaf below it."""
        if node == 0:
            return ""
        cur = node
        while self.kids[cur]:
            cur = self.kids[cur][0]
        return self.s[self.leaf_start[cur]:self.leaf_start[cur] + self.depth[node]]

    def distinct_substrings(self):
        """Loci of text + sentinel minus the leaves; each leaf adds one sentinel locus."""
        return (sum(self.edge_end[v] - self.edge_start[v] for v in range(1, len(self.depth)))
                - len(self.leaves))

    def longest_repeated_substring(self):
        best, node = 0, -1
        for v in range(1, len(self.depth)):
            if self.kids[v] and self.depth[v] > best:
                best, node = self.depth[v], v
        return "" if node == -1 else self.path_label(node)


class SuffixAutomaton:
    """Independent oracle for the number of distinct substrings."""

    def __init__(self, text):
        length, link, trans = [0], [-1], [{}]
        last = 0
        for ch in text:
            cur = len(length)
            length.append(length[last] + 1)
            link.append(-1)
            trans.append({})
            p = last
            while p != -1 and ch not in trans[p]:
                trans[p][ch] = cur
                p = link[p]
            if p == -1:
                link[cur] = 0
            elif length[p] + 1 == length[trans[p][ch]]:
                link[cur] = trans[p][ch]
            else:
                q = trans[p][ch]                     # clone: split the equivalence class
                clone = len(length)
                length.append(length[p] + 1)
                link.append(link[q])
                trans.append(dict(trans[q]))
                while p != -1 and trans[p].get(ch) == q:
                    trans[p][ch] = clone
                    p = link[p]
                link[q] = link[cur] = clone
            last = cur
        self.size, self.length, self.link = len(length), length, link

    def distinct_substrings(self):
        return sum(self.length[v] - self.length[self.link[v]] for v in range(1, self.size))


def brute_substrings(text):
    return {text[i:j] for i in range(len(text)) for j in range(i + 1, len(text) + 1)}


def brute_positions(text, pattern):
    out = []
    start = text.find(pattern)
    while start != -1:
        out.append(start)
        start = text.find(pattern, start + 1)
    return out


def brute_longest_repeated_length(text):
    """Exact: binary search over lengths, with one set of substrings per probe length."""
    def has_repeat(length):
        seen = set()
        for i in range(len(text) - length + 1):
            if text[i:i + length] in seen:
                return True
            seen.add(text[i:i + length])
        return False

    lo, hi = 0, len(text)                       # monotone: a repeat of length L implies L - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if has_repeat(mid) else (lo, mid - 1)
    return lo


def check_small(text):
    """Exhaustive verification for short texts (n <= 40)."""
    tree = SuffixTree(text)
    n = len(text)
    substrings = brute_substrings(text)
    for leaf in tree.leaves:                    # a leaf's path label is a whole suffix of text + $
        check(tree.depth[leaf] == n + 1 - tree.leaf_start[leaf], (text, leaf))
        check(tree.path_label(leaf) == (text + SENTINEL)[tree.leaf_start[leaf]:], (text, leaf))
    check(len(tree.leaves) == n + 1
          and sorted(tree.leaf_start[v] for v in tree.leaves) == list(range(n + 1)), text)
    check(tree.distinct_substrings() == len(substrings)
          == SuffixAutomaton(text).distinct_substrings(), f"distinct for {text!r}")
    for piece in substrings:
        node = tree.node_for(piece)
        check(node != -1 and tree.path_label(node).startswith(piece), (text, piece))
        check(tree.occurrences(piece) == len(brute_positions(text, piece)), (text, piece))
        check(tree.positions(piece) == brute_positions(text, piece), (text, piece))
    alphabet = sorted(set(text) | {"x"})
    for probe in ["".join(p) for size in (1, 2, 3) for p in product(alphabet, repeat=size)]:
        inside = probe in substrings
        check(tree.contains(probe) == inside, (text, probe))
        check(tree.occurrences(probe) == (inside and len(brute_positions(text, probe))), probe)
    check(not tree.contains(text + "z") and tree.occurrences("") == 0
          and tree.positions("") == [], text)

    # suffix links: an internal node at depth d > 0 links to the node of path(v)[1:], depth d - 1
    for node in range(1, len(tree.depth)):
        if tree.kids[node]:
            label = tree.path_label(node)
            check(tree.depth[tree.link[node]] == tree.depth[node] - 1, (text, label))
            check(tree.path_label(tree.link[node]) == label[1:], (text, label))

    if n <= 12:                                 # internal nodes are the right-maximal repeats
        after = {p: {text[i + len(p)] if i + len(p) < n else SENTINEL
                     for i in brute_positions(text, p)} for p in substrings}
        right_maximal = {p for p in substrings
                         if len(brute_positions(text, p)) >= 2 and len(after[p]) >= 2}
        nodes = {tree.path_label(v) for v in range(1, len(tree.depth)) if tree.kids[v]}
        check(nodes == right_maximal, (text, nodes ^ right_maximal))

    lrs = tree.longest_repeated_substring()
    longest = max([len(x) for x in substrings if len(brute_positions(text, x)) >= 2], default=0)
    check(len(lrs) == longest == brute_longest_repeated_length(text), (text, lrs))
    if lrs:
        check(len(brute_positions(text, lrs)) >= 2, (text, lrs))
    return tree


def main():
    banana = check_small("banana")
    check(banana.occurrences("ana") == 2 and banana.positions("ana") == [1, 3], "banana ana")
    check(banana.occurrences("na") == 2 and banana.occurrences("a") == 3
          and banana.occurrences("banana") == 1, "banana counts")
    check(banana.distinct_substrings() == 15 and banana.longest_repeated_substring() == "ana"
          and banana.path_label(banana.node_for("ana")) == "ana", "banana")

    mississippi = check_small("mississippi")
    check(mississippi.occurrences("issi") == 2 and mississippi.occurrences("i") == 4, "mississippi")
    check(mississippi.positions("ssi") == [2, 5]
          and mississippi.longest_repeated_substring() == "issi", "mississippi 2")
    for text in ("", "z", "a", "aaaa", "aaaaaaaa", "abracadabra", "abcabxabcd", "aaaabaaaab"):
        check_small(text)
    for alphabet in ("ab", "abc"):
        for trial in range(10):
            check_small(lcg_string(40, alphabet, 601 + trial * 7919)[0])
    # medium texts: a long unary run, a periodic text and a random one; sentinel paths too
    for text in ("a" * 500, "abcab" * 120, lcg_string(600, "ab", 4711)[0]):
        tree = SuffixTree(text)
        check(tree.distinct_substrings() == SuffixAutomaton(text).distinct_substrings()
              and len(tree.longest_repeated_substring()) == brute_longest_repeated_length(text), text)
        check(tree.occurrences(text[-40:]) == len(brute_positions(text, text[-40:])), "suffix query")
    # large text: Ukkonen has to stay linear, and both oracles stay exact at this size
    large, _ = lcg_string(12000, "abcd", 20260101)
    large = large[:2000] + "abababababababab" + large[2000:] + large[:1500]
    tree = SuffixTree(large)
    n = len(large)
    check(len(tree.leaves) == n + 1, "large leaf count")
    check(tree.distinct_substrings() == SuffixAutomaton(large).distinct_substrings(), "large distinct")
    lrs = tree.longest_repeated_substring()
    check(len(lrs) == brute_longest_repeated_length(large), "large LRS")
    check(tree.occurrences(lrs) == len(brute_positions(large, lrs)) >= 2, "large LRS occurrences")
    for window in range(0, n - 25, 613):        # sampled positions and counts along the text
        piece = large[window:window + 25]
        check(tree.occurrences(piece) == len(brute_positions(large, piece)), piece)
        check(tree.positions(piece) == brute_positions(large, piece), piece)
    check(tree.positions("abababababababab")[0] <= 2000, "injected literal recognised")
    check(tree.occurrences("ababababababababq") == 0 and not tree.contains("zzzzz"), "absent")
    print(f"large text: {n} chars, {len(tree.depth)} nodes, {tree.distinct_substrings()} distinct, "
          f"longest repeat {len(lrs)}")


if __name__ == "__main__":
    main()
    print("PASS")
