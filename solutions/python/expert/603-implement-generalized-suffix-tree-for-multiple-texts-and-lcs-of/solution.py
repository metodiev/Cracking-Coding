"""Task 603 - Implement generalized suffix tree for multiple texts and LCS of many strings. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

The k texts are concatenated as s_0 #0 s_1 #1 ... s_(k-1) #(k-1) with one *distinct* sentinel
character per text (chr(1 + doc), all below any real character). Because the last character of the
concatenation is a sentinel that occurs nowhere else, Ukkonen's construction over that string yields
a full suffix tree: every suffix ends at its own leaf, and a leaf's suffix start is |T| + 1 - depth.

The generalized tree is used through the documents, not the concatenation:
  * occurrences(pattern) - the leaves below the node where a sentinel-free pattern ends all start at
    real occurrences of the pattern, and the (document, offset) pair of the leaf's suffix start is an
    occurrence; a node below a sentinel-containing path has exactly one leaf, so it can never show up
    for a non-empty pattern;
  * document masks - each node carries the bitmask of the documents its subtree reaches, computed
    bottom-up (<= 64 documents, which the constructor asserts);
  * longest common substring of a document set - the deepest node whose mask covers the set; the node
    is *dirty* if any edge on its path holds a sentinel, and dirty nodes are ignored. A longest
    common substring of two or more documents is always right-maximal in the concatenation (each
    occurrence at the end of a document is followed by that document's own sentinel), so it is a node
    of the tree; the single-document case is returned directly, since that answer is the text itself.

Checks: hand sets (["banana","ananas","bandana"] -> 3, ["ab","ba","aba"] -> 1, ["aaaa","aa","a"] -> 1,
single text, empty texts, duplicates); the longest common substring of every document subset against
a brute-force intersection of substring sets; occurrences and document masks of every substring of
small sets against a naive per-document scan; and a 40-document case where the answer is verified to
occur in every document and no substring one character longer is common to all of them.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


class GeneralizedSuffixTree:
    """Full suffix tree of the concatenation with distinct sentinels, built by Ukkonen."""

    def __init__(self, texts, max_documents=64):
        check(len(texts) <= max_documents, "the document masks are limited to 64-bit integers")
        self.texts = list(texts)
        parts, doc_of, doc_start, offset = [], [], [], 0
        for i, text in enumerate(self.texts):
            doc_start.append(offset)                # character offset of the document in the text
            parts.append(text + chr(1 + i))         # the document's own, distinct sentinel
            offset += len(text) + 1
            doc_of.extend([i] * (len(text) + 1))
        self.s = "".join(parts)
        self.doc_of, self.doc_start = doc_of, doc_start
        self.sentinels = {chr(1 + i) for i in range(len(self.texts))}
        n = len(self.s)
        self.children = [{}]                # node -> {first character of the edge label: child}
        self.edge_start = [0]
        self.edge_end = [0]                 # leaves end at n, i.e. their label is open to the right
        self.link = [0]                     # suffix links; the root links to itself
        self.depth = [0]                    # length of the node's path label
        self.leaf_start = [-1]              # leaves: the suffix start, else -1
        self.mask = [0]                     # bitmask of the documents below the node
        self.dirty = [False]                # True once an edge on the path holds a sentinel
        if n:
            self._build(n)
        self._finalize()

    def _new_node(self, start, end):
        self.children.append({})
        for series, value in ((self.edge_start, start), (self.edge_end, end), (self.link, 0),
                              (self.depth, 0), (self.leaf_start, -1), (self.mask, 0)):
            series.append(value)
        self.dirty.append(False)
        return len(self.depth) - 1

    def _build(self, n):
        s = self.s
        active_node, active_edge, active_length, remaining = 0, -1, 0, 0
        for i in range(n):
            remaining += 1
            last_new = -1                   # internal node created last: needs a suffix link
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
        """Depths and sentinel prefixes top-down, leaf starts and document masks bottom-up."""
        total = len(self.s)
        prefix = [0] * (total + 1)          # number of sentinels in s[:p], so an edge is dirty in O(1)
        for p, ch in enumerate(self.s):
            prefix[p + 1] = prefix[p] + (1 if ch in self.sentinels else 0)
        self.kids = [list(c.values()) for c in self.children]
        order = [0]
        for node in order:                  # breadth-first: parents before their children
            order.extend(self.kids[node])
        for node in order:
            for child in self.kids[node]:
                self.depth[child] = self.depth[node] + self.edge_end[child] - self.edge_start[child]
                self.dirty[child] = self.dirty[node] or (
                    prefix[self.edge_end[child]] - prefix[self.edge_start[child]] > 0)
        for node in reversed(order):
            if node and not self.kids[node]:
                self.leaf_start[node] = total - self.depth[node]
                self.mask[node] = 1 << self.doc_of[self.leaf_start[node]]
            else:
                mask = 0
                for child in self.kids[node]:
                    mask |= self.mask[child]
                self.mask[node] = mask
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

    def path_label(self, node):
        """Text of the path from the root to `node`, recovered from any leaf below it."""
        if node == 0:
            return ""
        cur = node
        while self.kids[cur]:
            cur = self.kids[cur][0]
        return self.s[self.leaf_start[cur]:self.leaf_start[cur] + self.depth[node]]

    def leaves_below(self, node):
        stack, out = [node], []
        while stack:
            cur = stack.pop()
            if self.kids[cur]:
                stack.extend(self.kids[cur])
            elif cur:
                out.append(cur)
        return out

    def occurrences(self, pattern):
        """All (document, offset) occurrences of a sentinel-free pattern."""
        node = self.node_for(pattern) if pattern else -1
        if node == -1:
            return []
        out = []
        for leaf in self.leaves_below(node):
            start = self.leaf_start[leaf]
            doc = self.doc_of[start]
            out.append((doc, start - self.doc_start[doc]))
        return sorted(out)

    def documents(self, pattern):
        """Bitmask of the documents that contain `pattern`."""
        node = self.node_for(pattern) if pattern else -1
        return 0 if node == -1 else self.mask[node]

    def longest_common_substring_of(self, docs):
        """Longest string that occurs in every document of `docs` (>= 2 documents)."""
        if not docs:
            return ""
        if len(docs) == 1:                  # one document: the longest common substring is itself
            return self.texts[docs[0]]
        want = 0
        for doc in docs:
            want |= 1 << doc
        best, node = 0, -1
        for v in range(1, len(self.depth)):
            if not self.dirty[v] and self.depth[v] > best and self.mask[v] & want == want:
                best, node = self.depth[v], v
        return "" if node == -1 else self.path_label(node)

    def longest_common_substring_of_all(self):
        return self.longest_common_substring_of(list(range(len(self.texts))))


def substrings(text):
    return {text[i:j] for i in range(len(text)) for j in range(i + 1, len(text) + 1)}


def brute_lcs(texts):
    """Brute force: intersect the substring sets, then take the longest member."""
    if not texts:
        return ""
    common = substrings(texts[0])
    for text in texts[1:]:
        common &= substrings(text)
    return max(common, key=lambda p: (len(p), p), default="")


def naive_occurrences(texts, pattern):
    out = []
    for doc, text in enumerate(texts):
        start = text.find(pattern)
        while start != -1:
            out.append((doc, start))
            start = text.find(pattern, start + 1)
    return out


def main():
    tree = GeneralizedSuffixTree(["banana", "ananas", "bandana"])
    check(len(tree.leaves) == len(tree.s), "one leaf per suffix of the concatenation")
    check(tree.longest_common_substring_of_all() == "ana", "banana/ananas/bandana LCS")
    check(tree.longest_common_substring_of([0, 1]) == "anana", "pair 0/1")
    check(tree.longest_common_substring_of([0, 2]) in ("ban", "ana"), "pair 0/2 (tie at length 3)")
    check(tree.longest_common_substring_of([1, 2]) == "ana", "pair 1/2")
    check(tree.occurrences("ana") == [(0, 1), (0, 3), (1, 0), (1, 2), (2, 4)], "ana")
    check(tree.occurrences("ban") == [(0, 0), (2, 0)], "ban")
    check(tree.occurrences("bandana") == [(2, 0)], "bandana")
    check(tree.occurrences("nanas") == [(1, 1)], "nanas")
    check(tree.occurrences("bananaz") == [] and tree.occurrences("") == [], "rejected")
    check(tree.documents("ana") == 0b111 and tree.documents("band") == 0b100, "document masks")
    check(tree.documents("xyz") == 0, "absent pattern mask")

    for texts in (["ab", "ba", "aba"], ["aaaa", "aa", "a"], ["abc"], ["", "ab"],
                  ["abc", "abc"], ["x", "y"], ["a", "b", "c"], ["zz", "zzz", "zzzz"]):
        tree = GeneralizedSuffixTree(texts)
        k = len(texts)
        answer = tree.longest_common_substring_of_all()
        check(len(answer) == len(brute_lcs(texts)), texts)
        check(all(answer in text for text in texts), texts)
        for first in range(k):
            for second in range(first + 1, k):
                sub = tree.longest_common_substring_of([first, second])
                check(len(sub) == len(brute_lcs([texts[first], texts[second]])), (texts, first, second))
                check(sub in texts[first] and sub in texts[second], (texts, first, second))
        for piece in substrings("".join(texts)) | {""}:
            if any(ch in piece for ch in tree.sentinels):
                continue
            expected = [(doc, start) for doc, text in enumerate(texts)
                        for start in range(len(text) - len(piece) + 1)
                        if text[start:start + len(piece)] == piece] if piece else []
            check(tree.occurrences(piece) == expected, (texts, piece))
            check(tree.documents(piece) == sum(1 << doc for doc in {d for d, _ in expected}),
                  (texts, piece))

    for trial in range(8):
        texts = [lcg_string(30, "ab", 6030 + trial * 7919 + shift)[0] for shift in range(3)]
        tree = GeneralizedSuffixTree(texts)
        check(len(tree.longest_common_substring_of_all()) == len(brute_lcs(texts)), texts)
        check(len(tree.longest_common_substring_of([0, 1])) == len(brute_lcs(texts[:2])), texts)
        piece = texts[0][10:14]
        check(tree.occurrences(piece) == naive_occurrences(texts, piece), piece)

    # many documents: one 14-character block is injected into all 40 texts; maximality is proven
    # exactly, because every common substring is necessarily a substring of the first document
    texts = [lcg_string(100, "abc", 900 + trial * 104729)[0] for trial in range(40)]
    common = texts[0][20:34]
    texts = [text[:30] + common + text[30:] for text in texts]
    tree = GeneralizedSuffixTree(texts)
    answer = tree.longest_common_substring_of_all()
    check(len(answer) >= 14 and all(answer in text for text in texts), "40-document LCS")
    longer = [texts[0][i:i + len(answer) + 1] for i in range(len(texts[0]) - len(answer))]
    check(not any(all(candidate in text for text in texts) for candidate in longer),
          "no longer common substring")
    check(tree.documents(answer) == (1 << len(texts)) - 1, "all documents reached")
    check(tree.mask[0] == (1 << len(texts)) - 1, "root mask covers every document")

    print(f"banana/ananas/bandana: {len(GeneralizedSuffixTree(['banana', 'ananas', 'bandana']).depth)}"
          " nodes, longest common substring 'ana'")
    print(f"40 documents of 100 characters: longest common substring of length {len(answer)}")


if __name__ == "__main__":
    main()
    print("PASS")
