"""Task 406 - Implement an online suffix tree (Ukkonen outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Outline: a real Ukkonen tree builds the *compressed* suffix tree online in O(n) by extending
the last suffix, following a suffix link when a split happens, and keeping the suffix link of
every new node correct. This outline keeps the same skeleton for tiny strings: an explicit
suffix trie of the text (every node is a distinct substring, edges are single characters)
plus the suffix link of each node, which points at the node of its path label minus the first
character. That link is exactly the invariant Ukkonen restores when it splits an edge, and it
is what makes "ends with suffix link" navigation possible here: following links from a node
walks through all suffixes of its path in O(depth) steps.

Links are wired breadth-first: a depth-1 node links to the root, and a node whose parent p is
reached over character c links to children[link[p]][c], which always exists in a suffix trie.

Checks on tiny texts only (banana, mississippi, abab, aaaa): every substring is findable and
every generated non-substring fails, node count matches the brute-force distinct-substring
set, each link target really is the node of path[1:], link chains enumerate all suffixes, and
subtree terminal counts reproduce naive occurrence counts.
"""

class SuffixTreeOutline:
    """Explicit suffix tree: nodes = distinct substrings, suffix link = path[1:]."""

    def __init__(self, text):
        self.text = text
        self.children = [{}]
        self.parent = [-1]
        self.parent_char = [""]
        self.depth = [0]
        self.is_terminal = [False]       # a suffix of the text ends at this node
        self.link = [-1]                 # root has no suffix link
        for start in range(len(text)):
            node = 0
            for ch in text[start:]:
                if ch not in self.children[node]:
                    self.children[node][ch] = len(self.depth)
                    self.children.append({})
                    self.parent.append(node)
                    self.parent_char.append(ch)
                    self.depth.append(self.depth[node] + 1)
                    self.is_terminal.append(False)
                    self.link.append(-1)
                node = self.children[node][ch]
            self.is_terminal[node] = True
        self._wire_suffix_links()

    def _wire_suffix_links(self):
        """BFS: link of a depth-1 node is the root; else children[link[parent]][char]."""
        order = [0]
        for node in order:
            order.extend(self.children[node].values())
        for node in order:
            if node == 0:
                continue
            if self.depth[node] == 1:
                self.link[node] = 0
            else:
                self.link[node] = self.children[self.link[self.parent[node]]][self.parent_char[node]]

    def path_label(self, node):
        chars = []
        while node != 0:
            chars.append(self.parent_char[node])
            node = self.parent[node]
        return "".join(reversed(chars))

    def find(self, pattern):
        """Descend from the root; return the node or -1 (explicit navigation)."""
        node = 0
        for ch in pattern:
            node = self.children[node].get(ch, -1)
            if node == -1:
                return -1
        return node

    def contains(self, pattern):
        return self.find(pattern) != -1

    def suffix_chain(self, node):
        chain = []
        while node != -1:
            chain.append(node)
            node = self.link[node]
        return chain

    def occurrences(self, pattern):
        node = self.find(pattern)
        if node == -1:
            return 0
        count = 0
        stack = [node]
        while stack:
            cur = stack.pop()
            if self.is_terminal[cur]:
                count += 1
            stack.extend(self.children[cur].values())
        return count

    def distinct_substrings(self):
        return len(self.depth) - 1


def all_substrings(s):
    return {s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)}


def count_occurrences(s, t):
    if not t:
        return 0
    return sum(1 for i in range(len(s) - len(t) + 1) if s[i:i + len(t)] == t)


def enumerate_strings(alphabet, max_len):
    out = []
    frontier = [""]
    for _ in range(max_len + 1):
        out.extend(frontier)
        frontier = [w + ch for w in frontier for ch in alphabet]
    return out


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def check_outline(text, alphabet, probe_len):
    tree = SuffixTreeOutline(text)
    substrings = all_substrings(text)

    assert tree.distinct_substrings() == len(substrings), text
    assert tree.contains("") and tree.find("") == 0
    for t in substrings:
        node = tree.find(t)
        assert node != -1, (text, t)
        assert tree.path_label(node) == t, (text, t)
        assert tree.occurrences(t) == count_occurrences(text, t), (text, t)
    for t in enumerate_strings(alphabet, probe_len):
        if t and t not in substrings:          # "" is the root and always found
            assert not tree.contains(t), (text, t)

    for node in range(1, len(tree.depth)):
        label = tree.path_label(node)
        assert tree.link[node] != -1
        assert tree.path_label(tree.link[node]) == label[1:], (text, label)
        chain = tree.suffix_chain(node)
        assert len(chain) == len(label) + 1        # node, its suffixes..., root
        assert [tree.path_label(n) for n in chain] == [label[k:] for k in range(len(label))] + [""]
        assert tree.depth[tree.link[node]] == tree.depth[node] - 1
    return tree


def main():
    banana = check_outline("banana", "abn", 4)
    assert banana.occurrences("ana") == 2
    assert banana.occurrences("a") == 3
    assert banana.occurrences("na") == 2
    assert banana.occurrences("banana") == 1
    assert not banana.contains("bananas") and not banana.contains("nana-")

    mississippi = check_outline("mississippi", "imps", 3)
    assert mississippi.occurrences("issi") == 2
    assert mississippi.occurrences("ssi") == 2
    assert mississippi.occurrences("i") == 4

    check_outline("abab", "ab", 6)
    check_outline("aaaa", "a", 6)
    check_outline("abcab", "abc", 5)
    check_outline("", "ab", 2)
    check_outline("z", "ab", 2)

    for trial in range(20):
        text, seed = lcg_string(8, "ab", 5000 + trial * 7919)
        check_outline(text, "ab", 3)
    for trial in range(10):
        text, seed = lcg_string(7, "abc", 7000 + trial * 7919)
        check_outline(text, "abc", 3)

    # online flavour: inserting one character extends the outline and keeps every invariant
    base = SuffixTreeOutline("aban")
    assert base.occurrences("a") == 2
    extended = SuffixTreeOutline("aban" + "a")
    assert extended.distinct_substrings() == len(all_substrings("abana"))
    assert extended.occurrences("a") == 3
    assert extended.path_label(extended.link[extended.find("bana")]) == "ana"


if __name__ == "__main__":
    main()
    print("PASS")
