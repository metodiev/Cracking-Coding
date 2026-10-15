"""Task 253 - Implement compressed trie (radix tree) basics. (Tier: Intermediate, CTCI Ch 4)

A radix tree stores runs of single-child characters as one edge label. insert() splits an
edge when the new word shares only part of it. Inserting "cart", then "car", then "care":

    after "cart":  root --cart--> (leaf)
    after "car":   root --car--> (word) --t--> (word)      ("cart" edge split at 3 chars)
    after "care":  root --car--> (word) --e--> (word)
                                     \\--t--> (word)

shape() renders that compressed form as "[car*[e*,t*]]" ('*' marks a stored word), and
node_count() counts edges, so it can be compared with a plain trie over the same words.
"""


class RadixNode:
    __slots__ = ("label", "children", "is_word")

    def __init__(self, label):
        self.label = label
        self.children = {}
        self.is_word = False


def insert(root, word):
    """Inserts word; True iff it was not already stored."""
    node = root
    rest = word
    while True:
        if not rest:
            was_new = not node.is_word
            node.is_word = True
            return was_new
        child = node.children.get(rest[0])
        if child is None:
            fresh = RadixNode(rest)
            fresh.is_word = True
            node.children[rest[0]] = fresh
            return True
        common = 0
        while (common < len(rest) and common < len(child.label)
               and rest[common] == child.label[common]):
            common += 1
        if common == len(child.label):
            node, rest = child, rest[common:]
            continue
        split = RadixNode(child.label[:common])
        child.label = child.label[common:]
        split.children[child.label[0]] = child
        node.children[rest[0]] = split
        rest = rest[common:]
        if not rest:
            split.is_word = True
        else:
            fresh = RadixNode(rest)
            fresh.is_word = True
            split.children[rest[0]] = fresh
        return True


def contains(root, word):
    node = root
    i = 0
    while i < len(word):
        child = node.children.get(word[i])
        if child is None or word[i:i + len(child.label)] != child.label:
            return False
        i += len(child.label)
        node = child
    return node.is_word


def words_with_prefix(root, prefix):
    """Stored words starting with prefix, lexicographically sorted."""
    node = root
    so_far = ""
    i = 0
    while i < len(prefix):
        child = node.children.get(prefix[i])
        if child is None:
            return []
        want = len(prefix) - i
        take = min(want, len(child.label))
        if child.label[:take] != prefix[i:i + take]:
            return []
        node = child
        so_far += child.label
        if want <= len(child.label):
            break                                         # prefix ended inside this edge
        i += len(child.label)
    out = []
    _collect(node, so_far, out)
    return out


def _collect(node, so_far, out):
    if node.is_word:
        out.append(so_far)
    for ch in sorted(node.children):
        child = node.children[ch]
        _collect(child, so_far + child.label, out)


def starts_with(root, prefix):
    return bool(words_with_prefix(root, prefix))


def shape(node):
    """Deterministic compressed rendering: labels glued by commas, '*' = stored word."""
    text = node.label + ("*" if node.is_word else "")
    if not node.children:
        return text
    inner = ",".join(shape(node.children[ch]) for ch in sorted(node.children))
    return text + "[" + inner + "]"


def node_count(root):
    """Number of nodes excluding the root, i.e. the number of edges."""
    return len(root.children) + sum(node_count(child) for child in root.children.values())


def plain_trie_edges(words):
    """Edges of an uncompressed trie holding the same words (prefix set size)."""
    prefixes = set()
    for word in words:
        for i in range(1, len(word) + 1):
            prefixes.add(word[:i])
    return len(prefixes)


def main():
    root = RadixNode("")
    assert insert(root, "cart") is True
    assert shape(root) == "[cart*]"
    assert insert(root, "car") is True                      # split: car(*)+t(*)
    assert shape(root) == "[car*[t*]]"
    assert insert(root, "care") is True                     # "care" splits under car
    assert shape(root) == "[car*[e*,t*]]"
    assert insert(root, "car") is False                     # duplicate word

    for word in ("car", "care", "cart"):
        assert contains(root, word)
    for word in ("ca", "c", "carts", "caret", "card", ""):
        assert not contains(root, word)

    assert node_count(root) == 3
    assert plain_trie_edges(["cart", "car", "care"]) == 5
    assert node_count(root) < plain_trie_edges(["cart", "car", "care"])

    assert insert(root, "do") is True
    assert insert(root, "dot") is True
    assert insert(root, "dog") is True
    assert shape(root) == "[car*[e*,t*],do*[g*,t*]]"

    assert words_with_prefix(root, "car") == ["car", "care", "cart"]
    assert words_with_prefix(root, "ca") == ["car", "care", "cart"]   # ends inside an edge
    assert words_with_prefix(root, "care") == ["care"]
    assert words_with_prefix(root, "do") == ["do", "dog", "dot"]
    assert words_with_prefix(root, "d") == ["do", "dog", "dot"]
    assert words_with_prefix(root, "") == ["car", "care", "cart", "do", "dog", "dot"]
    assert words_with_prefix(root, "z") == []
    assert words_with_prefix(root, "carts") == []
    assert starts_with(root, "car") and not starts_with(root, "cat")

    assert node_count(root) == 6
    assert plain_trie_edges(["car", "care", "cart", "do", "dog", "dot"]) == 9
    for word in ("car", "care", "cart", "do", "dog", "dot"):
        assert contains(root, word)
    for word in ("co", "dat", "dodge", "done"):
        assert not contains(root, word)


if __name__ == "__main__":
    main()
    print("PASS")
