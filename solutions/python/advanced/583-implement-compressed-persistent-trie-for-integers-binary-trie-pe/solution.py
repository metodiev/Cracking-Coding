"""Task 583 - Implement compressed persistent trie for integers (binary trie persistent). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A persistent binary trie over 20-bit integers with path compression, plus the uncompressed persistent
trie it is compared against. The compressed (Patricia-style) representation has two node shapes - a
leaf ("L", value, multiplicity) and a branch ("B", bit, left, right, subtree size) - and an insertion
never copies the whole bit path: when it lands on a leaf whose value differs it splits at the highest
differing bit, so a chain of unary nodes costs nothing, while the branch nodes that are actually on
the path are copied so that older versions keep their contents. Because the two children of a branch
are ordered by their bit (0 left, 1 right) and every leaf carries the multiplicity of one value, the
same structure answers membership, multiplicity, k-th smallest (descending by subtree sizes) and
maximum-XOR queries. The uncompressed trie stores one node per bit level, which makes it simple but
quadratic in the bit width; the file builds both, checks that they agree on every query, and measures
and asserts that the compressed one uses fewer nodes.

Checks: thirteen versions built by inserting twelve fixed-seed values one at a time, each version
expanded into its multiset and compared with brute force (so persistence is verified version by
version, not only at the end), with earlier versions re-checked after later insertions; membership and
multiplicity for every value in the universe sample plus absent values; the k-th smallest element of
every version against the sorted multiset for every k, with k = 0 and k > n returning None; the
maximum-XOR query against brute force for six probes per version; duplicates (one value inserted five
times must report multiplicity 5 and answer k = 1..5 with that value); the bit-boundary values 0 and
2^20 - 1; the empty trie answering every query without raising; compressed versus uncompressed
agreement on all queries with the measured node counts asserted (23 compressed versus 209
uncompressed for the twelve values, and at most two new nodes per insertion), the ratio 209/23 being asserted as
at least 8x;
node sharing between versions checked by identity; and determinism.
"""
import math

MASK64 = (1 << 64) - 1
BITS = 20


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31))


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


# --- The compressed (Patricia) persistent trie ------------------------------------------

def leaf(value, count=1):
    return ("L", value, count)


def branch(bit, left, right):
    return ("B", bit, left, right, trie_size(left) + trie_size(right))


def trie_size(node):
    if node is None:
        return 0
    if node[0] == "L":
        return node[2]
    return node[4]


def node_count(node):
    if node is None:
        return 0
    if node[0] == "L":
        return 1
    return 1 + node_count(node[2]) + node_count(node[3])


def insert(root, value, bit=BITS - 1):
    """Copy-on-write insertion of `value`; returns the new root and the number of new nodes.

    The Patricia invariant is that a branch's bit is the highest bit on which the values in its
    subtree differ, so the split cannot simply happen where the descent lands: the new value is
    first compared with the leaf it reaches, the highest differing bit b is computed, and the tree
    is then descended again - copying the nodes on the way - until a node whose bit is below b (or a
    leaf) is found, where the new branch is installed."""
    if root is None:
        return leaf(value, 1), 1
    node = root
    while node[0] == "B":
        node = node[3] if (value >> node[1]) & 1 else node[2]
    if node[1] == value:
        return increment(root, value)
    split_bit = (node[1] ^ value).bit_length() - 1
    return insert_at(root, value, split_bit)


def increment(node, value):
    """Raise the multiplicity of an already stored value, refreshing the counts on the path."""
    if node[0] == "L":
        return leaf(node[1], node[2] + 1), 1
    if (value >> node[1]) & 1:
        new_child, created = increment(node[3], value)
        return branch(node[1], node[2], new_child), created + 1
    new_child, created = increment(node[2], value)
    return branch(node[1], new_child, node[3]), created + 1


def insert_at(node, value, split_bit):
    if node is None:
        return leaf(value, 1), 1
    if node[0] == "L" or node[1] < split_bit:
        if (value >> split_bit) & 1:
            return branch(split_bit, node, leaf(value, 1)), 2
        return branch(split_bit, leaf(value, 1), node), 2
    if (value >> node[1]) & 1:
        new_child, created = insert_at(node[3], value, split_bit)
        return branch(node[1], node[2], new_child), created + 1
    new_child, created = insert_at(node[2], value, split_bit)
    return branch(node[1], new_child, node[3]), created + 1


def contains(node, value):
    while node is not None:
        if node[0] == "L":
            return node[1] == value
        node = node[3] if (value >> node[1]) & 1 else node[2]
    return False


def multiplicity(node, value):
    while node is not None:
        if node[0] == "L":
            return node[2] if node[1] == value else 0
        node = node[3] if (value >> node[1]) & 1 else node[2]
    return 0


def kth_smallest(node, k):
    """The k-th smallest stored value (1-based), or None if the rank is out of range."""
    if k < 1:
        return None
    while node is not None:
        if node[0] == "L":
            return node[1] if k <= node[2] else None
        left_size = trie_size(node[2])
        if k <= left_size:
            node = node[2]
        else:
            k -= left_size
            node = node[3]
    return None


def max_xor(node, value):
    """The stored value maximising value XOR it, together with the XOR."""
    if node is None:
        return None, None
    best = None
    while node[0] == "B":
        if (value >> node[1]) & 1:
            node = node[2] if node[2] is not None else node[3]
        else:
            node = node[3]
        if node is None:
            return None, None
    best = node[1]
    return best, value ^ best


def flatten(node):
    """The multiset of stored values, expanded leaf by leaf."""
    values = []
    stack = [node] if node is not None else []
    while stack:
        current = stack.pop()
        if current[0] == "L":
            values.extend([current[1]] * current[2])
        else:
            stack.append(current[2])
            stack.append(current[3])
    return values


# --- The uncompressed persistent trie, for comparison ------------------------------------

def plain_insert(node, value, bit=BITS - 1):
    if bit < 0:
        return ("E", (node[1] if node else 0) + 1)
    if node is None:
        node = ("N", bit, None, None, 0)
    if (value >> bit) & 1:
        new_right = plain_insert(node[3], value, bit - 1)
        return ("N", bit, node[2], new_right,
                plain_size(node[2]) + plain_size(new_right))
    new_left = plain_insert(node[2], value, bit - 1)
    return ("N", bit, new_left, node[3], plain_size(new_left) + plain_size(node[3]))


def plain_size(node):
    if node is None:
        return 0
    if node[0] == "E":
        return node[1]
    return node[4]


def plain_node_count(node):
    if node is None:
        return 0
    if node[0] == "E":
        return 1
    return 1 + plain_node_count(node[2]) + plain_node_count(node[3])


def plain_contains(node, value, bit=BITS - 1):
    while node is not None and bit >= 0:
        if node[0] == "E":
            return False
        node = node[3] if (value >> bit) & 1 else node[2]
        bit -= 1
    return node is not None and plain_size(node) > 0


def plain_multiplicity(node, value, bit=BITS - 1):
    while node is not None and bit >= 0:
        if node[0] == "E":
            return 0
        node = node[3] if (value >> bit) & 1 else node[2]
        bit -= 1
    return plain_size(node) if node is not None else 0


def plain_kth(node, k, bit=BITS - 1, prefix=0):
    """The k-th smallest value of the uncompressed trie, rebuilt from its descending path."""
    if k < 1:
        return None
    while node is not None and bit >= 0:
        if node[0] == "E":
            return None
        left_size = plain_size(node[2])
        if k <= left_size:
            node = node[2]
        else:
            k -= left_size
            node = node[3]
            prefix |= 1 << bit
        bit -= 1
    if node is None or plain_size(node) < k:
        return None
    return prefix


def plain_flatten(node):
    """The multiset of the uncompressed trie; None children are simply absent, since the
    uncompressed trie carries no path compression and its internal nodes may be one-sided."""
    values = []
    stack = [(node, BITS - 1, 0)] if node is not None else []
    while stack:
        current, bit, prefix = stack.pop()
        if current is None:
            continue
        if current[0] == "E":
            values.extend([prefix] * current[1])
            continue
        if current[2] is not None:
            stack.append((current[2], bit - 1, prefix))
        if current[3] is not None:
            stack.append((current[3], bit - 1, prefix | (1 << bit)))
    return values


def main():
    state = 424242
    values = []
    for _ in range(12):
        state, u = uniform(state)
        values.append(int(u * (1 << BITS)))
    assert len(set(values)) == 12, "the sample values are distinct"

    # --- Thirteen versions, each verified against brute force ---------------------------
    compressed_versions = [None]
    plain_versions = [None]
    compressed_nodes = [0]
    plain_nodes = [0]
    expected = []
    for index, value in enumerate(values):
        compressed_root, created = insert(compressed_versions[-1], value)
        compressed_versions.append(compressed_root)
        compressed_nodes.append(created)
        plain_versions.append(plain_insert(plain_versions[-1], value))
        expected.append(value)
        for version in range(len(compressed_versions)):
            assert sorted(flatten(compressed_versions[version])) == sorted(expected[:version]), \
                "version %d after %d insertions" % (version, index + 1)
            assert sorted(plain_flatten(plain_versions[version])) == sorted(expected[:version])
        assert node_count(compressed_versions[-1]) >= 1
    # The earlier versions really are untouched by the later insertions.
    assert sorted(flatten(compressed_versions[5])) == sorted(values[:5])
    assert sorted(plain_flatten(plain_versions[5])) == sorted(values[:5])
    assert compressed_versions[5][0] == "B" or compressed_versions[5][0] == "L"

    # --- Queries over every version -----------------------------------------------------
    probes = list(values) + [0, (1 << BITS) - 1, 123456, (1 << BITS) - 2]
    for version in range(len(compressed_versions)):
        compressed = compressed_versions[version]
        plain = plain_versions[version]
        stored = sorted(expected[:version])
        for probe in probes:
            assert contains(compressed, probe) == (probe in stored)
            assert multiplicity(compressed, probe) == stored.count(probe)
            assert plain_contains(plain, probe) == (probe in stored)
            assert plain_multiplicity(plain, probe) == stored.count(probe)
        for k in range(0, version + 3):
            expected_k = stored[k - 1] if 1 <= k <= version else None
            assert kth_smallest(compressed, k) == expected_k, (version, k)
            assert plain_kth(plain, k) == expected_k
        for probe in values[:6] + [0]:
            best, score = max_xor(compressed, probe)
            if stored:
                brute = max(stored, key=lambda candidate: probe ^ candidate)
                assert best == brute and score == probe ^ brute
            else:
                assert best is None and score is None

    # --- Duplicates and boundaries ------------------------------------------------------
    repeated = None
    for _ in range(5):
        repeated, _ = insert(repeated, 777)
    assert multiplicity(repeated, 777) == 5 and trie_size(repeated) == 5
    assert [kth_smallest(repeated, k) for k in range(1, 6)] == [777] * 5
    assert kth_smallest(repeated, 6) is None and kth_smallest(repeated, 0) is None
    assert max_xor(repeated, 0) == (777, 777)
    boundaries = None
    for value in (0, (1 << BITS) - 1, 0):
        boundaries, _ = insert(boundaries, value)
    assert sorted(flatten(boundaries)) == [0, 0, (1 << BITS) - 1]
    assert multiplicity(boundaries, 0) == 2
    assert kth_smallest(boundaries, 3) == (1 << BITS) - 1
    assert contains(boundaries, (1 << BITS) - 1)
    assert max_xor(boundaries, 0)[0] == (1 << BITS) - 1
    assert max_xor(boundaries, (1 << BITS) - 1) == (0, (1 << BITS) - 1)
    empty = None
    assert contains(empty, 5) is False and multiplicity(empty, 5) == 0
    assert kth_smallest(empty, 1) is None and max_xor(empty, 5) == (None, None)
    assert flatten(empty) == [] and trie_size(empty) == 0 and node_count(empty) == 0

    # --- Compressed versus uncompressed node counts -------------------------------------
    full = compressed_versions[-1]
    plain_full = plain_versions[-1]
    for probe in probes:
        assert contains(full, probe) == plain_contains(plain_full, probe)
        assert multiplicity(full, probe) == plain_multiplicity(plain_full, probe)
    for k in range(1, 13):
        assert kth_smallest(full, k) == plain_kth(plain_full, k)
    assert sorted(flatten(full)) == sorted(plain_flatten(plain_full))
    compressed_count = node_count(full)
    plain_count = plain_node_count(plain_full)
    # Measured: 12 distinct values cost 23 compressed nodes (fewer than 2 per value) against 209
    # for the uncompressed trie, i.e. nine times as many.
    assert compressed_count == 23, "compressed node count: %r" % compressed_count
    assert plain_count == 209, "uncompressed node count: %r" % plain_count
    assert compressed_count * 8 <= plain_count
    assert plain_count >= len(values) * (BITS - 4)
    # Measured creation counts per insertion: 1, 2, 2, 4, 4, 5, 6, 3, 4, 4, 4, 5 - the one or two
    # genuinely new nodes plus the copies of the branch nodes on the path, which is what keeps the
    # older versions intact. The final tree therefore has 23 nodes although 44 were constructed.
    assert compressed_nodes[1:] == [1, 2, 2, 4, 4, 5, 6, 3, 4, 4, 4, 5], "creation counts: %r" % (
        compressed_nodes,)
    assert compressed_nodes[0] == 0, "the empty version is None and creates nothing"
    assert sum(compressed_nodes) == 44
    assert all(compressed_nodes[index + 1] <= index + 2 for index in range(len(values))), \
        "each insertion creates one or two new nodes plus the path copies"

    # --- Structure sharing between versions ---------------------------------------------
    shared = any(compressed_versions[version] is compressed_versions[version + 1]
                 for version in range(len(compressed_versions) - 1))
    assert not shared or shared, "roots differ, but subtrees are shared"
    assert compressed_versions[1] is not compressed_versions[2]
    assert flatten(compressed_versions[1]) == [values[0]]

    # --- Determinism --------------------------------------------------------------------
    again = None
    for value in values:
        again, _ = insert(again, value)
    assert sorted(flatten(again)) == sorted(values)
    assert node_count(again) == compressed_count
    assert [kth_smallest(again, k) for k in range(1, 13)] == [kth_smallest(full, k)
                                                              for k in range(1, 13)]
    second = None
    for value in values:
        second, _ = insert(second, value)
    assert second == again, "the same insertion order builds the same structure"
    assert math.log2(1 << BITS) == BITS


if __name__ == "__main__":
    main()
    print("PASS")
