"""Task 683 - Implement compacted trie (Patricia trie) for IP routing table simulation.  (Tier: Expert, Beyond — Networking, OS & Real-Time)

A path-compressed binary trie (Patricia trie) for CIDR longest-prefix-match routing.

Structure
  * every node stores a prefix (`bits` plus its `length`) and an optional next hop; nodes
    that only exist to branch have no next hop;
  * insertion descends while the stored prefix is a proper prefix of the new one; on a
    divergence the two are split at their common prefix and a new intermediate node takes
    over, and a new prefix that is an *ancestor* of an existing node is inserted above it,
    so the trie stays compressed: no node has a single child unless it also carries a next
    hop;
  * lookup walks down while the node's prefix matches the address, remembering the longest
    match with a next hop, which is exactly longest-prefix-match;
  * deletion removes the node, reattaching its only child to its parent, and re-verifies the
    compression invariant.

Verified: an independent linear scan of the routing table (mask, compare, keep the longest)
agrees with every lookup on random tables; /0, /32 and duplicate-prefix updates behave
correctly; deleting a prefix falls back to the covering route; the structure invariants hold
after every mutation (children labelled by bit, single-child nodes always carry a next hop,
node count linear in the number of prefixes); and the compressed trie uses strictly fewer
nodes than the uncompressed bit trie built from the same table, which is the whole point of
the compaction. IPv4 text parsing and formatting round-trip.

Simplified: IPv4 only, one next hop per prefix, no metric/tie-breaking between equal-length
prefixes and no incremental updates from a routing protocol.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


WIDTH = 32


def parse_ipv4(text):
    """'10.1.0.0/16' -> (address, length); a bare address means a /32 host route."""
    address_text, _, length_text = text.partition("/")
    octets = address_text.split(".")
    if len(octets) != 4:
        raise ValueError("an IPv4 address needs four octets: %r" % (text,))
    if length_text and not length_text.isdigit():
        raise ValueError("a prefix length must be numeric: %r" % (text,))
    length = int(length_text) if length_text else WIDTH
    if not 0 <= length <= WIDTH:
        raise ValueError("a prefix length must be between 0 and 32: %r" % (text,))
    value = 0
    for octet in octets:
        if not octet.isdigit() or int(octet) > 255:
            raise ValueError("bad octet in %r" % (text,))
        value = (value << 8) | int(octet)
    return value, length


def format_ipv4(address, length):
    octets = [(address >> shift) & 0xFF for shift in (24, 16, 8, 0)]
    return "%d.%d.%d.%d/%d" % (octets[0], octets[1], octets[2], octets[3], length)


def mask_of(length):
    return (0xFFFFFFFF << (WIDTH - length)) & 0xFFFFFFFF if length else 0


def matches(address, prefix, length):
    return (address & mask_of(length)) == (prefix & mask_of(length))


class Node:
    __slots__ = ("bits", "length", "nexthop", "left", "right")

    def __init__(self, prefix, length, nexthop=None):
        self.bits = prefix >> (WIDTH - length)     # the prefix's own bits, left aligned
        self.length = length
        self.nexthop = nexthop
        self.left = None
        self.right = None

    def children(self):
        return [child for child in (self.left, self.right) if child is not None]


class PatriciaTrie:
    """Compressed binary trie keyed by CIDR prefixes."""

    def __init__(self):
        self.root = None
        self.size = 0

    # ---- helpers ------------------------------------------------------------
    def _child_bit(self, node, bit):
        return node.right if bit else node.left

    def _set_child(self, node, bit, child):
        if bit:
            node.right = child
        else:
            node.left = child

    def _bit_at(self, prefix, index):
        return (prefix >> (WIDTH - 1 - index)) & 1

    def _common_prefix(self, node, prefix, length):
        """Number of leading bits shared by the node's prefix and the new one."""
        limit = min(node.length, length)
        for index in range(limit):
            if self._bit_at(node.bits << (WIDTH - node.length), index) != \
                    self._bit_at(prefix, index):
                return index
        return limit

    def insert(self, prefix, length, nexthop):
        """Insert or update a prefix; returns True when the prefix is new."""
        if nexthop is None:
            raise ValueError("a route needs a next hop")
        if self.root is None:
            self.root = Node(prefix, length, nexthop)
            self.size += 1
            return True
        node, parent, parent_bit = self.root, None, 0
        while node is not None and node.length <= length and \
                (prefix >> (WIDTH - node.length)) == node.bits:
            if node.length == length:
                new = node.nexthop is None
                node.nexthop = nexthop
                self.size += int(new)
                return new
            parent, parent_bit = node, self._bit_at(prefix, node.length)
            node = self._child_bit(node, parent_bit)
        if node is None:
            leaf = Node(prefix, length, nexthop)
            if parent is None:
                self.root = leaf
            else:
                self._set_child(parent, parent_bit, leaf)
            self.size += 1
            return True
        if node.length > length and \
                (node.bits >> (node.length - length)) == (prefix >> (WIDTH - length)):
            split = Node(prefix, length, nexthop)
            self._set_child(split, self._bit_at(node.bits << (WIDTH - node.length), length), node)
        else:
            shared = self._common_prefix(node, prefix, length)
            split = Node(prefix, shared)
            self._set_child(split, self._bit_at(node.bits << (WIDTH - node.length), shared), node)
            leaf = Node(prefix, length, nexthop)
            self._set_child(split, self._bit_at(prefix, shared), leaf)
        if parent is None:
            self.root = split
        else:
            self._set_child(parent, parent_bit, split)
        self.size += 1
        return True

    def search(self, address):
        """Longest-prefix match; returns the next hop or None."""
        node = self.root
        best = None
        while node is not None:
            if (address >> (WIDTH - node.length)) != node.bits:
                break
            if node.nexthop is not None:
                best = node.nexthop
            if node.length >= WIDTH:
                break
            node = self._child_bit(node, self._bit_at(address, node.length))
        return best

    def delete(self, prefix, length):
        """Remove a route; returns True when it existed."""
        path = []                      # (node, parent, bit) from the root downwards
        node, parent, parent_bit = self.root, None, 0
        while node is not None:
            if (prefix >> (WIDTH - node.length)) != node.bits:
                return False
            path.append((node, parent, parent_bit))
            if node.length == length:
                break
            if node.length >= WIDTH:
                return False
            parent, parent_bit = node, self._bit_at(prefix, node.length)
            node = self._child_bit(node, parent_bit)
        if node is None or node.length != length or node.nexthop is None:
            return False
        children = node.children()
        if len(children) == 2:
            node.nexthop = None            # still a branch point, just not a route
            self.size -= 1
            return True
        self._splice(path, children[0] if children else None)
        self.size -= 1
        return True

    def _splice(self, path, replacement):
        """Install `replacement` where the deleted node was and keep the trie compressed by
        also removing any transit node left with a single child and no next hop."""
        for index in range(len(path) - 1, -1, -1):
            node, parent, parent_bit = path[index]
            if parent is None:
                self.root = replacement
            else:
                self._set_child(parent, parent_bit, replacement)
            if index == 0 or parent.nexthop is not None or len(parent.children()) == 2:
                return
            replacement = parent.children()[0]     # splice the transit node out

    # ---- inspection ---------------------------------------------------------
    def entries(self):
        found = []

        def walk(node):
            if node is None:
                return
            if node.nexthop is not None:
                found.append((node.bits << (WIDTH - node.length), node.length, node.nexthop))
            walk(node.left)
            walk(node.right)

        walk(self.root)
        return found

    def node_count(self):
        def walk(node):
            return 0 if node is None else 1 + walk(node.left) + walk(node.right)

        return walk(self.root)

    def problems(self):
        """Structural invariants; an empty list means the trie is a valid Patricia trie."""
        issues = []

        def walk(node, parent):
            if node is None:
                return
            children = node.children()
            if parent is not None:
                if node.length <= parent.length:
                    issues.append("child prefix is not longer than its parent's")
                elif (node.bits >> (node.length - parent.length)) != parent.bits:
                    issues.append("child prefix does not extend the parent prefix")
            if len(children) == 1 and node.nexthop is None:
                issues.append("a single-child node without a next hop is not compressed")
            if node.left is not None and self._bit_at(node.left.bits << (WIDTH - node.left.length),
                                                      node.length) != 0:
                issues.append("a left child must be reached by a zero bit")
            if node.right is not None and self._bit_at(node.right.bits << (WIDTH - node.right.length),
                                                        node.length) != 1:
                issues.append("a right child must be reached by a one bit")
            walk(node.left, node)
            walk(node.right, node)

        walk(self.root, None)
        return issues


class UncompressedTrie:
    """One node per bit of every prefix: the baseline the compaction must beat."""

    def __init__(self):
        self.nodes = 0
        self.root = {}

    def insert(self, prefix, length):
        node = self.root
        if self.nodes == 0:
            self.nodes = 1
        for index in range(length):
            bit = (prefix >> (WIDTH - 1 - index)) & 1
            if bit not in node:
                node[bit] = {}
                self.nodes += 1
            node = node[bit]


def brute_force_match(address, table):
    """Independent oracle: scan the table, keep the longest matching prefix."""
    best, best_length = None, -1
    for prefix, length, nexthop in table:
        if matches(address, prefix, length) and length > best_length:
            best, best_length = nexthop, length
    return best


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_table(seed, prefixes, lengths=(8, 12, 16, 20, 24, 28, 32)):
    """A table of distinct prefixes (duplicates would only re-test the update path)."""
    state = seed
    table = []
    seen = set()
    attempts = 0
    while len(table) < prefixes and attempts < 20 * prefixes:
        attempts += 1
        state, raw = splitmix64(state)
        length = lengths[raw % len(lengths)]
        state, raw = splitmix64(state)
        prefix = raw & 0xFFFFFFFF
        key = (prefix & mask_of(length), length)
        if key in seen:
            continue
        seen.add(key)
        table.append((prefix, length, "nh%d" % len(table)))
    return table


def random_addresses(seed, count):
    state = seed
    addresses = []
    for _ in range(count):
        state, raw = splitmix64(state)
        addresses.append(raw & 0xFFFFFFFF)
    return addresses


def main():
    # ---- text round trip --------------------------------------------------------------
    address, length = parse_ipv4("10.1.2.0/24")
    check(format_ipv4(address, length) == "10.1.2.0/24", "text round trip")
    check(parse_ipv4("0.0.0.0/0") == (0, 0), "the default route parses")
    check(parse_ipv4("255.255.255.255/32")[1] == 32, "a host route parses")
    check(parse_ipv4("10.1.2.3") == (parse_ipv4("10.1.2.3/32")[0], 32),
          "a bare address means a host route")
    check(mask_of(0) == 0 and mask_of(32) == 0xFFFFFFFF and mask_of(8) == 0xFF000000,
          "masks: /0, /32 and /8")
    rejected = 0
    for bad in ("10.0.0.0/x", "10.0.0.0/33", "10.0.0/8", "10.0.0.256/8", "", "10.0.0.0/-1"):
        try:
            parse_ipv4(bad)
        except ValueError:
            rejected += 1
    check(rejected == 6, "malformed prefixes must be rejected (%d of 6)" % rejected)

    # ---- the classic table ------------------------------------------------------------
    trie = PatriciaTrie()
    routes = [("0.0.0.0/0", "default"), ("10.0.0.0/8", "A"), ("10.1.0.0/16", "B"),
              ("10.1.2.0/24", "C"), ("192.168.0.0/16", "D"), ("10.1.2.3/32", "host")]
    for text, nexthop in routes:
        prefix, prefix_length = parse_ipv4(text)
        check(trie.insert(prefix, prefix_length, nexthop), "%s must be new" % text)
    check(trie.size == len(routes), "the table must hold every route")
    check(trie.problems() == [], "the trie must satisfy its invariants: %r" % (trie.problems(),))
    check(trie.search(parse_ipv4("10.1.2.3")[0]) == "host", "a /32 beats the /24")
    check(trie.search(parse_ipv4("10.1.2.99")[0]) == "C", "the /24 covers the rest of its range")
    check(trie.search(parse_ipv4("10.1.9.9")[0]) == "B", "the /16 covers the middle range")
    check(trie.search(parse_ipv4("10.9.9.9")[0]) == "A", "the /8 covers 10/8")
    check(trie.search(parse_ipv4("8.8.8.8")[0]) == "default", "everything else hits the default")
    check(trie.search(parse_ipv4("192.168.5.5")[0]) == "D", "the other /16 matches")
    for text, nexthop in routes:
        prefix, prefix_length = parse_ipv4(text)
        check(trie.search(prefix) is not None, "a route must match itself: %s" % text)

    # ---- duplicates, updates and deletion --------------------------------------------
    nodes_before = trie.node_count()
    check(not trie.insert(parse_ipv4("10.1.2.0/24")[0], 24, "C2"),
          "re-inserting an existing prefix is an update, not a new route")
    check(trie.size == len(routes) and trie.node_count() == nodes_before,
          "an update must not add nodes")
    check(trie.search(parse_ipv4("10.1.2.99")[0]) == "C2", "the update takes effect")

    check(trie.delete(*parse_ipv4("10.1.2.0/24")[:2]), "the /24 can be deleted")
    check(trie.search(parse_ipv4("10.1.2.99")[0]) == "B",
          "after deleting the /24 the /16 covers the address")
    check(trie.problems() == [], "the invariants survive deletion")
    check(not trie.delete(parse_ipv4("10.1.2.0/24")[0], 24), "deleting twice reports failure")
    check(not trie.delete(parse_ipv4("172.16.0.0/12")[0], 12),
          "deleting a route that was never installed reports failure")
    check(trie.delete(*parse_ipv4("10.1.2.3/32")[:2]) and trie.search(parse_ipv4("10.1.2.3")[0]) == "B",
          "removing the host route falls back to the /16")
    check(trie.delete(*parse_ipv4("0.0.0.0/0")[:2]) and trie.search(parse_ipv4("8.8.8.8")[0]) is None,
          "without a default route, unmatched traffic has no next hop")

    # ---- the compaction actually compacts --------------------------------------------
    uncompressed = UncompressedTrie()
    for text, _ in routes:
        prefix, prefix_length = parse_ipv4(text)
        uncompressed.insert(prefix, prefix_length)
    compacted = PatriciaTrie()
    for text, nexthop in routes:
        prefix, prefix_length = parse_ipv4(text)
        compacted.insert(prefix, prefix_length, nexthop)
    check(compacted.node_count() <= 2 * len(routes),
          "a compressed trie is linear in the number of prefixes: %d" % compacted.node_count())
    check(compacted.node_count() < uncompressed.nodes,
          "compaction must beat one node per bit (%d < %d)"
          % (compacted.node_count(), uncompressed.nodes))
    deep = PatriciaTrie()
    deep.insert(*parse_ipv4("128.0.0.0/1")[:2], "half")
    deep.insert(*parse_ipv4("10.0.0.0/32")[:2], "far")
    check(deep.problems() == [] and deep.node_count() <= 3,
          "two far-apart prefixes share one branch node: %d" % deep.node_count())

    # ---- randomized agreement with the linear scan -----------------------------------
    comparisons = 0
    for i in range(60):
        table = random_table(0x1000 + i * 7919, 8 + i % 12)
        built = PatriciaTrie()
        for prefix, prefix_length, nexthop in table:
            built.insert(prefix, prefix_length, nexthop)
        check(built.problems() == [], "invariants broken on random table %d" % i)
        check(built.size == len(table), "size mismatch on random table %d" % i)
        for address in random_addresses(0x2000 + i, 30):
            expected = brute_force_match(address, table)
            check(built.search(address) == expected,
                  "LPM mismatch on table %d address %08x: %r vs %r"
                  % (i, address, built.search(address), expected))
            comparisons += 1
        for prefix, prefix_length, nexthop in table[:4]:
            check(built.delete(prefix, prefix_length), "deletion failed on table %d" % i)
            check(built.problems() == [], "invariants broken after deletion (%d)" % i)
        remaining = [route for route in table[4:]]
        for address in random_addresses(0x3000 + i, 20):
            check(built.search(address) == brute_force_match(address, remaining),
                  "LPM mismatch after deletion on table %d" % i)
            comparisons += 1
    check(comparisons > 2000, "the sweep must be substantial (%d lookups)" % comparisons)

    # ---- emptying and determinism ----------------------------------------------------
    wiped = PatriciaTrie()
    for prefix, prefix_length, nexthop in random_table(0x777, 20):
        wiped.insert(prefix, prefix_length, nexthop)
    removed = 0
    for prefix, prefix_length, nexthop in list(wiped.entries()):
        removed += wiped.delete(prefix, prefix_length)
    check(removed == 20 and wiped.size == 0 and wiped.node_count() == 0,
          "deleting every route must empty the trie")
    check(wiped.search(0x0A000001) is None, "an empty trie matches nothing")
    check(wiped.problems() == [], "an empty trie is valid")

    twice = PatriciaTrie()
    for prefix, prefix_length, nexthop in random_table(0x999, 15):
        twice.insert(prefix, prefix_length, nexthop)
    same = PatriciaTrie()
    for prefix, prefix_length, nexthop in random_table(0x999, 15):
        same.insert(prefix, prefix_length, nexthop)
    check(sorted(twice.entries()) == sorted(same.entries()),
          "building the same table twice must give the same structure")
    check(twice.node_count() == same.node_count(), "and the same node count")

    print("classic table: %d nodes for %d routes (uncompressed would need %d)"
          % (compacted.node_count(), len(routes), uncompressed.nodes))
    print("random sweep: 60 tables, %d lookup comparisons against the linear scan" % comparisons)
    print("PASS")


if __name__ == "__main__":
    main()
