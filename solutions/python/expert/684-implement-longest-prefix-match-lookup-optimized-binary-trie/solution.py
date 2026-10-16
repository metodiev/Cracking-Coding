"""Task 684 - Implement longest-prefix-match lookup optimized (binary trie).  (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)

Two lookup structures are built over the same IPv4 prefix table and cross-checked:

* `BinaryTrie` — the textbook bit-by-bit trie.  A node whose bit index is `b` tests bit `b` of the
  address, and a prefix of length `L` is stored at the node reached after exactly `L` tests.  Every
  node therefore *matches* the address that walks through it, so longest-prefix match is a single
  downward pass that keeps the deepest next hop seen: the "other" child never has to be explored,
  which is precisely the property a plain sorted list of prefixes lacks.

* `CompressedTrie` — the optimized variant.  Chains of nodes that carry no next hop and have a single
  child are fused away.  The surviving node keeps the absolute bit index it tests plus `rep`, the
  `bit` address bits that led to it in the plain trie.  A lookup verifies `rep` (one mask compare)
  instead of walking the fused bits, and a node whose `rep` disagrees also rules out its whole
  subtree, because prefixes on a root-to-leaf path are nested proper extensions.  Hence the descent
  stops early and visits at most as many nodes as the plain trie.

Invariants checked by `BinaryTrie.validate()`: a child sits exactly one bit below its parent, a node
carrying a next hop sits at bit index == prefix length, and a stored prefix agrees with the trie path
that reaches it.  `brute_force_lpm` (a linear scan keeping the longest match) is the oracle for every
lookup, on hand-written tables and on pseudo-random tables drawn from a fixed-seed splitmix64 PRNG.
"""

BITS = 32
FULL_MASK = 0xFFFFFFFF


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def mask_for(plen):
    """The /plen netmask; /0 is all zeroes, /32 all ones."""
    return (FULL_MASK << (BITS - plen)) & FULL_MASK if plen else 0


def ip_to_int(text):
    parts = text.split(".")
    if len(parts) != 4:
        raise ValueError(f"an IPv4 address needs four octets: {text!r}")
    value = 0
    for part in parts:
        if not part.isdigit() or (len(part) > 1 and part[0] == "0") or int(part) > 255:
            raise ValueError(f"bad octet {part!r} in {text!r}")
        value = value << 8 | int(part)
    return value


def int_to_ip(value):
    if not 0 <= value <= FULL_MASK:
        raise ValueError(f"address out of range: {value}")
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def parse_prefix(text):
    """'10.1.2.0/24' -> (masked address, 24).  A missing '/len' means a host route (/32)."""
    addr_text, slash, plen_text = text.partition("/")
    if slash and not plen_text.isdigit():
        raise ValueError(f"bad prefix length in {text!r}")
    if not slash:
        plen = BITS
    else:
        plen = int(plen_text)
    if not 0 <= plen <= BITS:
        raise ValueError(f"prefix length out of range in {text!r}")
    addr = ip_to_int(addr_text)
    return addr & mask_for(plen), plen


def brute_force_lpm(prefixes, addr):
    """Oracle: scan every prefix and keep the longest that matches (ties cannot happen)."""
    best, best_len = None, -1
    for paddr, plen, hop in prefixes:
        if plen > best_len and addr & mask_for(plen) == paddr:
            best, best_len = hop, plen
    return best


class _Node:
    __slots__ = ("bit", "addr", "plen", "hop", "children")

    def __init__(self, bit):
        self.bit = bit              # absolute bit index tested at this node
        self.addr = 0               # the prefix address (masked) when a next hop is stored here
        self.plen = -1
        self.hop = None
        self.children = [None, None]


class BinaryTrie:
    """One node per bit level; every node on the root-to-store path matches the queried address."""

    def __init__(self):
        self.root = None
        self.visits = 0             # nodes touched by lookup(), used to measure the optimisation

    def insert(self, addr, plen, hop):
        if self.root is None:
            self.root = _Node(0)
        node = self.root
        while node.bit < plen:                     # create the missing levels, one bit at a time
            index = (addr >> (BITS - 1 - node.bit)) & 1
            if node.children[index] is None:
                node.children[index] = _Node(node.bit + 1)
            node = node.children[index]
        node.hop = hop
        node.plen = plen
        node.addr = addr & mask_for(plen)

    def lookup(self, addr):
        best = None
        node = self.root
        while node is not None:
            self.visits += 1
            if node.hop is not None:               # the deepest stored hop is the longest match
                best = node.hop
            if node.bit >= BITS:
                break
            node = node.children[(addr >> (BITS - 1 - node.bit)) & 1]
        return best

    def delete(self, addr, plen):
        """Remove a prefix and prune the hop-less unary chain it left behind."""
        node, path = self.root, []
        while node is not None and node.bit < plen:
            path.append(node)
            node = node.children[(addr >> (BITS - 1 - node.bit)) & 1]
        if node is None or node.hop is None:
            return False
        node.hop, node.plen, node.addr = None, -1, 0
        while path:
            parent = path.pop()
            index = (addr >> (BITS - 1 - parent.bit)) & 1
            child = parent.children[index]
            if child.hop is None and child.children == [None, None]:
                parent.children[index] = None
            else:
                break
        return True

    def _collect(self, node, rep, out):
        if node.hop is not None:
            check(node.plen == node.bit, f"hop node at bit {node.bit} claims plen {node.plen}")
            check(node.addr & mask_for(node.plen) == rep & mask_for(node.plen),
                  f"prefix {int_to_ip(node.addr)}/{node.plen} disagrees with its trie path")
            out.append((node.addr, node.plen, node.hop))
            self.depth = max(self.depth, node.bit)
        for index in (0, 1):
            child = node.children[index]
            if child is None:
                continue
            check(node.bit < BITS, "children only exist below the last bit")
            check(child.bit == node.bit + 1, "a plain trie child sits one bit below its parent")
            self.nodes += 1
            self._collect(child, rep | (index << (BITS - 1 - node.bit)), out)

    def validate(self):
        """Structural audit; returns the stored prefixes.  Raises on any invariant violation."""
        self.nodes, self.depth = (0, 0) if self.root is None else (1, 0)
        out = []
        if self.root is not None:
            check(self.root.bit == 0, "the plain trie root must test bit 0")
            self._collect(self.root, 0, out)
        return out


class _CNode:
    __slots__ = ("bit", "rep", "plen", "hop", "children")

    def __init__(self, bit, rep, plen, hop):
        self.bit = bit              # bit index tested here (unchanged by compression)
        self.rep = rep              # the `bit` path bits that reach this node in the plain trie
        self.plen = plen
        self.hop = hop
        self.children = [None, None]

    def matches(self, addr):
        return addr & mask_for(self.bit) == self.rep & mask_for(self.bit)


class CompressedTrie:
    """`BinaryTrie` with hop-less unary chains fused into their descendants."""

    def __init__(self, prefixes):
        plain = BinaryTrie()
        for addr, plen, hop in prefixes:
            plain.insert(addr, plen, hop)
        self.visits = 0
        self.root = None if plain.root is None else self._build(plain.root, 0)

    def _build(self, node, rep):
        if node.hop is None and node.bit < BITS:
            only = [i for i in (0, 1) if node.children[i] is not None]
            if len(only) == 1:
                index = only[0]
                return self._build(node.children[index], rep | (index << (BITS - 1 - node.bit)))
        fused = _CNode(node.bit, rep, node.plen, node.hop)
        for index in (0, 1):
            if node.children[index] is not None:
                child_rep = rep | (index << (BITS - 1 - node.bit))
                fused.children[index] = self._build(node.children[index], child_rep)
        return fused

    def lookup(self, addr):
        best, node = None, self.root
        while node is not None:
            self.visits += 1
            if not node.matches(addr):         # rep disagrees: this node and its subtree are out
                break
            if node.hop is not None:
                best = node.hop
            if node.bit >= BITS:
                break
            node = node.children[(addr >> (BITS - 1 - node.bit)) & 1]
        return best

    def stats(self):
        """(node count, deepest tested bit, node count on the longest root-to-leaf path)."""
        nodes = 0
        max_bit = 0
        max_path = 0

        def walk(node, path_len):
            nonlocal nodes, max_bit, max_path
            nodes += 1
            max_bit = max(max_bit, node.bit)
            max_path = max(max_path, path_len)
            for index in (0, 1):
                child = node.children[index]
                if child is not None:
                    check(child.bit > node.bit, "compression must keep bit indices increasing")
                    walk(child, path_len + 1)

        if self.root is not None:
            walk(self.root, 1)
        return nodes, max_bit, max_path


def make_prng(seed):
    """splitmix64 — deterministic, so every run generates exactly the same tables."""
    state = seed & 0xFFFFFFFFFFFFFFFF

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    return nxt


def random_table(prng, count, max_plen=BITS):
    table = {}
    for _ in range(count):
        plen = prng() % (max_plen + 1)
        addr = (prng() & FULL_MASK) & mask_for(plen)
        table[(addr, plen)] = prng() % 1000
    return [(addr, plen, hop) for (addr, plen), hop in sorted(table.items())]


def main():
    check(int_to_ip(ip_to_int("0.0.0.0")) == "0.0.0.0", "IPv4 round trip at zero")
    check(int_to_ip(ip_to_int("255.255.255.255")) == "255.255.255.255", "IPv4 round trip at max")
    check(mask_for(0) == 0 and mask_for(32) == FULL_MASK and mask_for(24) == 0xFFFFFF00,
          "netmask table")
    for bad in ("10.0.0.256/8", "10.0.0/8", "10.0.0.0/33", "10.0.0.0/-1", "/8", "10.0.0.a/8",
                "010.0.0.1/8", "10.0.0.0/2x", "1.2.3.4/5/6"):
        try:
            parse_prefix(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid prefix {bad!r}")
    check(parse_prefix("10.0.0.0") == (ip_to_int("10.0.0.0"), 32), "bare address is a host route")
    check(parse_prefix("10.1.2.77/24")[0] == ip_to_int("10.1.2.0"), "host bits are masked off")

    routes = [("0.0.0.0/0", 1), ("10.0.0.0/8", 10), ("10.1.0.0/16", 20),
              ("10.1.2.0/24", 30), ("10.1.2.3/32", 40), ("192.168.0.0/16", 50)]
    prefixes = [(parse_prefix(text)[0], parse_prefix(text)[1], hop) for text, hop in routes]

    plain = BinaryTrie()
    for addr, plen, hop in prefixes:
        plain.insert(addr, plen, hop)
    check(plain.lookup(ip_to_int("10.1.2.3")) == 40, "a /32 host route beats the enclosing /24")
    check(plain.lookup(ip_to_int("10.1.2.9")) == 30, "a /24 beats its /16 parent")
    check(plain.lookup(ip_to_int("10.1.9.9")) == 20, "a /16 beats its /8 parent")
    check(plain.lookup(ip_to_int("10.9.9.9")) == 10, "the /8 is the longest match here")
    check(plain.lookup(ip_to_int("172.16.0.1")) == 1, "the default route catches the rest")
    check(plain.lookup(ip_to_int("192.168.5.5")) == 50, "a disjoint /16")

    stored = plain.validate()
    check(len(stored) == len(prefixes), "every inserted prefix is stored exactly once")
    plain.delete(parse_prefix("0.0.0.0/0")[0], 0)
    check(plain.lookup(ip_to_int("172.16.0.1")) is None, "no default route, no match")
    plain.delete(*parse_prefix("10.1.2.0/24"))
    check(plain.lookup(ip_to_int("10.1.2.9")) == 20, "after deleting the /24 the /16 wins")
    check(plain.delete(*parse_prefix("10.1.2.0/24")) is False, "deleting twice is a no-op")
    plain.delete(*parse_prefix("10.1.0.0/16"))
    check(plain.lookup(ip_to_int("10.1.2.9")) == 10, "after deleting the /16 the /8 wins")
    plain.insert(*parse_prefix("10.1.0.0/16"), 20)
    plain.insert(*parse_prefix("10.1.2.0/24"), 30)
    plain.insert(0, 0, 1)
    check(len(plain.validate()) == len(prefixes), "delete then re-insert restores the table")

    for addr in (ip_to_int("10.1.2.1"), ip_to_int("8.8.8.8"), ip_to_int("255.255.255.255")):
        check(plain.lookup(addr) == brute_force_lpm(prefixes, addr), "hand table vs brute force")

    prng = make_prng(0xC0FFEE)
    for max_plen in (8, 16, 24, 32):
        table = random_table(prng, 220, max_plen)
        trie = BinaryTrie()
        for addr, plen, hop in table:
            trie.insert(addr, plen, hop)
        check(len(trie.validate()) == len(table), f"small table with /{max_plen} prefixes")
        for _ in range(220):
            addr = prng() & FULL_MASK
            check(trie.lookup(addr) == brute_force_lpm(table, addr),
                  f"trie lookup mismatch at {int_to_ip(addr)} (/{max_plen} table)")
        compressed = CompressedTrie(table)
        queries = [prng() & FULL_MASK for _ in range(120)]
        for addr in queries:
            check(compressed.lookup(addr) == brute_force_lpm(table, addr),
                  f"compressed lookup mismatch at {int_to_ip(addr)}")
        c_nodes, _, _ = compressed.stats()
        check(c_nodes <= trie.nodes, "compression cannot add nodes")
        trie.visits = compressed.visits = 0
        for addr in queries:
            plain_before = trie.visits
            trie.lookup(addr)
            compressed_before = compressed.visits
            compressed.lookup(addr)
            check(compressed.visits - compressed_before <= trie.visits - plain_before,
                  "a compressed descent never visits more nodes than the plain descent")

    chain = [(ip_to_int("10.0.0.0") | (i << 8) | 1, 32, i) for i in range(0, 64)]
    chain_trie = BinaryTrie()
    for addr, plen, hop in chain:
        chain_trie.insert(addr, plen, hop)
    chain_trie.validate()
    chain_compressed = CompressedTrie(chain)
    chain_nodes, chain_bit, _ = chain_compressed.stats()
    check(chain_trie.nodes > len(chain), "64 host routes share long unary chains")
    check(chain_nodes == 2 * len(chain) - 1, "fusing leaves only the branch nodes and the leaves")
    chain_trie.visits = chain_compressed.visits = 0
    for addr, plen, hop in chain:
        check(chain_trie.lookup(addr) == hop, "host route reachable through the plain trie")
        check(chain_compressed.lookup(addr) == hop, "host route survives compression")
    check(chain_compressed.visits < chain_trie.visits, "fusing unary chains saves node visits")
    chain_compressed.visits = 0
    check(chain_compressed.lookup(ip_to_int("10.0.99.0")) is None, "unrouted address after fusing")
    check(chain_compressed.visits == 1, "the fused root rejects 10.0.99.0 in a single step")
    check(CompressedTrie([]).lookup(ip_to_int("1.2.3.4")) is None, "an empty table matches nothing")
    check(BinaryTrie().validate() == [], "an empty trie has no prefixes")

    print(f"plain trie: {plain.nodes} nodes, max test bit {plain.depth}")
    print(f"random table: {chain_trie.nodes} plain nodes vs {chain_nodes} compressed nodes")
    print(f"checked {len(prefixes)} hand-written and {220 * 4} generated prefixes")
    print("PASS")


if __name__ == "__main__":
    main()
