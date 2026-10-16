"""Task 685 - Implement fast IP routing simulation with TCAM-like behavior (bitwise).  (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)

A TCAM (ternary content-addressable memory) is simulated honestly at the bit level:

* Every rule is a (value, mask) pair; a rule *matches* an address when `addr & mask == value`, and
  bits that are zero in the mask are "don't care".  Validating `value & ~mask == 0` rejects rules
  that can never match, exactly like a table manager programming the array.

* The array is evaluated **column by column**, the way the hardware does: the rows of one bit
  position are packed into a Python integer used as a bit-vector (`want_one[bit]` / `want_zero[bit]`,
  precomputed when a rule is written), and a lookup ANDs one precomputed column vector per address
  bit.  The cost per lookup is therefore O(width) regardless of the number of rows, and the resulting
  bit-vector `match_lines` has exactly one bit set per matching rule.  A priority encoder picks the
  lowest set bit (`lines & -lines`), i.e. the rule with the smallest index wins.

* Longest-prefix-match is a *consequence*, not a special case: prefix rules are installed sorted by
  prefix length descending, and then first-match over the array equals LPM.  Installing in a different
  order demonstrably returns a shorter prefix, which is why the manager sorts before programming.

Oracle checks: `sequential_match_vector` re-implements the comparison row by row (no bit packing) and
must produce an identical vector; `brute_force_lpm` (linear longest-match scan) must agree with a
length-sorted TCAM on random tables; non-contiguous masks exercise the ternary (non-prefix) power a
TCAM has and a plain trie does not.  All random data comes from a fixed-seed splitmix64 PRNG, so the
program is deterministic.
"""

BITS = 32
FULL_MASK = 0xFFFFFFFF


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def mask_for(plen):
    return (FULL_MASK << (BITS - plen)) & FULL_MASK if plen else 0


def ip_to_int(text):
    parts = text.split(".")
    if len(parts) != 4:
        raise ValueError(f"an IPv4 address needs four octets: {text!r}")
    value = 0
    for part in parts:
        if not part.isdigit() or int(part) > 255:
            raise ValueError(f"bad octet {part!r} in {text!r}")
        value = value << 8 | int(part)
    return value


def int_to_ip(value):
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def parse_prefix(text):
    addr_text, slash, plen_text = text.partition("/")
    plen = BITS if not slash else int(plen_text)
    if not 0 <= plen <= BITS:
        raise ValueError(f"prefix length out of range in {text!r}")
    return ip_to_int(addr_text) & mask_for(plen), plen


def brute_force_lpm(prefixes, addr):
    """Oracle: longest prefix that matches, by linear scan."""
    best, best_len = None, -1
    for paddr, plen, hop in prefixes:
        if plen > best_len and addr & mask_for(plen) == paddr:
            best, best_len = hop, plen
    return best


class Rule:
    __slots__ = ("value", "mask", "hop", "plen", "label")

    def __init__(self, value, mask, hop, plen=None, label=""):
        if value & ~mask:
            raise ValueError(f"rule {label or int_to_ip(value)} has value bits outside its mask")
        self.value = value
        self.mask = mask
        self.hop = hop
        self.plen = plen if plen is not None else bin(mask).count("1")
        self.label = label or f"{int_to_ip(value)}/{self.plen}"

    def matches(self, addr):
        return addr & self.mask == self.value

    @staticmethod
    def prefix(text, hop):
        addr, plen = parse_prefix(text)
        return Rule(addr, mask_for(plen), hop, plen=plen, label=text)


class TCAM:
    """Ternary array with column-parallel matching and a lowest-index priority encoder."""

    def __init__(self, width=BITS):
        self.width = width
        self.entries = []
        self.program_ops = 0            # rule writes; every write rebuilds the column vectors
        self._rebuild()
        self.program_ops = 0            # an empty array is not a write

    def _rebuild(self):
        self.want_one = [0] * self.width
        self.want_zero = [0] * self.width
        for row, entry in enumerate(self.entries):
            bit_value = 1 << row
            for bit in range(self.width):
                if (entry.mask >> bit) & 1:
                    if (entry.value >> bit) & 1:
                        self.want_one[bit] |= bit_value
                    else:
                        self.want_zero[bit] |= bit_value
        self.row_mask = (1 << len(self.entries)) - 1
        self.program_ops += 1

    def write(self, entry):
        self.entries.append(entry)
        self._rebuild()

    def load(self, entries):
        """Program the array left to right; index order is the priority order."""
        check(len(entries) < 4096, "the simulated array holds fewer than 4096 rows")
        self.entries = list(entries)
        self._rebuild()

    def match_lines(self, addr):
        """Bit-vector of matching rows: one column vector ANDed per address bit."""
        lines = self.row_mask
        for bit in range(self.width):
            if (addr >> bit) & 1:
                lines &= ~self.want_zero[bit]
            else:
                lines &= ~self.want_one[bit]
        return lines & self.row_mask

    def lookup(self, addr, default=None):
        """Priority encoder: the lowest set bit of the match vector is the winning row."""
        lines = self.match_lines(addr)
        if lines == 0:
            return default
        index = (lines & -lines).bit_length() - 1
        return self.entries[index].hop


def sequential_match_vector(entries, addr):
    """Row-by-row oracle for `TCAM.match_lines`; integer bit r means "row r matched"."""
    lines = 0
    for row, entry in enumerate(entries):
        if entry.matches(addr):
            lines |= 1 << row
    return lines


def make_prng(seed):
    state = seed & 0xFFFFFFFFFFFFFFFF

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    return nxt


def prefix_table(prng, count, max_plen=BITS):
    """Deterministic prefix set, kept as index order = longest prefix first (the LPM programming)."""
    table = {}
    for _ in range(count):
        plen = prng() % (max_plen + 1)
        addr = (prng() & FULL_MASK) & mask_for(plen)
        table[(addr, plen)] = prng() % 1000
    prefixes = [(addr, plen, hop) for (addr, plen), hop in sorted(table.items())]
    prefixes.sort(key=lambda item: -item[1])            # stable: equal lengths keep table order
    return prefixes


def main():
    check(mask_for(0) == 0 and mask_for(16) == 0xFFFF0000, "netmask table")

    try:
        Rule(0xFF, 0x0F, 1, label="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("a rule with value bits outside the mask must be rejected")
    check(Rule.prefix("10.0.0.0/8", 7).plen == 8, "the prefix helper keeps the prefix length")
    check(Rule(0x0A, 0xFF, 1).matches(0x0A) and not Rule(0x0A, 0xFF, 1).matches(0x0B),
          "ternary match on a masked bit")

    routes = [("10.1.2.3/32", 40), ("10.1.2.0/24", 30), ("10.1.0.0/16", 20),
              ("10.0.0.0/8", 10), ("192.168.0.0/16", 50), ("0.0.0.0/0", 1)]
    prefixes = [(parse_prefix(text)[0], parse_prefix(text)[1], hop) for text, hop in routes]

    tcam = TCAM()
    tcam.load([Rule.prefix(text, hop) for text, hop in routes])
    check(tcam.lookup(ip_to_int("10.1.2.3")) == 40, "the /32 wins inside the /24")
    check(tcam.lookup(ip_to_int("10.1.2.9")) == 30, "the /24 wins inside the /16")
    check(tcam.lookup(ip_to_int("10.1.9.9")) == 20, "the /16 wins inside the /8")
    check(tcam.lookup(ip_to_int("10.9.9.9")) == 10, "the /8 wins over the default route")
    check(tcam.lookup(ip_to_int("172.16.0.1")) == 1, "the catch-all /0 matches last")

    lines = tcam.match_lines(ip_to_int("10.1.2.9"))
    check(lines == sequential_match_vector(tcam.entries, ip_to_int("10.1.2.9")),
          "bit-parallel columns disagree with the row-by-row oracle")
    check(bin(lines).count("1") == 4, "10.1.2.9 matches the /0, /8, /16 and /24 rows")
    check(lines == (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5),
          "the /32 host route of 10.1.2.3 must not match 10.1.2.9")

    unsorted = TCAM()
    unsorted.load([Rule.prefix("10.0.0.0/8", 10), Rule.prefix("10.1.0.0/16", 20)])
    check(unsorted.lookup(ip_to_int("10.1.2.3")) == 10, "priority order can shadow a longer prefix")
    sorted_tcam = TCAM()
    sorted_tcam.load([Rule.prefix("10.1.0.0/16", 20), Rule.prefix("10.0.0.0/8", 10)])
    check(sorted_tcam.lookup(ip_to_int("10.1.2.3")) == 20, "sorting by length restores LPM")
    check(sorted_tcam.program_ops == 1, "one vector rebuild per load, not one per row")

    tcam2 = TCAM()
    tcam2.write(Rule.prefix("10.0.0.0/8", 10))
    check(tcam2.program_ops == 1, "the first write rebuilds the vectors once")
    tcam2.write(Rule.prefix("10.1.0.0/16", 20))
    check(len(tcam2.entries) == 2 and tcam2.program_ops == 2, "each write costs one rebuild")
    check(tcam2.lookup(ip_to_int("10.1.2.3")) == 10, "row 0 keeps priority after an append")

    wildcard = TCAM()
    wildcard.load([Rule(0, 0, 99, plen=0, label="catch-all")])
    for addr in (ip_to_int("0.0.0.0"), ip_to_int("10.1.2.3"), ip_to_int("255.255.255.255")):
        check(wildcard.lookup(addr) == 99, "an all-zero mask matches every address")
    check(wildcard.match_lines(ip_to_int("8.8.8.8")) == 1, "exactly one row matches the catch-all")
    check(TCAM().lookup(ip_to_int("8.8.8.8")) is None, "an empty array matches nothing")
    check(TCAM().lookup(ip_to_int("8.8.8.8"), default=77) == 77, "the default route is returned")

    odd = TCAM()
    odd.load([Rule(ip_to_int("0.1.0.1") & 0x00FF00FF, 0x00FF00FF, 5, label="sparse mask")])
    check(odd.lookup(ip_to_int("9.1.7.1")) == 5, "a non-contiguous mask matches positionally")
    check(odd.lookup(ip_to_int("9.2.7.1")) is None, "a non-contiguous mask still discriminates")

    ports = TCAM(width=16)
    ports.load([Rule(0x0100, 0xFF00, "ssh", plen=8, label="port 1"), Rule(0x0080, 0xFFF0, "dns",
                                                                         plen=12, label="port 2")])
    check(ports.lookup(0x0180) == "ssh", "a 16-bit-wide array behaves the same way")
    check(ports.lookup(0x0080) == "dns", "the second rule wins where the first does not match")

    prng = make_prng(0xBEEF)
    for index, max_plen in enumerate((8, 16, 24, 32)):
        table = prefix_table(prng, 180, max_plen)
        array = TCAM()
        array.load([Rule(addr, mask_for(plen), hop, plen=plen) for addr, plen, hop in table])
        check(all(array.entries[k].plen >= array.entries[k + 1].plen
                  for k in range(len(array.entries) - 1)), "rows are ordered longest prefix first")
        for _ in range(150):
            addr = prng() & FULL_MASK
            check(array.match_lines(addr) == sequential_match_vector(array.entries, addr),
                  f"vector mismatch at {int_to_ip(addr)} in table {index}")
            check(array.lookup(addr) == brute_force_lpm(table, addr),
                  f"TCAM/LPM mismatch at {int_to_ip(addr)} in table {index}")

    print(f"hand table: {len(routes)} rules, {bin(tcam.match_lines(ip_to_int('10.1.2.9'))).count('1')}"
          " matching rows for 10.1.2.9")
    print(f"simulated {4 * 180} prefix rules in four arrays of width {BITS} and one of width 16")
    print(f"column vectors rebuilt {tcam.program_ops + ports.program_ops} times for 2 tables")
    print("PASS")


if __name__ == "__main__":
    main()
