"""Task 754 - Memory-sanitizer style shadow memory detector (toy).  (Tier: Expert, Beyond - Security & Privacy)

MemorySanitizer's idea: every *bit* of the application's memory has a shadow bit that says whether
that bit is initialized.  A store of a value propagates the value's shadow (bit for bit, so a
partially initialized integer stays partially initialized), arithmetic combines the operand shadows
with OR (an over-approximation: the result bit is uninitialized if either operand bit was), and the
sanitizer only *reports* when an uninitialized value reaches a branch or a syscall - not on every
load.  Additionally every poisoned byte carries an *origin* (the allocation that produced it), so
the report can say "this byte came from allocation #1".

This file implements that engine twice, with two different shadow representations:

    BitShadowMemory    a dense byte per byte of data, one shadow bit per data bit, plus an
                       origin byte per byte - the shape a real sanitizer uses
    SparseShadowMemory a dict from address to the *set of poisoned bit positions*, plus a dict of
                       origins - a completely different representation of the same state

A small driver interprets a list of operations (alloc / store_const / store_bits / memcpy / add /
branch / poison / unpoison / syscall_print) against an engine, so the same program can be replayed
through both engines.  Replaying is the oracle: the reports must be identical (kind, address,
origin) and the final shadow state must be identical when the sparse state is expanded into the
dense form.  Any divergence between the two representations is a bug in the propagation rules.

Simplified versus MSan: memory is a flat byte array with no allocator metadata, alignment or
redzones; the OR-combination is the toy rule for all arithmetic (real MSan propagates through a
compiler-generated IR and handles selects/shifts specially); origins are allocation ids instead of
a graph; only a handful of "report sites" (branch, syscall argument) are modelled, and there is no
call-stack origin or per-function suppression.  Verified: the exact address and origin of every
report, bit-exact shadow propagation through memcpy and add, the delay between propagation and
reporting, that writing a constant clears the shadow (no false positive), and full agreement with
the sparse oracle on every operation of the program.
"""
MASK64 = (1 << 64) - 1   # kept for the splitmix64-free variant: no randomness is needed here


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class BitShadowMemory:
    """Dense engine: data byte + shadow byte (1 bit = uninitialized) + origin id per byte."""

    def __init__(self):
        self.data = bytearray()
        self.shadow = bytearray()          # 0xFF in a byte means "all 8 bits uninitialized"
        self.origin = []                   # allocation id responsible for the byte's poison
        self.reports = []
        self.bases = {}

    # ------------------------------------------------------------------ raw helpers
    def _range(self, addr, n):
        if addr < 0 or n < 0 or addr + n > len(self.data):
            raise IndexError("address range [%d,%d) outside the arena" % (addr, addr + n))
        return range(addr, addr + n)

    def alloc(self, name, size, tag):
        base = len(self.data)
        self.data.extend(bytearray(size))
        self.shadow.extend(bytearray([0xFF]) * size)       # fresh memory: uninitialized
        self.origin.extend([tag] * size)
        self.bases[name] = base
        return base

    def report(self, kind, addr):
        self.reports.append((kind, addr, self.origin[addr]))

    # ------------------------------------------------------------------- operations
    def store_const(self, addr, payload):
        for offset, value in enumerate(payload):
            self.data[addr + offset] = value
            self.shadow[addr + offset] = 0                 # a constant is fully initialized
            self.origin[addr + offset] = 0

    def store_bits(self, addr, value, init_mask, tag):
        """Write one byte but only the bits in init_mask; the others stay uninitialized."""
        self.data[addr] = value & 0xFF
        self.shadow[addr] = (~init_mask) & 0xFF
        self.origin[addr] = tag if self.shadow[addr] else 0

    def memcpy(self, dst, src, n):
        if dst < src:                                      # copy forward unless they overlap backwards
            order = range(n)
        else:
            order = range(n - 1, -1, -1)
        for offset in order:
            self.data[dst + offset] = self.data[src + offset]
            self.shadow[dst + offset] = self.shadow[src + offset]   # bit-exact, per byte
            self.origin[dst + offset] = self.origin[src + offset]

    def add(self, dst, a, b, n):
        left = int.from_bytes(bytes(self.data[a:a + n]), "little")
        right = int.from_bytes(bytes(self.data[b:b + n]), "little")
        total = (left + right) & ((1 << (8 * n)) - 1)
        for offset in range(n):
            self.data[dst + offset] = (total >> (8 * offset)) & 0xFF
            shadow = self.shadow[a + offset] | self.shadow[b + offset]
            self.shadow[dst + offset] = shadow
            first = self.origin[a + offset] if self.shadow[a + offset] else 0
            second = self.origin[b + offset] if self.shadow[b + offset] else 0
            self.origin[dst + offset] = first if first else second

    def branch(self, addr, n, kind="branch"):
        for index in self._range(addr, n):
            if self.shadow[index]:
                self.report(kind, index)
                return False
        return True

    def poison(self, addr, n, tag):
        for index in self._range(addr, n):
            self.shadow[index] = 0xFF
            self.origin[index] = tag

    def unpoison(self, addr, n):
        for index in self._range(addr, n):
            self.shadow[index] = 0
            self.origin[index] = 0

    # ---------------------------------------------------------------------- queries
    def first_poisoned(self, addr, n):
        for index in self._range(addr, n):
            if self.shadow[index]:
                return index, self.shadow[index]
        return None

    def snapshot(self):
        return {addr: frozenset(bit for bit in range(8)
                                if (mask >> bit) & 1)
                for addr, mask in enumerate(self.shadow) if mask}


class SparseShadowMemory:
    """Oracle engine: identical semantics, shadow kept as sets of poisoned bit positions."""

    def __init__(self):
        self.data = {}
        self.poisoned = {}                 # addr -> set of poison bit indices
        self.origin = {}                   # addr -> allocation id
        self.reports = []
        self.bases = {}
        self.size = 0

    def _check(self, addr, n):
        if addr < 0 or n < 0 or addr + n > self.size:
            raise IndexError("address range [%d,%d) outside the arena" % (addr, addr + n))

    def _bits(self, addr):
        return self.poisoned.get(addr, frozenset())

    def alloc(self, name, size, tag):
        base = self.size
        for offset in range(size):
            self.poisoned[base + offset] = frozenset(range(8))
            self.origin[base + offset] = tag
        self.size += size
        self.bases[name] = base
        return base

    def report(self, kind, addr):
        self.reports.append((kind, addr, self.origin.get(addr, 0)))

    def store_const(self, addr, payload):
        for offset, value in enumerate(payload):
            self.data[addr + offset] = value
            self.poisoned[addr + offset] = frozenset()
            self.origin[addr + offset] = 0

    def store_bits(self, addr, value, init_mask, tag):
        self.data[addr] = value & 0xFF
        bits = frozenset(bit for bit in range(8) if not (init_mask >> bit) & 1)
        self.poisoned[addr] = bits
        self.origin[addr] = tag if bits else 0

    def memcpy(self, dst, src, n):
        order = range(n) if dst < src else range(n - 1, -1, -1)
        for offset in order:
            self.data[dst + offset] = self.data.get(src + offset, 0)
            self.poisoned[dst + offset] = self._bits(src + offset)
            self.origin[dst + offset] = self.origin.get(src + offset, 0)

    def add(self, dst, a, b, n):
        for offset in range(n):
            left = self.data.get(a + offset, 0)
            right = self.data.get(b + offset, 0)
            self.data[dst + offset] = (left + right) & 0xFF
            self.poisoned[dst + offset] = self._bits(a + offset) | self._bits(b + offset)
            first = self.origin.get(a + offset, 0) if self._bits(a + offset) else 0
            second = self.origin.get(b + offset, 0) if self._bits(b + offset) else 0
            self.origin[dst + offset] = first if first else second

    def branch(self, addr, n, kind="branch"):
        self._check(addr, n)
        for index in range(addr, addr + n):
            if self._bits(index):
                self.report(kind, index)
                return False
        return True

    def poison(self, addr, n, tag):
        self._check(addr, n)
        for index in range(addr, addr + n):
            self.poisoned[index] = frozenset(range(8))
            self.origin[index] = tag

    def unpoison(self, addr, n):
        self._check(addr, n)
        for index in range(addr, addr + n):
            self.poisoned[index] = frozenset()
            self.origin[index] = 0

    def first_poisoned(self, addr, n):
        self._check(addr, n)
        for index in range(addr, addr + n):
            bits = self._bits(index)
            if bits:
                mask = sum(1 << bit for bit in bits)
                return index, mask
        return None

    def snapshot(self):
        return {addr: bits for addr, bits in self.poisoned.items() if bits}


# ------------------------------------------------------------------------ the program

def drive(engine, ops):
    """Replay a fixed operation list against any engine with the documented interface."""
    for op in ops:
        kind = op[0]
        if kind == "alloc":
            engine.alloc(op[1], op[2], op[3])
        elif kind == "store_const":
            name, offset = op[1]
            engine.store_const(engine.bases[name] + offset, op[2])
        elif kind == "store_bits":
            name, offset = op[1]
            engine.store_bits(engine.bases[name] + offset, op[2], op[3], op[4])
        elif kind == "memcpy":
            dst, src, n = op[1], op[2], op[3]
            engine.memcpy(engine.bases[dst[0]] + dst[1], engine.bases[src[0]] + src[1], n)
        elif kind == "add":
            dst, left, right, n = op[1], op[2], op[3], op[4]
            engine.add(engine.bases[dst[0]] + dst[1], engine.bases[left[0]] + left[1],
                       engine.bases[right[0]] + right[1], n)
        elif kind == "branch":
            name, offset = op[1]
            engine.branch(engine.bases[name] + offset, op[2], op[3])
        elif kind == "poison":
            name, offset = op[1]
            engine.poison(engine.bases[name] + offset, op[2], op[3])
        elif kind == "unpoison":
            name, offset = op[1]
            engine.unpoison(engine.bases[name] + offset, op[2])
        else:
            raise ValueError("unknown operation %r" % (kind,))
    return engine


PROGRAM = [
    ("alloc", "src", 8, 1),
    ("alloc", "dst", 8, 2),
    ("store_const", ("src", 0), b"\x01\x02\x03\x04"),
    ("store_bits", ("src", 4), 0xAB, 0x0F, 1),      # only the low nibble of byte 4 is written
    ("memcpy", ("dst", 0), ("src", 0), 6),
    ("branch", ("dst", 0), 6, "copied poison"),
    ("add", ("dst", 6), ("dst", 0), ("src", 0), 2),
    ("branch", ("dst", 6), 2, "arithmetic result"),
    ("unpoison", ("dst", 4), 1),
    ("branch", ("dst", 0), 6, "still virgin after unpoison"),
    ("memcpy", ("dst", 0), ("src", 4), 4),         # re-poison from the nibble store
    ("branch", ("dst", 1), 4, "second-hand poison"),
    ("store_const", ("dst", 1), b"\x00\x00\x00\x00"),
    ("branch", ("dst", 0), 6, "after overwrite"),
    ("poison", ("dst", 7), 1, 2),
    ("branch", ("dst", 7), 1, "explicit poison"),
]


def main():
    # 1. a clean program produces no report at all (no false positives)
    clean = BitShadowMemory()
    drive(clean, [("alloc", "b", 4, 1), ("store_const", ("b", 0), b"\xde\xad\xbe\xef"),
                  ("branch", ("b", 0), 4, "clean")])
    check(clean.reports == [], "a fully initialized value must not be reported")

    # 2. branch on freshly allocated memory: one report, at the byte and with the allocation origin
    fresh = drive(BitShadowMemory(), [("alloc", "b", 4, 7)])
    base = fresh.bases["b"]
    check(fresh.branch(base, 4) is False, "uninitialized memory must be reported")
    check(fresh.reports == [("branch", base, 7)],
          "the report must name the first poisoned byte and its origin: %r" % (fresh.reports,))
    check(fresh.first_poisoned(base, 4) == (base, 0xFF), "every byte of a fresh allocation is poison")

    # 3. partially initialized byte: only the missing bits are poison, and the report points at it
    partial = BitShadowMemory()
    partial.alloc("p", 8, 3)
    partial.store_bits(0, 0x0F, 0x0F, 3)               # low nibble written, high nibble not
    check(partial.shadow[0] == 0xF0 and partial.first_poisoned(0, 1) == (0, 0xF0),
          "a nibble store must leave exactly the unwritten bits poisoned")
    partial.unpoison(0, 1)
    check(partial.branch(0, 1) is True and partial.reports == [],
          "unpoison must make the byte observably clean")
    check(partial.branch(1, 1) is False, "the neighbouring byte is still uninitialized")

    # 4. memcpy is bit-exact, and the origin travels with the data through chained copies
    chain = BitShadowMemory()
    chain.alloc("a", 4, 11)
    chain.alloc("b", 4, 12)
    chain.alloc("c", 4, 13)
    chain.store_bits(0, 0xAA, 0x03, 11)                # byte 0: bits 0,1 initialized only
    chain.memcpy(4, 0, 4)
    chain.memcpy(8, 4, 4)
    check(chain.shadow[8] == 0xFC, "chained copies must preserve the exact shadow bits")
    check(chain.first_poisoned(8, 4) == (8, 0xFC), "the copied poison must be visible at the copy")
    check(chain.origin[8] == 11, "the origin must follow the data, not the copy (origin 11)")
    check(chain.branch(8, 4) is False and chain.reports[0][2] == 11,
          "a report after two copies must still name the original allocation")

    # 5. add combines operand shadows with OR - checked bit for bit against independent arithmetic
    arith = BitShadowMemory()
    arith.alloc("x", 4, 21)
    arith.alloc("y", 4, 22)
    arith.alloc("z", 4, 23)
    arith.store_bits(0, 0x11, 0x0F, 21)                # byte 0: high nibble poisoned -> 0xF0
    arith.store_bits(4, 0x22, 0x3C, 22)                # byte 4: bits 2..5 written -> 0xC3 poisoned
    arith.add(8, 0, 4, 4)
    expected = [0xF0 | 0xC3, 0xFF, 0xFF, 0xFF]         # OR is per bit, not per byte
    check([arith.shadow[8 + i] for i in range(4)] == expected,
          "add must OR the per-bit shadows: %r" % ([arith.shadow[8 + i] for i in range(4)],))
    check(arith.first_poisoned(8, 4) == (8, expected[0]), "the arithmetic result is poisoned")

    # 6. the sparse oracle replays the whole program identically - reports and final shadow state
    dense = drive(BitShadowMemory(), PROGRAM)
    sparse = drive(SparseShadowMemory(), PROGRAM)
    check(dense.reports == sparse.reports,
          "the two shadow representations disagree on reports:\n%r\n%r"
          % (dense.reports, sparse.reports))
    check(dense.snapshot() == sparse.snapshot(),
          "the two shadow representations disagree on the final shadow state")

    # 7. reporting is delayed: poison stays silent until it reaches a branch, and every report
    #    carries the allocation the data originally came from
    check(len(dense.reports) == 5, "the program must produce five reports: %r" % (dense.reports,))
    check([kind for kind, _, _ in dense.reports] ==
          ["copied poison", "still virgin after unpoison", "second-hand poison",
           "after overwrite", "explicit poison"],
          "the reports must be the branches that really observed poison")
    check([origin for _, _, origin in dense.reports] == [1, 1, 1, 1, 2],
          "origins must follow the data: %r" % ([o for _, _, o in dense.reports],))
    dst_base = dense.bases["dst"]
    check([addr - dst_base for _, addr, _ in dense.reports] == [4, 5, 1, 0, 7],
          "every report must point at the offending byte of the destination")
    check("arithmetic result" not in [kind for kind, _, _ in dense.reports],
          "the OR of two initialized bytes must not be reported")

    # 8. writing a constant over poisoned memory clears it - no false positive survives
    overwrite = BitShadowMemory()
    overwrite.alloc("w", 4, 31)
    overwrite.store_const(0, b"\x01\x02\x03\x04")
    check(overwrite.branch(0, 4) is True and overwrite.reports == [],
          "an overwritten buffer must be clean")
    overwrite.poison(1, 1, 31)
    check(overwrite.branch(0, 4) is False, "an explicit poison must be reported")
    overwrite.store_const(1, b"\x09")
    check(overwrite.branch(0, 4) is True and len(overwrite.reports) == 1,
          "the constant store must clear the poison again")

    # 9. boundary behaviour: empty allocation, zero-length copy, last byte, out-of-range access
    edge = BitShadowMemory()
    zero = edge.alloc("zero", 0, 41)
    last = edge.alloc("last", 3, 42)
    check(zero == 0 and edge.branch(last, 0) is True, "zero-length ranges are trivially clean")
    edge.store_const(last, b"\x0a\x0b\x0c")
    check(edge.branch(last + 2, 1) is True, "the last byte of an allocation is addressable")
    for bad in (lambda: edge.branch(last, 4), lambda: edge.store_const(last + 2, b"\x00\x00")):
        try:
            bad()
            raise AssertionError("an out-of-range access must raise IndexError")
        except IndexError:
            pass

    # 10. the sparse engine is a real second implementation: it also rejects out-of-range accesses
    sparse_edge = SparseShadowMemory()
    sparse_edge.alloc("s", 2, 51)
    try:
        sparse_edge.branch(1, 2, "clean")
        raise AssertionError("the oracle engine must validate ranges too")
    except IndexError:
        pass
    check(sparse_edge.branch(0, 2, "clean") is False, "the oracle engine reports poison as well")

    # 11. determinism: identical programs produce identical reports and states on both engines
    again_dense = drive(BitShadowMemory(), PROGRAM)
    again_sparse = drive(SparseShadowMemory(), PROGRAM)
    check(again_dense.reports == dense.reports and again_sparse.reports == sparse.reports,
          "the engines must be deterministic")
    check(again_dense.snapshot() == dense.snapshot(), "shadow state must be reproducible")

    print("754 reports=%d (first=%r)" % (len(dense.reports), dense.reports[0]))
    print("754 poisoned bytes at the end: dense=%d sparse=%d"
          % (len(dense.snapshot()), len(sparse.snapshot())))
    print("754 nibble store leaves shadow 0x%02x; chained copies keep 0x%02x with origin %d"
          % (partial.shadow[0], chain.shadow[8], chain.origin[8]))
    print("754 dense/sparse agreement on %d operations of the program" % len(PROGRAM))
    print("PASS")


if __name__ == "__main__":
    main()
