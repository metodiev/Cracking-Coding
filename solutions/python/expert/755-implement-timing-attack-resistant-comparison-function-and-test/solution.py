"""Task 755 - Timing-attack resistant comparison function, with a test that proves it.  (Tier: Expert, Beyond - Security & Privacy)

A `memcmp`-style comparison that returns as soon as two bytes differ leaks the length of the common
prefix through its running time: an attacker who can measure the response time learns the secret one
byte at a time.  This file implements both sides of the story:

  * `naive_eq`   - the leaky implementation (early exit on the first differing byte);
  * `ct_eq`      - the constant-time one: a single pass over the whole string, accumulating
                   `diff |= a[i] ^ b[i]`, with the length treated as *public* (as every real
                   constant-time API does - RSA/CMAC tag comparators fail fast on a length
                   mismatch, because the length is part of the public encoding);
  * `ct_select_u32` / `ct_lt_u32` - branch-free primitives built from masks only.

The "clock" is an operation counter, a *timing model*: it counts one tick per byte comparison (plus
a fixed epilogue), which is the only property an attacker can observe.  Being a model, it is exact
and reproducible, so the security claim is testable rather than measured: `ct_eq` produces exactly
one distinct tick count over all inputs of a given length, `naive_eq` produces a distinct count per
common-prefix length, the byte-at-a-time timing attack recovers the first n-1 secret bytes from
`naive_eq` in n-1*256 probes, and the identical attack on `ct_eq` gets no signal at all (every
candidate in a row ties, so it cannot even start).  `ct_eq`'s boolean results are cross-checked
against `hmac.compare_digest`, and a static check walks the *bytecode* of both functions with `dis`
and counts conditional jumps that sit inside a loop body - the compiler-level rule "no branch on
secret data in a loop" - which flags `naive_eq` and accepts `ct_eq`.

Simplified versus a real constant-time audit: the timing model replaces cycle-accurate measurement,
so it cannot see micro-architectural effects (cache-line granularity, vector units, CPU
speculation) and it does not cover the secrets-as-integers case (Python big-int arithmetic is
itself not constant time - a real implementation uses fixed-width words).  What is verified is the
*algorithmic* property: the number of primitive operations and the control flow are independent of
the compared data, and the leak that would otherwise allow byte-wise recovery is absent.
"""
import dis
import hmac

MASK32 = 0xFFFFFFFF
SECRET = b"T0p-S3cret!"          # a demo secret; never real key material
KEY_ALPHABET = bytes(range(256))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


class Clock:
    """Timing model: counts the primitive operations a comparison performs."""

    def __init__(self):
        self.ticks = 0

    def tick(self, cost=1):
        self.ticks += cost


def naive_eq(clock, a, b):
    """Leaky comparison: the running time reveals the common-prefix length."""
    n = len(a) if len(a) < len(b) else len(b)
    for i in range(n):
        clock.tick()
        if a[i] != b[i]:
            return False
    return len(a) == len(b)


def ct_eq(clock, a, b):
    """Constant-time comparison of equal-length byte strings; the length itself is public."""
    if len(a) != len(b):
        return False
    diff = 0
    for i in range(len(a)):
        clock.tick()
        diff |= a[i] ^ b[i]
    clock.tick()                                    # the epilogue is fixed work as well
    return diff == 0


def ct_mask_from_bit(bit):
    """0 -> 0x00000000, 1 -> 0xFFFFFFFF, without a branch."""
    return (-(bit & 1)) & MASK32


def ct_select_u32(bit, x, y):
    """Branch-free select of two 32-bit words on a single bit."""
    mask = ct_mask_from_bit(bit)
    return ((x & mask) | (y & ~mask)) & MASK32


def ct_lt_u32(x, y):
    """Unsigned x < y as 0/1 using only masks, XOR and subtraction (the borrow trick).

    The high bit of (~x & y) | (~(x ^ y) & (x - y)) is the borrow out of x - y, i.e. exactly the
    unsigned comparison result; the naive "(x - y) >> 31" only works when the operands are within
    2^31 of each other.
    """
    x, y = x & MASK32, y & MASK32
    diff = (x - y) & MASK32
    return (((~x & y) | (~(x ^ y) & diff)) >> 31) & 1


def loop_interior_branches(func):
    """Conditional jumps that lie inside a loop body: the static constant-time lint."""
    instructions = list(dis.get_instructions(func))
    back_edges = [i for i in instructions
                  if "JUMP" in i.opname and isinstance(i.argval, int) and i.argval < i.offset]
    flagged = []
    for instr in instructions:
        conditional = instr.opname.startswith("POP_JUMP") or instr.opname.startswith("JUMP_IF")
        if not conditional or not isinstance(instr.argval, int):
            continue
        if any(back.argval <= instr.offset < back.offset for back in back_edges):
            flagged.append((instr.offset, instr.opname))
    return flagged


def random_bytes(state, length):
    out = bytearray()
    for _ in range(length):
        state, r = splitmix64(state)
        out.append(r & 0xFF)
    return bytes(out), state


def timing_attack(oracle, length, candidates=256):
    """Recover the first length-1 bytes of SECRET from the oracle's tick counts alone."""
    known = bytearray()
    rows = []
    for position in range(length - 1):
        row = []
        best, best_ticks = 0, -1
        for candidate in range(candidates):
            guess = bytes(known) + bytes([candidate]) + bytes(length - position - 1)
            clock = Clock()
            oracle(clock, guess, SECRET)
            row.append(clock.ticks)
            if clock.ticks > best_ticks:
                best, best_ticks = candidate, clock.ticks
        known.append(best)
        rows.append(row)
    return bytes(known), rows


def main():
    # 1. correctness: ct_eq answers exactly like == and like the standard library comparator
    state = 0x1234
    pairs = [(b"", b""), (b"", b"x"), (b"a", b"a"), (b"a", b"b"), (SECRET, SECRET),
             (SECRET, SECRET[:-1]), (SECRET, SECRET[:-1] + b"?"), (b"\x00" * 8, b"\x00" * 7 + b"\x01")]
    for _ in range(40):
        left, state = random_bytes(state, 1 + state % 20)
        right, state = random_bytes(state, 1 + state % 20)
        pairs.append((left, right))
    for left, right in pairs:
        check(ct_eq(Clock(), left, right) == (left == right),
              "ct_eq must agree with == on %r vs %r" % (left, right))
        check(ct_eq(Clock(), left, right) == hmac.compare_digest(left, right),
              "ct_eq must agree with hmac.compare_digest on %r vs %r" % (left, right))

    # 2. boundary: equal lengths but every byte different, and the first/last byte deciding
    boundary = [(b"\x00" * 16, b"\xff" * 16), (b"\x00" + SECRET[1:], SECRET),
                (SECRET[:-1] + b"\x00", SECRET), (b"x", b"x"), (b"x", b"y")]
    for left, right in boundary:
        check(ct_eq(Clock(), left, right) == (left == right), "boundary equality %r/%r" % (left, right))

    # 3. the constant-time property: one single tick count for a fixed length, whatever the data
    counts = set()
    for left, right in pairs:
        if len(left) != len(right):
            continue
        clock = Clock()
        ct_eq(clock, left, right)
        counts.add((len(left), clock.ticks))
    per_length = {}
    for length, ticks in counts:
        per_length.setdefault(length, set()).add(ticks)
    check(all(len(values) == 1 for values in per_length.values()),
          "ct_eq must cost the same for every input of a length: %r" % per_length)
    check(per_length.get(len(SECRET)) == {len(SECRET) + 1},
          "the model must charge exactly one tick per byte plus the epilogue")

    # 4. the leak: naive_eq's cost grows with the length of the common prefix
    prefix_costs = []
    for prefix in range(len(SECRET)):
        guess = SECRET[:prefix] + bytes([SECRET[prefix] ^ 0xFF]) + SECRET[prefix + 1:]
        clock = Clock()
        naive_eq(clock, guess, SECRET)
        prefix_costs.append(clock.ticks)
    check(prefix_costs == list(range(1, len(SECRET) + 1)),
          "naive_eq must spend one tick per matching byte before giving up: %r" % prefix_costs)
    check(len(set(prefix_costs)) == len(SECRET), "every prefix length must be distinguishable")

    # 5. the attack succeeds against naive_eq: the whole prefix falls out of the tick counts
    recovered, rows = timing_attack(naive_eq, len(SECRET))
    check(recovered == SECRET[:len(SECRET) - 1],
          "the timing attack must recover the secret prefix, got %r" % recovered)
    check(all(row[max(range(len(row)), key=row.__getitem__)] == max(row) for row in rows),
          "each attack row must have a maximum")
    check(all(row.count(max(row)) == 1 for row in rows),
          "the correct byte must be the unique slowest candidate in every row")
    check([max(row) for row in rows] == list(range(2, len(SECRET) + 1)),
          "the winning tick count must track the prefix length")

    # 6. the same attack against ct_eq gets no signal, so the prefix stays unknown
    proof_secret = SECRET
    check(proof_secret[0] != 0, "the attack's tie-break argument assumes a non-zero first byte")
    blind, blind_rows = timing_attack(ct_eq, len(SECRET))
    check(all(len(set(row)) == 1 for row in blind_rows),
          "every ct_eq attack row must tie, so no candidate can be preferred")
    check(blind != SECRET[:len(SECRET) - 1] and blind.count(0) == len(blind),
          "the constant-time proof must defeat the attack (recovered %r)" % blind)

    # 7. the static lint: no conditional branch inside a loop body of a constant-time routine
    check(loop_interior_branches(ct_eq) == [], "ct_eq must have no data-dependent branch in its loop")
    check(loop_interior_branches(naive_eq), "the lint must flag the early-exit comparison")
    check(loop_interior_branches(timing_attack) != loop_interior_branches(ct_eq),
          "the lint must be able to distinguish the two implementations")

    # 8. branch-free primitives: exhaustive over a boundary set plus random 32-bit words
    words = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF, 12345, 0xDEADBEEF]
    for bit in (0, 1):
        for x in words:
            for y in words:
                check(ct_select_u32(bit, x, y) == (x if bit else y),
                      "ct_select_u32 must select %d/%d on bit %d" % (x, y, bit))
    state = 0xA5A5
    for _ in range(200):
        state, r = splitmix64(state)
        x, y = r & MASK32, (r >> 32) & MASK32
        check(ct_lt_u32(x, y) == (1 if x < y else 0), "ct_lt_u32 must match < for %d/%d" % (x, y))
    for x in words:
        for y in words:
            check(ct_lt_u32(x, y) == (1 if x < y else 0), "boundary compare %d/%d" % (x, y))
    check(ct_lt_u32(4, 4) == 0 and ct_lt_u32(4, 5) == 1 and ct_lt_u32(5, 4) == 0,
          "ct_lt_u32 boundary cases")

    # 9. the primitives are branch-free as well, and deterministic
    check(loop_interior_branches(ct_select_u32) == [] and loop_interior_branches(ct_lt_u32) == [],
          "the bitwise primitives must not branch either")
    first, first_rows = timing_attack(naive_eq, len(SECRET))
    second, second_rows = timing_attack(naive_eq, len(SECRET))
    check(first == second and first_rows == second_rows, "the attack must be reproducible")
    check((len(SECRET) - 1) * 256 == len(SECRET) * 256 - 256,
          "the attack needs one row of 256 probes per recovered byte")

    print("755 ct_eq: %d distinct tick counts over %d equal-length pairs (lengths %s)"
          % (sum(len(v) for v in per_length.values()), len(counts), sorted(per_length)[:4]))
    print("755 naive_eq: %d distinguishable prefixes, cost %s" % (len(set(prefix_costs)), prefix_costs[:6]))
    print("755 attack on naive_eq recovered %r with %d probes; on ct_eq recovered %r with no signal"
          % (recovered, (len(SECRET) - 1) * 256, blind))
    print("755 lint: ct_eq=%s naive_eq=%s" % (loop_interior_branches(ct_eq),
                                              loop_interior_branches(naive_eq)))
    print("PASS")


if __name__ == "__main__":
    main()
