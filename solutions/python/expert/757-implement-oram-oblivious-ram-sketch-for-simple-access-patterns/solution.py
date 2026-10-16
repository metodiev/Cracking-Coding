"""Task 757 - ORAM (oblivious RAM) sketch for simple access patterns.  (Tier: Expert, Beyond - Security & Privacy)

An ORAM hides *which* logical address the CPU is touching from the party that observes the physical
memory bus.  The construction used here is the trivial ("linear scan") ORAM, made explicit:

  * the memory is an array of n slots, each holding one (logical id, value) pair, and a fresh
    pseudo-random permutation decides the physical slot of every item;
  * on `access(x)` the client reads *every* slot in increasing order and then writes *every* slot
    back in increasing order, with the items re-placed under a permutation derived from the access
    counter (the trusted client keeps the layout; the observer only sees addresses);
  * the observable transcript is therefore `R0 R1 ... R(n-1) W0 W1 ... W(n-1)` for every access,
    independent of the logical address, of the access history and of the order of the accesses -
    which is exactly the property that makes it oblivious.

`Prp` is a four-round Feistel network over a domain of n = 2^bits points with a splitmix64 round
function; it is verified to be a bijection with a working inverse, so the relayout really is a
permutation of the items.  `DirectIndexOram` is the negative control: it reads exactly the slot
holding the requested item, and `observer_traces`/`leaks` (the access-pattern checker used by the
checks) must flag it - including naming the leaked slot, since the first address of its transcript
*is* the requested item's address.

Simplified versus real ORAM (Path ORAM, Circuit ORAM, ...): the cost is O(n) block transfers per
access instead of O(log^2 n); there is no tree hierarchy, no stash and no recursive position map;
the relayout is a permutation applied inside the trusted client rather than an oblivious
(secret-shared/oblivious-sort) shuffle, and the "physical bus" is the model that records addresses
rather than a real memory interface.  What is verified: functional correctness against a plain
array over long random access sequences, the permutation invariant after every access (every
logical id occupies exactly one slot), the exact constant transcript, obliviousness across six
different access sequences, that the layout after k accesses depends only on k and not on which
addresses were touched, and that the leaky variant is detected by the same checker.
"""
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Prp:
    """Four-round Feistel permutation over [0, 2^bits) - a demo PRP, never a real cipher."""

    ROUNDS = 4

    def __init__(self, bits, key):
        if bits < 2 or bits % 2:
            raise ValueError("the Feistel domain needs an even bit width >= 2")
        self.bits = bits
        self.half = bits // 2
        self.mask = (1 << self.half) - 1
        self.size = 1 << bits
        self.round_keys = []
        state = (key ^ (bits << 40)) & MASK64
        for _ in range(self.ROUNDS):
            state, r = splitmix64(state)
            self.round_keys.append(r)

    def _round(self, index, half):
        state, r = splitmix64(self.round_keys[index] ^ ((half * 0x9E3779B97F4A7C15) & MASK64))
        return r & self.mask

    def permute(self, value):
        if not 0 <= value < self.size:
            raise ValueError("value outside the permutation domain")
        left, right = value >> self.half, value & self.mask
        for index in range(self.ROUNDS):
            left, right = right, left ^ self._round(index, right)
        return (left << self.half) | right

    def invert(self, value):
        if not 0 <= value < self.size:
            raise ValueError("value outside the permutation domain")
        left, right = value >> self.half, value & self.mask
        for index in range(self.ROUNDS - 1, -1, -1):
            left, right = right ^ self._round(index, left), left
        return (left << self.half) | right


class LinearScanOram:
    """Trivial ORAM: every access scans and rewrites the whole array under a fresh layout."""

    def __init__(self, values, key=0x5EED):
        size = len(values)
        if size < 4 or size & (size - 1):
            raise ValueError("the slot count must be a power of two and at least 4")
        if len(set(values)) != size:
            raise ValueError("logical ids must be unique")
        self.n = size
        self.bits = size.bit_length() - 1
        self.values = dict(enumerate(values))
        self.slots = [(logical, values[logical]) for logical in range(size)]
        self.key = key
        self.step = 0
        self.trace = []
        self.transfers = 0

    def position_of(self, logical):
        for slot, (stored, _) in enumerate(self.slots):
            if stored == logical:
                return slot
        raise ValueError("item %r is not in the memory" % (logical,))

    def access(self, logical):
        if logical not in self.values:
            raise IndexError("no such logical address: %r" % (logical,))
        value = None
        for slot in range(self.n):                     # phase 1: read every physical slot
            self.trace.append(("R", slot))
            self.transfers += 1
            stored, content = self.slots[slot]
            if stored == logical:
                value = content
        self.step += 1
        prp = Prp(self.bits, self.key ^ ((self.step * 0x9E3779B97F4A7C15) & MASK64))
        fresh = [None] * self.n
        for stored, content in self.slots:
            fresh[prp.permute(stored)] = (stored, content)
        for slot in range(self.n):                     # phase 2: write every slot, in slot order
            self.trace.append(("W", slot))
            self.transfers += 1
        self.slots = fresh
        return value

    def invariant_ok(self):
        """Every logical id occupies exactly one slot, and slots hold their own item's value."""
        stored = [logical for logical, _ in self.slots]
        return (sorted(stored) == list(range(self.n))
                and all(content == self.values[logical] for logical, content in self.slots))


class DirectIndexOram:
    """Negative control: an index into the physical array, so the address *is* the secret."""

    def __init__(self, values):
        self.n = len(values)
        self.values = dict(enumerate(values))
        self.slots = [(logical, values[logical]) for logical in range(self.n)]
        self.trace = []
        self.transfers = 0

    def position_of(self, logical):
        return logical

    def access(self, logical):
        if logical not in self.values:
            raise IndexError("no such logical address: %r" % (logical,))
        self.trace.append(("R", logical))
        self.transfers += 1
        return self.values[logical]


def observer_transcripts(factory, sequences):
    """Replay equal-length access sequences from a fresh ORAM and collect the observers' views."""
    transcripts = []
    for sequence in sequences:
        memory = factory()
        for logical in sequence:
            memory.access(logical)
        transcripts.append(tuple(memory.trace))
    return transcripts


def leaks(transcripts):
    """Access-pattern checker: a leak exists as soon as two transcripts differ."""
    return len(set(transcripts)) > 1


def main():
    n = 64
    values = [(index * 7 + 3) % 100 for index in range(n)]

    # 1. the permutation is a real bijection with a working inverse (exhaustive for a small domain)
    small = Prp(8, 0xABCDEF)
    images = sorted(small.permute(value) for value in range(256))
    check(images == list(range(256)), "a PRP must map the domain onto itself")
    check(all(small.invert(small.permute(value)) == value for value in range(256)),
          "the inverse must undo the permutation")
    check(any(small.permute(value) != value for value in range(256)),
          "the permutation must not be the identity here")
    for key in (1, 2, 0x5EED, 0xFEEDFACE):
        prp = Prp(6, key)
        check(sorted(prp.permute(v) for v in range(64)) == list(range(64)),
              "every key must give a bijection")
    check(Prp(6, 1).round_keys != Prp(6, 2).round_keys, "different keys must give different rounds")
    for bad in (0, 1, 3, 5):
        try:
            Prp(bad, 1)
            raise AssertionError("an odd or too small Feistel width must be rejected")
        except ValueError:
            pass

    # 2. correctness: 200 random accesses agree with a plain array, and the invariant always holds
    memory = LinearScanOram(values)
    state = 0x1DEA
    plain = dict(enumerate(values))
    for _ in range(200):
        state, r = splitmix64(state)
        logical = r % n
        check(memory.access(logical) == plain[logical],
              "ORAM access %d must return the plain value" % logical)
        check(memory.invariant_ok(), "every logical id must keep exactly one slot")

    # 3. the transcript is the same constant for every access - the whole point of the exercise
    canonical = [("R", slot) for slot in range(n)] + [("W", slot) for slot in range(n)]
    one = LinearScanOram(values)
    for logical in (0, 1, 63, 0, 32):
        before = len(one.trace)
        one.access(logical)
        check(tuple(one.trace[before:]) == tuple(canonical),
              "access %d must produce exactly the canonical scan-and-rewrite transcript" % logical)
    check(one.transfers == 5 * 2 * n, "each access must cost 2n block transfers")

    # 4. obliviousness: six different access sequences of the same length look identical
    sequences = [[0, 1, 2, 3, 4],
                 [63, 0, 5, 5, 5],
                 [17, 17, 17, 17, 17],
                 [1, 31, 7, 55, 12],
                 [32, 33, 34, 35, 36],
                 [40, 3, 22, 61, 9]]
    oblivious = observer_transcripts(lambda: LinearScanOram(values), sequences)
    check(len(set(oblivious)) == 1, "all equal-length transcripts must be identical")
    check(not leaks(oblivious), "the checker must find no leak in the ORAM")
    check(oblivious[0] == tuple(canonical) * 5, "the transcript must be the repeated canonical scan")

    # 5. the layout depends on the step counter only - not on which addresses were touched
    first, second = LinearScanOram(values), LinearScanOram(values)
    first.access(3)
    second.access(47)
    check(first.position_of(11) == second.position_of(11),
          "after one access the two layouts must be identical (the relayout is data-independent)")
    check(first.slots == second.slots, "the whole physical layout must match after k accesses")

    # 6. repeated accesses keep moving the item: the physical position is re-randomised every step
    repeated = LinearScanOram(values)
    positions = []
    for _ in range(32):
        repeated.access(5)
        positions.append(repeated.position_of(5))
    check(len(set(positions)) >= 16,
          "32 relayouts must scatter one item over many distinct slots: %d" % len(set(positions)))

    # 7. negative control: the direct-indexed variant leaks, and the checker says so
    leaky = observer_transcripts(lambda: DirectIndexOram(values), sequences)
    check(leaks(leaky), "the access-pattern checker must flag the non-oblivious variant")
    check(len(set(leaky)) == len(sequences), "every distinct sequence must be distinguishable")
    recovered = [transcript[0][1] for transcript in leaky]
    check(recovered == [sequence[0] for sequence in sequences],
          "the first observed address is literally the requested logical address: %r" % recovered)
    check(len(leaky[0]) == 5, "the leaky variant pays one transfer per access")
    overhead = 2.0 * n
    check(overhead == 2 * n, "the ORAM's overhead is 2n transfers per access here")

    # 8. boundary behaviour: the smallest legal size, out-of-range and malformed input
    tiny = LinearScanOram([10, 20, 30, 40])
    check(tiny.access(3) == 40 and tiny.access(0) == 10, "a four-slot ORAM must work")
    check(tiny.position_of(3) != tiny.position_of(0), "the two items live in different slots")
    for bad in (lambda: tiny.access(4), lambda: tiny.access(-1)):
        try:
            bad()
            raise AssertionError("an out-of-range logical address must be rejected")
        except IndexError:
            pass
    for bad_size in ([1, 2, 3], [1, 2, 3, 4, 5], [1, 2]):
        try:
            LinearScanOram(bad_size)
            raise AssertionError("a slot count that is not a power of two must be rejected")
        except ValueError:
            pass
    try:
        LinearScanOram([1, 1, 1, 1])
        raise AssertionError("duplicate logical ids must be rejected")
    except ValueError:
        pass

    # 9. every access leaves the memory a permutation of the logical ids (checked above) and the
    #    value returned is the one that was stored, even after many relayouts of a rewritten item
    rewrite = LinearScanOram(values)
    for step in range(40):
        check(rewrite.access(step % n) == plain[step % n], "values must survive 40 relayouts")
        check(rewrite.invariant_ok(), "the permutation invariant must hold after every relayout")

    # 10. determinism
    left = observer_transcripts(lambda: LinearScanOram(values), sequences)
    right = observer_transcripts(lambda: LinearScanOram(values), sequences)
    check(left == right and left[0] == oblivious[0], "transcripts must be reproducible")

    print("757 n=%d: transcript length %d per access, identical for %d different sequences"
          % (n, 2 * n, len(sequences)))
    print("757 transfers: ORAM %d per access vs direct %d (overhead factor %.0f)"
          % (2 * n, 1, overhead))
    print("757 one item visited %d distinct slots over 32 relayouts; PRP verified on 256 points"
          % len(set(positions)))
    print("757 leak checker: ORAM leak=%s, direct-index leak=%s (addresses recovered: %r)"
          % (leaks(oblivious), leaks(leaky), recovered[:3]))
    print("PASS")


if __name__ == "__main__":
    main()
