"""Task 756 - Side-channel resistant constant-time modular exponentiation.  (Tier: Expert, Beyond - Security & Privacy)

Naive square-and-multiply is the textbook timing leak of public-key crypto: it squares every bit of
the exponent but multiplies only when that bit is set, so both the total number of modular
multiplications and the branch taken depend on the secret exponent - a classic "square-and-multiply
vs. Montgomery ladder" analysis question.  This file implements and *instruments* three variants:

    naive_pow    textbook square-and-multiply, with an `if bit:` branch and a per-bit cost that
                 equals the Hamming weight of the exponent
    ladder_pow   the Montgomery ladder: per bit it computes R0^2, R0*R1 and R1^2 unconditionally
                 and then selects with a branch-free mask, so its invariant
                 R0 = base^prefix, R1 = base^(prefix+1) holds after every step
    window_pow   fixed-window (w bits) exponentiation whose precomputed table is read with a
                 full-table scan using a branch-free equality mask (`ct_eq_mask` / `ct_select`)

`Model` is the cost/leakage model under test: it counts modular multiplications, records a *trace*
of the operations (including which table index was read), records whether a branch was taken on a
secret bit, and records the selects.  Everything runs on toy moduli (a few tens of bits, never key
material) with `pow(base, exponent, modulus)` as the correctness oracle.

What is verified: results equal `pow` for random and boundary inputs (exponent 0, modulus 1, base 0,
all-ones exponent, powers of two, leading-zero padding); the cost and the trace of `ladder_pow` and
of the constant-time window variant are *identical for every exponent of the same padded bit
length*, while the naive variants' cost traces the Hamming weight; the number of secret-dependent
branches is 0 for the constant-time variants and equals the Hamming weight for `naive_pow`; the
table scan of the constant-time window version reads every index once per window, whereas the
"constant_time=False" variant leaks each window digit in its trace - and the checks *recover the
whole exponent* from that leak, which is what makes the claim testable rather than asserted.

Simplified versus real side-channel-hardened code: the model is an operation/access counter, not a
cycle-accurate simulator, so it cannot see cache-line granularity, micro-architectural side effects
or power consumption; exponent blinding, Montgomery/CIOS multiplication and modular inversion are
not modelled; the modular arithmetic itself uses Python big integers (which are not constant time),
so the *implementation* would need fixed-width words to be production-safe - the property verified
here is algorithmic (uniform operation count, uniform memory access pattern, no secret branches).
"""
MASK32 = 0xFFFFFFFF

# toy moduli: 31-bit and 17-bit, small enough to reason about, never key material
MODULI = (0x7FFFFFFF, 65537, 0xFFFF, 1000003)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


class Model:
    """Cost and leakage model: multiplication count, operation trace, secret branches."""

    def __init__(self):
        self.muls = 0
        self.selects = 0
        self.secret_branches = 0
        self.trace = []

    def mul(self, left, right):
        self.muls += 1
        self.trace.append("mul(%s,%s)" % (left, right))

    def select(self):
        self.selects += 1

    def reset_trace(self):
        self.trace = []

    def table_reads(self):
        return [event for event in self.trace if event.startswith("read(T[")]

    def leaked_table_indices(self):
        """Table indices exposed by a non-scanning lookup, in order (the leak an attacker reads)."""
        leaked = []
        for event in self.trace:
            if event.startswith("lookup(T["):
                leaked.append(int(event[event.index("[") + 1:event.index("]")]))
        return leaked


def ct_mul(model, left_value, left_name, right_value, right_name, modulus):
    """One modular multiplication, recorded under the given operand names."""
    model.mul(left_name, right_name)
    return (left_value * right_value) % modulus


def ct_eq_mask(a, b):
    """1 when a == b else 0, using bitwise operations only (no comparison branch)."""
    diff = (a ^ b) & MASK32
    return (((diff | -diff) >> 31) & 1) ^ 1


def ct_select(model, bit, left, right, modulus):
    """Branch-free select of two residues: bit == 1 picks `left`."""
    model.select()
    return (left * bit + right * (1 - bit)) % modulus


def ct_table_lookup(model, table, digit, modulus):
    """Read *every* table entry and combine with masks, so the access pattern is fixed."""
    result = table[0]
    for index, entry in enumerate(table):
        model.trace.append("read(T[%d])" % index)
        result = ct_select(model, ct_eq_mask(index, digit), entry, result, modulus)
    return result


def naive_pow(model, base, exponent, modulus):
    """Textbook square-and-multiply: the running time traces the Hamming weight."""
    result = 1 % modulus
    base %= modulus
    for bit_index in range(exponent.bit_length() - 1, -1, -1):
        result = ct_mul(model, result, "R", result, "R", modulus)
        if (exponent >> bit_index) & 1:                 # a branch on a secret bit
            model.secret_branches += 1
            result = ct_mul(model, result, "R", base, "B", modulus)
    return result


def ladder_pow(model, base, exponent, modulus, bit_length=None, on_step=None):
    """Montgomery ladder: three multiplications and two selects per bit, whatever the bits are.

    Invariant after processing the prefix P of the exponent: R0 = base^P and R1 = base^(P+1);
    `on_step(prefix, R0, R1)` is called after every bit when supplied.
    """
    if bit_length is None:
        bit_length = exponent.bit_length()
    if exponent.bit_length() > bit_length:
        raise ValueError("bit_length must cover the exponent; the length itself is public")
    base %= modulus
    r0, r1 = 1 % modulus, base
    prefix = 0
    for bit_index in range(bit_length - 1, -1, -1):
        bit = (exponent >> bit_index) & 1
        square = ct_mul(model, r0, "R0", r0, "R0", modulus)
        cross = ct_mul(model, r0, "R0", r1, "R1", modulus)
        doubled = ct_mul(model, r1, "R1", r1, "R1", modulus)
        r0 = ct_select(model, bit, cross, square, modulus)
        r1 = ct_select(model, bit, doubled, cross, modulus)
        prefix = (prefix << 1) | bit
        if on_step is not None:
            on_step(prefix, r0, r1)
    return r0


def window_pow(model, base, exponent, modulus, window=4, bit_length=None, constant_time=True):
    """Fixed-window exponentiation; the table lookup either scans everything or leaks the digit."""
    if bit_length is None:
        bit_length = exponent.bit_length()
    padded = -(-bit_length // window) * window             # pad to whole windows (public)
    if exponent.bit_length() > padded:
        raise ValueError("bit_length must cover the exponent")
    base %= modulus
    table = [1 % modulus]
    for _ in range(1, 1 << window):
        table.append((table[-1] * base) % modulus)
    result = 1 % modulus
    for start in range(padded - window, -1, -window):
        for _ in range(window):
            result = ct_mul(model, result, "R", result, "R", modulus)
        digit = (exponent >> start) & ((1 << window) - 1)
        if constant_time:
            factor = ct_table_lookup(model, table, digit, modulus)
        else:
            model.secret_branches += 1
            model.trace.append("lookup(T[%d])" % digit)      # the leak: only this index is touched
            factor = table[digit]
        result = ct_mul(model, result, "R", factor, "T", modulus)
    return result


def run(exp_func, base, exponent, modulus, **kwargs):
    model = Model()
    value = exp_func(model, base, exponent, modulus, **kwargs)
    return value, model


def random_case(state, bits):
    state, r = splitmix64(state)
    modulus = MODULI[r % len(MODULI)]
    state, r = splitmix64(state)
    base = r % (modulus + 1)
    state, r = splitmix64(state)
    exponent = r % (1 << bits)
    return (base, exponent, modulus), state


def main():
    # 1. correctness of the ladder against pow, for random cases and for the boundary exponents
    state = 0x5EED
    cases = []
    for _ in range(60):
        case, state = random_case(state, 20)
        cases.append(case)
    for modulus in MODULI:
        cases += [(0, 0, modulus), (0, 1, modulus), (1, 0, modulus), (1, 1 << 16, modulus),
                  (modulus - 1, 0, modulus), (modulus + 5, 12, modulus), (2, 1 << 15, modulus)]
    for base, exponent, modulus in cases:
        expected = pow(base, exponent, modulus)
        value, _ = run(ladder_pow, base, exponent, modulus)
        check(value == expected, "ladder_pow(%d, %d, %d) = %d, expected %d"
              % (base, exponent, modulus, value, expected))

    # 2. the ladder invariant: R0 = base^P and R1 = base^(P+1) after every processed bit
    seen = []
    base, exponent, modulus = 0x1234, 0xABCDEF, MODULI[0]
    ladder_pow(Model(), base, exponent, modulus, 24,
               on_step=lambda prefix, r0, r1: seen.append(
                   (r0 == pow(base, prefix, modulus) and r1 == pow(base, prefix + 1, modulus), prefix)))
    check(seen and all(flag for flag, _ in seen), "the ladder invariant must hold at every step")
    check([prefix for _, prefix in seen] == [exponent >> shift for shift in range(23, -1, -1)],
          "the recorded prefixes must be the successive prefixes of the exponent")

    # 3. the window variant agrees with the ladder and with pow, for several window sizes
    for base, exponent, modulus in cases[:30]:
        for window in (1, 2, 4):
            value, _ = run(window_pow, base, exponent, modulus, window=window)
            check(value == pow(base, exponent, modulus),
                  "window_pow(w=%d) on (%d, %d, %d)" % (window, base, exponent, modulus))

    # 4. constant cost: for one padded bit length, every exponent costs the same
    exponents = [0, 1, 2, 3, 0xFFFF, 0xAAAA, 0x5555, 0x8000, 0x7FFF, 0x1234]
    ladder_costs = set()
    window_costs = set()
    ladder_traces = set()
    for exponent in exponents:
        _, ladder_model = run(ladder_pow, 3, exponent, MODULI[0], bit_length=16)
        _, window_model = run(window_pow, 3, exponent, MODULI[0], window=4, bit_length=16)
        ladder_costs.add(ladder_model.muls)
        window_costs.add(window_model.muls)
        ladder_traces.add(tuple(ladder_model.trace))
        check(ladder_model.secret_branches == 0 and window_model.secret_branches == 0,
              "the constant-time variants must not branch on secret bits")
    check(len(ladder_costs) == 1 and ladder_costs == {3 * 16},
          "the ladder must spend 3 multiplications per bit, always: %r" % (ladder_costs,))
    check(len(window_costs) == 1 and window_costs == {16 + 4},
          "the window variant must spend w squarings plus one multiplication per window: %r"
          % (window_costs,))
    check(len(ladder_traces) == 1, "the ladder's operation trace must not depend on the exponent")

    # 5. the naive variant leaks both the cost and the take-branch count: exactly the popcount
    naive_costs = []
    for exponent in exponents:
        _, model = run(naive_pow, 3, exponent, MODULI[0])
        naive_costs.append((model.muls, model.secret_branches))
        check(model.muls == exponent.bit_length() + bin(exponent).count("1"),
              "naive cost must be the exponent's bit length plus its Hamming weight")
        check(model.secret_branches == bin(exponent).count("1"),
              "naive_pow must branch once per set bit of the secret exponent")
    check(len(set(cost for cost, _ in naive_costs)) > 1,
          "the naive cost must differ between exponents of the same length")
    check(naive_costs[4][0] != naive_costs[8][0],
          "all-ones and a single-bit exponent must be distinguishable by cost alone")

    # 6. the constant-time window scan touches every table index, in a fixed order
    _, scan_model = run(window_pow, 3, 0xABCD, MODULI[0], window=4, bit_length=16)
    reads = scan_model.table_reads()
    check(reads == ["read(T[%d])" % i for _ in range(4) for i in range(16)],
          "the constant-time lookup must scan all 16 entries once per window")
    check(scan_model.leaked_table_indices() == [],
          "the scanning lookup must not expose the window digit")

    # 7. the attacker's view: a non-scanning lookup leaks the exponent digits, which are recovered
    digits = [(0xABCD >> start) & 0xF for start in range(12, -1, -4)]
    _, leaky_model = run(window_pow, 3, 0xABCD, MODULI[0], window=4, bit_length=16,
                         constant_time=False)
    recovered = leaky_model.leaked_table_indices()
    check(recovered == digits, "the leaky lookup must expose the real window digits %r" % (digits,))
    rebuilt = 0
    for digit in recovered:
        rebuilt = (rebuilt << 4) | digit
    check(rebuilt == 0xABCD, "the recovered digits must rebuild the secret exponent exactly")
    _, guarded_model = run(window_pow, 3, 0xABCD, MODULI[0], window=4, bit_length=16)
    check(guarded_model.leaked_table_indices() == [],
          "the constant-time variant must not leak the digits the naive one does")

    # 8. padding: the trace depends on the public bit length, the result does not
    padded_value, padded_model = run(ladder_pow, 3, 5, MODULI[0], bit_length=8)
    short_value, short_model = run(ladder_pow, 3, 5, MODULI[0], bit_length=3)
    check(padded_value == short_value == pow(3, 5, MODULI[0]), "padding must not change the result")
    check(padded_model.muls > short_model.muls and padded_model.muls == 24,
          "a longer padded length must cost proportionally more work")
    try:
        ladder_pow(Model(), 3, 1 << 9, MODULI[0], 8)
        raise AssertionError("an exponent longer than the declared bit length must be rejected")
    except ValueError:
        pass
    try:
        window_pow(Model(), 3, 1 << 9, MODULI[0], window=4, bit_length=8)
        raise AssertionError("the window variant must reject a too small bit length")
    except ValueError:
        pass

    # 9. the mask primitives: exhaustive on the boundary set, and branch-free by construction
    boundary = [0, 1, 2, 15, 16, 255, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    for a in boundary:
        for b in boundary:
            check(ct_eq_mask(a, b) == (1 if a == b else 0), "ct_eq_mask(%d,%d)" % (a, b))
    for bit in (0, 1):
        for left in boundary:
            for right in boundary:
                for modulus in MODULI:
                    check(ct_select(Model(), bit, left % modulus, right % modulus, modulus)
                          == (left % modulus if bit else right % modulus),
                          "ct_select must be exact for bit %d, %d, %d" % (bit, left, right))

    # 10. degenerate moduli and boundaries that the loop must survive
    check(run(ladder_pow, 7, 0, 1)[0] == pow(7, 0, 1) == 0, "modulus 1 always yields 0")
    check(run(window_pow, 7, 0, 1, window=2, bit_length=2)[0] == 0, "modulus 1 in the window path")
    check(run(ladder_pow, 0, 0, 7)[0] == 1 and run(ladder_pow, 0, 3, 7)[0] == 0,
          "base 0 with exponent 0 is 1, with exponent 3 is 0")
    check(run(ladder_pow, 5, 0, 13, bit_length=0)[0] == 1, "a zero-length exponent is the identity")
    check(run(naive_pow, 5, 0, 13)[0] == 1 and run(naive_pow, 5, 1, 13)[0] == 5,
          "the naive path must agree on the identity and the single-bit exponent")

    # 11. determinism: identical inputs give identical results and traces
    first_value, first_model = run(ladder_pow, 0x1234, 0xDEADBEEF, MODULI[0], bit_length=32)
    second_value, second_model = run(ladder_pow, 0x1234, 0xDEADBEEF, MODULI[0], bit_length=32)
    check(first_value == second_value == pow(0x1234, 0xDEADBEEF, MODULI[0]),
          "the ladder must be deterministic")
    check(first_model.trace == second_model.trace, "traces must be reproducible")

    print("756 ladder: %d muls for every 16-bit exponent, %d secret branches, 1 distinct trace"
          % (ladder_costs.pop(), 0))
    print("756 window(w=4,16 bits): %d muls for every exponent, %d fixed table reads per window"
          % (window_costs.pop(), 16))
    print("756 naive: cost bits+popcount, e.g. %r; leaky window recovered exponent 0x%X from %d reads"
          % (naive_costs[:4], rebuilt, len(recovered)))
    print("756 mask primitives verified on %d boundary values; pow used as the oracle"
          % len(boundary))
    print("PASS")


if __name__ == "__main__":
    main()
