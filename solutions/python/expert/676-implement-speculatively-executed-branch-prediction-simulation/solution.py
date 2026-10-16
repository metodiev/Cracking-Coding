"""Task 676 - Implement speculatively executed branch prediction simulation.  (Tier: Expert, Beyond — Compilers, Languages & Solvers)

A three-stage in-order front end that fetches speculatively along the predicted path and
squashes everything behind a bad guess, driven by four deterministic predictors:

  * `OneBitPredictor`  - one bit per table entry (predict what happened last time);
  * `BimodalPredictor` - two-bit saturating counters, the classic four-state machine;
  * `GsharePredictor`  - the same counters indexed by pc XOR the global history of the last
    h outcomes, which is what makes history-correlated patterns predictable;
  * `TournamentPredictor` - two-bit chooser counters that pick between bimodal and gshare
    per branch.

`simulate` walks a trace of (pc, taken, target) records: direction comes from the predictor,
the target of a taken branch from a direct-mapped BTB, and every misprediction flushes
`penalty - 1` cycles and squashes just as many speculatively fetched instructions. Reported:
mispredictions, rate, cycles, CPI and squash count.

Verified: the saturating state machine against explicitly scripted sequences, the GHR update
rule against a hand-computed bit pattern, every table index inside its table, an independent
reference implementation of gshare (dictionaries plus masking instead of lists plus shifts)
reproducing the misprediction count exactly, the classic "2-bit beats 1-bit" result on
repeated loops, gshare beating bimodal on a history-correlated pattern, near-50% rates on an
uncorrelated trace, and the pipeline arithmetic
cycles = instructions + flushes * (penalty - 1).

Simplified: a fixed two-cycle flush, no fetch queue or I-cache effects, and traces are
generated deterministically (fixed-seed PRNG) rather than read from a binary.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def validated(size, name):
    if size < 1:
        raise ValueError("the %s table needs at least one entry" % name)
    return size


class OneBitPredictor:
    def __init__(self, size=16, shift=2):
        self.name = "1-bit"
        self.size = validated(size, "one-bit")
        self.shift = shift
        self.table = [0] * self.size
        self.mispredictions = 0

    def index(self, pc):
        return (pc >> self.shift) % self.size

    def predict(self, pc):
        return bool(self.table[self.index(pc)])

    def update(self, pc, taken):
        self.table[self.index(pc)] = 1 if taken else 0


class BimodalPredictor:
    """Two-bit saturating counter, 0..3; taken once the counter reaches 2."""

    def __init__(self, size=16, shift=2, initial=1):
        self.name = "bimodal"
        self.size = validated(size, "bimodal")
        self.shift = shift
        self.table = [initial] * self.size
        self.mispredictions = 0

    def index(self, pc):
        return (pc >> self.shift) % self.size

    def predict(self, pc):
        return self.table[self.index(pc)] >= 2

    def update(self, pc, taken):
        i = self.index(pc)
        self.table[i] = min(3, self.table[i] + 1) if taken else max(0, self.table[i] - 1)


class GsharePredictor:
    """Counters indexed by pc XOR the global history register."""

    def __init__(self, size=16, shift=2, history_bits=4):
        self.name = "gshare"
        self.size = validated(size, "gshare")
        if history_bits < 1:
            raise ValueError("gshare needs at least one history bit")
        self.shift = shift
        self.history_bits = history_bits
        self.history_mask = (1 << history_bits) - 1
        self.history = 0
        self.table = [1] * self.size
        self.mispredictions = 0

    def index(self, pc):
        return ((pc >> self.shift) ^ self.history) % self.size

    def predict(self, pc):
        return self.table[self.index(pc)] >= 2

    def update(self, pc, taken):
        i = self.index(pc)
        self.table[i] = min(3, self.table[i] + 1) if taken else max(0, self.table[i] - 1)
        self.history = ((self.history << 1) | (1 if taken else 0)) & self.history_mask


class TournamentPredictor:
    """A chooser counter decides per branch which of bimodal / gshare to trust."""

    def __init__(self, size=16, shift=2, history_bits=4):
        self.name = "tournament"
        self.size = validated(size, "chooser")
        self.shift = shift
        self.bimodal = BimodalPredictor(size, shift)
        self.gshare = GsharePredictor(size, shift, history_bits)
        self.chooser = [1] * size
        self.mispredictions = 0

    def index(self, pc):
        return (pc >> self.shift) % self.size

    def predict(self, pc):
        if self.chooser[self.index(pc)] >= 2:
            return self.gshare.predict(pc)
        return self.bimodal.predict(pc)

    def update(self, pc, taken):
        bimodal_wrong = self.bimodal.predict(pc) != taken
        gshare_wrong = self.gshare.predict(pc) != taken
        i = self.index(pc)
        if gshare_wrong and not bimodal_wrong:
            self.chooser[i] = max(0, self.chooser[i] - 1)      # trust bimodal here
        elif bimodal_wrong and not gshare_wrong:
            self.chooser[i] = min(3, self.chooser[i] + 1)      # trust gshare here
        self.bimodal.update(pc, taken)
        self.gshare.update(pc, taken)


class Btb:
    """Direct-mapped branch target buffer; only taken branches consult it."""

    def __init__(self, entries=128):
        self.entries = entries
        self.hits = 0
        self.misses = 0
        self.table = [None] * entries

    def lookup(self, pc, target):
        slot = (pc >> 2) % self.entries
        ok = self.table[slot] == target
        self.table[slot] = target
        self.hits += ok
        self.misses += not ok
        return ok


def reference_gshare(trace, size=16, shift=2, history_bits=4):
    """Independent gshare: dicts plus explicit masking instead of lists and shifts."""
    mask = (1 << history_bits) - 1
    counters, history, misses = {}, 0, 0
    for pc, taken, _ in trace:
        key = ((pc >> shift) ^ history) % size
        state = counters.get(key, 1)
        if (state >= 2) != taken:
            misses += 1
        counters[key] = min(3, state + 1) if taken else max(0, state - 1)
        history = ((history * 2) + (1 if taken else 0)) & mask
    return misses


def simulate(trace, predictor, btb=None, penalty=3):
    """Run the trace and report the speculative-execution accounting."""
    direction_misses = target_misses = 0
    for pc, taken, target in trace:
        predicted = predictor.predict(pc)
        if predicted != taken:
            direction_misses += 1
            predictor.mispredictions += 1
        elif taken and btb is not None and not btb.lookup(pc, target):
            target_misses += 1
        predictor.update(pc, taken)
    flushes = direction_misses + target_misses
    instructions = len(trace)
    cycles = instructions + flushes * (penalty - 1)
    return {"instructions": instructions, "direction_misses": direction_misses,
            "target_misses": target_misses, "flushes": flushes, "cycles": cycles,
            "squashed": flushes * (penalty - 1),
            "rate": direction_misses / instructions if instructions else 0.0,
            "cpi": cycles / instructions if instructions else 0.0}


def loop_trace(iterations, pc=0x100, inner=0x200):
    """A loop-back branch plus a periodic inner branch (taken every third iteration)."""
    trace = []
    for i in range(iterations):
        take_loop = i < iterations - 1
        trace.append((pc, take_loop, pc if take_loop else pc + 4))
        inner_taken = i % 3 == 0
        trace.append((inner, inner_taken, inner + 0x40 if inner_taken else inner + 4))
    return trace


def repeated_loop_trace(inner, repeats, pc=0x100):
    trace = []
    for _ in range(repeats):
        for i in range(inner):
            trace.append((pc, i < inner - 1, pc if i < inner - 1 else pc + 4))
    return trace


def correlated_trace(iterations, pc=0x300):
    """The (T, T, N) pattern: memorable for history, awkward for a single counter."""
    trace = []
    for i in range(iterations):
        taken = i % 3 != 2
        trace.append((pc, taken, pc + 0x40 if taken else pc + 4))
    return trace


def random_trace(count, seed, pc=0x400):
    state = seed
    trace = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        taken = ((state >> 33) & 1) == 1
        trace.append((pc, taken, pc + 0x40 if taken else pc + 4))
    return trace


def main():
    # ---- the two-bit state machine ----------------------------------------------------
    saturating = BimodalPredictor(4, 2)
    for _ in range(3):
        saturating.update(0x10, False)
    check(saturating.table[0] == 0, "three not-taken branches saturate the counter at zero")
    check(saturating.predict(0x10) is False, "a saturated-down counter predicts not taken")
    for _ in range(3):
        saturating.update(0x10, True)
    check(saturating.table[0] == 3, "three taken branches saturate it at three")
    check(saturating.predict(0x10) is True, "a saturated-up counter predicts taken")

    script = [(0x10, False, 0), (0x10, False, 0), (0x10, False, 0), (0x10, True, 0x20),
              (0x10, True, 0x20), (0x10, False, 0)]
    bimodal = BimodalPredictor(4, 2)
    predictions = []
    for pc, taken, _ in script:
        predictions.append(bimodal.predict(pc))
        bimodal.update(pc, taken)
    check(predictions == [False, False, False, False, False, True],
          "it takes two taken branches to flip the prediction: %r" % (predictions,))
    check(bimodal.table[0] == 1, "the final not-taken eases the counter back to 1")

    onebit = OneBitPredictor(4, 2)
    bits = []
    for pc, taken, _ in script:
        bits.append(onebit.predict(pc))
        onebit.update(pc, taken)
    check(bits == [False, False, False, False, True, True],
          "the one-bit predictor flips on the first change: %r" % (bits,))

    # ---- gshare bookkeeping -----------------------------------------------------------
    gshare = GsharePredictor(16, 2, 4)
    for pc, taken, _ in script:
        gshare.predict(pc)
        gshare.update(pc, taken)
    check(gshare.history == 0b0110,
          "the global history keeps the last four outcomes: %r" % (bin(gshare.history),))
    check(gshare.index(0x10) == (((0x10 >> 2) ^ 0b0110) % 16),
          "the gshare index is pc XOR history folded into the table: %d" % gshare.index(0x10))
    check(all(0 <= predictor.index(pc) < predictor.size for pc, _, _ in script
              for predictor in (bimodal, onebit, gshare)),
          "every table index stays inside its table")
    bounded = GsharePredictor(16, 2, 2)
    for _ in range(10):
        bounded.update(0x20, True)
    check(bounded.history == 0b11, "the history register is masked to its width")

    # ---- an independent gshare implementation ----------------------------------------
    for label, trace in (("loop", loop_trace(12)), ("correlated", correlated_trace(30)),
                         ("random", random_trace(200, 0xBEEF))):
        stats = simulate(trace, GsharePredictor(32, 2, 4))
        check(stats["direction_misses"] == reference_gshare(trace, 32, 2, 4),
              "table and reference gshare disagree on the %s trace (%d vs %d)"
              % (label, stats["direction_misses"], reference_gshare(trace, 32, 2, 4)))

    # ---- the classic results ----------------------------------------------------------
    nested = repeated_loop_trace(5, 8)
    nested_one = simulate(nested, OneBitPredictor(32, 2))
    nested_two = simulate(nested, BimodalPredictor(32, 2))
    check(nested_two["direction_misses"] < nested_one["direction_misses"],
          "two-bit counters beat one bit on repeated loops: %d < %d"
          % (nested_two["direction_misses"], nested_one["direction_misses"]))
    check(nested_two["direction_misses"] <= 8 + 2,
          "one misprediction per loop execution, not per iteration: %d"
          % nested_two["direction_misses"])
    check(nested_one["direction_misses"] >= 2 * 8 - 2,
          "the one-bit predictor pays two per loop: %d" % nested_one["direction_misses"])

    correlated = correlated_trace(60)
    corr_two = simulate(correlated, BimodalPredictor(64, 2))
    corr_gshare = simulate(correlated, GsharePredictor(64, 2, 4))
    check(corr_gshare["direction_misses"] < corr_two["direction_misses"],
          "history disambiguates the TTN pattern: gshare %d < bimodal %d"
          % (corr_gshare["direction_misses"], corr_two["direction_misses"]))
    check(corr_gshare["rate"] < 0.1,
          "the pattern is almost perfectly predictable with history: %.3f" % corr_gshare["rate"])
    check(corr_two["rate"] > 0.25,
          "without history the pattern costs about one in three: %.3f" % corr_two["rate"])

    noisy = random_trace(400, 0x1234567)
    noisy_two = simulate(noisy, BimodalPredictor(64, 2))
    noisy_gshare = simulate(noisy, GsharePredictor(64, 2, 6))
    for stats in (noisy_two, noisy_gshare):
        check(0.3 < stats["rate"] < 0.7,
              "an uncorrelated trace must stay near 50%%: %.3f" % stats["rate"])
    static_taken = sum(1 for _, taken, _ in noisy if not taken) / len(noisy)
    check(abs(noisy_two["rate"] - static_taken) < 0.1,
          "with no correlation a dynamic predictor is no better than static taken: %.3f vs %.3f"
          % (noisy_two["rate"], static_taken))

    # ---- the tournament ---------------------------------------------------------------
    loop = loop_trace(40)
    tournament_loop = simulate(loop, TournamentPredictor(64, 2, 4))
    loop_two = simulate(loop, BimodalPredictor(64, 2))
    tournament_corr = simulate(correlated, TournamentPredictor(64, 2, 4))
    check(tournament_loop["direction_misses"] <= loop_two["direction_misses"] + 4,
          "the chooser follows bimodal on the loop (%d vs %d)"
          % (tournament_loop["direction_misses"], loop_two["direction_misses"]))
    check(tournament_corr["direction_misses"] <= corr_gshare["direction_misses"] + 6,
          "the chooser follows gshare on the correlated pattern (%d vs %d)"
          % (tournament_corr["direction_misses"], corr_gshare["direction_misses"]))
    check(tournament_corr["direction_misses"] < corr_two["direction_misses"],
          "the tournament must not trail the weaker component here (%d vs %d)"
          % (tournament_corr["direction_misses"], corr_two["direction_misses"]))

    # ---- targets, pipeline arithmetic and boundary cases ------------------------------
    btb = Btb(128)
    stats = simulate(loop, BimodalPredictor(64, 2), btb)
    probe = BimodalPredictor(64, 2)          # replay to count correct taken predictions
    correct_taken = 0
    for pc, taken, _ in loop:
        if taken and probe.predict(pc) == taken:
            correct_taken += 1
        probe.update(pc, taken)
    check(btb.misses == 2,
          "each of the two taken targets is unknown exactly once: %d" % btb.misses)
    check(btb.hits == correct_taken - 2,
          "only correctly predicted taken branches consult the buffer: %d vs %d"
          % (btb.hits, correct_taken - 2))
    check(stats["flushes"] == stats["direction_misses"] + stats["target_misses"], "flush count")
    check(stats["cycles"] == stats["instructions"] + stats["flushes"] * 2,
          "cycles = instructions + flushes * (penalty - 1): %r" % (stats,))
    check(stats["squashed"] == stats["flushes"] * 2,
          "each flush squashes two speculatively executed instructions")
    check(stats["cpi"] > 1.0, "a mispredicting front end runs above one CPI")

    empty = simulate([], BimodalPredictor(4, 2))
    check(empty["cycles"] == 0 and empty["rate"] == 0.0 and empty["cpi"] == 0.0,
          "an empty trace costs nothing: %r" % (empty,))
    check(simulate([(0x40, True, 0x80)], OneBitPredictor(4, 2))["rate"] == 1.0,
          "a cold one-bit predictor always mispredicts the first branch")
    check(simulate([(0x40, False, 0x84)], OneBitPredictor(4, 2), Btb(8))["target_misses"] == 0,
          "a not-taken branch never consults the target buffer")
    rejected = 0
    for bad in (lambda: OneBitPredictor(0),
                lambda: BimodalPredictor(-1),
                lambda: GsharePredictor(4, 2, 0),
                lambda: TournamentPredictor(0)):
        try:
            bad()
        except ValueError:
            rejected += 1
    check(rejected == 4, "degenerate table sizes must be rejected (%d of 4)" % rejected)

    check(loop_trace(5) == loop_trace(5) and random_trace(9, 7) == random_trace(9, 7),
          "the trace generators must be deterministic")
    check(simulate(correlated, GsharePredictor(64, 2, 4)) == corr_gshare,
          "the simulation must be deterministic")

    print("repeated loop (8x5): 1-bit %d vs 2-bit %d misses"
          % (nested_one["direction_misses"], nested_two["direction_misses"]))
    print("correlated(60): 2-bit %.3f, gshare %.3f, tournament %.3f; random(400): %.3f / %.3f"
          % (corr_two["rate"], corr_gshare["rate"], tournament_corr["rate"],
             noisy_two["rate"], noisy_gshare["rate"]))
    print("loop with a BTB: CPI %.3f over %d branches" % (stats["cpi"], len(loop)))
    print("PASS")


if __name__ == "__main__":
    main()
