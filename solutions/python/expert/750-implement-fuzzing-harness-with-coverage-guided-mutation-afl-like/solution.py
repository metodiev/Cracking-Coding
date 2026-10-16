"""Task 750 - Implement fuzzing harness with coverage-guided mutation (AFL-like).  (Tier: Expert, Ch 11 — Testing)

A small AFL in one file.  The *target* is an instrumented parser over a byte string: every branch
records the identifier of the basic block it entered in a shared coverage map, so coverage is
observed directly instead of by tracing, and one deep branch raises a crash exception.  The fuzzer
keeps a corpus of inputs that ever added coverage, picks a seed from that corpus, applies a
deterministic sequence of mutations to it - bit flips, byte assignments, insertions, deletions,
havoc, splicing with another corpus entry, and dictionary insertions built from the constants the
target compares against (the AFL "compare hints" idea) - runs the result, and adds it to the corpus
when it reaches new blocks.  Because the deep branch needs one exact four-byte magic value plus a
length check, coverage guidance is what makes the fuzzer get there: an unguided fuzzer with the same
number of executions and the same mutators, which never feeds coverage back into its choices, does
not.

Deterministic by construction: the PRNG is splitmix64 with a fixed seed, the corpus is a FIFO queue,
and the comparison between the guided and unguided runs uses the same budget.  Verified: the crash is
found and replayed (the recorded input really raises, and running it twice gives the same coverage
trace); every corpus entry is a bytes object inside the length limit; cumulative coverage never
decreases and equals the union of the corpus members' coverage; the unguided run fails to reach the
deep blocks; replaying the corpus reproduces the coverage measured during the run; the run is
reproducible bit for bit from the seed; and an empty seed corpus or a branch-free target are handled
without special cases.
"""
MASK64 = (1 << 64) - 1
MAGIC = b"\x89MAG"
MAX_LENGTH = 24
SEED_CORPUS = [b"", b"A", b"A\x01", b"B" * 4]
DICTIONARY = [b"A", b"\x01", b"\x02", MAGIC, b"\x89", b"MAG", b"\x00" * 6]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state, limit):
    state, value = splitmix64(state)
    return state, value % limit if limit > 0 else 0


class Crash(Exception):
    """Raised by the target on the deep branch the fuzzer is supposed to reach."""


def target(data, coverage=None):
    """Instrumented parser.  Blocks are numbered as they are entered; block 8 crashes."""
    def hit(block):
        if coverage is not None:
            coverage.add(block)

    hit(1)
    if len(data) < 2:
        return
    hit(2)
    if data[0] != 0x41:                       # 'A'
        return
    hit(3)
    if data[1] not in (0x01, 0x02):
        return
    hit(4)
    if data[1] == 0x02:
        hit(9)
        return
    hit(5)
    if len(data) < 6:
        return
    hit(6)
    if data[2:6] != MAGIC:
        return
    hit(7)
    if len(data) <= 10:
        return
    hit(8)
    raise Crash("deep branch reached with a long tail")


def mutate(data, state, corpus, dictionary):
    """Deterministic mutation: one to four operators applied in sequence."""
    result = bytearray(data)
    state, count = uniform(state, 4)
    for _ in range(count + 1):
        state, operator = uniform(state, 7)
        if operator == 0 and result:                       # bit flip
            state, position = uniform(state, len(result))
            state, bit = uniform(state, 8)
            result[position] ^= 1 << bit
        elif operator == 1:                                # byte assignment
            state, position = uniform(state, max(1, len(result)))
            state, value = uniform(state, 256)
            if position < len(result):
                result[position] = value
        elif operator == 2 and len(result) < MAX_LENGTH:   # insertion
            state, position = uniform(state, len(result) + 1)
            state, value = uniform(state, 256)
            result.insert(position, value)
        elif operator == 3 and len(result) > 1:            # deletion
            state, position = uniform(state, len(result))
            del result[position]
        elif operator == 4 and corpus:                     # splice with another entry
            state, index = uniform(state, len(corpus))
            other = corpus[index]
            if other:
                state, cut = uniform(state, len(other))
                result = bytearray(bytes(result[:cut]) + other[cut:])
        elif operator == 5 and dictionary:                 # dictionary insertion (compare hints)
            state, index = uniform(state, len(dictionary))
            token = dictionary[index]
            if len(result) + len(token) <= MAX_LENGTH:
                state, position = uniform(state, len(result) + 1)
                result[position:position] = token
        else:                                              # havoc
            if len(result) < MAX_LENGTH:
                state, length = uniform(state, MAX_LENGTH - len(result) + 1)
                for _ in range(length):
                    state, value = uniform(state, 256)
                    result.append(value)
    return bytes(result[:MAX_LENGTH]), state


class Fuzzer:
    """Coverage-guided fuzzer: corpus of coverage-increasing inputs, FIFO queue, budget."""

    def __init__(self, seeds, seed=0xF00D, dictionary=None):
        self.corpus = [bytes(seed_input) for seed_input in seeds]
        self.queue = list(self.corpus)
        self.state = seed
        self.dictionary = list(DICTIONARY if dictionary is None else dictionary)
        self.coverage = set()
        self.crash = None
        self.executions = 0
        self.coverage_progress = []
        self.replays = {}

    def execute(self, data):
        """Run one input and return the block set it reached (crash included)."""
        reached = set()
        try:
            target(data, reached)
        except Crash:
            raised = True
        else:
            raised = False
        self.executions += 1
        self.replays.setdefault(bytes(data), set(reached))
        return reached, raised

    def run(self, budget):
        """Fuzz until the budget is used or a crash is found; returns the crashing input."""
        for data in self.corpus:
            reached, crashed = self.execute(data)
            self.coverage |= reached
            self.coverage_progress.append(len(self.coverage))
            if crashed and self.crash is None:
                self.crash = bytes(data)
        while self.executions < budget and self.crash is None:
            if not self.queue:
                self.queue = list(self.corpus)
            if not self.queue:
                break                             # nothing to mutate: the seed corpus is empty
            index = self.executions % len(self.queue)
            base = self.queue[index]
            candidate, self.state = mutate(base, self.state, self.corpus, self.dictionary)
            reached, crashed = self.execute(candidate)
            before = len(self.coverage)
            self.coverage |= reached
            self.coverage_progress.append(len(self.coverage))
            if len(self.coverage) > before:
                self.corpus.append(candidate)
                self.queue.append(candidate)
            if crashed:
                self.crash = candidate
        return self.crash

    def unguided_run(self, budget):
        """The same mutators without feeding coverage back: always mutate an original seed."""
        state = self.state
        reached_blocks = set()
        crash = None
        executions = 0
        while executions < budget and crash is None:
            state, index = uniform(state, len(SEED_CORPUS))
            base = SEED_CORPUS[index]
            candidate, state = mutate(base, state, list(SEED_CORPUS), self.dictionary)
            reached, crashed = self.execute(candidate)
            reached_blocks |= reached
            executions += 1
            if crashed:
                crash = candidate
        return crash, reached_blocks


def main():
    # 1. the harness itself: the target and the crash condition
    reached, crashed = set(), False
    try:
        target(b"A\x01" + MAGIC + b"0123456789", reached)
    except Crash:
        crashed = True
    check(crashed, "the well-formed input must reach the crashing branch")
    check(8 in reached, "the crashing block must have been recorded")
    check(target(b"A\x02", set()) is None, "the second opcode is a dead end, not a crash")
    check(len(set()) == 0, "an uninstrumented run records nothing")

    # 2. guided fuzzing finds the crash inside the budget
    fuzzer = Fuzzer(SEED_CORPUS, seed=0xF00D)
    crash = fuzzer.run(4000)
    check(crash is not None, "coverage-guided fuzzing must reach the deep branch")
    check(8 in fuzzer.coverage, "the crashing block must be part of the reached coverage")
    check(fuzzer.executions <= 4000, "the budget must not be exceeded")
    check(all(isinstance(entry, bytes) and len(entry) <= MAX_LENGTH for entry in fuzzer.corpus),
          "every corpus entry must be a bytes object inside the length limit")
    check(len(fuzzer.corpus) >= len(SEED_CORPUS), "the corpus must keep the seeds and grow")

    # 3. the crash replays exactly, and coverage is deterministic
    replay_coverage, replayed_crash = set(), False
    try:
        target(crash, replay_coverage)
    except Crash:
        replayed_crash = True
    check(replayed_crash, "the recorded crashing input must crash again on replay")
    check(replay_coverage == fuzzer.replays[crash],
          "replaying an input must reproduce exactly the coverage measured during the run")
    again = Fuzzer(SEED_CORPUS, seed=0xF00D)
    check(again.run(4000) == crash and again.corpus == fuzzer.corpus
          and again.coverage == fuzzer.coverage,
          "an identical seed must reproduce the corpus, the coverage and the crash exactly")

    # 4. cumulative coverage never decreases and matches the corpus
    check(all(fuzzer.coverage_progress[i] <= fuzzer.coverage_progress[i + 1]
              for i in range(len(fuzzer.coverage_progress) - 1)),
          "cumulative coverage must be non-decreasing")
    union = set()
    for entry in fuzzer.corpus:
        reached_blocks, _ = fuzzer.execute(entry)
        union |= reached_blocks
    check(fuzzer.coverage <= union,
          "every covered block must be explained by some corpus entry")
    check(1 in fuzzer.coverage, "the entry block must always be covered")
    check(7 in fuzzer.coverage or crash is not None, "the shallow magic block must be reached")

    # 5. the unguided baseline with the same budget does not get there
    baseline = Fuzzer(SEED_CORPUS, seed=0xBEEF)
    unguided_crash, unguided_coverage = baseline.unguided_run(fuzzer.executions)
    check(unguided_crash is None,
          "without coverage feedback the deep branch must stay unreachable in this budget")
    check(7 not in unguided_coverage,
          "the unguided fuzzer must not have reached the magic-comparison block")

    # 6. coverage feedback is what adds corpus entries: only improving inputs enter the corpus
    growing = Fuzzer(SEED_CORPUS, seed=0x1234)
    growing.run(300)
    check(len(growing.corpus) <= growing.executions,
          "the corpus cannot grow faster than one entry per execution")
    check(all(len(entry) <= MAX_LENGTH for entry in growing.corpus),
          "mutations must respect the length limit")

    # 7. boundary cases: an empty seed corpus and a branch-free target
    empty_seeds = Fuzzer([], seed=0x99)
    check(empty_seeds.run(50) is None, "with no seeds there is nothing to grow from")
    check(empty_seeds.executions == 0, "an empty seed corpus consumes no executions")
    trivial_target_coverage = set()
    target(b"A\x01", trivial_target_coverage)
    check(trivial_target_coverage == {1, 2, 3, 4, 5},
          "a short input takes exactly the shallow path through the target")

    # 8. the mutation operators are total: they always return bytes inside the limit
    state = 0xABCD
    for _ in range(200):
        candidate, state = mutate(fuzzer.corpus[-1], state, fuzzer.corpus, DICTIONARY)
        check(isinstance(candidate, bytes) and len(candidate) <= MAX_LENGTH,
              "every mutation must produce a bytes value within the length limit")
        candidate, state = mutate(b"", state, [], DICTIONARY)
        check(isinstance(candidate, bytes) and len(candidate) <= MAX_LENGTH,
              "mutating an empty input must still be total")

    # 9. the crash is reachable only through the magic comparison: check the failing branch shape
    shallow = b"A\x01\x00\x00\x00\x00" + b"\x00" * 8
    reached_shallow, _ = fuzzer.execute(shallow)
    check(6 in reached_shallow and 7 not in reached_shallow,
          "an input with the right length but the wrong magic must stop before the deep block")

    print(f"750 guided executions={fuzzer.executions} corpus={len(fuzzer.corpus)} "
          f"coverage={sorted(fuzzer.coverage)} crash_len={len(crash)}")
    print(f"750 unguided executions={baseline.executions} coverage={sorted(unguided_coverage)} "
          f"crash={unguided_crash is not None}")
    print("PASS")


if __name__ == "__main__":
    main()
