"""Task 720 - Implement ring-allreduce simulation and communication cost modeling.  (Tier: Expert, Ch 9 — System Design and Scalability)

All-reduce (every rank ends with the element-wise sum of all ranks' buffers) is the primitive
behind data-parallel gradient synchronisation.  The ring algorithm splits each buffer into N
chunks and runs two N-1 step phases over a logical ring:

* reduce-scatter: at step s rank r sends chunk (r - s) mod N to rank (r + 1) mod N and the
  receiver adds it into the same chunk slot.  After N-1 steps rank r owns the *complete* sum
  of chunk (r + 1) mod N - the chunks have rotated so that one fully reduced chunk sits on
  each rank;
* all-gather: at step s rank r sends chunk (r - s + 1) mod N to its successor, so after
  another N-1 steps every rank holds all N reduced chunks.

That gives 2(N-1) sequential messages per rank and 2(N-1)/N * S elements moved per rank -
within a factor of two of the information-theoretic minimum and, unlike a broadcast
all-reduce that moves (N-1) * S per rank, independent of the number of ranks.

The simulation really passes messages: each step builds a snapshot of every outgoing chunk
before touching any buffer (so a rank can never read a half-updated neighbour), and both the
message log and the byte counters are inspected by the checks.  Buffers are padded to a
multiple of N with zeros; the padding is trimmed from the result but is still charged to the
wire, which is exactly what the hand-computed volume formula predicts.

Simplified: sequential step-synchronous execution on a perfect ring (no congestion, no
overlap of communication with computation, no tree or recursive-halving-doubling variant, no
reduction operators other than addition, and no datatype conversion or in-flight
compression).  Bandwidth is modelled by a flat bytes-per-element charge plus a per-message
latency alpha, i.e. the classic alpha + beta*bytes model.  What is verified: the reduced
buffer equals a brute-force sum exactly (integer-valued payloads make the comparison exact),
every rank ends with identical results, the reduce-scatter ownership mapping and the message
schedule match the formulas above, the byte counters equal the volume model, and the boundary
cases (one rank, two ranks, empty payload, payload smaller than the ring, non-divisible
payload) all behave.
"""
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def integer_buffer(size, seed, low=-9, high=9):
    """Integer-valued payloads: float addition of small integers is exact, so the reduced
    buffer can be compared to the brute-force sum with == rather than a tolerance."""
    out = []
    state = seed & MASK64
    span = high - low + 1
    for _ in range(size):
        state, r = splitmix64(state)
        out.append(float(low + r % span))
    return out, state


class RingAllReduce:
    """Step-synchronous ring all-reduce with an inspectable message log."""

    def __init__(self, buffers):
        if not buffers:
            raise ValueError("at least one rank is required")
        size = len(buffers[0])
        if any(len(b) != size for b in buffers):
            raise ValueError("all ranks must hold the same number of elements")
        self.nranks = len(buffers)
        self.size = size
        self.chunk = -(-size // self.nranks) if size else 0
        self.padded = self.chunk * self.nranks
        self.state = []
        for b in buffers:
            full = list(b) + [0.0] * (self.padded - size)
            self.state.append([full[i * self.chunk:(i + 1) * self.chunk]
                               for i in range(self.nranks)])
        self.log = []                  # (phase, step, sender, receiver, chunk)
        self.bytes_sent = [0] * self.nranks
        self.steps = 0

    def _send(self, phase, step):
        """One ring step: snapshot every outgoing chunk first, then let receivers add/copy."""
        sender_of = {r: list(self.state[r][(r - step) % self.nranks]) for r in range(self.nranks)}
        for r in range(self.nranks):
            receiver = (r + 1) % self.nranks
            slot = (r - step) % self.nranks
            if phase == "reduce-scatter":
                target = self.state[receiver][slot]
                for j, value in enumerate(sender_of[r]):
                    target[j] += value
            else:
                self.state[receiver][slot] = list(sender_of[r])
            self.bytes_sent[r] += self.chunk
            self.log.append((phase, step, r, receiver, slot))
        self.steps += 1

    def reduce_scatter(self):
        for step in range(self.nranks - 1):
            self._send("reduce-scatter", step)
        return self

    def all_gather(self):
        for step in range(self.nranks - 1):
            # sending chunk (r - s + 1) mod N is the same rule with the step shifted by one
            self._send("all-gather", step - 1)
        return self

    def run(self):
        self.reduce_scatter()
        self.all_gather()
        return self

    def results(self):
        flattened = []
        for rank in range(self.nranks):
            flat = [v for chunk in self.state[rank] for v in chunk]
            flattened.append(flat[:self.size])
        return flattened

    def owned_chunks(self):
        """After reduce-scatter rank r fully owns chunk (r + 1) mod N (that is the mapping)."""
        return [[v for v in self.state[r][(r + 1) % self.nranks]] for r in range(self.nranks)]


def broadcast_allreduce(buffers):
    """Baseline: rank r ships its whole buffer to every other rank (fully connected fabric)."""
    nranks = len(buffers)
    size = len(buffers[0])
    totals = [sum(b[i] for b in buffers) for i in range(size)]
    volume_per_rank = (nranks - 1) * size
    return totals, volume_per_rank


def ring_volume_per_rank(nranks, size):
    """Hand formula: 2 * (N - 1) messages of ceil(size / N) elements each."""
    chunk = -(-size // nranks) if size else 0
    return 2 * (nranks - 1) * chunk


def ring_cost(nranks, size, alpha, beta):
    """Classic alpha + beta * bytes model: 2(N-1) sequential messages per rank."""
    return 2 * (nranks - 1) * alpha + ring_volume_per_rank(nranks, size) * beta


def brute_sum(buffers):
    size = len(buffers[0])
    return [sum(b[i] for b in buffers) for i in range(size)]


def main():
    # 1. correctness on a small ring with integer payloads: every rank must end with the
    #    exact brute-force sum, and the padding must be trimmed.
    buffers = [[float(i + r) for i in range(12)] for r in range(4)]
    expected = brute_sum(buffers)
    run = RingAllReduce(buffers).run().results()
    check(all(result == expected for result in run), "every rank must hold the exact sum")
    check(all(len(result) == 12 for result in run), "the padded tail must be trimmed away")

    # 2. cost model: 2(N-1) sequential steps and the hand-computed per-rank volume, including
    #    the bytes spent on padding when the payload is not divisible by N.
    model = RingAllReduce(buffers)
    model.run()
    check(model.steps == 2 * (4 - 1), "a 4-rank ring must need 6 sequential steps")
    check(model.bytes_sent == [ring_volume_per_rank(4, 12)] * 4, "per-rank volume must match")
    check(set(model.bytes_sent) == {2 * 3 * 3}, "each rank sends 6 chunks of 3 elements")
    total_moved = sum(model.bytes_sent)
    check(total_moved == model.nranks * 2 * (model.nranks - 1) * model.chunk,
          "total traffic must be N * 2(N-1) * chunk")
    check(len(model.log) == model.nranks * 2 * (model.nranks - 1),
          "every rank sends one message per step")

    # 3. the message schedule: always to the successor, and the chunk index is (sender - step)
    #    mod N - for the gather phase the recorded step is shifted by one, which is exactly the
    #    (r - s + 1) mod N rule, so the gather re-uses the scatter schedule.
    for phase, step, sender, receiver, slot in model.log:
        check(receiver == (sender + 1) % 4, "messages may only travel to the successor")
        check(slot == (sender - step) % 4, "the chunk index must follow the ring schedule")
    check(sorted({e[1] for e in model.log if e[0] == "all-gather"}) == [-1, 0, 1],
          "the gather phase must be the scatter schedule shifted by one chunk")

    # 4. reduce-scatter ownership: rank r ends up owning the full sum of chunk (r + 1) mod N,
    #    verified chunk by chunk against a brute-force reduction.
    scatter = RingAllReduce(buffers).reduce_scatter()
    owned = scatter.owned_chunks()
    for r in range(4):
        chunk_id = (r + 1) % 4
        reference = [sum(buffers[b][chunk_id * 3 + j] for b in range(4)) for j in range(3)]
        check(owned[r] == reference, "reduce-scatter ownership must follow (r + 1) mod N")
    check(scatter.steps == 3, "reduce-scatter alone must take N-1 steps")

    # 5. the pay-off: a ring moves 2(N-1)/N * S per rank regardless of N, whereas a
    #    broadcast all-reduce moves (N-1) * S per rank.
    for nranks in (3, 4, 8, 16):
        ring_small = RingAllReduce([[1.0] * 32 for _ in range(nranks)]).run()
        _, naive_volume = broadcast_allreduce([[1.0] * 32 for _ in range(nranks)])
        check(ring_small.bytes_sent[0] == ring_volume_per_rank(nranks, 32),
              "ring volume must be the volume model, not a lookup")
        check(ring_small.bytes_sent[0] * nranks < naive_volume * nranks,
              "the ring must move less traffic than a broadcast all-reduce")
    check(ring_volume_per_rank(16, 32) == 2 * 15 * 2, "N=16 with 32 elements: 2 chunks each")
    check(ring_volume_per_rank(3, 32) == 2 * 2 * 11, "ceil(32/3) = 11 padded elements")

    # 6. the cost model is monotone in both alpha and beta, and the latency term dominates for
    #    small payloads while the bandwidth term dominates for large ones.
    check(ring_cost(8, 1024, 1.0, 0.0) < ring_cost(8, 1024, 1.0, 0.1),
          "bandwidth must cost more than nothing")
    check(ring_cost(8, 8, 1.0, 0.5) > ring_cost(4, 8, 1.0, 0.5),
          "more ranks must cost more latency")
    check(ring_cost(4, 8, 1.0, 0.5) - 6 * 1.0 == ring_volume_per_rank(4, 8) * 0.5,
          "the model must decompose into latency plus bandwidth")

    # 7. a randomized cross-check against the brute-force sum, with random ring sizes and
    #    payload sizes (including payloads smaller than the ring and non-divisible ones).
    seed = 0x7201
    for nranks, size in ((1, 7), (2, 7), (3, 10), (5, 3), (6, 13), (8, 1), (4, 40)):
        buffers = []
        for _ in range(nranks):
            values, seed = integer_buffer(size, seed)
            buffers.append(values)
        result = RingAllReduce(buffers).run().results()
        reference = brute_sum(buffers)
        check(all(r == reference for r in result),
              f"ring all-reduce must be exact for N={nranks} and S={size}")

    # 8. boundary cases: a single rank transfers nothing, and an empty payload moves no bytes.
    alone = RingAllReduce([[3.0, 4.0, 5.0]])
    alone.run()
    check(alone.steps == 0 and alone.bytes_sent == [0], "one rank needs no communication")
    check(alone.results() == [[3.0, 4.0, 5.0]], "one rank is already reduced")
    empty = RingAllReduce([[], [], []])
    empty.run()
    check(empty.results() == [[], [], []], "an empty payload must stay empty")
    check(empty.bytes_sent == [0, 0, 0], "an empty payload must cost no bytes")
    check(ring_volume_per_rank(4, 0) == 0, "the volume model must handle a zero payload")
    pair = RingAllReduce([[1.0, 2.0], [3.0, 4.0]])
    pair.run()
    check(pair.results() == [[4.0, 6.0], [4.0, 6.0]], "two ranks must exchange whole buffers")
    check(pair.bytes_sent == [2, 2] and pair.steps == 2,
          "a 2-rank ring sends the whole buffer once per phase")

    # 9. malformed configurations and determinism.
    expect_value_error(lambda: RingAllReduce([]), "a ring needs at least one rank")
    expect_value_error(lambda: RingAllReduce([[1.0], [2.0, 3.0]]), "buffers must be equal sized")
    first = RingAllReduce(buffers).run()
    second = RingAllReduce(buffers).run()
    check(first.log == second.log and first.results() == second.results(),
          "identical inputs must produce identical messages and results")
    check(first.bytes_sent == second.bytes_sent, "identical inputs must produce identical costs")

    exact = RingAllReduce(buffers).run()
    check(exact.results()[0] == brute_sum(buffers), "the trunk of every result is the sum")
    print(f"720 ring N={exact.nranks} S={exact.size} steps={exact.steps} "
          f"per_rank_bytes={exact.bytes_sent[0]} total={sum(exact.bytes_sent)}")
    print(f"720 volume N=16 S=32: ring={ring_volume_per_rank(16, 32)} "
          f"broadcast=[{broadcast_allreduce([[1.0] * 32 for _ in range(16)])[1]}]")
    print("PASS")


if __name__ == "__main__":
    main()
