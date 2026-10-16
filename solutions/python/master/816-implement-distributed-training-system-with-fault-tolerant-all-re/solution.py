"""Task 816 - Implement distributed training system with fault-tolerant all-reduce and elastic workers.  (Tier: Master, CTCI Beyond - Machine Learning & AI)

Data-parallel training of a large model is one collective performed over and over: every worker
computes the gradient of its own shard, an all-reduce averages the gradients across the world, and the
optimiser steps.  The all-reduce here is the bandwidth-optimal ring: a reduce-scatter phase and an
all-gather phase, 2 * (S - 1) steps in which every worker sends exactly 2 * (S - 1) / S of the tensor,
so no worker is ever the bottleneck.  Messages travel over an explicit in-memory network whose cost is
latency + ceil(words / bandwidth) ticks; the schedule is driven by advancing that tick counter, never
by sleeping, threads or a clock, so a run is fully deterministic and reproducible.

Fault tolerance is layered on the same collective:

  * every step has a deadline; a worker whose message arrives after it is declared failed *at exactly
    that tick* (a straggler and a crashed worker are the same event in a synchronous trainer);
  * the collective aborts, the partial accumulators are thrown away, and training resumes from the last
    checkpoint with the surviving members only - so no committed version ever contains a partial
    aggregate;
  * elastic workers are supported: a member that joins later is initialised from the coordinator's
    parameters, and the world size used in the gradient averaging is the membership of that step.

Parameters and gradients are `fractions.Fraction`, so every comparison against the reference is exact
(no float tolerance, no drift): the model is the same arithmetic a serial trainer would run.

Verified: the network's latency/bandwidth model and delivery order, ring all-reduce against a
straightforward elementwise-sum oracle for world sizes 1, 2, 3, 5 and 8, correctness under padding
(a tensor that is not a multiple of the world size) and for an empty and a zero tensor, the measured
per-worker traffic and message count against the closed form 2 * (S - 1) / S * words and 2 * (S - 1),
gradient-sum correctness, K steps of data-parallel SGD agreeing *exactly* with an independent serial
oracle for even and uneven shards, the failure tick matching the deadline exactly, the safety
invariant that every committed parameter version equals the oracle over that step's membership, the
loss-accounting identity (the recovered sum differs from the full sum by exactly the failed worker's
gradient), elastic join equivalence against a static run, straggler eviction, survival when all but
one worker die, determinism of two identical runs, and the makespan bound.

Simplified: one process, no real UDP/NCCL, no floating point (exact rationals instead of bf16/fp32),
bulk-synchronous steps with a fixed per-step deadline rather than asynchronous streaming, one model
tensor and a quadratic loss so the gradients are checkable by hand, and checkpoints taken every step
while a real trainer would checkpoint every few minutes - the invariants verified are the ones a real
elastic trainer must hold at every commit.
"""
from fractions import Fraction

LATENCY_TICKS = 2
BANDWIDTH_WORDS_PER_TICK = 4
STEP_DEADLINE_TICKS = 3


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ------------------------------------------------------------------ the network
class Message:
    __slots__ = ("src", "dst", "payload", "send_tick", "deliver_tick", "sequence")

    def __init__(self, src, dst, payload, send_tick, deliver_tick, sequence):
        self.src = src
        self.dst = dst
        self.payload = payload
        self.send_tick = send_tick
        self.deliver_tick = deliver_tick
        self.sequence = sequence

    def label(self):
        return f"{self.src}->{self.dst}@{self.deliver_tick}"


class Network:
    """In-memory message bus: latency + size/bandwidth ticks, deterministic delivery order."""

    def __init__(self, latency=LATENCY_TICKS, bandwidth=BANDWIDTH_WORDS_PER_TICK):
        self.latency = latency
        self.bandwidth = bandwidth
        self.tick = 0
        self.queue = []
        self.sent_messages = 0
        self.sent_words = 0
        self.dropped = 0
        self.sequence = 0

    def cost(self, words, speed=1):
        return self.latency * speed + -(-words // self.bandwidth)

    def send(self, src, dst, payload, speed=1):
        self.sequence += 1
        deliver = self.tick + self.cost(len(payload), speed)
        message = Message(src, dst, list(payload), self.tick, deliver, self.sequence)
        self.queue.append(message)
        self.sent_messages += 1
        self.sent_words += len(payload)
        return message

    def drain(self, limit=None, events=None):
        """Deliver everything in (deliver_tick, sequence) order; drop anything past `limit` and
        advance the tick to `limit` (the deadline) or to the last delivery."""
        self.queue.sort(key=lambda message: (message.deliver_tick, message.sequence))
        delivered = []
        while self.queue:
            message = self.queue.pop(0)
            if limit is not None and message.deliver_tick > limit:
                self.queue.insert(0, message)
                self.dropped += 1
                if events is not None:
                    events.append((limit, "dropped", message.label()))
                self.tick = max(self.tick, limit)
                return delivered
            self.tick = max(self.tick, message.deliver_tick)
            delivered.append(message)
        if limit is not None:
            self.tick = max(self.tick, limit)
        return delivered


def evict_late(network, deadline, events):
    """Anything still in flight after the deadline belongs to a failed member."""
    late = sorted({message.src for message in network.queue
                   if message.deliver_tick > deadline})
    network.queue = [message for message in network.queue if message.deliver_tick <= deadline]
    for name in late:
        events.append((deadline, "failed", name))
    return set(late)


# ------------------------------------------------------------------ ring all-reduce
def pad_to(tensor, multiple):
    """Zero-pad so the ring can split the tensor into equal chunks (real rings do the same)."""
    remainder = len(tensor) % multiple
    if remainder == 0:
        return list(tensor), 0
    padding = multiple - remainder
    return list(tensor) + [0] * padding, padding


def ring_all_reduce(network, members, tensors, speeds=None, deadline_ticks=None, events=None):
    """Bulk-synchronous ring all-reduce.  Returns (results, failed).

    Each step: every member sends one chunk to its successor and the step waits until the deadline.
    A member whose chunk misses the deadline is failed and the whole collective is aborted (the
    caller then resumes from the last checkpoint with the survivors)."""
    events = events if events is not None else []
    speeds = speeds or {}
    members = list(members)
    size = len(members)
    check(size >= 1, "a collective needs at least one member")
    index_of = {name: index for index, name in enumerate(members)}
    length = len(tensors[members[0]]) if members else 0
    for name in members:
        check(len(tensors[name]) == length, "every member must hold the same tensor shape")
    padded, _ = pad_to(tensors[members[0]], size)
    chunked = {}
    for name in members:
        member_padded, _ = pad_to(tensors[name], size)
        check(len(member_padded) == len(padded),
              "every member must hold the same tensor shape")
        chunked[name] = [member_padded[index * len(padded) // size:
                                       (index + 1) * len(padded) // size]
                         for index in range(size)]
    accumulator = {name: [list(chunk) for chunk in chunked[name]] for name in members}
    deadline_ticks = STEP_DEADLINE_TICKS if deadline_ticks is None else deadline_ticks
    for phase in ("reduce-scatter", "all-gather"):
        for step in range(size - 1):
            deadline = network.tick + deadline_ticks
            for index, name in enumerate(members):
                if phase == "reduce-scatter":
                    chunk_index = (index - step) % size
                else:
                    chunk_index = (index - step + 1) % size
                network.send(name, members[(index + 1) % size], accumulator[name][chunk_index],
                             speeds.get(name, 1))
            delivered = network.drain(limit=deadline)
            failed = evict_late(network, deadline, events)
            if failed:
                events.append((deadline, "abort", phase))
                return None, failed
            for message in delivered:
                if phase == "reduce-scatter":
                    index = (index_of[message.dst] - step - 1) % size
                    existing = accumulator[message.dst][index]
                    accumulator[message.dst][index] = [a + b for a, b in
                                                       zip(existing, message.payload)]
                else:
                    index = (index_of[message.dst] - step) % size
                    accumulator[message.dst][index] = list(message.payload)
            events.append((deadline, f"{phase}-step-{step}", deadline))
    results = {}
    for name in members:
        flat = [value for chunk in accumulator[name] for value in chunk]
        results[name] = flat[:length]
    return results, set()


def all_reduce_sum(members, tensors):
    """Elementwise-sum oracle: the value every member must end up with."""
    length = len(tensors[members[0]])
    return [sum(tensors[name][index] for name in members) for index in range(length)]


# ------------------------------------------------------------------ the trainer
def gradients_from_shard(name, params, shard):
    """grad_i = mean over the shard of 2 * (w - x); an empty shard contributes nothing."""
    if not shard:
        return [Fraction(0) for _ in params]
    return [Fraction(2) * sum((params[index] - point) for point in shard) / len(shard)
            for index in range(len(params))]


class Trainer:
    """One parameter vector, one shard per member, one all-reduce per step."""

    def __init__(self, shards, learning_rate=Fraction(1, 4), latency=LATENCY_TICKS,
                 bandwidth=BANDWIDTH_WORDS_PER_TICK):
        self.shards = dict(shards)
        self.members = sorted(shards)
        self.params = [Fraction(1, 2)] * 2
        self.learning_rate = learning_rate
        self.network = Network(latency, bandwidth)
        self.events = []
        self.history = []
        self.dead = {}
        self.traffic = []

    def active(self):
        return [name for name in self.members if name not in self.dead]

    def step(self, speeds=None, deadline_ticks=STEP_DEADLINE_TICKS):
        """One data-parallel step.  Returns False when the collective aborted."""
        members = self.active()
        if not members:
            raise ValueError("every worker has failed")
        grads = {name: gradients_from_shard(name, self.params, self.shards[name])
                 for name in members}
        before = self.network.sent_words
        reduced, failed = ring_all_reduce(self.network, members, grads, speeds,
                                          deadline_ticks, self.events)
        self.traffic.append(self.network.sent_words - before)
        if failed:
            for name in failed:
                self.dead[name] = self.network.tick
            return False
        scale = Fraction(1, len(members))
        summed = reduced[members[0]]
        for name in members[1:]:
            check(reduced[name] == summed, "all-reduce must leave every member with the same tensor")
        check(summed == all_reduce_sum(members, grads),
              "the collective must equal the elementwise-sum oracle")
        self.params = [value - self.learning_rate * scale * total
                       for value, total in zip(self.params, summed)]
        self.history.append(list(self.params))
        return True

    def run(self, steps, failures=None, joins=None, speeds=None, deadline_ticks=None):
        """Drive `steps` optimiser steps, applying an explicit membership schedule."""
        failures = failures or {}
        joins = joins or {}
        deadline_ticks = STEP_DEADLINE_TICKS if deadline_ticks is None else deadline_ticks
        committed = 0
        attempts = 0
        while committed < steps and attempts < 10 * steps + 10:
            attempts += 1
            for name in sorted(joins.get(committed, [])):
                self.members.append(name)
                self.members.sort()
                self.events.append((self.network.tick, "join", name))
            for name in sorted(failures.get(committed, [])):
                if name in self.active():
                    self.dead[name] = self.network.tick
                    self.events.append((self.network.tick, "crash", name))
            if not self.active():
                raise ValueError("every worker has failed")
            if self.step(speeds, deadline_ticks):
                committed += 1
        check(committed == steps, f"the job stalled after {committed} of {steps} steps")
        return self.params


def serial_oracle(shards, steps, membership, learning_rate=Fraction(1, 4), initial=Fraction(1, 2)):
    """Independent reference: the same schedule evaluated serially, without any network."""
    params = [initial, initial]
    for step in range(steps):
        members = membership(step)
        summed = [Fraction(0), Fraction(0)]
        for name in members:
            grad = gradients_from_shard(name, params, shards[name])
            summed = [total + value for total, value in zip(summed, grad)]
        scale = Fraction(1, len(members))
        params = [value - learning_rate * scale * total
                  for value, total in zip(params, summed)]
    return params


# ----------------------------------------------------------------------- checks
def fixed_membership(*names):
    return lambda step: list(names)


def variable_membership(schedule):
    return lambda step: schedule[min(step, len(schedule) - 1)]


def main():
    # 1. the network model: cost, ordering, monotone ticks
    net = Network()
    net.tick = 5
    first = net.send("A", "B", [1, 2, 3, 4])
    second = net.send("B", "A", [5])
    check(first.deliver_tick == 5 + LATENCY_TICKS + 1 and second.deliver_tick == 5 + LATENCY_TICKS +
          1, "the cost must be latency + ceil(words / bandwidth)")
    check(net.cost(9) == LATENCY_TICKS + 3, "nine words at four words per tick take three ticks")
    delivered = net.drain()
    check([message.sequence for message in delivered] == [1, 2],
          "messages with equal arrival ticks must be delivered in send order")
    check(net.tick == first.deliver_tick and all(message.deliver_tick <= net.tick
                                                 for message in delivered),
          "the tick must advance to the last delivery, never past it")
    check(net.sent_messages == 2 and net.sent_words == 5, "traffic must be counted")
    quiet = Network()
    quiet.send("A", "B", [1])
    check(quiet.drain(limit=quiet.tick) == [] and quiet.dropped == 1,
          "a drain with a limit must drop whatever is still in flight")
    check(Network().cost(0) == LATENCY_TICKS, "an empty payload still costs the latency")

    # 2. ring all-reduce equals the elementwise-sum oracle for many world sizes
    for size in (1, 2, 3, 5, 8):
        members = ["w%d" % index for index in range(size)]
        tensors = {name: [index + 1, (index * 7) % 5, -index] for index, name in enumerate(members)}
        network = Network()
        results, failed = ring_all_reduce(network, members, tensors, events=[])
        check(not failed, f"a healthy ring of {size} must not lose members")
        expected = all_reduce_sum(members, tensors)
        check(all(results[name] == expected for name in members),
              f"every member of a ring of {size} must hold the sum")
        check(network.sent_messages == size * 2 * (size - 1),
              f"a ring of {size} sends 2(S-1) messages per member, got "
              f"{network.sent_messages} total")
        length = len(tensors[members[0]])
        padded_length = length + (-length % size)
        expected_words = 2 * (size - 1) / float(size) * padded_length
        check(abs(network.sent_words / size - expected_words) < 1e-9,
              f"a ring moves 2(S-1)/S of the tensor per member: {network.sent_words / size} vs "
              f"{expected_words}")

    # 3. padding, empty and zero tensors
    members = ["a", "b", "c"]
    ragged = {name: list(range(index + 1, index + 8)) for index, name in enumerate(members)}
    network = Network()
    results, _ = ring_all_reduce(network, members, ragged, events=[])
    check(all(len(results[name]) == 7 for name in members), "padding must not lengthen the result")
    check(results["a"] == all_reduce_sum(members, ragged), "a ragged tensor must still sum exactly")
    zeros = {name: [0, 0, 0] for name in members}
    results, _ = ring_all_reduce(Network(), members, zeros, events=[])
    check(results["b"] == [0, 0, 0], "the all-reduce of zeros must be zeros")
    empty = {name: [] for name in members}
    results, _ = ring_all_reduce(Network(), members, empty, events=[])
    check(results["c"] == [], "an empty tensor must all-reduce to an empty tensor")
    solo = Network()
    results, _ = ring_all_reduce(solo, ["only"], {"only": [4, 5]}, events=[])
    check(results["only"] == [4, 5] and solo.sent_messages == 0,
          "a world of one needs no messages and returns its own tensor")

    # 4. gradient sums: the collective result equals the plain sequential sum
    shards = {"w0": [Fraction(0), Fraction(1), Fraction(3)],
              "w1": [Fraction(-2), Fraction(1)],
              "w2": []}
    params = [Fraction(1, 2), Fraction(-1, 4)]
    grads = {name: gradients_from_shard(name, params, shard) for name, shard in sorted(shards.items())}
    sequential = [sum(grads[name][index] for name in shards) for index in range(2)]
    check(all_reduce_sum(sorted(shards), grads) == sequential,
          "the gradient sum must equal the sequential sum")
    check(grads["w2"] == [Fraction(0), Fraction(0)], "an empty shard must contribute nothing")

    # 5. K steps of data-parallel SGD agree exactly with the serial oracle (even and uneven shards)
    shards = {"w0": [Fraction(0), Fraction(2)], "w1": [Fraction(1), Fraction(-1)],
              "w2": [Fraction(3, 2), Fraction(1, 2)], "w3": [Fraction(-1, 2), Fraction(0)]}
    trainer = Trainer(shards)
    trainer.run(steps=6)
    expected = serial_oracle(shards, 6, fixed_membership(*sorted(shards)))
    check(trainer.params == expected,
          f"six steps must match the serial oracle exactly: {trainer.params} vs {expected}")
    check(len(trainer.history) == 6, "every committed step must be recorded")
    uneven = {"w0": [Fraction(1)] * 5, "w1": [Fraction(0)]}
    thin = Trainer(uneven)
    thin.run(steps=4)
    check(thin.params == serial_oracle(uneven, 4, fixed_membership("w0", "w1")),
          "uneven shards must match the oracle (the average is over members, not over points)")

    # 6. a failure is detected at exactly the deadline, and no partial aggregate is ever committed
    straggler = Trainer(shards)
    alive = straggler.step(speeds={"w2": 4}, deadline_ticks=3)
    check(not alive, "a straggler must abort the collective")
    check(straggler.dead == {"w2": 3},
          f"the failure must be recorded at the deadline tick, got {straggler.dead}")
    check(straggler.network.tick == 3, "the aborted step must stop at the deadline")
    check(straggler.history == [], "an aborted collective must not commit anything")
    check(("abort" in [event[1] for event in straggler.events]),
          "the abort must be part of the event log")
    check(straggler.params == [Fraction(1, 2)] * 2, "the parameters must be untouched")

    # 7. recovery: the resumed run equals the oracle over the surviving members
    recovered = Trainer(shards)
    for step in range(2):
        check(recovered.step(), "the healthy steps must commit")
    committed_before = list(recovered.params)
    check(recovered.step(speeds={"w1": 9}, deadline_ticks=3) is False,
          "the slow member must abort the collective")
    survivors = [name for name in sorted(shards) if name not in recovered.dead]
    check(sorted(survivors) == ["w0", "w2", "w3"], f"the world must shrink, survivors={survivors}")
    check(recovered.params == committed_before,
          "recovery must restart from the last committed checkpoint")
    check(recovered.step(), "the reduced world must make progress")
    check(recovered.step(), "the reduced world must keep making progress")
    check(len(recovered.history) == 4, f"four steps must be committed, got {len(recovered.history)}")
    reduced_oracle = serial_oracle(shards, 4, variable_membership([
        ["w0", "w1", "w2", "w3"], ["w0", "w1", "w2", "w3"], ["w0", "w2", "w3"],
        ["w0", "w2", "w3"]]))
    check(recovered.params == reduced_oracle,
          f"the recovered run must equal the oracle over the survivors: "
          f"{recovered.params} vs {reduced_oracle}")

    # 8. loss accounting: the recovered gradient differs by exactly the failed member's gradient
    failed_grad = gradients_from_shard("w1", committed_before, shards["w1"])
    full = [sum(gradients_from_shard(name, committed_before, shards[name])[index]
                for name in shards) for index in range(2)]
    surviving = [sum(gradients_from_shard(name, committed_before, shards[name])[index]
                     for name in survivors) for index in range(2)]
    check([a - b for a, b in zip(full, surviving)] == failed_grad,
          "the lost contribution must be exactly the failed member's gradient")
    check(failed_grad != [Fraction(0), Fraction(0)], "the accounting check must be non-trivial")

    # 9. elastic join: a member that joins later matches a static run with the same schedule
    schedule = [["w0", "w1"], ["w0", "w1"], ["w0", "w1", "w2"], ["w0", "w1", "w2"]]
    elastic = Trainer({"w0": shards["w0"], "w1": shards["w1"]})
    elastic.shards["w2"] = shards["w2"]
    elastic.run(steps=4, joins={2: ["w2"]})
    check(elastic.params == serial_oracle({"w0": shards["w0"], "w1": shards["w1"],
                                           "w2": shards["w2"]}, 4, variable_membership(schedule)),
          "an elastic join must reproduce the oracle for that membership schedule")
    check(("join" in [event[1] for event in elastic.events]), "the join must be logged")

    # 10. a crash is handled the same way as a straggler, and the world may shrink to one
    crashy = Trainer(shards)
    crashy.run(steps=4, failures={1: ["w1", "w2", "w3"]})
    check(crashy.params == serial_oracle(shards, 4, variable_membership([
        ["w0", "w1", "w2", "w3"], ["w0"], ["w0"], ["w0"]])),
        "a job that loses all but one worker must still finish with the right value")
    check(len(crashy.dead) == 3 and crashy.active() == ["w0"], "the world must shrink to one")
    doomed = Trainer(shards)
    try:
        doomed.run(steps=2, failures={0: sorted(shards)})
        check(False, "a job whose whole world dies must report failure")
    except ValueError:
        pass

    # 11. determinism and the makespan bound
    first_run = Trainer(shards)
    first_run.run(steps=5)
    second_run = Trainer(shards)
    second_run.run(steps=5)
    check(first_run.events == second_run.events, "two identical runs must produce identical events")
    check(first_run.params == second_run.params and first_run.traffic == second_run.traffic,
          "two identical runs must move the same bytes and commit the same parameters")
    ticks = first_run.network.tick
    per_step = 2 * (len(shards) - 1) * STEP_DEADLINE_TICKS
    check(ticks == 5 * per_step,
          f"the makespan must be exactly steps * 2(S-1) * deadline: {ticks} vs {5 * per_step}")
    check(first_run.traffic == [2 * (len(shards) - 1) * 4] * 5,
          f"every step must move 2(S-1)/S of the tensor per member: {first_run.traffic}")

    # 12. cost is independent of how much data a worker holds
    small_data = Trainer({"w0": [Fraction(0)], "w1": [Fraction(1)]})
    small_data.run(steps=3)
    large_data = Trainer({"w0": [Fraction(index, 7) for index in range(50)],
                          "w1": [Fraction(-index, 5) for index in range(50)]})
    large_data.run(steps=3)
    check(small_data.traffic == large_data.traffic,
          f"the traffic must depend on the tensor size only: {small_data.traffic} vs "
          f"{large_data.traffic}")
    check(small_data.traffic == [2 * (2 - 1) * 2] * 3,
          f"the traffic must match 2(S-1)/S per member per step, got {small_data.traffic}")

    print(f"all-reduce: world sizes 1,2,3,5,8 verified; traffic per member = 2(S-1)/S * words")
    print(f"training: 6 steps exact vs serial oracle; failures caught at tick {3}; "
          f"elastic join verified; makespan={ticks} ticks")
    print("PASS")


if __name__ == "__main__":
    main()
