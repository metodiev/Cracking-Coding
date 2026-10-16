"""Task 723 - Implement automatic batching (microbatch merging) skeleton.  (Tier: Expert, CTCI Beyond — Machine Learning & AI)

An inference/ training server that executes one request at a time wastes its device: the fixed
per-kernel cost (launch, weight fetch, memory traffic setup) is paid over and over.  Automatic
batching keeps arriving requests in a pending queue and dispatches them as one *microbatch*,
so the fixed cost is amortised over every item in the batch, at the price of letting each
request wait.

The scheduler implemented here is the standard dynamic batcher: a batch is dispatched as soon
as either the pending queue reaches `max_batch` items (a size-triggered flush, immediate) or
the oldest pending request has waited `max_delay` (a timeout flush at
first_pending_arrival + max_delay).  Execution cost is `base_cost + per_item_cost * size`, so
the per-item cost falls with batch size while the latency of every member rises - that
tradeoff is measured, not assumed.

The batched workload is real per-item work (each request carries a payload and the microbatch
returns payload^2 for every item, which must be routed back to the right request), so the
checks can verify results as well as bookkeeping.  Verified invariants: every request is served
exactly once with the correct result and in arrival order, no batch exceeds max_batch, each
request waits at most max_delay before dispatch, size-triggered flushes happen exactly when the
queue fills, timeout flushes happen exactly at the deadline, and batching lowers the total
cost (and raises the mean latency) relative to one-request-per-batch.

Simplified versus a production batcher (Triton dynamic batching, TensorFlow Serving): a
deterministic single-queue, single-device model with no queueing across devices, no
priority or fairness policy, no per-request deadlines or cancellation, no padded/ragged
tensor construction, and no concurrency - the batch is executed synchronously in the cost
model rather than on a device.  The scheduling logic, the flush rules and the measured cost
and latency tradeoff are real.
"""
import math


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


class MicroBatcher:
    """Merge arrivals into microbatches under a size cap and a queue-latency cap."""

    def __init__(self, max_batch, max_delay, base_cost, per_item_cost):
        if max_batch < 1:
            raise ValueError("max_batch must be at least one")
        if max_delay < 0.0:
            raise ValueError("max_delay must be non-negative")
        if base_cost < 0.0 or per_item_cost < 0.0:
            raise ValueError("costs must be non-negative")
        self.max_batch = max_batch
        self.max_delay = max_delay
        self.base_cost = base_cost
        self.per_item_cost = per_item_cost
        self.batches = []          # (batch id, reason, dispatch time, size, completion time)
        self.completions = []      # (request id, arrival, completion, latency, result)
        self.cost = 0.0

    def flush(self, pending, dispatch_time, reason):
        """Execute one microbatch: real per-item work, routed back to the request ids."""
        size = len(pending)
        if size == 0:
            raise ValueError("an empty microbatch must never be dispatched")
        duration = self.base_cost + self.per_item_cost * size
        completion = dispatch_time + duration
        batch_id = len(self.batches)
        for rid, arrival, payload in pending:
            self.completions.append((rid, arrival, completion, completion - arrival,
                                     payload * payload, size, duration))
        self.batches.append((batch_id, reason, dispatch_time, size, completion))
        self.cost += duration
        return completion

    def run(self, requests):
        """`requests` is a list of (request id, arrival time, payload) in arrival order."""
        pending = []
        clock_events = []
        for rid, arrival, payload in requests:
            if pending and arrival >= pending[0][1] + self.max_delay:
                deadline = pending[0][1] + self.max_delay
                clock_events.append((deadline, "timeout"))
                self.flush(pending, deadline, "timeout")
                pending = []
            pending.append((rid, arrival, payload))
            if len(pending) == self.max_batch:
                clock_events.append((arrival, "size"))
                self.flush(pending, arrival, "size")
                pending = []
        if pending:
            deadline = pending[0][1] + self.max_delay
            clock_events.append((deadline, "timeout"))
            self.flush(pending, deadline, "timeout")
        return self

    def mean_latency(self):
        if not self.completions:
            return 0.0
        return sum(entry[3] for entry in self.completions) / len(self.completions)

    def mean_batch_size(self):
        if not self.batches:
            return 0.0
        return sum(entry[3] for entry in self.batches) / len(self.batches)

    def cost_per_item(self):
        return self.cost / max(len(self.completions), 1)


def make_stream(payloads, gap, burst=1, idle_every=0, idle=0.0):
    """Deterministic arrival stream: `burst` requests share a timestamp, then a gap; every
    `idle_every` group the clock also jumps by `idle`, which is how idle periods are modelled."""
    requests = []
    clock = 0.0
    for index, payload in enumerate(payloads):
        if index and index % burst == 0:
            clock += gap
            if idle_every and (index // burst) % idle_every == 0:
                clock += idle
        requests.append((index, round(clock, 6), payload))
    return requests


def oracle_results(requests):
    return {rid: payload * payload for rid, _, payload in requests}


def main():
    payloads = [0.5 * (i % 9) + 0.25 for i in range(200)]

    # 1. serving correctness: every request exactly once, in arrival order, with the right
    #    result - the microbatch must route each output back to its own request.
    stream = make_stream(payloads, 0.05, burst=2, idle_every=5, idle=0.6)
    batcher = MicroBatcher(max_batch=8, max_delay=0.2, base_cost=1.0, per_item_cost=0.05).run(stream)
    served = [entry[0] for entry in batcher.completions]
    check(served == [rid for rid, _, _ in stream], "every request must be served once, in order")
    check({entry[0]: entry[4] for entry in batcher.completions} == oracle_results(stream),
          "each request must receive the result of its own payload")

    # 2. batch structure: sizes are within the cap, batches are never empty, and the batch log
    #    covers every request exactly once.
    check(all(1 <= entry[3] <= 8 for entry in batcher.batches), "batch sizes must respect the cap")
    check(sum(entry[3] for entry in batcher.batches) == len(stream), "batches must cover the stream")
    check(len(batcher.batches) == len({entry[0] for entry in batcher.batches}),
          "batch ids must be unique")
    check(sum(1 for entry in batcher.batches if entry[3] == 8) > 0,
          "a bursty stream must produce full batches")

    # 3. flush rules: a size trigger dispatches at the arrival instant, a timeout trigger at
    #    exactly first_pending_arrival + max_delay, and both are used here.
    arrivals = {rid: arrival for rid, arrival, _ in stream}
    for batch_id, reason, dispatch, size, completion in batcher.batches:
        check(reason in ("size", "timeout"), "the flush reason must be known")
        check(completion >= dispatch, "a batch cannot complete before it is dispatched")
        check(abs(completion - dispatch - (1.0 + 0.05 * size)) < 1e-12,
              "batch duration must follow the base plus per-item cost model")
        if reason == "size":
            check(size == 8 and dispatch in arrivals.values(),
                  "a size flush must happen at the arrival that filled the queue")
    check(any(entry[1] == "timeout" for entry in batcher.batches), "timeouts must be exercised")

    # 4. the latency guarantee: the wait before dispatch never exceeds max_delay, hence a
    #    request's latency is bounded by max_delay plus the duration of its own batch.
    for rid, arrival, completion, latency, result, size, duration in batcher.completions:
        check(completion - duration - arrival <= 0.2 + 1e-9,
              "a request may not wait longer than max_delay before dispatch")
        check(abs(latency - (completion - arrival)) < 1e-12,
              "latency must be completion minus arrival")
        check(abs(duration - (1.0 + 0.05 * size)) < 1e-12, "the duration model must hold")
        check(latency <= 0.2 + duration + 1e-9,
              "latency must be bounded by max_delay plus the batch duration")
        check(abs(result - oracle_results(stream)[rid]) < 1e-12, "the result must be correct")
    check(batcher.mean_latency() > 0.2, "batching must actually make requests wait")
    check(max(entry[5] for entry in batcher.completions) == 8,
          "the widest batch in the log must be the cap")

    # 5. the tradeoff: batching cuts the total cost but raises the mean latency; the per-item
    #    cost is strictly lower than executing one request at a time.
    solo = MicroBatcher(max_batch=1, max_delay=0.2, base_cost=1.0, per_item_cost=0.05).run(stream)
    check(solo.mean_latency() <= batcher.mean_latency(),
          "unbatched execution must have the lower mean latency")
    check(batcher.cost < solo.cost, "batching must reduce the total cost")
    check(batcher.cost_per_item() < solo.cost_per_item() / 2,
          "batched execution must cost less than half per item")
    check(abs(solo.cost - len(stream) * 1.05) < 1e-9, "unbatched cost must be the model, exactly")

    # 6. batch size effect: larger caps mean larger batches, cheaper items and higher latency.
    curve = []
    for cap in (1, 2, 4, 8, 16, 32):
        run = MicroBatcher(cap, 0.2, 1.0, 0.05).run(stream)
        curve.append((cap, run.mean_batch_size(), run.cost_per_item(), run.mean_latency()))
    check(all(curve[i][1] <= curve[i + 1][1] + 1e-9 for i in range(len(curve) - 1)),
          "mean batch size must grow with the cap")
    check(all(curve[i][2] >= curve[i + 1][2] - 1e-9 for i in range(len(curve) - 1)),
          "cost per item must fall as the cap grows")
    check(all(curve[i][3] <= curve[i + 1][3] + 1e-9 for i in range(len(curve) - 1)),
          "mean latency must grow as the cap grows")

    # 7. boundary cases: a single request, an empty stream, a cap of one, and an arrival that
    #    lands exactly on the deadline (the pending batch must flush first, deterministically).
    single = MicroBatcher(8, 0.5, 1.0, 0.1).run([(0, 1.0, 2.0)])
    check(single.batches == [(0, "timeout", 1.5, 1, 2.6)], "a lone request waits for the timeout")
    check(single.completions[0][4] == 4.0, "a lone request must still be computed")
    empty = MicroBatcher(4, 1.0, 1.0, 0.1).run([])
    check(empty.batches == [] and empty.completions == [], "an empty stream must do nothing")
    check(empty.mean_latency() == 0.0 and empty.cost == 0.0, "an empty stream must cost nothing")
    one = MicroBatcher(1, 5.0, 1.0, 0.1).run(stream)
    check(all(entry[3] == 1 for entry in one.batches), "a cap of one means no merging at all")
    check(all(abs(entry[2] - entry[6] - arrivals[entry[0]]) < 1e-12 for entry in one.completions),
          "with a cap of one every request dispatches immediately")
    edge = MicroBatcher(4, 0.4, 1.0, 0.1).run([(0, 0.0, 1.0), (1, 0.4, 2.0), (2, 0.4, 3.0)])
    check(edge.batches[0] == (0, "timeout", 0.4, 1, 1.5),
          "an arrival exactly at the deadline must not join the expiring batch")
    check(edge.batches[1][3] == 2, "the late arrivals must batch together")

    # 8. pattern sensitivity: bursts batch better than a spread-out stream with identical
    #    payloads, and the cost difference is what a server operator would tune for.
    spread = make_stream(payloads, 0.5, burst=1)
    bursty = MicroBatcher(8, 0.2, 1.0, 0.05).run(make_stream(payloads, 0.05, burst=8))
    sparse = MicroBatcher(8, 0.2, 1.0, 0.05).run(spread)
    check(bursty.mean_batch_size() > sparse.mean_batch_size(),
          "bursts must merge into larger batches than a spread stream")
    check(bursty.cost_per_item() < sparse.cost_per_item(),
          "and therefore cost less per item")
    check({entry[0]: entry[4] for entry in sparse.completions} == oracle_results(spread),
          "the spread stream must be served correctly too")

    # 9. rejected configurations and determinism.
    expect_value_error(lambda: MicroBatcher(0, 1.0, 1.0, 0.1), "max_batch 0 is invalid")
    expect_value_error(lambda: MicroBatcher(4, -1.0, 1.0, 0.1), "a negative delay is invalid")
    expect_value_error(lambda: MicroBatcher(4, 1.0, -1.0, 0.1), "a negative cost is invalid")
    left = MicroBatcher(8, 0.2, 1.0, 0.05).run(stream)
    right = MicroBatcher(8, 0.2, 1.0, 0.05).run(stream)
    check(left.batches == right.batches and left.completions == right.completions,
          "the same stream must produce the same batches and completions")

    check(math.isclose(batcher.mean_batch_size(), sum(e[3] for e in batcher.batches) / len(batcher.batches)),
          "mean batch size must equal the batch log average")
    print(f"723 stream={len(stream)} batches={len(batcher.batches)} "
          f"mean_size={batcher.mean_batch_size():.2f} mean_latency={batcher.mean_latency():.3f}")
    print(f"723 cost batched={batcher.cost:.2f} unbatched={solo.cost:.2f} "
          f"per_item={batcher.cost_per_item():.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
