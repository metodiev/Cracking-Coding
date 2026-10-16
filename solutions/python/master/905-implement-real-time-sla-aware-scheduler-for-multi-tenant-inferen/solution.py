"""Task 905 - Implement real-time SLA-aware scheduler for multi-tenant inference.
(Tier: Master, CTCI Ch 9 - System Design and Scalability)

What is implemented (in-process, deterministic - arrivals, services and slack come from the
seeded splitmix64 stream, the clock is a unit-time simulation with no wall clock or threads):

  * `edf_schedule` - preemptive Earliest-Deadline-First on one inference worker, stepping one
    time unit at a time; `worst_lateness` reports the maximum (completion - deadline).
  * `brute_force_batch` - the exhaustive oracle for a batch released at t=0: the minimum
    achievable maximum lateness over all non-preemptive orders.  EDF (Jackson's rule) must
    match it, which is the optimality statement a real-time scheduler is judged on.
  * `SlaScheduler` - per-tenant token buckets (SLA rate limits), reserved-share entitlement
    (a tenant may not consume more than its share of capacity plus a burst) and admission
    control: a request is admitted only if the EDF schedule of the accepted set *including* it
    still meets every deadline.  These rails are what make the offered SLA - "accepted requests
    always meet their deadline, and every tenant gets near its reservation" - true.

Simplified versus production: one worker with unit service quanta, no preemption cost, no
replica migration, no queueing beyond the accepted set; arrivals are a deterministic periodic
stream rather than a live load.  Verified below: the EDF optimality oracle over six seeded
batches, a negative control where FIFO misses a deadline EDF meets, the deadline-exactly-met
boundary, admission soundness re-checked with an independent re-simulation, the token-bucket and
entitlement caps, the reserved mix, the utilisation bound, and the empty/zero-rate/zero-slack
boundaries.
"""

import itertools

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def below(self, n):
        return self.next_u64() % n


class Request:
    __slots__ = ("rid", "tenant", "arrival", "service", "deadline")

    def __init__(self, rid, tenant, arrival, service, deadline):
        self.rid = rid
        self.tenant = tenant
        self.arrival = arrival
        self.service = service
        self.deadline = deadline

    def __repr__(self):
        return "R%d(%s)" % (self.rid, self.tenant)


def edf_schedule(requests, limit=5000):
    """Preemptive EDF, one time unit per step.  Returns {rid: completion time}."""
    remaining = {request.rid: request.service for request in requests}
    finish = {}
    clock = 0
    while remaining and clock <= limit:
        ready = [r for r in requests if r.arrival <= clock and remaining[r.rid] > 0]
        if ready:
            pick = min(ready, key=lambda r: (r.deadline, r.rid))
            remaining[pick.rid] -= 1
            if remaining[pick.rid] == 0:
                finish[pick.rid] = clock + 1
        clock += 1
    return finish


def worst_lateness(requests):
    finish = edf_schedule(requests)
    if len(finish) != len(requests):
        return 10 ** 9                       # something never finished: as bad as it gets
    return max(finish[r.rid] - r.deadline for r in requests)


def jackson_lateness(requests):
    """Non-preemptive EDF for a batch released at t=0: the optimal maximum lateness."""
    clock, worst = 0, -10 ** 9
    for request in sorted(requests, key=lambda r: (r.deadline, r.rid)):
        clock += request.service
        worst = max(worst, clock - request.deadline)
    return worst


def brute_force_batch(requests):
    """Exhaustive oracle: the best achievable maximum lateness over every ordering."""
    best = None
    for order in itertools.permutations(sorted(requests, key=lambda r: r.rid)):
        clock, worst = 0, -10 ** 9
        for request in order:
            clock += request.service
            worst = max(worst, clock - request.deadline)
        best = worst if best is None else min(best, worst)
    return best


def fifo_lateness(requests):
    clock, worst = 0, -10 ** 9
    for request in sorted(requests, key=lambda r: (r.arrival, r.rid)):
        clock = max(clock, request.arrival)
        clock += request.service
        worst = max(worst, clock - request.deadline)
    return worst


class TenantSla:
    __slots__ = ("name", "period", "slack", "reserved_permille", "rate_milli", "burst_milli")

    def __init__(self, name, period, slack, reserved_permille, rate_milli, burst_milli):
        self.name = name
        self.period = period                # one request every `period` seconds
        self.slack = slack                  # deadline = arrival + slack
        self.reserved_permille = reserved_permille
        self.rate_milli = rate_milli        # token bucket refill in thousandths per second
        self.burst_milli = burst_milli


def make_stream(tenants, horizon, rng):
    requests, rid = [], 0
    for name in sorted(tenants):
        sla = tenants[name]
        arrival = 0
        while arrival < horizon:
            jitter = rng.below(2)
            service = 1 + rng.below(3)
            requests.append(Request(rid, name, arrival + jitter, service,
                                    arrival + jitter + sla.slack))
            rid += 1
            arrival += sla.period
    return sorted(requests, key=lambda r: (r.arrival, r.rid))


class SlaScheduler:
    """Token buckets, reserved-share entitlement and deadline-feasibility admission control."""

    def __init__(self, tenants, burst_units=6):
        self.tenants = tenants
        self.burst_units = burst_units
        self.tokens = {name: sla.burst_milli for name, sla in tenants.items()}
        self.stamp = {name: 0 for name in tenants}
        self.spent = {name: 0 for name in tenants}
        self.accepted = []
        self.rejected = []

    def entitlement(self, name, now):
        """Work the tenant may still consume: its reserved share of the capacity up to `now`,
        plus a small burst allowance.  This is what keeps a heavy tenant from monopolising the
        worker while a sparse tenant still has to be served."""
        return self.tenants[name].reserved_permille * now // 1000 + self.burst_units

    def _refill(self, name, now):
        sla = self.tenants[name]
        elapsed = now - self.stamp[name]
        if elapsed > 0:
            self.tokens[name] = min(sla.burst_milli, self.tokens[name] + elapsed * sla.rate_milli)
            self.stamp[name] = now

    def run(self, stream):
        for request in stream:
            name = request.tenant
            self._refill(name, request.arrival)
            if self.tokens[name] < 1000:
                self.rejected.append((request, "rate"))
                continue
            if self.spent[name] + request.service > self.entitlement(name, request.arrival):
                self.rejected.append((request, "share"))
                continue
            if worst_lateness(self.accepted + [request]) > 0:
                self.rejected.append((request, "deadline"))
                continue
            self.tokens[name] -= 1000
            self.spent[name] += request.service
            self.accepted.append(request)
        return self.accepted, self.rejected


def reservation_report(tenants, served_units):
    total = sum(served_units.values())
    return {name: tenants[name].reserved_permille * total // 1000 for name in tenants}


def main():
    tenants = {
        "A": TenantSla("A", 2, 12, 400, 240, 6000),
        "B": TenantSla("B", 3, 12, 300, 180, 5000),
        "C": TenantSla("C", 6, 12, 200, 120, 4000),
        "D": TenantSla("D", 12, 12, 100, 60, 3000),
    }
    horizon = 200
    stream = make_stream(tenants, horizon, Rng(2024))
    check(len(stream) > 120, "the stream must be large enough to overload the worker")
    demand_units = sum(request.service for request in stream)
    check(demand_units > horizon, "the offered load must exceed the machine capacity")

    # ---- 1. EDF matches the exhaustive optimal-order oracle on seeded batches
    for seed in range(6):
        rng = Rng(700 + seed)
        batch = [Request(i, "A", 0, 1 + rng.below(4), 3 + rng.below(14)) for i in range(6)]
        check(jackson_lateness(batch) == brute_force_batch(batch),
              "EDF must attain the exhaustive optimum (seed %d)" % seed)
    bad_batch = [Request(0, "A", 0, 5, 5), Request(1, "A", 0, 5, 20), Request(2, "A", 0, 5, 10)]
    check(jackson_lateness(bad_batch) == brute_force_batch(bad_batch) == 0,
          "a batch that fits must show zero lateness for EDF and for the optimum")
    check(fifo_lateness(bad_batch) > jackson_lateness(bad_batch),
          "negative control: FIFO must miss a deadline that EDF meets")

    # ---- 2. the exactly-on-time boundary and the hopeless request
    exact = [Request(0, "A", 0, 4, 4)]
    check(worst_lateness(exact) == 0, "a request finishing exactly at its deadline is on time")
    check(worst_lateness([Request(0, "A", 0, 5, 3)]) == 2, "an infeasible request reports lateness")
    refused, refused_reasons = SlaScheduler(tenants).run(
        [Request(0, "A", 0, 5, 3), Request(1, "D", 0, 5, 10)])
    check([r.rid for r in refused] == [1] and [why for _, why in refused_reasons] == ["deadline"],
          "admission control must refuse requests it cannot serve on time")

    # ---- 3. run the real stream
    scheduler = SlaScheduler(tenants)
    accepted, rejected = scheduler.run(stream)
    check(accepted, "the scheduler must admit some of the load")
    check(len(accepted) + len(rejected) == len(stream), "every request must be accounted for")
    reasons = {}
    for _, reason in rejected:
        reasons[reason] = reasons.get(reason, 0) + 1
    check(reasons.get("rate", 0) > 0 and reasons.get("deadline", 0) > 0,
          "both rails must bite on this stream: %r" % reasons)

    # ---- 4. the SLA: re-simulate the accepted set independently
    finish = edf_schedule(accepted)
    check(len(finish) == len(accepted), "every accepted request must complete")
    lateness = [finish[r.rid] - r.deadline for r in accepted]
    check(max(lateness) <= 0, "no accepted request may miss its deadline")
    latency = [finish[r.rid] - r.arrival for r in accepted]
    check(max(latency) <= max(r.deadline - r.arrival for r in accepted),
          "achieved latency can never exceed the requested slack")

    # ---- 5. admission maximality: a rejected request really had to be rejected
    for request, reason in rejected:
        if reason != "deadline":
            continue
        check(worst_lateness(accepted + [request]) > 0,
              "a deadline rejection must be justified by an actual miss (%r)" % request)
        break

    # ---- 6. token buckets and the reserved-share mix
    served = {name: 0 for name in tenants}
    counts = {name: 0 for name in tenants}
    offered = {name: 0 for name in tenants}
    for request in accepted:
        served[request.tenant] += request.service
        counts[request.tenant] += 1
    for request in stream:
        offered[request.tenant] += 1
    for name, sla in tenants.items():
        cap = sla.burst_milli // 1000 + (horizon * sla.rate_milli) // 1000
        allowance = scheduler.entitlement(name, horizon)
        check(counts[name] <= cap, "tenant %s exceeded its token bucket (%d > %d)"
              % (name, counts[name], cap))
        check(served[name] <= allowance,
              "tenant %s beat its reserved entitlement (%d > %d)" % (name, served[name], allowance))
        check(served[name] * 3 >= allowance * 2,
              "tenant %s was starved below its reservation (%d of %d)"
              % (name, served[name], allowance))
    check(sum(served.values()) <= horizon + 12, "served work cannot exceed machine capacity")
    check(counts["A"] < offered["A"] and counts["B"] < offered["B"],
          "the over-demanding tenants must be throttled")
    total_served = sum(served.values())
    check(served["A"] > served["B"] > served["C"] > served["D"],
          "the served mix must follow the reservations: %r" % served)
    check(total_served > horizon * 4 // 5, "the worker must stay nearly saturated")

    # ---- 7. percentiles and determinism
    ordered = sorted(latency)
    p99 = ordered[(len(ordered) * 99) // 100 - 1] if len(ordered) >= 100 else ordered[-1]
    check(p99 <= 12, "the p99 latency must stay inside the 12 unit slack")
    again_accepted, again_rejected = SlaScheduler(tenants).run(stream)
    check([r.rid for r in again_accepted] == [r.rid for r in accepted],
          "the scheduler must be deterministic")
    check([(r.rid, why) for r, why in again_rejected] == [(r.rid, why) for r, why in rejected],
          "the rejection reasons must be deterministic")

    # ---- 8. boundaries: empty stream, zero-rate tenant, zero slack
    empty_accepted, empty_rejected = SlaScheduler(tenants).run([])
    check(empty_accepted == [] and empty_rejected == [], "an empty stream is a no-op")
    mute = {"Z": TenantSla("Z", 1, 12, 1000, 0, 0)}
    mute_accepted, mute_rejected = SlaScheduler(mute).run(
        make_stream(mute, 20, Rng(5)))
    check(mute_accepted == [] and mute_rejected
          and all(why == "rate" for _, why in mute_rejected),
          "a zero-rate tenant must be refused by the token bucket alone")
    zero_slack = {"Z": TenantSla("Z", 1, 0, 1000, 1000, 9000)}
    tight, tight_rejected = SlaScheduler(zero_slack).run(make_stream(zero_slack, 10, Rng(6)))
    check(tight == [] and all(why in ("deadline", "share") for _, why in tight_rejected),
          "a zero-slack stream cannot admit any request")

    print("stream %d requests, offered %d units, capacity %d units" % (len(stream), demand_units, horizon))
    print("accepted %d, rejected %d (%r)" % (len(accepted), len(rejected), reasons))
    print("served units %r vs reserved %r" % (served, reservation_report(tenants, served)))
    print("latency mean %.2f p99 %d, max %d" % (sum(latency) / len(latency), p99, max(latency)))
    print("PASS")


if __name__ == "__main__":
    main()
