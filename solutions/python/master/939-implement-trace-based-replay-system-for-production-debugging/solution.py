"""Task 939 - Implement trace-based replay system for production debugging.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A deterministic trace recorder and replay engine.  A request fans out over a microservice call tree;
every span records its service, the digest of its input, the digest of the payload it produced, its
parent and its start tick.  The recorder builds such a trace by executing the services, and the
replayer re-executes the recorded call structure against a (possibly modified) service map, feeding
each span either the *recorded* input - which isolates a change to one service and keeps the demo
deterministic, exactly like a replay against recorded payloads does - or, in recompute mode, the
input derived from the replayed parent payload, which is how a code change propagates to downstream
spans.  Every payload difference yields a divergence record with the span id, service, recorded and
replayed payload, sorted by start tick then id.

Simplified versus a real tracer (Jaeger/tcpdump-style replay): payloads are 32-bit digests instead of
bytes, clocks are integer ticks and there is no sampling, retry or async continuation.  Verified: a
fault-free replay is identical to the recording; an isolated fault diverges exactly at the modified
span; in recompute mode the divergent set equals the descendants closure of that span computed by an
independent BFS over the parent links (and sibling branches stay clean); a conditional fault diverges
only inside that closure; empty, single-span and orphaned traces are tolerated; every recorded span
is replayed exactly once; and the engine is deterministic.
"""
SERVICES = ("gateway", "orders", "payments", "inventory")
CHILDREN = {"gateway": ("orders", "inventory"), "orders": ("payments", "inventory"),
            "payments": (), "inventory": ()}
SALT = {"gateway": 11, "orders": 23, "payments": 37, "inventory": 41}
MODULUS = 1_000_003


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Span:
    __slots__ = ("span_id", "parent", "service", "input", "payload", "start", "end")

    def __init__(self, span_id, parent, service, input_digest, payload, start, end):
        self.span_id = span_id
        self.parent = parent
        self.service = service
        self.input = input_digest
        self.payload = payload
        self.start = start
        self.end = end

    def key(self):
        return (self.start, self.span_id)


def default_services():
    """The service bodies: injective transforms, so a changed input always changes the payload."""
    return {name: (lambda digest, name=name: (digest * 3 + SALT[name] * 7 + 1) % MODULUS)
            for name in SERVICES}


def faulted_services(target, only=None):
    """Copy of the service map with `target` perturbed (optionally only for some inputs)."""
    services = default_services()
    original = services[target]

    def perturbed(digest):
        if only is not None and not only(digest):
            return original(digest)
        return (original(digest) + 1) % MODULUS

    services[target] = perturbed
    return services


def record(services, root_input=1337):
    """Execute the call tree and return the recorded spans in deterministic (DFS) order."""
    spans = []

    def walk(service, parent_id, input_digest, start):
        payload = services[service](input_digest)
        span_id = "s%d" % len(spans)
        spans.append(Span(span_id, parent_id, service, input_digest, payload, start, start + 1))
        for offset, child in enumerate(CHILDREN[service]):
            walk(child, span_id, (payload + SALT[child]) % MODULUS, start + offset + 1)
        return payload

    walk("gateway", None, root_input, 0)
    return spans


def descendants(spans, span_id):
    """Independent oracle: BFS over the recorded parent links."""
    children = {}
    for span in spans:
        children.setdefault(span.parent, []).append(span.span_id)
    found, queue = set(), [span_id]
    while queue:
        for child in children.get(queue.pop(), ()):
            if child not in found:
                found.add(child)
                queue.append(child)
    return found


def replay(spans, services, recompute=False):
    """Re-execute a recorded trace; returns (replayed spans, divergences, visits, orphans)."""
    recorded = {span.span_id: span for span in spans}
    orphans = [span.span_id for span in spans
               if span.parent is not None and span.parent not in recorded]
    replayed, divergences, visits = {}, [], 0
    for span in spans:
        parent = replayed.get(span.parent)
        if recompute and parent is not None:
            input_digest = (parent.payload + SALT[span.service]) % MODULUS
        else:
            input_digest = span.input
        payload = services[span.service](input_digest)
        visits += 1
        replayed[span.span_id] = Span(span.span_id, span.parent, span.service, input_digest,
                                      payload, span.start, span.end)
        if payload != span.payload:
            divergences.append((span.span_id, span.service, span.payload, payload))
    divergences.sort(key=lambda entry: recorded[entry[0]].key())
    return replayed, divergences, visits, orphans


def divergent_ids(divergences):
    return tuple(entry[0] for entry in divergences)


def main():
    services = default_services()
    spans = record(services)
    # The trace: gateway with an orders subtree (payments) and an inventory leaf.
    by_service = {span.span_id: span.service for span in spans}
    orders_id = next(span.span_id for span in spans if span.service == "orders")
    payments_id = next(span.span_id for span in spans if span.service == "payments")

    # 1) happy path: a fault-free replay reproduces the recording in both modes.
    for recompute in (False, True):
        replayed, divergences, visits, orphans = replay(spans, services, recompute)
        check(divergences == [] and orphans == [] and visits == len(spans)
              and len(replayed) == len(spans)
              and all(replayed[span.span_id].payload == span.payload for span in spans),
              f"clean replay (recompute={recompute}) must match the recording")

    # 2) an isolated fault diverges at exactly one span, and nothing else moves.
    broken_payments = faulted_services("payments")
    _, isolated_div, visits, _ = replay(spans, broken_payments, recompute=False)
    check(divergent_ids(isolated_div) == (payments_id,)
          and isolated_div[0][1] == "payments" and isolated_div[0][2] != isolated_div[0][3]
          and visits == len(spans),
          f"an isolated change must diverge only at the payments span: {isolated_div}")

    # 3) recompute mode: the divergent set is the independent descendants closure of the fault.
    cascade = descendants(spans, orders_id) | {orders_id}
    _, divergences, _, _ = replay(spans, faulted_services("orders"), recompute=True)
    check(set(divergent_ids(divergences)) == cascade and len(divergences) == len(cascade),
          f"a changed orders span must cascade to its closure {sorted(cascade)}: "
          f"{divergent_ids(divergences)}")

    # 4) everything outside the cascade replays identically, and the cascade is the subtree only.
    replayed, _, _, _ = replay(spans, faulted_services("orders"), recompute=True)
    outside = [span for span in spans if span.span_id not in cascade]
    check(outside and all(replayed[span.span_id].payload == span.payload for span in outside),
          "an unrelated branch must replay identically")
    check(all(by_service[span_id] in ("orders", "payments", "inventory") for span_id in cascade),
          f"the closure of orders must only contain its own subtree: {cascade}")

    # 5) conditional fault: only the spans reached by the perturbed input diverge.
    conditional = faulted_services("payments", only=lambda digest: digest % 2 == 0)
    _, divergences, _, _ = replay(spans, conditional, recompute=True)
    inside = {payments_id} | descendants(spans, payments_id)
    check(set(divergent_ids(divergences)) <= inside
          and (not divergences or set(divergent_ids(divergences)) == {payments_id}),
          f"a conditional fault must stay inside the payments span: {divergent_ids(divergences)}")

    # 6) boundaries: an empty trace, a truncated single-span trace and an orphaned span.
    check(replay([], services) == ({}, [], 0, []), "an empty trace replays to nothing")
    single = [spans[0]]
    replayed, divergences, visits, orphans = replay(single, services)
    check(divergences == [] and visits == 1 and orphans == [] and replayed["s0"].payload
          == single[0].payload,
          "a trace with only the root span must replay on its own")
    orphaned = record(services, 4242)
    orphaned[2] = Span("orphan", "s404", "payments", 7, 0, 9, 10)
    replayed, divergences, visits, orphans = replay(orphaned, services)
    check(orphans == ["orphan"] and visits == len(orphaned)
          and divergent_ids(divergences) == ("orphan",),
          f"an orphaned span must be replayed and flagged, not crash: {orphans} {divergences}")

    # 7) determinism and stable divergence ordering.
    def projection(result):
        replayed, divergences, visits, orphans = result
        return ([(sid, span.input, span.payload) for sid, span in sorted(replayed.items())],
                divergences, visits, orphans)

    first = replay(spans, faulted_services("orders"), True)
    second = replay(spans, faulted_services("orders"), True)
    starts = [entry[0] for entry in first[1]]
    index = {span.span_id: span for span in spans}
    check(projection(first) == projection(second)
          and starts == sorted(starts, key=lambda sid: index[sid].key()),
          "replays must be deterministic and sorted by (start tick, span id)")

    print(f"spans={len(spans)} services={len(SERVICES)} visits={first[2]} "
          f"orphans={len(orphans)}")
    print(f"isolated fault diverts {divergent_ids(isolated_div)}; orders cascade "
          f"{sorted(cascade)}")
    print("PASS")


if __name__ == "__main__":
    main()
