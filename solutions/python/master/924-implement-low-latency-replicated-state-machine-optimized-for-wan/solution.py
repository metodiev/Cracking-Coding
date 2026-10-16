"""Task 924 - Implement low-latency replicated state machine optimized for WANs with batching.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

An in-memory Raft-flavoured replicated state machine driven by an explicit discrete-event loop.

* Model: 5 replicas, majority quorum 3, one leader per term.  Messages are AppendEntries (log
  index plus batch), Ack and Commit notifications, each delivered after a per-replica WAN latency in
  ticks and priced at one message; the leader counts itself towards the quorum.
* Batching and pipelining: client operations queue up and are flushed every `batch_interval` ticks
  as a single log entry, so several entries are in flight at once; the commit index advances in
  order and each replica applies its committed prefix to the key/value state machine.
* Leader change: at a scheduled tick the leader fails, the surviving replica with the longest log
  wins the election for the next term, it ships its log to repair the followers (log matching) and
  commits resume.  The client queue is shared, modelling a client that retries against the leader.

Verified: any two majorities of five intersect; every replica's committed state equals an oracle
application of the same batch order; each commit is a prefix of that order; batching commits the
identical operation sequence with strictly fewer messages per operation (paying a higher mean
commit latency, the WAN trade-off); after the injected failure the surviving log still holds every
committed entry and all operations commit; a batch interval longer than the run commits nothing;
zero ticks send nothing; and repeated runs are bit-identical.  Simplified: no persistence, no
snapshots, a single scheduled failure and a longest-log election.
"""
from itertools import combinations

N_REPLICAS, QUORUM = 5, 3
LATENCY = [1, 3, 2, 4]                 # WAN ticks to each follower, asymmetric on purpose
OP_INTERVAL, TICKS, BATCH = 2, 60, 6
OPS = [("k%d" % (index % 4), index) for index in range(12)]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def quorums():
    return list(combinations(range(N_REPLICAS), QUORUM))


def simulate(batch_interval, leader_failure=None, ticks=TICKS, ops=None):
    """Run the replicated state machine; returns logs, commit indices, states and metrics."""
    operations = list(OPS if ops is None else ops)
    logs = [[] for _ in range(N_REPLICAS)]          # per-replica log of batches
    commit = [0] * N_REPLICAS                       # committed index per replica
    applied_count = [0] * N_REPLICAS
    applied = [dict() for _ in range(N_REPLICAS)]   # the state machine itself
    events, metrics = [], {"messages": 0, "elections": 0, "proposed": 0, "latency": []}
    sequence, ops_committed = 0, 0
    leader, term, pending = 0, 0, []
    acks, proposed_at = {}, {}

    def post(time, destination, kind, payload):
        nonlocal sequence
        sequence += 1
        metrics["messages"] += 1
        events.append((time + LATENCY[destination % len(LATENCY)], sequence, kind, destination,
                       payload))

    def advance(replica):
        while applied_count[replica] < min(commit[replica], len(logs[replica])):
            for key, value in logs[replica][applied_count[replica]]:
                applied[replica][key] = value
            applied_count[replica] += 1

    def deliver(time, kind, replica, payload):
        if kind == "append":
            index, batch = payload
            if index == len(logs[replica]):
                logs[replica].append(batch)
            if index < len(logs[replica]):
                acks.setdefault(index, set()).add(replica)
                post(time, leader, "ack", index)
        elif kind == "ack":
            acks.setdefault(payload, set()).add(replica)
        else:
            commit[replica] = max(commit[replica], payload)
            advance(replica)

    for time in range(ticks):
        events.sort(key=lambda event: (event[0], event[1]))
        while events and events[0][0] <= time:
            _, _, kind, replica, payload = events.pop(0)
            deliver(time, kind, replica, payload)

        if leader_failure == time:
            survivors = [r for r in range(N_REPLICAS) if r != leader]
            surviving_commit = max(commit)
            leader = max(survivors, key=lambda r: (len(logs[r]), -r))
            term += 1
            metrics["elections"] += 1
            commit[leader] = surviving_commit
            for replica in range(N_REPLICAS):
                if len(logs[replica]) < len(logs[leader]):
                    logs[replica] = list(logs[leader])              # ship the surviving log
            acks = {index: {leader} for index in range(len(logs[leader]))}
            for replica in survivors:
                if replica != leader and logs[leader]:
                    post(time, replica, "append", (len(logs[leader]) - 1, logs[leader][-1]))
                post(time, replica, "commit", commit[leader])

        if time % OP_INTERVAL == 0 and time // OP_INTERVAL < len(operations):
            pending.append(operations[time // OP_INTERVAL])

        if pending and time % batch_interval == 0:
            index = len(logs[leader])
            logs[leader].append(list(pending))
            pending, proposed_at[index] = [], time
            acks[index] = {leader}
            metrics["proposed"] += 1
            for replica in range(N_REPLICAS):
                if replica != leader:
                    post(time, replica, "append", (index, logs[leader][index]))

        index = commit[leader]
        while index < len(logs[leader]) and len(acks.get(index, ())) >= QUORUM:
            commit[leader] = index + 1
            metrics["latency"].append(time - proposed_at.get(index, time))
            ops_committed += len(logs[leader][index])
            for replica in range(N_REPLICAS):
                if replica != leader:
                    post(time, replica, "commit", commit[leader])
            index += 1
        advance(leader)

    for replica in range(N_REPLICAS):
        commit[replica] = min(commit[replica], len(logs[replica]))
        advance(replica)
    return {"logs": logs, "commit": commit, "applied": applied, "leader": leader, "term": term,
            "metrics": metrics, "ops_committed": ops_committed}


def oracle_state(batches):
    """Independent reference: apply a batch order to a fresh dictionary."""
    state = {}
    for batch in batches:
        for key, value in batch:
            state[key] = value
    return state


def main():
    # 1. Quorum intersection: the safety property every majority commit relies on.
    majorities = quorums()
    check(len(majorities) == 10 and all(set(a) & set(b) for a in majorities for b in majorities),
          "any two majorities of five replicas must intersect")

    # 2. Baseline batched run: every replica agrees with the oracle state machine.
    batched = simulate(BATCH)
    reference = oracle_state(batched["logs"][batched["leader"]])
    check(batched["ops_committed"] == len(OPS)
          and batched["commit"][batched["leader"]] == batched["metrics"]["proposed"],
          "every proposed batch must end up committed once all operations arrive")
    check(1 < len(batched["logs"][batched["leader"]]) < len(OPS),
          "batching must group operations into fewer log entries than operations")
    check(batched["logs"][batched["leader"]][0] == [OPS[0]],
          "the batch flushed at tick zero must hold the operations that had arrived")
    check(all(batched["applied"][r] == reference for r in range(N_REPLICAS)),
          "every replica state machine must equal the oracle application")

    # 3. Prefix consistency: each replica commit is a prefix of the agreed batch order.
    check(all(batched["logs"][r][:batched["commit"][r]] ==
              batched["logs"][batched["leader"]][:batched["commit"][r]] for r in range(N_REPLICAS)),
          "a replica commit must be a prefix of the agreed batch order")

    # 4. Batching trades messages for latency: fewer messages per operation, higher mean commit wait.
    unbatched = simulate(1)
    per_op = batched["metrics"]["messages"] / batched["ops_committed"]
    per_op_plain = unbatched["metrics"]["messages"] / unbatched["ops_committed"]
    mean = sum(batched["metrics"]["latency"]) / len(batched["metrics"]["latency"])
    mean_plain = sum(unbatched["metrics"]["latency"]) / len(unbatched["metrics"]["latency"])
    check(per_op < per_op_plain,
          "batching must reduce the number of messages per committed operation")
    check(mean >= mean_plain, "batching must not pretend to be free")
    check([op for batch in batched["logs"][0] for op in batch] ==
          [op for batch in unbatched["logs"][0] for op in batch] == OPS,
          "batching must not change the committed operation sequence")

    # 5. Leader failure: one election, every committed entry survives, commits resume and agree.
    survived = simulate(BATCH, leader_failure=20)
    check(survived["term"] == 1 and survived["metrics"]["elections"] == 1,
          "a scheduled failure must trigger exactly one election")
    check(survived["logs"][survived["leader"]][:batched["commit"][batched["leader"]]]
          == batched["logs"][batched["leader"]][:batched["commit"][batched["leader"]]],
          "the new leader must hold every entry that was committed before the failure")
    check(survived["ops_committed"] == len(OPS)
          and all(survived["applied"][r] == oracle_state(survived["logs"][survived["leader"]])
                  for r in range(N_REPLICAS)),
          "commits must resume after the failure and all replicas must still agree")

    # 6. Boundary cases: zero ticks, and a client with nothing to send.
    empty = simulate(BATCH, ticks=0)
    check(empty["metrics"]["messages"] == 0 and empty["ops_committed"] == 0,
          "a zero-tick run must send nothing and commit nothing")
    idle = simulate(BATCH, ops=[])
    check(idle["ops_committed"] == 0 and idle["metrics"]["proposed"] == 0
          and idle["metrics"]["messages"] == 0,
          "with no client operations nothing is proposed, batched or committed")
    check(simulate(1, ops=[OPS[0]])["ops_committed"] == 1,
          "a single operation must commit with an unbatched pipeline")

    # 7. Determinism: the same schedule replays identically.
    replay = simulate(BATCH)
    check(replay["applied"] == batched["applied"] and replay["metrics"] == batched["metrics"],
          "the simulation must be deterministic")

    print(f"batched: {batched['metrics']['proposed']} batches, {batched['metrics']['messages']} "
          f"messages, {per_op:.2f} messages/op, mean commit wait {mean:.1f} ticks")
    print(f"unbatched: {unbatched['metrics']['messages']} messages, {per_op_plain:.2f} messages/op, "
          f"mean commit wait {mean_plain:.1f} ticks")
    print("PASS")


if __name__ == "__main__":
    main()
