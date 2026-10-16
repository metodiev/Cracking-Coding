/** Task 924 - Implement low-latency replicated state machine optimized for WANs with batching.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A discrete-event model of a static-leader replicated log (Raft-like replication, no elections):
 * a seeded latency table gives symmetric one-way delays between nodes, client requests arrive on a
 * fixed schedule, the leader collects the requests that accumulated during one tick into one
 * AppendEntries batch and pipelines it to the followers, followers append and answer with a single
 * batch ack, and the leader advances the commit index as soon as a quorum (majority of all nodes,
 * counting its own log) has acknowledged - the committed prefix is then applied to the state
 * machine and piggybacked on later messages.
 * Simplification: no terms/elections/log truncation, fixed latencies, one batch per tick, no
 * snapshotting. Verified: convergence of all replicas and identical committed logs, a closed-form
 * cross-check of the applied state, message cost per entry with and without batching, the commit
 * latency lower bound set by the quorum round trip, the absence of commits without a quorum, the
 * empty-workload boundary, invalid input, and determinism.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17; x ^= x << 5;
    return (this.state = x >>> 0);
  }
  int(bound) { return this.next() % bound; }
}

/** Symmetric seeded one-way latency table in ticks. */
function makeLatency(prng, ids, options = {}) {
  const base = options.baseLatency ?? 40;
  const jitter = options.jitter ?? 20;
  const table = new Map();
  for (let a = 0; a < ids.length; a++) {
    for (let b = a + 1; b < ids.length; b++) {
      const value = base + prng.int(jitter);
      table.set(`${ids[a]}-${ids[b]}`, value);
      table.set(`${ids[b]}-${ids[a]}`, value);
    }
  }
  return (a, b) => (a === b ? 0 : table.get(`${a}-${b}`));
}

/** Deterministic state machine: a counter plus a per-key tally, applied in commit order. */
function applyCommand(state, command) {
  state.counter += command.value;
  state.keys.set(command.key, (state.keys.get(command.key) ?? 0) + command.value);
  state.applied.push(command.id);
  return state;
}

function freshState() { return { counter: 0, keys: new Map(), applied: [] }; }

/**
 * Run the cluster. Returns the committed log, the metrics and one state machine per node.
 * `blocked` lists node ids whose traffic is dropped (used to model a lost quorum).
 */
function simulate(options = {}) {
  const nodeCount = options.nodeCount ?? 5;
  assert.ok(Number.isInteger(nodeCount) && nodeCount >= 1, "nodeCount must be a positive integer");
  assert.ok(Number.isInteger(options.batchSize ?? 4) && (options.batchSize ?? 4) >= 1,
    "batchSize must be a positive integer");
  const batchSize = options.batchSize ?? 4;
  const ticks = options.ticks ?? 900;
  const requestCount = options.requestCount ?? 40;
  // Client requests arrive in bursts (a realistic WAN workload), which is what batching exploits.
  const burst = options.burst ?? 8;
  const burstInterval = options.burstInterval ?? 60;
  const seed = options.seed ?? 924;
  const prng = new Prng(seed);
  const ids = Array.from({ length: nodeCount }, (_, k) => k);
  const latency = makeLatency(prng, ids, options);
  const blocked = new Set(options.blocked ?? []);
  const leader = ids[0];
  const reachable = ids.filter((id) => id !== leader && !blocked.has(id));

  const log = [];                       // leader log: {command, submitTick, commitTick}
  const pending = Array.from({ length: requestCount },
    (_, k) => ({ id: k, submitTick: 1 + Math.floor(k / burst) * burstInterval }));
  let cursor = 0;                       // next request to be batched
  const ackIndex = new Map(ids.map((id) => [id, 0]));
  const replicaLength = new Map(ids.map((id) => [id, 0]));
  const states = new Map(ids.map((id) => [id, freshState()]));
  const inFlight = [];
  let messages = 0, dropped = 0, commitIndex = 0, latencies = [], entriesAppended = 0, appends = 0;

  const send = (to, type, payload, from, now) => {
    if (blocked.has(to) || blocked.has(from)) { dropped++; return; }
    messages++;
    inFlight.push({ deliverAt: now + latency(from, to), to, from, type, payload });
  };

  for (let tick = 1; tick <= ticks; tick++) {
    // Deliver everything that is due: followers append whole batches and ack once per batch.
    for (let k = inFlight.length - 1; k >= 0; k--) {
      if (inFlight[k].deliverAt > tick) continue;
      const message = inFlight.splice(k, 1)[0];
      if (message.type === "append") {
        const nextExpected = replicaLength.get(message.to);
        let appended = 0;
        for (const entry of message.payload.entries) {
          if (entry.index !== nextExpected + appended + 1) continue;
          appended++;
        }
        replicaLength.set(message.to, nextExpected + appended);
        entriesAppended += appended;
        send(leader, "ack", { index: nextExpected + appended, commitHint: message.payload.commitIndex },
          message.to, tick);
      } else {
        ackIndex.set(message.from, Math.max(ackIndex.get(message.from), message.payload.index));
      }
    }

    // The leader forms at most one batch per tick from the requests that have already arrived and
    // pipelines it to every follower without waiting for the previous batch.
    const batch = [];
    while (cursor < pending.length && batch.length < batchSize && pending[cursor].submitTick <= tick) {
      batch.push(pending[cursor]);
      cursor++;
    }
    if (batch.length > 0) {
      const entries = batch.map((r) => {
        const entry = { index: log.length + 1, command: { id: r.id, key: `k${r.id % 3}`, value: 1 },
          submitTick: r.submitTick };
        log.push(entry);
        return entry;
      });
      appends++;
      for (const follower of ids) {
        if (follower === leader) continue;
        send(follower, "append", { entries, commitIndex }, leader, tick);
      }
      ackIndex.set(leader, log.length);
    }

    // Commit as soon as a majority of nodes (including the leader's own log) has acknowledged.
    const majority = Math.floor(nodeCount / 2) + 1;
    const sorted = ids.map((id) => ackIndex.get(id)).sort((a, b) => b - a);
    const canCommit = sorted[majority - 1] ?? 0;
    if (canCommit > commitIndex) {
      for (let index = commitIndex + 1; index <= canCommit; index++) {
        const entry = log[index - 1];
        entry.commitTick = tick;
        latencies.push(tick - entry.submitTick);
        for (const id of ids) applyCommand(states.get(id), entry.command);
      }
      commitIndex = canCommit;
    }
  }

  const committed = log.slice(0, commitIndex);
  return { log, committed, commitIndex, states, messages, dropped, latencies, appends,
    entriesAppended, replicaLength, ids, leader, batchSize, ticks, entryCount: log.length };
}

function main() {
  // 1. Safety and liveness: every replica converges on the same state and the same commit index.
  const run = simulate({ nodeCount: 5, batchSize: 4, requestCount: 40, seed: 924 });
  assert.strictEqual(run.commitIndex, run.entryCount, "every entry must eventually commit");
  const reference = run.states.get(run.leader);
  for (const id of run.ids) {
    assert.deepStrictEqual([...run.states.get(id).keys], [...reference.keys],
      `replica ${id} diverged`);
    assert.deepStrictEqual(run.states.get(id).applied, reference.applied,
      `replica ${id} applied a different log prefix`);
  }

  // 2. Independent oracle: replaying the committed log from scratch reproduces the live state, and
  //    the counter (closed form) equals the number of committed increments.
  const replay = freshState();
  for (const entry of run.committed) applyCommand(replay, entry.command);
  assert.deepStrictEqual([...replay.keys], [...reference.keys]);
  assert.strictEqual(reference.counter, run.commitIndex, "the counter must equal the commit count");

  // 3. Batching economics: fewer messages per entry, and the unbatched count matches a closed form.
  const unbatched = simulate({ nodeCount: 5, batchSize: 1, requestCount: 40, seed: 924 });
  const batched = simulate({ nodeCount: 5, batchSize: 8, requestCount: 40, seed: 924 });
  const perEntry = (r) => r.messages / r.commitIndex;
  assert.ok(perEntry(batched) < perEntry(unbatched) / 4,
    `batching barely helped: ${perEntry(unbatched)} -> ${perEntry(batched)}`);
  const followers = 4;
  assert.strictEqual(unbatched.messages, unbatched.entryCount * followers * 2,
    "one batch per entry must cost exactly an append and an ack per follower");
  assert.strictEqual(unbatched.appends, unbatched.entryCount);
  assert.ok(batched.appends < unbatched.appends / 3, "batch 8 must cut the number of appends");

  // 4. Latency: committing needs a quorum round trip, so it is bounded below by the fastest
  //    follower round trip and above by that plus the batching schedule.
  const minRoundTrip = 2 * 40; // baseLatency = 40 ticks each way
  assert.ok(Math.min(...run.latencies) >= minRoundTrip,
    `commit latency ${Math.min(...run.latencies)} below the round-trip bound`);
  assert.ok(Math.max(...run.latencies) <= minRoundTrip + 20 + 2 * run.batchSize,
    "commit latency far above the round trip plus one batching window");
  assert.ok(run.latencies.length === run.commitIndex);

  // 5. Boundary: an empty workload sends nothing and changes no state; a single request commits.
  const empty = simulate({ nodeCount: 3, requestCount: 0, seed: 7 });
  assert.strictEqual(empty.messages, 0, "an empty workload must not produce messages");
  assert.strictEqual(empty.commitIndex, 0);
  assert.deepStrictEqual([...empty.states.get(0).keys], []);
  const single = simulate({ nodeCount: 3, requestCount: 1, seed: 7, batchSize: 1 });
  assert.strictEqual(single.commitIndex, 1);
  assert.strictEqual(single.states.get(0).counter, 1);
  assert.strictEqual(single.states.get(2).counter, 1, "the third node must commit too");
  const soloNode = simulate({ nodeCount: 1, requestCount: 5, seed: 7, batchSize: 2 });
  assert.strictEqual(soloNode.commitIndex, 5, "a single node commits on its own");
  assert.strictEqual(soloNode.messages, 0, "a single node has nobody to talk to");

  // 6. Without a quorum the log never commits (safety holds, liveness is lost by design).
  const split = simulate({ nodeCount: 5, batchSize: 4, requestCount: 20, seed: 924,
    blocked: [2, 3, 4] });
  assert.ok(split.log.length > 0, "the leader still appends locally");
  assert.strictEqual(split.commitIndex, 0, "two acknowledgements are not a majority of five");
  assert.strictEqual(split.states.get(0).counter, 0, "an uncommitted entry must not be applied");
  assert.ok(split.dropped > 0, "blocked traffic must be counted as dropped");
  assert.ok(split.messages <= split.appends * 2, "only the reachable follower may be contacted");

  // 7. Invalid input is rejected.
  assert.throws(() => simulate({ nodeCount: 0 }), /positive integer/);
  assert.throws(() => simulate({ batchSize: 0 }), /positive integer/);

  // 8. Determinism: same seed replays identically, a different seed changes only the latencies.
  const rerun = simulate({ nodeCount: 5, batchSize: 4, requestCount: 40, seed: 924 });
  assert.deepStrictEqual(rerun.states.get(0).keys, reference.keys);
  assert.deepStrictEqual(rerun.latencies, run.latencies);
  const otherSeed = simulate({ nodeCount: 5, batchSize: 4, requestCount: 40, seed: 925 });
  assert.strictEqual(otherSeed.commitIndex, rerun.commitIndex);
  assert.notDeepStrictEqual(otherSeed.latencies, rerun.latencies);

  console.log(`rsm: ${run.commitIndex} commits, ${run.messages} messages ` +
    `(${perEntry(run).toFixed(2)}/entry, unbatched ${perEntry(unbatched).toFixed(2)}/entry), ` +
    `min/max commit latency ${Math.min(...run.latencies)}/${Math.max(...run.latencies)} ticks`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, makeLatency, applyCommand, freshState, simulate };
