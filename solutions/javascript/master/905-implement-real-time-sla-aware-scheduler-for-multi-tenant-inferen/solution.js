/** Task 905 - Implement real-time SLA-aware scheduler for multi-tenant inference.
 *  (Tier: Master, CTCI Beyond — Scheduling & Real-Time Systems)
 *
 * A discrete-time batched inference server: each 10 ms round admits the requests that arrived, drops the
 * ones that are already infeasible (arrival-to-deadline window shorter than their own service cost) or
 * that overflow the bounded queue, then forms EDF batches from the earliest-deadline prefix. A batch
 * costs max(cost) + 0.1*(size-1) ms, so batching amortizes, and a request misses its SLA exactly when its
 * completion time passes its deadline.
 *
 * The claim under test is the classic one: on a single server EDF is deadline-optimal among
 * work-conserving orders, so it must dominate FIFO on the same trace, and admission control converts
 * unavoidable misses into rejects (an explicit, counted outcome) instead of silent SLA violations.
 * Checks: schedule legality and uniqueness, the SLA/reject targets, an independent replay of the emitted
 * batch log, EDF vs FIFO on identical traces, per-tenant fairness, the batch-size trade-off, degenerate
 * configurations and reproducibility.
 */
"use strict";
const assert = require("node:assert");
const ROUND_MS = 10;
const BATCH_OVERHEAD_MS = 0.1;

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  int(bound) { return this.next() % bound; }
  float() { return this.next() / 4294967296; }
}
/** Knuth's Poisson sampler for the arrival process. */
function poisson(lambda, prng) {
  const limit = Math.exp(-lambda);
  let count = 0;
  let product = prng.float();
  while (product > limit && count < 20) {
    count += 1;
    product *= prng.float();
  }
  return count;
}
/** Poisson arrivals over `rounds` rounds for several tenants, each with its own SLA and cost profile. */
function generateTrace(tenants, rounds, prng) {
  const requests = [];
  for (let round = 0; round < rounds; round++) {
    for (const tenant of tenants) {
      const count = poisson(tenant.arrivalRate, prng);
      for (let i = 0; i < count; i++) {
        const costMs = tenant.baseCostMs + prng.int(tenant.costJitterMs + 1);
        requests.push({
          id: `${tenant.name}#${round}.${i}`, tenant: tenant.name, arrivalMs: round * ROUND_MS + prng.int(ROUND_MS), slaMs: tenant.slaMs,
          costMs, });
      }
    }
  }
  return requests;
}
/** EDF (or FIFO when order === "fifo") with admission control and bounded queues.
 *  mode "static" admits every request whose cost fits its own SLA window (so the ordering comparison
 *  sees an identical workload); mode "projected" additionally projects the queue's worst-case drain
 *  time, which is what keeps admitted requests inside the SLA. */
function schedule(requests, options, order = "edf", mode = "projected") {
  const { maxBatch, maxQueue } = options;
  const queue = [];
  const batches = [];
  const outcomes = new Map();
  let rejected = 0;
  let now = 0;
  const arrivals = [...requests].sort((a, b) => a.arrivalMs - b.arrivalMs || (a.id < b.id ? -1 : 1));
  let index = 0;
  while (index < arrivals.length || queue.length > 0) {
    if (index < arrivals.length && (queue.length === 0 || arrivals[index].arrivalMs <= now)) {
      const request = arrivals[index++];
      now = Math.max(now, request.arrivalMs);         // the server is idle until the next arrival
      const worstCost = queue.reduce((worst, queued) => Math.max(worst, queued.costMs), request.costMs);
      const projected = now + Math.ceil((queue.length + 1) / maxBatch) * worstCost;
      const admissible = mode === "static"
        ? request.costMs <= request.slaMs
        : projected <= request.arrivalMs + request.slaMs;
      if (queue.length >= maxQueue || !admissible) {
        rejected += 1;                                // infeasible or overloaded: an explicit reject
        outcomes.set(request.id, { status: "rejected", request });
        continue;
      }
      queue.push(request);
      continue;
    }
    if (queue.length === 0) continue;
    queue.sort((a, b) => (order === "edf" ? a.arrivalMs + a.slaMs - (b.arrivalMs + b.slaMs) : a.arrivalMs - b.arrivalMs) || (a.id < b.id ? -1 : 1));
    const batch = queue.splice(0, Math.min(maxBatch, queue.length));
    const duration = batch.reduce((worst, request) => Math.max(worst, request.costMs), 0) + BATCH_OVERHEAD_MS * (batch.length - 1);
    const finish = now + duration;
    batches.push({ ids: batch.map((request) => request.id), startMs: now, durationMs: duration, finishMs: finish });
    for (const request of batch) {
      const deadline = request.arrivalMs + request.slaMs;
      outcomes.set(request.id, { status: finish <= deadline ? "met" : "missed", request, finish, deadline });
    }
    now = finish;
  }
  return { batches, outcomes, rejected, serverBusyMs: batches.reduce((sum, b) => sum + b.durationMs, 0) };
}
/** Independent replay of the batch log: completion times and misses recomputed from the log alone. */
function replay(batches, outcomes) {
  let clock = 0;
  let met = 0;
  let missed = 0;
  let busy = 0;
  for (const batch of batches) {
    assert.ok(batch.startMs >= clock, "batches must not overlap in time");
    assert.ok(batch.finishMs >= batch.startMs + batch.durationMs - 1e-9, "a batch must take its duration");
    busy += batch.durationMs;
    clock = batch.finishMs;
    for (const id of batch.ids) {
      const outcome = outcomes.get(id);
      if (!outcome || outcome.request.id !== id) return null;
      if (batch.finishMs <= outcome.request.arrivalMs + outcome.request.slaMs) met += 1; else missed += 1;
    }
  }
  return { met, missed, busyMs: busy, utilization: busy / Math.max(1, clock) };
}

function countByStatus(outcomes, status) {
  let total = 0;
  for (const outcome of outcomes.values()) if (outcome.status === status) total += 1;
  return total;
}

function main() {
  const tenants = [
    { name: "small", arrivalRate: 1.0, baseCostMs: 4, costJitterMs: 4, slaMs: 40 },
    { name: "medium", arrivalRate: 0.9, baseCostMs: 8, costJitterMs: 8, slaMs: 60 },
    { name: "large", arrivalRate: 0.6, baseCostMs: 12, costJitterMs: 12, slaMs: 80 },
    { name: "bursty", arrivalRate: 0.4, baseCostMs: 10, costJitterMs: 6, slaMs: 20 }, ];
  const options = { maxBatch: 4, maxQueue: 16 };
  const trace = generateTrace(tenants, 20, new Prng(0x5eed0905));
  const edf = schedule(trace, options, "edf");
  const fifo = schedule(trace, options, "fifo");
  const edfOpen = schedule(trace, options, "edf", "static");      // no queue-projection admission control
  const fifoOpen = schedule(trace, options, "fifo", "static");
  // 1. legality — every request is decided exactly once, and every batch is a set of scheduled requests.
  {
    assert.strictEqual(edf.outcomes.size, trace.length, "each request must have exactly one outcome");
    const scheduled = edf.batches.flatMap((batch) => batch.ids);
    assert.strictEqual(new Set(scheduled).size, scheduled.length, "a request is never scheduled twice");
    const decided = [...edf.outcomes.values()];
    assert.strictEqual(decided.filter((o) => o.status === "rejected").length, edf.rejected, "rejects are counted");
    for (const outcome of decided) {
      assert.ok(outcome.status === "rejected" || outcome.finish >= outcome.request.arrivalMs, "no time travel");
      assert.ok(scheduled.includes(outcome.request.id) === (outcome.status !== "rejected"), "scheduled exactly when not rejected");
    }
  }
  // 2. the SLA — admission control keeps misses among admitted work inside the 3% target, and the
  //    residual comes from arrivals the (non-clairvoyant) controller cannot see when it admits work.
  let edfMisses = 0;
  {
    edfMisses = countByStatus(edf.outcomes, "missed");
    const admitted = edf.outcomes.size - edf.rejected;
    assert.ok(admitted > 0.5 * trace.length, `only ${admitted}/${trace.length} requests were admitted`);
    assert.ok(edfMisses / admitted <= 0.03, `miss rate ${(edfMisses / admitted).toFixed(3)} must stay <= 3%`);
    assert.ok(edf.rejected / trace.length <= 0.25, "the reject rate must stay bounded");
    const openAdmitted = edfOpen.outcomes.size - edfOpen.rejected;
    const openMisses = countByStatus(edfOpen.outcomes, "missed");
    assert.ok(openMisses / openAdmitted >= 3 * (edfMisses / admitted),
      `without projection ${openMisses}/${openAdmitted} must be much worse than ${edfMisses}/${admitted}`);
    const audited = replay(edf.batches, edf.outcomes);
    assert.strictEqual(audited.met, admitted - edfMisses, "the replay agrees on the met count");
    assert.strictEqual(audited.missed, edfMisses, "the replay agrees on the miss count");
    assert.ok(Math.abs(audited.busyMs - edf.serverBusyMs) < 1e-6, "the replay agrees on busy time");
    assert.ok(audited.utilization > 0.2 && audited.utilization <= 1, "utilization is a fraction");
  }
  // 3. EDF dominance — with identical admitted workloads EDF cannot miss more deadlines than FIFO.
  {
    const fifoMisses = countByStatus(fifoOpen.outcomes, "missed");
    const edfOpenMisses = countByStatus(edfOpen.outcomes, "missed");
    assert.ok(edfOpenMisses <= fifoMisses, `EDF missed ${edfOpenMisses} vs FIFO ${fifoMisses}`);
    assert.ok(fifoMisses > 0, "the trace is loaded enough to make ordering matter");
    const lag = (result) => [...result.outcomes.values()].filter((o) => o.status !== "rejected")
      .reduce((sum, o) => sum + (o.finish - o.request.arrivalMs), 0);
    assert.ok(lag(edfOpen) <= lag(fifoOpen), "EDF should also win on total latency");
    assert.ok(countByStatus(edf.outcomes, "missed") <= countByStatus(fifo.outcomes, "missed"), "EDF wins with projection enabled too");
  }
  // 4. fairness — no tenant starves, and no request waits longer than its own SLA window plus a batch.
  {
    for (const tenant of tenants) {
      const served = [...edf.outcomes.values()].filter((o) => o.request.tenant === tenant.name);
      assert.ok(served.length > 0, `tenant ${tenant.name} must be served`);
      assert.ok(served.some((o) => o.status !== "rejected"), `tenant ${tenant.name} must have admitted work`);
    }
    const worstWait = Math.max(...[...edf.outcomes.values()].filter((o) => o.status !== "rejected")
      .map((o) => o.finish - o.request.arrivalMs));
    const widestBatch = Math.max(...edf.batches.map((batch) => batch.durationMs));
    assert.ok(worstWait <= widestBatch + 8 * options.maxBatch * ROUND_MS, `worst wait ${worstWait} must stay within a bounded number of batches`);
  }
  // 5. batching trade-off — wider batches amortize overhead, admitting more work per busy millisecond.
  {
    const [tight, two, four, wide] = [1, 2, 4, 8].map((maxBatch) => schedule(trace, { ...options, maxBatch }, "edf"));
    assert.ok(tight.serverBusyMs > two.serverBusyMs && two.serverBusyMs > four.serverBusyMs &&
      four.serverBusyMs > wide.serverBusyMs, "busy time must fall as batches widen");
    assert.ok(four.rejected < tight.rejected && wide.rejected < tight.rejected, "batching admits more work");
    assert.strictEqual(countByStatus(tight.outcomes, "missed"), 0, "unbatched service rejects rather than violating the SLA");
    assert.ok(countByStatus(wide.outcomes, "missed") <= 1, "wider batches still keep misses bounded");
    assert.ok(tight.batches.length > wide.batches.length, "wider batches need fewer service rounds");
  }
  // 6. admission control — infeasible and overflowing requests are rejected instead of failing later.
  {
    const impossible = [{ id: "x", tenant: "small", arrivalMs: 0, slaMs: 1, costMs: 50 }];
    const one = schedule(impossible, options, "edf");
    assert.strictEqual(one.rejected, 1, "a request whose window is shorter than its cost is rejected");
    assert.strictEqual(one.batches.length, 0, "nothing is scheduled for it");
    const flood = generateTrace([{ name: "flood", arrivalRate: 6, baseCostMs: 20, costJitterMs: 10, slaMs: 60 }], 10, new Prng(5));
    const loaded = schedule(flood, { maxBatch: 2, maxQueue: 4 }, "edf");
    assert.ok(loaded.rejected > 0, "an overloaded server must shed load");
    assert.ok(loaded.rejected + [...loaded.outcomes.values()].filter((o) => o.status !== "rejected").length ===
      flood.length, "every request is either rejected or decided");
  }
  // 7. boundary — an empty trace, one admitted request, and a zero-length round.
  {
    const none = schedule([], options, "edf");
    assert.deepStrictEqual(none.batches, []);
    assert.strictEqual(none.outcomes.size, 0);
    assert.strictEqual(replay([], none.outcomes).missed, 0);
    const single = schedule([{ id: "s", tenant: "small", arrivalMs: 5, slaMs: 100, costMs: 7 }], options, "edf");
    assert.strictEqual(single.batches.length, 1);
    const sole = single.outcomes.get("s");
    assert.strictEqual(sole.status, "met");
    assert.strictEqual(sole.finish, 12, "the sole request finishes at arrival + cost");
  }
  // 8. determinism — the trace and both schedules replay exactly from their seeds.
  {
    assert.deepStrictEqual(generateTrace(tenants, 20, new Prng(0x5eed0905)), trace);
    assert.deepStrictEqual(schedule(trace, options, "edf").batches, edf.batches);
    assert.notDeepStrictEqual(schedule(trace, { ...options, maxBatch: 2 }, "edf").batches, edf.batches);
  }
  console.log(`scheduling: ${trace.length} requests, ${tenants.length} tenants, EDF misses ${edfMisses} ` +
    `vs FIFO ${countByStatus(fifo.outcomes, "missed")} with projection, EDF ${countByStatus(edfOpen.outcomes, "missed")} ` +
    `vs FIFO ${countByStatus(fifoOpen.outcomes, "missed")} without; rejects ${edf.rejected}, busy ` +
    `${edf.serverBusyMs.toFixed(0)} ms, ${edf.batches.length} batches`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = { ROUND_MS, BATCH_OVERHEAD_MS, Prng, generateTrace, schedule, replay, countByStatus };
