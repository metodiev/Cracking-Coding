/** Task 935 - Implement multi-tenant GPU scheduler with preemption and memory reclamation.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * In-memory cluster: each GPU has a memory budget (MiB) and a compute budget (SMs). Classified jobs
 * (free < standard < premium < critical) carry a footprint, compute width, gang size and duration in
 * ticks; a job runs on device ids whose free budget satisfies every dimension. When a higher class
 * cannot fit, the scheduler preempts a greedy minimal prefix of strictly lower-class jobs, reclaims
 * their memory/compute and resumes them later with the remaining work preserved. Logical ticks only
 * (no timers, threads or wall clock) and seeded splitmix64 is the sole randomness, so every run is
 * byte-identical. Simplified: a device is two scalar budgets rather than MIG instances, preemption
 * is free, and the victim set is a greedy prefix, not a knapsack. Verified: the ledger rebuilt
 * independently from the schedule log, per-job executed device-ticks, preemption monotonicity
 * (victims strictly lower class, none between equals), liveness, 1.35x fairness, and the closed-form
 * makespan bound; unschedulable and invalid jobs are rejected.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the only source of randomness; identical stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

const FREE = 0, STANDARD = 1, PREMIUM = 2, CRITICAL = 3;
/** Lowest device ids that can each host one gang slot of `job`. */
function placementFor(caps, job) {
  const fits = caps.map((c, i) => (c.mem >= job.mem && c.sm >= job.sm ? i : -1)).filter((i) => i >= 0);
  return fits.length >= job.gang ? fits.slice(0, job.gang) : null;
}

class Cluster {
  constructor({ devices, memCap, smCap }) {
    if (devices <= 0 || memCap <= 0 || smCap <= 0) throw new Error("positive capacity required");
    this.memCap = memCap; this.smCap = smCap;
    this.devices = Array.from({ length: devices }, (_, id) => ({ id, mem: memCap, sm: smCap }));
  }
  caps() { return this.devices.map((d) => ({ mem: d.mem, sm: d.sm })); }
  /** Add (sign 1) or give back (sign -1) `job`'s footprint on its device ids. */
  move(ids, job, sign) {
    for (const id of ids) { this.devices[id].mem -= sign * job.mem; this.devices[id].sm -= sign * job.sm; }
  }
}

class Scheduler {
  constructor(cluster, { preemption = true } = {}) {
    this.cluster = cluster; this.preemption = preemption;
    this.queue = []; this.running = []; this.done = []; this.rejected = [];
    this.log = []; this.events = []; this.tenantTicks = new Map();
    this.tick = 0; this.seq = 0; this.preemptions = 0;
  }
  submit(job) {
    if (!(job.mem > 0) || !(job.sm > 0) || !(job.ticks > 0)) throw new Error(`invalid job ${job.id}`);
    if (!Number.isInteger(job.gang) || job.gang < 1 || job.gang > this.cluster.devices.length) throw new Error(`job ${job.id}: gang cannot fit`);
    if (job.mem > this.cluster.memCap || job.sm > this.cluster.smCap) { this.rejected.push({ job }); return false; }
    this.seq += 1;
    this.queue.push({ job, remaining: job.ticks, preemptions: 0, seq: this.seq, submitTick: this.tick });
    return true;
  }
  tenantLoad(tenant) { return this.running.reduce((a, r) => a + (r.job.tenant === tenant ? r.job.gang : 0), 0);   }
  /** Admission order: class desc, then least-loaded tenant (max-min fairness), then FIFO. */
  order() {
    this.queue.sort((a, b) => (b.job.priority - a.job.priority)
      || (this.tenantLoad(a.job.tenant) - this.tenantLoad(b.job.tenant)) || (a.seq - b.seq));
  }
  /**
   * Admit ready jobs. Preemption runs only when the arriving job does not fit *and* reclaiming
   * strictly lower-class work makes it fit, so capacity is reclaimed only when immediately reused.
   * Each round admits at most one job and then re-orders; the round cap is a liveness assertion.
   */
  admit() {
    const placed = [];
    for (let round = 1; ; round += 1) {
      assert(round <= 512, "admission loop must terminate");
      this.order();
      let progressed = false;
      for (let i = 0; i < this.queue.length && !progressed; i += 1) {
        const entry = this.queue[i];
        let ids = placementFor(this.cluster.caps(), entry.job);
        const victims = !ids && this.preemption ? this.selectVictims(entry.job) : [];
        if (victims.length) {
          this.preempt(victims, entry.job);
          ids = placementFor(this.cluster.caps(), entry.job);
          assert(ids, "reclaimed capacity must admit the incoming job");
        }
        if (!ids) continue;
        this.queue.splice(i, 1);
        entry.ids = ids; entry.startedTick = this.tick;
        this.cluster.move(ids, entry.job, 1);
        this.running.push(entry);
        placed.push(entry.job.id);
        progressed = true;
      }
      if (!progressed) return placed;
    }
  }
  /** Greedy minimal prefix of strictly lower-class victims whose reclamation admits `incoming`. */
  selectVictims(incoming) {
    const candidates = this.running.filter((r) => r.job.priority < incoming.priority)
      .sort((a, b) => (a.job.priority - b.job.priority) || (this.tenantLoad(b.job.tenant) - this.tenantLoad(a.job.tenant))
        || (b.remaining - a.remaining) || (a.job.id < b.job.id ? -1 : 1));
    const victims = [];
    for (const r of candidates) { victims.push(r); if (placementFor(this.view(victims), incoming)) return victims; }
    return [];
  }
  /** Device budgets as they would be with every victim in `victims` reclaimed. */
  view(victims) {
    return this.cluster.caps().map((c, id) => {
      let { mem, sm } = c;
      for (const v of victims) if (v.ids.includes(id)) { mem += v.job.mem; sm += v.job.sm; }
      return { mem, sm };
    });
  }
  preempt(victims, incoming) {
    for (const victim of victims) {
      this.cluster.move(victim.ids, victim.job, -1);
      this.running.splice(this.running.indexOf(victim), 1);
      victim.ids = []; victim.preemptions += 1;
      this.queue.push(victim);
      this.events.push({ tick: this.tick, victim: victim.job, by: incoming });
      this.preemptions += 1;
    }
  }
  execute() {
    const finished = [];
    for (const r of this.running) {
      r.remaining -= 1;
      this.tenantTicks.set(r.job.tenant, (this.tenantTicks.get(r.job.tenant) ?? 0) + r.job.gang);
      for (const id of r.ids) this.log.push({ tick: this.tick, device: id, job: r.job.id, mem: r.job.mem, sm: r.job.sm });
      if (r.remaining === 0) finished.push(r);
    }
    for (const r of finished) {
      this.cluster.move(r.ids, r.job, -1);
      r.ids = []; r.finishedTick = this.tick;
      this.running.splice(this.running.indexOf(r), 1);
      this.done.push(r);
    }
  }
  step() { const placed = this.admit(); this.execute(); this.tick += 1; return placed; }
  deviceTicks(tenant) { return this.tenantTicks.get(tenant) ?? 0; }
  makespan() { return Math.max(0, ...this.done.map((r) => r.finishedTick + 1)); }
}

/** Run a `[{ at, jobs }]` timeline to completion; the cluster must drain inside the horizon. */
function simulate(spec, timeline, { preemption = true, horizon = 400 } = {}) {
  const scheduler = new Scheduler(new Cluster(spec), { preemption });
  const byTick = new Map();
  for (const event of timeline) byTick.set(event.at, (byTick.get(event.at) ?? []).concat(event.jobs));
  const lastSubmit = timeline.reduce((m, e) => Math.max(m, e.at), 0);
  while (scheduler.tick <= horizon) {
    for (const job of byTick.get(scheduler.tick) ?? []) scheduler.submit(job);
    if (scheduler.queue.length === 0 && scheduler.running.length === 0 && scheduler.tick > lastSubmit) break;
    scheduler.step();
  }
  assert(scheduler.queue.length === 0 && scheduler.running.length === 0, "workload must drain inside the horizon");
  return scheduler;
}
/** Independent oracle: rebuild per-(tick, device) occupancy from the log alone. */
function replayLedger(log, spec) {
  const cells = new Map();
  for (const e of log) {
    const key = `${e.tick}:${e.device}`;
    const cell = cells.get(key) ?? { mem: 0, sm: 0, jobs: new Set() };
    assert(!cell.jobs.has(e.job), `job ${e.job} double-booked on device ${e.device} at tick ${e.tick}`);
    cell.jobs.add(e.job); cell.mem += e.mem; cell.sm += e.sm;
    assert(cell.mem <= spec.memCap && cell.sm <= spec.smCap, `device ${e.device} over-committed at tick ${e.tick}`);
    cells.set(key, cell);
  }
  return cells;
}
/** Closed-form lower bound: the longest job, and memory / compute work against capacity. */
function makespanBound(spec, jobs) {
  const work = (f) => jobs.reduce((a, j) => a + j.ticks * f(j) * j.gang, 0);
  return Math.max(Math.max(0, ...jobs.map((j) => j.ticks)),
    Math.ceil(work((j) => j.sm) / (spec.devices * spec.smCap)),
    Math.ceil(work((j) => j.mem) / (spec.devices * spec.memCap)));
}

/** Ledger oracle plus work conservation, tenant accounting and full release of every device. */
function audit(scheduler, spec) {
  replayLedger(scheduler.log, spec);
  let logged = 0;
  for (const r of scheduler.done) {
    const entries = scheduler.log.filter((e) => e.job === r.job.id).length;
    assert(entries === r.job.ticks * r.job.gang, `job ${r.job.id} ran ${entries} of ${r.job.ticks * r.job.gang} ticks`);
    logged += entries;
  }  assert([...scheduler.tenantTicks.values()].reduce((a, b) => a + b, 0) === logged, "tenant accounting equals the log");
  for (const d of scheduler.cluster.devices) assert(d.mem === spec.memCap && d.sm === spec.smCap, `device ${d.id} released`);
  return logged;
}
const job = (id, tenant, priority, mem, sm, ticks, gang = 1) => ({ id, tenant, priority, mem, sm, ticks, gang });
function main() {
  const roomy = { devices: 4, memCap: 16, smCap: 8 }, one = { devices: 1, memCap: 8, smCap: 4 };
  // 1. Room to spare: six jobs overlap, nothing is preempted, the packing is optimal.
  const batch = Array.from({ length: 6 }, (_, i) => job(`b${i}`, "batch", STANDARD, 4, 2, 3));
  const run1 = simulate(roomy, [{ at: 0, jobs: batch }]);
  assert(run1.done.length === 6 && run1.preemptions === 0, "roomy run completes without preemption");
  assert(run1.makespan() === 3 && run1.makespan() === makespanBound(roomy, batch), "optimal packing at the lower bound");
  assert(audit(run1, roomy) === 18, "each job executed its three device-ticks");
  // 2. Boundary: a footprint bigger than one device is rejected; invalid jobs throw.
  const run2 = simulate(roomy, [{ at: 0, jobs: [job("huge", "batch", CRITICAL, 64, 8, 5), ...batch] }]);
  const fresh = () => new Scheduler(new Cluster(roomy));
  assert(run2.rejected[0]?.job.id === "huge" && run2.preemptions === 0 && run2.done.length === 6, "oversized job rejected, cluster untouched");
  for (const bad of [job("z", "t", 1, 0, 2, 1), job("z", "t", 1, 4, 0, 1), job("z", "t", 1, 4, 2, 0)]) assert.throws(() => fresh().submit(bad), /invalid job/);
  assert.throws(() => fresh().submit(job("g", "t", 1, 4, 2, 1, 5)), /gang cannot fit/);
  // 3. Preemption, memory reclamation and resume on a single device.
  const lowA = job("A", "batch", STANDARD, 8, 4, 4), lowB = job("B", "batch", STANDARD, 8, 4, 4);
  const run3 = simulate(one, [{ at: 0, jobs: [lowA, lowB] }, { at: 1, jobs: [job("C", "rt", CRITICAL, 8, 4, 2)] }]);
  const [event] = run3.events;
  assert(run3.preemptions === 1 && event.victim.id === "A" && event.by.priority > event.victim.priority, "critical arrival preempts strictly lower-class A");
  const victim = run3.done.find((r) => r.job.id === "A");
  assert(victim.preemptions === 1 && victim.remaining === 0, "victim resumes and keeps its remaining work");
  assert(victim.startedTick === 3 && victim.finishedTick === 5, "victim restarts right after the urgent job");
  assert(run3.done.find((r) => r.job.id === "C").finishedTick === 2 && run3.done.length === 3, "urgent job finishes first, queue drains");
  assert(audit(run3, one) === 10, "preempted work is conserved (4+4+2 device-ticks)");
  // 4. Equals never preempt each other: three standard jobs share one slot.
  const run4 = simulate(one, [{ at: 0, jobs: [lowA, lowB, job("D", "batch", STANDARD, 8, 4, 2)] }]);
  assert(run4.preemptions === 0 && run4.done.length === 3, "equal classes never preempt one another");

  // 5. Max-min fairness: three equal-weight tenants share two slots of one device.
  const tenants = ["t1", "t2", "t3"], fair = tenants.flatMap((t) => [0, 1].map((i) => job(`${t}-${i}`, t, STANDARD, 4, 2, 4)));
  const run5 = simulate(one, [{ at: 0, jobs: fair }]);
  const shares = tenants.map((t) => run5.deviceTicks(t));
  assert(shares.reduce((a, b) => a + b, 0) === 24 && Math.max(...shares) / Math.min(...shares) <= 1.35, `shares balanced: ${shares}`);
  assert(run5.makespan() === makespanBound(one, fair), "fair run reaches the lower bound too");
  // 6. Seeded mixed-class workload: policy-independent work, preemption always upward.
  const rng = splitmix64(20240935);
  const mixed = Array.from({ length: 14 }, (_, i) => job(`m${i}`, `tenant${rng.int(3)}`, rng.int(4), 4 + 4 * rng.int(3), 2 + 2 * rng.int(2), 2 + rng.int(5)));
  const spec = { devices: 2, memCap: 16, smCap: 8 };
  const preemptive = simulate(spec, [{ at: 0, jobs: mixed }]);
  const delivered = audit(simulate(spec, [{ at: 0, jobs: mixed }], { preemption: false }), spec);
  assert(delivered === audit(preemptive, spec), "delivered device-ticks are policy independent");
  assert(delivered === mixed.reduce((a, j) => a + j.ticks * j.gang, 0), "every submitted job is executed in full");
  const waits = preemptive.done.map((r) => r.startedTick - r.submitTick);
  assert(preemptive.events.every((e) => e.by.priority > e.victim.priority) && preemptive.done.length === mixed.length && Math.max(...waits) <= 60, `seeded run drains, waits bounded (${Math.max(...waits)})`);
  console.log(`gpu-scheduler: preemptions=${preemptive.preemptions + run3.preemptions} device-ticks=${delivered} shares=${shares.join("/")}`);
  console.log("PASS");
}
module.exports = { splitmix64, Cluster, Scheduler, simulate, replayLedger, makespanBound, audit, placementFor,
  FREE, STANDARD, PREMIUM, CRITICAL };
if (require.main === module) main();
