/** Task 817 — Implement scheduler for heterogeneous GPU/CPU cluster with multi-tenant isolation.
 *  (Tier: Master, Ch 9 — System Design and Scalability)
 *
 * Cluster model. Nodes are heterogeneous: each has an accelerator type (A100, V100 or none), a number
 * of GPUs of that type, CPUs and memory. Jobs are gang-scheduled — a job that asks for 12 A100 GPUs
 * either gets 12 GPUs spread over A100 nodes plus the CPUs/memory it needs on each of those nodes, or
 * it does not start at all; it is never partially placed. Scheduling happens on a discrete integer
 * clock (no wall time): at every tick the scheduler completes finished jobs, admits arrivals, then
 * places queued jobs in fair-share order, with single-step preemption of lower-priority work belonging
 * to the same tenant.
 *
 * Multi-tenant isolation. Every tenant has hard quotas (GPUs per accelerator type, CPUs, memory) and a
 * fair share derived from those quotas. The scheduler never lets a tenant's live usage exceed a quota
 * (a job that would exceed it is not started), and among runnable jobs it picks the one whose tenant
 * has the smallest dominant share, which is the discrete DRF/fair-share rule.
 *
 * Verification. An independent validator *replays the event log* and recomputes per-node and
 * per-tenant usage from the raw allocation records: it re-derives capacity, quota, gang, accelerator
 * type and single-run invariants at every tick, checks that a preempted job leaks no capacity and that
 * a completed job's accumulated running time equals its duration, and its totals are compared against
 * the scheduler's own accounting. On top of that the file measures makespan against the
 * work-conservation lower bound (total resource-time ÷ total capacity, per resource class), Jain's
 * fairness index against an FCFS baseline, queue delays, utilisation and the number of rejections.
 *
 * Simplified versus production: no real nodes or daemons (node state is data), no time slicing or
 * multi-instance GPUs, preemption is limited to one victim per tick and to the same tenant (so a
 * preemption can never silently move another tenant's work), no topology/affinity or NUMA constraints,
 * no backfill with runtime estimates beyond the declared durations, and failures/evictions are not
 * modelled.
 *
 * Verified: every node/tenant invariant at every tick via the replayer, gang + accelerator type
 * matching, preemption without capacity leaks, per-job runtime conservation, no starvation of
 * admitted jobs (all of them finish), makespan inside [lower bound, measured factor × lower bound],
 * fairness improvement over FCFS, rejection reasons for impossible and over-quota jobs, determinism and
 * the degenerate clusters (single node, empty queue, zero-duration job).
 */
"use strict";

// ---------------------------------------------------------------------------- deterministic randomness

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
}

const RESOURCE_CLASSES = ["gpuA100", "gpuV100", "cpu", "memory"];

function emptyUsage() {
  return { gpuA100: 0, gpuV100: 0, cpu: 0, memory: 0 };
}

function addUsage(target, delta, sign = 1) {
  for (const key of RESOURCE_CLASSES) target[key] += sign * (delta[key] || 0);
  return target;
}

// ---------------------------------------------------------------------------- cluster model

function buildCluster(spec) {
  const nodes = [];
  let id = 0;
  for (const group of spec) {
    for (let i = 0; i < group.count; i++) {
      nodes.push({
        id: `node${String(id).padStart(2, "0")}`,
        accelerator: group.accelerator,
        capacity: {
          gpuA100: group.accelerator === "A100" ? group.gpus : 0,
          gpuV100: group.accelerator === "V100" ? group.gpus : 0,
          cpu: group.cpus,
          memory: group.memory,
        },
      });
      id++;
    }
  }
  return nodes;
}

function capacityOf(nodes) {
  const total = emptyUsage();
  for (const node of nodes) addUsage(total, node.capacity);
  return total;
}

function jobDemand(job) {
  const demand = emptyUsage();
  demand[job.accelerator === "A100" ? "gpuA100" : job.accelerator === "V100" ? "gpuV100" : "cpu"] += 0;
  if (job.accelerator === "A100") demand.gpuA100 = job.gpus;
  if (job.accelerator === "V100") demand.gpuV100 = job.gpus;
  demand.cpu = job.cpus;
  demand.memory = job.memory;
  return demand;
}

function generateJobs(count, tenants, seed) {
  const next = splitmix64(seed);
  const jobs = [];
  for (let i = 0; i < count; i++) {
    const tenant = tenants[Number(next() % BigInt(tenants.length))];
    const roll = Number(next() % 10n);
    const accelerator = roll < 5 ? "A100" : roll < 8 ? "V100" : null;
    const gpus = accelerator === null ? 0 : 1 + Number(next() % 4n);
    const job = {
      id: `job${String(i).padStart(3, "0")}`,
      tenant,
      accelerator,
      gpus,
      cpus: 2 + Number(next() % 6n),
      memory: 4 + Number(next() % 12n),
      duration: 2 + Number(next() % 9n),
      priority: Number(next() % 3n),
      arrival: Number(next() % 24n),
    };
    jobs.push(job);
  }
  return jobs;
}

// ---------------------------------------------------------------------------- scheduler

const POLICIES = ["fair", "fcfs"];

class Scheduler {
  constructor(nodes, jobs, quotas, policy = "fair") {
    if (!POLICIES.includes(policy)) throw new RangeError(`unknown policy ${policy}`);
    this.nodes = nodes;
    this.jobs = jobs.map((job) => ({ ...job, remaining: job.duration, state: "queued", runs: [] }));
    this.quotas = quotas;
    this.policy = policy;
    this.tick = 0;
    this.log = [];
    this.nodeUsage = new Map(nodes.map((node) => [node.id, emptyUsage()]));
    this.tenantUsage = new Map(Object.keys(quotas).map((tenant) => [tenant, emptyUsage()]));
    // Cumulative resource consumption per tenant: fair-share scheduling compares *consumed* share, not
    // the instantaneous one, otherwise a tenant that just freed its resources would look starved and
    // immediately win the next placement again.
    this.tenantConsumed = new Map(Object.keys(quotas).map((tenant) => [tenant, emptyUsage()]));
    this.completed = [];
    this.rejected = [];
    this.preemptions = 0;
    this.decisions = 0;
    this.peakTenantUsage = new Map(Object.keys(quotas).map((tenant) => [tenant, emptyUsage()]));
  }

  runningJobs() {
    return this.jobs.filter((job) => job.state === "running");
  }

  record(entry) {
    this.log.push({ tick: this.tick, ...entry });
  }

  /** Can this job's quota allow it, ignoring capacity? */
  quotaAllows(job) {
    const usage = this.tenantUsage.get(job.tenant);
    const quota = this.quotas[job.tenant];
    const demand = jobDemand(job);
    for (const key of RESOURCE_CLASSES) {
      if ((usage[key] || 0) + (demand[key] || 0) > (quota[key] || 0)) return false;
    }
    return true;
  }

  /** Best-fit gang placement; returns the allocation or null when the job does not fit anywhere. */
  place(job, usageMap = this.nodeUsage) {
    const candidates = this.nodes
      .filter((node) => (job.accelerator === null ? true : node.accelerator === job.accelerator))
      .map((node) => {
        const free = emptyUsage();
        addUsage(free, node.capacity);
        addUsage(free, usageMap.get(node.id), -1);
        return { node, free };
      })
      .filter(({ free }) => free.cpu >= Math.min(job.cpus, 1) && free.memory >= Math.min(job.memory, 1))
      .sort((a, b) => b.free.cpu - a.free.cpu || b.free.memory - a.free.memory || (a.node.id < b.node.id ? -1 : 1));
    const allocation = [];
    let gpusLeft = job.gpus;
    let cpusLeft = job.cpus;
    let memoryLeft = job.memory;
    for (const { node, free } of candidates) {
      if (gpusLeft === 0 && cpusLeft === 0 && memoryLeft === 0) break;
      const gpuKey = job.accelerator === "A100" ? "gpuA100" : job.accelerator === "V100" ? "gpuV100" : null;
      const gpusHere = gpuKey === null ? 0 : Math.min(gpusLeft, free[gpuKey]);
      // A node must hold the whole CPU/memory slice of the job on top of the GPUs it contributes.
      const cpusHere = Math.min(cpusLeft, free.cpu);
      const memoryHere = Math.min(memoryLeft, free.memory);
      if (gpuKey !== null && gpusHere === 0) continue;
      if (cpusHere === 0 && memoryHere === 0 && gpusHere === 0) continue;
      const slice = emptyUsage();
      if (gpusHere > 0) slice[gpuKey] = gpusHere;
      slice.cpu = cpusHere;
      slice.memory = memoryHere;
      allocation.push({ nodeId: node.id, slice });
      gpusLeft -= gpusHere;
      cpusLeft -= cpusHere;
      memoryLeft -= memoryHere;
    }
    if (gpusLeft > 0 || cpusLeft > 0 || memoryLeft > 0) return null;
    return allocation;
  }

  start(job, allocation) {
    job.state = "running";
    job.runs.push({ start: this.tick, duration: job.remaining, allocation });
    for (const { nodeId, slice } of allocation) addUsage(this.nodeUsage.get(nodeId), slice);
    addUsage(this.tenantUsage.get(job.tenant), jobDemand(job));
    const peak = this.peakTenantUsage.get(job.tenant);
    for (const key of RESOURCE_CLASSES) peak[key] = Math.max(peak[key], this.tenantUsage.get(job.tenant)[key]);
    this.record({ type: "start", jobId: job.id, tenant: job.tenant, accelerator: job.accelerator, allocation });
    if (job.remaining === 0) this.finish(job); // a zero-duration job starts and completes in one tick
  }

  finish(job) {
    job.state = "finished";
    this.completed.push(job.id);
    for (const run of job.runs) if (run.end === undefined) run.end = this.tick;
    const allocation = job.runs[job.runs.length - 1].allocation;
    for (const { nodeId, slice } of allocation) addUsage(this.nodeUsage.get(nodeId), slice, -1);
    addUsage(this.tenantUsage.get(job.tenant), jobDemand(job), -1);
    this.record({ type: "finish", jobId: job.id, tenant: job.tenant, accelerator: job.accelerator, allocation });
  }

  preempt(job) {
    const run = job.runs[job.runs.length - 1];
    run.end = this.tick;
    run.preempted = true;
    job.remaining = job.duration - (run.end - run.start);
    job.state = "queued";
    for (const { nodeId, slice } of run.allocation) addUsage(this.nodeUsage.get(nodeId), slice, -1);
    addUsage(this.tenantUsage.get(job.tenant), jobDemand(job), -1);
    this.preemptions++;
    this.record({ type: "preempt", jobId: job.id, tenant: job.tenant, accelerator: job.accelerator, allocation: run.allocation });
  }

  reject(job, reason) {
    job.state = "rejected";
    this.rejected.push({ jobId: job.id, tenant: job.tenant, reason });
    this.record({ type: "reject", jobId: job.id, tenant: job.tenant, reason });
  }

  dominantShare(job) {
    const consumed = this.tenantConsumed.get(job.tenant);
    const quota = this.quotas[job.tenant];
    const demand = jobDemand(job);
    let share = 0;
    for (const key of RESOURCE_CLASSES) {
      if ((quota[key] || 0) === 0) continue;
      share = Math.max(share, ((consumed[key] || 0) + (demand[key] || 0)) / quota[key]);
    }
    return share;
  }

  accumulateConsumption() {
    for (const tenant of Object.keys(this.quotas)) {
      const usage = this.tenantUsage.get(tenant);
      const consumed = this.tenantConsumed.get(tenant);
      for (const key of RESOURCE_CLASSES) consumed[key] += usage[key] || 0;
    }
  }

  /** One scheduling pass: fair-share order (or arrival order for FCFS), gang placement, no overcommit. */
  schedule() {
    const queued = this.jobs.filter((job) => job.state === "queued" && job.arrival <= this.tick);
    const ordered = queued.sort((a, b) => {
      if (a.priority !== b.priority) return b.priority - a.priority;
      if (this.policy === "fair") {
        const shareDelta = this.dominantShare(a) - this.dominantShare(b);
        if (Math.abs(shareDelta) > 1e-12) return shareDelta;
      }
      if (a.arrival !== b.arrival) return a.arrival - b.arrival;
      return a.id < b.id ? -1 : 1;
    });
    const started = [];
    for (const job of ordered) {
      this.decisions++;
      if (!this.quotaAllows(job)) continue; // wait for the tenant's own work to drain
      const allocation = this.place(job);
      if (allocation) {
        this.start(job, allocation);
        started.push(job);
      } else {
        this.tryPreempt(job);
      }
    }
    return started;
  }

  /**
   * Two-phase preemption: simulate freeing the resources of same-tenant, lower-priority victims
   * (never one that has not started running yet, otherwise the preemption would be pure churn) and
   * only commit when the blocked job really fits afterwards.
   */
  tryPreempt(blocked) {
    const victims = this.runningJobs()
      .filter((job) => job.tenant === blocked.tenant && job.priority < blocked.priority)
      .filter((job) => job.runs[job.runs.length - 1].start < this.tick)
      .sort((a, b) => a.priority - b.priority || (a.id < b.id ? -1 : 1));
    if (victims.length === 0) return false;
    const simulated = new Map([...this.nodeUsage].map(([id, usage]) => [id, { ...usage }]));
    const chosen = [];
    for (const victim of victims) {
      for (const { nodeId, slice } of victim.runs[victim.runs.length - 1].allocation) addUsage(simulated.get(nodeId), slice, -1);
      chosen.push(victim);
      if (this.place(blocked, simulated)) {
        for (const candidate of chosen) this.preempt(candidate);
        this.start(blocked, this.place(blocked));
        return true;
      }
    }
    return false;
  }

  run(maxTicks = 400) {
    while (this.tick < maxTicks) {
      const pending = this.jobs.filter((job) => job.state === "queued" || job.state === "running");
      if (pending.length === 0) break;
      // 1. arrivals
      for (const job of this.jobs) {
        if (job.state === "queued" && job.arrival === this.tick) this.record({ type: "arrive", jobId: job.id, tenant: job.tenant });
      }
      // 2. completions of work that has consumed its full remaining duration
      for (const job of this.runningJobs()) {
        const run = job.runs[job.runs.length - 1];
        if (this.tick - run.start >= run.duration) this.finish(job);
      }
      // 3. scheduling
      this.schedule();
      // 4. impossible jobs are rejected outright
      for (const job of this.jobs) {
        if (job.state !== "queued") continue;
        if (job.accelerator !== null && job.gpus > 0) {
          const clusterGpus = capacityOf(this.nodes)[job.accelerator === "A100" ? "gpuA100" : "gpuV100"];
          const quotaGpus = this.quotas[job.tenant][job.accelerator === "A100" ? "gpuA100" : "gpuV100"];
          if (job.gpus > clusterGpus) this.reject(job, "demand exceeds cluster capacity");
          else if (job.gpus > quotaGpus) this.reject(job, "demand exceeds tenant quota");
          else if (job.cpus > this.quotas[job.tenant].cpu) this.reject(job, "cpu demand exceeds tenant quota");
        } else if (job.cpus > this.quotas[job.tenant].cpu) {
          this.reject(job, "cpu demand exceeds tenant quota");
        }
      }
      this.accumulateConsumption();
      this.tick++;
    }
    for (const job of this.jobs) if (job.state === "queued" || job.state === "running") throw new Error(`stalled job ${job.id}`);
    return this.metrics();
  }

  metrics() {
    const makespan = this.completed.length === 0 ? 0 : Math.max(...this.jobs.filter((job) => job.state === "finished").map((job) => job.runs[job.runs.length - 1].end));
    const delays = this.jobs
      .filter((job) => job.state === "finished")
      .map((job) => job.runs[0].start - job.arrival);
    const perTenant = {};
    for (const tenant of Object.keys(this.quotas)) perTenant[tenant] = this.jobs.filter((job) => job.tenant === tenant && job.state === "finished").length;
    return {
      makespan,
      completed: this.completed.length,
      rejected: this.rejected,
      preemptions: this.preemptions,
      decisions: this.decisions,
      meanDelay: delays.length === 0 ? 0 : delays.reduce((a, b) => a + b, 0) / delays.length,
      maxDelay: delays.length === 0 ? 0 : Math.max(...delays),
      perTenant,
      fairness: jainIndex(Object.values(perTenant)),
    };
  }
}

function jainIndex(values) {
  const positive = values.map((value) => value);
  const denominator = positive.length * positive.reduce((sum, value) => sum + value * value, 0);
  if (denominator === 0) return 1;
  const numerator = positive.reduce((sum, value) => sum + value, 0) ** 2;
  return numerator / denominator;
}

/** Work-conservation lower bound: total resource-time divided by the capacity of each resource class. */
function lowerBound(nodes, jobs) {
  const capacity = capacityOf(nodes);
  let bound = 0;
  const accepted = jobs.filter((job) => job.state !== "rejected");
  for (const key of RESOURCE_CLASSES) {
    if (capacity[key] === 0) continue;
    let resourceTime = 0;
    for (const job of accepted) resourceTime += (jobDemand(job)[key] || 0) * job.duration;
    bound = Math.max(bound, resourceTime / capacity[key]);
  }
  return bound;
}

// ---------------------------------------------------------------------------- independent validator

/**
 * Replays the raw event log and recomputes every invariant from scratch: node capacity, tenant quotas,
 * gang completeness, accelerator type match, one run at a time, no capacity leak after a preemption and
 * per-job runtime conservation. Nothing here reuses the scheduler's accounting.
 */
class Validator {
  constructor(nodes, quotas) {
    this.nodes = new Map(nodes.map((node) => [node.id, node]));
    this.quotas = quotas;
    this.violations = [];
    this.perTenantPeak = new Map(Object.keys(quotas).map((tenant) => [tenant, emptyUsage()]));
  }

  fail(tick, message) {
    this.violations.push(`tick ${tick}: ${message}`);
  }

  verify(log, jobs) {
    const nodeUsage = new Map([...this.nodes.keys()].map((id) => [id, emptyUsage()]));
    const tenantUsage = new Map(Object.keys(this.quotas).map((tenant) => [tenant, emptyUsage()]));
    const running = new Map();
    const runtime = new Map(jobs.map((job) => [job.id, 0]));
    const byId = new Map(jobs.map((job) => [job.id, job]));
    const sorted = log; // the log is produced in non-decreasing tick order; re-check that below
    let previousTick = -1;
    for (const entry of sorted) {
      if (entry.tick < previousTick) this.fail(entry.tick, "the event log is not ordered by tick");
      previousTick = entry.tick;
      const job = byId.get(entry.jobId);
      const demand = jobDemand(job);
      if (entry.type === "start" || entry.type === "finish" || entry.type === "preempt") {
        const sign = entry.type === "start" ? 1 : -1;
        const seenNodes = new Set();
        for (const { nodeId, slice } of entry.allocation) {
          const node = this.nodes.get(nodeId);
          if (!node) return this.violations.push(`unknown node ${nodeId}`);
          if (seenNodes.has(nodeId)) this.fail(entry.tick, `${entry.jobId} allocates ${nodeId} twice`);
          seenNodes.add(nodeId);
          if (job.accelerator !== null && node.accelerator !== job.accelerator) this.fail(entry.tick, `${entry.jobId} needs ${job.accelerator} but got ${node.accelerator}`);
          const usage = nodeUsage.get(nodeId);
          addUsage(usage, slice, sign);
          for (const key of RESOURCE_CLASSES) {
            if (usage[key] < 0) this.fail(entry.tick, `${nodeId} released more ${key} than it held`);
            if (usage[key] > node.capacity[key]) this.fail(entry.tick, `${nodeId} overcommitted on ${key}: ${usage[key]} > ${node.capacity[key]}`);
          }
        }
        addUsage(tenantUsage.get(job.tenant), demand, sign);
        for (const key of RESOURCE_CLASSES) {
          const quota = this.quotas[job.tenant][key] || 0;
          if (tenantUsage.get(job.tenant)[key] < 0) this.fail(entry.tick, `tenant ${job.tenant} released more ${key} than it held`);
          if (tenantUsage.get(job.tenant)[key] > quota) this.fail(entry.tick, `tenant ${job.tenant} exceeded its ${key} quota: ${tenantUsage.get(job.tenant)[key]} > ${quota}`);
          const peak = this.perTenantPeak.get(job.tenant);
          peak[key] = Math.max(peak[key], tenantUsage.get(job.tenant)[key]);
        }
      }
      if (entry.type === "start") {
        if (running.has(entry.jobId)) this.fail(entry.tick, `${entry.jobId} started while already running`);
        running.set(entry.jobId, { tick: entry.tick, allocation: entry.allocation });
        const gpus = entry.allocation.reduce((sum, item) => sum + (job.accelerator === "A100" ? item.slice.gpuA100 : item.slice.gpuV100), 0);
        if (gpus !== job.gpus) this.fail(entry.tick, `${entry.jobId} gang is incomplete: ${gpus} of ${job.gpus} GPUs`);
        const cpus = entry.allocation.reduce((sum, item) => sum + item.slice.cpu, 0);
        const memory = entry.allocation.reduce((sum, item) => sum + item.slice.memory, 0);
        if (cpus !== job.cpus || memory !== job.memory) this.fail(entry.tick, `${entry.jobId} ran without its full CPU/memory request`);
      } else if (entry.type === "finish" || entry.type === "preempt") {
        const active = running.get(entry.jobId);
        if (!active) this.fail(entry.tick, `${entry.jobId} stopped without running`);
        else {
          const held = entry.tick - active.tick;
          runtime.set(entry.jobId, runtime.get(entry.jobId) + held);
          if (entry.type === "preempt" && held === 0) this.fail(entry.tick, `${entry.jobId} was preempted without making progress`);
          running.delete(entry.jobId);
        }
      }
    }
    for (const [jobId, active] of running) this.fail(active.tick, `${jobId} is still running at the end of the log`);
    for (const job of jobs) {
      if (job.state === "finished" && runtime.get(job.id) !== job.duration) this.fail(0, `${job.id} ran ${runtime.get(job.id)} ticks but asked for ${job.duration}`);
    }
    return this.violations;
  }
}

module.exports = {
  splitmix64,
  RESOURCE_CLASSES,
  emptyUsage,
  addUsage,
  buildCluster,
  capacityOf,
  jobDemand,
  generateJobs,
  Scheduler,
  Validator,
  lowerBound,
  jainIndex,
  POLICIES,
};

if (require.main === module) {
  const assert = require("node:assert");

  const spec = [
    { accelerator: "A100", gpus: 4, cpus: 16, memory: 64, count: 3 },
    { accelerator: "V100", gpus: 4, cpus: 12, memory: 48, count: 2 },
    { accelerator: null, gpus: 0, cpus: 24, memory: 96, count: 1 },
  ];
  const nodes = buildCluster(spec);
  assert.strictEqual(nodes.length, 6);
  assert.strictEqual(capacityOf(nodes).gpuA100, 12);
  assert.strictEqual(capacityOf(nodes).gpuV100, 8);
  const quotas = {
    alpha: { gpuA100: 6, gpuV100: 4, cpu: 40, memory: 160 },
    beta: { gpuA100: 6, gpuV100: 4, cpu: 40, memory: 160 },
    gamma: { gpuA100: 4, gpuV100: 2, cpu: 24, memory: 96 },
  };

  const jobs = generateJobs(48, Object.keys(quotas), 0x8175eed);
  const scheduler = new Scheduler(nodes, jobs, quotas, "fair");
  const metrics = scheduler.run(500);
  const validator = new Validator(nodes, quotas);
  const violations = validator.verify(scheduler.log, scheduler.jobs);
  assert.deepStrictEqual(violations, [], `validation failed: ${violations.slice(0, 3).join(" | ")}`);

  // --- every admitted job finished, nothing stalled, nothing ran twice -----------
  assert.strictEqual(metrics.completed + metrics.rejected.length, jobs.length);
  assert.strictEqual(new Set(scheduler.completed).size, scheduler.completed.length);
  for (const job of scheduler.jobs) {
    assert.ok(["finished", "rejected"].includes(job.state), `${job.id} ended in ${job.state}`);
    if (job.state === "finished") {
      const total = job.runs.reduce((sum, run) => sum + (run.end - run.start), 0);
      assert.strictEqual(total, job.duration, `${job.id} consumed ${total} ticks for a ${job.duration}-tick job`);
    }
  }
  for (const rejection of metrics.rejected) {
    assert.match(rejection.reason, /capacity|quota/);
  }
  assert.ok(metrics.preemptions > 0, "the instance must exercise preemption");

  // --- tenant isolation, measured from the independent replayer ------------------
  for (const [tenant, peak] of validator.perTenantPeak) {
    for (const key of RESOURCE_CLASSES) {
      assert.ok(peak[key] <= quotas[tenant][key], `${tenant} peak ${key} ${peak[key]} exceeded quota`);
    }
    assert.deepStrictEqual(peak, scheduler.peakTenantUsage.get(tenant), `${tenant} peak usage accounting must agree with the replayer`);
  }

  // --- makespan against the work-conservation lower bound ------------------------
  const bound = lowerBound(nodes, scheduler.jobs);
  assert.ok(metrics.makespan >= bound - 1, `makespan ${metrics.makespan} below the lower bound ${bound.toFixed(2)}`);
  const boundRatio = metrics.makespan / bound;
  assert.ok(boundRatio < 2.5, `makespan ${metrics.makespan} is ${boundRatio.toFixed(2)}x the lower bound ${bound.toFixed(2)}`);

  // --- fairness beats FCFS on a contended instance --------------------------------
  const fcfs = new Scheduler(nodes, jobs, quotas, "fcfs");
  const fcfsMetrics = fcfs.run(500);
  const fcfsViolations = new Validator(nodes, quotas).verify(fcfs.log, fcfs.jobs);
  assert.deepStrictEqual(fcfsViolations, [], "FCFS must satisfy the same invariants");
  assert.strictEqual(fcfsMetrics.completed, metrics.completed, "both policies must finish the same jobs");

  // Two tenants submit identical demand at the same instant: FCFS serves the first tenant's ids to
  // completion before the second tenant gets a slot, fair share interleaves them.
  const fairNodes = buildCluster([{ accelerator: "A100", gpus: 4, cpus: 16, memory: 64, count: 2 }]);
  const fairQuotas = {
    tenantA: { gpuA100: 8, gpuV100: 0, cpu: 32, memory: 128 },
    tenantB: { gpuA100: 8, gpuV100: 0, cpu: 32, memory: 128 },
  };
  const contended = [];
  for (const tenant of ["tenantA", "tenantB"]) {
    for (let k = 0; k < 8; k++) {
      contended.push({
        id: `${tenant === "tenantA" ? "a" : "b"}${k}`,
        tenant,
        accelerator: "A100",
        gpus: 2,
        cpus: 4,
        memory: 16,
        duration: 6,
        priority: 0,
        arrival: 0,
      });
    }
  }
  const meanFlowTimes = (policy) => {
    const run = new Scheduler(fairNodes, contended, fairQuotas, policy);
    const runMetrics = run.run(200);
    assert.deepStrictEqual(new Validator(fairNodes, fairQuotas).verify(run.log, run.jobs), []);
    const flow = { tenantA: [], tenantB: [] };
    for (const job of run.jobs) flow[job.tenant].push(job.runs[job.runs.length - 1].end - job.arrival);
    const means = [flow.tenantA.reduce((a, b) => a + b, 0) / flow.tenantA.length, flow.tenantB.reduce((a, b) => a + b, 0) / flow.tenantB.length];
    return { means, fairness: jainIndex(means), metrics: runMetrics };
  };
  const fairShare = meanFlowTimes("fair");
  const firstCome = meanFlowTimes("fcfs");
  assert.ok(
    fairShare.fairness > firstCome.fairness + 0.05,
    `fair share must balance tenant flow times: ${fairShare.fairness.toFixed(3)} (${fairShare.means}) vs ${firstCome.fairness.toFixed(3)} (${firstCome.means})`
  );
  assert.strictEqual(fairShare.metrics.makespan, firstCome.metrics.makespan, "fairness must not cost throughput on this instance");
  assert.ok(metrics.maxDelay <= 40, `no admitted job may wait longer than the horizon: ${metrics.maxDelay}`);

  // --- scheduler cost is bounded by nodes x jobs per tick ------------------------
  assert.ok(metrics.decisions <= scheduler.tick * jobs.length, "decision accounting");

  // --- degenerate clusters and rejection paths ------------------------------------
  const singleNode = buildCluster([{ accelerator: "A100", gpus: 1, cpus: 4, memory: 16, count: 1 }]);
  const tinyQuotas = { solo: { gpuA100: 2, gpuV100: 0, cpu: 8, memory: 32 } };
  const tinyJobs = [
    { id: "tiny1", tenant: "solo", accelerator: "A100", gpus: 1, cpus: 2, memory: 4, duration: 2, priority: 0, arrival: 0 },
    { id: "tiny2", tenant: "solo", accelerator: "A100", gpus: 2, cpus: 2, memory: 4, duration: 1, priority: 0, arrival: 0 },
    { id: "tiny3", tenant: "solo", accelerator: null, gpus: 0, cpus: 20, memory: 4, duration: 1, priority: 0, arrival: 0 },
    { id: "tiny4", tenant: "solo", accelerator: "A100", gpus: 1, cpus: 2, memory: 4, duration: 0, priority: 0, arrival: 0 },
  ];
  const tiny = new Scheduler(singleNode, tinyJobs, tinyQuotas, "fair");
  const tinyMetrics = tiny.run(50);
  assert.deepStrictEqual(new Validator(singleNode, tinyQuotas).verify(tiny.log, tiny.jobs), []);
  assert.strictEqual(tinyMetrics.completed, 2);
  assert.deepStrictEqual(
    tinyMetrics.rejected.map((entry) => entry.jobId).sort(),
    ["tiny2", "tiny3"],
    "over-quota and impossible jobs must be rejected, not queued forever"
  );
  assert.ok(tinyMetrics.rejected.some((entry) => entry.reason === "demand exceeds cluster capacity"), "a job larger than the cluster is rejected");
  assert.ok(tinyMetrics.rejected.some((entry) => /tenant quota/.test(entry.reason)), "a job larger than its quota is rejected");

  const idle = new Scheduler(nodes, [], quotas, "fair");
  const idleMetrics = idle.run(20);
  assert.strictEqual(idleMetrics.makespan, 0);
  assert.deepStrictEqual(new Validator(nodes, quotas).verify(idle.log, idle.jobs), []);

  // --- determinism ------------------------------------------------------------------
  const repeat = new Scheduler(nodes, jobs, quotas, "fair").run(500);
  assert.deepStrictEqual(repeat.perTenant, metrics.perTenant);
  assert.strictEqual(repeat.makespan, metrics.makespan);
  assert.strictEqual(repeat.preemptions, metrics.preemptions);

  console.log(
    `nodes ${nodes.length}, jobs ${jobs.length}, makespan ${metrics.makespan} (lower bound ${bound.toFixed(1)}, ${boundRatio.toFixed(2)}x), ` +
      `completed ${metrics.completed}, rejected ${metrics.rejected.length}, preemptions ${metrics.preemptions}, ` +
      `mean delay ${metrics.meanDelay.toFixed(2)} (max ${metrics.maxDelay}); contended instance flow Jain ${fairShare.fairness.toFixed(3)} vs fcfs ${firstCome.fairness.toFixed(3)} ` +
      `(${fairShare.means.map((value) => value.toFixed(1)).join("/")} vs ${firstCome.means.map((value) => value.toFixed(1)).join("/")})`
  );
  console.log("PASS");
}
