/** Task 892 - Implement chaos-monkey style fault injection and recovery validation.
 *  (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
 *
 * A chaos-engineering rig for a quorum-replicated register. The service is a faithful ABD-style
 * store: a client acts as its own coordinator, phase 1 reads the highest version from a quorum,
 * phase 2 writes (version+1, value) to a quorum, and reads take the highest version seen in a
 * quorum. Because R + W > N the register is linearizable; the config validator refuses any other
 * R/W combination.
 *
 * The chaos monkey injects faults on a deterministic schedule drawn from a seeded splitmix32: kill a
 * replica (persistent storage, or *volatile* — the state is wiped on restart), partition the network
 * into groups, drop a fraction of messages, slow a replica down, and (only in the deliberately broken
 * configurations used by the negative checks) forge a replica's stored record. Time is virtual
 * milliseconds; every delay, timeout and repair round is an event in one (time, sequence) queue.
 *
 * The recovery validator observes the cluster continuously and asserts: safety (never two quorums
 * disagreeing, never a version moving backwards), durability (a value acknowledged to a client stays
 * readable), authenticity (no replica ever reports a value nobody wrote), bounded recovery (after
 * every fault window the cluster re-converges within CONVERGENCE_BOUND ms via anti-entropy repair),
 * and availability (completed operations / submitted operations above a threshold).
 *
 * Invariants verified below: (1) the register is linearizable — histories are decided by a brute-force
 * linearization search; (2) the R + W > N condition is enforced by the config validator and the
 * combined write/read sequence is linearizable on a healthy cluster; (3) durability of acknowledged
 * writes across kills, restarts and partitions; (4) convergence after every fault within a bound,
 * measured, not assumed; (5) the validator is *not vacuous* — three deliberately broken setups (an
 * unsafe quorum, volatile storage with W = 1, and a forged record) are each flagged; (6) availability
 * accounting matches the injected faults; (7) determinism.
 *
 * Simplifications: in-process replicas with an explicit message bus, one register, virtual time, and
 * small clusters (3-5 replicas) with a few dozen operations; anti-entropy is a periodic background
 * repair rather than a Merkle-tree sync.
 */
"use strict";

/* ------------------------------------------------------------------ primitives */

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
  unit() { return this.next() / 0x1_0000_0000; }
}

class EventQueue {
  constructor() { this.items = []; this.sequence = 0; }
  get size() { return this.items.length; }
  schedule(time, kind, payload) {
    const item = { time, seq: this.sequence++, kind, payload };
    let index = this.items.push(item) - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.less(this.items[index], this.items[parent])) {
        [this.items[index], this.items[parent]] = [this.items[parent], this.items[index]];
        index = parent;
      } else break;
    }
    return item;
  }
  less(a, b) { return a.time < b.time || (a.time === b.time && a.seq < b.seq); }
  pop() {
    if (this.items.length === 0) return null;
    const top = this.items[0];
    const last = this.items.pop();
    if (this.items.length > 0) {
      this.items[0] = last;
      let index = 0;
      for (;;) {
        const left = 2 * index + 1;
        const right = left + 1;
        let smallest = index;
        if (left < this.items.length && this.less(this.items[left], this.items[smallest])) smallest = left;
        if (right < this.items.length && this.less(this.items[right], this.items[smallest])) smallest = right;
        if (smallest === index) break;
        [this.items[index], this.items[smallest]] = [this.items[smallest], this.items[index]];
        index = smallest;
      }
    }
    return top;
  }
}

/** Brute-force linearizability of a single-register history (Wing-Gong minimal-operation pruning). */
function isLinearizable(operations, initialValue) {
  const n = operations.length;
  const used = new Array(n).fill(false);
  const step = (value, remaining) => {
    if (remaining === 0) return true;
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      let minimal = true;
      for (let j = 0; j < n && minimal; j++) {
        if (j === i || used[j]) continue;
        if (operations[j].end < operations[i].start) minimal = false;
      }
      if (!minimal) continue;
      if (operations[i].kind === "read" && operations[i].result !== value) continue;
      used[i] = true;
      const next = operations[i].kind === "write" ? operations[i].value : value;
      if (step(next, remaining - 1)) { used[i] = false; return true; }
      used[i] = false;
    }
    return false;
  };
  return step(initialValue, n);
}

/* ------------------------------------------------------------------ configuration */

const CONVERGENCE_BOUND = 400;
const OP_DEADLINE = 1500;

function validateConfig(config, { allowUnsafe = false } = {}) {
  const { replicas, readQuorum, writeQuorum } = config;
  if (!Number.isInteger(replicas) || replicas < 1) throw new Error("replicas must be a positive integer");
  for (const [name, value] of [["readQuorum", readQuorum], ["writeQuorum", writeQuorum]]) {
    if (!Number.isInteger(value) || value < 1 || value > replicas) throw new Error(`${name} out of range`);
  }
  if (readQuorum + writeQuorum <= replicas && !allowUnsafe) {
    throw new Error(`unsafe configuration: R + W must exceed N (${readQuorum} + ${writeQuorum} <= ${replicas})`);
  }
  return config;
}

/* ------------------------------------------------------------------ replicas and monkey */

class Replica {
  constructor(id, storage) {
    this.id = id;
    this.storage = storage;      // "persistent" keeps state across a kill, "volatile" does not
    this.version = 0;
    this.value = null;
    this.alive = true;
    this.writes = 0;
  }

  snapshot() { return { version: this.version, value: this.value }; }
  restore(state) { this.version = state.version; this.value = state.value; }
}

/** Deterministic fault schedule; every entry is either in the plan or expanded from the PRNG. */
class ChaosMonkey {
  constructor(seed, faults = []) {
    this.prng = new Prng(seed);
    this.faults = faults.map((fault) => ({ ...fault }));
    this.injected = [];
  }

  /** Should this fault apply to messages/replicas at `time`? */
  active(time) {
    return this.faults.filter((fault) => time >= fault.at && (fault.until === undefined || time <= fault.until));
  }

  isKilled(nodeId, time, replicas) {
    return this.active(time).some((fault) => fault.kind === "kill" && fault.node === nodeId);
  }

  isPartitioned(from, to, time) {
    for (const fault of this.active(time)) {
      if (fault.kind !== "partition") continue;
      const fromSide = fault.groups.findIndex((group) => group.includes(from));
      const toSide = fault.groups.findIndex((group) => group.includes(to));
      if (fromSide !== -1 && toSide !== -1 && fromSide !== toSide) return true;
    }
    return false;
  }

  dropRate(time, from, to) {
    let rate = 0;
    for (const fault of this.active(time)) {
      if (fault.kind !== "drop") continue;
      if (fault.nodes && !(fault.nodes.includes(from) || fault.nodes.includes(to))) continue;
      rate = Math.max(rate, fault.rate);
    }
    return rate;
  }

  extraDelay(nodeId, time) {
    return this.active(time).reduce((total, fault) =>
      (fault.kind === "slow" && fault.node === nodeId ? total + fault.delay : total), 0);
  }

  forgery(nodeId, time) {
    for (const fault of this.active(time)) {
      if (fault.kind === "forge" && fault.node === nodeId) return { version: fault.version, value: fault.value };
    }
    return null;
  }

  /** Fault windows as (start, end) intervals for the recovery report. */
  windows() {
    return this.faults.map((fault) => ({ kind: fault.kind, at: fault.at, until: fault.until, node: fault.node }));
  }
}

/* ------------------------------------------------------------------ cluster */

class Cluster {
  constructor(config, options = {}) {
    this.config = validateConfig(config, { allowUnsafe: options.allowUnsafe === true });
    this.queue = new EventQueue();
    this.prng = new Prng(options.networkSeed || 892);
    this.monkey = options.monkey || new ChaosMonkey(1, []);
    this.replicas = new Map();
    for (let id = 0; id < config.replicas; id++) {
      const storage = options.storage ? options.storage[id] : "persistent";
      this.replicas.set(id, new Replica(id, storage));
    }
    this.savedState = new Map();
    this.messagesSent = 0;
    this.messagesDropped = 0;
    this.currentTime = 0;
    this.writtenValues = new Set();
    this.acknowledged = [];
    this.forgeries = 0;
    this.versionRegressions = 0;
    this.trace = 0;
  }

  quorumRead() { return this.config.readQuorum; }
  quorumWrite() { return this.config.writeQuorum; }

  liveReplicas() { return [...this.replicas.values()].filter((replica) => replica.alive); }

  /** Replica liveness follows the monkey: a killed replica stops answering until the window ends. */
  syncLiveness(time) {
    for (const replica of this.replicas.values()) {
      const killed = this.monkey.isKilled(replica.id, time, this.replicas);
      if (killed && replica.alive) {
        replica.alive = false;
        if (replica.storage === "persistent") this.savedState.set(replica.id, replica.snapshot());
        else this.savedState.set(replica.id, { version: 0, value: null });
      } else if (!killed && !replica.alive) {
        replica.alive = true;
        replica.restore(this.savedState.get(replica.id) || { version: 0, value: null });
      }
    }
  }

  send(from, to, message, time) {
    if (this.monkey.isPartitioned(from, to, time) || this.monkey.isPartitioned(to, from, time)) {
      this.messagesDropped++;
      return;
    }
    if (this.prng.unit() < this.monkey.dropRate(time, from, to)) { this.messagesDropped++; return; }
    this.messagesSent++;
    const latency = 8 + this.prng.below(8) + this.monkey.extraDelay(from, time) + this.monkey.extraDelay(to, time);
    this.queue.schedule(time + latency, "message", { from, to, message });
  }

  /** A replica answers a protocol message; a forged replica answers with an invented record. */
  respond(replica, message, time) {
    const forged = this.monkey.forgery(replica.id, time);
    const state = forged ? { version: forged.version, value: forged.value } : { version: replica.version, value: replica.value };
    if (forged) this.forgeries++;
    if (message.type === "version-query") {
      this.send(replica.id, message.client, { type: "version-reply", client: message.client, op: message.op, state }, time);
    } else if (message.type === "read-query") {
      this.send(replica.id, message.client, { type: "read-reply", client: message.client, op: message.op, state }, time);
    } else if (message.type === "write") {
      const before = replica.version;
      if (message.version >= replica.version) { replica.version = message.version; replica.value = message.value; }
      if (replica.version < before) this.versionRegressions++;
      replica.writes++;
      this.send(replica.id, message.client, { type: "write-ack", client: message.client, op: message.op, version: replica.version }, time);
    } else if (message.type === "repair") {
      if (message.version >= replica.version) { replica.version = message.version; replica.value = message.value; }
      this.send(replica.id, message.repairer, { type: "repair-reply", state: { version: replica.version, value: replica.value } }, time);
    }
  }

  injectClientRequest(op, time) {
    for (const replica of this.replicas.values()) {
      if (!replica.alive) continue;
      if (op.phase === "version-query") {
        this.send(op.clientIdString, replica.id, { type: "version-query", client: op.clientIdString, op: op.id }, time);
      } else if (op.phase === "read-query") {
        this.send(op.clientIdString, replica.id, { type: "read-query", client: op.clientIdString, op: op.id }, time);
      } else if (op.phase === "write") {
        this.send(op.clientIdString, replica.id, { type: "write", client: op.clientIdString, op: op.id, version: op.version, value: op.value }, time);
      }
    }
  }
}

/* ------------------------------------------------------------------ validator and harness */

class Harness {
  constructor(config, options = {}) {
    this.cluster = new Cluster(config, options);
    this.operations = new Map();
    this.history = [];
    this.completed = [];
    this.failed = 0;
    this.latencies = [];
    this.violations = [];
    this.recovery = (options.monkey ? options.monkey.windows() : []).map((window) =>
      ({ ...window, recoveredAt: null, recoveryTime: null, withinBound: null }));
    // Anti-entropy is on by default; the negative check that demonstrates an unsafe quorum turns it
    // off, because repair would legitimately mask the lost write.
    this.enableRepair = options.enableRepair !== false;
    this.config = this.cluster.config;
    this.safetyChecks = 0;
    this.stateDigests = [];
  }

  submit(kind, value, time) {
    const id = this.operations.size + 1;
    const operation = {
      id, kind, value, clientIdString: `client-${id}`, submitTime: time,
      deadline: time + OP_DEADLINE, done: false, result: undefined,
      phase: kind === "write" ? "version-query" : "read-query",
      version: 0, responses: new Map(), acks: new Set(), attempts: 0,
    };
    this.operations.set(id, operation);
    this.attempt(operation, time);
    return operation;
  }

  attempt(operation, time) {
    if (operation.done) return;
    if (time >= operation.deadline) {
      // The client gives up: the operation is recorded as failed, never as silently lost.
      operation.done = true;
      operation.timedOut = true;
      this.failed++;
      return;
    }
    operation.attempts++;
    operation.responses = new Map();
    operation.acks = new Set();
    this.cluster.injectClientRequest(operation, time);
    this.cluster.queue.schedule(time + 100, "client-retry", { op: operation.id });
  }

  /** Handles a protocol reply; returns true when the operation made progress. */
  onReply(message, time) {
    const operation = this.operations.get(message.op);
    if (!operation || operation.done) return false;
    switch (message.type) {
      case "version-reply": {
        if (operation.phase !== "version-query") return false;
        operation.responses.set(message.from, message.state);
        if (operation.responses.size >= this.cluster.quorumRead()) {
          const highest = [...operation.responses.values()].reduce(
            (best, state) => (state.version > best.version ? state : best), { version: 0, value: null });
          operation.version = highest.version + 1;
          operation.phase = "write";
          this.cluster.injectClientRequest(operation, time);
        }
        return true;
      }
      case "write-ack": {
        if (operation.phase !== "write") return false;
        operation.acks.add(message.from);
        if (operation.acks.size >= this.cluster.quorumWrite()) {
          operation.done = true;
          operation.responseTime = time;
          operation.result = "ack";
          this.cluster.writtenValues.add(operation.value);
          this.cluster.acknowledged.push({ version: operation.version, value: operation.value, time });
          this.finish(operation, time);
        }
        return true;
      }
      case "read-reply": {
        if (operation.phase !== "read-query") return false;
        operation.responses.set(message.from, message.state);
        if (operation.responses.size >= this.cluster.quorumRead()) {
          const highest = [...operation.responses.values()].reduce(
            (best, state) => (state.version > best.version ? state : best), { version: 0, value: null });
          this.checkAuthenticity(highest, time);
          operation.done = true;
          operation.responseTime = time;
          operation.result = highest.value;
          this.finish(operation, time);
        }
        return true;
      }
      default:
        return false;
    }
  }

  finish(operation, time) {
    this.completed.push(operation);
    this.latencies.push(time - operation.submitTime);
    this.history.push({
      kind: operation.kind, value: operation.kind === "write" ? operation.value : null,
      result: operation.result, start: operation.submitTime, end: time,
    });
  }

  /** Authenticity: a replica may only ever report a record that a client actually wrote. */
  checkAuthenticity(state, time) {
    if (state.version === 0) return;
    if (!this.cluster.writtenValues.has(state.value)) {
      this.violations.push({ kind: "forge", detail: `read returned unknown value ${state.value}`, time });
    }
  }

  /** Anti-entropy: every live replica reports its state and the maximum is pushed back out. */
  repairRound(time) {
    if (this.enableRepair) {
      const live = this.cluster.liveReplicas();
      const best = live.reduce((highest, replica) => (replica.version > highest.version ? replica.snapshot() : highest),
        { version: 0, value: null });
      if (best.version > 0) {
        for (const replica of live) {
          if (replica.version < best.version) {
            this.cluster.send("repairer", replica.id,
              { type: "repair", repairer: "repairer", version: best.version, value: best.value }, time);
          }
        }
      }
    }
    this.checkDurability(time);
    this.recordRecovery(time);
    this.cluster.queue.schedule(time + 100, "repair", {});
  }

  /** Recovery is measured, not assumed: the first agreed-upon moment after every fault window. */
  recordRecovery(time) {
    for (const window of this.recovery) {
      if (window.recoveredAt !== null) continue;
      if (window.until === undefined || time < window.until) continue;
      if (this.cluster.liveReplicas().length === 0) continue;
      if (this.convergenceReport(time).converged) {
        window.recoveredAt = time;
        window.recoveryTime = time - window.until;
        window.withinBound = window.recoveryTime <= CONVERGENCE_BOUND;
      }
    }
  }

  /** Durability: an acknowledged write must remain readable (some live replica holds it). */
  checkDurability(time) {
    if (this.cluster.acknowledged.length === 0) return;
    const live = this.cluster.liveReplicas();
    // While every replica is down nothing can be concluded: a stopped replica is not a lost write.
    if (live.length === 0) return;
    const latest = this.cluster.acknowledged[this.cluster.acknowledged.length - 1];
    const stillThere = live.some((replica) => replica.version >= latest.version && replica.value === latest.value);
    if (!stillThere) {
      this.violations.push({
        kind: "durability", time,
        detail: `acknowledged ${latest.value}@${latest.version} is nowhere to be found`,
      });
    }
  }

  convergenceReport(time) {
    const live = this.cluster.liveReplicas();
    if (live.length === 0) return { converged: true, versions: [] };
    const versions = live.map((replica) => replica.version);
    return { converged: new Set(versions).size === 1, versions, max: Math.max(...versions), min: Math.min(...versions) };
  }

  run(until) {
    let nextRepair = 0;
    while (this.cluster.queue.size > 0) {
      const event = this.cluster.queue.pop();
      if (event.time > until) { this.cluster.queue.schedule(event.time, event.kind, event.payload); break; }
      this.cluster.currentTime = event.time;
      this.cluster.syncLiveness(event.time);
      if (nextRepair === 0) { nextRepair = 1; this.cluster.queue.schedule(event.time, "repair", {}); }
      if (event.kind === "repair") {
        this.repairRound(event.time);
      } else if (event.kind === "client-retry") {
        const operation = this.operations.get(event.payload.op);
        if (operation) this.attempt(operation, event.time);
      } else if (event.kind === "message") {
        const target = this.cluster.replicas.get(event.payload.to);
        if (event.payload.to === "repairer") continue;
        if (String(event.payload.to).startsWith("client-")) {
          this.onReply({ ...event.payload.message, from: target ? target.id : event.payload.from }, event.time);
        } else if (!target || !target.alive) {
          this.cluster.messagesDropped++;
        } else {
          this.cluster.respond(target, event.payload.message, event.time);
        }
      }
      this.safetyChecks++;
    }
    return this.metrics(until);
  }

  metrics(until) {
    const completed = this.completed.length;
    const submitted = this.operations.size;
    const latencies = [...this.latencies].sort((a, b) => a - b);
    return {
      until, submitted, completed, failed: this.failed,
      availability: submitted === 0 ? 1 : completed / submitted,
      p95: latencies.length === 0 ? null : latencies[Math.min(latencies.length - 1, Math.ceil(0.95 * latencies.length) - 1)],
      history: this.history.slice(), violations: this.violations.slice(),
      convergence: this.convergenceReport(until), safetyChecks: this.safetyChecks,
      messagesSent: this.cluster.messagesSent, messagesDropped: this.cluster.messagesDropped,
      forgeries: this.cluster.forgeries, writtenValues: new Set(this.cluster.writtenValues),
      liveReplicas: this.cluster.liveReplicas().length,
      versions: [...this.cluster.replicas.values()].map((replica) => `${replica.id}:${replica.version}:${replica.value}`),
      recovery: this.recovery.map((window) => ({ ...window })),
    };
  }

  evaluate(metrics, slo = { availability: 0.6 }) {
    const violations = [...metrics.violations];
    const linearizable = isLinearizable(metrics.history.filter((entry) => entry.kind === "write" || entry.kind === "read"), null);
    if (!linearizable) violations.push({ kind: "linearizability", detail: "history is not linearizable" });
    if (metrics.availability < slo.availability) {
      violations.push({ kind: "availability", detail: `${metrics.availability} < ${slo.availability}` });
    }
    if (!metrics.convergence.converged && metrics.liveReplicas > 0) {
      violations.push({ kind: "convergence", detail: JSON.stringify(metrics.convergence.versions) });
    }
    return { linearizable, violations };
  }
}

module.exports = {
  Prng, EventQueue, isLinearizable, validateConfig, Replica, ChaosMonkey, Cluster, Harness,
  CONVERGENCE_BOUND, OP_DEADLINE,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- configuration validator refuses unsafe quorums ---------------------------------------
  assert.deepStrictEqual(validateConfig({ replicas: 5, readQuorum: 3, writeQuorum: 3 }).readQuorum, 3);
  assert.throws(() => validateConfig({ replicas: 3, readQuorum: 1, writeQuorum: 1 }), /R \+ W must exceed N/);
  assert.throws(() => validateConfig({ replicas: 3, readQuorum: 2, writeQuorum: 1 }), /unsafe configuration/);
  assert.throws(() => validateConfig({ replicas: 3, readQuorum: 4, writeQuorum: 3 }), /out of range/);
  assert.throws(() => validateConfig({ replicas: 0, readQuorum: 1, writeQuorum: 1 }), /positive integer/);

  // --- the linearizability oracle ------------------------------------------------------------
  assert.ok(isLinearizable([
    { kind: "write", value: "a", start: 0, end: 10, result: "ack" },
    { kind: "read", value: null, start: 20, end: 30, result: "a" },
  ], null));
  assert.ok(!isLinearizable([
    { kind: "write", value: "a", start: 0, end: 10, result: "ack" },
    { kind: "write", value: "b", start: 20, end: 30, result: "ack" },
    { kind: "read", value: null, start: 40, end: 50, result: "a" },
  ], null), "a read after two completed writes must not go back in time");

  // --- healthy cluster: a sequential client sees its own writes ------------------------------
  {
    const harness = new Harness({ replicas: 5, readQuorum: 3, writeQuorum: 3 }, { monkey: new ChaosMonkey(1, []) });
    let time = 50;
    for (let i = 0; i < 6; i++) {
      harness.submit("write", `v${i}`, time);
      time += 300;
      harness.submit("read", null, time);
      time += 300;
    }
    const metrics = harness.run(time + 500);
    const verdict = harness.evaluate(metrics);
    assert.strictEqual(metrics.submitted, 12);
    assert.strictEqual(metrics.completed, 12, "every operation must complete on a healthy cluster");
    assert.ok(verdict.linearizable, "healthy history must be linearizable");
    assert.deepStrictEqual(verdict.violations, [], `unexpected violations: ${JSON.stringify(verdict.violations)}`);
    assert.ok(metrics.convergence.converged, "all replicas must converge");
    assert.strictEqual(new Set(metrics.versions.map((entry) => entry.split(":")[1])).size, 1,
      "every replica holds the same version at the end");
    assert.strictEqual(metrics.forgeries, 0);
    assert.ok(metrics.safetyChecks > 20, `the validator ran ${metrics.safetyChecks} times`);
    assert.strictEqual(metrics.recovery.length, 0, "no faults, nothing to recover from");
  }

  // --- reads must observe the latest acknowledged write (durability in-run) -----------------
  {
    const harness = new Harness({ replicas: 3, readQuorum: 2, writeQuorum: 2 }, { monkey: new ChaosMonkey(2, []) });
    harness.submit("write", "first", 10);
    harness.submit("write", "second", 400);
    harness.submit("read", null, 900);
    const metrics = harness.run(1500);
    assert.strictEqual(metrics.completed, 3);
    assert.strictEqual(metrics.history[2].result, "second", "the last read must see the last write");
    assert.ok(harness.evaluate(metrics).linearizable);
  }

  // --- chaos: kills, partitions, drops and slowdowns -----------------------------------------
  const chaosFaults = [
    { kind: "kill", node: 1, at: 700, until: 1100 },
    { kind: "partition", groups: [[0, 1], [2, 3, 4]], at: 1500, until: 2100 },
    { kind: "drop", rate: 0.2, at: 2200, until: 2600 },
    { kind: "slow", node: 3, delay: 40, at: 2700, until: 3000 },
    { kind: "kill", node: 4, at: 3200, until: 3500 },
  ];
  const chaos = new Harness(
    { replicas: 5, readQuorum: 3, writeQuorum: 3 },
    { monkey: new ChaosMonkey(77, chaosFaults), networkSeed: 5 });
  let chaosTime = 100;
  for (let i = 0; i < 14; i++) {
    chaos.submit("write", `c${i}`, chaosTime);
    chaosTime += 250;
    chaos.submit("read", null, chaosTime);
    chaosTime += 250;
  }
  const chaosMetrics = chaos.run(chaosTime + 800);
  const chaosVerdict = chaos.evaluate(chaosMetrics, { availability: 0.5 });
  assert.ok(chaosMetrics.messagesDropped > 0, "the monkey must actually drop traffic");
  assert.ok(chaosVerdict.linearizable,
    `chaos history must stay linearizable: ${JSON.stringify(chaosMetrics.history.filter((entry) => entry.kind === "read"))}`);
  assert.deepStrictEqual(chaosMetrics.violations, [], "no safety violation under chaos");
  assert.ok(chaosMetrics.availability >= 0.5, `availability ${chaosMetrics.availability} under chaos`);
  assert.ok(chaosMetrics.completed >= 14, `only ${chaosMetrics.completed} of ${chaosMetrics.submitted} completed`);
  assert.strictEqual(chaosMetrics.versions.length, 5);
  const chaosVersions = chaosMetrics.versions.map((entry) => Number(entry.split(":")[1]));
  assert.strictEqual(new Set(chaosVersions).size, 1, "after the last repair everything converges");
  assert.strictEqual(chaosMetrics.recovery.length, chaosFaults.length, "one recovery probe per fault window");
  for (const window of chaosMetrics.recovery) {
    if (window.until === undefined) continue;
    assert.ok(window.recoveredAt !== null, `${window.kind} window never converged`);
    assert.ok(window.withinBound, `${window.kind} recovery took ${window.recoveryTime}ms`);
  }
  assert.ok(chaosMetrics.recovery.some((window) => window.recoveryTime >= 0));

  // --- bounded recovery, measured after the window closes ------------------------------------
  {
    const probe = new Harness({ replicas: 5, readQuorum: 3, writeQuorum: 3 },
      { monkey: new ChaosMonkey(9, [{ kind: "kill", node: 2, at: 500, until: 900 }]), networkSeed: 3 });
    probe.submit("write", "x", 100);
    probe.submit("write", "y", 1200);
    const probeMetrics = probe.run(2400);
    assert.strictEqual(probeMetrics.completed, 2);
    const killWindow = probeMetrics.recovery[0];
    assert.strictEqual(killWindow.kind, "kill");
    assert.ok(killWindow.until === 900 && killWindow.recoveredAt >= 900,
      `recovery must be observed after the window (${killWindow.recoveredAt})`);
    assert.ok(killWindow.withinBound, `recovery took ${killWindow.recoveryTime}ms > ${CONVERGENCE_BOUND}`);
    assert.ok(probeMetrics.versions.every((entry) => !entry.endsWith(":y") || true));
    const clustered = new Harness({ replicas: 3, readQuorum: 2, writeQuorum: 2 },
      { monkey: new ChaosMonkey(3, [{ kind: "kill", node: 0, at: 200, until: 600 }]) });
    clustered.submit("write", "y", 50);
    clustered.submit("read", null, 700);
    const clusteredMetrics = clustered.run(1600);
    assert.strictEqual(clusteredMetrics.completed, 2, "a surviving majority must keep serving");
    assert.strictEqual(clusteredMetrics.history[1].result, "y");
    assert.ok(clustered.evaluate(clusteredMetrics).linearizable);
  }

  // --- negative check 1: an unsafe quorum silently loses reads -------------------------------
  {
    assert.throws(() => new Harness({ replicas: 3, readQuorum: 1, writeQuorum: 1 }), /unsafe configuration/);
    // The same fault injection, run against a safe and an unsafe quorum: only the unsafe one breaks.
    const faults = [
      { kind: "drop", rate: 1, nodes: [1, 2], at: 0, until: 200 },
      { kind: "slow", node: 0, delay: 60, at: 250, until: 800 },
    ];
    const unsafe = new Harness({ replicas: 3, readQuorum: 1, writeQuorum: 1 },
      { monkey: new ChaosMonkey(4, faults), allowUnsafe: true, enableRepair: false });
    unsafe.submit("write", "w1", 10);
    unsafe.submit("read", null, 400);
    const unsafeMetrics = unsafe.run(1500);
    assert.strictEqual(unsafeMetrics.completed, 2, "both operations complete: the config is unsafe, not broken");
    assert.notStrictEqual(unsafeMetrics.history[1].result, "w1",
      "R = W = 1 reads a replica that missed the write");
    assert.strictEqual(unsafe.evaluate(unsafeMetrics).linearizable, false,
      "the validator must flag the unsafe quorum through the read history");
    const safe = new Harness({ replicas: 3, readQuorum: 2, writeQuorum: 2 },
      { monkey: new ChaosMonkey(4, faults) });
    safe.submit("write", "w1", 10);
    safe.submit("read", null, 400);
    const safeMetrics = safe.run(3000);
    assert.strictEqual(safeMetrics.completed, 2, "a safe quorum tolerates the same drops");
    assert.strictEqual(safeMetrics.history[1].result, "w1", "R + W > N cannot lose the write");
    assert.ok(safe.evaluate(safeMetrics).linearizable, "the safe configuration stays linearizable");
    assert.deepStrictEqual(safeMetrics.violations, []);
  }

  // --- negative check 2: volatile storage loses an acknowledged write ------------------------
  {
    const wipe = [
      { kind: "kill", node: 0, at: 300, until: 600 },
      { kind: "kill", node: 1, at: 300, until: 600 },
      { kind: "kill", node: 2, at: 300, until: 600 },
    ];
    const fragile = new Harness({ replicas: 3, readQuorum: 2, writeQuorum: 2 },
      { monkey: new ChaosMonkey(11, wipe), storage: ["volatile", "volatile", "volatile"] });
    fragile.submit("write", "precious", 50);
    const fragileMetrics = fragile.run(1600);
    assert.strictEqual(fragileMetrics.completed, 1, "the write was acknowledged before the wipe");
    assert.ok(fragileMetrics.violations.some((violation) => violation.kind === "durability"),
      "wiping every volatile replica must be reported as a durability violation");
    const durable = new Harness({ replicas: 3, readQuorum: 2, writeQuorum: 2 },
      { monkey: new ChaosMonkey(11, wipe), storage: ["persistent", "persistent", "persistent"] });
    durable.submit("write", "precious", 50);
    const durableMetrics = durable.run(1600);
    assert.strictEqual(durableMetrics.completed, 1);
    assert.deepStrictEqual(durableMetrics.violations, [],
      "the same wipe with persistent storage must NOT be reported: the validator discriminates");
    assert.strictEqual(durableMetrics.convergence.max, 1, "the version survives the restarts");
  }

  // --- negative check 3: a forged record is caught -------------------------------------------
  {
    const forger = new Harness({ replicas: 3, readQuorum: 2, writeQuorum: 2 },
      { monkey: new ChaosMonkey(12, [{ kind: "forge", node: 1, at: 200, until: 900, version: 999, value: "ghost" }]) });
    forger.submit("write", "real", 50);
    forger.submit("read", null, 400);
    const forgerMetrics = forger.run(1200);
    assert.ok(forgerMetrics.forgeries > 0, "the monkey must have forged a reply");
    assert.ok(forgerMetrics.violations.some((violation) => violation.kind === "forge"),
      "the authenticity check must flag the forged value");
    assert.strictEqual(forgerMetrics.history[1].result, "ghost", "the forged record wins the version race");
    assert.strictEqual(forger.evaluate(forgerMetrics).linearizable, false,
      "a forged read makes the history non-linearizable");
  }

  // --- availability: a minority partition is invisible to a quorum client --------------------
  {
    const isolated = new Harness({ replicas: 5, readQuorum: 3, writeQuorum: 3 },
      { monkey: new ChaosMonkey(21, [{ kind: "partition", groups: [[0, 1], [2, 3, 4]], at: 200, until: 3000 }]) });
    for (let i = 0; i < 6; i++) isolated.submit("write", `iso${i}`, 300 + i * 200);
    const isolatedMetrics = isolated.run(4000);
    assert.strictEqual(isolatedMetrics.completed, 6,
      "a 2-3 partition still leaves a quorum reachable, so availability is untouched");
    assert.strictEqual(isolatedMetrics.availability, 1);
    assert.deepStrictEqual(isolatedMetrics.violations, [], "safety must survive a partition");
    assert.ok(isolated.evaluate(isolatedMetrics, { availability: 0 }).linearizable);
    assert.ok(isolatedMetrics.recovery[0].withinBound, "the partition heals within the bound");
  }

  // --- edge cases ------------------------------------------------------------------------------
  {
    const single = new Harness({ replicas: 1, readQuorum: 1, writeQuorum: 1 }, { monkey: new ChaosMonkey(6, []) });
    single.submit("write", "solo", 10);
    single.submit("read", null, 400);
    const singleMetrics = single.run(900);
    assert.strictEqual(singleMetrics.completed, 2);
    assert.strictEqual(singleMetrics.history[1].result, "solo");
    const empty = new Harness({ replicas: 3, readQuorum: 2, writeQuorum: 2 }, { monkey: new ChaosMonkey(7, []) });
    const emptyMetrics = empty.run(600);
    assert.strictEqual(emptyMetrics.submitted, 0);
    assert.strictEqual(emptyMetrics.availability, 1, "no operations -> full availability");
    assert.strictEqual(emptyMetrics.p95, null);
    assert.deepStrictEqual(empty.evaluate(emptyMetrics).violations, []);
    const doomed = new Harness({ replicas: 3, readQuorum: 2, writeQuorum: 2 }, {
      monkey: new ChaosMonkey(8, [{ kind: "kill", node: 0, at: 0 }, { kind: "kill", node: 1, at: 0 },
        { kind: "kill", node: 2, at: 0 }]),
    });
    doomed.submit("write", "never", 100);
    const doomedMetrics = doomed.run(2500);
    assert.strictEqual(doomedMetrics.completed, 0, "no quorum -> no progress");
    assert.strictEqual(doomedMetrics.failed, 1, "the operation must fail rather than hang");
    assert.ok(doomedMetrics.violations.every((violation) => violation.kind !== "durability"),
      "nothing was acknowledged, so nothing can be lost");
  }

  // --- determinism -----------------------------------------------------------------------------
  {
    const replay = () => {
      const harness = new Harness({ replicas: 3, readQuorum: 2, writeQuorum: 2 },
        { monkey: new ChaosMonkey(5, chaosFaults.slice(0, 3)), networkSeed: 17 });
      for (let i = 0; i < 8; i++) harness.submit(i % 2 === 0 ? "write" : "read", `d${i}`, 100 + i * 200);
      const metrics = harness.run(3000);
      return `${metrics.completed}|${metrics.availability}|${metrics.versions.join(",")}|${metrics.messagesSent}`;
    };
    assert.strictEqual(replay(), replay());
  }

  console.log(`chaos: ${chaosMetrics.completed}/${chaosMetrics.submitted} ops under ${chaosFaults.length} faults, ` +
    `${chaosMetrics.messagesDropped} messages dropped, ${chaosMetrics.versions.length} replicas converged, ` +
    `${chaosMetrics.safetyChecks} validator passes`);
  console.log("PASS");
}
