/** Task 852 - Implement global snapshotting for distributed streaming pipeline (Chandy-Lamport at scale).
 *  (Tier: Master, CTCI Ch 9 — System Design and Scalability)
 *
 * Model. A streaming pipeline of P processes on a complete graph of FIFO channels, each channel with
 * a fixed propagation delay. Every process repeatedly performs application events: an "internal"
 * event injects one token into the process, a "send" event moves up to `take` tokens to a peer (a
 * send with an empty balance is a no-op and is counted as starved). All application events are
 * pre-scheduled at deterministic times so the application trace is independent of the snapshot
 * algorithm; one or more initiators then start the Chandy-Lamport algorithm at time t0.
 *
 * Chandy-Lamport as implemented: an initiator records its state and sends a marker on every outgoing
 * channel; a process that receives its first marker records its state (a prefix of its event history)
 * and forwards a marker on all outgoing channels; while a process has recorded but the marker of
 * channel p->q has not arrived yet, every message arriving on that channel is appended to the
 * channel's recorded state; the channel closes when its marker arrives.
 *
 * Simplified versus a real deployment: an in-process discrete-event network with no failures, no
 * reordering (per-channel FIFO is a modelling assumption, not an emergent property), constant link
 * delay, one marker per channel, no wall clock and no threads — time is an integer tick advanced by
 * explicitly scheduled events. Randomness is a single seeded mulberry32 stream.
 *
 * Invariants verified (all of them from a verifier that reads only the recorded trace):
 *   safety      the recorded cut is consistent: no message received inside the cut was sent outside
 *               it; and for every channel the recorded channel state is *exactly* the set of messages
 *               sent outside the cut that are received inside it (no message lost, none invented).
 *   liveness    every process records exactly once, exactly one marker traverses each channel, and
 *               the run terminates with a complete snapshot for any number of initiators.
 *   accounting  token conservation across the cut: tokens held by the recorded process states plus
 *               tokens inside recorded channel states equals the tokens injected before the cut
 *               (each token is in exactly one place).
 *   oracle      the recorded per-process state is replayed independently from the raw trace (fold the
 *               first `delivered` events) and must match the recorded counters; the cut must contain
 *               every pre-scheduled event that happened before t0.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — the only source of randomness here. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Binary min-heap ordered by (time, insertion order) — deterministic event scheduling. */
class Timer {
  constructor() {
    this.heap = [];
    this.seq = 0;
    this.now = 0;
  }

  static before(a, b) {
    return a.t < b.t || (a.t === b.t && a.order < b.order);
  }

  schedule(t, run) {
    const event = { t, order: this.seq++, run };
    const heap = this.heap;
    heap.push(event);
    let i = heap.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!Timer.before(heap[i], heap[parent])) break;
      const tmp = heap[i];
      heap[i] = heap[parent];
      heap[parent] = tmp;
      i = parent;
    }
    return event;
  }

  drain() {
    while (this.heap.length > 0) {
      const top = this.heap[0];
      const last = this.heap.pop();
      if (this.heap.length > 0) {
        this.heap[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1;
          const r = l + 1;
          let small = i;
          if (l < this.heap.length && Timer.before(this.heap[l], this.heap[small])) small = l;
          if (r < this.heap.length && Timer.before(this.heap[r], this.heap[small])) small = r;
          if (small === i) break;
          const tmp = this.heap[i];
          this.heap[i] = this.heap[small];
          this.heap[small] = tmp;
          i = small;
        }
      }
      this.now = top.t;
      top.run();
    }
  }
}

/**
 * Streaming pipeline with a deterministic application trace. The action clock and the link delay are
 * both multiples of 3 and t0 is congruent to 1 (mod 3), so an application event never shares an
 * instant with a marker event — that keeps the FIFO argument of Chandy-Lamport exact instead of
 * relying on tie-breaking. The delay (6 * processes) is deliberately longer than the action spacing,
 * so a marker wave really does straddle messages in flight.
 */
class Pipeline {
  constructor({ processes = 8, actionsPerProcess = 30, delay = undefined, seed = 852 } = {}) {
    if (!Number.isInteger(processes) || processes < 1) throw new RangeError("processes must be >= 1");
    if (!Number.isInteger(actionsPerProcess) || actionsPerProcess < 1) throw new RangeError("actionsPerProcess must be >= 1");
    if (!Number.isInteger(seed)) throw new RangeError("seed must be an integer");
    const linkDelay = delay === undefined ? 6 * processes : delay;
    if (!Number.isInteger(linkDelay) || linkDelay < 3 || linkDelay % 3 !== 0) {
      throw new RangeError("delay must be a positive multiple of 3");
    }
    this.processes = processes;
    this.actionsPerProcess = actionsPerProcess;
    this.delay = linkDelay;
    const rand = mulberry32(seed);
    this.actions = [];
    for (let p = 0; p < processes; p++) {
      const list = [];
      for (let k = 0; k < actionsPerProcess; k++) {
        const at = p * 3 + k * 3 * processes;
        if (processes > 1 && rand() < 0.55) {
          const offset = 1 + Math.floor(rand() * (processes - 1));
          list.push({ kind: "send", peer: (p + offset) % processes, take: 1 + Math.floor(rand() * 2), at });
        } else {
          list.push({ kind: "internal", at });
        }
      }
      this.actions.push(list);
    }
    // t0 is congruent to 1 (mod 3), hence never an application event time.
    this.t0 = 1 + 3 * Math.floor((actionsPerProcess * processes) / 2);
    this.span = 3 * (processes - 1) + 3 * (actionsPerProcess - 1) * processes;
  }

  /** Number of this process's pre-scheduled actions that happen strictly before time t. */
  actionsBefore(p, t) {
    return this.actions[p].filter((a) => a.at < t).length;
  }

  /**
   * Runs the application plus the snapshot algorithm. `initiators` is a list of process ids or
   * {process, at} records. Returns a plain-data run record: the snapshot plus the full trace the
   * verifier needs.
   */
  run(initiators = [0]) {
    const starts = initiators.map((x) => (typeof x === "number" ? { process: x, at: this.t0 } : x));
    for (const s of starts) {
      if (!Number.isInteger(s.process) || s.process < 0 || s.process >= this.processes) {
        throw new RangeError("initiator out of range");
      }
    }
    const P = this.processes;
    const timer = new Timer();
    const procs = [];
    for (let i = 0; i < P; i++) {
      procs.push({ id: i, tokens: 0, delivered: 0, recorded: null, markersSent: 0, noopSends: 0 });
    }
    const logs = [];
    for (let q = 0; q < P; q++) {
      const row = [];
      for (let p = 0; p < P; p++) row.push({ closed: false, messages: [] });
      logs.push(row);
    }
    const trace = [];
    for (let i = 0; i < P; i++) trace.push([]);
    const msgs = [];
    const markers = [];
    for (let i = 0; i < P; i++) markers.push(new Array(P).fill(0));
    const injections = [];
    for (let i = 0; i < P; i++) injections.push([]);
    let markerCount = 0;
    let markerDeliveries = 0;

    const record = (p, at) => {
      const proc = procs[p];
      proc.recorded = { time: at, delivered: proc.delivered, tokens: proc.tokens };
      for (let q = 0; q < P; q++) {
        if (q === p) continue;
        markers[p][q] += 1;
        markerCount += 1;
        proc.markersSent += 1;
        const from = p;
        const to = q;
        timer.schedule(at + this.delay, () => {
          markerDeliveries += 1;
          if (procs[to].recorded === null) record(to, timer.now);
          logs[to][from].closed = true;
        });
      }
    };

    const deliver = (from, to, tokens, id, recvTime) => {
      timer.schedule(recvTime, () => {
        const proc = procs[to];
        proc.delivered += 1;
        proc.tokens += tokens;
        trace[to].push({ kind: "recv", from, tokens });
        const channel = logs[to][from];
        if (proc.recorded !== null && !channel.closed) channel.messages.push(id);
      });
    };

    // The whole application trace is scheduled up front, so the application is independent of the
    // snapshot algorithm and both are driven by the same deterministic event order.
    for (let p = 0; p < P; p++) {
      for (const action of this.actions[p]) {
        timer.schedule(action.at, () => {
          const proc = procs[p];
          proc.delivered += 1;
          if (action.kind === "internal") {
            proc.tokens += 1;
            injections[p].push(timer.now);
            trace[p].push({ kind: "internal", tokens: 1 });
            return;
          }
          const take = Math.min(proc.tokens, action.take);
          if (take === 0) {
            proc.noopSends += 1;
            trace[p].push({ kind: "send", to: action.peer, tokens: 0 });
            return;
          }
          proc.tokens -= take;
          trace[p].push({ kind: "send", to: action.peer, tokens: take });
          const id = msgs.length;
          msgs.push({ id, from: p, to: action.peer, tokens: take, sendTime: timer.now, recvTime: timer.now + this.delay });
          deliver(p, action.peer, take, id, timer.now + this.delay);
        });
      }
    }
    for (const s of starts) timer.schedule(s.at, () => record(s.process, timer.now));
    timer.drain();

    return {
      processes: P,
      startedAt: Math.min(...starts.map((s) => s.at)),
      logs,
      trace,
      msgs,
      injections,
      markers,
      markerCount,
      markerDeliveries,
      noopSends: procs.reduce((a, c) => a + c.noopSends, 0),
      // the snapshot itself
      snapshot: procs.map((proc) => ({ id: proc.id, recorded: proc.recorded, markersSent: proc.markersSent })),
    };
  }
}

/**
 * Independent verifier. It only looks at the recorded run and re-derives every expected quantity
 * from message timestamps, so it shares no logic with the algorithm above.
 */
function verifySnapshot(pipeline, run) {
  const problems = [];
  const P = run.processes;
  const rec = run.snapshot.map((s) => s.recorded);
  const fail = (msg) => problems.push(msg);

  // 1. liveness: every process recorded exactly once, after the snapshot started.
  for (let p = 0; p < P; p++) {
    if (rec[p] === null) {
      fail(`process ${p} never recorded its state`);
      continue;
    }
    if (rec[p].time < run.startedAt) fail(`process ${p} recorded before the snapshot started`);
  }

  // 2. consistency of the cut: no receive inside the cut whose send is outside it.
  for (const m of run.msgs) {
    const sendInside = m.sendTime < rec[m.from].time;
    const recvInside = m.recvTime < rec[m.to].time;
    if (recvInside && !sendInside) {
      fail(`message ${m.id} (${m.from}->${m.to}) received inside the cut but sent outside it`);
    }
    if (!sendInside && recvInside) continue;
  }

  // 3. channel states are exactly the messages crossing the cut on that channel.
  let loggedMessages = 0;
  let loggedTokens = 0;
  for (let q = 0; q < P; q++) {
    for (let p = 0; p < P; p++) {
      if (p === q) continue;
      const expected = run.msgs
        .filter((m) => m.from === p && m.to === q && m.sendTime < rec[p].time && m.recvTime > rec[q].time)
        .map((m) => m.id)
        .sort((a, b) => a - b);
      const actual = run.logs[q][p].messages.slice().sort((a, b) => a - b);
      if (expected.length !== actual.length || expected.some((id, i) => id !== actual[i])) {
        fail(`channel ${p}->${q}: recorded [${actual}] but the cut requires [${expected}]`);
      }
      for (const id of actual) {
        loggedMessages += 1;
        loggedTokens += run.msgs[id].tokens;
      }
    }
  }

  // 4. token conservation across the cut.
  let held = 0;
  let injectedBefore = 0;
  for (let p = 0; p < P; p++) {
    held += rec[p].tokens;
    injectedBefore += run.injections[p].filter((t) => t < rec[p].time).length;
  }
  if (held + loggedTokens !== injectedBefore) {
    fail(`token conservation broken: held ${held} + in-flight ${loggedTokens} != injected ${injectedBefore}`);
  }

  // 5. replay oracle: the recorded state is the fold of the first `delivered` trace entries.
  for (let p = 0; p < P; p++) {
    if (rec[p].delivered > run.trace[p].length) {
      fail(`process ${p} recorded more events (${rec[p].delivered}) than it processed`);
      continue;
    }
    let tokens = 0;
    for (let i = 0; i < rec[p].delivered; i++) {
      const e = run.trace[p][i];
      tokens += e.kind === "recv" ? e.tokens : e.kind === "internal" ? 1 : -e.tokens;
    }
    if (tokens !== rec[p].tokens) fail(`process ${p}: replay gives ${tokens} tokens, snapshot has ${rec[p].tokens}`);
  }

  // 6. the cut contains everything that certainly happened before the snapshot started.
  for (let p = 0; p < P; p++) {
    const certain = pipeline.actionsBefore(p, run.startedAt);
    if (rec[p].delivered < certain) fail(`process ${p} cut misses pre-start events (${rec[p].delivered} < ${certain})`);
  }

  // 7. bounded marker traffic: one marker per channel, forever (a process forwards only once).
  for (let p = 0; p < P; p++) {
    for (let q = 0; q < P; q++) {
      if (p === q) continue;
      if (run.markers[p][q] !== 1) fail(`channel ${p}->${q} carried ${run.markers[p][q]} markers, expected 1`);
    }
  }
  if (run.markerCount !== P * (P - 1)) fail(`total markers ${run.markerCount}, expected ${P * (P - 1)}`);
  if (run.markerDeliveries !== P * (P - 1)) fail(`delivered markers ${run.markerDeliveries}`);
  for (let p = 0; p < P; p++) {
    const expected = rec[p] === null ? 0 : P - 1;
    if (run.snapshot[p].markersSent !== expected) fail(`process ${p} sent ${run.snapshot[p].markersSent} markers`);
  }
  if (problems.length === 0) {
    return { ok: true, problems, metrics: { loggedMessages, loggedTokens, held, injectedBefore } };
  }
  return { ok: false, problems, metrics: { loggedMessages, loggedTokens, held, injectedBefore } };
}

/** JSON round-trip clone of a plain run record (used to build deliberately corrupted snapshots). */
function cloneRun(run) {
  return JSON.parse(JSON.stringify(run));
}

function selfCheck() {
  // --- configuration validation ---------------------------------------------------------------
  assert.throws(() => new Pipeline({ processes: 0 }), /processes/);
  assert.throws(() => new Pipeline({ actionsPerProcess: 0 }), /actionsPerProcess/);
  assert.throws(() => new Pipeline({ processes: 4, delay: 4 }), /multiple of 3/);
  assert.throws(() => new Pipeline({ seed: 1.5 }), /seed/);
  const one = new Pipeline({ processes: 3, actionsPerProcess: 4, seed: 1 });
  assert.throws(() => one.run([3]), /initiator/);
  assert.strictEqual(one.t0 % 3, 1);
  assert.strictEqual(one.delay % 3, 0);
  assert.ok(one.actions.every((list, p) => list.every((a) => a.at % 3 === 0)));
  assert.ok(one.actions.every((list) => list.every((a, k) => k === 0 || a.at > list[k - 1].at)));

  // --- base case: 8 processes, single initiator ------------------------------------------------
  const pipeline = new Pipeline({ processes: 8, actionsPerProcess: 30, seed: 852 });
  const run = pipeline.run([0]);
  const check = verifySnapshot(pipeline, run);
  assert.deepStrictEqual(check.problems, []);
  assert.ok(check.ok);
  assert.ok(run.msgs.length > 40, "the trace needs real traffic");
  assert.ok(check.metrics.loggedTokens > 0, "some tokens must be in flight across the cut");
  assert.ok(check.metrics.injectedBefore > check.metrics.held, "tokens must also be in flight or delivered");
  assert.ok(run.noopSends > 0, "starved sends exercise the token accounting path");

  // --- degenerate pipeline: one process, no channels ---------------------------------------------
  const solo = new Pipeline({ processes: 1, actionsPerProcess: 6, seed: 9 });
  const soloRun = solo.run([0]);
  assert.strictEqual(soloRun.markerCount, 0);
  assert.strictEqual(soloRun.snapshot[0].markersSent, 0);
  const soloCheck = verifySnapshot(solo, soloRun);
  assert.ok(soloCheck.ok && soloCheck.metrics.loggedMessages === 0);
  assert.strictEqual(soloCheck.metrics.held, soloCheck.metrics.injectedBefore);
  assert.ok(soloRun.snapshot[0].recorded.delivered > 0);

  // --- initiator at the very end: cut == final state, nothing in flight --------------------------
  const late = new Pipeline({ processes: 5, actionsPerProcess: 8, seed: 77 });
  const lateRun = late.run([{ process: 2, at: late.span + late.delay + 1 }]);
  assert.ok(verifySnapshot(late, lateRun).ok);
  assert.strictEqual(verifySnapshot(late, lateRun).metrics.loggedTokens, 0);
  assert.ok(lateRun.snapshot.every((s, p) => s.recorded.delivered === lateRun.trace[p].length));

  // --- initiator at the very beginning: the cut is a true prefix of the history ------------------
  const early = new Pipeline({ processes: 5, actionsPerProcess: 8, seed: 78 });
  const earlyRun = early.run([{ process: 0, at: 1 }]);
  const earlyCheck = verifySnapshot(early, earlyRun);
  assert.ok(earlyCheck.ok, earlyCheck.problems.join("; "));
  assert.strictEqual(earlyRun.snapshot[0].recorded.delivered, 1); // only the action at t=0
  assert.ok(earlyRun.snapshot.every((s, p) => s.recorded.delivered < earlyRun.trace[p].length));
  const covered = (r) => r.snapshot.reduce((acc, s) => acc + s.recorded.delivered, 0);
  assert.ok(covered(earlyRun) < covered(lateRun), "an early cut must cover fewer events than a late one");
  assert.strictEqual(covered(lateRun), lateRun.trace.reduce((a, t) => a + t.length, 0));

  // --- multiple initiators: still one marker per channel, still consistent -----------------------
  const multi = new Pipeline({ processes: 6, actionsPerProcess: 12, seed: 99 });
  const multiRun = multi.run([0, 3, 5]);
  const multiCheck = verifySnapshot(multi, multiRun);
  assert.ok(multiCheck.ok, multiCheck.problems.join("; "));
  assert.strictEqual(multiRun.markerCount, 6 * 5);
  const staggered = multi.run([{ process: 1, at: multi.t0 }, { process: 4, at: multi.t0 + 3 }]);
  assert.ok(verifySnapshot(multi, staggered).ok);

  // --- scale: 16 processes / 40 actions ----------------------------------------------------------
  const wide = new Pipeline({ processes: 16, actionsPerProcess: 40, seed: 4242 });
  const wideRun = wide.run([0, 8]);
  const wideCheck = verifySnapshot(wide, wideRun);
  assert.ok(wideCheck.ok, wideCheck.problems.join("; "));
  assert.strictEqual(wideRun.markerCount, 16 * 15);
  assert.ok(wideRun.msgs.length > 100);

  // --- negative checks: the verifier must reject corrupted snapshots ------------------------------
  const dropped = cloneRun(run);
  const channelWithState = dropped.logs.find((row, q) => row.some((c, p) => p !== q && c.messages.length > 0));
  for (let q = 0; q < dropped.processes && channelWithState; q++) {
    for (let p = 0; p < dropped.processes; p++) {
      if (p !== q && dropped.logs[q][p].messages.length > 0) {
        dropped.logs[q][p].messages.pop(); // lose an in-flight message
        assert.strictEqual(verifySnapshot(pipeline, dropped).ok, false, "dropping a message must be detected");
        dropped.logs[q][p].messages = [];
        q = dropped.processes;
        break;
      }
    }
  }
  const invented = cloneRun(run);
  invented.logs[1][0].messages.push(0);
  assert.strictEqual(verifySnapshot(pipeline, invented).ok, false, "an invented in-flight message must be detected");

  const shifted = cloneRun(run);
  shifted.snapshot[3].recorded.delivered += 1; // pretend the cut is one event further along
  const shiftedResult = verifySnapshot(pipeline, shifted);
  assert.strictEqual(shiftedResult.ok, false, "a moved cut boundary must be detected");

  const doubleMarker = cloneRun(run);
  doubleMarker.markers[0][1] = 2;
  assert.strictEqual(verifySnapshot(pipeline, doubleMarker).ok, false, "extra marker traffic must be detected");

  // --- determinism --------------------------------------------------------------------------------
  const again = pipeline.run([0]);
  assert.deepStrictEqual(again.snapshot.map((s) => s.recorded), run.snapshot.map((s) => s.recorded));
  assert.deepStrictEqual(again.msgs, run.msgs);
  assert.notDeepStrictEqual(new Pipeline({ processes: 8, actionsPerProcess: 30, seed: 853 }).run([0]).msgs, run.msgs);

  console.log(`run: P=${pipeline.processes} actions=${pipeline.actionsPerProcess * pipeline.processes} messages=${run.msgs.length} markers=${run.markerCount}`);
  console.log(`cut: in-flight messages=${check.metrics.loggedMessages} in-flight tokens=${check.metrics.loggedTokens} held=${check.metrics.held} injected-before-cut=${check.metrics.injectedBefore}`);
  console.log(`scale: P=${wide.processes} messages=${wideRun.msgs.length} markers=${wideRun.markerCount}`);
  console.log("PASS");
}

module.exports = { Pipeline, Timer, verifySnapshot, cloneRun, mulberry32 };

if (require.main === module) {
  selfCheck();
}
