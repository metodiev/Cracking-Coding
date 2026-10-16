/** Task 851 - Implement latency-slash-throughput tradeoff benchmarks for various consistency levels.
 *  (Tier: Master, CTCI Ch 9 — System Design and Scalability)
 *
 * Model. A replicated key-value store: N replicas sit behind a coordinator, replica i acknowledges
 * after rtt[i] abstract time units. A write with quorum W must collect W acknowledgements, a read
 * with quorum R waits for R responses and returns the highest version among them (coordinator-side
 * read repair). Quorums are drawn uniformly at random per operation. Consistency levels are quorum
 * configurations plus a replication mode:
 *
 *   eventual : W = 1, R = 1, the coordinator answers after the first ack and the value reaches the
 *              remaining replicas ASYNC_DELAY later (asynchronous replication).
 *   quorum   : W = R = floor(N/2) + 1   (majority quorums, W + R > N)
 *   all      : W = R = N               (every replica must ack)
 *
 * Three checks are pure safety/liveness, the rest are cost bounds measured against independent
 * oracles.
 *
 * What is simplified compared with a real deployment: no clock skew, no node failures or retries,
 * no partial partitions, no per-replica queueing, one value per key, replication delays are exact
 * integers, and the discrete-event core advances only on explicitly scheduled events (no wall
 * clock, no timers, no threads, no I/O). All choices come from one seeded mulberry32 stream, so a
 * run is bit-for-bit reproducible.
 *
 * Invariants verified:
 *   safety      W + R > N  =>  no read ever misses a completed write, and a client that writes and
 *               then reads can never observe an older version than its own write. The observed
 *               stale-read rate of the eventual level is compared with the exact combinatorial
 *               probability C(N-W,R)/C(N,R) that a random read quorum misses a random write quorum.
 *   liveness    after the asynchronous replication queue drains, every replica holds the latest
 *               version (convergence).
 *   cost bounds mean operation latency equals the exact expectation E[max rtt over a random quorum
 *               subset], enumerated exhaustively over all C(N,k) subsets — an oracle that shares no
 *               code with the simulator; mean latency and p95 grow monotonically and throughput
 *               falls as the quorum grows.
 *   determinism two identical runs produce identical metrics.
 */
"use strict";

const assert = require("node:assert");

const READ = "read";
const WRITE = "write";
const MIXED = [WRITE, READ, READ, READ];
const WRITE_ONLY = [WRITE];
const READ_ONLY = [READ];
const WRITE_THEN_READ = [WRITE, READ];

/** mulberry32 — the single source of randomness in this file (deterministic, seedable). */
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

/** Exact binomial coefficient; N <= 32 keeps the floating point arithmetic exact. */
function choose(n, k) {
  if (k < 0 || k > n) return 0;
  let out = 1;
  for (let i = 0; i < k; i++) out = (out * (n - i)) / (i + 1);
  return Math.round(out);
}

/** Deterministic per-replica acknowledgement latency, in abstract time units. */
function replicaRtts(n, seed) {
  const rnd = mulberry32(seed);
  const rtts = [];
  for (let i = 0; i < n; i++) rtts.push(6 + (i % 3) * 4 + Math.floor(rnd() * 3));
  return rtts;
}

/** Level name -> quorum configuration. Levels are the only difference between benchmarks. */
const LEVELS = {
  eventual: (n) => ({ w: 1, r: 1 }),
  quorum: (n) => ({ w: Math.floor(n / 2) + 1, r: Math.floor(n / 2) + 1 }),
  all: (n) => ({ w: n, r: n }),
};

/** Binary min-heap over (time, insertionOrder) — deterministic tie-breaking. */
class Timer {
  constructor() {
    this.heap = [];
    this.seq = 0;
    this.now = 0;
    this.busy = true;
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

  next() {
    const heap = this.heap;
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let small = i;
        if (l < heap.length && Timer.before(heap[l], heap[small])) small = l;
        if (r < heap.length && Timer.before(heap[r], heap[small])) small = r;
        if (small === i) break;
        const tmp = heap[i];
        heap[i] = heap[small];
        heap[small] = tmp;
        i = small;
      }
    }
    return top;
  }

  /** Runs the simulation until the event queue is empty (that drain is the liveness step). */
  drain() {
    while (this.heap.length > 0) {
      const event = this.next();
      this.now = event.t;
      event.run();
    }
    this.busy = false;
  }
}

/**
 * Discrete-event replicated store. One closed-loop client pool issues a fixed pattern of operations;
 * every operation's latency is the maximum acknowledgement latency of its randomly drawn quorum,
 * and each operation completion frees the client slot (a closed queueing system, so throughput
 * saturates and the latency/throughput tradeoff is observable).
 */
class QuorumStore {
  constructor({ n, w, r, rtts, seed, pattern, concurrency, opsPerClient, asyncDelay }) {
    if (!Number.isInteger(n) || n < 1) throw new RangeError("replica count must be a positive integer");
    if (!Number.isInteger(w) || w < 1 || w > n) throw new RangeError("write quorum out of range");
    if (!Number.isInteger(r) || r < 1 || r > n) throw new RangeError("read quorum out of range");
    if (!Number.isInteger(concurrency) || concurrency < 1) throw new RangeError("concurrency must be >= 1");
    if (!Number.isInteger(opsPerClient) || opsPerClient < 0) throw new RangeError("opsPerClient must be >= 0");
    if (!Array.isArray(pattern) || pattern.length === 0) throw new RangeError("pattern must not be empty");
    this.n = n;
    this.w = w;
    this.r = r;
    this.rtts = rtts.slice();
    this.asyncDelay = asyncDelay === undefined ? 3 * Math.max(...this.rtts) + 5 : asyncDelay;
    this.rand = mulberry32(seed);
    this.pattern = pattern.slice();
    this.concurrency = concurrency;
    this.opsPerClient = opsPerClient;
    this.timers = new Timer();
    this.replicas = [];
    for (let i = 0; i < n; i++) this.replicas.push({ version: 0, value: null });
    this.clients = [];
    for (let c = 0; c < concurrency; c++) this.clients.push({ id: c, done: 0, sessionVersion: 0 });
    this.version = 0;
    this.committed = 0;
    this.staleReads = 0;
    this.reads = 0;
    this.rywViolations = 0;
    this.latencies = [];
    this.makespan = 0;
  }

  /** Uniformly random k-subset of replica ids (partial Fisher-Yates on the seeded stream). */
  pickQuorum(k) {
    const idx = [];
    for (let i = 0; i < this.n; i++) idx.push(i);
    for (let i = 0; i < k; i++) {
      const j = i + Math.floor(this.rand() * (this.n - i));
      const tmp = idx[i];
      idx[i] = idx[j];
      idx[j] = tmp;
    }
    return idx.slice(0, k).sort((a, b) => a - b);
  }

  apply(replica, version, value) {
    if (version > this.replicas[replica].version) this.replicas[replica] = { version, value };
  }

  issueWrite(t, client) {
    const version = ++this.version;
    const value = `v${version}`;
    const quorum = this.pickQuorum(this.w);
    let latency = 0;
    for (const i of quorum) {
      latency = Math.max(latency, this.rtts[i]);
      this.timers.schedule(t + this.rtts[i], () => this.apply(i, version, value));
    }
    // Completion is scheduled before the client's next step, so at equal timestamps the commit is
    // visible to the following operation.
    this.timers.schedule(t + latency, () => {
      if (version > this.committed) this.committed = version;
      if (version > client.sessionVersion) client.sessionVersion = version;
    });
    // Asynchronous fan-out to the replicas outside the quorum — the only replication path the
    // eventual level has, and the liveness path that makes every level converge.
    for (let i = 0; i < this.n; i++) {
      if (!quorum.includes(i)) this.timers.schedule(t + this.asyncDelay, () => this.apply(i, version, value));
    }
    return latency;
  }

  /** Replica state is sampled when the request is issued; the latency then delays the response. */
  issueRead(t, client) {
    const quorum = this.pickQuorum(this.r);
    let latency = 0;
    let version = 0;
    for (const i of quorum) {
      latency = Math.max(latency, this.rtts[i]);
      if (this.replicas[i].version > version) version = this.replicas[i].version;
    }
    this.reads += 1;
    if (version < this.committed) this.staleReads += 1;
    if (version < client.sessionVersion) this.rywViolations += 1;
    return latency;
  }

  step(clientId) {
    const client = this.clients[clientId];
    const kind = this.pattern[client.done % this.pattern.length];
    const t = this.timers.now;
    const latency = kind === WRITE ? this.issueWrite(t, client) : this.issueRead(t, client);
    this.latencies.push({ kind, value: latency });
    client.done += 1;
    if (client.done < this.opsPerClient) this.timers.schedule(t + latency, () => this.step(clientId));
    else this.makespan = Math.max(this.makespan, t + latency);
  }

  run() {
    if (this.opsPerClient > 0) {
      for (let c = 0; c < this.concurrency; c++) this.timers.schedule(0, () => this.step(c));
    }
    this.timers.drain();
    return this;
  }

  /** Every replica holds the newest version once the replication queue has drained. */
  converged() {
    return this.replicas.every((replica) => replica.version === this.version);
  }

  mean(values) {
    return values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length;
  }

  metrics(label) {
    const all = this.latencies.map((x) => x.value);
    const reads = this.latencies.filter((x) => x.kind === READ).map((x) => x.value);
    const writes = this.latencies.filter((x) => x.kind === WRITE).map((x) => x.value);
    const sorted = all.slice().sort((a, b) => a - b);
    const p95 = sorted.length === 0 ? 0 : sorted[Math.min(sorted.length - 1, Math.floor(0.95 * sorted.length))];
    const ops = all.length;
    return {
      label,
      w: this.w,
      r: this.r,
      ops,
      reads: this.reads,
      makespan: this.makespan,
      meanLatency: this.mean(all),
      meanReadLatency: this.mean(reads),
      meanWriteLatency: this.mean(writes),
      p95Latency: p95,
      throughput: this.makespan === 0 ? 0 : ops / this.makespan,
      staleReads: this.staleReads,
      staleRate: this.reads === 0 ? 0 : this.staleReads / this.reads,
      rywViolations: this.rywViolations,
      converged: this.converged(),
      replicaVersions: this.replicas.map((replica) => replica.version),
    };
  }
}

/** Runs one benchmark configuration and returns its metrics. */
function benchmark(level, { n = 5, seed = 851, concurrency = 4, opsPerClient = 600, pattern = MIXED } = {}) {
  const cfg = LEVELS[level](n);
  const rtts = replicaRtts(n, 1000 + seed);
  const store = new QuorumStore({
    n,
    w: cfg.w,
    r: cfg.r,
    rtts,
    seed,
    pattern,
    concurrency,
    opsPerClient,
  });
  return store.run().metrics(level);
}

/** Oracle 1: exact expectation of max rtt over a uniformly random k-subset (exhaustive). */
function expectedMaxRtt(rtts, k) {
  const n = rtts.length;
  let sum = 0;
  let count = 0;
  const pick = [];
  const walk = (start) => {
    if (pick.length === k) {
      let best = 0;
      for (const i of pick) best = Math.max(best, rtts[i]);
      sum += best;
      count += 1;
      return;
    }
    for (let i = start; i < n; i++) {
      pick.push(i);
      walk(i + 1);
      pick.pop();
    }
  };
  walk(0);
  return { expected: sum / count, subsets: count };
}

/** Oracle 2: exact probability that a random R-quorum misses a random W-quorum (disjoint). */
function staleProbability(n, w, r) {
  return choose(n - w, r) / choose(n, r);
}

function selfCheck() {
  // --- configuration validation: only legal quorum shapes are accepted -------------------------
  assert.throws(() => new QuorumStore({ n: 0, w: 1, r: 1 }), /replica count/);
  assert.throws(() => benchmark("linearizable"), TypeError);
  assert.throws(() => new QuorumStore({ n: 5, w: 6, r: 1 }), /write quorum/);
  assert.throws(() => new QuorumStore({ n: 5, w: 1, r: 6 }), /read quorum/);
  assert.throws(() => new QuorumStore({ n: 5, w: 1, r: 5, concurrency: 0 }), /concurrency/);
  assert.throws(() => new QuorumStore({ n: 4, w: 2, r: 2, concurrency: 1, opsPerClient: 10, pattern: [] }), /pattern/);

  // LEVELS is the whole definition of the tradeoff; majority quorums must intersect.
  const n5 = 5;
  assert.strictEqual(LEVELS.quorum(n5).w + LEVELS.quorum(n5).r > n5, true);
  assert.strictEqual(LEVELS.eventual(n5).w + LEVELS.eventual(n5).r, 2);
  assert.strictEqual(LEVELS.all(n5).w, n5);

  // --- edge case N = 1: every level degenerates to the single replica --------------------------
  const one = ["eventual", "quorum", "all"].map((level) => benchmark(level, { n: 1, opsPerClient: 40, concurrency: 1 }));
  for (const m of one) {
    assert.strictEqual(m.staleReads, 0, `${m.label} single replica must never be stale`);
    assert.strictEqual(m.rywViolations, 0);
    assert.ok(m.converged);
    assert.strictEqual(m.meanLatency, 1 * replicaRtts(1, 1000 + 851)[0]);
  }
  assert.strictEqual(one[0].meanLatency, one[2].meanLatency);

  // --- safety: W + R > N never returns a stale read (sequential client, many write/read pairs) --
  const strongest = benchmark("all", { pattern: WRITE_THEN_READ, opsPerClient: 400, concurrency: 1 });
  const majority = benchmark("quorum", { pattern: WRITE_THEN_READ, opsPerClient: 400, concurrency: 1 });
  for (const m of [strongest, majority]) {
    assert.strictEqual(m.staleReads, 0, `${m.label}: W+R>N must be linearizable-ish`);
    assert.strictEqual(m.rywViolations, 0, `${m.label}: read-your-writes must hold`);
    assert.strictEqual(m.reads, 200);
    assert.ok(m.converged, `${m.label} must converge`);
  }

  // --- eventual level: staleness is real, and its rate matches the exact combinatorial model ----
  const weak = benchmark("eventual", { pattern: WRITE_THEN_READ, opsPerClient: 3000, concurrency: 1 });
  const exactRate = staleProbability(5, 1, 1); // C(4,1)/C(5,1) = 0.8
  assert.strictEqual(exactRate, 0.8);
  assert.ok(weak.staleReads > 0, "eventual reads must be able to miss a completed write");
  assert.ok(weak.rywViolations > 0, "eventual readers must be able to lose their own write");
  assert.ok(
    Math.abs(weak.staleRate - exactRate) < 0.05,
    `stale rate ${weak.staleRate.toFixed(3)} should match the exact ${exactRate}`,
  );
  // C(N-W,R)/C(N,R) is monotone: intersecting quorums remove staleness entirely.
  assert.strictEqual(staleProbability(5, 3, 3), 0);
  assert.strictEqual(staleProbability(5, 4, 2), 0);

  // --- liveness: full convergence after the async queue drains ---------------------------------
  assert.ok(weak.converged && weak.replicaVersions.every((v) => v === weak.replicaVersions[0]));
  assert.ok(weak.replicaVersions[0] > 0);

  // --- cost bounds: latency matches the exhaustive expectation, and ordering is monotone --------
  const rtts = replicaRtts(5, 1000 + 851);
  const writeOracle = expectedMaxRtt(rtts, 3);
  assert.strictEqual(writeOracle.subsets, choose(5, 3));
  const readOracle = expectedMaxRtt(rtts, 3);
  const wBench = benchmark("quorum", { pattern: WRITE_ONLY, opsPerClient: 2000, concurrency: 1 });
  const rBench = benchmark("quorum", { pattern: READ_ONLY, opsPerClient: 2000, concurrency: 1 });
  assert.ok(
    Math.abs(wBench.meanWriteLatency - writeOracle.expected) < 0.35,
    `simulated write latency ${wBench.meanWriteLatency.toFixed(3)} vs oracle ${writeOracle.expected.toFixed(3)}`,
  );
  assert.ok(Math.abs(rBench.meanReadLatency - readOracle.expected) < 0.35);
  assert.ok(rBench.meanLatency >= Math.min(...rtts) && rBench.meanLatency <= Math.max(...rtts));

  const bench = ["eventual", "quorum", "all"].map((level) =>
    benchmark(level, { opsPerClient: 700, concurrency: 4 }),
  );
  const [even, maj, all] = bench;
  assert.ok(even.meanLatency < maj.meanLatency, "weaker quorums must be faster");
  assert.ok(maj.meanLatency < all.meanLatency, "all-replica acks must be the slowest");
  assert.ok(even.throughput > maj.throughput && maj.throughput > all.throughput, "throughput must fall");
  for (const m of bench) {
    assert.ok(m.p95Latency >= m.meanLatency, `${m.label}: p95 must bound the mean`);
    assert.ok(m.p95Latency <= Math.max(...rtts) * Math.max(...[1, 2, 3]), "p95 must stay bounded");
    assert.ok(m.ops === 700 * 4);
  }
  // 'all' draws the single full quorum, so its latency distribution is degenerate.
  const allWrites = benchmark("all", { pattern: WRITE_ONLY, opsPerClient: 500, concurrency: 1 });
  assert.strictEqual(allWrites.meanLatency, Math.max(...replicaRtts(5, 1000 + 851)));

  // --- degenerate workload: no operations at all ------------------------------------------------
  const idle = benchmark("quorum", { opsPerClient: 0, concurrency: 3 });
  assert.strictEqual(idle.ops, 0);
  assert.strictEqual(idle.throughput, 0);
  assert.strictEqual(idle.meanLatency, 0);
  assert.strictEqual(idle.staleRate, 0);
  assert.ok(Number.isFinite(idle.p95Latency) && idle.converged);

  // --- determinism -------------------------------------------------------------------------------
  assert.deepStrictEqual(benchmark("quorum", { opsPerClient: 300, concurrency: 3 }), benchmark("quorum", { opsPerClient: 300, concurrency: 3 }));
  assert.notDeepStrictEqual(benchmark("quorum", { seed: 7, opsPerClient: 300 }), benchmark("quorum", { seed: 8, opsPerClient: 300 }));

  console.log(`oracle  write latency E[max rtt | W=3] = ${writeOracle.expected.toFixed(4)} over ${writeOracle.subsets} subsets`);
  console.log(`latency mean  eventual=${even.meanLatency.toFixed(2)}  quorum=${maj.meanLatency.toFixed(2)}  all=${all.meanLatency.toFixed(2)}`);
  console.log(`throughput    eventual=${even.throughput.toFixed(4)}  quorum=${maj.throughput.toFixed(4)}  all=${all.throughput.toFixed(4)}`);
  console.log(`staleness     eventual=${weak.staleRate.toFixed(3)} (exact ${exactRate})  quorum=0  all=0`);
  console.log("PASS");
}

module.exports = { QuorumStore, Timer, LEVELS, benchmark, expectedMaxRtt, staleProbability, replicaRtts, mulberry32, choose };

if (require.main === module) {
  selfCheck();
}
