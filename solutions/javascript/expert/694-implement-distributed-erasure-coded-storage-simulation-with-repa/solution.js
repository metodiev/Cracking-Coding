/** Task 694 - Implement distributed erasure-coded storage simulation with repair. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Model: a small storage cluster that keeps every object as a Reed-Solomon (k, k+m) stripe spread
 * over k + m distinct nodes (the cluster must hold at least that many nodes, so one node failure can
 * never take two chunks of the same object). An object is `k * chunkSize` bytes: it is cut into k data
 * chunks and, for every byte offset inside a chunk, the k bytes are RS-encoded over GF(2^8) into m
 * parity bytes, so chunk index i is the stripe polynomial evaluated at the field point i + 1. Reading
 * needs any k of the k + m chunks: the coefficient solve is done once per erasure pattern and then
 * applied to every byte offset, exactly what a real storage node does.
 *
 * Repair: the lost chunks are regenerated from k surviving chunks and written to a healthy node that
 * holds no chunk of that object (a replacement node if one has been added, otherwise the placement
 * degrades to fewer nodes holding more chunks - which is what real clusters do under pressure). The
 * simulator counts read and written bytes so the classic trade-off can be measured: storage overhead
 * (k + m) / k instead of factor-3 replication, but k chunk reads to rebuild one lost chunk instead of
 * one replica copy.
 *
 * Simplified versus a real deployment: no network or disk latency, instant deterministic repairs, no
 * background scrubbing or rate limiting, no end-to-end checksums (integrity is checked by comparing
 * against the pre-failure chunks), fixed membership with an explicit addNode/drop simulation, and a
 * single repair-target policy. Verified: the stripe round-trip, all simultaneous failure patterns of
 * size m leaving every object readable and byte-identical, an m + 1 failure that really loses data,
 * refusal to serve fewer than k chunks, byte-identical repair of data *and* parity chunks, the repair
 * bandwidth accounting, the replication trade-off arithmetic, repeated fail/replace/repair cycles
 * leaving the data intact, and determinism.
 */
"use strict";

const FIELD_SIZE = 256;
const MODULUS = 0x11d;

function buildField() {
  const exp = new Uint8Array(FIELD_SIZE * 2);
  const log = new Uint8Array(FIELD_SIZE);
  let value = 1;
  for (let power = 0; power < FIELD_SIZE - 1; power++) {
    exp[power] = value;
    log[value] = power;
    value <<= 1;
    if (value & FIELD_SIZE) value ^= MODULUS;
    value &= 0xff;
  }
  for (let power = FIELD_SIZE - 1; power < FIELD_SIZE * 2; power++) exp[power] = exp[power - (FIELD_SIZE - 1)];
  return { exp, log };
}

const FIELD = buildField();

function mul(a, b) {
  if (a === 0 || b === 0) return 0;
  return FIELD.exp[FIELD.log[a] + FIELD.log[b]];
}

function inv(a) {
  if (a === 0) throw new RangeError("no inverse for zero");
  return FIELD.exp[FIELD_SIZE - 1 - FIELD.log[a]];
}

/** Evaluate the degree < k polynomial with these coefficients at x (Horner). */
function evalAt(coefficients, x) {
  let acc = 0;
  for (let i = coefficients.length - 1; i >= 0; i--) acc = (mul(acc, x) ^ coefficients[i]) & 0xff;
  return acc;
}

/** Field points of a stripe: 1, 2, ..., n. */
function stripePoints(n) {
  return Array.from({ length: n }, (_, i) => i + 1);
}

/** Inverse of the k x k Vandermonde matrix built on `points`; null when it is singular. */
function invertVandermonde(points, k) {
  const rows = points.map((x, rowIndex) => {
    const row = [1];
    for (let j = 1; j < k; j++) row.push(mul(row[j - 1], x));
    return [...row, ...Array.from({ length: k }, (_, index) => (index === rowIndex ? 1 : 0))];
  });
  for (let column = 0; column < k; column++) {
    let pivot = -1;
    for (let row = column; row < k; row++) {
      if (rows[row][column] !== 0) { pivot = row; break; }
    }
    if (pivot === -1) return null;
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const scale = inv(rows[column][column]);
    rows[column] = rows[column].map((value) => mul(value, scale));
    for (let row = 0; row < k; row++) {
      if (row === column || rows[row][column] === 0) continue;
      const factor = rows[row][column];
      rows[row] = rows[row].map((value, index) => (value ^ mul(factor, rows[column][index])) & 0xff);
    }
  }
  return rows.map((row) => row.slice(k));
}

/** Solve the k x k system once, then apply the same coefficients to every byte offset. */
function solveCoefficients(inverse, values) {
  const coefficients = new Array(values.length).fill(0);
  for (let row = 0; row < values.length; row++) {
    let sum = 0;
    for (let column = 0; column < values.length; column++) sum ^= mul(inverse[row][column], values[column]);
    coefficients[row] = sum & 0xff;
  }
  return coefficients;
}

/** Systematic stripe encoder: k data chunks in, m parity chunks out. */
function encodeStripe(dataChunks, m) {
  const k = dataChunks.length;
  const chunkSize = dataChunks[0].length;
  const points = stripePoints(k + m);
  const inverse = invertVandermonde(points.slice(0, k), k);
  const parity = Array.from({ length: m }, () => new Uint8Array(chunkSize));
  for (let offset = 0; offset < chunkSize; offset++) {
    const coefficients = solveCoefficients(inverse, dataChunks.map((chunk) => chunk[offset]));
    for (let j = 0; j < m; j++) parity[j][offset] = evalAt(coefficients, points[k + j]);
  }
  return parity;
}

/** Independent Lagrange evaluation used as the oracle for the encoder. */
function evaluateViaLagrange(points, values, x) {
  let total = 0;
  for (let i = 0; i < points.length; i++) {
    let basis = 1;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      basis = mul(basis, x ^ points[j]);
      basis = mul(basis, inv(points[i] ^ points[j]));
    }
    total ^= mul(values[i], basis);
  }
  return total & 0xff;
}

/**
 * Rebuild the chunks in `missing` from the survivors of `chunks` (an array where a missing chunk is
 * null). Returns a Map(index -> chunk), or null when fewer than k chunks survive.
 */
function decodeStripe(chunks, missing, k, m, chunkSize) {
  const points = stripePoints(k + m);
  const survivors = [];
  for (let i = 0; i < k + m; i++) if (chunks[i] !== null) survivors.push(i);
  if (survivors.length < k) return null;
  const basis = survivors.slice(0, k);
  const inverse = invertVandermonde(basis.map((index) => points[index]), k);
  const rebuilt = new Map();
  for (const index of missing) rebuilt.set(index, new Uint8Array(chunkSize));
  for (let offset = 0; offset < chunkSize; offset++) {
    const coefficients = solveCoefficients(inverse, basis.map((index) => chunks[index][offset]));
    for (const index of missing) rebuilt.get(index)[offset] = evalAt(coefficients, points[index]);
  }
  return rebuilt;
}

class Cluster {
  constructor({ nodes, k, m, chunkSize }) {
    if (!Number.isInteger(k) || k < 1) throw new RangeError("k must be a positive integer");
    if (!Number.isInteger(m) || m < 1) throw new RangeError("m must be a positive integer");
    if (!Number.isInteger(chunkSize) || chunkSize < 1) throw new RangeError("chunkSize must be positive");
    if (nodes.length < k + m) throw new RangeError("the cluster needs at least k + m nodes");
    this.nodes = [...nodes].sort((a, b) => a - b);
    this.failed = new Set();
    this.k = k;
    this.m = m;
    this.chunkSize = chunkSize;
    this.objects = new Map();                        // objectId -> Array(k + m) of { node, data|null }
    this.bytesRead = 0;
    this.bytesWritten = 0;
    this.repairs = 0;
  }

  /** Chunk i of an object goes to the i-th node after a deterministic rotation per object. */
  placement(objectId) {
    const shift = [...String(objectId)].reduce((acc, ch) => acc + ch.charCodeAt(0), 0) % this.nodes.length;
    return Array.from({ length: this.k + this.m }, (_, i) => this.nodes[(i + shift) % this.nodes.length]);
  }

  store(objectId, data) {
    if (data.length !== this.k * this.chunkSize) {
      throw new RangeError(`objects are exactly ${this.k * this.chunkSize} bytes`);
    }
    const dataChunks = [];
    for (let i = 0; i < this.k; i++) dataChunks.push(data.slice(i * this.chunkSize, (i + 1) * this.chunkSize));
    const parity = encodeStripe(dataChunks, this.m);
    const placement = this.placement(objectId);
    const stripe = [];
    for (let i = 0; i < this.k; i++) stripe.push({ node: placement[i], data: Uint8Array.from(dataChunks[i]) });
    for (let j = 0; j < this.m; j++) stripe.push({ node: placement[this.k + j], data: parity[j] });
    this.objects.set(objectId, stripe);
    this.bytesWritten += data.length;
    return stripe.length;
  }

  /** Reassemble an object from any k available chunks; null when the object is unreadable. */
  read(objectId) {
    const stripe = this.requireObject(objectId);
    const missing = stripe.map((chunk, index) => (chunk.data === null ? index : -1)).filter((i) => i >= 0);
    if (stripe.length - missing.length < this.k) return null;
    let chunks = stripe.map((chunk) => chunk.data);
    if (missing.length > 0) {
      const rebuilt = decodeStripe(chunks, missing, this.k, this.m, this.chunkSize);
      if (rebuilt === null) return null;
      chunks = chunks.map((chunk, index) => (chunk === null ? rebuilt.get(index) : chunk));
    }
    const total = new Uint8Array(this.k * this.chunkSize);
    for (let i = 0; i < this.k; i++) total.set(chunks[i], i * this.chunkSize);
    return total;
  }

  requireObject(objectId) {
    const stripe = this.objects.get(objectId);
    if (stripe === undefined) throw new Error(`unknown object ${objectId}`);
    return stripe;
  }

  /** Fail a node: every chunk of every object it holds is lost. Returns the number of chunks lost. */
  failNode(nodeId) {
    if (!this.nodes.includes(nodeId)) throw new Error(`unknown node ${nodeId}`);
    this.failed.add(nodeId);
    let lost = 0;
    for (const stripe of this.objects.values()) {
      for (const chunk of stripe) {
        if (chunk.node === nodeId && chunk.data !== null) {
          chunk.data = null;
          lost++;
        }
      }
    }
    return lost;
  }

  /** Add a replacement node - exactly what an operator does after a disk failure. */
  addNode(nodeId) {
    if (this.nodes.includes(nodeId)) throw new Error(`node ${nodeId} is already a member`);
    this.nodes = [...this.nodes, nodeId].sort((a, b) => a - b);
    return this;
  }

  /** Healthy nodes that hold no chunk of this object (preferred repair targets); may be empty. */
  freeNodes(objectId) {
    const held = new Set(this.requireObject(objectId)
      .filter((chunk) => chunk.data !== null)
      .map((chunk) => chunk.node));
    const healthy = this.nodes.filter((node) => !this.failed.has(node));
    const preferred = healthy.filter((node) => !held.has(node));
    return preferred.length > 0 ? preferred : healthy;
  }

  lostChunks(objectId) {
    return this.requireObject(objectId).map((chunk, index) => (chunk.data === null ? index : -1)).filter((i) => i >= 0);
  }

  /** Rebuild every lost chunk of one object from k survivors and place it on a free node. */
  repair(objectId) {
    const stripe = this.requireObject(objectId);
    const missing = this.lostChunks(objectId);
    if (missing.length === 0) return { chunksRepaired: 0, bytesRead: 0, bytesWritten: 0 };
    if (stripe.length - missing.length < this.k) return null;      // data is gone for good
    const rebuilt = decodeStripe(stripe.map((chunk) => chunk.data), missing, this.k, this.m, this.chunkSize);
    if (rebuilt === null) return null;
    const bytesRead = missing.length * this.k * this.chunkSize;    // k chunk reads per repaired chunk
    let written = 0;
    for (const index of missing) {
      const free = this.freeNodes(objectId);
      if (free.length === 0) throw new Error("no healthy node to repair onto");
      stripe[index] = { node: free[0], data: rebuilt.get(index) };
      written += this.chunkSize;
    }
    this.bytesRead += bytesRead;
    this.bytesWritten += written;
    this.repairs++;
    return { chunksRepaired: missing.length, bytesRead, bytesWritten: written };
  }

  status(objectId) {
    const stripe = this.requireObject(objectId);
    const live = stripe.filter((chunk) => chunk.data !== null);
    return { total: stripe.length, available: live.length, nodes: new Set(live.map((chunk) => chunk.node)).size };
  }

  /** Storage overhead of this code against plain factor-3 replication. */
  storageOverhead() {
    return (this.k + this.m) / this.k;
  }
}

class Lcg {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  byte() {
    return this.next() & 0xff;
  }

  bytes(count) {
    const out = [];
    for (let i = 0; i < count; i++) out.push(this.byte());
    return out;
  }
}

function* combinations(items, choose, start = 0, prefix = []) {
  if (prefix.length === choose) {
    yield prefix.slice();
    return;
  }
  for (let i = start; i <= items.length - (choose - prefix.length); i++) {
    prefix.push(items[i]);
    yield* combinations(items, choose, i + 1, prefix);
    prefix.pop();
  }
}

module.exports = {
  MODULUS, mul, inv, evalAt, invertVandermonde, encodeStripe, decodeStripe, evaluateViaLagrange,
  Cluster, Lcg, combinations,
};

if (require.main === module) {
  const assert = require("node:assert");

  const lcg = new Lcg(694);
  const NODES = [1, 2, 3, 4, 5, 6];
  const K = 4;
  const M = 2;
  const CHUNK = 4;
  const OBJECT_BYTES = K * CHUNK;
  const POINTS = [1, 2, 3, 4, 5, 6];

  // --- field and stripe primitives ----------------------------------------------------------
  assert.strictEqual(mul(0, 0x7f), 0);
  assert.strictEqual(mul(1, 0x7f), 0x7f);
  for (let a = 1; a < FIELD_SIZE; a++) assert.strictEqual(mul(a, inv(a)), 1, `inverse ${a}`);
  assert.strictEqual(evalAt([3, 0, 1], 2), 3 ^ 4, "p(x) = 3 + x^2");
  assert.strictEqual(invertVandermonde([1, 1, 1], 3), null, "repeated points make the matrix singular");
  const identity = invertVandermonde([1, 2, 3], 3);
  const trialCoefficients = [0x31, 0x7a, 0x0e];
  assert.deepStrictEqual(solveCoefficients(identity, [1, 2, 3].map((x) => evalAt(trialCoefficients, x))),
    trialCoefficients, "the inverse matrix undoes the evaluation");

  const dataChunks = [
    Uint8Array.from([1, 2, 3, 4]), Uint8Array.from([5, 6, 7, 8]),
    Uint8Array.from([9, 10, 11, 12]), Uint8Array.from([13, 14, 15, 16]),
  ];
  const parity = encodeStripe(dataChunks, M);
  assert.strictEqual(parity.length, M);
  assert.strictEqual(parity[0].length, CHUNK);
  for (let offset = 0; offset < CHUNK; offset++) {
    const values = dataChunks.map((chunk) => chunk[offset]);
    for (let j = 0; j < M; j++) {
      assert.strictEqual(parity[j][offset], evaluateViaLagrange(POINTS.slice(0, K), values, POINTS[K + j]),
        `parity ${j} offset ${offset} matches Lagrange interpolation`);
    }
  }
  const full = [...dataChunks, ...parity];
  for (const missing of combinations([0, 1, 2, 3, 4, 5], 2)) {
    const damaged = full.map((chunk, index) => (missing.includes(index) ? null : chunk));
    const rebuilt = decodeStripe(damaged, missing, K, M, CHUNK);
    for (const index of missing) {
      assert.deepStrictEqual([...rebuilt.get(index)], [...full[index]], `stripe repair ${missing}`);
    }
  }
  assert.strictEqual(decodeStripe(full.map(() => null), [0], K, M, CHUNK), null, "no survivor, no data");
  assert.strictEqual(decodeStripe([full[0], full[1], full[2], null, null, null], [3, 4, 5], K, M, CHUNK), null,
    "three survivors cannot rebuild a four-chunk stripe");
  assert.deepStrictEqual([...decodeStripe([full[0], null, full[2], full[3], null, full[5]], [1, 4], K, M, CHUNK).get(4)],
    [...full[4]], "four survivors rebuild the two gaps");

  // --- the cluster ---------------------------------------------------------------------------
  const cluster = new Cluster({ nodes: NODES, k: K, m: M, chunkSize: CHUNK });
  assert.strictEqual(cluster.storageOverhead(), 1.5, "1.5x overhead against factor-3 replication");
  assert.throws(() => new Cluster({ nodes: [1, 2, 3], k: K, m: M, chunkSize: CHUNK }), RangeError);
  assert.throws(() => cluster.store("o-odd", new Uint8Array(OBJECT_BYTES - 1)), RangeError);
  const original = lcg.bytes(OBJECT_BYTES);
  assert.strictEqual(cluster.store("o1", Uint8Array.from(original)), K + M);
  assert.deepStrictEqual([...cluster.read("o1")], original, "round trip");
  assert.strictEqual(new Set(cluster.objects.get("o1").map((chunk) => chunk.node)).size, K + M,
    "every chunk of one object lives on a different node");
  assert.deepStrictEqual(cluster.status("o1"), { total: 6, available: 6, nodes: 6 });
  for (const subset of combinations([0, 1, 2, 3, 4, 5], K)) {
    const scratch = new Cluster({ nodes: NODES, k: K, m: M, chunkSize: CHUNK });
    scratch.store("s", Uint8Array.from(original));
    const stripe = scratch.objects.get("s");
    for (let i = 0; i < K + M; i++) if (!subset.includes(i)) stripe[i].data = null;
    assert.deepStrictEqual([...scratch.read("s")], original, `read from survivors ${subset}`);
  }

  // --- durability: every set of m simultaneous failures is survivable ------------------------
  for (const failed of combinations(NODES, M)) {
    const test = new Cluster({ nodes: NODES, k: K, m: M, chunkSize: CHUNK });
    const payloads = [];
    for (let index = 0; index < 3; index++) {
      const data = Uint8Array.from(lcg.bytes(OBJECT_BYTES));
      payloads.push(data);
      test.store(`obj${index}`, data);
    }
    assert.strictEqual(failed.reduce((acc, node) => acc + test.failNode(node), 0), 3 * M,
      "each node failure costs one chunk per object");
    for (let index = 0; index < 3; index++) {
      assert.deepStrictEqual([...test.read(`obj${index}`)], [...payloads[index]], `read after ${failed}`);
    }
  }

  // --- m + 1 failures on the same object, and exactly m of them ------------------------------
  const doomed = new Cluster({ nodes: NODES, k: K, m: M, chunkSize: CHUNK });
  doomed.store("o1", Uint8Array.from(original));
  const chunkNodes = doomed.objects.get("o1").map((chunk) => chunk.node);
  for (const node of chunkNodes.slice(0, M + 1)) doomed.failNode(node);
  assert.strictEqual(doomed.status("o1").available, K - 1, "k - 1 surviving chunks");
  assert.strictEqual(doomed.read("o1"), null, "below the reconstruction threshold");
  assert.strictEqual(doomed.repair("o1"), null, "and the object is unrepairable");
  assert.strictEqual(doomed.status("o1").available, K - 1, "the failed repair changed nothing");
  const survivorCase = new Cluster({ nodes: NODES, k: K, m: M, chunkSize: CHUNK });
  survivorCase.store("o1", Uint8Array.from(original));
  for (const node of chunkNodes.slice(0, M)) survivorCase.failNode(node);
  assert.deepStrictEqual([...survivorCase.read("o1")], original, "exactly m failures are tolerated");

  // --- repair with and without a replacement node --------------------------------------------
  const healed = new Cluster({ nodes: NODES, k: K, m: M, chunkSize: CHUNK });
  healed.store("o1", Uint8Array.from(original));
  const before = healed.objects.get("o1").map((chunk) => Uint8Array.from(chunk.data));
  healed.failNode(2);
  assert.strictEqual(healed.lostChunks("o1").length, 1, "one chunk per object is lost per node failure");
  healed.addNode(7);
  const report = healed.repair("o1");
  assert.deepStrictEqual([report.chunksRepaired, report.bytesRead, report.bytesWritten],
    [1, K * CHUNK, CHUNK], "a repair reads exactly k chunks and writes one");
  assert.strictEqual(healed.bytesRead, report.bytesRead);
  assert.strictEqual(healed.repairs, 1);
  assert.deepStrictEqual(healed.objects.get("o1").map((chunk) => [...chunk.data]), before.map((chunk) => [...chunk]),
    "the repaired chunk is byte-identical");
  assert.deepStrictEqual([...healed.read("o1")], original, "and the object still reads");
  assert.deepStrictEqual(healed.status("o1"), { total: 6, available: 6, nodes: 6 },
    "the replacement node restores the node spread");
  assert.strictEqual(healed.repair("o1").chunksRepaired, 0, "a healthy object needs no repair");
  assert.strictEqual(healed.bytesRead, report.bytesRead, "and consumes no bandwidth");
  assert.throws(() => healed.addNode(7), Error, "a node cannot be added twice");
  const congested = new Cluster({ nodes: NODES, k: K, m: M, chunkSize: CHUNK });
  congested.store("o1", Uint8Array.from(original));
  congested.failNode(3);
  assert.strictEqual(congested.repair("o1").chunksRepaired, 1);
  assert.strictEqual(congested.status("o1").available, K + M, "the data is whole again");
  assert.strictEqual(congested.status("o1").nodes, K + M - 1, "but only five nodes hold chunks now");
  assert.deepStrictEqual([...congested.read("o1")], original);
  assert.ok(healed.storageOverhead() < 3, "EC stores less than factor-3 replication");
  assert.strictEqual(report.bytesRead / CHUNK, K, "and pays k repair reads per lost chunk");

  // --- repeated fail / replace / repair cycles ------------------------------------------------
  const cyclic = new Cluster({ nodes: NODES, k: K, m: M, chunkSize: CHUNK });
  const payloads = [];
  for (let index = 0; index < 3; index++) {
    const data = lcg.bytes(OBJECT_BYTES);
    payloads.push(data);
    cyclic.store(`c${index}`, Uint8Array.from(data));
  }
  let totalRepaired = 0;
  [3, 5, 1, 6].forEach((victim, round) => {
    assert.strictEqual(cyclic.failNode(victim) > 0, true, `node ${victim} held chunks`);
    cyclic.addNode(100 + round);
    for (let index = 0; index < 3; index++) {
      totalRepaired += cyclic.repair(`c${index}`).chunksRepaired;
      assert.strictEqual(cyclic.status(`c${index}`).available, K + M, `c${index} is whole after ${victim} failed`);
      assert.strictEqual(cyclic.status(`c${index}`).nodes, K + M, "and scattered over six nodes again");
      assert.deepStrictEqual([...cyclic.read(`c${index}`)], payloads[index], `c${index} still reads back`);
    }
  });
  assert.strictEqual(totalRepaired, 12, "four failures, one chunk each, three objects");
  assert.strictEqual(cyclic.bytesRead, totalRepaired * K * CHUNK);
  assert.strictEqual(cyclic.bytesWritten, 3 * OBJECT_BYTES + totalRepaired * CHUNK);

  // --- smaller shapes and errors ---------------------------------------------------------------
  const tiny = new Cluster({ nodes: [10, 20, 30, 40, 50], k: 3, m: 2, chunkSize: 2 });
  assert.strictEqual(tiny.storageOverhead().toFixed(3), "1.667");
  const small = Uint8Array.from(lcg.bytes(6));
  tiny.store("t", small);
  for (const node of [10, 20]) tiny.failNode(node);
  assert.deepStrictEqual([...tiny.read("t")], [...small], "k survivors are enough");
  tiny.repair("t");
  assert.deepStrictEqual([...tiny.read("t")], [...small]);
  assert.throws(() => tiny.read("missing"), Error);
  assert.throws(() => tiny.failNode(99), Error);

  // --- determinism -----------------------------------------------------------------------------
  const buildRun = () => {
    const run = new Cluster({ nodes: NODES, k: K, m: M, chunkSize: CHUNK });
    const generator = new Lcg(694);
    for (let index = 0; index < 3; index++) run.store(`d${index}`, Uint8Array.from(generator.bytes(OBJECT_BYTES)));
    run.failNode(4);
    run.addNode(9);
    for (let index = 0; index < 3; index++) run.repair(`d${index}`);
    return { payload: [...run.read("d0")], stats: [run.bytesRead, run.bytesWritten, run.repairs] };
  };
  const runA = buildRun();
  const runB = buildRun();
  assert.deepStrictEqual(runB.payload, runA.payload);
  assert.deepStrictEqual(runB.stats, runA.stats, "identical traffic on a second run");

  console.log(`failureSets=15 repaired=${totalRepaired} overhead=${cluster.storageOverhead()}`);
  console.log("PASS");
}
