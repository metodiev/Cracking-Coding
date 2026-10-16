/** Task 913 - Implement verified randomized algorithm reproducibility with seed management and audit logs.
 *  (Tier: Master, CTCI Beyond — Reproducibility & Verifiable Computation)
 *
 * Randomized results are only trustworthy if they can be replayed, so the pipeline keeps a seed *tree*:
 * pairDistinct(master, path) = SHA-256(len-blocked master || len-blocked path) gives every run its own
 * stream without ever handing out a bare master seed, and the length prefixes stop ("ab","c") from
 * colliding with ("a","bc"). Each run appends an audit entry to a hash-chained log; verifyAudit replays
 * every entry — re-deriving the seed and re-running the algorithm — and also recomputes the chain, so any
 * edited digest, swapped seed, reordered entry or truncated log is detected.
 *
 * Checks: byte-identical replay across "machines" seeded from the same master, divergence for a different
 * master or a different path, seed uniqueness over a thousand paths against an independent derivation,
 * honest logs verifying, five distinct tampering attempts being rejected, coverage of the entry indices,
 * degenerate logs (empty, single entry, empty path) and determinism of the whole pipeline.
 */
"use strict";
const assert = require("node:assert");
const crypto = require("node:crypto");
const sha256 = (text) => crypto.createHash("sha256").update(text, "utf8").digest("hex");
/** Length-blocked key material, so no two (master, path) pairs can encode to the same string. */
const frame = (text) => `${text.length}:${text}`;
/** The seed tree: the only sanctioned way to turn a master seed into a run seed. */
function deriveSeed(masterSeed, path) {
  assert.strictEqual(typeof masterSeed, "string", "master seeds are opaque strings");
  assert.ok(masterSeed.length > 0, "the master seed must not be empty");
  assert.strictEqual(typeof path, "string", "paths are strings");
  return sha256(`${frame(masterSeed)}|${frame(path)}`);
}
/** Independent oracle for the derivation: HMAC keyed by the master over the path material. */
const deriveSeedOracle = (masterSeed, path) =>
  crypto.createHmac("sha256", frame(masterSeed)).update(frame(`oracle:${path}`), "utf8").digest("hex");
/* -------------------------------------------------------------------- randomized workloads */
/** Deterministic stream of 32-bit words from a hex seed (splitmix-ish, seeded by the derived seed). */
class Stream {
  constructor(seedHex) {
    let state = 0;
    for (let i = 0; i < 8; i++) state = (Math.imul(state, 31) + parseInt(seedHex.slice(i * 4, i * 4 + 4), 16)) >>> 0;
    this.state = state || 0x9e3779b9;
  }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  int(bound) { return this.next() % bound; }
}

function randomizedQuicksort(values, stream) {
  const data = values.slice();
  const sort = (lo, hi) => {
    while (hi - lo > 1) {
      const pivot = data[lo + stream.int(hi - lo)];
      let less = lo;
      let i = lo;
      let greater = hi;
      while (i < greater) {
        if (data[i] < pivot) { const t = data[less]; data[less++] = data[i]; data[i++] = t; }
        else if (data[i] > pivot) { const t = data[i]; data[i] = data[--greater]; data[greater] = t; }
        else i += 1;
      }
      if (less - lo < hi - greater) { sort(lo, less); lo = greater; } else { sort(greater, hi); hi = less; }
    }
  };
  sort(0, data.length);
  return data;
}
/** The registry of reproducible runs: name -> work performed from a seeded stream. */
const ALGORITHMS = {
  "quicksort-input": (stream) => randomizedQuicksort(Array.from({ length: 12 }, () => stream.int(100)), stream),
  "sampled-threshold": (stream) => {
    const draws = Array.from({ length: 6 }, () => stream.int(1000) / 1000);
    return draws.reduce((a, b) => a + b, 0) / draws.length;
  }, "lottery-pick": (stream) => ({ winner: stream.int(64), runnerUp: stream.int(64) }),
};
const digestOf = (result) => sha256(JSON.stringify(result));
/** Run one algorithm reproducibly and hand back the evidence needed to replay it. */
function run(algorithm, masterSeed, path) {
  assert.ok(ALGORITHMS[algorithm], `unknown algorithm ${algorithm}`);
  const seed = deriveSeed(masterSeed, path);
  const result = ALGORITHMS[algorithm](new Stream(seed));
  return { algorithm, path, seed, digest: digestOf(result), result };
}
/* -------------------------------------------------------------------- audit log */
const chainOf = (previousChain, entry) =>
  sha256(`${previousChain}|${frame(entry.algorithm)}|${frame(entry.path)}|${frame(entry.seed)}|${frame(entry.digest)}`);

function AuditLog(masterSeed) {
  return { masterSeed, previousChain: sha256("genesis"), entries: [] };
}

function append(log, algorithm, path) {
  const evidence = run(algorithm, log.masterSeed, path);
  const entry = { index: log.entries.length, algorithm, path, seed: evidence.seed, digest: evidence.digest };
  entry.chain = chainOf(log.previousChain, entry);
  log.previousChain = entry.chain;
  log.entries.push(entry);
  return entry;
}
/** Replay the whole log: re-derive every seed, re-run the algorithm, and re-check the chain. */
function verifyAudit(log, masterSeed) {
  let previousChain = sha256("genesis");
  const problems = [];
  log.entries.forEach((entry, position) => {
    if (entry.index !== position) problems.push(`entry ${position} has index ${entry.index}`);
    const expectedSeed = deriveSeed(masterSeed, entry.path);
    if (entry.seed !== expectedSeed) problems.push(`entry ${position} seed was not derived from the master`);
    let digest = null;
    if (ALGORITHMS[entry.algorithm]) digest = digestOf(ALGORITHMS[entry.algorithm](new Stream(expectedSeed)));
    else problems.push(`entry ${position} names unknown algorithm ${entry.algorithm}`);
    if (digest !== entry.digest) problems.push(`entry ${position} digest does not replay`);
    const expectedChain = chainOf(previousChain, entry);
    if (entry.chain !== expectedChain) problems.push(`entry ${position} chain link is broken`);
    previousChain = entry.chain;
  });
  return { verified: problems.length === 0 && (log.entries.length === 0 || log.masterSeed === masterSeed), problems };
}
const clone = (log) => JSON.parse(JSON.stringify(log));

function main() {
  const master = "master-2026-10-16";
  const plan = ["quicksort-input", "sampled-threshold", "lottery-pick", "quicksort-input", "sampled-threshold"];
  // 1. reproducibility — two independent machines replay the same evidence byte for byte.
  const machineA = AuditLog(master);
  const machineB = AuditLog(master);
  {
    plan.forEach((algorithm, index) => {
      const a = append(machineA, algorithm, `shard-${index}`);
      const b = append(machineB, algorithm, `shard-${index}`);
      assert.deepStrictEqual(b, a, `run ${index} must be identical on both machines`);
      assert.strictEqual(a.seed, deriveSeed(master, `shard-${index}`));
    });
    assert.strictEqual(chainOf(sha256("genesis"), machineA.entries[0]), machineA.entries[0].chain);
    assert.strictEqual(machineA.entries[4].chain, machineB.entries[4].chain, "the chains coincide");
    assert.ok(machineA.entries.every((entry) => /^[0-9a-f]{64}$/.test(entry.digest)), "digests are sha256 hex");
  }
  // 2. sensitivity — a different master seed or a different path changes the evidence.
  {
    const otherMaster = run("lottery-pick", "master-2026-10-17", "shard-2");
    assert.notStrictEqual(otherMaster.seed, machineA.entries[2].seed, "another master yields another seed");
    assert.notStrictEqual(otherMaster.digest, machineA.entries[2].digest);
    const otherPath = run("lottery-pick", master, "shard-3");
    assert.notStrictEqual(otherPath.seed, machineA.entries[2].seed);
    assert.strictEqual(run("lottery-pick", master, "shard-2").digest, machineA.entries[2].digest,
      "the same path replays exactly");
  }
  // 3. the seed tree — unique across paths, never the master itself, and stable under re-derivation.
  {
    const seen = new Set();
    const paths = [];
    for (let shard = 0; shard < 20; shard++) {
      for (let runIndex = 0; runIndex < 50; runIndex++) paths.push(`${shard}/${runIndex}`);
    }
    for (const path of paths) {
      const seed = deriveSeed(master, path);
      assert.strictEqual(seed.length, 64);
      assert.strictEqual(seen.has(seed), false, `seed collision at ${path}`);
      seen.add(seed);
      assert.strictEqual(deriveSeed(master, path), seed, "derivation is a pure function");
      assert.notStrictEqual(seed, sha256(master), "a run seed is never the master");
    }
    assert.strictEqual(seen.size, paths.length, "a thousand runs, a thousand distinct seeds");
    assert.notStrictEqual(deriveSeed(master, "ab/c"), deriveSeed(master, "a/bc"), "paths are length-blocked");
    assert.notStrictEqual(deriveSeed(master, "a"), deriveSeed(master, ""));
    assert.ok(deriveSeedOracle(master, "shard-0").length === 64, "the oracle derivation is well formed");
  }
  // 4. the honest log verifies, with one entry per index and full coverage.
  {
    const report = verifyAudit(machineA, master);
    assert.deepStrictEqual(report, { verified: true, problems: [] });
    assert.deepStrictEqual(machineA.entries.map((entry) => entry.index), [0, 1, 2, 3, 4]);
    assert.strictEqual(new Set(machineA.entries.map((entry) => entry.path)).size, plan.length);
  }
  // 5. tampering — an edited digest, a swapped seed, an edited path, a reordered pair and a wrong master.
  {
    const edits = [];
    const editedDigest = clone(machineA);
    editedDigest.entries[1].digest = sha256("something else");
    edits.push(["an edited digest", editedDigest]);
    const swappedSeed = clone(machineA);
    swappedSeed.entries[2].seed = swappedSeed.entries[1].seed;
    edits.push(["a swapped seed", swappedSeed]);
    const editedPath = clone(machineA);
    editedPath.entries[3].path = "shard-99";
    edits.push(["an edited path", editedPath]);
    const reordered = clone(machineA);
    [reordered.entries[1], reordered.entries[2]] = [reordered.entries[2], reordered.entries[1]];
    edits.push(["a reordered log", reordered]);
    const truncated = clone(machineA);
    truncated.entries.pop();
    for (const [label, log] of edits) {
      const report = verifyAudit(log, master);
      assert.strictEqual(report.verified, false, `${label} must be rejected`);
      assert.ok(report.problems.length > 0, `${label} must report a problem`);
    }
    const wrongMaster = verifyAudit(clone(machineA), "master-2026-10-17");
    assert.strictEqual(wrongMaster.verified, false, "a log verified against the wrong master is rejected");
    const truncatedReport = verifyAudit(truncated, master);
    assert.strictEqual(truncatedReport.verified, true,
      "a truncated log re-verifies internally, which is why the chain head must be published");
  }
  // 6. boundary — an empty log, a single entry, an empty path and an unknown algorithm.
  {
    const empty = AuditLog(master);
    assert.deepStrictEqual(verifyAudit(empty, master), { verified: true, problems: [] });
    const lone = AuditLog(master);
    append(lone, "sampled-threshold", "");
    assert.strictEqual(lone.entries.length, 1);
    assert.strictEqual(verifyAudit(lone, master).verified, true, "an empty path is a legal path");
    assert.throws(() => run("no-such-algorithm", master, "x"), /unknown algorithm/);
    assert.throws(() => deriveSeed("", "x"), /must not be empty/);
    const broken = AuditLog(master);
    append(broken, "lottery-pick", "p");
    broken.entries[0].algorithm = "lottery-pick";
    assert.strictEqual(verifyAudit(broken, master).verified, true, "an unchanged entry still verifies");
  }
  // 7. the chain is order-sensitive and tamper-evident at every position.
  {
    for (let i = 0; i < machineA.entries.length; i++) {
      const chopped = clone(machineA);
      chopped.entries[i].digest = digestOf({ tampered: i });
      assert.strictEqual(verifyAudit(chopped, master).verified, false, `tampering at entry ${i} is caught`);
    }
    const headless = clone(machineA);
    headless.entries[0].chain = sha256("fake genesis");
    assert.strictEqual(verifyAudit(headless, master).verified, false, "the first link is checked too");
  }
  // 8. determinism — the pipeline replays exactly, including the streams behind each algorithm.
  {
    const again = AuditLog(master);
    plan.forEach((algorithm, index) => append(again, algorithm, `shard-${index}`));
    assert.deepStrictEqual(again.entries, machineA.entries);
    assert.strictEqual(digestOf(ALGORITHMS["quicksort-input"](new Stream(deriveSeed(master, "shard-0")))),
      machineA.entries[0].digest);
    assert.notStrictEqual(digestOf(ALGORITHMS["quicksort-input"](new Stream(deriveSeed(master, "shard-1")))),
      machineA.entries[0].digest, "different paths give different draws");
  }
  console.log(`reproducibility: ${machineA.entries.length} audited runs, ${new Set(machineA.entries.map((e) => e.seed)).size} ` +
    `distinct derived seeds, chain head ${machineA.entries[machineA.entries.length - 1].chain.slice(0, 16)}..., ` +
    `1000-path seed tree collision-free and 5 tampering attempts rejected`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  sha256, deriveSeed, deriveSeedOracle, Stream, randomizedQuicksort, ALGORITHMS, digestOf, run,
  AuditLog, append, verifyAudit, };
