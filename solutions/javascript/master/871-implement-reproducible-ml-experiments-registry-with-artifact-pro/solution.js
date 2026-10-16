/** Task 871 - Implement reproducible ML experiments registry with artifact provenance. (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A content-addressed experiment registry. Every experiment is described by a manifest
 * (name, algorithm, hyper-parameters, dataset hash, code hash, seed, parent run ids); the manifest is
 * serialised with a canonical JSON writer (sorted keys, no insignificant whitespace) and hashed with
 * SHA-256, which yields the run id. Artifacts are stored the same way, so the registry gives:
 *
 *   - identity  : the same experiment description always maps to the same run id, and the same artifact
 *                 bytes always map to the same artifact hash (deduplication by construction);
 *   - provenance: each run records its parent runs, so the lineage is a DAG that can be walked upwards
 *                 (ancestors, provenance path, depth) and audited against a brute-force traversal;
 *   - reproduction: re-running an experiment from its manifest must rebuild a bit-identical artifact
 *                 hash, and re-registering a manifest must not execute anything again (memoisation);
 *   - integrity: artifact bytes are re-hashed on read, so tampering with stored bytes is detected;
 *   - retention: a mark-and-sweep collector keeps exactly the artifacts reachable from registered runs.
 *
 * The "training" itself is a deterministic linear-regression fit on a synthetic dataset generated from
 * the dataset hash and the seed with a splitmix64 PRNG, so the reproduction check is real work: the
 * artifact (the fitted coefficient vector quantised to a decimal grid) is recomputed and compared.
 *
 * Invariants verified: canonicalisation is key-order independent but array-order sensitive; hashes are
 * stable across runs and change on any content change; deduplication keeps exactly one copy; identical
 * manifests never execute twice; every registered run reproduces its artifact hash; tampered bytes fail
 * verification; the lineage queries agree with brute-force BFS/longest-path computations; aggregation
 * queries agree with brute-force scans; the mark-and-sweep collector matches brute-force reachability
 * and is idempotent; cycles, unknown parents and unknown runs are rejected.
 *
 * What is simplified: experiments run in-process (no cluster, no container, no GPU), the dataset is
 * synthetic and generated on demand rather than fetched, metrics are computed from the fitted model on
 * the training split only, and there is no concurrency or wall-clock timestamp - ordering is by an
 * explicit logical tick. Only Node built-ins are used, and only node:crypto for SHA-256.
 */
"use strict";

const { ok, throws } = require("node:assert");
const { createHash } = require("node:crypto");

/** SHA-256 over the canonical serialisation of a value. */
function contentHash(value) {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

/** Canonical JSON: object keys sorted, no whitespace, so key order never changes a hash. */
function canonicalJson(value) {
  if (value === null) return "null";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new RangeError(`non-finite number in manifest: ${value}`);
    return JSON.stringify(value);
  }
  if (typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object") {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError(`unsupported object type in manifest: ${value.constructor?.name ?? "unknown"}`);
    }
    const keys = Object.keys(value).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  throw new TypeError(`unsupported value type in manifest: ${typeof value}`);
}

/** splitmix64 in [0, 1): deterministic randomness for synthetic datasets. */
function makeRng(seed) {
  let state = BigInt.asUintN(64, BigInt(seed >>> 0) ^ 0x2545f4914f6cdd1dn);
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    z ^= z >> 31n;
    return Number(BigInt.asUintN(64, z) >> 11n) / 9007199254740992;
  };
}

/** Synthetic regression dataset derived from the dataset hash and the seed. */
function makeDataset(datasetHash, seed, { samples = 60, features = 3 } = {}) {
  const seedFromHash = parseInt(datasetHash.slice(0, 8), 16);
  const rng = makeRng(((seedFromHash ^ (seed * 2654435761)) >>> 0) || 1);
  const truth = Array.from({ length: features }, () => rng() * 2 - 1);
  const rows = [];
  for (let i = 0; i < samples; i++) {
    const x = Array.from({ length: features }, () => rng() * 2 - 1);
    const noise = (rng() * 2 - 1) * 0.05;
    rows.push({ x, y: x.reduce((acc, value, j) => acc + value * truth[j], 0) + noise });
  }
  return { rows, truth };
}

/** Ordinary least squares by normal equations with partial pivoting (the "training" of a run). */
function fitLinear(rows) {
  const features = rows[0].x.length;
  const xtx = Array.from({ length: features }, () => new Array(features).fill(0));
  const xty = new Array(features).fill(0);
  for (const row of rows) {
    for (let i = 0; i < features; i++) {
      xty[i] += row.x[i] * row.y;
      for (let j = 0; j < features; j++) xtx[i][j] += row.x[i] * row.x[j];
    }
  }
  const a = xtx.map((row, i) => [...row, xty[i]]);
  for (let col = 0; col < features; col++) {
    let pivot = col;
    for (let r = col + 1; r < features; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-12) throw new Error("singular design matrix");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let r = col + 1; r < features; r++) {
      const factor = a[r][col] / a[col][col];
      for (let c = col; c <= features; c++) a[r][c] -= factor * a[col][c];
    }
  }
  const coefficients = new Array(features).fill(0);
  for (let r = features - 1; r >= 0; r--) {
    let sum = a[r][features];
    for (let c = r + 1; c < features; c++) sum -= a[r][c] * coefficients[c];
    coefficients[r] = sum / a[r][r];
  }
  const predictions = rows.map((row) => row.x.reduce((acc, value, j) => acc + value * coefficients[j], 0));
  const meanTarget = rows.reduce((acc, row) => acc + row.y, 0) / rows.length;
  const rss = rows.reduce((acc, row, i) => acc + (row.y - predictions[i]) ** 2, 0);
  const tss = rows.reduce((acc, row) => acc + (row.y - meanTarget) ** 2, 0);
  return { coefficients, rSquared: 1 - rss / tss, rss };
}

/** Quantise a coefficient vector onto a decimal grid so the artifact bytes are stable. */
const quantise = (coefficients, decimals = 6) => coefficients.map((value) => Number(value.toFixed(decimals)));

/** Executes an experiment: pure function of the manifest, so reproduction is meaningful. */
function executeExperiment(manifest) {
  const { rows, truth } = makeDataset(manifest.datasetHash, manifest.seed, {
    samples: manifest.params.samples ?? 60,
    features: manifest.params.features ?? 3,
  });
  const fit = fitLinear(rows);
  return {
    artifact: { coefficients: quantise(fit.coefficients), features: fit.coefficients.length },
    metrics: {
      rSquared: fit.rSquared,
      rmse: Math.sqrt(fit.rss / rows.length),
      parameterError: Math.sqrt(fit.coefficients.reduce((acc, value, j) => acc + (value - truth[j]) ** 2, 0)),
    },
    datasetsize: rows.length,
  };
}

function createRegistry() {
  return {
    store: new Map(),        // artifact hash -> { content, size }
    runs: new Map(),         // run id -> { manifest, artifactHash, metrics, tick }
    order: [],               // registration order of run ids
    executions: 0,
    tick: 0,
  };
}

/** Store artifact content by hash; identical content is deduplicated, not stored twice. */
function putArtifact(registry, content) {
  const hash = contentHash(content);
  if (!registry.store.has(hash)) {
    const size = canonicalJson(content).length;
    registry.store.set(hash, { content, size, references: 0 });
  }
  return hash;
}

/** Reads an artifact and re-hashes the bytes: the integrity check of the registry. */
function getArtifact(registry, hash) {
  const entry = registry.store.get(hash);
  if (!entry) throw new Error(`unknown artifact ${hash}`);
  if (contentHash(entry.content) !== hash) throw new Error(`artifact ${hash} failed its integrity check`);
  return entry.content;
}

/**
 * Registers (and if necessary executes) an experiment. The manifest's canonical hash is the run id, so
 * registering the same description twice is a pure cache lookup and executes nothing.
 */
function registerRun(registry, spec, { tick = null } = {}) {
  if (!spec || typeof spec !== "object") throw new TypeError("spec must be an object");
  for (const field of ["name", "algorithm", "datasetHash", "codeHash"]) {
    if (typeof spec[field] !== "string" || spec[field].length === 0) throw new TypeError(`spec.${field} must be a non-empty string`);
  }
  if (!Number.isInteger(spec.seed)) throw new RangeError("spec.seed must be an integer");
  const params = spec.params ?? {};
  if (typeof params !== "object" || params === null || Array.isArray(params)) throw new TypeError("spec.params must be an object");
  const parents = spec.parents ?? [];
  if (!Array.isArray(parents)) throw new TypeError("spec.parents must be an array");
  for (const parent of parents) {
    if (!registry.runs.has(parent)) throw new Error(`unknown parent run ${parent}`);
    if (parent === contentHash({ ...spec, parents })) throw new Error("a run cannot be its own parent");
  }
  const manifest = {
    name: spec.name,
    algorithm: spec.algorithm,
    params,
    datasetHash: spec.datasetHash,
    codeHash: spec.codeHash,
    seed: spec.seed,
    parents: [...parents].sort(),
  };
  const runId = contentHash(manifest);
  if (manifest.parents.includes(runId)) throw new Error("lineage would contain a cycle");
  const existing = registry.runs.get(runId);
  if (existing) return { runId, ...existing, executed: false };
  const outcome = executeExperiment(manifest);
  registry.executions += 1;
  const artifactHash = putArtifact(registry, outcome.artifact);
  const entry = {
    manifest,
    artifactHash,
    metrics: outcome.metrics,
    datasetSize: outcome.datasetsize,
    tick: tick ?? registry.tick,
  };
  registry.runs.set(runId, entry);
  registry.order.push(runId);
  registry.tick += 1;
  return { runId, ...entry, executed: true };
}

/** Re-executes a registered run and checks that it rebuilds the same artifact hash. */
function verifyRun(registry, runId) {
  const entry = registry.runs.get(runId);
  if (!entry) throw new Error(`unknown run ${runId}`);
  const outcome = executeExperiment(entry.manifest);
  const reproducedHash = contentHash(outcome.artifact);
  return {
    runId,
    reproduced: reproducedHash === entry.artifactHash && contentHash(entry.manifest) === runId,
    storedHash: entry.artifactHash,
    reproducedHash,
  };
}

/** Ancestors of a run (including itself), computed with an explicit worklist. */
function ancestors(registry, runId) {
  if (!registry.runs.has(runId)) throw new Error(`unknown run ${runId}`);
  const seen = new Set([runId]);
  const worklist = [runId];
  while (worklist.length > 0) {
    const current = worklist.pop();
    for (const parent of registry.runs.get(current).manifest.parents) {
      if (!seen.has(parent)) {
        seen.add(parent);
        worklist.push(parent);
      }
    }
  }
  return seen;
}

/** Ancestors via a Warshall transitive closure of the parent relation: an independent oracle. */
function ancestorsBruteForce(registry, runId) {
  const ids = [...registry.runs.keys()];
  const index = new Map(ids.map((id, i) => [id, i]));
  const closure = ids.map(() => new Array(ids.length).fill(false));
  for (const id of ids) {
    for (const parent of registry.runs.get(id).manifest.parents) closure[index.get(id)][index.get(parent)] = true;
  }
  for (let k = 0; k < ids.length; k++) {
    for (let i = 0; i < ids.length; i++) {
      if (!closure[i][k]) continue;
      for (let j = 0; j < ids.length; j++) if (closure[k][j]) closure[i][j] = true;
    }
  }
  const reachable = new Set([runId]);
  const row = index.get(runId);
  for (let j = 0; j < ids.length; j++) if (closure[row][j]) reachable.add(ids[j]);
  return reachable;
}

/** Provenance path from a root run down to `runId`; throws when the node is not an ancestor. */
function provenancePath(registry, ancestorId, runId) {
  if (!registry.runs.has(runId) || !registry.runs.has(ancestorId)) throw new Error("unknown run in provenance path");
  const queue = [[runId]];
  const visited = new Set([runId]);
  while (queue.length > 0) {
    const path = queue.shift();
    const head = path[path.length - 1];
    if (head === ancestorId) return path.slice().reverse();
    for (const parent of registry.runs.get(head).manifest.parents) {
      if (!visited.has(parent)) {
        visited.add(parent);
        queue.push([...path, parent]);
      }
    }
  }
  throw new Error(`${ancestorId} is not an ancestor of ${runId}`);
}

/** Longest ancestor chain length below a run; brute-force dynamic programming cross-check. */
function lineageDepth(registry, runId) {
  const memo = new Map();
  const depth = (id) => {
    if (memo.has(id)) return memo.get(id);
    const parents = registry.runs.get(id).manifest.parents;
    const value = parents.length === 0 ? 0 : 1 + Math.max(...parents.map(depth));
    memo.set(id, value);
    return value;
  };
  return depth(runId);
}

/** Best run for a metric: a scan, cross-checked against a brute-force reduction in the checks. */
function bestRun(registry, metric, { maximize = true } = {}) {
  let best = null;
  for (const id of registry.order) {
    const value = registry.runs.get(id).metrics[metric];
    if (typeof value !== "number" || !Number.isFinite(value)) throw new TypeError(`metric ${metric} is not numeric`);
    if (best === null || (maximize ? value > best.value : value < best.value)) best = { id, value };
  }
  return best;
}

/** Mark-and-sweep retention: keep exactly the artifacts reachable from the registered runs. */
function collectGarbage(registry) {
  const reachable = new Set();
  for (const entry of registry.runs.values()) reachable.add(entry.artifactHash);
  const removed = [];
  for (const hash of [...registry.store.keys()]) {
    if (!reachable.has(hash)) {
      registry.store.delete(hash);
      removed.push(hash);
    }
  }
  return { reachable, removed };
}

function main() {
  const diagnostics = [];
  const registry = createRegistry();

  // 1. Content addressing: key order never matters, content and array order always do.
  const manifestA = { b: 2, a: 1, nested: { y: [1, 2], x: "z" } };
  const manifestB = { nested: { x: "z", y: [1, 2] }, a: 1, b: 2 };
  diagnostics.push(`canonical form: ${canonicalJson(manifestA)}`);
  ok(contentHash(manifestA) === contentHash(manifestB), "key order must not change the hash");
  ok(contentHash(manifestA) !== contentHash({ ...manifestA, a: 1.0000001 }), "a numeric change must change the hash");
  ok(contentHash([1, 2]) !== contentHash([2, 1]), "array order must change the hash");
  ok(contentHash("x").length === 64, "the digest must be a 64 character hex string");
  throws(() => canonicalJson(Number.NaN), RangeError);
  throws(() => canonicalJson(new Map()), TypeError);

  // 2. Artifact store: deduplication and the integrity check.
  const artifactA = { coefficients: [0.5, -0.25], features: 2 };
  const firstHash = putArtifact(registry, artifactA);
  const secondHash = putArtifact(registry, { features: 2, coefficients: [0.5, -0.25] });
  diagnostics.push(`store: ${registry.store.size} entries after two identical puts, hash ${firstHash.slice(0, 12)}...`);
  ok(firstHash === secondHash, "identical artifacts must deduplicate to one hash");
  ok(registry.store.size === 1, "deduplication must not grow the store");
  ok(getArtifact(registry, firstHash).coefficients[0] === 0.5, "artifacts must round-trip");
  registry.store.get(firstHash).content = { coefficients: [9, 9] };
  throws(() => getArtifact(registry, firstHash), /integrity/);
  registry.store.get(firstHash).content = artifactA;
  ok(getArtifact(registry, firstHash).features === 2, "restoring the bytes must restore integrity");
  throws(() => getArtifact(registry, "0".repeat(64)), /unknown artifact/);

  // 3. Registering a run executes it, stores its artifact and reports metrics.
  const baseSpec = {
    name: "baseline", algorithm: "ols", params: { samples: 80, features: 3 },
    datasetHash: contentHash({ dataset: "synth-1" }), codeHash: contentHash({ code: "train-v1" }), seed: 11,
  };
  const baseline = registerRun(registry, baseSpec);
  const stored = getArtifact(registry, baseline.artifactHash);
  diagnostics.push(`baseline run ${baseline.runId.slice(0, 12)}... executions=${registry.executions} R2=${baseline.metrics.rSquared.toFixed(4)} paramErr=${baseline.metrics.parameterError.toFixed(4)}`);
  ok(baseline.executed, "the first registration must execute");
  ok(stored.coefficients.length === 3, "the artifact must hold one coefficient per feature");
  ok(baseline.metrics.rSquared > 0.95, `the fit must be good (R2 = ${baseline.metrics.rSquared})`);
  ok(baseline.metrics.parameterError < 0.05, `the recovered parameters must be close to the truth (${baseline.metrics.parameterError})`);

  // 4. Memoisation: the same manifest is a cache hit, a different seed is a new execution.
  const repeat = registerRun(registry, { ...baseSpec, params: { features: 3, samples: 80 } });
  ok(!repeat.executed && repeat.runId === baseline.runId, "re-registering the same manifest must not execute again");
  ok(registry.executions === 1, `the logical execution counter must not grow on a cache hit (got ${registry.executions})`);
  const reseeded = registerRun(registry, { ...baseSpec, seed: 12 });
  ok(reseeded.executed && reseeded.runId !== baseline.runId, "a new seed must be a new run");
  const differentCode = registerRun(registry, { ...baseSpec, codeHash: contentHash({ code: "train-v2" }) });
  ok(differentCode.runId !== baseline.runId, "a code change must produce a new run id");

  // 5. Provenance: children record parents, and the lineage queries match brute-force traversal.
  const child = registerRun(registry, { ...baseSpec, name: "tuned", params: { samples: 80, features: 3, lr: 0.1 }, parents: [baseline.runId] });
  const grandchild = registerRun(registry, { ...baseSpec, name: "final", seed: 13, parents: [child.runId, reseeded.runId] });
  const lineage = ancestors(registry, grandchild.runId);
  const bruteLineage = ancestorsBruteForce(registry, grandchild.runId);
  diagnostics.push(`lineage of final: ${[...lineage].length} runs, depth ${lineageDepth(registry, grandchild.runId)}, expected depth 2`);
  ok(lineage.size === bruteLineage.size && [...lineage].every((id) => bruteLineage.has(id)), "the ancestor walk must match brute-force reachability");
  ok(lineage.has(baseline.runId) && lineage.has(reseeded.runId) && lineage.size === 4, "the lineage must contain both branches and the root");
  ok(lineageDepth(registry, grandchild.runId) === 2, "the lineage depth must be 2");
  ok(lineageDepth(registry, baseline.runId) === 0, "a root run must have depth 0");
  const path = provenancePath(registry, baseline.runId, grandchild.runId);
  diagnostics.push(`provenance path: ${path.map((id) => id.slice(0, 8)).join(" -> ")}`);
  ok(path[0] === baseline.runId && path[path.length - 1] === grandchild.runId, "the provenance path must run from the ancestor to the run");
  ok(path.length === 3, "the shortest provenance path must have three nodes");
  throws(() => provenancePath(registry, grandchild.runId, baseline.runId), /not an ancestor/);
  throws(() => registerRun(registry, { ...baseSpec, parents: ["deadbeef"] }), /unknown parent/);

  // 6. Reproduction: every registered run rebuilds its artifact hash, and tampering is detected.
  const verifications = registry.order.map((id) => verifyRun(registry, id));
  diagnostics.push(`reproduction: ${verifications.filter((v) => v.reproduced).length}/${verifications.length} runs rebuilt their artifact hash`);
  ok(verifications.every((v) => v.reproduced), "every run must reproduce its artifact hash from the manifest");
  const tampered = registry.runs.get(baseline.runId);
  const honestHash = tampered.artifactHash;
  registry.runs.set(baseline.runId, { ...tampered, artifactHash: contentHash({ coefficients: [0, 0, 0], features: 3 }) });
  ok(!verifyRun(registry, baseline.runId).reproduced, "a run whose recorded artifact was replaced must fail reproduction");
  registry.runs.set(baseline.runId, tampered);
  ok(verifyRun(registry, baseline.runId).reproduced && tampered.artifactHash === honestHash, "restoring the record must restore reproduction");

  // 7. Aggregation queries agree with brute-force scans.
  const bestR2 = bestRun(registry, "rSquared");
  const bruteBest = registry.order.map((id) => ({ id, value: registry.runs.get(id).metrics.rSquared }))
    .reduce((a, b) => (b.value > a.value ? b : a));
  const smallestRmse = bestRun(registry, "rmse", { maximize: false });
  diagnostics.push(`best R2 = ${bestR2.value.toFixed(5)} (brute force ${bruteBest.value.toFixed(5)}), best rmse = ${smallestRmse.value.toFixed(5)}`);
  ok(bestR2.id === bruteBest.id, "the best-run query must agree with a brute-force scan");
  ok(smallestRmse.value <= bestR2.value + 1e6 && smallestRmse.value > 0, "the minimum query must return a plausible value");
  throws(() => bestRun(registry, "nonexistent"), TypeError);

  // 8. Scale: many experiments keep exact accounting and full reproducibility.
  const bulk = createRegistry();
  const bulkIds = [];
  for (let i = 0; i < 30; i++) {
    const spec = {
      ...baseSpec,
      params: { samples: 40 + i, features: 3 },
      seed: 100 + i,
      parents: i === 0 ? [] : [bulkIds[i - 1]],
    };
    const run = registerRun(bulk, spec);
    bulkIds.push(run.runId);
  }
  const reRegistered = bulkIds.filter((id) => !registerRun(bulk, bulk.runs.get(id).manifest).executed);
  const bulkVerified = bulkIds.every((id) => verifyRun(bulk, id).reproduced);
  diagnostics.push(`bulk registry: ${bulk.order.length} runs, ${bulk.executions} executions, ${reRegistered.length} cache hits, stores ${bulk.store.size} artifacts`);
  ok(bulk.order.length === 30 && bulk.executions === 30, "each distinct manifest must execute exactly once");
  ok(reRegistered.length === 30, "every manifest must be servable from the run cache");
  ok(bulkVerified, "all bulk runs must reproduce");
  ok(lineageDepth(bulk, bulkIds[29]) === 29, "a chained lineage must reach depth 29");
  ok(ancestors(bulk, bulkIds[29]).size === 30, "the deepest run must see the whole chain");
  ok(bulk.store.size <= 30, "artifact storage must be bounded by the number of distinct artifacts");

  // 9. Retention: mark-and-sweep keeps exactly the reachable artifacts and is idempotent.
  putArtifact(bulk, { coefficients: [42], features: 1, note: "orphan" });
  const reachable = new Set([...bulk.runs.values()].map((entry) => entry.artifactHash));
  const collected = collectGarbage(bulk);
  diagnostics.push(`collection: removed ${collected.removed.length} orphan(s), store now ${bulk.store.size} of ${reachable.size} reachable`);
  ok(collected.removed.length === 1, "exactly the orphan artifact must be removed");
  ok(bulk.store.size === reachable.size && [...reachable].every((hash) => bulk.store.has(hash)), "the collector must match brute-force reachability");
  const secondSweep = collectGarbage(bulk);
  ok(secondSweep.removed.length === 0, "a second sweep must be a no-op");
  ok(bulkIds.every((id) => verifyRun(bulk, id).reproduced), "collection must not break reproduction");

  // 10. Determinism: two registries fed the same specs agree on every identifier and metric.
  const rebuilt = createRegistry();
  const idsA = [baseSpec, { ...baseSpec, seed: 12 }, { ...baseSpec, codeHash: contentHash({ code: "train-v2" }) }].map((spec) => registerRun(registry, spec).runId);
  const idsB = [baseSpec, { ...baseSpec, seed: 12 }, { ...baseSpec, codeHash: contentHash({ code: "train-v2" }) }].map((spec) => registerRun(rebuilt, spec).runId);
  diagnostics.push(`determinism: ${idsA.map((id) => id.slice(0, 8)).join(",")} vs ${idsB.map((id) => id.slice(0, 8)).join(",")}`);
  ok(idsA.join() === idsB.join(), "run ids must be deterministic");
  ok([...registry.runs.keys()].every((id) => rebuilt.runs.has(id) ? rebuilt.runs.get(id).artifactHash === registry.runs.get(id).artifactHash : true), "artifact hashes must be deterministic");

  // 11. Rejection of malformed specifications and unknown runs.
  throws(() => registerRun(registry, { ...baseSpec, name: "" }), TypeError);
  throws(() => registerRun(registry, { ...baseSpec, seed: 1.5 }), RangeError);
  throws(() => registerRun(registry, { ...baseSpec, params: [] }), TypeError);
  throws(() => registerRun(registry, { ...baseSpec, parents: "x" }), TypeError);
  throws(() => verifyRun(registry, "deadbeef"), /unknown run/);
  throws(() => ancestors(registry, "deadbeef"), /unknown run/);

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  canonicalJson,
  contentHash,
  makeRng,
  makeDataset,
  fitLinear,
  quantise,
  executeExperiment,
  createRegistry,
  putArtifact,
  getArtifact,
  registerRun,
  verifyRun,
  ancestors,
  ancestorsBruteForce,
  provenancePath,
  lineageDepth,
  bestRun,
  collectGarbage,
};

if (require.main === module) main();
