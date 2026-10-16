/** Task 900 - Implement a full-featured reproducible research platform with hardware config capture.
 *  (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
 *
 * A pipeline runner whose output is a *manifest*, not just a result. Everything that can influence a
 * run is content-addressed: the step implementation and its parameters, the recursive hashes of its
 * inputs, the code revision, the numerics policy, and the captured hardware description. A step's key
 * is the hash of all of that, so the cache is exact and the manifest is a proof of what was run.
 *
 * Hardware capture matters because it is what usually breaks reproducibility: the same code on 4 and on
 * 64 cores can disagree in the last bits if the reduction order depends on the machine. The platform
 * therefore carries an explicit determinism policy — `fixed-order` (the platform's default, which sums
 * in canonical index order and is bit-identical across core counts) versus `hardware-order` (which sums
 * per core and is not). A verifier compares two manifests and reports which steps diverged, so a
 * non-reproducible run is *detected* rather than silently accepted.
 *
 * Invariants verified below: (1) content hashing is stable under key reordering and sensitive to every
 * value, including the code revision and the hardware descriptor; (2) a second identical run executes
 * nothing (pure cache hits) and produces the same root hash; (3) a parameter change re-executes exactly
 * the dirty set predicted by an independent reachability computation over the dependency graph;
 * (4) a code-revision change invalidates the whole pipeline while a hardware change under `fixed-order`
 * does *not* change any artifact hash — the numeric results are bit-identical across core counts, which
 * is asserted on the actual sums; (5) under `hardware-order` the same change *is* detected by the
 * verifier and reported as non-reproducible; (6) tampering with a cached artifact is caught by its
 * hash; (7) the manifest root hash changes whenever any step's inputs change; (8) degenerate pipelines
 * (empty, single step, diamond dependencies, cycles, unknown names) behave; (9) execution counts and
 * runs are deterministic.
 *
 * Simplifications: no filesystem or processes — artifacts live in an in-process content-addressed store,
 * the "code revision" is a string, and the hardware description is a structured record rather than a
 * probe (only the fields a numerics policy can depend on are captured).
 */
"use strict";

const crypto = require("node:crypto");

/* ------------------------------------------------------------------ content addressing */

/** Canonical serialisation: object keys sorted, so key order never changes a hash. */
function canonical(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
}

const contentHash = (value) => crypto.createHash("sha256").update(canonical(value)).digest("hex");
const shortHash = (value) => contentHash(value).slice(0, 12);

class ArtifactStore {
  constructor() { this.items = new Map(); this.reads = 0; this.writes = 0; }
  put(value) {
    const hash = contentHash(value);
    this.items.set(hash, canonical(value));
    this.writes++;
    return hash;
  }
  /** A read re-verifies the content, so a tampered store cannot return a clean answer. */
  get(hash) {
    this.reads++;
    if (!this.items.has(hash)) throw new Error(`artifact ${hash} is missing`);
    const stored = this.items.get(hash);
    if (contentHash(JSON.parse(stored)) !== hash) throw new Error(`artifact ${hash} failed its hash check`);
    return JSON.parse(stored);
  }
  tamper(hash, value) {
    if (!this.items.has(hash)) throw new Error("nothing to tamper with");
    this.items.set(hash, canonical(value));
  }
}

/* ------------------------------------------------------------------ hardware and numerics */

const REQUIRED_HARDWARE_FIELDS = ["cpu", "cores", "simdWidth", "memoryGb", "os"];

function captureHardware(description) {
  for (const field of REQUIRED_HARDWARE_FIELDS) {
    if (description[field] === undefined) throw new Error(`hardware description is missing ${field}`);
  }
  if (!Number.isInteger(description.cores) || description.cores < 1) throw new Error("cores must be a positive integer");
  if (!Number.isInteger(description.simdWidth) || description.simdWidth < 1) throw new Error("simdWidth must be a positive integer");
  return { description: { ...description }, hash: shortHash(description) };
}

const POLICIES = new Set(["fixed-order", "hardware-order"]);

function assertPolicy(policy) {
  if (!POLICIES.has(policy)) throw new Error(`unknown determinism policy ${policy}`);
  return policy;
}

/**
 * Floating-point reduction under an explicit policy. `fixed-order` walks the canonical index order and
 * is therefore bit-identical on any machine; `hardware-order` emulates a parallel reduction whose
 * partial sums depend on the core count, which is exactly the failure mode this platform must expose.
 */
function reduceValues(values, policy, hardware) {
  assertPolicy(policy);
  if (policy === "fixed-order") {
    let total = 0;
    for (const value of values) total += value;
    return total;
  }
  const cores = Math.max(1, Math.min(hardware.description.cores, values.length));
  const partials = new Array(cores).fill(0);
  for (let i = 0; i < values.length; i++) partials[i % cores] += values[i];
  let total = 0;
  for (const partial of partials) total += partial;
  return total;
}

/* ------------------------------------------------------------------ pipeline */

const IMPLEMENTATIONS = {
  /** Sums all of its inputs (or an explicit parameter list) under the configured policy. */
  sum: ({ params, inputs, policy, hardware }) => reduceValues(
    params.values !== undefined ? params.values : inputs.flat(), policy, hardware),
  /** Doubles every number of its input. */
  scale: ({ params, inputs }) => (inputs[0] || []).map((value) => value * (params.factor === undefined ? 1 : params.factor)),
  /** Repeats its input `times` times. */
  repeat: ({ params, inputs }) => {
    const out = [];
    for (let i = 0; i < (params.times || 1); i++) out.push(...(inputs[0] || []));
    return out;
  },
  /** Emits a literal, to have a source with no inputs at all. */
  literal: ({ params }) => params.value,
  /** Emits an irregular vector, so a hardware-dependent reduction order really does differ. */
  ramp: ({ params }) => Array.from({ length: params.length || 8 }, (_, i) =>
    (i + 1) * (params.step || 0.1) * (1 + (i % 7) * (params.jitter || 0))),
};

function validatePipeline(spec) {
  if (!spec || !Array.isArray(spec.steps)) throw new Error("a pipeline needs a steps array");
  const names = new Set(spec.steps.map((step) => step.name));
  const seen = new Set();
  for (const step of spec.steps) {
    if (typeof step.name !== "string" || step.name === "") throw new Error("every step needs a name");
    if (seen.has(step.name)) throw new Error(`duplicate step ${step.name}`);
    seen.add(step.name);
    if (!Object.prototype.hasOwnProperty.call(IMPLEMENTATIONS, step.impl)) {
      throw new Error(`step ${step.name} uses unknown implementation ${step.impl}`);
    }
    for (const input of step.inputs || []) {
      if (!names.has(input)) throw new Error(`step ${step.name} depends on unknown step ${input}`);
    }
  }
  // Cycle detection by traversal: a pipeline that never finishes is rejected up front.
  const order = topologicalOrder(spec);
  if (order.length !== spec.steps.length) throw new Error("the pipeline has a cycle");
  return spec;
}

function topologicalOrder(spec) {
  const byName = new Map(spec.steps.map((step) => [step.name, step]));
  const state = new Map();
  const order = [];
  const visit = (name) => {
    const status = state.get(name);
    if (status === "done") return true;
    if (status === "open") return false;                 // a back edge: cycle
    const step = byName.get(name);
    if (!step) return false;
    state.set(name, "open");
    for (const input of step.inputs || []) if (!visit(input)) return false;
    state.set(name, "done");
    order.push(step);
    return true;
  };
  for (const step of spec.steps) {
    if (!visit(step.name)) return order;
  }
  return order;
}

/** The dirty set after changing steps in `changed`, by reachability (an independent oracle). */
function expectedDirtySteps(spec, changed) {
  const changedSet = new Set(changed);
  const reverse = new Map(spec.steps.map((step) => [step.name, []]));
  for (const step of spec.steps) {
    for (const input of step.inputs || []) reverse.get(input).push(step.name);
  }
  const dirty = new Set();
  const queue = [...changedSet];
  while (queue.length > 0) {
    const name = queue.pop();
    if (dirty.has(name)) continue;
    dirty.add(name);
    for (const downstream of reverse.get(name) || []) queue.push(downstream);
  }
  return dirty;
}

class ResearchPlatform {
  constructor(options = {}) {
    // The artifact store and the key cache are *shared* resources (a cache directory in reality): two
    // machines running the same experiment share them, which is what makes a warm run free.
    this.store = options.store || new ArtifactStore();
    this.cache = options.cache || new Map();
    this.executions = [];
    this.policy = assertPolicy(options.policy || "fixed-order");
    this.codeRevision = options.codeRevision || "rev-1";
    this.hardware = captureHardware(options.hardware || {
      cpu: "generic-x86", cores: 4, simdWidth: 8, memoryGb: 16, os: "linux",
    });
    this.runs = 0;
  }

  /**
   * The key covers everything that can change a step's output. Under `fixed-order` the hardware cannot
   * influence the numbers, so it is deliberately *not* part of the key; under `hardware-order` it is,
   * so results from different machines can never collide in the cache.
   */
  stepKey(step, inputHashes) {
    return shortHash({
      name: step.name, impl: step.impl, params: step.params || {},
      inputs: inputHashes, code: this.codeRevision, policy: this.policy,
      hardware: this.policy === "fixed-order" ? null : this.hardware.hash,
    });
  }

  run(spec) {
    validatePipeline(spec);
    this.runs++;
    const order = topologicalOrder(spec);
    const artifacts = new Map();
    const manifestSteps = [];
    let executed = 0;
    let cached = 0;
    for (const step of order) {
      const inputHashes = (step.inputs || []).map((input) => artifacts.get(input));
      const key = this.stepKey(step, inputHashes);
      let artifactHash = this.cache.get(key);
      if (artifactHash === undefined) {
        const value = IMPLEMENTATIONS[step.impl]({
          params: step.params || {},
          inputs: (step.inputs || []).map((input) => this.store.get(artifacts.get(input))),
          policy: this.policy,
          hardware: this.hardware,
        });
        artifactHash = this.store.put(value);
        this.cache.set(key, artifactHash);
        executed++;
        this.executions.push(step.name);
      } else {
        cached++;
      }
      artifacts.set(step.name, artifactHash);
      manifestSteps.push({ name: step.name, key, artifactHash });
    }
    const manifest = {
      revision: this.codeRevision,
      policy: this.policy,
      hardware: this.hardware,
      steps: manifestSteps,
      root: shortHash(manifestSteps.map((step) => `${step.name}:${step.key}:${step.artifactHash}`)),
    };
    return { manifest, artifacts, executed, cached, store: this.store };
  }

  value(result, stepName) {
    return this.store.get(result.artifacts.get(stepName));
  }
}

/** Compares two manifests and reports the steps whose inputs or outputs diverged. */
function verifyReproduction(before, after) {
  const diverged = [];
  const beforeByName = new Map(before.steps.map((step) => [step.name, step]));
  for (const step of after.steps) {
    const previous = beforeByName.get(step.name);
    if (!previous) { diverged.push({ name: step.name, reason: "new step" }); continue; }
    if (previous.key !== step.key) diverged.push({ name: step.name, reason: "different key" });
    else if (previous.artifactHash !== step.artifactHash) diverged.push({ name: step.name, reason: "different artifact" });
  }
  const afterNames = new Set(after.steps.map((step) => step.name));
  for (const step of before.steps) if (!afterNames.has(step.name)) diverged.push({ name: step.name, reason: "removed" });
  return {
    reproducible: diverged.length === 0 && before.root === after.root,
    rootMatches: before.root === after.root,
    diverged,
    hardwareChanged: before.hardware.hash !== after.hardware.hash,
    revisionChanged: before.revision !== after.revision,
    policyChanged: before.policy !== after.policy,
  };
}

/* ------------------------------------------------------------------ fixture */

function demoSpec(overrides = {}) {
  return {
    steps: [
      { name: "seed", impl: "ramp", params: { length: 64, step: 0.1, jitter: 0.37 }, inputs: [] },
      { name: "scaled", impl: "scale", params: { factor: 3 }, inputs: ["seed"] },
      { name: "doubled", impl: "repeat", params: { times: 2 }, inputs: ["scaled"] },
      { name: "seed2", impl: "ramp", params: { length: 64, step: 0.05, jitter: 0.41 }, inputs: [] },
      {
        name: "total",
        impl: "sum",
        params: {},
        inputs: ["doubled", "seed2"],
        ...(overrides.total || {}),
      },
    ],
  };
}

module.exports = {
  canonical, contentHash, shortHash, ArtifactStore, captureHardware, POLICIES, assertPolicy,
  reduceValues, IMPLEMENTATIONS, validatePipeline, topologicalOrder, expectedDirtySteps,
  ResearchPlatform, verifyReproduction, demoSpec, REQUIRED_HARDWARE_FIELDS,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- content addressing ---------------------------------------------------------------------
  {
    assert.strictEqual(contentHash({ a: 1, b: 2 }), contentHash({ b: 2, a: 1 }), "key order must not matter");
    assert.notStrictEqual(contentHash({ a: 1 }), contentHash({ a: 2 }));
    assert.notStrictEqual(contentHash([1, 2]), contentHash([2, 1]));
    assert.strictEqual(contentHash(null), contentHash(null));
    assert.strictEqual(shortHash("x").length, 12);
    const store = new ArtifactStore();
    const hash = store.put({ nested: [1, { two: 2 }] });
    assert.deepStrictEqual(store.get(hash), { nested: [1, { two: 2 }] });
    assert.strictEqual(store.writes, 1);
    store.tamper(hash, { nested: [9, 9] });
    assert.throws(() => store.get(hash), /failed its hash check/, "tampering must be caught");
    assert.throws(() => store.get("deadbeef"), /missing/);
  }

  // --- hardware capture ------------------------------------------------------------------------
  {
    const captured = captureHardware({ cpu: "zen4", cores: 8, simdWidth: 16, memoryGb: 32, os: "linux" });
    assert.strictEqual(captured.description.cores, 8);
    assert.strictEqual(captured.hash.length, 12);
    assert.strictEqual(captureHardware({ cpu: "zen4", cores: 8, simdWidth: 16, memoryGb: 32, os: "linux" }).hash,
      captured.hash, "capture is deterministic");
    assert.notStrictEqual(captureHardware({ cpu: "zen4", cores: 16, simdWidth: 16, memoryGb: 32, os: "linux" }).hash,
      captured.hash);
    assert.throws(() => captureHardware({ cpu: "zen4" }), /missing memoryGb|missing/);
    assert.throws(() => captureHardware({ cpu: "z", cores: 0, simdWidth: 8, memoryGb: 8, os: "linux" }), /cores/);
    assert.throws(() => assertPolicy("whatever"), /unknown determinism policy/);
  }

  // --- the numerics policy is what makes a run reproducible across hardware --------------------
  {
    const values = [0.1, 0.2, 0.3, 0.7, 0.11, 0.13, 0.17, 0.19, 0.23, 0.29, 0.31, 0.37];
    const small = captureHardware({ cpu: "generic", cores: 2, simdWidth: 4, memoryGb: 8, os: "linux" });
    const large = captureHardware({ cpu: "generic", cores: 16, simdWidth: 4, memoryGb: 8, os: "linux" });
    assert.strictEqual(reduceValues(values, "fixed-order", small), reduceValues(values, "fixed-order", large),
      "a fixed reduction order is bit-identical across core counts");
    assert.notStrictEqual(reduceValues(values, "hardware-order", small),
      reduceValues(values, "hardware-order", large),
      "a hardware-ordered reduction is not, which is the failure mode being detected");
    assert.strictEqual(reduceValues([1, 2, 3], "fixed-order", small), 6);
    assert.strictEqual(reduceValues([], "fixed-order", small), 0);
    assert.throws(() => reduceValues([1], "nope", small), /unknown determinism policy/);
  }

  // --- pipeline validation ---------------------------------------------------------------------
  {
    validatePipeline(demoSpec());
    assert.throws(() => validatePipeline({}), /steps array/);
    assert.throws(() => validatePipeline({ steps: [{ name: "a", impl: "sum", inputs: [] }, { name: "a", impl: "sum", inputs: [] }] }), /duplicate step/);
    assert.throws(() => validatePipeline({ steps: [{ name: "a", impl: "ghost", inputs: [] }] }), /unknown implementation/);
    assert.throws(() => validatePipeline({ steps: [{ name: "a", impl: "sum", inputs: ["missing"] }] }), /unknown step/);
    assert.throws(() => validatePipeline({
      steps: [{ name: "a", impl: "sum", inputs: ["b"] }, { name: "b", impl: "sum", inputs: ["a"] }],
    }), /cycle/);
    assert.throws(() => validatePipeline({ steps: [{ name: "a", impl: "sum", inputs: ["a"] }] }), /cycle/);
    const order = topologicalOrder(demoSpec()).map((step) => step.name);
    assert.deepStrictEqual(order.indexOf("seed") < order.indexOf("scaled"), true, "the order is topological");
    assert.deepStrictEqual(order.indexOf("scaled") < order.indexOf("doubled"), true);
    assert.ok(order.indexOf("doubled") < order.indexOf("total") && order.indexOf("seed2") < order.indexOf("total"));
    assert.strictEqual(order.length, 5);
  }

  // --- a cold run, then a fully cached rerun ---------------------------------------------------
  const hardwareSmall = { cpu: "generic", cores: 4, simdWidth: 8, memoryGb: 16, os: "linux" };
  const hardwareLarge = { cpu: "generic", cores: 64, simdWidth: 8, memoryGb: 256, os: "linux" };
  const hardwareshared = new ArtifactStore();
  const sharedCache = new Map();
  const platform = new ResearchPlatform({
    codeRevision: "rev-a", hardware: hardwareSmall, policy: "fixed-order",
    store: hardwareshared, cache: sharedCache,
  });
  const first = platform.run(demoSpec());
  {
    assert.strictEqual(first.executed, 5, "a cold run executes every step");
    assert.strictEqual(first.cached, 0);
    assert.deepStrictEqual(platform.executions, ["seed", "scaled", "doubled", "seed2", "total"]);
    const total = platform.value(first, "total");
    assert.ok(Number.isFinite(total));
    const second = platform.run(demoSpec());
    assert.strictEqual(second.executed, 0, "a rerun executes nothing");
    assert.strictEqual(second.cached, 5);
    assert.strictEqual(second.manifest.root, first.manifest.root, "the manifest is identical");
    const verdict = verifyReproduction(first.manifest, second.manifest);
    assert.ok(verdict.reproducible && verdict.rootMatches && verdict.diverged.length === 0);
    assert.strictEqual(verdict.hardwareChanged, false);
  }

  // --- hardware changes do not perturb a fixed-order run, and the manifest records them --------
  {
    const copy = new ResearchPlatform({
      codeRevision: "rev-a", hardware: hardwareLarge, policy: "fixed-order",
      store: hardwareshared, cache: sharedCache,
    });
    const result = copy.run(demoSpec());
    assert.strictEqual(result.executed, 0, "the cache is shared across hardware when the policy is fixed");
    const totalSmall = platform.value(first, "total");
    const totalLarge = copy.value(result, "total");
    assert.strictEqual(totalSmall, totalLarge, "the numbers must be bit-identical across core counts");
    const verdict = verifyReproduction(first.manifest, result.manifest);
    assert.ok(verdict.reproducible, "and the run is therefore reproducible");
    assert.ok(verdict.hardwareChanged, "even though the captured hardware differs");
    assert.notStrictEqual(first.manifest.hardware.hash, result.manifest.hardware.hash);
  }

  // --- the same experiment under a hardware-order policy is detected as non-reproducible ------
  {
    const honest = new ResearchPlatform({
      codeRevision: "rev-a", hardware: hardwareSmall, policy: "hardware-order",
      store: new ArtifactStore(), cache: new Map(),
    });
    const runSmall = honest.run(demoSpec());
    const other = new ResearchPlatform({
      codeRevision: "rev-a", hardware: hardwareLarge, policy: "hardware-order",
      store: hardwareshared, cache: sharedCache,
    });
    const runLarge = other.run(demoSpec());
    assert.ok(runLarge.executed > 0, "a hardware-ordered run cannot reuse another machine's cache");
    const verdict = verifyReproduction(runSmall.manifest, runLarge.manifest);
    assert.strictEqual(verdict.reproducible, false, "a hardware-ordered pipeline must not claim reproducibility");
    assert.strictEqual(verdict.policyChanged, false);
    assert.ok(verdict.hardwareChanged);
    assert.ok(verdict.diverged.some((entry) => entry.name === "total"),
      "the divergent step must be named in the report");
    assert.notStrictEqual(honest.value(runSmall, "total"), other.value(runLarge, "total"),
      "and the numbers really do differ");
  }

  // --- incremental invalidation matches the independent oracle ---------------------------------
  {
    const fresh = new ResearchPlatform({ codeRevision: "rev-a", hardware: hardwareSmall });
    const baseline = fresh.run(demoSpec());
    assert.strictEqual(baseline.executed, 5);
    const edits = [
      { label: "seed length", changed: ["seed"], patch: (spec) => { spec.steps[0].params.length = 32; } },
      { label: "scale factor", changed: ["scaled"], patch: (spec) => { spec.steps[1].params.factor = 4; } },
      { label: "repeat times", changed: ["doubled"], patch: (spec) => { spec.steps[2].params.times = 3; } },
    ];
    for (const edit of edits) {
      const spec = demoSpec();
      edit.patch(spec);
      const expected = expectedDirtySteps(demoSpec(), edit.changed);
      const result = fresh.run(spec);
      assert.strictEqual(result.executed, expected.size,
        `${edit.label}: executed ${result.executed}, expected ${expected.size}`);
      const executedNames = new Set(fresh.executions.slice(-result.executed));
      for (const name of expected) {
        assert.ok(executedNames.has(name), `${edit.label}: ${name} should have re-run`);
      }
      assert.notStrictEqual(result.manifest.root, baseline.manifest.root, `${edit.label}: the root must change`);
      assert.notStrictEqual(verifyReproduction(baseline.manifest, result.manifest).reproducible, true);
    }
    // A change to the untouched branch re-runs only that branch.
    const branchSpec = demoSpec();
    branchSpec.steps[3].params.step = 0.07;
    const branchResult = fresh.run(branchSpec);
    assert.strictEqual(branchResult.executed, 2, "seed2 and total");
    assert.deepStrictEqual(fresh.executions.slice(-2), ["seed2", "total"]);
  }

  // --- a code revision invalidates everything ---------------------------------------------------
  {
    const revised = new ResearchPlatform({ codeRevision: "rev-b", hardware: hardwareSmall });
    const result = revised.run(demoSpec());
    assert.strictEqual(result.executed, 5, "a new revision re-executes the whole pipeline");
    assert.strictEqual(result.cached, 0);
    const verdict = verifyReproduction(first.manifest, result.manifest);
    assert.strictEqual(verdict.reproducible, false);
    assert.ok(verdict.revisionChanged);
    assert.strictEqual(verdict.diverged.length, 5);
    assert.ok(verdict.diverged.every((entry) => entry.reason === "different key"));
  }

  // --- diamond dependencies and degenerate pipelines --------------------------------------------
  {
    const diamond = {
      steps: [
        { name: "source", impl: "ramp", params: { length: 4, step: 1 }, inputs: [] },
        { name: "left", impl: "scale", params: { factor: 2 }, inputs: ["source"] },
        { name: "right", impl: "scale", params: { factor: 3 }, inputs: ["source"] },
        { name: "join", impl: "sum", params: {}, inputs: ["left", "right"] },
      ],
    };
    const platform2 = new ResearchPlatform({ hardware: hardwareSmall });
    const diamondRun = platform2.run(diamond);
    assert.strictEqual(diamondRun.executed, 4, "a shared dependency runs once");
    assert.deepStrictEqual(platform2.value(diamondRun, "left"), [2, 4, 6, 8]);
    assert.deepStrictEqual(platform2.value(diamondRun, "right"), [3, 6, 9, 12]);
    assert.strictEqual(platform2.value(diamondRun, "join"), 50, "the join sums both branches");
    assert.strictEqual(platform2.run(diamond).executed, 0);
    const empty = new ResearchPlatform({ hardware: hardwareSmall });
    const emptyRun = empty.run({ steps: [] });
    assert.strictEqual(emptyRun.manifest.steps.length, 0);
    assert.strictEqual(emptyRun.manifest.root.length, 12, "an empty pipeline still has a manifest");
    assert.strictEqual(emptyRun.executed, 0);
    assert.ok(verifyReproduction(empty.run({ steps: [] }).manifest, empty.run({ steps: [] }).manifest).reproducible);
    const single = new ResearchPlatform({ hardware: hardwareSmall });
    const singleRun = single.run({ steps: [{ name: "only", impl: "literal", params: { value: 7 }, inputs: [] }] });
    assert.strictEqual(single.value(singleRun, "only"), 7);
    assert.strictEqual(singleRun.executed, 1);
    const noParams = new ResearchPlatform({ hardware: hardwareSmall });
    const noParamsRun = noParams.run({ steps: [{ name: "np", impl: "sum", params: {}, inputs: [] }] });
    assert.strictEqual(noParams.value(noParamsRun, "np"), 0);
  }

  // --- determinism ---------------------------------------------------------------------------------
  {
    const build = () => {
      const platform3 = new ResearchPlatform({ codeRevision: "rev-z", hardware: hardwareSmall, policy: "fixed-order" });
      const result = platform3.run(demoSpec());
      return `${result.manifest.root}|${result.executed}|${platform3.value(result, "total")}|${platform3.executions.join(",")}`;
    };
    assert.strictEqual(build(), build());
  }

  console.log(`reproducible: ${first.manifest.steps.length} steps, root ${first.manifest.root}, ` +
    `rerun executed ${platform.run(demoSpec()).executed}, ` +
    `fixed-order total ${platform.value(first, "total")} stable across 4 and 64 cores, ` +
    `hardware-order run flagged as non-reproducible`);
  console.log("PASS");
}
