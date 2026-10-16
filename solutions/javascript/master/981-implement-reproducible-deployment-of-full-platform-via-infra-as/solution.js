/** Task 981 - Implement reproducible deployment of full platform via infra-as-code.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * The desired state is a declarative resource document (type, name, spec, explicit `dependsOn`) and
 * the current state is what the last apply left behind. The planner diffs the two: resources only in
 * the document become `create`, changed specs become `update` with an attribute-level diff, equal
 * specs become `no-op`, and resources that vanished become `delete`. Creates and updates are ordered
 * so that every dependency is applied first; deletes run in reverse dependency order. Applying a
 * plan produces exactly the desired state, so a second plan is empty - that is the idempotency and
 * reproducibility claim, underscored by a stable plan hash.
 * Simplified: an in-memory provider (no cloud API, no state file encryption) with string specs.
 * Verified: plan shape against an independent per-attribute diff, apply order, the empty second
 * plan, drift detection, delete ordering, spec-key-order insensitivity, purity and determinism.
 */
"use strict";
const assert = require("node:assert");

const DESIRED = { resources: [
  { type: "network", name: "vpc", spec: { cidr: "10.0.0.0/16" } },
  { type: "database", name: "primary", dependsOn: ["network.vpc"], spec: { engine: "postgres", replicas: 1 } },
  { type: "service", name: "api", dependsOn: ["network.vpc", "database.primary"],
    spec: { image: "api:1.2.0", replicas: 3, cpu: "500m", env: "prod" } },
  { type: "service", name: "worker", dependsOn: ["database.primary"], spec: { image: "worker:0.9.1", replicas: 2 } },
] };
const CURRENT = [
  { type: "network", name: "vpc", spec: { cidr: "10.0.0.0/16" } },
  { type: "database", name: "primary", dependsOn: ["network.vpc"], spec: { engine: "postgres", replicas: 1 } },
  { type: "service", name: "api", dependsOn: ["network.vpc", "database.primary"],
    spec: { image: "api:1.1.0", replicas: 3, cpu: "500m" } },
  { type: "service", name: "legacy", spec: { image: "legacy:4.0" } },
];

const keyOf = (resource) => `${resource.type}.${resource.name}`;

/** Attribute map in a canonical order, so specs compare independently of key insertion order. */
const canonical = (spec) => Object.fromEntries(Object.entries(spec).sort(([a], [b]) => (a < b ? -1 : 1)));

/** Attribute-level diff between two specs; missing and added keys are both reported. */
function diffSpec(before = {}, after = {}) {
  const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort();
  return keys.filter((key) => before[key] !== after[key])
    .map((key) => ({ attribute: key, from: before[key] ?? null, to: after[key] ?? null }));
}

/** Kahn topological order (dependencies first, alphabetical among ready resources). */
function topoOrder(resources) {
  const byKey = new Map(resources.map((resource) => [keyOf(resource), resource]));
  const indegree = new Map([...byKey.keys()].map((key) => [key, 0]));
  const dependents = new Map([...byKey.keys()].map((key) => [key, []]));
  for (const [key, resource] of byKey) {
    for (const dependency of resource.dependsOn || []) {
      if (!byKey.has(dependency)) throw new Error(`${key} depends on unknown resource ${dependency}`);
      indegree.set(key, indegree.get(key) + 1);
      dependents.get(dependency).push(key);
    }
  }
  const ready = [...indegree].filter(([, degree]) => degree === 0).map(([key]) => key).sort();
  const order = [];
  while (ready.length) {
    const key = ready.shift();
    order.push(key);
    for (const next of dependents.get(key)) {
      indegree.set(next, indegree.get(next) - 1);
      if (indegree.get(next) === 0) { ready.push(next); ready.sort(); }
    }
  }
  if (order.length !== byKey.size) throw new Error("the resource graph has a cycle");
  return order;
}

/** FNV-1a hash of the canonical plan, so two identical plans share one identity. */
function planHash(steps) {
  let hash = 0x811c9dc5;
  for (const character of JSON.stringify(steps)) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

/** Plan the transition from `current` to `desired`. */
function plan({ desired, current }) {
  const currentByKey = new Map(current.map((resource) => [keyOf(resource), resource]));
  const desiredByKey = new Map(desired.resources.map((resource) => [keyOf(resource), resource]));
  const steps = [];
  for (const key of topoOrder(desired.resources)) {
    const wanted = desiredByKey.get(key);
    const existing = currentByKey.get(key);
    if (!existing) steps.push({ action: "create", id: key, changes: diffSpec({}, wanted.spec) });
    else {
      const changes = diffSpec(existing.spec, wanted.spec);
      steps.push({ action: changes.length ? "update" : "no-op", id: key, changes });
    }
  }
  const removals = current.filter((resource) => !desiredByKey.has(keyOf(resource)));
  const removalOrder = topoOrder(current.filter((resource) => currentByKey.has(keyOf(resource))))
    .filter((key) => !desiredByKey.has(key)).reverse();
  for (const key of removalOrder.length ? removalOrder : removals.map(keyOf).reverse()) {
    steps.push({ action: "delete", id: key, changes: diffSpec(currentByKey.get(key).spec, {}) });
  }
  const summary = steps.reduce((counts, step) => ({ ...counts, [step.action]: counts[step.action] + 1 }),
    { create: 0, update: 0, "no-op": 0, delete: 0 });
  return { steps, summary, hash: planHash(steps) };
}

/** Apply a plan; the provider is pure, so the inputs are never mutated. */
function apply({ desired, current }) {
  const steps = plan({ desired, current });
  const next = new Map(current.map((resource) => [keyOf(resource), resource]));
  for (const step of steps.steps) {
    if (step.action === "delete") next.delete(step.id);
    else if (step.action !== "no-op") {
      const wanted = desired.resources.find((resource) => keyOf(resource) === step.id);
      next.set(step.id, { ...wanted, spec: canonical(wanted.spec) });
    }
  }
  return { resources: [...next.values()].sort((a, b) => (keyOf(a) < keyOf(b) ? -1 : 1)), plan: steps };
}

const drift = (resources, id, patch) => resources.map((resource) => (keyOf(resource) === id
  ? { ...resource, spec: { ...resource.spec, ...patch } } : resource));

function main() {
  const empty = { resources: [] };

  // 1. Happy path: a first apply creates everything in dependency order.
  const bootstrap = plan({ desired: DESIRED, current: [] });
  assert.deepStrictEqual(bootstrap.steps.map((step) => step.action), ["create", "create", "create", "create"],
    "an empty account creates every resource");
  assert.deepStrictEqual(bootstrap.summary, { create: 4, update: 0, "no-op": 0, delete: 0 }, "summary counts");
  assert.deepStrictEqual(bootstrap.steps.map((step) => step.id),
    ["network.vpc", "database.primary", "service.api", "service.worker"], "dependencies come first");

  // 2. Idempotency: applying the plan leaves a state whose next plan is empty.
  const applied = apply({ desired: DESIRED, current: [] });
  assert.strictEqual(applied.resources.length, 4, "the platform now exists");
  const second = plan({ desired: DESIRED, current: applied.resources });
  assert(second.steps.every((step) => step.action === "no-op"), "a second plan changes nothing");
  assert.deepStrictEqual(second.summary, { create: 0, update: 0, "no-op": 4, delete: 0 }, "all no-ops");

  // 3. Drift detection with an independent attribute diff.
  const drifted = drift(applied.resources, "service.api", { image: "api:1.1.9", replicas: 5 });
  const repair = plan({ desired: DESIRED, current: drifted });
  const updateStep = repair.steps.find((step) => step.id === "service.api");
  assert.strictEqual(updateStep.action, "update", "drift becomes an update");
  const driftedApi = drifted.find((resource) => keyOf(resource) === "service.api").spec;
  const wantedApi = DESIRED.resources.find((resource) => keyOf(resource) === "service.api").spec;
  const manual = [...new Set([...Object.keys(driftedApi), ...Object.keys(wantedApi)])]
    .filter((attribute) => driftedApi[attribute] !== wantedApi[attribute]).sort();
  assert.deepStrictEqual(updateStep.changes.map((change) => change.attribute), manual, "the diff lists exactly the drift");
  assert.deepStrictEqual(manual, ["image", "replicas"], "two attributes drifted");
  assert.deepStrictEqual(repair.steps.map((step) => step.action), ["no-op", "no-op", "update", "no-op"], "only one update");

  // 4. Deletes run in reverse dependency order and are planned before nothing else changes.
  const shrink = { resources: DESIRED.resources.filter((resource) => keyOf(resource) === "network.vpc") };
  const teardown = plan({ desired: shrink, current: applied.resources });
  assert.deepStrictEqual(teardown.steps.filter((step) => step.action === "delete").map((step) => step.id),
    ["service.worker", "service.api", "database.primary"], "dependents are deleted before their dependency");
  assert.deepStrictEqual(teardown.summary, { create: 0, update: 0, "no-op": 1, delete: 3 }, "teardown counts");
  const gone = apply({ desired: { resources: [] }, current: applied.resources });
  assert.deepStrictEqual(gone.resources, [], "an empty document tears everything down");
  assert.strictEqual(plan({ desired: empty, current: [] }).steps.length, 0, "nothing to do on an empty platform");

  // 5. The reference transition: no-op vpc/primary, update api, delete legacy.
  const reference = plan({ desired: DESIRED, current: CURRENT });
  assert.deepStrictEqual(reference.steps.map((step) => `${step.action}:${step.id}`),
    ["no-op:network.vpc", "no-op:database.primary", "update:service.api", "create:service.worker",
      "delete:service.legacy"], "the migration plan");
  assert.deepStrictEqual(reference.summary, { create: 1, update: 1, "no-op": 2, delete: 1 }, "migration summary");
  assert.deepStrictEqual(reference.steps[2].changes.map((change) => change.attribute), ["env", "image"],
    "the api diff names the changed attributes");
  assert.deepStrictEqual(reference.steps[2].changes[1], { attribute: "image", from: "api:1.1.0", to: "api:1.2.0" },
    "a change carries both sides");

  // 6. Boundary: spec key order does not create drift, and extra keys are reported as removals.
  const reordered = applied.resources.map((resource) => (keyOf(resource) === "network.vpc"
    ? { ...resource, spec: { cidr: "10.0.0.0/16" } } : resource));
  assert.strictEqual(plan({ desired: DESIRED, current: reordered }).summary.update, 0, "key order is irrelevant");
  const extra = drift(applied.resources, "service.worker", { spot: "true" });
  assert.deepStrictEqual(plan({ desired: DESIRED, current: extra }).steps[3].changes,
    [{ attribute: "spot", from: "true", to: null }], "an extra attribute is removed");

  // 7. Reproducibility: plan identity is stable, and it changes when the plan changes.
  assert.strictEqual(plan({ desired: DESIRED, current: CURRENT }).hash, reference.hash, "same input, same hash");
  assert.notStrictEqual(plan({ desired: DESIRED, current: applied.resources }).hash, reference.hash, "different plan");
  assert.strictEqual(apply({ desired: DESIRED, current: CURRENT }).plan.hash, reference.hash, "apply reuses the plan");
  assert.deepStrictEqual(apply({ desired: DESIRED, current: CURRENT }).resources.map(keyOf),
    ["database.primary", "network.vpc", "service.api", "service.worker"], "the migration result");

  // 8. Purity and graph errors: inputs are untouched and impossible graphs are rejected.
  const before = JSON.stringify(CURRENT);
  apply({ desired: DESIRED, current: CURRENT });
  assert.strictEqual(JSON.stringify(CURRENT), before, "planning never mutates the current state");
  assert.throws(() => topoOrder([{ type: "a", name: "1", dependsOn: ["a.2"] }, { type: "a", name: "2", dependsOn: ["a.1"] }]),
    /cycle/, "cyclic dependencies are rejected");
  assert.throws(() => topoOrder([{ type: "a", name: "1", dependsOn: ["b.9"] }]), /unknown resource/,
    "dangling dependencies are rejected");
  assert.deepStrictEqual(topoOrder(DESIRED.resources), ["network.vpc", "database.primary", "service.api", "service.worker"],
    "the order is deterministic");
  assert.strictEqual(DESIRED.resources.length, 4, "the desired document is never mutated either");

  console.log(`infra-as-code: migration ${reference.summary.create}c/${reference.summary.update}u/` +
    `${reference.summary.delete}d hash ${reference.hash}, second plan all no-op, ` +
    `delete order ${teardown.steps.filter((s) => s.action === "delete").map((s) => s.id).join(" > ")}`);
  console.log("PASS");
}
module.exports = { DESIRED, CURRENT, keyOf, canonical, diffSpec, topoOrder, planHash, plan, apply, drift };
if (require.main === module) main();
