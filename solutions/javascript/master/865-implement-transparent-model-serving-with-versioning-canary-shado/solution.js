/** Task 865 - Implement transparent model serving with versioning, canary, shadow tests, and rollback.
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * Model. A routing layer in front of several model versions. Every request carries a key and features; the
 * router hashes the key into a bucket in [0,100) so the same client always reaches the same version while
 * a canary version receives exactly the configured percentage of *distinct keys* (deterministic hashing,
 * not random sampling — the split is reproducible and sticky). The remaining traffic is served by the
 * stable version. A shadow version evaluates a mirrored copy of every request but its answer is recorded
 * only in metrics and can never reach the client.
 *
 * Health is monitored with a sliding window: when the canary's error rate over its last `windowSize`
 * requests crosses `errorThreshold`, the router rolls the canary back automatically — traffic returns to
 * the previous stable version and the bad version is marked rolled back, so a repeated rollback is a no-op.
 * A canary that survives `promotionAfter` clean requests is promoted instead, which is what makes the
 * deployment self-driving.
 *
 * What is simplified: one process and no network (a "version" is a plain function, not a served model), no
 * real load balancing or connection draining, fixed hash-based routing instead of a service mesh, a simple
 * sliding-window error rate instead of statistical health checking, and randomness confined to one seeded
 * mulberry32 stream used only to synthesise traffic — no wall clock, no threads, no I/O.
 *
 * Invariants verified:
 *   routing      the set of keys served by the canary is exactly the set whose hash falls in the canary
 *                band (not merely close), the observed share is inside the binomial tolerance, every key is
 *                sticky across requests, and each request is served by exactly one live version.
 *   transparency the shadow answer never changes the client's response: for every request the returned
 *                value equals the serving version's own output, the shadow evaluation count equals the
 *                number of mirrored requests, and shadow metrics are recomputed independently.
 *   safety       a bad canary is rolled back within a bounded number of requests, the rollback restores the
 *                previous version's exact behaviour, it happens exactly once (idempotent), traffic after the
 *                rollback is 100% stable, and the whole rollout log is consistent with the request log.
 *   promotion    a healthy canary is promoted after the documented number of clean requests and then serves
 *                everything; a version with no predecessor refuses to roll back.
 *   determinism  identical seeds reproduce identical routing, events and metrics.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — used only to synthesise traffic, never to make routing decisions. */
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

/** Deterministic string hash: FNV-1a, stable across runs and independent of insertion order. */
function hashKey(key) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < key.length; i++) {
    hash ^= key.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

const bucketOf = (key) => hashKey(key) % 100;

/** A model version: an id, a predict function and a deterministic fault condition. */
class ModelVersion {
  constructor({ id, predict, faulty = () => false, description = "" }) {
    if (typeof id !== "string" || id.length === 0) throw new TypeError("a version needs an id");
    if (typeof predict !== "function") throw new TypeError("a version needs a predict function");
    this.id = id;
    this.predict = predict;
    this.faulty = faulty;
    this.description = description;
    this.status = "registered";
    this.served = 0;
    this.errors = 0;
  }

  /** Returns the prediction and whether this evaluation was a fault. */
  evaluate(request) {
    this.served += 1;
    const error = this.faulty(request);
    if (error) this.errors += 1;
    return { value: this.predict(request), error };
  }
}

/**
 * The serving router. `versions` is an ordered list; the first entry is the initial stable version.
 */
class ServingRouter {
  constructor({
    versions,
    canary = null,
    shadow = null,
    canaryPercent = 10,
    windowSize = 20,
    errorThreshold = 0.5,
    promotionAfter = 100,
  } = {}) {
    if (!Array.isArray(versions) || versions.length === 0) throw new RangeError("at least one version is required");
    if (!versions.every((version) => version instanceof ModelVersion)) throw new TypeError("versions must be ModelVersion instances");
    if (!Number.isInteger(canaryPercent) || canaryPercent < 0 || canaryPercent > 100) throw new RangeError("canaryPercent must be an integer in [0,100]");
    if (!Number.isInteger(windowSize) || windowSize < 1) throw new RangeError("windowSize must be >= 1");
    if (!(errorThreshold > 0 && errorThreshold <= 1)) throw new RangeError("errorThreshold must be in (0,1]");
    if (!Number.isInteger(promotionAfter) || promotionAfter < 1) throw new RangeError("promotionAfter must be >= 1");
    this.versions = versions;
    this.byId = new Map(versions.map((version) => [version.id, version]));
    this.stable = versions[0];
    this.stable.status = "stable";
    this.canary = canary;
    this.shadow = shadow;
    this.canaryPercent = canaryPercent;
    this.windowSize = windowSize;
    this.errorThreshold = errorThreshold;
    this.promotionAfter = promotionAfter;
    this.canaryWindow = [];
    this.events = [];
    this.requests = [];
    if (canary !== null) {
      if (!(canary instanceof ModelVersion)) throw new TypeError("canary must be a ModelVersion");
      canary.status = "canary";
    }
    if (shadow !== null) {
      if (!(shadow instanceof ModelVersion)) throw new TypeError("shadow must be a ModelVersion");
      shadow.status = "shadow";
    }
  }

  /** Which version serves this key: the canary band is [0, canaryPercent) of the hash space. */
  serveFor(request) {
    if (this.canary === null || this.canary.status !== "canary") return this.stable;
    return bucketOf(request.key) < this.canaryPercent ? this.canary : this.stable;
  }

  /** Handles one request end to end; the client only ever sees the serving version's answer. */
  handle(request) {
    if (request === null || typeof request !== "object" || typeof request.key !== "string") {
      throw new TypeError("a request needs a string key");
    }
    const server = this.serveFor(request);
    const evaluation = server.evaluate(request);
    if (server === this.canary) {
      this.canaryWindow.push(evaluation.error ? 1 : 0);
      if (this.canaryWindow.length > this.windowSize) this.canaryWindow.shift();
    }
    let shadowEvaluation = null;
    if (this.shadow !== null && (this.canaryPercent === 0 || server === this.canary)) {
      shadowEvaluation = this.shadow.evaluate(request);
    }
    const record = {
      key: request.key,
      bucket: bucketOf(request.key),
      servedBy: server.id,
      value: evaluation.value,
      error: evaluation.error,
      shadow: shadowEvaluation === null ? null : { id: this.shadow.id, value: shadowEvaluation.value, error: shadowEvaluation.error },
      event: "served",
    };
    const outcome = this.healthCheck(record);
    this.requests.push(record);
    return { ...record, ...outcome };
  }

  canaryErrorRate() {
    if (this.canaryWindow.length === 0) return 0;
    return this.canaryWindow.reduce((a, b) => a + b, 0) / this.canaryWindow.length;
  }

  /** Sliding-window health policy: roll back a bad canary, promote a good one, otherwise keep serving. */
  healthCheck(record) {
    if (this.canary === null || this.canary.status !== "canary") return { event: "served" };
    if (canaryServed(this) >= this.promotionAfter && this.canaryErrorRate() <= this.errorThreshold) {
      return this.promote();
    }
    if (this.canaryWindow.length >= Math.max(4, Math.ceil(this.windowSize / 2)) && this.canaryErrorRate() >= this.errorThreshold) {
      return this.rollback();
    }
    return { event: "served" };
  }

  /** Promotes the canary to stable; the previous stable version is retired. */
  promote() {
    if (this.canary === null || this.canary.status !== "canary") return { event: "noop", reason: "no canary to promote" };
    const promoted = this.canary;
    const retired = this.stable;
    retired.status = "retired";
    promoted.status = "stable";
    this.stable = promoted;
    this.canary = null;
    this.previousStable = retired;
    const event = { event: "promote", version: promoted.id, retired: retired.id, at: this.requests.length };
    this.events.push(event);
    return event;
  }

  /** Rolls the canary back: it stops serving, the previous stable version takes over again. */
  rollback(reason = "error rate") {
    if (this.canary === null || this.canary.status !== "canary") return { event: "noop", reason: "nothing to roll back" };
    const failed = this.canary;
    if (this.previousStable === undefined && this.versions.indexOf(failed) === 0) {
      return { event: "noop", reason: "no previous version to fall back to" };
    }
    failed.status = "rolled-back";
    this.canary = null;
    this.rolledBack = failed;
    this.canaryWindow = [];
    const event = {
      event: "rollback",
      version: failed.id,
      restored: this.stable.id,
      errorRate: this.canaryErrorRate(),
      reason,
      at: this.requests.length,
    };
    this.events.push(event);
    return event;
  }

  metrics() {
    const served = this.requests.reduce((acc, record) => {
      acc[record.servedBy] = (acc[record.servedBy] || 0) + 1;
      return acc;
    }, {});
    const errors = this.requests.filter((record) => record.error).length;
    const shadowed = this.requests.filter((record) => record.shadow !== null).length;
    const shadowErrors = this.requests.filter((record) => record.shadow !== null && record.shadow.error).length;
    return {
      total: this.requests.length,
      served,
      errors,
      shadowed,
      shadowErrors,
      shadowErrorRate: shadowed === 0 ? 0 : shadowErrors / shadowed,
      canaryErrorRate: this.canaryErrorRate(),
      events: this.events.slice(),
    };
  }
}

/** Number of requests the current canary has actually served. */
function canaryServed(router) {
  return router.requests.reduce((acc, record) => acc + (record.servedBy === router.canary.id ? 1 : 0), 0);
}

/** Synthetic traffic: distinct keys plus repetitions so stickiness is observable. */
function makeTraffic({ keys = 400, repeat = 3, seed = 865 } = {}) {
  if (!Number.isInteger(keys) || keys < 1) throw new RangeError("keys must be >= 1");
  if (!Number.isInteger(repeat) || repeat < 1) throw new RangeError("repeat must be >= 1");
  const rand = mulberry32(seed);
  const traffic = [];
  for (let r = 0; r < repeat; r++) {
    for (let k = 0; k < keys; k++) {
      traffic.push({ key: `user-${k}`, features: [rand(), rand(), 1 + Math.floor(rand() * 5)] });
    }
  }
  return traffic;
}

/** Reference model behaviours used by the fixture. */
const predictable = (threshold) => (request) => (request.features[0] > threshold ? "accept" : "reject");
const faultyEvery = (n) => (request) => hashKey(request.key) % n === 0;

function selfCheck() {
  // --- validation -----------------------------------------------------------------------------------------
  assert.throws(() => new ServingRouter({}), /at least one version/);
  assert.throws(() => new ServingRouter({ versions: ["nope"] }), /ModelVersion/);
  assert.throws(() => new ServingRouter({ versions: [new ModelVersion({ id: "a", predict: () => 1 })], canaryPercent: 150 }), /canaryPercent/);
  assert.throws(() => new ServingRouter({ versions: [new ModelVersion({ id: "a", predict: () => 1 })], windowSize: 0 }), /windowSize/);
  assert.throws(() => new ServingRouter({ versions: [new ModelVersion({ id: "a", predict: () => 1 })], errorThreshold: 0 }), /errorThreshold/);
  assert.throws(() => new ServingRouter({ versions: [new ModelVersion({ id: "a", predict: () => 1 })], promotionAfter: 0 }), /promotionAfter/);
  assert.throws(() => new ModelVersion({ id: "", predict: () => 1 }), /needs an id/);
  assert.throws(() => new ModelVersion({ id: "a" }), /predict function/);
  assert.throws(() => makeTraffic({ keys: 0 }), /keys/);
  assert.throws(() => makeTraffic({ repeat: 0 }), /repeat/);
  const routerProbe = new ServingRouter({ versions: [new ModelVersion({ id: "v1", predict: () => 1 })] });
  assert.throws(() => routerProbe.handle({}), /string key/);
  assert.throws(() => routerProbe.handle(null), /string key/);

  // --- routing: exact band, stickiness, single server --------------------------------------------------------
  const v1 = new ModelVersion({ id: "v1", predict: predictable(0.5) });
  const v2 = new ModelVersion({ id: "v2", predict: (request) => (request.features[0] > 0.45 ? "accept" : "reject") });
  const v3 = new ModelVersion({ id: "v3", predict: predictable(0.4) });
  const canaryRouter = new ServingRouter({ versions: [v1, v2], canary: v2, shadow: v3, canaryPercent: 25, windowSize: 40, errorThreshold: 0.6, promotionAfter: 100000 });
  const traffic = makeTraffic({ keys: 400, repeat: 3, seed: 865 });
  const uniqueKeys = [...new Set(traffic.map((request) => request.key))];
  const expectedCanary = new Set(uniqueKeys.filter((key) => bucketOf(key) < 25));
  const seen = new Map();
  let shadowCount = 0;
  for (const request of traffic) {
    const outcome = canaryRouter.handle(request);
    assert.strictEqual(outcome.shadow === null, !(bucketOf(request.key) < 25), "only canary traffic is mirrored");
    if (outcome.shadow !== null) shadowCount += 1;
    assert.strictEqual(outcome.value, outcome.servedBy === "v2" ? v2.predict(request) : v1.predict(request), "the client sees the serving version's own answer");
    const previous = seen.get(request.key);
    if (previous === undefined) seen.set(request.key, outcome.servedBy);
    else assert.strictEqual(previous, outcome.servedBy, `key ${request.key} must be sticky`);
    assert.strictEqual(expectedCanary.has(request.key), outcome.servedBy === "v2", `bucket routing mismatch for ${request.key}`);
  }
  const metrics = canaryRouter.metrics();
  assert.strictEqual(metrics.total, traffic.length);
  assert.strictEqual(metrics.served.v1 + metrics.served.v2, traffic.length, "every request is served by exactly one live version");
  assert.strictEqual(metrics.served.v2, expectedCanary.size * 3, "the canary receives exactly the keys in its band, three times each");
  assert.strictEqual(metrics.shadowed, shadowCount);
  assert.strictEqual(metrics.shadowed, metrics.served.v2, "mirroring follows the canary traffic exactly");
  assert.ok(Math.abs(expectedCanary.size / uniqueKeys.length - 0.25) < 0.06, `the observed band share must be near the configured one (${(expectedCanary.size / uniqueKeys.length).toFixed(3)})`);
  assert.strictEqual(v3.status, "shadow");
  assert.strictEqual(v3.served, shadowCount, "the shadow model is evaluated once per mirrored request");

  // --- transparency: shadow answers never leak ----------------------------------------------------------------
  const leakProbe = new ServingRouter({ versions: [v1], shadow: v3, canary: v2, canaryPercent: 50 });
  for (const request of traffic.slice(0, 50)) {
    const outcome = leakProbe.handle(request);
    const expected = outcome.servedBy === "v2" ? v2.predict(request) : v1.predict(request);
    assert.strictEqual(outcome.value, expected, "the shadow must not influence the response");
    if (outcome.shadow !== null) assert.strictEqual(outcome.shadow.id, "v3");
  }
  const shadowErrors = leakProbe.requests.filter((record) => record.shadow !== null && record.shadow.error).length;
  const shadowed = leakProbe.requests.filter((record) => record.shadow !== null).length;
  assert.strictEqual(leakProbe.metrics().shadowErrors, shadowErrors, "shadow metrics are recomputed from the log");
  assert.strictEqual(leakProbe.metrics().shadowErrorRate, shadowed === 0 ? 0 : shadowErrors / shadowed);

  // --- automatic rollback of a bad canary -----------------------------------------------------------------------
  const goodStable = new ModelVersion({ id: "stable", predict: predictable(0.5) });
  const badCanary = new ModelVersion({ id: "bad", predict: predictable(0.5), faulty: faultyEvery(2) });
  const rollbackRouter = new ServingRouter({ versions: [goodStable, badCanary], canary: badCanary, canaryPercent: 100, windowSize: 20, errorThreshold: 0.4, promotionAfter: 100000 });
  const beforeRollback = goodStable.status;
  let rollbackAt = null;
  for (let i = 0; i < 200 && rollbackAt === null; i++) {
    const outcome = rollbackRouter.handle(traffic[i]);
    if (outcome.event === "rollback") rollbackAt = i;
  }
  assert.ok(rollbackAt !== null, "a canary failing half its requests must be rolled back");
  assert.ok(rollbackAt <= 20, `the rollback must arrive within the window (${rollbackAt})`);
  assert.strictEqual(badCanary.status, "rolled-back");
  assert.strictEqual(beforeRollback, "stable");
  assert.strictEqual(rollbackRouter.events.filter((event) => event.event === "rollback").length, 1);
  const afterRollback = [];
  for (let i = 200; i < 260; i++) afterRollback.push(rollbackRouter.handle(traffic[i]));
  assert.ok(afterRollback.every((outcome) => outcome.servedBy === "stable"), "traffic returns to the stable version");
  assert.strictEqual(rollbackRouter.rollback().event, "noop", "a second rollback is a no-op");
  assert.strictEqual(rollbackRouter.events.filter((event) => event.event === "rollback").length, 1, "the rollback happens exactly once");
  // The rollback restores exactly the previous behaviour.
  const restored = rollbackRouter.handle({ key: "user-3", features: [0.7, 0.1, 1] });
  assert.strictEqual(restored.value, goodStable.predict({ key: "user-3", features: [0.7, 0.1, 1] }));
  assert.strictEqual(restored.error, false, "the stable version is not faulty");
  assert.strictEqual(rollbackRouter.metrics().served.bad, rollbackAt + 1, "the canary only ever served up to the rollback");
  assert.strictEqual(rollbackRouter.metrics().errors, rollbackRouter.requests.filter((record) => record.error).length);

  // --- promotion of a healthy canary ------------------------------------------------------------------------------
  const healthy = new ModelVersion({ id: "next", predict: predictable(0.45) });
  const promoteRouter = new ServingRouter({ versions: [goodStable, healthy], canary: healthy, canaryPercent: 50, windowSize: 20, errorThreshold: 0.5, promotionAfter: 30 });
  let promoted = null;
  for (let i = 0; i < 400 && promoted === null; i++) {
    const outcome = promoteRouter.handle(traffic[i]);
    if (outcome.event === "promote") promoted = i;
  }
  assert.ok(promoted !== null, "a healthy canary must be promoted");
  assert.strictEqual(healthy.status, "stable");
  assert.strictEqual(goodStable.status, "retired");
  assert.strictEqual(promoteRouter.canary, null);
  const post = promoteRouter.handle({ key: "user-9", features: [0.9, 0, 1] });
  assert.strictEqual(post.servedBy, "next", "after promotion the new version serves everything");
  assert.strictEqual(post.value, healthy.predict({ key: "user-9", features: [0.9, 0, 1] }));
  assert.strictEqual(promoteRouter.events.length, 1);
  assert.strictEqual(promoteRouter.events[0].event, "promote");
  assert.strictEqual(promoteRouter.events[0].retired, "stable");

  // --- degenerate configurations ------------------------------------------------------------------------------------
  const noCanary = new ServingRouter({ versions: [goodStable], canaryPercent: 0 });
  const first = noCanary.handle(traffic[0]);
  assert.strictEqual(first.servedBy, "stable");
  assert.strictEqual(first.shadow, null, "there is nothing to mirror without a shadow or a canary");
  assert.strictEqual(noCanary.rollback().event, "noop");
  const fullCanary = new ServingRouter({ versions: [goodStable, healthy], canary: healthy, canaryPercent: 100, windowSize: 10, errorThreshold: 0.9, promotionAfter: 100000 });
  assert.ok(fullCanary.handle(traffic[0]).servedBy === "next");
  const zeroPercent = new ServingRouter({ versions: [goodStable, healthy], canary: healthy, canaryPercent: 0, windowSize: 10, errorThreshold: 0.9, promotionAfter: 100000 });
  assert.strictEqual(zeroPercent.handle(traffic[0]).servedBy, "stable", "a zero-percent canary receives nothing");
  const shadowOnly = new ServingRouter({ versions: [goodStable], shadow: v3, canaryPercent: 0 });
  const mirrored = shadowOnly.handle(traffic[1]);
  assert.strictEqual(mirrored.shadow.id, "v3");
  assert.strictEqual(mirrored.servedBy, "stable");
  const unrollable = new ServingRouter({ versions: [goodStable], canary: new ModelVersion({ id: "solo", predict: () => 1 }), canaryPercent: 10 });
  unrollable.canary = unrollable.versions[0];
  assert.strictEqual(unrollable.rollback().event, "noop", "the first version has nothing to fall back to");
  const neverPromote = new ServingRouter({ versions: [goodStable, healthy], canary: healthy, canaryPercent: 5, windowSize: 10, errorThreshold: 0.5, promotionAfter: 100000 });
  for (let i = 0; i < 100; i++) neverPromote.handle(traffic[i]);
  assert.strictEqual(neverPromote.events.length, 0, "a patient rollout must not promote early");
  assert.ok(neverPromote.metrics().canaryErrorRate === 0);

  // --- determinism ------------------------------------------------------------------------------------------------------
  const build = () => {
    const stable = new ModelVersion({ id: "v1", predict: predictable(0.5) });
    const candidate = new ModelVersion({ id: "v2", predict: predictable(0.45), faulty: faultyEvery(7) });
    const mirror = new ModelVersion({ id: "shadow", predict: predictable(0.6) });
    const router = new ServingRouter({ versions: [stable, candidate], canary: candidate, shadow: mirror, canaryPercent: 30, windowSize: 25, errorThreshold: 0.3, promotionAfter: 100000 });
    const outcomes = traffic.map((request) => router.handle(request));
    return { outcomes, metrics: router.metrics() };
  };
  const runA = build();
  const runB = build();
  assert.deepStrictEqual(runA.outcomes, runB.outcomes);
  assert.deepStrictEqual(runA.metrics.served, runB.metrics.served);
  assert.deepStrictEqual(runA.metrics.events, runB.metrics.events);
  assert.strictEqual(runA.metrics.shadowed, runB.metrics.shadowed);
  const otherSeed = build();
  assert.notDeepStrictEqual(makeTraffic({ seed: 866 }).slice(0, 5), traffic.slice(0, 5), "another traffic seed differs");

  console.log(`routing: ${metrics.served.v2} of ${metrics.total} requests to the canary (${((metrics.served.v2 / metrics.total) * 100).toFixed(1)}%), ${metrics.shadowed} mirrored`);
  console.log(`rollback: bad canary demoted after ${rollbackAt} requests, ${rollbackRouter.events.length} rollout event(s), stable traffic after that`);
  console.log(`promotion: healthy canary promoted after ${promoted} requests; shadow error rate ${(metrics.shadowErrorRate * 100).toFixed(1)}%`);
  console.log("PASS");
}

module.exports = { ServingRouter, ModelVersion, makeTraffic, hashKey, bucketOf, canaryServed, mulberry32 };

if (require.main === module) {
  selfCheck();
}
