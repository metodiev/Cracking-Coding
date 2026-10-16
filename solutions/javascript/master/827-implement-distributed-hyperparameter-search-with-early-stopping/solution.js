/** Task 827 - Implement distributed hyperparameter search with early stopping and resource allocation
 * (BOHB). (Tier: Master, Beyond — Machine Learning & AI)
 *
 * BOHB = Hyperband's bracket schedule + TPE-on-KDE model-based sampling. Every bracket runs the
 * successive-halving structure (a rung evaluates a population at one budget, promotes the best 1/eta
 * to eta times that budget); brackets after the first draw their configurations from a kernel-density
 * model of the earlier observations, where candidates are proposed from a Gaussian mixture centred on
 * the best quartile and accepted by the TPE ratio p_good/p_bad, then snapped to the discrete grid.
 *
 * Distribution: an evaluation is a job of length `budget`; jobs are dispatched to a pool of `workers`
 * by a work-conserving discrete-event scheduler, so "at scale" is expressed through the worker count
 * and measured makespan/max-concurrency instead of real parallelism (no timers, no threads).
 *
 * What is simplified: the objective is a deterministic synthetic response surface
 * base(cfg) + z/sqrt(budget) with z from a seeded LCG (validation noise shrinks with the budget), not
 * a real training run; cost is linear in budget; the KDE bandwidth is a fixed heuristic rather than
 * BOHB's multilevel rule. No wall-clock speed-up is claimed — the verified properties are the
 * schedule, the promotion rule, resource safety, cost, recall and determinism.
 *
 * Invariants verified:
 *   - schedule: bracket sizes n_s and start budgets R*eta^-s follow the Hyperband closed form, rung
 *     populations shrink by eta, rung budgets grow by eta, every bracket ends at the maximum budget
 *     and the per-bracket budget equals the successive-halving total;
 *   - promotion (safety): for every rung the promoted set is exactly the top ceil(k/eta) by observed
 *     loss with ties by grid index, recomputed by an independent replay; an eliminated configuration
 *     is never re-evaluated inside its bracket and its budget sequence only grows;
 *   - resource safety: never more than `workers` evaluations running at once, the scheduler is
 *     work-conserving, and a rung of k equal-length jobs takes exactly ceil(k/workers)*budget;
 *   - cost: the spent budget equals the sum of the evaluation budgets and stays far below the
 *     full-sweep cost of evaluating every configuration at the maximum budget;
 *   - recall: the incumbent is the best configuration observed at the maximum budget and reaches the
 *     exact optimum of the response surface, beating random search on the same budget;
 *   - determinism: the same seed reproduces the whole history bit-for-bit.
 */
"use strict";

const PARAMETERS = ["lr", "batch", "depth"];

/** Discrete grid: log-spaced learning rates, powers-of-two batches and integer depths. */
const GRID = {
  lr: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
  batch: [32, 64, 128, 256],
  depth: [1, 2, 3, 4],
};

/** The synthetic response surface is minimised at (1e-2, 64, 3). */
const OPTIMUM = { lr: 1e-2, batch: 64, depth: 3 };

function enumerateSpace(grid = GRID) {
  const configs = [];
  for (const lr of grid.lr) {
    for (const batch of grid.batch) {
      for (const depth of grid.depth) configs.push({ lr, batch, depth });
    }
  }
  return configs;
}

const indexOfConfig = (config, configs) => configs.findIndex((candidate) =>
  candidate.lr === config.lr && candidate.batch === config.batch && candidate.depth === config.depth);

/** Normalised coordinate in [0,1]^3; the scale parameters are log-encoded. */
function encode(config) {
  const span = (value, low, high) => (value - low) / (high - low);
  return [
    span(Math.log10(config.lr), Math.log10(GRID.lr[0]), Math.log10(GRID.lr[GRID.lr.length - 1])),
    span(Math.log2(config.batch), Math.log2(GRID.batch[0]), Math.log2(GRID.batch[GRID.batch.length - 1])),
    span(config.depth, GRID.depth[0], GRID.depth[GRID.depth.length - 1]),
  ];
}

/** Nearest grid configuration to a normalised point (Euclidean in [0,1]^3). */
function snap(point, configs) {
  let best = configs[0];
  let bestDistance = Infinity;
  for (const config of configs) {
    const encoded = encode(config);
    let distance = 0;
    for (let d = 0; d < 3; d++) distance += (encoded[d] - point[d]) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = config;
    }
  }
  return best;
}

/** True quality of a configuration, independent of the budget (the search's hidden target). */
function baseLoss(config) {
  return 1.0 * Math.abs(Math.log10(config.lr / OPTIMUM.lr)) +
    0.4 * Math.abs(Math.log2(config.batch / OPTIMUM.batch)) +
    0.2 * Math.abs(config.depth - OPTIMUM.depth);
}

/** Deterministic LCG (Numerical Recipes constants) — the only source of "randomness". */
function lcg(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Standard normal deviate from two uniforms (Box-Muller), deterministic per stream. */
function gaussian(rng) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Validation objective: base(cfg) plus +/-0.5/sqrt(budget) of noise, memoised per (config, budget). */
function makeObjective(configs) {
  const cache = new Map();
  return (config, budget) => {
    const index = indexOfConfig(config, configs);
    const key = `${index}@${budget}`;
    if (cache.has(key)) return cache.get(key);
    const stream = lcg((0x9e3779b9 ^ (index * 2654435761) ^ budget) >>> 0);
    const value = baseLoss(config) + 0.5 * (stream() * 2 - 1) / Math.sqrt(budget);
    cache.set(key, value);
    return value;
  };
}

/** Work-conserving discrete-event scheduler; reports finish times plus the safety trace. */
function runCluster(jobs, workers) {
  if (workers <= 0) throw new Error("workers must be positive");
  if (jobs.length === 0) {
    return { makespan: 0, start: [], finish: [], maxConcurrent: 0, idleWhileReady: 0 };
  }
  const start = new Array(jobs.length).fill(0);
  const finish = new Array(jobs.length).fill(0);
  const waiting = jobs.map((job, index) => index);
  const running = [];
  let clock = 0;
  let maxConcurrent = 0;
  let idleWhileReady = 0;
  while (waiting.length > 0 || running.length > 0) {
    while (running.length < workers && waiting.length > 0) {
      const index = waiting.shift();
      start[index] = clock;
      finish[index] = clock + jobs[index].duration;
      running.push(index);
    }
    maxConcurrent = Math.max(maxConcurrent, running.length);
    if (running.length === 0) {
      if (waiting.length > 0) idleWhileReady += 1; // unreachable, kept as an invariant probe
      break;
    }
    if (running.length < workers && waiting.length > 0) idleWhileReady += 1;
    let nextClock = Infinity;
    for (const index of running) nextClock = Math.min(nextClock, finish[index]);
    clock = nextClock;
    for (let i = running.length - 1; i >= 0; i--) if (finish[running[i]] <= clock) running.splice(i, 1);
  }
  let makespan = 0;
  for (const value of finish) makespan = Math.max(makespan, value);
  return { makespan, start, finish, maxConcurrent, idleWhileReady };
}

/**
 * Hyperband schedule: sMax+1 brackets, bracket s starts with n_s = ceil((sMax+1)/(s+1) * eta^s)
 * configurations at budget R*eta^-s and runs exactly s+1 rungs, each halving the population while
 * multiplying the budget by eta, so every bracket ends at the maximum budget. `capacity` caps the
 * population by the number of distinct configurations that exist.
 */
function bracketSchedule({ eta, r0, budgetMax, capacity = Infinity }) {
  if (eta < 2) throw new Error("eta must be at least 2");
  if (r0 <= 0) throw new Error("r0 must be positive");
  const sMax = Math.round(Math.log(budgetMax / r0) / Math.log(eta));
  if (Math.abs(r0 * eta ** sMax - budgetMax) > 1e-9) {
    throw new Error("budgetMax must equal r0 * eta^s for an integer s");
  }
  const brackets = [];
  for (let s = 0; s <= sMax; s++) {
    const n = Math.min(capacity, Math.ceil((sMax + 1) / (s + 1) * eta ** s));
    const startBudget = r0 * eta ** (sMax - s);
    const rungs = [];
    for (let i = 0; i <= s; i++) {
      const count = Math.floor(n / eta ** i);
      if (count < 1) break;
      rungs.push({ count, budget: startBudget * eta ** i });
    }
    const rungBudget = rungs.reduce((acc, rung) => acc + rung.count * rung.budget, 0);
    brackets.push({ s, n, startBudget, rungs, rungBudget });
  }
  return { sMax, brackets };
}

/** KDE log-density with an isotropic Gaussian kernel in normalised coordinates. */
function kdeLogDensity(points, point, bandwidth) {
  if (points.length === 0) return -Infinity;
  let total = 0;
  for (const other of points) {
    let distance = 0;
    for (let d = 0; d < 3; d++) distance += (other[d] - point[d]) ** 2;
    total += Math.exp(-distance / (2 * bandwidth * bandwidth));
  }
  return Math.log(total / points.length + 1e-300);
}

/** Bandwidth = a quarter of the observed spread, floored so the kernel never collapses. */
function bandwidthFor(points) {
  if (points.length < 2) return 0.25;
  let worst = 0;
  for (let d = 0; d < 3; d++) {
    let low = Infinity;
    let high = -Infinity;
    for (const point of points) {
      low = Math.min(low, point[d]);
      high = Math.max(high, point[d]);
    }
    worst = Math.max(worst, high - low);
  }
  return Math.max(0.05, worst * 0.25);
}

/**
 * TPE-on-KDE sampler. The best quartile of the observations forms the "good" mixture; candidates are
 * jittered draws around good points (every fourth draw is uniform for exploration) and the one with
 * the largest log p_good - log p_bad is returned, snapped to the grid.
 */
function makeModelSampler(observations, configs, rng, candidates = 64) {
  const scored = observations.map((observation) => ({ config: observation.config,
    rank: observation.rank, point: encode(observation.config) }));
  scored.sort((a, b) => (a.rank - b.rank) || (indexOfConfig(a.config, configs) - indexOfConfig(b.config, configs)));
  const goodCount = Math.max(1, Math.floor(scored.length / 4));
  const goodPoints = scored.slice(0, goodCount).map((entry) => entry.point);
  const badPoints = scored.map((entry) => entry.point);
  const goodBandwidth = bandwidthFor(goodPoints);
  const badBandwidth = Math.max(goodBandwidth, 0.1);
  return () => {
    let best = null;
    let bestRatio = -Infinity;
    for (let attempt = 0; attempt < candidates; attempt++) {
      let point;
      if (attempt % 4 === 3) {
        point = [rng(), rng(), rng()];
      } else {
        const anchor = goodPoints[Math.floor(rng() * goodPoints.length) % goodPoints.length];
        point = anchor.map((value) =>
          Math.min(1, Math.max(0, value + gaussian(rng) * goodBandwidth)));
      }
      const ratio = kdeLogDensity(goodPoints, point, goodBandwidth) -
        kdeLogDensity(badPoints, point, badBandwidth);
      if (ratio > bestRatio) {
        bestRatio = ratio;
        best = snap(point, configs);
      }
    }
    return best;
  };
}

/**
 * BOHB driver. Rungs run through the cluster; promotions come from the observed losses; later
 * brackets share the observation pool through the KDE sampler. Configurations are sampled without
 * replacement until the grid is exhausted, after which re-use is allowed.
 */
function runBohb(options) {
  const { configs = enumerateSpace(), eta = 3, r0 = 1, budgetMax = 81, workers = 4, seed = 827,
    modelFromBracket = 1, candidates = 32, maxObservations = 256 } = options ?? {};
  const objective = makeObjective(configs);
  const schedule = bracketSchedule({ eta, r0, budgetMax, capacity: configs.length });
  const rng = lcg(seed);
  const history = [];
  const observations = [];
  let totalBudget = 0;
  let maxConcurrent = 0;
  let idleWhileReady = 0;
  let totalMakespan = 0;
  let incumbent = null;
  for (const bracket of schedule.brackets) {
    const window = observations.slice(-maxObservations);
    const sampler = bracket.s >= modelFromBracket && window.length > 0
      ? makeModelSampler(window, configs, rng, candidates) : null;
    const used = new Set(); // configurations are not repeated inside one bracket
    const population = [];
    for (let slot = 0; slot < bracket.n; slot++) {
      let config = null;
      for (let attempt = 0; attempt < 8 && config === null; attempt++) {
        const candidate = sampler
          ? sampler()
          : configs[Math.floor(rng() * configs.length) % configs.length];
        if (!used.has(indexOfConfig(candidate, configs)) || used.size >= configs.length) {
          config = candidate;
        }
      }
      if (config === null) {
        // exploration fallback: a uniform draw among the configurations unused in this bracket
        const unseen = configs.filter((unused, index) => !used.has(index));
        config = unseen[Math.floor(rng() * unseen.length) % unseen.length];
      }
      used.add(indexOfConfig(config, configs));
      population.push(config);
    }
    if (population.length !== bracket.n) throw new Error("bracket population could not be filled");
    let current = population;
    for (const rung of bracket.rungs) {
      if (current.length === 0) break;
      const cluster = runCluster(current.map(() => ({ duration: rung.budget })), workers);
      maxConcurrent = Math.max(maxConcurrent, cluster.maxConcurrent);
      idleWhileReady += cluster.idleWhileReady;
      totalMakespan += cluster.makespan;
      const results = current.map((config, index) => ({
        config,
        loss: objective(config, rung.budget),
        index,
        start: cluster.start[index],
        finish: cluster.finish[index],
      }));
      const floor = Math.min(...results.map((result) => result.loss));
      const keep = Math.max(1, Math.floor(results.length / eta));
      const ranked = results.slice().sort((a, b) => (a.loss - b.loss) || (a.index - b.index));
      const promotedIndices = new Set(ranked.slice(0, keep).map((result) => result.index));
      for (const result of results) {
        totalBudget += rung.budget;
        const entry = {
          bracket: bracket.s,
          budget: rung.budget,
          config: result.config,
          loss: result.loss,
          promoted: promotedIndices.has(result.index),
          rungCount: results.length,
          keep,
          start: result.start,
          finish: result.finish,
        };
        history.push(entry);
        observations.push({ config: result.config, rank: result.loss - floor, budget: rung.budget });
        if (rung.budget === budgetMax && (incumbent === null || result.loss < incumbent.loss)) {
          incumbent = entry;
        }
      }
      current = ranked.slice(0, keep).map((result) => result.config);
    }
  }
  const bestByBase = history.reduce((best, entry) =>
    baseLoss(entry.config) < baseLoss(best.config) ? entry : best, history[0]);
  return { history, observations, totalBudget, fullSweepCost: configs.length * budgetMax,
    maxConcurrent, idleWhileReady, makespan: totalMakespan, incumbent, bestByBase, schedule,
    budgetMax, workers, evaluated: new Set(history.map((entry) => indexOfConfig(entry.config, configs))) };
}

/** Independent random-search baseline: same objective, same total budget, every trial at max budget. */
function runRandomSearch({ configs = enumerateSpace(), budgetMax = 81, totalBudget, seed = 4242 }) {
  const objective = makeObjective(configs);
  const rng = lcg(seed);
  let spent = 0;
  let best = null;
  let trials = 0;
  while (spent + budgetMax <= totalBudget) {
    const config = configs[Math.floor(rng() * configs.length) % configs.length];
    const loss = objective(config, budgetMax);
    spent += budgetMax;
    trials += 1;
    if (best === null || loss < best.loss) best = { config, loss };
  }
  return { best, spent, trials };
}

module.exports = { GRID, PARAMETERS, OPTIMUM, enumerateSpace, indexOfConfig, encode, snap, baseLoss,
  lcg, gaussian, makeObjective, runCluster, bracketSchedule, kdeLogDensity, bandwidthFor,
  makeModelSampler, runBohb, runRandomSearch };

if (require.main === module) {
  const assert = require("node:assert");
  const configs = enumerateSpace();
  const metrics = [];

  // ---- 1. bracket schedule --------------------------------------------------------------
  const schedule = bracketSchedule({ eta: 3, r0: 1, budgetMax: 81 });
  assert.strictEqual(schedule.sMax, 4);
  assert.deepStrictEqual(schedule.brackets.map((b) => b.n), [5, 8, 15, 34, 81]);
  assert.deepStrictEqual(schedule.brackets.map((b) => b.startBudget), [81, 27, 9, 3, 1]);
  for (const bracket of schedule.brackets) {
    const rungs = bracket.rungs;
    assert.strictEqual(rungs.length, bracket.s + 1, `bracket ${bracket.s} has s+1 rungs`);
    assert.strictEqual(rungs[0].count, bracket.n);
    for (let i = 1; i < rungs.length; i++) {
      assert.strictEqual(rungs[i].count, Math.floor(bracket.n / 3 ** i), "populations shrink by eta");
      assert.strictEqual(rungs[i].budget, rungs[i - 1].budget * 3, "budgets grow by eta");
      assert.ok(rungs[i].count >= 1, "the final rung keeps at least one configuration");
    }
    assert.strictEqual(rungs[rungs.length - 1].budget, 81, "every bracket ends at the maximum budget");
    // independent count: population at rung i is floor(n/eta^i), budget r0*eta^(sMax-s+i)
    let expected = 0;
    for (let i = 0; i <= bracket.s; i++) {
      expected += Math.floor(bracket.n / 3 ** i) * (1 * 3 ** (schedule.sMax - bracket.s + i));
    }
    assert.strictEqual(bracket.rungBudget, expected, `bracket ${bracket.s} total budget`);
    assert.ok(bracket.rungBudget > 0);
  }
  assert.throws(() => bracketSchedule({ eta: 3, r0: 1, budgetMax: 80 }), /budgetMax/);
  assert.throws(() => bracketSchedule({ eta: 1, r0: 1, budgetMax: 81 }), /eta/);
  const twoEta = bracketSchedule({ eta: 2, r0: 1, budgetMax: 8 });
  assert.deepStrictEqual(twoEta.brackets.map((b) => b.n), [4, 4, 6, 8]);
  assert.deepStrictEqual(twoEta.brackets.map((b) => b.startBudget), [8, 4, 2, 1]);
  assert.deepStrictEqual(twoEta.brackets.map((b) => b.rungs.length), [1, 2, 3, 4]);
  const capped = bracketSchedule({ eta: 3, r0: 1, budgetMax: 81, capacity: 20 });
  assert.deepStrictEqual(capped.brackets.map((b) => b.n), [5, 8, 15, 20, 20]);

  // ---- 2. work-conserving cluster ---------------------------------------------------------
  const nine = runCluster(Array.from({ length: 9 }, () => ({ duration: 6 })), 4);
  assert.strictEqual(nine.makespan, Math.ceil(9 / 4) * 6, "equal jobs finish in ceil(k/W) waves");
  assert.strictEqual(nine.maxConcurrent, 4);
  assert.strictEqual(nine.idleWhileReady, 0);
  for (let i = 0; i < 9; i++) assert.strictEqual(nine.finish[i], (Math.floor(i / 4) + 1) * 6);
  const mixed = runCluster([{ duration: 5 }, { duration: 1 }, { duration: 4 }, { duration: 2 },
    { duration: 3 }], 2);
  assert.strictEqual(mixed.maxConcurrent, 2);
  assert.strictEqual(mixed.idleWhileReady, 0, "no worker idles while a job waits");
  assert.strictEqual(mixed.makespan, 8);
  assert.throws(() => runCluster([{ duration: 1 }], 0), /workers/);
  assert.strictEqual(runCluster([], 3).makespan, 0);

  // ---- 3. determinism ---------------------------------------------------------------------
  const run = runBohb({ seed: 827, workers: 4 });
  const rerun = runBohb({ seed: 827, workers: 4 });
  assert.deepStrictEqual(rerun.history.map((h) => [h.bracket, h.budget, h.loss, h.promoted]),
    run.history.map((h) => [h.bracket, h.budget, h.loss, h.promoted]));
  assert.ok(run.history.length > 0 && run.totalBudget > 0);

  // ---- 4. promotion replay (independent oracle) --------------------------------------------
  const rungs = new Map();
  for (const entry of run.history) {
    const key = `${entry.bracket}@${entry.budget}`;
    if (!rungs.has(key)) rungs.set(key, []);
    rungs.get(key).push(entry);
  }
  for (const entries of rungs.values()) {
    const keep = Math.max(1, Math.floor(entries.length / 3));
    const expected = entries.slice()
      .sort((a, b) => (a.loss - b.loss) ||
        (indexOfConfig(a.config, configs) - indexOfConfig(b.config, configs)))
      .slice(0, keep).map((entry) => indexOfConfig(entry.config, configs))
      .sort((a, b) => a - b);
    const observed = entries.filter((entry) => entry.promoted)
      .map((entry) => indexOfConfig(entry.config, configs)).sort((a, b) => a - b);
    assert.strictEqual(observed.length, keep, `rung size ${entries.length}`);
    assert.deepStrictEqual(observed, expected, `rung replay ${entries[0].bracket}@${entries[0].budget}`);
    for (const entry of entries) {
      assert.strictEqual(entry.keep, keep);
      assert.strictEqual(entry.rungCount, entries.length);
    }
  }
  assert.ok(rungs.size >= 5);
  // Structural invariant: inside a bracket every configuration survives a prefix of the rungs and is
  // eliminated exactly once (the last rung of the bracket may also end with a promotion).
  let chainsChecked = 0;
  for (const bracket of schedule.brackets) {
    const byConfig = new Map();
    for (const entry of run.history.filter((h) => h.bracket === bracket.s)) {
      const key = indexOfConfig(entry.config, configs);
      if (!byConfig.has(key)) byConfig.set(key, []);
      byConfig.get(key).push(entry);
    }
    for (const chain of byConfig.values()) {
      chain.sort((a, b) => a.budget - b.budget);
      for (let i = 0; i + 1 < chain.length; i++) {
        assert.ok(chain[i].promoted, `configuration survives rung ${chain[i].budget} or leaves the bracket`);
      }
      const last = chain[chain.length - 1];
      const isFinalRung = last.budget === bracket.rungs[bracket.rungs.length - 1].budget;
      assert.ok(isFinalRung ? last.rungCount === bracket.rungs[bracket.rungs.length - 1].count
        : last.promoted === false, "an eliminated configuration is not resurrected in its bracket");
      chainsChecked += 1;
    }
  }
  assert.ok(chainsChecked > 50);

  // ---- 5. budget monotonicity and per-bracket accounting -----------------------------------
  for (const bracket of schedule.brackets) {
    const perConfig = new Map();
    const entries = run.history.filter((h) => h.bracket === bracket.s);
    for (const entry of entries) {
      const key = indexOfConfig(entry.config, configs);
      if (!perConfig.has(key)) perConfig.set(key, []);
      const budgets = perConfig.get(key);
      assert.ok(!budgets.includes(entry.budget), "each budget is evaluated once per configuration");
      budgets.push(entry.budget);
    }
    for (const budgets of perConfig.values()) {
      for (let i = 1; i < budgets.length; i++) assert.ok(budgets[i] > budgets[i - 1], "budgets only grow");
    }
    const expectedEntries = bracket.rungs.reduce((acc, rung) => acc + rung.count, 0);
    assert.strictEqual(entries.length, expectedEntries, `bracket ${bracket.s} rung sizes`);
    const spent = entries.reduce((acc, entry) => acc + entry.budget, 0);
    assert.strictEqual(spent, bracket.rungBudget, `bracket ${bracket.s} budget accounting`);
    assert.ok(perConfig.size > 0 && perConfig.size <= bracket.n);
  }
  assert.strictEqual(run.totalBudget, run.history.reduce((acc, entry) => acc + entry.budget, 0));

  // ---- 6. resource and cost bounds ----------------------------------------------------------
  assert.ok(run.maxConcurrent <= 4, "the worker pool is never oversubscribed");
  assert.strictEqual(run.idleWhileReady, 0);
  assert.ok(run.totalBudget < run.fullSweepCost);
  const saving = 1 - run.totalBudget / run.fullSweepCost;
  assert.ok(saving > 0.6, `Hyperband saves ${(saving * 100).toFixed(1)}% of the full sweep`);
  const singleWorker = runBohb({ seed: 827, workers: 1, budgetMax: 27 });
  const fourWorkers = runBohb({ seed: 827, workers: 4, budgetMax: 27 });
  assert.deepStrictEqual(singleWorker.history.map((h) => h.loss), fourWorkers.history.map((h) => h.loss),
    "the worker count changes the timing but not the search trajectory");
  assert.ok(singleWorker.makespan > fourWorkers.makespan, "more workers finish sooner");

  // ---- 7. recall / search quality ------------------------------------------------------------
  const random = runRandomSearch({ totalBudget: run.totalBudget, seed: 4242 });
  assert.ok(random.spent <= run.totalBudget);
  assert.strictEqual(baseLoss(run.bestByBase.config), 0, `BOHB reaches the optimum (${baseLoss(run.bestByBase.config)})`);
  assert.strictEqual(baseLoss(run.incumbent.config), 0, "the incumbent is the true optimum");
  assert.ok(baseLoss(run.incumbent.config) < baseLoss(random.best.config),
    "better than random search with the same budget");
  const rankedByBase = configs.slice().sort((a, b) => baseLoss(a) - baseLoss(b));
  assert.ok(baseLoss(rankedByBase[0]) === 0 && baseLoss(rankedByBase[1]) > 0, "the optimum is unique");
  assert.ok(run.evaluated.size > configs.length * 0.75, `coverage ${run.evaluated.size}/${configs.length}`);    // The noise model justifies early stopping: at budget 1 the observed ranking barely agrees with
  // the true quality, while at the maximum budget it agrees almost perfectly.
  const concordance = (budget) => {
    const sample = run.history.filter((h) => h.budget === budget);
    let concordant = 0;
    let discordant = 0;
    for (let i = 0; i < sample.length; i++) {
      for (let j = i + 1; j < sample.length; j++) {
        const lossGap = sample[i].loss - sample[j].loss;
        const baseGap = baseLoss(sample[i].config) - baseLoss(sample[j].config);
        if (lossGap === 0 || baseGap === 0) continue;
        if (Math.sign(lossGap) === Math.sign(baseGap)) concordant += 1;
        else discordant += 1;
      }
    }
    return { rate: concordant / Math.max(1, concordant + discordant), pairs: concordant + discordant };
  };
  const lowBudget = concordance(1);
  const highBudget = concordance(run.budgetMax);
  assert.ok(lowBudget.pairs > 500 && highBudget.pairs > 20);
  assert.ok(highBudget.rate > 0.9, `max-budget concordance ${highBudget.rate}`);
  assert.ok(lowBudget.rate < highBudget.rate - 0.05, `budget-1 concordance ${lowBudget.rate}`);

  // ---- 8. model sampler favours the good region --------------------------------------------
  const observations = [
    { config: { lr: 1e-2, batch: 64, depth: 3 }, rank: 0.0 },
    { config: { lr: 3e-2, batch: 128, depth: 2 }, rank: 0.05 },
    { config: { lr: 1e-3, batch: 32, depth: 1 }, rank: 0.9 },
    { config: { lr: 1e-4, batch: 256, depth: 4 }, rank: 1.4 },
    { config: { lr: 3e-4, batch: 128, depth: 1 }, rank: 1.1 },
    { config: { lr: 3e-3, batch: 32, depth: 4 }, rank: 0.7 },
    { config: { lr: 1e-2, batch: 256, depth: 2 }, rank: 0.3 },
    { config: { lr: 3e-2, batch: 64, depth: 1 }, rank: 0.4 },
  ];
  const GOOD = 0.6;
  const sampler = makeModelSampler(observations, configs, lcg(11), 64);
  const uniform = lcg(99);
  let modelGood = 0;
  let uniformGood = 0;
  for (let i = 0; i < 400; i++) {
    const config = sampler();
    assert.ok(indexOfConfig(config, configs) >= 0, "the sampler returns grid configurations");
    if (baseLoss(config) <= GOOD) modelGood += 1;
    if (baseLoss(configs[Math.floor(uniform() * configs.length)]) <= GOOD) uniformGood += 1;
  }
  assert.ok(modelGood / 400 > uniformGood / 400 * 2, `model ${modelGood} vs uniform ${uniformGood}`);
  assert.ok(modelGood / 400 > 0.2, `model hit rate ${modelGood / 400}`);
  const samplerA = makeModelSampler(observations, configs, lcg(11), 64);
  const samplerB = makeModelSampler(observations, configs, lcg(11), 64);
  assert.deepStrictEqual(Array.from({ length: 20 }, () => samplerA()),
    Array.from({ length: 20 }, () => samplerB()), "the sampler is deterministic in its stream");

  // ---- 9. degenerate configurations ---------------------------------------------------------
  const tiny = runBohb({ configs: [{ lr: 1e-2, batch: 64, depth: 3 }], eta: 2, r0: 1, budgetMax: 4,
    workers: 1, seed: 5 });
  assert.strictEqual(tiny.history.length, 3); // one single-configuration rung per bracket
  assert.strictEqual(tiny.totalBudget, 4 + 2 + 1);
  const oneBracket = runBohb({ eta: 3, r0: 9, budgetMax: 9, workers: 2, seed: 5 });
  assert.strictEqual(oneBracket.schedule.brackets.length, 1);
  assert.strictEqual(oneBracket.schedule.brackets[0].rungs.length, 1);
  for (const entry of oneBracket.history) assert.strictEqual(entry.budget, 9);
  const noModel = runBohb({ seed: 5, modelFromBracket: 99 });
  assert.ok(noModel.history.length > 0, "brackets run with random sampling when no model is allowed");
  assert.ok(Number.isFinite(noModel.incumbent.loss));
  assert.ok(noModel.incumbent.budget === 81);

  metrics.push(`schedule: ${schedule.brackets.length} brackets, n = ${schedule.brackets.map((b) => b.n).join(",")}, start budgets = ${schedule.brackets.map((b) => b.startBudget).join(",")}`);
  metrics.push(`evaluations: ${run.history.length} rung entries, ${run.totalBudget} budget units (full sweep ${run.fullSweepCost}, saving ${(saving * 100).toFixed(1)}%)`);
  metrics.push(`resources: max concurrency ${run.maxConcurrent}/4, idle-while-ready ${run.idleWhileReady}, simulated time ${run.makespan} (1 worker: ${singleWorker.makespan}, 4 workers: ${fourWorkers.makespan})`);
  metrics.push(`recall: incumbent base loss ${baseLoss(run.incumbent.config).toFixed(6)}, random search ${baseLoss(random.best.config).toFixed(6)}, coverage ${run.evaluated.size}/${configs.length} configurations`);
  metrics.push(`sampler: ${(modelGood / 400 * 100).toFixed(1)}% of draws in the good region vs ${(uniformGood / 400 * 100).toFixed(1)}% uniform`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
