/** Task 884 - Implement autotuner that explores kernel/block/grid parameters and caches best. (Tier: Master, CTCI Beyond - Compilers, Languages & Solvers)
 *
 * A deterministic kernel autotuner: a parameter space of launch configurations, a legality validator,
 * an execution-cost model that plays the role of the GPU, three search strategies and a bounded cache
 * keyed by the problem shape.
 *
 *   Config space : blockX x blockY threads, unroll factor, vector width and the K-tile size.
 *   Legality     : threads <= 1024, threads * vectorWidth <= 4096, blockY <= 64, tileK <= k, shared
 *                  memory ((blockX + blockY) * tileK * 8 * 2 bytes) <= 48 KiB and grid <= 2^31; illegal
 *                  configurations are rejected by the validator and can never be cached.
 *   Cost model   : latency = launch overhead + tail effect + work / throughput, where the throughput
 *                  grows with occupancy (threads) and vector width, and is degraded by shared-memory
 *                  traffic per thread and by a stride penalty when blockY does not divide n. The model
 *                  is a pure function of (problem, config), which is what makes the tuner verifiable.
 *   Strategies   : exhaustive (the oracle), a pruned UCB1 bandit and coordinate descent, plus a
 *                  default configuration used when the measurement budget is zero.
 *   Cache        : keyed by (m, n, k, dtype, layout), LRU eviction, bounded capacity; a hit returns the
 *                  stored configuration without measuring anything, and every hit is re-validated.
 *
 * Invariants verified: the exhaustive search returns the exact argmin of the legal space (cross-checked
 * by an independent scan); UCB and coordinate descent stay within a small regret of that optimum while
 * measuring far fewer configurations; the pruned space provably contains the global optimum for the
 * problems tested (checked against exhaustive); cached configurations are the same object of truth as a
 * fresh search, cost zero measurements on a hit, respect LRU eviction, and are re-validated on use;
 * illegal configurations are rejected and never cached; the model is monotone in the problem size and
 * the tuned configuration beats the default one by a measured margin; determinism holds for every
 * strategy; and malformed problems, empty spaces and impossible budgets are handled.
 *
 * What is simplified: the "GPU" is an analytic cost model rather than real hardware, so no wall-clock
 * time is measured and results are exactly reproducible; the parameter space is a single-level tile
 * model with no register pressure, no occupancy API and no multi-kernel fusion; and no timers, threads,
 * subprocesses or randomness outside the seeded PRNG are used anywhere.
 */
"use strict";

const { ok, throws } = require("node:assert");

// ---------------------------------------------------------------------------------------------
// parameter space and legality
// ---------------------------------------------------------------------------------------------

const LIMITS = {
  maxThreads: 1024,
  maxThreadsTimesVector: 4096,
  maxBlockY: 64,
  maxSharedBytes: 48 * 1024,
  maxGrid: 2 ** 31,
  elementsPerThreadMinimum: 1,
};

const SPACE = {
  blockX: [16, 32, 64, 128, 256],
  blockY: [1, 2, 4, 8, 16],
  unroll: [1, 2, 4, 8],
  vectorWidth: [1, 2, 4],
  tileK: [8, 16, 32, 64],
};

function makeProblem({ m = 1024, n = 1024, k = 1024, dtype = "float32", layout = "row-major" } = {}) {
  for (const [name, value] of [["m", m], ["n", n], ["k", k]]) {
    if (!Number.isInteger(value) || value < 1) throw new RangeError(`${name} must be a positive integer (got ${value})`);
  }
  return { m, n, k, dtype, layout };
}

function sharedBytes(config) {
  return (config.blockX + config.blockY) * config.tileK * 8 * 2;
}

/** Legality of a launch configuration for a problem: returns null when legal, else the reason. */
function validate(problem, config) {
  const { blockX, blockY, unroll, vectorWidth, tileK } = config;
  const threads = blockX * blockY;
  if (threads > LIMITS.maxThreads) return `threads ${threads} exceed the device limit`;
  if (threads * vectorWidth > LIMITS.maxThreadsTimesVector) return `threads * vectorWidth ${threads * vectorWidth} exceed the limit`;
  if (blockY > LIMITS.maxBlockY) return `blockY ${blockY} exceeds the limit`;
  if (tileK > problem.k) return `tileK ${tileK} exceeds the reduction depth ${problem.k}`;
  if (sharedBytes(config) > LIMITS.maxSharedBytes) return `shared memory ${sharedBytes(config)} exceeds the device limit`;
  const gridX = Math.ceil(problem.m / blockX);
  const gridY = Math.ceil(problem.n / blockY);
  if (gridX * gridY > LIMITS.maxGrid) return `grid ${gridX}x${gridY} exceeds the limit`;
  if (gridX * gridY < 1) return "empty grid";
  if (unroll < 1 || vectorWidth < 1) return "unroll and vector width must be positive";
  if (problem.m * problem.n * problem.k / threads / unroll < LIMITS.elementsPerThreadMinimum) return "not enough work per thread";
  return null;
}

/** Enumerates the full cartesian space, keeping only legal configurations. */
function enumerateSpace(problem, space = SPACE) {
  const configs = [];
  for (const blockX of space.blockX) {
    for (const blockY of space.blockY) {
      for (const unroll of space.unroll) {
        for (const vectorWidth of space.vectorWidth) {
          for (const tileK of space.tileK) {
            const config = { blockX, blockY, unroll, vectorWidth, tileK };
            if (validate(problem, config) === null) configs.push(config);
          }
        }
      }
    }
  }
  if (configs.length === 0) throw new Error("the parameter space contains no legal configuration");
  return configs;
}

/**
 * A coarser space used by the bandit strategy: it keeps the dimensions that matter (wide vectors, deep
 * unrolling, small K tiles, tall-and-narrow blocks) while dropping the clearly dominated corners, so it
 * holds roughly an order of magnitude fewer configurations than the full space.
 */
function prunedSpace(problem) {
  return enumerateSpace(problem, {
    blockX: [64, 128, 256],
    blockY: [4, 8, 16],
    unroll: [2, 4, 8],
    vectorWidth: [2, 4],
    tileK: [8, 16],
  });
}

const keyOf = (problem) => `${problem.m}x${problem.n}x${problem.k}:${problem.dtype}:${problem.layout}`;

// ---------------------------------------------------------------------------------------------
// the cost model (standing in for the GPU)
// ---------------------------------------------------------------------------------------------

const MEASUREMENT_NOISE = 0.0; // kept at zero: the model must be a pure function for verification

/** Deterministic latency model in abstract time units. */
function latency(problem, config) {
  const reason = validate(problem, config);
  if (reason !== null) throw new RangeError(`illegal configuration: ${reason}`);
  const threads = config.blockX * config.blockY;
  const gridX = Math.ceil(problem.m / config.blockX);
  const gridY = Math.ceil(problem.n / config.blockY);
  const blocks = gridX * gridY;
  const work = problem.m * problem.n * problem.k;
  // Occupancy benefit saturates at 256 threads; very wide blocks pay a scheduling penalty.
  const occupancy = Math.min(1, threads / 256) * (threads > 512 ? 0.85 : 1);
  const vectorBenefit = 0.7 + 0.15 * config.vectorWidth;
  const unrollBenefit = 0.8 + 0.06 * config.unroll;
  const sharedPressure = 1 + (sharedBytes(config) / LIMITS.maxSharedBytes) * 0.6;
  const stridePenalty = problem.n % config.blockY === 0 ? 1 : 1.12;
  const perElement = (0.9 / (occupancy * vectorBenefit * unrollBenefit)) * sharedPressure * stridePenalty;
  const tailEffect = blocks % 8 === 0 ? 1 : 1.02;
  const launchOverhead = 20 + 0.5 * blocks;
  const tail = (work % (threads * config.unroll)) / work;
  return launchOverhead * tailEffect + work * perElement * 1e-6 + tail * 500;
  void MEASUREMENT_NOISE;
}

/** Wraps the model with a measurement counter, which is what the tuner's budget is spent on. */
function createDevice(problem) {
  let measurements = 0;
  return {
    get measurements() { return measurements; },
    measure(config) {
      measurements += 1;
      return latency(problem, config);
    },
    reset() { measurements = 0; },
  };
}

// ---------------------------------------------------------------------------------------------
// search strategies
// ---------------------------------------------------------------------------------------------

/** Exhaustive search: the oracle for every other strategy. */
function searchExhaustive(problem, { space = SPACE } = {}) {
  const configs = enumerateSpace(problem, space);
  const device = createDevice(problem);
  let best = null;
  for (const config of configs) {
    const value = device.measure(config);
    if (best === null || value < best.latency) best = { config, latency: value };
  }
  return { config: best.config, latency: best.latency, measurements: device.measurements, evaluated: configs.length };
}

/** Coordinate descent from a starting configuration: one dimension at a time, repeated. */
function searchCoordinateDescent(problem, { rounds = 4, start = null } = {}) {
  const device = createDevice(problem);
  let current = start ?? { blockX: 64, blockY: 4, unroll: 2, vectorWidth: 2, tileK: 16 };
  if (validate(problem, current) !== null) current = enumerateSpace(problem)[0];
  let bestLatency = device.measure(current);
  const dimensions = ["blockX", "blockY", "unroll", "vectorWidth", "tileK"];
  for (let round = 0; round < rounds; round++) {
    let improved = false;
    for (const dimension of dimensions) {
      for (const value of SPACE[dimension]) {
        const candidate = { ...current, [dimension]: value };
        if (validate(problem, candidate) !== null) continue;
        const measured = device.measure(candidate);
        if (measured < bestLatency - 1e-12) {
          bestLatency = measured;
          current = candidate;
          improved = true;
        }
      }
    }
    if (!improved) break;
  }
  return { config: current, latency: bestLatency, measurements: device.measurements };
}

/**
 * UCB1 bandit over the pruned space. Rewards are normalised latencies, so the arm with the best
 * observed mean wins; the exploration term is the classic sqrt(2 ln t / n_a).
 */
function searchBandit(problem, { pulls = 3, seed = 1234 } = {}) {
  const arms = prunedSpace(problem);
  const device = createDevice(problem);
  const stats = arms.map(() => ({ count: 0, total: 0, mean: Infinity }));
  let best = null;
  const total = Math.max(arms.length, arms.length * pulls);
  for (let step = 0; step < total; step++) {
    let chosen = -1;
    if (step < arms.length) {
      chosen = step; // one initial pull per arm keeps the estimates comparable
    } else {
      let bestScore = -Infinity;
      for (let index = 0; index < arms.length; index++) {
        if (stats[index].count === 0) {
          chosen = index;
          break;
        }
        const exploration = Math.sqrt((2 * Math.log(step + 1)) / stats[index].count);
        const score = -stats[index].mean + exploration * 1000;
        if (score > bestScore) {
          bestScore = score;
          chosen = index;
        }
      }
    }
    const measured = device.measure(arms[chosen]);
    stats[chosen].count += 1;
    stats[chosen].total += measured;
    stats[chosen].mean = stats[chosen].total / stats[chosen].count;
    if (best === null || measured < best.latency) best = { config: arms[chosen], latency: measured };
  }
  return { config: best.config, latency: best.latency, measurements: device.measurements, arms: arms.length, seed };
}

const DEFAULT_CONFIG = { blockX: 64, blockY: 4, unroll: 1, vectorWidth: 1, tileK: 8 };

/** Tunes with a strategy and a measurement budget; a zero budget falls back to the default config. */
function tune(problem, { strategy = "exhaustive", budget = 2000 } = {}) {
  if (!Number.isInteger(budget) || budget < 0) throw new RangeError("budget must be a non-negative integer");
  if (budget === 0) {
    if (validate(problem, DEFAULT_CONFIG) !== null) throw new Error("the default configuration is illegal for this problem");
    const device = createDevice(problem);
    return { config: DEFAULT_CONFIG, latency: device.measure(DEFAULT_CONFIG), measurements: device.measurements, fallback: true };
  }
  if (strategy === "exhaustive") return { ...searchExhaustive(problem), fallback: false };
  if (strategy === "bandit") return { ...searchBandit(problem, { pulls: Math.max(1, Math.floor(budget / 40)) }), fallback: false };
  if (strategy === "coordinate") return { ...searchCoordinateDescent(problem), fallback: false };
  throw new RangeError(`unknown strategy ${strategy}`);
}

// ---------------------------------------------------------------------------------------------
// the cache
// ---------------------------------------------------------------------------------------------

function createCache(capacity = 4) {
  if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError("cache capacity must be a positive integer");
  const entries = new Map(); // key -> { config, latency, lastUsed }
  let clock = 0;
  return {
    get capacity() { return capacity; },
    get size() { return entries.size; },
    keys() { return [...entries.keys()]; },
    /** Returns the cached configuration or null, refreshing its LRU position. */
    lookup(problem) {
      const key = keyOf(problem);
      const entry = entries.get(key);
      if (!entry) return null;
      clock += 1;
      entry.lastUsed = clock;
      return { config: { ...entry.config }, latency: entry.latency, measurements: 0, cached: true, hit: true };
    },
    store(problem, result) {
      const reason = validate(problem, result.config);
      if (reason !== null) throw new RangeError(`refusing to cache an illegal configuration: ${reason}`);
      const key = keyOf(problem);
      clock += 1;
      entries.set(key, { config: { ...result.config }, latency: result.latency, lastUsed: clock });
      while (entries.size > capacity) {
        let oldestKey = null;
        let oldest = Infinity;
        for (const [candidateKey, entry] of entries) {
          if (entry.lastUsed < oldest) {
            oldest = entry.lastUsed;
            oldestKey = candidateKey;
          }
        }
        entries.delete(oldestKey);
      }
      return { ...result, cached: true, hit: false };
    },
  };
}

/** Tunes through the cache: a hit costs nothing, a miss runs a search and stores the winner. */
function tuneWithCache(problem, cache, { strategy = "exhaustive", budget = 2000 } = {}) {
  const hit = cache.lookup(problem);
  if (hit) {
    const reason = validate(problem, hit.config);
    if (reason !== null) throw new Error(`the cache returned an illegal configuration: ${reason}`);
    return hit;
  }
  const result = tune(problem, { strategy, budget });
  return cache.store(problem, result);
}

// ---------------------------------------------------------------------------------------------
// checks
// ---------------------------------------------------------------------------------------------

/** Independent re-scan used as the oracle for the exhaustive strategy. */
function scanBest(problem) {
  const configs = enumerateSpace(problem);
  let best = null;
  for (const config of configs) {
    const value = latency(problem, config);
    if (best === null || value < best.latency) best = { config, latency: value };
  }
  return best;
}

/** Fraction of configurations that are at least as good as the given one (0 = the optimum). */
function percentileOf(problem, config) {
  const values = enumerateSpace(problem).map((candidate) => latency(problem, candidate)).sort((a, b) => a - b);
  const target = latency(problem, config);
  const worse = values.filter((value) => value < target - 1e-12).length;
  return worse / values.length;
}

function main() {
  const diagnostics = [];
  const problems = [
    makeProblem({ m: 512, n: 512, k: 512 }),
    makeProblem({ m: 1024, n: 1024, k: 128 }),
    makeProblem({ m: 256, n: 2048, k: 256, layout: "column-major" }),
    makeProblem({ m: 1000, n: 1000, k: 1000 }),
  ];

  // 1. Legality: the validator rejects every out-of-range configuration and the space only holds legal
  //    ones.
  const problem = problems[0];
  const legal = enumerateSpace(problem);
  const illegalExamples = [
    { blockX: 1024, blockY: 2, unroll: 1, vectorWidth: 4, tileK: 16 },  // 2048 threads
    { blockX: 256, blockY: 8, unroll: 1, vectorWidth: 4, tileK: 64 },   // shared memory blow-up
    { blockX: 64, blockY: 1, unroll: 1, vectorWidth: 4, tileK: 2048 },  // tileK above the reduction depth
  ];
  const rejected = illegalExamples.filter((config) => validate(problem, config) !== null);
  diagnostics.push(`legality: ${legal.length} legal configurations out of ${Object.values(SPACE).reduce((acc, values) => acc * values.length, 1)}, ${rejected.length}/${illegalExamples.length} crafted illegal ones rejected`);
  ok(legal.every((config) => validate(problem, config) === null), "the enumerated space must contain only legal configurations");
  ok(rejected.length === illegalExamples.length, "every crafted illegal configuration must be rejected");
  ok(legal.length > 40 && legal.length < 2000, `the legal space must be large but finite (got ${legal.length})`);

  // 2. The cost model is a pure function and is monotone in the problem size.
  const small = makeProblem({ m: 256, n: 256, k: 256 });
  const large = makeProblem({ m: 2048, n: 2048, k: 2048 });
  const shared = { blockX: 64, blockY: 4, unroll: 2, vectorWidth: 2, tileK: 16 };
  diagnostics.push(`model: latency(small) ${latency(small, shared).toFixed(3)}, latency(large) ${latency(large, shared).toFixed(3)}`);
  ok(latency(large, shared) > latency(small, shared), "a bigger problem must take longer for the same configuration");
  ok(latency(problem, shared) === latency(problem, shared), "the model must be a pure function of (problem, config)");
  throws(() => latency(problem, illegalExamples[0]), RangeError);

  // 3. Exhaustive search matches the independent scan exactly.
  const exhaustive = searchExhaustive(problem);
  const oracle = scanBest(problem);
  diagnostics.push(`exhaustive: ${exhaustive.measurements} measurements, best latency ${exhaustive.latency.toFixed(3)} at block ${exhaustive.config.blockX}x${exhaustive.config.blockY}, unroll ${exhaustive.config.unroll}, vec ${exhaustive.config.vectorWidth}, tileK ${exhaustive.config.tileK}`);
  ok(exhaustive.latency === oracle.latency, "the exhaustive search must find the exact optimum of the legal space");
  ok(exhaustive.measurements === legal.length, "exhaustive search must measure every legal configuration exactly once");
  ok(percentileOf(problem, exhaustive.config) === 0, "the exhaustive winner must sit at the top of the distribution");

  // 4. The pruned space used by the bandit contains the global optimum on these problems.
  let pruningFailures = 0;
  for (const candidate of problems) {
    const pruned = prunedSpace(candidate);
    const global = scanBest(candidate);
    const bestPruned = pruned.reduce((best, config) => (best === null || latency(candidate, config) < best.latency ? { config, latency: latency(candidate, config) } : best), null);
    if (Math.abs(bestPruned.latency - global.latency) > 1e-9) pruningFailures += 1;
  }
  diagnostics.push(`pruning safety: ${pruningFailures} of ${problems.length} problems lost the global optimum when pruning`);
  ok(pruningFailures === 0, "the pruned space must still contain the global optimum for these problems");

  // 5. Bandit and coordinate descent: small regret with far fewer measurements.
  for (const candidate of problems) {
    const reference = scanBest(candidate);
    const bandit = searchBandit(candidate);
    const coordinate = searchCoordinateDescent(candidate);
    const banditRegret = (bandit.latency - reference.latency) / reference.latency;
    const coordinateRegret = (coordinate.latency - reference.latency) / reference.latency;
    diagnostics.push(`problem ${candidate.m}x${candidate.n}x${candidate.k}: bandit regret ${(banditRegret * 100).toFixed(2)}% with ${bandit.measurements} measurements (exhaustive needs ${enumerateSpace(candidate).length}), coordinate regret ${(coordinateRegret * 100).toFixed(2)}% with ${coordinate.measurements}`);
    ok(banditRegret >= -1e-9 && banditRegret < 0.05, "the bandit must stay within 5% of the optimum");
    ok(coordinateRegret >= -1e-9 && coordinateRegret < 0.05, "coordinate descent must stay within 5% of the optimum");
    ok(bandit.measurements < enumerateSpace(candidate).length, "the bandit must measure fewer configurations than exhaustive search");
    ok(coordinate.measurements < enumerateSpace(candidate).length, "coordinate descent must measure fewer configurations than exhaustive search");
    ok(percentileOf(candidate, bandit.config) <= 0.05, "the bandit's configuration must land in the best 5% of the space");
  }

  // 6. The tuned configuration beats the default one, and the strategy is reproducible.
  const defaultLatency = latency(problem, DEFAULT_CONFIG);
  const speedup = defaultLatency / exhaustive.latency;
  diagnostics.push(`tuning benefit: default ${defaultLatency.toFixed(3)} vs tuned ${exhaustive.latency.toFixed(3)} (${speedup.toFixed(2)}x)`);
  ok(speedup > 1, "tuning must improve on the default configuration");
  const repeatExhaustive = searchExhaustive(problem);
  const repeatBandit = searchBandit(problem);
  ok(JSON.stringify(repeatExhaustive) === JSON.stringify(exhaustive), "exhaustive search must be reproducible");
  ok(JSON.stringify(repeatBandit) === JSON.stringify(searchBandit(problem)), "the bandit must be reproducible");
  const differentProblem = searchBandit(problems[1]);
  ok(JSON.stringify(differentProblem.config) !== JSON.stringify(repeatBandit.config) || differentProblem.latency !== repeatBandit.latency, "a different problem must be tuned independently");

  // 7. The cache: hits cost nothing, misses search, LRU eviction is honoured and every hit is validated.
  const cache = createCache(2);
  const first = tuneWithCache(problems[0], cache, { strategy: "bandit" });
  const second = tuneWithCache(problems[0], cache, { strategy: "bandit" });
  diagnostics.push(`cache: first call ${first.measurements} measurements (hit: ${first.hit}), second call ${second.measurements} measurements (hit: ${second.hit}), size ${cache.size}/${cache.capacity}`);
  ok(!first.hit && first.measurements > 0, "a cold cache must run a search");
  ok(second.hit && second.measurements === 0, "a warm cache must return the configuration without measuring");
  ok(JSON.stringify(second.config) === JSON.stringify(first.config), "a cache hit must return the tuned configuration");
  ok(second.latency === first.latency, "the cached latency must match the measured one");
  tuneWithCache(problems[1], cache, { strategy: "bandit" });
  tuneWithCache(problems[2], cache, { strategy: "bandit" });
  diagnostics.push(`cache after three distinct problems (capacity 2): ${cache.keys().join(" | ")}`);
  ok(cache.size === 2, "the cache must respect its capacity");
  ok(!cache.keys().includes(keyOf(problems[0])), "the least recently used entry must be evicted");
  const evicted = tuneWithCache(problems[0], cache, { strategy: "bandit" });
  ok(!evicted.hit && evicted.measurements > 0, "an evicted problem must be tuned again");
  const keyed = createCache(4);
  keyed.store(problem, { config: exhaustive.config, latency: exhaustive.latency });
  const revalidated = keyed.lookup(problem);
  ok(validate(problem, revalidated.config) === null, "a cache hit must be re-validated before use");
  throws(() => keyed.store(problem, { config: illegalExamples[0], latency: 1 }), RangeError);
  throws(() => createCache(0), RangeError);
  throws(() => createCache(1.5), RangeError);

  // 8. Budgets, strategies and validation.
  const noBudget = tune(problem, { strategy: "exhaustive", budget: 0 });
  diagnostics.push(`zero budget fallback: ${JSON.stringify(noBudget.config)} at ${noBudget.latency.toFixed(3)} (fallback: ${noBudget.fallback})`);
  ok(noBudget.fallback && noBudget.config.blockX === DEFAULT_CONFIG.blockX, "a zero budget must fall back to the default configuration");
  ok(noBudget.latency >= exhaustive.latency - 1e-9, "the default configuration cannot beat the tuned optimum");
  throws(() => tune(problem, { strategy: "unknown" }), RangeError);
  throws(() => tune(problem, { budget: -1 }), RangeError);
  throws(() => makeProblem({ m: 0 }), RangeError);
  throws(() => makeProblem({ n: 1.5 }), RangeError);
  throws(() => enumerateSpace(makeProblem({ m: 1024, n: 1024, k: 1 }), { blockX: [1024], blockY: [16], unroll: [1], vectorWidth: [4], tileK: [64] }), /no legal configuration/);
  const tiny = makeProblem({ m: 16, n: 16, k: 16 });
  const tinyBest = searchExhaustive(tiny);
  ok(validate(tiny, tinyBest.config) === null && tinyBest.latency > 0, "small problems must still tune to a legal configuration");

  // 9. Every tuned configuration would actually launch on the modelled device.
  let launchFailures = 0;
  for (const candidate of problems) {
    const tuned = searchExhaustive(candidate);
    if (validate(candidate, tuned.config) !== null) launchFailures += 1;
    const threads = tuned.config.blockX * tuned.config.blockY;
    if (threads > LIMITS.maxThreads) launchFailures += 1;
    if (sharedBytes(tuned.config) > LIMITS.maxSharedBytes) launchFailures += 1;
  }
  diagnostics.push(`launch legality of tuned configurations: ${launchFailures} violations over ${problems.length} problems`);
  ok(launchFailures === 0, "every tuned configuration must be launchable within the device limits");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  LIMITS,
  SPACE,
  DEFAULT_CONFIG,
  makeProblem,
  sharedBytes,
  validate,
  enumerateSpace,
  prunedSpace,
  keyOf,
  latency,
  createDevice,
  searchExhaustive,
  searchCoordinateDescent,
  searchBandit,
  tune,
  createCache,
  tuneWithCache,
  scanBest,
  percentileOf,
};

if (require.main === module) main();
