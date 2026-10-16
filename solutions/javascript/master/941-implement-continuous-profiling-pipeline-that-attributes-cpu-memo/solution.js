/** Task 941 - Implement continuous profiling pipeline that attributes CPU/memory hotspots to code
 * and inputs. (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * A sampling profiler over a service call tree. One sample descends from the root: at every frame it
 * stops in that frame's own work with probability proportional to its exclusive weight, otherwise it
 * descends into a child subtree proportionally - exactly how a real stack sample lands on the
 * instruction being executed. Each sample therefore yields a stack, the terminal frame an exclusive
 * hit and every frame on the stack an inclusive hit. Samples carry an input label, so the pipeline
 * produces a differential profile: per-label exclusive/inclusive attribution that shows which
 * function got slower because the input got heavier.
 *
 * Simplified: weights are an analytic model instead of measured CPU cycles, the tree is static and
 * four levels deep, and samples are drawn one by one rather than in periodic bursts. Verified:
 * sample conservation, inclusive >= exclusive per frame, exclusive and inclusive shares against the
 * analytic weights, the differential ratio between input labels, profile merging, and empty,
 * single-sample and degenerate-label boundaries.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the only source of randomness; identical stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}
const unit = (rng) => Number(rng.next() >> 11n) / 2 ** 53;

/** Call tree with analytic exclusive weights per input label. */
const TREE = {
  root: "main",
  nodes: {
    main: { self: { small: 8, large: 8 }, children: ["parse", "compute", "encode"] },
    parse: { self: { small: 20, large: 10 }, children: ["lex", "build"] },
    lex: { self: { small: 30, large: 12 }, children: [] },
    build: { self: { small: 5, large: 3 }, children: [] },
    compute: { self: { small: 30, large: 120 }, children: ["dot", "norm"] },
    dot: { self: { small: 40, large: 260 }, children: [] },
    norm: { self: { small: 12, large: 20 }, children: [] },
    encode: { self: { small: 25, large: 25 }, children: [] },
  },
};
const LABELS = ["small", "large"];

/** Total analytic cost of a subtree, memoised per label - the descent probability denominator. */
function subtreeCost(nodes, node, label, cache = new Map()) {
  const key = `${node}:${label}`;
  if (cache.has(key)) return cache.get(key);
  const own = nodes[node].self[label];
  const total = own + nodes[node].children.reduce((a, c) => a + subtreeCost(nodes, c, label, cache), 0);
  cache.set(key, total);
  return total;
}

/** Independent oracle: analytic exclusive and inclusive shares of a label. */
function analyticShares(nodes, root, label) {
  const cache = new Map();
  const total = subtreeCost(nodes, root, label, cache);
  const inclusive = new Map();
  const walk = (node, probability) => {
    inclusive.set(node, (inclusive.get(node) ?? 0) + probability);
    const below = nodes[node].children.reduce((a, c) => a + subtreeCost(nodes, c, label, cache), 0);
    for (const child of nodes[node].children) {
      walk(child, (probability * subtreeCost(nodes, child, label, cache)) / (below + nodes[node].self[label]));
    }
  };
  walk(root, 1);
  return { exclusive: new Map(Object.entries(nodes).map(([n, v]) => [n, v.self[label] / total])), inclusive };
}

/** One stack sample: descend until the sample lands in a frame's own work. */
function takeSample(rng, nodes, root, label, cache) {
  const stack = [];
  let node = root;
  for (;;) {
    stack.push(node);
    const self = nodes[node].self[label];
    const below = nodes[node].children.reduce((a, c) => a + subtreeCost(nodes, c, label, cache), 0);
    let r = unit(rng) * (self + below);
    if (r < self) return { stack, leaf: node };
    r -= self;
    for (const child of nodes[node].children) {
      const share = subtreeCost(nodes, child, label, cache);
      if (r < share) { node = child; break; }
      r -= share;
    }
  }
}

function newProfile() { return { samples: 0, exclusive: new Map(), inclusive: new Map(), byLabel: new Map() }; }

/** Attribute samples: the terminal frame is exclusive, every frame on the stack is inclusive. */
function collect(seed, label, samples, nodes = TREE.nodes, root = TREE.root) {
  const rng = splitmix64(seed);
  const cache = new Map();
  const profile = newProfile();
  for (let i = 0; i < samples; i += 1) {
    const { stack, leaf } = takeSample(rng, nodes, root, label, cache);
    profile.samples += 1;
    profile.exclusive.set(leaf, (profile.exclusive.get(leaf) ?? 0) + 1);
    for (const frame of new Set(stack)) profile.inclusive.set(frame, (profile.inclusive.get(frame) ?? 0) + 1);
  }
  profile.byLabel.set(label, profile);
  return profile;
}

/** Merge interval profiles: a continuous pipeline aggregates many small windows. */
function mergeProfiles(profiles) {
  const merged = newProfile();
  for (const profile of profiles) {
    merged.samples += profile.samples;
    for (const map of ["exclusive", "inclusive"]) {
      for (const [key, value] of profile[map]) merged[map].set(key, (merged[map].get(key) ?? 0) + value);
    }
    for (const [label, nested] of profile.byLabel) merged.byLabel.set(label, nested);
  }
  return merged;
}

const share = (profile, map, frame) => (profile[map].get(frame) ?? 0) / Math.max(1, profile.samples);
const sorted = (profile, map) => [...profile[map]].sort((a, b) => b[1] - a[1]);

function main() {
  const small = collect(941, "small", 20000);
  const large = collect(941, "large", 20000);
  const truth = { small: analyticShares(TREE.nodes, TREE.root, "small"), large: analyticShares(TREE.nodes, TREE.root, "large") };

  // 1. Conservation: exclusive hits equal the sample count and the root is always on the stack.
  const exclusiveTotal = [...small.exclusive.values()].reduce((a, b) => a + b, 0);
  assert(exclusiveTotal === small.samples && small.samples === 20000, `exclusive hits must equal samples, got ${exclusiveTotal}`);
  assert(share(small, "inclusive", "main") === 1, "the root is on every stack");

  // 2. Invariant: inclusive attribution is never below exclusive attribution.
  for (const [frame, hits] of small.exclusive) assert(small.inclusive.get(frame) >= hits, `inclusive < exclusive for ${frame}`);

  // 3. Oracle: exclusive shares match the analytic weights of this label.
  for (const [frame, expected] of truth.small.exclusive) {
    assert(Math.abs(share(small, "exclusive", frame) - expected) < 0.02, `${frame} exclusive ${share(small, "exclusive", frame)} vs ${expected}`);
  }

  // 4. Oracle: inclusive shares match the analytic path probabilities.
  for (const [frame, expected] of truth.small.inclusive) {
    assert(Math.abs(share(small, "inclusive", frame) - expected) < 0.02, `${frame} inclusive ${share(small, "inclusive", frame)} vs ${expected}`);
  }

  // 5. Differential attribution: the input label shifts cost into compute only.
  assert(share(large, "exclusive", "compute") > 1.3 * share(small, "exclusive", "compute"), "a heavier input moves cost into compute");
  const stable = (p) => share(p, "exclusive", "encode") / share(p, "exclusive", "main");
  assert(Math.abs(stable(large) - stable(small)) < 0.1, "frames whose weight does not depend on the input keep their ratio");
  const ratio = share(large, "inclusive", "dot") / share(small, "inclusive", "dot");
  const truthRatio = truth.large.inclusive.get("dot") / truth.small.inclusive.get("dot");
  assert(Math.abs(ratio - truthRatio) / truthRatio < 0.05, `dot scales with the input: ${ratio} vs ${truthRatio}`);

  // 6. Hotspot ranking: the top exclusive frame and the top inclusive frame are as predicted.
  assert(sorted(small, "exclusive")[0][0] === [...truth.small.exclusive].sort((a, b) => b[1] - a[1])[0][0], "top exclusive frame matches the model");
  assert(sorted(large, "inclusive")[0][0] === "main" && sorted(large, "exclusive")[0][0] === "dot", "a heavy input makes dot the hotspot");

  // 7. Merging: interval profiles aggregate into the sum of their parts.
  const merged = mergeProfiles([collect(941, "small", 5000), collect(942, "small", 5000), collect(943, "small", 5000)]);
  assert(merged.samples === 15000, "merged sample count is the sum");
  assert([...merged.exclusive.values()].reduce((a, b) => a + b, 0) === 15000, "merged exclusive hits are conserved");
  assert(Math.abs(share(merged, "exclusive", "compute") - truth.small.exclusive.get("compute")) < 0.04, "a merged profile stays accurate");

  // 8. Boundary: no samples yield a well-formed profile with zero shares and no NaN.
  const none = collect(941, "small", 0);
  assert(none.samples === 0 && share(none, "exclusive", "main") === 0 && !Number.isNaN(share(none, "exclusive", "main")), "an empty profile is safe");

  // 9. Boundary: a single sample lands in exactly one frame.
  const one = collect(941, "small", 1);
  assert(one.samples === 1 && [...one.exclusive.values()].reduce((a, b) => a + b, 0) === 1, "one sample produces one exclusive hit");
  assert([...one.exclusive.values()].every((v) => v === 1) && one.exclusive.size === 1, "one sample attributes to a single frame");
  assert(one.inclusive.get("main") === 1 && one.inclusive.size >= 2, "one sample keeps its whole stack");

  // 10. Determinism and degenerate labels: same seed, same profile; an unused label stays empty.
  assert(JSON.stringify([...collect(941, "small", 4000).exclusive]) === JSON.stringify([...collect(941, "small", 4000).exclusive]), "the profiler is deterministic");
  const degenerate = collect(941, "tiny", 500, { root: { self: { tiny: 1 }, children: [] } }, "root");
  assert(degenerate.samples === 500 && degenerate.exclusive.get("root") === 500, "a single-frame tree attributes every sample to itself");

  console.log(`profiler: samples=${small.samples + large.samples} topExclusive=${sorted(large, "exclusive")[0][0]} dotRatio=${ratio.toFixed(2)} merged=${merged.samples}`);
  console.log("PASS");
}
module.exports = { splitmix64, collect, mergeProfiles, analyticShares, subtreeCost, takeSample, TREE, LABELS };
if (require.main === module) main();
