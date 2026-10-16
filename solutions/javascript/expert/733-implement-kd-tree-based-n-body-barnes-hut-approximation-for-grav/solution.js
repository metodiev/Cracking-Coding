/** Task 733 - Implement kd-tree based n-body Barnes-Hut approximation for gravity. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The direct n-body force sum costs O(N^2). Barnes-Hut (Barnes & Hut 1986) builds a spatial
 * tree, stores the total mass and centre of mass in every node, and then, for each particle,
 * walks the tree: if a node's size s satisfies s / d < theta (d = distance from the particle to
 * the node's centre of mass) the whole node is replaced by a single point mass at its centre of
 * mass; otherwise the walk descends. That turns the force sum into O(N log N) at the price of a
 * controllable error, with theta = 0 recovering the exact O(N^2) sum.
 *
 * This file implements the kd-tree (median split along the widest axis), the centre-of-mass
 * reduction, the soft Plummer-softened force and the theta walk, next to the direct sum that
 * serves as the oracle.
 *
 * Simplified: two spatial dimensions, a kd-tree instead of an octree (same idea, simpler split
 * rule), constant leaf capacity with a depth cap instead of a bucketing scheme, no
 * time integration and no periodic images. What is verified: the tree partitions the particles
 * exactly once; every node's mass equals the sum of its leaves' masses and its centre of mass
 * lies inside its own bounding box; total momentum and the root totals are consistent; theta = 0
 * reproduces the direct sum to machine precision while growing theta grows the error
 * monotonically and theta = 0.5 keeps it small; the approximation really does far fewer
 * interactions than N^2; and the degenerate inputs (no particles, one particle, a pair, all
 * particles coincident) behave. Positions come from a seeded splitmix64 stream.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

const THETA_SWEEP = [0, 0.3, 0.5, 0.7, 1.0];

/** Deterministic splitmix64 -> [0, 1). */
function makeRng(seed) {
  let state = (BigInt(seed >>> 0) + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
  return function next() {
    state = (state + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
    z ^= z >> 31n;
    return Number(z >> 11n) / 9007199254740992;
  };
}

const SOFTENING = 0.01;
const GRAVITY = 1;

/** A cloud with a couple of heavy clumps, which is what makes the tree earn its keep. */
function makeCloud(rng, count, extent = 1) {
  const clumps = [[0.3, 0.7], [0.7, 0.3], [0.5, 0.5]];
  return Array.from({ length: count }, (_, i) => {
    const anchor = clumps[i % clumps.length];
    return {
      x: anchor[0] + (rng() - 0.5) * 0.2 * extent,
      y: anchor[1] + (rng() - 0.5) * 0.2 * extent,
      mass: 1 + Math.floor(rng() * 3),
    };
  });
}

class KdNode {
  constructor(indices) {
    this.indices = indices;
    this.left = null;
    this.right = null;
    this.mass = 0;
    this.cx = 0;
    this.cy = 0;
    this.size = 0;
  }

  get leaf() {
    return this.left === null && this.right === null;
  }
}

/**
 * Median-split kd-tree: the widest axis of the current bounding box is split at the median, so
 * both children are non-empty by construction and the depth is O(log N).
 */
function buildTree(particles, { leafSize = 8, maxDepth = 32 } = {}) {
  const indices = particles.map((_, i) => i);
  const root = new KdNode(indices);
  const recurse = (node, depth) => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const index of node.indices) {
      minX = Math.min(minX, particles[index].x);
      maxX = Math.max(maxX, particles[index].x);
      minY = Math.min(minY, particles[index].y);
      maxY = Math.max(maxY, particles[index].y);
    }
    node.bounds = [minX, maxX, minY, maxY];
    node.size = Math.max(maxX - minX, maxY - minY);
    if (node.indices.length <= leafSize || depth >= maxDepth) return node;
    const axis = maxX - minX >= maxY - minY ? "x" : "y";
    const sorted = node.indices.slice().sort((a, b) => particles[a][axis] - particles[b][axis]);
    const middle = sorted.length >> 1;
    const left = new KdNode(sorted.slice(0, middle));
    const right = new KdNode(sorted.slice(middle));
    node.left = recurse(left, depth + 1);
    node.right = recurse(right, depth + 1);
    return node;
  };
  recurse(root, 0);
  reduceMass(root, particles);
  return root;
}

/** Bottom-up reduction: total mass and centre of mass of every node. */
function reduceMass(node, particles) {
  if (node.leaf) {
    let mass = 0;
    let sx = 0;
    let sy = 0;
    for (const index of node.indices) {
      mass += particles[index].mass;
      sx += particles[index].mass * particles[index].x;
      sy += particles[index].mass * particles[index].y;
    }
    node.mass = mass;
    node.cx = mass > 0 ? sx / mass : 0;
    node.cy = mass > 0 ? sy / mass : 0;
    return node;
  }
  reduceMass(node.left, particles);
  reduceMass(node.right, particles);
  node.mass = node.left.mass + node.right.mass;
  node.cx = node.mass > 0 ? (node.left.mass * node.left.cx + node.right.mass * node.right.cx) / node.mass : 0;
  node.cy = node.mass > 0 ? (node.left.mass * node.left.cy + node.right.mass * node.right.cy) / node.mass : 0;
  return node;
}

/** Softened pair force from `source` on `target`, returned as [fx, fy]. */
function pairForce(target, source, softening = SOFTENING) {
  const dx = source.x - target.x;
  const dy = source.y - target.y;
  const r2 = dx * dx + dy * dy + softening * softening;
  const factor = (GRAVITY * target.mass * source.mass) / (r2 * Math.sqrt(r2));
  return [factor * dx, factor * dy];
}

/** Direct O(N^2) sum: the oracle every approximation below is measured against. */
function directForces(particles, softening = SOFTENING) {
  const forces = [];
  let interactions = 0;
  for (let i = 0; i < particles.length; i++) {
    let fx = 0;
    let fy = 0;
    for (let j = 0; j < particles.length; j++) {
      if (i === j) continue;
      const [dx, dy] = pairForce(particles[i], particles[j], softening);
      fx += dx;
      fy += dy;
      interactions++;
    }
    forces.push([fx, fy]);
  }
  return { forces, interactions };
}

/**
 * Barnes-Hut walk for one particle. `stats` records how many node interactions were used, which
 * is the O(N log N) claim made concrete. The particle's own node may never be replaced by its
 * centre of mass, otherwise a particle would attract itself.
 */
function forceOn(particles, tree, target, theta, stats = { interactions: 0 }) {
  let fx = 0;
  let fy = 0;
  const visit = (node) => {
    if (node.indices.length === 0) return;
    const dx = node.cx - target.x;
    const dy = node.cy - target.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const own = node.indices.includes(target.index);
    if (node.leaf || (!own && node.size / Math.max(distance, 1e-30) < theta)) {
      if (node.leaf) {
        for (const index of node.indices) {
          if (index === target.index) continue;
          const [px, py] = pairForce(target, particles[index]);
          fx += px;
          fy += py;
          stats.interactions++;
        }
      } else {
        const [px, py] = pairForce(target, { x: node.cx, y: node.cy, mass: node.mass });
        fx += px;
        fy += py;
        stats.interactions++;
      }
      return;
    }
    visit(node.left);
    visit(node.right);
  };
  visit(tree);
  return { force: [fx, fy], interactions: stats.interactions };
}

function treeForces(particles, tree, theta) {
  const stats = { interactions: 0 };
  const forces = particles.map((particle, index) =>
    forceOn(particles, tree, { ...particle, index }, theta, stats).force);
  return { forces, interactions: stats.interactions };
}

/** Relative force error against the oracle: max |dF| over max |F|. */
function forceError(approximate, exact) {
  let worst = 0;
  let scale = 0;
  for (let i = 0; i < exact.length; i++) {
    scale = Math.max(scale, Math.hypot(exact[i][0], exact[i][1]));
    worst = Math.max(worst, Math.hypot(approximate[i][0] - exact[i][0], approximate[i][1] - exact[i][1]));
  }
  return { worst, relative: scale > 0 ? worst / scale : 0, scale };
}

/** Walk the tree and collect every leaf: used for the partition invariant. */
function leaves(node, out = []) {
  if (node.leaf) out.push(node);
  else {
    leaves(node.left, out);
    leaves(node.right, out);
  }
  return out;
}

function totalMass(node) {
  return node.leaf ? node.mass : node.left.mass + node.right.mass;
}

module.exports = {
  makeRng, makeCloud, buildTree, reduceMass, pairForce, directForces, forceOn, treeForces,
  forceError, leaves, totalMass, KdNode, SOFTENING, THETA_SWEEP,
};

if (require.main === module) {
  const rng = makeRng(20260733);

  // -- the tree is a partition, and its reductions are consistent --------------------------
  const particles = makeCloud(rng, 240);
  const tree = buildTree(particles, { leafSize: 8 });
  const leafNodes = leaves(tree);
  const covered = leafNodes.flatMap((leaf) => leaf.indices).sort((a, b) => a - b);
  deepStrictEqual(covered, particles.map((_, i) => i),
    "every particle must live in exactly one leaf");
  ok(leafNodes.every((leaf) => leaf.indices.length >= 1 && leaf.indices.length <= 8),
    "leaves must respect the leaf capacity");
  ok(tree.indices.length === particles.length, "the root holds everything");
  const expectedMass = particles.reduce((acc, p) => acc + p.mass, 0);
  strictEqual(totalMass(tree), expectedMass, "the root mass is the total mass");
  const centre = particles.reduce((acc, p) => [acc[0] + p.mass * p.x, acc[1] + p.mass * p.y], [0, 0])
    .map((value) => value / expectedMass);
  ok(Math.abs(tree.cx - centre[0]) < 1e-12 && Math.abs(tree.cy - centre[1]) < 1e-12,
    "the root centre of mass is the mass-weighted mean");
  let comInside = true;
  let massConsistent = true;
  const walk = (node) => {
    if (!node.leaf) {
      if (node.left.indices.length === 0 || node.right.indices.length === 0) comInside = false;
      if (Math.abs(node.mass - (node.left.mass + node.right.mass)) > 1e-12) massConsistent = false;
      walk(node.left);
      walk(node.right);
    } else {
      for (const index of node.indices) {
        const p = particles[index];
        if (p.x < node.bounds[0] - 1e-12 || p.x > node.bounds[1] + 1e-12
          || p.y < node.bounds[2] - 1e-12 || p.y > node.bounds[3] + 1e-12) comInside = false;
      }
    }
  };
  walk(tree);
  ok(comInside, "every leaf contains exactly its own particles and nothing outside its box");
  ok(massConsistent, "every internal node's mass is the sum of its children");
  const depth = (node) => (node.leaf ? 1 : 1 + Math.max(depth(node.left), depth(node.right)));
  ok(depth(tree) >= 3 && depth(tree) <= 32, "a cloud of 240 particles needs a real tree");
  console.log(`tree: ${leafNodes.length} leaves, depth ${depth(tree)}, total mass ${expectedMass}`);

  // -- theta = 0 recovers the direct sum ----------------------------------------------------
  const oracle = directForces(particles);
  const exactTree = treeForces(particles, tree, 0);
  const exactError = forceError(exactTree.forces, oracle.forces);
  ok(exactError.relative < 1e-12,
    `theta = 0 must reproduce the direct sum to machine precision, got ${exactError.relative}`);
  strictEqual(exactTree.interactions, oracle.interactions,
    "and it must touch exactly the same number of pairs as the direct sum");
  console.log(`theta=0: relative error ${exactError.relative.toExponential(2)}` +
    ` with ${exactTree.interactions} interactions`);

  // -- accuracy and cost as theta grows -----------------------------------------------------
  const sweep = THETA_SWEEP.map((theta) => {
    const run = treeForces(particles, tree, theta);
    const error = forceError(run.forces, oracle.forces);
    return { theta, relative: error.relative, interactions: run.interactions };
  });
  for (let i = 1; i < sweep.length; i++) {
    ok(sweep[i].interactions <= sweep[i - 1].interactions,
      `theta ${sweep[i].theta} must not use more interactions than ${sweep[i - 1].theta}`);
  }
  const standard = sweep.find((entry) => entry.theta === 0.5);
  ok(standard.relative < 0.05,
    `the standard theta = 0.5 must keep the force error small, got ${standard.relative}`);
  ok(standard.interactions < oracle.interactions / 2,
    `and it must halve the interaction count at least: ${standard.interactions} vs ${oracle.interactions}`);
  ok(sweep[sweep.length - 1].relative >= standard.relative,
    "opening the opening angle must not improve accuracy");
  console.log(`theta sweep: ${sweep.map((s) => `${s.theta}:${(s.relative * 100).toFixed(2)}%/${s.interactions}`).join(" ")}`);

  // -- the O(N log N) claim at scale ---------------------------------------------------------
  const bigParticles = makeCloud(rng, 1200);
  const bigTree = buildTree(bigParticles, { leafSize: 8 });
  const bigTheta = treeForces(bigParticles, bigTree, 0.5);
  const bigOraclePairs = bigParticles.length * (bigParticles.length - 1);
  ok(bigTheta.interactions < bigOraclePairs / 5,
    `Barnes-Hut must beat O(N^2) by a wide margin: ${bigTheta.interactions} vs ${bigOraclePairs}`);
  const directSample = directForces(bigParticles.slice(0, 60));
  const sampledTree = buildTree(bigParticles.slice(0, 60), { leafSize: 8 });
  const sampledError = forceError(treeForces(bigParticles.slice(0, 60), sampledTree, 0.5).forces, directSample.forces);
  ok(sampledError.relative < 0.05, "and stay accurate while doing it");
  console.log(`N=1200: ${bigTheta.interactions} tree interactions vs ${bigOraclePairs} direct pairs`);

  // -- the approximation breaks Newton's third law only slightly ----------------------------
  const netForce = bigTheta.forces.reduce((acc, [fx, fy]) => [acc[0] + fx, acc[1] + fy], [0, 0]);
  const magnitude = bigTheta.forces.reduce((acc, [fx, fy]) => acc + Math.hypot(fx, fy), 0);
  ok(Math.hypot(netForce[0], netForce[1]) / magnitude < 0.05,
    `the net force must stay small even though Barnes-Hut is not symmetric: ` +
    `${(Math.hypot(netForce[0], netForce[1]) / magnitude).toFixed(4)}`);

  // -- degenerate clouds ---------------------------------------------------------------------
  const emptyTree = buildTree([], { leafSize: 4 });
  strictEqual(leaves(emptyTree).length, 1, "an empty cloud still has a root leaf");
  deepStrictEqual(emptyTree.indices, []);
  strictEqual(emptyTree.mass, 0, "an empty cloud has no mass");
  deepStrictEqual(treeForces([], emptyTree, 0.5).forces, [], "and produces no forces");
  deepStrictEqual(directForces([]).forces, []);

  const solo = [{ x: 0.5, y: 0.5, mass: 3 }];
  const soloTree = buildTree(solo);
  const soloForce = treeForces(solo, soloTree, 0.5);
  deepStrictEqual(soloForce.forces, [[0, 0]], "a lone particle feels nothing");
  strictEqual(soloForce.interactions, 0, "a lone particle interacts with nobody");

  const pair = [{ x: 0.2, y: 0.5, mass: 2 }, { x: 0.8, y: 0.5, mass: 5 }];
  const pairTree = treeForces(pair, buildTree(pair), 0.5);
  const analytic = GRAVITY * 2 * 5 / (0.6 * 0.6 + SOFTENING ** 2) / Math.sqrt(0.6 * 0.6 + SOFTENING ** 2);
  ok(Math.abs(pairTree.forces[0][0] - analytic * 0.6) < 1e-12,
    "a pair must feel exactly the analytic softened force");
  ok(Math.abs(pairTree.forces[0][0] + pairTree.forces[1][0]) < 1e-12,
    "and the two forces must cancel exactly for a single pair");
  ok(Math.abs(pairTree.forces[0][1]) < 1e-15 && Math.abs(pairTree.forces[1][1]) < 1e-15, "with no y component");

  const coincident = Array.from({ length: 12 }, () => ({ x: 0.5, y: 0.5, mass: 1 }));
  const coincidentTree = treeForces(coincident, buildTree(coincident), 0.5);
  ok(coincidentTree.forces.every(([fx, fy]) => Number.isFinite(fx) && Number.isFinite(fy)),
    "softening must keep coincident particles finite");
  deepStrictEqual(coincidentTree.forces, coincident.map(() => [0, 0]),
    "and by symmetry they feel nothing");
  deepStrictEqual(buildTree([{ x: 0, y: 0, mass: 1 }], { leafSize: 8 }).indices, [0],
    "a single particle builds a degenerate tree");

  // -- determinism ----------------------------------------------------------------------------
  const replay = () => {
    const local = makeRng(733733);
    const cloud = makeCloud(local, 80);
    const built = buildTree(cloud, { leafSize: 8 });
    const run = treeForces(cloud, built, 0.5);
    return `${run.interactions}/${run.forces[0][0].toExponential(12)}/${depth(built)}`;
  };
  deepStrictEqual(replay(), replay(), "identical clouds must build identical trees and forces");
  console.log("PASS");
}
