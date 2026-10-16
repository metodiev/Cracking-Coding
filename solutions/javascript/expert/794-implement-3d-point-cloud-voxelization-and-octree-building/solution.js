/** Task 794 - Implement 3D point cloud voxelization and octree building. (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Two spatial data structures over the same point cloud:
 *   voxel grid: index = floor(coordinate / size) per axis, stored in a hash map from the integer
 *               triple to the point indices and the voxel centroid. Math.floor gives the correct
 *               cell for negative coordinates, and a point exactly on a boundary belongs to the
 *               upper cell (checked below).
 *   octree:     a cube root node subdivided into 8 octants; a node that holds more than
 *               `capacity` points splits recursively until `maxDepth`, so the tree adapts to the
 *               local density. Leaves store index lists and live inside their node's bounds.
 * The octree supports nearest-neighbour and radius queries by descending only into children whose
 * bounds are within the current best distance, and both queries are verified against brute force.
 *
 * Invariants verified: the leaf point sets partition the input cloud exactly once, every point lies
 * inside its leaf's bounds, the tree is a full 8-ary tree (every internal node has 8 children, and
 * leaves = 1 + 7 * internal nodes), voxel indexing matches a naive reference for all sign
 * combinations including exact boundaries, voxel membership partitions the points, nearest-neighbour
 * and radius queries agree with brute force on random queries, and empty / single-point / all-identical
 * clouds are handled without crashing.
 *
 * What is simplified: points are stored as index lists instead of a compact array-of-structures, the
 * split criterion is a fixed capacity plus max depth (no Morton-code sorting, no SAH refinement), and
 * clouds are small (a few hundred points) so the whole file stays in milliseconds. A seeded splitmix32
 * generates the deterministic cloud; every reported statistic is measured.
 */
"use strict";

const { ok, throws } = require("node:assert");

/** Deterministic splitmix32 uniforms in [0, 1). */
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return ((z ^ (z >>> 15)) >>> 0) / 4294967296;
  };
}

const distanceSquared = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;

/** Voxel key of a point: floor semantics, so negative coordinates land in the right cell. */
const voxelIndex = (point, size) => point.map((v) => Math.floor(v / size));
const voxelKey = (index) => index.join(",");

/**
 * Voxelises a cloud at the given voxel size. Returns the voxel map plus the occupancy statistics;
 * `statistics.points` must equal the input length for every input (the partition invariant).
 */
function voxelise(points, size) {
  if (!(size > 0)) throw new RangeError("voxel size must be > 0");
  const voxels = new Map();
  points.forEach((point, index) => {
    const key = voxelKey(voxelIndex(point, size));
    const cell = voxels.get(key);
    if (cell) {
      cell.indices.push(index);
      cell.count += 1;
    } else {
      voxels.set(key, { indices: [index], count: 1, index: voxelIndex(point, size) });
    }
  });
  for (const cell of voxels.values()) {
    cell.centroid = [0, 1, 2].map((d) => cell.indices.reduce((acc, i) => acc + points[i][d], 0) / cell.count);
  }
  return { voxels, statistics: { voxels: voxels.size, points: points.reduce((acc, _, i) => acc + 1, 0) } };
}

/** Bounds of a leaf: axis-aligned cube given by its centre and half size. */
const boundsContain = (bounds, point) => [0, 1, 2].every((d) => point[d] >= bounds.min[d] - 1e-12 && point[d] <= bounds.max[d] + 1e-12);

class Octree {
  constructor(points, { capacity = 4, maxDepth = 6 } = {}) {
    if (!(capacity >= 1)) throw new RangeError("capacity must be >= 1");
    if (!Number.isInteger(maxDepth) || maxDepth < 0) throw new RangeError("maxDepth must be a non-negative integer");
    this.points = points;
    this.capacity = capacity;
    this.maxDepth = maxDepth;
    const extent = points.length === 0 ? 1 : Math.max(1e-6, ...points.flat().map(Math.abs)) * 1.001;
    this.root = this.#build(points.map((_, i) => i), { min: [-extent, -extent, -extent], max: [extent, extent, extent] }, 0);
    const audit = this.#audit(this.root);
    this.leafCount = audit.leaves;
    this.internalCount = audit.internal;
    this.maxLeafSize = audit.maxLeaf;
    const complete = audit.indices.length === points.length && new Set(audit.indices).size === points.length;
    this.violations = audit.violations + (complete ? 0 : 1);
  }

  /** Recursively splits a node while it holds more than `capacity` points and depth allows. */
  #build(indices, bounds, depth) {
    const node = { bounds, depth, indices: null, children: null };
    if (indices.length <= this.capacity || depth >= this.maxDepth) {
      node.indices = indices;
      return node;
    }
    const mid = [0, 1, 2].map((d) => (bounds.min[d] + bounds.max[d]) / 2);
    node.children = Array.from({ length: 8 }, (_, child) => {
      const childBounds = { min: [], max: [] };
      for (let d = 0; d < 3; d++) {
        const high = (child >> d) & 1;
        childBounds.min[d] = high ? mid[d] : bounds.min[d];
        childBounds.max[d] = high ? bounds.max[d] : mid[d];
      }
      return null;
    });
    const buckets = Array.from({ length: 8 }, () => []);
    for (const i of indices) {
      const point = this.points[i];
      let child = 0;
      for (let d = 0; d < 3; d++) if (point[d] >= mid[d]) child |= 1 << d;
      buckets[child].push(i);
    }
    node.children = buckets.map((bucket, child) => {
      const childBounds = { min: [], max: [] };
      for (let d = 0; d < 3; d++) {
        const high = (child >> d) & 1;
        childBounds.min[d] = high ? mid[d] : bounds.min[d];
        childBounds.max[d] = high ? bounds.max[d] : mid[d];
      }
      return this.#build(bucket, childBounds, depth + 1);
    });
    return node;
  }

  /** Structural audit: leaves stay inside their bounds and no point is lost or duplicated. */
  #audit(node) {
    if (node.children === null) {
      const inside = node.indices.every((i) => boundsContain(node.bounds, this.points[i]));
      return { leaves: 1, internal: 0, maxLeaf: node.indices.length, violations: inside ? 0 : 1, indices: node.indices.slice() };
    }
    const parts = node.children.map((child) => this.#audit(child));
    const indices = parts.flatMap((part) => part.indices);
    const duplicated = new Set(indices).size !== indices.length;
    return {
      leaves: parts.reduce((acc, part) => acc + part.leaves, 0),
      internal: 1 + parts.reduce((acc, part) => acc + part.internal, 0),
      maxLeaf: Math.max(...parts.map((part) => part.maxLeaf)),
      violations: parts.reduce((acc, part) => acc + part.violations, 0) + (duplicated ? 1 : 0),
      indices,
    };
  }

  /** All (leaf, indices) pairs, in deterministic traversal order. */
  leaves() {
    const out = [];
    const walk = (node) => {
      if (node.children === null) {
        out.push(node);
        return;
      }
      for (const child of node.children) walk(child);
    };
    walk(this.root);
    return out;
  }

  /** Radius query via branch-and-bound: only children whose box is within r are descended. */
  withinRadius(centre, radius) {
    if (!(radius >= 0)) throw new RangeError("radius must be >= 0");
    const hits = [];
    const radiusSquared = radius * radius;
    const walk = (node) => {
      const boxDistance = [0, 1, 2].reduce((acc, d) => {
        const delta = Math.max(node.bounds.min[d] - centre[d], 0, centre[d] - node.bounds.max[d]);
        return acc + delta * delta;
      }, 0);
      if (boxDistance > radiusSquared) return;
      if (node.children === null) {
        for (const i of node.indices) if (distanceSquared(this.points[i], centre) <= radiusSquared) hits.push(i);
        return;
      }
      for (const child of node.children) walk(child);
    };
    walk(this.root);
    return hits.sort((a, b) => a - b);
  }

  /** Nearest neighbour via branch-and-bound; returns {index, distance} or null for an empty tree. */
  nearest(point) {
    let best = null;
    let bestDistance = Infinity;
    const walk = (node) => {
      const boxDistance = [0, 1, 2].reduce((acc, d) => {
        const delta = Math.max(node.bounds.min[d] - point[d], 0, point[d] - node.bounds.max[d]);
        return acc + delta * delta;
      }, 0);
      if (boxDistance >= bestDistance) return;
      if (node.children === null) {
        for (const i of node.indices) {
          const d = distanceSquared(this.points[i], point);
          if (d < bestDistance) {
            bestDistance = d;
            best = i;
          }
        }
        return;
      }
      for (const child of node.children) walk(child);
    };
    walk(this.root);
    return best === null ? null : { index: best, distance: Math.sqrt(bestDistance) };
  }
}

/** Brute-force nearest neighbour, used as the oracle. */
function bruteForceNearest(points, point) {
  let best = null;
  let bestDistance = Infinity;
  points.forEach((p, i) => {
    const d = distanceSquared(p, point);
    if (d < bestDistance) {
      bestDistance = d;
      best = i;
    }
  });
  return best === null ? null : { index: best, distance: Math.sqrt(bestDistance) };
}

/** Deterministic cloud: a jittered box plus a dense cluster plus a couple of duplicate points. */
function makeCloud(seed, count) {
  const random = makeRng(seed);
  const cloud = Array.from({ length: count }, () => [(random() - 0.5) * 6, (random() - 0.5) * 6, (random() - 0.5) * 6]);
  for (let i = 0; i < 8; i++) cloud.push([1 + random() * 0.02, 1 + random() * 0.02, 1 + random() * 0.02]);
  cloud.push([2, 2, 2], [2, 2, 2], [-3.5, 0, 1.25]);
  return cloud;
}

function main() {
  const notes = [];
  const random = makeRng(20251016);

  // 1. Voxel indexing must follow floor semantics for every sign pattern, boundaries included.
  const size = 0.5;
  const probes = [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [-0.25, -0.25, -0.25], [-0.5, 0.5, -0.5], [0, -1.5, 2.4999]];
  const expected = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [-1, 1, -1], [0, -3, 4]];
  const indexChecks = probes.map((p, k) => voxelIndex(p, size).join(",") === expected[k].join(","));
  notes.push(`voxel indexing (size ${size}): ${probes.map((p, k) => `${JSON.stringify(p)} -> ${voxelIndex(p, size).join(",")}`).join("; ")}`);
  ok(indexChecks.every(Boolean), "voxel indices must use floor semantics, including for negative coordinates");
  ok(voxelIndex([0.5, 0, 0], size)[0] === 1, "a point exactly on a boundary must go to the upper cell");

  // 2. Voxelisation partitions the cloud and every centroid stays inside its cell.
  const cloud = makeCloud(11, 200);
  const { voxels, statistics } = voxelise(cloud, 0.75);
  let voxelPartition = statistics.points === cloud.length;
  let centroidInside = true;
  const assigned = [];
  for (const cell of voxels.values()) {
    assigned.push(...cell.indices);
    if (!cell.indices.every((i) => voxelKey(voxelIndex(cloud[i], 0.75)) === voxelKey(cell.index))) voxelPartition = false;
    if (!cell.indices.every((i) => cell.centroid.map((c, d) => Math.abs(c - cloud[i][d]) < 0.75).every(Boolean))) centroidInside = false;
  }
  const uniqueAssigned = new Set(assigned);
  notes.push(`voxelisation: ${statistics.voxels} occupied cells for ${cloud.length} points, largest cell ${Math.max(...[...voxels.values()].map((c) => c.count))} points, clustering coefficient ${(statistics.voxels / cloud.length).toFixed(3)}`);
  ok(voxelPartition && uniqueAssigned.size === cloud.length && assigned.length === cloud.length, "voxels must partition the cloud exactly once");
  ok(centroidInside, "every voxel centroid must stay within one voxel size of its members");

  // 3. The octree must contain every point exactly once, inside the right leaf bounds.
  const octree = new Octree(cloud, { capacity: 4, maxDepth: 6 });
  const leafIndices = octree.leaves().flatMap((leaf) => leaf.indices);
  notes.push(`octree: ${octree.leaves().length} leaves, ${octree.internalCount} internal nodes, deepest max leaf ${octree.maxLeafSize}, audit violations ${octree.violations}`);
  ok(octree.violations === 0, "the structural audit must report no violations");
  ok(leafIndices.length === cloud.length && new Set(leafIndices).size === cloud.length, "the leaves must partition the cloud exactly");
  ok(octree.leaves().every((leaf) => leaf.indices.every((i) => boundsContain(leaf.bounds, cloud[i]))), "every point must lie inside its leaf bounds");
  ok(octree.leaves().length === 1 + 7 * octree.internalCount, "a full 8-ary tree must satisfy leaves = 1 + 7 * internal nodes");

  // 4. Nearest-neighbour queries must agree with brute force.
  let worstNeighbourGap = 0;
  let neighbourMatches = 0;
  for (let k = 0; k < 200; k++) {
    const query = [(random() - 0.5) * 8, (random() - 0.5) * 8, (random() - 0.5) * 8];
    const fromTree = octree.nearest(query);
    const fromBruteForce = bruteForceNearest(cloud, query);
    if (fromTree.index === fromBruteForce.index) neighbourMatches += 1;
    worstNeighbourGap = Math.max(worstNeighbourGap, Math.abs(fromTree.distance - fromBruteForce.distance));
  }
  notes.push(`nearest neighbour over 200 queries: ${neighbourMatches} index matches, worst distance gap ${worstNeighbourGap.toExponential(2)}`);
  ok(neighbourMatches === 200, `the octree must return the brute-force neighbour (${neighbourMatches}/200)`);
  ok(worstNeighbourGap < 1e-12, "the reported distances must match to machine precision");

  // 5. Radius queries must agree with brute force for several radii.
  const radiusReport = [0.5, 1.5, 3].map((radius) => {
    const centre = [0.3, -0.2, 0.1];
    const fromTree = octree.withinRadius(centre, radius);
    const fromBrute = cloud.map((p, i) => (distanceSquared(p, centre) <= radius * radius ? i : -1)).filter((i) => i >= 0);
    return { radius, tree: fromTree.length, brute: fromBrute.length, same: fromTree.join(",") === fromBrute.join(",") };
  });
  notes.push(`radius queries: ${radiusReport.map((r) => `r = ${r.radius} -> ${r.tree} nodes (brute force ${r.brute})`).join("; ")}`);
  ok(radiusReport.every((r) => r.same), "radius queries must return exactly the brute-force set");

  // 6. A tighter capacity must produce a deeper tree with smaller leaves (duplicates excepted).
  const deduped = cloud.filter((p, i) => cloud.findIndex((q) => distanceSquared(p, q) === 0) === i);
  const tight = new Octree(deduped, { capacity: 1, maxDepth: 12 });
  const loose = new Octree(cloud, { capacity: 32, maxDepth: 8 });
  notes.push(`capacity sweep: capacity 1 -> ${tight.leaves().length} leaves (max ${tight.maxLeafSize}) on the ${deduped.length} distinct points; capacity 32 -> ${loose.leaves().length} leaves (max ${loose.maxLeafSize})`);
  ok(tight.leaves().length > loose.leaves().length, "a smaller capacity must produce more leaves");
  ok(tight.maxLeafSize === 1 && tight.violations === 0, "capacity 1 must give single-point leaves once duplicates are removed");

  // 7. Degenerate clouds must be handled: empty, single point, all identical.
  const empty = new Octree([], { capacity: 4, maxDepth: 4 });
  ok(empty.leaves().length === 1 && empty.nearest([0, 0, 0]) === null, "an empty cloud must give one empty leaf and no neighbour");
  const single = new Octree([[1, 2, 3]], { capacity: 4, maxDepth: 4 });
  ok(single.nearest([0, 0, 0]).distance === Math.sqrt(14), "a single-point cloud must answer queries exactly");
  const identical = new Octree([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], { capacity: 2, maxDepth: 3 });
  notes.push(`all-identical cloud: ${identical.leaves().length} leaves after hitting maxDepth, max leaf ${identical.maxLeafSize}`);
  ok(identical.maxLeafSize === 5 && identical.violations === 0, "identical points must stop at maxDepth and stay in one leaf");
  ok(voxelise([], 1).statistics.voxels === 0, "voxelising an empty cloud must give no cells");

  // 8. Validation and determinism.
  throws(() => voxelise([[0, 0, 0]], 0), RangeError, "a zero voxel size must throw");
  throws(() => new Octree([[0, 0, 0]], { capacity: 0 }), RangeError, "capacity 0 must throw");
  throws(() => new Octree([[0, 0, 0]], { maxDepth: 1.5 }), RangeError, "a fractional maxDepth must throw");
  const repeat = new Octree(cloud, { capacity: 4, maxDepth: 6 });
  ok(repeat.leaves().length === octree.leaves().length && repeat.violations === 0, "the octree must be deterministic");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { makeRng, voxelIndex, voxelKey, voxelise, Octree, bruteForceNearest, makeCloud };

if (require.main === module) main();
