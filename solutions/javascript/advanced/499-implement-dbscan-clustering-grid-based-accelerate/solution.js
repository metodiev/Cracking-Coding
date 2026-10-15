/**
 * Task 499 - Implement DBSCAN clustering (grid-based accelerate). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Density-based clustering (eps-neighbourhoods count points with squared distance <= eps^2,
 * minPts core threshold) with two interchangeable neighbourhood engines: the naive one scans
 * all n points, the accelerated one hashes points into a grid of cell size eps and probes only
 * the 3x3 block of cells around a query - every point within eps must lie in that block, so the
 * results are identical while far fewer distances are evaluated (the self-test counts both and
 * asserts the grid strictly wins). Clusters are grown by the textbook queue expansion in index
 * order, which fixes the processing order of ambiguous border points, so the two engines must
 * produce element-wise identical labels (noise = -1); the comparison is also made
 * order-insensitively (sets of point-index sets per cluster plus the noise set). Test sets are
 * LCG-generated 2-D integer points, including duplicate-heavy sets, a mostly-noise sparse set,
 * a hand-checked chain, and points exactly eps apart (the <= boundary is inclusive).
 */
"use strict";

const NOISE = -1;

function sqdist(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

function lcgPoints(n, seed, limit = 40) {
  const points = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    const x = seed % limit;
    seed = (seed * 75 + 74) % 65537;
    const y = seed % limit;
    points.push([x, y]);
  }
  return points;
}

class NaiveEngine {
  constructor(points, eps) {
    this.points = points;
    this.eps2 = eps * eps;
    this.checks = 0;
  }

  region(index) {
    const out = [];
    const [px, py] = this.points[index];
    for (let j = 0; j < this.points.length; j++) {
      this.checks += 1;
      const dx = px - this.points[j][0];
      const dy = py - this.points[j][1];
      if (dx * dx + dy * dy <= this.eps2) out.push(j);
    }
    return out;
  }
}

class GridEngine {
  constructor(points, eps) {
    this.points = points;
    this.eps2 = eps * eps;
    this.eps = eps;
    this.cells = new Map();
    points.forEach(([x, y], j) => {
      const key = `${Math.floor(x / eps)},${Math.floor(y / eps)}`;
      if (!this.cells.has(key)) this.cells.set(key, []);
      this.cells.get(key).push(j);
    });
    this.checks = 0;
  }

  region(index) {
    const [px, py] = this.points[index];
    const cx = Math.floor(px / this.eps);
    const cy = Math.floor(py / this.eps);
    const out = [];
    for (let gx = cx - 1; gx <= cx + 1; gx++) {
      for (let gy = cy - 1; gy <= cy + 1; gy++) {
        const bucket = this.cells.get(`${gx},${gy}`);
        if (bucket === undefined) continue;
        for (const j of bucket) {
          this.checks += 1;
          const dx = px - this.points[j][0];
          const dy = py - this.points[j][1];
          if (dx * dx + dy * dy <= this.eps2) out.push(j);
        }
      }
    }
    out.sort((a, b) => a - b);
    return out;
  }
}

function dbscan(points, eps, minPts, engine) {
  const n = points.length;
  const labels = new Array(n).fill(NOISE);
  const visited = new Array(n).fill(false);
  let cluster = 0;
  for (let start = 0; start < n; start++) {
    if (visited[start]) continue;
    visited[start] = true;
    const neighbours = engine.region(start);
    if (neighbours.length < minPts) continue;  // provisional noise; may become a border point
    labels[start] = cluster;
    const queue = neighbours.filter((q) => q !== start);
    for (let head = 0; head < queue.length; head++) {
      const current = queue[head];
      if (!visited[current]) {
        visited[current] = true;
        const currentNeighbours = engine.region(current);
        if (currentNeighbours.length >= minPts) {
          for (const q of currentNeighbours) {
            if (!visited[q]) queue.push(q);
          }
        }
      }
      if (labels[current] === NOISE) labels[current] = cluster;   // border point
    }
    cluster += 1;
  }
  return labels;
}

/** Renumber clusters by first appearance so two runs compare label-for-label. */
function canonical(labels) {
  const mapping = new Map();
  return labels.map((label) => {
    if (label === NOISE) return NOISE;
    if (!mapping.has(label)) mapping.set(label, mapping.size);
    return mapping.get(label);
  });
}

/** Order-insensitive partition: per-cluster index sets (sorted keys) plus the noise set. */
function partition(labels) {
  const clusters = new Map();
  const noise = [];
  labels.forEach((label, i) => {
    if (label === NOISE) {
      noise.push(i);
    } else {
      if (!clusters.has(label)) clusters.set(label, []);
      clusters.get(label).push(i);
    }
  });
  const parts = [...clusters.values()].map((members) => members.join(",")).sort();
  return `${parts.join("|")}#${noise.join(",")}`;
}

function compareSet(points, eps, minPts, label) {
  const naive = new NaiveEngine(points, eps);
  const grid = new GridEngine(points, eps);
  const naiveLabels = dbscan(points, eps, minPts, naive);
  const gridLabels = dbscan(points, eps, minPts, grid);
  const assert = require("node:assert");
  assert.deepStrictEqual(canonical(naiveLabels), canonical(gridLabels),
    `${label}: engines disagree label-for-label`);
  assert.strictEqual(partition(naiveLabels), partition(gridLabels), `${label}: partitions`);
  assert.ok(grid.checks < naive.checks, `${label}: grid must evaluate fewer distances`);
  return [naiveLabels, naive.checks, grid.checks];
}

function main() {
  const assert = require("node:assert");

  const chain = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 10.0]];
  assert.deepStrictEqual(dbscan(chain, 1.2, 2, new NaiveEngine(chain, 1.2)), [0, 0, 0, NOISE]);
  assert.deepStrictEqual(dbscan(chain, 1.2, 3, new NaiveEngine(chain, 1.2)), [0, 0, 0, NOISE],
    "all three points are core at minPts = 3");

  const pair = [[0.0, 0.0], [2.0, 0.0], [20.0, 20.0]];
  assert.deepStrictEqual(dbscan(pair, 2.0, 2, new NaiveEngine(pair, 2.0)), [0, 0, NOISE]);
  assert.deepStrictEqual(dbscan(pair, 1.999, 2, new NaiveEngine(pair, 1.999)),
    [NOISE, NOISE, NOISE], "just inside eps everything is noise");

  const sets = [
    ["dense", lcgPoints(60, 11), 3.0, 3],
    ["fine", lcgPoints(80, 23), 2.0, 2],
    ["coarse", lcgPoints(50, 37), 5.0, 4],
    ["sparse", lcgPoints(40, 53, 100), 1.5, 5],
    ["duplicates", lcgPoints(30, 67, 10), 2.0, 3],
  ];
  const report = [];
  for (const [name, points, eps, minPts] of sets) {
    const [labels, naiveChecks, gridChecks] = compareSet(points, eps, minPts, name);
    const clusterCount = new Set(labels.filter((label) => label !== NOISE)).size;
    const noiseCount = labels.filter((label) => label === NOISE).length;
    report.push([name, points.length, eps, minPts, clusterCount, noiseCount, naiveChecks,
      gridChecks]);
  }

  const dense = report[0];
  assert.ok(dense[4] >= 1 && dense[4] <= 12, "dense set: a handful of clusters");
  assert.ok(dense[5] >= 0 && dense[5] < dense[1], "dense set: not everything is noise");
  for (const [name, , , , , , naiveChecks, gridChecks] of report) {
    assert.ok(gridChecks <= Math.floor(naiveChecks / 2), `${name}: the grid saves at least half`);
  }

  const lonely = lcgPoints(12, 5, 1000);
  assert.deepStrictEqual([...new Set(dbscan(lonely, 1.0, 3, new GridEngine(lonely, 1.0)))],
    [NOISE], "scattered points are all noise");
}

module.exports = { NOISE, sqdist, lcgPoints, NaiveEngine, GridEngine, dbscan, canonical,
  partition, compareSet };

if (require.main === module) {
  main();
  console.log("PASS");
}
