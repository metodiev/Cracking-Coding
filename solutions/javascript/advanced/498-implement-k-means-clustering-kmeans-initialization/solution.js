/**
 * Task 498 - Implement k-means clustering (kmeans++ initialization). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * k-means over fixed 2-D blobs with a fully deterministic k-means++ seeding: the first centre is
 * an LCG-uniform data point, then each next centre is drawn with probability proportional to
 * D(x)^2 (the squared distance to the closest chosen centre; the draw is one fixed-LCG uniform
 * against the cumulative D^2 weights). Lloyd iterations then alternate exact assignment
 * (squared Euclidean distance, ties to the lowest cluster index) with mean updates (an empty
 * cluster keeps its old centre) until the labels stop changing. Data: three planted discs of
 * jittered points around (0,0), (4,0) and (0,4), 20 points each, jitter in 0.01 steps of
 * +/-0.5. Documented expectations: with 100 fixed random restarts (distinct-index sampling,
 * same LCG family) the k-means++ objective is at most the first restart's objective and within
 * 5% of the best restart (measured 10.613 vs 176.38 for the first restart and 10.613 for the
 * best of 100); every cluster is pure (all its points come from one planted blob); and
 * re-running with the same seed reproduces the labels and centroids exactly (bit for bit).
 */
"use strict";

const BLOBS = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
const PER_BLOB = 20;
const K = 3;
const SEED = 7;
const RESTARTS = 100;

function lcgPoints(seed) {
  const points = [];
  const truth = [];
  for (let blob = 0; blob < BLOBS.length; blob++) {
    const [cx, cy] = BLOBS[blob];
    for (let i = 0; i < PER_BLOB; i++) {
      seed = (seed * 75 + 74) % 65537;
      const x = cx + (seed % 101 - 50) / 100.0;
      seed = (seed * 75 + 74) % 65537;
      const y = cy + (seed % 101 - 50) / 100.0;
      points.push([x, y]);
      truth.push(blob);
    }
  }
  return [points, truth];
}

function sqdist(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

function kmeansPlusPlus(points, k, seed) {
  const n = points.length;
  seed = (seed * 75 + 74) % 65537;
  const first = seed % n;
  const centers = [points[first]];
  let d2 = points.map((p) => sqdist(p, centers[0]));
  while (centers.length < k) {
    const total = d2.reduce((a, b) => a + b, 0);
    seed = (seed * 75 + 74) % 65537;
    const target = ((seed + 0.5) / 65537.0) * total;
    let acc = 0.0;
    let choice = n - 1;
    for (let i = 0; i < n; i++) {
      acc += d2[i];
      if (acc >= target) {
        choice = i;
        break;
      }
    }
    centers.push(points[choice]);
    d2 = d2.map((value, i) => Math.min(value, sqdist(points[i], centers[centers.length - 1])));
  }
  return centers;
}

/** Deterministic distinct-index sampling from the same LCG family. */
function randomInit(points, k, seed) {
  const n = points.length;
  const chosen = [];
  while (chosen.length < k) {
    seed = (seed * 75 + 74) % 65537;
    const index = seed % n;
    if (!chosen.includes(index)) chosen.push(index);
  }
  return chosen.map((index) => points[index]);
}

function assign(point, centers) {
  let best = 0;
  let bestDistance = sqdist(point, centers[0]);
  for (let c = 1; c < centers.length; c++) {
    const distance = sqdist(point, centers[c]);
    if (distance < bestDistance) {
      best = c;
      bestDistance = distance;
    }
  }
  return best;
}

function lloyd(points, centers, maxIter = 100) {
  let labels = null;
  for (let iter = 0; iter < maxIter; iter++) {
    const newLabels = points.map((p) => assign(p, centers));
    const newCenters = [];
    for (let c = 0; c < centers.length; c++) {
      const members = points.filter((_, i) => newLabels[i] === c);
      if (members.length > 0) {
        newCenters.push([
          members.reduce((a, p) => a + p[0], 0.0) / members.length,
          members.reduce((a, p) => a + p[1], 0.0) / members.length,
        ]);
      } else {
        newCenters.push(centers[c].slice());          // empty cluster keeps its centre
      }
    }
    const same = labels !== null && labels.every((value, i) => value === newLabels[i]);
    labels = newLabels;
    centers = newCenters;
    if (same) break;
  }
  return [labels, centers];
}

function kmeans(points, k, seed) {
  return lloyd(points, kmeansPlusPlus(points, k, seed));
}

function objective(points, centers) {
  let total = 0.0;
  for (const p of points) {
    let best = Infinity;
    for (const c of centers) best = Math.min(best, sqdist(p, c));
    total += best;
  }
  return total;
}

function main() {
  const assert = require("node:assert");

  const [points, truth] = lcgPoints(SEED);
  assert.strictEqual(points.length, 3 * PER_BLOB);
  assert.strictEqual(new Set(points.map((p) => p.join(","))).size, points.length,
    "jittered blobs are duplicate-free");

  const [labels, centers] = kmeans(points, K, SEED + 1);
  assert.deepStrictEqual([...new Set(labels)].sort(), [0, 1, 2], "three non-empty clusters");

  const purity = new Map();
  labels.forEach((label, i) => {
    if (!purity.has(label)) purity.set(label, new Set());
    purity.get(label).add(truth[i]);
  });
  for (const blobs of purity.values()) assert.strictEqual(blobs.size, 1, "clusters must be pure");
  assert.strictEqual(new Set([...purity.values()].map((blobs) => [...blobs][0])).size, 3,
    "one blob per cluster");

  const kppObjective = objective(points, centers);

  const [firstRandomLabels, firstRandomCenters] = lloyd(points, randomInit(points, K, 9100));
  assert.ok(firstRandomLabels.length === points.length);
  const firstRandom = objective(points, firstRandomCenters);
  assert.ok(kppObjective <= firstRandom, "k-means++ must beat the first random init");
  let bestRandom = Infinity;
  for (let r = 0; r < RESTARTS; r++) {
    const [, centersR] = lloyd(points, randomInit(points, K, 1000 + 17 * r));
    bestRandom = Math.min(bestRandom, objective(points, centersR));
  }
  assert.ok(kppObjective <= 1.05 * bestRandom, "within 5% of the best of 100 restarts");

  const [repeatLabels, repeatCenters] = kmeans(points, K, SEED + 1);
  assert.deepStrictEqual(repeatLabels, labels, "labels must be stable under re-run");
  assert.deepStrictEqual(repeatCenters, centers, "centroids must be identical bit for bit");

  const centersPp = kmeansPlusPlus(points, K, SEED + 1);
  for (const c of centersPp) {
    assert.ok(points.some((p) => p[0] === c[0] && p[1] === c[1]), "centres are data points");
  }
  assert.strictEqual(new Set(centersPp.map((c) => c.join(","))).size, K, "centres are distinct");
  assert.strictEqual(kmeansPlusPlus(points, points.length, SEED + 1).length, points.length);

  const tiny = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
                [3.0, 3.0], [3.1, 3.0], [3.0, 3.1], [3.1, 3.1]];
  const [tinyLabels] = kmeans(tiny, 2, 55);
  assert.deepStrictEqual(tinyLabels.slice(0, 4), new Array(4).fill(tinyLabels[0]));
  assert.deepStrictEqual(tinyLabels.slice(4), new Array(4).fill(tinyLabels[4]));
  assert.notStrictEqual(tinyLabels[0], tinyLabels[4]);
}

module.exports = { BLOBS, PER_BLOB, K, SEED, lcgPoints, sqdist, kmeansPlusPlus, randomInit,
  assign, lloyd, kmeans, objective };

if (require.main === module) {
  main();
  console.log("PASS");
}
