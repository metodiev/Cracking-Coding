/** Task 460 - Implement small epsilon-approximate nearest neighbor in high-dim (LSH idea). (Tier: Advanced, CTCI Beyond) */
"use strict";

const DIM = 10;
const TABLES = 6;
const BUCKET_WIDTH = 700;
const NUM_POINTS = 40;

let seed = 4601;

function lcg() {
  seed = (seed * 75 + 74) % 65537;
  return seed;
}

function makePoints(count) {
  const points = [];
  for (let p = 0; p < count; p++) {
    const coords = [];
    for (let j = 0; j < DIM; j++) coords.push((lcg() % 61) - 30);
    points.push(coords);
  }
  return points;
}

function makeTablesFromSeed(tableSeed) {
  const tables = [];
  for (let t = 0; t < TABLES; t++) {
    const projection = [];
    for (let j = 0; j < DIM; j++) {
      tableSeed = (tableSeed * 75 + 74) % 65537;
      projection.push(1 + (tableSeed % 20));
    }
    tableSeed = (tableSeed * 75 + 74) % 65537;
    const offset = tableSeed % BUCKET_WIDTH;
    tables.push({ projection, offset, buckets: new Map() });
  }
  return tables;
}

function makeTables() {
  const tableSeed = seed;
  for (let i = 0; i < TABLES * (DIM + 1); i++) lcg();
  return makeTablesFromSeed(tableSeed);
}

function hashOf(table, point) {
  let total = table.offset;
  for (let j = 0; j < DIM; j++) total += table.projection[j] * point[j];
  return Math.floor(total / BUCKET_WIDTH);
}

function buildIndex(points, tables) {
  for (const table of tables) {
    for (let idx = 0; idx < points.length; idx++) {
      const h = hashOf(table, points[idx]);
      if (!table.buckets.has(h)) table.buckets.set(h, []);
      table.buckets.get(h).push(idx);
    }
  }
}

function squaredDistance(p, q) {
  let total = 0;
  for (let j = 0; j < p.length; j++) {
    const diff = p[j] - q[j];
    total += diff * diff;
  }
  return total;
}

/** Exact nearest neighbour (the epsilon-oracle). Returns [index, squared dist]. */
function bruteForceNn(points, query) {
  let bestIdx = 0;
  let bestDist = squaredDistance(points[0], query);
  for (let idx = 1; idx < points.length; idx++) {
    const dist = squaredDistance(points[idx], query);
    if (dist < bestDist) {
      bestIdx = idx;
      bestDist = dist;
    }
  }
  return [bestIdx, bestDist];
}

/** Returns [index, squared dist, usedBucket]; falls back to exact when all buckets empty. */
function lshApproxNn(points, tables, query) {
  const candidates = new Set();
  for (const table of tables) {
    const bucket = table.buckets.get(hashOf(table, query));
    if (bucket !== undefined) for (const idx of bucket) candidates.add(idx);
  }
  if (candidates.size === 0) {
    const [idx, dist] = bruteForceNn(points, query);
    return [idx, dist, false];
  }
  let bestIdx = -1;
  let bestDist = Infinity;
  for (const idx of Array.from(candidates).sort((a, b) => a - b)) {
    const dist = squaredDistance(points[idx], query);
    if (dist < bestDist) {
      bestIdx = idx;
      bestDist = dist;
    }
  }
  return [bestIdx, bestDist, true];
}

function checkQuery(assert, points, tables, query, counters, label) {
  const [trueIdx, trueDist] = bruteForceNn(points, query);
  const [approxIdx, approxDist, usedBucket] = lshApproxNn(points, tables, query);
  assert.ok(approxDist >= trueDist, `${label} never closer than true NN`);
  assert.ok(4 * approxDist <= 9 * trueDist, `${label} epsilon bound`);
  assert.strictEqual(squaredDistance(points[approxIdx], query), approxDist, `${label} distance matches`);
  counters[usedBucket ? "bucket" : "fallback"]++;
  return [trueIdx, trueDist, approxIdx, approxDist, usedBucket];
}

module.exports = { makePoints, makeTables, buildIndex, bruteForceNn, lshApproxNn, squaredDistance };

if (require.main === module) {
  const assert = require("node:assert");
  const points = makePoints(NUM_POINTS);
  const tables = makeTables();
  buildIndex(points, tables);

  let bucketCount = 0;
  let maxBucket = 0;
  for (const table of tables) {
    for (const bucket of table.buckets.values()) {
      bucketCount++;
      maxBucket = Math.max(maxBucket, bucket.length);
    }
  }
  assert.ok(bucketCount > 0, "buckets exist");
  assert.ok(maxBucket >= 2, "collisions exist");

  const counters = { bucket: 0, fallback: 0 };
  for (let q = 0; q < 12; q++) {
    const query = [];
    for (let j = 0; j < DIM; j++) query.push((lcg() % 61) - 30);
    checkQuery(assert, points, tables, query, counters, `LCG query ${q}`);
  }

  const outlier = new Array(DIM).fill(10000);
  const result = checkQuery(assert, points, tables, outlier, counters, "outlier");
  assert.strictEqual(result[4], false, "outlier falls back");
  assert.ok(result[2] === result[0] && result[3] === result[1], "fallback is exact");

  const handPoints = [new Array(DIM).fill(0), new Array(DIM).fill(0), new Array(DIM).fill(0)];
  handPoints[1][0] = 10;
  handPoints[2][1] = 10;
  const handTables = makeTablesFromSeed(99999);
  buildIndex(handPoints, handTables);
  const handQuery = new Array(DIM).fill(0);
  handQuery[0] = 9;
  const [handIdx, handDist] = bruteForceNn(handPoints, handQuery);
  assert.ok(handIdx === 1 && handDist === 1, "hand true NN");
  const [approxIdx, approxDist] = lshApproxNn(handPoints, handTables, handQuery);
  assert.ok(approxDist >= 1, "hand never closer");
  assert.ok(4 * approxDist <= 9 * 1, "hand epsilon bound");
  assert.strictEqual(squaredDistance(handPoints[approxIdx], handQuery), approxDist, "hand distance matches");

  assert.ok(counters.bucket >= 1, "bucket path exercised");
  assert.ok(counters.fallback >= 1, "fallback path exercised");
  console.log("PASS");
}
