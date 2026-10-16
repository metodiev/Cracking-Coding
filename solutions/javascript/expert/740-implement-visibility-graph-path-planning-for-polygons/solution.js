/** Task 740 - Implement visibility graph path planning for polygons.
 *  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Obstacles are simple polygons; free space is everything that is not strictly inside one of
 * them, so a route may slide along a wall but not cut a corner. The visibility graph has one
 * node per polygon vertex plus the start and the goal, and an edge between two nodes when the
 * open segment between them is unobstructed: no proper (interior) crossing with any polygon
 * edge, no polygon vertex in the interior of the segment, and the midpoint outside every
 * polygon - the midpoint test is what rejects a segment that lies wholly inside an obstacle and
 * therefore crosses no edge. Dijkstra on that graph yields the shortest polygonal route, which
 * is taut by construction (every bend is an obstacle vertex), so no shortcutting is needed.
 *
 * What is simplified: the graph is rebuilt per query instead of being maintained incrementally,
 * the O(V^2) visibility loop is not accelerated with a rotational sweep (documented; vertex
 * counts here are small), and vertices are never inspected in a triangulation-aware order. What
 * is verified: the predicate is cross-checked against an independent dense sampler in its sound
 * direction (a sampled strictly-interior point proves obstruction) and against analytic
 * single-obstacle distances; the returned route is taut (a greedy shortcut pass cannot remove a
 * single vertex), every edge survives both the exact predicate and the sampler, the cost lies
 * between the Euclidean bound and an independent grid-Dijkstra route, and unreachable or
 * invalid queries behave as specified.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

const EPS = 1e-9;
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
const sign = (value) => (value > EPS ? 1 : value < -EPS ? -1 : 0);

function onSegment(a, b, p) {
  return sign(cross(a, b, p)) === 0 &&
    Math.min(a[0], b[0]) - EPS <= p[0] && p[0] <= Math.max(a[0], b[0]) + EPS &&
    Math.min(a[1], b[1]) - EPS <= p[1] && p[1] <= Math.max(a[1], b[1]) + EPS;
}

/** The open segment (a, b) and the open segment (c, d) cross properly in their interiors. */
function properCross(a, b, c, d) {
  const d1 = sign(cross(c, d, a));
  const d2 = sign(cross(c, d, b));
  const d3 = sign(cross(a, b, c));
  const d4 = sign(cross(a, b, d));
  return d1 * d2 < 0 && d3 * d4 < 0;
}

/** Crossing number of a ray to the right; returns 1 inside, 0 outside, 2 on the boundary. */
function pointInPolygon(p, polygon) {
  let inside = false;
  const n = polygon.length;
  for (let i = 0; i < n; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % n];
    if (onSegment(a, b, p)) return 2;
    if ((a[1] > p[1]) !== (b[1] > p[1])) {
      const x = a[0] + ((p[1] - a[1]) * (b[0] - a[0])) / (b[1] - a[1]);
      if (x > p[0]) inside = !inside;
    }
  }
  return inside ? 1 : 0;
}

function strictlyInside(p, polygons) {
  return polygons.some((polygon) => pointInPolygon(p, polygon) === 1);
}

function pointFree(p, polygons) {
  return !strictlyInside(p, polygons);
}

const midpoint = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

/** Visibility of the open segment (a, b) in free space. */
function visible(a, b, polygons) {
  if (!pointFree(midpoint(a, b), polygons)) return false;
  for (const polygon of polygons) {
    const n = polygon.length;
    for (let i = 0; i < n; i++) {
      const c = polygon[i];
      const d = polygon[(i + 1) % n];
      if (properCross(a, b, c, d)) return false;
      // a vertex strictly inside the segment blocks the line of sight
      if (onSegment(a, b, c) && !((Math.abs(c[0] - a[0]) < EPS && Math.abs(c[1] - a[1]) < EPS) ||
        (Math.abs(c[0] - b[0]) < EPS && Math.abs(c[1] - b[1]) < EPS))) return false;
    }
  }
  return true;
}

/** Independent sampler: a strictly interior point on the segment proves obstruction. */
function sampledObstruction(a, b, polygons, samples = 400) {
  for (let i = 1; i < samples; i++) {
    const t = i / samples;
    const p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    if (strictlyInside(p, polygons)) return true;
  }
  return false;
}

/** Nodes = start, goal and every polygon vertex; edge = visible pair. */
function buildVisibilityGraph(start, goal, polygons) {
  const nodes = [start, goal];
  const vertexNode = [];
  polygons.forEach((polygon, p) => {
    vertexNode[p] = [];
    polygon.forEach((v) => {
      vertexNode[p].push(nodes.length);
      nodes.push(v);
    });
  });
  const adjacency = nodes.map(() => []);
  let testedPairs = 0;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      testedPairs++;
      if (!visible(nodes[i], nodes[j], polygons)) continue;
      const length = distance(nodes[i], nodes[j]);
      adjacency[i].push([j, length]);
      adjacency[j].push([i, length]);
    }
  }
  return { nodes, adjacency, vertexNode, testedPairs };
}

/** Dijkstra with a naive linear-scan frontier (the graphs here have a few dozen nodes). */
function dijkstra(adjacency, source) {
  const n = adjacency.length;
  const best = new Array(n).fill(Infinity);
  const previous = new Array(n).fill(-1);
  const settled = new Array(n).fill(false);
  best[source] = 0;
  for (let step = 0; step < n; step++) {
    let pick = -1;
    for (let i = 0; i < n; i++) {
      if (!settled[i] && (pick === -1 || best[i] < best[pick] - EPS)) pick = i;
    }
    if (pick === -1 || !Number.isFinite(best[pick])) break;
    settled[pick] = true;
    for (const [next, weight] of adjacency[pick]) {
      if (best[pick] + weight < best[next] - EPS) {
        best[next] = best[pick] + weight;
        previous[next] = pick;
      }
    }
  }
  return { best, previous };
}

function planVisibility(start, goal, polygons) {
  if (!pointFree(start, polygons) || !pointFree(goal, polygons)) {
    throw new RangeError("start and goal must lie outside the obstacles");
  }
  const graph = buildVisibilityGraph(start, goal, polygons);
  const { best, previous } = dijkstra(graph.adjacency, 0);
  if (!Number.isFinite(best[1])) return { path: null, cost: Infinity, graph };
  const path = [];
  for (let node = 1; node !== -1; node = previous[node]) path.push(graph.nodes[node]);
  path.reverse();
  return { path, cost: best[1], graph };
}

function pathLength(path) {
  let total = 0;
  for (let i = 1; i < path.length; i++) total += distance(path[i - 1], path[i]);
  return total;
}

/** Greedy shortcut: a taut graph path must be returned unchanged. */
function shortcut(path, polygons) {
  const out = [path[0]];
  let current = 0;
  while (current < path.length - 1) {
    let next = current + 1;
    for (let candidate = path.length - 1; candidate > current; candidate--) {
      if (visible(path[current], path[candidate], polygons)) {
        next = candidate;
        break;
      }
    }
    out.push(path[next]);
    current = next;
  }
  return out;
}

/** Independent oracle: Dijkstra on a fine grid of free cell centres. */
function gridShortestPath(start, goal, polygons, bounds, cell) {
  const cols = Math.floor((bounds[2] - bounds[0]) / cell) + 1;
  const rows = Math.floor((bounds[3] - bounds[1]) / cell) + 1;
  const toPoint = (c) => [bounds[0] + c[0] * cell, bounds[1] + c[1] * cell];
  const cells = [];
  for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) cells.push([i, j]);
  const index = new Map(cells.map((c, i) => [`${c[0]},${c[1]}`, i]));
  const adjacency = cells.map(() => []);
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      const from = toPoint([i, j]);
      if (!pointFree(from, polygons)) continue;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;
          const ni = i + dx;
          const nj = j + dy;
          if (ni < 0 || nj < 0 || ni >= cols || nj >= rows) continue;
          const to = toPoint([ni, nj]);
          if (!pointFree(to, polygons)) continue;
          if (sampledObstruction(from, to, polygons, 40)) continue;
          adjacency[index.get(`${i},${j}`)].push([index.get(`${ni},${nj}`), distance(from, to)]);
        }
      }
    }
  }
  const cellOf = (p) => [Math.round((p[0] - bounds[0]) / cell), Math.round((p[1] - bounds[1]) / cell)];
  const source = index.get(`${cellOf(start)[0]},${cellOf(start)[1]}`);
  const target = index.get(`${cellOf(goal)[0]},${cellOf(goal)[1]}`);
  if (source === undefined || target === undefined) return Infinity;
  return dijkstra(adjacency, source).best[target];
}

const rectangle = (x0, y0, x1, y1) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];

module.exports = { visible, planVisibility, buildVisibilityGraph, pointInPolygon, gridShortestPath, shortcut };

if (require.main === module) {
  const wall = [rectangle(40, 0, 50, 60)];
  const start = [10, 50];
  const goal = [90, 50];

  // -- predicate hand cases ---------------------------------------------------------------
  ok(visible([0, 0], [30, 0], wall), "a segment clear of the wall is visible");
  ok(!visible([0, 30], [100, 30], wall), "a segment through the wall is blocked");
  ok(!visible([45, 90], [45, 10], wall), "a segment through the wall is blocked vertically");
  ok(!visible([30, 30], [60, 30], wall), "a segment starting outside and ending inside is blocked");
  ok(visible([0, 61], [100, 61], wall), "a segment that passes above the wall is visible");
  ok(!visible([40, 60], [100, 60], wall), "touching the corner from outside blocks the sight line");
  ok(visible([30, 30], [40, 30], wall), "ending exactly on a wall vertex is allowed");
  ok(visible([40, 70], [40, 80], wall), "a segment grazing the wall from above stays visible");
  ok(!visible([42, 10], [48, 50], wall), "a segment inside the wall is blocked by the midpoint test");
  ok(visible([80, 80], [90, 90], wall), "far away segments are visible");

  // -- the sampler can only ever confirm obstruction ---------------------------------------
  const complex = [rectangle(20, 20, 40, 70), rectangle(55, 30, 75, 90), [[5, 90], [18, 95], [30, 88], [12, 82]]];
  let confirmations = 0;
  let pairs = 0;
  for (let i = 0; i < 60; i++) {
    for (let j = i + 1; j < 60; j++) {
      const a = [(i * 7) % 97, (i * 13) % 91];
      const b = [(j * 11) % 97, (j * 17) % 91];
      pairs++;
      if (sampledObstruction(a, b, complex, 300)) {
        ok(!visible(a, b, complex), `sampled obstruction at pair ${i},${j} must block visibility`);
        confirmations++;
      }
    }
  }
  ok(confirmations > 100, `the sampler must confirm many blocked pairs (got ${confirmations} of ${pairs})`);

  // -- one rectangular obstacle: the optimal length is analytic ----------------------------
  const single = [rectangle(30, 30, 70, 70)];
  const route = planVisibility([10, 50], [90, 50], single);
  ok(route.path !== null, "the single obstacle must be circumnavigable");
  const corner = [30, 30];
  const analytic = 2 * distance([10, 50], corner) + 40;
  ok(Math.abs(route.cost - analytic) < 1e-9,
    `the shortest route must round two corners (${route.cost.toFixed(6)} vs ${analytic.toFixed(6)})`);
  deepStrictEqual(route.path, [[10, 50], [30, 30], [70, 30], [90, 50]], "the taut route hugs the bottom");
  deepStrictEqual(shortcut(route.path, single), route.path, "a graph route must already be taut");
  ok(route.path.every((p) => !strictlyInside(p, single)), "no bend may lie inside the obstacle");

  // -- wall with a gap -----------------------------------------------------------------------
  const gapWall = [rectangle(40, 0, 50, 60), rectangle(40, 70, 50, 100)];
  const gaps = planVisibility(start, goal, gapWall);
  ok(gaps.path !== null, "the gap must be usable");
  deepStrictEqual(gaps.path, [[10, 50], [40, 60], [50, 60], [90, 50]], "the taut route clips both gap corners");
  for (let i = 1; i < gaps.path.length; i++) {
    ok(visible(gaps.path[i - 1], gaps.path[i], gapWall), `route edge ${i} must be visible`);
    ok(!sampledObstruction(gaps.path[i - 1], gaps.path[i], gapWall, 400), `route edge ${i} must be clean`);
  }
  const gapOracle = gridShortestPath(start, goal, gapWall, [0, 0, 100, 100], 2);
  ok(Number.isFinite(gapOracle), "the grid oracle must find the gap as well");
  ok(gaps.cost <= gapOracle + 1e-9, `the graph route must beat the grid route (${gaps.cost.toFixed(3)} vs ${gapOracle.toFixed(3)})`);
  ok(gaps.cost >= distance(start, goal) - 1e-9, "no route can beat the Euclidean distance");
  deepStrictEqual(shortcut(gaps.path, gapWall), gaps.path, "the gap route must be taut");

  // -- a corridor of two walls forces a specific detour ---------------------------------------
  const corridor = [rectangle(30, 0, 40, 45), rectangle(60, 55, 70, 100)];
  const corridorRoute = planVisibility([10, 20], [90, 80], corridor);
  ok(corridorRoute.path !== null, "the slalom must be solvable");
  deepStrictEqual(corridorRoute.path,
    [[10, 20], [30, 45], [70, 55], [90, 80]], "the taut slalom bends at the two wall corners");
  const corridorOracle = gridShortestPath([10, 20], [90, 80], corridor, [0, 0, 100, 100], 2);
  ok(!(corridorOracle < corridorRoute.cost - 1e-6), "the grid can never beat the visibility graph");

  // -- graph structure ------------------------------------------------------------------------
  const graph = gaps.graph;
  strictEqual(graph.nodes.length, 2 + 8, "the graph holds the endpoints plus every polygon vertex");
  let asymmetric = 0;
  for (let i = 0; i < graph.adjacency.length; i++) {
    for (const [j] of graph.adjacency[i]) {
      if (!graph.adjacency[j].some(([back]) => back === i)) asymmetric++;
    }
  }
  strictEqual(asymmetric, 0, "the visibility graph must be undirected");
  ok(graph.adjacency[0].length > 0, "the start must see something");
  const repeat = buildVisibilityGraph(start, goal, gapWall);
  strictEqual(repeat.testedPairs, graph.testedPairs, "the pair count must be deterministic");
  deepStrictEqual(planVisibility(start, goal, gapWall).path, gaps.path, "planning must be deterministic");

  // -- unreachable and invalid queries ----------------------------------------------------------
  const closedRing = [rectangle(20, 20, 80, 26), rectangle(20, 74, 80, 80), rectangle(20, 20, 26, 80),
    rectangle(74, 20, 80, 80)];
  const trapped = planVisibility([5, 5], [50, 50], closedRing);
  strictEqual(trapped.path, null, "a goal inside a closed ring is unreachable");
  strictEqual(gridShortestPath([5, 5], [50, 50], closedRing, [0, 0, 100, 100], 2), Infinity,
    "the grid oracle agrees that the ring is closed");
  const outside = planVisibility([5, 5], [95, 95], closedRing);
  ok(outside.path !== null, "two points outside the ring are still connected");
  let threw = false;
  try {
    planVisibility([23, 50], [95, 95], closedRing);
  } catch (error) {
    threw = error instanceof RangeError;
  }
  ok(threw, "a start inside a wall polygon must be rejected");
  strictEqual(pointInPolygon([23, 50], closedRing[2]), 1, "the point really is strictly inside that wall");
  strictEqual(pointInPolygon([50, 50], closedRing[2]), 0, "the enclosed courtyard itself is free space");
  const touching = planVisibility([10, 50], [90, 50], []);
  deepStrictEqual(touching.path, [[10, 50], [90, 50]], "an empty world gives the straight line");
  ok(Math.abs(touching.cost - 80) < 1e-9, "the straight line cost is the distance");

  console.log(`blocked_pairs_confirmed=${confirmations}/${pairs} graph_nodes=${graph.nodes.length}`);
  console.log(`single_obstacle=${route.cost.toFixed(4)} analytic=${analytic.toFixed(4)} gap=${gaps.cost.toFixed(4)}`);
  console.log(`gap_oracle=${gapOracle.toFixed(4)} corridor=${corridorRoute.cost.toFixed(4)}`);
  console.log("PASS");
}
