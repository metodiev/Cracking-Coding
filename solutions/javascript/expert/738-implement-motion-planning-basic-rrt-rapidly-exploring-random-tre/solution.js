/** Task 738 - Implement motion planning: basic RRT (rapidly exploring random tree).
 *  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * The workspace is a rectangle of axis-aligned obstacles; a segment is free when it does not
 * touch any closed obstacle box (the slab/interval test). The tree grows from the start: a
 * deterministic xorshift32 sample replaces the goal every GOAL_BIAS-th iteration (goal bias),
 * the nearest node is found by a linear scan (documented simplification - no kd-tree), the
 * sample is clamped to STEP units along the connecting segment (steer) and the edge is added
 * only when it is collision free; the goal is inserted as soon as a steered point can reach it.
 * The resulting path is then shortcut greedily ("connect to the furthest ancestor whose edge is
 * free"), which removes the staircase artefacts of the raw tree.
 *
 * What is simplified: two dimensions, axis-aligned box obstacles, one tree (not RRT*), nearest
 * neighbour by linear scan with no rewiring or cost optimisation, and a fixed iteration budget.
 * What is verified: every returned edge is collision free under the exact checker *and* under an
 * independent dense sampler (a sampled point strictly inside an obstacle proves a collision, so
 * the sampler is only ever used in that sound direction), no edge exceeds STEP, the path starts
 * at the start and ends at the goal, its length is at least the Euclidean distance and within a
 * documented factor of a grid-BFS oracle path, the tree is a tree (one root, edges = nodes - 1,
 * all nodes inside the bounds), an enclosed goal yields no path, invalid starts are rejected,
 * and the whole run is reproducible from the seed.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

/** Deterministic xorshift32 in [0, 2^32); `bound` restricts the result. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x2545f491;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

const EPS = 1e-9;

const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

function pointInObstacle(point, box, strict) {
  const insideX = strict ? point[0] > box.x0 && point[0] < box.x1 : point[0] >= box.x0 && point[0] <= box.x1;
  const insideY = strict ? point[1] > box.y0 && point[1] < box.y1 : point[1] >= box.y0 && point[1] <= box.y1;
  return insideX && insideY;
}

function pointFree(point, obstacles) {
  return !obstacles.some((box) => pointInObstacle(point, box, false));
}

/** Slab test: does the closed segment touch the closed box? Touching counts as a collision. */
function segmentHitsBox(a, b, box) {
  if (pointInObstacle(a, box, false) || pointInObstacle(b, box, false)) return true;
  let tLow = 0;
  let tHigh = 1;
  const d = [b[0] - a[0], b[1] - a[1]];
  const low = [box.x0, box.y0];
  const high = [box.x1, box.y1];
  for (let axis = 0; axis < 2; axis++) {
    if (Math.abs(d[axis]) < EPS) {
      if (a[axis] < low[axis] || a[axis] > high[axis]) return false;
      continue;
    }
    let t1 = (low[axis] - a[axis]) / d[axis];
    let t2 = (high[axis] - a[axis]) / d[axis];
    if (t1 > t2) {
      const swap = t1;
      t1 = t2;
      t2 = swap;
    }
    tLow = Math.max(tLow, t1);
    tHigh = Math.min(tHigh, t2);
    if (tLow > tHigh + EPS) return false;
  }
  return tLow <= 1 + EPS && tHigh >= -EPS;
}

function segmentFree(a, b, obstacles) {
  return !obstacles.some((box) => segmentHitsBox(a, b, box));
}

/** Independent sampler: dense points along the segment, checked with a strict inside test. */
function sampledCollision(a, b, obstacles, samples = 512) {
  for (let i = 0; i <= samples; i++) {
    const point = lerp(a, b, i / samples);
    if (obstacles.some((box) => pointInObstacle(point, box, true))) return true;
  }
  return false;
}

/** Steer from `from` towards `to`, stopping STEP units short or at `to` when it is closer. */
function steer(from, to, step) {
  const span = distance(from, to);
  if (span <= step) return [to[0], to[1]];
  const t = step / span;
  return lerp(from, to, t);
}

/**
 * Basic RRT. Returns { path, nodes, edges, iterations, goalReached } or path === null when the
 * budget runs out. `start`/`goal` being inside an obstacle is an invalid input.
 */
function rrt(start, goal, obstacles, options = {}) {
  const bounds = options.bounds ?? [0, 0, 100, 100];
  const step = options.step ?? 3;
  const maxIterations = options.maxIterations ?? 4000;
  const goalBiasEvery = options.goalBiasEvery ?? 5;
  const rng = options.rng ?? makeRng(1);
  if (!pointFree(start, obstacles) || !pointFree(goal, obstacles)) {
    throw new RangeError("start and goal must be outside the obstacles");
  }
  const nodes = [[start[0], start[1]]];
  const parent = [-1];
  let iterations = 0;
  let goalIndex = -1;
  while (iterations < maxIterations) {
    iterations++;
    const sample = iterations % goalBiasEvery === 0
      ? goal
      : [bounds[0] + rng(bounds[2] - bounds[0] + 1), bounds[1] + rng(bounds[3] - bounds[1] + 1)];
    let nearestIndex = 0;
    let nearestDistance = Infinity;
    for (let i = 0; i < nodes.length; i++) {
      const d = distance(nodes[i], sample);
      if (d < nearestDistance) {
        nearestDistance = d;
        nearestIndex = i;
      }
    }
    const candidate = steer(nodes[nearestIndex], sample, step);
    if (!segmentFree(nodes[nearestIndex], candidate, obstacles)) continue;
    nodes.push(candidate);
    parent.push(nearestIndex);
    const newIndex = nodes.length - 1;
    if (distance(candidate, goal) <= step && segmentFree(candidate, goal, obstacles)) {
      nodes.push([goal[0], goal[1]]);
      parent.push(newIndex);
      goalIndex = nodes.length - 1;
      break;
    }
  }
  const raw = [];
  for (let i = goalIndex; i !== -1; i = parent[i]) raw.push(nodes[i]);
  raw.reverse();
  const path = goalIndex === -1 ? null : shortcut(raw, obstacles);
  return { path, nodes: nodes.length, parents: parent, edges: nodes.length - 1, iterations, rawPath: raw };
}

/** Greedy shortcutting: repeatedly jump to the furthest ancestor reachable by a free segment. */
function shortcut(path, obstacles) {
  if (path === null) return null;
  const out = [path[0]];
  let current = 0;
  while (current < path.length - 1) {
    let next = current + 1;
    for (let candidate = path.length - 1; candidate > current; candidate--) {
      if (segmentFree(path[current], path[candidate], obstacles)) {
        next = candidate;
        break;
      }
    }
    out.push(path[next]);
    current = next;
  }
  return out;
}

function pathLength(path) {
  let total = 0;
  for (let i = 1; i < path.length; i++) total += distance(path[i - 1], path[i]);
  return total;
}

/** Validates the structural promises of a path independently of the planner. */
function validatePath(path, start, goal, obstacles) {
  strictEqual(path[0][0], start[0], "the path must start at the start");
  strictEqual(path[0][1], start[1], "the path must start at the start");
  const last = path[path.length - 1];
  strictEqual(last[0], goal[0], "the path must end at the goal");
  strictEqual(last[1], goal[1], "the path must end at the goal");
  for (let i = 1; i < path.length; i++) {
    const a = path[i - 1];
    const b = path[i];
    ok(segmentFree(a, b, obstacles), `edge ${i} must be collision free`);
    ok(!sampledCollision(a, b, obstacles), `edge ${i} must survive the dense sampler`);
    ok(pointFree(a, obstacles) && pointFree(b, obstacles), "path vertices must be free");
  }
  ok(pathLength(path) >= distance(start, goal) - 1e-9, "a path cannot beat the Euclidean distance");
}

/** The raw tree edges are the ones that must respect the steering step. */
function validateTreeEdges(rawPath, obstacles, step) {
  for (let i = 1; i < rawPath.length; i++) {
    ok(distance(rawPath[i - 1], rawPath[i]) <= step + 1e-6, `tree edge ${i} must respect the step size`);
    ok(segmentFree(rawPath[i - 1], rawPath[i], obstacles), `tree edge ${i} must be collision free`);
  }
}

/** Oracle: Dijkstra on an 8-connected grid of free cell centres, connected by free segments. */
function gridShortestPath(start, goal, obstacles, bounds, step) {
  const cols = Math.floor((bounds[2] - bounds[0]) / step) + 1;
  const rows = Math.floor((bounds[3] - bounds[1]) / step) + 1;
  const toCell = (p) => [Math.round((p[0] - bounds[0]) / step), Math.round((p[1] - bounds[1]) / step)];
  const toPoint = (c) => [bounds[0] + c[0] * step, bounds[1] + c[1] * step];
  const startCell = toCell(start);
  const goalCell = toCell(goal);
  const best = new Map();
  const key = (c) => `${c[0]},${c[1]}`;
  const queue = [{ cell: startCell, cost: 0 }];
  best.set(key(startCell), 0);
  while (queue.length > 0) {
    let pick = 0;
    for (let i = 1; i < queue.length; i++) if (queue[i].cost < queue[pick].cost) pick = i;
    const { cell, cost } = queue.splice(pick, 1)[0];
    if (cost > (best.get(key(cell)) ?? Infinity)) continue;
    if (key(cell) === key(goalCell)) return cost;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        if (dx === 0 && dy === 0) continue;
        const next = [cell[0] + dx, cell[1] + dy];
        if (next[0] < 0 || next[1] < 0 || next[0] >= cols || next[1] >= rows) continue;
        if (!pointFree(toPoint(next), obstacles)) continue;
        if (!segmentFree(toPoint(cell), toPoint(next), obstacles)) continue;
        const candidate = cost + distance(toPoint(cell), toPoint(next));
        if (candidate < (best.get(key(next)) ?? Infinity)) {
          best.set(key(next), candidate);
          queue.push({ cell: next, cost: candidate });
        }
      }
    }
  }
  return Infinity;
}

module.exports = { rrt, segmentFree, sampledCollision, gridShortestPath, pathLength, makeRng, shortcut };

if (require.main === module) {
  const box = (x0, y0, x1, y1) => ({ x0, y0, x1, y1 });
  const bounds = [0, 0, 100, 100];

  // -- hand checked collision checker ----------------------------------------------------
  const wall = [box(40, 0, 50, 60)];
  ok(segmentFree([0, 0], [30, 0], wall), "a segment that stops before the wall is free");
  ok(segmentFree([0, 80], [100, 80], wall), "a segment above the wall is free");
  ok(!segmentFree([0, 30], [100, 30], wall), "a segment through the wall is blocked");
  ok(!segmentFree([45, 90], [45, 10], wall), "a segment through the wall is blocked vertically");
  ok(!segmentFree([20, 20], [45, 30], wall), "an endpoint inside the wall is blocked");
  ok(!segmentFree([30, 40], [50, 40], wall), "touching the wall's face counts as a collision");
  ok(!segmentFree([40, 60], [60, 60], wall), "touching the corners counts as a collision");
  ok(segmentFree([-5, 30], [0, 30], wall), "outside segments are fine");
  for (const boxTest of wall) {
    ok(!segmentFree([boxTest.x0 - EPS, 30], [boxTest.x0 + EPS, 30], [boxTest]),
      "a hair-thin crossing still counts");
  }
  // the dense sampler is only trustworthy in the sound direction
  let samplerAgreements = 0;
  const rng = makeRng(738738);
  for (let trial = 0; trial < 300; trial++) {
    const a = [rng(101), rng(101)];
    const b = [rng(101), rng(101)];
    if (sampledCollision(a, b, wall, 400)) {
      ok(!segmentFree(a, b, wall), "a sampled point inside the wall must imply a collision");
      samplerAgreements++;
    }
  }
  ok(samplerAgreements > 0, "the sampler must find collisions sometimes");

  // -- empty workspace: the shortcut path is the straight line ---------------------------
  const open = rrt([5, 5], [95, 95], [], { bounds, step: 5, rng: makeRng(11) });
  ok(open.path !== null, "the open workspace must be solvable");
  deepStrictEqual(open.path, [[5, 5], [95, 95]], "shortcutting must collapse the staircase");
  validatePath(open.path, [5, 5], [95, 95], []);
  validateTreeEdges(open.rawPath, [], 5);
  ok(Math.abs(pathLength(open.path) - distance([5, 5], [95, 95])) < 1e-9, "the line is optimal here");
  strictEqual(open.edges, open.nodes - 1, "a tree with n nodes has n - 1 edges");

  // -- wall with a gap: the RRT must find and shorten a path -----------------------------
  const gapWall = [box(40, 0, 50, 60), box(40, 70, 50, 100)];
  const start = [10, 50];
  const goal = [90, 50];
  const plan = rrt(start, goal, gapWall, { bounds, step: 4, goalBiasEvery: 5, rng: makeRng(2024) });
  ok(plan.path !== null, "the wall with a gap must be solvable");
  validatePath(plan.path, start, goal, gapWall);
  validateTreeEdges(plan.rawPath, gapWall, 4);
  const oracle = gridShortestPath(start, goal, gapWall, bounds, 2);
  ok(Number.isFinite(oracle), "the grid oracle must find the gap too");
  ok(oracle >= distance(start, goal) - 1e-9, "the grid oracle cannot beat the straight line");
  const planned = pathLength(plan.path);
  ok(planned <= oracle * 1.5, `the RRT path must stay near optimal (${planned.toFixed(2)} vs ${oracle.toFixed(2)})`);
  ok(plan.path.length < plan.rawPath.length, "shortcutting must remove at least one waypoint");
  const rawLength = pathLength(plan.rawPath);
  ok(planned <= rawLength + 1e-9, "shortcutting must never lengthen the path");
  for (const node of plan.rawPath) {
    ok(node[0] >= bounds[0] - 1e-9 && node[0] <= bounds[2] + 1e-9 && node[1] >= bounds[1] - 1e-9 &&
      node[1] <= bounds[3] + 1e-9, "every sampled node must stay inside the bounds");
  }

  // -- three walls: a longer detour, still collision free and near the oracle -------------
  const maze = [box(20, 0, 28, 70), box(46, 30, 54, 100), box(72, 0, 80, 65)];
  const mazePlan = rrt([5, 95], [95, 5], maze, { bounds, step: 5, rng: makeRng(9), maxIterations: 6000 });
  ok(mazePlan.path !== null, "the three-wall maze must be solvable");
  validatePath(mazePlan.path, [5, 95], [95, 5], maze);
  validateTreeEdges(mazePlan.rawPath, maze, 5);
  const mazeOracle = gridShortestPath([5, 95], [95, 5], maze, bounds, 2.5);
  ok(Number.isFinite(mazeOracle), "the grid oracle must solve the maze");
  ok(pathLength(mazePlan.path) <= mazeOracle * 1.5,
    `maze path ${pathLength(mazePlan.path).toFixed(2)} vs oracle ${mazeOracle.toFixed(2)}`);

  // -- enclosed goal: no path, and the budget is respected -------------------------------
  const enclosure = [box(20, 20, 80, 26), box(20, 74, 80, 80), box(20, 20, 26, 80), box(74, 20, 80, 80)];
  const trapped = rrt([10, 10], [50, 50], enclosure, { bounds, step: 4, rng: makeRng(5), maxIterations: 800 });
  strictEqual(trapped.path, null, "a goal inside a closed ring must be unreachable");
  strictEqual(trapped.iterations, 800, "the budget must be spent exactly");
  ok(trapped.nodes > 1, "the tree must still have grown");

  // -- invalid inputs --------------------------------------------------------------------
  throws(() => rrt([25, 50], [90, 50], [box(20, 0, 30, 100)]), RangeError,
    "a start inside an obstacle is rejected");
  throws(() => rrt([10, 50], [25, 50], [box(20, 0, 30, 100)]), RangeError,
    "a goal inside an obstacle is rejected");
  const tiny = rrt([1, 1], [2, 2], [], { bounds: [0, 0, 3, 3], step: 5, rng: makeRng(3) });
  deepStrictEqual(tiny.path, [[1, 1], [2, 2]], "a goal closer than the step is reached directly");

  // -- determinism ------------------------------------------------------------------------
  const replay = () => rrt(start, goal, gapWall, { bounds, step: 4, rng: makeRng(2024) });
  const firstRun = replay();
  const secondRun = replay();
  deepStrictEqual(firstRun.path, secondRun.path, "identical seeds must produce identical paths");
  strictEqual(firstRun.nodes, secondRun.nodes, "identical seeds must produce identical trees");
  strictEqual(firstRun.iterations, secondRun.iterations, "identical seeds must use the same budget");

  console.log(`open_nodes=${open.nodes} gap_nodes=${plan.nodes} iterations=${plan.iterations}`);
  console.log(`gap_path=${planned.toFixed(3)} oracle=${oracle.toFixed(3)} raw=${rawLength.toFixed(3)}`);
  console.log(`maze_nodes=${mazePlan.nodes} maze_path=${pathLength(mazePlan.path).toFixed(3)} sampler_hits=${samplerAgreements}`);
  console.log("PASS");
}
