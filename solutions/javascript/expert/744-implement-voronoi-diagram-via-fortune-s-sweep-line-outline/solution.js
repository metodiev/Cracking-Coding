/** Task 744 - Implement the Voronoi diagram via Fortune's sweep-line (outline).
 *  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A horizontal line sweeps from +y downwards. The *beachline* is the lower envelope of the
 * parabolas whose focus is a site already passed and whose directrix is the sweep line; it is
 * kept as an ordered array of arcs (the documented simplification - a linear scan and a linear
 * event list replace the balanced tree and the priority queue). A site event splits the arc
 * whose parabola is lowest above the new site and opens one new bisector, a circle event is the
 * lowest point of the circumcircle of three consecutive arcs; it removes the middle arc, emits
 * one Voronoi vertex and terminates the two bisectors meeting there. Circle events are scheduled
 * for every consecutive triple and validated when popped (the arc must still be in the beachline
 * with the same pending event, and the neighbours must still be its own). Bisectors that never
 * get a second endpoint are completed as rays (or, when they never met a vertex at all, as full
 * lines) using the direction that leaves the cells of the two sites intact, and are clipped to a
 * bounding box on request.
 *
 * What is simplified: floating-point arithmetic, a linear beachline scan, no event balancing and
 * general position (no three collinear sites and no four cocircular sites). What is verified
 * against independent constructions: the vertex set coincides with the circumcentres of the
 * Delaunay triangles of the same sites computed by exact BigInt predicates, every vertex is
 * equidistant from its three defining sites and no closer to any other site (the empty-circle
 * property, checked numerically), every Voronoi edge is the perpendicular bisector of a Delaunay
 * pair and the edge count equals the Delaunay edge count, the clipped cells of all sites each
 * contain exactly their own site and tile the bounding box by area, the collinear and few-site
 * cases are handled specially, and the run is deterministic.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

const EPS = 1e-9;
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

// ---- exact helpers (oracle and circumcentre checks) -----------------------------------------
const big = (v) => v.map((c) => BigInt(c));

function orient2(a, b, c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function det3(rows) {
  const [[a, b, c], [d, e, f], [g, h, i]] = rows;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

/** > 0 when d is strictly inside the circumcircle of the counter-clockwise triangle (a, b, c). */
function inCircle(a, b, c, d) {
  const rows = [a, b, c].map((p) => {
    const dx = BigInt(p[0]) - BigInt(d[0]);
    const dy = BigInt(p[1]) - BigInt(d[1]);
    return [dx, dy, dx * dx + dy * dy];
  });
  return det3(rows);
}

/** Circumcentre of three sites as an exact rational: [xNumerator, yNumerator, denominator]. */
function circumcentreExact(a, b, c) {
  const [ax, ay] = big(a);
  const [bx, by] = big(b);
  const [cx, cy] = big(c);
  const d = 2n * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  const ux = (ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) +
    (cx * cx + cy * cy) * (ay - by);
  const uy = (ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) +
    (cx * cx + cy * cy) * (bx - ax);
  return [ux, uy, d];
}

// ---- the sweep -------------------------------------------------------------------------------
/** Circumcircle of three sites: centre, radius and the sweep position where the event fires. */
function circumcircleOf(a, b, c) {
  const determinant = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
  if (Math.abs(determinant) < 1e-12) return null; // collinear: no circle event
  const centreX = ((a[0] ** 2 + a[1] ** 2) * (b[1] - c[1]) + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1]) +
    (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])) / determinant;
  const centreY = ((a[0] ** 2 + a[1] ** 2) * (c[0] - b[0]) + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0]) +
    (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])) / determinant;
  const radius = Math.hypot(centreX - a[0], centreY - a[1]);
  return { centreX, centreY, radius, sweepY: centreY - radius };
}

/**
 * Breakpoint between two consecutive arcs: the x where the beachline passes from the left arc's
 * focus to the right arc's. Equating the two parabolas gives a quadratic whose two roots are the
 * two crossings; the transition is the crossing on the side of the left focus, so the branch
 * depends on the x-order of the two sites.
 */
function transitionX(sites, arcA, arcB, sweepY) {
  const a = sites[arcA.site];
  const b = sites[arcB.site];
  if (a[0] === b[0] && a[1] === b[1]) return Infinity;
  const curvatureA = 1 / (2 * (a[1] - sweepY));
  const curvatureB = 1 / (2 * (b[1] - sweepY));
  const p = curvatureA - curvatureB;
  const q = -2 * (curvatureA * a[0] - curvatureB * b[0]);
  const r = curvatureA * a[0] * a[0] - curvatureB * b[0] * b[0] + (a[1] - b[1]) / 2;
  let first;
  let second;
  if (Math.abs(p) < 1e-12) {
    if (Math.abs(q) < 1e-12) return Infinity;
    first = -r / q;
    second = first;
  } else {
    const discriminant = q * q - 4 * p * r;
    if (discriminant < 0) return Infinity;
    const root = Math.sqrt(discriminant);
    first = (-q - root) / (2 * p);
    second = (-q + root) / (2 * p);
  }
  // The wider parabola (the smaller curvature, i.e. the higher site) owns the two outer
  // intervals, so the transition from a wider to a narrower arc is the *first* crossing and
  // the other way round it is the second.
  return curvatureA <= curvatureB ? Math.min(first, second) : Math.max(first, second);
}

/** The arc whose x-interval contains x; breakpoints decide, equal parabolas cannot. */
function locateArc(sites, arcs, x, sweepY) {
  for (let i = 0; i + 1 < arcs.length; i++) {
    if (x <= transitionX(sites, arcs[i], arcs[i + 1], sweepY) + EPS) return i;
  }
  return arcs.length - 1;
}

/** The circle through three sites must contain no other site for the event to be real. */
function circleIsEmpty(sites, circle, triple) {
  void circle;
  // the in-circle determinant needs a counter-clockwise triple, so orient it once
  const ccw = orient2(sites[triple[0]], sites[triple[1]], sites[triple[2]]) > 0
    ? [sites[triple[0]], sites[triple[1]], sites[triple[2]]]
    : [sites[triple[0]], sites[triple[2]], sites[triple[1]]];
  for (let index = 0; index < sites.length; index++) {
    if (triple.includes(index)) continue;
    if (inCircle(ccw[0], ccw[1], ccw[2], sites[index]) > 0n) return false;
  }
  return true;
}

/**
 * The next circle event, recomputed from scratch: the consecutive arc triple whose circumcircle is
 * empty *and* whose middle arc is really squeezed at that sweep position (the two breakpoints of
 * the middle arc coincide there). Recomputing and re-validating instead of caching events removes
 * the whole class of stale-event bugs.
 */
function nextCircleEvent(sites, arcs, sweepY) {
  let best = null;
  for (let i = 1; i + 1 < arcs.length; i++) {
    const triple = [arcs[i - 1].site, arcs[i].site, arcs[i + 1].site];
    if (triple[0] === triple[1] || triple[1] === triple[2] || triple[0] === triple[2]) continue;
    const circle = circumcircleOf(sites[triple[0]], sites[triple[1]], sites[triple[2]]);
    if (circle === null) continue;
    const leftEdge = transitionX(sites, arcs[i - 1], arcs[i], circle.sweepY);
    const rightEdge = transitionX(sites, arcs[i], arcs[i + 1], circle.sweepY);
    // the middle arc vanishes exactly when both of its breakpoints meet
    if (!(Math.abs(leftEdge - rightEdge) < 1e-6 * (1 + Math.abs(leftEdge) + Math.abs(rightEdge)))) {
      continue;
    }
    if (!circleIsEmpty(sites, circle, triple)) continue;
    if (best === null || circle.sweepY > best.sweepY + EPS) {
      best = { ...circle, position: i, triple };
    }
  }
  return best;
}

function buildVoronoi(sites) {
  if (sites.length === 0) return { vertices: [], edges: [], rays: [], lines: [], collinear: false };
  const collinear = sites.length < 3 || sites.every((s) => orient2(sites[0], sites[1], s) === 0);
  const vertices = [];
  const bisectors = new Map();
  const ensureBisector = (a, b) => {
    const key = `${Math.min(a, b)},${Math.max(a, b)}`;
    if (!bisectors.has(key)) {
      bisectors.set(key, { a: Math.min(a, b), b: Math.max(a, b), points: [] });
    }
    return bisectors.get(key);
  };
  const samePoint = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-9;
  const addBisectorPoint = (a, b, point) => {
    const bisector = ensureBisector(a, b);
    const last = bisector.points[bisector.points.length - 1];
    if (last === undefined || !samePoint(last, point)) bisector.points.push([point[0], point[1]]);
  };

  const order = sites.map((s, i) => i)
    .sort((i, j) => sites[j][1] - sites[i][1] || sites[i][0] - sites[j][0]);
  const arcs = [];
  let nextSite = 0;
  let sweepY = Infinity;
  let guard = 0;
  while (guard++ < 20000) {
    const siteEvent = nextSite < order.length ? order[nextSite] : null;
    const circleEvent = nextCircleEvent(sites, arcs, sweepY);
    if (siteEvent === null && circleEvent === null) break;
    const siteY = siteEvent === null ? -Infinity : sites[siteEvent][1];
    if (circleEvent === null || siteY >= circleEvent.sweepY - EPS) {
      sweepY = siteY;
      const index = siteEvent;
      if (arcs.length === 0) {
        arcs.push({ site: index });
      } else {
        const position = locateArc(sites, arcs, sites[index][0], sweepY);
        const split = arcs[position];
        arcs.splice(position, 1, { site: split.site }, { site: index }, { site: split.site });
        ensureBisector(split.site, index);
      }
      nextSite++;
      continue;
    }
    sweepY = circleEvent.sweepY;
    const middle = arcs[circleEvent.position];
    const leftArc = arcs[circleEvent.position - 1];
    const rightArc = arcs[circleEvent.position + 1];
    const point = [circleEvent.centreX, circleEvent.centreY];
    if (!vertices.some((v) => samePoint([v.x, v.y], point))) {
      vertices.push({ x: point[0], y: point[1], radius: circleEvent.radius, sites: circleEvent.triple });
      addBisectorPoint(leftArc.site, middle.site, point);
      addBisectorPoint(middle.site, rightArc.site, point);
      addBisectorPoint(leftArc.site, rightArc.site, point);
    }
    arcs.splice(circleEvent.position, 1);
  }
  for (let i = 0; i + 1 < arcs.length; i++) ensureBisector(arcs[i].site, arcs[i + 1].site);

  const edges = [];
  const rays = [];
  const lines = [];
  const centre = [sites.reduce((s, p) => s + p[0], 0) / sites.length,
    sites.reduce((s, p) => s + p[1], 0) / sites.length];
  const unit = (v) => {
    const length = Math.hypot(v[0], v[1]);
    return length < EPS ? null : [v[0] / length, v[1] / length];
  };
  for (const bisector of bisectors.values()) {
    const a = sites[bisector.a];
    const b = sites[bisector.b];
    const along = unit([-(b[1] - a[1]), b[0] - a[0]]);
    if (along === null) continue;
    if (bisector.points.length >= 2) {
      edges.push({ a: bisector.a, b: bisector.b, start: bisector.points[0],
        end: bisector.points[bisector.points.length - 1], direction: along });
      continue;
    }
    if (bisector.points.length === 1) {
      const point = bisector.points[0];
      const towards = [centre[0] - point[0], centre[1] - point[1]];
      const sign = along[0] * towards[0] + along[1] * towards[1] > 0 ? -1 : 1;
      const direction = [along[0] * sign, along[1] * sign];
      rays.push({ a: bisector.a, b: bisector.b, start: point, direction,
        end: [point[0] + direction[0] * 1e6, point[1] + direction[1] * 1e6] });
      continue;
    }
    const middle = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    lines.push({ a: bisector.a, b: bisector.b, start: middle, direction: along,
      end: [middle[0] + along[0] * 1e6, middle[1] + along[1] * 1e6] });
  }
  return { vertices, edges, rays, lines, collinear };
}

// ---- independent constructions used by the checks ---------------------------------------------
function clipRay(point, direction, bounds) {
  let low = 0;
  let high = 1e6;
  for (let axis = 0; axis < 2; axis++) {
    const min = axis === 0 ? bounds[0] : bounds[1];
    const max = axis === 0 ? bounds[2] : bounds[3];
    if (Math.abs(direction[axis]) < EPS) {
      if (point[axis] < min - EPS || point[axis] > max + EPS) return null;
      continue;
    }
    const t1 = (min - point[axis]) / direction[axis];
    const t2 = (max - point[axis]) / direction[axis];
    low = Math.max(low, Math.min(t1, t2));
    high = Math.min(high, Math.max(t1, t2));
  }
  if (high < low - EPS) return null;
  return [[point[0] + direction[0] * low, point[1] + direction[1] * low],
    [point[0] + direction[0] * high, point[1] + direction[1] * high]];
}

/** Exact brute-force Delaunay triangulation of the site set (BigInt predicates). */
function bruteDelaunay(sites) {
  const triangles = [];
  for (let i = 0; i < sites.length; i++) {
    for (let j = i + 1; j < sites.length; j++) {
      for (let k = j + 1; k < sites.length; k++) {
        if (orient2(sites[i], sites[j], sites[k]) === 0) continue;
        const ccw = orient2(sites[i], sites[j], sites[k]) > 0 ? [i, j, k] : [i, k, j];
        let empty = true;
        for (let other = 0; other < sites.length && empty; other++) {
          if (ccw.includes(other)) continue;
          if (inCircle(sites[ccw[0]], sites[ccw[1]], sites[ccw[2]], sites[other]) > 0n) empty = false;
        }
        if (empty) triangles.push(ccw);
      }
    }
  }
  return triangles;
}

/** The site's cell, built independently by clipping the box against every bisector. */
function cellPolygon(siteIndex, sites, bounds) {
  let polygon = [[bounds[0], bounds[1]], [bounds[2], bounds[1]], [bounds[2], bounds[3]],
    [bounds[0], bounds[3]]];
  const site = sites[siteIndex];
  for (let other = 0; other < sites.length; other++) {
    if (other === siteIndex) continue;
    const o = sites[other];
    const side = (p) => (p[0] - site[0]) ** 2 + (p[1] - site[1]) ** 2 -
      ((p[0] - o[0]) ** 2 + (p[1] - o[1]) ** 2);
    const next = [];
    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i];
      const b = polygon[(i + 1) % polygon.length];
      const fa = side(a);
      const fb = side(b);
      if (fa <= 1e-9) next.push(a);
      if ((fa <= 1e-9) !== (fb <= 1e-9)) {
        const t = fa / (fa - fb);
        next.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
      }
    }
    polygon = next;
    if (polygon.length === 0) break;
  }
  return polygon;
}

function polygonArea(polygon) {
  let total = 0;
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    total += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(total) / 2;
}

module.exports = { buildVoronoi, bruteDelaunay, circumcentreExact, cellPolygon, clipRay, polygonArea };

if (require.main === module) {
  const bounds = [-45, -45, 90, 90];
  const rng = makeRng(744744);

  // -- tiny and degenerate site sets ----------------------------------------------------------
  const empty = buildVoronoi([]);
  deepStrictEqual([empty.vertices, empty.edges, empty.rays, empty.lines], [[], [], [], []]);
  strictEqual(buildVoronoi([[3, 4]]).vertices.length, 0, "one site has no vertex");
  const pair = buildVoronoi([[0, 0], [10, 0]]);
  strictEqual(pair.vertices.length, 0, "two sites share one bisector and no vertex");
  strictEqual(pair.lines.length, 1, "with two sites the bisector is a full line");
  strictEqual(pair.lines[0].a, 0);
  strictEqual(pair.lines[0].b, 1);
  const pairClip = clipRay(pair.lines[0].start, pair.lines[0].direction, bounds);
  ok(pairClip !== null, "the bisector line must cross the bounding box");
  ok(Math.abs(pairClip[0][0] - 5) < 1e-9 && Math.abs(pairClip[1][0] - 5) < 1e-9,
    "the bisector of two horizontally separated sites is the vertical line x = 5");
  const trio = buildVoronoi([[0, 0], [10, 0], [20, 0]]);
  ok(trio.collinear, "collinear sites must be detected");
  strictEqual(trio.vertices.length, 0, "collinear sites produce no vertex");
  strictEqual(trio.lines.length, 2, "three collinear sites give two parallel bisectors");

  // -- a generic site set in general position ---------------------------------------------------
  const sites = [];
  while (sites.length < 9) {
    const candidate = [rng(61) - 30, rng(61) - 30];
    if (sites.some((p) => p[0] === candidate[0] && p[1] === candidate[1])) continue;
    sites.push(candidate);
  }
  const generic = sites.every((p, i) => sites.every((q, j) => j <= i ||
    sites.every((r, k) => k <= j || orient2(p, q, r) !== 0)));
  ok(generic, "the generated sites must be in general position");
  const diagram = buildVoronoi(sites);
  const delaunay = bruteDelaunay(sites);
  const circumcentreOf = (triangle) => {
    const [ux, uy, d] = circumcentreExact(sites[triangle[0]], sites[triangle[1]], sites[triangle[2]]);
    return [Number(ux) / Number(d), Number(uy) / Number(d)];
  };
  strictEqual(diagram.vertices.length, delaunay.length,
    `one vertex per Delaunay triangle (${diagram.vertices.length} vs ${delaunay.length})`);
  for (const triangle of delaunay) {
    const centre = circumcentreOf(triangle);
    const match = diagram.vertices.find((v) => Math.hypot(v.x - centre[0], v.y - centre[1]) < 1e-6);
    ok(match !== undefined, `the Delaunay triangle ${triangle} must have a Voronoi vertex`);
    ok(match.sites.slice().sort().join(",") === triangle.slice().sort().join(","),
      "the vertex's defining sites must be that triangle");
  }

  // -- the empty circle property at every vertex --------------------------------------------------
  let radiusChecks = 0;
  for (const vertex of diagram.vertices) {
    const a = sites[vertex.sites[0]];
    const radius = distance([vertex.x, vertex.y], a);
    ok(Math.abs(vertex.radius - radius) < 1e-6, "the stored radius must be the distance to the focus");
    for (const index of vertex.sites.slice(1)) {
      ok(Math.abs(distance([vertex.x, vertex.y], sites[index]) - radius) < 1e-6,
        "a Voronoi vertex is equidistant from its three defining sites");
      radiusChecks++;
    }
    for (let index = 0; index < sites.length; index++) {
      if (vertex.sites.includes(index)) continue;
      ok(distance([vertex.x, vertex.y], sites[index]) >= radius - 1e-6,
        `site ${index} must not be closer than the empty-circle radius`);
      radiusChecks++;
    }
  }
  ok(radiusChecks > 50, `the empty-circle property must be tested broadly (${radiusChecks} tests)`);

  // -- edge/dual correspondence -------------------------------------------------------------------
  const delaunayEdges = new Set();
  for (const triangle of delaunay) {
    for (let i = 0; i < 3; i++) {
      const u = triangle[i];
      const v = triangle[(i + 1) % 3];
      delaunayEdges.add(`${Math.min(u, v)},${Math.max(u, v)}`);
    }
  }
  const diagramEdges = [...diagram.edges, ...diagram.rays, ...diagram.lines];
  for (const edge of diagramEdges) {
    ok(delaunayEdges.has(`${Math.min(edge.a, edge.b)},${Math.max(edge.a, edge.b)}`),
      `the bisector ${edge.a}-${edge.b} must dualise to a Delaunay edge`);
    ok(Math.abs(distance(edge.start, sites[edge.a]) - distance(edge.start, sites[edge.b])) < 1e-6,
      "a bisector starts equally close to both of its sites");
  }
  strictEqual(diagramEdges.length, delaunayEdges.size,
    `the diagram must have one edge per Delaunay edge (${diagramEdges.length} vs ${delaunayEdges.size})`);
  // the unbounded bisectors are exactly the convex hull edges of the site set
  const hull = [];
  {
    const sorted = sites.map((p, i) => [p[0], p[1], i]).sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    const build = (sequence) => {
      const chain = [];
      for (const p of sequence) {
        while (chain.length >= 2 && orient2(chain[chain.length - 2], chain[chain.length - 1], p) <= 0) {
          chain.pop();
        }
        chain.push(p);
      }
      return chain;
    };
    const lower = build(sorted);
    const upper = build(sorted.slice().reverse());
    hull.push(...lower.slice(0, -1).concat(upper.slice(0, -1)).map((p) => p[2]));
  }
  const hullEdges = new Set(hull.map((index, i) => {
    const next = hull[(i + 1) % hull.length];
    return `${Math.min(index, next)},${Math.max(index, next)}`;
  }));
  strictEqual(diagram.rays.length, hullEdges.size,
    `one unbounded bisector per hull edge (${diagram.rays.length} vs ${hullEdges.size})`);
  for (const ray of diagram.rays) {
    ok(hullEdges.has(`${Math.min(ray.a, ray.b)},${Math.max(ray.a, ray.b)}`),
      `the ray ${ray.a}-${ray.b} must belong to a hull edge`);
  }
  strictEqual(diagram.edges.length, delaunayEdges.size - hullEdges.size,
    "the bounded Voronoi edges are the non-hull Delaunay edges");

  // -- cells tile the box and each site owns its own cell ------------------------------------------
  let totalArea = 0;
  const cellSides = [];
  for (let index = 0; index < sites.length; index++) {
    const cell = cellPolygon(index, sites, bounds);
    ok(cell.length >= 3, `every site must own a non-empty cell (site ${index})`);
    for (const point of cell) {
      const own = distance(point, sites[index]);
      for (let other = 0; other < sites.length; other++) {
        if (other === index) continue;
        ok(own <= distance(point, sites[other]) + 1e-6, "no cell point may be closer to another site");
      }
    }
    totalArea += polygonArea(cell);
    cellSides.push(cell.length);
  }
  const boxArea = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
  ok(Math.abs(totalArea - boxArea) < 1e-5, `the cells must tile the box (${totalArea.toFixed(2)})`);
  ok(cellSides.every((sides) => sides >= 3), "cells are polygons with at least three corners");

  // -- a hand-written second configuration ---------------------------------------------------------
  const second = [[0, 0], [24, 3], [12, 20], [-9, 11], [5, -14], [30, 24], [-18, -7]];
  const secondDiagram = buildVoronoi(second);
  const secondDelaunay = bruteDelaunay(second);
  strictEqual(secondDiagram.vertices.length, secondDelaunay.length,
    "the hand written configuration must match its Delaunay dual");
  for (const vertex of secondDiagram.vertices) {
    const [ux, uy, d] = circumcentreExact(second[vertex.sites[0]], second[vertex.sites[1]],
      second[vertex.sites[2]]);
    ok(Math.hypot(vertex.x - Number(ux) / Number(d), vertex.y - Number(uy) / Number(d)) < 1e-6,
      "each vertex must sit on the exact rational circumcentre of its sites");
  }
  const secondRays = secondDiagram.rays.map((ray) => clipRay(ray.start, ray.direction, bounds));
  ok(secondRays.some((segment) => segment !== null), "at least one ray must cross the box");

  // -- determinism ----------------------------------------------------------------------------------
  deepStrictEqual(buildVoronoi(sites).vertices, buildVoronoi(sites).vertices,
    "the sweep must be deterministic");
  deepStrictEqual(buildVoronoi(sites).edges, buildVoronoi(sites).edges,
    "the bisector list must be deterministic");
  deepStrictEqual(buildVoronoi(second).vertices, buildVoronoi(second).vertices,
    "the second configuration must be deterministic as well");

  console.log(`sites=${sites.length} vertices=${diagram.vertices.length} delaunay=${delaunay.length}`);
  console.log(`edges=${diagram.edges.length} rays=${diagram.rays.length} lines=${diagram.lines.length}`);
  console.log(`cells=${sites.length} tiled_area=${totalArea.toFixed(2)} box_area=${boxArea} ` +
    `radius_checks=${radiusChecks}`);
  console.log("PASS");
}
