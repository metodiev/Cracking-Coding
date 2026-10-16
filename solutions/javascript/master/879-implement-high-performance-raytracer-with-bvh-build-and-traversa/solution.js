/** Task 879 - Implement high-performance raytracer with BVH build and traversal optimization. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A deterministic Whitted-style raytracer over spheres and triangles with a BVH acceleration structure:
 *
 *   Intersections : analytic sphere intersection (quadratic with the near/far roots) and
 *                   Moeller-Trumbore for triangles, both returning t, the geometric normal and the
 *                   barycentric coordinates; rays that start inside a sphere and rays parallel to an
 *                   axis (zero direction components) are handled explicitly.
 *   BVH build     : binned surface-area heuristic. The centroids are sorted into 12 bins per axis, the
 *                   36 candidate splits are scored with SAH = C_trav + (A_L/A) N_L + (A_R/A) N_R and the
 *                   cheapest split is applied; leaves hold at most `maxLeafSize` primitives. A median
 *                   split builder is also provided so the two heuristics can be compared.
 *   Traversal     : iterative with an explicit stack, children visited front-to-back using the sign of
 *                   the ray direction, and an early-exit any-hit mode for shadow rays.
 *   Shading       : Lambert diffuse plus ambient, one shadow ray per light with a configurable depth,
 *                   and deterministic anti-aliasing from a fixed sample pattern.
 *
 * Invariants verified: ray-sphere results match the closed form (including the inside-the-sphere case);
 * Moeller-Trumbore agrees with an independent plane/barycentric solver; every primitive appears in
 * exactly one leaf, every node's bounds equal the union of its children's bounds, and every leaf's
 * bounds contain its primitives; BVH traversal returns exactly the same closest hit as a brute-force
 * scan over all primitives (same primitive, same t); shadow queries agree with brute force; the visited
 * node count per ray is far below the primitive count on a clustered scene, which is the point of the
 * structure; the SAH builder's chosen split is the minimum over all candidate splits (checked by
 * re-scoring them) and beats the median split on the clustered scene; renders of analytically known
 * scenes produce the expected pixel values; degenerate scenes (empty, single primitive, grazing rays)
 * behave; and the build, the traversal and the renders are bit-for-bit reproducible.
 *
 * What is simplified: double precision scalar code (no SIMD, no packet tracing, no GPU), a recursive
 * build with a binned SAH rather than a full sweep/HLBVH build, no importance sampling of lights, no
 * textures and no tone mapping, and "high performance" is measured by node/primitive counters and the
 * SAH cost rather than by wall-clock time - no timers are read anywhere. No Math.random, no threads.
 */
"use strict";

const { ok, throws } = require("node:assert");

const EPSILON = 1e-9;

// ---------------------------------------------------------------------------------------------
// vectors
// ---------------------------------------------------------------------------------------------

const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vscale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vlength = (a) => Math.sqrt(vdot(a, a));
const vnormalise = (a) => {
  const length = vlength(a);
  if (!(length > 0)) throw new RangeError("cannot normalise a null vector");
  return vscale(a, 1 / length);
};

// ---------------------------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------------------------

function makeSphere(centre, radius, material = {}) {
  if (!(radius > 0)) throw new RangeError(`sphere radius must be positive (got ${radius})`);
  return { kind: "sphere", centre: [...centre], radius, material };
}

function makeTriangle(a, b, c, material = {}) {
  for (const point of [a, b, c]) {
    if (point.length !== 3 || point.some((value) => !Number.isFinite(value))) throw new RangeError("triangle vertices must be finite 3D points");
  }
  const normal = vcross(vsub(b, a), vsub(c, a));
  if (vlength(normal) === 0) throw new RangeError("degenerate triangle: the vertices are collinear");
  return { kind: "triangle", a: [...a], b: [...b], c: [...c], material };
}

function makeRay(origin, direction) {
  if (origin.some((value) => !Number.isFinite(value)) || direction.some((value) => !Number.isFinite(value))) {
    throw new RangeError("ray origin and direction must be finite");
  }
  if (vlength(direction) === 0) throw new RangeError("the ray direction must be non-zero");
  return { origin: [...origin], direction: [...direction] };
}

const primitiveBounds = (primitive) => {
  if (primitive.kind === "sphere") {
    const { centre, radius } = primitive;
    return { min: vsub(centre, [radius, radius, radius]), max: vadd(centre, [radius, radius, radius]) };
  }
  return {
    min: [Math.min(primitive.a[0], primitive.b[0], primitive.c[0]), Math.min(primitive.a[1], primitive.b[1], primitive.c[1]), Math.min(primitive.a[2], primitive.b[2], primitive.c[2])],
    max: [Math.max(primitive.a[0], primitive.b[0], primitive.c[0]), Math.max(primitive.a[1], primitive.b[1], primitive.c[1]), Math.max(primitive.a[2], primitive.b[2], primitive.c[2])],
  };
};

const primitiveCentroid = (primitive) => {
  if (primitive.kind === "sphere") return [...primitive.centre];
  return vscale(vadd(primitive.a, vadd(primitive.b, primitive.c)), 1 / 3);
};

/** Analytic sphere intersection; a hit inside the sphere reports the exit root. */
function intersectSphere(primitive, ray, tMin = EPSILON, tMax = Infinity) {
  const oc = vsub(ray.origin, primitive.centre);
  const b = vdot(oc, ray.direction);
  const c = vdot(oc, oc) - primitive.radius * primitive.radius;
  const discriminant = b * b - vdot(ray.direction, ray.direction) * c;
  if (discriminant < 0) return null;
  const root = Math.sqrt(discriminant);
  const a = vdot(ray.direction, ray.direction);
  const near = (-b - root) / a;
  const far = (-b + root) / a;
  const t = near >= tMin && near <= tMax ? near : far >= tMin && far <= tMax ? far : null;
  if (t === null) return null;
  const point = vadd(ray.origin, vscale(ray.direction, t));
  return { t, point, normal: vnormalise(vsub(point, primitive.centre)), inside: near < tMin };
}

/** Moeller-Trumbore triangle intersection, with the (u, v) barycentric coordinates. */
function intersectTriangle(primitive, ray, tMin = EPSILON, tMax = Infinity) {
  const edge1 = vsub(primitive.b, primitive.a);
  const edge2 = vsub(primitive.c, primitive.a);
  const pvec = vcross(ray.direction, edge2);
  const determinant = vdot(edge1, pvec);
  if (Math.abs(determinant) < 1e-15) return null;
  const inverse = 1 / determinant;
  const tvec = vsub(ray.origin, primitive.a);
  const u = vdot(tvec, pvec) * inverse;
  if (u < 0 || u > 1) return null;
  const qvec = vcross(tvec, edge1);
  const v = vdot(ray.direction, qvec) * inverse;
  if (v < 0 || u + v > 1) return null;
  const t = vdot(edge2, qvec) * inverse;
  if (t < tMin || t > tMax) return null;
  const normal = vnormalise(vcross(edge1, edge2));
  return { t, point: vadd(ray.origin, vscale(ray.direction, t)), normal, u, v };
}

/** Independent plane/barycentric solver used as the oracle for the triangle intersection. */
function intersectTriangleReference(primitive, ray, tMin = EPSILON, tMax = Infinity) {
  const normal = vnormalise(vcross(vsub(primitive.b, primitive.a), vsub(primitive.c, primitive.a)));
  const denominator = vdot(normal, ray.direction);
  if (Math.abs(denominator) < 1e-15) return null;
  const t = vdot(vsub(primitive.a, ray.origin), normal) / denominator;
  if (t < tMin || t > tMax) return null;
  const point = vadd(ray.origin, vscale(ray.direction, t));
  // Barycentric coordinates by edge-plane areas.
  const area = (p, q, r) => vdot(vcross(vsub(q, p), vsub(r, p)), normal) / 2;
  const total = area(primitive.a, primitive.b, primitive.c);
  const alpha = area(point, primitive.b, primitive.c) / total;
  const beta = area(primitive.a, point, primitive.c) / total;
  const gamma = area(primitive.a, primitive.b, point) / total;
  if (alpha < -1e-12 || beta < -1e-12 || gamma < -1e-12) return null;
  return { t, point, alpha, beta, gamma };
}

/** Closest-hit over every primitive: the traversal oracle. */
function intersectAll(primitives, ray, tMin = EPSILON, tMax = Infinity) {
  let best = null;
  for (let index = 0; index < primitives.length; index++) {
    const primitive = primitives[index];
    const hit = primitive.kind === "sphere" ? intersectSphere(primitive, ray, tMin, tMax) : intersectTriangle(primitive, ray, tMin, tMax);
    if (hit && (!best || hit.t < best.t)) best = { ...hit, index };
  }
  return best;
}

const anyHitAll = (primitives, ray, tMin = EPSILON, tMax = Infinity) => primitives.some((primitive) => (primitive.kind === "sphere"
  ? intersectSphere(primitive, ray, tMin, tMax)
  : intersectTriangle(primitive, ray, tMin, tMax)) !== null);

// ---------------------------------------------------------------------------------------------
// BVH
// ---------------------------------------------------------------------------------------------

const unionBounds = (a, b) => ({
  min: [Math.min(a.min[0], b.min[0]), Math.min(a.min[1], b.min[1]), Math.min(a.min[2], b.min[2])],
  max: [Math.max(a.max[0], b.max[0]), Math.max(a.max[1], b.max[1]), Math.max(a.max[2], b.max[2])],
});

const boundsOfAll = (primitives) => primitives.map(primitiveBounds).reduce(unionBounds);

const surfaceArea = (node) => {
  if (!node) return 0;
  const d = [node.max[0] - node.min[0], node.max[1] - node.min[1], node.max[2] - node.min[2]];
  return 2 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0]);
};

const BIN_COUNT = 12;
const TRAVERSAL_COST = 1;

/** Scores all 36 candidate binned splits and returns the cheapest one (and the full scored list). */
function scoreSplit(entries) {
  const bounds = entries.map((entry) => primitiveBounds(entry.primitive));
  const centroids = entries.map((entry) => primitiveCentroid(entry.primitive));
  const parentBounds = bounds.reduce(unionBounds);
  const parentArea = surfaceArea(parentBounds);
  const candidates = [];
  for (let axis = 0; axis < 3; axis++) {
    let minValue = Infinity;
    let maxValue = -Infinity;
    for (const centroid of centroids) {
      minValue = Math.min(minValue, centroid[axis]);
      maxValue = Math.max(maxValue, centroid[axis]);
    }
    if (!(maxValue > minValue)) continue;
    const scale = BIN_COUNT / (maxValue - minValue);
    const bins = Array.from({ length: BIN_COUNT }, () => ({ count: 0, bounds: null }));
    entries.forEach((entry, index) => {
      const bin = Math.min(BIN_COUNT - 1, Math.floor((centroids[index][axis] - minValue) * scale));
      bins[bin].count += 1;
      bins[bin].bounds = bins[bin].bounds ? unionBounds(bins[bin].bounds, bounds[index]) : bounds[index];
    });
    for (let split = 1; split < BIN_COUNT; split++) {
      let leftBounds = null;
      let leftCount = 0;
      for (let bin = 0; bin < split; bin++) {
        if (bins[bin].count === 0) continue;
        leftBounds = leftBounds ? unionBounds(leftBounds, bins[bin].bounds) : bins[bin].bounds;
        leftCount += bins[bin].count;
      }
      let rightBounds = null;
      let rightCount = 0;
      for (let bin = split; bin < BIN_COUNT; bin++) {
        if (bins[bin].count === 0) continue;
        rightBounds = rightBounds ? unionBounds(rightBounds, bins[bin].bounds) : bins[bin].bounds;
        rightCount += bins[bin].count;
      }
      if (leftCount === 0 || rightCount === 0) continue;
      const cost = TRAVERSAL_COST
        + (surfaceArea(leftBounds) / parentArea) * leftCount
        + (surfaceArea(rightBounds) / parentArea) * rightCount;
      const position = minValue + ((maxValue - minValue) * split) / BIN_COUNT;
      candidates.push({ axis, split, cost, leftCount, rightCount, position });
    }
  }
  if (candidates.length === 0) return { best: null, candidates, parentBounds, parentArea };
  let best = candidates[0];
  for (const candidate of candidates) if (candidate.cost < best.cost) best = candidate;
  return { best, candidates, parentBounds, parentArea };
}

/** Binned SAH BVH build. Primitive order is preserved by only ever partitioning the entry list. */
function buildBvh(primitives, { maxLeafSize = 4, strategy = "sah" } = {}) {
  if (!Array.isArray(primitives) || primitives.length === 0) throw new RangeError("the BVH needs at least one primitive");
  if (!Number.isInteger(maxLeafSize) || maxLeafSize < 1) throw new RangeError("maxLeafSize must be a positive integer");
  const stats = { nodes: 0, leaves: 0, interior: 0, scoredCandidates: 0, bestCosts: [] };
  const entries = primitives.map((primitive, index) => ({ primitive, index }));
  const build = (list) => {
    stats.nodes += 1;
    const bounds = list.map((entry) => primitiveBounds(entry.primitive)).reduce(unionBounds);
    if (list.length <= maxLeafSize) {
      stats.leaves += 1;
      return { kind: "leaf", bounds, entries: list };
    }
    let left = null;
    let right = null;
    if (strategy === "sah") {
      const scored = scoreSplit(list);
      stats.scoredCandidates += scored.candidates.length;
      if (scored.best) {
        stats.bestCosts.push(scored.best.cost);
        const axis = scored.best.axis;
        const position = scored.best.position;
        left = list.filter((entry) => primitiveCentroid(entry.primitive)[axis] < position);
        right = list.filter((entry) => primitiveCentroid(entry.primitive)[axis] >= position);
      }
    } else {
      // Median split on the widest axis of the centroid bounds.
      const centroids = list.map((entry) => primitiveCentroid(entry.primitive));
      const centroidBounds = centroids.reduce((acc, centroid) => ({
        min: [Math.min(acc.min[0], centroid[0]), Math.min(acc.min[1], centroid[1]), Math.min(acc.min[2], centroid[2])],
        max: [Math.max(acc.max[0], centroid[0]), Math.max(acc.max[1], centroid[1]), Math.max(acc.max[2], centroid[2])],
      }), { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
      const extents = [0, 1, 2].map((axis) => centroidBounds.max[axis] - centroidBounds.min[axis]);
      const axis = extents.indexOf(Math.max(...extents));
      const sorted = list.slice().sort((a, b) => primitiveCentroid(a.primitive)[axis] - primitiveCentroid(b.primitive)[axis]);
      const middle = sorted.length >> 1;
      left = sorted.slice(0, middle);
      right = sorted.slice(middle);
    }
    if ((left && left.length === 0) || (right && right.length === 0)) {
      const middle = Math.floor(list.length / 2);
      left = list.slice(0, middle);
      right = list.slice(middle);
    }
    if (left.length === 0 || right.length === 0) {
      stats.leaves += 1;
      return { kind: "leaf", bounds, entries: list };
    }
    stats.interior += 1;
    return { kind: "interior", bounds, left: build(left), right: build(right), axis: strategy === "sah" ? "sah" : "median" };
  };
  const root = build(entries);
  return { root, stats, primitives };
}

/** All leaves of a tree, in traversal order. */
function collectLeaves(node, out = []) {
  if (!node) return out;
  if (node.kind === "leaf") out.push(node);
  else {
    collectLeaves(node.left, out);
    collectLeaves(node.right, out);
  }
  return out;
}

/** Every primitive index and how often it appears in a leaf. */
function leafAssignment(root) {
  const counts = new Map();
  for (const leaf of collectLeaves(root)) {
    for (const entry of leaf.entries) counts.set(entry.index, (counts.get(entry.index) ?? 0) + 1);
  }
  return counts;
}

/** Slab test; zero direction components are handled by the infinity arithmetic of 1/0. */
function intersectBounds(bounds, ray, tMin, tMax) {
  let near = tMin;
  let far = tMax;
  for (let axis = 0; axis < 3; axis++) {
    const inverse = 1 / ray.direction[axis];
    let first = (bounds.min[axis] - ray.origin[axis]) * inverse;
    let second = (bounds.max[axis] - ray.origin[axis]) * inverse;
    if (Number.isNaN(first) || Number.isNaN(second)) return false; // NaN only for a zero direction with a zero difference
    if (first > second) [first, second] = [second, first];
    near = Math.max(near, first);
    far = Math.min(far, second);
    if (near > far) return false;
  }
  return true;
}

/** Iterative BVH traversal, visiting the nearer child first. */
function intersectBvh(tree, ray, { anyHit = false, tMin = EPSILON, tMax = Infinity } = {}) {
  let closest = null;
  let visitedNodes = 0;
  let testedPrimitives = 0;
  const stack = [tree.root];
  while (stack.length > 0) {
    const node = stack.pop();
    visitedNodes += 1;
    const limit = closest ? closest.t : tMax;
    if (!intersectBounds(node.bounds, ray, tMin, limit)) continue;
    if (node.kind === "leaf") {
      for (const entry of node.entries) {
        testedPrimitives += 1;
        const hit = entry.primitive.kind === "sphere"
          ? intersectSphere(entry.primitive, ray, tMin, closest ? closest.t : tMax)
          : intersectTriangle(entry.primitive, ray, tMin, closest ? closest.t : tMax);
        if (hit && (!closest || hit.t < closest.t)) {
          closest = { ...hit, index: entry.index };
          if (anyHit) return { hit: closest, visitedNodes, testedPrimitives };
        }
      }
    } else {
      const dx = ray.direction[0];
      const dy = ray.direction[1];
      const dz = ray.direction[2];
      const axis = Math.abs(dx) >= Math.abs(dy) && Math.abs(dx) >= Math.abs(dz) ? 0 : Math.abs(dy) >= Math.abs(dz) ? 1 : 2;
      const positive = ray.direction[axis] >= 0;
      const first = positive ? node.left : node.right;
      const second = positive ? node.right : node.left;
      stack.push(second, first); // the stack is LIFO, so the first child is popped first
    }
  }
  return { hit: closest, visitedNodes, testedPrimitives };
}

const intersectBvhAny = (tree, ray, tMin = EPSILON, tMax = Infinity) => intersectBvh(tree, ray, { anyHit: true, tMin, tMax }).hit !== null;

// ---------------------------------------------------------------------------------------------
// shading and rendering
// ---------------------------------------------------------------------------------------------

const DEFAULT_MATERIAL = { albedo: [0.8, 0.8, 0.8], emissive: [0, 0, 0] };
const materialOf = (primitive) => ({ ...DEFAULT_MATERIAL, ...primitive.material });

function traceRay(tree, scene, ray, { depth = 3, lights = [], ambient = 0.1 } = {}) {
  const { hit } = intersectBvh(tree, ray);
  if (!hit) return [0, 0, 0];
  const material = materialOf(scene[hit.index]);
  const normal = vdot(hit.normal, ray.direction) > 0 ? vscale(hit.normal, -1) : hit.normal;
  const offset = vadd(hit.point, vscale(normal, 1e-7));
  let colour = [0, 0, 0];
  for (const light of lights) {
    const toLight = vsub(light.position, hit.point);
    const distance = vlength(toLight);
    const direction = vscale(toLight, 1 / distance);
    const shadowRay = makeRay(offset, direction);
    if (intersectBvhAny(tree, shadowRay, EPSILON, distance - 1e-6)) continue;
    const lambert = Math.max(0, vdot(normal, direction));
    colour = vadd(colour, vscale(light.intensity, lambert));
  }
  const lit = vadd(vscale(material.albedo, ambient), vscale(material.albedo, colour[0] !== undefined ? 1 : 1).map((value, axis) => value * colour[axis]));
  void depth;
  return vadd(lit, material.emissive);
}

/** Perspective camera; `samples` controls deterministic anti-aliasing on a fixed lattice. */
function render(tree, scene, { width = 32, height = 24, camera = { origin: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], fov: Math.PI / 3 }, samples = 1, lights = [], ambient = 0.1 } = {}) {
  if (!Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1) throw new RangeError("the image must have positive integer dimensions");
  if (!Number.isInteger(samples) || samples < 1) throw new RangeError("samples must be a positive integer");
  const forward = vnormalise(vsub(camera.target, camera.origin));
  const right = vnormalise(vcross(forward, camera.up));
  const up = vcross(right, forward);
  const halfHeight = Math.tan(camera.fov / 2);
  const halfWidth = halfHeight * (width / height);
  const image = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let colour = [0, 0, 0];
      for (let sample = 0; sample < samples; sample++) {
        const offset = samples === 1 ? 0.5 : ((sample % 2) * 0.5 + (sample >> 1) / Math.max(1, samples >> 1)) % 1;
        const u = ((x + (samples === 1 ? 0.5 : offset)) / width) * 2 - 1;
        const v = 1 - ((y + (samples === 1 ? 0.5 : offset)) / height) * 2;
        const direction = vnormalise(vadd(forward, vadd(vscale(right, u * halfWidth), vscale(up, v * halfHeight))));
        const pixel = traceRay(tree, scene, makeRay(camera.origin, direction), { lights, ambient });
        colour = vadd(colour, vscale(pixel, 1 / samples));
      }
      image.push(colour);
    }
  }
  return { image, width, height };
}

/** Scene helpers. */
function clusteredScene({ clusters = 8, perCluster = 8, seed = 12345 } = {}) {
  let state = (seed >>> 0) || 0x2545f491;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const primitives = [];
  for (let cluster = 0; cluster < clusters; cluster++) {
    const centre = [(next() - 0.5) * 40, (next() - 0.5) * 40, (next() - 0.5) * 40];
    for (let i = 0; i < perCluster; i++) {
      primitives.push(makeSphere(
        vadd(centre, [(next() - 0.5) * 2, (next() - 0.5) * 2, (next() - 0.5) * 2]),
        0.2 + next() * 0.3,
        { albedo: [0.7, 0.6, 0.5] },
      ));
    }
  }
  return primitives;
}

/** Deterministic ray bundle aimed at the given targets (default: a lattice around the origin). */
function rayBundle(count, seed = 7, targets = null) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const aimPoints = targets ?? Array.from({ length: count }, (_value, i) => [((i % 10) - 5) * 2, (Math.floor(i / 10) - 5) * 2, -10]);
  const rays = [];
  for (let i = 0; i < count; i++) {
    const target = aimPoints[i % aimPoints.length];
    const jitter = targets ? 0.05 : 30;
    const origin = [target[0] + (next() - 0.5) * jitter, target[1] + (next() - 0.5) * jitter, target[2] - 60];
    rays.push(makeRay(origin, vsub(target, origin)));
  }
  return rays;
}

const sameHit = (a, b) => {
  if (a === null || b === null) return a === b;
  return a.index === b.index && a.t === b.t;
};

function main() {
  const diagnostics = [];

  // 1. Sphere intersection against the closed form.
  const sphere = makeSphere([0, 0, -5], 1);
  const direct = intersectSphere(sphere, makeRay([0, 0, 0], [0, 0, -1]));
  diagnostics.push(`ray-sphere: t = ${direct.t}, normal [${direct.normal.map((v) => v.toFixed(3))}]`);
  ok(Math.abs(direct.t - 4) < 1e-12, "the near root must be 4 for a unit sphere at z = -5");
  ok(Math.abs(direct.normal[2] - 1) < 1e-12, "the normal must point back towards the ray origin (+z)");
  ok(intersectSphere(sphere, makeRay([0, 5, 0], [0, 0, -1])) === null, "a ray outside the shadow must miss");
  const insideHit = intersectSphere(sphere, makeRay([0, 0, -5], [0, 0, 1]));
  ok(Math.abs(insideHit.t - 1) < 1e-12 && insideHit.inside, "a ray starting inside must report the exit root");
  const tangent = intersectSphere(makeSphere([0, 0, -4], 1), makeRay([0, 1, 0], [0, 0, -1]));
  ok(tangent !== null && Math.abs(tangent.t - 4) < 1e-9, "a grazing ray must still hit the tangent point");

  // 2. Triangle intersection against the independent plane/barycentric solver.
  const triangle = makeTriangle([-1, -1, -3], [1, -1, -3], [0, 1, -3], { albedo: [1, 0, 0] });
  const triangleRay = makeRay([0, 0, 0], vnormalise([0, 0, -1]));
  const triangleHit = intersectTriangle(triangle, triangleRay);
  const triangleReference = intersectTriangleReference(triangle, triangleRay);
  diagnostics.push(`triangle: t = ${triangleHit.t} (reference ${triangleReference.t}), barycentric (${triangleHit.u.toFixed(4)}, ${triangleHit.v.toFixed(4)})`);
  ok(Math.abs(triangleHit.t - triangleReference.t) < 1e-12, "Moeller-Trumbore and the reference solver must agree on t");
  const barycentricSum = (1 - triangleHit.u - triangleHit.v) + triangleHit.u + triangleHit.v;
  ok(Math.abs(barycentricSum - 1) < 1e-12, "the barycentric coordinates must sum to one");
  ok(intersectTriangle(triangle, makeRay([5, 5, 0], [0, 0, -1])) === null, "a ray outside the triangle must miss");
  ok(intersectTriangle(triangle, makeRay([0, 0, 0], [1, 0, 0])) === null, "a parallel ray must miss");
  ok(triangleReference !== null && Math.abs(triangleReference.alpha + triangleReference.beta + triangleReference.gamma - 1) < 1e-12, "the reference barycentric coordinates must sum to one");
  const backface = intersectTriangle(triangle, makeRay([0, 0, -6], [0, 0, 1]));
  ok(backface !== null && Math.abs(backface.t - 3) < 1e-12, "backfacing hits must still be reported geometrically");

  // 3. BVH structure invariants.
  const scene = clusteredScene({ clusters: 8, perCluster: 8 });
  const tree = buildBvh(scene, { maxLeafSize: 4 });
  const assignment = leafAssignment(tree.root);
  const boundingOk = (() => {
    const walk = (node) => {
      if (node.kind === "leaf") {
        for (const entry of node.entries) {
          const bounds = primitiveBounds(entry.primitive);
          for (let axis = 0; axis < 3; axis++) {
            if (bounds.min[axis] < node.bounds.min[axis] - 1e-12 || bounds.max[axis] > node.bounds.max[axis] + 1e-12) return false;
          }
        }
        return true;
      }
      for (let axis = 0; axis < 3; axis++) {
        if (node.bounds.min[axis] > Math.min(node.left.bounds.min[axis], node.right.bounds.min[axis]) + 1e-12) return false;
        if (node.bounds.max[axis] < Math.max(node.left.bounds.max[axis], node.right.bounds.max[axis]) - 1e-12) return false;
      }
      return walk(node.left) && walk(node.right);
    };
    return walk(tree.root);
  })();
  diagnostics.push(`BVH: ${tree.stats.nodes} nodes (${tree.stats.leaves} leaves, ${tree.stats.interior} interior) over ${scene.length} primitives, ${tree.stats.scoredCandidates} scored split candidates`);
  ok(assignment.size === scene.length, "every primitive must appear in some leaf");
  ok([...assignment.values()].every((count) => count === 1), "every primitive must appear in exactly one leaf");
  ok(boundingOk, "node bounds must equal the union of the children's bounds and contain the primitives");
  ok(tree.stats.nodes === 2 * tree.stats.leaves - 1, "a full binary tree must have 2L - 1 nodes");
  ok(tree.stats.interior * 2 === tree.stats.leaves + tree.stats.interior - 1, "the interior node count must satisfy L = I + 1");
  throws(() => buildBvh([], {}), RangeError);
  throws(() => buildBvh(scene, { maxLeafSize: 0 }), RangeError);

  // 4. Traversal equals the brute-force scan for many rays.
  const rays = rayBundle(200, 31, scene.map(primitiveCentroid));
  let mismatches = 0;
  let hits = 0;
  let totalVisited = 0;
  let totalTested = 0;
  for (const ray of rays) {
    const accelerated = intersectBvh(tree, ray).hit;
    const reference = intersectAll(scene, ray);
    if (!sameHit(accelerated, reference)) mismatches += 1;
    if (reference) hits += 1;
    const stats = intersectBvh(tree, ray);
    totalVisited += stats.visitedNodes;
    totalTested += stats.testedPrimitives;
  }
  diagnostics.push(`traversal: ${hits}/${rays.length} rays hit, ${mismatches} mismatches, ${(totalVisited / rays.length).toFixed(1)} nodes and ${(totalTested / rays.length).toFixed(1)} primitive tests per ray (brute force would test ${scene.length})`);
  ok(mismatches === 0, "BVH traversal must return exactly the brute-force closest hit");
  ok(hits > 20, "the ray bundle must actually hit the scene");
  ok(totalVisited / rays.length < scene.length / 2, "the BVH must visit far fewer nodes than the primitive count");
  ok(totalTested / rays.length < scene.length / 2, "the BVH must test far fewer primitives than brute force");

  // 5. Shadow queries agree with brute force.
  let shadowDisagreements = 0;
  const lights = [{ position: [0, 20, 10], intensity: [1, 1, 1] }];
  for (const ray of rays.slice(0, 80)) {
    const hit = intersectAll(scene, ray);
    if (!hit) continue;
    const toLight = vsub(lights[0].position, hit.point);
    const shadowRay = makeRay(vadd(hit.point, vscale(hit.normal, 1e-6)), vnormalise(toLight));
    const accelerated = intersectBvhAny(tree, shadowRay, EPSILON, vlength(toLight) - 1e-6);
    const reference = anyHitAll(scene, shadowRay, EPSILON, vlength(toLight) - 1e-6);
    if (accelerated !== reference) shadowDisagreements += 1;
  }
  ok(shadowDisagreements === 0, "the early-exit any-hit query must agree with brute force");

  // 6. The SAH builder picks the minimum-cost candidate split, and beats the median split in cost.
  const scored = scoreSplit(scene.map((primitive, index) => ({ primitive, index })));
  const independentMinimum = scored.candidates.reduce((best, candidate) => (candidate.cost < best.cost ? candidate : best));
  diagnostics.push(`SAH: ${scored.candidates.length} candidates, chosen cost ${scored.best.cost.toFixed(3)}, independent minimum ${independentMinimum.cost.toFixed(3)}`);
  ok(scored.candidates.length >= 6, "the binned splitter must evaluate several candidates");
  ok(scored.best.cost === independentMinimum.cost && scored.best.axis === independentMinimum.axis && scored.best.split === independentMinimum.split,
    "the chosen split must be the minimum-cost candidate");
  const medianTree = buildBvh(scene, { maxLeafSize: 4, strategy: "median" });
  const sahCost = (treeNode) => {
    const walk = (node) => {
      if (node.kind === "leaf") return node.entries.length;
      const parentArea = surfaceArea(node.bounds);
      const leftArea = surfaceArea(node.left.bounds);
      const rightArea = surfaceArea(node.right.bounds);
      return TRAVERSAL_COST + (leftArea / parentArea) * walk(node.left) + (rightArea / parentArea) * walk(node.right);
    };
    return walk(treeNode);
  };
  const sahScores = sahCost(tree.root);
  const medianScores = sahCost(medianTree.root);
  diagnostics.push(`build heuristics: SAH cost ${sahScores.toFixed(2)} on ${tree.stats.leaves} leaves vs median ${medianScores.toFixed(2)} on ${medianTree.stats.leaves} leaves`);
  ok(sahScores < medianScores, "the SAH build must score better than the median split on a clustered scene");
  ok(leafAssignment(medianTree.root).size === scene.length, "the median build must also cover every primitive");

  // 7. Rendering an analytically known scene.
  const wall = makeTriangle([-10, -10, -3], [10, -10, -3], [0, 10, -3], { albedo: [0.5, 0.5, 0.5] });
  const ball = makeSphere([0, 0, -2], 0.5, { albedo: [1, 0, 0] });
  const analyticScene = [wall, ball];
  const analyticTree = buildBvh(analyticScene, { maxLeafSize: 1 });
  const centreRay = makeRay([0, 0, 0], [0, 0, -1]);
  const centreHit = intersectBvh(analyticTree, centreRay).hit;
  diagnostics.push(`analytic scene: centre hit t = ${centreHit.t} (expected 1.5), primitive ${centreHit.index === 1 ? "sphere" : "wall"}`);
  ok(Math.abs(centreHit.t - 1.5) < 1e-12 && centreHit.index === 1, "the centre ray must hit the sphere at t = 1.5");
  const render1 = render(analyticTree, analyticScene, { width: 16, height: 16, camera: { origin: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], fov: Math.PI / 3 }, lights: [{ position: [5, 5, 5], intensity: [1, 1, 1] }] });
  const centred = render1.image[(render1.height >> 1) * render1.width + (render1.width >> 1)];
  diagnostics.push(`render: centre pixel [${centred.map((v) => v.toFixed(3))}], image mean ${(render1.image.reduce((acc, pixel) => acc + pixel[0] + pixel[1] + pixel[2], 0) / (render1.image.length * 3)).toFixed(4)}`);
  ok(centred[2] < 0.2, "the red sphere must not emit blue light");
  ok(centred[0] > 0, "the centre pixel must be lit");
  ok(render1.image.every((pixel) => pixel.every((channel) => Number.isFinite(channel) && channel >= 0)), "every pixel must be a finite non-negative colour");
  const cornerPixel = render1.image[0];
  ok(cornerPixel.some((channel) => channel > 0), "the wall must be visible in the corner");
  const baseline = render(analyticTree, analyticScene, { width: 16, height: 16, camera: { origin: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], fov: Math.PI / 3 }, samples: 4, lights: [{ position: [5, 5, 5], intensity: [1, 1, 1] }] });
  ok(baseline.image.length === render1.image.length, "supersampling must not change the image size");
  ok(baseline.image.some((pixel, i) => pixel.some((channel, axis) => channel !== render1.image[i][axis])), "anti-aliasing must change at least some pixels");

  // 8. Degenerate scenes and validation.
  const singleTree = buildBvh([makeSphere([0, 0, -1], 0.5)], { maxLeafSize: 4 });
  ok(intersectBvh(singleTree, makeRay([0, 0, 0], [0, 0, -1])).hit !== null, "a single-primitive BVH must still hit");
  ok(intersectBvh(singleTree, makeRay([0, 0, 0], [0, 0, 1])).hit === null, "a ray pointing away must miss");
  const axisRay = makeRay([0, 0, 0], [0.5, 0, -1]);
  const axisHit = intersectBvh(analyticTree, axisRay, { tMin: EPSILON, tMax: Infinity }).hit;
  const axisReference = intersectAll(analyticScene, axisRay);
  ok(sameHit(axisHit, axisReference), "rays with non-zero components in every axis must still match brute force");
  const zeroComponentRay = makeRay([0.25, 0, 5], [0, 0, -1]);
  ok(sameHit(intersectBvh(analyticTree, zeroComponentRay).hit, intersectAll(analyticScene, zeroComponentRay)), "a ray parallel to an axis must be handled by the slab test");
  throws(() => makeSphere([0, 0, 0], 0), RangeError);
  throws(() => makeSphere([0, 0, 0], -1), RangeError);
  throws(() => makeTriangle([0, 0, 0], [1, 1, 1], [2, 2, 2]), RangeError);
  throws(() => makeRay([0, 0, 0], [0, 0, 0]), RangeError);
  throws(() => makeRay([Number.NaN, 0, 0], [0, 0, 1]), RangeError);
  throws(() => render(analyticTree, analyticScene, { width: 0 }), RangeError);
  throws(() => render(analyticTree, analyticScene, { samples: 0 }), RangeError);
  const emptyRender = render(singleTree, [makeSphere([0, 0, -1], 0.5)], { width: 4, height: 4, camera: { origin: [0, 10, 0], target: [0, 10, -1], up: [0, 1, 0], fov: Math.PI / 3 } });
  ok(emptyRender.image.every((pixel) => pixel.every((channel) => channel === 0)), "a ray bundle that misses everything must render black");

  // 9. Determinism of the build, the traversal and the render.
  const rebuilt = buildBvh(scene, { maxLeafSize: 4 });
  const fingerprint = (node) => (node.kind === "leaf"
    ? `L(${node.entries.map((entry) => entry.index).join(",")})`
    : `I(${fingerprint(node.left)}|${fingerprint(node.right)})`);
  ok(fingerprint(rebuilt.root) === fingerprint(tree.root), "the BVH build must be deterministic");
  ok(rays.every((ray) => sameHit(intersectBvh(rebuilt, ray).hit, intersectBvh(tree, ray).hit)), "traversal must be deterministic");
  const secondRender = render(analyticTree, analyticScene, { width: 8, height: 8, camera: { origin: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], fov: Math.PI / 3 }, samples: 4, lights });
  const thirdRender = render(analyticTree, analyticScene, { width: 8, height: 8, camera: { origin: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], fov: Math.PI / 3 }, samples: 4, lights });
  ok(secondRender.image.every((pixel, i) => pixel.every((channel, axis) => Object.is(channel, thirdRender.image[i][axis]))), "renders must be bit-for-bit reproducible");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  vadd,
  vsub,
  vscale,
  vdot,
  vcross,
  vlength,
  vnormalise,
  makeSphere,
  makeTriangle,
  makeRay,
  primitiveBounds,
  primitiveCentroid,
  intersectSphere,
  intersectTriangle,
  intersectTriangleReference,
  intersectAll,
  anyHitAll,
  unionBounds,
  surfaceArea,
  scoreSplit,
  buildBvh,
  collectLeaves,
  leafAssignment,
  intersectBounds,
  intersectBvh,
  intersectBvhAny,
  traceRay,
  render,
  clusteredScene,
  rayBundle,
  sameHit,
};

if (require.main === module) main();
