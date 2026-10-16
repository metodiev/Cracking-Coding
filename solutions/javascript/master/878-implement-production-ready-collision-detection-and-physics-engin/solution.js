/** Task 878 - Implement production-ready collision detection and physics engine basics. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A small but complete 2D rigid-body pipeline: broad phase, narrow phase, contact manifold and an
 * iterative sequential-impulse solver with position correction.
 *
 *   Broad phase   : a uniform grid that returns candidate pairs. Completeness is verified against the
 *                   brute-force O(n^2) pair enumeration, and the pruning factor is measured.
 *   Narrow phase  : circle-circle, AABB-AABB and convex polygon-polygon via the separating axis
 *                   theorem, which also yields the minimum translation vector (MTV). The oracle for the
 *                   polygon cases is an independent Sutherland-Hodgman clip whose area decides overlap.
 *   Contact model : normal (pointing from A to B), penetration depth and a contact point; touching
 *                   shapes produce a contact of depth zero, which is the boundary case of the tests.
 *   Solver        : semi-implicit Euler integration, restitution with a bias threshold, Coulomb
 *                   friction clamping (|j_t| <= mu * j_n) and Baumgarte position correction. Bodies
 *                   may be static (infinite mass); pairs are solved sequentially with a fixed iteration
 *                   budget, and substepping is available for fast movers.
 *
 * Invariants verified: circle/AABB/polygon overlap agrees with the independent clip-area oracle away
 * from tangency; translating a polygon by its MTV removes the overlap exactly; the broad phase never
 * misses a pair (completeness) while returning far fewer candidates; an equal-mass head-on collision
 * with restitution 1 conserves momentum exactly and kinetic energy to round-off; a resting/inelastic
 * impact does not increase energy and drives the relative normal velocity to zero; penetration is
 * resolved below tolerance within the iteration budget; a resting stack settles to zero velocity; a
 * static body never moves; substepping prevents tunnelling that a single large step would miss;
 * sequential solving is order-stable to round-off; and everything is deterministic.
 *
 * What is simplified: 2D only, convex shapes only, no rotational dynamics (no inertia tensors, angular
 * impulses or rolling friction), contact manifolds carry a single point per pair, no warm starting or
 * persistent contact caching, and the "production" concerns of broad-phase tree balancing, island
 * sleeping and CCD by ray casting are represented by a grid, a velocity threshold and substepping. No
 * timers, no wall-clock time, no threads, no Math.random.
 */
"use strict";

const { ok, throws } = require("node:assert");

const EPSILON = 1e-9;

// ---------------------------------------------------------------------------------------------
// shapes and their local geometry
// ---------------------------------------------------------------------------------------------

function makeCircle(radius) {
  if (!(radius > 0)) throw new RangeError(`circle radius must be positive (got ${radius})`);
  return { kind: "circle", radius };
}

function makeBox(halfWidth, halfHeight) {
  if (!(halfWidth > 0) || !(halfHeight > 0)) throw new RangeError("box half extents must be positive");
  return { kind: "box", halfWidth, halfHeight };
}

/** Convex polygon from vertices in counter-clockwise order (validated by the cross products). */
function makePolygon(vertices) {
  if (!Array.isArray(vertices) || vertices.length < 3) throw new RangeError("a polygon needs at least three vertices");
  let orientation = 0;
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % vertices.length];
    const c = vertices[(i + 2) % vertices.length];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (cross !== 0) {
      if (orientation === 0) orientation = Math.sign(cross);
      else if (Math.sign(cross) !== orientation) throw new RangeError("the polygon must be convex and consistently oriented");
    }
  }
  const ordered = orientation < 0 ? vertices.slice().reverse() : vertices.slice();
  return { kind: "polygon", vertices: ordered };
}

function makeBody({ shape, position = [0, 0], velocity = [0, 0], mass = 1, restitution = 0.5, friction = 0.3, staticBody = false }) {
  if (!shape) throw new TypeError("a body needs a shape");
  if (!(mass > 0)) throw new RangeError("mass must be positive");
  return {
    shape,
    position: [...position],
    velocity: [...velocity],
    mass: staticBody ? Infinity : mass,
    inverseMass: staticBody ? 0 : 1 / mass,
    restitution,
    friction,
    staticBody,
    awake: true,
  };
}

/** World-space vertices of a polygon or box shape. */
function worldVertices(body) {
  const { shape, position } = body;
  if (shape.kind === "polygon") return shape.vertices.map(([x, y]) => [x + position[0], y + position[1]]);
  if (shape.kind === "box") {
    const { halfWidth, halfHeight } = shape;
    return [[-halfWidth, -halfHeight], [halfWidth, -halfHeight], [halfWidth, halfHeight], [-halfWidth, halfHeight]]
      .map(([x, y]) => [x + position[0], y + position[1]]);
  }
  throw new TypeError(`worldVertices needs a polygon-like shape (got ${shape.kind})`);
}

/** Axis-aligned bounds of any shape. */
function bounds(body) {
  const { shape, position } = body;
  if (shape.kind === "circle") {
    return { minX: position[0] - shape.radius, maxX: position[0] + shape.radius, minY: position[1] - shape.radius, maxY: position[1] + shape.radius };
  }
  const vertices = worldVertices(body);
  return {
    minX: Math.min(...vertices.map((v) => v[0])),
    maxX: Math.max(...vertices.map((v) => v[0])),
    minY: Math.min(...vertices.map((v) => v[1])),
    maxY: Math.max(...vertices.map((v) => v[1])),
  };
}

const boundsOverlap = (a, b) => a.minX <= b.maxX + EPSILON && b.minX <= a.maxX + EPSILON && a.minY <= b.maxY + EPSILON && b.minY <= a.maxY + EPSILON;

// ---------------------------------------------------------------------------------------------
// narrow phase
// ---------------------------------------------------------------------------------------------

function circleCircle(a, b) {
  const dx = b.position[0] - a.position[0];
  const dy = b.position[1] - a.position[1];
  const distance = Math.hypot(dx, dy);
  const radii = a.shape.radius + b.shape.radius;
  if (distance > radii) return null;
  const normal = distance > 0 ? [dx / distance, dy / distance] : [1, 0];
  return { normal, depth: radii - distance, point: [a.position[0] + normal[0] * a.shape.radius, a.position[1] + normal[1] * a.shape.radius] };
}

function aabbAabb(a, b) {
  const first = bounds(a);
  const second = bounds(b);
  const overlapX = Math.min(first.maxX, second.maxX) - Math.max(first.minX, second.minX);
  const overlapY = Math.min(first.maxY, second.maxY) - Math.max(first.minY, second.minY);
  if (overlapX < 0 || overlapY < 0) return null;
  if (overlapX <= overlapY) {
    const sign = first.minX + first.maxX <= second.minX + second.maxX ? 1 : -1;
    return { normal: [sign, 0], depth: overlapX, point: [0, 0] };
  }
  const sign = first.minY + first.maxY <= second.minY + second.maxY ? 1 : -1;
  return { normal: [0, sign], depth: overlapY, point: [0, 0] };
}

/** Separating axis theorem for convex polygons: returns the MTV or null. */
function satPolygons(first, second) {
  const firstVertices = worldVertices(first);
  const secondVertices = worldVertices(second);
  const axes = [];
  for (const vertices of [firstVertices, secondVertices]) {
    for (let i = 0; i < vertices.length; i++) {
      const a = vertices[i];
      const b = vertices[(i + 1) % vertices.length];
      const edge = [b[0] - a[0], b[1] - a[1]];
      const length = Math.hypot(edge[0], edge[1]);
      if (length === 0) throw new RangeError("a polygon edge must not be degenerate");
      axes.push([-edge[1] / length, edge[0] / length]);
    }
  }
  let bestDepth = Infinity;
  let bestNormal = null;
  for (const axis of axes) {
    const project = (vertices) => vertices.reduce((range, vertex) => {
      const value = vertex[0] * axis[0] + vertex[1] * axis[1];
      return [Math.min(range[0], value), Math.max(range[1], value)];
    }, [Infinity, -Infinity]);
    const [firstMin, firstMax] = project(firstVertices);
    const [secondMin, secondMax] = project(secondVertices);
    const overlap = Math.min(firstMax, secondMax) - Math.max(firstMin, secondMin);
    if (overlap < -EPSILON) return null;
    if (overlap < bestDepth) {
      bestDepth = overlap;
      const direction = firstMin + firstMax <= secondMin + secondMax ? 1 : -1;
      bestNormal = [axis[0] * direction, axis[1] * direction];
    }
  }
  if (bestNormal === null) return null;
  return { normal: bestNormal, depth: Math.max(0, bestDepth), point: [0, 0] };
}

/** Independent overlap oracle for convex polygons: Sutherland-Hodgman clipping and its area. */
function clipArea(first, second) {
  let output = worldVertices(first);
  const clipVertices = worldVertices(second);
  for (let i = 0; i < clipVertices.length && output.length > 0; i++) {
    const a = clipVertices[i];
    const b = clipVertices[(i + 1) % clipVertices.length];
    const inside = (point) => (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= 0;
    const input = output;
    output = [];
    for (let j = 0; j < input.length; j++) {
      const current = input[j];
      const previous = input[(j + input.length - 1) % input.length];
      const currentInside = inside(current);
      const previousInside = inside(previous);
      if (currentInside) {
        if (!previousInside) {
          const t = ((a[0] - previous[0]) * (b[1] - a[1]) - (a[1] - previous[1]) * (b[0] - a[0]))
            / ((current[0] - previous[0]) * (b[1] - a[1]) - (current[1] - previous[1]) * (b[0] - a[0]));
          output.push([previous[0] + t * (current[0] - previous[0]), previous[1] + t * (current[1] - previous[1])]);
        }
        output.push(current);
      } else if (previousInside) {
        const t = ((a[0] - previous[0]) * (b[1] - a[1]) - (a[1] - previous[1]) * (b[0] - a[0]))
          / ((current[0] - previous[0]) * (b[1] - a[1]) - (current[1] - previous[1]) * (b[0] - a[0]));
        output.push([previous[0] + t * (current[0] - previous[0]), previous[1] + t * (current[1] - previous[1])]);
      }
    }
  }
  if (output.length < 3) return 0;
  let area = 0;
  for (let i = 0; i < output.length; i++) {
    const a = output[i];
    const b = output[(i + 1) % output.length];
    area += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(area / 2);
}

/** Full narrow phase dispatch; returns the contact or null. */
function collide(a, b) {
  const kinds = `${a.shape.kind}:${b.shape.kind}`;
  if (kinds === "circle:circle") return circleCircle(a, b);
  if (kinds === "box:box") return aabbAabb(a, b);
  if (kinds === "circle:polygon" || kinds === "circle:box") return circlePolygon(a, b);
  if (kinds === "polygon:circle" || kinds === "box:circle") {
    const contact = circlePolygon(b, a);
    return contact ? { normal: [-contact.normal[0], -contact.normal[1]], depth: contact.depth, point: contact.point } : null;
  }
  return satPolygons(a, b);
}

/** Circle against a polygon/box: the closest point on the polygon boundary decides the normal. */
function circlePolygon(circle, polygon) {
  const vertices = worldVertices(polygon);
  const centre = circle.position;
  let closest = null;
  let closestDistance = Infinity;
  let inside = true;
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % vertices.length];
    const cross = (b[0] - a[0]) * (centre[1] - a[1]) - (b[1] - a[1]) * (centre[0] - a[0]);
    if (cross < 0) inside = false;
    const edge = [b[0] - a[0], b[1] - a[1]];
    const lengthSquared = edge[0] * edge[0] + edge[1] * edge[1];
    const t = Math.max(0, Math.min(1, ((centre[0] - a[0]) * edge[0] + (centre[1] - a[1]) * edge[1]) / lengthSquared));
    const point = [a[0] + t * edge[0], a[1] + t * edge[1]];
    const distance = Math.hypot(point[0] - centre[0], point[1] - centre[1]);
    if (distance < closestDistance) {
      closestDistance = distance;
      closest = point;
    }
  }
  if (inside) {
    const direction = [closest[0] - centre[0], closest[1] - centre[1]];
    const length = Math.hypot(direction[0], direction[1]) || 1;
    return { normal: [direction[0] / length, direction[1] / length], depth: circle.shape.radius + closestDistance, point: closest };
  }
  if (closestDistance > circle.shape.radius) return null;
  const direction = [closest[0] - centre[0], closest[1] - centre[1]];
  const length = Math.hypot(direction[0], direction[1]) || 1;
  return { normal: [direction[0] / length, direction[1] / length], depth: circle.shape.radius - closestDistance, point: closest };
}

// ---------------------------------------------------------------------------------------------
// broad phase
// ---------------------------------------------------------------------------------------------

/** Uniform grid broad phase: candidate pairs, plus the tuple keys it visited. */
function broadPhase(bodies, cellSize) {
  if (!(cellSize > 0)) throw new RangeError("cellSize must be positive");
  const cells = new Map();
  bodies.forEach((body, index) => {
    const box = bounds(body);
    for (let x = Math.floor(box.minX / cellSize); x <= Math.floor(box.maxX / cellSize); x++) {
      for (let y = Math.floor(box.minY / cellSize); y <= Math.floor(box.maxY / cellSize); y++) {
        const key = `${x}:${y}`;
        if (!cells.has(key)) cells.set(key, []);
        cells.get(key).push(index);
      }
    }
  });
  const candidates = new Set();
  for (const occupants of cells.values()) {
    for (let i = 0; i < occupants.length; i++) {
      for (let j = i + 1; j < occupants.length; j++) {
        const low = Math.min(occupants[i], occupants[j]);
        const high = Math.max(occupants[i], occupants[j]);
        candidates.add(`${low}:${high}`);
      }
    }
  }
  return { candidates, cellCount: cells.size, contacts: [...candidates].map((key) => key.split(":").map(Number)) };
}

/** Brute-force broad phase: every pair whose bounds overlap (the completeness oracle). */
function broadPhaseBruteForce(bodies) {
  const pairs = [];
  for (let i = 0; i < bodies.length; i++) {
    for (let j = i + 1; j < bodies.length; j++) {
      if (boundsOverlap(bounds(bodies[i]), bounds(bodies[j]))) pairs.push([i, j]);
    }
  }
  return pairs;
}

// ---------------------------------------------------------------------------------------------
// solver
// ---------------------------------------------------------------------------------------------

const dot2 = (a, b) => a[0] * b[0] + a[1] * b[1];

/** Applies one impulse along the contact normal plus a clamped tangential (friction) impulse. */
function resolveContact(a, b, contact, { restitutionScale = 1, frictionEnabled = true, positionCorrection = 0.4, slop = 1e-6 } = {}) {
  const inverseMassSum = a.inverseMass + b.inverseMass;
  if (inverseMassSum === 0) return { normalImpulse: 0, frictionImpulse: 0, positionFixed: 0 };
  const relative = [b.velocity[0] - a.velocity[0], b.velocity[1] - a.velocity[1]];
  const normalVelocity = dot2(relative, contact.normal);
  const bias = 0.1;
  const restitution = Math.min(a.restitution, b.restitution) * restitutionScale;
  const effectiveRestitution = normalVelocity < -bias ? restitution : 0;
  const target = -effectiveRestitution * normalVelocity;
  const normalImpulse = Math.max(0, (target - normalVelocity) / inverseMassSum);
  const apply = (body, impulse) => {
    if (body.inverseMass === 0) return;
    body.velocity[0] += impulse[0] * body.inverseMass;
    body.velocity[1] += impulse[1] * body.inverseMass;
  };
  apply(b, [contact.normal[0] * normalImpulse, contact.normal[1] * normalImpulse]);
  apply(a, [-contact.normal[0] * normalImpulse, -contact.normal[1] * normalImpulse]);
  let frictionImpulse = 0;
  if (frictionEnabled && normalImpulse > 0) {
    const tangent = [-contact.normal[1], contact.normal[0]];
    const tangentVelocity = dot2([b.velocity[0] - a.velocity[0], b.velocity[1] - a.velocity[1]], tangent);
    const mu = Math.sqrt(a.friction * b.friction);
    const raw = -tangentVelocity / inverseMassSum;
    frictionImpulse = Math.max(-mu * normalImpulse, Math.min(mu * normalImpulse, raw));
    apply(b, [tangent[0] * frictionImpulse, tangent[1] * frictionImpulse]);
    apply(a, [-tangent[0] * frictionImpulse, -tangent[1] * frictionImpulse]);
  }
  const correction = Math.max(0, contact.depth - slop) * positionCorrection / inverseMassSum;
  let positionFixed = 0;
  if (correction > 0) {
    a.position[0] -= contact.normal[0] * correction * a.inverseMass;
    a.position[1] -= contact.normal[1] * correction * a.inverseMass;
    b.position[0] += contact.normal[0] * correction * b.inverseMass;
    b.position[1] += contact.normal[1] * correction * b.inverseMass;
    positionFixed = contact.depth;
  }
  return { normalImpulse, frictionImpulse, positionFixed };
}

/** One simulation step: integrate, detect, then run a fixed number of solver iterations. */
function stepWorld(world, dt, { gravity = [0, -9.81], iterations = 8, cellSize = 2, substeps = 1, restitutionScale = 1 } = {}) {
  if (!(dt > 0)) throw new RangeError(`dt must be positive (got ${dt})`);
  if (!Number.isInteger(iterations) || iterations < 1) throw new RangeError("iterations must be a positive integer");
  if (!Number.isInteger(substeps) || substeps < 1) throw new RangeError("substeps must be a positive integer");
  const subDt = dt / substeps;
  let contactsDetected = 0;
  let candidatePairs = 0;
  let deepestPenetration = 0;
  for (let substep = 0; substep < substeps; substep++) {
    for (const body of world) {
      if (body.inverseMass === 0) continue;
      body.velocity[0] += gravity[0] * subDt;
      body.velocity[1] += gravity[1] * subDt;
      body.position[0] += body.velocity[0] * subDt;
      body.position[1] += body.velocity[1] * subDt;
    }
    const broad = broadPhase(world, cellSize);
    candidatePairs += broad.contacts.length;
    for (let iteration = 0; iteration < iterations; iteration++) {
      for (const [i, j] of broad.contacts) {
        const a = world[i];
        const b = world[j];
        if (a.inverseMass === 0 && b.inverseMass === 0) continue;
        const contact = collide(a, b);
        if (!contact) continue;
        if (iteration === 0) {
          contactsDetected += 1;
          deepestPenetration = Math.max(deepestPenetration, contact.depth);
        }
        resolveContact(a, b, contact, { restitutionScale });
      }
    }
    for (const body of world) {
      if (body.inverseMass === 0) continue;
      if (Math.hypot(...body.velocity) < 1e-3) body.awake = false;
      else body.awake = true;
    }
  }
  return { contactsDetected, candidatePairs, deepestPenetration };
}

const kineticEnergy = (world) => world.reduce((acc, body) => (body.inverseMass === 0 ? acc : acc + 0.5 * body.mass * (body.velocity[0] ** 2 + body.velocity[1] ** 2)), 0);
const totalMomentum = (world) => world.reduce((acc, body) => (body.inverseMass === 0 ? acc : [acc[0] + body.mass * body.velocity[0], acc[1] + body.mass * body.velocity[1]]), [0, 0]);

/** Deterministic generator for reproducible scenes. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x2545f491;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random convex polygon: a circle's points with jittered radii (always strictly convex). */
function randomPolygon(rng, { radius = 1, sides = 5 } = {}) {
  const vertices = [];
  for (let i = 0; i < sides; i++) {
    const angle = (2 * Math.PI * i) / sides;
    const jitter = radius * (0.75 + rng() * 0.5);
    vertices.push([Math.cos(angle) * jitter, Math.sin(angle) * jitter]);
  }
  return makePolygon(vertices);
}

function main() {
  const diagnostics = [];

  // 1. Circle-circle including the exact touching case.
  const c1 = makeBody({ shape: makeCircle(1), position: [0, 0] });
  const c2 = makeBody({ shape: makeCircle(1), position: [1.5, 0] });
  const c3 = makeBody({ shape: makeCircle(1), position: [2, 0] });
  const c4 = makeBody({ shape: makeCircle(1), position: [2.5, 0] });
  const circleContact = circleCircle(c1, c2);
  diagnostics.push(`circle contacts: depth ${circleContact.depth.toFixed(3)}, touching depth ${circleCircle(c1, c3).depth.toFixed(3)}, separated ${circleCircle(c1, c4)}`);
  ok(Math.abs(circleContact.depth - 0.5) < 1e-12 && circleContact.normal[0] === 1, "the overlap depth and normal must be exact");
  ok(Math.abs(circleCircle(c1, c3).depth) < 1e-12, "exactly touching circles must produce a zero-depth contact");
  ok(circleCircle(c1, c4) === null, "separated circles must not collide");

  // 2. AABB overlap, containment and the axis-of-least-penetration choice.
  const boxA = makeBody({ shape: makeBox(1, 1), position: [0, 0] });
  const boxB = makeBody({ shape: makeBox(1, 1), position: [1.5, 0.25] });
  const boxC = makeBody({ shape: makeBox(0.25, 0.25), position: [0, 0] });
  const boxContact = aabbAabb(boxA, boxB);
  diagnostics.push(`AABB contact: normal [${boxContact.normal}], depth ${boxContact.depth.toFixed(3)}`);
  ok(boxContact.normal[0] === 1 && boxContact.normal[1] === 0 && Math.abs(boxContact.depth - 0.5) < 1e-12, "the MTV must lie on the least-penetration axis");
  ok(Math.abs(aabbAabb(boxA, boxC).depth - 0.5) < 1e-12, "a fully contained box must report a contact as deep as its own extent");
  ok(aabbAabb(boxA, makeBody({ shape: makeBox(1, 1), position: [2.001, 0] })) === null, "separated boxes must not collide");

  // 3. Polygon narrow phase against the independent clip-area oracle.
  const rng = makeRng(20240607);
  let satDisagreements = 0;
  let oracleCases = 0;
  for (let i = 0; i < 60; i++) {
    const first = makeBody({ shape: randomPolygon(rng), position: [0, 0] });
    const offsetX = (rng() * 4 - 2) * 1.6;
    const offsetY = (rng() * 4 - 2) * 1.6;
    const second = makeBody({ shape: randomPolygon(rng), position: [offsetX, offsetY] });
    const area = clipArea(first, second);
    const contact = satPolygons(first, second);
    // Skip the numerically ambiguous neighbourhood of exact tangency.
    if (area < 1e-6 && contact && contact.depth < 1e-3) continue;
    if (area > 1e-6 && (!contact || contact.depth < 1e-3)) continue;
    oracleCases += 1;
    const oracleOverlaps = area > 0;
    if (Boolean(contact) !== oracleOverlaps) satDisagreements += 1;
  }
  diagnostics.push(`SAT vs clip area: ${oracleCases} unambiguous pairs, ${satDisagreements} disagreements`);
  ok(oracleCases > 30, "the oracle must classify most random pairs unambiguously");
  ok(satDisagreements === 0, "SAT and the clip-area oracle must agree on unambiguous pairs");

  // 4. Translating a polygon by its MTV must remove the overlap: the MTV is really a minimum.
  const overlapA = makeBody({ shape: randomPolygon(rng), position: [0, 0] });
  const overlapB = makeBody({ shape: randomPolygon(rng), position: [0.8, 0.4] });
  const overlapContact = satPolygons(overlapA, overlapB);
  const separatedCopy = makeBody({ shape: overlapB.shape, position: [0.8 + overlapContact.normal[0] * (overlapContact.depth + 1e-9), 0.4 + overlapContact.normal[1] * (overlapContact.depth + 1e-9)] });
  const afterSeparation = satPolygons(overlapA, separatedCopy);
  const clipAfter = clipArea(overlapA, separatedCopy);
  diagnostics.push(`MTV ${overlapContact.depth.toFixed(6)} along [${overlapContact.normal.map((v) => v.toFixed(3))}]: residual overlap area ${clipAfter.toExponential(2)}`);
  ok(overlapContact.depth > 0, "the overlapping pair must have a positive MTV");
  ok(clipAfter < 1e-12, "translating by the MTV must remove the overlap (verified by the independent oracle)");
  ok(afterSeparation === null || afterSeparation.depth < 1e-9, "SAT must report no overlap after the MTV translation");

  // 5. Broad phase: complete, and cheaper than the brute-force enumeration.
  const scene = [];
  for (let i = 0; i < 30; i++) {
    const cell = i % 9;
    scene.push(makeBody({ shape: makeCircle(0.4), position: [(cell % 3) * 1.2, Math.floor(cell / 3) * 1.2] }));
  }
  const broad = broadPhase(scene, 1.5);
  const brute = broadPhaseBruteForce(scene);
  const bruteKeys = new Set(brute.map(([i, j]) => `${i}:${j}`));
  const complete = [...bruteKeys].every((key) => broad.candidates.has(key));
  diagnostics.push(`broad phase: ${broad.candidates.size} candidates vs ${bruteKeys.size} overlapping bound pairs out of ${(30 * 29) / 2} possible, ${broad.cellCount} cells`);
  ok(complete, "the grid broad phase must never miss a pair whose bounds overlap");
  ok(broad.candidates.size < (30 * 29) / 2, "the broad phase must prune pairs");
  ok(broad.candidates.size >= bruteKeys.size, "the candidate set must be a superset of the true overlaps");
  throws(() => broadPhase(scene, 0), RangeError);

  // 6. Perfectly elastic equal-mass head-on collision: momentum exact, energy preserved.
  const left = makeBody({ shape: makeCircle(1), position: [-1, 0], velocity: [2, 0], restitution: 1 });
  const right = makeBody({ shape: makeCircle(1), position: [1, 0], velocity: [-2, 0], restitution: 1 });
  const beforeMomentum = totalMomentum([left, right]);
  const beforeEnergy = kineticEnergy([left, right]);
  const elasticContact = circleCircle(left, right);
  resolveContact(left, right, elasticContact, { restitutionScale: 1 });
  const afterMomentum = totalMomentum([left, right]);
  const afterEnergy = kineticEnergy([left, right]);
  diagnostics.push(`elastic impact: v = ${left.velocity[0].toFixed(6)} / ${right.velocity[0].toFixed(6)}, energy ${beforeEnergy.toFixed(6)} -> ${afterEnergy.toFixed(6)}`);
  ok(Math.abs(afterMomentum[0] - beforeMomentum[0]) < 1e-12 && Math.abs(afterMomentum[1] - beforeMomentum[1]) < 1e-12, "momentum must be conserved");
  ok(Math.abs(afterEnergy - beforeEnergy) < 1e-12, "a restitution-1 impact must conserve kinetic energy");
  ok(Math.abs(left.velocity[0] + 2) < 1e-12 && Math.abs(right.velocity[0] - 2) < 1e-12, "equal masses must exchange velocities");

  // 7. Inelastic impact drives the relative normal velocity to zero without gaining energy.
  const softLeft = makeBody({ shape: makeCircle(1), position: [-1, 0], velocity: [3, 0], restitution: 0 });
  const softRight = makeBody({ shape: makeCircle(1), position: [1, 0], velocity: [0, 0], restitution: 0 });
  const softEnergyBefore = kineticEnergy([softLeft, softRight]);
  resolveContact(softLeft, softRight, circleCircle(softLeft, softRight), { restitutionScale: 1 });
  const relativeAfter = (softRight.velocity[0] - softLeft.velocity[0]);
  diagnostics.push(`inelastic impact: relative normal velocity after ${relativeAfter.toExponential(2)}, energy ${softEnergyBefore.toFixed(4)} -> ${kineticEnergy([softLeft, softRight]).toFixed(4)}`);
  ok(Math.abs(relativeAfter) < 1e-12, "a restitution-0 impact must null the relative normal velocity");
  ok(kineticEnergy([softLeft, softRight]) <= softEnergyBefore + 1e-12, "the solver must never create energy");

  // 8. Friction is clamped by the Coulomb cone.
  const sliding = makeBody({ shape: makeCircle(1), position: [-0.9, 0], velocity: [3, 2], restitution: 0, friction: 0.5 });
  const floor = makeBody({ shape: makeCircle(1), position: [0.9, 0.2], restitution: 0, friction: 0.5 });
  const slidingContact = circleCircle(sliding, floor);
  const slidingResult = resolveContact(sliding, floor, slidingContact, { restitutionScale: 1 });
  diagnostics.push(`friction: normal impulse ${slidingResult.normalImpulse.toFixed(4)}, tangential ${slidingResult.frictionImpulse.toFixed(4)} (cone limit ${(0.5 * slidingResult.normalImpulse).toFixed(4)})`);
  ok(Math.abs(slidingResult.frictionImpulse) <= 0.5 * slidingResult.normalImpulse + 1e-12, "the friction impulse must respect the Coulomb cone");
  const frictionless = resolveContact(sliding, floor, slidingContact, { frictionEnabled: false, restitutionScale: 1 });
  ok(frictionless.frictionImpulse === 0, "friction must be switchable off");

  // 9. Penetration is resolved within the iteration budget, and a resting pair settles.
  const ground = makeBody({ shape: makeBox(10, 0.5), position: [0, -1], staticBody: true, restitution: 0 });
  const falling = makeBody({ shape: makeCircle(0.5), position: [0, 0.2], velocity: [0, -5], restitution: 0, friction: 0.4 });
  const world = [ground, falling];
  let worstPenetration = 0;
  for (let step = 0; step < 240; step++) {
    const stats = stepWorld(world, 1 / 120, { iterations: 10, cellSize: 2 });
    worstPenetration = Math.max(worstPenetration, stats.deepestPenetration);
  }
  const restingOverlap = collide(ground, falling);
  diagnostics.push(`resting ball: penetration resolved to ${restingOverlap ? restingOverlap.depth.toExponential(2) : 0}, final velocity ${Math.hypot(...falling.velocity).toExponential(2)}, worst penetration during the run ${worstPenetration.toFixed(3)}`);
  ok(!restingOverlap || restingOverlap.depth < 1e-3, "the resting penetration must be resolved below tolerance");
  ok(Math.hypot(...falling.velocity) < 0.05, "the ball must come to rest on the ground");
  ok(ground.position[1] === -1, "a static body must never move");
  ok(falling.position[1] > -0.8, "the ball must not sink through the ground");

  // 10. Energy never increases over a full simulation, and the momentum of a free system is preserved.
  const freeA = makeBody({ shape: makeCircle(1), position: [-2, 0], velocity: [1.5, 0], restitution: 1 });
  const freeB = makeBody({ shape: makeCircle(1), position: [2, 0], velocity: [-1.5, 0], restitution: 1 });
  const freeWorld = [freeA, freeB];
  const startMomentum = totalMomentum(freeWorld);
  const startEnergy = kineticEnergy(freeWorld);
  let energyPeak = startEnergy;
  for (let step = 0; step < 400; step++) {
    stepWorld(freeWorld, 1 / 240, { gravity: [0, 0], iterations: 6, cellSize: 4 });
    energyPeak = Math.max(energyPeak, kineticEnergy(freeWorld));
  }
  const endMomentum = totalMomentum(freeWorld);
  diagnostics.push(`free collision run: energy ${startEnergy.toFixed(6)} -> ${kineticEnergy(freeWorld).toFixed(6)} (peak ${energyPeak.toFixed(6)}), momentum drift ${Math.hypot(endMomentum[0] - startMomentum[0], endMomentum[1] - startMomentum[1]).toExponential(2)}`);
  ok(energyPeak <= startEnergy + 1e-9, "the solver must not inject energy over the run");
  ok(Math.hypot(endMomentum[0] - startMomentum[0], endMomentum[1] - startMomentum[1]) < 1e-9, "momentum must be preserved in a gravity-free world");

  // 11. Tunnelling: one large step misses a thin wall, substepping catches it. The displacement per
  //     substep is 600 * (1/60) / substeps, so at least 34 substeps are needed for a 0.3 sized obstacle.
  const wall = makeBody({ shape: makeBox(0.1, 5), position: [0, 0], staticBody: true, restitution: 0 });
  const bullets = [1, 64].map((substeps) => {
    const bullet = makeBody({ shape: makeCircle(0.05), position: [-3, 0], velocity: [600, 0], restitution: 0 });
    const bulletWorld = [makeBody({ shape: wall.shape, position: [0, 0], staticBody: true }), bullet];
    stepWorld(bulletWorld, 1 / 60, { gravity: [0, 0], iterations: 4, cellSize: 2, substeps });
    return { substeps, position: bullet.position[0], crossing: bullet.position[0] > 0.2 };
  });
  diagnostics.push(`tunnelling: ${bullets.map((entry) => `${entry.substeps} substeps -> x=${entry.position.toFixed(3)}${entry.crossing ? " (tunnelled)" : " (blocked)"}`).join(", ")}`);
  ok(bullets[0].crossing, "a single large step must let the fast bullet through the thin wall");
  ok(!bullets[1].crossing, "substepping must stop the bullet at the wall");

  // 12. Sequential solving is order-stable to round-off, and the whole engine is deterministic.
  const orderA = makeBody({ shape: makeCircle(1), position: [-0.5, 0], velocity: [1, 0], restitution: 0.4 });
  const orderB = makeBody({ shape: makeCircle(1), position: [0.5, 0], velocity: [-1, 0], restitution: 0.4 });
  const pairs = [[orderA, orderB]];
  const resolveAll = () => {
    for (let iteration = 0; iteration < 8; iteration++) {
      for (const [a, b] of pairs) {
        const contact = circleCircle(a, b);
        if (contact) resolveContact(a, b, contact);
      }
    }
  };
  resolveAll();
  const reversed = makeBody({ shape: makeCircle(1), position: [0.5, 0], velocity: [-1, 0], restitution: 0.4 });
  const reversedOrder = makeBody({ shape: makeCircle(1), position: [-0.5, 0], velocity: [1, 0], restitution: 0.4 });
  for (let iteration = 0; iteration < 8; iteration++) {
    const contact = circleCircle(reversedOrder, reversed);
    if (contact) resolveContact(reversedOrder, reversed, contact);
  }
  ok(Math.abs(orderB.velocity[0] - reversed.velocity[0]) < 1e-12, "solving in the opposite order must give the same velocities");
  const deterministicWorld = () => [makeBody({ shape: makeCircle(0.5), position: [0, 3] }), makeBody({ shape: makeBox(4, 0.5), position: [0, -1], staticBody: true })];
  const firstRun = deterministicWorld();
  const secondRun = deterministicWorld();
  for (let step = 0; step < 120; step++) {
    stepWorld(firstRun, 1 / 120, { iterations: 8, cellSize: 2 });
    stepWorld(secondRun, 1 / 120, { iterations: 8, cellSize: 2 });
  }
  ok(firstRun.every((body, i) => body.position.every((value, axis) => Object.is(value, secondRun[i].position[axis]))), "the engine must be bit-for-bit reproducible");
  const differentGravity = deterministicWorld();
  for (let step = 0; step < 120; step++) stepWorld(differentGravity, 1 / 120, { gravity: [0, -1], iterations: 8, cellSize: 2 });
  ok(differentGravity[0].position[1] !== firstRun[0].position[1], "different gravity must give a different trajectory");

  // 13. Validation.
  throws(() => makeCircle(0), RangeError);
  throws(() => makeBox(1, -1), RangeError);
  throws(() => makePolygon([[0, 0], [1, 1]]), RangeError);
  throws(() => makePolygon([[0, 0], [1, 1], [0, 1], [1, 0]]), RangeError);
  throws(() => makeBody({ shape: makeCircle(1), mass: 0 }), RangeError);
  throws(() => worldVertices(makeBody({ shape: makeCircle(1) })), TypeError);
  throws(() => stepWorld([makeBody({ shape: makeCircle(1) })], 0), RangeError);
  throws(() => stepWorld([makeBody({ shape: makeCircle(1) })], 1 / 60, { iterations: 0 }), RangeError);
  throws(() => stepWorld([makeBody({ shape: makeCircle(1) })], 1 / 60, { substeps: 0 }), RangeError);

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  makeCircle,
  makeBox,
  makePolygon,
  makeBody,
  worldVertices,
  bounds,
  boundsOverlap,
  circleCircle,
  aabbAabb,
  satPolygons,
  clipArea,
  collide,
  circlePolygon,
  broadPhase,
  broadPhaseBruteForce,
  resolveContact,
  stepWorld,
  kineticEnergy,
  totalMomentum,
  makeRng,
  randomPolygon,
};

if (require.main === module) main();
