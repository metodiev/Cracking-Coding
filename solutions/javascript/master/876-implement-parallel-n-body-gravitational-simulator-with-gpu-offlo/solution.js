/** Task 876 - Implement parallel N-body gravitational simulator with GPU offload sketch. (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
 *
 * A direct-sum gravitational N-body integrator with a deterministic offload model:
 *
 *   - forces use Plummer softening, a_ij = G m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2), and every
 *     pair is evaluated once and applied symmetrically (a_i += f, a_j -= f), so the interaction count is
 *     exactly N(N-1)/2 per step and Newton's third law holds by construction;
 *   - time integration is leapfrog kick-drift-kick, which is symplectic and therefore keeps the energy
 *     error bounded instead of drifting secularly;
 *   - "parallelism" is explicit: interaction tiles (row blocks x column blocks) are produced by a
 *     deterministic scheduler and distributed round-robin over a configurable number of devices, which
 *     only reorders the summation, so a parameter sweep must agree to round-off and repeated runs with
 *     identical parameters must agree bit-for-bit - both are checked rather than assumed;
 *   - "GPU offload" is a modelled device: each device has a declared throughput in interactions per
 *     tick and a launch overhead. The coordinator reports the modelled wall ticks, so the cost of
 *     additional devices can be measured without any real hardware, threads or timers.
 *
 * Invariants verified: a two-body circular orbit keeps its radius for a full period; Kepler's third law
 * holds across separations; total energy is conserved to a tight bound with a bounded (not secular)
 * drift; momentum and centre-of-mass velocity drift stay at round-off level; forces computed by an
 * independent naive per-body summation agree with the tiled symmetric kernel; the total force on the
 * system cancels; the interaction counter equals N(N-1)/2 exactly and every tile is scheduled exactly
 * once per step; the device load is balanced to within one tile and the modelled wall time falls as
 * devices are added; degenerate systems (empty, single body, invalid softening or timestep) behave as
 * documented; and everything is reproducible bit-for-bit.
 *
 * What is simplified: no real threads, no GPU, no device memory - offload is a cost model plus a
 * deterministic tile scheduler; there is no tree code (Barnes-Hut), no individual time steps and no
 * collision handling, so the cost is always O(N^2) and N is kept small. Wall-clock time is never read;
 * "ticks" are logical. No Math.random, no timers, no subprocesses.
 */
"use strict";

const { ok, throws } = require("node:assert");

const GRAVITATIONAL_CONSTANT = 1;

/** Body factory with validation: masses must be positive and the state must be finite. */
function makeBody(mass, position, velocity = [0, 0, 0]) {
  if (!(mass > 0)) throw new RangeError(`mass must be positive (got ${mass})`);
  for (const value of [...position, ...velocity]) {
    if (!Number.isFinite(value)) throw new RangeError("positions and velocities must be finite");
  }
  return { mass, position: [...position], velocity: [...velocity] };
}

const cloneBodies = (bodies) => bodies.map((body) => ({ mass: body.mass, position: [...body.position], velocity: [...body.velocity] }));

/** Pairwise separation vector and the softened inverse cube factor. */
function pairKernel(a, b, softening) {
  const dx = b.position[0] - a.position[0];
  const dy = b.position[1] - a.position[1];
  const dz = b.position[2] - a.position[2];
  const distanceSquared = dx * dx + dy * dy + dz * dz + softening * softening;
  const inverseDistance = 1 / Math.sqrt(distanceSquared);
  const factor = inverseDistance * inverseDistance * inverseDistance;
  return { dx, dy, dz, factor };
}

/**
 * Deterministic tile scheduler: interaction tiles are (row block, column block) pairs with
 * column block >= row block (the symmetric half) and are handed to devices round-robin.
 */
function scheduleTiles(bodyCount, { tileSize = 8, devices = 1 } = {}) {
  if (!Number.isInteger(tileSize) || tileSize < 1) throw new RangeError("tileSize must be a positive integer");
  if (!Number.isInteger(devices) || devices < 1) throw new RangeError("devices must be a positive integer");
  const blocks = [];
  for (let start = 0; start < bodyCount; start += tileSize) blocks.push({ start, end: Math.min(start + tileSize, bodyCount) });
  const tiles = [];
  for (let row = 0; row < blocks.length; row++) {
    for (let column = row; column < blocks.length; column++) {
      tiles.push({ row, column, rowBlock: blocks[row], columnBlock: blocks[column] });
    }
  }
  const perDevice = Array.from({ length: devices }, () => []);
  tiles.forEach((tile, index) => perDevice[index % devices].push(tile));
  return { tiles, perDevice };
}

/**
 * Accumulates accelerations over the symmetric half of the interaction matrix. The iteration order is
 * the tile order, which makes the result independent of the tiling parameters.
 */
function computeAccelerations(bodies, { softening = 1e-9, G = GRAVITATIONAL_CONSTANT, tileSize = 8, devices = 1 } = {}) {
  if (!(softening > 0)) throw new RangeError(`softening must be positive (got ${softening})`);
  const accelerations = bodies.map(() => [0, 0, 0]);
  const { tiles, perDevice } = scheduleTiles(bodies.length, { tileSize, devices });
  let interactions = 0;
  const visited = new Set();
  const deviceLoads = perDevice.map(() => 0);
  for (let device = 0; device < perDevice.length; device++) {
    for (const tile of perDevice[device]) {
      const key = `${tile.row}:${tile.column}`;
      if (visited.has(key)) throw new Error(`tile ${key} was scheduled twice`);
      visited.add(key);
      for (let i = tile.rowBlock.start; i < tile.rowBlock.end; i++) {
        for (let j = tile.columnBlock.start; j < tile.columnBlock.end; j++) {
          if (j <= i) continue; // only the strict upper triangle of the interaction matrix
          const { dx, dy, dz, factor } = pairKernel(bodies[i], bodies[j], softening);
          const onI = G * bodies[j].mass * factor;
          const onJ = G * bodies[i].mass * factor;
          accelerations[i][0] += onI * dx;
          accelerations[i][1] += onI * dy;
          accelerations[i][2] += onI * dz;
          accelerations[j][0] -= onJ * dx;
          accelerations[j][1] -= onJ * dy;
          accelerations[j][2] -= onJ * dz;
          interactions += 1;
          deviceLoads[device] += 1;
        }
      }
    }
  }
  return { accelerations, interactions, tiles: tiles.length, deviceLoads, scheduledTiles: visited.size };
}

/** Independent naive reference: every body sums over every other body (no symmetry, no tiling). */
function accelerationsNaive(bodies, { softening = 1e-9, G = GRAVITATIONAL_CONSTANT } = {}) {
  return bodies.map((body) => {
    const total = [0, 0, 0];
    for (const other of bodies) {
      if (other === body) continue;
      const { dx, dy, dz, factor } = pairKernel(body, other, softening);
      const scaled = G * other.mass * factor;
      total[0] += scaled * dx;
      total[1] += scaled * dy;
      total[2] += scaled * dz;
    }
    return total;
  });
}

/** One leapfrog kick-drift-kick step, in place on a cloned state. */
function leapfrogStep(bodies, dt, options = {}) {
  if (!(dt > 0)) throw new RangeError(`dt must be positive (got ${dt})`);
  const half = dt / 2;
  const first = computeAccelerations(bodies, options);
  for (let i = 0; i < bodies.length; i++) {
    for (let axis = 0; axis < 3; axis++) bodies[i].velocity[axis] += half * first.accelerations[i][axis];
  }
  for (const body of bodies) {
    for (let axis = 0; axis < 3; axis++) body.position[axis] += dt * body.velocity[axis];
  }
  const second = computeAccelerations(bodies, options);
  for (let i = 0; i < bodies.length; i++) {
    for (let axis = 0; axis < 3; axis++) bodies[i].velocity[axis] += half * second.accelerations[i][axis];
  }
  return { interactions: first.interactions + second.interactions, tiles: first.tiles + second.tiles };
}

/** Runs the simulation for a number of steps and returns the state plus accumulated counters. */
function simulate(bodies, { dt = 0.001, steps = 100, options = {} } = {}) {
  if (!Number.isInteger(steps) || steps < 0) throw new RangeError("steps must be a non-negative integer");
  const state = cloneBodies(bodies);
  let interactions = 0;
  let tiles = 0;
  for (let step = 0; step < steps; step++) {
    const stats = leapfrogStep(state, dt, options);
    interactions += stats.interactions;
    tiles += stats.tiles;
  }
  return { bodies: state, interactions, tiles };
}

/** Kinetic + potential energy of the system. */
function totalEnergy(bodies, { softening = 1e-9, G = GRAVITATIONAL_CONSTANT } = {}) {
  let kinetic = 0;
  for (const body of bodies) {
    kinetic += 0.5 * body.mass * (body.velocity[0] ** 2 + body.velocity[1] ** 2 + body.velocity[2] ** 2);
  }
  let potential = 0;
  for (let i = 0; i < bodies.length; i++) {
    for (let j = i + 1; j < bodies.length; j++) {
      const { dx, dy, dz } = pairKernel(bodies[i], bodies[j], softening);
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      potential -= (G * bodies[i].mass * bodies[j].mass) / distance;
    }
  }
  return kinetic + potential;
}

/** Total linear momentum and the centre-of-mass velocity. */
function momentum(bodies) {
  const total = [0, 0, 0];
  let mass = 0;
  for (const body of bodies) {
    mass += body.mass;
    for (let axis = 0; axis < 3; axis++) total[axis] += body.mass * body.velocity[axis];
  }
  return { momentum: total, mass, centreOfMassVelocity: total.map((value) => value / mass) };
}

/** Modelled offload wall time in logical device ticks. */
function modelledWallTicks(interactionsByDevice, { throughput = 1000, launchOverhead = 2 } = {}) {
  const busy = interactionsByDevice.map((load) => Math.ceil(load / throughput));
  return { ticks: Math.max(0, ...busy) + (interactionsByDevice.length > 0 ? launchOverhead : 0), perDevice: busy };
}

/** Two equal masses on a circular orbit about their common centre of mass. */
function circularTwoBody({ separation = 2, mass = 1, G = GRAVITATIONAL_CONSTANT } = {}) {
  const speed = Math.sqrt((G * mass) / (2 * separation));
  return [
    makeBody(mass, [-separation / 2, 0, 0], [0, 0, speed]),
    makeBody(mass, [separation / 2, 0, 0], [0, 0, -speed]),
  ];
}

/** Period of the circular two-body orbit: T = 2*pi*r/v with r = separation/2 and v = sqrt(G m /(2 d)). */
const orbitalPeriod = ({ separation = 2, mass = 1, G = GRAVITATIONAL_CONSTANT } = {}) =>
  (2 * Math.PI * (separation / 2)) / Math.sqrt((G * mass) / (2 * separation));

function main() {
  const diagnostics = [];

  // 1. The interaction kernel: pair symmetry, and agreement with the independent naive summation.
  const cluster = [
    makeBody(1.0, [0, 0, 0]),
    makeBody(0.5, [1.5, 0.3, -0.2], [0.01, -0.02, 0]),
    makeBody(2.0, [-1.2, 0.9, 0.4], [-0.03, 0.01, 0.02]),
    makeBody(0.75, [0.4, -1.7, 0.6], [0.02, 0.04, -0.01]),
  ];
  const tiled = computeAccelerations(cluster, { tileSize: 2, devices: 3 });
  const naive = accelerationsNaive(cluster);
  let worstAcceleration = 0;
  tiled.accelerations.forEach((value, i) => {
    for (let axis = 0; axis < 3; axis++) worstAcceleration = Math.max(worstAcceleration, Math.abs(value[axis] - naive[i][axis]));
  });
  const totalForce = cluster.reduce((acc, body, i) => acc.map((value, axis) => value + body.mass * tiled.accelerations[i][axis]), [0, 0, 0]);
  diagnostics.push(`tiled vs naive accelerations: worst deviation ${worstAcceleration.toExponential(2)}, total force ${Math.hypot(...totalForce).toExponential(2)}`);
  ok(worstAcceleration < 1e-12, "the tiled symmetric kernel must agree with the naive per-body sum");
  ok(Math.hypot(...totalForce) < 1e-12, "Newton's third law must cancel the total force");
  ok(tiled.interactions === 6, `four bodies must need six pair interactions (got ${tiled.interactions})`);
  ok(tiled.scheduledTiles === tiled.tiles, "every tile must be scheduled exactly once per step");

  // 2. Tiling and device count only reorder the summation, so the state must agree to round-off and be
  //    bit-identical for repeated runs with the same parameters.
  const configurations = [
    { tileSize: 1, devices: 1 }, { tileSize: 2, devices: 2 }, { tileSize: 4, devices: 1 }, { tileSize: 3, devices: 4 },
  ];
  const runs = configurations.map((options) => simulate(cluster, { dt: 0.002, steps: 40, options }));
  const referenceRun = runs[0];
  let worstStateDifference = 0;
  runs.forEach((run) => run.bodies.forEach((body, i) => body.position.forEach((value, axis) => {
    worstStateDifference = Math.max(worstStateDifference, Math.abs(value - referenceRun.bodies[i].position[axis]));
  })));
  diagnostics.push(`tiling/device sweep ${configurations.map((c) => `${c.tileSize}x${c.devices}`).join(", ")}: worst state difference ${worstStateDifference.toExponential(2)}`);
  ok(worstStateDifference < 1e-10, `tiling must not change the result beyond round-off (worst ${worstStateDifference})`);
  ok(runs.every((run) => run.interactions === 40 * 2 * 6), "each step must evaluate exactly N(N-1)/2 pairs twice (kick-drift-kick)");
  const identicalRepeat = simulate(cluster, { dt: 0.002, steps: 40, options: { tileSize: 1, devices: 1 } });
  ok(identicalRepeat.bodies.every((body, i) => body.position.every((value, axis) => Object.is(value, referenceRun.bodies[i].position[axis]))),
    "identical parameters must reproduce the state bit-for-bit");

  // 3. Two-body circular orbit: radius, period and energy over a full revolution.
  const orbit = circularTwoBody({ separation: 2, mass: 1 });
  const period = orbitalPeriod({ separation: 2, mass: 1 });
  const initialEnergy = totalEnergy(orbit);
  const orbitSteps = 4000;
  const orbitRun = simulate(orbit, { dt: period / orbitSteps, steps: orbitSteps, options: { softening: 1e-6 } });
  const radiusAfter = Math.hypot(orbitRun.bodies[0].position[0], orbitRun.bodies[0].position[1], orbitRun.bodies[0].position[2]);
  const energyAfter = totalEnergy(orbitRun.bodies, { softening: 1e-6 });
  diagnostics.push(`circular orbit over one period (T=${period.toFixed(4)}): radius ${radiusAfter.toFixed(6)} (expected 1), energy drift ${((energyAfter - initialEnergy) / Math.abs(initialEnergy)).toExponential(2)}`);
  ok(period > 12 && period < 13, `the analytic period must be about 12.57 (got ${period})`);
  ok(Math.abs(radiusAfter - 1) < 1e-3, `the circular orbit must keep its radius (got ${radiusAfter})`);
  ok(Math.abs(energyAfter - initialEnergy) / Math.abs(initialEnergy) < 1e-9, "the leapfrog energy error must stay tiny on a smooth orbit");

  // 4. Kepler's third law: T^2 / a^3 is the same constant for different separations.
  const separations = [1.6, 2, 3];
  const kepler = separations.map((separation) => {
    const system = circularTwoBody({ separation, mass: 1 });
    const analytic = orbitalPeriod({ separation, mass: 1 });
    const steps = 3000;
    const run = simulate(system, { dt: analytic / steps, steps, options: { softening: 1e-6 } });
    const radius = Math.hypot(...run.bodies[0].position);
    return { separation, analytic, ratio: (analytic * analytic) / (separation / 2) ** 3, radius };
  });
  diagnostics.push(`Kepler: ${kepler.map((entry) => `a=${entry.separation} T=${entry.analytic.toFixed(4)} T^2/a^3=${entry.ratio.toFixed(5)}`).join(", ")}`);
  ok(kepler.every((entry) => Math.abs(entry.ratio - kepler[0].ratio) < 1e-12), "the Kepler ratio must be constant across separations");
  ok(kepler.every((entry) => Math.abs(entry.radius - entry.separation / 2) < 2e-3), "every circular orbit must stay on its radius (separation/2)");

  // 5. Energy drift is bounded rather than secular: the drift after 4000 steps must not be ten times
  //    the drift after 1000 steps.
  const shortRun = simulate(orbit, { dt: period / 1000, steps: 1000, options: { softening: 1e-6 } });
  const longRun = simulate(orbit, { dt: period / 4000, steps: 4000, options: { softening: 1e-6 } });
  const shortDrift = Math.abs(totalEnergy(shortRun.bodies, { softening: 1e-6 }) - initialEnergy);
  const longDrift = Math.abs(totalEnergy(longRun.bodies, { softening: 1e-6 }) - initialEnergy);
  diagnostics.push(`bounded drift: ${shortDrift.toExponential(2)} after 1000 steps vs ${longDrift.toExponential(2)} after 4000`);
  ok(longDrift < 10 * Math.max(shortDrift, 1e-12), "the symplectic drift must not grow linearly with the number of steps");

  // 6. Momentum and centre-of-mass velocity.
  const momentumBefore = momentum(cluster);
  const momentumRun = simulate(cluster, { dt: 0.002, steps: 200 });
  const momentumAfter = momentum(momentumRun.bodies);
  const momentumDrift = Math.hypot(...momentumAfter.momentum.map((value, axis) => value - momentumBefore.momentum[axis]));
  const comDrift = Math.hypot(...momentumAfter.centreOfMassVelocity.map((value, axis) => value - momentumBefore.centreOfMassVelocity[axis]));
  diagnostics.push(`momentum drift ${momentumDrift.toExponential(2)}, centre-of-mass velocity drift ${comDrift.toExponential(2)}`);
  ok(momentumDrift < 1e-12, `the momentum drift must stay at round-off level (${momentumDrift})`);
  ok(comDrift < 1e-12, "the centre-of-mass velocity must stay constant");

  // 7. Offload model: balanced loads, monotone speed-up, and a documented launch overhead.
  const many = Array.from({ length: 64 }, (_value, i) => makeBody(
    1 + (i % 5) * 0.1,
    [(i % 8) - 3.5, Math.floor(i / 8) - 3.5, Math.sin(i) * 0.5],
    [Math.cos(i) * 0.01, Math.sin(i) * 0.01, 0],
  ));
  const loads = [1, 2, 4, 8].map((devices) => {
    const scheduled = computeAccelerations(many, { tileSize: 4, devices });
    const wall = modelledWallTicks(scheduled.deviceLoads, { throughput: 200, launchOverhead: 3 });
    return { devices, loads: scheduled.deviceLoads, wall: wall.ticks, total: scheduled.interactions };
  });
  diagnostics.push(`offload sweep (throughput 200 ticks/interaction): ${loads.map((entry) => `${entry.devices}d -> ${entry.wall} ticks`).join(", ")}`);
  ok(loads.every((entry) => entry.total === (64 * 63) / 2), "the interaction count must stay N(N-1)/2 for every device count");
  ok(loads.every((entry) => Math.max(...entry.loads) - Math.min(...entry.loads) <= 1), "round-robin scheduling must balance the tile loads to within one tile");
  for (let i = 1; i < loads.length; i++) {
    ok(loads[i].wall <= loads[i - 1].wall, `more devices must not increase the modelled wall time (${loads[i - 1].devices} -> ${loads[i].devices})`);
  }
  ok(loads[loads.length - 1].wall < loads[0].wall, "adding devices must actually reduce the modelled wall time");
  const noOverhead = modelledWallTicks([100], { throughput: 100, launchOverhead: 0 });
  ok(noOverhead.ticks === 1 && modelledWallTicks([100], { throughput: 100, launchOverhead: 5 }).ticks === 6, "the launch overhead must be added exactly once");

  // 8. Degenerate systems and validation.
  const empty = simulate([], { dt: 0.1, steps: 5 });
  ok(empty.bodies.length === 0 && empty.interactions === 0, "an empty system must be a no-op");
  const single = simulate([makeBody(3, [1, 2, 3], [0.5, 0, 0])], { dt: 1, steps: 10 });
  ok(single.bodies[0].position.every(Number.isFinite) && Object.is(single.bodies[0].position[0], 6), "a lone body must drift at constant velocity without NaN");
  ok(single.interactions === 0, "a single body needs no interactions");
  const pair = computeAccelerations([makeBody(1, [0, 0, 0]), makeBody(1, [1e6, 0, 0])]);
  ok(Math.hypot(...pair.accelerations[0]) < 1e-11, "a very distant pair must exert a negligible force");
  throws(() => makeBody(0, [0, 0, 0]), RangeError);
  throws(() => makeBody(1, [Number.NaN, 0, 0]), RangeError);
  throws(() => computeAccelerations(cluster, { softening: 0 }), RangeError);
  throws(() => scheduleTiles(4, { tileSize: 0 }), RangeError);
  throws(() => scheduleTiles(4, { devices: 0 }), RangeError);
  throws(() => simulate(cluster, { dt: 0 }), RangeError);
  throws(() => simulate(cluster, { steps: -1 }), RangeError);

  // 9. Determinism, and the counters follow the documented formulas.
  const repeatA = simulate(many, { dt: 0.001, steps: 30, options: { tileSize: 4, devices: 2 } });
  const repeatB = simulate(many, { dt: 0.001, steps: 30, options: { tileSize: 4, devices: 2 } });
  ok(repeatA.bodies.every((body, i) => body.position.every((value, axis) => Object.is(value, repeatB.bodies[i].position[axis]))), "simulations must be reproducible");
  ok(repeatA.interactions === 30 * 2 * ((64 * 63) / 2), "the interaction counter must follow 2 * steps * N(N-1)/2");
  const differentDt = simulate(many, { dt: 0.002, steps: 30, options: { tileSize: 4, devices: 2 } });
  ok(differentDt.bodies.some((body, i) => body.position.some((value, axis) => value !== repeatA.bodies[i].position[axis])), "a different timestep must give a different trajectory");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  GRAVITATIONAL_CONSTANT,
  makeBody,
  cloneBodies,
  pairKernel,
  scheduleTiles,
  computeAccelerations,
  accelerationsNaive,
  leapfrogStep,
  simulate,
  totalEnergy,
  momentum,
  modelledWallTicks,
  circularTwoBody,
  orbitalPeriod,
};

if (require.main === module) main();
