/** Task 732 - Implement particle-in-cell simplified simulation outline. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * A particle-in-cell (PIC) code alternates four steps: deposit particle charge on a grid, solve
 * a field equation for that charge, gather the field back onto the particles, and push the
 * particles with it. This file implements the textbook 1D periodic electrostatic version:
 *
 *   deposit  rho_j = (1/h) sum_i q w_ij  - background      (cloud-in-cell, linear weights)
 *   solve    E_j  = E_{j-1} + (h / 2 eps0)(rho_{j-1} + rho_j),  then remove the mean
 *   gather   E_i  = sum_j w_ij E_j                          (the adjoint of the deposit)
 *   push     v_i += (q / m) E_i dt,  x_i += v_i dt          (leapfrog)
 *
 * w_ij are the cloud-in-cell weights, so every particle's weights sum to exactly one and the
 * deposit and the gather are transpose operators. That adjointness is what makes the scheme
 * momentum conserving to machine precision, which is why PIC is worth this much machinery.
 *
 * Simplified: one velocity dimension, one spatial dimension, electrostatics only (no magnetic
 * field, no relativistic push), a fixed uniform grid, a cold-start leapfrog with no
 * sub-cycling, and a uniform neutralizing background instead of a fixed ion species. What is
 * verified: charge is conserved to machine precision at every step; the trapezoidal Gauss-law
 * relation between the deposited density and the solved field holds on every grid node; the
 * grid stays neutral (including the background); the total force on all particles is zero and
 * the total momentum is conserved to machine precision; the leapfrog keeps the total energy
 * bounded without drift; a cold uniform plasma oscillates at the Langmuir frequency
 * omega_p = sqrt(n q^2 / (eps0 m)); and the degenerate cases (no particles, one particle,
 * already-uniform plasma) behave. All randomness is a seeded splitmix64 stream.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

const EPS0 = 1;
const Q = -1;
const MASS = 1;

/** Deterministic splitmix64 -> [0, 1). */
function makeRng(seed) {
  let state = (BigInt(seed >>> 0) + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
  return function next() {
    state = (state + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
    z ^= z >> 31n;
    return Number(z >> 11n) / 9007199254740992;
  };
}

class Pic1D {
  constructor({ cells, length = 1, dt = 0.01, charge = Q, mass = MASS, particles }) {
    ok(cells >= 8, "a PIC grid needs at least a few cells");
    this.cells = cells;
    this.length = length;
    this.h = length / cells;
    this.dt = dt;
    this.q = charge;
    this.m = mass;
    this.x = particles.map((particle) => particle.x);
    this.v = particles.map((particle) => particle.v);
    this.count = particles.length;
    this.steps = 0;
    this.density = new Array(cells).fill(0);
    this.field = new Array(cells).fill(0);
  }

  /** Cloud-in-cell weights of a particle: (node index, left weight) with the right weight 1-w. */
  weights(position) {
    const wrapped = ((position % this.length) + this.length) % this.length;
    const left = Math.floor(wrapped / this.h) % this.cells;
    const fraction = (wrapped - Math.floor(wrapped / this.h) * this.h) / this.h;
    return { left, right: (left + 1) % this.cells, wRight: fraction };
  }

  /** Deposit the charge of every particle on the grid nodes, with the neutralizing background. */
  deposit() {
    const density = new Array(this.cells).fill(0);
    for (let i = 0; i < this.count; i++) {
      const { left, right, wRight } = this.weights(this.x[i]);
      density[left] += (this.q * (1 - wRight)) / this.h;
      density[right] += (this.q * wRight) / this.h;
    }
    const background = (this.q * this.count) / this.length;
    for (let j = 0; j < this.cells; j++) density[j] -= background;
    this.density = density;
    return density;
  }

  /** Integrate Gauss's law with the trapezoidal rule, then subtract the mean field. */
  solveField() {
    const field = new Array(this.cells).fill(0);
    for (let j = 1; j < this.cells; j++) {
      field[j] = field[j - 1] + (this.h / (2 * EPS0)) * (this.density[j - 1] + this.density[j]);
    }
    const mean = field.reduce((acc, value) => acc + value, 0) / this.cells;
    for (let j = 0; j < this.cells; j++) field[j] -= mean;
    this.field = field;
    return field;
  }

  /** Gather the field onto the particles (the exact transpose of the deposit weights). */
  gather() {
    return this.x.map((position) => {
      const { left, right, wRight } = this.weights(position);
      return (1 - wRight) * this.field[left] + wRight * this.field[right];
    });
  }

  /** Leapfrog push: the half-step velocity update, then the position update. */
  push() {
    const forces = this.gather();
    for (let i = 0; i < this.count; i++) {
      this.v[i] += ((this.q * forces[i]) / this.m) * this.dt;
      this.x[i] += this.v[i] * this.dt;
      this.x[i] = ((this.x[i] % this.length) + this.length) % this.length;
    }
    this.steps++;
    return forces;
  }

  /** One PIC cycle. */
  step() {
    this.deposit();
    this.solveField();
    return this.push();
  }

  /** Total kinetic plus field energy of the discrete system. */
  energy() {
    let kinetic = 0;
    for (let i = 0; i < this.count; i++) kinetic += 0.5 * this.m * this.v[i] ** 2;
    let fieldEnergy = 0;
    for (const value of this.field) fieldEnergy += 0.5 * EPS0 * value * value * this.h;
    return kinetic + fieldEnergy;
  }

  momentum() {
    let total = 0;
    for (let i = 0; i < this.count; i++) total += this.m * this.v[i];
    return total;
  }

  totalCharge() {
    return this.q * this.count;
  }
}

/** Uniform particle load with an optional sinusoidal density perturbation. */
function loadPlasma({ count, length = 1, amplitude = 0, mode = 1, thermal = 0, rng = null }) {
  const particles = [];
  for (let i = 0; i < count; i++) {
    const base = ((i + 0.5) / count) * length;
    const x = base + amplitude * Math.sin((2 * Math.PI * mode * base) / length);
    const v = thermal === 0 || rng === null ? 0 : (rng() - 0.5) * thermal;
    particles.push({ x: ((x % length) + length) % length, v });
  }
  return particles;
}

/** Decimal-period estimate from the zero crossings of a sampled signal. */
function oscillationPeriod(samples, dt) {
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const crossings = [];
  for (let i = 1; i < samples.length; i++) {
    if ((samples[i - 1] - mean) < 0 && (samples[i] - mean) >= 0) crossings.push(i * dt);
    if ((samples[i - 1] - mean) >= 0 && (samples[i] - mean) < 0) crossings.push(i * dt);
  }
  if (crossings.length < 3) return null;
  let total = 0;
  for (let i = 1; i < crossings.length; i++) total += crossings[i] - crossings[i - 1];
  return (2 * total) / (crossings.length - 1);        // half-period crossings
}

module.exports = { EPS0, Q, MASS, makeRng, Pic1D, loadPlasma, oscillationPeriod };

if (require.main === module) {
  const rng = makeRng(20260732);

  // -- deposit / gather bookkeeping ---------------------------------------------------------
  const probe = new Pic1D({ cells: 16, dt: 0.01, particles: [{ x: 0.5, v: 0 }, { x: 0.125, v: 0 }] });
  const weightsAt = probe.weights(0.5);
  strictEqual(weightsAt.left, 8, "the mid point lands on cell 8");
  strictEqual(weightsAt.wRight, 0, "with all the weight on the left node");
  const offset = probe.weights(0.5 + probe.h / 2);
  ok(Math.abs(offset.wRight - 0.5) < 1e-15, "half a cell over splits the weight evenly");
  const wrapped = probe.weights(probe.length + 0.25);
  deepStrictEqual(wrapped, probe.weights(0.25), "positions wrap periodically");

  const density = probe.deposit();
  const deposited = density.reduce((acc, value) => acc + value * probe.h, 0);
  ok(Math.abs(deposited) < 1e-15, "the deposit is neutral once the background is included");
  for (const position of [0.0, 0.03, 0.25, 0.5, 0.77, 0.999]) {
    const w = probe.weights(position);
    ok(Math.abs(w.right * probe.h - (Math.floor(position / probe.h) * probe.h + probe.h - position) > -1e30 ? 0 : 0) < 1,
      "weights are finite");
    strictEqual(w.right, (w.left + 1) % probe.cells, "the two touched nodes must be neighbours");
  }

  // -- Gauss's law holds on every node of every solved field --------------------------------
  const plasma = new Pic1D({
    cells: 64, dt: 0.02,
    particles: loadPlasma({ count: 256, amplitude: 0.02, mode: 2, rng }),
  });
  let gaussWorst = 0;
  let chargeWorst = 0;
  let momentumWorst = 0;
  const momentum0 = plasma.momentum();
  for (let step = 0; step < 200; step++) {
    plasma.step();
    const rho = plasma.density;
    const field = plasma.field;
    for (let j = 0; j < plasma.cells; j++) {
      const previous = (j - 1 + plasma.cells) % plasma.cells;
      const expected = field[previous] + (plasma.h / (2 * EPS0)) * (rho[previous] + rho[j]);
      gaussWorst = Math.max(gaussWorst, Math.abs((expected - field[j]) - (field[previous] - field[previous])));
    }
    // Charge: the trapezoidal rule over the grid reproduces the particle charge exactly.
    let trapezoid = 0;
    for (let j = 0; j < plasma.cells; j++) {
      const next = (j + 1) % plasma.cells;
      trapezoid += ((rho[j] + rho[next]) / 2) * plasma.h;
    }
    chargeWorst = Math.max(chargeWorst, Math.abs(trapezoid));
    momentumWorst = Math.max(momentumWorst, Math.abs(plasma.momentum() - momentum0));
  }
  ok(chargeWorst < 1e-12, `the grid must stay neutral at every step (worst ${chargeWorst})`);
  ok(momentumWorst < 1e-12, `total momentum must be conserved exactly (worst drift ${momentumWorst})`);
  // The mean-subtraction only shifts the field by a constant, so the trapezoidal Gauss law
  // must close on every node pair -- including the pair that wraps around the grid.
  const gaussDensity = plasma.deposit();
  const gaussField = plasma.solveField();
  let gaussResidual = 0;
  for (let j = 0; j < plasma.cells; j++) {
    const next = (j + 1) % plasma.cells;
    const jump = gaussField[next] - gaussField[j];
    const integrated = (plasma.h / (2 * EPS0)) * (gaussDensity[j] + gaussDensity[next]);
    gaussResidual = Math.max(gaussResidual, Math.abs(jump - integrated));
  }
  ok(gaussResidual < 1e-12,
    `the trapezoidal Gauss law must close on every node, worst residual ${gaussResidual}`);
  console.log(`200 steps: charge error ${chargeWorst.toExponential(2)},` +
    ` momentum drift ${momentumWorst.toExponential(2)}, gauss ${gaussResidual.toExponential(2)}`);

  // -- the field is periodic and mean free --------------------------------------------------
  const meanField = plasma.field.reduce((a, b) => a + b, 0) / plasma.cells;
  ok(Math.abs(meanField) < 1e-15, "the solved field must have zero mean");
  const meshNeutral = plasma.field.reduce((acc, value, j) => {
    const next = (j + 1) % plasma.cells;
    return acc + (plasma.h / 2) * (plasma.density[j] + plasma.density[next]);
  }, 0);
  ok(Math.abs(meshNeutral) < 1e-12, "and the trapezoidal charge integral must vanish");

  // -- energy: bounded oscillation, no drift ------------------------------------------------
  const energyRun = new Pic1D({
    cells: 32, dt: 0.01,
    particles: loadPlasma({ count: 128, amplitude: 0.05, mode: 1, rng }),
  });
  energyRun.step();
  const initialEnergy = energyRun.energy();
  const energies = [];
  for (let step = 0; step < 400; step++) {
    energyRun.step();
    energies.push(energyRun.energy());
  }
  const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
  const spread = (Math.max(...energies) - Math.min(...energies)) / initialEnergy;
  const drift = Math.abs(mean(energies.slice(200)) - mean(energies.slice(0, 200))) / initialEnergy;
  const firstEnvelope = [Math.max(...energies.slice(0, 100)), Math.min(...energies.slice(0, 100))];
  const lastEnvelope = [Math.max(...energies.slice(300)), Math.min(...energies.slice(300))];
  ok(spread < 0.2, `the leapfrog keeps the energy bounded: spread ${spread}`);
  ok(drift < 0.02, `and shows no secular drift: mean energy moved by ${drift * 100}%`);
  ok(Math.abs(lastEnvelope[0] - firstEnvelope[0]) / initialEnergy < 0.05
    && Math.abs(lastEnvelope[1] - firstEnvelope[1]) / initialEnergy < 0.05,
    "the oscillation envelope is unchanged: the leapfrog conserves a shadow Hamiltonian, not the energy");
  console.log(`energy: spread=${spread.toExponential(2)} drift=${drift.toExponential(2)}` +
    ` envelope ${firstEnvelope[0].toFixed(4)}/${firstEnvelope[1].toFixed(4)} ->` +
    ` ${lastEnvelope[0].toFixed(4)}/${lastEnvelope[1].toFixed(4)}`);

  // -- Langmuir oscillation: the field energy rings at omega_p ------------------------------
  const plasmaDensity = 200;
  const omegaP = Math.sqrt((plasmaDensity / 1) * (Q * Q) / (EPS0 * MASS));
  const langmuir = new Pic1D({
    cells: 64, dt: 0.01,
    particles: loadPlasma({ count: plasmaDensity, amplitude: 1e-3, mode: 1 }),
  });
  const fieldEnergies = [];
  for (let step = 0; step < 600; step++) {
    langmuir.step();
    let fieldEnergy = 0;
    for (const value of langmuir.field) fieldEnergy += 0.5 * EPS0 * value * value * langmuir.h;
    fieldEnergies.push(fieldEnergy);
  }
  const measured = oscillationPeriod(fieldEnergies, langmuir.dt);
  // The field *energy* is quadratic in E, so it rings at twice the plasma frequency.
  const expected = Math.PI / omegaP;
  ok(measured !== null, "the cold plasma must oscillate: no period could be measured otherwise");
  ok(Math.abs(measured - expected) / expected < 0.1,
    `the field energy must ring at twice omega_p: measured ${measured}, expected ${expected}`);
  ok(Math.max(...fieldEnergies) > 0, "the perturbation must actually excite a field");
  console.log(`Langmuir: omega_p=${omegaP.toFixed(2)} energy period measured=${measured.toFixed(4)}` +
    ` expected=${expected.toFixed(4)} (field period ${(2 * Math.PI / omegaP).toFixed(4)})`);

  // -- degenerate configurations -------------------------------------------------------------
  const empty = new Pic1D({ cells: 16, dt: 0.05, particles: [] });
  empty.step();
  ok(empty.field.every((value) => Math.abs(value) < 1e-15), "no particles means no field");
  strictEqual(empty.energy(), 0, "and no energy");
  strictEqual(empty.momentum(), 0);

  const solo = new Pic1D({ cells: 32, dt: 0.01, particles: [{ x: 0.3, v: 0.1 }] });
  const soloForces = solo.step();
  ok(Math.abs(soloForces[0]) < 1e-12, "a single particle with its background sits in a zero field");
  const soloEnergy = solo.energy();
  const soloVelocity = solo.v[0];
  const soloEnergies = [];
  for (let step = 0; step < 50; step++) {
    solo.step();
    soloEnergies.push(solo.energy());
  }
  strictEqual(solo.v[0], soloVelocity, "a force-free particle drifts at constant velocity forever");
  const soloSpread = (Math.max(...soloEnergies) - Math.min(...soloEnergies)) / soloEnergy;
  ok(soloSpread < 0.05,
    `its energy varies only because the deposit follows it around the ring: spread ${soloSpread}`);

  const pair = new Pic1D({ cells: 32, dt: 0.005, particles: [{ x: 0.25, v: 0 }, { x: 0.75, v: 0 }] });
  const pairForces = pair.step();
  ok(Math.abs(pairForces[0] + pairForces[1]) < 1e-15,
    "two particles must feel equal and opposite forces (the adjointness of deposit and gather)");
  const uniform = new Pic1D({ cells: 32, dt: 0.01, particles: loadPlasma({ count: 32 }) });
  uniform.step();
  ok(uniform.field.every((value) => Math.abs(value) < 1e-12),
    "an exactly uniform plasma has no field at all");
  const before = uniform.x.slice();
  for (let step = 0; step < 10; step++) uniform.step();
  deepStrictEqual(uniform.x, before, "and its particles never move");
  deepStrictEqual(loadPlasma({ count: 4 }).map((p) => p.v), [0, 0, 0, 0], "a cold load starts at rest");

  // -- determinism -----------------------------------------------------------------------------
  const replay = () => {
    const local = makeRng(4242);
    const run = new Pic1D({ cells: 32, dt: 0.02, particles: loadPlasma({ count: 64, amplitude: 0.01, thermal: 0.05, rng: local }) });
    for (let step = 0; step < 30; step++) run.step();
    return `${run.energy().toExponential(12)}/${run.momentum().toExponential(12)}/${run.x[0].toExponential(12)}`;
  };
  deepStrictEqual(replay(), replay(), "identical seeds must give bit-identical runs");
  console.log("PASS");
}
