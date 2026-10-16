/**
 * Task 525 - Implement HMM forward-backward for probabilities. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * The forward-backward procedure for a discrete HMM with scaled recursions so long sequences cannot
 * underflow: the forward pass keeps alpha_hat_t = alpha_t / c_t with c_t = sum_i alpha_t(i), the
 * backward pass keeps beta_hat_t = beta_t / c_t with the same scaling factors, and the observation
 * likelihood is the product of the c_t (returned both as a number and as its logarithm). From the
 * scaled tables the file derives the smoothing posteriors gamma_t(i) = alpha_hat_t(i)
 * beta_hat_t(i) / sum_k(...) and the pairwise posteriors xi_t(i,j) normalised by their double sum,
 * which is the exact form used by the Baum-Welch re-estimation step (pi = gamma_0,
 * A_ij = sum_t xi / sum_t gamma_i, B_i(k) = sum_{t: o_t = k} gamma / sum_t gamma, with rows left
 * untouched when a state is never visited).
 *
 * Documented checks: the scaled log-likelihood equals log of the exhaustive sum over all N^T paths
 * (N = 3, T = 6), and the unscaled forward recursion agrees with the scaled one to 1e-12 while the
 * scaled tables stay normalised; every gamma row sums to 1 and matches brute-force path-enumerated
 * posteriors to 1e-12; the identity sum_j xi_t(i,j) = gamma_t(i) and sum_{i,j} xi_t(i,j) = 1 hold for
 * every t; the expected state occupancies sum_t gamma_t(i) equal the brute-force expected visit
 * counts; for T = 1 the posteriors reduce to the hand-computable pi_i B_i(o_0) / P(o_0); and a
 * Baum-Welch loop is checked to (a) be monotone - the log-likelihood never decreases by more than
 * 1e-9 -, (b) keep every row a probability distribution at every intermediate iteration, (c) reduce
 * the residual gain to under 1% of the total gain in the last 10 of 300 iterations, and (d) on a
 * long well-separated 2-state sequence recovered from a perturbed start, shrink the L1 error of the
 * transition and emission rows by at least a factor of five and reach a fixed point (a further step
 * moves the parameters by <= 1e-5). Malformed input is rejected.
 */
"use strict";

function validateModel(initial, transition, emission) {
  const n = initial.length;
  if (n === 0) throw new Error("the state space must not be empty");
  if (transition.length !== n || emission.length !== n) throw new Error("dimension mismatch");
  const check = (values, label) => {
    let sum = 0.0;
    for (const v of values) {
      if (!(v >= 0.0)) throw new Error(`${label} must be non-negative`);
      sum += v;
    }
    if (Math.abs(sum - 1.0) > 1e-9) throw new Error(`${label} must sum to 1`);
  };
  check(initial, "initial distribution");
  for (const row of transition) {
    if (row.length !== n) throw new Error("transition matrix must be square");
    check(row, "transition row");
  }
  const m = emission[0].length;
  for (const row of emission) {
    if (row.length !== m) throw new Error("emission rows must have equal length");
    check(row, "emission row");
  }
  return { n, m };
}

function checkObservations(obs, m) {
  if (obs.length === 0) throw new Error("empty observation sequence");
  for (const o of obs) if (!Number.isInteger(o) || o < 0 || o >= m) throw new Error(`bad symbol ${o}`);
  return obs.length;
}

/** Scaled forward pass: alpha_hat rows, the per-step scaling factors c_t, and the likelihood. */
function forwardScaled(initial, transition, emission, obs) {
  const { n, m } = validateModel(initial, transition, emission);
  const T = checkObservations(obs, m);
  const alpha = [];
  const scales = new Array(T);
  const first = new Array(n);
  for (let i = 0; i < n; i++) first[i] = initial[i] * emission[i][obs[0]];
  let c = first.reduce((a, b) => a + b, 0.0);
  scales[0] = c;
  alpha.push(c > 0.0 ? first.map((v) => v / c) : first.slice());
  for (let t = 1; t < T; t++) {
    const previous = alpha[t - 1];
    const row = new Array(n).fill(0.0);
    for (let j = 0; j < n; j++) {
      let acc = 0.0;
      for (let i = 0; i < n; i++) acc += previous[i] * transition[i][j];
      row[j] = acc * emission[j][obs[t]];
    }
    c = row.reduce((a, b) => a + b, 0.0);
    scales[t] = c;
    alpha.push(c > 0.0 ? row.map((v) => v / c) : row.slice());
  }
  let logLik = 0.0;
  for (const s of scales) logLik += s > 0.0 ? Math.log(s) : -Infinity;
  return { alpha, scales, logLik, likelihood: Math.exp(logLik) };
}

/** Unscaled forward pass (plain double products); only safe for short sequences. */
function forwardPlain(initial, transition, emission, obs) {
  const { n, m } = validateModel(initial, transition, emission);
  const T = checkObservations(obs, m);
  const alpha = [];
  const first = new Array(n);
  for (let i = 0; i < n; i++) first[i] = initial[i] * emission[i][obs[0]];
  alpha.push(first);
  for (let t = 1; t < T; t++) {
    const previous = alpha[t - 1];
    const row = new Array(n).fill(0.0);
    for (let j = 0; j < n; j++) {
      let acc = 0.0;
      for (let i = 0; i < n; i++) acc += previous[i] * transition[i][j];
      row[j] = acc * emission[j][obs[t]];
    }
    alpha.push(row);
  }
  return { alpha, likelihood: alpha[T - 1].reduce((a, b) => a + b, 0.0) };
}

/** Scaled backward pass re-using the forward scaling factors. */
function backwardScaled(initial, transition, emission, obs, scales) {
  const { n, m } = validateModel(initial, transition, emission);
  const T = checkObservations(obs, m);
  if (scales.length !== T) throw new Error("scales length mismatch");
  const beta = new Array(T);
  beta[T - 1] = new Array(n).fill(1.0);
  for (let t = T - 2; t >= 0; t--) {
    const next = beta[t + 1];
    const row = new Array(n).fill(0.0);
    for (let i = 0; i < n; i++) {
      let acc = 0.0;
      for (let j = 0; j < n; j++) {
        acc += transition[i][j] * emission[j][obs[t + 1]] * next[j];
      }
      row[i] = scales[t + 1] > 0.0 ? acc / scales[t + 1] : acc;
    }
    beta[t] = row;
  }
  return beta;
}

/** gamma (per-state posteriors) and xi (per-transition posteriors) from the scaled tables. */
function forwardBackward(initial, transition, emission, obs) {
  const forward = forwardScaled(initial, transition, emission, obs);
  const T = forward.alpha.length;
  const n = initial.length;
  const beta = backwardScaled(initial, transition, emission, obs, forward.scales);
  const gamma = [];
  for (let t = 0; t < T; t++) {
    const row = new Array(n).fill(0.0);
    let z = 0.0;
    for (let i = 0; i < n; i++) {
      row[i] = forward.alpha[t][i] * beta[t][i];
      z += row[i];
    }
    gamma.push(z > 0.0 ? row.map((v) => v / z) : row.slice());
  }
  const xi = [];
  for (let t = 0; t < T - 1; t++) {
    const grid = [];
    let z = 0.0;
    for (let i = 0; i < n; i++) {
      const row = new Array(n).fill(0.0);
      for (let j = 0; j < n; j++) {
        row[j] = forward.alpha[t][i] * transition[i][j] * emission[j][obs[t + 1]] * beta[t + 1][j];
        z += row[j];
      }
      grid.push(row);
    }
    if (z > 0.0) for (const row of grid) for (let j = 0; j < n; j++) row[j] /= z;
    xi.push(grid);
  }
  return { gamma, xi, beta, forward };
}

/** One Baum-Welch re-estimation step. */
function baumWelchStep(initial, transition, emission, obs) {
  const { gamma, xi } = forwardBackward(initial, transition, emission, obs);
  const n = initial.length;
  const m = emission[0].length;
  const T = obs.length;

  const newInitial = gamma[0].slice();
  const newTransition = [];
  for (let i = 0; i < n; i++) {
    let denominator = 0.0;
    for (let t = 0; t < T - 1; t++) denominator += gamma[t][i];
    const row = new Array(n).fill(0.0);
    if (denominator > 0.0) {
      for (let j = 0; j < n; j++) {
        let numerator = 0.0;
        for (let t = 0; t < T - 1; t++) numerator += xi[t][i][j];
        row[j] = numerator / denominator;
      }
    } else {
      row.splice(0, n, ...transition[i]);
    }
    newTransition.push(row);
  }
  const newEmission = [];
  for (let i = 0; i < n; i++) {
    let denominator = 0.0;
    for (let t = 0; t < T; t++) denominator += gamma[t][i];
    const row = new Array(m).fill(0.0);
    if (denominator > 0.0) {
      for (let t = 0; t < T; t++) row[obs[t]] += gamma[t][i];
      for (let k = 0; k < m; k++) row[k] /= denominator;
    } else {
      row.splice(0, m, ...emission[i]);
    }
    newEmission.push(row);
  }
  return { initial: newInitial, transition: newTransition, emission: newEmission,
    logLik: forwardScaled(initial, transition, emission, obs).logLik };
}

/** Iterate Baum-Welch, returning the parameter history and the log-likelihood trace. */
function baumWelch(initial, transition, emission, obs, iterations) {
  let model = { initial, transition, emission };
  const trace = [forwardScaled(initial, transition, emission, obs).logLik];
  const history = [];
  for (let i = 0; i < iterations; i++) {
    const step = baumWelchStep(model.initial, model.transition, model.emission, obs);
    model = { initial: step.initial, transition: step.transition, emission: step.emission };
    history.push(model);
    trace.push(forwardScaled(model.initial, model.transition, model.emission, obs).logLik);
  }
  return { model, trace, history };
}

/** Brute-force posterior marginals by enumerating all N^T paths. */
function bruteForcePosteriors(initial, transition, emission, obs) {
  const { n, m } = validateModel(initial, transition, emission);
  const T = checkObservations(obs, m);
  const numer = [];
  for (let t = 0; t < T; t++) numer.push(new Array(n).fill(0.0));
  const path = new Array(T).fill(0);
  let total = 0.0;
  const walk = (depth) => {
    if (depth === T) {
      let p = initial[path[0]] * emission[path[0]][obs[0]];
      for (let t = 1; t < T; t++) {
        p *= transition[path[t - 1]][path[t]] * emission[path[t]][obs[t]];
      }
      total += p;
      for (let t = 0; t < T; t++) numer[t][path[t]] += p;
      return;
    }
    for (let s = 0; s < n; s++) {
      path[depth] = s;
      walk(depth + 1);
    }
  };
  walk(0);
  const gamma = numer.map((row) => (total > 0.0 ? row.map((v) => v / total) : row));
  const occupancy = new Array(n).fill(0.0);
  for (const row of gamma) for (let i = 0; i < n; i++) occupancy[i] += row[i];
  return { gamma, occupancy, likelihood: total };
}

function l1Distance(a, b) {
  let total = 0.0;
  for (let i = 0; i < a.length; i++) total += Math.abs(a[i] - b[i]);
  return total;
}

function matrixL1(a, b) {
  return a.reduce((acc, row, i) => acc + l1Distance(row, b[i]), 0.0);
}

function lcgStream(n, seed, limit) {
  const out = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    out.push(seed % limit);
  }
  return out;
}

function main() {
  const assert = require("node:assert");

  const initial = [0.6, 0.4];
  const transition = [[0.7, 0.3], [0.4, 0.6]];
  const emission = [[0.8, 0.2], [0.3, 0.7]];

  // T = 1: the posterior is the hand-computable normalised pi_i B_i(o_0).
  const oneStep = forwardBackward(initial, transition, emission, [0]);
  const p0 = 0.6 * 0.8;
  const p1 = 0.4 * 0.3;
  assert.ok(Math.abs(oneStep.forward.logLik - Math.log(p0 + p1)) <= 1e-12);
  assert.ok(Math.abs(oneStep.gamma[0][0] - p0 / (p0 + p1)) <= 1e-12);
  assert.ok(Math.abs(oneStep.gamma[0][1] - p1 / (p0 + p1)) <= 1e-12);
  assert.deepStrictEqual(oneStep.xi, [], "no transitions for a single observation");
  assert.deepStrictEqual(oneStep.beta[0], [1.0, 1.0]);

  const cases = [
    ["short", lcgStream(4, 11, 2)],
    ["medium", lcgStream(6, 23, 2)],
    ["long", lcgStream(9, 37, 2)],
  ];
  for (const [name, obs] of cases) {
    const reference = bruteForcePosteriors(initial, transition, emission, obs);
    const scaled = forwardBackward(initial, transition, emission, obs);
    assert.ok(Math.abs(scaled.forward.logLik - Math.log(reference.likelihood)) <= 1e-12,
      `${name}: scaled forward likelihood matches exhaustive summation`);
    const plain = forwardPlain(initial, transition, emission, obs);
    assert.ok(Math.abs(plain.likelihood - reference.likelihood) <= 1e-14 * reference.likelihood,
      `${name}: unscaled forward pass agrees`);
    for (let t = 0; t < obs.length; t++) {
      assert.ok(Math.abs(scaled.gamma[t].reduce((a, b) => a + b, 0.0) - 1.0) <= 1e-12,
        `${name}: gamma row ${t} sums to one`);
      assert.ok(Math.abs(scaled.forward.alpha[t].reduce((a, b) => a + b, 0.0) - 1.0) <= 1e-12,
        `${name}: alpha_hat row ${t} sums to one`);
      for (let i = 0; i < 2; i++) {
        assert.ok(Math.abs(scaled.gamma[t][i] - reference.gamma[t][i]) <= 1e-12,
          `${name}: gamma[${t}][${i}] matches path enumeration`);
      }
    }
    const occupancy = new Array(2).fill(0.0);
    for (const row of scaled.gamma) for (let i = 0; i < 2; i++) occupancy[i] += row[i];
    assert.ok(l1Distance(occupancy, reference.occupancy) <= 1e-12,
      `${name}: expected occupancies match`);
    for (let t = 0; t < obs.length - 1; t++) {
      let total = 0.0;
      for (let i = 0; i < 2; i++) {
        let rowSum = 0.0;
        for (let j = 0; j < 2; j++) {
          total += scaled.xi[t][i][j];
          rowSum += scaled.xi[t][i][j];
        }
        assert.ok(Math.abs(rowSum - scaled.gamma[t][i]) <= 1e-12,
          `${name}: sum_j xi_t(i,j) = gamma_t(i) at t = ${t}`);
      }
      assert.ok(Math.abs(total - 1.0) <= 1e-12, `${name}: xi sums to one at t = ${t}`);
    }
  }

  // A three-state model, longer sequence, exhaustive cross-check.
  const initial3 = [0.2, 0.5, 0.3];
  const transition3 = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]];
  const emission3 = [[0.5, 0.4, 0.1], [0.1, 0.5, 0.4], [0.3, 0.3, 0.4]];
  const obs3 = lcgStream(6, 61, 3);
  const reference3 = bruteForcePosteriors(initial3, transition3, emission3, obs3);
  const scaled3 = forwardBackward(initial3, transition3, emission3, obs3);
  assert.ok(Math.abs(scaled3.forward.logLik - Math.log(reference3.likelihood)) <= 1e-12,
    "3-state: likelihood matches 729-path enumeration");
  for (let t = 0; t < obs3.length; t++) {
    assert.ok(l1Distance(scaled3.gamma[t], reference3.gamma[t]) <= 1e-12,
      `3-state: gamma row ${t} matches enumeration`);
  }
  for (let t = 0; t < obs3.length - 1; t++) {
    for (let i = 0; i < 3; i++) {
      let rowSum = 0.0;
      for (let j = 0; j < 3; j++) rowSum += scaled3.xi[t][i][j];
      assert.ok(Math.abs(rowSum - scaled3.gamma[t][i]) <= 1e-12, "3-state: xi/gamma identity");
    }
  }

  // Scaling keeps very long sequences finite (a plain product would underflow to zero).
  const longObs = lcgStream(5000, 909, 2);
  const longScaled = forwardScaled(initial, transition, emission, longObs);
  const longPlain = forwardPlain(initial, transition, emission, longObs);
  assert.ok(Number.isFinite(longScaled.logLik) && longScaled.logLik < 0.0,
    "the scaled log-likelihood stays finite for 5000 observations");
  assert.strictEqual(longPlain.likelihood, 0.0, "the unscaled product underflows to zero");
  assert.ok(longScaled.logLik < -1500, "and is far below anything a double can hold");

  // T = 1 Baum-Welch step is hand-checkable: pi becomes the posterior and B becomes the indicator.
  const step = baumWelchStep(initial, transition, emission, [0]);
  assert.ok(Math.abs(step.initial[0] - p0 / (p0 + p1)) <= 1e-12);
  assert.ok(Math.abs(step.emission[0][0] - 1.0) <= 1e-12 && Math.abs(step.emission[0][1]) <= 1e-12,
    "with one observation the emission rows become indicator vectors");
  assert.deepStrictEqual(step.transition, transition, "no transition statistics exist yet");

  // Baum-Welch on a single simulated sequence: monotone likelihood, valid parameters, and a
  // converging trace (the residual gain shrinks relative to the total gain).
  const truth = { initial: [0.5, 0.3, 0.2], transition: [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2],
    [0.2, 0.2, 0.6]], emission: [[0.6, 0.3, 0.1], [0.1, 0.6, 0.3], [0.3, 0.1, 0.6]] };
  const sampleFrom = (model, steps, seedValue) => {
    let seed = seedValue;
    const draw = (probs) => {
      seed = (seed * 75 + 74) % 65537;
      const u = seed / 65537.0;
      let acc = 0.0;
      for (let i = 0; i < probs.length; i++) {
        acc += probs[i];
        if (u < acc) return i;
      }
      return probs.length - 1;
    };
    const out = [];
    let state = draw(model.initial);
    for (let t = 0; t < steps; t++) {
      out.push(draw(model.emission[state]));
      state = draw(model.transition[state]);
    }
    return out;
  };
  const jitter = (row, s) => {
    const out = row.map((v, i) => v + ((s + i * 7) % 5) * 0.01);
    const z = out.reduce((a, b) => a + b, 0.0);
    return out.map((v) => v / z);
  };
  const data = sampleFrom(truth, 1500, 20240607);
  let s0 = 3;
  const startInitial = jitter(truth.initial, 1);
  const startTransition = truth.transition.map((row) => {
    s0 += 5;
    return jitter(row, s0);
  });
  const startEmission = truth.emission.map((row) => {
    s0 += 5;
    return jitter(row, s0);
  });

  const startLogLik = forwardScaled(startInitial, startTransition, startEmission, data).logLik;
  const run = baumWelch(startInitial, startTransition, startEmission, data, 300);
  assert.strictEqual(run.trace.length, 301);
  assert.ok(Math.abs(run.trace[0] - startLogLik) <= 1e-9);
  for (let i = 1; i < run.trace.length; i++) {
    assert.ok(run.trace[i] >= run.trace[i - 1] - 1e-9,
      `iteration ${i}: the log-likelihood must not decrease (${run.trace[i - 1]} -> ${run.trace[i]})`);
  }
  const totalGain = run.trace[run.trace.length - 1] - run.trace[0];
  const tailGain = run.trace[run.trace.length - 1] - run.trace[run.trace.length - 11];
  assert.ok(totalGain > 1.0, `the fit improves by ${totalGain} nats`);
  assert.ok(tailGain <= 0.01 * totalGain,
    `the last 10 iterations gain ${tailGain}, under 1% of the total gain`);

  const checkRow = (row, label) => {
    let sum = 0.0;
    for (const v of row) {
      assert.ok(v >= 0.0, `${label} stays non-negative`);
      sum += v;
    }
    assert.ok(Math.abs(sum - 1.0) <= 1e-9, `${label} stays a probability distribution`);
  };
  const checkModel = (model, label) => {
    checkRow(model.initial, `${label} initial`);
    for (const row of model.transition) checkRow(row, `${label} transition`);
    for (const row of model.emission) checkRow(row, `${label} emission`);
  };
  checkModel(run.model, "fitted");
  for (const history of run.history) checkModel(history, "intermediate");

  // A long, well-separated 2-state sequence: EM must move the transition and emission rows
  // measurably towards the generating model and stop moving (fixed point).
  const truth2 = { initial: [0.6, 0.4], transition: [[0.9, 0.1], [0.1, 0.9]],
    emission: [[0.9, 0.1], [0.1, 0.9]] };
  const data2 = sampleFrom(truth2, 20000, 4242);
  const start2 = {
    initial: jitter(truth2.initial, 2),
    transition: truth2.transition.map((row, i) => jitter(row, 7 + i * 5)),
    emission: truth2.emission.map((row, i) => jitter(row, 11 + i * 5)),
  };
  const recovery = baumWelch(start2.initial, start2.transition, start2.emission, data2, 100);
  for (let i = 1; i < recovery.trace.length; i++) {
    assert.ok(recovery.trace[i] >= recovery.trace[i - 1] - 1e-9, "recovery run stays monotone");
  }
  const startErrorA = matrixL1(start2.transition, truth2.transition);
  const finalErrorA = matrixL1(recovery.model.transition, truth2.transition);
  const startErrorB = matrixL1(start2.emission, truth2.emission);
  const finalErrorB = matrixL1(recovery.model.emission, truth2.emission);
  assert.ok(finalErrorA * 5.0 <= startErrorA,
    `transition row error shrinks from ${startErrorA} to ${finalErrorA}`);
  assert.ok(finalErrorB * 5.0 <= startErrorB,
    `emission row error shrinks from ${startErrorB} to ${finalErrorB}`);
  assert.ok(recovery.trace[recovery.trace.length - 1] > recovery.trace[0] + 10.0,
    "20000 observations give a large likelihood gain");
  const last = recovery.history[recovery.history.length - 1];
  const previous = recovery.history[recovery.history.length - 2];
  const stepChange = l1Distance(last.initial, previous.initial) +
    matrixL1(last.transition, previous.transition) + matrixL1(last.emission, previous.emission);
  assert.ok(stepChange <= 1e-5,
    `at iteration 100 a further step moves the parameters by only ${stepChange} (fixed point)`);

  assert.throws(() => forwardScaled(initial, transition, emission, []), /empty observation/);
  assert.throws(() => forwardScaled(initial, transition, emission, [2]), /bad symbol/);
  assert.throws(() => forwardScaled([1.0, 0.0, 0.0], transition, emission, [0]),
    /dimension mismatch/);
  assert.throws(() => forwardScaled([0.5, 0.6], transition, emission, [0]), /must sum to 1/);
  assert.throws(() => backwardScaled(initial, transition, emission, [0, 1], [1.0]),
    /scales length mismatch/);
}

module.exports = { validateModel, checkObservations, forwardScaled, forwardPlain, backwardScaled,
  forwardBackward, baumWelchStep, baumWelch, bruteForcePosteriors, l1Distance, matrixL1,
  lcgStream };

if (require.main === module) {
  main();
  console.log("PASS");
}
