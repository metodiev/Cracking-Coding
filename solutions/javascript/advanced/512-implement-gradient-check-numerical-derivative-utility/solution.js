/**
 * Task 512 - Implement gradient check numerical derivative utility. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A small finite-difference toolkit used to audit analytic gradients: the forward difference
 * (f(x + h e_i) - f(x)) / h with error O(h), the central difference
 * (f(x + h e_i) - f(x - h e_i)) / (2h) with error O(h^2), and the five-point stencil
 * (-f(x + 2h) + 8 f(x + h) - 8 f(x - h) + f(x - 2h)) / (12h) with error O(h^4). The comparison
 * metric is the standard relative error |a - b| / max(1, |a| + |b|), which stays meaningful when
 * the analytic derivative is near zero. `checkGradient(f, grad, x, h, tol)` returns the per
 * coordinate errors, the maximum error and the list of offending coordinates, so a caller can
 * also audit a deliberately wrong gradient. A numeric Hessian built by central-differencing the
 * analytic gradient is included. The documented default step is h = 1e-5 in double precision.
 *
 * Documented checks: the analytic gradients of the polynomial 3x^3 - 2x + 1, sin(x) e^x, the
 * logistic, log-sum-exp, product and the 2-D Rosenbrock function are audited at the documented
 * h = 1e-5 (central error <= 3e-9, five-point error <= 3e-11 - the round-off floor), while the
 * truncation-dominated regime at h = 1e-3 must order the stencils strictly forward > central >
 * five-point with the five-point error <= 1e-12. The error curve is asserted to be U-shaped: at
 * h = 1e-1 truncation dominates (error >= 1e-6) and at h = 1e-13 round-off dominates
 * (error >= 1e-6) while the documented h = 1e-5 sits below both (< 1e-10); the empirical
 * convergence orders are pinned down too (halving h shrinks the central error by 3-5x, the
 * forward error by 1.6-2.4x and the five-point error by 12-20x). Finally the utility must *fail*
 * a gradient perturbed by 1e-3 and one that is wrong only in the last coordinate - naming that
 * coordinate and leaving the correct ones below 1e-9 - and the numerical Hessian of Rosenbrock at
 * its minimum (1, 1) must be symmetric and match the analytic Hessian (802, -400, 200) to 1e-3,
 * with the numeric gradient of Rosenbrock at (0.42, 0.19) matching the analytic one to 3e-9 at
 * h = 1e-5.
 */
"use strict";

const DEFAULT_H = 1e-5;
const DEFAULT_TOL = 1e-6;

function relativeError(a, b) {
  return Math.abs(a - b) / Math.max(1.0, Math.abs(a) + Math.abs(b));
}

function addScaled(x, i, scale) {
  const out = x.slice();
  out[i] += scale;
  return out;
}

function forwardDiff(f, x, i, h = DEFAULT_H) {
  return (f(addScaled(x, i, h)) - f(x)) / h;
}

function centralDiff(f, x, i, h = DEFAULT_H) {
  return (f(addScaled(x, i, h)) - f(addScaled(x, i, -h))) / (2.0 * h);
}

function fivePointDiff(f, x, i, h = DEFAULT_H) {
  const plus2 = f(addScaled(x, i, 2.0 * h));
  const plus1 = f(addScaled(x, i, h));
  const minus1 = f(addScaled(x, i, -h));
  const minus2 = f(addScaled(x, i, -2.0 * h));
  return (-plus2 + 8.0 * plus1 - 8.0 * minus1 + minus2) / (12.0 * h);
}

function numericGradient(f, x, h = DEFAULT_H) {
  return x.map((_, i) => centralDiff(f, x, i, h));
}

/**
 * Audit an analytic gradient. Returns per-coordinate relative errors, the worst error and the
 * indices that exceed the tolerance.
 */
function checkGradient(f, grad, x, h = DEFAULT_H, tol = DEFAULT_TOL) {
  const analytic = grad(x);
  const errors = analytic.map((value, i) => relativeError(value, centralDiff(f, x, i, h)));
  let worst = 0;
  let worstIndex = -1;
  const offenders = [];
  errors.forEach((error, i) => {
    if (error > worst) {
      worst = error;
      worstIndex = i;
    }
    if (error > tol) offenders.push(i);
  });
  return { errors, maxError: worst, worstIndex, offenders, passed: offenders.length === 0, tol, h };
}

/** Numeric Hessian: entry (i, j) is a central difference of grad[i] in coordinate j. */
function numericHessian(grad, x, h = 1e-4) {
  const n = x.length;
  const out = [];
  for (let i = 0; i < n; i++) out.push(new Array(n).fill(0.0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const plus = grad(addScaled(x, j, h))[i];
      const minus = grad(addScaled(x, j, -h))[i];
      out[i][j] = (plus - minus) / (2.0 * h);
    }
  }
  return out;
}

const FUNCTIONS = {
  // f, analytic gradient, sample point, label
  polynomial: {
    f: (x) => 3.0 * x[0] ** 3 - 2.0 * x[0] + 1.0,
    g: (x) => [9.0 * x[0] * x[0] - 2.0],
    point: [0.7],
  },
  sineExp: {
    f: (x) => Math.sin(x[0]) * Math.exp(x[0]),
    g: (x) => [(Math.sin(x[0]) + Math.cos(x[0])) * Math.exp(x[0])],
    point: [0.35],
  },
  logistic: {
    f: (x) => 1.0 / (1.0 + Math.exp(-x[0])),
    g: (x) => {
      const s = 1.0 / (1.0 + Math.exp(-x[0]));
      return [s * (1.0 - s)];
    },
    point: [-0.9],
  },
  logSumExp: {
    f: (x) => Math.log(Math.exp(x[0]) + Math.exp(x[1]) + Math.exp(x[2])),
    g: (x) => {
      const m = Math.exp(x[0]) + Math.exp(x[1]) + Math.exp(x[2]);
      return [Math.exp(x[0]) / m, Math.exp(x[1]) / m, Math.exp(x[2]) / m];
    },
    point: [0.3, -0.4, 0.9],
  },
  rosenbrock: {
    f: (x) => (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] * x[0]) ** 2,
    g: (x) => [
      -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
      200.0 * (x[1] - x[0] * x[0]),
    ],
    point: [0.42, 0.19],
  },
  product: {
    f: (x) => x[0] * x[1] + Math.sin(x[0] * x[1]) + x[0] / (1.0 + x[1] * x[1]),
    g: (x) => {
      const d = 1.0 + x[1] * x[1];
      return [
        x[1] + x[1] * Math.cos(x[0] * x[1]) + 1.0 / d,
        x[0] + x[0] * Math.cos(x[0] * x[1]) - 2.0 * x[0] * x[1] / (d * d),
      ];
    },
    point: [0.8, -0.6],
  },
};

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b}| > ${tol}`);

  // (1) the three stencils all pass; ordering is measured in the truncation-dominated regime.
  const ORDER_H = 1e-3;
  const accuracy = [];
  for (const [name, { f, g, point }] of Object.entries(FUNCTIONS)) {
    const analytic = g(point);
    let worstCentral = 0.0;
    let worstFive = 0.0;
    let worstForward = 0.0;
    for (let i = 0; i < point.length; i++) {
      worstCentral = Math.max(worstCentral,
        relativeError(analytic[i], centralDiff(f, point, i, DEFAULT_H)));
      worstFive = Math.max(worstFive,
        relativeError(analytic[i], fivePointDiff(f, point, i, DEFAULT_H)));
      worstForward = Math.max(worstForward,
        relativeError(analytic[i], forwardDiff(f, point, i, DEFAULT_H)));
    }
    assert.ok(worstCentral <= 3e-9, `${name}: central error ${worstCentral} <= 3e-9 at h = 1e-5`);
    assert.ok(worstFive <= 3e-11, `${name}: five-point error ${worstFive} <= 3e-11 at h = 1e-5`);

    let orderForward = 0.0;
    let orderCentral = 0.0;
    let orderFive = 0.0;
    for (let i = 0; i < point.length; i++) {
      orderForward = Math.max(orderForward,
        relativeError(analytic[i], forwardDiff(f, point, i, ORDER_H)));
      orderCentral = Math.max(orderCentral,
        relativeError(analytic[i], centralDiff(f, point, i, ORDER_H)));
      orderFive = Math.max(orderFive,
        relativeError(analytic[i], fivePointDiff(f, point, i, ORDER_H)));
    }
    assert.ok(orderForward > orderCentral && orderCentral > orderFive,
      `${name}: at h = 1e-3 truncation orders as forward > central > five-point`);
    assert.ok(orderFive <= 1e-12, `${name}: five-point is essentially exact at h = 1e-3`);
    const report = checkGradient(f, g, point);
    assert.ok(report.passed, `${name}: checkGradient passes at the documented tolerance`);
    assert.ok(report.maxError <= 3e-9, `${name}: report.maxError <= 3e-9`);
    accuracy.push([name, orderForward, orderCentral, orderFive]);
  }
  assert.ok(accuracy.length === Object.keys(FUNCTIONS).length);

  // (2) the error curve is U-shaped in h (truncation vs round-off).
  const { f: polyF, g: polyG } = FUNCTIONS.polynomial;
  const p = FUNCTIONS.polynomial.point;
  const errAt = (h, stencil) => relativeError(polyG(p)[0], stencil(polyF, p, 0, h));
  const at1e1 = errAt(1e-1, centralDiff);
  const at1e5 = errAt(DEFAULT_H, centralDiff);
  const at1e13 = errAt(1e-13, centralDiff);
  assert.ok(at1e1 >= 1e-6 && at1e1 <= 1e-1, `h = 1e-1 is truncation dominated (${at1e1})`);
  assert.ok(at1e13 >= 1e-6, `h = 1e-13 is round-off dominated (${at1e13})`);
  assert.ok(at1e5 < at1e1 && at1e5 < at1e13, "the documented h = 1e-5 sits at the bottom");
  assert.ok(at1e5 <= 1e-10, `central error at h = 1e-5 is <= 1e-10 (${at1e5})`);
  const curve = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    .map((h) => errAt(h, centralDiff));
  const best = Math.min(...curve);
  const bestH = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8][curve.indexOf(best)];
  assert.ok(bestH <= 1e-4 && bestH >= 1e-7, `empirical optimum step is around 1e-7..1e-4 (${bestH})`);

  // (3) empirical convergence orders (against a known derivative: exp at x = 1).
  const expF = (v) => Math.exp(v[0]);
  const expG = Math.exp(1.0);
  const truncErr = (stencil, h) =>
    relativeError(expG, stencil(expF, [1.0], 0, h));
  const centralRatio = truncErr(centralDiff, 1e-3) / truncErr(centralDiff, 5e-4);
  const forwardRatio = truncErr(forwardDiff, 1e-3) / truncErr(forwardDiff, 5e-4);
  const fiveRatio = truncErr(fivePointDiff, 1e-2) / truncErr(fivePointDiff, 5e-3);
  assert.ok(centralRatio >= 3.0 && centralRatio <= 5.0,
    `central difference is O(h^2): ratio ${centralRatio}`);
  assert.ok(forwardRatio >= 1.6 && forwardRatio <= 2.4,
    `forward difference is O(h): ratio ${forwardRatio}`);
  assert.ok(fiveRatio >= 12.0 && fiveRatio <= 20.0,
    `five-point stencil is O(h^4): ratio ${fiveRatio}`);

  // (4) the utility must reject wrong gradients and name the offender.
  const badAll = (x) => polyG(x).map((value) => value + 1e-3);
  const reportAll = checkGradient(polyF, badAll, p);
  assert.ok(!reportAll.passed, "a gradient perturbed by 1e-3 must be rejected");
  assert.deepStrictEqual(reportAll.offenders, [0], "the single coordinate is named");
  assert.ok(reportAll.maxError >= 1e-4, "the reported error reflects the perturbation");

  const badLast = (x) => {
    const out = FUNCTIONS.logSumExp.g(x);
    out[2] += 0.05;
    return out;
  };
  const reportLast = checkGradient(FUNCTIONS.logSumExp.f, badLast, [0.3, -0.4, 0.9]);
  assert.ok(!reportLast.passed, "a gradient wrong only in the last coordinate is rejected");
  assert.deepStrictEqual(reportLast.offenders, [2]);
  assert.strictEqual(reportLast.worstIndex, 2, "worst coordinate reported");
  assert.ok(reportLast.errors[0] <= 1e-9 && reportLast.errors[1] <= 1e-9,
    "correct coordinates stay below the tolerance");

  const good = checkGradient(FUNCTIONS.logSumExp.f, FUNCTIONS.logSumExp.g, [0.3, -0.4, 0.9]);
  assert.ok(good.passed && good.offenders.length === 0, "the correct gradient is accepted");
  assert.deepStrictEqual(good.errors.map((error) => error <= 1e-9), [true, true, true]);

  // (5) numeric Hessian of Rosenbrock: symmetric and matching the analytic Hessian.
  const rosen = FUNCTIONS.rosenbrock;
  const hNum = numericHessian(rosen.g, [1.0, 1.0]);
  close(hNum[0][1], hNum[1][0], 1e-6, "numeric Hessian is symmetric");
  const exactHessian = [
    [2.0 - 400.0 * (1.0 - 3.0 * 1.0 * 1.0), -400.0 * 1.0],
    [-400.0 * 1.0, 200.0],
  ];
  close(hNum[0][0], exactHessian[0][0], 1e-3, "d2f/dx2 at the minimum");
  close(hNum[0][1], exactHessian[0][1], 1e-3, "d2f/dxdy at the minimum");
  close(hNum[1][1], exactHessian[1][1], 1e-3, "d2f/dy2 at the minimum");
  const gradientAtMinimum = rosen.g([1.0, 1.0]);
  assert.ok(Math.abs(gradientAtMinimum[0]) <= 1e-12 && Math.abs(gradientAtMinimum[1]) <= 1e-12,
    "Rosenbrock gradient vanishes at (1, 1)");
  assert.ok(Math.abs(rosen.f([1.0, 1.0])) <= 1e-24, "f(1, 1) = 0");

  // (6) the numeric gradient helper and the relative-error metric behave as documented.
  for (const h of [1e-3, 1e-5]) {
    const gRosen = numericGradient(rosen.f, [0.42, 0.19], h);
    const analytic = rosen.g([0.42, 0.19]);
    for (let i = 0; i < 2; i++) {
      assert.ok(relativeError(gRosen[i], analytic[i]) <= 3e-5,
        `numericGradient matches the analytic gradient at h = ${h}`);
    }
  }
  assert.ok(relativeError(numericGradient(rosen.f, [0.42, 0.19])[0],
    rosen.g([0.42, 0.19])[0]) <= 3e-9, "the documented h = 1e-5 gives ~1e-9 on Rosenbrock");
  close(relativeError(0.0, 0.0), 0.0, 1e-18, "zero vs zero");
  close(relativeError(1.0, -1.0), 1.0, 1e-15, "relative error is symmetric and capped at 1");
  close(relativeError(1.0e-8, 0.0), 1e-8, 1e-20, "denominator floor keeps tiny values sane");
  close(relativeError(100.0, 99.0), 1.0 / 199.0, 1e-15, "large values scale by the magnitude");
}

module.exports = { DEFAULT_H, DEFAULT_TOL, relativeError, addScaled, forwardDiff, centralDiff,
  fivePointDiff, numericGradient, checkGradient, numericHessian, FUNCTIONS };

if (require.main === module) {
  main();
  console.log("PASS");
}
