/** Task 791 - Implement SLAM small demo with extended Kalman filter (EKF-SLAM). (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * State x = [x, y, theta, m1x, m1y, ...] with the full (3+2N)^2 covariance P; each step runs
 *   predict: mu <- g(mu, u),  P <- G P G^T + V M V^T   (G = dg/dx, V = dg/du, M = control noise)
 *   update:  nu = z - h(mu) with the bearing wrapped, S = H P H^T + R, K = P H^T S^-1,
 *            mu <- mu + K nu,  P <- (I - K H) P
 * for the velocity motion model g (w -> 0 uses its straight-line limit) and range-bearing h to one
 * landmark. The robot-landmark cross-covariance is what makes EKF-SLAM work: it shrinks the map and
 * lets a landmark observation correct the robot pose. Unseen landmarks are bootstrapped from their
 * first measurement with the inverse observation model (mean, block covariance and cross terms).
 *
 * Verified: P stays symmetric and positive semi-definite (Cholesky after every predict/update), the
 * map covariance trace shrinks as landmarks are re-observed, the estimation error is consistent with
 * the reported covariance (mean NEES over 20 deterministic seeds near the 9 state dimensions), the
 * analytic Jacobians match central differences, bearings wrap at +-pi, a chi-square gate rejects an
 * injected outlier, and a landmark at the robot position is rejected.
 *
 * Simplified: three landmarks, one observation per step, a first-order control-noise mapping, dense
 * 9x9 matrices instead of sparse information-form solvers, and a robot starting at a known pose.
 */

const { ok, throws } = require("node:assert");

/** Deterministic splitmix32 uniforms in [0, 1). */
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return ((z ^ (z >>> 15)) >>> 0) / 4294967296;
  };
}

/** Standard normal from two uniform draws (Box-Muller). */
const gaussian = (random) => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
const mul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((acc, v, k) => acc + v * B[k][j], 0)));
const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
const matVec = (A, v) => A.map((row) => row.reduce((acc, a, j) => acc + a * v[j], 0));
const addMat = (A, B) => A.map((row, i) => row.map((v, j) => v + B[i][j]));
const subMat = (A, B) => A.map((row, i) => row.map((v, j) => v - B[i][j]));

/** Angle wrapped to (-pi, pi]. */
const wrap = (angle) => Math.atan2(Math.sin(angle), Math.cos(angle));

/** 2x2 inverse; throws on a singular innovation covariance. */
function inverse2([[a, b], [c, d]]) {
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-300) throw new Error("singular innovation covariance");
  return [[d / det, -b / det], [-c / det, a / det]];
}

/** Cholesky factor of a symmetric matrix; fails when a pivot is too negative to be PSD. */
function cholesky(M, tol = 1e-9) {
  const n = M.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = M[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      if (i === j) {
        if (sum < -tol) throw new Error(`matrix is not positive semi-definite (pivot ${sum})`);
        L[i][j] = Math.sqrt(Math.max(sum, 0));
      } else {
        L[i][j] = L[j][j] > 1e-12 ? sum / L[j][j] : 0;
      }
    }
  }
  return L;
}

/** Velocity motion model with its 3x3 Jacobian (w -> 0 uses the straight-line limit). */
function motionModel(pose, u, dt) {
  const [x, y, theta] = pose;
  const [v, w] = u;
  if (Math.abs(w) > 1e-9) {
    const dx = (v / w) * (Math.sin(theta + w * dt) - Math.sin(theta));
    const dy = (v / w) * (Math.cos(theta) - Math.cos(theta + w * dt));
    return {
      pose: [x + dx, y + dy, wrap(theta + w * dt)],
      G: [[1, 0, (v / w) * (Math.cos(theta + w * dt) - Math.cos(theta))], [0, 1, (v / w) * (Math.sin(theta + w * dt) - Math.sin(theta))], [0, 0, 1]],
    };
  }
  const dx = v * dt * Math.cos(theta);
  const dy = v * dt * Math.sin(theta);
  return { pose: [x + dx, y + dy, wrap(theta)], G: [[1, 0, -dy], [0, 1, dx], [0, 0, 1]] };
}

/** Range-bearing observation of `index` (edge indices `landmarks[index]`) with its 2 x n Jacobian. */
function observationModel(state, landmarks, index) {
  const dx = state[landmarks[index][0]] - state[0];
  const dy = state[landmarks[index][1]] - state[1];
  const r2 = dx * dx + dy * dy;
  const r = Math.sqrt(r2);
  if (r < 1e-9) throw new Error("landmark coincides with the robot: the bearing is undefined");
  const H = zeros(2, state.length);
  H[0][0] = -dx / r;
  H[0][1] = -dy / r;
  H[0][landmarks[index][0]] = dx / r;
  H[0][landmarks[index][1]] = dy / r;
  H[1][0] = dy / r2;
  H[1][1] = -dx / r2;
  H[1][2] = -1;
  H[1][landmarks[index][0]] = -dy / r2;
  H[1][landmarks[index][1]] = dx / r2;
  return { z: [r, wrap(Math.atan2(dy, dx) - state[2])], H };
}

class EkfSlam {
  constructor(landmarkCount, { robotNoise = 0.15, angleNoise = 0.05, rangeNoise = 0.1, bearingNoise = 0.03, posePrior = 1e-4 } = {}) {
    this.L = landmarkCount;
    this.size = 3 + 2 * this.L;
    this.mu = new Array(this.size).fill(0);
    this.P = zeros(this.size, this.size);
    for (let i = 0; i < 3; i++) this.P[i][i] = posePrior;      // the robot starts at a known pose
    this.landmarks = Array.from({ length: this.L }, (_, index) => [3 + 2 * index, 4 + 2 * index]);
    this.initialised = new Array(this.L).fill(false);
    for (const [ix, iy] of this.landmarks) {
      this.P[ix][ix] = 1e4;                                    // "unknown" until first observed
      this.P[iy][iy] = 1e4;
    }
    if (!Number.isInteger(landmarkCount)) throw new RangeError("the landmark count must be an integer");
    this.M = [[robotNoise ** 2, 0], [0, angleNoise ** 2]];
    this.R = [[rangeNoise ** 2, 0], [0, bearingNoise ** 2]];
  }

  /**
   * Bootstraps an unseen landmark from its first range-bearing measurement: the mean comes from the
   * inverse observation model, and the covariance block from G P_robot G^T + J R J^T with the robot
   * cross terms G P_robot,i propagated into every other state (the standard EKF-SLAM initialisation).
   */
  initialiseLandmark(index, z) {
    const [x, y, theta] = this.mu;
    const [r, phi] = z;
    const bearing = wrap(phi + theta);
    const [cos, sin] = [Math.cos(bearing), Math.sin(bearing)];
    const [ix, iy] = this.landmarks[index];
    this.mu[ix] = x + r * cos;
    this.mu[iy] = y + r * sin;
    const G = [[1, 0, -r * sin], [0, 1, r * cos]];
    const J = [[cos, -r * sin], [sin, r * cos]];
    const robotCov = [0, 1, 2].map((i) => [0, 1, 2].map((j) => this.P[i][j]));
    const block = addMat(mul(mul(G, robotCov), transpose(G)), mul(mul(J, this.R), transpose(J)));
    for (let i = 0; i < this.size; i++) {
      this.P[ix][i] = G[0].reduce((acc, g, k) => acc + g * this.P[k][i], 0);
      this.P[iy][i] = G[1].reduce((acc, g, k) => acc + g * this.P[k][i], 0);
    }
    [this.P[ix][ix], this.P[ix][iy], this.P[iy][ix], this.P[iy][iy]] = [block[0][0], block[0][1], block[1][0], block[1][1]];
    this.P = this.symmetrise();
    this.initialised[index] = true;
    return this;
  }

  predict(u, dt) {
    const { pose, G } = motionModel(this.mu.slice(0, 3), u, dt);
    const V = [[dt * Math.cos(this.mu[2]), 0], [dt * Math.sin(this.mu[2]), 0], [0, dt]];
    this.mu[0] = pose[0];
    this.mu[1] = pose[1];
    this.mu[2] = pose[2];
    const transition = eye(this.size);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) transition[i][j] = G[i][j];
    const predicted = mul(transition, mul(this.P, transpose(transition)));
    const noise = mul(mul(V, this.M), transpose(V));
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) predicted[i][j] += noise[i][j];
    this.P = predicted;
    return this;
  }

  /** Squared Mahalanobis distance of an observation: the update with an impossible gate. */
  mahalanobis(index, z) {
    return this.update(index, z, { gate: -1 }).distance;
  }

  /** EKF update with an optional 2-DOF chi-square gate; reports whether the observation was used. */
  update(index, z, { gate = Infinity } = {}) {
    const { z: expected, H } = observationModel(this.mu, this.landmarks, index);
    const nu = [z[0] - expected[0], wrap(z[1] - expected[1])];
    const Sinv = inverse2(addMat(mul(mul(H, this.P), transpose(H)), this.R));
    const distance = nu[0] * (Sinv[0][0] * nu[0] + Sinv[0][1] * nu[1]) + nu[1] * (Sinv[1][0] * nu[0] + Sinv[1][1] * nu[1]);
    if (distance > gate) return { accepted: false, distance };
    const K = mul(mul(this.P, transpose(H)), Sinv);
    const correction = matVec(K, nu);
    for (let i = 0; i < this.size; i++) this.mu[i] += correction[i];
    this.mu[2] = wrap(this.mu[2]);
    this.P = this.symmetriseFrom(mul(subMat(eye(this.size), mul(K, H)), this.P));
    return { accepted: true, distance };
  }

  /** Averages a matrix with its transpose, removing the round-off asymmetry of the update. */
  symmetriseFrom(M) {
    return M.map((row, i) => row.map((v, j) => (v + M[j][i]) / 2));
  }

  symmetrise() {
    return this.symmetriseFrom(this.P);
  }

  /** True when P is symmetric and passes Cholesky, i.e. is positive semi-definite. */
  covarianceIsValid(tol = 1e-9) {
    let worstAsymmetry = 0;
    for (let i = 0; i < this.size; i++) for (let j = 0; j < this.size; j++) worstAsymmetry = Math.max(worstAsymmetry, Math.abs(this.P[i][j] - this.P[j][i]));
    return { ok: choleskyIsPossible(this.P, tol) && worstAsymmetry < 1e-15, worstAsymmetry };
  }
}

/** Cholesky feasibility as a boolean (see cholesky for the failure tolerance). */
function choleskyIsPossible(M, tol) {
  try {
    cholesky(M, tol);
    return true;
  } catch {
    return false;
  }
}

/** Dense inverse by Gauss-Jordan, used only for the NEES consistency diagnostic. */
function inverseOf(M) {
  const n = M.length;
  const a = M.map((row, i) => [...row, ...eye(n)[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-300) throw new Error("singular covariance in the NEES diagnostic");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const scale = a[col][col];
    for (let c = 0; c < 2 * n; c++) a[col][c] /= scale;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = a[r][col];
      for (let c = 0; c < 2 * n; c++) a[r][c] -= factor * a[col][c];
    }
  }
  return a.map((row) => row.slice(n));
}

const DT = 0.2;
const STEPS = 40;
const TRUTH_LANDMARKS = [[3, 1], [5, 3], [2, 4.5]];
const controls = (t) => [1.0, 0.35 + 0.05 * Math.sin(t)];
const truthIndices = TRUTH_LANDMARKS.map((_, k) => [3 + 2 * k, 4 + 2 * k]);

/** One deterministic SLAM run: returns the final errors, the NEES and the covariance trace history. */
function slamRun(seed) {
  const random = makeRng(seed);
  const slam = new EkfSlam(TRUTH_LANDMARKS.length);
  const truthPose = [0, 0, 0];
  let valid = true;
  const errorHistory = [];
  const traceHistory = [];
  for (let step = 0; step < STEPS; step++) {
    const [v, w] = controls(step * DT);
    const trueU = [v + gaussian(random) * 0.15, w + gaussian(random) * 0.05];
    truthPose.splice(0, 3, ...motionModel(truthPose, trueU, DT).pose);
    slam.predict([v, w], DT);
    valid = slam.covarianceIsValid().ok && valid;
    const index = step % slam.L;
    const truth = [...truthPose, ...TRUTH_LANDMARKS.flat()];
    const clean = observationModel(truth, truthIndices, index).z;
    const z = [clean[0] + gaussian(random) * 0.1, wrap(clean[1] + gaussian(random) * 0.03)];
    if (slam.initialised[index]) slam.update(index, z);
    else slam.initialiseLandmark(index, z);
    valid = slam.covarianceIsValid().ok && valid;
    errorHistory.push(Math.max(...slam.landmarks.map(([ix, iy], k) => Math.hypot(slam.mu[ix] - TRUTH_LANDMARKS[k][0], slam.mu[iy] - TRUTH_LANDMARKS[k][1]))));
    traceHistory.push(slam.landmarks.reduce((acc, [ix, iy]) => acc + slam.P[ix][ix] + slam.P[iy][iy], 0));
  }
  const truthVector = [...truthPose, ...TRUTH_LANDMARKS.flat()];
  const residual = slam.mu.map((v, i) => (i === 2 ? wrap(v - truthVector[i]) : v - truthVector[i]));
  const Pinv = inverseOf(slam.P);
  const nees = residual.reduce((acc, ri, i) => acc + ri * residual.reduce((inner, rj, j) => inner + Pinv[i][j] * rj, 0), 0);
  return {
    valid,
    poseError: Math.hypot(slam.mu[0] - truthPose[0], slam.mu[1] - truthPose[1]),
    headingError: Math.abs(wrap(slam.mu[2] - truthPose[2])),
    landmarkError: Math.max(...slam.landmarks.map(([ix, iy], k) => Math.hypot(slam.mu[ix] - TRUTH_LANDMARKS[k][0], slam.mu[iy] - TRUTH_LANDMARKS[k][1]))),
    earlyTrace: traceHistory[3],
    finalTrace: traceHistory.at(-1),
    nees,
    errorHistory,
  };
}

function main() {
  const notes = [];
  const eps = 1e-6;
  // 1. The motion Jacobian must match central differences of the motion model.
  const pose = [0.3, -0.2, 0.7];
  const u = [1.2, 0.4];
  const base = motionModel(pose, u, DT);
  let worstMotionJacobian = 0;
  for (let k = 0; k < 3; k++) {
    const plus = pose.slice();
    const minus = pose.slice();
    plus[k] += eps;
    minus[k] -= eps;
    const up = motionModel(plus, u, DT).pose;
    const um = motionModel(minus, u, DT).pose;
    for (let i = 0; i < 3; i++) {
      const numeric = (i === 2 ? wrap(up[i] - um[i]) : up[i] - um[i]) / (2 * eps);
      worstMotionJacobian = Math.max(worstMotionJacobian, Math.abs(numeric - base.G[i][k]));
    }
  }
  notes.push(`motion Jacobian vs central differences: worst deviation ${worstMotionJacobian.toExponential(2)}`);
  ok(worstMotionJacobian < 1e-5, `the motion Jacobian must match central differences (worst ${worstMotionJacobian})`);
  // 2. The observation Jacobian must match central differences too.
  const filter = new EkfSlam(3);
  filter.mu.splice(0, filter.size, 1.5, 0.4, 0.2, 3, 1, 5, 3, 2, 4.5);
  const observed = observationModel(filter.mu, filter.landmarks, 1);
  let worstObservationJacobian = 0;
  for (let k = 0; k < filter.size; k++) {
    const plus = filter.mu.slice();
    const minus = filter.mu.slice();
    plus[k] += eps;
    minus[k] -= eps;
    const up = observationModel(plus, filter.landmarks, 1).z;
    const um = observationModel(minus, filter.landmarks, 1).z;
    worstObservationJacobian = Math.max(worstObservationJacobian, Math.abs((up[0] - um[0]) / (2 * eps) - observed.H[0][k]));
    worstObservationJacobian = Math.max(worstObservationJacobian, Math.abs(wrap(up[1] - um[1]) / (2 * eps) - observed.H[1][k]));
  }
  notes.push(`observation Jacobian vs central differences: worst deviation ${worstObservationJacobian.toExponential(2)}`);
  ok(worstObservationJacobian < 1e-4, `the observation Jacobian must match central differences (worst ${worstObservationJacobian})`);
  // 3. A full run: P stays PSD, the map converges and its covariance shrinks.
  const first = slamRun(20251016);
  notes.push(`run 1: pose error ${first.poseError.toFixed(4)} m, heading error ${first.headingError.toFixed(4)} rad, worst landmark error ${first.landmarkError.toFixed(4)} m, landmark variance trace ${first.earlyTrace.toFixed(5)} -> ${first.finalTrace.toFixed(5)}`);
  ok(first.valid, "P must stay symmetric and positive semi-definite through every predict/update");
  ok(first.poseError < 0.3 && first.headingError < 0.06, "the robot pose must track the truth");
  ok(first.landmarkError < 0.4, `the landmarks must converge to the truth (worst ${first.landmarkError})`);
  ok(first.finalTrace < first.earlyTrace, "re-observing landmarks must shrink the map covariance");
  ok(first.errorHistory.at(-1) < first.errorHistory[0], "the landmark error must improve over the run");
  // 4. Consistency: the mean NEES over 20 deterministic seeds must be near the 9 dimensions.
  const runs = Array.from({ length: 20 }, (_, k) => slamRun(1000 + 37 * k));
  const meanNees = runs.reduce((acc, r) => acc + r.nees, 0) / runs.length;
  notes.push(`consistency over ${runs.length} seeds: mean NEES ${meanNees.toFixed(2)} of 9 dimensions, worst pose error ${Math.max(...runs.map((r) => r.poseError)).toFixed(4)} m`);
  ok(meanNees > 4 && meanNees < 16, `estimation error must be consistent with P (mean NEES ${meanNees})`);
  ok(runs.every((r) => r.valid), "every run must keep P positive semi-definite");
  ok(Math.max(...runs.map((r) => r.poseError)) < 0.8, "no seed may diverge");
  // 5. Bearing innovations must wrap at +-pi instead of reporting a huge difference.
  const wrapFilter = new EkfSlam(3);
  wrapFilter.mu.splice(0, wrapFilter.size, 0, 0, 3.1, 3, 1, 0, 0, 0, 0);
  const wrapObs = observationModel(wrapFilter.mu, wrapFilter.landmarks, 0);
  const raw = Math.atan2(1, 3) - 3.1;
  notes.push(`bearing wrap: measured ${wrapObs.z[1].toFixed(6)}, raw difference ${raw.toFixed(6)}, wrap(pi + 0.001) = ${wrap(Math.PI + 0.001).toFixed(6)}`);
  ok(Math.abs(wrapObs.z[1] - wrap(raw)) < 1e-12 && Math.abs(wrapObs.z[1]) <= Math.PI, "bearings must be wrapped into (-pi, pi]");
  ok(Math.abs(wrap(Math.PI + 0.001) + Math.PI - 0.001) < 1e-12, "wrap must fold angles across +-pi");
  // 6. The chi-square gate must reject an injected outlier and accept a consistent observation.
  const gated = new EkfSlam(3);
  for (let step = 0; step < 6; step++) gated.predict(controls(step), DT);
  gated.initialiseLandmark(0, observationModel([0, 0, 0, ...TRUTH_LANDMARKS.flat()], truthIndices, 0).z);
  const good = observationModel(gated.mu, gated.landmarks, 0);
  const outlier = [good.z[0] + 12, wrap(good.z[1] + 1.5)];
  const distanceOutlier = gated.mahalanobis(0, outlier);
  const rejected = gated.update(0, outlier, { gate: 9.21 });
  const accepted = gated.update(0, good.z, { gate: 9.21 });
  notes.push(`gate: outlier distance^2 ${distanceOutlier.toFixed(2)} rejected = ${!rejected.accepted}; genuine observation distance^2 ${accepted.distance.toExponential(2)} accepted = ${accepted.accepted}`);
  ok(!rejected.accepted && distanceOutlier > 9.21, "the gate must reject the injected outlier");
  ok(accepted.accepted, "a consistent observation must pass the gate");
  // 7. Degenerate input: a landmark at the robot position has no defined bearing.
  throws(() => observationModel([0, 0, 0, 0, 0], [[3, 4]], 0), Error, "a landmark at the robot position must throw");
  throws(() => inverse2([[1, 2], [2, 4]]), Error, "a singular innovation covariance must throw");
  // 8. Determinism.
  const repeat = slamRun(20251016);
  ok(repeat.poseError === first.poseError && repeat.landmarkError === first.landmarkError, "the same seed must reproduce the run exactly");
  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}
module.exports = { makeRng, gaussian, wrap, cholesky, motionModel, observationModel, inverse2, EkfSlam, slamRun };

if (require.main === module) main();
