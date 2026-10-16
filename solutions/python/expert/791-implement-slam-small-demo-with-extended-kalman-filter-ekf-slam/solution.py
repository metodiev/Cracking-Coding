"""Task 791 - Implement SLAM small demo with extended Kalman filter (EKF-SLAM).  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)

Full SLAM keeps the robot pose *and* every landmark in one state vector
mu = (x, y, theta, l_1x, l_1y, ..., l_Mx, l_My) with covariance P.  The demo uses the
standard 2D velocity motion model and range-bearing observations:

    x' = x + v dt cos(theta)          r   = |l - p|
    y' = y + v dt sin(theta)          phi = atan2(l_y - y, l_x - x) - theta
    theta' = theta + w dt

Prediction propagates mu with the nonlinear model and P with the Jacobians
F = d f / d mu and G = d f / d (v, w), so P <- F P F^T + G diag(sv^2, sw^2) G^T.  An
observation of landmark j uses H = d h / d mu (the 2x3 robot block and the 2x2 block of that
landmark), S = H P H^T + R, K = P H^T S^{-1}, mu <- mu + K nu and P <- (I - K H) P.  A
landmark is initialised the first time it is seen by inverting the measurement
(l = p + r (cos(phi + theta), sin(phi + theta))) and injecting J_pose P_rr J_pose^T +
J_meas R J_meas^T into its diagonal block plus J_pose P_rr into its cross-covariance; the
correlation with landmarks already in the map is filled in the same way (J_pose P_rm), which
is exactly what links the new landmark to the current pose uncertainty.  The update keeps the
Joseph form P <- (I - K H) P (I - K H)^T + K R K^T for numerical symmetry/definiteness.

What is verified: P stays symmetric and positive definite at every step, landmark variances
shrink once a landmark is re-observed, the *gauge-free* geometry is accurate (pairwise
landmark distances and robot-to-landmark ranges match the truth within a few centimetres,
while the absolute drift that the rigid-transform gauge allows is reported separately), the
fraction of measurement innovations inside the 3-sigma gate stays high, the cross-covariance
between the robot and the landmarks stays non-trivial (the information link that distinguishes
SLAM from plain localisation), the pose-known version of the filter reproduces an independent
Gauss-Newton batch fit of the same range-bearing data, and NEES (normalised estimation error
squared) averaged over a small Monte-Carlo sweep stays inside a defensible consistency band.

Simplified versus production EKF-SLAM: 2D, no loop closure / no data association problem
(landmark identities are given), measurements are simply skipped when the range is
degenerate, the first pose is anchored at the truth (the standard gauge choice - without it
the map and trajectory are only determined up to a rigid transform, which is why the accuracy
checks are gauge free), and the Monte-Carlo consistency sweep is small (12 runs) so it is
reported with a wide band rather than a tight chi-square test.
"""
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def gauss(state, sigma):
    """Box-Muller pair driven by the deterministic PRNG; returns (state, sample)."""
    state, u1 = splitmix64(state)
    state, u2 = splitmix64(state)
    r = (u1 >> 11) / 9007199254740992.0
    phi = (u2 >> 11) / 9007199254740992.0
    if r < 1e-12:
        r = 1e-12
    return state, sigma * math.sqrt(-2.0 * math.log(r)) * math.cos(2.0 * math.pi * phi)


def wrap(angle):
    """Wrap to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------- matrix helpers
def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def matmul(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    out = zeros(rows, cols)
    for i in range(rows):
        ai = a[i]
        for k in range(inner):
            if ai[k] == 0.0:
                continue
            factor = ai[k]
            bk = b[k]
            for j in range(cols):
                out[i][j] += factor * bk[j]
    return out


def transposed(a):
    return [list(row) for row in zip(*a)]


def add(a, b):
    return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def sub(a, b):
    return [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def solve2(matrix, rhs):
    """Explicit 2x2 solve; returns the solution as a list of column vectors.

    With rhs of shape 2 x c the result is c columns, i.e. result[col][row]; using rhs = H P
    gives (S^-1 H P) transposed, which is the shape the Kalman gain wants (n x 2).
    """
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if abs(det) < 1e-300:
        return None
    columns = len(rhs[0])
    return [[(matrix[1][1] * rhs[0][j] - matrix[0][1] * rhs[1][j]) / det,
             (-matrix[1][0] * rhs[0][j] + matrix[0][0] * rhs[1][j]) / det]
            for j in range(columns)]


def symmetric(matrix):
    n = len(matrix)
    out = [list(row) for row in matrix]
    for i in range(n):
        for j in range(i + 1, n):
            value = 0.5 * (matrix[i][j] + matrix[j][i])
            out[i][j] = out[j][i] = value
    return out


def is_positive_definite(matrix):
    """Cholesky-free check: all leading principal minors > 0 via Gaussian elimination."""
    n = len(matrix)
    a = [list(row) for row in matrix]
    for col in range(n):
        if a[col][col] <= 0.0:
            return False
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for k in range(col, n):
                a[row][k] -= factor * a[col][k]
    return True


# ------------------------------------------------------------------- EKF-SLAM
class EKFSLAM:
    """Range-bearing EKF-SLAM with landmark injection on first observation."""

    def __init__(self, num_landmarks, dt, noise, initial_pose, initial_cov):
        self.dt = dt
        self.sigma_v, self.sigma_w, self.sigma_r, self.sigma_phi = noise
        self.n = 3 + 2 * num_landmarks
        self.mu = [0.0] * (3 + 2 * num_landmarks)
        self.mu[0], self.mu[1], self.mu[2] = initial_pose
        self.P = zeros(self.n, self.n)
        for i in range(3):
            self.P[i][i] = initial_cov[i]
        self.seen = [False] * num_landmarks
        self.innovation_ratios = []

    # ------------------------------------------------------------------ predict
    def predict(self, velocity, turn_rate):
        dt = self.dt
        c, s = math.cos(self.mu[2]), math.sin(self.mu[2])
        self.mu[0] += velocity * dt * c
        self.mu[1] += velocity * dt * s
        self.mu[2] = wrap(self.mu[2] + turn_rate * dt)
        f = identity(self.n)
        f[0][2] = -velocity * dt * s
        f[1][2] = velocity * dt * c
        # control-noise covariance Q = G diag(sv^2, sw^2) G^T with G = d f / d(v, w)
        q = zeros(3, 3)
        q[0][0] = (dt * c) ** 2 * self.sigma_v ** 2
        q[0][1] = q[1][0] = dt * c * dt * s * self.sigma_v ** 2
        q[1][1] = (dt * s) ** 2 * self.sigma_v ** 2
        q[2][2] = dt ** 2 * self.sigma_w ** 2
        propagated = matmul(matmul(f, self.P), transposed(f))
        for i in range(3):
            for j in range(3):
                propagated[i][j] += q[i][j]
        self.P = symmetric(propagated)

    # ------------------------------------------------------------------- update
    def _initialise_landmark(self, index, rng_measurement):
        r, phi = rng_measurement
        theta = self.mu[2]
        a = wrap(phi + theta)
        lx = self.mu[0] + r * math.cos(a)
        ly = self.mu[1] + r * math.sin(a)
        row = 3 + 2 * index
        self.mu[row], self.mu[row + 1] = lx, ly
        robot_cov = [[self.P[i][j] for j in range(3)] for i in range(3)]
        j_pose = [[1.0, 0.0, -r * math.sin(a)], [0.0, 1.0, r * math.cos(a)]]
        j_meas = [[math.cos(a), -r * math.sin(a)], [math.sin(a), r * math.cos(a)]]
        r_noise = [[self.sigma_r ** 2, 0.0], [0.0, self.sigma_phi ** 2]]
        block = add(matmul(matmul(j_pose, robot_cov), transposed(j_pose)),
                    matmul(matmul(j_meas, r_noise), transposed(j_meas)))
        cross = matmul(j_pose, robot_cov)                       # 2 x 3
        for i in range(2):
            for j in range(2):
                self.P[row + i][row + j] = block[i][j]
            for j in range(3):
                self.P[row + i][j] = cross[i][j]
                self.P[j][row + i] = cross[i][j]
        # the new landmark inherits its correlation with the landmarks already in the map
        for other in range(len(self.seen)):
            if other == index or not self.seen[other]:
                continue
            other_row = 3 + 2 * other
            other_cross = [[self.P[i][other_row + k] for k in range(2)] for i in range(3)]
            block_lm = matmul(j_pose, other_cross)              # 2 x 2
            for i in range(2):
                for j in range(2):
                    self.P[row + i][other_row + j] = block_lm[i][j]
                    self.P[other_row + j][row + i] = block_lm[i][j]
        self.seen[index] = True

    def update(self, index, measurement):
        row = 3 + 2 * index
        if not self.seen[index]:
            self._initialise_landmark(index, measurement)
            return True
        r, phi = measurement
        dx = self.mu[row] - self.mu[0]
        dy = self.mu[row + 1] - self.mu[1]
        q = dx * dx + dy * dy
        rho = math.sqrt(q)
        if rho < 1e-6:
            return False                                            # degenerate: skip
        predicted = [rho, wrap(math.atan2(dy, dx) - self.mu[2])]
        h = zeros(2, self.n)
        h[0][0], h[0][1] = -dx / rho, -dy / rho
        h[0][row], h[0][row + 1] = dx / rho, dy / rho
        h[1][0], h[1][1], h[1][2] = dy / q, -dx / q, -1.0
        h[1][row], h[1][row + 1] = -dy / q, dx / q
        resid = [wrap(r - predicted[0]), wrap(phi - predicted[1])]
        r_noise = [[self.sigma_r ** 2, 0.0], [0.0, self.sigma_phi ** 2]]
        s = add(matmul(matmul(h, self.P), transposed(h)), r_noise)
        ph_t = matmul(self.P, transposed(h))                        # n x 2 = P H^T
        # solve2 returns one column vector of length 2 per right-hand-side column, i.e. the
        # n x 2 matrix K = (S^-1 (H P))^T = P H^T S^-1 directly
        gain = solve2(s, transposed(ph_t))
        if gain is None:
            return False
        for i in range(self.n):
            for j in range(2):
                self.mu[i] += gain[i][j] * resid[j]
        self.mu[2] = wrap(self.mu[2])
        # Joseph form: preserves symmetry and positive definiteness numerically
        correction = sub(identity(self.n), matmul(gain, h))
        first = matmul(matmul(correction, self.P), transposed(correction))
        second = matmul(matmul(gain, r_noise), transposed(gain))
        self.P = symmetric(add(first, second))
        self.innovation_ratios.append(max(abs(resid[0]) / math.sqrt(s[0][0]),
                                          abs(resid[1]) / math.sqrt(s[1][1])))
        return True

    def nees(self, truth_state):
        """Normalised estimation error squared against a full true state."""
        diff = [self.mu[i] - truth_state[i] for i in range(self.n)]
        diff[2] = wrap(diff[2])
        # solve P x = diff by Gaussian elimination, then NEES = diff . x
        n = self.n
        a = [list(self.P[i]) + [diff[i]] for i in range(n)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
            if abs(a[pivot][col]) < 1e-300:
                return float("inf")
            a[col], a[pivot] = a[pivot], a[col]
            inv = 1.0 / a[col][col]
            for row in range(col + 1, n):
                factor = a[row][col] * inv
                for k in range(col, n + 1):
                    a[row][k] -= factor * a[col][k]
        x = [0.0] * n
        for row in range(n - 1, -1, -1):
            x[row] = (a[row][n] - sum(a[row][k] * x[k] for k in range(row + 1, n))) / a[row][row]
        return math.fsum(d * v for d, v in zip(diff, x))


# --------------------------------------------------------------- batch oracle
def batch_landmark_fit(poses, measurements, landmark, initial, iterations=40,
                       sigma_r=0.05, sigma_phi=0.02):
    """Gauss-Newton fit of ONE landmark from range-bearing data with the poses known."""
    point = list(initial)
    weights = (1.0 / sigma_r ** 2, 1.0 / sigma_phi ** 2)
    for _ in range(iterations):
        normal = zeros(2, 2)
        gradient = [0.0, 0.0]
        for (x, y, theta), (r, phi) in zip(poses, measurements):
            dx = point[0] - x
            dy = point[1] - y
            rho2 = dx * dx + dy * dy
            rho = math.sqrt(rho2)
            residual_r = rho - r
            residual_phi = wrap(math.atan2(dy, dx) - theta - phi)
            j = [[dx / rho, dy / rho], [-dy / rho2, dx / rho2]]
            for i in range(2):
                for k in range(2):
                    gradient[i] += weights[k] * j[k][i] * (residual_r if k == 0
                                                           else residual_phi)
                    for m in range(2):
                        normal[i][m] += weights[k] * j[k][i] * j[k][m]
        step = solve2([[normal[0][0], normal[0][1]], [normal[1][0], normal[1][1]]],
                      [[-gradient[0]], [-gradient[1]]])
        if step is None:
            break
        point[0] += step[0][0]                     # one column: [delta_x, delta_y]
        point[1] += step[0][1]
    return point


def simulate(seed, steps=60, dt=0.1, noise=(0.12, 0.09, 0.05, 0.02), landmarks=None):
    """Drive the robot around the landmarks; returns the true states and observations."""
    sigma_v, sigma_w, sigma_r, sigma_phi = noise
    landmarks = landmarks or [(3.0, 0.4), (2.2, 2.6), (-0.6, 2.4), (-2.1, 0.2),
                              (-0.9, -2.3), (2.4, -1.9)]
    state = seed
    pose = [0.5, -0.4, 0.2]
    history = [list(pose)]
    observations = []
    for step in range(steps):
        velocity = 0.9 + 0.15 * math.sin(0.3 * step)
        turn = 0.35 + 0.10 * math.cos(0.21 * step)
        state, noise_v = gauss(state, sigma_v)
        state, noise_w = gauss(state, sigma_w)
        pose[0] += (velocity + noise_v) * dt * math.cos(pose[2])
        pose[1] += (velocity + noise_v) * dt * math.sin(pose[2])
        pose[2] = wrap(pose[2] + (turn + noise_w) * dt)
        history.append(list(pose))
        readings = {}
        for index, (lx, ly) in enumerate(landmarks):
            dx = lx - pose[0]
            dy = ly - pose[1]
            if math.hypot(dx, dy) > 3.6:
                continue
            state, nr = gauss(state, sigma_r)
            state, np = gauss(state, sigma_phi)
            readings[index] = (math.hypot(dx, dy) + nr,
                               wrap(math.atan2(dy, dx) - pose[2] + np))
        observations.append((list(pose), (velocity, turn), readings))
    return landmarks, history, observations, state


def run_filter(seed, steps=60, dt=0.1, noise=(0.12, 0.09, 0.05, 0.02), check_invariants=False):
    """Run EKF-SLAM over a simulated drive, with the pose prior anchored at the true start.

    Anchoring the first pose is the standard gauge choice: without it the whole map and
    trajectory are only determined up to a rigid transform (and to the scale-free prior), so
    absolute drift is expected and the meaningful accuracy measures are the gauge-free ones.
    """
    sigma_v, sigma_w, sigma_r, sigma_phi = noise
    del sigma_v, sigma_w, sigma_r, sigma_phi
    landmarks, history, observations, _state = simulate(seed, steps=steps, dt=dt, noise=noise)
    slam = EKFSLAM(len(landmarks), dt, noise, history[0], [0.01, 0.01, 0.005])
    trace = []
    init_cov = [None] * len(landmarks)
    for pose, control, readings in observations:
        slam.predict(control[0], control[1])
        for index, measurement in sorted(readings.items()):
            slam.update(index, measurement)
            if init_cov[index] is None and slam.seen[index]:
                init_cov[index] = (slam.P[3 + 2 * index][3 + 2 * index],
                                   slam.P[4 + 2 * index][4 + 2 * index])
        if check_invariants:
            n = slam.n
            symmetry = max(abs(slam.P[i][j] - slam.P[j][i]) for i in range(n) for j in range(n))
            check(symmetry < 1e-12, "P must stay symmetric, worst asymmetry %.3e" % symmetry)
            check(is_positive_definite(slam.P),
                  "P must stay positive definite after every predict/update step")
        pose_error = math.hypot(slam.mu[0] - pose[0], slam.mu[1] - pose[1])
        trace.append((pose_error, list(slam.mu), [list(row) for row in slam.P]))
    return slam, landmarks, history, observations, trace, init_cov


def pairwise_distance_error(slam, landmarks):
    """Largest disagreement between estimated and true landmark distances (gauge free)."""
    worst = 0.0
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            estimated = math.hypot(slam.mu[3 + 2 * i] - slam.mu[3 + 2 * j],
                                   slam.mu[4 + 2 * i] - slam.mu[4 + 2 * j])
            true = math.hypot(landmarks[i][0] - landmarks[j][0],
                              landmarks[i][1] - landmarks[j][1])
            worst = max(worst, abs(estimated - true))
    return worst


def relative_range_error(slam, landmarks, pose):
    """Largest robot-to-landmark range error (gauge free)."""
    worst = 0.0
    for i in range(len(landmarks)):
        estimated = math.hypot(slam.mu[3 + 2 * i] - slam.mu[0], slam.mu[4 + 2 * i] - slam.mu[1])
        true = math.hypot(landmarks[i][0] - pose[0], landmarks[i][1] - pose[1])
        worst = max(worst, abs(estimated - true))
    return worst


def main():
    NOISE = (0.12, 0.09, 0.05, 0.02)
    slam, landmarks, history, observations, trace, init_cov = run_filter(791001,
                                                                        check_invariants=True)
    n = slam.n

    # --- 1. state layout, invariants and the SLAM information link ------------
    check(slam.n == 3 + 2 * len(landmarks), "the state must hold the pose plus 2D per landmark")
    check(all(slam.seen[index] for index in range(len(landmarks))),
          "every landmark must have been observed at least once")
    check(abs(slam.P[0][3]) > 1e-9 and abs(slam.P[0][9]) > 1e-9,
          "robot/landmark cross-covariance must be non-trivial (that is the SLAM link)")
    check(all(cov is not None for cov in init_cov), "every landmark must record its birth variance")

    # --- 2. gauge-free map accuracy and covariance shrinkage -----------------
    distance_error = pairwise_distance_error(slam, landmarks)
    check(distance_error < 0.1,
          "estimated landmark distances must match the truth within 0.1 m, worst %.4f m"
          % distance_error)
    range_error = relative_range_error(slam, landmarks, history[-1])
    check(range_error < 0.1,
          "robot-to-landmark ranges must match the truth within 0.1 m, worst %.4f m" % range_error)
    final_cov = [(slam.P[3 + 2 * index][3 + 2 * index], slam.P[4 + 2 * index][4 + 2 * index])
                 for index in range(len(landmarks))]
    check(all(birth[0] > final[0] and birth[1] > final[1]
              for birth, final in zip(init_cov, final_cov)),
          "re-observing a landmark must shrink its variance below the birth value")
    check(all(value[0] + value[1] < 0.1 for value in final_cov),
          "final landmark covariances must be small: %r"
          % [[round(a, 5), round(b, 5)] for a, b in final_cov])

    # --- 3. innovation gate and the degenerate measurement -------------------
    ratios = slam.innovation_ratios
    inside = sum(1 for value in ratios if value <= 3.0) / len(ratios)
    check(inside > 0.95,
          "at least 95%% of innovations must fall inside 3 sigma, got %.3f" % inside)
    fresh = EKFSLAM(1, 0.1, NOISE, [0.0, 0.0, 0.0], [0.1, 0.1, 0.05])
    fresh._initialise_landmark(0, (1.0, 0.0))
    fresh.mu[3], fresh.mu[4] = 0.0, 0.0                   # landmark on top of the robot
    check(fresh.update(0, (0.0, 0.0)) is False,
          "a landmark at the robot's own position must be skipped, not divided by zero")

    # --- 4. independent oracle: batch Gauss-Newton on the same data ----------
    poses = [pose for pose, _control, _readings in observations]
    known_pose = EKFSLAM(len(landmarks), 0.1, NOISE, [poses[0][0], poses[0][1], poses[0][2]],
                         [1e-12, 1e-12, 1e-12])
    for pose, _control, readings in observations:
        known_pose.mu[0], known_pose.mu[1], known_pose.mu[2] = pose
        for index, measurement in sorted(readings.items()):
            known_pose.update(index, measurement)
    batch_errors = []
    for index in range(len(landmarks)):
        series = [(pose, readings[index]) for pose, _c, readings in observations
                  if index in readings]
        if len(series) < 2:
            continue
        fit = batch_landmark_fit([pose for pose, _m in series], [m for _p, m in series],
                                 index, [known_pose.mu[3 + 2 * index],
                                         known_pose.mu[4 + 2 * index]],
                                 sigma_r=NOISE[2], sigma_phi=NOISE[3])
        batch_errors.append(math.hypot(fit[0] - known_pose.mu[3 + 2 * index],
                                       fit[1] - known_pose.mu[4 + 2 * index]))
    check(len(batch_errors) >= 4, "at least four landmarks must have a batch comparison")
    check(max(batch_errors) < 1e-3,
          "pose-known EKF must reproduce the batch Gauss-Newton optimum, worst gap %.3e"
          % max(batch_errors))

    # --- 5. Monte-Carlo consistency (NEES) ----------------------------------
    nees_samples = []
    for run in range(12):
        other, other_landmarks, other_history, _obs, _trace, _cov = run_filter(791100 + run)
        true_state = list(other_history[-1]) + [value for pair in other_landmarks
                                                for value in pair]
        nees_samples.append(other.nees(true_state) / other.n)
    average_nees = math.fsum(nees_samples) / len(nees_samples)
    check(0.3 < average_nees < 3.0,
          "average NEES per dimension must stay consistent, got %.4f" % average_nees)
    check(all(0.05 < value < 6.0 for value in nees_samples),
          "no single run may be wildly inconsistent: %r" % [round(v, 3) for v in nees_samples])
    check(len(nees_samples) == 12, "the Monte-Carlo sweep must cover all runs")

    # --- 6. determinism -----------------------------------------------------
    repeat, _l, _h, _o, repeat_trace, _cov = run_filter(791001)
    check(repeat.mu == slam.mu, "an identical seed must reproduce the estimate exactly")
    check(repeat_trace[0][0] == trace[0][0], "identical seeds must give identical pose errors")
    check(repeat.innovation_ratios == ratios, "identical seeds must give identical innovations")

    print("gauge-free accuracy: landmark distances %.4f m, robot ranges %.4f m"
          % (distance_error, range_error))
    print("absolute drift (gauge): worst pose error %.4f m over the run" % max(e[0] for e in trace))
    print("innovations inside 3 sigma: %.3f of %d" % (inside, len(ratios)))
    print("pose-known EKF vs batch Gauss-Newton: worst landmark gap %.3e m" % max(batch_errors))
    print("average NEES per dimension over 12 runs: %.4f" % average_nees)
    print("PASS")



if __name__ == "__main__":
    main()
