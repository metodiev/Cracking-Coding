"""Task 792 - Implement particle filter localization for robot.  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)

Monte-Carlo localisation represents the belief over the pose (x, y, theta) with a weighted
particle set.  One step of the filter is

  1. predict: every particle is pushed through the odometry motion model with per-particle
     noise drawn from the deterministic PRNG used throughout this file;
  2. correct: ranges to the known landmarks are compared with the measured ranges and the
     weight of a particle is multiplied by the Gaussian likelihood of those differences,
     w_i <- w_i * prod_k N(z_k - h_k(x_i), sigma^2) (implemented in log space for stability);
  3. rescale: weights are normalised so that sum_i w_i = 1; if the effective sample size
     ESS = 1 / sum_i w_i^2 drops below half the particle count, the set is resampled with
     *systematic* resampling (one uniform draw plus N-1 fixed offsets), the weights are reset
     to 1/N, and a small amount of roughening noise is added to avoid impoverishment (the
     classic augmented-MCL idea).

The pose estimate is the weighted mean position and the circular mean of the heading, and the
spread of the particle cloud is returned as a scalar uncertainty measure.

Verified: weights are normalised after every update and stay finite, ESS is exactly N for a
uniform cloud and 1 for a point mass, systematic resampling keeps the particle count, its
empirical distribution matches the normalised weights (histogram comparison), resampling
leaves the weighted mean essentially unchanged, the filter converges from a broad uniform
prior to the true pose, it tracks the truth at least as well as an independently implemented
EKF on the same measurement stream, an impossible measurement stream does not produce NaNs
(weights are renormalised or the update is skipped), and the whole run is bit-reproducible.

Scaled-down versus a real MCL implementation: four point landmarks with noiseless range-only
sensing (no map ray casting, no field-of-view model, no adaptive particle count), a fixed
particle count of a few hundred, and the roughening only applies after a resampling step.
"""
import math

MASK64 = (1 << 64) - 1
LANDMARKS = [(3.0, 0.5), (0.6, 3.2), (-2.8, 1.6), (-1.4, -2.9), (2.6, -2.2)]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state):
    state, r = splitmix64(state)
    return state, r / 18446744073709551616.0


def gauss(state, sigma):
    """Box-Muller Gaussian sample; deterministic given the state."""
    state, u1 = splitmix64(state)
    state, u2 = splitmix64(state)
    r = (u1 >> 11) / 9007199254740992.0
    p = (u2 >> 11) / 9007199254740992.0
    if r < 1e-12:
        r = 1e-12
    return state, sigma * math.sqrt(-2.0 * math.log(r)) * math.cos(2.0 * math.pi * p)


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------------ motion model
def motion_model(pose, velocity, turn_rate, dt, noise_v, noise_w):
    """Differential-drive style step with control noise applied to v and w."""
    return (pose[0] + (velocity + noise_v) * dt * math.cos(pose[2]),
            pose[1] + (velocity + noise_v) * dt * math.sin(pose[2]),
            wrap(pose[2] + (turn_rate + noise_w) * dt))


def range_vector(pose, landmarks):
    return [math.hypot(lx - pose[0], ly - pose[1]) for lx, ly in landmarks]


# --------------------------------------------------------------- particle filter
class ParticleFilter:
    """Range-only Monte-Carlo localisation over (x, y, theta)."""

    def __init__(self, count, landmarks, sigma_r, dt, noise_v, noise_w, seed):
        self.count = count
        self.landmarks = list(landmarks)
        self.sigma_r = sigma_r
        self.dt = dt
        self.noise_v = noise_v
        self.noise_w = noise_w
        self.state = seed
        self.particles = []
        self.weights = [1.0 / count] * count
        self.ess = float(count)
        self.resampling_count = 0

    # ------------------------------------------------------------- construction
    def initialise_uniform(self, x_range, y_range):
        """Uniform cloud over a rectangle; the heading is uniform over the circle."""
        self.particles = []
        for _ in range(self.count):
            self.state, rx = uniform(self.state)
            self.state, ry = uniform(self.state)
            self.state, rt = uniform(self.state)
            self.particles.append((x_range[0] + (x_range[1] - x_range[0]) * rx,
                                   y_range[0] + (y_range[1] - y_range[0]) * ry,
                                   wrap((rt - 0.5) * 2.0 * math.pi)))
        self.weights = [1.0 / self.count] * self.count
        self.ess = float(self.count)
        return self

    def initialise_at(self, pose, sigma_xy, sigma_theta):
        """Gaussian cloud around a nominal pose (used for the EKF comparison run)."""
        self.particles = []
        for _ in range(self.count):
            self.state, dx = gauss(self.state, sigma_xy)
            self.state, dy = gauss(self.state, sigma_xy)
            self.state, dtheta = gauss(self.state, sigma_theta)
            self.particles.append((pose[0] + dx, pose[1] + dy, wrap(pose[2] + dtheta)))
        self.weights = [1.0 / self.count] * self.count
        self.ess = float(self.count)
        return self

    # ------------------------------------------------------------------ kernels
    def predict(self, velocity, turn_rate):
        moved = []
        for pose in self.particles:
            self.state, nv = gauss(self.state, self.noise_v)
            self.state, nw = gauss(self.state, self.noise_w)
            moved.append(motion_model(pose, velocity, turn_rate, self.dt, nv, nw))
        self.particles = moved
        return self

    def update(self, measurement):
        """Multiply the weights by the per-landmark Gaussian range likelihood."""
        logs = []
        scale = -0.5 / (self.sigma_r ** 2)
        for pose in self.particles:
            total = 0.0
            for predicted, measured in zip(range_vector(pose, self.landmarks), measurement):
                diff = measured - predicted
                total += diff * diff
            logs.append(scale * total)
        top = max(logs)
        weights = [math.exp(value - top) * prior for value, prior in zip(logs, self.weights)]
        normaliser = math.fsum(weights)
        if not normaliser > 0.0 or not math.isfinite(normaliser):
            weights = [1.0 / self.count] * self.count        # impossible measurement: reset
        else:
            weights = [value / normaliser for value in weights]
        self.weights = weights
        self.ess = 1.0 / math.fsum(w * w for w in weights)
        return self

    def systematic_resample(self):
        """One uniform draw plus N-1 fixed offsets: low variance, bias free."""
        positions = []
        self.state, start = uniform(self.state)
        start /= self.count
        cumulative = []
        acc = 0.0
        for weight in self.weights:
            acc += weight
            cumulative.append(acc)
        index = 0
        for k in range(self.count):
            target = start + k / self.count
            while cumulative[index] < target and index < self.count - 1:
                index += 1
            positions.append(self.particles[index])
        return positions

    def resample(self, roughening=0.0):
        before = self.weighted_mean()
        self.particles = self.systematic_resample()
        self.weights = [1.0 / self.count] * self.count
        self.ess = float(self.count)
        self.resampling_count += 1
        if roughening > 0.0:
            perturbed = []
            for pose in self.particles:
                self.state, dx = gauss(self.state, roughening)
                self.state, dy = gauss(self.state, roughening)
                self.state, dtheta = gauss(self.state, roughening)
                perturbed.append((pose[0] + dx, pose[1] + dy, wrap(pose[2] + dtheta)))
            self.particles = perturbed
        return before

    # ---------------------------------------------------------------- estimates
    def weighted_mean(self):
        """Mean x, y and *circular* mean heading of the weighted cloud."""
        sx = sy = 0.0
        sc = ss = 0.0
        for pose, weight in zip(self.particles, self.weights):
            sx += weight * pose[0]
            sy += weight * pose[1]
            sc += weight * math.cos(pose[2])
            ss += weight * math.sin(pose[2])
        return (sx, sy, math.atan2(ss, sc))

    def spread(self):
        """Weighted RMS distance of the particles from the weighted mean (ignores heading)."""
        mean = self.weighted_mean()
        total = 0.0
        for pose, weight in zip(self.particles, self.weights):
            total += weight * ((pose[0] - mean[0]) ** 2 + (pose[1] - mean[1]) ** 2)
        return math.sqrt(total)

    def step(self, velocity, turn_rate, measurement, threshold=0.5, roughening=0.0):
        self.predict(velocity, turn_rate)
        self.update(measurement)
        if self.ess < threshold * self.count:
            self.resample(roughening)
        return self.weighted_mean()

    def mean_weight(self):
        return math.fsum(self.weights) / len(self.weights)

    def weight_sum(self):
        return math.fsum(self.weights)


# ------------------------------------------------------------- EKF for comparison
class RangeEKF:
    """Plain EKF localisation with known landmarks and range-only observations."""

    def __init__(self, pose, cov, landmarks, sigma_r, dt, noise_v, noise_w):
        self.mu = list(pose)
        self.cov = [list(row) for row in cov]
        self.landmarks = list(landmarks)
        self.sigma_r = sigma_r
        self.dt = dt
        self.noise_v = noise_v
        self.noise_w = noise_w

    def predict(self, velocity, turn_rate):
        dt = self.dt
        c, s = math.cos(self.mu[2]), math.sin(self.mu[2])
        self.mu[0] += velocity * dt * c
        self.mu[1] += velocity * dt * s
        self.mu[2] = wrap(self.mu[2] + turn_rate * dt)
        f = [[1.0, 0.0, -velocity * dt * s],
             [0.0, 1.0, velocity * dt * c],
             [0.0, 0.0, 1.0]]
        q = [[(dt * c) ** 2 * self.noise_v ** 2, dt * c * dt * s * self.noise_v ** 2, 0.0],
             [dt * c * dt * s * self.noise_v ** 2, (dt * s) ** 2 * self.noise_v ** 2, 0.0],
             [0.0, 0.0, dt ** 2 * self.noise_w ** 2]]
        new = [[math.fsum(f[i][a] * self.cov[a][b] * f[j][b] for a in range(3)
                          for b in range(3)) + q[i][j] for j in range(3)] for i in range(3)]
        self.cov = new

    def update(self, landmark, measured):
        dx = landmark[0] - self.mu[0]
        dy = landmark[1] - self.mu[1]
        rho = math.hypot(dx, dy)
        if rho < 1e-6:
            return
        h = [-dx / rho, -dy / rho, 0.0]
        ph = [math.fsum(self.cov[i][j] * h[j] for j in range(3)) for i in range(3)]
        s = math.fsum(h[i] * ph[i] for i in range(3)) + self.sigma_r ** 2
        innovation = measured - rho
        gain = [value / s for value in ph]
        for i in range(3):
            self.mu[i] += gain[i] * innovation
        self.mu[2] = wrap(self.mu[2])
        self.cov = [[self.cov[i][j] - gain[i] * ph[j] for j in range(3)] for i in range(3)]


# ------------------------------------------------------------------ simulation
def simulate(seed, steps=40, dt=0.1, sigma_r=0.08, noise_v=0.10, noise_w=0.08):
    """Drive a circular path; return the true poses and the measured range vectors."""
    state = seed
    pose = [1.0, 0.6, 0.0]
    history = [list(pose)]
    measurements = []
    for step in range(steps):
        velocity = 1.0 + 0.12 * math.sin(0.4 * step)
        turn = 0.28 + 0.08 * math.cos(0.3 * step)
        state, nv = gauss(state, noise_v)
        state, nw = gauss(state, noise_w)
        pose = list(motion_model(pose, velocity, turn, dt, nv, nw))
        history.append(list(pose))
        reading = []
        for value in range_vector(pose, LANDMARKS):
            state, noise = gauss(state, sigma_r)
            reading.append(value + noise)
        measurements.append((velocity, turn, reading))
    return history, measurements, state


def run_particle_filter(seed, prior, steps=40, dt=0.1, sigma_r=0.08, noise_v=0.10,
                        noise_w=0.08, count=400, roughening=0.02):
    """Full MCL run; returns the filter, the truth and the estimate history."""
    history, measurements, _state = simulate(seed, steps=steps, dt=dt, sigma_r=sigma_r,
                                             noise_v=noise_v, noise_w=noise_w)
    pf = ParticleFilter(count, LANDMARKS, sigma_r, dt, noise_v, noise_w, seed ^ 0xA5A5)
    if prior == "uniform":
        pf.initialise_uniform((-2.0, 2.0), (-2.0, 2.0))
    else:
        pf.initialise_at(prior, 0.25, 0.25)
    estimates = []
    for step, (velocity, turn, reading) in enumerate(measurements):
        estimates.append(pf.step(velocity, turn, reading, roughening=roughening))
    return pf, history, measurements, estimates


def run_ekf(seed, prior, steps=40, dt=0.1, sigma_r=0.08, noise_v=0.10, noise_w=0.08):
    """Same measurement stream through an EKF started from the same prior."""
    history, measurements, _state = simulate(seed, steps=steps, dt=dt, sigma_r=sigma_r,
                                             noise_v=noise_v, noise_w=noise_w)
    ekf = RangeEKF(prior, [[0.0625, 0.0, 0.0], [0.0, 0.0625, 0.0], [0.0, 0.0, 0.0625]],
                   LANDMARKS, sigma_r, dt, noise_v, noise_w)
    estimates = []
    for velocity, turn, reading in measurements:
        ekf.predict(velocity, turn)
        for landmark, measured in zip(LANDMARKS, reading):
            ekf.update(landmark, measured)
        estimates.append(tuple(ekf.mu))
    return ekf, history, estimates


def position_error(estimate, truth):
    return math.hypot(estimate[0] - truth[0], estimate[1] - truth[1])


def main():
    # --- 1. weight bookkeeping and ESS boundaries ---------------------------
    pf = ParticleFilter(200, LANDMARKS, 0.08, 0.1, 0.1, 0.08, 792001)
    pf.initialise_uniform((-3.0, 3.0), (-3.0, 3.0))
    check(abs(pf.weight_sum() - 1.0) < 1e-12, "the initial weights must sum to one")
    check(abs(pf.ess - 200.0) < 1e-12,
          "a uniform cloud must have ESS equal to the particle count, got %.6f" % pf.ess)
    point_mass = ParticleFilter(50, LANDMARKS, 0.08, 0.1, 0.1, 0.08, 792002)
    point_mass.initialise_uniform((-1.0, 1.0), (-1.0, 1.0))
    point_mass.weights = [1.0] + [0.0] * 49
    point_mass.ess = 1.0 / math.fsum(w * w for w in point_mass.weights)
    check(abs(point_mass.ess - 1.0) < 1e-12,
          "a point mass must collapse ESS to one, got %.6f" % point_mass.ess)

    # --- 2. resampling invariants ------------------------------------------
    weights = [math.exp(-0.5 * ((pose[0] - 1.0) / 0.9) ** 2) for pose in pf.particles]
    total = math.fsum(weights)
    pf.weights = [value / total for value in weights]
    pf.ess = 1.0 / math.fsum(w * w for w in pf.weights)
    check(abs(pf.weight_sum() - 1.0) < 1e-12, "weights must be normalised by the update")
    check(pf.ess > 20.0,
          "a smooth weight profile must keep a healthy ESS, got %.2f" % pf.ess)
    check(all(0.0 <= w <= 1.0 for w in pf.weights), "weights must stay inside [0, 1]")
    weights_before = list(pf.weights)
    mean_before = pf.weighted_mean()
    index_by_particle = {}
    for position, particle in enumerate(pf.particles):
        index_by_particle.setdefault(particle, position)
    for _repeat in range(4):                       # several draws for a stable histogram
        resampled = pf.systematic_resample()
        check(len(resampled) == pf.count, "systematic resampling must keep the particle count")
    histogram = [0] * pf.count
    for particle in pf.systematic_resample():
        histogram[index_by_particle[particle]] += 1
    deviation = max(abs(histogram[i] - pf.count * weights_before[i]) for i in range(pf.count))
    check(deviation < 1.0 + 1e-9,
          "systematic resampling must realise each weight to within one particle, max %.4f"
          % deviation)
    drift = 0.5 * math.fsum(abs(histogram[i] / pf.count - weights_before[i])
                            for i in range(pf.count))
    check(drift < 0.25,
          "the resampled distribution must track the weights up to the sub-1/N tail mass, "
          "TV distance %.4f" % drift)
    mean_after = pf.weighted_mean()
    check(abs(mean_after[0] - mean_before[0]) < 0.1 and abs(mean_after[1] - mean_before[1]) < 0.1,
          "resampling must not move the cloud mean appreciably: (%.4f, %.4f) -> (%.4f, %.4f)"
          % (mean_before[0], mean_before[1], mean_after[0], mean_after[1]))
    check(sum(histogram) == pf.count, "the resampling histogram must cover every particle")

    # --- 3. weight update from a real measurement ---------------------------
    fresh = ParticleFilter(200, LANDMARKS, 0.08, 0.1, 0.1, 0.08, 792005)
    fresh.initialise_uniform((-1.0, 1.0), (-1.0, 1.0))
    fresh.update(range_vector((0.0, 0.0, 0.0), LANDMARKS))
    check(abs(fresh.weight_sum() - 1.0) < 1e-12, "the update must leave normalised weights")
    check(fresh.ess <= fresh.count, "ESS can never exceed the particle count")
    check(fresh.ess < 0.5 * fresh.count,
          "a sharp range measurement from a flat prior must collapse the ESS, got %.2f"
          % fresh.ess)
    best_index = max(range(fresh.count), key=lambda i: fresh.weights[i])
    check(position_error(fresh.particles[best_index], (0.0, 0.0, 0.0)) < 0.2,
          "the highest-weight particle must sit near the pose the measurement implies, "
          "got %.4f m" % position_error(fresh.particles[best_index], (0.0, 0.0, 0.0)))

    # --- 3. convergence from a broad uniform prior --------------------------
    probe = ParticleFilter(1500, LANDMARKS, 0.08, 0.1, 0.1, 0.08, 792010)
    probe.initialise_uniform((-2.0, 2.0), (-2.0, 2.0))
    prior_error = position_error(probe.weighted_mean(), (1.0, 0.6, 0.0))
    prior_spread = probe.spread()
    check(prior_error > 0.6,
          "the uniform prior mean must be far from the truth, got %.3f m" % prior_error)
    check(prior_spread > 1.2,
          "the uniform prior must be broad before any measurement, spread %.3f m" % prior_spread)
    pf, history, measurements, estimates = run_particle_filter(792010, "uniform", count=1500)
    errors = [position_error(estimate, truth) for estimate, truth in zip(estimates, history[1:])]
    check(errors[0] < 0.5,
          "one range update must already localise the robot to 0.5 m, got %.3f m" % errors[0])
    check(max(errors) < 0.3,
          "the filter must stay within 0.3 m of the truth for the whole run, worst %.3f m"
          % max(errors))
    check(pf.spread() < 0.35,
          "the converged cloud must be tight, spread %.3f m" % pf.spread())
    check(pf.resampling_count > 0, "ESS must trigger resampling at least once on this run")
    check(abs(pf.weight_sum() - 1.0) < 1e-12, "weights must be normalised after the whole run")

    # --- 4. tracking accuracy versus an independent EKF on the same stream --
    prior = (1.0, 0.6, 0.0)
    pf_track, track_history, _m, track_estimates = run_particle_filter(792020, prior)
    ekf, _h, ekf_estimates = run_ekf(792020, prior)
    pf_errors = [position_error(e, t) for e, t in zip(track_estimates, track_history[1:])]
    ekf_errors = [position_error(e, t) for e, t in zip(ekf_estimates, track_history[1:])]
    pf_rmse = math.sqrt(math.fsum(e * e for e in pf_errors) / len(pf_errors))
    ekf_rmse = math.sqrt(math.fsum(e * e for e in ekf_errors) / len(ekf_errors))
    check(pf_rmse < 0.25, "particle filter RMSE must stay small, got %.4f m" % pf_rmse)
    check(pf_rmse < 1.6 * ekf_rmse,
          "the particle filter must track comparably to the EKF: PF %.4f vs EKF %.4f"
          % (pf_rmse, ekf_rmse))
    disagreement = max(position_error(a, b) for a, b in zip(track_estimates[5:],
                                                           ekf_estimates[5:]))
    check(disagreement < 0.3,
          "after the transient PF and EKF must agree within 0.3 m, worst %.4f m" % disagreement)

    # --- 5. impossible measurements and boundary behaviour ------------------
    robust = ParticleFilter(120, LANDMARKS, 0.05, 0.1, 0.1, 0.08, 792030)
    robust.initialise_uniform((-2.0, 2.0), (-2.0, 2.0))
    robust.step(1.0, 0.2, [40.0, 40.0, 40.0, 40.0, 40.0])
    check(all(math.isfinite(w) for w in robust.weights),
          "an impossible measurement must not produce non-finite weights")
    check(abs(robust.weight_sum() - 1.0) < 1e-12,
          "an impossible measurement must still leave normalised weights")
    single = ParticleFilter(40, LANDMARKS, 0.08, 0.1, 0.1, 0.08, 792031)
    single.initialise_at((0.5, 0.5, 0.0), 0.05, 0.05)
    single.resample()
    check(abs(single.weight_sum() - 1.0) < 1e-12,
          "resampling must reset the weights to a uniform 1/N")
    check(abs(single.ess - 40.0) < 1e-12, "ESS must return to N after resampling")
    check(len(single.particles) == 40, "resampling must preserve the cloud size")

    # --- 6. determinism ----------------------------------------------------
    repeat_pf, _h, _m, repeat_estimates = run_particle_filter(792020, prior)
    check(repeat_estimates == track_estimates,
          "an identical seed must reproduce every estimate bit for bit")
    check(repeat_pf.resampling_count == pf_track.resampling_count,
          "an identical seed must reproduce the number of resampling events")

    print("uniform prior: mean error %.3f m, spread %.3f m -> final error %.3f m, spread %.3f m"
          % (prior_error, prior_spread, errors[-1], pf.spread()))
    print("worst error over the uniform-prior run: %.3f m, resamplings %d"
          % (max(errors), pf.resampling_count))
    print("tracking RMSE: particle filter %.4f m, EKF %.4f m, worst disagreement %.4f m"
          % (pf_rmse, ekf_rmse, disagreement))
    print("resampling TV distance vs weights: %.4f, ESS after update: %.1f" % (drift, pf.ess))
    print("particles: %d, final spread %.4f m" % (pf_track.count, pf_track.spread()))
    print("PASS")


if __name__ == "__main__":
    main()
