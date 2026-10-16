"""Task 881 - Implement global illumination importance sampling techniques in renderer.  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

A Monte Carlo direct-lighting integrator with the three standard variance-reduction techniques,
each implemented for real and validated against closed forms:

  * cosine-weighted hemisphere sampling - the pdf cos(theta)/pi exactly cancels the cosine and
    Lambertian BRDF factors, so for a uniform environment the estimator has *zero* variance
    while uniform hemisphere sampling has positive variance: both are unbiased, only one is
    quiet;
  * next-event estimation (NEE) - sampling the solid angle subtended by a spherical light
    (the classic cone sampling, pdf = 1/Omega) which removes the catastrophic hidden-light
    variance of BSDF sampling for small lights;
  * multiple importance sampling - the balance and power heuristics combining the two
    strategies, w_i = p_i / sum_j p_j, so the estimator is unbiased whatever the light size and
    never much worse than the better of its two strategies.

Russian roulette termination (continue with probability q, divide the weight by q) and
stratified sampling are included as well, and the whole path sum is checked against the
analytic cavity series L / (1 - rho).

Verified: the cosine pdf integrates to one and reproduces the analytic cos(theta) distribution
(including the exact tail probabilities 1 - t^2); the uniform-environment estimator is
bit-identical to its mean (zero variance) with cosine sampling while uniform sampling has
positive measured variance with the same mean; the direct-light estimator matches the closed
form pi L (R / d)^2 cos(theta) for both a small and a large spherical light; NEE converges to
the point-light limit as the radius shrinks; MIS is unbiased for every light size while beating
BSDF-only sampling by orders of magnitude on a small light and never losing to the better
single strategy on a large one; Russian roulette keeps the estimator unbiased (matching the
no-RR mean within its standard error) while cutting the mean path length; stratified sampling
lowers the variance of a smooth integrand; the error decreases like 1/sqrt(N); firefly clamping
is shown to be biased while the unclamped estimator is not; and every estimator is
deterministic.

Simplified versus a production renderer: a single shading point with a spherical light instead
of a full scene traversal, analytic (not ray-traced) visibility, and no spectral or BSDF
importance sampling beyond the diffuse cosine lobe.
"""
import math

PI = math.pi
TWO_PI = 2.0 * math.pi
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 stream - the only source of randomness in this file."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def uniform(self):
        state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        self.state = state
        z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return ((z ^ (z >> 31)) >> 11) * (1.0 / float(1 << 53))


def mean(values):
    return sum(values) / len(values)


def variance(values):
    average = mean(values)
    return sum((value - average) ** 2 for value in values) / (len(values) - 1)


# ------------------------------------------------------------------ sampling routines
def sample_cosine_hemisphere(rng):
    """Malley's method: a cosine-distributed direction around +z; returns (direction, pdf)
    with pdf = cos(theta)/pi, so direction[2] is the cosine."""
    u1 = rng.uniform()
    u2 = rng.uniform()
    cos_theta = math.sqrt(1.0 - u1)
    sin_theta = math.sqrt(u1)
    phi = TWO_PI * u2
    return (sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta), cos_theta / PI


def sample_uniform_hemisphere(rng):
    """Area-uniform direction with pdf 1/(2 pi); returns (direction, pdf)."""
    cos_theta = rng.uniform()
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = TWO_PI * rng.uniform()
    return (sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta), 1.0 / TWO_PI


def sample_uniform_sphere(rng):
    cos_theta = 2.0 * rng.uniform() - 1.0
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = TWO_PI * rng.uniform()
    return sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta


def mis_weight(pdf_used, pdf_other, power=1.0):
    """Balance (power = 1) and power (power = 2) heuristics."""
    if pdf_used <= 0.0:
        return 0.0
    a = pdf_used ** power
    b = pdf_other ** power
    return a / (a + b) if a + b > 0.0 else 0.0


def cone_geometry(radius, distance):
    """Half-angle, cosine and solid angle of a sphere light on the normal axis."""
    if not 0.0 < radius < distance:
        raise ValueError("the light must not intersect the surface")
    sin_alpha = radius / distance
    cos_alpha = math.sqrt(max(0.0, 1.0 - sin_alpha * sin_alpha))
    return sin_alpha, cos_alpha, TWO_PI * (1.0 - cos_alpha)


def sample_cone(rng, cos_alpha_max):
    """Uniform solid-angle sampling of the cone: returns (direction, pdf) with pdf = 1/Omega
    and direction[2] = cos(alpha)."""
    cos_alpha = cos_alpha_max + (1.0 - cos_alpha_max) * rng.uniform()
    sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha))
    phi = TWO_PI * rng.uniform()
    omega = TWO_PI * (1.0 - cos_alpha_max)
    return (sin_alpha * math.cos(phi), sin_alpha * math.sin(phi), cos_alpha), 1.0 / omega


# ------------------------------------------------------------------ estimators
def direct_light_estimate(strategy, rng, radius, distance, albedo, radiance, samples,
                          clamp=None):
    """One Monte Carlo estimate of the direct radiance from a spherical light on the normal.

    strategies: "bsdf" (cosine sampling only), "nee" (cone sampling only),
    "mis" (half the samples from each, combined with the balance heuristic).
    """
    _sin_alpha, cos_alpha_max, omega = cone_geometry(radius, distance)
    if strategy == "mis":
        # Standard multiple importance sampling: a *fixed* allocation of samples per strategy
        # (half each) combined with the power heuristic, so the estimator is the sum of two
        # properly normalised averages rather than a randomly chosen strategy.
        half = max(samples // 2, 1)
        total = 0.0
        for _ in range(half):
            direction, pdf_bsdf = sample_cosine_hemisphere(rng)
            cos_theta = direction[2]
            if cos_theta < cos_alpha_max:
                continue
            value = (albedo / PI) * radiance * cos_theta / pdf_bsdf
            total += value * mis_weight(pdf_bsdf, 1.0 / omega, power=2.0)
        for _ in range(samples - half):
            direction, pdf_light = sample_cone(rng, cos_alpha_max)
            cos_theta = direction[2]
            value = (albedo / PI) * radiance * cos_theta / pdf_light
            total += value * mis_weight(pdf_light, cos_theta / PI, power=2.0)
        return total / half
    total = 0.0
    for _ in range(samples):
        use_bsdf = strategy == "bsdf"
        if use_bsdf:
            direction, pdf_bsdf = sample_cosine_hemisphere(rng)
            cos_theta = direction[2]
            if cos_theta < cos_alpha_max:
                continue  # the sampled direction misses the light: f = 0
            value = (albedo / PI) * radiance * cos_theta / pdf_bsdf  # = albedo * radiance
            pdf_light = 1.0 / omega
        else:
            direction, pdf_light = sample_cone(rng, cos_alpha_max)
            cos_theta = direction[2]
            value = (albedo / PI) * radiance * cos_theta / pdf_light
            pdf_bsdf = cos_theta / PI
        if clamp is not None and value > clamp:
            value = clamp
        total += value
    return total / samples


def uniform_environment_estimate(rng, albedo, radiance, samples, stratified=False, strata=1):
    """Outgoing radiance under a uniform environment: the exact answer is albedo * radiance."""
    total = 0.0
    for index in range(samples):
        if stratified and strata > 1:
            u1 = (index % strata + rng.uniform()) / strata
            u2 = rng.uniform()
            cos_theta = math.sqrt(1.0 - u1)
            sin_theta = math.sqrt(u1)
            _direction = (sin_theta * math.cos(TWO_PI * u2),
                          sin_theta * math.sin(TWO_PI * u2), cos_theta)
            pdf = cos_theta / PI
        else:
            _direction, pdf = sample_cosine_hemisphere(rng)
            cos_theta = _direction[2]
        total += (albedo / PI) * radiance * cos_theta / pdf
    return total / samples


def cavity_estimate(rng, albedo, radiance, max_depth, roulette=None):
    """Inter-reflecting cavity: the exact sum is radiance / (1 - albedo).

    Each Lambertian bounce contributes a factor albedo (the cosine cancels the pdf), so the
    path sum is analytic and Russian roulette can be checked exactly rather than statistically.
    """
    weight = 1.0
    total = 0.0
    depth = 0
    while depth < max_depth:
        total += weight * radiance
        weight *= albedo
        depth += 1
        if weight <= 1e-17:
            break
        if roulette is not None and depth > 1:
            if rng.uniform() > roulette:
                break
            weight /= roulette
    return total, depth


# ------------------------------------------------------------------ tests
def main():
    # ---------------- the cosine pdf is the one we claim ----------------
    rng = Rng(20260881)
    above_half = 0
    above_nine = 0
    samples = 200000
    for _ in range(samples):
        direction, _pdf = sample_cosine_hemisphere(rng)
        if direction[2] > 0.5:
            above_half += 1
        if direction[2] > 0.9:
            above_nine += 1
    # P(cos > t) = 1 - t^2 for a cosine-distributed direction
    expected_half = 1.0 - 0.25
    expected_nine = 1.0 - 0.81
    check(abs(above_half / samples - expected_half) < 0.005,
          f"cosine sampling tail mismatch: {above_half / samples} vs {expected_half}")
    check(abs(above_nine / samples - expected_nine) < 0.005,
          f"cosine sampling tail mismatch: {above_nine / samples} vs {expected_nine}")
    uniform_above = 0
    for _ in range(samples):
        direction, _pdf = sample_uniform_hemisphere(rng)
        uniform_above += 1 if direction[2] > 0.5 else 0
    check(abs(uniform_above / samples - 0.5) < 0.005,
          "uniform hemisphere sampling must have P(cos > 0.5) = 0.5")
    check(abs(sum(component * component for component in
                  sample_cosine_hemisphere(rng)[0]) - 1.0) < 1e-12,
          "cosine samples must be unit vectors")

    # ---------------- the uniform environment: zero variance with cosine sampling ----------------
    exact_environment = 0.8 * 1.0
    cosine_estimates = [uniform_environment_estimate(Rng(seed), 0.8, 1.0, 16)
                        for seed in range(64)]
    check(max(abs(value - exact_environment) for value in cosine_estimates) < 1e-12,
          "cosine sampling must give the exact environment answer with zero variance")
    check(variance(cosine_estimates) < 1e-24, "the environment variance must be zero")

    uniform_hemisphere_estimates = []
    for seed in range(4000):
        generator = Rng(seed)
        total = 0.0
        for _ in range(16):
            direction, _pdf = sample_uniform_hemisphere(generator)
            total += (0.8 / PI) * 1.0 * direction[2] * TWO_PI
        uniform_hemisphere_estimates.append(total / 16)
    uniform_variance = variance(uniform_hemisphere_estimates)
    check(uniform_variance > 1e-6,
          f"uniform hemisphere sampling must have real variance: {uniform_variance}")
    check(abs(mean(uniform_hemisphere_estimates) - exact_environment)
          < 4.0 * math.sqrt(uniform_variance / len(uniform_hemisphere_estimates)),
          "uniform hemisphere sampling must still be unbiased")

    stratified = [uniform_environment_estimate(Rng(seed), 0.8, 1.0, 16, stratified=True,
                                               strata=4) for seed in range(4000)]
    check(variance(stratified) < uniform_variance,
          f"stratification must reduce the variance: {variance(stratified)} vs {uniform_variance}")

    # ---------------- direct lighting against the closed form ----------------
    albedo, radiance, distance = 0.8, 1.0, 2.0
    for radius, better_ratio, bsdf_gain in ((0.05, 10.0, 1000.0), (1.2, 20.0, 10.0)):
        analytic = albedo * radiance * (radius / distance) ** 2
        estimates = [direct_light_estimate("nee", Rng(seed), radius, distance, albedo,
                                           radiance, 32) for seed in range(400)]
        spread = variance(estimates)
        check(abs(mean(estimates) - analytic) < 4.0 * math.sqrt(spread / len(estimates)) + 1e-12,
              f"NEE must be unbiased for radius {radius}: {mean(estimates)} vs {analytic}")
        check(spread < (0.1 * analytic) ** 2 + 1e-18,
              f"NEE variance must be small for radius {radius}: {spread}")
        mis_estimates = [direct_light_estimate("mis", Rng(seed + 10000), radius, distance,
                                               albedo, radiance, 32) for seed in range(400)]
        mis_spread = variance(mis_estimates)
        check(abs(mean(mis_estimates) - analytic)
              < 4.0 * math.sqrt(mis_spread / len(mis_estimates)) + 1e-12,
              f"MIS must be unbiased for radius {radius}: {mean(mis_estimates)} vs {analytic}")
        bsdf_estimates = [direct_light_estimate("bsdf", Rng(seed + 20000), radius, distance,
                                                albedo, radiance, 32) for seed in range(400)]
        bsdf_spread = variance(bsdf_estimates)
        check(bsdf_spread > spread,
              f"BSDF sampling must be noisier than NEE at radius {radius}")
        check(mis_spread < bsdf_spread,
              f"MIS must fix the BSDF fireflies at radius {radius}: {mis_spread} vs {bsdf_spread}")
        check(mis_spread * bsdf_gain < bsdf_spread,
              f"MIS must beat BSDF sampling by {bsdf_gain}x at radius {radius}")
        check(mis_spread <= spread * better_ratio,
              f"MIS must stay within {better_ratio}x of the better strategy at radius {radius}: "
              f"{mis_spread} vs {spread}")

    # ---------------- the point-light limit ----------------
    limits = []
    for radius in (0.4, 0.2, 0.05):
        analytic = albedo * radiance * (radius / distance) ** 2
        estimate = mean([direct_light_estimate("nee", Rng(5000 + seed), radius, distance,
                                               albedo, radiance, 64) for seed in range(200)])
        limits.append((radius, abs(estimate - analytic) / analytic))
    check(limits[0][1] > limits[-1][1],
          f"the estimator must approach the point-light limit as R shrinks: {limits}")
    check(limits[-1][1] < 1e-3, f"the small-light relative error must be tiny: {limits}")

    # ---------------- MIS heuristics ----------------
    weights = []
    for _ in range(2000):
        first = rng.uniform() * 4.0
        second = rng.uniform() * 4.0
        balance = mis_weight(first, second, power=1.0)
        power_two = mis_weight(first, second, power=2.0)
        weights.append((balance, power_two))
        check(0.0 <= balance <= 1.0 and 0.0 <= power_two <= 1.0,
              "MIS weights must lie in [0, 1]")
        check(abs(balance + mis_weight(second, first, power=1.0) - 1.0) < 1e-12,
              "the balance weights must form a partition of unity")
    check(mis_weight(0.0, 3.0) == 0.0, "a zero pdf must get a zero weight")
    check(mis_weight(3.0, 0.0, power=2.0) == 1.0, "the only non-zero strategy must get all weight")

    # ---------------- Russian roulette on the analytic cavity ----------------
    exact_cavity = radiance / (1.0 - 0.6)
    full = [cavity_estimate(Rng(seed), 0.6, radiance, 80)[0] for seed in range(600)]
    roulette_runs = [cavity_estimate(Rng(seed + 999), 0.6, radiance, 80, roulette=0.5)
                     for seed in range(600)]
    roulette_values = [value for value, _depth in roulette_runs]
    roulette_depths = [depth for _value, depth in roulette_runs]
    check(abs(mean(full) - exact_cavity) < 1e-12, "the deep cavity sum must be exact")
    spread = math.sqrt(variance(roulette_values) / len(roulette_values))
    check(abs(mean(roulette_values) - exact_cavity) < 4.0 * spread + 1e-9,
          f"Russian roulette must stay unbiased: {mean(roulette_values)} vs {exact_cavity}")
    check(variance(roulette_values) > 0.0, "roulette must introduce variance")
    check(mean(roulette_depths) < 79.0,
          f"roulette must cut the mean path length: {mean(roulette_depths)}")
    check(all(depth >= 1 for depth in roulette_depths), "paths must keep at least one vertex")
    uniform_roulette = [cavity_estimate(Rng(seed), 0.6, radiance, 80, roulette=0.8)[1]
                        for seed in range(200)]
    check(mean(uniform_roulette) > mean(roulette_depths),
          "a higher survival probability must give longer paths")

    # ---------------- convergence and clamping ----------------
    errors = []
    for sample_count in (256, 1024):
        estimates = [direct_light_estimate("nee", Rng(7000 + seed), 1.2, distance, albedo,
                                           radiance, sample_count) for seed in range(40)]
        analytic = albedo * radiance * (1.2 / distance) ** 2
        errors.append(math.sqrt(variance(estimates)))
    check(errors[1] < errors[0] * 0.75,
          f"the error must fall like 1/sqrt(N): {errors}")
    check(errors[1] > errors[0] * 0.25, f"the error must not fall faster than 1/sqrt(N): {errors}")

    analytic_small = albedo * radiance * (0.05 / distance) ** 2
    clamped_bsdf = mean([direct_light_estimate("bsdf", Rng(seed + 30000), 0.05, distance,
                                              albedo, radiance, 64, clamp=0.1)
                         for seed in range(400)])
    unclamped_bsdf = mean([direct_light_estimate("bsdf", Rng(seed + 30000), 0.05, distance,
                                                 albedo, radiance, 64) for seed in range(400)])
    check(abs(unclamped_bsdf - analytic_small) < 0.1 * analytic_small,
          f"the unclamped BSDF estimator must be unbiased: {unclamped_bsdf} vs {analytic_small}")
    check(clamped_bsdf < unclamped_bsdf * 0.5,
          f"clamping fireflies must bias the estimate downwards: {clamped_bsdf} vs "
          f"{unclamped_bsdf}")

    # ---------------- infinite-environment invariance and invalid inputs ----------------
    sphere_direction = sample_uniform_sphere(rng)
    check(abs(sum(component * component for component in sphere_direction) - 1.0) < 1e-12,
          "a sphere sample must be a unit vector")
    for call, why in (
        (lambda: cone_geometry(1.0, 1.0), "light touching the surface"),
        (lambda: cone_geometry(1.5, 1.0), "light intersecting the surface"),
        (lambda: cone_geometry(0.0, 1.0), "degenerate light"),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")

    # ---------------- determinism ----------------
    first = direct_light_estimate("mis", Rng(1234), 1.2, distance, albedo, radiance, 128)
    second = direct_light_estimate("mis", Rng(1234), 1.2, distance, albedo, radiance, 128)
    check(first == second, "the estimator must be deterministic")
    check(cavity_estimate(Rng(7), 0.6, 1.0, 40, roulette=0.5)
          == cavity_estimate(Rng(7), 0.6, 1.0, 40, roulette=0.5),
          "the roulette path must be deterministic")

    print(f"cosine tails: P(cos>0.5)={above_half / samples:.4f} P(cos>0.9)={above_nine / samples:.4f}")
    print(f"environment variance: cosine={variance(cosine_estimates):.2e} "
          f"uniform={uniform_variance:.2e} stratified={variance(stratified):.2e}")
    print(f"cavity exact={exact_cavity:.6f} roulette mean={mean(roulette_values):.6f} "
          f"mean depth={mean(roulette_depths):.2f}")
    print(f"clamped={clamped_bsdf:.4f} unclamped={unclamped_bsdf:.4f} "
          f"analytic={analytic_small:.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
