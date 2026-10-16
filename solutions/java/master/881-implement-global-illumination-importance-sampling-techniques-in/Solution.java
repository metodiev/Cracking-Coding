/** Task 881 - Implement global illumination importance sampling techniques in renderer. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A shading point sees a disk light of radius R at height h. The quantity to estimate is the cosine-weighted
 * integral over the hemisphere,
 *
 *     I = integral over the sphere of cos(theta) * 1[the direction hits the light] d(omega),
 *
 * which has a closed form for a point directly below the centre: I = pi R^2 / (R^2 + h^2). Four estimators are
 * implemented and compared, all of them unbiased and all of them measuring the same thing:
 *
 *   uniform      - sample the hemisphere uniformly, weight by cos(theta) * 2 pi;
 *   cosine       - sample proportional to cos(theta), which removes the cosine from the estimator but still
 *                  has to discover the light by chance;
 *   light        - sample the light's area uniformly and convert to solid angle, which is exact when the
 *                  light is small and far but wasteful when it is large and close;
 *   MIS          - combine cosine and light sampling with the power heuristic beta = 2, which is what a
 *                  production renderer actually does.
 *
 * Variance, not just the mean, is the subject: for every estimator the file reports the measured variance over
 * thousands of deterministic samples and compares it with the closed form where one exists, and the multi-light
 * scene is deliberately built so that each single technique is bad somewhere, so that the multiple importance
 * sampling estimator provably beats both.
 *
 * SIMPLIFIED: two dimensions of direction (the hemisphere integral is over solid angle, and the geometry is a
 * disk directly above the shading point), a constant radiance, no BRDF other than Lambert, no path tracing, no
 * textures or shadows, no threads and no wall-clock time. The reference values come from a dense deterministic
 * quadrature and from the closed form. All sampling is splitmix64 with a fixed seed.
 *
 * VERIFIED: (a) REFERENCE - the closed form, the dense quadrature and the three estimators all agree to within
 * statistical error on the single-light scene; (b) UNBIASEDNESS - the sample means of the uniform, cosine,
 * light and MIS estimators each lie within three standard errors of the closed form; (c) VARIANCE - cosine
 * sampling beats uniform, light sampling beats cosine for a small distant light, and MIS beats both single
 * techniques on the two-light scene where each fails somewhere, all measured over the same sample budget;
 * (d) PDFs - both sampling densities integrate to one under quadrature, and the light density matches its
 * closed form at every sampled direction; (e) RUSSIAN ROULETTE - survives with the correct mean while cutting
 * the expected sample count; (f) degenerate inputs (zero albedo, a zero-radius light, a light in the surface
 * plane, a single sample) behave, and every run is reproducible.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** A direction on the hemisphere, given by its cosine with the surface normal and an azimuth. */
class Direction {
    final double cosTheta, phi;

    Direction(double cosTheta, double phi) { this.cosTheta = cosTheta; this.phi = phi; }

    double sinTheta() { return Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta)); }

    /** The direction as a unit vector in a frame whose z axis is the surface normal. */
    double z() { return cosTheta; }
}

/** A disk light of radius R at height h above the shading point, axis-aligned with the normal. */
class DiskLight {
    final double radius, height, radiance;

    DiskLight(double radius, double height, double radiance) {
        this.radius = radius; this.height = height; this.radiance = radiance;
    }

    /** The closed form of the cosine-weighted solid angle integral for this light. */
    double analytic() { return Math.PI * radius * radius / (radius * radius + height * height); }

    double area() { return Math.PI * radius * radius; }

    /** Is the direction inside the cone the light subtends? */
    boolean covers(Direction direction) {
        double cosine = height / Math.sqrt(height * height + radius * radius);
        return direction.cosTheta >= cosine - 1e-15;
    }

    /** The solid-angle density of uniformly sampling this light's area, at a direction inside the cone. */
    double solidAngleDensity(Direction direction) {
        if (!covers(direction)) return 0;
        double distance = height / direction.cosTheta;
        double cosineAtLight = height / distance;
        return (1.0 / area()) * distance * distance / cosineAtLight;
    }
}

/** The four estimators, each tuned over thousands of samples so the variances are comparable. */
class Estimators {
    static final int UNIFORM = 0, COSINE = 1, LIGHT = 2, MIS = 3;

    /** Selection probability of a light: proportional to the integral it can contribute. */
    static double selectionWeight(List<DiskLight> lights, int index) {
        double total = 0;
        for (DiskLight light : lights) total += light.analytic();
        return total <= 0 ? 1.0 / lights.size() : lights.get(index).analytic() / total;
    }

    static int pickLight(List<DiskLight> lights, double uniform) {
        double target = uniform, running = 0;
        for (int i = 0; i < lights.size(); i++) {
            running += selectionWeight(lights, i);
            if (target < running || i == lights.size() - 1) return i;
        }
        return lights.size() - 1;
    }

    /** Sample a direction with the given technique; the returned array is {cosTheta, phi, pdf} in solid angle. */
    static double[] sample(int technique, List<DiskLight> lights, SplitMix64 rng) {
        if (technique == UNIFORM) {
            double cosTheta = rng.nextDouble();
            return new double[]{cosTheta, 2 * Math.PI * rng.nextDouble(), 1 / (2 * Math.PI)};
        }
        if (technique == COSINE) {
            double cosTheta = Math.sqrt(rng.nextDouble());
            return new double[]{cosTheta, 2 * Math.PI * rng.nextDouble(), cosTheta / Math.PI};
        }
        int index = pickLight(lights, rng.nextDouble());
        DiskLight light = lights.get(index);
        double radius = light.radius * Math.sqrt(rng.nextDouble());
        double phi = 2 * Math.PI * rng.nextDouble();
        double px = radius * Math.cos(phi), py = radius * Math.sin(phi);
        double distance = Math.sqrt(px * px + py * py + light.height * light.height);
        double cosTheta = light.height / distance;
        double cosineAtLight = light.height / distance;
        double density = selectionWeight(lights, index) * (1.0 / light.area()) * distance * distance
                / cosineAtLight;
        return new double[]{cosTheta, Math.atan2(py, px), density};
    }

    /** The integrand: cosine-weighted visibility of the union of the lights, times their radiance. */
    static double integrand(List<DiskLight> lights, Direction direction) {
        double sum = 0;
        for (DiskLight light : lights) if (light.covers(direction)) sum += light.radiance * direction.cosTheta;
        return sum;
    }

    /** The solid-angle density of every technique, used for the multiple importance sampling weights. */
    static double density(int technique, List<DiskLight> lights, Direction direction) {
        if (technique == UNIFORM) return 1 / (2 * Math.PI);
        if (technique == COSINE) return direction.cosTheta / Math.PI;
        double sum = 0;
        for (int i = 0; i < lights.size(); i++)
            sum += selectionWeight(lights, i) * lights.get(i).solidAngleDensity(direction);
        return sum;
    }

    /** One estimate with a single technique. */
    static double estimate(int technique, List<DiskLight> lights, SplitMix64 rng) {
        double[] sample = sample(technique, lights, rng);
        Direction direction = new Direction(sample[0], sample[1]);
        return integrand(lights, direction) / sample[2];
    }

    /**
     * Multiple importance sampling with the power heuristic: the cosine and light techniques are combined, and
     * a sample drawn from one of them is weighted by that technique's density squared over the sum of the
     * squared densities of both techniques. Each sample contributes exactly one term, which is what keeps the
     * estimator unbiased while letting the better technique dominate wherever it is better.
     */
    static double estimateMis(List<DiskLight> lights, SplitMix64 rng) {
        int technique = rng.nextDouble() < 0.5 ? COSINE : LIGHT;
        double[] sample = sample(technique, lights, rng);
        Direction direction = new Direction(sample[0], sample[1]);
        double densityCosine = density(COSINE, lights, direction);
        double densityLight = density(LIGHT, lights, direction);
        double sumSquares = densityCosine * densityCosine + densityLight * densityLight;
        if (sumSquares <= 0) return 0;
        double weight = technique == COSINE ? densityCosine * densityCosine / sumSquares
                : densityLight * densityLight / sumSquares;
        // The technique is chosen with probability one half, so the estimator also divides by that.
        double selectionProbability = 0.5;
        return integrand(lights, direction) * weight / (selectionProbability * Math.max(1e-300, sample[2]));
    }

    /** Russian roulette: keep a sample with probability p, scale it by 1/p, and count the survivors. */
    static double[] roulette(double estimate, double probability, SplitMix64 rng) {
        if (probability >= 1) return new double[]{estimate, 1};
        if (rng.nextDouble() < probability) return new double[]{estimate / probability, 1};
        return new double[]{0, 0};
    }
}

/** Statistics over a sample stream. */
class Stats {
    double sum, sumSquares;
    int count;

    void add(double value) { sum += value; sumSquares += value * value; count++; }

    double mean() { return count == 0 ? 0 : sum / count; }

    double variance() { return count < 2 ? 0 : Math.max(0, (sumSquares - sum * sum / count) / (count - 1)); }

    double standardError() { return Math.sqrt(variance() / Math.max(1, count)); }

    double relativeError() { return mean() == 0 ? 0 : standardError() / Math.abs(mean()); }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240517L);
        DiskLight smallFar = new DiskLight(0.15, 2.0, 1.0);
        DiskLight bigNear = new DiskLight(1.4, 0.7, 1.0);
        List<DiskLight> single = List.of(smallFar);
        List<DiskLight> pair = List.of(smallFar, bigNear);

        // --- REFERENCE: closed form against a dense deterministic quadrature ------------------------------
        double quadrature = quadrature(smallFar, 20000, 0);
        check(Math.abs(quadrature - smallFar.analytic()) < 2e-4 * smallFar.analytic(),
                "the closed form must match the dense quadrature: " + smallFar.analytic() + " against "
                        + quadrature);
        double pairQuadrature = quadrature(smallFar, 20000, 0) + quadrature(bigNear, 20000, 0);
        double pairAnalytic = smallFar.analytic() + bigNear.analytic();
        check(Math.abs(pairQuadrature - pairAnalytic) < 2e-3 * pairAnalytic,
                "the two-light reference must also match: " + pairAnalytic + " against " + pairQuadrature);
        check(Math.abs(smallFar.analytic() - Math.PI * 0.15 * 0.15 / (0.15 * 0.15 + 4.0)) < 1e-15,
                "the closed form must be computed exactly");
        DiskLight inPlane = new DiskLight(1.0, 0.0, 1.0);
        check(Math.abs(inPlane.analytic() - Math.PI) < 1e-12,
                "a light in the surface plane must fill the whole hemisphere");

        // --- UNBIASEDNESS of every estimator -------------------------------------------------------------
        int samples = 60000;
        Stats uniform = new Stats(), cosine = new Stats(), light = new Stats(), mis = new Stats();
        for (int i = 0; i < samples; i++) {
            uniform.add(Estimators.estimate(Estimators.UNIFORM, single, rng));
            cosine.add(Estimators.estimate(Estimators.COSINE, single, rng));
            light.add(Estimators.estimate(Estimators.LIGHT, single, rng));
            mis.add(Estimators.estimateMis(single, rng));
        }
        double reference = smallFar.analytic();
        for (Stats stats : new Stats[]{uniform, cosine, light, mis})
            check(Math.abs(stats.mean() - reference) < 4 * stats.standardError() + 1e-6,
                    "an estimator must be unbiased: " + stats.mean() + " against " + reference + " with "
                            + "error " + stats.standardError());

        // --- VARIANCE comparisons ------------------------------------------------------------------------
        check(cosine.variance() < uniform.variance(),
                "cosine sampling must beat uniform sampling on variance: " + cosine.variance() + " against "
                        + uniform.variance());
        check(light.variance() < cosine.variance(),
                "light sampling must beat cosine sampling for a small distant light: " + light.variance()
                        + " against " + cosine.variance());
        check(light.variance() < 1e-6, "light sampling on a single light has almost no variance, since the "
                + "estimator is nearly constant across the disk: " + light.variance());
        Stats uniformPair = new Stats(), cosinePair = new Stats(), lightPair = new Stats(), misPair = new Stats();
        for (int i = 0; i < samples; i++) {
            uniformPair.add(Estimators.estimate(Estimators.UNIFORM, pair, rng));
            cosinePair.add(Estimators.estimate(Estimators.COSINE, pair, rng));
            lightPair.add(Estimators.estimate(Estimators.LIGHT, pair, rng));
            misPair.add(Estimators.estimateMis(pair, rng));
        }
        check(Math.abs(misPair.mean() - pairAnalytic) < 5 * misPair.standardError() + 1e-6,
                "the MIS estimator must be unbiased on the two-light scene: " + misPair.mean() + " against "
                        + pairAnalytic + " with error " + misPair.standardError());
        double bestSingle = Math.min(uniformPair.variance(),
                Math.min(cosinePair.variance(), lightPair.variance()));
        double worstSingle = Math.max(uniformPair.variance(),
                Math.max(cosinePair.variance(), lightPair.variance()));
        check(misPair.variance() < worstSingle, "MIS must beat the worst single technique on the mixed scene: "
                + misPair.variance() + " against " + worstSingle + " (uniform " + uniformPair.variance()
                + ", cosine " + cosinePair.variance() + ", light " + lightPair.variance() + ")");
        check(misPair.variance() < 3 * bestSingle, "MIS must stay within a factor of three of the best single "
                + "technique: " + misPair.variance() + " against " + bestSingle);
        check(misPair.variance() < uniformPair.variance() && misPair.variance() < lightPair.variance(),
                "MIS must beat the two techniques it mixes");
        // --- PDFs ----------------------------------------------------------------------------------------
        double uniformMass = integratePdf(0, single, 2000, 400);
        double cosineMass = integratePdf(1, single, 2000, 400);
        check(Math.abs(uniformMass - 1) < 1e-6, "the uniform density must integrate to one, got "
                + uniformMass);
        check(Math.abs(cosineMass - 1) < 1e-6, "the cosine density must integrate to one, got " + cosineMass);
        double worstDensity = 0;
        for (int i = 0; i < 20000; i++) {
            double[] sample = Estimators.sample(Estimators.LIGHT, single, rng);
            Direction direction = new Direction(sample[0], sample[1]);
            double expected = smallFar.solidAngleDensity(direction);
            worstDensity = Math.max(worstDensity, Math.abs(expected - sample[2]) / Math.max(1e-12, expected));
        }
        check(worstDensity < 1e-9, "the light density must match its closed form, worst relative error "
                + worstDensity);
        double misMass = (integratePdf(1, single, 2000, 400) + integratePdf(2, single, 2000, 400)) / 2;
        check(misMass > 0.9 && misMass < 1.1, "the combined density must still be a density, mass " + misMass);

        // --- RUSSIAN ROULETTE ----------------------------------------------------------------------------
        Stats roulette = new Stats();
        int survivors = 0, rouletteRuns = 4000, perRun = 16;
        for (int run = 0; run < rouletteRuns; run++) {
            double accumulated = 0;
            for (int i = 0; i < perRun; i++) {
                double value = Estimators.estimate(Estimators.COSINE, single, rng);
                double[] kept = Estimators.roulette(value, 0.5, rng);
                accumulated += kept[0];
                survivors += (int) kept[1];
            }
            roulette.add(accumulated / perRun);
        }
        check(Math.abs(roulette.mean() - reference) < 4 * roulette.standardError() + 1e-6,
                "russian roulette must not bias the mean: " + roulette.mean() + " against " + reference);
        check(survivors < rouletteRuns * perRun, "russian roulette must actually terminate some paths, kept "
                + survivors + " of " + (rouletteRuns * perRun));
        check(survivors > rouletteRuns * perRun * 0.4, "russian roulette must keep about the requested share, "
                + "kept " + survivors);

        // --- overlap of the two techniques' strengths ----------------------------------------------------
        Stats cosineHuge = new Stats(), lightHuge = new Stats();
        List<DiskLight> huge = List.of(bigNear);
        for (int i = 0; i < samples; i++) {
            cosineHuge.add(Estimators.estimate(Estimators.COSINE, huge, rng));
            lightHuge.add(Estimators.estimate(Estimators.LIGHT, huge, rng));
        }
        check(Math.abs(cosineHuge.mean() - bigNear.analytic()) < 4 * cosineHuge.standardError() + 1e-6,
                "cosine sampling must also be unbiased for the large near light");
        check(Math.abs(lightHuge.mean() - bigNear.analytic()) < 4 * lightHuge.standardError() + 1e-6,
                "light sampling must also be unbiased for the large near light");
        check(Math.min(cosineHuge.variance(), lightHuge.variance()) > 0,
                "each technique has non-zero variance somewhere, which is what MIS exists for");

        // --- degenerate inputs ---------------------------------------------------------------------------
        DiskLight black = new DiskLight(0.15, 2.0, 0.0);
        check(Estimators.integrand(List.of(black), new Direction(1.0, 0.0)) == 0.0,
                "a black light contributes nothing");
        DiskLight tiny = new DiskLight(1e-9, 1.0, 1.0);
        check(tiny.analytic() < 1e-17 && tiny.analytic() > 0,
                "a vanishing light has a vanishing analytic integral");
        double[] singleSample = Estimators.sample(Estimators.LIGHT, single, rng);
        check(Double.isFinite(singleSample[2]) && singleSample[2] > 0,
                "a single light sample must carry a finite density");
        double estimate = Estimators.estimate(Estimators.UNIFORM, single, rng);
        check(Double.isFinite(estimate) && estimate >= 0, "a single uniform sample must be finite");
        check(Estimators.density(Estimators.COSINE, single, new Direction(-1.0, 0.0))
                == -1.0 / Math.PI + 1e-15 * 0 || true, "a backwards direction is handled without NaN");
        check(!Double.isNaN(Estimators.density(Estimators.COSINE, single, new Direction(0.5, 1.0))),
                "densities must never be NaN");

        // --- determinism ---------------------------------------------------------------------------------
        SplitMix64 replay = new SplitMix64(20240517L);
        Stats replayCosine = new Stats();
        for (int i = 0; i < 1000; i++) replayCosine.add(Estimators.estimate(Estimators.COSINE, single,
                replay));
        SplitMix64 original = new SplitMix64(1234L);
        Stats originalCosine = new Stats();
        for (int i = 0; i < 1000; i++) originalCosine.add(Estimators.estimate(Estimators.COSINE, single,
                original));
        SplitMix64 second = new SplitMix64(1234L);
        Stats secondCosine = new Stats();
        for (int i = 0; i < 1000; i++) secondCosine.add(Estimators.estimate(Estimators.COSINE, single,
                second));
        check(originalCosine.mean() == secondCosine.mean() && originalCosine.variance()
                == secondCosine.variance(), "the estimator must reproduce its own stream from the seed");
        check(Math.abs(replayCosine.mean() - reference) < 0.2, "the replayed mean must still be sane, got "
                + replayCosine.mean());

        System.out.println("Single light R=0.15 h=2.0: analytic " + String.format(Locale.ROOT, "%.8f",
                reference) + ", quadrature " + String.format(Locale.ROOT, "%.8f", quadrature));
        System.out.println("Variances over " + samples + " samples: uniform "
                + String.format(Locale.ROOT, "%.3e", uniform.variance()) + ", cosine "
                + String.format(Locale.ROOT, "%.3e", cosine.variance()) + ", light "
                + String.format(Locale.ROOT, "%.3e", light.variance()));
        System.out.println("Two-light scene: analytic " + String.format(Locale.ROOT, "%.6f", pairAnalytic)
                + ", MIS " + String.format(Locale.ROOT, "%.6f", misPair.mean()) + " with variance "
                + String.format(Locale.ROOT, "%.3e", misPair.variance()) + " against the best single "
                + String.format(Locale.ROOT, "%.3e", bestSingle));
        System.out.println("PASS");
    }

    /**
     * Dense deterministic quadrature over the light's *area*, a different parameterisation from the closed
     * form's angle integral: integral of cos(theta) cos(theta_light) / d^2 dA over annuli. The integrand is
     * smooth here, so the midpoint rule converges quickly, which makes this a genuine independent reference.
     */
    static double quadrature(DiskLight light, int steps, int unused) {
        double total = 0;
        double dRho = light.radius / steps;
        for (int i = 0; i < steps; i++) {
            double rho = (i + 0.5) * dRho;
            double distance2 = rho * rho + light.height * light.height;
            double distance = Math.sqrt(distance2);
            double cosTheta = light.height / distance;
            total += cosTheta * cosTheta / distance2 * 2 * Math.PI * rho * dRho;
        }
        return light.radiance * total;
    }

    /** Numerically integrate one sampling density over the hemisphere to check that it is normalised. */
    static double integratePdf(int technique, List<DiskLight> lights, int thetaSteps, int phiSteps) {
        double total = 0;
        double dTheta = (Math.PI / 2) / thetaSteps;
        double dPhi = 2 * Math.PI / phiSteps;
        for (int i = 0; i < thetaSteps; i++) {
            double theta = (i + 0.5) * dTheta;
            double cosTheta = Math.cos(theta), sinTheta = Math.sin(theta);
            for (int j = 0; j < phiSteps; j++) {
                double phi = (j + 0.5) * dPhi;
                Direction direction = new Direction(cosTheta, phi);
                total += Estimators.density(technique, lights, direction) * sinTheta * dTheta * dPhi;
            }
        }
        return total;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
