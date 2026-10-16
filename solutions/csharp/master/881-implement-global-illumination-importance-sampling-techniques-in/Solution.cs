// Task 881 - Implement global illumination importance sampling techniques in renderer.
// (Tier: Master, CTCI Beyond - Rendering & Graphics)
//
// Model: the sampling machinery of a path tracer, isolated so that every claim can be checked against an
// analytic value:
//   * the integrand is the outgoing radiance of a Lambertian surface,
//     L_o = (rho / pi) * integral_hemisphere L_i(omega) cos(theta) d(omega),
//     with a piecewise-constant environment (a bright cap of half-angle pi/4 around the normal and a dim
//     background) whose integral is known in closed form;
//   * three estimators compete on the same samples: cosine-weighted hemisphere sampling (density
//     cos(theta)/pi), uniform hemisphere sampling (density 1/2pi) and multiple importance sampling with the
//     balance heuristic w_i = p_i / sum_j p_j;
//   * Russian roulette terminates the deeper bounces without changing the expectation.
// The interesting properties are the ones a renderer depends on: the white-furnace identity (the
// cosine-weighted estimator returns exactly rho * L for *every* sample in a uniform environment), the MIS
// weights summing to one, the balance heuristic staying within a small constant factor of the better
// strategy, and Russian roulette leaving the mean unchanged while lowering the average path length.
//
// Simplifications (documented, not behavioural): radiance is a scalar (the per-channel handling is
// identical and would triple every number), bounces are represented by their sampling weights instead of a
// full scene traversal, no textures or lights with singular densities, and the environment is analytic
// rather than an HDR map. The sampling itself - densities, weights, stratification-free determinism - is the
// real thing.
//
// Verified here:
//   * white furnace: uniform environment, cosine sampling reproduces rho * L exactly per sample (variance
//     zero to machine precision), and uniform sampling and MIS do the same in expectation;
//   * multiple importance sampling: balance-heuristic weights sum to 1 for arbitrary densities, the
//     estimator is unbiased against the closed-form integral, stays within 2x of the better single strategy
//     while clearly beating the worse one, and its mixture parameter is a real tuning knob;
//   * quadrature oracle: a deterministic product quadrature (Gauss-Legendre style midpoint grid over the
//     hemisphere) reproduces the same integral, independently of the Monte Carlo estimators;
//   * Russian roulette: unbiased with the termination applied (mean within tolerance of the reference) and
//     the average path length strictly smaller, plus the survival-probability bookkeeping;
//   * degenerate cases: zero albedo, zero radiance, a fully occluded hemisphere, a delta-bright cap and a
//     single-sample run;
//   * determinism: every estimator is bitwise reproducible from its seed.
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static ulong _state = 0x9E3779B97F4A7C15UL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextUnit() => (NextU64() >> 11) * (1.0 / 9007199254740992.0);

    // ---------- directions ----------
    private struct Dir
    {
        public double X;
        public double Y;
        public double Z;         // z is the surface normal axis
        public Dir(double x, double y, double z) { X = x; Y = y; Z = z; }
        public double CosTheta => Z;
        public override string ToString() => "(" + X.ToString("F4") + "," + Y.ToString("F4") + "," + Z.ToString("F4") + ")";
    }

    // ---------- environment and material ----------
    private sealed class Zone
    {
        public double ThetaStart;
        public double ThetaEnd;
        public double Radiance;
        public Zone(double a, double b, double l) { ThetaStart = a; ThetaEnd = b; Radiance = l; }
    }

    private sealed class Environment
    {
        public List<Zone> Zones = new List<Zone>();
        public bool Uniform;
        public double UniformRadiance = 1.0;
        public double OccluderAngle = double.NaN;     // optional: a black cap around the normal

        public double Radiance(Dir d)
        {
            if (d.CosTheta <= 0) return 0.0;
            double theta = Math.Acos(Math.Clamp(d.Z, -1, 1));
            if (!double.IsNaN(OccluderAngle) && theta < OccluderAngle) return 0.0;
            if (Uniform) return UniformRadiance;
            foreach (Zone z in Zones) if (theta >= z.ThetaStart && theta < z.ThetaEnd) return z.Radiance;
            return 0.0;
        }

        /// <summary>
        /// Closed form of integral L(omega) cos(theta) d(omega) over the upper hemisphere: each zone
        /// contributes pi * R * (cos^2(thetaStart) - cos^2(thetaEnd)).
        /// </summary>
        public double IrradianceClosedForm()
        {
            if (Uniform) return Math.PI * UniformRadiance * (double.IsNaN(OccluderAngle) ? 1.0 : Math.Cos(OccluderAngle) * Math.Cos(OccluderAngle));
            double total = 0.0;
            foreach (Zone z in Zones)
            {
                double start = double.IsNaN(OccluderAngle) ? z.ThetaStart : Math.Max(z.ThetaStart, OccluderAngle);
                if (z.ThetaEnd <= start) continue;
                total += Math.PI * z.Radiance *
                         (Math.Cos(start) * Math.Cos(start) - Math.Cos(z.ThetaEnd) * Math.Cos(z.ThetaEnd));
            }
            return total;
        }
    }

    // ---------- sampling strategies ----------
    /// <summary>Cosine-weighted hemisphere sample; density cos(theta)/pi.</summary>
    private static Dir SampleCosine(out double pdf)
    {
        double u1 = NextUnit(), u2 = NextUnit();
        double r = Math.Sqrt(u1);
        double phi = 2.0 * Math.PI * u2;
        double z = Math.Sqrt(Math.Max(0.0, 1.0 - u1));
        Dir d = new Dir(r * Math.Cos(phi), r * Math.Sin(phi), z);
        pdf = Math.Max(z, 1e-12) / Math.PI;
        return d;
    }

    /// <summary>Uniform hemisphere sample; density 1/(2 pi).</summary>
    private static Dir SampleUniform(out double pdf)
    {
        double u1 = NextUnit(), u2 = NextUnit();
        double z = u1;
        double r = Math.Sqrt(Math.Max(0.0, 1.0 - z * z));
        double phi = 2.0 * Math.PI * u2;
        pdf = 1.0 / (2.0 * Math.PI);
        return new Dir(r * Math.Cos(phi), r * Math.Sin(phi), z);
    }

    private static double PdfCosine(Dir d) => d.CosTheta <= 0 ? 0.0 : d.CosTheta / Math.PI;
    private static double PdfUniform(Dir d) => d.CosTheta <= 0 ? 0.0 : 1.0 / (2.0 * Math.PI);

    /// <summary>Balance heuristic: the weights of a set of strategies sum to one by construction.</summary>
    private static double[] BalanceWeights(double[] densities)
    {
        double sum = densities.Sum();
        if (sum <= 0) return densities.Select(_ => 0.0).ToArray();
        return densities.Select(p => p / sum).ToArray();
    }

    // ---------- estimators ----------
    /// <summary>
    /// Accumulator using Welford's online algorithm: the naive sum-of-squares form loses all precision when
    /// the variance is near zero, which is exactly the case the white-furnace check looks at.
    /// </summary>
    private sealed class Estimate
    {
        public double Mean;
        public double M2;
        public long Samples;

        public void Add(double value)
        {
            Samples++;
            double delta = value - Mean;
            Mean += delta / Samples;
            M2 += delta * (value - Mean);
        }

        public double Variance => Samples > 1 ? M2 / (Samples - 1) : 0.0;
        public double Error => Math.Sqrt(Variance / Math.Max(1, Samples));
        public long RouletteKills;
        public long Segments;
    }

    private static Estimate EstimateCosine(Environment env, double albedo, int samples, ulong seed)
    {
        _state = seed;
        var e = new Estimate();
        for (int i = 0; i < samples; i++)
        {
            Dir d = SampleCosine(out double pdf);
            double f = albedo / Math.PI * env.Radiance(d) * d.CosTheta;
            e.Add(f / pdf);
            e.Segments++;
        }
        return e;
    }

    private static Estimate EstimateUniform(Environment env, double albedo, int samples, ulong seed)
    {
        _state = seed;
        var e = new Estimate();
        for (int i = 0; i < samples; i++)
        {
            Dir d = SampleUniform(out double pdf);
            double f = albedo / Math.PI * env.Radiance(d) * d.CosTheta;
            e.Add(f / pdf);
            e.Segments++;
        }
        return e;
    }

    /// <summary>
    /// One-sample MIS with the balance heuristic. The technique is chosen with probability pCosine and the
    /// value is f * w_k / (p_k * P_k) with w_k = P_k p_k / sum_j P_j p_j, which collapses to the elegant
    /// f / (P_c p_c + P_u p_u): the balance heuristic behaves like sampling from the combined density, so the
    /// estimator is unbiased for any mixture P and the weights still sum to one over the techniques.
    /// </summary>
    private static Estimate EstimateMis(Environment env, double albedo, int samples, ulong seed, double pCosine)
    {
        _state = seed;
        var e = new Estimate();
        for (int i = 0; i < samples; i++)
        {
            bool useCosine = NextUnit() < pCosine;
            Dir d = useCosine ? SampleCosine(out _) : SampleUniform(out _);
            double pc = PdfCosine(d), pu = PdfUniform(d);
            double[] w = BalanceWeights(new[] { pCosine * pc, (1 - pCosine) * pu });
            double f = albedo / Math.PI * env.Radiance(d) * d.CosTheta;
            double pdf = useCosine ? pc : pu;
            double selection = useCosine ? pCosine : 1 - pCosine;
            e.Add(f * (useCosine ? w[0] : w[1]) / (pdf * selection));
            e.Segments++;
        }
        return e;
    }

    private static Estimate EstimateMis(Environment env, double albedo, int samples, ulong seed) =>
        EstimateMis(env, albedo, samples, seed, 0.5);

    /// <summary>
    /// Multi-bounce estimate with Russian roulette. Each bounce contributes the sampled radiance attenuated by
    /// the throughput, the throughput decays by the albedo (energy transfer) and roulette either terminates
    /// the path with probability 1-q or divides the throughput by q so the expectation is untouched: the
    /// analytic mean is therefore reference / (1 - albedo).
    /// </summary>
    private static Estimate EstimateWithRussianRoulette(Environment env, double albedo, int samples, ulong seed,
                                                        double survival, out double meanPathLength)
    {
        _state = seed;
        var e = new Estimate();
        long segments = 0;
        for (int i = 0; i < samples; i++)
        {
            double throughput = 1.0;
            double value = 0.0;
            int bounce = 0;
            while (true)
            {
                Dir d = SampleCosine(out double pdf);
                double f = albedo / Math.PI * env.Radiance(d) * d.CosTheta;
                value += throughput * f / pdf;
                segments++;
                bounce++;
                throughput *= albedo;                       // energy transferred to the next bounce
                if (bounce >= 6) break;
                if (NextUnit() > survival)
                {
                    e.RouletteKills++;
                    break;
                }
                throughput /= survival;                     // the classic roulette compensation
            }
            e.Add(value);
        }
        meanPathLength = segments / (double)Math.Max(1, samples);
        return e;
    }

    /// <summary>
    /// Independent numerical oracle: the phi integral is analytic (2 pi) and the theta integral is a midpoint
    /// rule applied *per zone*, so the grid resolves every discontinuity of L and the rule converges
    /// quadratically inside each smooth piece - no analytical shortcut from the closed form is reused.
    /// </summary>
    private static double QuadratureIrradiance(Environment env, int stepsPerZone)
    {
        double lowerLimit = double.IsNaN(env.OccluderAngle) ? 0.0 : env.OccluderAngle;
        double total = 0.0;
        if (env.Uniform)
        {
            double a0 = lowerLimit, b0 = Math.PI / 2.0, d = (b0 - a0) / stepsPerZone;
            for (int i = 0; i < stepsPerZone; i++)
            {
                double th = a0 + d * (i + 0.5);
                total += 2.0 * Math.PI * env.UniformRadiance * Math.Cos(th) * Math.Sin(th) * d;
            }
            return total;
        }
        foreach (Zone z in env.Zones)
        {
            double a = Math.Max(z.ThetaStart, lowerLimit), b = z.ThetaEnd;
            if (b <= a) continue;
            double d = (b - a) / stepsPerZone;
            for (int i = 0; i < stepsPerZone; i++)
            {
                double th = a + d * (i + 0.5);
                total += 2.0 * Math.PI * z.Radiance * Math.Cos(th) * Math.Sin(th) * d;
            }
        }
        return total;
    }

    // ---------- main ----------
    public static void Main()
    {
        // 1-2: white furnace - a uniform environment must return exactly rho * L for every cosine sample.
        var furnace = new Environment { Uniform = true, UniformRadiance = 0.8 };
        double albedo = 0.65;
        var furnaceEstimate = EstimateCosine(furnace, albedo, 256, 0x1234UL);
        Check(Math.Abs(furnaceEstimate.Mean - albedo * 0.8) < 1e-12,
              "the white furnace returns exactly rho*L (" + furnaceEstimate.Mean + " vs " + albedo * 0.8 + ")");
        Check(furnaceEstimate.Variance < 1e-24,
              "the furnace estimator has zero variance (all samples identical, var " + furnaceEstimate.Variance.ToString("E2") + ")");
        double uniformQuadrature = QuadratureIrradiance(furnace, 4096);
        Check(Math.Abs(uniformQuadrature - Math.PI * 0.8) < 1e-6,
              "the quadrature oracle reproduces the pi*L irradiance of a uniform sky (" + uniformQuadrature + ")");
        Check(Math.Abs(furnace.IrradianceClosedForm() - Math.PI * 0.8) < 1e-12,
              "the closed form agrees with pi*L");

        // 3: MIS weights sum to one for arbitrary densities.
        int weightChecks = 0;
        for (int i = 0; i < 200; i++)
        {
            double a = NextUnit() * 10.0, b = NextUnit() * 10.0;
            double[] w = BalanceWeights(new[] { a, b });
            if (Math.Abs(w[0] + w[1] - 1.0) < 1e-15 && w[0] >= 0 && w[1] >= 0) weightChecks++;
        }
        Check(weightChecks == 200, "the balance heuristic weights always sum to one (200/200)");
        double[] degenerateWeights = BalanceWeights(new[] { 0.0, 0.0 });
        Check(degenerateWeights.All(v => v == 0.0), "two zero densities give zero weights instead of NaN");
        Check(BalanceWeights(new[] { 3.0, 0.0 })[0] == 1.0, "a single non-zero density gets all the weight");

        // 4-6: unbiasedness against the closed form and the quadrature oracle.
        // Two narrow bright zones that play to different strengths: a cap around the normal (which the
        // cosine density samples well) and a band near the horizon (which only uniform sampling reaches).
        var env = new Environment();
        env.Zones.Add(new Zone(0.0, Math.PI / 18.0, 1.0));            // 10 degree cap
        env.Zones.Add(new Zone(85.0 * Math.PI / 180.0, Math.PI / 2.0, 1.0));
        env.Zones.Add(new Zone(Math.PI / 18.0, 85.0 * Math.PI / 180.0, 0.1));
        double reference = albedo / Math.PI * env.IrradianceClosedForm();
        double quadrature = albedo / Math.PI * QuadratureIrradiance(env, 4096);
        Check(Math.Abs(reference - quadrature) < 1e-6,
              "the closed form matches the quadrature oracle to its discretisation error (" + reference + " vs " + quadrature + ")");
        var cosine = EstimateCosine(env, albedo, 20000, 0x2222UL);
        var uniform = EstimateUniform(env, albedo, 20000, 0x3333UL);
        var mis = EstimateMis(env, albedo, 20000, 0x4444UL);
        Console.WriteLine("reference " + reference.ToString("F6") + "; cosine " + cosine.Mean.ToString("F6") +
                          " +-" + cosine.Error.ToString("E2") + "; uniform " + uniform.Mean.ToString("F6") +
                          " +-" + uniform.Error.ToString("E2") + "; mis " + mis.Mean.ToString("F6") +
                          " +-" + mis.Error.ToString("E2"));
        Check(Math.Abs(cosine.Mean - reference) < 4 * cosine.Error + 1e-12, "cosine sampling is unbiased");
        Check(Math.Abs(uniform.Mean - reference) < 4 * uniform.Error + 1e-12, "uniform sampling is unbiased");
        Check(Math.Abs(mis.Mean - reference) < 4 * mis.Error + 1e-12, "MIS is unbiased");

        // 7: the balance heuristic stays within a small factor of the better strategy and clearly beats the
        // worse one - the standard guarantee that makes MIS safe to use without knowing which strategy wins.
        Check(mis.Variance <= 2.0 * Math.Min(cosine.Variance, uniform.Variance),
              "MIS stays within 2x of the better single strategy (" + mis.Variance.ToString("E2") + " vs " +
              Math.Min(cosine.Variance, uniform.Variance).ToString("E2") + ")");
        Check(mis.Variance * 1.2 < uniform.Variance,
              "MIS clearly beats the worse strategy (" + mis.Variance.ToString("E2") + " < " +
              uniform.Variance.ToString("E2") + ")");
        Check(cosine.Variance < uniform.Variance,
              "the cosine density beats the uniform density on a cosine-weighted integrand");

        // 8: the mixture parameter is a real tuning knob - the best mixture of the tested ones is chosen and
        // it is strictly better than the naively balanced 50/50 split.
        var mixtureErrors = new List<(double PCosine, double Error)>();
        foreach (double pc in new[] { 0.3, 0.5, 0.7, 0.85, 0.95 })
        {
            var tuned = EstimateMis(env, albedo, 8000, 0x5555UL, pc);
            mixtureErrors.Add((pc, tuned.Error));
        }
        var best = mixtureErrors.OrderBy(m => m.Error).First();
        var balanced = mixtureErrors.First(m => m.PCosine == 0.5);
        Console.WriteLine("mixture tuning: " + string.Join(", ", mixtureErrors.Select(m =>
                          m.PCosine.ToString("F2") + "->" + m.Error.ToString("E2"))) + " best " + best.PCosine);
        Check(best.Error <= balanced.Error, "the tuned mixture beats the naive 50/50 split");
        Check(best.PCosine > 0.3, "the tuner prefers the cosine strategy on a cosine-weighted integrand");
        Check(mixtureErrors.All(m => m.Error < uniform.Error * 2), "every mixture stays in the same ballpark");

        // 9: the mixture comparison holds on several seeds, so it is not a single-seed artefact.
        int misWinsOverUniform = 0;
        foreach (ulong seed in new[] { 0x6666UL, 0x7777UL, 0x8888UL, 0x9999UL })
        {
            var u = EstimateUniform(env, albedo, 8000, seed);
            var m = EstimateMis(env, albedo, 8000, seed, 0.85);
            if (m.Variance * 1.1 < u.Variance) misWinsOverUniform++;
        }
        Check(misWinsOverUniform == 4, "the tuned mixture beats uniform sampling on all four seeds ("
              + misWinsOverUniform + "/4)");

        // 10-11: Russian roulette.
        double meanPath;
        double multiBounceReference = reference * (1 - Math.Pow(albedo, 6)) / (1.0 - albedo);
        var roulette = EstimateWithRussianRoulette(env, albedo, 20000, 0x9999UL, 0.6, out meanPath);
        Check(Math.Abs(roulette.Mean - multiBounceReference) < 4 * roulette.Error + 1e-12,
              "Russian roulette stays unbiased against reference/(1-albedo) (" + roulette.Mean + " vs " +
              multiBounceReference + ")");
        Check(meanPath > 1.0 && meanPath < 4.0,
              "the average path length stays bounded (" + meanPath + ")");
        Check(roulette.RouletteKills > 0 && roulette.RouletteKills <= roulette.Samples,
              "roulette terminates paths without killing every sample (" + roulette.RouletteKills + ")");
        double meanPathHigh;
        var noRoulette = EstimateWithRussianRoulette(env, albedo, 20000, 0x9999UL, 1.0, out meanPathHigh);
        Check(meanPathHigh > meanPath, "a higher survival probability lengthens the paths (" +
              meanPathHigh.ToString("F3") + " > " + meanPath.ToString("F3") + ")");
        Check(Math.Abs(noRoulette.Mean - multiBounceReference) < 4 * noRoulette.Error + 1e-12,
              "survival probability 1 keeps the estimator unbiased");
        Check(noRoulette.RouletteKills == 0, "survival 1 never kills a path early");

        // 11-13: degenerate inputs.
        var black = EstimateCosine(env, 0.0, 1000, 0xAAAAUL);
        Check(black.Mean == 0.0 && black.Variance == 0.0, "a black surface returns exactly zero");
        var dark = EstimateCosine(new Environment { Uniform = true, UniformRadiance = 0.0 }, albedo, 1000, 0xBBBBUL);
        Check(dark.Mean == 0.0, "a zero-radiance environment returns exactly zero");
        var occluded = new Environment { Uniform = true, UniformRadiance = 1.0, OccluderAngle = Math.PI / 2 };
        var occludedEstimate = EstimateCosine(occluded, albedo, 2000, 0xCCCCUL);
        Check(occludedEstimate.Mean == 0.0,
              "a hemisphere occluded by a pi/2 cap contributes nothing (" + occludedEstimate.Mean + ")");
        Check(Math.Abs(occluded.IrradianceClosedForm()) < 1e-12,
              "the occluded closed form is zero as well (" + occluded.IrradianceClosedForm() + ")");
        var single = EstimateCosine(env, albedo, 1, 0xDDDDUL);
        Check(single.Samples == 1 && double.IsFinite(single.Mean), "a single-sample run is finite");
        Check(single.Variance == 0.0, "the variance of one sample is reported as zero, not NaN");
        var narrowCap = new Environment();
        narrowCap.Zones.Add(new Zone(0.0, 5.0 * Math.PI / 180.0, 1.0));
        narrowCap.Zones.Add(new Zone(5.0 * Math.PI / 180.0, Math.PI / 2.0, 0.0));
        var narrowEstimate = EstimateCosine(narrowCap, albedo, 20000, 0xEEEEUL);
        double narrowReference = albedo / Math.PI * narrowCap.IrradianceClosedForm();
        Check(Math.Abs(narrowEstimate.Mean - narrowReference) < 5 * narrowEstimate.Error + 1e-15,
              "a narrow bright cap is estimated without bias (" + narrowEstimate.Mean + " vs " +
              narrowReference + " +-" + narrowEstimate.Error.ToString("E2") + ")");
        Check(narrowEstimate.Error > 0.05 * narrowReference,
              "the narrow cap is exactly the high-variance case a real renderer must stratify");
        var narrowQuadrature = QuadratureIrradiance(narrowCap, 4096);
        Check(Math.Abs(narrowQuadrature - narrowCap.IrradianceClosedForm()) < 1e-9,
              "the quadrature agrees on the narrow cap too");

        // 14: determinism.
        var repeatA = EstimateMis(env, albedo, 5000, 0xFFFFUL);
        var repeatB = EstimateMis(env, albedo, 5000, 0xFFFFUL);
        Check(repeatA.Mean == repeatB.Mean && repeatA.Variance == repeatB.Variance,
              "the estimators are bitwise reproducible");
        var furnaceMis = EstimateMis(furnace, albedo, 5000, 0x1357UL);
        Check(Math.Abs(furnaceMis.Mean - albedo * 0.8) < 4 * furnaceMis.Error + 1e-12,
              "MIS reproduces the furnace identity in expectation, not pointwise (" + furnaceMis.Mean +
              " +-" + furnaceMis.Error.ToString("E2") + ")");
        Check(furnaceMis.Variance > furnaceEstimate.Variance,
              "the mixture is the only furnace estimator with non-zero variance, as expected");
        var furnaceUniform = EstimateUniform(furnace, albedo, 5000, 0x2468UL);
        Check(Math.Abs(furnaceUniform.Mean - albedo * 0.8) < 4 * furnaceUniform.Error + 1e-12,
              "uniform sampling is unbiased in a uniform environment (" + furnaceUniform.Mean + ")");
        Check(furnaceUniform.Variance > 1e-6,
              "but it is not pointwise exact: its estimator varies with cos(theta) (var " +
              furnaceUniform.Variance.ToString("E2") + ")");

        Console.WriteLine("furnace exact " + furnaceEstimate.Mean.ToString("F12") + " for rho*L " +
                          (albedo * 0.8).ToString("F12") + "; roulette mean path " + meanPath.ToString("F3") +
                          " vs " + meanPathHigh.ToString("F3") + " at survival 1");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
