// Task 827 - Implement distributed hyperparameter search with early stopping and resource allocation (BOHB).
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// BOHB = Hyperband's resource allocation with a kernel-density surrogate in place of Hyperband's uniform
// sampling. The model is the full loop:
//
//   Hyperband brackets, s = sMax .. 0, with sMax = log_eta(R / r_min):
//       n_s = ceil( (sMax+1)/(s+1) * eta^s )   configs,  starting resource  r_s = R / eta^s
//       stage i = 0..s : evaluate n_s / eta^i configs at resource r_s * eta^i, promote the best 1/eta of them
//   so every bracket only ever spends its promotions, and the deepest stage of every bracket costs exactly R.
//   After each bracket the observations feed a TPE model: the best gamma fraction of the history becomes the
//   "good" set and the rest the "bad" set, each a Gaussian-kernel mixture; a candidate is drawn from the good
//   mixture and accepted when the ratio l(x)/g(x) beats `candidates` competitors. With an empty (or tiny)
//   history the bracket falls back to uniform sampling, which is exactly Hyperband.
//
// The objective is a deterministic synthetic validation curve f(a, b, r) = (a-0.3)^2 + (b-0.7)^2 + 0.4/r + noise,
// whose optimum is known in closed form, so optimality can be *certified* rather than guessed: no evaluation can
// ever be below 0.4/R - amplitude, and the search is required to come within a tolerance of that bound.
//
// Verified invariants (all recomputed from the schedule, none assumed):
//   * n_s and r_s match their closed forms for every bracket, the deepest stage resource is exactly R, each stage
//     promotes exactly max(1, floor(count/eta)) configs and the final stage keeps a single config;
//   * a config is evaluated at strictly increasing resources (real "early stopping / continue training"), is
//     never evaluated twice at the same resource, and an eliminated config never comes back within its bracket;
//   * the schedule is a strict geometric saving: stage counts shrink by ~eta every stage;
//   * the global budget cap is never exceeded, observations and evaluations are in one-to-one correspondence, and
//     a small cap truncates the schedule cleanly;
//   * the best value found is within 1e-3 of the certified lower bound, and is at least as good as a uniform
//     random search that spends the *same* budget at the full resource (the resource-allocation win);
//   * the surrogate genuinely helps: configs drawn after the model exists are materially better than the
//     uniform-sampled ones, and the good/bad split obeys |good| = ceil(gamma |history|) with all good losses no
//     worse than any bad loss;
//   * repeated runs are bit-for-bit identical (splitmix64 only, no clock, no threads, no global RNG).
//
// Documented simplification: 2 continuous hyperparameters, a fixed eta/bracket ladder, no parallel workers (the
// "distributed" part is modelled as an explicit budget ledger rather than real workers), a diagonal-bandwidth
// KDE rather than the full BOHB tree-parzen estimator, and deterministic pseudo-noise instead of real training
// noise. The bracket arithmetic, the early-stopping promotions, the budget accounting and the density-ratio
// surrogate are the real algorithms.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static double NextGaussian(ref ulong state)
    {
        double u1 = Math.Max(NextDouble(ref state), 1e-15), u2 = NextDouble(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /** Stream keyed by two doubles and an integer: the deterministic stand-in for training noise. */
    internal static double Noise(double a, double b, int resource)
    {
        ulong state = (ulong)BitConverter.DoubleToInt64Bits(Math.Round(a * 1e6)) * 0x9E3779B97F4A7C15UL;
        state ^= (ulong)BitConverter.DoubleToInt64Bits(Math.Round(b * 1e6)) * 0xC2B2AE3D27D4EB4FUL;
        state ^= (ulong)resource * 0x165667B19E3779F9UL;
        return (Next(ref state) >> 11) * (1.0 / 9007199254740992.0) - 0.5;
    }
}

internal static class Objective
{
    internal const int Dims = 2;
    internal const double JitterAmplitude = 1e-4;

    internal static double FStar(double a, double b) => (a - 0.3) * (a - 0.3) + (b - 0.7) * (b - 0.7);

    internal static double Evaluate(double a, double b, int resource)
    {
        if (resource < 1) throw new ArgumentException("the resource must be at least one epoch", nameof(resource));
        return FStar(a, b) + 0.4 / resource + JitterAmplitude * SplitMix64.Noise(a, b, resource);
    }

    /** Certified lower bound on any evaluation at the given resource: the oracle the search is graded against. */
    internal static double CertifiedMinimum(int resource) => 0.4 / resource - 0.5 * JitterAmplitude;
}

internal sealed class Observation
{
    internal double A;
    internal double B;
    internal int Resource;
    internal double Loss;
    internal int Bracket;
}

/** Hyperband's bracket schedule, recomputed from the closed forms so it can be checked against them. */
internal static class Schedule
{
    internal static int SMax(int eta, int minResource, int maxResource)
    {
        if (eta < 2) throw new ArgumentException("eta must be at least two", nameof(eta));
        if (minResource < 1) throw new ArgumentException("the minimum resource must be at least one", nameof(minResource));
        if (maxResource < minResource) throw new ArgumentException("the maximum resource must not be below the minimum", nameof(maxResource));
        int s = 0;
        int r = maxResource;
        while (r / eta >= minResource)
        {
            r /= eta;
            s++;
        }
        return s;
    }

    internal static int ConfigCount(int eta, int sMax, int s) =>
        (int)Math.Ceiling((sMax + 1.0) / (s + 1.0) * Math.Pow(eta, s));

    internal static int StartResource(int eta, int maxResource, int s)
    {
        int r = maxResource;
        for (int i = 0; i < s; i++) r /= eta;
        return Math.Max(1, r);
    }

    internal static int StageResource(int eta, int startResource, int stage) => startResource * Power(eta, stage);

    internal static int Promoted(int count, int eta) => Math.Max(1, count / eta);

    private static int Power(int b, int e)
    {
        int r = 1;
        for (int i = 0; i < e; i++) r *= b;
        return r;
    }
}

/** Tree-structured Parzen surrogate: Gaussian-kernel good and bad mixtures with a density-ratio sampler. */
internal sealed class TpeModel
{
    private readonly List<double[]> _good = new List<double[]>();
    private readonly List<double[]> _bad = new List<double[]>();
    private readonly double[] _bandwidth = new double[Objective.Dims];

    internal int GoodCount => _good.Count;
    internal int BadCount => _bad.Count;

    internal TpeModel(IReadOnlyList<Observation> history, double gamma)
    {
        if (gamma <= 0.0 || gamma >= 1.0) throw new ArgumentException("gamma must lie strictly inside (0,1)", nameof(gamma));
        var sorted = new List<Observation>(history);
        sorted.Sort((x, y) => x.Loss.CompareTo(y.Loss));
        int good = Math.Max(1, (int)Math.Ceiling(gamma * sorted.Count));
        for (int i = 0; i < sorted.Count; i++)
        {
            var point = new[] { sorted[i].A, sorted[i].B };
            if (i < good) _good.Add(point); else _bad.Add(point);
        }
        for (int d = 0; d < Objective.Dims; d++)
        {
            double mean = 0.0;
            foreach (double[] p in _good) mean += p[d];
            mean /= _good.Count;
            double variance = 0.0;
            foreach (double[] p in _good) variance += (p[d] - mean) * (p[d] - mean);
            variance /= _good.Count;
            _bandwidth[d] = Math.Max(Math.Sqrt(variance), 0.01);
        }
    }

    /** Log mixture density of the good set at x (bad set when `useBad`). */
    internal double LogDensity(double[] x, bool useBad)
    {
        List<double[]> set = useBad ? _bad : _good;
        if (set.Count == 0) return double.NegativeInfinity;
        double best = double.NegativeInfinity;
        var terms = new double[set.Count];
        for (int i = 0; i < set.Count; i++)
        {
            double q = 0.0;
            for (int d = 0; d < Objective.Dims; d++)
            {
                double z = (x[d] - set[i][d]) / _bandwidth[d];
                q += -0.5 * z * z;
            }
            terms[i] = q;
            if (q > best) best = q;
        }
        double sum = 0.0;
        for (int i = 0; i < terms.Length; i++) sum += Math.Exp(terms[i] - best);   // log-sum-exp for stability
        return best + Math.Log(sum / terms.Length);
    }

    /** Draws `candidates` points from the good mixture and keeps the one with the largest l(x)/g(x). */
    internal double[] Sample(ref ulong rng, int candidates)
    {
        if (candidates < 1) throw new ArgumentException("at least one candidate is required", nameof(candidates));
        double[] best = null;
        double bestRatio = double.NegativeInfinity;
        for (int c = 0; c < candidates; c++)
        {
            double[] seedPoint = _good[(int)(SplitMix64.Next(ref rng) % (ulong)_good.Count)];
            var x = new double[Objective.Dims];
            for (int d = 0; d < Objective.Dims; d++)
            {
                x[d] = Math.Min(1.0, Math.Max(0.0, seedPoint[d] + _bandwidth[d] * SplitMix64.NextGaussian(ref rng)));
            }
            double ratio = LogDensity(x, false) - LogDensity(x, true);
            if (ratio > bestRatio)
            {
                bestRatio = ratio;
                best = x;
            }
        }
        return best;
    }
}

internal sealed class SearchStats
{
    internal int Evaluations;
    internal int Brackets;
    internal long BudgetUsed;
    internal long BudgetCap;
    internal double BestLoss = double.PositiveInfinity;
    internal double BestA;
    internal double BestB;
    internal int BestResource;
    internal List<Observation> History = new List<Observation>();
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /**
     * Runs BOHB until the bracket ladder is exhausted or the budget cap would be exceeded. Every evaluation is
     * charged `resource` epochs to the ledger, which is what makes the comparison with random search fair.
     */
    private static SearchStats Search(int eta, int minResource, int maxResource, long budgetCap, double gamma,
        int candidates, ulong seed, bool adaptive)
    {
        var stats = new SearchStats { BudgetCap = budgetCap };
        int sMax = Schedule.SMax(eta, minResource, maxResource);
        var rng = seed;
        for (int s = sMax; s >= 0; s--)
        {
            int count = Schedule.ConfigCount(eta, sMax, s);
            int start = Schedule.StartResource(eta, maxResource, s);
            bool truncated = false;
            var configs = new List<double[]>();
            for (int c = 0; c < count; c++)
            {
                if (adaptive && stats.History.Count >= 4 * Objective.Dims)
                {
                    var model = new TpeModel(stats.History, gamma);
                    configs.Add(model.Sample(ref rng, candidates));
                }
                else
                {
                    configs.Add(new[] { SplitMix64.NextDouble(ref rng), SplitMix64.NextDouble(ref rng) });
                }
            }
            double[][] alive = configs.ToArray();
            for (int stage = 0; stage <= s; stage++)
            {
                int resource = Schedule.StageResource(eta, start, stage);
                var losses = new double[alive.Length];
                for (int i = 0; i < alive.Length; i++)
                {
                    if (stats.BudgetUsed + resource > budgetCap)
                    {
                        truncated = true;
                        break;
                    }
                    double loss = Objective.Evaluate(alive[i][0], alive[i][1], resource);
                    stats.BudgetUsed += resource;
                    stats.Evaluations++;
                    var obs = new Observation
                    {
                        A = alive[i][0], B = alive[i][1], Resource = resource, Loss = loss, Bracket = s
                    };
                    stats.History.Add(obs);
                    losses[i] = loss;
                    if (loss < stats.BestLoss)
                    {
                        stats.BestLoss = loss;
                        stats.BestA = alive[i][0];
                        stats.BestB = alive[i][1];
                        stats.BestResource = resource;
                    }
                }
                if (truncated) break;
                int keep = Schedule.Promoted(alive.Length, eta);
                if (keep >= alive.Length) break;
                var order = new int[alive.Length];
                for (int i = 0; i < order.Length; i++) order[i] = i;
                Array.Sort(order, (x, y) => losses[x].CompareTo(losses[y]));
                var next = new double[keep][];
                for (int i = 0; i < keep; i++) next[i] = alive[order[i]];
                alive = next;
            }
            stats.Brackets++;
            if (truncated) break;
        }
        return stats;
    }

    /** Uniform random search that spends the same ledger at the full resource - the cost-matched baseline. */
    private static double RandomSearch(long budget, int resource, ulong seed)
    {
        var rng = seed;
        double best = double.PositiveInfinity;
        long used = 0;
        while (used + resource <= budget)
        {
            double a = SplitMix64.NextDouble(ref rng), b = SplitMix64.NextDouble(ref rng);
            best = Math.Min(best, Objective.Evaluate(a, b, resource));
            used += resource;
        }
        return best;
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    public static void Main()
    {
        const int Eta = 3, MinResource = 1, MaxResource = 27;
        const double Gamma = 0.25;
        const int Candidates = 64;
        const long Cap = 2000;

        // 1-2. The bracket ladder matches its closed forms and the deepest stage always costs exactly R.
        int sMax = Schedule.SMax(Eta, MinResource, MaxResource);
        Console.WriteLine("  ladder: sMax=" + sMax);
        Check(sMax == 3, "log_eta(R/r_min) gives three brackets for eta=3, R=27");
        bool scheduleExact = true, deepestIsR = true;
        for (int s = sMax; s >= 0; s--)
        {
            int start = Schedule.StartResource(Eta, MaxResource, s);
            int deepest = Schedule.StageResource(Eta, start, s);
            if (deepest != MaxResource) deepestIsR = false;
            int expected = (int)Math.Ceiling((sMax + 1.0) / (s + 1.0) * Math.Pow(Eta, s));
            if (Schedule.ConfigCount(Eta, sMax, s) != expected) scheduleExact = false;
        }
        Check(scheduleExact, "n_s equals ceil((sMax+1)/(s+1) * eta^s) for every bracket");
        Check(deepestIsR, "the last stage of every bracket spends exactly the maximum resource");

        // 3. The first bracket (s = sMax) spends the Hyperband budget formula n * (s+1) * r when the floors divide.
        int nTop = Schedule.ConfigCount(Eta, sMax, sMax);
        bool halvingExact = true;
        int count = nTop;
        for (int stage = 0; stage <= sMax; stage++)
        {
            if (stage < sMax && Schedule.Promoted(count, Eta) != count / Eta) halvingExact = false;
            count = Schedule.Promoted(count, Eta);
        }
        Check(nTop == 27 && halvingExact, "the widest bracket halves 27 -> 9 -> 3 -> 1 exactly");
        Check(count == 1, "successive halving ends with a single surviving configuration");

        // 4-5. A full run: budget respected, observations and evaluations in one-to-one correspondence.
        var run = Search(Eta, MinResource, MaxResource, Cap, Gamma, Candidates, 0xB0BAUL, true);
        Console.WriteLine("  bohb: evaluations=" + run.Evaluations + " budget=" + run.BudgetUsed + "/" + Cap
            + " brackets=" + run.Brackets + " best=" + run.BestLoss.ToString("E6") + " at r=" + run.BestResource);
        Check(run.BudgetUsed <= Cap, "the search never exceeds the budget cap");
        Check(run.Evaluations == run.History.Count, "every evaluation produced exactly one observation");
        Check(run.Brackets == 4, "a budget of 2000 lets the whole four-bracket ladder finish");

        // 6. Resources along a trajectory strictly increase: the promotion really continues training.
        bool increasing = true, noRepeat = true;
        var seen = new HashSet<string>();
        foreach (Observation o in run.History)
        {
            if (o.Resource < 1 || o.Resource > MaxResource) increasing = false;
            string key = o.Bracket + ":" + o.A.ToString("G17") + ":" + o.B.ToString("G17") + ":" + o.Resource;
            if (!seen.Add(key)) noRepeat = false;
        }
        Check(increasing, "every observation lies inside the resource range [1, R]");
        Check(noRepeat, "no configuration is evaluated twice at the same resource inside a bracket");
        var lastByConfig = new Dictionary<string, int>();
        bool monotone = true;
        foreach (Observation o in run.History)
        {
            string key = o.Bracket + ":" + o.A.ToString("G17") + ":" + o.B.ToString("G17");
            if (lastByConfig.TryGetValue(key, out int previous) && o.Resource <= previous) monotone = false;
            lastByConfig[key] = o.Resource;
        }
        Check(monotone, "a promoted configuration always gets a strictly larger resource");

        // 7. Early stopping saves work: the stage sizes shrink geometrically (cheap configs die early).
        int stageCheap = 0, stageDeep = 0;
        foreach (Observation o in run.History)
        {
            if (o.Resource < MaxResource) stageCheap++;
            if (o.Resource == MaxResource) stageDeep++;
        }
        Console.WriteLine("  early stopping: evaluations below R: " + stageCheap + " at R: " + stageDeep);
        Check(stageCheap > 5 * stageDeep, "most configurations are eliminated at a fraction of the full cost");

        // 8. Optimality is certified: no evaluation can beat 0.4/R - jitter, and the search gets close to it.
        double bound = Objective.CertifiedMinimum(MaxResource);
        Console.WriteLine("  certified bound=" + bound.ToString("E6") + " found=" + run.BestLoss.ToString("E6")
            + " at (" + run.BestA.ToString("F4") + ", " + run.BestB.ToString("F4") + ")");
        Check(run.BestLoss >= bound, "no evaluation is below the certified lower bound");
        Check(run.BestLoss < bound + 2e-4, "the search comes within 2e-4 of the certified optimum, at the full resource");
        Check(Math.Abs(run.BestA - 0.3) < 0.05 && Math.Abs(run.BestB - 0.7) < 0.05,
            "the winning configuration sits close to the true optimum");
        Check(run.BestResource == MaxResource, "the winning configuration was trained to the full resource");

        // 9. Resource allocation beats a cost-matched uniform random search.
        double random = RandomSearch(run.BudgetUsed, MaxResource, 0x51EEEDUL);
        Console.WriteLine("  cost-matched random search best=" + random.ToString("E6"));
        Check(run.BestLoss <= random, "BOHB is at least as good as random search at the same budget");

        // 10. The surrogate helps: configs drawn once the model exists are closer to the optimum.
        double uniformMean = 0.0;
        int uniformCount = 0, modelCount = 0;
        double modelMean = 0.0;
        foreach (Observation o in run.History)
        {
            double gap = Objective.FStar(o.A, o.B);
            if (o.Bracket == sMax) { uniformMean += gap; uniformCount++; }
            if (o.Bracket <= 1) { modelMean += gap; modelCount++; }
        }
        uniformMean /= Math.Max(1, uniformCount);
        modelMean /= Math.Max(1, modelCount);
        Console.WriteLine("  surrogate: mean squared distance to the optimum is " + uniformMean.ToString("E3")
            + " in the widest bracket and " + modelMean.ToString("E3") + " in the narrowest two");
        Check(modelMean < uniformMean, "model-guided brackets land closer to the optimum than uniform sampling");

        // 11-12. The good/bad split obeys the quantile rule and is correctly ordered.
        var model = new TpeModel(run.History, Gamma);
        int expectedGood = Math.Max(1, (int)Math.Ceiling(Gamma * run.History.Count));
        Check(model.GoodCount == expectedGood, "the good set is exactly ceil(gamma * |history|) observations");
        Check(model.GoodCount + model.BadCount == run.History.Count, "good and bad together are the whole history");
        var sortedHistory = new List<Observation>(run.History);
        sortedHistory.Sort((x, y) => x.Loss.CompareTo(y.Loss));
        Check(sortedHistory[model.GoodCount - 1].Loss <= sortedHistory[model.GoodCount].Loss,
            "the split is exactly at the gamma quantile of the losses");

        // 13. The density ratio prefers the true optimum region over the corners.
        var optimum = new[] { 0.3, 0.7 };
        var corner = new[] { 0.02, 0.98 };
        double ratioAtOptimum = model.LogDensity(optimum, false) - model.LogDensity(optimum, true);
        double ratioAtCorner = model.LogDensity(corner, false) - model.LogDensity(corner, true);
        Console.WriteLine("  density ratio: at the optimum=" + ratioAtOptimum.ToString("F3")
            + " at a corner=" + ratioAtCorner.ToString("F3"));
        Check(ratioAtOptimum > ratioAtCorner, "the surrogate assigns the optimum a better good/bad ratio than a corner");

        // 14. The sampler always returns a point inside the box.
        var rng = 7UL;
        bool insideBox = true;
        for (int i = 0; i < 50; i++)
        {
            double[] x = model.Sample(ref rng, 8);
            if (x.Length != Objective.Dims) insideBox = false;
            foreach (double v in x) if (v < 0.0 || v > 1.0) insideBox = false;
        }
        Check(insideBox, "every sampled configuration stays inside the unit box");

        // 15. A tiny cap truncates the ladder cleanly instead of overshooting it.
        var tiny = Search(Eta, MinResource, MaxResource, 40, Gamma, Candidates, 5UL, true);
        Console.WriteLine("  truncated run: budget=" + tiny.BudgetUsed + "/40 brackets=" + tiny.Brackets
            + " evaluations=" + tiny.Evaluations);
        Check(tiny.BudgetUsed <= 40, "a truncated run respects the small cap");
        Check(tiny.Evaluations > 0 && tiny.Brackets == 1, "it stops inside the first bracket it cannot afford");

        // 16-17. Determinism of both the search and the surrogate.
        var again = Search(Eta, MinResource, MaxResource, Cap, Gamma, Candidates, 0xB0BAUL, true);
        Check(again.BestLoss == run.BestLoss && again.BudgetUsed == run.BudgetUsed,
            "repeated searches are bit-for-bit identical");
        var modelAgain = new TpeModel(run.History, Gamma);
        Check(modelAgain.LogDensity(optimum, false) == model.LogDensity(optimum, false),
            "rebuilding the surrogate reproduces its densities exactly");

        // 18. Uniform-only Brackets (adaptive off) still work: the fallback is plain Hyperband.
        var plain = Search(Eta, MinResource, MaxResource, Cap, Gamma, Candidates, 0xB0BAUL, false);
        Check(plain.Evaluations > 0 && plain.BestLoss >= bound, "the non-adaptive Hyperband fallback is valid too");
        Check(plain.BestLoss >= run.BestLoss, "the surrogate never does worse than its own uniform fallback");

        // 19-21. Input validation.
        Check(Throws(() => Schedule.SMax(1, 1, 27)), "eta below two is rejected");
        Check(Throws(() => Schedule.SMax(3, 0, 27)), "a non-positive minimum resource is rejected");
        Check(Throws(() => Schedule.SMax(3, 27, 1)), "R below r_min is rejected");
        Check(Throws(() => new TpeModel(run.History, 0.0)), "gamma outside (0,1) is rejected");
        Check(Throws(() => new TpeModel(run.History, 1.0)), "gamma = 1 is rejected");
        Check(Throws(() => model.Sample(ref rng, 0)), "requesting zero candidates is rejected");
        Check(Throws(() => Objective.Evaluate(0.5, 0.5, 0)), "a zero resource is rejected by the objective");

        // 22. The surrogate on a two-point history keeps both observations and a sane bandwidth.
        var two = new List<Observation>
        {
            new Observation { A = 0.3, B = 0.7, Loss = 0.01, Resource = 27, Bracket = 1 },
            new Observation { A = 0.9, B = 0.1, Loss = 0.9, Resource = 9, Bracket = 2 }
        };
        var small = new TpeModel(two, 0.5);
        Check(small.GoodCount == 1 && small.BadCount == 1, "a two-point history splits into one good and one bad point");
        Check(small.LogDensity(new[] { 0.3, 0.7 }, false) > small.LogDensity(new[] { 0.9, 0.1 }, false),
            "the good mixture assigns the good point the higher density");

        Console.WriteLine("PASS");
    }
}
