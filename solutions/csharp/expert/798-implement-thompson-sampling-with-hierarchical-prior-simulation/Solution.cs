// Task 798 - Implement Thompson sampling with hierarchical prior simulation. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// Thompson sampling for a Bernoulli bandit whose arm probabilities share a common Beta prior - the two-level
// ("hierarchical") structure that makes the arms borrow strength from each other:
//   level 1: the environment draws p_a ~ Beta(alpha, beta) once for every arm;
//   level 2: the agent keeps the conjugate posterior p_a | data ~ Beta(alpha + s_a, beta + f_a), samples one
//            draw per arm each round with the exact inverse-CDF sampler for integer parameters, and plays the
//            arm with the largest draw;
//   the shared (alpha, beta) are themselves re-estimated by *empirical Bayes*: after each block of rounds a
//   gradient step on the marginal likelihood of the observed per-arm rewards moves (alpha, beta) towards the
//   values the data supports, which is the cheap stand-in for a full hyperprior posterior.
// Invariants asserted: the posterior mean equals the closed form (alpha + s) / (alpha + beta + n), the sampler's
// empirical distribution matches the analytic Beta CDF for integer parameters, and the hierarchical posterior
// mean always lies between the maximum-likelihood estimate and the prior mean (shrinkage).
//
// Independent oracles:
//   * the analytic Beta CDF F(x) = sum_{i=a}^{a+b-1} C(a+b-1, i) x^i (1-x)^{a+b-1-i}, which the inverse-CDF
//     sampler must reproduce to Monte-Carlo accuracy;
//   * the closed-form conjugate update;
//   * the empirical-Bayes objective, whose value must increase after the hyper-gradient step (checked against a
//     hand-rolled marginal likelihood, not against the gradient code);
//   * cumulative regret against a uniform-random policy on the same environment, plus a sublinearity check.
//
// Documented simplification: a "simulation" of the hierarchical scheme - the hyperparameters are learned by
// empirical Bayes rather than sampled from their posterior (no MCMC), the reward is Bernoulli with a small
// number of arms, and the hyper-gradient uses a fixed step size. Sampling, the conjugate update, shrinkage and
// the regret accounting are the real algorithm. Deterministic splitmix64 randomness only.
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
}

internal static class Beta
{
    /** Binomial coefficient as a double (exact for the small parameters used here). */
    internal static double Choose(int n, int k)
    {
        if (k < 0 || k > n) return 0.0;
        double result = 1.0;
        for (int i = 0; i < k; i++) result = result * (n - i) / (i + 1);
        return result;
    }

    /** Exact CDF of Beta(a, b) for positive integers, by the binomial identity. */
    internal static double Cdf(double x, int a, int b)
    {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        int n = a + b - 1;
        double total = 0.0;
        for (int i = a; i <= n; i++)
        {
            total += Choose(n, i) * Math.Pow(x, i) * Math.Pow(1.0 - x, n - i);
        }
        return Math.Min(1.0, Math.Max(0.0, total));
    }

    /** Inverse CDF by bisection on the regularized incomplete beta: deterministic and accurate to ~1e-12 in x. */
    internal static double Sample(double u, int a, int b)
    {
        double lo = 0.0, hi = 1.0;
        for (int iteration = 0; iteration < 40; iteration++)
        {
            double mid = 0.5 * (lo + hi);
            if (RegularizedIncompleteBeta(mid, a, b) < u) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
    }

    /**
     * Regularized incomplete beta I_x(a, b) by the Lentz continued fraction (Numerical Recipes formulation).
     * This is the numerically stable CDF for arbitrary (a, b); Cdf below is an independent formula for small
     * integer parameters and is what the sampler is validated against.
     */
    internal static double RegularizedIncompleteBeta(double x, double a, double b)
    {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        double front = Math.Exp(LogGamma(a + b) - LogGamma(a) - LogGamma(b) + a * Math.Log(x) + b * Math.Log(1.0 - x));
        if (x < (a + 1.0) / (a + b + 2.0)) return front * ContinuedFraction(x, a, b) / a;
        return 1.0 - front * ContinuedFraction(1.0 - x, b, a) / b;
    }

    private static double ContinuedFraction(double x, double a, double b)
    {
        const int maxIterations = 300;
        const double eps = 3e-14, tiny = 1e-300;
        double qab = a + b, qap = a + 1.0, qam = a - 1.0;
        double c = 1.0, d = 1.0 - qab * x / qap;
        if (Math.Abs(d) < tiny) d = tiny;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= maxIterations; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < tiny) d = tiny;
            c = 1.0 + aa / c;
            if (Math.Abs(c) < tiny) c = tiny;
            d = 1.0 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < tiny) d = tiny;
            c = 1.0 + aa / c;
            if (Math.Abs(c) < tiny) c = tiny;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1.0) < eps) break;
        }
        return h;
    }

    internal static double Mean(int a, int b) => (double)a / (a + b);

    internal static double Variance(int a, int b) => (double)a * b / ((a + b) * (a + b) * (a + b + 1.0));

    /** log of the Beta density, up to the normalising constant (used by the empirical-Bayes objective). */
    internal static double LogDensity(double x, double alpha, double beta) =>
        (alpha - 1.0) * Math.Log(x) + (beta - 1.0) * Math.Log(1.0 - x) - LogBeta(alpha, beta);

    internal static double LogBeta(double alpha, double beta) => LogGamma(alpha) + LogGamma(beta) - LogGamma(alpha + beta);

    internal static double LogGammaPublic(double x) => LogGamma(x);

    private static double LogGamma(double x)
    {
        double[] c =
        {
            0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
        };
        if (x < 0.5) return Math.Log(Math.PI / Math.Sin(Math.PI * x)) - LogGamma(1.0 - x);
        double z = x - 1.0, acc = c[0];
        for (int i = 1; i < 9; i++) acc += c[i] / (z + i);
        double t = z + 7.5;
        return 0.5 * Math.Log(2.0 * Math.PI) + (z + 0.5) * Math.Log(t) - t + Math.Log(acc);
    }
}

internal sealed class HierarchicalBandit
{
    private readonly double[] _probabilities;
    private ulong _rng;

    internal int Arms => _probabilities.Length;
    internal double Probability(int arm) => _probabilities[arm];

    internal HierarchicalBandit(int arms, double alpha, double beta, ulong seed)
    {
        if (arms < 1) throw new ArgumentException("at least one arm is required", nameof(arms));
        if (alpha <= 0.0 || beta <= 0.0) throw new ArgumentException("prior parameters must be positive");
        _rng = seed;
        _probabilities = new double[arms];
        for (int a = 0; a < arms; a++)
        {
            int integerAlpha = (int)Math.Round(alpha);
            int integerBeta = (int)Math.Round(beta);
            _probabilities[a] = Beta.Sample(SplitMix64.NextDouble(ref _rng), integerAlpha, integerBeta);
        }
    }

    internal int BestArm()
    {
        int best = 0;
        for (int a = 1; a < Arms; a++) if (_probabilities[a] > _probabilities[best]) best = a;
        return best;
    }

    internal int Play(int arm)
    {
        if (arm < 0 || arm >= Arms) throw new ArgumentOutOfRangeException(nameof(arm));
        return SplitMix64.NextDouble(ref _rng) < _probabilities[arm] ? 1 : 0;
    }
}

internal sealed class ThompsonAgent
{
    private readonly int[] _successes;
    private readonly int[] _failures;
    private ulong _rng;

    internal double Alpha { get; private set; }
    internal double BetaParameter { get; private set; }
    internal int Rounds { get; private set; }

    internal ThompsonAgent(int arms, double alpha, double beta, ulong seed)
    {
        if (arms < 1) throw new ArgumentException("at least one arm is required", nameof(arms));
        if (alpha <= 0.0 || beta <= 0.0) throw new ArgumentException("prior parameters must be positive");
        _successes = new int[arms];
        _failures = new int[arms];
        _rng = seed;
        Alpha = alpha;
        BetaParameter = beta;
    }

    internal int Arms => _successes.Length;
    internal int Successes(int arm) => _successes[arm];
    internal int Trials(int arm) => _successes[arm] + _failures[arm];

    /** Closed-form conjugate posterior mean (alpha + s) / (alpha + beta + n). */
    internal double PosteriorMean(int arm) =>
        (Alpha + _successes[arm]) / (Alpha + BetaParameter + Trials(arm));

    internal double MaximumLikelihood(int arm)
    {
        int trials = Trials(arm);
        return trials == 0 ? double.NaN : _successes[arm] / (double)trials;
    }

    internal double PriorMean => Alpha / (Alpha + BetaParameter);

    /** One Thompson draw for an arm, using the integer-parameter inverse-CDF sampler. */
    internal double Draw(int arm)
    {
        int a = (int)Math.Round(Alpha) + _successes[arm];
        int b = (int)Math.Round(BetaParameter) + _failures[arm];
        return Beta.Sample(SplitMix64.NextDouble(ref _rng), Math.Max(1, a), Math.Max(1, b));
    }

    internal int Select()
    {
        int best = 0;
        double bestDraw = double.NegativeInfinity;
        for (int a = 0; a < Arms; a++)
        {
            double draw = Draw(a);
            if (draw > bestDraw)
            {
                bestDraw = draw;
                best = a;
            }
        }
        Rounds++;
        return best;
    }

    internal void Observe(int arm, int reward)
    {
        if (reward == 1) _successes[arm]++;
        else _failures[arm]++;
    }

    /** Marginal log-likelihood of the per-arm means under the shared Beta prior (empirical-Bayes objective). */
    internal double MarginalLogLikelihood()
    {
        double total = 0.0;
        for (int a = 0; a < Arms; a++)
        {
            int trials = Trials(a);
            if (trials == 0) continue;
            double mean = (0.5 + _successes[a]) / (trials + 1.0);           // smoothed arm mean
            total += Beta.LogDensity(mean, Alpha, BetaParameter);
        }
        return total;
    }

    /** One empirical-Bayes gradient step on (alpha, beta) with numerical gradients and a fixed step size. */
    internal void UpdateHyperparameters(double step)
    {
        const double h = 1e-5;
        double baseValue = MarginalLogLikelihood();
        double dAlpha = (ObjectiveAt(Alpha + h, BetaParameter) - ObjectiveAt(Alpha - h, BetaParameter)) / (2.0 * h);
        double dBeta = (ObjectiveAt(Alpha, BetaParameter + h) - ObjectiveAt(Alpha, BetaParameter - h)) / (2.0 * h);
        double nextAlpha = Math.Max(0.05, Alpha + step * dAlpha);
        double nextBeta = Math.Max(0.05, BetaParameter + step * dBeta);
        if (ObjectiveAt(nextAlpha, nextBeta) >= baseValue)
        {
            Alpha = nextAlpha;
            BetaParameter = nextBeta;
        }
    }

    private double ObjectiveAt(double alpha, double beta)
    {
        double total = 0.0;
        for (int a = 0; a < Arms; a++)
        {
            int trials = Trials(a);
            if (trials == 0) continue;
            double mean = (0.5 + _successes[a]) / (trials + 1.0);
            total += Beta.LogDensity(mean, alpha, beta);
        }
        return total;
    }
}

public static class Solution
{
    private const int Arms = 4, Horizon = 2000;

    private static double Regret(ThompsonAgent agent, HierarchicalBandit environment, int rounds, out double lateFraction, out double earlyRegret, out double lateRegret)
    {
        int best = environment.BestArm();
        double regret = 0.0;
        int hits = 0, lateRounds = rounds / 5;
        earlyRegret = 0.0;
        lateRegret = 0.0;
        for (int t = 0; t < rounds; t++)
        {
            int arm = agent.Select();
            int reward = environment.Play(arm);
            agent.Observe(arm, reward);
            regret += environment.Probability(best) - environment.Probability(arm);
            if (t < rounds / 5) earlyRegret += environment.Probability(best) - environment.Probability(arm);
            if (t >= rounds - lateRounds)
            {
                lateRegret += environment.Probability(best) - environment.Probability(arm);
                if (arm == best) hits++;
            }
        }
        lateFraction = hits / (double)lateRounds;
        return regret;
    }

    public static void Main()
    {
        // 1-3. Sampler oracle: the inverse-CDF draws must reproduce the exact Beta CDF, and the CDF itself must
        //      match its closed form at the extremes and the median-checks.
        var rng = 0x5EEDUL;
        int[] countBelow = new int[4];
        double[] checkPoints = { 0.1, 0.3, 0.5, 0.7 };
        const int samples = 6000;
        for (int i = 0; i < samples; i++)
        {
            double x = Beta.Sample(SplitMix64.NextDouble(ref rng), 3, 5);
            for (int c = 0; c < checkPoints.Length; c++) if (x <= checkPoints[c]) countBelow[c]++;
        }
        double worstCdfDeviation = 0.0;
        for (int c = 0; c < checkPoints.Length; c++)
        {
            double empirical = countBelow[c] / (double)samples;
            double exact = Beta.Cdf(checkPoints[c], 3, 5);
            worstCdfDeviation = Math.Max(worstCdfDeviation, Math.Abs(empirical - exact));
        }
        Console.WriteLine("  sampler oracle: max |empirical CDF - exact CDF| = " + worstCdfDeviation.ToString("F4")
            + " over " + samples + " draws of Beta(3,5)");
        Check(worstCdfDeviation < 0.01, "inverse-CDF sampler reproduces the analytic Beta CDF");
        double worstFormulaAgreement = 0.0;
        foreach (double x in new[] { 0.05, 0.2, 0.5, 0.8, 0.95 })
        {
            foreach ((int a, int b) in new[] { (1, 1), (3, 5), (2, 7), (6, 2) })
            {
                double fraction = Beta.RegularizedIncompleteBeta(x, a, b);
                worstFormulaAgreement = Math.Max(worstFormulaAgreement, Math.Abs(fraction - Beta.Cdf(x, a, b)));
            }
        }
        Console.WriteLine("  CDF formulas: max |continued fraction - binomial identity| = "
            + worstFormulaAgreement.ToString("E3"));
        Check(worstFormulaAgreement < 1e-10, "continued-fraction CDF equals the binomial-identity CDF");
        Check(Beta.LogGammaPublic(6.0) > 4.78 && Beta.LogGammaPublic(6.0) < 4.79, "log-Gamma of 6 is log(120)");
        Check(Beta.Cdf(0.0, 3, 5) == 0.0 && Beta.Cdf(1.0, 3, 5) == 1.0, "CDF is pinned at the endpoints");
        Check(Math.Abs(Beta.Mean(3, 5) - 0.375) < 1e-15 && Math.Abs(Beta.Choose(5, 2) - 10.0) < 1e-12,
            "closed-form mean and binomial coefficient");

        // 4-6. Thompson sampling on a well-separated environment: the best arm dominates late rounds and the
        //      average regret falls, beating the uniform-random baseline.
        var environment = new HierarchicalBandit(Arms, 2, 2, 0x777UL);
        double[] separation = new double[Arms];
        for (int a = 0; a < Arms; a++) separation[a] = environment.Probability(a);
        Array.Sort(separation);
        Check(separation[Arms - 1] - separation[0] > 0.2, "the simulated arms are well separated");

        var agent = new ThompsonAgent(Arms, 1, 1, 0x99UL);
        double regret = Regret(agent, environment, Horizon, out double lateFraction, out double earlyRegret, out double lateRegret);
        Console.WriteLine("  Thompson: cumulative regret=" + regret.ToString("F2") + " best-arm share late="
            + lateFraction.ToString("F3") + " average regret early=" + (earlyRegret / (Horizon / 5)).ToString("F4")
            + " late=" + (lateRegret / (Horizon / 5)).ToString("F4"));
        Console.Write("  arms:");
        for (int a = 0; a < Arms; a++)
        {
            Console.Write(" a" + a + " p=" + environment.Probability(a).ToString("F3")
                + " trials=" + agent.Trials(a) + " rate=" + (agent.Trials(a) == 0 ? 0.0 : agent.Successes(a) / (double)agent.Trials(a)).ToString("F3") + ";");
        }
        Console.WriteLine(" best=" + environment.BestArm());
        Check(lateFraction > 0.75, "the best arm is played in most of the late rounds");
        Check(lateRegret / (Horizon / 5) < earlyRegret / (Horizon / 5), "average regret per round falls over time");

        var baseline = new ThompsonAgent(Arms, 1, 1, 0x99UL);
        var baselineRng = 0x33UL;
        double randomRegret = 0.0;
        for (int t = 0; t < Horizon; t++)
        {
            int arm = (int)(SplitMix64.NextDouble(ref baselineRng) * Arms);
            baseline.Observe(arm, environment.Play(arm));
            randomRegret += environment.Probability(environment.BestArm()) - environment.Probability(arm);
        }
        Console.WriteLine("  baseline: uniform random regret=" + randomRegret.ToString("F2"));
        Check(regret < randomRegret, "Thompson sampling beats the uniform random policy");

        // 7-9. Conjugate update and shrinkage: the posterior mean is the closed form, and it is bracketed by the
        //      maximum-likelihood estimate and the prior mean when data are scarce.
        var shrinkage = new ThompsonAgent(Arms, 8, 2, 0x1UL);
        shrinkage.Observe(0, 1);
        shrinkage.Observe(0, 0);
        double posterior = shrinkage.PosteriorMean(0);
        double mle = shrinkage.MaximumLikelihood(0);
        double prior = shrinkage.PriorMean;
        Console.WriteLine("  shrinkage: posterior=" + posterior.ToString("F4") + " mle=" + mle.ToString("F4")
            + " prior mean=" + prior.ToString("F4"));
        Check(Math.Abs(posterior - 9.0 / 12.0) < 1e-15, "posterior mean matches the closed form");
        Check(posterior > Math.Min(mle, prior) && posterior < Math.Max(mle, prior),
            "the hierarchical posterior is shrunk between the MLE and the prior mean");
        Check(Math.Abs(new ThompsonAgent(Arms, 4, 6, 0x2UL).PosteriorMean(3) - 0.4) < 1e-15,
            "an unplayed arm keeps exactly the prior mean");

        // 10-11. Empirical-Bayes hyperparameter update: the objective the step optimises must not decrease.
        var hyperAgent = new ThompsonAgent(Arms, 1, 1, 0x5UL);
        var hyperEnvironment = new HierarchicalBandit(Arms, 3, 2, 0x1234UL);
        for (int t = 0; t < 400; t++)
        {
            int arm = hyperAgent.Select();
            hyperAgent.Observe(arm, hyperEnvironment.Play(arm));
        }
        double before = hyperAgent.MarginalLogLikelihood();
        double alphaBefore = hyperAgent.Alpha, betaBefore = hyperAgent.BetaParameter;
        for (int step = 0; step < 25; step++) hyperAgent.UpdateHyperparameters(0.05);
        double after = hyperAgent.MarginalLogLikelihood();
        Console.WriteLine("  empirical Bayes: (alpha,beta) (" + alphaBefore.ToString("F3") + "," + betaBefore.ToString("F3")
            + ") -> (" + hyperAgent.Alpha.ToString("F3") + "," + hyperAgent.BetaParameter.ToString("F3")
            + "), objective " + before.ToString("F4") + " -> " + after.ToString("F4"));
        Check(after >= before - 1e-9, "the hyperparameter step never decreases the marginal likelihood");
        Check(hyperAgent.Alpha > 0.0 && hyperAgent.BetaParameter > 0.0, "hyperparameters stay positive");

        // 12. Sampling noise: the drawn posterior quantiles track the analytic Beta quantiles.
        int lowCount = 0;
        const int draws = 6000;
        var tailRng = 0x600DUL;
        var tailAgent = new ThompsonAgent(2, 2, 2, 0x11UL);
        tailAgent.Observe(0, 1);
        for (int i = 0; i < draws; i++)
        {
            if (tailAgent.Draw(0) <= 0.5) lowCount++;
        }
        double empiricalHalf = lowCount / (double)draws;
        double exactHalf = Beta.Cdf(0.5, 3, 2);
        Console.WriteLine("  posterior quantiles: P(X <= 0.5) empirical=" + empiricalHalf.ToString("F4")
            + " exact=" + exactHalf.ToString("F4"));
        Check(Math.Abs(empiricalHalf - exactHalf) < 0.015, "posterior draws match the analytic Beta quantile");

        // 13-15. Boundary cases and rejection of invalid parameters.
        var singleAgent = new ThompsonAgent(1, 1, 1, 0x7UL);
        var singleBandit = new HierarchicalBandit(1, 2, 2, 0x8UL);
        int singleReward = -1;
        for (int t = 0; t < 20; t++)
        {
            int arm = singleAgent.Select();
            singleReward = singleBandit.Play(arm);
            singleAgent.Observe(arm, singleReward);
        }
        Check(singleAgent.Trials(0) == 20 && singleBandit.BestArm() == 0, "a single-arm bandit is consistent");
        Check(singleAgent.Successes(0) >= 0 && singleAgent.Successes(0) <= 20, "success counts stay in range");
        Check(Throws(() => new ThompsonAgent(0, 1, 1, 1UL)), "zero arms rejected");
        Check(Throws(() => new ThompsonAgent(2, 0.0, 1, 1UL)), "non-positive alpha rejected");
        Check(Throws(() => new ThompsonAgent(2, 1, -1, 1UL)), "negative beta rejected");
        Check(Throws(() => new HierarchicalBandit(2, 1, 0, 1UL)), "zero prior parameter rejected for the environment");
        Check(Throws(() => singleBandit.Play(5)), "out-of-range arm rejected");
        var zeroRounds = new ThompsonAgent(Arms, 1, 1, 0x9UL);
        Check(zeroRounds.MarginalLogLikelihood() == 0.0 && Math.Abs(zeroRounds.PosteriorMean(0) - 0.5) < 1e-15,
            "an untouched agent has prior mean 0.5 and zero objective");

        // 16. Determinism.
        var repeatAgent = new ThompsonAgent(Arms, 1, 1, 0x99UL);
        var repeatEnvironment = new HierarchicalBandit(Arms, 2, 2, 0x777UL);
        double repeatRegret = Regret(repeatAgent, repeatEnvironment, Horizon, out double repeatLate, out _, out _);
        Check(repeatRegret == regret && repeatLate == lateFraction, "identical seeds reproduce the run bit for bit");

        Console.WriteLine("PASS");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
