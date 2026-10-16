// Task 977 - Implement long-term retention estimator using learner response models.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Each learner is described by a half-life h: predicted recall after t days is 0.5^(t/h), so the
// curve passes through 0.5 at exactly t = h. Every observation (gap t, whether the recall
// succeeded) nudges log h along the log-likelihood gradient of that model,
//   h' = clamp(h * exp(eta * (correct ? 1-p : -p) * ln2 * t / h), 0.5, 3650),
// and the next review is planned for the day where recall drops to a target threshold.
//
// Verified: the endpoints p(0)=1 and p(h)=0.5, the review-planning round trip, an independent
// Math.Pow closed form as the oracle for the exponential curve, update direction, the clamp
// invariant re-checked after 400 random observations, recovery of three injected half-lives within
// 30%, plus the negative-gap, zero-half-life and empty-history boundaries.
// Simplification: a single scalar half-life per learner, no item difficulty effect and no
// per-learner feature regression; the response stream is simulated with a seeded PRNG.
using System;
using System.Collections.Generic;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static double NextUnit() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private const double MinHalfLife = 0.5;
    private const double MaxHalfLife = 3650.0;

    private sealed class Observation
    {
        public readonly double GapDays;
        public readonly bool Recalled;

        public Observation(double gapDays, bool recalled)
        {
            GapDays = gapDays;
            Recalled = recalled;
        }
    }

    /** Probability that the learner still recalls the item t days after the last review. */
    private static double Predict(double halfLife, double days)
    {
        if (halfLife <= 0.0) throw new ArgumentException("half-life must be positive");
        if (days <= 0.0) return 1.0;
        return Math.Exp(-Math.Log(2.0) * days / halfLife);
    }

    private static double UpdateHalfLife(double halfLife, double days, bool correct, double learningRate)
    {
        double predicted = Predict(halfLife, days);
        double gradient = Math.Log(2.0) * Math.Max(days, 0.0) / halfLife;
        double step = learningRate * (correct ? 1.0 - predicted : -predicted) * gradient;
        double updated = halfLife * Math.Exp(step);
        return Math.Min(MaxHalfLife, Math.Max(MinHalfLife, updated));
    }

    private static double EstimateFromHistory(double prior, List<Observation> history, double learningRate)
    {
        double halfLife = prior;
        foreach (Observation observation in history)
            halfLife = UpdateHalfLife(halfLife, observation.GapDays, observation.Recalled, learningRate);
        return halfLife;
    }

    /** Day at which recall decays to `threshold`: t = h * ln(threshold) / ln(0.5). */
    private static double NextReviewDay(double halfLife, double threshold)
    {
        if (threshold <= 0.0 || threshold >= 1.0) throw new ArgumentException("threshold must be inside (0,1)");
        return halfLife * Math.Log(threshold) / Math.Log(0.5);
    }

    private static List<Observation> Replay(double trueHalfLife, ulong seed, int rounds, double span)
    {
        _state = seed;
        var history = new List<Observation>();
        for (int i = 0; i < rounds; i++)
        {
            double gap = 1.0 + NextUnit() * span;
            history.Add(new Observation(gap, NextUnit() < Predict(trueHalfLife, gap)));
        }
        return history;
    }

    public static void Main()
    {
        // Endpoints and the half-life point itself.
        Check(Predict(10.0, 0.0) == 1.0, "recall immediately after a review is 1.0");
        Check(Math.Abs(Predict(10.0, 10.0) - 0.5) < 1e-15, "recall at t = half-life is exactly 0.5");
        Check(Math.Abs(Predict(10.0, 20.0) - 0.25) < 1e-15, "two half-lives leave a quarter of the recall");

        // Oracle: the curve must equal the independent power form 0.5^(t/h).
        double worst = 0.0;
        double[] lives = { 0.5, 2.0, 7.5, 40.0, 365.0 };
        double[] gaps = { 0.25, 1.0, 5.0, 30.0, 400.0 };
        foreach (double life in lives)
            foreach (double gap in gaps)
                worst = Math.Max(worst, Math.Abs(Predict(life, gap) - Math.Pow(0.5, gap / life)));
        Check(worst < 1e-15, "the exponential form matches the independent power oracle to 1e-15 (worst "
            + worst.ToString("E2") + ")");

        bool decreasing = true;
        for (double day = 0.0; day < 60.0; day += 0.5) decreasing &= Predict(12.0, day) >= Predict(12.0, day + 0.5);
        Check(decreasing, "predicted recall is non-increasing in the gap for a 12 day half-life");
        Check(Predict(5.0, 7.0) < Predict(10.0, 7.0), "a longer half-life retains more after a week");

        // Update direction and the clamp invariant after every mutation.
        Check(UpdateHalfLife(10.0, 4.0, true, 0.15) > 10.0, "an on-time recall lengthens the half-life");
        Check(UpdateHalfLife(10.0, 4.0, false, 0.15) < 10.0, "a lapse shortens the half-life");
        Check(UpdateHalfLife(10.0, 0.0, true, 0.15) == 10.0, "a recall at zero gap produces no update");

        _state = 977UL;
        double halfLife = 20.0;
        double lowest = halfLife, highest = halfLife;
        int escaped = 0;
        for (int i = 0; i < 400; i++)
        {
            double gap = NextUnit() * 90.0;
            bool correct = NextUnit() < Predict(30.0, gap);
            halfLife = UpdateHalfLife(halfLife, gap, correct, 0.2);
            lowest = Math.Min(lowest, halfLife);
            highest = Math.Max(highest, halfLife);
            if (halfLife < MinHalfLife - 1e-12 || halfLife > MaxHalfLife + 1e-12) escaped++;
            double predicted = Predict(halfLife, gap);
            if (predicted < 0.0 || predicted > 1.0) escaped++;
        }
        Check(escaped == 0, "the half-life and the predicted recall stay inside their bounds over 400 updates");
        Check(lowest >= MinHalfLife && highest <= MaxHalfLife, "the observed half-life band is ["
            + lowest.ToString("F2") + ", " + highest.ToString("F2") + "]");

        // Review planning round trip: the planned day really is where recall hits the threshold.
        double worstTrip = 0.0;
        foreach (double life in new[] { 3.0, 12.0, 60.0 })
        {
            double day = NextReviewDay(life, 0.8);
            worstTrip = Math.Max(worstTrip, Math.Abs(Predict(life, day) - 0.8));
        }
        Check(worstTrip < 1e-12, "planning a review at 80% recall round trips through the curve (worst "
            + worstTrip.ToString("E2") + ")");
        Check(NextReviewDay(10.0, 0.5) == 10.0, "a 50% threshold plans the next review at the half-life");
        Check(NextReviewDay(10.0, 0.9) < NextReviewDay(10.0, 0.8),
            "a stricter recall threshold plans the review earlier");

        // Recovery of three injected half-lives from simulated response streams.
        _state = 0UL;
        double[] truths = { 2.0, 10.0, 40.0 };
        double[] estimates = new double[truths.Length];
        for (int i = 0; i < truths.Length; i++)
        {
            List<Observation> history = Replay(truths[i], 9770UL + (ulong)i, 1000, 9.0);
            estimates[i] = EstimateFromHistory(5.0, history, 0.15);
            double ratio = estimates[i] / truths[i];
            Check(ratio > 0.7 && ratio < 1.3, "the estimate for half-life " + truths[i]
                + " lands within 30% of the truth (got " + estimates[i].ToString("F2") + ")");
        }

        // Boundaries: an empty history keeps the prior, negative gaps are harmless, h<=0 is rejected.
        Check(EstimateFromHistory(8.0, new List<Observation>(), 0.15) == 8.0,
            "a learner with no observations keeps the prior half-life");
        Check(Predict(4.0, -5.0) == 1.0, "a negative gap is treated as an immediate recall");
        bool rejected = false;
        try { Predict(0.0, 3.0); }
        catch (ArgumentException) { rejected = true; }
        Check(rejected, "a non-positive half-life is rejected instead of dividing by zero");

        Console.WriteLine("recovery: truths 2/10/40 -> " + estimates[0].ToString("F2") + "/"
            + estimates[1].ToString("F2") + "/" + estimates[2].ToString("F2"));
        Console.WriteLine("planning: 80% threshold at " + NextReviewDay(10.0, 0.8).ToString("F2")
            + " days for h=10, 50% at " + NextReviewDay(10.0, 0.5).ToString("F2") + " days");
        Console.WriteLine("curve oracle agreement: worst deviation " + worst.ToString("E2") + " over 25 pairs");
        Console.WriteLine("PASS");
    }
}
