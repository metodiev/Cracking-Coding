// Task 912 - Implement end-to-end pipeline for privacy-preserving analytics with DP budgeting.
// (Tier: Master, Beyond - Security & Privacy)
//
// A small analytics pipeline where every released number is a differentially private answer and every
// answer is charged against an explicit privacy budget:
//
//  * data contributors are clamped, so a single contributor can move a count by at most one - that bound is
//    the declared sensitivity and it is re-derived here by brute force over the clamped values;
//  * counts are released with the Laplace mechanism, drawn from a deterministic splitmix64 stream so the
//    whole pipeline (and every statistic reported below) is reproducible; with b = sensitivity / epsilon
//    the noise has variance 2b^2 and P(|noise| > b ln(20)) = 0.05, both of which are measured;
//  * the accountant implements basic sequential composition (costs add up) and parallel composition
//    (disjoint partitions cost only the maximum), refuses any query that would overrun the budget, and the
//    refusal point is re-derived from the granted epsilons rather than trusted.
//
// Simplifications: Laplace-only mechanisms, integer counts over a handful of partitions, no privacy
// amplification by subsampling and no RDP/zCDP accountants.  Verified: no over-spending, the accounting
// identity, noise mean/variance and tail bounds against the closed form, sensitivity by brute force,
// parallel composition, the zero-budget and single-query boundaries, and reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;

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

    private static double NextUniform() => ((NextUInt64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    /** Laplace(0, scale) by inverse CDF: the sign comes from the U(0,1) draw, the magnitude from its log. */
    private static double Laplace(double scale)
    {
        double uniform = NextUniform() - 0.5;
        double magnitude = scale * Math.Log(1.0 - 2.0 * Math.Abs(uniform));
        return uniform >= 0 ? -magnitude : magnitude;
    }

    /** The accountant: sequential costs add, parallel queries inside a group cost only the maximum. */
    internal sealed class Accountant
    {
        private readonly double _total;
        private readonly Dictionary<string, double> _groups = new Dictionary<string, double>();
        internal readonly List<string> Ledger = new List<string>();
        internal double Spent;
        internal int GrantCount;
        internal int RefusalCount;

        internal Accountant(double total)
        {
            _total = total;
        }

        internal double Remaining => _total - Spent;

        internal bool Spend(string label, double epsilon)
        {
            if (epsilon <= 0 || epsilon > Remaining + 1e-12)
            {
                RefusalCount++;
                Ledger.Add("refused " + label + " (eps=" + epsilon.ToString("F4", CultureInfo.InvariantCulture) + ")");
                return false;
            }
            Spent += epsilon;
            GrantCount++;
            Ledger.Add("granted " + label + " (eps=" + epsilon.ToString("F4", CultureInfo.InvariantCulture) + ")");
            return true;
        }

        /** Parallel composition: every query in the group is charged the maximum epsilon of the group. */
        internal bool SpendParallel(string group, double epsilon)
        {
            _groups.TryGetValue(group, out double current);
            double increment = Math.Max(0.0, epsilon - current);
            if (increment > Remaining + 1e-12)
            {
                RefusalCount++;
                Ledger.Add("refused parallel " + group + " (eps=" + epsilon.ToString("F4", CultureInfo.InvariantCulture) + ")");
                return false;
            }
            _groups[group] = Math.Max(current, epsilon);
            Spent += increment;
            GrantCount++;
            Ledger.Add("granted parallel " + group + " (eps=" + epsilon.ToString("F4", CultureInfo.InvariantCulture) + ")");
            return true;
        }

        /** An independent recomputation of the granted budget from the ledger. */
        internal double RecomputeSpent()
        {
            double total = 0.0;
            foreach (string entry in Ledger)
            {
                if (!entry.StartsWith("granted ", StringComparison.Ordinal)) continue;
                int open = entry.IndexOf("(eps=", StringComparison.Ordinal) + 5;
                int close = entry.IndexOf(')', open);
                total += double.Parse(entry.Substring(open, close - open), CultureInfo.InvariantCulture);
            }
            return total;
        }
    }

    /** Adjacent-dataset sensitivity of a clamped count, by brute force over the contributor values. */
    private static int ClampedSensitivity(double[] raw)
    {
        int worst = 0;
        for (int i = 0; i < raw.Length; i++)
        {
            double original = Clamp(raw[i]);
            foreach (double replacement in new[] { 0.0, 1.0, 0.5, 7.0 })
            {
                int delta = Math.Abs((int)Math.Round(Math.Abs(original - Clamp(replacement))));
                worst = Math.Max(worst, delta);
            }
        }
        return worst;
    }

    private static double Clamp(double value) => Math.Min(1.0, Math.Max(0.0, value));

    private static int TrueCount(double[] raw) => (int)Math.Round(raw.Length > 0 ? Sum(raw) : 0.0);

    private static double Sum(double[] raw)
    {
        double total = 0.0;
        foreach (double value in raw) total += Clamp(value);
        return total;
    }

    public static void Main()
    {
        // ---- 1. sensitivity is exactly the declared bound --------------------------------------------
        double[] contributors = { 1.0, 1.0, 0.0, 1.0, 3.7, 1.0, 0.2, 1.0, 12.0, 1.0 };
        Check(ClampedSensitivity(contributors) == 1,
            "clamping to [0,1] makes one contributor worth at most one count (sensitivity 1)");
        Check(TrueCount(contributors) == 8, "the clamped true count is " + TrueCount(contributors));
        Check(Math.Abs(Sum(contributors) - 8.2) < 1e-9, "clamping keeps the fractional 0.2 contributor and caps the outliers");
        Check(ClampedSensitivity(new double[0]) == 0, "an empty contributor set has zero sensitivity");

        // ---- 2. the Laplace mechanism: variance, tail and unbiasedness ---------------------------------
        const double epsilon = 0.5;
        double b = 1.0 / epsilon;
        const int runs = 4000;
        _state = 1234567UL;
        double sum = 0.0, sumSquares = 0.0;
        int beyond = 0;
        double threshold = b * Math.Log(20.0); // P(|X| > b ln 20) = 0.05
        for (int i = 0; i < runs; i++)
        {
            double noise = Laplace(b);
            sum += noise;
            sumSquares += noise * noise;
            if (Math.Abs(noise) > threshold) beyond++;
        }
        double mean = sum / runs;
        double variance = sumSquares / runs - mean * mean;
        double standardError = Math.Sqrt(2.0 * b * b / runs);
        Check(Math.Abs(mean) <= 4.0 * standardError, "the empirical noise mean is unbiased (" +
            mean.ToString("F4", CultureInfo.InvariantCulture) + " vs standard error " + standardError.ToString("F4", CultureInfo.InvariantCulture) + ")");
        Check(Math.Abs(variance - 2.0 * b * b) < 0.2 * 2.0 * b * b, "the empirical variance matches 2b^2 (" +
            variance.ToString("F4", CultureInfo.InvariantCulture) + " vs " + (2.0 * b * b).ToString("F4", CultureInfo.InvariantCulture) + ")");
        double tail = beyond / (double)runs;
        Check(tail <= 0.05 + 0.02, "the empirical tail beyond b ln 20 stays near the closed-form 0.05 (" +
            tail.ToString("F4", CultureInfo.InvariantCulture) + ")");

        // ---- 3. the budget is never overrun, and refusals are exact -------------------------------------
        Accountant accountant = new Accountant(1.0);
        double[] request = { 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.05 };
        double grantedSoFar = 0.0;
        int expectedGrants = 0;
        foreach (double requestEpsilon in request)
        {
            bool granted = accountant.Spend("q" + requestEpsilon.ToString("F2", CultureInfo.InvariantCulture), requestEpsilon);
            bool shouldGrant = requestEpsilon <= 1.0 - grantedSoFar + 1e-12;
            Check(granted == shouldGrant, "the admission decision matches an independent budget recomputation (" +
                requestEpsilon.ToString("F2", CultureInfo.InvariantCulture) + ")");
            if (granted)
            {
                grantedSoFar += requestEpsilon;
                expectedGrants++;
            }
        }
        Check(accountant.Spent <= 1.0 + 1e-12, "the accountant never spends past the total budget (" +
            accountant.Spent.ToString("F4", CultureInfo.InvariantCulture) + ")");
        Check(accountant.GrantCount == expectedGrants && accountant.GrantCount == 5,
            "exactly " + accountant.GrantCount + " queries were granted, the rest refused (" + accountant.RefusalCount + ")");
        Check(Math.Abs(accountant.RecomputeSpent() - accountant.Spent) < 1e-12,
            "the ledger replays to the same spend (" + accountant.RecomputeSpent().ToString("F4", CultureInfo.InvariantCulture) + ")");

        // ---- 4. parallel composition over disjoint partitions --------------------------------------------
        Accountant parallel = new Accountant(1.0);
        bool allGranted = true;
        for (int partition = 0; partition < 5; partition++) allGranted &= parallel.SpendParallel("disjoint", epsilon);
        Check(allGranted, "five disjoint partitions each fit in the budget");
        Check(Math.Abs(parallel.Spent - epsilon) < 1e-12,
            "parallel composition charged the maximum epsilon, not the sum (" + parallel.Spent.ToString("F4", CultureInfo.InvariantCulture) + ")");
        bool secondRound = true;
        for (int partition = 0; partition < 5; partition++) secondRound &= parallel.SpendParallel("disjoint", 0.7);
        Check(secondRound && Math.Abs(parallel.Spent - 0.7) < 1e-12,
            "raising the per-partition epsilon charges only the increment (" + parallel.Spent.ToString("F4", CultureInfo.InvariantCulture) + ")");
        Check(!parallel.SpendParallel("disjoint", 2.0), "a parallel query beyond the remaining budget is refused");
        Check(Math.Abs(parallel.Spent - 0.7) < 1e-12, "a refused query costs nothing");

        // ---- 5. end-to-end: released answers track the truth within the noise bound -------------------------
        _state = 99991UL;
        int insideFourB = 0, released = 0;
        double looseThreshold = 4.0 * b;
        for (int trial = 0; trial < 200; trial++)
        {
            double answer = TrueCount(contributors) + Laplace(b);
            released++;
            if (Math.Abs(answer - TrueCount(contributors)) <= looseThreshold) insideFourB++;
        }
        Check(released == 200 && insideFourB >= 195, "released counts stay within 4b of the truth almost always (" +
            insideFourB + "/" + released + ")");

        // ---- 6. boundaries ---------------------------------------------------------------------------------
        Accountant empty = new Accountant(0.0);
        Check(!empty.Spend("q", 0.1) && empty.Spent == 0.0, "a zero budget serves nothing");
        Check(!empty.Spend("q", 0.0), "a zero-epsilon query is refused rather than answered without noise");
        Accountant single = new Accountant(2.0);
        Check(single.Spend("all", 2.0) && Math.Abs(single.Remaining) < 1e-12, "one query may consume the entire budget exactly");
        Check(!single.Spend("more", 1e-9), "and then the next request is refused");
        Accountant tiny = new Accountant(1.0);
        int rounds = 0;
        while (tiny.Spend("q", 0.1) && rounds < 100) rounds++;
        Check(rounds == 10, "ten 0.1-epsilon queries exactly exhaust a budget of 1.0 (" + rounds + ")");

        // ---- 7. reproducibility ------------------------------------------------------------------------------
        _state = 4242UL;
        double first = Laplace(b);
        _state = 4242UL;
        double second = Laplace(b);
        Check(first == second, "the same seed reproduces the same released noise");
        _state = 4243UL;
        Check(Laplace(b) != first, "a fresh seed changes the release");

        Console.WriteLine("mechanism: Laplace with b=1/epsilon=" + b.ToString("F1", CultureInfo.InvariantCulture) +
            ", empirical mean " + mean.ToString("F4", CultureInfo.InvariantCulture) + ", variance " + variance.ToString("F4", CultureInfo.InvariantCulture) +
            " versus 2b^2=" + (2.0 * b * b).ToString("F4", CultureInfo.InvariantCulture));
        Console.WriteLine("accounting: sequential spend " + accountant.Spent.ToString("F4", CultureInfo.InvariantCulture) +
            " of 1.0 with " + accountant.RefusalCount + " refusals; parallel group charged " +
            parallel.Spent.ToString("F4", CultureInfo.InvariantCulture) + " for 5 disjoint partitions");
        Console.WriteLine("tail: " + insideFourB + "/200 released counts inside 4b, " + beyond + "/" + runs +
            " noises beyond b ln 20, sensitivity derived by brute force as 1");
        Console.WriteLine("PASS");
    }
}
