// Task 901 - Implement provable bounds for randomized algorithms used in your projects, with tests.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Two randomized algorithms are paired with the closed-form bounds that justify them:
//
//  * Karger's contraction min-cut: contract a uniformly random edge until two super-vertices remain.
//    Every cut it returns is a genuine cut, so its size is at least the global minimum, and the
//    probability that it returns the minimum is at least 2/(n(n-1)).  The returned value is cross-checked
//    against a brute-force minimum over all 2^n vertex bipartitions, the "never below the optimum"
//    invariant is re-checked after every trial, and the success frequency over thousands of
//    deterministic seeds is compared with the theorem.
//  * Randomized rounding: n independent fair coins drive a placement decision.  The upper tail
//    P(X >= (1+d)np) <= exp(-np d^2/3), the lower tail P(X <= (1-d)np) <= exp(-np d^2/2) and Hoeffding's
//    two-sided bound are verified against the exact binomial tail computed by dynamic programming (an
//    independent oracle), and the empirical frequency of the same event is checked to sit below the bound.
//
// Simplifications (documented, not behavioural): graphs stay tiny (n <= 8) so the exhaustive oracle and
// the trial counts remain cheap, and randomness is a hand-written splitmix64 stream rather than a system
// RNG so every measurement is reproducible.  Verified: the bound inequalities on a parameter grid, cut
// optimality and validity, the Karger lower bound, pmf normalisation and stream determinism.
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

    private static int NextInt(int bound) => (int)(NextUInt64() % (ulong)bound);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    // ---- exact binomial tails: the independent oracle for the closed-form bounds ----

    private static double[] BinomialPmf(int n, double p)
    {
        double[] pmf = new double[n + 1];
        pmf[0] = Math.Pow(1.0 - p, n);
        for (int k = 1; k <= n; k++) pmf[k] = pmf[k - 1] * (n - k + 1) / k * (p / (1.0 - p));
        return pmf;
    }

    private static double ExactTail(int n, double p, double delta, bool upper)
    {
        double[] pmf = BinomialPmf(n, p);
        double edge = upper ? (1.0 + delta) * n * p : (1.0 - delta) * n * p;
        double sum = 0.0;
        for (int k = 0; k <= n; k++)
        {
            if (upper ? k >= edge : k <= edge) sum += pmf[k];
        }
        return sum;
    }

    private static double ChernoffUpper(int n, double p, double delta) => Math.Exp(-n * p * delta * delta / 3.0);

    private static double ChernoffLower(int n, double p, double delta) => Math.Exp(-n * p * delta * delta / 2.0);

    private static double Hoeffding(int n, double t) => 2.0 * Math.Exp(-2.0 * t * t / n);

    // ---- Karger's randomized contraction ----

    private static int Find(int[] parent, int v)
    {
        while (parent[v] != v) v = parent[v] = parent[parent[v]];
        return v;
    }

    /** One contraction run; returns the size of the cut induced by the two surviving super-vertices. */
    private static int KargerCut(int n, List<int[]> edges)
    {
        int[] parent = new int[n];
        for (int v = 0; v < n; v++) parent[v] = v;
        List<int[]> live = new List<int[]>(edges);
        int groups = n;
        while (groups > 2)
        {
            int[] pick = live[NextInt(live.Count)];
            int a = Find(parent, pick[0]);
            int b = Find(parent, pick[1]);
            if (a == b)
            {
                live.Remove(pick); // a self-loop produced by an earlier contraction: delete, do not contract
                continue;
            }
            parent[b] = a;
            groups--;
        }
        int cut = 0;
        foreach (int[] e in live)
        {
            if (Find(parent, e[0]) != Find(parent, e[1])) cut++;
        }
        return cut;
    }

    /** Exhaustive minimum cut: the oracle Karger's answer must never beat. */
    private static int BruteForceMinCut(int n, List<int[]> edges)
    {
        int best = int.MaxValue;
        for (int mask = 1; mask < (1 << n) - 1; mask++)
        {
            int cut = 0;
            foreach (int[] e in edges)
            {
                bool left = (mask & (1 << e[0])) != 0;
                bool right = (mask & (1 << e[1])) != 0;
                if (left != right) cut++;
            }
            best = Math.Min(best, cut);
        }
        return best;
    }

    private static List<int[]> Cycle(int n)
    {
        List<int[]> edges = new List<int[]>();
        for (int v = 0; v < n; v++) edges.Add(new[] { v, (v + 1) % n });
        return edges;
    }

    private static List<int[]> Complete(int n)
    {
        List<int[]> edges = new List<int[]>();
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++) edges.Add(new[] { u, v });
        }
        return edges;
    }

    private static List<int[]> RandomGraph(int n, double density)
    {
        List<int[]> edges = new List<int[]>();
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++)
            {
                if (NextUInt64() % 1000UL < (ulong)(density * 1000.0)) edges.Add(new[] { u, v });
            }
        }
        if (edges.Count == 0) edges.Add(new[] { 0, 1 });
        return edges;
    }

    public static void Main()
    {
        // ---- 1. closed-form bounds versus the exact dp tail ------------------------------------------
        int gridPoints = 0;
        foreach (int n in new[] { 10, 50, 200 })
        {
            foreach (double p in new[] { 0.2, 0.5, 0.9 })
            {
                foreach (double delta in new[] { 0.05, 0.2, 0.5 })
                {
                    Check(ExactTail(n, p, delta, true) <= ChernoffUpper(n, p, delta) + 1e-15,
                        "chernoff upper dominates the exact tail at n=" + n + " p=" + p + " d=" + delta);
                    Check(ExactTail(n, p, delta, false) <= ChernoffLower(n, p, delta) + 1e-15,
                        "chernoff lower dominates the exact tail at n=" + n + " p=" + p + " d=" + delta);
                    gridPoints += 2;
                }
            }
        }
        double[] pmf = BinomialPmf(64, 0.37);
        double mass = 0.0;
        foreach (double probability in pmf) mass += probability;
        Check(Math.Abs(mass - 1.0) < 1e-9, "the oracle pmf is normalised (sum = " + mass.ToString("F12", CultureInfo.InvariantCulture) + ")");
        Check(Math.Abs(ExactTail(1, 0.5, 1.0, true) - 0.5) < 1e-15 && Math.Abs(ExactTail(1, 0.5, 1.0, false) - 0.5) < 1e-15,
            "the single-coin boundary splits the mass 0.5/0.5 between the two directions");
        Check(ExactTail(0, 0.5, 0.5, true) == 1.0, "an empty experiment makes the tail event certain");

        // ---- 2. Hoeffding bound ----------------------------------------------------------------
        int hoeffdingPoints = 0;
        foreach (int n in new[] { 8, 32, 128 })
        {
            double p = 0.5;
            for (double t = 0.5; t <= n / 4.0; t += 1.0)
            {
                double exact = 0.0;
                double[] tail = BinomialPmf(n, p);
                for (int k = 0; k <= n; k++)
                {
                    if (Math.Abs(k - n * p) >= t) exact += tail[k];
                }
                Check(exact <= Hoeffding(n, t) + 1e-15, "hoeffding covers the exact two-sided tail (n=" + n + ", t=" + t + ")");
                hoeffdingPoints++;
            }
        }

        // ---- 3. Karger: optimality, validity and the 2/(n(n-1)) lower bound --------------------------
        _state = 20240901UL;
        var graphs = new List<(string Name, int N, List<int[]> Edges)>
        {
            ("path P4", 4, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 } }),
            ("cycle C5", 5, Cycle(5)),
            ("complete K5", 5, Complete(5)),
            ("cycle C8", 8, Cycle(8)),
            ("random n=6", 6, RandomGraph(6, 0.55)),
        };
        var report = new List<string>();
        foreach ((string name, int n, List<int[]> edges) in graphs)
        {
            int optimum = BruteForceMinCut(n, edges);
            const int trials = 4000;
            int successes = 0;
            int minSeen = int.MaxValue;
            for (int trial = 0; trial < trials; trial++)
            {
                int cut = KargerCut(n, edges);
                minSeen = Math.Min(minSeen, cut);
                if (cut < optimum) Check(false, "Karger returned a cut below the optimum on " + name);
                if (cut == optimum) successes++;
            }
            double lowerBound = 2.0 / (n * (n - 1));
            double frequency = successes / (double)trials;
            Check(minSeen >= optimum, "no trial beat the brute-force optimum on " + name + " (min seen " + minSeen + ")");
            Check(frequency >= lowerBound,
                "empirical Karger success " + frequency.ToString("F4", CultureInfo.InvariantCulture) + " >= 2/(n(n-1))=" +
                lowerBound.ToString("F4", CultureInfo.InvariantCulture) + " on " + name);
            report.Add(name + ":" + frequency.ToString("F3", CultureInfo.InvariantCulture));
        }
        var single = new List<int[]> { new[] { 0, 1 } };
        Check(KargerCut(2, single) == 1 && BruteForceMinCut(2, single) == 1,
            "the boundary graph with two vertices and one edge has cut 1 in every run");

        // ---- 4. empirical tail below the analytic bound ---------------------------------------
        _state = 987654321UL;
        const int coins = 200;
        const int rounds = 600;
        double bound = ChernoffUpper(coins, 0.5, 0.2);
        int exceedances = 0;
        for (int round = 0; round < rounds; round++)
        {
            int heads = 0;
            for (int i = 0; i < coins; i++) heads += (int)(NextUInt64() & 1UL);
            if (heads >= (int)(1.2 * coins * 0.5)) exceedances++;
        }
        double empirical = exceedances / (double)rounds;
        Check(empirical <= bound, "empirical exceedance " + empirical.ToString("F4", CultureInfo.InvariantCulture) +
            " stays below the chernoff bound " + bound.ToString("E3", CultureInfo.InvariantCulture));

        // ---- 5. determinism -----------------------------------------------------------------------
        _state = 7UL;
        var firstRun = new List<int>();
        for (int i = 0; i < 5; i++) firstRun.Add(KargerCut(6, Complete(6)));
        _state = 7UL;
        var secondRun = new List<int>();
        for (int i = 0; i < 5; i++) secondRun.Add(KargerCut(6, Complete(6)));
        Check(firstRun.Count == secondRun.Count, "the same seed yields the same number of trials");
        bool identical = true;
        for (int i = 0; i < firstRun.Count; i++) identical &= firstRun[i] == secondRun[i];
        Check(identical, "restarting the seed reproduces the identical cut sequence");
        _state = 8UL;
        var thirdRun = new List<int>();
        for (int i = 0; i < 5; i++) thirdRun.Add(KargerCut(6, Complete(6)));
        bool differs = false;
        for (int i = 0; i < thirdRun.Count; i++) differs |= thirdRun[i] != firstRun[i];
        Check(differs, "a different seed explores a different contraction schedule");

        Console.WriteLine("bounds: " + gridPoints + " chernoff inequalities and " + hoeffdingPoints +
            " hoeffding inequalities verified against the dp tail; pmf mass " + mass.ToString("F9", CultureInfo.InvariantCulture));
        Console.WriteLine("karger: success frequency per graph -> " + string.Join(", ", report));
        Console.WriteLine("randomized rounding: " + exceedances + "/" + rounds + " exceedances at delta=0.2 versus bound " +
            bound.ToString("E3", CultureInfo.InvariantCulture));
        Console.WriteLine("PASS");
    }
}
