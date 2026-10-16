// Task 940 - Implement high-fidelity load generator with correlated user behavior models.
// (Tier: Master, CTCI Ch 11 — Testing)
//
// A closed behavioural model of an e-commerce session: a six-state Markov chain (browse, search,
// view, add-to-cart, checkout, exit) with per-state exponential think times, driven by an
// open-loop Poisson arrival process (inter-arrivals sampled by inverting the exponential CDF).
// Each session walks the chain until it draws the absorbing exit state, so the generated load
// carries the *correlation* the chain encodes - a session that searched is far more likely to view
// a product, and only add-to-cart sessions can check out.
//
// The generator is validated statistically and against closed forms rather than by inspection:
// the empirical transition matrix must converge to the configured one; the expected number of
// steps and the expected per-state visit counts come from solving (I - Q)^-1 e_start for the
// transient submatrix Q, and the probability of ever reaching checkout comes from an independent
// hitting-probability linear system - both compared with the simulation. The arrival process is
// checked against the Poisson prediction (per-window counts inside four standard deviations) and
// the exponential prediction (mean and standard deviation of the gaps). Degenerate matrices,
// sessions that start already exited, and reproducibility for a fixed seed are boundary checks.
// Simplified: one traffic class, no ramp shape, no per-user state reuse, think times are memoryless
// and arrival rate is constant; the chain, its closed forms and the arrival process are real.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class Session
{
    internal readonly List<int> States = new List<int>();
    internal readonly List<double> ThinkTimesMs = new List<double>();
    internal double ArrivalGapMs;
}

internal sealed class MarkovWorkload
{
    internal const int Exit = 5;
    internal const int Checkout = 4;
    internal const int AddToCart = 3;
    internal const int StateCount = 6;
    internal const int TransientCount = 5;

    private readonly double[,] transitions;
    internal readonly double[] ThinkMeanMs;
    private ulong state;

    internal MarkovWorkload(double[,] transitions, double[] thinkMeanMs, ulong seed)
    {
        for (int s = 0; s < StateCount; s++)
        {
            double sum = 0.0;
            for (int t = 0; t < StateCount; t++) sum += transitions[s, t];
            if (Math.Abs(sum - 1.0) > 1e-9) throw new ArgumentException("row " + s + " does not sum to 1");
        }
        this.transitions = transitions;
        ThinkMeanMs = thinkMeanMs;
        state = seed;
    }

    internal double Transition(int from, int to) => transitions[from, to];

    /** Uniform in (0,1): the open interval keeps the exponential inversion finite. */
    private double Uniform() => ((SplitMix64.Next(ref state) >> 11) + 0.5) * (1.0 / 9007199254740992.0);

    private int Sample(int from)
    {
        double draw = Uniform();
        double cumulative = 0.0;
        for (int to = 0; to < StateCount; to++)
        {
            cumulative += transitions[from, to];
            if (draw <= cumulative) return to;
        }
        return StateCount - 1;
    }

    internal Session GenerateSession(int startState, double rate)
    {
        var session = new Session { ArrivalGapMs = -1000.0 * Math.Log(Uniform()) / rate };
        int current = startState;
        while (current != Exit)
        {
            session.States.Add(current);
            session.ThinkTimesMs.Add(-ThinkMeanMs[current] * Math.Log(Uniform()));
            current = Sample(current);
        }
        return session;
    }
}

internal sealed class WorkloadReport
{
    internal readonly List<Session> Sessions = new List<Session>();
    internal readonly long[,] TransitionCounts = new long[MarkovWorkload.StateCount, MarkovWorkload.StateCount];
    internal readonly long[] VisitCounts = new long[MarkovWorkload.StateCount];
    internal readonly double[] ThinkTimeSum = new double[MarkovWorkload.StateCount];
    internal readonly long[] ThinkTimeCount = new long[MarkovWorkload.StateCount];
    internal long Transitions;
    internal long CartSessions;
    internal double TotalArrivalMs;

    internal void Add(IEnumerable<Session> sessions)
    {
        foreach (Session session in sessions)
        {
            Sessions.Add(session);
            TotalArrivalMs += session.ArrivalGapMs;
            bool reachedCart = false;
            for (int i = 0; i < session.States.Count; i++)
            {
                int current = session.States[i];
                VisitCounts[current]++;
                ThinkTimeSum[current] += session.ThinkTimesMs[i];
                ThinkTimeCount[current]++;
                if (current == MarkovWorkload.AddToCart) reachedCart = true;
                int next = i + 1 < session.States.Count ? session.States[i + 1] : MarkovWorkload.Exit;
                TransitionCounts[current, next]++;
                Transitions++;
            }
            if (reachedCart) CartSessions++;
        }
    }

    internal double MeanSteps => Sessions.Count == 0 ? 0.0 : (double)Sessions.Sum(s => s.States.Count) / Sessions.Count;
    internal double CartFraction => Sessions.Count == 0 ? 0.0 : (double)CartSessions / Sessions.Count;
}

internal static class LinearAlgebra
{
    /** Gaussian elimination with partial pivoting; null when the matrix is singular. */
    internal static double[]? Solve(double[,] matrix, double[] rhs)
    {
        int n = rhs.Length;
        var a = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) a[i, j] = matrix[i, j];
            a[i, n] = rhs[i];
        }
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(a[row, col]) > Math.Abs(a[pivot, col])) pivot = row;
            if (Math.Abs(a[pivot, col]) < 1e-12) return null;
            for (int j = 0; j <= n; j++) { double swap = a[col, j]; a[col, j] = a[pivot, j]; a[pivot, j] = swap; }
            for (int row = 0; row < n; row++)
            {
                if (row == col) continue;
                double factor = a[row, col] / a[col, col];
                if (factor == 0.0) continue;
                for (int j = col; j <= n; j++) a[row, j] -= factor * a[col, j];
            }
        }
        var solution = new double[n];
        for (int i = 0; i < n; i++) solution[i] = a[i, n] / a[i, i];
        return solution;
    }
}

public static class Solution
{
    private const int Transient = MarkovWorkload.TransientCount;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static double[,] ShopModel() => new double[,]
    {
        { 0.30, 0.30, 0.35, 0.02, 0.00, 0.03 },
        { 0.10, 0.25, 0.50, 0.10, 0.00, 0.05 },
        { 0.05, 0.10, 0.20, 0.50, 0.10, 0.05 },
        { 0.05, 0.05, 0.15, 0.25, 0.45, 0.05 },
        { 0.02, 0.02, 0.03, 0.03, 0.10, 0.80 },
        { 0.00, 0.00, 0.00, 0.00, 0.00, 1.00 },
    };

    private static double[] ShopThinkTimes() => new[] { 900.0, 1400.0, 2200.0, 3000.0, 5000.0, 0.0 };

    /** Expected transient visits from a start state: solve (I - Q^T) v = e_start. */
    private static double[] ExpectedVisits(MarkovWorkload model, int startState)
    {
        var a = new double[Transient, Transient];
        var b = new double[Transient];
        for (int i = 0; i < Transient; i++)
        {
            // The visit count of state i is "started here" plus the flows entering i from every
            // state j, so the system is indexed by the incoming transitions: A = I - Q^T.
            for (int j = 0; j < Transient; j++) a[i, j] = (i == j ? 1.0 : 0.0) - model.Transition(j, i);
            b[i] = i == startState ? 1.0 : 0.0;
        }
        return LinearAlgebra.Solve(a, b) ?? new double[Transient];
    }

    /** Probability of visiting add-to-cart before exit, solved independently of the visit counts. */
    private static double CheckoutProbability(MarkovWorkload model)
    {
        int[] kept = { 0, 1, 2, MarkovWorkload.Checkout };
        int n = kept.Length;
        var a = new double[n, n];
        var b = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) a[i, j] = (i == j ? 1.0 : 0.0) - model.Transition(kept[i], kept[j]);
            b[i] = model.Transition(kept[i], MarkovWorkload.AddToCart);
        }
        double[]? h = LinearAlgebra.Solve(a, b);
        return h == null ? double.NaN : h[0];
    }

    public static void Main()
    {
        const double Rate = 20.0;
        const int SessionCount = 20000;
        double[,] matrix = ShopModel();
        var model = new MarkovWorkload(matrix, ShopThinkTimes(), 0x9401UL);

        // 1-2. Model validation, then the empirical transition matrix against the configured one.
        Check(!Throws(() => new MarkovWorkload(ShopModel(), ShopThinkTimes(), 1UL)), "every configured row sums to 1");
        Check(Throws(() => new MarkovWorkload(new double[,] { { 0.5, 0.5, 0, 0, 0, 0.5 }, { 0, 1, 0, 0, 0, 0 },
            { 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 0, 1 } }, ShopThinkTimes(), 1UL)),
            "a row that does not sum to 1 is rejected");

        var sessions = new List<Session>();
        for (int i = 0; i < SessionCount; i++) sessions.Add(model.GenerateSession(0, Rate));
        var report = new WorkloadReport();
        report.Add(sessions);
        double worst = 0.0;
        for (int s = 0; s < MarkovWorkload.StateCount; s++)
        {
            long total = 0;
            for (int t = 0; t < MarkovWorkload.StateCount; t++) total += report.TransitionCounts[s, t];
            if (total == 0) continue;
            for (int t = 0; t < MarkovWorkload.StateCount; t++)
            {
                double empirical = (double)report.TransitionCounts[s, t] / total;
                worst = Math.Max(worst, Math.Abs(empirical - matrix[s, t]));
            }
        }
        Check(report.Sessions.Count == SessionCount && report.Transitions > 50000 && worst < 0.02,
            "the empirical transition matrix tracks the configured one (max deviation " + worst.ToString("F4") + ")");

        // 3-4. Closed-form oracle: expected steps and per-state visit counts from (I - Q)^-1 e_start.
        double[] visits = ExpectedVisits(model, 0);
        double expectedSteps = visits.Sum();
        Check(Math.Abs(report.MeanSteps - expectedSteps) / expectedSteps < 0.03,
            "the mean session length " + report.MeanSteps.ToString("F3") + " matches the closed form "
            + expectedSteps.ToString("F3"));
        double worstVisit = 0.0;
        for (int s = 0; s < Transient; s++)
        {
            double empirical = (double)report.VisitCounts[s] / SessionCount;
            worstVisit = Math.Max(worstVisit, Math.Abs(empirical - visits[s]));
        }
        Check(worstVisit < 0.03, "per-state visit rates match the fundamental matrix (max deviation "
            + worstVisit.ToString("F4") + ")");

        // 5. Correlation check: reaching the cart is predicted by an independent hitting probability.
        double hit = CheckoutProbability(model);
        Check(!double.IsNaN(hit) && Math.Abs(report.CartFraction - hit) < 0.02,
            "the share of sessions reaching add-to-cart " + report.CartFraction.ToString("F3")
            + " matches the hitting probability " + hit.ToString("F3"));

        // 6. Arrival process: Poisson window counts and exponential gap statistics.
        double windowMs = 50000.0;
        int windows = 20;
        var counts = new int[windows];
        double clock = 0.0;
        foreach (Session session in report.Sessions)
        {
            clock += session.ArrivalGapMs;
            int bucket = (int)(clock / windowMs);
            if (bucket < windows) counts[bucket]++;
        }
        double expectedPerWindow = Rate * windowMs / 1000.0;
        double tolerance = 4.0 * Math.Sqrt(expectedPerWindow);
        Check(counts.All(c => Math.Abs(c - expectedPerWindow) <= tolerance),
            "every window holds a Poisson-plausible arrival count (expected " + expectedPerWindow.ToString("F0")
            + " +/- " + tolerance.ToString("F0") + ", min " + counts.Min() + ", max " + counts.Max()
            + ", last " + counts[windows - 1] + ")");
        double meanGap = report.TotalArrivalMs / SessionCount;
        double variance = report.Sessions.Sum(s => (s.ArrivalGapMs - meanGap) * (s.ArrivalGapMs - meanGap)) / SessionCount;
        double expectedGap = 1000.0 / Rate;
        Check(Math.Abs(meanGap - expectedGap) / expectedGap < 0.03
              && Math.Abs(Math.Sqrt(variance) - expectedGap) / expectedGap < 0.03,
            "inter-arrivals have mean " + meanGap.ToString("F2") + " ms and sd "
            + Math.Sqrt(variance).ToString("F2") + " ms against the exponential " + expectedGap.ToString("F1") + " ms");

        // 7. Think times per state follow the configured exponential means.
        double worstThink = 0.0;
        for (int s = 0; s < Transient; s++)
        {
            double observed = report.ThinkTimeSum[s] / report.ThinkTimeCount[s];
            worstThink = Math.Max(worstThink, Math.Abs(observed - ShopThinkTimes()[s]) / ShopThinkTimes()[s]);
        }
        Check(worstThink < 0.05, "per-state think times match the configured means (max error "
            + (worstThink * 100).ToString("F2") + "%)");

        // 8-9. Boundaries: a degenerate chain and a session that starts already exited.
        var degenerate = new MarkovWorkload(new double[,]
        {
            { 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 0, 0, 1 },
            { 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 0, 0, 1 },
        }, ShopThinkTimes(), 7UL);
        var degenerateReport = new WorkloadReport();
        var degenerateSessions = new List<Session>();
        for (int i = 0; i < 200; i++) degenerateSessions.Add(degenerate.GenerateSession(0, Rate));
        degenerateReport.Add(degenerateSessions);
        Check(degenerateReport.Sessions.All(s => s.States.Count == 1 && s.States[0] == 0)
              && Math.Abs(degenerateReport.MeanSteps - ExpectedVisits(degenerate, 0).Sum()) < 1e-9,
            "a degenerate chain produces single-step sessions and the closed form agrees");
        var exitedReport = new WorkloadReport();
        var exitedSessions = new List<Session>();
        for (int i = 0; i < 50; i++) exitedSessions.Add(model.GenerateSession(MarkovWorkload.Exit, Rate));
        exitedReport.Add(exitedSessions);
        Check(exitedReport.MeanSteps == 0.0 && ExpectedVisits(model, MarkovWorkload.Exit).Sum() == 0.0
              && exitedReport.Sessions.All(s => s.States.Count == 0),
            "a session that starts at the exit state has length zero and no visits");

        // 10. Reproducibility: a fixed seed replays exactly, a different seed does not.
        var sameA = new MarkovWorkload(ShopModel(), ShopThinkTimes(), 99UL).GenerateSession(0, Rate);
        var sameB = new MarkovWorkload(ShopModel(), ShopThinkTimes(), 99UL).GenerateSession(0, Rate);
        var other = new MarkovWorkload(ShopModel(), ShopThinkTimes(), 100UL).GenerateSession(0, Rate);
        Check(sameA.States.SequenceEqual(sameB.States) && sameA.ArrivalGapMs == sameB.ArrivalGapMs
              && !sameA.States.SequenceEqual(other.States),
            "a fixed seed replays identically while another seed diverges");

        Console.WriteLine("  sessions: " + SessionCount + ", transitions " + report.Transitions
            + ", mean length " + report.MeanSteps.ToString("F3") + " (closed form " + expectedSteps.ToString("F3") + ")"
            + ", cart share " + report.CartFraction.ToString("F3"));
        Console.WriteLine("PASS");
    }
}
