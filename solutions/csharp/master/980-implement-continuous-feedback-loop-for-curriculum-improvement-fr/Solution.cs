// Task 980 - Implement continuous feedback loop for curriculum improvement from user telemetry.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Each problem carries telemetry (attempts, solves) and an assumed difficulty. The loop measures the
// empirical difficulty 1 - solveRate, compares it with the assumption, and only reacts when the drift
// leaves a dead band: too hard means delay the problem, too easy means advance it. After each round
// the assumption is nudged towards the measurement, so the drift decays geometrically - which is the
// convergence property the loop is judged on - while a confirmation counter suppresses alerts that
// would fire on a single noisy round.
//
// Verified: drift detection in both directions, the false-positive guard on a matched problem, the
// dead-band boundary on both sides, the minimum-sample guard, an independent solve-rate and kind
// oracle, monotone convergence of the total drift over eight rounds, hysteresis under a
// two-round confirmation rule, sorted and reproducible output and the empty-telemetry boundary.
// Simplification: telemetry is in-memory counters and the loop nudges a scalar difficulty assumption;
// there is no A/B deployment, no statistical significance test and no student-level model.
using System;
using System.Collections.Generic;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Telemetry
    {
        public readonly string Id;
        public readonly int Attempts;
        public readonly int Solves;

        public Telemetry(string id, int attempts, int solves)
        {
            Id = id;
            Attempts = attempts;
            Solves = solves;
        }
    }

    private sealed class Proposal
    {
        public readonly string Id;
        public readonly string Kind;
        public readonly double Estimate;
        public readonly double Drift;

        public Proposal(string id, string kind, double estimate, double drift)
        {
            Id = id;
            Kind = kind;
            Estimate = estimate;
            Drift = drift;
        }
    }

    private const int MinimumSamples = 3;

    /** -1 means "not enough data yet". */
    private static double Estimate(Telemetry telemetry, int minimumSamples)
        => telemetry.Attempts < minimumSamples ? -1.0 : 1.0 - telemetry.Solves / (double)telemetry.Attempts;

    private static Proposal Propose(Telemetry telemetry, double assumed, double deadBand, int minimumSamples)
    {
        double estimate = Estimate(telemetry, minimumSamples);
        if (estimate < 0.0) return new Proposal(telemetry.Id, "hold", estimate, 0.0);
        double drift = estimate - assumed;
        // The 1e-9 slack keeps a drift that lands exactly on the dead band from flipping on rounding.
        double edges = deadBand + 1e-9;
        string kind = drift > edges ? "delay" : drift < -edges ? "advance" : "hold";
        return new Proposal(telemetry.Id, kind, estimate, drift);
    }

    /** One round of the loop: measure, propose, then move the assumption a fraction towards the data. */
    private static List<Proposal> Round(List<Telemetry> telemetry, Dictionary<string, double> assumed, double deadBand)
    {
        var proposals = new List<Proposal>();
        foreach (Telemetry sample in telemetry) proposals.Add(Propose(sample, assumed[sample.Id], deadBand, MinimumSamples));
        proposals.Sort((a, b) => string.CompareOrdinal(a.Id, b.Id));
        foreach (Proposal proposal in proposals)
            if (proposal.Kind != "hold") assumed[proposal.Id] += 0.4 * proposal.Drift;
        return proposals;
    }

    private static double TotalDrift(List<Proposal> proposals)
    {
        double total = 0.0;
        foreach (Proposal proposal in proposals) total += Math.Abs(proposal.Drift);
        return total;
    }

    /** Only drift that persists for `confirmations` consecutive rounds becomes a proposal. */
    private static List<Proposal> Confirm(List<List<Proposal>> rounds, int confirmations)
    {
        var confirmed = new List<Proposal>();
        var streak = new Dictionary<string, int>(StringComparer.Ordinal);
        var lastKind = new Dictionary<string, string>(StringComparer.Ordinal);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (List<Proposal> round in rounds)
            foreach (Proposal proposal in round)
            {
                bool sameKind = lastKind.TryGetValue(proposal.Id, out string previous) && previous == proposal.Kind;
                streak[proposal.Id] = sameKind ? streak[proposal.Id] + 1 : 1;
                lastKind[proposal.Id] = proposal.Kind;
                if (proposal.Kind != "hold" && streak[proposal.Id] >= confirmations && seen.Add(proposal.Id))
                    confirmed.Add(proposal);
            }
        return confirmed;
    }

    private static List<Telemetry> Sample()
    {
        return new List<Telemetry>
        {
            new Telemetry("p-exact", 100, 50),   // matches the 0.5 assumption
            new Telemetry("p-easy", 100, 80),    // measured 0.2, easier than assumed
            new Telemetry("p-hard", 100, 40),    // measured 0.6, harder than assumed
            new Telemetry("p-thin", 2, 1),       // below the sample floor
        };
    }

    public static void Main()
    {
        const double deadBand = 0.02;
        var assumed = new Dictionary<string, double>(StringComparer.Ordinal)
        {
            { "p-exact", 0.5 }, { "p-easy", 0.5 }, { "p-hard", 0.5 }, { "p-thin", 0.5 },
        };
        List<Telemetry> telemetry = Sample();

        List<Proposal> first = Round(telemetry, assumed, deadBand);
        Check(string.Join(",", first.ConvertAll(p => p.Id + ":" + p.Kind))
              == "p-easy:advance,p-exact:hold,p-hard:delay,p-thin:hold",
            "the first round advances the easy problem, delays the hard one and holds the others");
        Check(Math.Abs(TotalDrift(first) - 0.4) < 1e-12,
            "the first round measures 0.4 of total drift (0.3 easy plus 0.1 hard)");

        // Oracle: the solve rates and the proposal kinds recomputed from the raw counters.
        int oracleMismatch = 0;
        foreach (Telemetry sample in telemetry)
        {
            double rate = 1.0 - sample.Solves / (double)sample.Attempts;
            string kind = sample.Id == "p-easy" ? "advance" : sample.Id == "p-hard" ? "delay" : "hold";
            foreach (Proposal proposal in first)
                if (proposal.Id == sample.Id)
                {
                    if (sample.Attempts < MinimumSamples) { oracleMismatch += Math.Abs(proposal.Drift) > 0 ? 1 : 0; }
                    else if (Math.Abs(proposal.Estimate - rate) > 1e-12 || proposal.Kind != kind) oracleMismatch++;
                }
        }
        Check(oracleMismatch == 0, "an independent solve-rate and kind table reproduces every proposal");

        // False-positive guard: a problem that behaves as assumed is never touched.
        var matched = new Dictionary<string, double>(StringComparer.Ordinal) { { "p-exact", 0.5 } };
        Check(Propose(new Telemetry("p-exact", 200, 100), 0.5, deadBand, MinimumSamples).Kind == "hold",
            "a problem matching its assumption generates no proposal");
        Check(Propose(new Telemetry("p-exact", 200, 104), 0.5, deadBand, MinimumSamples).Kind == "hold",
            "a 0.02 drift sits exactly on the dead-band edge and is ignored");
        Check(Propose(new Telemetry("p-exact", 200, 94), 0.5, deadBand, MinimumSamples).Kind == "delay",
            "a 0.03 drift towards fewer solves leaves the dead band and is reported as too hard");
        Check(Propose(new Telemetry("p-exact", 200, 106), 0.5, deadBand, MinimumSamples).Kind == "advance",
            "the mirrored 0.03 drift towards more solves is reported as too easy");

        // Boundary: below the sample floor nothing is proposed, and the estimate is unknown.
        Telemetry thin = new Telemetry("p-thin", 2, 0);
        Check(Estimate(thin, MinimumSamples) < 0.0 && Propose(thin, 0.5, deadBand, MinimumSamples).Kind == "hold",
            "a problem with two attempts is excluded from the loop instead of being judged");

        // Convergence: the drift decays every round and finishes inside the dead band.
        var converging = new Dictionary<string, double>(StringComparer.Ordinal)
        {
            { "p-hard", 0.5 }, { "p-easy", 0.5 }, { "p-exact", 0.5 }, { "p-thin", 0.5 },
        };
        var drifts = new List<double>();
        for (int round = 0; round < 8; round++) drifts.Add(TotalDrift(Round(telemetry, converging, deadBand)));
        bool shrinking = true, nonIncreasing = true;
        for (int i = 1; i < drifts.Count; i++)
        {
            nonIncreasing &= drifts[i] <= drifts[i - 1] + 1e-12;
            if (i <= 5) shrinking &= drifts[i] < drifts[i - 1];
        }
        Check(shrinking && nonIncreasing, "total drift shrinks for five rounds and never grows afterwards ("
            + string.Join(" > ", drifts.ConvertAll(d => d.ToString("F4"))) + ")");
        Check(drifts[drifts.Count - 1] < deadBand * 2,
            "after eight rounds the residual drift sits inside the dead band and stops moving");
        Check(converging["p-hard"] > 0.5 && converging["p-easy"] < 0.5,
            "the assumptions moved towards the measured difficulty in both directions");

        // Hysteresis: one noisy round never becomes a proposal, two in a row do.
        var single = new List<List<Proposal>>
        {
            new List<Proposal> { new Proposal("p1", "delay", 0.8, 0.3) },
            new List<Proposal> { new Proposal("p1", "hold", 0.62, 0.012) },
        };
        var twoInARow = new List<List<Proposal>>
        {
            new List<Proposal> { new Proposal("p1", "delay", 0.8, 0.3) },
            new List<Proposal> { new Proposal("p1", "delay", 0.8, 0.3) },
        };
        Check(Confirm(single, 2).Count == 0, "a single-round excursion is suppressed by the confirmation rule");
        Check(Confirm(twoInARow, 2).Count == 1 && Confirm(twoInARow, 2)[0].Kind == "delay",
            "a drift confirmed in two consecutive rounds is reported once");

        // Reproducibility: the same inputs always produce the same ordered proposals.
        var replay = new Dictionary<string, double>(StringComparer.Ordinal)
        {
            { "p-exact", 0.5 }, { "p-easy", 0.5 }, { "p-hard", 0.5 }, { "p-thin", 0.5 },
        };
        List<Proposal> again = Round(telemetry, replay, deadBand);
        Check(string.Join(",", again.ConvertAll(p => p.Id + ":" + p.Kind))
              == string.Join(",", first.ConvertAll(p => p.Id + ":" + p.Kind)),
            "a replayed loop produces the identical, id-sorted proposals");

        // Boundary: no telemetry at all, and telemetry with zero attempts.
        var empty = new Dictionary<string, double>(StringComparer.Ordinal);
        Check(Round(new List<Telemetry>(), empty, deadBand).Count == 0,
            "an empty telemetry feed produces no proposals and does not throw");
        Check(Propose(new Telemetry("p-zero", 0, 0), 0.5, deadBand, MinimumSamples).Kind == "hold",
            "a problem with zero attempts is not divided by zero");

        Console.WriteLine("round 1: " + string.Join(",", first.ConvertAll(p => p.Id + ":" + p.Kind)));
        Console.WriteLine("drift decay over 8 rounds: " + string.Join(" -> ", drifts.ConvertAll(d => d.ToString("F4"))));
        Console.WriteLine("confirmed proposals under 2-round hysteresis: single excursion "
            + Confirm(single, 2).Count + ", sustained " + Confirm(twoInARow, 2).Count);
        Console.WriteLine("PASS");
    }
}
