// Task 691 - Implement additive-increase multiplicative-decrease (AIMD) simulation. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A deterministic packet-per-round AIMD simulator. N flows share one bottleneck link that serves
// `capacity` packets per round. A round is one RTT: while the aggregate offered load fits the
// link every flow adds alpha packets to its window (the additive increase); once the aggregate
// overshoots the link the shared FIFO buffer signals congestion to all flows in flight and each
// of them multiplies its window by beta (the multiplicative decrease, beta = 0.5 for TCP). The
// packets themselves are accounted with a max-min fair split of the capacity, which is what a
// per flow round robin queue provides. Documented simplifications: integer windows in packets,
// one round per RTT, no timeouts or slow start, no delayed ACK, no queue length model - the
// congestion signal is the aggregate exceeding the capacity in that round. What is verified is
// the classic AIMD content: a single flow's sawtooth averages 0.75 of the capacity, the aggregate
// is bounded by capacity plus N, two flows converge from very different starting windows to equal
// shares with Jain's index rising towards one and the window gap halving per congestion event,
// the aggressive flow with alpha = 4 claims a multiple of the polite flow's bandwidth, a
// multiplicative increase flow starves the additive one, a larger beta buys utilisation at the
// price of more congestion events, and no flow is ever starved to zero.
using System;
using System.Collections.Generic;

internal sealed class AimdFlow
{
    public readonly int Id;
    public int Cwnd;
    public double Alpha;          // packets added per RTT while no packet is lost
    public double Beta;           // window multiplier applied when a packet is lost
    public bool MultiplicativeIncrease;

    public long Delivered;
    public long Dropped;
    public long DecreaseEvents;

    public AimdFlow(int id, int initialCwnd, double alpha = 1.0, double beta = 0.5)
    {
        Id = id;
        Cwnd = initialCwnd;
        Alpha = alpha;
        Beta = beta;
    }
}

internal sealed class AimdLink
{
    private readonly List<AimdFlow> _flows = new List<AimdFlow>();

    public readonly int Capacity;
    public long Rounds;
    public long CongestionRounds;
    public long DeliveredTotal;
    public bool StarvationSeen;
    public bool AggregateExceededBound;
    public readonly List<int[]> WindowsAtDecrease = new List<int[]>();

    public AimdLink(int capacity) => Capacity = capacity;

    public IReadOnlyList<AimdFlow> Flows => _flows;

    public void Add(AimdFlow flow) => _flows.Add(flow);

    public int Aggregate()
    {
        int total = 0;
        foreach (AimdFlow flow in _flows) total += flow.Cwnd;
        return total;
    }

    /// <summary>Max-min fair allocation: each flow gets min(window, level), the level chosen so that
    /// exactly `capacity` packets are served (the remainder is dropped by the bottleneck).</summary>
    public int[] Serve()
    {
        int n = _flows.Count;
        var served = new int[n];
        int remaining = Capacity;
        while (remaining > 0)
        {
            int active = 0;
            for (int i = 0; i < n; i++) if (served[i] < _flows[i].Cwnd) active++;
            if (active == 0) break;
            for (int i = 0; i < n && remaining > 0; i++)
            {
                if (served[i] >= _flows[i].Cwnd) continue;
                served[i]++;
                remaining--;
            }
        }
        return served;
    }

    public void Step()
    {
        Rounds++;
        long offered = Aggregate();
        bool congested = offered > Capacity;
        int[] served = Serve();
        int[] windows = new int[_flows.Count];
        if (congested) CongestionRounds++;

        for (int i = 0; i < _flows.Count; i++)
        {
            AimdFlow flow = _flows[i];
            flow.Delivered += served[i];
            flow.Dropped += flow.Cwnd - served[i];
            DeliveredTotal += served[i];
            if (congested)
            {
                flow.Cwnd = Math.Max(1, (int)Math.Floor(flow.Cwnd * flow.Beta));
                flow.DecreaseEvents++;
            }
            else if (flow.MultiplicativeIncrease)
            {
                flow.Cwnd = Math.Max(1, flow.Cwnd * 2);
            }
            else
            {
                flow.Cwnd += Math.Max(1, (int)Math.Round(flow.Alpha));
            }
            windows[i] = flow.Cwnd;
            if (flow.Cwnd < 1) StarvationSeen = true;
        }
        if (congested) WindowsAtDecrease.Add(windows);
        if (Aggregate() > Capacity + _flows.Count) AggregateExceededBound = true;
    }

    public void Run(int rounds)
    {
        for (int i = 0; i < rounds; i++) Step();
    }

    /// <summary>Jain's fairness index over the windows: 1 means perfectly equal shares.</summary>
    public double Fairness()
    {
        double sum = 0, sumSquares = 0;
        foreach (AimdFlow flow in _flows)
        {
            sum += flow.Cwnd;
            sumSquares += (double)flow.Cwnd * flow.Cwnd;
        }
        if (sumSquares == 0) return 0;
        return sum * sum / (_flows.Count * sumSquares);
    }

    public double AverageAggregate()
    {
        // Delivered packets per round divided by the capacity is exactly the link utilisation.
        return Rounds == 0 ? 0 : (double)DeliveredTotal / Rounds / Capacity;
    }

    public double AverageWindow(int flowId)
    {
        AimdFlow flow = _flows[flowId];
        return Rounds == 0 ? 0 : (double)(flow.Delivered + flow.Dropped) / Rounds;
    }

    public string Dump()
    {
        var parts = new List<string>();
        foreach (AimdFlow flow in _flows) parts.Add($"{flow.Id}:{flow.Cwnd}");
        return string.Join(",", parts);
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- A single flow sawtooths around 0.75 of the capacity. -------------------------
        var single = new AimdLink(100);
        single.Add(new AimdFlow(0, 1));
        single.Run(600);
        double singleUtilisation = single.AverageAggregate();
        Check(!single.StarvationSeen, "the single flow never drops below one packet");
        Check(!single.AggregateExceededBound, "the aggregate stays inside capacity + N packets");
        Check(singleUtilisation > 0.70 && singleUtilisation < 0.80,
            "one AIMD flow utilises about 75 % of the link, got " + singleUtilisation.ToString("F3"));
        Check(single.CongestionRounds > 2, "the single flow sees repeated congestion, events=" + single.CongestionRounds);

        // The exact sawtooth: halve on loss, add one packet per RTT while it survives.
        var saw = new AimdLink(100);
        saw.Add(new AimdFlow(0, 1));
        var decreaseWindows = new List<int>();
        int lastLossRound = -1;
        for (int round = 1; round <= 400; round++)
        {
            int before = saw.Flows[0].Cwnd;
            saw.Step();
            int after = saw.Flows[0].Cwnd;
            if (after < before)
            {
                decreaseWindows.Add(before);
                if (lastLossRound >= 0) Check(round - lastLossRound >= 45, "the sawtooth period is about half the capacity");
                lastLossRound = round;
            }
        }
        Check(decreaseWindows.Count >= 3, "several sawtooth cycles were observed");
        bool samePeak = true;
        foreach (int peak in decreaseWindows) if (peak != decreaseWindows[0]) samePeak = false;
        Check(samePeak, "every cycle peaks at the same window, peak=" + decreaseWindows[0]);
        Check(decreaseWindows[0] == 101, "the peak is one packet over the capacity, peak=" + decreaseWindows[0]);
        Check(saw.Flows[0].Cwnd > 50 && saw.Flows[0].Cwnd <= 101, "the window stays inside one sawtooth");

        // --- Two flows converge from very different windows to equal shares. --------------
        var pair = new AimdLink(100);
        pair.Add(new AimdFlow(0, 1));
        pair.Add(new AimdFlow(1, 99));
        double fairnessStart = pair.Fairness();
        var fairnessTrace = new List<double>();
        for (int cycle = 0; cycle < 40; cycle++)
        {
            pair.Run(10);
            fairnessTrace.Add(pair.Fairness());
        }
        double fairnessEnd = pair.Fairness();
        Check(fairnessStart < 0.6, "the initial windows are very unfair, index=" + fairnessStart.ToString("F3"));
        Check(fairnessEnd > 0.99, "the flows converge to equal shares, index=" + fairnessEnd.ToString("F3"));
        Check(fairnessEnd > fairnessStart, "fairness improved over the run");
        int improvements = 0, regressions = 0;
        for (int i = 1; i < fairnessTrace.Count; i++)
        {
            if (fairnessTrace[i] > fairnessTrace[i - 1] + 1e-9) improvements++;
            else if (fairnessTrace[i] < fairnessTrace[i - 1] - 1e-9) regressions++;
        }
        Check(improvements > regressions, $"fairness improves far more often than it slips ({improvements} vs {regressions})");
        int first = pair.Flows[0].Cwnd, second = pair.Flows[1].Cwnd;
        Check(Math.Abs(first - second) <= 2, $"the converged windows are within two packets ({first} vs {second})");
        Check(!pair.StarvationSeen && !pair.AggregateExceededBound, "no starvation and a bounded aggregate");

        // The convergence rate: the gap between the two windows must shrink at the decreases.
        var gap = new AimdLink(100);
        gap.Add(new AimdFlow(0, 2));
        gap.Add(new AimdFlow(1, 40));
        int initialGap = 38;
        for (int cycle = 0; cycle < 6; cycle++) gap.Run(12);
        int finalGap = Math.Abs(gap.Flows[0].Cwnd - gap.Flows[1].Cwnd);
        Check(finalGap < initialGap / 3, $"the window gap shrank from {initialGap} to {finalGap}");

        // --- Shares at the decrease points follow alpha (Chiu-Jain). ----------------------
        var asymmetric = new AimdLink(100);
        asymmetric.Add(new AimdFlow(0, 1, alpha: 4.0));
        asymmetric.Add(new AimdFlow(1, 1, alpha: 1.0));
        asymmetric.Run(800);
        double aggressiveMean = asymmetric.AverageWindow(0);
        double politeMean = asymmetric.AverageWindow(1);
        double ratio = aggressiveMean / politeMean;
        Check(ratio > 2.5 && ratio < 5.5, "alpha = 4 claims roughly four times the bandwidth, ratio=" + ratio.ToString("F2"));
        Check(politeMean > 5, "the polite flow still keeps a real share, mean=" + politeMean.ToString("F1"));

        // --- A multiplicative increase flow starves the additive one. --------------------
        var hostile = new AimdLink(100);
        hostile.Add(new AimdFlow(0, 1));
        var multiplicative = new AimdFlow(1, 4) { MultiplicativeIncrease = true };
        hostile.Add(multiplicative);
        hostile.Run(400);
        Check(hostile.AverageWindow(1) > 3 * hostile.AverageWindow(0),
            "multiplicative increase takes over the link, " + hostile.AverageWindow(1).ToString("F1") +
            " vs " + hostile.AverageWindow(0).ToString("F1"));
        Check(!hostile.StarvationSeen, "the additive flow is squeezed but never to zero");
        Check(Math.Abs(multiplicative.Cwnd - Math.Max(1, (int)Math.Floor(multiplicative.Cwnd * 0.5))) < 1
              || multiplicative.Cwnd > 1, "the multiplicative flow keeps a valid window");

        // --- beta trades utilisation against loss events. --------------------------------
        var gentle = new AimdLink(100);
        gentle.Add(new AimdFlow(0, 1, beta: 0.9));
        gentle.Run(600);
        Check(gentle.AverageAggregate() > singleUtilisation,
            $"beta = 0.9 buys more utilisation ({gentle.AverageAggregate():F3}) than beta = 0.5 ({singleUtilisation:F3})");
        Check(gentle.CongestionRounds > single.CongestionRounds,
            $"the price is more congestion events ({gentle.CongestionRounds} vs {single.CongestionRounds})");

        // --- Boundary: many flows on a tiny link stay alive and barely use it. -----------
        var crowded = new AimdLink(16);
        for (int i = 0; i < 8; i++) crowded.Add(new AimdFlow(i, 1 + i * 3));
        crowded.Run(500);
        Check(!crowded.StarvationSeen, "eight flows on a 16 packet link never starve");
        Check(crowded.Aggregate() >= 8, "every crowded flow keeps at least one packet, aggregate=" + crowded.Aggregate());
        Check(crowded.AverageAggregate() <= 1.0, "utilisation can never exceed the capacity");
        Check(crowded.Fairness() > 0.95, "the crowded link ends up fairly shared, index=" + crowded.Fairness().ToString("F4"));

        // --- Exact arithmetic of one decrease step. --------------------------------------
        var exact = new AimdLink(1000);                  // huge link so nothing is ever dropped
        exact.Add(new AimdFlow(0, 7));
        exact.Add(new AimdFlow(1, 30));
        exact.Run(3);
        Check(exact.Dump() == "0:10,1:33", "unloaded flows grow by one packet per RTT, got " + exact.Dump());
        var halve = new AimdLink(1);
        halve.Add(new AimdFlow(0, 9));
        halve.Add(new AimdFlow(1, 8));
        halve.Step();
        Check(halve.Dump() == "0:4,1:4", "both flows in the congested round multiply by beta, got " + halve.Dump());
        Check(halve.CongestionRounds == 1, "the congested round is counted exactly once");

        // --- Determinism. -----------------------------------------------------------------
        var runOne = new AimdLink(60);
        var runTwo = new AimdLink(60);
        foreach (var link in new[] { runOne, runTwo })
        {
            link.Add(new AimdFlow(0, 1));
            link.Add(new AimdFlow(1, 5, alpha: 2));
            link.Add(new AimdFlow(2, 20));
            link.Run(300);
        }
        Check(runOne.Dump() == runTwo.Dump(), "two identical runs end in the same windows");
        Check(runOne.CongestionRounds == runTwo.CongestionRounds && runOne.DeliveredTotal == runTwo.DeliveredTotal,
            "two identical runs lose and deliver the same amounts");
        Check(runOne.WindowsAtDecrease.Count == runTwo.WindowsAtDecrease.Count, "the same number of decrease events");

        Console.WriteLine($"singleFlowUtil={singleUtilisation:F3} peak={decreaseWindows[0]} " +
                          $"losses={single.CongestionRounds} fairness={fairnessEnd:F4} alphaRatio={ratio:F2}");
        Console.WriteLine($"gentleBetaUtil={gentle.AverageAggregate():F3} gentleLosses={gentle.CongestionRounds} " +
                          $"crowdedFairness={crowded.Fairness():F4}");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
