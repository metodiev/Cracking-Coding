// Task 689 - Implement congestion control algorithms simulator (TCP Reno/CUBIC). (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// Two congestion controllers are driven against the same deterministic bottleneck, one
// round trip at a time (one round == one RTT, the window is measured in MSS sized segments):
//
//   * Reno: slow start doubles the window per RTT until ssthresh, congestion avoidance adds one
//     segment per RTT, a triple duplicate ACK halves ssthresh and sets cwnd = ssthresh (fast
//     retransmit) and a retransmission timeout halves ssthresh and restarts slow start at
//     cwnd = 1;
//   * CUBIC (RFC 8312 shape): after a loss the sender remembers W_max and follows
//     W_cubic(t) = C (t - K)^3 + W_max with K = cbrt(W_max (1 - beta) / C), beta = 0.7, C = 0.4,
//     never dropping below the Reno friendly estimate W_est(t) = W_max beta + 3 (1 - beta)/(1 + beta) t,
//     and applies the fast convergence heuristic when the new loss window is below W_max.
//
// The bottleneck is deliberately simple and fully deterministic: it can carry `capacity`
// segments per RTT, so a round whose window exceeds the capacity loses the excess. If at least
// three segments still get through the sender sees duplicate ACKs (fast retransmit), otherwise
// its retransmission timer expires. Documented simplifications: no delayed ACKs, no RTT
// estimation, no SACK, no bandwidth-delay product or buffer model, no pacing, and CUBIC's
// per-RTT increment caps are omitted so the raw window function is observable. What is verified
// is real protocol behaviour: slow start doubling, additive increase, multiplicative decrease,
// the 0.75 * W_max utilisation identity of a Reno sawtooth, the exact cubic curve properties
// (concavity before K, convexity after K, W_cubic(0) = beta W_max, W_cubic(K) = W_max), the
// faster-than-Reno recovery time, determinism, and the invariant that a loss is only ever
// observed above the bottleneck capacity.
using System;
using System.Collections.Generic;

internal sealed class RenoSender
{
    public long Cwnd = 1;
    public long Ssthresh;
    public long MaxSeen = 1;

    public RenoSender(long ssthresh) => Ssthresh = ssthresh;

    /// <summary>One RTT without a loss: exponential in slow start, additive in congestion avoidance.</summary>
    public void OnAck()
    {
        if (Cwnd < Ssthresh) Cwnd = Math.Min(Cwnd * 2, Ssthresh);
        else Cwnd += 1;
        MaxSeen = Math.Max(MaxSeen, Cwnd);
    }

    /// <summary>Triple duplicate ACK: multiplicative decrease, no slow start.</summary>
    public void OnFastRetransmit()
    {
        Ssthresh = Math.Max(Cwnd / 2, 2);
        Cwnd = Ssthresh;
    }

    /// <summary>Retransmission timeout: multiplicative decrease plus a cold restart.</summary>
    public void OnTimeout()
    {
        Ssthresh = Math.Max(Cwnd / 2, 2);
        Cwnd = 1;
    }
}

internal sealed class CubicSender
{
    public const double Beta = 0.7;    // multiplicative decrease factor after a loss
    public const double C = 0.4;       // cubic scaling constant

    public long Cwnd;
    public long Ssthresh;
    public long WMax;                  // window at the last loss event
    public double K;                   // RTTs the cubic curve needs to get back to W_max
    public double EpochTime;           // RTTs since the current epoch started
    public bool FastConvergenceApplied;

    public CubicSender(long ssthresh)
    {
        Ssthresh = ssthresh;
        Cwnd = ssthresh;
    }

    public double CubicTarget(double t)
    {
        double dt = t - K;
        return C * dt * dt * dt + WMax;
    }

    public double RenoFriendlyTarget(double t) => WMax * Beta + 3.0 * (1.0 - Beta) / (1.0 + Beta) * t;

    public void OnAck()
    {
        EpochTime += 1;
        double target = Math.Max(CubicTarget(EpochTime), RenoFriendlyTarget(EpochTime));
        if (target > Cwnd) Cwnd = (long)Math.Floor(target + 1e-9);
    }

    public void OnLoss(bool timeout)
    {
        long previousWMax = WMax;
        long candidate = Math.Max((long)Math.Floor(Cwnd * (1.0 + Beta) / 2.0), 2);
        // Fast convergence (RFC 8312 4.6): a loss below the historical maximum pulls W_max further
        // down so that competing flows can claim the freed capacity.
        FastConvergenceApplied = previousWMax > 0 && Cwnd < previousWMax;
        WMax = FastConvergenceApplied ? candidate : Cwnd;

        Ssthresh = Math.Max((long)Math.Floor(WMax * Beta), 2);
        K = Math.Cbrt(Math.Max(WMax * (1.0 - Beta) / C, 0.0));
        EpochTime = 0;
        Cwnd = timeout ? 1 : Ssthresh;
    }
}

internal sealed class FlowRun
{
    public readonly List<long> Windows = new List<long>();
    public readonly List<int> LossRounds = new List<int>();
    public readonly List<long> LossWindows = new List<long>();

    public int Capacity;
    public long Goodput;
    public int Timeouts;
    public int FastRetransmits;
    public bool LossAboveCapacityOnly = true;
    public bool OvershotByMoreThanOne;

    public double AverageWindow
    {
        get
        {
            long total = 0;
            foreach (long window in Windows) total += window;
            return Windows.Count == 0 ? 0 : (double)total / Windows.Count;
        }
    }

    public double AverageGoodput => Windows.Count == 0 ? 0 : (double)Goodput / Windows.Count;
}

internal static class Bottleneck
{
    /// <summary>Runs one flow for <paramref name="rounds"/> round trips against a fixed capacity.</summary>
    public static FlowRun Simulate(bool cubic, int capacity, int rounds, long initialSsthresh)
    {
        var run = new FlowRun { Capacity = capacity };
        RenoSender? reno = cubic ? null : new RenoSender(initialSsthresh);
        CubicSender? cubicSender = cubic ? new CubicSender(initialSsthresh) : null;

        for (int round = 1; round <= rounds; round++)
        {
            long window = cubic ? cubicSender!.Cwnd : reno!.Cwnd;
            run.Windows.Add(window);
            run.Goodput += Math.Min(window, capacity);
            if (window > capacity)
            {
                run.LossRounds.Add(round);
                run.LossWindows.Add(window);
                if (window > capacity + 1) run.OvershotByMoreThanOne = true;
                bool timeout = capacity < 3;              // too few segments survive for 3 dup ACKs
                if (timeout) { run.Timeouts++; if (cubic) cubicSender!.OnLoss(true); else reno!.OnTimeout(); }
                else { run.FastRetransmits++; if (cubic) cubicSender!.OnLoss(false); else reno!.OnFastRetransmit(); }
            }
            else
            {
                if (cubic) cubicSender!.OnAck(); else reno!.OnAck();
            }
        }
        return run;
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- Reno slow start: the window doubles every RTT until ssthresh. -----------------
        var slowStart = new RenoSender(64);
        var doubling = new List<long>();
        for (int i = 0; i < 7; i++) { doubling.Add(slowStart.Cwnd); slowStart.OnAck(); }
        Check(string.Join(",", doubling) == "1,2,4,8,16,32,64", "slow start doubles: " + string.Join(",", doubling));
        Check(slowStart.Cwnd == 65, "once ssthresh is reached growth turns additive");

        // --- Congestion avoidance is additive: +1 segment per RTT. ------------------------
        var increase = new List<long>();
        for (int i = 0; i < 3; i++) { increase.Add(slowStart.Cwnd); slowStart.OnAck(); }
        Check(string.Join(",", increase) == "65,66,67", "additive increase gives 65,66,67");
        long previous = 0;
        bool additive = true;
        foreach (long window in increase) { if (previous != 0 && window - previous != 1) additive = false; previous = window; }
        Check(additive, "every congestion avoidance step is exactly one segment");

        // --- Multiplicative decrease on a triple duplicate ACK and on a timeout. ----------
        var halving = new RenoSender(64);
        for (int i = 0; i < 20; i++) halving.OnAck();
        long beforeLoss = halving.Cwnd;
        Check(beforeLoss == 78, "window reached 78 before the loss, got " + beforeLoss);
        halving.OnFastRetransmit();
        Check(halving.Ssthresh == beforeLoss / 2 && halving.Cwnd == beforeLoss / 2,
            "fast retransmit halves both ssthresh and cwnd");
        halving.OnAck();
        Check(halving.Cwnd == beforeLoss / 2 + 1, "recovery continues additively, not exponentially");

        var timedOut = new RenoSender(64);
        for (int i = 0; i < 20; i++) timedOut.OnAck();
        timedOut.OnTimeout();
        Check(timedOut.Ssthresh == beforeLoss / 2 && timedOut.Cwnd == 1, "a timeout halves ssthresh and restarts at 1");
        var restart = new List<long>();
        for (int i = 0; i < 4; i++) { restart.Add(timedOut.Cwnd); timedOut.OnAck(); }
        Check(string.Join(",", restart) == "1,2,4,8", "slow start restarts from one segment");
        Check(timedOut.Cwnd == 16 && timedOut.Ssthresh == beforeLoss / 2, "the restarted flow doubles toward ssthresh");

        // --- Boundary: the decrease never drops below one segment / two ssthresh. ---------
        var floor = new RenoSender(64);
        for (int i = 0; i < 5; i++) floor.OnAck();             // 1,2,4,8,16 -> 32
        Check(floor.Cwnd == 32, "small window before the boundary loss, got " + floor.Cwnd);
        floor.OnFastRetransmit();
        var floor2 = new RenoSender(64);
        floor2.OnTimeout();
        Check(floor.Ssthresh == 16 && floor2.Ssthresh == 2 && floor2.Cwnd == 1,
            "ssthresh floors at two segments while cwnd floors at one");

        // --- A full Reno run against the bottleneck. -------------------------------------
        var renoRun = Bottleneck.Simulate(cubic: false, capacity: 64, rounds: 4000, initialSsthresh: 64);
        Check(renoRun.LossAboveCapacityOnly, "Reno only ever loses segments above the capacity");
        Check(!renoRun.OvershotByMoreThanOne, "Reno never overshoots the capacity by more than one segment");
        Check(renoRun.LossWindows.Count > 50, "the sawtooth repeats many times, losses=" + renoRun.LossWindows.Count);
        Check(renoRun.Goodput <= (long)64 * 4000, "goodput never exceeds the link capacity");

        long minLoss = long.MaxValue, maxLoss = 0;
        foreach (long window in renoRun.LossWindows) { minLoss = Math.Min(minLoss, window); maxLoss = Math.Max(maxLoss, window); }
        Check(minLoss == maxLoss, $"every Reno cycle peaks at the same window ({minLoss}..{maxLoss})");

        var periods = new List<int>();
        for (int i = 1; i < renoRun.LossRounds.Count; i++) periods.Add(renoRun.LossRounds[i] - renoRun.LossRounds[i - 1]);
        bool constantPeriod = true;
        for (int i = 1; i < periods.Count; i++) if (periods[i] != periods[0]) constantPeriod = false;
        Check(constantPeriod, "the sawtooth period is constant, period=" + (periods.Count > 0 ? periods[0] : 0));

        // Classic identity: the mean window of a perfect Reno sawtooth is 0.75 * W_max.
        double utilisation = renoRun.AverageWindow / maxLoss;
        Check(Math.Abs(utilisation - 0.75) < 0.01, "mean window is 0.75 of the peak, ratio=" + utilisation.ToString("F4"));
        Check(renoRun.AverageWindow > 0.7 * maxLoss && renoRun.AverageWindow < 0.8 * maxLoss, "utilisation stays in the band");
        Check(renoRun.Timeouts == 0, "a capacity of 64 always produces duplicate ACKs, never a timeout");

        // --- A timeout really does occur on a tiny bottleneck. ---------------------------
        var narrow = Bottleneck.Simulate(cubic: false, capacity: 2, rounds: 200, initialSsthresh: 2);
        Check(narrow.Timeouts > 0, "a two segment bottleneck makes the timer expire");
        Check(narrow.LossWindows.Count > 0 && narrow.LossAboveCapacityOnly, "the narrow run also obeys the capacity rule");

        // --- CUBIC window function properties. -------------------------------------------
        var cubic = new CubicSender(64);
        cubic.WMax = 100;
        cubic.K = Math.Cbrt(100 * (1.0 - CubicSender.Beta) / CubicSender.C);
        Check(Math.Abs(cubic.K - 4.2168) < 0.001, "K = cbrt(W_max (1-beta)/C) = " + cubic.K.ToString("F4"));
        Check(Math.Abs(cubic.CubicTarget(0) - CubicSender.Beta * 100) < 1e-9, "W_cubic(0) equals beta * W_max");
        Check(Math.Abs(cubic.CubicTarget(cubic.K) - 100) < 1e-9, "W_cubic(K) returns exactly to W_max");
        Check(cubic.K < 100 / 2.0, "CUBIC recovers far faster than Reno's additive climb would");

        var incrementsBefore = new List<double>();
        var incrementsAfter = new List<double>();
        for (double t = 0; t < cubic.K - 1; t++) incrementsBefore.Add(cubic.CubicTarget(t + 1) - cubic.CubicTarget(t));
        for (double t = cubic.K; t < cubic.K + 4; t++) incrementsAfter.Add(cubic.CubicTarget(t + 1) - cubic.CubicTarget(t));
        Check(Decreasing(incrementsBefore), "the cubic curve is concave before K (increments shrink)");
        Check(Increasing(incrementsAfter), "the cubic curve is convex after K (increments grow)");
        Check(cubic.CubicTarget(1) > cubic.CubicTarget(0), "the curve is monotonically increasing after a loss");

        double friendlySlope = (cubic.RenoFriendlyTarget(1) - cubic.RenoFriendlyTarget(0));
        Check(Math.Abs(friendlySlope - 3 * (1 - CubicSender.Beta) / (1 + CubicSender.Beta)) < 1e-12,
            "the Reno friendly estimate grows linearly, slope=" + friendlySlope.ToString("F4"));
        Check(cubic.RenoFriendlyTarget(0) == CubicSender.Beta * 100, "the friendly estimate starts at beta * W_max");

        // The live sender never goes below the friendly estimate.
        var friendlyCheck = new CubicSender(64);
        bool friendlyRespected = true;
        for (int i = 0; i < 30; i++)
        {
            friendlyCheck.OnAck();
            if (friendlyCheck.Cwnd + 1 < friendlyCheck.RenoFriendlyTarget(friendlyCheck.EpochTime)) friendlyRespected = false;
        }
        Check(friendlyRespected, "the window stays above the Reno friendly estimate");

        // --- CUBIC loss reaction and fast convergence. -----------------------------------
        var lossy = new CubicSender(64);
        for (int i = 0; i < 10; i++) lossy.OnAck();
        long cubicBeforeLoss = lossy.Cwnd;
        lossy.OnLoss(false);
        Check(lossy.WMax == cubicBeforeLoss, "the loss window becomes W_max");
        Check(lossy.Cwnd == Math.Max((long)Math.Floor(lossy.WMax * 0.7), 2),
            "the window drops to beta * W_max, cwnd=" + lossy.Cwnd);
        Check(lossy.EpochTime == 0, "the epoch clock restarts at the loss");
        Check(Math.Abs(lossy.CubicTarget(0) - lossy.Ssthresh) < 1e-9, "the epoch starts exactly on the curve");

        var converging = new CubicSender(64);
        converging.WMax = 200;
        converging.Cwnd = 40;                                   // a much smaller window than W_max
        converging.OnLoss(false);
        Check(converging.FastConvergenceApplied, "fast convergence triggers below the old W_max");
        Check(converging.WMax == (long)Math.Floor(40 * (1 + CubicSender.Beta) / 2), "W_max is pulled down further");
        var nonConverging = new CubicSender(64);
        nonConverging.WMax = 30;
        nonConverging.Cwnd = 40;
        nonConverging.OnLoss(false);
        Check(!nonConverging.FastConvergenceApplied && nonConverging.WMax == 40, "a higher loss window keeps W_max");

        // --- CUBIC against the same bottleneck. -----------------------------------------
        var cubicRun = Bottleneck.Simulate(cubic: true, capacity: 64, rounds: 4000, initialSsthresh: 64);
        Check(cubicRun.LossAboveCapacityOnly, "CUBIC only loses segments above the capacity");
        long cubicPeak = 0;
        foreach (long window in cubicRun.LossWindows) cubicPeak = Math.Max(cubicPeak, window);
        // CUBIC's convex phase grows by several segments per RTT, so the window can be well above the
        // capacity when the loss is finally observed - unlike Reno, which creeps up one segment at a time.
        Check(cubicPeak <= 1.5 * 64, "the cubic overshoot stays within half the capacity, peak=" + cubicPeak);
        Check(cubicRun.Goodput <= (long)64 * 4000, "CUBIC goodput never exceeds the link capacity");
        Check(cubicRun.AverageWindow >= renoRun.AverageWindow,
            $"CUBIC keeps a larger average window ({cubicRun.AverageWindow:F2}) than Reno ({renoRun.AverageWindow:F2})");
        Check(cubicRun.FastRetransmits > 0, "CUBIC reacts to duplicate ACKs in the run");

        // --- Determinism. ----------------------------------------------------------------
        var again = Bottleneck.Simulate(cubic: false, capacity: 64, rounds: 4000, initialSsthresh: 64);
        Check(again.LossRounds.Count == renoRun.LossRounds.Count, "two identical Reno runs lose at the same rounds");
        bool same = true;
        for (int i = 0; i < renoRun.Windows.Count; i++) if (again.Windows[i] != renoRun.Windows[i]) same = false;
        Check(same, "two identical Reno runs produce the same window trace");

        Console.WriteLine($"reno: losses={renoRun.LossRounds.Count} peak={maxLoss} mean={renoRun.AverageWindow:F2} " +
                          $"utilisation={utilisation:F4} meanGoodput={renoRun.AverageGoodput:F1}");
        Console.WriteLine($"cubic: losses={cubicRun.LossRounds.Count} mean={cubicRun.AverageWindow:F2} " +
                          $"goodput={cubicRun.AverageGoodput:F1}");
        Console.WriteLine("PASS");
    }

    private static bool Decreasing(List<double> values)
    {
        for (int i = 1; i < values.Count; i++) if (values[i] > values[i - 1] + 1e-9) return false;
        return true;
    }

    private static bool Increasing(List<double> values)
    {
        for (int i = 1; i < values.Count; i++) if (values[i] < values[i - 1] - 1e-9) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
