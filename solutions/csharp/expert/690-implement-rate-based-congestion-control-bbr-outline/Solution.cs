// Task 690 - Implement rate-based congestion control (BBR outline). (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A scaled-down but faithful BBR model: unlike loss based controllers this one paces by an
// estimated delivery rate instead of a window, and it decides from two windowed filters:
//
//   * BtlBw  - the maximum delivery rate observed in the last BandwidthWindow rounds (a max filter);
//   * RTprop - the minimum round trip time observed in the same window (a min filter);
//   * pacing rate  = pacing_gain * BtlBw,  inflight cap = cwnd_gain * BtlBw * RTprop;
//   * a four state machine: Startup (pacing_gain 2.885, exits after three rounds without 25 %
//     rate growth), Drain (pacing_gain 1/2.885 until the queue emptied), ProbeBW (the eight phase
//     gain cycle 1.25, 0.75, 1, 1, 1, 1, 1, 1 whose mean is exactly one), and ProbeRTT (drop the
//     inflight cap to four segments for one round so the queue empties and RTprop is refreshed).
//
// The link is a deterministic fluid model in units of "segments per round trip": it delivers
// `capacity` segments per round, holds up to `buffer` segments of queue, and drops anything that
// does not fit. Documented simplifications: one RTT granularity (no sub-round pacing clock), no
// packet level ACK train, no loss recovery (dropped segments are only counted, not retransmitted),
// the gain cycle advances one phase per round instead of per ProbeBW round of a real BBR clock,
// and no BBRv2 loss/QUEUE styles. What is verified is the real algorithmic content: the rate
// estimate converges to the link capacity during Startup, the mean of the gain cycle is one,
// ProbeBW's "up" phase builds a queue that its "down" phase drains, the standing queue stays
// bounded (the point of a rate based controller), ProbeRTT restores the RTprop floor, the
// inflight cap is respected every round, and two runs are bit identical.
using System;
using System.Collections.Generic;

internal enum BbrState
{
    Startup,
    Drain,
    ProbeBw,
    ProbeRtt,
}

internal sealed class BbrSample
{
    public int Round;
    public BbrState State;
    public double PacingGain;
    public double PacingRate;
    public double CwndLimit;
    public long Sent;
    public long Delivered;
    public long Queue;
    public long Dropped;
    public double Rtt;
    public double BtlbBw;
    public double RtProp;
    public double BtlBwUsed;    // the estimate the round was paced with (before the filters updated)
    public double RtPropUsed;
}

internal sealed class BbrFlow
{
    public const double StartupPacingGain = 2.885;
    public const double DrainPacingGain = 1.0 / 2.885;
    public const double CwndGain = 2.0;
    public const double ProbeRttCap = 4.0;
    public const int ProbeRttInterval = 10;     // a scaled down 10 second ProbeRTT period
    public const int FilterWindow = 10;
    public static readonly double[] GainCycle = { 1.25, 0.75, 1, 1, 1, 1, 1, 1 };

    public BbrState State = BbrState.Startup;
    public double BtlbBw = 1.0;                 // segments per round
    public double RtProp = 1.0;                 // rounds, starts at the base RTT assumption
    public int Phase;
    public long Queue;
    public int PlateauRounds;
    public int ProbeRttRound = -1;

    private long _lastDelivered;
    private readonly Queue<double> _rateWindow = new Queue<double>();
    private readonly Queue<double> _rttWindow = new Queue<double>();
    private readonly List<BbrSample> _trace = new List<BbrSample>();

    public List<BbrSample> Trace => _trace;
    public long TotalDelivered { get; private set; }
    public long TotalDropped { get; private set; }
    public int LossRounds { get; private set; }

    public double PacingGain() => State switch
    {
        BbrState.Startup => StartupPacingGain,
        BbrState.Drain => DrainPacingGain,
        BbrState.ProbeBw => GainCycle[Phase],
        _ => 1.0,
    };

    public double CwndLimit(double btlbw, double rtprop)
    {
        if (State == BbrState.ProbeRtt) return ProbeRttCap;
        return Math.Max(CwndGain * btlbw * rtprop, ProbeRttCap);
    }

    /// <summary>One round trip against a link with the given capacity, buffer and base RTT.</summary>
    public BbrSample Step(int round, int capacity, int buffer, double baseRtt)
    {
        double gain = PacingGain();
        double estimateUsed = BtlbBw;
        double rtPropUsed = RtProp;
        double pacingRate = gain * estimateUsed;
        double cwndLimit = CwndLimit(estimateUsed, rtPropUsed);

        long budget = (long)Math.Floor(pacingRate);
        long offer = Math.Min(budget, (long)Math.Floor(cwndLimit));
        long space = Math.Max(0, buffer - Queue);
        long accepted = Math.Min(offer, capacity + space);
        long dropped = offer - accepted;
        // The link drains at most `capacity` segments per round, so it serves the queue as well.
        long inSystem = accepted + Queue;
        long delivered = Math.Min(inSystem, capacity);
        Queue = inSystem - delivered;
        double rtt = baseRtt + (double)Queue / capacity;

        _rateWindow.Enqueue(delivered);
        if (_rateWindow.Count > FilterWindow) _rateWindow.Dequeue();
        double best = 0;
        foreach (double rate in _rateWindow) best = Math.Max(best, rate);
        BtlbBw = best;

        _rttWindow.Enqueue(rtt);
        if (_rttWindow.Count > FilterWindow) _rttWindow.Dequeue();
        double lowest = double.MaxValue;
        foreach (double sample in _rttWindow) lowest = Math.Min(lowest, sample);
        RtProp = lowest;

        var entry = new BbrSample
        {
            Round = round, State = State, PacingGain = gain, PacingRate = pacingRate, CwndLimit = cwndLimit,
            Sent = offer, Delivered = delivered, Queue = Queue, Dropped = dropped, Rtt = rtt,
            BtlbBw = BtlbBw, RtProp = RtProp, BtlBwUsed = estimateUsed, RtPropUsed = rtPropUsed,
        };
        _trace.Add(entry);
        TotalDelivered += delivered;
        TotalDropped += dropped;
        if (dropped > 0) LossRounds++;

        Advance(round, delivered);
        return entry;
    }

    private void Advance(int round, long delivered)
    {
        switch (State)
        {
            case BbrState.Startup:
                // Startup ends when the delivery rate stops growing by 25 % for three rounds in a row.
                if (delivered >= 1.25 * _lastDelivered) PlateauRounds = 0;
                else PlateauRounds++;
                _lastDelivered = delivered;
                if (PlateauRounds >= 3) State = BbrState.Drain;
                break;

            case BbrState.Drain:
                // Stop pacing faster than the pipe once the queue we built has drained away.
                if (Queue == 0) State = BbrState.ProbeBw;
                break;

            case BbrState.ProbeBw:
                Phase = (Phase + 1) % GainCycle.Length;
                if (round % ProbeRttInterval == 0)
                {
                    ProbeRttRound = round;
                    State = BbrState.ProbeRtt;
                }
                break;

            case BbrState.ProbeRtt:
                if (round - ProbeRttRound >= 1) State = BbrState.ProbeBw;
                break;
        }
    }
}

public static class Solution
{
    private const int Capacity = 100;
    private const int Buffer = 200;
    private const double BaseRtt = 1.0;

    public static void Main()
    {
        var flow = Run(240);

        // --- Startup discovers the bandwidth and paces at 2.885x the estimate. ------------
        var startup = flow.Trace.FindAll(s => s.State == BbrState.Startup);
        Check(startup.Count > 4, "startup lasts several rounds, rounds=" + startup.Count);
        bool startupGain = true, estimateGrows = true;
        double previousEstimate = 0;
        foreach (var sample in startup)
        {
            if (Math.Abs(sample.PacingGain - BbrFlow.StartupPacingGain) > 1e-12) startupGain = false;
            if (Math.Abs(sample.PacingRate - BbrFlow.StartupPacingGain * sample.BtlBwUsed) > 1e-9) startupGain = false;
            if (sample.BtlbBw < previousEstimate - 1e-9) estimateGrows = false;
            previousEstimate = sample.BtlbBw;
        }
        Check(startupGain, "startup paces at 2.885 * BtlBw and nothing else");
        Check(estimateGrows, "the max filter makes the bandwidth estimate non-decreasing");

        var lastStartup = startup[startup.Count - 1];
        Check(Math.Abs(lastStartup.BtlbBw - Capacity) <= 0.05 * Capacity,
            "startup ends within 5 % of the link capacity, estimate=" + lastStartup.BtlbBw);

        // The exit rule itself: three consecutive rounds below 25 % growth.
        var trace = flow.Trace;
        int exitIndex = trace.FindIndex(s => s.State == BbrState.Drain);
        Check(exitIndex >= 3, "drain follows at least three startup rounds");
        bool plateau = true;
        for (int i = exitIndex - 3; i < exitIndex; i++)
        {
            if (trace[i].Delivered >= 1.25 * trace[i - 1].Delivered) plateau = false;
        }
        Check(plateau, "the last three startup rounds all grew by less than 25 %");

        // --- Drain: pace at 1/2.885 until the queue is empty. ----------------------------
        var drain = trace.FindAll(s => s.State == BbrState.Drain);
        Check(drain.Count > 0, "the flow actually entered drain");
        bool drainGain = true;
        foreach (var sample in drain) if (Math.Abs(sample.PacingGain - BbrFlow.DrainPacingGain) > 1e-12) drainGain = false;
        Check(drainGain, "drain paces at 1/2.885 of the estimate");
        Check(drain[drain.Count - 1].Queue == 0, "drain ends with an empty queue");

        // --- ProbeBW gain cycle. ---------------------------------------------------------
        double gainSum = 0;
        foreach (double gain in BbrFlow.GainCycle) gainSum += gain;
        Check(Math.Abs(gainSum - BbrFlow.GainCycle.Length) < 1e-12, "the ProbeBW gain cycle means exactly one, sum=" + gainSum);
        Check(BbrFlow.GainCycle[0] > 1 && BbrFlow.GainCycle[1] < 1, "the cycle probes up first and then drains");

        var probeBw = trace.FindAll(s => s.State == BbrState.ProbeBw);
        Check(probeBw.Count > 8, "ProbeBW covers several cycles, rounds=" + probeBw.Count);
        var gainsSeen = new List<double>();
        for (int i = 0; i < BbrFlow.GainCycle.Length; i++) gainsSeen.Add(probeBw[i].PacingGain);
        bool cycleOrder = true;
        for (int i = 0; i < gainsSeen.Count; i++)
            if (Math.Abs(gainsSeen[i] - BbrFlow.GainCycle[i]) > 1e-12) cycleOrder = false;
        Check(cycleOrder, "the first ProbeBW rounds walk the gain cycle in order");

        // The up phase must build a queue and the down phase must drain it again.
        var up = probeBw[0];
        Check(Math.Abs(up.PacingGain - 1.25) < 1e-12, "the probe up phase uses gain 1.25");
        Check(up.Queue > 0, "the 1.25x phase builds a standing queue, queue=" + up.Queue);

        var probeUps = probeBw.FindAll(s => Math.Abs(s.PacingGain - 1.25) < 1e-12);
        var probeDowns = probeBw.FindAll(s => Math.Abs(s.PacingGain - 0.75) < 1e-12);

        // Steady state keeps the queue bounded - the whole point of a rate based controller.
        Check(probeUps.Count >= 3, "several probe up phases were observed, count=" + probeUps.Count);
        long steadyQueue = probeUps[0].Queue;
        bool sameProbe = true;
        foreach (var sample in probeUps) if (sample.Queue != steadyQueue) sameProbe = false;
        Check(sameProbe, "every 1.25x probe builds exactly the same queue, queue=" + steadyQueue);
        Check(steadyQueue <= Capacity, "the probe queue stays inside one bandwidth-delay product");
        bool allDrained = true;
        foreach (var sample in probeDowns) if (sample.Queue > 0) allDrained = false;
        Check(allDrained, "the 0.75x phase always empties the queue again");
        var down = probeBw[1];
        Check(down.Queue < up.Queue, "the 0.75x phase drains part of that queue");

        // --- Inflight cap and pacing bound every round. ----------------------------------
        bool capRespected = true, noLossInSteadyState = true;
        foreach (var sample in trace)
        {
            if (sample.Sent > sample.CwndLimit + 1e-9) capRespected = false;
            double expectedCap = Math.Max(BbrFlow.CwndGain * sample.BtlBwUsed * sample.RtPropUsed, BbrFlow.ProbeRttCap);
            if (sample.State != BbrState.ProbeRtt && Math.Abs(sample.CwndLimit - expectedCap) > 1e-9) capRespected = false;
        }
        Check(capRespected, "the inflight cap is 2 * BtlBw * RTprop and is never exceeded");
        foreach (var sample in probeBw) if (sample.Dropped > 0) noLossInSteadyState = false;
        Check(noLossInSteadyState, "the steady state needs no loss at all to stay efficient");

        // --- RTprop is a windowed minimum and ProbeRTT restores it. ----------------------
        bool rttMinimum = true;
        foreach (var sample in trace) if (sample.RtProp > sample.Rtt + 1e-12) rttMinimum = false;
        Check(rttMinimum, "RTprop never exceeds an observed round trip time");
        Check(Math.Abs(flow.RtProp - BaseRtt) < 1e-9, "RTprop settles on the base RTT of the link");

        var probeRtt = trace.FindAll(s => s.State == BbrState.ProbeRtt);
        Check(probeRtt.Count >= 4, "ProbeRTT fires periodically, times=" + probeRtt.Count);
        bool capDropped = true;
        foreach (var sample in probeRtt) if (Math.Abs(sample.CwndLimit - BbrFlow.ProbeRttCap) > 1e-12) capDropped = false;
        Check(capDropped, "ProbeRTT drops the inflight cap to four segments");
        Check(probeRtt[0].Dropped == 0 && probeRtt[0].Queue < Capacity, "ProbeRTT drains the queue instead of overrunning it");
        var afterProbeRtt = probeBw.FindAll(s => s.Round > probeRtt[0].Round);
        Check(afterProbeRtt.Count > 0 && afterProbeRtt[afterProbeRtt.Count - 1].CwndLimit > BbrFlow.ProbeRttCap,
            "the cap recovers after ProbeRTT");

        // --- Efficiency, loss accounting and state ordering. ----------------------------
        Check(flow.TotalDropped > 0, "startup overshooting the buffer is visible in the loss counter");
        Check(flow.LossRounds < trace.Count / 10, "losses are confined to the startup overshoot, rounds=" + flow.LossRounds);
        long deliveredInProbeBw = 0;
        int busyRounds = 0;
        foreach (var sample in probeBw)
        {
            deliveredInProbeBw += sample.Delivered;
            if (sample.Delivered == Capacity) busyRounds++;
        }
        Check((double)busyRounds / probeBw.Count >= 0.9, "ProbeBW keeps the link busy, busy=" + busyRounds + "/" + probeBw.Count);
        Check(deliveredInProbeBw <= (long)Capacity * probeBw.Count, "delivery never exceeds the link capacity");

        int[] order = { 0, 1, 2, 3 };
        bool neverBackwards = true;
        int highest = 0;
        foreach (var sample in trace)
        {
            int rank = Array.IndexOf(order, (int)sample.State);
            if (sample.State == BbrState.ProbeRtt) continue;
            if (rank < highest && sample.State != BbrState.ProbeRtt) neverBackwards = false;
            if (rank > highest) highest = rank;
        }
        Check(neverBackwards, "the state machine never falls back to an earlier state");

        // --- Determinism. -----------------------------------------------------------------
        var repeat = Run(240);
        bool identical = repeat.Trace.Count == flow.Trace.Count;
        for (int i = 0; identical && i < flow.Trace.Count; i++)
        {
            if (repeat.Trace[i].Sent != flow.Trace[i].Sent || repeat.Trace[i].Queue != flow.Trace[i].Queue) identical = false;
            if (Math.Abs(repeat.Trace[i].BtlbBw - flow.Trace[i].BtlbBw) > 1e-12) identical = false;
        }
        Check(identical, "two identical runs produce the same trace");

        Console.WriteLine($"startupRounds={startup.Count} drainRounds={drain.Count} probeBwRounds={probeBw.Count} " +
                          $"lossRounds={flow.LossRounds} delivered={flow.TotalDelivered}");
        Console.WriteLine("PASS");
    }

    private static BbrFlow Run(int rounds)
    {
        var flow = new BbrFlow();
        for (int round = 1; round <= rounds; round++) flow.Step(round, Capacity, Buffer, BaseRtt);
        return flow;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
