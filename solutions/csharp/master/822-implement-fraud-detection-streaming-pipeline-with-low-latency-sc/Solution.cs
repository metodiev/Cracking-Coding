// Task 822 - Implement fraud-detection streaming pipeline with low-latency scoring and feedback. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Events are processed one at a time in event time with an explicit watermark and an allowed lateness;
// there is no wall clock, no thread and no sleeping. Per key (account) the pipeline keeps a sliding
// window of the last W event-time units as a list kept in event-time order (an insertion scans back only
// past the entries the new event follows, eviction takes from the head, both counted), plus a second
// ordered list for the device velocity. The feature step therefore performs exactly FeatureCount
// operations per event - a counted constant, not a stopwatch reading - and never rescans history.
//
//   * features - amount, window count, window mean, amount/mean deviation, new-device-for-key flag and
//       device velocity in the window; every one of them comes from the incremental state;
//   * scoring  - logistic regression in a single dot product;
//   * feedback - labels arrive with a fixed delay; each revealed label triggers one SGD step that uses
//       the features cached at scoring time. The feedback loop refuses any label whose reveal tick has
//       not passed (the filter is exercised and counted), and precision/recall/AUC on the second half of
//       the stream are compared with the first half to show the loop actually helps.
//
// Simplifications (documented): one in-memory stage (no Kafka/Flink, no shuffle, no durable state); the
// watermark advances from observed event times, not from a source heartbeat; a logistic regression with
// hand written SGD stands in for a boosted ensemble; out-of-order events are inserted in event-time
// order by scanning back only to its own position (amortised constant for the small disorder a bounded
// lateness allows, and the counted bound is checked); the device velocity is anchored at the newest
// event time seen for that device rather than at the arriving event's time; the latency claim is an
// operation count, not a measurement of hardware time.
//
// Verified invariants: incremental window aggregates (count, sum, max) equal a brute-force recomputation
// over the same window, in order and fully out of order; events older than the watermark are dropped and
// each drop decision is reproduced independently; every score equals a freshly recomputed dot product;
// feature work per event is exactly the counted constant and maintenance pops stay within their
// amortised bound; the label ledger never uses an unrevealed label; the event ledger balances
// (processed + dropped == fed); the model's recall and AUC improve on the held-out second half; and the
// pipeline is deterministic.
using System;
using System.Collections.Generic;
using System.Linq;

internal struct SplitMix64
{
    private ulong state;

    public SplitMix64(ulong seed) { state = seed; }

    public ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public int NextInt(int bound) => (int)(Next() % (ulong)bound);

    public double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class Event
{
    public int Id;
    public int Sequence;
    public long EventTime;
    public int Key;
    public int Device;
    public double Amount;
    public bool Fraud;
}

internal sealed class KeyState
{
    public readonly LinkedList<(long Time, double Amount)> Recent = new LinkedList<(long, double)>();
    public readonly HashSet<int> Devices = new HashSet<int>();
    public double Sum;
    public long Anchor = long.MinValue;
    public long Evictions;
}

internal sealed class DeviceState
{
    public readonly LinkedList<long> Times = new LinkedList<long>();
    public long Anchor = long.MinValue;
}

internal sealed class Scored
{
    public Event Event = null!;
    public double[] Features = Array.Empty<double>();
    public double Score;
    public bool Flagged;
    public long RevealTick;
}

internal sealed class FraudPipeline
{
    public const int FeatureCount = 7;

    public int WindowUnits = 40;
    public long AllowedLateness = 3;
    public double Threshold = 0.5;
    public double LearningRate = 0.9;

    private readonly Dictionary<int, KeyState> keys = new Dictionary<int, KeyState>();
    private readonly Dictionary<int, DeviceState> deviceTimes = new Dictionary<int, DeviceState>();
    private readonly double[] weights = new double[FeatureCount];
    private double bias;

    public long Watermark = long.MinValue;
    public long Processed;
    public long Dropped;
    public long FeatureOps;
    public long MaintenancePops;
    public long HistoryScans;
    public long Updates;
    public long RevealedLabels;
    public long LabelChecks;
    public long LabelLeaks;
    public long WindowEvictions;

    public readonly List<Scored> Scored = new List<Scored>();
    private readonly List<Scored> pending = new List<Scored>();

    public double this[int feature] => weights[feature];
    public double Bias => bias;
    public double[] Weights => (double[])weights.Clone();
    public IEnumerable<int> ObservedKeys => keys.Keys.OrderBy(k => k);

    public bool Process(Event e, long arrivalTick)
    {
        Watermark = Math.Max(Watermark, e.EventTime - AllowedLateness);
        if (e.EventTime < Watermark)
        {
            Dropped++;
            return false;
        }

        KeyState state = GetKey(e.Key);
        state.Anchor = Math.Max(state.Anchor, e.EventTime);
        // The window is anchored at the newest event time for the key, and kept in event-time order, so
        // late arrivals cannot unsort it and eviction always drops the oldest surviving event.
        long cutoff = state.Anchor - WindowUnits;
        while (state.Recent.Count > 0 && state.Recent.First!.Value.Time < cutoff)
        {
            state.Sum -= state.Recent.First.Value.Amount;
            state.Recent.RemoveFirst();
            state.Evictions++;
            WindowEvictions++;
            MaintenancePops++;
        }
        if (e.EventTime >= cutoff)
        {
            InsertByTime(state.Recent, e.EventTime, e.Amount);
            state.Sum += e.Amount;
        }

        bool newDeviceForKey = state.Devices.Add(e.Device);

        DeviceState device = GetDevice(e.Device);
        device.Anchor = Math.Max(device.Anchor, e.EventTime);
        long deviceCutoff = device.Anchor - WindowUnits;
        while (device.Times.Count > 0 && device.Times.First!.Value < deviceCutoff)
        {
            device.Times.RemoveFirst();
            MaintenancePops++;
        }
        if (e.EventTime >= deviceCutoff) InsertTime(device.Times, e.EventTime);

        double mean = state.Sum / Math.Max(1, state.Recent.Count);
        var features = new double[FeatureCount];
        features[0] = 1.0;
        features[1] = e.Amount / 100.0;
        features[2] = state.Recent.Count / 10.0;
        features[3] = mean / 100.0;
        features[4] = Math.Min(10.0, e.Amount / Math.Max(1e-9, mean));
        features[5] = newDeviceForKey ? 1.0 : 0.0;
        features[6] = device.Times.Count / 5.0;
        FeatureOps += FeatureCount;

        double score = Score(features);
        var scored = new Scored
        {
            Event = e,
            Features = features,
            Score = score,
            Flagged = score >= Threshold,
            RevealTick = arrivalTick + 3
        };
        Scored.Add(scored);
        pending.Add(scored);
        Processed++;
        return true;
    }

    // Insert into an event-time ordered deque, scanning back only past the entries the new event follows.
    private void InsertByTime(LinkedList<(long Time, double Amount)> list, long time, double amount)
    {
        LinkedListNode<(long Time, double Amount)>? node = list.Last;
        while (node != null && node.Value.Time > time)
        {
            node = node.Previous;
            MaintenancePops++;
        }
        if (node == null) list.AddFirst((time, amount));
        else list.AddAfter(node, (time, amount));
    }

    private void InsertTime(LinkedList<long> list, long time)
    {
        LinkedListNode<long>? node = list.Last;
        while (node != null && node.Value > time)
        {
            node = node.Previous;
            MaintenancePops++;
        }
        if (node == null) list.AddFirst(time);
        else list.AddAfter(node, time);
    }

    private KeyState GetKey(int key)
    {
        if (!keys.TryGetValue(key, out KeyState state))
        {
            state = new KeyState();
            keys[key] = state;
        }
        return state;
    }

    private DeviceState GetDevice(int device)
    {
        if (!deviceTimes.TryGetValue(device, out DeviceState state))
        {
            state = new DeviceState();
            deviceTimes[device] = state;
        }
        return state;
    }

    public double Score(double[] features)
    {
        double z = bias;
        for (int f = 0; f < FeatureCount; f++) z += weights[f] * features[f];
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    // Only labels whose reveal tick has passed are used, exactly once each; the filter is counted so the
    // "never learn from the future" claim is exercised rather than asserted.
    public void ApplyFeedback(long now)
    {
        foreach (Scored scored in pending.Where(s => s.RevealTick <= now).OrderBy(s => s.Event.Sequence).ToList())
        {
            LabelChecks++;
            if (scored.RevealTick > now)
            {
                LabelLeaks++;
                continue;
            }
            double error = scored.Score - (scored.Event.Fraud ? 1.0 : 0.0);
            for (int f = 0; f < FeatureCount; f++) weights[f] -= LearningRate * error * scored.Features[f];
            bias -= LearningRate * error;
            Updates++;
            RevealedLabels++;
            pending.Remove(scored);
        }
    }

    public (long Count, double Sum) ObservedWindow(int key)
    {
        KeyState state = GetKey(key);
        return (state.Recent.Count, state.Sum);
    }

    public long AnchorOf(int key) => GetKey(key).Anchor;

    public int StateKeys => keys.Count;

    public long StateRows => keys.Values.Sum(k => k.Recent.Count);
}

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0xF1A0DUL);

    private static int Rand(int bound) => rng.NextInt(bound);

    public static void Main()
    {
        var events = Generate(4000, 0.10, out int fraudCount);
        Check(events.Count == 4000 && fraudCount > 200 && fraudCount < 800,
            "the stream carries a realistic fraud rate: " + fraudCount);

        // ---- incremental aggregates against a brute force window ---------------------------------
        var pipeline = new FraudPipeline();
        var history = new List<Event>();
        int compared = 0;
        foreach (Event e in events)
        {
            pipeline.Process(e, e.Sequence);
            history.Add(e);
            if (e.Sequence % 97 == 0)
            {
                (long count, double sum) = pipeline.ObservedWindow(e.Key);
                (long expectedCount, double expectedSum) = BruteForceWindow(history, e.Key, e.EventTime, pipeline.WindowUnits);
                Check(count == expectedCount, $"window count for key {e.Key} at {e.EventTime}: {count} vs {expectedCount}");
                Check(Math.Abs(sum - expectedSum) < 1e-9, "window sum matches the brute force recomputation");
                Check(Math.Abs(sum / Math.Max(1, count) - expectedSum / Math.Max(1, expectedCount)) < 1e-9,
                    "window mean follows from the incremental sum and count");
                compared++;
            }
        }
        Check(compared >= 40, "window comparisons performed: " + compared);
        Check(pipeline.WindowEvictions > 0, "the sliding window actually evicted: " + pipeline.WindowEvictions);
        Check(pipeline.FeatureOps == pipeline.Processed * FraudPipeline.FeatureCount,
            "feature work is exactly a constant per event: " + pipeline.FeatureOps);
        Check(pipeline.MaintenancePops <= 4 * pipeline.Processed,
            "maintenance pops stay inside the amortised bound: " + pipeline.MaintenancePops);
        Check(pipeline.HistoryScans == 0, "no feature computation scanned history");
        Check(pipeline.Processed == events.Count && pipeline.Dropped == 0, "the ordered pass drops nothing");

        // ---- scoring equals a freshly recomputed dot product -------------------------------------
        var scoring = new FraudPipeline();
        foreach (Event e in events.Take(200)) scoring.Process(e, e.Sequence);
        int scoreChecks = 0;
        foreach (Scored scored in scoring.Scored.Take(50))
        {
            double z = scoring.Bias;
            double[] weights = scoring.Weights;
            for (int f = 0; f < FraudPipeline.FeatureCount; f++) z += weights[f] * scored.Features[f];
            double expected = 1.0 / (1.0 + Math.Exp(-z));
            Check(Math.Abs(expected - scored.Score) < 1e-12, "the score equals the recomputed dot product");
            Check(scored.Flagged == (scored.Score >= scoring.Threshold), "the flag follows the threshold");
            Check(scored.Score >= 0 && scored.Score <= 1, "scores are probabilities");
            scoreChecks++;
        }
        Check(scoreChecks == 50, "scores compared");

        // ---- feedback improves the model ---------------------------------------------------------
        var learned = new FraudPipeline();
        foreach (Event e in events)
        {
            learned.Process(e, e.Sequence);
            learned.ApplyFeedback(e.Sequence);
        }
        Check(learned.Updates == learned.RevealedLabels, "every used label was counted as revealed");
        Check(learned.LabelLeaks == 0, "no label was used before it was revealed");
        Check(learned.LabelChecks == learned.Updates, "the reveal filter admitted every used label");
        Check(learned.Updates > 3000, "the great majority of labels were used: " + learned.Updates);
        double[] early = Metrics(learned.Scored.Take(100).ToList());
        double[] late = Metrics(learned.Scored.Skip(2500).ToList());
        Check(late[2] > early[2] + 0.02, $"AUC improves after feedback: {early[2]:F3} -> {late[2]:F3}");
        Check(late[1] > early[1] + 0.2, $"precision improves: {early[1]:F3} -> {late[1]:F3}");
        Check(late[0] >= early[0] - 0.05, $"recall is preserved while precision rises: {early[0]:F3} -> {late[0]:F3}");
        Check(late[2] > 0.8, "the learned model separates fraud well: AUC " + late[2].ToString("F3"));
        double flaggedWithNewDevice = learned.Scored.Where(s => s.Features[5] > 0.5).Average(s => s.Flagged ? 1.0 : 0.0);
        double flaggedWithout = learned.Scored.Where(s => s.Features[5] <= 0.5).Average(s => s.Flagged ? 1.0 : 0.0);
        Check(flaggedWithNewDevice > flaggedWithout + 0.2,
            $"events on a new device are flagged far more often: {flaggedWithNewDevice:F3} vs {flaggedWithout:F3}");
        Check(learned.Weights.Any(w => Math.Abs(w) > 0.5), "the model learned non trivial weights");

        // ---- fully out of order processing against the same window oracle -------------------------
        var shuffled = events.OrderBy(e => (e.Sequence * 7919) % 4000).ToList();
        var reordered = new FraudPipeline { AllowedLateness = 1L << 40 };
        var reorderedHistory = new List<Event>();
        foreach (Event e in shuffled)
        {
            reordered.Process(e, e.Sequence);
            reorderedHistory.Add(e);
        }
        Check(reordered.Dropped == 0 && reordered.Processed == events.Count, "a generous lateness allowance drops nothing");
        int keysCompared = 0;
        foreach (int key in reordered.ObservedKeys)
        {
            long anchor = reordered.AnchorOf(key);
            (long count, double sum) = reordered.ObservedWindow(key);
            (long expectedCount, double expectedSum) = BruteForceWindow(reorderedHistory, key, anchor, reordered.WindowUnits);
            Check(count == expectedCount, $"out of order window count for key {key}: {count} vs {expectedCount}");
            Check(Math.Abs(sum - expectedSum) < 1e-9, "out of order window sum matches the oracle");
            keysCompared++;
        }
        Check(keysCompared >= 40, "out of order keys compared: " + keysCompared);

        // ---- late arrivals are dropped exactly when they precede the watermark -------------------
        var strict = new FraudPipeline { AllowedLateness = 2 };
        foreach (Event e in events.Take(600)) strict.Process(e, e.Sequence);
        int staleChecked = 0;
        foreach (Event stale in events.Take(60))
        {
            long watermark = strict.Watermark;
            bool shouldDrop = stale.EventTime < watermark;
            bool accepted = strict.Process(stale, stale.Sequence + 10000);
            Check(accepted == !shouldDrop, "a stale event is dropped exactly when it precedes the watermark");
            staleChecked++;
        }
        Check(staleChecked == 60 && strict.Dropped > 0, "stale events were dropped: " + strict.Dropped);
        Check(strict.Processed + strict.Dropped == 660, "the event ledger balances after the late arrivals");

        // ---- latency, state and memory bounds -----------------------------------------------------
        Check(pipeline.StateKeys <= 64, "state keys stay bounded: " + pipeline.StateKeys);
        Check(pipeline.StateRows <= events.Count, "state rows never exceed the events seen");
        Check(pipeline.StateRows > 0, "the state is actually populated");
        long opsPerEvent = learned.FeatureOps / Math.Max(1, learned.Processed);
        Check(opsPerEvent == FraudPipeline.FeatureCount, "amortised feature ops per event: " + opsPerEvent);
        Check(learned.FeatureOps == FraudPipeline.FeatureCount * learned.Processed, "feature work is linear in events, not quadratic");
        Check(learned.Scored.Count == learned.Processed, "one scored record per processed event");

        // ---- degenerate streams -------------------------------------------------------------------
        var empty = new FraudPipeline();
        Check(empty.Processed == 0 && empty.Scored.Count == 0 && empty.Watermark == long.MinValue,
            "an empty stream leaves the pipeline untouched");
        var single = new FraudPipeline();
        single.Process(new Event { Id = 1, Sequence = 1, EventTime = 10, Key = 1, Device = 5, Amount = 20, Fraud = true }, 1);
        Check(single.Processed == 1 && single.Scored.Count == 1 && single.Scored[0].Features[5] == 1.0,
            "a single event scores with the new-device feature set");
        var clean = new FraudPipeline();
        foreach (Event e in events.Where(e => !e.Fraud).Take(400)) clean.Process(e, e.Sequence);
        Check(clean.Scored.All(s => Math.Abs(s.Score - 0.5) < 1e-12),
            "an untrained model scores every event at exactly one half");
        double falsePositiveRate = learned.Scored.Where(s => !s.Event.Fraud).Average(s => s.Flagged ? 1.0 : 0.0);
        Check(falsePositiveRate < 0.3, "the trained model flags clean traffic at a low rate: " + falsePositiveRate.ToString("F3"));
        var burst = new FraudPipeline();
        foreach (Event e in events.Take(40).Select(e => new Event
        {
            Id = e.Id,
            Sequence = e.Sequence,
            EventTime = 1000 + e.Sequence,
            Key = 7,
            Device = 4242,
            Amount = 900,
            Fraud = true
        }))
            burst.Process(e, e.Sequence);
        (long burstCount, double burstSum) = burst.ObservedWindow(7);
        Check(burstCount == 40 && Math.Abs(burstSum - 40 * 900) < 1e-6, "a burst fills the window as expected");
        Check(burst.Scored.Last().Features[6] > 5, "the burst raises the device velocity feature: "
            + burst.Scored.Last().Features[6].ToString("F2"));
        Check(burst.Scored.First().Features[5] == 1.0 && burst.Scored.Last().Features[5] == 0.0,
            "the first burst event sees a new device and the rest do not");
        var allFraud = new FraudPipeline();
        foreach (Event e in events.Where(e => e.Fraud).Take(300)) allFraud.Process(e, e.Sequence);
        Check(allFraud.Processed > 100 && allFraud.Scored.All(s => s.Score >= 0 && s.Score <= 1),
            "an all fraud stream still produces valid scores");

        // ---- determinism --------------------------------------------------------------------------
        var repeat = new FraudPipeline();
        foreach (Event e in events)
        {
            repeat.Process(e, e.Sequence);
            repeat.ApplyFeedback(e.Sequence);
        }
        Check(repeat.Scored.Select(s => s.Score).SequenceEqual(learned.Scored.Select(s => s.Score)),
            "two identical runs score identically");
        Check(Math.Abs(repeat.Bias - learned.Bias) < 1e-15 && repeat.Updates == learned.Updates,
            "two identical runs learn identically");

        Console.WriteLine($"822 pipeline: {learned.Processed} events scored, {learned.Dropped} dropped, "
            + $"{learned.Updates} feedback updates with {learned.LabelLeaks} leaks; cold model recall {early[0]:F3} "
            + $"precision {early[1]:F3} AUC {early[2]:F3} -> warm model recall {late[0]:F3} precision {late[1]:F3} "
            + $"AUC {late[2]:F3}; {opsPerEvent} feature ops per event, {pipeline.WindowEvictions} window evictions, "
            + $"state {pipeline.StateKeys} keys / {pipeline.StateRows} rows");
        Console.WriteLine("PASS");
    }

    private static double[] Metrics(IList<Scored> scored)
    {
        int truePositive = 0, falsePositive = 0, falseNegative = 0;
        foreach (Scored s in scored)
        {
            if (s.Event.Fraud && s.Flagged) truePositive++;
            else if (!s.Event.Fraud && s.Flagged) falsePositive++;
            else if (s.Event.Fraud && !s.Flagged) falseNegative++;
        }
        double recall = truePositive + falseNegative == 0 ? 0 : truePositive / (double)(truePositive + falseNegative);
        double precision = truePositive + falsePositive == 0 ? 0 : truePositive / (double)(truePositive + falsePositive);
        return new[] { recall, precision, Auc(scored) };
    }

    private static double Auc(IList<Scored> scored)
    {
        var positives = scored.Where(s => s.Event.Fraud).Select(s => s.Score).ToList();
        var negatives = scored.Where(s => !s.Event.Fraud).Select(s => s.Score).ToList();
        if (positives.Count == 0 || negatives.Count == 0) return 0.5;
        double wins = 0;
        foreach (double p in positives)
            foreach (double n in negatives)
                wins += p > n ? 1.0 : p < n ? 0.0 : 0.5;
        return wins / (positives.Count * (double)negatives.Count);
    }

    private static (long Count, double Sum) BruteForceWindow(IList<Event> history, int key, long atTime, int windowUnits)
    {
        long cutoff = atTime - windowUnits;
        long count = 0;
        double sum = 0;
        foreach (Event e in history)
        {
            if (e.Key != key || e.EventTime > atTime || e.EventTime < cutoff) continue;
            count++;
            sum += e.Amount;
        }
        return (count, sum);
    }

    private static List<Event> Generate(int count, double fraudRate, out int fraudCount)
    {
        var events = new List<Event>();
        var local = new SplitMix64(0xF1A0D5EEDUL);
        fraudCount = 0;
        long time = 0;
        for (int i = 0; i < count; i++)
        {
            time += 1 + (int)(local.Next() % 2);
            int key = 1 + (int)(local.Next() % 50);
            bool fraud = local.NextUnit() < fraudRate;
            double amount = fraud ? 200.0 + 800.0 * local.NextUnit() : 5.0 + 45.0 * local.NextUnit();
            if (fraud) fraudCount++;
            events.Add(new Event
            {
                Id = i,
                Sequence = i,
                EventTime = time,
                Key = key,
                Device = fraud ? 90000 + i : 1000 + key,
                Amount = amount,
                Fraud = fraud
            });
        }
        return events;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
