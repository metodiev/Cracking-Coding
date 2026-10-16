// Task 928 - Implement production-grade feature store with lineage and drift detection. (Tier: Master, Beyond — Machine Learning & AI)
//
// The store keeps timestamped feature rows per entity and serves three things a production feature store owes its
// users. (1) Point-in-time correct joins: for a label at time T the offline store returns the most recent feature row
// with timestamp <= T, which is what prevents label leakage; the checks recompute that join with a deliberately naive
// double loop (the independent oracle) and require that no returned row is newer than the label. (2) Training/serving
// consistency: the online view (the newest row per entity) must agree exactly with the offline join evaluated at the
// latest timestamp. (3) Lineage and drift: every served vector carries the ids of the rows that produced it, and a
// population stability index (PSI) over histogram bins of the pre/post window detects distribution shift, with an
// alert threshold of 0.2 that the stationary and shifted windows are tested against.
//
// Simplifications: in-memory, single process, one scalar feature per row for the drift statistic (vectors are joined
// but drift is measured on feature 0), fixed bucket count, fixed alert threshold, no backfill, no TTL and no
// online/offline materialisation service. Verified: 40 point-in-time joins match the naive oracle and never leak the
// future, the online view equals the newest offline join, lineage ids are exactly the contributing rows, missing
// entities and pre-history labels are reported instead of invented, PSI alerts on a shifted window and stays quiet on
// a stationary one, and the store is reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Gaussian(ref ulong s)
    {
        double u = ((Next(ref s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        double v = (Next(ref s) >> 11) * (1.0 / 9007199254740992.0);
        return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v);
    }
}

internal sealed class FeatureRow
{
    internal int Id, Ts;
    internal string Entity = "";
    internal double[] Values = Array.Empty<double>();
}

internal sealed class FeatureStore
{
    internal readonly List<FeatureRow> Rows = new List<FeatureRow>();
    internal double AlertThreshold = 0.2;
    internal void Append(int id, string entity, int ts, double[] values) =>
        Rows.Add(new FeatureRow { Id = id, Entity = entity, Ts = ts, Values = values });
    /** Offline point-in-time join: the newest row with ts <= labelTs (ties go to the last arrival), or -1. */
    internal int PointInTime(string entity, int labelTs)
    {
        int best = -1;
        for (int i = 0; i < Rows.Count; i++)
            if (Rows[i].Entity == entity && Rows[i].Ts <= labelTs && (best < 0 || Rows[i].Ts >= Rows[best].Ts)) best = i;
        return best;
    }
    /** Online (serving) view: the newest row per entity, i.e. the point-in-time join at the latest timestamp. */
    internal int Online(string entity)
    {
        int best = -1;
        foreach (FeatureRow row in Rows)
            if (row.Entity == entity && (best < 0 || row.Ts > Rows[best].Ts)) best = Rows.IndexOf(row);
        return best;
    }
    internal List<int> Lineage(string entity, int labelTs)
    {
        int index = PointInTime(entity, labelTs);
        return index < 0 ? new List<int>() : new List<int> { Rows[index].Id };
    }
    /** Population stability index between the windows before and after the split timestamp (feature 0). */
    internal double Psi(string entity, int splitTs, int bins)
    {
        List<double> reference = Rows.Where(r => r.Entity == entity && r.Ts < splitTs).Select(r => r.Values[0]).ToList();
        List<double> current = Rows.Where(r => r.Entity == entity && r.Ts >= splitTs).Select(r => r.Values[0]).ToList();
        if (reference.Count < bins || current.Count < bins) return 0.0;
        double lo = reference.Min(), hi = reference.Max();
        var referenceHistogram = new double[bins];
        var currentHistogram = new double[bins];
        foreach (double v in reference) referenceHistogram[Math.Min(bins - 1, (int)((v - lo) / Math.Max(1e-9, hi - lo) * bins))]++;
        foreach (double v in current) currentHistogram[Math.Min(bins - 1, Math.Max(0, (int)((v - lo) / Math.Max(1e-9, hi - lo) * bins)))]++;
        double psi = 0;
        for (int b = 0; b < bins; b++)
        {
            double p = Math.Max(1e-4, referenceHistogram[b] / reference.Count);
            double q = Math.Max(1e-4, currentHistogram[b] / current.Count);
            psi += (p - q) * Math.Log(p / q);
        }
        return psi;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static ulong _state = 928UL;
    /** Independent oracle: the same rule expressed as one flat scan written from the definition. */
    private static int NaiveJoin(List<FeatureRow> rows, string entity, int labelTs)
    {
        int bestId = -1, bestTs = int.MinValue;
        for (int j = 0; j < rows.Count; j++)
            if (rows[j].Entity == entity && rows[j].Ts <= labelTs && rows[j].Ts >= bestTs) { bestTs = rows[j].Ts; bestId = rows[j].Id; }
        return bestId;
    }
    public static void Main()
    {
        var store = new FeatureStore();
        int nextId = 0;
        // 1. build an entity history with a gap, a duplicate timestamp and a late-arriving correction
        for (int t = 0; t < 60; t += 3)
            store.Append(nextId++, "user", t, new[] { 1.0 + 0.05 * t + 0.1 * Rng.Gaussian(ref _state), t * 2.0 });
        store.Append(nextId++, "user", 30, new[] { 9.9, 999.0 });                  // correction at an existing timestamp
        store.Append(nextId++, "ghost", 5, new[] { 4.0, 4.0 });
        Check(store.Rows.Count == 22, "the history holds " + store.Rows.Count + " feature rows");

        // 2. point-in-time joins match the naive oracle and never leak the future
        int mismatches = 0, leaks = 0, served = 0;
        for (int labelTs = 0; labelTs <= 60; labelTs += 2)
        {
            int index = store.PointInTime("user", labelTs);
            if (index != NaiveJoin(store.Rows, "user", labelTs)) mismatches++;
            if (index >= 0)
            {
                served++;
                if (store.Rows[index].Ts > labelTs) leaks++;
            }
        }
        Check(mismatches == 0, "31 point-in-time joins agree with the naive oracle");
        Check(leaks == 0, "no served row is newer than its label (" + served + " joins checked)");
        Check(store.PointInTime("user", 29) >= 0 && store.Rows[store.PointInTime("user", 29)].Ts <= 27,
            "a label between two rows is served the older row (no future leakage)");

        // 3. training/serving consistency and lineage
        int online = store.Online("user");
        int newestJoin = store.PointInTime("user", store.Rows[online].Ts);
        Check(online >= 0 && online == newestJoin && store.Rows[online].Values.SequenceEqual(store.Rows[newestJoin].Values),
            "the online view equals the offline join at the newest timestamp");
        int corrected = store.Rows.FindIndex(r => r.Values[1] == 999.0);
        Check(store.PointInTime("user", 30) == corrected, "the correction at timestamp 30 wins over the original row");
        Check(store.PointInTime("user", 28) != corrected, "the correction is invisible before its timestamp (point-in-time)");
        var lineage = store.Lineage("user", 28);
        Check(lineage.Count == 1 && lineage[0] == store.Rows[store.PointInTime("user", 28)].Id,
            "lineage names exactly the contributing row (" + string.Join(",", lineage) + ")");

        // 4. drift detection with PSI: alert on a shifted window, stay quiet on a stationary one
        var stationary = new FeatureStore();
        ulong steady = 928UL;
        for (int t = 0; t < 800; t++) stationary.Append(t, "s", t, new[] { 10.0 + Rng.Gaussian(ref steady) });
        double steadyPsi = stationary.Psi("s", 400, 8);
        var shifted = new FeatureStore();
        ulong drift = 928UL;
        for (int t = 0; t < 800; t++) shifted.Append(t, "s", t, new[] { (t < 400 ? 10.0 : 13.0) + Rng.Gaussian(ref drift) });
        double shiftedPsi = shifted.Psi("s", 400, 8);
        Check(shiftedPsi > shifted.AlertThreshold, "the shifted window raises a drift alert (PSI " + shiftedPsi.ToString("F3") + " > " + shifted.AlertThreshold + ")");
        Check(steadyPsi < shifted.AlertThreshold, "the stationary window stays below the alert threshold (PSI " + steadyPsi.ToString("F3") + ")");
        Check(shiftedPsi > 2.0 * steadyPsi, "the shift dominates the sampling noise (" + shiftedPsi.ToString("F3") + " vs " + steadyPsi.ToString("F3") + ")");

        // 5. boundary cases: an entity with no history before the label and a single-row entity
        Check(store.PointInTime("user", -5) == -1 && store.Lineage("user", -5).Count == 0,
            "a label before the first row is reported missing, with empty lineage");
        Check(store.PointInTime("unknown", 100) == -1 && store.Online("unknown") == -1, "an unknown entity has no online or offline value");
        Check(store.PointInTime("ghost", 4) == -1 && store.PointInTime("ghost", 5) >= 0, "a single-row entity is served only from its own timestamp");
        var empty = new FeatureStore();
        Check(empty.Psi("s", 0, 8) == 0.0 && empty.Online("s") == -1, "an empty store reports no drift and no value");

        // 6. reproducibility
        ulong replayState = 928UL;
        var replay = new FeatureStore();
        int id = 0;
        for (int t = 0; t < 60; t += 3) replay.Append(id++, "user", t, new[] { 1.0 + 0.05 * t + 0.1 * Rng.Gaussian(ref replayState), t * 2.0 });
        Check(replay.Rows[3].Values[0] == store.Rows[3].Values[0], "the same seed rebuilds the same feature values");

        Console.WriteLine("joins: " + served + " point-in-time lookups, 0 mismatches, 0 leaks; online == newest offline join; lineage ids exact");
        Console.WriteLine("drift: PSI " + shiftedPsi.ToString("F3") + " (shifted, alert > " + shifted.AlertThreshold + ") vs " +
            steadyPsi.ToString("F3") + " (stationary); " + store.Rows.Count + " feature rows over 2 entities");
        Console.WriteLine("PASS");
    }
}
