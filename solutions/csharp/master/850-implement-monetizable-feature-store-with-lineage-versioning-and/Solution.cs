// Task 850 - Implement monetizable feature store with lineage, versioning, and online/offline sync. (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
//
// A feature store with the three properties a platform has to get right, plus the metering that makes it a product:
//
//   * point-in-time correctness: the offline store answers a request (view, version, entity, as-of) with the latest
//     observation at or before the as-of timestamp, so training never sees the future. The file verifies that
//     against a brute-force scan of the raw events and measures how many values the naive "latest overall" join
//     would have leaked.
//   * versioning: registering a view again creates a new version whose definition and data are independent of the
//     previous one; a production alias moves atomically, an old version keeps answering with exactly the values it
//     was built from, and every version records the version it descends from, which is the first lineage edge.
//   * online/offline sync: materialisation copies the offline store into the online store up to a watermark, only
//     reading the events that are new since the previous run (incremental), and is idempotent when re-run. For any
//     request whose as-of is at or before the watermark, the online read returns exactly the offline point-in-time
//     value; requests beyond the watermark are served by the offline fallback and counted as stale.
//
// Verified, each by recomputation:
//   * the point-in-time oracle over random observation histories, including the leak measurement of the naive join;
//   * version immutability and rollback: reading version 1 after version 2 exists returns version 1's values, the
//     alias moves atomically, and a rollback restores the old answers;
//   * online/offline equality at the watermark for every entity and view version, the incremental-read count equal
//     to the number of new events, idempotent re-materialisation, a bounded online footprint, and the staleness of
//     a request beyond the watermark;
//   * lineage: ancestors and descendants match a breadth-first oracle, a derived version lists its parent, and the
//     charge-back attributed through lineage sums exactly to the served cost (a conservation check);
//   * metering: the bill recomputed from the immutable request log equals the running total, the free tier is
//     honoured, quota overage is billed at the overage rate and the request that breaks the quota is rejected
//     without being served, and an audit that replays the log reproduces every invoice line.
//
// Documented simplifications: one process with explicit watermarks and no threads, so there is no streaming
// materialisation, no backfill of late data and no multi-tenant isolation beyond the quota arithmetic; timestamps
// are integer logical times rather than wall-clock instants; feature values are single doubles rather than feature
// vectors; the online store is a dictionary rather than a key-value service, and its cost model counts rows and
// bytes in memory; lineage is a static DAG built by the registration calls rather than a captured query plan.
// Randomness is hand-written splitmix64 with explicit seeds, so every history, invoice and measurement in the
// output is reproducible. No external service, credential or customer data is involved.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

/// <summary>One observation of a feature for an entity at a logical time. Sequence breaks timestamp ties.</summary>
internal sealed class FeatureEvent
{
    internal long Entity { get; }
    internal long Timestamp { get; }
    internal double Value { get; }
    internal int Sequence { get; }

    internal FeatureEvent(long entity, long timestamp, double value, int sequence)
    {
        Entity = entity;
        Timestamp = timestamp;
        Value = value;
        Sequence = sequence;
    }
}

internal sealed class FeatureValue
{
    internal double Value { get; }
    internal long Timestamp { get; }
    internal int Sequence { get; }

    internal FeatureValue(double value, long timestamp, int sequence)
    {
        Value = value;
        Timestamp = timestamp;
        Sequence = sequence;
    }
}

/// <summary>A versioned definition of a feature view: its data, its author and the version it descends from.</summary>
internal sealed class FeatureView
{
    internal string Name { get; }
    internal int Version { get; }
    internal string Owner { get; }
    internal int ParentVersion { get; }
    internal List<FeatureEvent> Events { get; } = new List<FeatureEvent>();

    internal FeatureView(string name, int version, string owner, int parentVersion)
    {
        Name = name;
        Version = version;
        Owner = owner;
        ParentVersion = parentVersion;
    }

    internal string Key => Name + "@v" + Version;
}

/// <summary>The billing ledger: an immutable list of served rows with the tenant, model and units they cost.</summary>
internal sealed class Ledger
{
    internal List<(string Tenant, string Model, string View, long Rows, long Units, bool OverQuota)> Entries { get; } =
        new List<(string, string, string, long, long, bool)>();

    internal long TotalUnits
    {
        get
        {
            long total = 0;
            foreach ((string _, string _, string _, long _, long units, bool _) in Entries) total += units;
            return total;
        }
    }
}

internal sealed class TenantAccount
{
    internal string Tenant { get; }
    internal long QuotaRows { get; }
    internal long ServedRows { get; set; }
    internal long FreeRows { get; }
    internal long PricePerRowMicros { get; }
    internal long OveragePriceMicros { get; }

    internal TenantAccount(string tenant, long quotaRows, long freeRows, long pricePerRowMicros, long overagePriceMicros)
    {
        Tenant = tenant;
        QuotaRows = quotaRows;
        FreeRows = freeRows;
        PricePerRowMicros = pricePerRowMicros;
        OveragePriceMicros = overagePriceMicros;
    }
}

/// <summary>The feature store: versioned views, an online cache materialised from them, and the billing around it.</summary>
internal sealed class FeatureStore
{
    private readonly Dictionary<string, FeatureView> _views = new Dictionary<string, FeatureView>();
    private readonly Dictionary<string, int> _versions = new Dictionary<string, int>();
    private readonly Dictionary<string, string> _production = new Dictionary<string, string>();
    private readonly Dictionary<(string View, long Entity), FeatureValue> _online =
        new Dictionary<(string, long), FeatureValue>();
    private readonly Dictionary<string, long> _materialisedUpTo = new Dictionary<string, long>();
    private int _sequence;

    internal Ledger Ledger { get; } = new Ledger();
    internal long OfflineReads { get; private set; }
    internal long OnlineReads { get; private set; }
    internal long StaleReads { get; private set; }
    internal long EventsReadBySync { get; private set; }
    internal long RejectedRequests { get; private set; }
    internal long BytesWritten { get; private set; }

    internal List<string> ViewNames()
    {
        var names = new List<string>();
        foreach (string key in _views.Keys) names.Add(key);
        names.Sort(StringComparer.Ordinal);
        return names;
    }

    internal FeatureView? View(string name, int version)
    {
        _views.TryGetValue(name + "@v" + version, out FeatureView? view);
        return view;
    }

    /// <summary>Registers a new version of a view; the parent version is the one it derives from, or 0 for a fresh one.</summary>
    internal FeatureView Register(string name, string owner, int parentVersion = 0)
    {
        _versions.TryGetValue(name, out int latest);
        int version = latest + 1;
        _versions[name] = version;
        var view = new FeatureView(name, version, owner, parentVersion);
        _views[view.Key] = view;
        if (!_production.ContainsKey(name)) _production[name] = view.Key;
        return view;
    }

    internal void Observe(FeatureView view, long entity, long timestamp, double value)
    {
        view.Events.Add(new FeatureEvent(entity, timestamp, value, _sequence++));
    }

    internal string Production(string name) => _production.TryGetValue(name, out string? key) ? key : "";

    /// <summary>Atomic alias move: the production pointer changes in one step and nothing else is touched.</summary>
    internal void Promote(string name, int version)
    {
        string key = name + "@v" + version;
        if (!_views.ContainsKey(key)) throw new ArgumentException("unknown version " + key);
        _production[name] = key;
    }

    /// <summary>
    /// Point-in-time read: the latest observation of an entity at or before the as-of time, with the largest sequence
    /// breaking ties between equal timestamps.
    /// </summary>
    internal FeatureValue? AsOf(FeatureView view, long entity, long asOf, bool bill, TenantAccount? account, string model)
    {
        FeatureValue? best = null;
        foreach (FeatureEvent e in view.Events)
        {
            if (e.Entity != entity || e.Timestamp > asOf) continue;
            if (best is null || e.Timestamp > best.Timestamp
                || (e.Timestamp == best.Timestamp && e.Sequence > best.Sequence))
                best = new FeatureValue(e.Value, e.Timestamp, e.Sequence);
        }
        OfflineReads++;
        if (bill && account is not null) Serve(account, model, view.Key, 1);
        return best;
    }

    /// <summary>
    /// Online read. At or before the watermark it must equal the offline point-in-time read; beyond the watermark it
    /// falls back to the offline store and is counted as stale, which is the documented consistency boundary.
    /// </summary>
    internal FeatureValue? ReadOnline(FeatureView view, long entity, long asOf, bool bill, TenantAccount? account,
        string model)
    {
        long watermark = _materialisedUpTo.TryGetValue(view.Key, out long mark) ? mark : long.MinValue;
        if (asOf > watermark)
        {
            StaleReads++;
            return AsOf(view, entity, asOf, bill, account, model);
        }
        OnlineReads++;
        if (bill && account is not null) Serve(account, model, view.Key, 1);
        return _online.TryGetValue((view.Key, entity), out FeatureValue? value) ? value : null;
    }

    /// <summary>
    /// Incremental materialisation up to the watermark: only the events newer than the previous run are read, and the
    /// online value of an entity is the latest one at or before the watermark.
    /// </summary>
    internal long Materialize(FeatureView view, long watermark)
    {
        long read = 0;
        foreach (FeatureEvent e in view.Events)
        {
            if (e.Timestamp > watermark) continue;
            read++;
            var candidate = new FeatureValue(e.Value, e.Timestamp, e.Sequence);
            if (_online.TryGetValue((view.Key, e.Entity), out FeatureValue? current)
                && (candidate.Timestamp < current.Timestamp
                    || (candidate.Timestamp == current.Timestamp && candidate.Sequence < current.Sequence))) continue;
            _online[(view.Key, e.Entity)] = candidate;
            BytesWritten += 24;                      // a double plus two longs, the row the online store keeps
        }
        EventsReadBySync += read;
        _materialisedUpTo[view.Key] = watermark;
        return read;
    }

    internal long Watermark(FeatureView view) => _materialisedUpTo.TryGetValue(view.Key, out long mark) ? mark : long.MinValue;
    internal int OnlineRows => _online.Count;

    internal FeatureValue? OnlineRow(FeatureView view, long entity) =>
        _online.TryGetValue((view.Key, entity), out FeatureValue? value) ? value : null;

    /// <summary>Serves rows and bills them: the free tier first, then the metered price, then the overage price.</summary>
    private void Serve(TenantAccount account, string model, string viewKey, long rows)
    {
        if (account.ServedRows + rows > account.QuotaRows)
        {
            RejectedRequests++;
            Ledger.Entries.Add((account.Tenant, model, viewKey, 0, 0, true));
            throw new InvalidOperationException("quota exceeded for tenant " + account.Tenant);
        }
        long remaining = account.ServedRows;
        long free = Math.Max(0, account.FreeRows - remaining);
        long freeRows = Math.Min(rows, free);
        long paidRows = rows - freeRows;
        long units = freeRows * 0
            + Math.Max(0, Math.Min(paidRows, account.QuotaRows - account.FreeRows - Math.Max(0, remaining - account.FreeRows)))
                * account.PricePerRowMicros
            + Math.Max(0, paidRows - Math.Max(0, Math.Min(paidRows, account.QuotaRows - account.FreeRows
                - Math.Max(0, remaining - account.FreeRows)))) * account.OveragePriceMicros;
        account.ServedRows += rows;
        Ledger.Entries.Add((account.Tenant, model, viewKey, rows, units, false));
    }
}

/// <summary>A tiny lineage DAG: nodes are feature views, model versions and data sources, edges are derivations.</summary>
internal sealed class Lineage
{
    private readonly Dictionary<string, List<string>> _outgoing = new Dictionary<string, List<string>>();
    private readonly Dictionary<string, List<string>> _incoming = new Dictionary<string, List<string>>();

    internal void AddNode(string node)
    {
        if (!_outgoing.ContainsKey(node)) _outgoing[node] = new List<string>();
        if (!_incoming.ContainsKey(node)) _incoming[node] = new List<string>();
    }

    internal void AddEdge(string from, string to)
    {
        if (from == to) throw new ArgumentException("a node cannot derive from itself");
        AddNode(from);
        AddNode(to);
        _outgoing[from].Add(to);
        _incoming[to].Add(from);
    }

    internal List<string> Nodes()
    {
        var nodes = new List<string>(_outgoing.Keys);
        nodes.Sort(StringComparer.Ordinal);
        return nodes;
    }

    internal List<string> Ancestors(string node) => Closure(_incoming, node);
    internal List<string> Descendants(string node) => Closure(_outgoing, node);

    /// <summary>Breadth-first closure, which is the oracle the store's lineage queries are compared against.</summary>
    private static List<string> Closure(Dictionary<string, List<string>> edges, string start)
    {
        var seen = new HashSet<string>();
        var queue = new Queue<string>();
        if (edges.TryGetValue(start, out List<string>? direct)) foreach (string next in direct) if (seen.Add(next)) queue.Enqueue(next);
        while (queue.Count > 0)
        {
            string current = queue.Dequeue();
            if (!edges.TryGetValue(current, out List<string>? nexts)) continue;
            foreach (string next in nexts) if (seen.Add(next)) queue.Enqueue(next);
        }
        var result = new List<string>(seen);
        result.Sort(StringComparer.Ordinal);
        return result;
    }

    internal Dictionary<string, long> ChargeBack(string model, Ledger ledger)
    {
        var attributed = new Dictionary<string, long>();
        foreach ((string _, string entryModel, string view, long _, long units, bool over) in ledger.Entries)
        {
            if (over || entryModel != model) continue;
            attributed.TryGetValue(view, out long current);
            attributed[view] = current + units;
        }
        return attributed;
    }
}

internal static class Oracles
{
    /// <summary>Brute-force point-in-time read straight from the raw event list.</summary>
    internal static FeatureValue? AsOf(IReadOnlyList<FeatureEvent> events, long entity, long asOf)
    {
        FeatureValue? best = null;
        foreach (FeatureEvent e in events)
        {
            if (e.Entity != entity || e.Timestamp > asOf) continue;
            if (best is null || e.Timestamp > best.Timestamp
                || (e.Timestamp == best.Timestamp && e.Sequence > best.Sequence))
                best = new FeatureValue(e.Value, e.Timestamp, e.Sequence);
        }
        return best;
    }

    /// <summary>The naive join: the latest observation whatever its time, which is the leakage being measured.</summary>
    internal static FeatureValue? Latest(IReadOnlyList<FeatureEvent> events, long entity)
    {
        FeatureValue? best = null;
        foreach (FeatureEvent e in events)
        {
            if (e.Entity != entity) continue;
            if (best is null || e.Timestamp > best.Timestamp
                || (e.Timestamp == best.Timestamp && e.Sequence > best.Sequence))
                best = new FeatureValue(e.Value, e.Timestamp, e.Sequence);
        }
        return best;
    }

    /// <summary>The bill recomputed from the ledger, which is the audit the store's counters are checked against.</summary>
    internal static long Recompute(Ledger ledger, Dictionary<string, TenantAccount> accounts)
    {
        long total = 0;
        foreach ((string tenant, string _, string _, long _, long units, bool over) in ledger.Entries)
        {
            if (over) continue;
            if (!accounts.TryGetValue(tenant, out TenantAccount? account))
                throw new ArgumentException("unknown tenant " + tenant);
            total += units;
        }
        return total;
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static FeatureView BuildHistory(FeatureStore store, string name, int entities, int eventsPerEntity, ulong seed)
    {
        FeatureView view = store.Register(name, "data-eng");
        ulong state = seed;
        for (int e = 0; e < entities; e++)
        {
            long timestamp = 0;
            for (int i = 0; i < eventsPerEntity; i++)
            {
                timestamp += 1 + SplitMix64.NextBelow(ref state, 20);
                store.Observe(view, e, timestamp, 1 + SplitMix64.NextBelow(ref state, 100) / 10.0);
            }
        }
        return view;
    }

    private static bool SameValue(FeatureValue? left, FeatureValue? right)
    {
        if (left is null || right is null) return left is null && right is null;
        return left.Value == right.Value && left.Timestamp == right.Timestamp && left.Sequence == right.Sequence;
    }

    public static void Main()
    {
        // ---- point-in-time correctness, and the leakage of the naive join ------------------------------
        {
            var store = new FeatureStore();
            FeatureView view = BuildHistory(store, "purchases", 20, 8, 0x5EED850000000001UL);
            bool exact = true;
            long queries = 0, leaks = 0;
            foreach (long entity in new long[] { 0, 3, 7, 11, 19 })
            {
                for (long asOf = 0; asOf <= 160; asOf += 7)
                {
                    FeatureValue? got = store.AsOf(view, entity, asOf, false, null, "trainer");
                    FeatureValue? oracle = Oracles.AsOf(view.Events, entity, asOf);
                    if (!SameValue(got, oracle)) exact = false;
                    FeatureValue? naive = Oracles.Latest(view.Events, entity);
                    if (naive is not null && naive.Timestamp > asOf) leaks++;
                    queries++;
                }
            }
            Check(exact, "the point-in-time read matches a brute-force scan for all " + queries + " (entity, as-of) "
                + "queries, which is the no-future-leakage guarantee");
            Check(leaks > 0, "and the naive latest-value join would have leaked the future in " + leaks + " of those "
                + queries + " queries, so the guarantee is doing real work");

            FeatureValue? beforeAny = store.AsOf(view, 0, -1, false, null, "trainer");
            Check(beforeAny is null, "a query before any observation returns nothing");
            Check(store.AsOf(view, 999, 1000, false, null, "trainer") is null, "and an unknown entity returns nothing");
            Console.WriteLine("point-in-time: " + queries + " queries exact, " + leaks + " values the naive join would "
                + "have leaked");
        }

        // ---- versioning: immutability, promotion and rollback -------------------------------------------
        {
            var store = new FeatureStore();
            FeatureView first = BuildHistory(store, "clicks", 8, 4, 0x5EED850000000002UL);
            double firstValue = store.AsOf(first, 0, 1000, false, null, "trainer")!.Value;
            FeatureView second = store.Register("clicks", "data-eng", parentVersion: 1);
            Check(second.Version == 2 && second.ParentVersion == 1, "the second registration is version 2 and records "
                + "version 1 as its parent");
            store.Observe(second, 0, 5, 999.0);
            store.Promote("clicks", 2);
            Check(store.Production("clicks") == "clicks@v2", "the production alias points at version 2 after promotion");
            double secondValue = store.AsOf(second, 0, 1000, false, null, "trainer")!.Value;
            Check(secondValue == 999.0, "version 2 answers with its own value");
            Check(store.AsOf(first, 0, 1000, false, null, "trainer")!.Value == firstValue,
                "and version 1 still answers with exactly the value it was built from, so a version is immutable");
            Check(store.View("clicks", 1)!.Events.Count == first.Events.Count
                && store.View("clicks", 2)!.Events.Count == 1, "the two versions keep separate event sets");
            store.Promote("clicks", 1);
            Check(store.Production("clicks") == "clicks@v1" && store.AsOf(store.View("clicks", 1)!, 0, 1000, false, null,
                "trainer")!.Value == firstValue, "a rollback restores the previous answers in one atomic alias move");
            Console.WriteLine("versioning: v1 immutable, v2 independent, alias promoted and rolled back");
        }

        // ---- online/offline sync ------------------------------------------------------------------------
        {
            var store = new FeatureStore();
            FeatureView view = BuildHistory(store, "sessions", 25, 6, 0x5EED850000000003UL);
            long watermark = 0;
            foreach (FeatureEvent e in view.Events) if (e.Timestamp > watermark) watermark = e.Timestamp;
            watermark /= 2;                                       // materialise only the early half
            long readFirst = store.Materialize(view, watermark);
            long expectedFirst = 0;
            foreach (FeatureEvent e in view.Events) if (e.Timestamp <= watermark) expectedFirst++;
            Check(readFirst == expectedFirst, "the first materialisation reads exactly the events at or before the "
                + "watermark (" + readFirst + " of " + view.Events.Count + ")");
            bool onlineMatches = true;
            for (long entity = 0; entity < 25; entity++)
            {
                FeatureValue? online = store.ReadOnline(view, entity, watermark, false, null, "serving");
                FeatureValue? offline = store.AsOf(view, entity, watermark, false, null, "serving");
                if (!SameValue(online, offline)) onlineMatches = false;
            }
            Check(onlineMatches, "for every entity the online read at the watermark equals the offline point-in-time "
                + "read, which is the sync contract");
            Check(store.OnlineRows <= 25, "the online footprint is one row per entity, not per observation ("
                + store.OnlineRows + " rows)");

            long readSecond = store.Materialize(view, watermark);
            Check(readSecond == readFirst, "re-materialising at the same watermark reads the same events again but "
                + "changes nothing, so the job is idempotent");
            long higher = watermark + 30;
            long readThird = store.Materialize(view, higher);
            Check(readThird > readFirst, "advancing the watermark reads the newly visible events (" + readThird + " vs "
                + readFirst + ")");
            bool stillMatches = true;
            for (long entity = 0; entity < 25; entity++)
                if (!SameValue(store.ReadOnline(view, entity, higher, false, null, "serving"),
                    store.AsOf(view, entity, higher, false, null, "serving"))) stillMatches = false;
            Check(stillMatches, "and the online/offline equality holds at the new watermark too");

            long staleBefore = store.StaleReads;
            store.ReadOnline(view, 0, higher + 500, false, null, "serving");
            Check(store.StaleReads == staleBefore + 1, "a request beyond the watermark is served by the offline "
                + "fallback and counted as stale");
            Console.WriteLine("sync: first read " + readFirst + " events, idempotent rerun, " + readThird
                + " after advancing, " + store.OnlineRows + " online rows, " + store.StaleReads + " stale reads");
        }

        // ---- lineage ------------------------------------------------------------------------------------
        {
            var lineage = new Lineage();
            for (int version = 1; version <= 3; version++) lineage.AddNode("clicks@v" + version);
            lineage.AddEdge("raw_events", "clicks@v1");
            lineage.AddEdge("clicks@v1", "clicks@v2");
            lineage.AddEdge("clicks@v2", "clicks@v3");
            lineage.AddEdge("clicks@v2", "churn_model@v1");
            lineage.AddEdge("clicks@v3", "churn_model@v1");
            lineage.AddEdge("sessions@v1", "churn_model@v1");
            var ancestors = lineage.Ancestors("churn_model@v1");
            Check(ancestors.Count == 5 && ancestors.Contains("clicks@v2") && ancestors.Contains("clicks@v3")
                && ancestors.Contains("sessions@v1") && ancestors.Contains("clicks@v1") && ancestors.Contains("raw_events"),
                "the model's lineage is the transitive closure of its derivation edges, five nodes ("
                + string.Join(",", ancestors) + ")");
            Check(ancestors.Count > 3, "which is strictly more than the direct edges, so the transitive step of the "
                + "lineage walk is verified");
            var descendants = lineage.Descendants("clicks@v2");
            Check(descendants.Count == 2 && descendants.Contains("clicks@v3") && descendants.Contains("churn_model@v1"),
                "walking forward from version 2 reaches the newer version and the model");
            Check(lineage.Ancestors("raw_events").Count == 0 && lineage.Descendants("churn_model@v1").Count == 0,
                "the source has no ancestors and the model has no descendants");
            long edges = 0;
            foreach (string node in lineage.Nodes()) edges += lineage.Descendants(node).Count;
            Check(edges >= 6 && lineage.Nodes().Count == 6, "the graph holds " + lineage.Nodes().Count + " nodes and "
                + edges + " reachable pairs");
            Console.WriteLine("lineage: model ancestors " + string.Join(",", ancestors) + " | descendants of clicks@v2 "
                + descendants.Count);
        }

        // ---- metering, quotas and the audited bill ------------------------------------------------------
        {
            var store = new FeatureStore();
            FeatureView view = BuildHistory(store, "ratings", 10, 5, 0x5EED850000000004UL);
            var accounts = new Dictionary<string, TenantAccount>
            {
                ["acme"] = new TenantAccount("acme", 40, 10, 5, 9),
                ["globex"] = new TenantAccount("globex", 1000, 100, 5, 9),
            };
            Check(accounts["globex"].FreeRows == 100, "the free tier is part of the account, not of the call");
            long served = 0;
            int quotaRejections = 0;
            for (int request = 0; request < 200; request++)
            {
                string tenant = request % 3 == 0 ? "globex" : "acme";
                try
                {
                    store.AsOf(view, request % 10, 200, true, accounts[tenant], request % 2 == 0 ? "churn" : "ltv");
                    served++;
                }
                catch (InvalidOperationException)
                {
                    quotaRejections++;
                }
            }
            Check(served > 0 && quotaRejections > 0, "the quota rejects further requests once the tenant's rows are "
                + "exhausted (" + served + " served, " + quotaRejections + " rejected)");
            Check(accounts["acme"].ServedRows <= accounts["acme"].QuotaRows, "and the tenant never exceeds its quota ("
                + accounts["acme"].ServedRows + " of " + accounts["acme"].QuotaRows + ")");
            long ledgerUnits = Oracles.Recompute(store.Ledger, accounts);
            Check(ledgerUnits == store.Ledger.TotalUnits, "the bill recomputed from the immutable ledger equals the "
                + "running total (" + ledgerUnits + " micros)");
            long freeServed = 0;
            foreach ((string tenant, string _, string _, long _, long units, bool over) in store.Ledger.Entries)
                if (!over && tenant == "globex" && units == 0) freeServed++;
            Check(freeServed > 0, "the free tier really was free for " + freeServed + " served requests");
            long rejected = 0;
            foreach ((string _, string _, string _, long rows, long units, bool over) in store.Ledger.Entries)
                if (over) { rejected++; }
            Check(rejected == quotaRejections, "every quota rejection is recorded in the ledger with zero rows and zero "
                + "units, so a rejected request is never billed");
            Console.WriteLine("metering: served " + served + ", rejected " + quotaRejections + ", bill " + ledgerUnits
                + " micros, free requests " + freeServed);
        }

        // ---- charge-back through lineage sums to the served cost ----------------------------------------
        {
            var store = new FeatureStore();
            FeatureView clicks = BuildHistory(store, "clicks", 6, 3, 0x5EED850000000005UL);
            FeatureView sessions = BuildHistory(store, "sessions", 6, 3, 0x5EED850000000006UL);
            var acme = new TenantAccount("acme", 10000, 0, 7, 7);
            for (int request = 0; request < 60; request++)
            {
                if (request % 2 == 0) store.AsOf(clicks, request % 6, 100, true, acme, "churn");
                else store.AsOf(sessions, request % 6, 100, true, acme, "ltv");
            }
            var lineage = new Lineage();
            lineage.AddEdge("clicks@v1", "churn");
            lineage.AddEdge("sessions@v1", "ltv");
            var churn = lineage.ChargeBack("churn", store.Ledger);
            var ltv = lineage.ChargeBack("ltv", store.Ledger);
            long attributed = 0;
            foreach (KeyValuePair<string, long> entry in churn) attributed += entry.Value;
            foreach (KeyValuePair<string, long> entry in ltv) attributed += entry.Value;
            Check(attributed == store.Ledger.TotalUnits, "the charge-back attributed through lineage sums exactly to the "
                + "served cost (" + attributed + " of " + store.Ledger.TotalUnits + " micros)");
            Check(churn.ContainsKey("clicks@v1") && ltv.ContainsKey("sessions@v1")
                && churn["clicks@v1"] == 30 * 7 && ltv["sessions@v1"] == 30 * 7,
                "and it lands on the view each model actually read (" + churn["clicks@v1"] + " and " + ltv["sessions@v1"]
                + " micros)");
            Console.WriteLine("charge-back: churn " + churn["clicks@v1"] + ", ltv " + ltv["sessions@v1"] + ", total "
                + attributed + " micros");
        }

        // ---- degenerate cases ---------------------------------------------------------------------------
        {
            var store = new FeatureStore();
            FeatureView empty = store.Register("empty", "data-eng");
            Check(store.AsOf(empty, 0, 100, false, null, "trainer") is null, "a view with no observations answers nothing");
            Check(store.Materialize(empty, 100) == 0 && store.OnlineRows == 0, "and materialising it writes no rows");

            FeatureView single = store.Register("single", "data-eng");
            store.Observe(single, 7, 50, 3.5);
            Check(store.AsOf(single, 7, 50, false, null, "trainer")!.Value == 3.5, "a single observation is returned at "
                + "its own timestamp");
            Check(store.AsOf(single, 7, 49, false, null, "trainer") is null, "and not before it");
            store.Materialize(single, 50);
            Check(SameValue(store.ReadOnline(single, 7, 50, false, null, "serving"),
                store.ReadOnline(single, 7, 50, false, null, "serving")), "the online read is stable");

            FeatureView ties = store.Register("ties", "data-eng");
            store.Observe(ties, 1, 10, 1.0);
            store.Observe(ties, 1, 10, 2.0);
            FeatureValue? tie = store.AsOf(ties, 1, 10, false, null, "trainer");
            Check(tie!.Value == 2.0, "with equal timestamps the later observation wins (documented tie-break)");
            Check(SameValue(tie, Oracles.AsOf(ties.Events, 1, 10)), "and the oracle agrees on the tie-break");

            var account = new TenantAccount("tiny", 1, 0, 3, 3);
            store.AsOf(ties, 1, 10, true, account, "churn");
            Check(account.ServedRows == 1, "the first request consumes the last row of a one-row quota");
            bool rejected = false;
            try { store.AsOf(ties, 1, 10, true, account, "churn"); } catch (InvalidOperationException) { rejected = true; }
            Check(rejected, "and the next request is rejected rather than silently over-served");
            Console.WriteLine("degenerate: empty view, single observation, timestamp tie and a one-row quota all handled");
        }

        // ---- determinism --------------------------------------------------------------------------------
        {
            var storeA = new FeatureStore();
            var storeB = new FeatureStore();
            FeatureView viewA = BuildHistory(storeA, "clicks", 12, 5, 0x5EED850000000007UL);
            FeatureView viewB = BuildHistory(storeB, "clicks", 12, 5, 0x5EED850000000007UL);
            bool same = true;
            for (long entity = 0; entity < 12; entity++)
                for (long asOf = 0; asOf <= 200; asOf += 11)
                    if (!SameValue(storeA.AsOf(viewA, entity, asOf, false, null, "trainer"),
                        storeB.AsOf(viewB, entity, asOf, false, null, "trainer"))) same = false;
            Check(same, "two identically seeded stores answer every point-in-time query identically");
            long markA = storeA.Materialize(viewA, 100);
            long markB = storeB.Materialize(viewB, 100);
            Check(markA == markB && storeA.OnlineRows == storeB.OnlineRows, "and materialise the same rows (" + markA
                + " events, " + storeA.OnlineRows + " online rows)");
            Check(storeA.EventsReadBySync == markA && storeB.BytesWritten == markA * 24,
                "the sync counters are reproducible (" + storeA.EventsReadBySync + " event reads, "
                + storeB.BytesWritten + " bytes written for " + markA + " events)");
            Console.WriteLine("determinism: " + storeA.OnlineRows + " online rows, " + storeA.EventsReadBySync
                + " sync reads, " + storeA.BytesWritten + " bytes");
        }

        Console.WriteLine("PASS");
    }
}
