// Task 888 - Implement incremental view maintenance with delta propagation and bounded memory. (Tier: Master, Ch 14 — Databases)
//
// A view  SELECT region, SUM(amount), COUNT(*) FROM orders JOIN customers ON customerId
//          WHERE amount > 50 GROUP BY region  is maintained incrementally from Z-set deltas.
//
// Changes are appended to a *bounded* pending log (a merged dictionary of table/tuple -> multiplicity,
// capacity K). The log is flushed when it overflows or when a query needs a consistent read, and the
// flush pushes the merged deltas through the pipeline:
//      Δ(filtered orders) = filter(Δorders)
//      Δ(filtered ⋈ customers) = ΔF ⋈ C  +  F ⋈ ΔC  +  ΔF ⋈ ΔC
//      Δaggregate = fold the joined delta into (sum, count) per group, dropping groups that reach zero
// The third join term is the one everybody forgets: a batch that changes both sides of a join in a
// matching way is exactly where two-term propagation goes wrong, and the tests demonstrate that.
//
// Transactional: a batch is validated as a whole (no multiplicity may become negative, the table and
// the amount must parse) and only then logged - a batch with one bad change leaves the state untouched.
//
// Simplifications (documented, not behavioural): string-encoded tuples (values may not contain '|'),
// a single join + filter + group-by pipeline instead of a general planner, and work is measured in
// tuple visits rather than milliseconds. Values are 64-bit; the aggregate is SUM/COUNT (AVG is derived).
//
// Verified: after every batch the incremental view equals an independent full-recompute oracle (also
// for duplicate-heavy bag data, whole-table deletes and groups that vanish); the flush cost equals an
// independently recomputed sum of |ΔF|*|C| + |F|*|ΔC| + |ΔF|*|ΔC| term products; the pending log never
// exceeds K; incremental work stays far below repeated full recomputation; a deliberately two-term
// propagation diverges exactly on the batch that changes both sides; rejected batches (negative
// multiplicity, unknown table, malformed amount) are atomic; the filter boundary (amount == 50
// excluded, 51 included) and empty relations behave; and the whole sequence is deterministic.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

internal sealed class Change
{
    internal string Table = "";
    internal string[] Values = Array.Empty<string>();
    internal long Delta;

    internal Change(string table, long delta, params string[] values)
    {
        Table = table;
        Delta = delta;
        Values = values;
    }
}

internal sealed class Table
{
    internal string[] Columns;
    private readonly SortedDictionary<string, long> _rows = new SortedDictionary<string, long>(StringComparer.Ordinal);

    internal Table(string[] columns) => Columns = columns;

    internal string Key(IReadOnlyList<string> values)
    {
        if (values.Count != Columns.Length) throw new InvalidOperationException("arity mismatch for table " + Columns[0]);
        foreach (string value in values)
            if (value.IndexOf('|') >= 0) throw new InvalidOperationException("'|' is not allowed inside a value");
        return string.Join("|", values);
    }

    internal static string[] Split(string tuple) => tuple.Split('|');

    internal long Multiplicity(string tuple) => _rows.TryGetValue(tuple, out long value) ? value : 0;

    internal void Apply(string tuple, long delta)
    {
        long next = Multiplicity(tuple) + delta;
        if (next == 0) _rows.Remove(tuple);
        else _rows[tuple] = next;
    }

    internal IEnumerable<KeyValuePair<string, long>> Live() => _rows.Where(kv => kv.Value > 0);

    internal int LiveCount() => _rows.Count(kv => kv.Value > 0);

    internal long TotalMultiplicity() => _rows.Where(kv => kv.Value > 0).Sum(kv => kv.Value);
}

internal struct FlushStats
{
    internal int OrderDelta;
    internal int FilteredDelta;
    internal int CustomerDelta;
    internal int FilteredRows;
    internal int CustomerRows;
    internal int JoinedGroups;
}

internal sealed class ViewPipeline
{
    internal const long MinAmount = 50;                 // filter is amount > 50
    internal const int PendingCapacity = 8;             // bounded memory: distinct pending tuples

    internal readonly Table Orders = new Table(new[] { "orderId", "customerId", "amount" });
    internal readonly Table Customers = new Table(new[] { "customerId", "region" });
    internal readonly Table Filtered = new Table(new[] { "customerId", "amount" });
    internal readonly SortedDictionary<string, (long Sum, long Count)> View = new SortedDictionary<string, (long, long)>(StringComparer.Ordinal);
    internal readonly List<FlushStats> Flushes = new List<FlushStats>();

    private readonly SortedDictionary<string, long> _pending = new SortedDictionary<string, long>(StringComparer.Ordinal);

    internal bool ThreeTermJoinDelta = true;
    internal long IncrementalOps;
    internal long PropagatedDeltas;
    internal long RejectedBatches;

    internal int PendingCount => _pending.Count;

    private Table TableByName(string name)
    {
        switch (name)
        {
            case "orders": return Orders;
            case "customers": return Customers;
            default: throw new InvalidOperationException("unknown table '" + name + "'");
        }
    }

    private static (string Table, string Tuple) SplitKey(string key)
    {
        int cut = key.IndexOf('\u0001');
        return (key.Substring(0, cut), key.Substring(cut + 1));
    }

    /** Atomic batch: validate everything, then log; overflow or a query drives the flush. */
    internal void ApplyBatch(IReadOnlyList<Change> changes)
    {
        var tentative = new SortedDictionary<string, long>(StringComparer.Ordinal);
        foreach (Change change in changes)
        {
            Table table = TableByName(change.Table);
            string tuple = table.Key(change.Values);
            if (change.Table == "orders" && !long.TryParse(Table.Split(tuple)[2], NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out long _))
                throw new InvalidOperationException("amount is not an integer: " + tuple);
            string key = change.Table + "\u0001" + tuple;
            tentative[key] = tentative.TryGetValue(key, out long current) ? current + change.Delta : change.Delta;
        }

        foreach (KeyValuePair<string, long> entry in tentative)
        {
            (string table, string tuple) = SplitKey(entry.Key);
            if (TableByName(table).Multiplicity(tuple) + entry.Value < 0)
            {
                RejectedBatches++;
                throw new InvalidOperationException("change would make multiplicity negative: " + table + " " + tuple);
            }
        }

        foreach (KeyValuePair<string, long> entry in tentative)
        {
            long merged = _pending.TryGetValue(entry.Key, out long current) ? current + entry.Value : entry.Value;
            if (merged == 0) _pending.Remove(entry.Key);
            else _pending[entry.Key] = merged;
        }

        if (_pending.Count > PendingCapacity) Flush();
    }

    internal void Flush()
    {
        if (_pending.Count == 0) return;

        var orderDelta = new SortedDictionary<string, long>(StringComparer.Ordinal);
        var customerDelta = new SortedDictionary<string, long>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, long> entry in _pending)
        {
            (string table, string tuple) = SplitKey(entry.Key);
            if (table == "orders") orderDelta[tuple] = entry.Value;
            else customerDelta[tuple] = entry.Value;
            PropagatedDeltas++;
        }
        _pending.Clear();

        // Term 0: push the filter down into Δorders.
        var filteredDelta = new SortedDictionary<string, long>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, long> entry in orderDelta)
        {
            string[] values = Table.Split(entry.Key);
            IncrementalOps++;
            if (long.Parse(values[2], CultureInfo.InvariantCulture) <= MinAmount) continue;
            string tuple = values[1] + "|" + values[2];
            filteredDelta[tuple] = filteredDelta.TryGetValue(tuple, out long current) ? current + entry.Value : entry.Value;
        }

        // The three join terms, all evaluated against the pre-flush state.
        var joinedDelta = new SortedDictionary<string, long>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, long> delta in filteredDelta)
        {
            string[] values = Table.Split(delta.Key);
            foreach (KeyValuePair<string, long> customer in Customers.Live())
            {
                IncrementalOps++;
                if (Table.Split(customer.Key)[0] != values[0]) continue;
                Add(joinedDelta, Table.Split(customer.Key)[1] + "|" + values[1], delta.Value * customer.Value);
            }
        }
        foreach (KeyValuePair<string, long> row in Filtered.Live())
        {
            string[] values = Table.Split(row.Key);
            foreach (KeyValuePair<string, long> delta in customerDelta)
            {
                IncrementalOps++;
                if (Table.Split(delta.Key)[0] != values[0]) continue;
                Add(joinedDelta, Table.Split(delta.Key)[1] + "|" + values[1], row.Value * delta.Value);
            }
        }
        if (ThreeTermJoinDelta)
        {
            foreach (KeyValuePair<string, long> delta in filteredDelta)
            {
                string[] values = Table.Split(delta.Key);
                foreach (KeyValuePair<string, long> customerDeltaEntry in customerDelta)
                {
                    IncrementalOps++;
                    if (Table.Split(customerDeltaEntry.Key)[0] != values[0]) continue;
                    Add(joinedDelta, Table.Split(customerDeltaEntry.Key)[1] + "|" + values[1], delta.Value * customerDeltaEntry.Value);
                }
            }
        }

        Flushes.Add(new FlushStats
        {
            OrderDelta = orderDelta.Count,
            FilteredDelta = filteredDelta.Count,
            CustomerDelta = customerDelta.Count,
            FilteredRows = Filtered.LiveCount(),
            CustomerRows = Customers.LiveCount(),
            JoinedGroups = joinedDelta.Count,
        });

        foreach (KeyValuePair<string, long> entry in orderDelta) Orders.Apply(entry.Key, entry.Value);
        foreach (KeyValuePair<string, long> entry in customerDelta) Customers.Apply(entry.Key, entry.Value);
        foreach (KeyValuePair<string, long> entry in filteredDelta) Filtered.Apply(entry.Key, entry.Value);

        foreach (KeyValuePair<string, long> entry in joinedDelta)
        {
            string[] values = Table.Split(entry.Key);
            long amount = long.Parse(values[1], CultureInfo.InvariantCulture);
            IncrementalOps++;
            (long Sum, long Count) group = View.TryGetValue(values[0], out (long, long) current) ? current : (0L, 0L);
            long sum = group.Sum + amount * entry.Value;
            long count = group.Count + entry.Value;
            if (count == 0) View.Remove(values[0]);
            else View[values[0]] = (sum, count);
        }
    }

    private static void Add(SortedDictionary<string, long> map, string key, long delta)
    {
        long merged = map.TryGetValue(key, out long current) ? current + delta : delta;
        if (merged == 0) map.Remove(key);
        else map[key] = merged;
    }

    /** A read must observe every logged change, so it flushes first. */
    internal string Query()
    {
        Flush();
        return Canonical(View);
    }

    internal static string Canonical(SortedDictionary<string, (long Sum, long Count)> view) =>
        string.Join(";", view.Select(kv => kv.Key + "=" + kv.Value.Sum + ":" + kv.Value.Count));
}

internal static class Oracle
{
    /** Independent nested-loop evaluation of the view from the base tables. */
    internal static SortedDictionary<string, (long Sum, long Count)> Recompute(Table orders, Table customers, out long ops)
    {
        var view = new SortedDictionary<string, (long, long)>(StringComparer.Ordinal);
        ops = 0;
        foreach (KeyValuePair<string, long> order in orders.Live())
        {
            string[] orderValues = Table.Split(order.Key);
            long amount = long.Parse(orderValues[2], CultureInfo.InvariantCulture);
            if (amount <= ViewPipeline.MinAmount) continue;
            foreach (KeyValuePair<string, long> customer in customers.Live())
            {
                ops++;
                if (Table.Split(customer.Key)[0] != orderValues[1]) continue;
                string region = Table.Split(customer.Key)[1];
                long multiplicity = order.Value * customer.Value;
                (long Sum, long Count) group = view.TryGetValue(region, out (long, long) current) ? current : (0L, 0L);
                view[region] = (group.Sum + amount * multiplicity, group.Count + multiplicity);
            }
        }
        return view;
    }

    internal static string Canonical(SortedDictionary<string, (long Sum, long Count)> view) => ViewPipeline.Canonical(view);
}

public static class Solution
{
    private static ulong _state = 0x2545F4914F6CDD1DUL;

    private static ulong NextRandom()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }

    private static Change Insert(string table, params string[] values) => new Change(table, 1, values);

    private static Change Delete(string table, params string[] values) => new Change(table, -1, values);

    /** The full-recompute oracle, with its work counted so the two strategies can be compared. */
    private static string OracleView(ViewPipeline pipeline, out long ops)
    {
        SortedDictionary<string, (long Sum, long Count)> recomputed = Oracle.Recompute(pipeline.Orders, pipeline.Customers, out ops);
        return Oracle.Canonical(recomputed);
    }

    private static List<Change[]> BuildWorkload(int batches, int width)
    {
        var workload = new List<Change[]>();
        for (int b = 0; b < batches; b++)
        {
            var batch = new List<Change>();
            for (int k = 0; k < width; k++)
            {
                long orderId = b * 10 + k;
                string customer = "c" + (orderId % 12);
                long amount = 20 + (long)(NextRandom() % 120);
                batch.Add(Insert("orders", "o" + orderId, customer, amount.ToString(CultureInfo.InvariantCulture)));
            }
            if (b % 3 == 1)
            {
                string region = new[] { "eu", "us", "apac" }[b % 3];
                batch.Add(Insert("customers", "c" + (b % 12), region));
            }
            workload.Add(batch.ToArray());
        }
        return workload;
    }

    public static void Main()
    {
        // ---- 1. the reference workload: three customers, four orders --------------------------
        var pipeline = new ViewPipeline();
        Check(pipeline.Query() == "", "an empty pipeline has an empty view");
        long oracleOps;
        Check(pipeline.Query() == OracleView(pipeline, out oracleOps), "empty view matches the oracle");

        pipeline.ApplyBatch(new[]
        {
            Insert("customers", "c1", "eu"), Insert("customers", "c2", "us"), Insert("customers", "c3", "eu"),
        });
        pipeline.ApplyBatch(new[]
        {
            Insert("orders", "o1", "c1", "100"),     // counted
            Insert("orders", "o2", "c1", "50"),      // filter boundary: not > 50
            Insert("orders", "o3", "c2", "51"),      // just above the boundary
            Insert("orders", "o4", "c3", "200"),
            Insert("orders", "o5", "c9", "300"),     // no matching customer: no group
        });
        Check(pipeline.Query() == OracleView(pipeline, out oracleOps), "first batch matches the oracle");
        Check(pipeline.View["eu"] == (300L, 2L), "eu is the sum of o1 and o4 with count 2");
        Check(pipeline.View["us"] == (51L, 1L), "the boundary order is included");
        Check(!pipeline.View.ContainsKey("apac"), "no group without matching rows");
        Check(pipeline.Query() == "eu=300:2;us=51:1", "canonical view text");

        // ---- 2. duplicate rows are a bag, not a set --------------------------------------------
        var bag = new ViewPipeline();
        bag.ApplyBatch(new[] { Insert("customers", "c1", "eu") });
        bag.ApplyBatch(new[] { Insert("orders", "o1", "c1", "70"), Insert("orders", "o1", "c1", "70") });
        Check(bag.Query() == OracleView(bag, out oracleOps), "duplicate inserts follow bag semantics");
        Check(bag.View["eu"] == (140L, 2L), "duplicates double both sum and count");
        bag.ApplyBatch(new[] { Delete("orders", "o1", "c1", "70") });
        Check(bag.Query() == "eu=70:1", "one duplicate removed");
        bag.ApplyBatch(new[] { Delete("orders", "o1", "c1", "70") });
        Check(bag.Query() == "" && bag.Filtered.LiveCount() == 0, "removing the last row empties the group");

        // ---- 3. retractions, whole-table deletes and vanishing groups --------------------------
        var shrink = new ViewPipeline();
        shrink.ApplyBatch(new[] { Insert("customers", "c1", "eu"), Insert("customers", "c2", "us") });
        shrink.ApplyBatch(new[]
        {
            Insert("orders", "o1", "c1", "80"), Insert("orders", "o2", "c2", "90"), Insert("orders", "o3", "c1", "10"),
        });
        Check(shrink.Query() == "eu=80:1;us=90:1", "two regions before the deletes");
        shrink.ApplyBatch(new[] { Delete("orders", "o1", "c1", "80") });
        Check(shrink.Query() == "us=90:1", "a region disappears when its only matching row goes");
        Check(shrink.Query() == OracleView(shrink, out oracleOps), "retraction matches the oracle");
        shrink.ApplyBatch(new[] { Delete("orders", "o3", "c1", "10"), Delete("orders", "o2", "c2", "90") });
        Check(shrink.Query() == "" && shrink.View.Count == 0, "deleting everything empties the view");
        Check(shrink.Query() == OracleView(shrink, out oracleOps), "empty state matches the oracle");

        // ---- 4. the third join term matters ----------------------------------------------------
        var both = new[]
        {
            Insert("orders", "n1", "c7", "300"),
            Insert("customers", "c7", "latam"),
            Insert("orders", "n2", "c8", "400"),
            Insert("customers", "c8", "latam"),
        };
        var correct = new ViewPipeline();
        correct.ApplyBatch(both);
        Check(correct.Query() == OracleView(correct, out oracleOps), "both-sided batch with three terms matches the oracle");
        Check(correct.View["latam"] == (700L, 2L), "both new rows land in the new region");

        var twoTerm = new ViewPipeline { ThreeTermJoinDelta = false };
        twoTerm.ApplyBatch(both);
        Check(twoTerm.Query() != OracleView(twoTerm, out oracleOps),
            "dropping ΔF ⋈ ΔC diverges when one batch changes both sides");
        Check(twoTerm.View.Count == 0 || twoTerm.View["latam"].Count != 2, "the two-term view is visibly wrong");

        // ---- 5. transactional batches ----------------------------------------------------------
        var guarded = new ViewPipeline();
        guarded.ApplyBatch(new[] { Insert("customers", "c1", "eu") });
        guarded.ApplyBatch(new[] { Insert("orders", "o1", "c1", "60") });
        string before = guarded.Query();
        long rejectedBefore = guarded.RejectedBatches;
        Check(Throws(() => guarded.ApplyBatch(new[]
            {
                Insert("orders", "o2", "c1", "70"),          // valid
                Delete("orders", "o9", "c1", "70"),          // would go negative
            })), "a batch with one bad change is rejected");
        Check(guarded.Query() == before, "a rejected batch leaves the view untouched");
        Check(guarded.Orders.Multiplicity("o1|c1|60") == 1 && guarded.Orders.Multiplicity("o2|c1|70") == 0,
            "a rejected batch leaves the tables untouched");
        Check(guarded.RejectedBatches == rejectedBefore + 1, "the rejection is counted once");
        Check(Throws(() => guarded.ApplyBatch(new[] { Insert("warehouse", "w1") })), "an unknown table is rejected");
        Check(Throws(() => guarded.ApplyBatch(new[] { Insert("orders", "o3", "c1", "not-a-number") })),
            "a malformed amount is rejected");
        Check(Throws(() => guarded.ApplyBatch(new[] { Insert("orders", "o4", "c1") })), "an arity mismatch is rejected");
        Check(Throws(() => guarded.ApplyBatch(new[] { Insert("orders", "o5", "c|1", "70") })),
            "a tuple containing the separator is rejected");
        guarded.ApplyBatch(new[] { Insert("orders", "o2", "c1", "70") });
        Check(guarded.Query() == OracleView(guarded, out oracleOps), "the pipeline recovers after rejections");

        // ---- 6. bounded memory and flush accounting -------------------------------------------
        var bounded = new ViewPipeline();
        var workload = BuildWorkload(60, 4);
        int maxPending = 0;
        long oracleWork = 0;
        foreach (Change[] batch in workload)
        {
            bounded.ApplyBatch(batch);
            maxPending = Math.Max(maxPending, bounded.PendingCount);
            Check(bounded.PendingCount <= ViewPipeline.PendingCapacity, "pending log stays within capacity");
            Check(bounded.Query() == OracleView(bounded, out oracleOps), "incremental view equals the oracle after every batch");
        }
        Check(maxPending <= ViewPipeline.PendingCapacity && maxPending > 0, "capacity was approached but never exceeded");
        Check(bounded.Flushes.Count >= workload.Count, "every batch that overflows or is queried flushes");
        Check(bounded.PendingCount == 0, "a query leaves nothing pending");

        long expectedOps = 0;
        foreach (FlushStats stats in bounded.Flushes)
            expectedOps += stats.OrderDelta + (long)stats.FilteredDelta * stats.CustomerRows +
                (long)stats.FilteredRows * stats.CustomerDelta + (long)stats.FilteredDelta * stats.CustomerDelta +
                stats.JoinedGroups;
        Check(expectedOps == bounded.IncrementalOps, "flush cost equals the summed term products: " +
            bounded.IncrementalOps + " vs " + expectedOps);

        long queries = workload.Count;
        long fullWork = bounded.Orders.TotalMultiplicity() * bounded.Customers.TotalMultiplicity() * queries;
        oracleWork = fullWork;
        Check(bounded.IncrementalOps < fullWork / 5, "incremental work is far below repeated recomputation: " +
            bounded.IncrementalOps + " vs " + fullWork);

        // A flush must be a no-op on an empty log, and compaction by querying must not change the view.
        string settled = bounded.Query();
        int flushesBefore = bounded.Flushes.Count;
        bounded.Flush();
        Check(bounded.Flushes.Count == flushesBefore, "an empty log does not flush");
        Check(bounded.Query() == settled, "a repeated query is stable");

        // ---- 7. determinism --------------------------------------------------------------------
        var repeat = new ViewPipeline();
        foreach (Change[] batch in workload) repeat.ApplyBatch(batch);
        Check(repeat.Query() == settled, "the same workload yields the same view");

        // ---- 8. degenerate shapes --------------------------------------------------------------
        var customerOnly = new ViewPipeline();
        customerOnly.ApplyBatch(new[] { Insert("customers", "c1", "eu") });
        Check(customerOnly.Query() == "" && customerOnly.Orders.LiveCount() == 0, "customers alone produce no view rows");
        customerOnly.ApplyBatch(new[] { Delete("customers", "c1", "eu") });
        Check(customerOnly.Query() == "" && customerOnly.Customers.LiveCount() == 0, "removing the customer is fine");
        var filterOnly = new ViewPipeline();
        filterOnly.ApplyBatch(new[] { Insert("orders", "o1", "c1", "50") });
        Check(filterOnly.Query() == "" && filterOnly.Filtered.LiveCount() == 0, "a row exactly at the filter bound is dropped");
        filterOnly.ApplyBatch(new[] { Insert("orders", "o2", "c1", "51") });
        filterOnly.Flush();
        Check(filterOnly.Filtered.LiveCount() == 1, "the next integer above the bound is kept once the log is flushed");

        // A new customer makes an already-stored order visible without touching the order.
        var joinLate = new ViewPipeline();
        joinLate.ApplyBatch(new[] { Insert("orders", "o1", "c4", "120") });
        Check(joinLate.Query() == "", "orders without a customer contribute nothing");
        joinLate.ApplyBatch(new[] { Insert("customers", "c4", "nordics") });
        Check(joinLate.Query() == "nordics=120:1", "the second join term admits the stored order");
        joinLate.ApplyBatch(new[] { Delete("customers", "c4", "nordics") });
        Check(joinLate.Query() == "" && joinLate.Orders.LiveCount() == 1, "removing the customer hides the order again");

        // ---- 9. wider scale: 500 orders, 40 customers -----------------------------------------
        var wide = new ViewPipeline();
        for (int c = 0; c < 40; c++) wide.ApplyBatch(new[] { Insert("customers", "c" + c, "r" + (c % 7)) });
        for (int b = 0; b < 50; b++)
        {
            var batch = new List<Change>();
            for (int k = 0; k < 10; k++)
            {
                long orderId = b * 10 + k;
                batch.Add(Insert("orders", "w" + orderId, "c" + (orderId % 40),
                    (30 + (long)(NextRandom() % 200)).ToString(CultureInfo.InvariantCulture)));
            }
            wide.ApplyBatch(batch.ToArray());
            Check(wide.PendingCount <= ViewPipeline.PendingCapacity, "wide workload keeps the log bounded");
        }
        Check(wide.Query() == OracleView(wide, out oracleOps), "wide workload matches the oracle");
        Check(wide.Orders.TotalMultiplicity() == 500, "500 orders were inserted");
        long wideFullWork = wide.Orders.TotalMultiplicity() * wide.Customers.TotalMultiplicity();
        Check(wide.IncrementalOps < wideFullWork, "wide incremental work is below one full recomputation: " +
            wide.IncrementalOps + " vs " + wideFullWork);
        Check(wide.View.Count == 7, "one group per region");

        Console.WriteLine("view: " + wide.View.Count + " groups over " + wide.Orders.TotalMultiplicity() + " orders and " +
            wide.Customers.TotalMultiplicity() + " customers; flushes=" + wide.Flushes.Count +
            ", propagated deltas=" + wide.PropagatedDeltas + ", incremental ops=" + wide.IncrementalOps +
            " vs one full recompute " + wideFullWork);
        Console.WriteLine("bounded workload: " + workload.Count + " batches, maxPending=" + maxPending +
            " of capacity " + ViewPipeline.PendingCapacity + ", incrementals=" + bounded.IncrementalOps +
            " vs oracle work " + fullWork + ", rejected batches=" + guarded.RejectedBatches);
        Console.WriteLine("oracle ops on the final wide query=" + oracleOps + ", two-term variant diverges=" +
            (twoTerm.Query() != OracleView(twoTerm, out long _)));
        Console.WriteLine("PASS");
    }
}
