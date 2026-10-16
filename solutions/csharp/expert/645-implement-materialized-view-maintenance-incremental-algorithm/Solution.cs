// Task 645 - Implement materialized view maintenance incremental algorithm. (Tier: Expert, CTCI Ch 14 - Databases)
//
// Incremental view maintenance over multiset base tables. Two views are kept:
//   * `order_region`  = pi_{orders.cust, customers.region} (orders |x| customers on cust = id),
//   * `cust_totals`   = group by orders.cust: (count(*), sum(amount)).
// Whenever a base tuple changes, the view is NOT recomputed; instead the *delta* is computed and
// applied. For an insert of tuple t into orders the join delta is t |x| customers (the rows of the
// other relation that join with t), for a delete it is the same set subtracted; a customer insert
// contributes customers_new |x| orders the same way. The grouped view is maintained by adjusting
// (count, sum) of the touched group, and a group whose count drops to zero is removed.
//
// Applying a delta can never succeed if it is wrong: the multiset subtracts refuse to go negative, so
// a maintenance bug surfaces immediately as an exception rather than as a silently wrong answer.
//
// Simplified: integer columns, two base tables, one equi-join view and one grouped view, no
// self-joins, no outer joins, no nested aggregates and no WAL - and the "work" figures are tuple
// counts (touched tuples per delta versus tuples scanned by a full recompute), not real IO.
//
// Verified: after every operation of a 200-step randomised insert/delete schedule the incrementally
// maintained views equal a from-scratch recompute; deleting a tuple that is not there is rejected
// without changing anything; duplicate tuples each appear once in the delta; empty groups vanish and
// reappear; and the incremental work stays below the recompute work.
using System;
using System.Collections.Generic;
using System.Linq;

internal sealed class Table
{
    private readonly List<int[]> _rows = new List<int[]>();

    public IReadOnlyList<int[]> Rows => _rows;
    public int Count => _rows.Count;

    public void Insert(int[] row) => _rows.Add(row);

    public bool Remove(int[] row)
    {
        for (int i = 0; i < _rows.Count; i++)
        {
            if (Same(_rows[i], row))
            {
                _rows.RemoveAt(i);
                return true;
            }
        }
        return false;
    }

    public static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    public static string Key(int[] row) => string.Join(",", row);
}

internal sealed class RowMultiset
{
    private readonly SortedDictionary<string, int> _counts = new SortedDictionary<string, int>(StringComparer.Ordinal);

    public int Total { get; private set; }

    public void Insert(int[] row)
    {
        _counts.TryGetValue(Table.Key(row), out int seen);
        _counts[Table.Key(row)] = seen + 1;
        Total++;
    }

    // Subtracting a row that is not there means the delta was wrong: fail loudly.
    public void Remove(int[] row)
    {
        string key = Table.Key(row);
        if (!_counts.TryGetValue(key, out int seen) || seen <= 0)
        {
            throw new Exception("delta underflow: cannot remove (" + key + ") from the view");
        }
        if (seen == 1) _counts.Remove(key);
        else _counts[key] = seen - 1;
        Total--;
    }

    public string Canonical()
        => string.Join(";", _counts.Where(kv => kv.Value > 0).Select(kv => kv.Key + "x" + kv.Value));

    public int CountOf(int[] row) => _counts.TryGetValue(Table.Key(row), out int seen) ? seen : 0;
}

internal sealed class ViewState
{
    public readonly string JoinRows;
    public readonly string GroupRows;
    public readonly int JoinTotal;
    public readonly int GroupTotal;

    public ViewState(string joinRows, string groupRows, int joinTotal, int groupTotal)
    {
        JoinRows = joinRows;
        GroupRows = groupRows;
        JoinTotal = joinTotal;
        GroupTotal = groupTotal;
    }
}

internal sealed class Maintainer
{
    public readonly Table Orders = new Table();
    public readonly Table Customers = new Table();

    private readonly RowMultiset _joinView = new RowMultiset();
    private readonly SortedDictionary<int, int[]> _groupView = new SortedDictionary<int, int[]>();

    public long IncrementalWork;
    public long RecomputeWork;
    public long DeltaApplications;

    // orders(cust, amount), customers(id, region)
    private const int OrderCust = 0;
    private const int OrderAmount = 1;
    private const int CustomerId = 0;
    private const int CustomerRegion = 1;

    public void InsertOrder(int cust, int amount) => ApplyOrder(new[] { cust, amount }, true);

    public void DeleteOrder(int cust, int amount)
    {
        if (!ApplyOrder(new[] { cust, amount }, false)) throw new Exception("delete of a nonexistent order tuple");
    }

    public void InsertCustomer(int id, int region) => ApplyCustomer(new[] { id, region }, true);

    public void DeleteCustomer(int id, int region)
    {
        if (!ApplyCustomer(new[] { id, region }, false)) throw new Exception("delete of a nonexistent customer tuple");
    }

    private bool ApplyOrder(int[] tuple, bool insert)
    {
        if (insert)
        {
            Orders.Insert(tuple);
        }
        else if (!Orders.Remove(tuple))
        {
            return false;
        }

        DeltaApplications++;
        // Join delta: the new/changed tuple joined with the *current* rows of the other relation.
        int matches = 0;
        foreach (int[] customer in Customers.Rows)
        {
            if (customer[CustomerId] != tuple[OrderCust]) continue;
            matches++;
            int[] viewRow = { tuple[OrderCust], customer[CustomerRegion] };
            if (insert) _joinView.Insert(viewRow);
            else _joinView.Remove(viewRow);
        }
        IncrementalWork += matches + 1;
        RecomputeWork += Orders.Count + Customers.Count;

        // Group delta: adjust (count, sum) of the single affected group.
        int group = tuple[OrderCust];
        if (!_groupView.TryGetValue(group, out int[]? state))
        {
            state = new int[3];
            state[0] = group;
            _groupView[group] = state;
        }
        state[1] += insert ? 1 : -1;
        state[2] += insert ? tuple[OrderAmount] : -tuple[OrderAmount];
        if (state[1] == 0)
        {
            if (state[2] != 0) throw new Exception("group " + group + " has count 0 but a nonzero sum");
            _groupView.Remove(group);
        }
        else if (state[1] < 0)
        {
            throw new Exception("group " + group + " went negative");
        }
        return true;
    }

    private bool ApplyCustomer(int[] tuple, bool insert)
    {
        if (insert)
        {
            Customers.Insert(tuple);
        }
        else if (!Customers.Remove(tuple))
        {
            return false;
        }

        DeltaApplications++;
        int matches = 0;
        foreach (int[] order in Orders.Rows)
        {
            if (order[OrderCust] != tuple[CustomerId]) continue;
            matches++;
            int[] viewRow = { order[OrderCust], tuple[CustomerRegion] };
            if (insert) _joinView.Insert(viewRow);
            else _joinView.Remove(viewRow);
        }
        IncrementalWork += matches + 1;
        RecomputeWork += Orders.Count + Customers.Count;
        return true;
    }

    public ViewState Snapshot() => new ViewState(_joinView.Canonical(), GroupCanonical(), _joinView.Total, GroupTotal());

    public ViewState FullRecompute()
    {
        var join = new RowMultiset();
        foreach (int[] order in Orders.Rows)
        {
            foreach (int[] customer in Customers.Rows)
            {
                if (order[OrderCust] == customer[CustomerId])
                {
                    join.Insert(new[] { order[OrderCust], customer[CustomerRegion] });
                }
            }
        }

        var groups = new SortedDictionary<int, int[]>();
        foreach (int[] order in Orders.Rows)
        {
            int group = order[OrderCust];
            if (!groups.TryGetValue(group, out int[]? state))
            {
                state = new int[3];
                state[0] = group;
                groups[group] = state;
            }
            state[1]++;
            state[2] += order[OrderAmount];
        }

        string groupText = string.Join(";", groups.Select(kv => kv.Key + ":" + kv.Value[1] + ":" + kv.Value[2]));
        int groupTotal = groups.Count;
        return new ViewState(join.Canonical(), groupText, join.Total, groupTotal);
    }

    private string GroupCanonical()
        => string.Join(";", _groupView.Select(kv => kv.Key + ":" + kv.Value[1] + ":" + kv.Value[2]));

    private int GroupTotal() => _groupView.Count;
}

public static class Solution
{
    private static ulong _seed = 0x853C49E6748FEA9BUL;

    private static ulong NextRand()
    {
        _seed += 0x9E3779B97F4A7C15UL;
        ulong z = _seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static int NextInt(int bound) => (int)(NextRand() % (ulong)bound);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckAgainstRecompute(Maintainer maintainer, string label)
    {
        ViewState maintained = maintainer.Snapshot();
        ViewState recomputed = maintainer.FullRecompute();
        Check(maintained.JoinRows == recomputed.JoinRows, label + ": join view matches the recompute (" + maintained.JoinRows + " vs " + recomputed.JoinRows + ")");
        Check(maintained.GroupRows == recomputed.GroupRows, label + ": grouped view matches the recompute (" + maintained.GroupRows + " vs " + recomputed.GroupRows + ")");
        Check(maintained.JoinTotal == recomputed.JoinTotal, label + ": join cardinality matches");
        Check(maintained.GroupTotal == recomputed.GroupTotal, label + ": group count matches");
    }

    private static void BasicDelta()
    {
        var maintainer = new Maintainer();
        CheckAgainstRecompute(maintainer, "empty tables");

        maintainer.InsertCustomer(1, 10);
        maintainer.InsertCustomer(2, 20);
        maintainer.InsertOrder(1, 5);
        CheckAgainstRecompute(maintainer, "after the first insert");
        Check(maintainer.Snapshot().JoinRows == "1,10x1", "the join delta produced exactly (cust 1, region 10)");

        // A customer inserted *after* the orders that reference it must contribute its rows too.
        maintainer.InsertOrder(3, 7);
        maintainer.InsertOrder(3, 8);
        Check(maintainer.Snapshot().JoinTotal == 1, "orders without a customer do not join");
        maintainer.InsertCustomer(3, 30);
        CheckAgainstRecompute(maintainer, "customer inserted after its orders");
        Check(maintainer.Snapshot().JoinTotal == 3, "the new customer pulled in both of its orders");
        Check(maintainer.Snapshot().GroupRows == "1:1:5;3:2:15", "grouped view: cust 1 count 1 sum 5, cust 3 count 2 sum 15");

        maintainer.DeleteOrder(3, 7);
        CheckAgainstRecompute(maintainer, "after deleting one order");
        Check(maintainer.Snapshot().GroupRows == "1:1:5;3:1:8", "the deleted order's group shrank to count 1");
        maintainer.DeleteCustomer(1, 10);
        CheckAgainstRecompute(maintainer, "after deleting a customer");
        Check(maintainer.Snapshot().JoinRows == "3,30x1", "only the surviving customer's joined row is left");
        Console.WriteLine("basic deltas: join and grouped views match the recompute after every step");
    }

    private static void DuplicatesAndEmptyGroups()
    {
        var maintainer = new Maintainer();
        maintainer.InsertCustomer(1, 10);
        maintainer.InsertOrder(1, 4);
        maintainer.InsertOrder(1, 4);
        maintainer.InsertOrder(1, 4);
        CheckAgainstRecompute(maintainer, "three identical orders");
        Check(maintainer.Snapshot().JoinRows == "1,10x3", "the multiset view keeps all three identical rows");
        Check(maintainer.Snapshot().GroupRows == "1:3:12", "the group sums three identical rows");

        maintainer.DeleteOrder(1, 4);
        Check(maintainer.Snapshot().JoinRows == "1,10x2", "deleting one occurrence leaves two");
        maintainer.DeleteOrder(1, 4);
        maintainer.DeleteOrder(1, 4);
        Check(maintainer.Snapshot().JoinRows == "", "deleting every occurrence empties the view");
        Check(maintainer.Snapshot().GroupRows == "", "the group disappears when its count reaches zero");
        CheckAgainstRecompute(maintainer, "after emptying the orders");

        maintainer.InsertOrder(1, 9);
        Check(maintainer.Snapshot().GroupRows == "1:1:9", "the group is recreated by a later insert");
        CheckAgainstRecompute(maintainer, "after recreating the group");

        bool threw = false;
        try { maintainer.DeleteOrder(1, 999); }
        catch (Exception) { threw = true; }
        Check(threw, "deleting a tuple that is not in the table is rejected");
        CheckAgainstRecompute(maintainer, "after the rejected delete");
    }

    private static void UpdateAsDeletePlusInsert()
    {
        var maintainer = new Maintainer();
        maintainer.InsertCustomer(1, 10);
        maintainer.InsertCustomer(1, 11);
        maintainer.InsertOrder(1, 5);
        maintainer.InsertOrder(1, 6);
        Check(maintainer.Snapshot().JoinTotal == 4, "one order joins both customers that share the id");

        maintainer.DeleteOrder(1, 5);
        maintainer.InsertOrder(1, 7);
        CheckAgainstRecompute(maintainer, "update as delete + insert");
        Check(maintainer.Snapshot().GroupRows == "1:2:13", "the group reflects the replacement value");

        maintainer.DeleteCustomer(1, 11);
        CheckAgainstRecompute(maintainer, "deleting one of two customers with the same id");
        Check(maintainer.Snapshot().JoinTotal == 2, "exactly two joined rows disappeared with that customer");
    }

    private static void RandomisedSchedule()
    {
        var maintainer = new Maintainer();
        for (int id = 0; id < 8; id++) maintainer.InsertCustomer(id, id % 3);
        CheckAgainstRecompute(maintainer, "seeded customers");

        int inserts = 0, deletes = 0;
        for (int step = 0; step < 200; step++)
        {
            int action = NextInt(4);
            if (action == 0 || action == 1 || maintainer.Orders.Count == 0)
            {
                maintainer.InsertOrder(NextInt(10), 1 + NextInt(5));
                inserts++;
            }
            else if (action == 2)
            {
                int[] victim = maintainer.Orders.Rows[NextInt(maintainer.Orders.Count)];
                maintainer.DeleteOrder(victim[0], victim[1]);
                deletes++;
            }
            else
            {
                if (NextInt(3) == 0)
                {
                    maintainer.InsertCustomer(NextInt(10), NextInt(4));
                }
                else if (maintainer.Customers.Count > 1)
                {
                    int[] victim = maintainer.Customers.Rows[NextInt(maintainer.Customers.Count)];
                    maintainer.DeleteCustomer(victim[0], victim[1]);
                    deletes++;
                }
            }
            CheckAgainstRecompute(maintainer, "random step " + step);
        }

        Check(inserts > 20 && deletes > 20, "the schedule really exercised inserts and deletes");
        Check(maintainer.IncrementalWork < maintainer.RecomputeWork,
            "delta maintenance touched fewer tuples than full recomputes would have (" + maintainer.IncrementalWork + " vs " + maintainer.RecomputeWork + ")");
        Console.WriteLine("random schedule: 200 steps, incremental tuples " + maintainer.IncrementalWork
            + " vs recompute tuples " + maintainer.RecomputeWork);
    }

    public static void Main()
    {
        BasicDelta();
        DuplicatesAndEmptyGroups();
        UpdateAsDeletePlusInsert();
        RandomisedSchedule();
        Console.WriteLine("PASS");
    }
}
