// Task 644 - Implement query execution engine operator pipelining simulation. (Tier: Expert, CTCI Ch 14 - Databases)
//
// A Volcano/iterator-model executor: every operator implements Open / Next / Close over int[] rows,
// parents pull one row at a time from their children, and the whole tree is instrumented with
// Open / Close / Next / RowsProduced counters so the *pipelining* behaviour can be asserted instead
// of gestured at. The operators are: SeqScan (re-openable, so it survives a re-scan), Filter,
// Project, Limit (early termination), Sort and HashAggregate (blocking: they drain their child on
// Open) and NestedLoopJoin (which re-opens its right child once per surviving left row).
//
// Simplified: in-memory tables, a single physical join method, no buffer pool, no vector batches
// (task 648 covers batch-at-a-time execution), no parallelism and no cost-based operator choices.
//
// Verified: every query result equals a hand-written naive reference over the raw table; a Limit on
// top of a scan really stops early (the scan is asked for fewer rows than it holds); a blocking Sort
// consumes its whole input no matter what the parent wants; NestedLoopJoin re-scans its right input
// once per left row, and a Limit above the join cuts both the re-scans and the left-side pulls;
// every opened operator is closed; and the row counters obey the tree conservation invariant.
using System;
using System.Collections.Generic;
using System.Linq;

internal abstract class Operator
{
    public readonly string Name;
    public int Opens;
    public int Closes;
    public int NextCalls;
    public int RowsProduced;

    private bool _open;

    protected Operator(string name) => Name = name;

    public bool IsOpen => _open;

    public void Open()
    {
        Opens++;
        _open = true;
        OpenInternal();
    }

    public int[]? Next()
    {
        NextCalls++;
        if (!_open) throw new Exception(Name + ": Next on a closed operator");
        int[]? row = NextInternal();
        if (row != null) RowsProduced++;
        return row;
    }

    public void Close()
    {
        if (!_open) return;
        _open = false;
        Closes++;
        CloseInternal();
    }

    public int Drain()
    {
        int rows = 0;
        while (Next() != null) rows++;
        return rows;
    }

    protected abstract void OpenInternal();
    protected abstract int[]? NextInternal();
    protected abstract void CloseInternal();

    public abstract IEnumerable<Operator> Children { get; }
}

internal sealed class SeqScan : Operator
{
    private readonly List<int[]> _rows;
    private int _position;

    public SeqScan(string name, List<int[]> rows) : base(name) => _rows = rows;

    public int TableRows => _rows.Count;

    public override IEnumerable<Operator> Children => Enumerable.Empty<Operator>();

    protected override void OpenInternal() => _position = 0;

    protected override int[]? NextInternal() => _position < _rows.Count ? _rows[_position++] : null;

    protected override void CloseInternal() { }
}

internal sealed class Filter : Operator
{
    private readonly Operator _child;
    private readonly Func<int[], bool> _predicate;

    public Filter(Operator child, Func<int[], bool> predicate) : base("filter")
    {
        _child = child;
        _predicate = predicate;
    }

    public override IEnumerable<Operator> Children => new[] { _child };

    protected override void OpenInternal() => _child.Open();

    protected override int[]? NextInternal()
    {
        while (true)
        {
            int[]? row = _child.Next();
            if (row == null) return null;
            if (_predicate(row)) return row;
        }
    }

    protected override void CloseInternal() => _child.Close();
}

internal sealed class Project : Operator
{
    private readonly Operator _child;
    private readonly int _column;

    public Project(Operator child, int column) : base("project")
    {
        _child = child;
        _column = column;
    }

    public override IEnumerable<Operator> Children => new[] { _child };

    protected override void OpenInternal() => _child.Open();

    protected override int[]? NextInternal()
    {
        int[]? row = _child.Next();
        return row == null ? null : new[] { row[_column] };
    }

    protected override void CloseInternal() => _child.Close();
}

internal sealed class Limit : Operator
{
    private readonly Operator _child;
    private readonly int _limit;
    private int _emitted;

    public Limit(Operator child, int limit) : base("limit")
    {
        _child = child;
        _limit = limit;
    }

    public override IEnumerable<Operator> Children => new[] { _child };

    protected override void OpenInternal()
    {
        _emitted = 0;
        _child.Open();
    }

    protected override int[]? NextInternal()
    {
        if (_emitted >= _limit) return null;
        int[]? row = _child.Next();
        if (row == null) return null;
        _emitted++;
        return row;
    }

    protected override void CloseInternal() => _child.Close();
}

internal sealed class Sort : Operator
{
    private readonly Operator _child;
    private readonly int _column;
    private List<int[]> _buffer = new List<int[]>();
    private int _position;

    public Sort(Operator child, int column) : base("sort")
    {
        _child = child;
        _column = column;
    }

    public override IEnumerable<Operator> Children => new[] { _child };

    // Blocking: the child is drained completely before the first row can be served.
    protected override void OpenInternal()
    {
        _child.Open();
        _buffer = new List<int[]>();
        while (true)
        {
            int[]? row = _child.Next();
            if (row == null) break;
            _buffer.Add(row);
        }
        _buffer = _buffer.OrderBy(r => r[_column]).ThenBy(r => r[0]).ToList();
        _position = 0;
    }

    protected override int[]? NextInternal() => _position < _buffer.Count ? _buffer[_position++] : null;

    protected override void CloseInternal() => _child.Close();
}

internal sealed class HashAggregate : Operator
{
    private readonly Operator _child;
    private readonly int _groupColumn;
    private readonly int _valueColumn;
    private List<int[]> _groups = new List<int[]>();
    private int _position;

    public HashAggregate(Operator child, int groupColumn, int valueColumn) : base("hash-aggregate")
    {
        _child = child;
        _groupColumn = groupColumn;
        _valueColumn = valueColumn;
    }

    public override IEnumerable<Operator> Children => new[] { _child };

    protected override void OpenInternal()
    {
        _child.Open();
        var counts = new SortedDictionary<int, int[]>();
        while (true)
        {
            int[]? row = _child.Next();
            if (row == null) break;
            int group = row[_groupColumn];
            if (!counts.TryGetValue(group, out int[]? state))
            {
                state = new int[3];
                state[0] = group;
                counts[group] = state;
            }
            state[1]++;
            state[2] += row[_valueColumn];
        }
        _groups = counts.Values.ToList();
        _position = 0;
    }

    protected override int[]? NextInternal() => _position < _groups.Count ? _groups[_position++] : null;

    protected override void CloseInternal() => _child.Close();
}

internal sealed class NestedLoopJoin : Operator
{
    private readonly Operator _left;
    private readonly Operator _right;
    private readonly int _leftColumn;
    private readonly int _rightColumn;
    private int[]? _leftRow;
    private bool _rightDrained = true;

    public NestedLoopJoin(Operator left, Operator right, int leftColumn, int rightColumn) : base("nested-loop-join")
    {
        _left = left;
        _right = right;
        _leftColumn = leftColumn;
        _rightColumn = rightColumn;
    }

    public Operator RightInput => _right;

    public override IEnumerable<Operator> Children => new[] { _left, _right };

    protected override void OpenInternal()
    {
        _leftRow = null;
        _rightDrained = true;
        _left.Open();
    }

    protected override int[]? NextInternal()
    {
        while (true)
        {
            if (_rightDrained)
            {
                if (_right.IsOpen) _right.Close();
                _leftRow = _left.Next();
                if (_leftRow == null) return null;
                _right.Open();
                _rightDrained = false;
            }
            int[]? rightRow = _right.Next();
            if (rightRow == null)
            {
                _rightDrained = true;
                continue;
            }
            if (_leftRow![_leftColumn] == rightRow[_rightColumn])
            {
                return new[] { _leftRow[0], rightRow[0] };
            }
        }
    }

    protected override void CloseInternal()
    {
        _left.Close();
        if (_right.IsOpen) _right.Close();
    }
}

internal static class OperatorTree
{
    public static List<Operator> Flatten(Operator root)
    {
        var result = new List<Operator>();
        Walk(root, result);
        return result;
    }

    private static void Walk(Operator node, List<Operator> result)
    {
        result.Add(node);
        foreach (Operator child in node.Children) Walk(child, result);
    }

    public static List<int[]> Run(Operator root)
    {
        var rows = new List<int[]>();
        root.Open();
        while (true)
        {
            int[]? row = root.Next();
            if (row == null) break;
            rows.Add(row);
        }
        root.Close();
        return rows;
    }

    public static string Canonical(IEnumerable<int[]> rows)
        => string.Join(";", rows.Select(r => string.Join(",", r)));
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    // 500 rows: id, group (7 values), value (0..99).
    private static List<int[]> Table()
    {
        var rows = new List<int[]>();
        for (int i = 0; i < 500; i++) rows.Add(new[] { i, i % 7, (i * 13) % 100 });
        return rows;
    }

    private static void FilterProjectLimitResults()
    {
        List<int[]> table = Table();
        var scan = new SeqScan("scan", table);
        var pipeline = new Limit(new Project(new Filter(scan, r => r[2] >= 90), 2), 5);
        List<int[]> rows = OperatorTree.Run(pipeline);

        List<int[]> expected = table.Where(r => r[2] >= 90).Select(r => new[] { r[2] }).Take(5).ToList();
        Check(OperatorTree.Canonical(rows) == OperatorTree.Canonical(expected), "filter/project/limit result matches the reference");
        Check(rows.Count == 5, "the limit emits exactly five rows");

        int scanned = 0;
        int matched = 0;
        while (matched < 5 && scanned < table.Count)
        {
            if (table[scanned][2] >= 90) matched++;
            scanned++;
        }
        Check(scan.RowsProduced == scanned, "the scan produced only the rows the pipeline demanded, got " + scan.RowsProduced + " of " + scanned);
        Check(scan.NextCalls == scanned, "the scan was asked for exactly " + scanned + " rows, not the whole table");
        Check(scan.NextCalls < table.Count, "pipelining with a limit avoids a full table scan");
        Check(OperatorTree.Flatten(pipeline).All(op => op.Closes == op.Opens && !op.IsOpen), "every opened operator was closed");
        Console.WriteLine("pipelining: limit 5 touched " + scan.NextCalls + " of " + table.Count + " rows");
    }

    private static void BlockingSortConsumesEverything()
    {
        List<int[]> table = Table();
        var scan = new SeqScan("scan", table);
        var sort = new Sort(scan, 2);
        var pipeline = new Limit(sort, 3);
        List<int[]> rows = OperatorTree.Run(pipeline);

        List<int[]> expected = table.OrderBy(r => r[2]).ThenBy(r => r[0]).Take(3).ToList();
        Check(OperatorTree.Canonical(rows) == OperatorTree.Canonical(expected), "sort + limit matches the reference");
        Check(scan.NextCalls == table.Count + 1, "the blocking sort drained the scan completely (" + scan.NextCalls + " calls)");
        Check(sort.RowsProduced == 3, "the sort only hands three rows to the limit");
        Check(sort.NextCalls == 3, "the limit never asked the sort for a fourth row");

        var scan2 = new SeqScan("scan2", table);
        var openSort = new Sort(scan2, 2);
        openSort.Open();
        Check(scan2.NextCalls == table.Count + 1, "the drain happens during Open, before the first Next");
        openSort.Close();
    }

    private static void Aggregation()
    {
        List<int[]> table = Table();
        var scan = new SeqScan("scan", table);
        var aggregate = new HashAggregate(scan, 1, 2);
        List<int[]> rows = OperatorTree.Run(aggregate);

        var reference = new SortedDictionary<int, int[]>();
        foreach (int[] row in table)
        {
            if (!reference.TryGetValue(row[1], out int[]? state))
            {
                state = new int[3];
                state[0] = row[1];
                reference[row[1]] = state;
            }
            state[1]++;
            state[2] += row[2];
        }
        Check(rows.Count == 7, "one output row per distinct group, got " + rows.Count);
        Check(OperatorTree.Canonical(rows) == OperatorTree.Canonical(reference.Values), "grouped counts and sums match the reference");
        Check(rows.All(r => r[1] == table.Count(r2 => r2[1] == r[0])), "each group's count equals a recount");
        Check(scan.NextCalls == table.Count + 1, "the aggregate drained its input");

        var empty = new HashAggregate(new SeqScan("empty", new List<int[]>()), 1, 2);
        Check(OperatorTree.Run(empty).Count == 0, "aggregating an empty table yields no groups");
    }

    private static void JoinRescansAndLimitPushdown()
    {
        var leftRows = new List<int[]>();
        for (int i = 0; i < 200; i++) leftRows.Add(new[] { i, i % 20 });
        var rightRows = new List<int[]>();
        for (int i = 0; i < 50; i++) rightRows.Add(new[] { i, i % 10 });

        var leftScan = new SeqScan("left", leftRows);
        var rightScan = new SeqScan("right", rightRows);
        var join = new NestedLoopJoin(leftScan, rightScan, 1, 1);

        int expected = 0;
        for (int key = 0; key < 20; key++)
        {
            int left = leftRows.Count(r => r[1] == key);
            int right = rightRows.Count(r => r[1] == key);
            expected += left * right;
        }

        List<int[]> all = OperatorTree.Run(join);
        Check(all.Count == expected, "the nested loop join returns " + expected + " rows, got " + all.Count);
        Check(rightScan.Opens == 200, "the right input was re-scanned once per left row, got " + rightScan.Opens);
        Check(rightScan.Opens == rightScan.Closes, "every re-scan was closed again");
        Check(leftScan.NextCalls == 201, "the left input was pulled to EOF exactly once");

        var leftScan2 = new SeqScan("left2", leftRows);
        var rightScan2 = new SeqScan("right2", rightRows);
        var join2 = new NestedLoopJoin(leftScan2, rightScan2, 1, 1);
        List<int[]> limited = OperatorTree.Run(new Limit(join2, 10));
        Check(limited.Count == 10, "the limit above the join yields ten rows");
        Check(rightScan2.Opens == 2, "the limit stopped the join after two left rows, so two re-scans, got " + rightScan2.Opens);
        Check(leftScan2.NextCalls == 2, "the limit also cut the number of left-side pulls, got " + leftScan2.NextCalls);
        Console.WriteLine("join: " + expected + " rows with " + rightScan.Opens + " re-scans; a limit of 10 needs only " + rightScan2.Opens);
    }

    private static void ConservationInvariant()
    {
        List<int[]> table = Table();
        var scan = new SeqScan("scan", table);
        var join = new NestedLoopJoin(new Filter(scan, r => r[1] == 3), new SeqScan("right", Table()), 1, 1);
        var aggregate = new HashAggregate(new Project(new Filter(join, r => r[0] >= 0), 0), 0, 0);
        var root = new Limit(aggregate, 4);
        OperatorTree.Run(root);

        List<Operator> operators = OperatorTree.Flatten(root);
        foreach (Operator op in operators)
        {
            Check(op.Closes == op.Opens, op.Name + ": closes match opens");
            Check(op.NextCalls >= op.RowsProduced, op.Name + ": more Next calls than rows produced");
            Check(!op.IsOpen, op.Name + ": closed after the query finished");
        }
        foreach (Operator parent in operators)
        {
            List<Operator> children = parent.Children.ToList();
            if (parent is NestedLoopJoin)
            {
                // A join emits one row per matching pair, so it is bounded by the cross product of its
                // inputs rather than by either input alone.
                Check(parent.RowsProduced <= children[0].RowsProduced * children[1].RowsProduced,
                    "nested loop join output is bounded by left * right");
            }
            else
            {
                foreach (Operator child in children)
                {
                    Check(parent.RowsProduced <= child.RowsProduced, parent.Name + " produced no more rows than its child");
                }
            }
        }
        Check(operators.Count == 8, "the tree has eight operators, got " + operators.Count);
    }

    private static void Boundaries()
    {
        var empty = new List<int[]>();
        Check(OperatorTree.Run(new SeqScan("scan", empty)).Count == 0, "an empty scan yields nothing");
        Check(OperatorTree.Run(new Filter(new SeqScan("scan", empty), r => true)).Count == 0, "filtering an empty input yields nothing");
        Check(OperatorTree.Run(new Sort(new SeqScan("scan", empty), 0)).Count == 0, "sorting an empty input yields nothing");
        Check(OperatorTree.Run(new Limit(new SeqScan("scan", empty), 5)).Count == 0, "a limit over an empty input yields nothing");
        Check(OperatorTree.Run(new NestedLoopJoin(new SeqScan("l", empty), new SeqScan("r", empty), 0, 0)).Count == 0,
            "joining two empty inputs yields nothing");

        List<int[]> table = Table();
        Check(OperatorTree.Run(new Filter(new SeqScan("scan", table), r => false)).Count == 0, "a predicate that rejects everything yields no rows");
        Check(OperatorTree.Run(new Limit(new SeqScan("scan", table), 0)).Count == 0, "a limit of zero yields no rows");

        var single = new List<int[]> { new[] { 1, 2, 3 } };
        Check(OperatorTree.Run(new Project(new SeqScan("scan", single), 1)).Count == 1, "a one-row table flows through the pipeline");
        Check(OperatorTree.Run(new Sort(new SeqScan("scan", single), 2)).Count == 1, "sorting one row is a no-op");
    }

    public static void Main()
    {
        FilterProjectLimitResults();
        BlockingSortConsumesEverything();
        Aggregation();
        JoinRescansAndLimitPushdown();
        ConservationInvariant();
        Boundaries();
        Console.WriteLine("PASS");
    }
}
