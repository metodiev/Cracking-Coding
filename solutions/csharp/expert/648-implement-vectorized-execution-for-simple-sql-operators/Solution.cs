// Task 648 - Implement vectorized execution for simple SQL operators. (Tier: Expert, CTCI Ch 14 - Databases)
//
// Batch-at-a-time ("vectorized") execution: rows travel through the operators as batches of column
// arrays instead of one tuple at a time, and every operator is instrumented so the amortisation claim
// can be checked rather than asserted - each operator counts *vector steps* (one pass over a batch)
// next to the row count, so a filter over 10 000 rows in batches of 512 needs 20 vector steps instead
// of 10 000 per-row steps.
//
// Two ideas beyond batching are implemented for real:
//   * a selection vector: the filter produces the matching row indices of a batch and then compacts the
//     batch from that selection, so downstream operators never touch rejected rows;
//   * masked short-circuit evaluation: for a conjunctive predicate the second conjunct is evaluated
//     only over the rows that survived the first one, which is counted to prove the saving.
// Operators: VectorScan, VectorFilter (selection vector + compaction), VectorProject (arithmetic
// expression over two columns), VectorAggregate (grouped count/sum via a hash table) and VectorSum.
//
// Simplified: int32 columns, no nulls / no selection-vector-aware join, no SIMD intrinsics and no
// dictionary-compressed batches (task 647 covers compression). The verified contract is that every
// vectorized result is byte-for-byte equal to a scalar row-at-a-time reference, including the tail
// batch, batch size 1, an empty input and a fully rejecting predicate.
using System;
using System.Collections.Generic;
using System.Linq;

internal sealed class Batch
{
    public readonly int Size;
    public readonly int[][] Columns;

    public Batch(int size, int[][] columns)
    {
        Size = size;
        Columns = columns;
    }

    public static Batch Empty => new Batch(0, Array.Empty<int[]>());
}

internal sealed class VectorScan
{
    private readonly List<int[]> _rows;
    private readonly int _batchSize;
    private int _position;

    public int VectorSteps;
    public int RowsEmitted;

    public VectorScan(List<int[]> rows, int batchSize)
    {
        _rows = rows;
        _batchSize = batchSize;
    }

    public Batch? NextBatch()
    {
        if (_position >= _rows.Count) return null;
        VectorSteps++;
        int size = Math.Min(_batchSize, _rows.Count - _position);
        var columns = new[] { new int[size], new int[size] };
        for (int i = 0; i < size; i++)
        {
            columns[0][i] = _rows[_position + i][0];
            columns[1][i] = _rows[_position + i][1];
        }
        _position += size;
        RowsEmitted += size;
        return new Batch(size, columns);
    }
}

internal sealed class VectorFilter
{
    private readonly int _groupColumn;
    private readonly int _groupValue;
    private readonly int _valueColumn;
    private readonly int _threshold;

    public int VectorSteps;
    public int RowsIn;
    public int RowsOut;
    public int FirstConjunctEvaluations;
    public int SecondConjunctEvaluations;

    public VectorFilter(int groupColumn, int groupValue, int valueColumn, int threshold)
    {
        _groupColumn = groupColumn;
        _groupValue = groupValue;
        _valueColumn = valueColumn;
        _threshold = threshold;
    }

    // Vector pass one: the cheap conjunct fills the selection vector. Vector pass two: the expensive
    // conjunct runs only over the selected rows.
    public int[] Screen(Batch batch, out int selected)
    {
        VectorSteps++;
        RowsIn += batch.Size;
        var selection = new int[batch.Size];
        selected = 0;
        int[] group = batch.Columns[_groupColumn];
        for (int i = 0; i < batch.Size; i++)
        {
            FirstConjunctEvaluations++;
            if (group[i] == _groupValue) selection[selected++] = i;
        }
        int kept = 0;
        int[] value = batch.Columns[_valueColumn];
        for (int i = 0; i < selected; i++)
        {
            SecondConjunctEvaluations++;
            if (value[selection[i]] > _threshold) selection[kept++] = selection[i];
        }
        selected = kept;
        RowsOut += kept;
        return selection;
    }

    public Batch? Apply(Batch batch)
    {
        int[] selection = Screen(batch, out int selected);
        if (selected == 0) return Batch.Empty;
        var columns = new int[batch.Columns.Length][];
        for (int c = 0; c < batch.Columns.Length; c++)
        {
            var column = new int[selected];
            for (int i = 0; i < selected; i++) column[i] = batch.Columns[c][selection[i]];
            columns[c] = column;
        }
        return new Batch(selected, columns);
    }
}

internal sealed class VectorProject
{
    public int VectorSteps;
    public int RowsEmitted;

    // column0 + 2 * column1 applied to a whole batch with one pass per column.
    public Batch Apply(Batch batch)
    {
        VectorSteps++;
        if (batch.Size == 0) return Batch.Empty;
        var result = new int[batch.Size];
        int[] a = batch.Columns[0];
        int[] b = batch.Columns[1];
        for (int i = 0; i < batch.Size; i++) result[i] = a[i] + 2 * b[i];
        RowsEmitted += batch.Size;
        return new Batch(batch.Size, new[] { result });
    }
}

internal sealed class VectorAggregate
{
    private readonly int _groupColumn;
    private readonly int _valueColumn;
    private readonly SortedDictionary<int, int[]> _groups = new SortedDictionary<int, int[]>();

    public int VectorSteps;

    public VectorAggregate(int groupColumn, int valueColumn)
    {
        _groupColumn = groupColumn;
        _valueColumn = valueColumn;
    }

    public void Accumulate(Batch batch)
    {
        VectorSteps++;
        if (batch.Size == 0) return;
        int[] group = batch.Columns[_groupColumn];
        int[] value = batch.Columns[_valueColumn];
        for (int i = 0; i < batch.Size; i++)
        {
            if (!_groups.TryGetValue(group[i], out int[]? state))
            {
                state = new int[3];
                state[0] = group[i];
                _groups[group[i]] = state;
            }
            state[1]++;
            state[2] += value[i];
        }
    }

    public List<int[]> Result() => _groups.Values.ToList();
}

internal sealed class VectorSum
{
    private readonly int _column;

    public int VectorSteps;
    public long Total;

    public VectorSum(int column) => _column = column;

    public void Accumulate(Batch batch)
    {
        VectorSteps++;
        if (batch.Size == 0) return;
        int[] values = batch.Columns[_column];
        long running = Total;
        for (int i = 0; i < batch.Size; i++) running += values[i];
        Total = running;
    }
}

public static class Solution
{
    private static ulong _seed = 0x9E3779B97F4A7C15UL;

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

    private static List<int[]> Table(int rows)
    {
        var table = new List<int[]>();
        for (int i = 0; i < rows; i++) table.Add(new[] { i % 17, NextInt(100) });
        return table;
    }

    // Row-at-a-time reference for the whole pipeline.
    private static string ScalarPipeline(List<int[]> table, int group, int threshold)
    {
        var groups = new SortedDictionary<int, int[]>();
        foreach (int[] row in table)
        {
            if (row[0] != group || row[1] <= threshold) continue;
            if (!groups.TryGetValue(row[0], out int[]? state))
            {
                state = new int[3];
                state[0] = row[0];
                groups[row[0]] = state;
            }
            state[1]++;
            state[2] += row[1];
        }
        return string.Join(";", groups.Values.Select(g => g[0] + ":" + g[1] + ":" + g[2]));
    }

    private static string VectorPipeline(List<int[]> table, int batchSize, int group, int threshold, out VectorFilter filter, out VectorScan scan)
    {
        scan = new VectorScan(table, batchSize);
        filter = new VectorFilter(0, group, 1, threshold);
        var aggregate = new VectorAggregate(0, 1);
        Batch? batch;
        while ((batch = scan.NextBatch()) != null)
        {
            Batch? selected = filter.Apply(batch);
            if (selected != null && selected.Size > 0) aggregate.Accumulate(selected);
        }
        return string.Join(";", aggregate.Result().Select(g => g[0] + ":" + g[1] + ":" + g[2]));
    }

    private static void PipelineMatchesScalar()
    {
        List<int[]> table = Table(10000);
        for (int group = 0; group < 17; group += 5)
        {
            for (int threshold = 0; threshold <= 100; threshold += 50)
            {
                string expected = ScalarPipeline(table, group, threshold);
                string actual = VectorPipeline(table, 512, group, threshold, out VectorFilter filter, out VectorScan scan);
                Check(actual == expected, "vectorized pipeline equals the scalar reference (group " + group + ", threshold " + threshold + ")");
                Check(scan.RowsEmitted == table.Count, "the scan emitted every row");
                Check(filter.RowsIn == table.Count, "the filter saw every row");
                Check(filter.RowsOut == table.Count(row => row[0] == group && row[1] > threshold), "the filter kept exactly the matching rows");
            }
        }
        Console.WriteLine("pipeline: vectorized group-by equals the row-at-a-time reference for 8 parameter combinations");
    }

    private static void BatchArithmeticAndTail()
    {
        var scan = new VectorScan(Table(13), 4);
        var sizes = new List<int>();
        Batch? batch;
        while ((batch = scan.NextBatch()) != null) sizes.Add(batch.Size);
        Check(sizes.SequenceEqual(new[] { 4, 4, 4, 1 }), "13 rows in batches of 4 give 4,4,4,1, got " + string.Join(",", sizes));
        Check(scan.VectorSteps == 4, "four vector steps for four batches, got " + scan.VectorSteps);
        Check(sizes.Sum() == 13, "the batches cover every row exactly once");

        var big = new VectorScan(Table(5), 64);
        Batch? only = big.NextBatch();
        Check(only != null && only.Size == 5 && big.NextBatch() == null, "a batch size larger than the table yields one short batch");

        var empty = new VectorScan(new List<int[]>(), 8);
        Check(empty.NextBatch() == null && empty.VectorSteps == 0, "an empty scan produces no batch at all");
    }

    private static void VectorisationAmortises()
    {
        List<int[]> table = Table(10000);
        const int batchSize = 512;

        string reference = VectorPipeline(table, batchSize, 3, 50, out VectorFilter filter, out VectorScan scan);
        Check(reference == ScalarPipeline(table, 3, 50), "the batched pipeline equals the scalar reference");
        int batches = scan.VectorSteps;
        Check(batches == (10000 + batchSize - 1) / batchSize, "batch count is ceil(rows / batchSize)");

        Check(scan.VectorSteps * batchSize >= table.Count, "every row is covered by a vector step");
        Check(filter.VectorSteps == batches, "the filter needs one vector step per batch, got " + filter.VectorSteps);
        Check(filter.VectorSteps < table.Count, "the filter uses far fewer steps than the " + table.Count + " rows");

        var rowWise = new VectorScan(table, 1);
        var rowFilter = new VectorFilter(0, 3, 1, 50);
        Batch? batch;
        while ((batch = rowWise.NextBatch()) != null)
        {
            Batch? selected = rowFilter.Apply(batch);
            if (selected != null && selected.Size > 0) { }
        }
        Check(rowWise.VectorSteps == table.Count, "batch size 1 degenerates to one step per row");
        Check(rowFilter.VectorSteps == table.Count, "the degenerate filter also does one step per row");
        Check(rowFilter.RowsOut == filter.RowsOut, "batch size does not change the result size");
        Console.WriteLine("amortisation: " + batches + " vector steps vs " + table.Count + " rows (batch " + batchSize + ")");
    }

    private static void MaskedShortCircuit()
    {
        List<int[]> table = Table(5000);
        VectorPipeline(table, 128, 3, 50, out VectorFilter filter, out _);

        Check(filter.FirstConjunctEvaluations == table.Count, "the cheap conjunct runs once per row");
        Check(filter.SecondConjunctEvaluations <= filter.FirstConjunctEvaluations, "the expensive conjunct runs at most once per row");
        Check(filter.SecondConjunctEvaluations < table.Count / 4,
            "the expensive conjunct is skipped for the rows the cheap one rejects, evaluated "
            + filter.SecondConjunctEvaluations + " times");

        int survivors = table.Count(row => row[0] == 3);
        Check(filter.SecondConjunctEvaluations == survivors, "the expensive conjunct runs exactly once per surviving row");
        Check(filter.RowsOut <= survivors, "the selection only shrinks");
        Console.WriteLine("masked evaluation: cheap conjunct " + filter.FirstConjunctEvaluations
            + " rows, expensive conjunct " + filter.SecondConjunctEvaluations + " rows");
    }

    private static void SelectionVectorAndProjection()
    {
        List<int[]> table = Table(1000);
        var scan = new VectorScan(table, 64);
        var filter = new VectorFilter(0, 7, 1, 0);
        var project = new VectorProject();
        var sum = new VectorSum(0);

        Batch? batch;
        var projected = new List<int>();
        while ((batch = scan.NextBatch()) != null)
        {
            int[] selection = filter.Screen(batch, out int selected);
            for (int i = 1; i < selected; i++)
            {
                Check(selection[i] > selection[i - 1], "the selection vector is strictly increasing");
            }
            Batch? kept = filter.Apply(batch);
            if (kept == null || kept.Size == 0) continue;
            Batch mapped = project.Apply(kept);
            for (int i = 0; i < mapped.Size; i++) projected.Add(mapped.Columns[0][i]);
            sum.Accumulate(mapped);
        }

        var expected = table.Where(row => row[0] == 7 && row[1] > 0).Select(row => row[0] + 2 * row[1]).ToList();
        Check(projected.SequenceEqual(expected), "projection over batched selection vectors matches the scalar expression");
        Check(sum.Total == expected.Sum(), "the vectorized sum matches the scalar sum");
        Check(sum.VectorSteps == project.VectorSteps, "sum and projection see the same number of batches");
        Check(project.VectorSteps < table.Count, "projection needs fewer steps than the table has rows");
        Check(projected.Count <= table.Count, "no row is duplicated by compaction");
    }

    private static void Boundaries()
    {
        List<int[]> table = Table(100);
        string rejecting = VectorPipeline(table, 32, 99, 200, out VectorFilter filter, out VectorScan scan);
        Check(rejecting.Length == 0, "a rejecting predicate yields an empty aggregate");
        Check(filter.RowsOut == 0 && scan.RowsEmitted == table.Count, "the rows were scanned but none survived");

        string everything = VectorPipeline(table, 32, 0, -1, out VectorFilter keeper, out VectorScan sweep);
        Check(keeper.RowsOut == table.Count(row => row[0] == 0), "an accepting predicate keeps every qualifying row");
        Check(everything.Length > 0, "the aggregate has at least one group");

        var single = new VectorScan(new List<int[]> { new[] { 1, 2 } }, 4);
        Batch? one = single.NextBatch();
        Check(one != null && one.Columns[0][0] == 1 && one.Columns[1][0] == 2, "columns keep row alignment in the tail batch");

        var project = new VectorProject();
        Batch projected = project.Apply(new Batch(3, new[] { new[] { 1, 2, 3 }, new[] { 10, 20, 30 } }));
        Check(projected.Columns[0].SequenceEqual(new[] { 21, 42, 63 }), "projection computes a + 2b elementwise");
        Check(project.Apply(Batch.Empty).Size == 0, "projecting an empty batch yields an empty batch");
    }

    public static void Main()
    {
        PipelineMatchesScalar();
        BatchArithmeticAndTail();
        VectorisationAmortises();
        MaskedShortCircuit();
        SelectionVectorAndProjection();
        Boundaries();
        Console.WriteLine("PASS");
    }
}
