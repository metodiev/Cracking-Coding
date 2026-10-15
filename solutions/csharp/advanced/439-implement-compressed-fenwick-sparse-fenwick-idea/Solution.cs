// Task 439 - Implement compressed Fenwick / sparse Fenwick idea. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A Fenwick tree that is touched at only a few coordinates does not need a dense array: the sparse
// version keeps a map of just the cells an update really visits (i, i + lowbit(i), ...) and treats
// every untouched cell as 0 while walking a prefix query (i, i - lowbit(i), ...). Documented setup:
// SIZE = 2^20 = 1048576, coordinates 1..SIZE, updates propagate while i <= SIZE. A dense Fenwick of
// the same size is the oracle: LCG updates are applied to both and every prefix query must agree,
// with a brute-force sweep over the recorded updates as a third opinion. The touched-cell count is
// asserted to stay within 21 * (#updates) - at most log2(SIZE) = 20 cells are visited per update -
// and far below SIZE, which is the evidence that the representation really is compressed.
using System;
using System.Collections.Generic;

internal class SparseFenwick
{
    private readonly int size;
    private readonly Dictionary<int, long> tree = new Dictionary<int, long>();

    public SparseFenwick(int size)
    {
        this.size = size;
    }

    public void Add(int index, long delta)
    {
        int i = index;
        while (i <= size)
        {
            tree[i] = tree.TryGetValue(i, out long current) ? current + delta : delta;
            i += i & -i;
        }
    }

    public long Prefix(int index)
    {
        long total = 0;
        int i = index;
        while (i > 0)
        {
            if (tree.TryGetValue(i, out long value)) total += value;
            i -= i & -i;
        }
        return total;
    }

    public int Touched() => tree.Count;
}

internal class DenseFenwick
{
    private readonly int size;
    private readonly long[] tree;

    public DenseFenwick(int size)
    {
        this.size = size;
        tree = new long[size + 1];
    }

    public void Add(int index, long delta)
    {
        int i = index;
        while (i <= size)
        {
            tree[i] += delta;
            i += i & -i;
        }
    }

    public long Prefix(int index)
    {
        long total = 0;
        int i = index;
        while (i > 0)
        {
            total += tree[i];
            i -= i & -i;
        }
        return total;
    }
}

public static class Solution
{
    private const int Size = 1 << 20;

    public static void Main()
    {
        var sparse = new SparseFenwick(16);
        var dense = new DenseFenwick(16);
        sparse.Add(1, 5);
        sparse.Add(3, 2);
        sparse.Add(16, 4);
        dense.Add(1, 5);
        dense.Add(3, 2);
        dense.Add(16, 4);
        Check(sparse.Prefix(0) == 0, "empty prefix");
        Check(sparse.Prefix(1) == 5, "prefix 1");
        Check(sparse.Prefix(2) == 5, "prefix 2");
        Check(sparse.Prefix(3) == 7, "prefix 3");
        Check(sparse.Prefix(15) == 7, "prefix 15");
        Check(sparse.Prefix(16) == 11, "prefix 16");
        Check(sparse.Touched() == 6, "touched cells " + sparse.Touched());
        for (int index = 0; index <= 16; index++)
        {
            Check(sparse.Prefix(index) == dense.Prefix(index), "small agreement " + index);
        }

        sparse = new SparseFenwick(Size);
        var bigDense = new DenseFenwick(Size);
        Check(sparse.Prefix(0) == 0 && sparse.Prefix(Size) == 0, "large empty");
        Check(sparse.Touched() == 0, "large empty touched");

        var recorded = new List<(int Coord, long Delta)>();
        int seed = 4390;
        int queries = 0;
        int updates = 0;
        for (int step = 0; step < 800; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            seed = (seed * 75 + 74) % 65537;
            int coord = 1 + (seed * 1024 + (seed >> 4)) % (Size - 1);
            seed = (seed * 75 + 74) % 65537;
            if (step % 2 == 0)
            {
                long delta = seed % 21 - 10;
                sparse.Add(coord, delta);
                bigDense.Add(coord, delta);
                recorded.Add((coord, delta));
                updates++;
            }
            else
            {
                Check(sparse.Prefix(coord) == bigDense.Prefix(coord), "random agreement step " + step);
                queries++;
            }
        }
        Check(updates == 400 && queries == 400, "operation counts");

        foreach (int coord in new[] { 1, 12345, Size / 2, Size - 1, Size })
        {
            long expected = 0;
            foreach (var (position, delta) in recorded)
            {
                if (position <= coord) expected += delta;
            }
            Check(sparse.Prefix(coord) == expected, "brute agreement " + coord);
            Check(sparse.Prefix(coord) == bigDense.Prefix(coord), "dense agreement " + coord);
        }
        Check(sparse.Touched() >= 1, "some cells touched");
        Check(sparse.Touched() <= 21 * updates, "compressed within bound " + sparse.Touched());
        Check(sparse.Touched() < Size / 100, "compressed far below dense " + sparse.Touched());
        long total = 0;
        foreach (var (_, delta) in recorded) total += delta;
        Check(sparse.Prefix(Size) == total, "total sum");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
