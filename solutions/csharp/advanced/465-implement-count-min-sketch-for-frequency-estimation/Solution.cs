// Task 465 - Implement count-min sketch for frequency estimation. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Count-min sketch with d = 4 rows and width = 48 cells, using FIXED hash seeds:
// column r = ((a_r * key + b_r) mod 1000003) mod width for r in 0..3. Counters only
// ever increase, so estimate(key) = min over the rows is NEVER below the true
// count; the overshoot is bounded by the collision mass.
//
// Checks: a hand-computed tiny sketch (width 4: columns(7) = columns(11) = [2, 0],
// columns(0) = [1, 3]) documents exact behaviours, then a fixed LCG stream of
// N = 5000 updates over ids 0..63 asserts for every id: estimate >= true count, and
// estimate - true count <= EPS * N with the documented EPS = 0.06 (>= the standard
// e/width ~ 0.0566); the measured max overshoot is 101 (~1/3 of the 300 bound) and
// 23 of the 64 ids do over-estimate, documenting that collisions really happen.
using System;

internal class CountMinSketch
{
    private const int P = 1000003;
    private static readonly int[] HashA = { 31, 131, 2711, 65521 };
    private static readonly int[] HashB = { 17, 91, 601, 4099 };

    private readonly int width, depth;
    private readonly int[][] table;

    public CountMinSketch(int width, int depth)
    {
        this.width = width;
        this.depth = depth;
        table = new int[depth][];
        for (int r = 0; r < depth; r++) table[r] = new int[width];
    }

    public int[] Columns(int key)
    {
        var cols = new int[depth];
        for (int r = 0; r < depth; r++)
            cols[r] = (int)(((long)HashA[r] * key + HashB[r]) % P % width);
        return cols;
    }

    public void Add(int key)
    {
        var cols = Columns(key);
        for (int r = 0; r < depth; r++) table[r][cols[r]]++;
    }

    public int Estimate(int key)
    {
        var cols = Columns(key);
        int best = int.MaxValue;
        for (int r = 0; r < depth; r++) best = Math.Min(best, table[r][cols[r]]);
        return best;
    }
}

public static class Solution
{
    private const double Eps = 0.06;
    private static int seed;

    private static int Lcg() => seed = (seed * 75 + 74) % 65537;

    public static void Main()
    {
        var sketch = new CountMinSketch(4, 2);
        Check(SameColumns(sketch.Columns(7), new[] { 2, 0 }), "columns(7)");
        Check(SameColumns(sketch.Columns(11), new[] { 2, 0 }), "columns(11)");
        sketch.Add(7);
        Check(sketch.Estimate(7) == 1, "tiny 1");
        sketch.Add(11);
        Check(sketch.Estimate(11) == 2, "tiny 2"); // both rows collide with key 7
        Check(sketch.Estimate(7) == 2, "tiny collision"); // over-estimate, never below
        sketch.Add(7);
        Check(sketch.Estimate(7) == 3, "tiny 3");
        Check(sketch.Estimate(0) == 0, "tiny untouched");

        sketch = new CountMinSketch(48, 4);
        seed = 7654321;
        const int total = 5000;
        var trueCounts = new int[64];
        for (int i = 0; i < total; i++) trueCounts[Lcg() % 64]++;
        seed = 7654321;
        for (int i = 0; i < total; i++) sketch.Add(Lcg() % 64);
        int maxOvershoot = 0;
        for (int item = 0; item < 64; item++)
        {
            int estimate = sketch.Estimate(item);
            Check(estimate >= trueCounts[item], "never under-estimates " + item);
            int overshoot = estimate - trueCounts[item];
            maxOvershoot = Math.Max(maxOvershoot, overshoot);
            Check(overshoot <= Eps * total, "overshoot bound " + item);
        }
        Check(maxOvershoot >= 1, "collisions occurred"); // documented
        int sum = 0;
        foreach (int c in trueCounts) sum += c;
        Check(sum == total, "bookkeeping");
        Console.WriteLine("PASS");
    }

    private static bool SameColumns(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
