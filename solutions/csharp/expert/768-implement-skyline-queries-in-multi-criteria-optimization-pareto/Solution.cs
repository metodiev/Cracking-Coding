// Task 768 - Implement skyline queries in multi-criteria optimization (Pareto front). (Tier: Expert, CTCI Ch 8 - Recursion and Dynamic Programming)
//
// Given n points in d dimensions and a per-dimension preference (lower-is-better or higher-is-better),
// point p dominates q when p is at least as good as q on every criterion and strictly better on at
// least one. The skyline (Pareto front) is the set of points that nobody dominates. Two structural
// facts hold for every input and are the invariants this file checks on all of them:
//   (1) the skyline is an antichain - no member dominates another member;
//   (2) it covers the input  - every input point is either on the skyline or dominated by a member.
//
// Four independent methods are implemented so that they can cross-check each other:
//   * BruteForce      - O(n^2) dominance scan used as the reference oracle.
//   * BlockNestedLoops- Bentley's BNL: a self-maintaining candidate window; a new point is dropped as
//                       soon as one window member dominates it, otherwise it evicts every window
//                       member it dominates and joins the window. Comparison count is tracked.
//   * DominatorCount  - skyline = points with zero dominators, counted by a plain double loop.
//   * MinPrefixSweep  - the O(n log n) 2D specialization: sort by x, then keep the running minimum of
//                       y over strictly smaller x. Equal-x points are handled as a group, so a
//                       duplicate is kept (a twin is never strictly better) while a same-x point with
//                       a smaller y does dominate.
// A fifth path maintains the skyline incrementally over a growing point stream and is re-validated
// against the brute force after every insertion.
//
// Coordinates are mapped into a canonical "smaller is better" plane before the sweep, so mixed
// preference vectors (minimize one criterion, maximize another) reuse the same code path.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Point
{
    internal readonly int[] Coords;
    internal readonly string Id;

    internal Point(string id, params int[] coords)
    {
        Id = id;
        Coords = coords;
    }

    internal int Dim => Coords.Length;
}

internal static class Skyline
{
    private static bool AtLeastAsGood(int candidate, int other, bool lower) => lower ? candidate <= other : candidate >= other;
    private static bool StrictlyBetter(int candidate, int other, bool lower) => lower ? candidate < other : candidate > other;

    /// <summary>True when p is no worse than q on every criterion and strictly better on one.</summary>
    internal static bool Dominates(Point p, Point q, bool[] lower)
    {
        bool strict = false;
        for (int j = 0; j < p.Dim; j++)
        {
            if (!AtLeastAsGood(p.Coords[j], q.Coords[j], lower[j])) return false;
            if (StrictlyBetter(p.Coords[j], q.Coords[j], lower[j])) strict = true;
        }
        return strict;
    }

    internal static void Validate(Point[] points, bool[] lower)
    {
        foreach (Point p in points)
        {
            if (p.Dim != lower.Length)
                throw new ArgumentException("point " + p.Id + " has " + p.Dim + " coordinates but the preference vector has " + lower.Length);
        }
    }

    /// <summary>Reference oracle: every point is checked against every other point.</summary>
    internal static List<int> BruteForce(Point[] points, bool[] lower, out long comparisons)
    {
        Validate(points, lower);
        var result = new List<int>();
        comparisons = 0;
        for (int i = 0; i < points.Length; i++)
        {
            bool dominated = false;
            for (int j = 0; j < points.Length && !dominated; j++)
            {
                if (i == j) continue;
                comparisons++;
                if (Dominates(points[j], points[i], lower)) dominated = true;
            }
            if (!dominated) result.Add(i);
        }
        return result;
    }

    /// <summary>Bentley's block nested loops: a window of candidates is maintained while scanning.</summary>
    internal static List<int> BlockNestedLoops(Point[] points, bool[] lower, out long comparisons)
    {
        Validate(points, lower);
        var window = new List<int>();
        comparisons = 0;
        for (int i = 0; i < points.Length; i++)
        {
            bool dominated = false;
            for (int w = 0; w < window.Count; w++)
            {
                comparisons++;
                if (Dominates(points[window[w]], points[i], lower)) { dominated = true; break; }
            }
            if (dominated) continue;
            for (int w = window.Count - 1; w >= 0; w--)
            {
                comparisons++;
                if (Dominates(points[i], points[window[w]], lower)) window.RemoveAt(w);
            }
            window.Add(i);
        }
        return window;
    }

    /// <summary>Number of points that dominate each input point; the skyline is exactly the zeros.</summary>
    internal static int[] DominatorCounts(Point[] points, bool[] lower)
    {
        Validate(points, lower);
        var counts = new int[points.Length];
        for (int i = 0; i < points.Length; i++)
            for (int j = 0; j < points.Length; j++)
                if (i != j && Dominates(points[j], points[i], lower)) counts[i]++;
        return counts;
    }

    /// <summary>O(n log n) sweep for d = 2; coordinates are first mapped so that smaller is better.</summary>
    internal static List<int> MinPrefixSweep(Point[] points, bool[] lower)
    {
        Validate(points, lower);
        if (lower.Length != 2) throw new ArgumentException("the prefix sweep only handles two criteria");
        int n = points.Length;
        var x = new long[n];
        var y = new long[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = lower[0] ? points[i].Coords[0] : -(long)points[i].Coords[0];
            y[i] = lower[1] ? points[i].Coords[1] : -(long)points[i].Coords[1];
        }
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int c = x[a].CompareTo(x[b]);
            if (c != 0) return c;
            c = y[a].CompareTo(y[b]);
            return c != 0 ? c : a.CompareTo(b);
        });

        var result = new List<int>();
        long prefixMinY = long.MaxValue;                    // minimum y over strictly smaller x
        int g = 0;
        while (g < n)
        {
            int groupStart = g;
            long groupMinY = long.MaxValue;
            while (g < n && x[order[g]] == x[order[groupStart]])
            {
                int idx = order[g];
                bool beatenBySmallerX = prefixMinY <= y[idx];
                bool beatenInsideGroup = groupMinY < y[idx]; // same x, strictly smaller y dominates
                if (!beatenBySmallerX && !beatenInsideGroup) result.Add(idx);
                if (y[idx] < groupMinY) groupMinY = y[idx];
                g++;
            }
            if (groupMinY < prefixMinY) prefixMinY = groupMinY;
        }
        return result;
    }

    /// <summary>Streaming skyline: the front is repaired on every insertion instead of rescanned.</summary>
    internal static List<int> Incremental(Point[] points, bool[] lower)
    {
        Validate(points, lower);
        var front = new List<int>();
        for (int i = 0; i < points.Length; i++)
        {
            bool dominated = false;
            for (int f = 0; f < front.Count; f++)
            {
                if (Dominates(points[front[f]], points[i], lower)) { dominated = true; break; }
            }
            if (dominated) continue;
            for (int f = front.Count - 1; f >= 0; f--)
            {
                if (Dominates(points[i], points[front[f]], lower)) front.RemoveAt(f);
            }
            front.Add(i);
        }
        return front;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static List<int> Sorted(List<int> xs)
    {
        var copy = new List<int>(xs);
        copy.Sort();
        return copy;
    }

    private static string Show(List<int> xs)
    {
        var sb = new System.Text.StringBuilder("[");
        for (int i = 0; i < xs.Count; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(xs[i]);
        }
        return sb.Append(']').ToString();
    }

    private static bool SameSet(List<int> a, List<int> b)
    {
        var x = Sorted(a);
        var y = Sorted(b);
        if (x.Count != y.Count) return false;
        for (int i = 0; i < x.Count; i++) if (x[i] != y[i]) return false;
        return true;
    }

    /// <summary>Invariants (1) antichain and (2) coverage, checked with the brute-force dominance test.</summary>
    private static string StructureViolation(Point[] points, bool[] lower, List<int> front)
    {
        var onFront = new bool[points.Length];
        foreach (int i in front) onFront[i] = true;
        for (int a = 0; a < front.Count; a++)
            for (int b = 0; b < front.Count; b++)
                if (a != b && Skyline.Dominates(points[front[a]], points[front[b]], lower))
                    return "front member " + front[a] + " dominates front member " + front[b];
        for (int i = 0; i < points.Length; i++)
        {
            if (onFront[i]) continue;
            bool covered = false;
            foreach (int f in front) if (Skyline.Dominates(points[f], points[i], lower)) { covered = true; break; }
            if (!covered) return "point " + i + " is neither on the front nor dominated by it";
        }
        return null;
    }

    public static void Main()
    {
        var min2 = new bool[] { true, true };
        var max2 = new bool[] { false, false };
        var mixed3 = new bool[] { true, false, true };

        // ---- hand-computed 2D example (lower is better on both axes) ----------------------------
        {
            var pts = new[]
            {
                new Point("a", 1, 5), new Point("b", 2, 2), new Point("c", 3, 3),
                new Point("d", 4, 1), new Point("e", 5, 4),
            };
            long bc, bn;
            int[] dominators = Skyline.DominatorCounts(pts, min2);
            var expected = new List<int> { 0, 1, 3 };            // (1,5), (2,2), (4,1): c and e are dominated
            var brute = Skyline.BruteForce(pts, min2, out bc);
            var bnl = Skyline.BlockNestedLoops(pts, min2, out bn);
            var sweep = Skyline.MinPrefixSweep(pts, min2);
            var incremental = Skyline.Incremental(pts, min2);
            Check(SameSet(brute, expected), "hand-computed 2D skyline is {(1,5),(2,2),(4,1)}, got " + Show(brute));
            Check(SameSet(bnl, expected), "block nested loops reproduces the hand-computed skyline");
            Check(SameSet(sweep, expected), "the 2D prefix sweep reproduces the hand-computed skyline");
            Check(SameSet(incremental, expected), "the incremental front reproduces the hand-computed skyline");
            var zero = new List<int>();
            for (int i = 0; i < pts.Length; i++) if (dominators[i] == 0) zero.Add(i);
            Check(SameSet(zero, expected), "points with zero dominators are exactly the skyline");
            Check(StructureViolation(pts, min2, brute) == null, "hand example: antichain + coverage hold");
            Console.WriteLine("2D example: bruteForceComparisons=" + bc + " bnlComparisons=" + bn + " frontSize=" + brute.Count);
        }

        // ---- boundary inputs ---------------------------------------------------------------------
        {
            var empty = Array.Empty<Point>();
            long c1, c2;
            Check(Skyline.BruteForce(empty, min2, out c1).Count == 0, "the empty input has an empty skyline");
            Check(Skyline.BlockNestedLoops(empty, min2, out c2).Count == 0, "block nested loops handles the empty input");
            Check(Skyline.MinPrefixSweep(empty, min2).Count == 0, "the sweep handles the empty input");

            var single = new[] { new Point("s", 7, 9) };
            Check(Skyline.MinPrefixSweep(single, min2).Count == 1, "a single point is its own skyline");

            var twins = new[] { new Point("t1", 4, 4), new Point("t2", 4, 4), new Point("t3", 4, 4) };
            var twinFront = Skyline.MinPrefixSweep(twins, min2);
            Check(twinFront.Count == 3, "identical points never strictly dominate each other, so all three survive");

            var oneDim = new[]
            {
                new Point("p0", 9), new Point("p1", 2), new Point("p2", 5), new Point("p3", 2),
            };
            long c3, c4;
            var oneDimFront = Skyline.BruteForce(oneDim, new bool[] { true }, out c3);
            Check(oneDimFront.Count == 2, "a 1-D skyline keeps only the best value (both copies of the minimum)");
            Check(SameSet(Skyline.BlockNestedLoops(oneDim, new bool[] { true }, out c4), oneDimFront),
                "BNL agrees with brute force on a 1-D input");

            var incomparable = new[] { new Point("u", 1, 9), new Point("v", 9, 1) };
            Check(Skyline.MinPrefixSweep(incomparable, min2).Count == 2, "two incomparable points are both on the front");

            bool threw = false;
            try { Skyline.BruteForce(new[] { new Point("bad", 1, 2, 3) }, min2, out _); }
            catch (ArgumentException) { threw = true; }
            Check(threw, "a point whose dimension does not match the preference vector is rejected");

            bool threwSweep = false;
            try { Skyline.MinPrefixSweep(new[] { new Point("bad", 1, 2, 3) }, new bool[] { true, true, true }); }
            catch (ArgumentException) { threwSweep = true; }
            Check(threwSweep, "the 2D sweep rejects three-dimensional input instead of silently ignoring a criterion");
        }

        // ---- randomized cross-validation of all four methods -------------------------------------
        {
            const int Size = 400, Dim = 4;
            ulong state = 0x5EED5EED1234ABCDUL;
            var pts = new Point[Size];
            for (int i = 0; i < Size; i++)
            {
                var coords = new int[Dim];
                for (int j = 0; j < Dim; j++) coords[j] = SplitMix64.NextInt(ref state, 1000);
                pts[i] = new Point("p" + i, coords);
            }
            var prefs = new bool[] { true, false, true, true };   // minimize criteria 0,2,3 and maximize 1
            long bc, bn;
            var brute = Skyline.BruteForce(pts, prefs, out bc);
            var bnl = Skyline.BlockNestedLoops(pts, prefs, out bn);
            var inc = Skyline.Incremental(pts, prefs);
            var counts = Skyline.DominatorCounts(pts, prefs);
            var zeros = new List<int>();
            for (int i = 0; i < Size; i++) if (counts[i] == 0) zeros.Add(i);
            Check(SameSet(brute, bnl), "BNL matches brute force on 400 four-dimensional points");
            Check(SameSet(brute, inc), "the incremental front matches brute force on the same 400 points");
            Check(SameSet(brute, zeros), "the zero-dominator set matches brute force on the same 400 points");
            Check(StructureViolation(pts, prefs, brute) == null, "invariants hold for the 4-D random set: " + StructureViolation(pts, prefs, brute));
            Check(bn < bc, "the BNL window inspects fewer pairs than the naive double loop (" + bn + " < " + bc + ")");
            Check(brute.Count > 0 && brute.Count < Size / 4,
                "a 4-D random front is non-trivial but small (got " + brute.Count + " of " + Size + ")");
            Console.WriteLine("4D random n=400: front=" + brute.Count + " bruteComparisons=" + bc + " bnlComparisons=" + bn);

            // Mixed preferences: the same point set with all criteria maximized must give the
            // front of the simultaneously negated set, which is the symmetry the sweep relies on.
            var maxPrefs = new bool[] { false, false, false, false };
            var negated = new Point[Size];
            for (int i = 0; i < Size; i++)
            {
                var coords = new int[Dim];
                for (int j = 0; j < Dim; j++) coords[j] = -pts[i].Coords[j];
                negated[i] = new Point("n" + i, coords);
            }
            long c5, c6;
            var maximized = Skyline.BruteForce(pts, maxPrefs, out c5);
            var negatedMin = Skyline.BruteForce(negated, new bool[] { true, true, true, true }, out c6);
            Check(SameSet(maximized, negatedMin), "maximizing every criterion equals minimizing the negated point set");
        }

        // ---- 2D sweep versus brute force, including ties and duplicates ---------------------------
        {
            const int Size = 350;
            ulong state = 0x0DDBA11CAFEF00DUL;
            var pts = new Point[Size];
            for (int i = 0; i < Size; i++)
            {
                // a small coordinate range deliberately creates many ties
                pts[i] = new Point("q" + i, SplitMix64.NextInt(ref state, 60), SplitMix64.NextInt(ref state, 60));
            }
            long bc;
            var brute = Skyline.BruteForce(pts, min2, out bc);
            var sweep = Skyline.MinPrefixSweep(pts, min2);
            long bn;
            var bnl = Skyline.BlockNestedLoops(pts, min2, out bn);
            Check(SameSet(sweep, brute), "the O(n log n) sweep matches brute force on 350 tied/duplicated 2D points");
            Check(SameSet(bnl, brute), "BNL matches brute force on the same tied 2D input");
            Check(StructureViolation(pts, min2, brute) == null, "invariants hold on the tied 2D input");
            Check(brute.Count > 0 && brute.Count <= Size, "the tied 2D front is a proper subset of the input");
            Console.WriteLine("2D ties n=350: front=" + brute.Count + " bruteComparisons=" + bc);
        }

        // ---- streaming maintenance: re-validate the front after every insertion -------------------
        {
            const int Size = 250;
            ulong state = 0xF00DF00DF00DF00DUL;
            var pts = new Point[Size];
            for (int i = 0; i < Size; i++)
                pts[i] = new Point("s" + i, SplitMix64.NextInt(ref state, 500), SplitMix64.NextInt(ref state, 500), SplitMix64.NextInt(ref state, 500));
            var prefs = new bool[] { true, true, false };

            var front = new List<int>();
            bool everDisagreed = false;
            int shrinks = 0;
            for (int i = 0; i < Size; i++)
            {
                // one incremental step, mirroring Skyline.Incremental
                bool dominated = false;
                foreach (int f in front) if (Skyline.Dominates(pts[f], pts[i], prefs)) { dominated = true; break; }
                if (!dominated)
                {
                    int before = front.Count;
                    for (int f = front.Count - 1; f >= 0; f--) if (Skyline.Dominates(pts[i], pts[front[f]], prefs)) front.RemoveAt(f);
                    if (front.Count < before) shrinks++;
                    front.Add(i);
                }
                var prefix = new Point[i + 1];
                Array.Copy(pts, prefix, i + 1);
                long ignored;
                var reference = Skyline.BruteForce(prefix, prefs, out ignored);
                if (!SameSet(front, reference)) { everDisagreed = true; break; }
                if (StructureViolation(prefix, prefs, front) != null) { everDisagreed = true; break; }
            }
            Check(!everDisagreed, "the streaming front equals the brute-force front after every one of the " + Size + " insertions");
            Check(shrinks > 0, "insertions really evicted dominated members (" + shrinks + " shrink events)");
            Console.WriteLine("streaming: insertions=" + Size + " evictions=" + shrinks + " finalFront=" + front.Count + " dims=" + pts[0].Dim);
        }

        Console.WriteLine("PASS");
    }
}
