// Task 371 - Line-segment intersection (Bentley-Ottmann sweep outline). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
using System;
using System.Collections.Generic;

/// <summary>
/// Detect all intersecting pairs among n segments.  The full Bentley-Ottmann algorithm
/// runs in O((n + k) log n) for k intersection points; this task implements the sweep
/// *skeleton* (event queue + status structure) and handles the small cases this repo
/// tests (n <= ~12) exactly, plus a plain O(n^2) fallback used as the oracle.
///
/// Bentley-Ottmann invariant (the full algorithm this skeleton approximates):
///   1. Events: every segment endpoint, ordered lexicographically by (x, y).  A left
///      endpoint inserts the segment into the status structure; a right endpoint
///      removes it.  Intersection points found during the sweep are also pushed as
///      events so that the order of the status structure is restored after a crossing.
///   2. Status structure: the segments currently crossing the vertical sweep line,
///      kept ordered by their y coordinate at the current sweep x.  In general
///      position no two of them cross between events, so the order is well defined
///      and stable there.
///   3. Crossing detection: only neighbours in the status can become adjacent-and-
///      swapped between events, so a crossing is reported when two segments are
///      adjacent in the status (on insertion/removal, and after processing a found
///      intersection event) and their exact intersection test succeeds.  Each found
///      intersection is reported once, at the first event x where the two segments
///      become adjacent.
///   4. Degeneracies (documented but only partially handled here): vertical segments
///      (processed entirely at one event x), several events with equal x, endpoint
///      touches, collinear overlaps and overlapping collinear bundles.
///
/// Small-case exact handling used here: at every event x, after all left endpoints of
/// that x have been inserted and finished segments dropped, every *pair* of segments in
/// the status structure is tested with the exact integer predicate.  An intersecting
/// pair always spans the event x = max(left_x1, left_x2) (the later segment's left
/// endpoint), so every intersecting pair is seen at least once, duplicate reports
/// collapse into a set, and the count matches the O(n^2) oracle.
///
/// All predicates use exact integer arithmetic (cross products); no floating point.
/// </summary>
public static class Solution
{
    private sealed class Lcg
    {
        private int seed;

        public Lcg(int seed)
        {
            this.seed = seed;
        }

        public int Next(int modulus)
        {
            seed = (seed * 75 + 74) % 65537;
            return seed % modulus;
        }
    }

    /// <summary>Twice the signed area of (p, q, r): >0 CCW, <0 CW, 0 collinear.</summary>
    private static long Orient(int px, int py, int qx, int qy, int rx, int ry)
    {
        return (long)(qx - px) * (ry - py) - (long)(qy - py) * (rx - px);
    }

    /// <summary>True when collinear q lies inside segment pr (endpoints included).</summary>
    private static bool OnSegment(int px, int py, int qx, int qy, int rx, int ry)
    {
        return Math.Min(px, rx) <= qx && qx <= Math.Max(px, rx)
            && Math.Min(py, ry) <= qy && qy <= Math.Max(py, ry);
    }

    /// <summary>Exact closed-segment intersection test (touches and overlaps count).</summary>
    public static bool SegmentsIntersect(int[] s, int[] t)
    {
        long o1 = Orient(s[0], s[1], s[2], s[3], t[0], t[1]);
        long o2 = Orient(s[0], s[1], s[2], s[3], t[2], t[3]);
        long o3 = Orient(t[0], t[1], t[2], t[3], s[0], s[1]);
        long o4 = Orient(t[0], t[1], t[2], t[3], s[2], s[3]);
        if (o1 * o2 < 0 && o3 * o4 < 0) return true;                   // proper crossing
        if (o1 == 0 && OnSegment(s[0], s[1], t[0], t[1], s[2], s[3])) return true;
        if (o2 == 0 && OnSegment(s[0], s[1], t[2], t[3], s[2], s[3])) return true;
        if (o3 == 0 && OnSegment(t[0], t[1], s[0], s[1], t[2], t[3])) return true;
        if (o4 == 0 && OnSegment(t[0], t[1], s[2], s[3], t[2], t[3])) return true;
        return false;
    }

    /// <summary>Normalise {x1,y1,x2,y2} so that (x1,y1) &lt;= (x2,y2) lexicographically.</summary>
    private static int[][] Normalized(int[][] segments)
    {
        var result = new int[segments.Length][];
        for (int i = 0; i < segments.Length; i++)
        {
            int[] s = (int[])segments[i].Clone();
            if (s[0] > s[2] || (s[0] == s[2] && s[1] > s[3]))
            {
                int tmp = s[0]; s[0] = s[2]; s[2] = tmp;
                tmp = s[1]; s[1] = s[3]; s[3] = tmp;
            }
            result[i] = s;
        }
        return result;
    }

    /// <summary>-1/0/1 comparing the segments' exact y values at the sweep line x.</summary>
    private static int CompareY(int[] a, int[] b, int x)
    {
        long na, da, nb, db;
        if (a[2] == a[0]) { na = a[1]; da = 1; }
        else { na = (long)a[1] * (a[2] - a[0]) + (long)(a[3] - a[1]) * (x - a[0]); da = a[2] - a[0]; }
        if (b[2] == b[0]) { nb = b[1]; db = 1; }
        else { nb = (long)b[1] * (b[2] - b[0]) + (long)(b[3] - b[1]) * (x - b[0]); db = b[2] - b[0]; }
        long left = na * db, right = nb * da;
        return left < right ? -1 : (left > right ? 1 : 0);
    }

    /// <summary>Sweep skeleton: event queue + status structure; returns the intersecting pairs.</summary>
    public static SortedSet<string> SweepIntersectionPairs(int[][] rawSegments)
    {
        int[][] segments = Normalized(rawSegments);
        var events = new List<int[]>();
        for (int idx = 0; idx < segments.Length; idx++)
        {
            events.Add(new[] { segments[idx][0], segments[idx][1], idx, 0 });   // left
            events.Add(new[] { segments[idx][2], segments[idx][3], idx, 1 });   // right
        }
        events.Sort((u, v) =>
        {
            if (u[0] != v[0]) return u[0].CompareTo(v[0]);
            if (u[1] != v[1]) return u[1].CompareTo(v[1]);
            if (u[2] != v[2]) return u[2].CompareTo(v[2]);
            return u[3].CompareTo(v[3]);
        });

        var status = new List<int>();                 // active ids, y-ordered
        var found = new SortedSet<string>();
        int i = 0;
        while (i < events.Count)
        {
            int x = events[i][0];
            int j = i;
            while (j < events.Count && events[j][0] == x)
            {
                int[] @event = events[j];
                if (@event[3] == 0)                   // insert, keeping y-order
                {
                    int pos = 0;
                    while (pos < status.Count && CompareY(segments[status[pos]], segments[@event[2]], x) <= 0) pos++;
                    status.Insert(pos, @event[2]);
                }
                j++;
            }
            status.RemoveAll(k => segments[k][2] < x);   // finish finished segments
            for (int a = 0; a < status.Count; a++)
            {
                for (int b = a + 1; b < status.Count; b++)
                {
                    if (SegmentsIntersect(segments[status[a]], segments[status[b]]))
                    {
                        int low = Math.Min(status[a], status[b]);
                        int high = Math.Max(status[a], status[b]);
                        found.Add(low + ":" + high);
                    }
                }
            }
            i = j;
        }
        return found;
    }

    /// <summary>O(n^2) oracle: test every pair with the same exact predicate.</summary>
    public static SortedSet<string> BruteForceIntersectionPairs(int[][] segments)
    {
        var found = new SortedSet<string>();
        for (int i = 0; i < segments.Length; i++)
        {
            for (int j = i + 1; j < segments.Length; j++)
            {
                if (SegmentsIntersect(segments[i], segments[j])) found.Add(i + ":" + j);
            }
        }
        return found;
    }

    private static int[][] LcgSegments(int count, Lcg lcg)
    {
        var segments = new int[count][];
        for (int i = 0; i < count; i++)
        {
            int x1, y1, x2, y2;
            do
            {
                x1 = lcg.Next(41); y1 = lcg.Next(41);
                x2 = lcg.Next(41); y2 = lcg.Next(41);
            } while (x1 == x2 && y1 == y2);
            segments[i] = new[] { x1, y1, x2, y2 };
        }
        return segments;
    }

    private static SortedSet<string> SetOf(params string[] keys)
    {
        return new SortedSet<string>(keys);
    }

    public static void Main()
    {
        // --- hand predicates ------------------------------------------------
        Check(SegmentsIntersect(new[] { 0, 0, 4, 4 }, new[] { 0, 4, 4, 0 }), "proper crossing");
        Check(SegmentsIntersect(new[] { 0, 0, 2, 2 }, new[] { 2, 2, 4, 0 }), "shared endpoint");
        Check(SegmentsIntersect(new[] { 0, 0, 4, 0 }, new[] { 2, 0, 6, 0 }), "collinear overlap");
        Check(!SegmentsIntersect(new[] { 0, 0, 2, 0 }, new[] { 0, 1, 2, 1 }), "parallel disjoint");
        Check(SegmentsIntersect(new[] { 0, 0, 4, 0 }, new[] { 2, 0, 2, 3 }), "T-touch");
        Check(!SegmentsIntersect(new[] { 0, 0, 1, 1 }, new[] { 3, 0, 4, 1 }), "separated");
        Check(SegmentsIntersect(new[] { 0, 0, 1, 1 }, new[] { 1, 1, 2, 0 }), "endpoint meet");

        // --- hand sweep set: 4 intersecting pairs ---------------------------
        int[][] hand =
        {
            new[] { 0, 0, 4, 4 }, new[] { 0, 4, 4, 0 }, new[] { 2, 2, 6, 2 },
            new[] { 6, 0, 6, 6 }, new[] { 8, 8, 9, 9 },
        };
        SortedSet<string> expected = SetOf("0:1", "0:2", "1:2", "2:3");
        Check(SweepIntersectionPairs(hand).SetEquals(expected), "hand sweep set");
        Check(BruteForceIntersectionPairs(hand).SetEquals(expected), "hand brute force");
        Check(expected.Count == 4, "hand count is 4");

        // --- collinear / touch family ----------------------------------------
        int[][] collinear =
        {
            new[] { 0, 0, 4, 0 }, new[] { 2, 0, 6, 0 }, new[] { 0, 1, 6, 1 }, new[] { 0, 0, 0, 1 },
        };
        Check(SweepIntersectionPairs(collinear).SetEquals(SetOf("0:1", "0:3", "2:3")), "collinear family");
        Check(SweepIntersectionPairs(collinear).SetEquals(BruteForceIntersectionPairs(collinear)),
            "collinear family vs oracle");

        int[][] disjoint = { new[] { 0, 0, 1, 1 }, new[] { 3, 0, 4, 1 }, new[] { 5, 5, 6, 6 } };
        Check(SweepIntersectionPairs(disjoint).Count == 0, "disjoint set");
        Check(SweepIntersectionPairs(new int[0][]).Count == 0, "empty input");

        // --- LCG cross-checks against the O(n^2) oracle -----------------------
        var lcg = new Lcg(37113);
        for (int trial = 0; trial < 40; trial++)
        {
            int count = 5 + lcg.Next(8);
            int[][] segments = LcgSegments(count, lcg);
            Check(SweepIntersectionPairs(segments).SetEquals(BruteForceIntersectionPairs(segments)),
                "random sweep vs brute force " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
