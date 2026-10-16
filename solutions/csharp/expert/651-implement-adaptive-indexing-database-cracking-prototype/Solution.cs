// Task 651 - Implement adaptive indexing (database cracking) prototype. (Tier: Expert, CTCI Ch 14 - Databases)
//
// Database cracking (Idreos et al.) builds an index *incrementally out of the workload* instead of paying one
// up-front sort. The column stays a plain array; the index is a partition of that array into *pieces* whose
// value ranges are globally ordered, so piece k only holds values <= those of piece k+1. A range query
// [lo, hi] walks the pieces: a disjoint piece is skipped without touching a single data element, a contained
// piece contributes its whole length, and a partially overlapping piece is *cracked* in place with a
// three-way (Dutch national flag) partition into [< lo | in [lo,hi] | > hi].
// Invariant maintained after every crack: the bounds are strictly increasing, the per-piece [min,max]
// metadata describes its piece exactly, max(piece k) <= min(piece k+1), and the array is still a permutation
// of the input (cracking only moves values). Because the permutation is preserved, a brute-force linear scan
// over the untouched input must agree with the crack index on every query - the cross-check this file leans
// on for the whole 120-query workload. A consequence of the invariant is that once every distinct value has
// been used as a point query, every piece holds a single value and the column has been sorted by the
// workload itself, which is asserted directly. Cost is accounted honestly (CrackWork = elements pushed by
// partitions and metadata rebuilds, QueryWork = pieces examined) and the file verifies the cumulative cost
// stays under the naive "rescan the column per query" baseline, while a four-query workload stays cheaper
// than the equivalent full sort. All data comes from a hand-written splitmix64 stream, so the run is
// deterministic.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal static class Cracking
{
    internal sealed class Index
    {
        private readonly int[] _data;
        // Piece k covers the index range [_bounds[k], _bounds[k + 1]), hence bounds count = pieces + 1.
        private readonly List<int> _bounds = new List<int>();
        private readonly List<int> _min = new List<int>();
        private readonly List<int> _max = new List<int>();

        internal long Queries, Cracks, CrackWork, QueryWork;

        internal Index(int[] data)
        {
            _data = data;
            _bounds.Add(0);
            if (data.Length == 0) return;
            _bounds.Add(data.Length);
            _min.Add(Min(data, 0, data.Length));
            _max.Add(Max(data, 0, data.Length));
        }

        internal int PieceCount => _bounds.Count - 1;

        internal int[] DataCopy() => (int[])_data.Clone();

        internal int PointCount(int value) => RangeCount(value, value);

        /// <summary>Number of values in [lo, hi], cracking every partially overlapping piece on demand.</summary>
        internal int RangeCount(int lo, int hi)
        {
            if (lo > hi) throw new ArgumentException("lo must not exceed hi", nameof(hi));
            Queries++;
            int total = 0;
            int k = 0;
            while (k < PieceCount)
            {
                int from = _bounds[k];
                int to = _bounds[k + 1];
                QueryWork++;
                if (_max[k] < lo || _min[k] > hi) { k++; continue; }                        // disjoint
                if (_min[k] >= lo && _max[k] <= hi) { total += to - from; k++; continue; }  // contained
                total += Crack(k, lo, hi);
                k = _bounds.BinarySearch(to);   // resume at the piece starting where the old one ended
            }
            return total;
        }

        /// <summary>Three-way partition of piece k around [lo, hi]; returns the size of the middle region.</summary>
        private int Crack(int k, int lo, int hi)
        {
            Cracks++;
            int from = _bounds[k];
            int to = _bounds[k + 1];
            int lt = from, i = from, gt = to;
            while (i < gt)
            {
                int v = _data[i];
                CrackWork++;
                if (v < lo) { Swap(i, lt); lt++; i++; }
                else if (v > hi) { gt--; Swap(i, gt); }
                else i++;
            }

            // The three regions are contiguous, so their starts (minus the redundant leading one) are the
            // new interior boundaries; empty regions simply contribute no piece.
            var boundsSub = new List<int> { from };
            var regionMins = new List<int>();
            var regionMaxs = new List<int>();
            AddRegion(from, lt, boundsSub, regionMins, regionMaxs);
            AddRegion(lt, gt, boundsSub, regionMins, regionMaxs);
            AddRegion(gt, to, boundsSub, regionMins, regionMaxs);
            boundsSub.Add(to);

            _bounds.RemoveRange(k, 2);
            _bounds.InsertRange(k, boundsSub);
            _min.RemoveAt(k);
            _max.RemoveAt(k);
            _min.InsertRange(k, regionMins);
            _max.InsertRange(k, regionMaxs);
            return gt - lt;
        }

        private void AddRegion(int from, int to, List<int> boundsSub, List<int> mins, List<int> maxs)
        {
            if (to <= from) return;
            if (boundsSub[boundsSub.Count - 1] != from) boundsSub.Add(from);
            mins.Add(Min(_data, from, to));
            maxs.Add(Max(_data, from, to));
            CrackWork += to - from;
        }

        private static int Min(int[] data, int from, int to)
        {
            int lo = int.MaxValue;
            for (int i = from; i < to; i++) if (data[i] < lo) lo = data[i];
            return lo;
        }

        private static int Max(int[] data, int from, int to)
        {
            int hi = int.MinValue;
            for (int i = from; i < to; i++) if (data[i] > hi) hi = data[i];
            return hi;
        }

        private void Swap(int a, int b)
        {
            if (a == b) return;
            int t = _data[a];
            _data[a] = _data[b];
            _data[b] = t;
        }

        /// <summary>Null when the crack invariant holds, otherwise the first violation found.</summary>
        internal string? InvariantViolation()
        {
            if (_bounds.Count != PieceCount + 1) return "bounds bookkeeping is inconsistent";
            if (_bounds[0] != 0) return "the first bound is not 0";
            if (_bounds[_bounds.Count - 1] != _data.Length) return "the last bound is not the column length";
            for (int k = 0; k < _bounds.Count - 1; k++)
                if (_bounds[k] >= _bounds[k + 1]) return "bounds are not strictly increasing at " + k;
            if (_min.Count != PieceCount || _max.Count != PieceCount) return "piece metadata has the wrong length";
            for (int k = 0; k < PieceCount; k++)
            {
                int lo = Min(_data, _bounds[k], _bounds[k + 1]);
                int hi = Max(_data, _bounds[k], _bounds[k + 1]);
                if (lo != _min[k] || hi != _max[k])
                    return "piece " + k + " claims [" + _min[k] + "," + _max[k] + "] but holds [" + lo + "," + hi + "]";
                if (k + 1 < PieceCount && _max[k] > _min[k + 1])
                    return "pieces " + k + "/" + (k + 1) + " are out of order (" + _max[k] + " > " + _min[k + 1] + ")";
            }
            return null;
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int BruteForceCount(int[] values, int lo, int hi)
    {
        int count = 0;
        foreach (int v in values) if (v >= lo && v <= hi) count++;
        return count;
    }

    private static int[] Sorted(int[] values)
    {
        int[] copy = (int[])values.Clone();
        Array.Sort(copy);
        return copy;
    }

    private static bool SameArray(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void Main()
    {
        // ---- degenerate inputs and a rejected query -------------------------------------------
        {
            var empty = new Cracking.Index(Array.Empty<int>());
            Check(empty.RangeCount(0, 100) == 0 && empty.PieceCount == 0, "an empty column matches nothing");
            Check(empty.InvariantViolation() == null, "the empty column satisfies the invariant trivially");

            var single = new Cracking.Index(new[] { 7 });
            Check(single.PointCount(7) == 1 && single.PointCount(6) == 0 && single.PointCount(8) == 0,
                "the single element is found exactly once and only for its own value");
            Check(single.InvariantViolation() == null, "one element keeps the invariant");
            Check(single.PointCount(7) == 1, "re-querying a cracked single-element column is stable");

            bool threw = false;
            try { single.RangeCount(9, 4); }
            catch (ArgumentException) { threw = true; }
            Check(threw, "an inverted range is rejected instead of silently returning 0");
        }

        const int Size = 400, Domain = 1000, Workload = 120;
        ulong state = 0x5EEDCAFE12345678UL;
        int[] original = new int[Size];
        for (int i = 0; i < Size; i++) original[i] = Rng.NextInt(ref state, Domain);
        int[] sortedOriginal = Sorted(original);

        // ---- the workload: every answer cross-checked against a brute-force scan ---------------
        var index = new Cracking.Index((int[])original.Clone());
        bool allAgree = true, invariantHeld = true, permutationHeld = true, refined = true;
        string? firstFailure = null;
        int previousPieces = index.PieceCount;
        for (int q = 0; q < Workload; q++)
        {
            int lo, hi;
            if (q % 3 == 0) { lo = Rng.NextInt(ref state, Domain); hi = lo; }
            else if (q % 3 == 1) { lo = Rng.NextInt(ref state, Domain); hi = Math.Min(Domain - 1, lo + Rng.NextInt(ref state, 5)); }
            else { int a = Rng.NextInt(ref state, Domain); int b = Rng.NextInt(ref state, Domain); lo = Math.Min(a, b); hi = Math.Max(a, b); }

            int got = index.RangeCount(lo, hi);
            int expected = BruteForceCount(original, lo, hi);
            if (got != expected) { allAgree = false; firstFailure ??= "query " + q + " [" + lo + "," + hi + "] gave " + got + " not " + expected; }
            if (index.InvariantViolation() is string violation) { invariantHeld = false; firstFailure ??= violation; }
            if (!SameArray(Sorted(index.DataCopy()), sortedOriginal)) { permutationHeld = false; firstFailure ??= "not a permutation at query " + q; }
            if (index.PieceCount < previousPieces) refined = false;
            previousPieces = index.PieceCount;
        }
        Check(allAgree, "every cracked range count matches the brute-force scan: " + (firstFailure ?? "ok"));
        Check(invariantHeld, "the piece invariant survives every crack: " + (firstFailure ?? "ok"));
        Check(permutationHeld, "cracking only permutes values, it never loses or invents one");
        Check(refined, "cracking only refines the partition, it never merges pieces");
        Check(index.Cracks > 0, "the workload really caused cracks");
        Check(index.Cracks <= 2L * index.Queries, "a query cracks at most two pieces (the two range borders)");
        Check(index.PieceCount <= 4 * index.Queries + 1, "each crack adds at most two new pieces");

        long naiveWork = (long)Workload * Size;
        long adaptiveWork = index.CrackWork + index.QueryWork;
        Console.WriteLine("queries=" + Workload + " cracks=" + index.Cracks + " pieces=" + index.PieceCount
            + " crackWork=" + index.CrackWork + " queryWork=" + index.QueryWork + " naiveFullScanWork=" + naiveWork);
        Check(adaptiveWork < naiveWork, "cracking inspects fewer elements than rescanning the column per query");

        // ---- a short workload is cheaper than the equivalent up-front sort ---------------------
        {
            var fresh = new Cracking.Index((int[])original.Clone());
            for (int q = 0; q < 4; q++) fresh.PointCount(Rng.NextInt(ref state, Domain));
            long sortWork = 0;
            for (int pass = Size; pass > 1; pass /= 2) sortWork += Size;   // ~log2(Size) merge levels
            Check(fresh.CrackWork + fresh.QueryWork < sortWork,
                "four queries crack more cheaply than a full sort costs (" + (fresh.CrackWork + fresh.QueryWork)
                + " < " + sortWork + ")");
        }

        // ---- point-cracking every distinct value sorts the column outright ---------------------
        {
            var contig = new Cracking.Index((int[])original.Clone());
            var seen = new bool[Domain];
            var distinct = new List<int>();
            foreach (int v in original) if (!seen[v]) { seen[v] = true; distinct.Add(v); }
            distinct.Sort();
            long accounted = 0;
            foreach (int v in distinct) accounted += contig.PointCount(v);
            Check(accounted == Size, "the point queries account for every element exactly once");
            Check(contig.InvariantViolation() == null, "the fully point-cracked column still satisfies the invariant");
            Check(SameArray(contig.DataCopy(), sortedOriginal),
                "point-cracking every distinct value leaves the column sorted");
            Check(contig.PieceCount == distinct.Count, "one single-valued piece per distinct value");
            Console.WriteLine("distinctValues=" + distinct.Count + " finalPieces=" + contig.PieceCount);
        }

        Console.WriteLine("PASS");
    }
}
