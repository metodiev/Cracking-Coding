// Task 652 - Implement bitmap index and fast AND/OR operations. (Tier: Expert, CTCI Ch 14 - Databases)
//
// A bitmap index stores one bitvector per distinct column value; bit i is set when row i holds that value.
// Predicates then become word-parallel set algebra: `region = a AND category = b` is a 64-bit-at-a-time AND
// of two bitvectors, `col IN {...}` is an OR-fold, `col <> a` is a complement inside the row universe. The
// implementation is a real Bitmap type over ulong[] words with word-by-word AND / OR / ANDNOT, a SWAR
// popcount (not a per-bit loop), a set-bit iterator, a complement that knows the valid bit length, and an
// invariant that no bit at index >= Length is ever set - which keeps AND/OR/NOT closed over the universe
// instead of leaking junk into the complement. Combine() reports how many 64-bit words it touched, so the
// checks can quantify the win: an AND over 512 rows reads 16 words instead of comparing 512 rows.
// Verified: the SWAR popcount equals a naive per-bit count; De Morgan holds; AND/OR are commutative,
// associative and idempotent; a word plus its complement partitions the 64 bit positions; inclusion-exclusion
// holds; the set-bit iterator is strictly ascending; the tail invariant survives complement; and, the
// load-bearing cross-check, every predicate result is compared row-by-row against a brute-force scan of the
// column for 240 deterministic workloads. All inputs come from a hand-written splitmix64 stream.
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

/// <summary>A fixed-length bitvector over 64-bit words; bits at index >= Length stay permanently zero.</summary>
internal sealed class Bitmap
{
    private readonly ulong[] _words;
    private readonly int _length;

    internal Bitmap(int length)
    {
        _length = length;
        _words = new ulong[(length + 63) / 64];
    }

    internal int WordCount => _words.Length;

    internal ulong Word(int index) => _words[index];

    /// <summary>Words read/written by the most recent whole-vector operation.</summary>
    internal long WordsTouched { get; private set; }

    internal void Set(int index)
    {
        if (index < 0 || index >= _length) throw new ArgumentOutOfRangeException(nameof(index));
        _words[index >> 6] |= 1UL << (index & 63);
    }

    internal bool Get(int index)
    {
        if (index < 0 || index >= _length) throw new ArgumentOutOfRangeException(nameof(index));
        return (_words[index >> 6] & (1UL << (index & 63))) != 0;
    }

    internal static Bitmap And(Bitmap a, Bitmap b) => Combine(a, b, (x, y) => x & y);

    internal static Bitmap Or(Bitmap a, Bitmap b) => Combine(a, b, (x, y) => x | y);

    internal static Bitmap AndNot(Bitmap a, Bitmap b) => Combine(a, b, (x, y) => x & ~y);

    private static Bitmap Combine(Bitmap a, Bitmap b, Func<ulong, ulong, ulong> op)
    {
        if (a._length != b._length) throw new ArgumentException("bitmaps must cover the same universe");
        var result = new Bitmap(a._length);
        for (int i = 0; i < a._words.Length; i++)
        {
            result._words[i] = op(a._words[i], b._words[i]);
            result.WordsTouched += 2;
        }
        result.Trim();
        return result;
    }

    /// <summary>Complement inside the row universe [0, Length).</summary>
    internal Bitmap Not()
    {
        var result = new Bitmap(_length);
        for (int i = 0; i < _words.Length; i++) result._words[i] = ~_words[i];
        result.Trim();
        result.WordsTouched = _words.Length;
        return result;
    }

    internal bool IsEmpty()
    {
        for (int i = 0; i < _words.Length; i++) if (_words[i] != 0) return false;
        return true;
    }

    /// <summary>Population count computed word-by-word with the classic SWAR masks.</summary>
    internal int Cardinality()
    {
        int total = 0;
        for (int i = 0; i < _words.Length; i++) total += PopCount(_words[i]);
        WordsTouched = _words.Length;
        return total;
    }

    internal static int PopCount(ulong value)
    {
        ulong v = value;
        v -= (v >> 1) & 0x5555555555555555UL;
        v = (v & 0x3333333333333333UL) + ((v >> 2) & 0x3333333333333333UL);
        v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FUL;
        return (int)((v * 0x0101010101010101UL) >> 56);
    }

    /// <summary>Ascending set-bit positions - the row identifiers a predicate answers with.</summary>
    internal List<int> Positions()
    {
        var positions = new List<int>();
        for (int w = 0; w < _words.Length; w++)
        {
            ulong bits = _words[w];
            while (bits != 0)
            {
                positions.Add((w << 6) + System.Numerics.BitOperations.TrailingZeroCount(bits));
                bits &= bits - 1;
            }
        }
        return positions;
    }

    /// <summary>Clears any bit beyond Length, restoring the tail invariant.</summary>
    internal void Trim()
    {
        int tail = _length & 63;
        if (tail != 0 && _words.Length > 0) _words[_words.Length - 1] &= (1UL << tail) - 1;
    }

    /// <summary>Null when every bit at index >= Length is clear, otherwise a description of the violation.</summary>
    internal string? InvariantViolation()
    {
        int tail = _length & 63;
        if (tail == 0) return null;
        ulong mask = (1UL << tail) - 1;
        return (_words[_words.Length - 1] & ~mask) == 0 ? null : "the last word has bits beyond the universe";
    }
}

/// <summary>One bitmap per distinct value of one column; range predicates are OR-folds of value bitmaps.</summary>
internal sealed class BitmapIndex
{
    private readonly int[] _column;
    private readonly Dictionary<int, Bitmap> _byValue = new Dictionary<int, Bitmap>();

    internal BitmapIndex(int[] column) => _column = column;

    internal int Length => _column.Length;

    internal int DistinctValues => _byValue.Count;

    internal void Build()
    {
        foreach (int v in _column) if (!_byValue.ContainsKey(v)) _byValue[v] = new Bitmap(_column.Length);
        for (int i = 0; i < _column.Length; i++) _byValue[_column[i]].Set(i);
    }

    internal Bitmap Values(int value) => _byValue.TryGetValue(value, out Bitmap? b) ? b : new Bitmap(Length);

    /// <summary>Bitmap of the rows whose value lies in [lo, hi], folded with word-parallel Ors.</summary>
    internal Bitmap Range(int lo, int hi)
    {
        if (lo > hi) throw new ArgumentException("lo must not exceed hi", nameof(hi));
        Bitmap? acc = null;
        for (int v = lo; v <= hi; v++)
        {
            if (!_byValue.TryGetValue(v, out Bitmap? bitmap)) continue;
            acc = acc is null ? bitmap : Bitmap.Or(acc, bitmap);
        }
        return acc ?? new Bitmap(Length);
    }

    /// <summary>Null when every value bitmap has a clean tail and every row is covered exactly once.</summary>
    internal string? InvariantViolation()
    {
        long total = 0;
        foreach (var pair in _byValue)
        {
            if (pair.Value.InvariantViolation() is string violation) return violation;
            total += pair.Value.Cardinality();
        }
        if (total != Length) return "per-value bitmaps cover " + total + " of " + Length + " rows";
        for (int i = 0; i < Length; i++)
        {
            int hits = 0;
            foreach (var pair in _byValue) if (pair.Value.Get(i)) hits++;
            if (hits != 1) return "row " + i + " is set in " + hits + " value bitmaps";
        }
        return null;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static bool Same(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>Brute-force row ids for `region = r AND category = c`.</summary>
    private static List<int> RowsWhere(int[] region, int[] category, int r, int c)
    {
        var rows = new List<int>();
        for (int i = 0; i < region.Length; i++) if (region[i] == r && category[i] == c) rows.Add(i);
        return rows;
    }

    /// <summary>Brute-force row ids for `column = value`.</summary>
    private static List<int> RowsWithValue(int[] column, int value)
    {
        var rows = new List<int>();
        for (int i = 0; i < column.Length; i++) if (column[i] == value) rows.Add(i);
        return rows;
    }

    private static void Main()
    {
        // ---- the Bitmap type itself -----------------------------------------------------------
        {
            var b = new Bitmap(200);
            Check(b.Cardinality() == 0 && b.IsEmpty() && b.InvariantViolation() == null,
                "a fresh bitmap is empty and satisfies the tail invariant");
            b.Set(0); b.Set(63); b.Set(64); b.Set(199);
            var positions = b.Positions();
            bool ascending = positions.Count == 4 && positions[0] == 0 && positions[3] == 199;
            for (int i = 1; i < positions.Count; i++) if (positions[i] <= positions[i - 1]) ascending = false;
            Check(ascending && b.Cardinality() == 4 && b.Get(64) && !b.Get(65),
                "four single-bit sets straddle the 64-bit word boundary and iterate in ascending order");
            var full = b.Not();
            Check(full.Cardinality() == 196 && full.InvariantViolation() == null && full.Not().Cardinality() == 4,
                "the complement covers exactly the remaining rows and keeps the tail invariant");
            Check(Same(full.Not().Positions(), positions), "double complement restores the exact bits");
            bool rejected = false;
            try { b.Set(200); } catch (ArgumentOutOfRangeException) { rejected = true; }
            bool mismatched = false;
            try { _ = Bitmap.And(b, new Bitmap(7)); } catch (ArgumentException) { mismatched = true; }
            Check(rejected && mismatched,
                "an out-of-universe bit and a mismatched-universe AND are both rejected");
        }

        // ---- SWAR popcount against a naive per-bit count ---------------------------------------
        {
            ulong seed = 0xA5A5A5A5DEADBEEFUL;
            bool agree = Bitmap.PopCount(0UL) == 0 && Bitmap.PopCount(ulong.MaxValue) == 64
                && Bitmap.PopCount(1UL << 63) == 1;
            for (int t = 0; t < 64; t++)
            {
                ulong value = Rng.Next(ref seed);
                int naive = 0;
                for (int bit = 0; bit < 64; bit++) if ((value & (1UL << bit)) != 0) naive++;
                if (Bitmap.PopCount(value) != naive || Bitmap.PopCount(~value) != 64 - naive) agree = false;
            }
            Check(agree, "the SWAR popcount matches a naive per-bit count on 64 random words and both extremes");
        }

        const int Rows = 512, Regions = 8, Categories = 6, Conjunctions = 200;
        ulong state = 0x1234ABCD5678EF90UL;
        int[] region = new int[Rows];
        int[] category = new int[Rows];
        for (int i = 0; i < Rows; i++)
        {
            region[i] = Rng.NextInt(ref state, Regions);
            category[i] = Rng.NextInt(ref state, Categories);
        }
        var regionIndex = new BitmapIndex(region);
        regionIndex.Build();
        var categoryIndex = new BitmapIndex(category);
        categoryIndex.Build();
        Check(regionIndex.InvariantViolation() == null && categoryIndex.InvariantViolation() == null
                && regionIndex.DistinctValues == Regions && categoryIndex.DistinctValues == Categories,
            "every row lands in exactly one bitmap per index, one bitmap per distinct value, tails clean");

        // ---- 200 two-column conjunctions, each cross-checked against a scan --------------------
        long wordsForOneAnd = 0;
        bool conjunctionsAgree = true, commutative = true, deMorgan = true, singleValueAgree = true;
        string? failure = null;
        for (int t = 0; t < Conjunctions; t++)
        {
            int r = Rng.NextInt(ref state, Regions);
            int c = Rng.NextInt(ref state, Categories);
            Bitmap both = Bitmap.And(regionIndex.Values(r), categoryIndex.Values(c));
            wordsForOneAnd = both.WordsTouched;
            var rows = both.Positions();
            if (!Same(rows, RowsWhere(region, category, r, c))) { conjunctionsAgree = false; failure ??= "region#" + r; }
            if (!Same(Bitmap.And(categoryIndex.Values(c), regionIndex.Values(r)).Positions(), rows)) commutative = false;
            if (!Same(both.Not().Positions(),
                    Bitmap.Or(regionIndex.Values(r).Not(), categoryIndex.Values(c).Not()).Positions()))
            {
                deMorgan = false;
                failure ??= "De Morgan";
            }
        }
        Check(conjunctionsAgree, "all 200 two-column conjunctions match the brute-force scan: " + (failure ?? "ok"));
        Check(commutative, "AND is commutative on those 200 pairs");
        Check(deMorgan, "NOT(a AND b) == NOT a OR NOT b on those 200 pairs");
        Check(wordsForOneAnd <= Rows / 32, "one AND reads at most " + Rows / 32 + " words, not " + Rows + " rows");
        for (int r = 0; r < Regions; r++)
            if (!Same(regionIndex.Values(r).Positions(), RowsWithValue(region, r))) singleValueAgree = false;
        Check(singleValueAgree, "every single-value bitmap equals exactly the rows holding that value");

        // ---- 40 range predicates via OR-folds, plus the degenerate cases ------------------------
        {
            bool rangesAgree = true;
            int rowsReturned = 0;
            for (int t = 0; t < 40; t++)
            {
                int a = Rng.NextInt(ref state, Regions);
                int b = Rng.NextInt(ref state, Regions);
                int lo = Math.Min(a, b), hi = Math.Max(a, b);
                var expected = new List<int>();
                for (int i = 0; i < Rows; i++) if (region[i] >= lo && region[i] <= hi) expected.Add(i);
                var got = regionIndex.Range(lo, hi).Positions();
                if (!Same(got, expected)) rangesAgree = false;
                rowsReturned += got.Count;
            }
            Check(rangesAgree, "all 40 range predicates match the brute-force scan");
            Check(regionIndex.Range(0, Regions - 1).Cardinality() == Rows, "the full-domain range covers every row");
            Check(Same(regionIndex.Range(0, 0).Positions(), RowsWithValue(region, 0))
                    && regionIndex.Range(Regions + 5, Regions + 9).IsEmpty(),
                "a degenerate single-value range is that value's own bitmap and an unindexed range is empty");
            Check(new BitmapIndex(Array.Empty<int>()).Range(0, 3).IsEmpty(), "an empty column produces no rows");
            var dup = new BitmapIndex(new[] { 3, 3, 3, 1, 2, 2, 0, 9, 9, 9, 9, 5 });
            dup.Build();
            Check(dup.InvariantViolation() == null && dup.DistinctValues == 6
                    && Same(dup.Values(9).Positions(), new List<int> { 7, 8, 9, 10 }) && dup.Values(4).IsEmpty(),
                "a duplicate-heavy column indexes cleanly, finds all repeats and reports absent values as empty");
            Check(dup.Range(1, 3).Cardinality() == 6 && dup.Range(9, 12).Cardinality() == 4,
                "ranges count the right rows and tolerate gaps in the value domain");
            Console.WriteLine("rangeRowsReturned=" + rowsReturned);
        }

        // ---- set-algebra identities over two random bitmaps -------------------------------------
        {
            var x = new Bitmap(Rows);
            var y = new Bitmap(Rows);
            for (int i = 0; i < Rows; i++)
            {
                if (Rng.NextInt(ref state, 3) == 0) x.Set(i);
                if (Rng.NextInt(ref state, 3) == 0) y.Set(i);
            }
            var z = new Bitmap(Rows);
            for (int i = 0; i < Rows; i += 5) z.Set(i);

            bool laws = Bitmap.And(x, x).Cardinality() == x.Cardinality()
                && Bitmap.Or(x, x).Cardinality() == x.Cardinality()
                && Bitmap.And(x, y).Cardinality() == Bitmap.And(y, x).Cardinality()
                && Bitmap.Or(x, y).Cardinality() == Bitmap.Or(y, x).Cardinality()
                && Bitmap.And(Bitmap.And(x, y), z).Cardinality() == Bitmap.And(x, Bitmap.And(y, z)).Cardinality()
                && Bitmap.Or(Bitmap.Or(x, y), z).Cardinality() == Bitmap.Or(x, Bitmap.Or(y, z)).Cardinality();
            Check(laws, "AND/OR are idempotent, commutative and associative on 512-bit vectors");

            int andCard = Bitmap.And(x, y).Cardinality();
            int orCard = Bitmap.Or(x, y).Cardinality();
            Check(andCard + orCard == x.Cardinality() + y.Cardinality(),
                "inclusion-exclusion: |a AND b| + |a OR b| == |a| + |b|");
            Check(Bitmap.AndNot(x, y).Cardinality() == x.Cardinality() - andCard
                    && Bitmap.AndNot(y, x).Cardinality() == y.Cardinality() - andCard,
                "ANDNOT removes exactly the shared rows from either side");
            bool partitions = true;
            for (int w = 0; w < x.WordCount; w++)
            {
                if (Bitmap.PopCount(x.Word(w)) + Bitmap.PopCount(x.Not().Word(w)) != 64) partitions = false;
                if (Bitmap.And(x, x.Not()).Word(w) != 0 || Bitmap.Or(x, x.Not()).Word(w) != ulong.MaxValue)
                    partitions = false;
            }
            Check(partitions, "each word and its complement partition all 64 bit positions");
            Console.WriteLine("xSet=" + x.Cardinality() + " ySet=" + y.Cardinality() + " intersect=" + andCard
                + " union=" + orCard + " wordsPerAnd=" + wordsForOneAnd);
        }

        Console.WriteLine("PASS");
    }
}
