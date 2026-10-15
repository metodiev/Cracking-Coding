// Task 374 - Bitset (bit array) with fast set operations for a small universe. (Tier: Intermediate, CTCI Ch 5)
using System;
using System.Collections.Generic;
using System.Numerics;

/// <summary>
/// A fixed universe of n_bits (here &lt;= 256) is packed into 32-bit words.  Every
/// operation touches only whole words, so add/remove/contains are O(1), Count() is
/// O(words) using a popcount, and Union/Intersection are O(words) instead of
/// O(universe).  Iteration extracts the lowest set bit of each word in turn, which
/// yields indices in strictly ascending order without scanning empty stretches.
///
/// Everything is cross-checked against the plain set oracle (SortedSet) on deterministic
/// LCG universes: membership of every index, Count() against the oracle size, ascending
/// iteration against the sorted oracle, and Union/Intersection against the oracle set
/// operations.  Boundary indices 0, 31, 32 and n_bits-1 are asserted explicitly.
///
/// The implementation only uses 32-bit words, so the sign quirks of each language are
/// handled identically: words are treated as unsigned bit patterns (uint arithmetic,
/// BitOperations.PopCount and BitOperations.TrailingZeroCount).
/// </summary>
public static class Solution
{
    private const int Universe = 256;

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

    /// <summary>Bitset over indices 0 .. bits-1, backed by 32-bit words.</summary>
    private sealed class IntBitset
    {
        private readonly int bits;
        private readonly uint[] words;

        public IntBitset(int bits)
        {
            if (bits <= 0) throw new ArgumentException("universe must be positive");
            this.bits = bits;
            words = new uint[(bits + 31) / 32];
        }

        public void Add(int index)
        {
            Validate(index);
            words[index >> 5] |= 1u << (index & 31);
        }

        public void Remove(int index)
        {
            Validate(index);
            words[index >> 5] &= ~(1u << (index & 31));
        }

        public bool Contains(int index)
        {
            Validate(index);
            return ((words[index >> 5] >> (index & 31)) & 1u) == 1u;
        }

        public int Count()
        {
            int total = 0;
            foreach (uint word in words) total += BitOperations.PopCount(word);
            return total;
        }

        public IntBitset Union(IntBitset other)
        {
            CheckSameUniverse(other);
            var result = new IntBitset(bits);
            for (int i = 0; i < words.Length; i++) result.words[i] = words[i] | other.words[i];
            return result;
        }

        public IntBitset Intersection(IntBitset other)
        {
            CheckSameUniverse(other);
            var result = new IntBitset(bits);
            for (int i = 0; i < words.Length; i++) result.words[i] = words[i] & other.words[i];
            return result;
        }

        /// <summary>All members as indices in ascending order.</summary>
        public int[] Iterate()
        {
            var members = new List<int>();
            for (int wi = 0; wi < words.Length; wi++)
            {
                uint word = words[wi];
                while (word != 0)
                {
                    uint low = word & (uint)(~word + 1u);       // lowest set bit
                    members.Add((wi << 5) + BitOperations.TrailingZeroCount(word));
                    word &= ~low;
                }
            }
            return members.ToArray();
        }

        private void Validate(int index)
        {
            if (index < 0 || index >= bits) throw new ArgumentException("index " + index);
        }

        private void CheckSameUniverse(IntBitset other)
        {
            if (bits != other.bits) throw new ArgumentException("universe size mismatch");
        }
    }

    private static int[] SortedMembers(SortedSet<int> oracle)
    {
        var result = new int[oracle.Count];
        oracle.CopyTo(result);
        return result;
    }

    public static void Main()
    {
        // --- boundary behaviour ----------------------------------------------
        var empty = new IntBitset(Universe);
        Check(empty.Count() == 0, "empty count");
        Check(empty.Iterate().Length == 0, "empty iterate");

        var boundary = new IntBitset(Universe);
        foreach (int index in new[] { 0, 31, 32, 33, 63, 64, 255 }) boundary.Add(index);
        Check(SequenceEqual(boundary.Iterate(), new[] { 0, 31, 32, 33, 63, 64, 255 }), "boundary iterate");
        Check(boundary.Count() == 7, "boundary count");
        Check(boundary.Contains(0) && boundary.Contains(255), "boundary contains");
        Check(!boundary.Contains(1) && !boundary.Contains(254), "boundary non-members");
        boundary.Add(0);
        Check(boundary.Count() == 7, "add is idempotent");
        boundary.Remove(1);
        Check(boundary.Count() == 7, "remove absent is a no-op");
        boundary.Remove(0);
        Check(!boundary.Contains(0) && boundary.Count() == 6, "remove works");

        // --- explicit small example -------------------------------------------
        var first = new IntBitset(64);
        var second = new IntBitset(64);
        foreach (int index in new[] { 1, 5, 40, 63 }) first.Add(index);
        foreach (int index in new[] { 5, 7, 40, 41 }) second.Add(index);
        Check(SequenceEqual(first.Iterate(), new[] { 1, 5, 40, 63 }), "first iterate");
        Check(SequenceEqual(second.Iterate(), new[] { 5, 7, 40, 41 }), "second iterate");
        Check(SequenceEqual(first.Union(second).Iterate(), new[] { 1, 5, 7, 40, 41, 63 }), "union");
        Check(SequenceEqual(first.Intersection(second).Iterate(), new[] { 5, 40 }), "intersection");
        Check(first.Union(second).Count() == 6, "union count");
        Check(first.Intersection(second).Count() == 2, "intersection count");
        Check(new IntBitset(64).Union(first).Count() == 4, "union with empty");
        Check(new IntBitset(64).Intersection(first).Count() == 0, "intersection with empty");

        var full = new IntBitset(Universe);
        for (int index = 0; index < Universe; index++) full.Add(index);
        Check(full.Count() == Universe, "full count");
        Check(full.Iterate().Length == Universe, "full iterate size");
        var spark = new IntBitset(Universe);
        foreach (int index in new[] { 1, 5, 40, 63, 255 }) spark.Add(index);
        Check(full.Union(spark).Count() == Universe, "full union stays full");
        Check(SequenceEqual(full.Union(spark).Iterate(), Range(Universe)), "full union iterate");
        Check(SequenceEqual(full.Intersection(spark).Iterate(), spark.Iterate()), "full intersection");
        Check(new IntBitset(Universe).Intersection(full).Count() == 0, "empty intersection full");

        // --- LCG cross-checks against the plain set oracle --------------------
        var lcg = new Lcg(37413);
        for (int trial = 0; trial < 40; trial++)
        {
            var oracleA = new SortedSet<int>();
            var oracleB = new SortedSet<int>();
            var bitsA = new IntBitset(Universe);
            var bitsB = new IntBitset(Universe);

            for (int step = 0; step < 60; step++)
            {
                int index = lcg.Next(Universe);
                int action = lcg.Next(Universe);
                if (action % 3 == 0)
                {
                    bitsA.Remove(index);
                    oracleA.Remove(index);
                }
                else
                {
                    bitsA.Add(index);
                    oracleA.Add(index);
                }
                index = lcg.Next(Universe);
                action = lcg.Next(Universe);
                if (action % 3 == 0)
                {
                    bitsB.Remove(index);
                    oracleB.Remove(index);
                }
                else
                {
                    bitsB.Add(index);
                    oracleB.Add(index);
                }
            }

            for (int index = 0; index < Universe; index++)
            {
                Check(bitsA.Contains(index) == oracleA.Contains(index), "membership A " + trial);
                Check(bitsB.Contains(index) == oracleB.Contains(index), "membership B " + trial);
            }
            Check(bitsA.Count() == oracleA.Count, "count A " + trial);
            Check(bitsB.Count() == oracleB.Count, "count B " + trial);
            Check(SequenceEqual(bitsA.Iterate(), SortedMembers(oracleA)), "iterate A " + trial);
            Check(SequenceEqual(bitsB.Iterate(), SortedMembers(oracleB)), "iterate B " + trial);

            var unionOracle = new SortedSet<int>(oracleA);
            unionOracle.UnionWith(oracleB);
            var intersectionOracle = new SortedSet<int>(oracleA);
            intersectionOracle.IntersectWith(oracleB);
            IntBitset union = bitsA.Union(bitsB);
            IntBitset intersection = bitsA.Intersection(bitsB);
            Check(union.Count() == unionOracle.Count, "union count " + trial);
            Check(intersection.Count() == intersectionOracle.Count, "intersection count " + trial);
            Check(SequenceEqual(union.Iterate(), SortedMembers(unionOracle)), "union iterate " + trial);
            Check(SequenceEqual(intersection.Iterate(), SortedMembers(intersectionOracle)),
                "intersection iterate " + trial);
            for (int index = 0; index < Universe; index += 7)
            {
                Check(union.Contains(index) == unionOracle.Contains(index), "union membership " + trial);
                Check(intersection.Contains(index) == intersectionOracle.Contains(index),
                    "intersection membership " + trial);
            }
        }
        Console.WriteLine("PASS");
    }

    private static int[] Range(int count)
    {
        var result = new int[count];
        for (int i = 0; i < count; i++) result[i] = i;
        return result;
    }

    private static bool SequenceEqual(int[] a, int[] b)
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
