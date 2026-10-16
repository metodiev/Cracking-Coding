// Task 755 - Implement timing attack resistant comparison function and test. (Tier: Expert, Beyond - Security & Privacy)
//
// A comparison that returns as soon as it finds a mismatch leaks *where* the mismatch is: an attacker who can
// measure the cost of "does this guess equal the secret" recovers the secret byte by byte. This file implements
// both sides of that story and makes the security claim testable.
//
// Timing model. Real wall-clock timing is not reproducible inside a unit-checked program, so the cost of an
// operation is modelled explicitly: every counted step (one byte loaded and XORed, one loop iteration) adds one
// to a meter that the implementations bump by hand. The model is exact for the straight-line code below, and it
// is the model - not a stopwatch - that the tests assert on. Documented simplification: this measures *executed
// counted operations*, not cache behaviour, branch prediction or speculative execution, which are the real
// reasons a branch-free implementation is still not perfectly constant time.
//
// Functions: NaiveEquals (early exit, the vulnerable one), ConstantTimeEquals (XOR-accumulate the whole array,
// branch on the result only), ConstantTimeCompare (a branch-free three-way ordinal comparison built from
// less/greater flag accumulators) and ConstantTimeSelect (a mask based conditional move). Equal-length inputs
// are required: the length of a token is public metadata in every real protocol, so the length check branches
// and that leak is documented and asserted, not hidden.
//
// Invariants verified: (a) correctness against plain brute-force references on thousands of random pairs;
// (b) the constant-time implementations have exactly one cost over inputs that differ at every position, while
// the naive one's cost *is* the mismatch position; (c) a real byte-at-a-time timing attack recovers 15 of the
// 16 secret bytes from the naive comparator (the final byte stays ambiguous because a mismatch there costs
// exactly as much as a full match), and against the constant-time comparator every one of the 256 candidates
// scores identically at every position, so the attack has no advantage at all.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    private static ulong state = 0x7557AE1C0FFEEUL;

    internal static ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(int bound) => (int)(Next() % (ulong)bound);

    internal static byte[] Bytes(int length)
    {
        var bytes = new byte[length];
        for (int i = 0; i < length; i++) bytes[i] = (byte)Next();
        return bytes;
    }
}

internal static class Meter
{
    internal static long Ops;

    internal static void Reset() => Ops = 0;
    internal static void Step() => Ops++;
}

internal static class Comparisons
{
    /** Vulnerable: the loop stops at the first difference, so the cost is the mismatch position. */
    internal static bool NaiveEquals(byte[] left, byte[] right)
    {
        Meter.Step();
        if (left.Length != right.Length) return false;
        for (int i = 0; i < left.Length; i++)
        {
            Meter.Step();
            if (left[i] != right[i]) return false;
        }
        return true;
    }

    /** Constant time: the whole array is folded into one accumulator, the only branch is on the final result. */
    internal static bool ConstantTimeEquals(byte[] left, byte[] right)
    {
        Meter.Step();
        if (left.Length != right.Length) return false;      // the length is public metadata, not a secret
        int difference = 0;
        for (int i = 0; i < left.Length; i++)
        {
            Meter.Step();
            difference |= left[i] ^ right[i];
        }
        return difference == 0;
    }

    /** Branch-free three-way ordinal comparison; -1, 0 or 1. The prefix decides first, the length only breaks a tie. */
    internal static int ConstantTimeCompare(byte[] left, byte[] right)
    {
        Meter.Step();
        int shared = Math.Min(left.Length, right.Length);
        int less = 0;
        int greater = 0;
        for (int i = 0; i < shared; i++)
        {
            Meter.Step();
            int x = left[i];
            int y = right[i];
            int lt = ((x - y) >> 8) & 1;                    // 1 when x < y, no branch
            int gt = ((y - x) >> 8) & 1;
            int undecided = (less | greater) ^ 1;           // 1 until the first difference decides the order
            less |= lt & undecided;
            greater |= gt & undecided;
        }
        int lengthLess = left.Length < right.Length ? 1 : 0;
        int lengthGreater = right.Length < left.Length ? 1 : 0;
        int stillUndecided = (less | greater) ^ 1;
        less |= lengthLess & stillUndecided;
        greater |= lengthGreater & stillUndecided;
        return greater - less;                              // "less" is left < right, so left < right gives -1
    }

    /** Conditional move: mask is 0x00 or 0xFF, and both values are touched regardless of the mask. */
    internal static byte ConstantTimeSelect(byte mask, byte whenMaskSet, byte whenMaskClear)
    {
        Meter.Step();
        return (byte)((whenMaskSet & mask) | (whenMaskClear & ~mask));
    }

    /** Plain brute-force reference, used only as an oracle. */
    internal static bool ReferenceEquals(byte[] left, byte[] right)
    {
        if (left.Length != right.Length) return false;
        bool same = true;
        for (int i = 0; i < left.Length; i++)
        {
            if (left[i] != right[i]) same = false;
        }
        return same;
    }

    internal static int ReferenceCompare(byte[] left, byte[] right)
    {
        int shared = Math.Min(left.Length, right.Length);
        for (int i = 0; i < shared; i++)
        {
            if (left[i] != right[i]) return left[i] < right[i] ? -1 : 1;
        }
        return left.Length.CompareTo(right.Length);
    }
}

public static class Solution
{
    private const int SecretLength = 16;

    public static void Main()
    {
        var secret = new byte[SecretLength];
        for (int i = 0; i < SecretLength; i++) secret[i] = (byte)(i * 31 + 7);

        Correctness(secret);
        TimingInvariance(secret);
        NaiveLeaksThePosition(secret);
        Attack(secret);
        Boundaries();

        Console.WriteLine("PASS");
    }

    private static void Correctness(byte[] secret)
    {
        int mismatches = 0, lengthCases = 0;
        for (int trial = 0; trial < 2000; trial++)
        {
            int length = 1 + SplitMix64.NextInt(24);
            byte[] left = SplitMix64.Bytes(length);
            byte[] right = (byte[])left.Clone();
            if (trial % 2 == 0 && length > 0) right[SplitMix64.NextInt(length)] ^= 0x5A;   // force a difference
            if (trial % 7 == 0)
            {
                right = new byte[length + 1];
                Array.Copy(left, right, length);
                lengthCases++;
            }
            bool expectedSame = Comparisons.ReferenceEquals(left, right);
            if (Comparisons.NaiveEquals(left, right) != expectedSame) mismatches++;
            if (Comparisons.ConstantTimeEquals(left, right) != expectedSame) mismatches++;
            if (Comparisons.ConstantTimeCompare(left, right) != Comparisons.ReferenceCompare(left, right)) mismatches++;
        }
        Check(mismatches == 0 && lengthCases > 200, "equality and ordering agree with brute force on 2000 pairs");

        Check(Comparisons.ConstantTimeSelect(0xFF, 0xAB, 0x12) == 0xAB
            && Comparisons.ConstantTimeSelect(0x00, 0xAB, 0x12) == 0x12,
            "conditional move picks the right operand");
        Check(Comparisons.ConstantTimeSelect(0x00, 0xAB, 0x12) == Comparisons.ConstantTimeSelect(0x00, 0xCD, 0x12),
            "the result depends only on the mask and the selected operand");

        var longer = new byte[20];
        Check(!Comparisons.ConstantTimeEquals(secret, longer) && !Comparisons.NaiveEquals(secret, longer),
            "a length mismatch is not equal");
        Check(Comparisons.ConstantTimeCompare(secret, longer) == Comparisons.ReferenceCompare(secret, longer),
            "a length mismatch still orders like the reference");
    }

    private static void TimingInvariance(byte[] secret)
    {
        var constantTimeCosts = new SortedSet<long>();
        var naiveCosts = new SortedSet<long>();
        for (int i = 0; i < secret.Length; i++)
        {
            var guess = (byte[])secret.Clone();
            guess[i] ^= 0x01;                                 // differs exactly at position i
            Meter.Reset();
            Comparisons.ConstantTimeEquals(guess, secret);
            constantTimeCosts.Add(Meter.Ops);
            Meter.Reset();
            Comparisons.NaiveEquals(guess, secret);
            naiveCosts.Add(Meter.Ops);
        }
        Meter.Reset();
        Comparisons.ConstantTimeEquals(secret, secret);
        constantTimeCosts.Add(Meter.Ops);
        Meter.Reset();
        Comparisons.NaiveEquals(secret, secret);
        naiveCosts.Add(Meter.Ops);

        Check(constantTimeCosts.Count == 1 && constantTimeCosts.Contains(1 + secret.Length),
            "constant-time equality costs exactly 1 + length for every input: {" + string.Join(",", constantTimeCosts) + "}");
        Check(naiveCosts.Count == secret.Length,
            "naive equality takes " + naiveCosts.Count + " distinct costs: a mismatch at the last byte costs "
            + "the same as a full match, which is itself a (smaller) leak");

        var ordering = new SortedSet<long>();
        for (int i = 0; i < 200; i++)
        {
            byte[] left = SplitMix64.Bytes(12);
            byte[] right = SplitMix64.Bytes(12);
            Meter.Reset();
            Comparisons.ConstantTimeCompare(left, right);
            ordering.Add(Meter.Ops);
        }
        Check(ordering.Count == 1 && ordering.Contains(1 + 12), "constant-time ordering is cost invariant too");

        var selectCosts = new SortedSet<long>();
        for (int mask = 0; mask < 2; mask++)
        {
            for (int i = 0; i < 50; i++)
            {
                Meter.Reset();
                Comparisons.ConstantTimeSelect((byte)(mask * 0xFF), (byte)i, (byte)(255 - i));
                selectCosts.Add(Meter.Ops);
            }
        }
        Check(selectCosts.Count == 1 && selectCosts.Contains(1), "conditional move executes one step for either mask");
    }

    private static void NaiveLeaksThePosition(byte[] secret)
    {
        var costs = new long[secret.Length + 1];
        for (int i = 0; i < secret.Length; i++)
        {
            var guess = (byte[])secret.Clone();
            guess[i] ^= 0x01;
            Meter.Reset();
            Comparisons.NaiveEquals(guess, secret);
            costs[i] = Meter.Ops;
        }
        Meter.Reset();
        Comparisons.NaiveEquals(secret, secret);
        costs[secret.Length] = Meter.Ops;

        bool monotone = true;
        for (int i = 0; i < secret.Length; i++) monotone &= costs[i] == i + 2;
        Check(monotone, "the naive cost is exactly the mismatch position plus 2: "
            + costs[0] + "... " + costs[secret.Length - 1]);
        Check(costs[secret.Length] == 1 + secret.Length && costs[secret.Length - 1] == 1 + secret.Length,
            "a mismatch at the last byte is indistinguishable from a full match");
    }

    private static void Attack(byte[] secret)
    {
        AttackResult leaky = Recover(secret, Comparisons.NaiveEquals);
        Check(SamePrefix(leaky.Recovered, secret, secret.Length - 1),
            "the timing attack recovers 15 of the 16 secret bytes: " + Hex(leaky.Recovered));
        bool uniqueEverywhere = true;
        for (int i = 0; i < secret.Length - 1; i++) uniqueEverywhere &= leaky.Ties[i] == 1;
        Check(uniqueEverywhere, "every position except the last has exactly one maximal candidate");
        Check(leaky.Ties[secret.Length - 1] == 256,
            "the last byte is entirely ambiguous: a mismatch there costs exactly as much as a full match");

        var withTrueTail = (byte[])leaky.Recovered.Clone();
        withTrueTail[secret.Length - 1] = secret[secret.Length - 1];
        Meter.Reset();
        Comparisons.NaiveEquals(withTrueTail, secret);
        long trueTailCost = Meter.Ops;
        Meter.Reset();
        Comparisons.NaiveEquals(leaky.Recovered, secret);
        long zeroTailCost = Meter.Ops;
        Check(trueTailCost == zeroTailCost,
            "the recovered tail and the true tail cost the same (" + trueTailCost + "), so 15 of 16 bytes leak");

        AttackResult resistant = Recover(secret, Comparisons.ConstantTimeEquals);
        bool allTie = true;
        for (int i = 0; i < secret.Length; i++) allTie &= resistant.Ties[i] == 256;
        Check(allTie, "against the constant-time comparator all 256 candidates tie at every position");
        Check(!SameBytes(resistant.Recovered, secret) && resistant.Recovered[0] == 0,
            "the attack learns nothing from the constant-time comparator: " + Hex(resistant.Recovered));
    }

    private sealed class AttackResult
    {
        internal byte[] Recovered = Array.Empty<byte>();
        internal int[] Ties = Array.Empty<int>();
    }

    /** Classic byte-at-a-time timing attack: pick, at each position, the candidate with the highest cost. */
    private static AttackResult Recover(byte[] secret, Func<byte[], byte[], bool> comparator)
    {
        var result = new AttackResult { Recovered = new byte[secret.Length], Ties = new int[secret.Length] };
        for (int position = 0; position < secret.Length; position++)
        {
            long best = -1;
            int bestByte = 0;
            int ties = 0;
            for (int candidate = 0; candidate < 256; candidate++)
            {
                var guess = (byte[])result.Recovered.Clone();
                guess[position] = (byte)candidate;
                Meter.Reset();
                comparator(guess, secret);
                long cost = Meter.Ops;
                if (cost > best)
                {
                    best = cost;
                    bestByte = candidate;
                    ties = 1;
                }
                else if (cost == best)
                {
                    ties++;
                }
            }
            result.Recovered[position] = (byte)bestByte;
            result.Ties[position] = ties;
        }
        return result;
    }

    private static void Boundaries()
    {
        var empty = Array.Empty<byte>();
        Check(Comparisons.ConstantTimeEquals(empty, empty) && Comparisons.NaiveEquals(empty, empty),
            "two empty arrays are equal");
        Meter.Reset();
        Comparisons.ConstantTimeEquals(empty, empty);
        Check(Meter.Ops == 1, "the empty comparison costs only the length step, observed " + Meter.Ops);

        var single = new byte[] { 0x42 };
        var other = new byte[] { 0x42 };
        var different = new byte[] { 0x43 };
        Check(Comparisons.ConstantTimeEquals(single, other) && !Comparisons.ConstantTimeEquals(single, different),
            "single byte arrays compare correctly");
        Check(Comparisons.ConstantTimeCompare(single, different) < 0 && Comparisons.ConstantTimeCompare(different, single) > 0,
            "single byte ordering");

        var zeros = new byte[64];
        var ones = new byte[64];
        for (int i = 0; i < 64; i++) ones[i] = 0xFF;
        Meter.Reset();
        Comparisons.ConstantTimeEquals(zeros, ones);
        long cost = Meter.Ops;
        Check(cost == 65 && !Comparisons.ConstantTimeEquals(zeros, ones), "64 byte all-different comparison costs 65");

        var longLeft = SplitMix64.Bytes(1024);
        var longRight = (byte[])longLeft.Clone();
        longRight[1023] ^= 0x80;
        var costs = new SortedSet<long>();
        for (int i = 0; i < 8; i++)
        {
            int flip = i * 128;
            var probe = (byte[])longLeft.Clone();
            probe[flip] ^= 0xFF;
            Meter.Reset();
            Comparisons.ConstantTimeEquals(probe, longLeft);
            costs.Add(Meter.Ops);
            Meter.Reset();
            Comparisons.ConstantTimeEquals(longRight, longLeft);
            costs.Add(Meter.Ops);
        }
        Check(costs.Count == 1 && costs.Contains(1 + 1024),
            "1024 byte comparisons never leak the position: {" + string.Join(",", costs) + "}");

        var lengthProbe = new byte[1025];
        Meter.Reset();
        Comparisons.ConstantTimeEquals(longLeft, lengthProbe);
        long lengthCost = Meter.Ops;
        Check(lengthCost == 1, "a length mismatch costs 1 step: the length is the documented residual leak");
    }

    private static bool SamePrefix(byte[] left, byte[] right, int length)
    {
        for (int i = 0; i < length; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }

    private static bool SameBytes(byte[] left, byte[] right)
    {
        if (left.Length != right.Length) return false;
        for (int i = 0; i < left.Length; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }

    private static string Hex(byte[] bytes)
    {
        var parts = new List<string>();
        for (int i = 0; i < Math.Min(bytes.Length, 8); i++) parts.Add(bytes[i].ToString("X2"));
        return string.Join(" ", parts) + (bytes.Length > 8 ? " ..." : "");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
