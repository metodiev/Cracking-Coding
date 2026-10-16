// Task 756 - Implement side-channel resistant constant-time modular exponentiation. (Tier: Expert, Beyond - Security & Privacy)
//
// Modular exponentiation is the hot spot of RSA and Diffie-Hellman, and a naive square-and-multiply leaks the
// exponent through its own control flow: the extra multiply happens only when the current exponent bit is set,
// so both the Hamming weight and (with windows) the bit pattern are visible to an attacker who can count
// multiplications or time the operation. This file implements the fix and the tests that make the claim checkable.
//
// * Montgomery-style binary ladder: every exponent bit performs exactly one multiply and one square, and the bit
//   only drives two branch-free conditional swaps, so the work per bit is identical for bit 0 and bit 1.
// * Conditional swap by mask: difference = (a ^ b) & mask with mask = 0 or all ones, no data dependent jump.
// * Exponent blinding: e' = e + t*(p-1) for a prime modulus p; Fermat's little theorem makes a^e' == a^e mod p,
//   and because t is random the bit length (the one thing the ladder still leaks) no longer reveals e's length.
// * Base blinding: b' = b*r and a final multiplication by r^-e hides which base value is being exponentiated.
//
// Timing model. Wall clock time is not reproducible in a checked program, so cost is modelled explicitly: every
// modular multiplication and every conditional swap adds one to a counter. The model is exact for the code below.
// Documented simplification: this measures counted operations, not cache/branch-predictor effects, and the toy
// values are not key material - moduli are a few small primes and the arithmetic is 64 bit, so nothing here is
// usable as cryptography. What is verified is that the operation count is a function of the exponent *bit
// length* alone, that the leaky implementation's count is a function of the exponent *value*, that the results
// agree with two independent oracles (plus Fermat's little theorem), and that both blinding strategies preserve
// the result while changing the work profile.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    private static ulong state = 0x756C0FFEE1BADD5EUL;
    internal static ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

internal static class Clock
{
    internal static long Ops;

    internal static void Reset() => Ops = 0;
    internal static void Multiply() => Ops++;
    internal static void Swap() => Ops++;
}

internal static class Field
{
    /** Multiplies modulo `modulus`. Valid while modulus < 2^32, so the product fits in 64 bits. */
    internal static ulong Multiply(ulong left, ulong right, ulong modulus)
    {
        Clock.Multiply();
        return left * right % modulus;
    }

    /** Branch-free conditional swap; bit is 0 or 1 and only feeds the mask. */
    internal static void ConditionalSwap(int bit, ref ulong high, ref ulong low)
    {
        Clock.Swap();
        ulong mask = (ulong)(-(long)bit);              // 0 or 0xFFFF...FF, no branch
        ulong difference = (high ^ low) & mask;
        high ^= difference;
        low ^= difference;
    }

    internal static int BitLength(ulong value)
    {
        int bits = 0;
        while (value > 0)
        {
            bits++;
            value >>= 1;
        }
        return bits;
    }
}

internal static class Exponentiation
{
    /**
     * Constant time binary ladder. Invariant: after processing exponents' bits from the top, {r0, r1} holds
     * {base^acc, base^(acc+1)} regardless of which bit was taken, so each iteration costs the same two
     * multiplications and two conditional swaps for a 0 bit and for a 1 bit.
     */
    internal static ulong PowConstantTime(ulong baseValue, ulong exponent, ulong modulus, out int bits)
    {
        baseValue %= modulus;
        ulong r0 = 1 % modulus;
        ulong r1 = baseValue;
        bits = Field.BitLength(exponent);
        for (int i = bits - 1; i >= 0; i--)
        {
            int bit = (int)((exponent >> i) & 1UL);
            Field.ConditionalSwap(bit, ref r0, ref r1);
            r1 = Field.Multiply(r0, r1, modulus);
            r0 = Field.Multiply(r0, r0, modulus);
            Field.ConditionalSwap(bit, ref r0, ref r1);
        }
        return r0;
    }

    /** The vulnerable shape, kept for the timing comparison: the extra multiply is data dependent. */
    internal static ulong PowLeaky(ulong baseValue, ulong exponent, ulong modulus)
    {
        baseValue %= modulus;
        ulong result = 1 % modulus;
        int bits = Field.BitLength(exponent);
        for (int i = bits - 1; i >= 0; i--)
        {
            result = Field.Multiply(result, result, modulus);
            if (((exponent >> i) & 1UL) == 1) result = Field.Multiply(result, baseValue, modulus);
        }
        return result;
    }

    /** e' = e + t*(p - 1): identical result modulo a prime p, but the bit length now depends on t. */
    internal static ulong PowExponentBlinded(ulong baseValue, ulong exponent, ulong prime, ulong t, out int bits)
        => PowConstantTime(baseValue, exponent + t * (prime - 1), prime, out bits);

    /** b' = b*r, then multiply by r^-e: hides which base value was exponentiated. */
    internal static ulong PowBaseBlinded(ulong baseValue, ulong exponent, ulong prime, ulong r)
    {
        ulong blinded = Field.Multiply(baseValue % prime, r % prime, prime);
        ulong inverse = Oracles.SquareAndMultiply(r % prime, prime - 2, prime);          // Fermat inverse
        ulong masked = PowConstantTime(blinded, exponent, prime, out _);
        return Field.Multiply(masked, Oracles.SquareAndMultiply(inverse, exponent, prime), prime);
    }
}

internal static class Oracles
{
    /** Oracle 1: exactly `exponent` multiplications, no squaring at all. */
    internal static ulong RepeatedMultiplication(ulong baseValue, int exponent, ulong modulus)
    {
        ulong result = 1 % modulus;
        ulong factor = baseValue % modulus;
        for (int i = 0; i < exponent; i++) result = result * factor % modulus;
        return result;
    }

    /** Oracle 2: recursive square and multiply, a different algorithm shape from the ladder. */
    internal static ulong SquareAndMultiply(ulong baseValue, ulong exponent, ulong modulus)
    {
        baseValue %= modulus;
        if (exponent == 0) return 1 % modulus;
        ulong half = SquareAndMultiply(baseValue, exponent / 2, modulus);
        ulong squared = half * half % modulus;
        return (exponent & 1UL) == 0 ? squared : squared * baseValue % modulus;
    }
}

public static class Solution
{
    private static readonly ulong[] Primes = { 101UL, 65537UL, 998244353UL };

    public static void Main()
    {
        Correctness();
        TimingModel();
        Blinding();

        Console.WriteLine("PASS");
    }

    private static void Correctness()
    {
        int smallMismatches = 0;
        for (int trial = 0; trial < 300; trial++)
        {
            ulong prime = Primes[trial % Primes.Length];
            ulong baseValue = SplitMix64.Next() % prime;
            int exponent = SplitMix64.NextInt(40);
            ulong expected = Oracles.RepeatedMultiplication(baseValue, exponent, prime);
            int bits;
            if (Exponentiation.PowConstantTime(baseValue, (ulong)exponent, prime, out bits) != expected) smallMismatches++;
            if (Exponentiation.PowLeaky(baseValue, (ulong)exponent, prime) != expected) smallMismatches++;
            if (bits != Field.BitLength((ulong)exponent)) smallMismatches++;
        }
        Check(smallMismatches == 0, "the ladder and the leaky loop both match repeated multiplication on 300 cases");

        int bigMismatches = 0;
        for (int trial = 0; trial < 300; trial++)
        {
            ulong prime = Primes[trial % Primes.Length];
            ulong baseValue = 1 + SplitMix64.Next() % (prime - 1);
            ulong exponent = SplitMix64.Next() % 0x80000000UL;             // up to 31 bits
            ulong expected = Oracles.SquareAndMultiply(baseValue, exponent, prime);
            if (Exponentiation.PowConstantTime(baseValue, exponent, prime, out _) != expected) bigMismatches++;
            if (Exponentiation.PowLeaky(baseValue, exponent, prime) != expected) bigMismatches++;
        }
        Check(bigMismatches == 0, "both implementations match recursive square and multiply on 300 large cases");

        ulong primeValue = 65537;
        int fermat = 0;
        for (int trial = 0; trial < 60; trial++)
        {
            ulong baseValue = 1 + SplitMix64.Next() % (primeValue - 1);
            if (Exponentiation.PowConstantTime(baseValue, primeValue - 1, primeValue, out _) != 1) fermat++;
        }
        Check(fermat == 0, "Fermat's little theorem holds for every exponentiation: a^(p-1) = 1 mod p");

        int homomorphism = 0;
        for (int trial = 0; trial < 60; trial++)
        {
            ulong prime = Primes[trial % Primes.Length];
            ulong baseValue = 1 + SplitMix64.Next() % (prime - 1);
            ulong first = SplitMix64.Next() % 0x40000UL;
            ulong second = SplitMix64.Next() % 0x40000UL;
            ulong combined = Exponentiation.PowConstantTime(baseValue, first + second, prime, out _);
            ulong split = Field.Multiply(
                Exponentiation.PowConstantTime(baseValue, first, prime, out _),
                Exponentiation.PowConstantTime(baseValue, second, prime, out _),
                prime);
            if (combined != split) homomorphism++;
        }
        Check(homomorphism == 0, "a^(e1+e2) = a^e1 * a^e2 mod p on 60 random splits");

        // Boundary values: 0, 1 and a base that is already reduced, plus the degenerate moduli 1 and 2.
        Check(Exponentiation.PowConstantTime(12345, 0, 65537, out int zeroBits) == 1 && zeroBits == 0,
            "e = 0 gives 1 (0^0 = 1 by the loop invariant) with one bit length zero");
        Check(Exponentiation.PowConstantTime(12345, 1, 65537, out _) == 12345, "e = 1 returns the base");
        Check(Exponentiation.PowConstantTime(0, 7, 65537, out _) == 0, "0^7 = 0");
        Check(Exponentiation.PowConstantTime(65537 + 7, 5, 65537, out _)
            == Exponentiation.PowConstantTime(7, 5, 65537, out _), "a base larger than the modulus is reduced");
        Check(Exponentiation.PowConstantTime(5, 3, 1, out _) == 0, "modulus 1 collapses everything to 0");
        Check(Exponentiation.PowConstantTime(7, 5, 2, out _) == 1
            && Exponentiation.PowLeaky(7, 5, 2) == 1, "modulus 2 matches the oracle");
    }

    private static void TimingModel()
    {
        int bits = 31;
        ulong modulus = 998244353;
        var constantCosts = new SortedSet<long>();
        var probes = new List<ulong>();
        for (int trial = 0; trial < 200; trial++)
        {
            ulong exponent = (1UL << (bits - 1)) + SplitMix64.Next() % (1UL << (bits - 1));
            probes.Add(exponent);
        }
        probes.Add(1UL << (bits - 1));                 // a single set bit
        probes.Add((1UL << bits) - 1);                 // every bit set
        probes.Add(0x66666666UL);                      // alternating bits, same bit length
        probes.Add(0x75555555UL);

        ulong baseValue = 1234567 % modulus;
        int leakyMatches = 0;
        foreach (ulong exponent in probes)
        {
            Clock.Reset();
            Exponentiation.PowConstantTime(baseValue, exponent, modulus, out int observedBits);
            constantCosts.Add(Clock.Ops);
            if (observedBits != bits) throw new Exception("FAILED: unexpected bit length " + observedBits);
            Clock.Reset();
            Exponentiation.PowLeaky(baseValue, exponent, modulus);
            if (Clock.Ops == bits + PopCount(exponent)) leakyMatches++;
        }
        Check(constantCosts.Count == 1 && constantCosts.Contains(4L * bits),
            "the ladder costs exactly 4 operations per exponent bit for every exponent: {"
            + string.Join(",", constantCosts) + "}");
        Check(leakyMatches == probes.Count,
            "the leaky loop costs bits + Hamming weight for each of the " + probes.Count + " exponents");
        Check(leakyCostOf(baseValue, modulus, 1UL << (bits - 1)) == bits + 1
            && leakyCostOf(baseValue, modulus, (1UL << bits) - 1) == 2L * bits,
            "the leaky spread is the whole character of the exponent: 1 set bit costs " + (bits + 1)
            + ", 31 set bits cost " + (2L * bits));

        var baseIndependentCosts = new SortedSet<long>();
        foreach (ulong probeBase in new[] { 0UL, 1UL, 2UL, modulus - 1, modulus - 2 })
        {
            Clock.Reset();
            Exponentiation.PowConstantTime(probeBase, 0x7FFFFFFFUL, modulus, out _);
            baseIndependentCosts.Add(Clock.Ops);
        }
        Check(baseIndependentCosts.Count == 1 && baseIndependentCosts.Contains(4L * 31),
            "the cost does not depend on the base value either: {" + string.Join(",", baseIndependentCosts) + "}");

        // The residual leak: the bit length of the exponent is visible in the work performed.
        Clock.Reset();
        Exponentiation.PowConstantTime(baseValue, 0xFFFFUL, modulus, out _);
        long sixteenBitCost = Clock.Ops;
        Clock.Reset();
        Exponentiation.PowConstantTime(baseValue, 0xFFFFFFFFUL, modulus, out _);
        long thirtyTwoBitCost = Clock.Ops;
        Check(sixteenBitCost == 64 && thirtyTwoBitCost == 128 && thirtyTwoBitCost == 2 * sixteenBitCost,
            "cost is linear in the bit length, which is why blinding is needed (" + sixteenBitCost + " vs "
            + thirtyTwoBitCost + ")");

        int swaps = 0;
        ulong left = 11, right = 22;
        Field.ConditionalSwap(1, ref left, ref right);
        if (left == 22 && right == 11) swaps++;
        Field.ConditionalSwap(0, ref left, ref right);
        if (left == 22 && right == 11) swaps++;
        Field.ConditionalSwap(1, ref left, ref right);
        if (left == 11 && right == 22) swaps++;
        Check(swaps == 3, "conditional swap honours both masks and is its own inverse");
    }

    private static void Blinding()
    {
        ulong prime = 65537;
        int mismatches = 0;
        var rawLengths = new SortedSet<int>();
        var blindedLengths = new SortedSet<int>();
        var blindOffsets = new SortedSet<ulong>();
        for (int trial = 0; trial < 60; trial++)
        {
            ulong baseValue = 1 + SplitMix64.Next() % (prime - 1);          // coprime to the prime, as Fermat needs
            ulong exponent = 0x8000UL + SplitMix64.Next() % 0x8000UL;       // every raw exponent is exactly 16 bits
            ulong expected = Exponentiation.PowConstantTime(baseValue, exponent, prime, out _);
            ulong t = 1 + SplitMix64.Next() % 64;
            ulong blinded = Exponentiation.PowExponentBlinded(baseValue, exponent, prime, t, out int bits);
            if (blinded != expected) mismatches++;
            blindOffsets.Add(exponent + t * (prime - 1));
            rawLengths.Add(Field.BitLength(exponent));
            blindedLengths.Add(bits);
        }
        Check(mismatches == 0, "exponent blinding preserves the result on 60 random cases");
        Check(rawLengths.Count == 1 && blindedLengths.Count >= 4,
            "all raw exponents share one bit length while the blinded length spans " + blindedLengths.Count
            + " values, which is exactly the leak blinding removes");
        Check(blindOffsets.Min > 0x10000UL && blindOffsets.Count == 60,
            "blinded exponents are offsets above the raw range (toy values only, no key material)");

        int baseBlindMismatches = 0;
        int changedBase = 0;
        for (int trial = 0; trial < 40; trial++)
        {
            ulong baseValue = 1 + SplitMix64.Next() % (prime - 1);
            ulong exponent = 1 + SplitMix64.Next() % 0x2000UL;
            ulong r = 1 + SplitMix64.Next() % (prime - 1);
            ulong expected = Exponentiation.PowConstantTime(baseValue, exponent, prime, out _);
            if (Exponentiation.PowBaseBlinded(baseValue, exponent, prime, r) != expected) baseBlindMismatches++;
            if (Field.Multiply(baseValue, r % prime, prime) != baseValue) changedBase++;
        }
        Check(baseBlindMismatches == 0, "base blinding preserves the result on 40 random cases");
        Check(changedBase == 40, "base blinding really changes the base that gets exponentiated");

        // With a fixed offset the ladder's flat profile per bit is unchanged: the cost tracks the bit length only.
        var fixedOffsetCosts = new SortedSet<long>();
        for (int trial = 0; trial < 100; trial++)
        {
            ulong exponent = SplitMix64.Next() % 0xFFFFFFFFUL;
            ulong blindedExponent = exponent + (prime - 1);                 // t = 1
            Clock.Reset();
            Exponentiation.PowConstantTime(3, blindedExponent, prime, out int length);
            long cost = Clock.Ops;
            Check(cost == 4L * length, "blinded ladder cost matches its bit length");
            fixedOffsetCosts.Add(cost);
        }
        Check(fixedOffsetCosts.Count >= 3,
            "the blinded cost still varies only with the (now hidden) bit length: "
            + fixedOffsetCosts.Count + " distinct costs");
    }

    private static long leakyCostOf(ulong baseValue, ulong modulus, ulong exponent)
    {
        Clock.Reset();
        Exponentiation.PowLeaky(baseValue, exponent, modulus);
        return Clock.Ops;
    }

    private static int PopCount(ulong value)
    {
        int count = 0;
        while (value > 0)
        {
            count += (int)(value & 1UL);
            value >>= 1;
        }
        return count;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
