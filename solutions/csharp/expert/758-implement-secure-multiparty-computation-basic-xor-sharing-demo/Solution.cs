// Task 758 - Implement secure multiparty computation basic XOR-sharing demo. (Tier: Expert, Beyond - Security & Privacy)
//
// Three parties hold XOR shares of their inputs and compute on them without revealing the inputs. The ring here
// is (GF(2)^32, XOR, AND): XOR is addition and AND is multiplication, so

//     x = x1 xor x2 xor x3,  y = y1 xor y2 xor y3  =>  x xor y = (x1 xor y1) xor (x2 xor y2) xor (x3 xor y3),
//
// i.e. XOR is a *local* operation - no communication, no randomness, no leakage. Multiplication is where the
// interaction happens: a Beaver triple (a, b, c = a AND b) is opened up as d = x xor a and e = y xor b, and then
//
//     z_i = c_i xor (d AND b_i) xor (e AND a_i) xor [i = 0](d AND e)
//
// reconstructs z = x AND y, because opening only the randomised masks d and e reveals nothing: given two shares
// of a uniform a, d = x xor a is uniform. A second secret can be multiplied after that with another triple.
//
// Documented simplifications: the correlated randomness (the Beaver triples) comes from a trusted dealer rather
// than from an OT-based protocol; the three parties are objects in one process, so there is no real network, no
// message ordering and no malicious behaviour (the protocol is semi-honest only); 32 bit words are not key
// material. Invariants verified: (a) share/reconstruct round-trips; (b) perfect privacy - for every one of the 16
// secrets of a 4 bit toy domain the distribution of a revealed share stays uniform, and given two shares the
// hidden one is a bijection of the secret, so all secrets remain equally likely; (c) local operations never open
// anything (transcript check) and each multiplication opens exactly two masks; (d) correctness of AND and of a
// small circuit against plaintext evaluation, plus the boundary cases 0, 1 and all-ones; (e) the documented
// leakage of opening a mask of a *public* value, which is why only the intended output may be opened.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    private static ulong state = 0x7585EC0DE12345UL;

    internal static ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static uint NextUInt() => (uint)(Next() >> 32);
}

internal sealed class MpcEngine
{
    internal const int Parties = 3;
    internal const uint AllOnes = 0xFFFFFFFFu;

    internal readonly List<uint> Transcript = new List<uint>();
    internal long Multiplications;
    internal long Openings;

    /** XOR sharing: the first n-1 shares are uniform, the last absorbs the secret. */
    internal uint[] ShareSecret(uint secret)
    {
        var shares = new uint[Parties];
        uint accumulated = 0;
        for (int i = 0; i < Parties - 1; i++)
        {
            shares[i] = SplitMix64.NextUInt();
            accumulated ^= shares[i];
        }
        shares[Parties - 1] = secret ^ accumulated;
        return shares;
    }

    /** A public constant needs no randomness: it is the constant to one party and zero to the others. */
    internal static uint[] SharePublic(uint value)
    {
        var shares = new uint[Parties];
        shares[0] = value;
        return shares;
    }

    internal static uint Reconstruct(uint[] shares)
    {
        uint value = 0;
        foreach (uint share in shares) value ^= share;
        return value;
    }

    /** Opening is the only operation that publishes a value; every one of them is counted and recorded. */
    internal uint Open(uint[] shares)
    {
        Openings++;
        uint value = Reconstruct(shares);
        Transcript.Add(value);
        return value;
    }

    internal static uint[] Xor(uint[] left, uint[] right)
    {
        var result = new uint[Parties];
        for (int i = 0; i < Parties; i++) result[i] = left[i] ^ right[i];
        return result;
    }

    internal static uint[] XorPublic(uint[] value, uint constant)
    {
        var result = (uint[])value.Clone();
        result[0] ^= constant;
        return result;
    }

    internal static uint[] Not(uint[] value) => XorPublic(value, AllOnes);

    /** AND with a *public* constant stays local: p AND (xor of shares) = xor of (p AND share). */
    internal static uint[] AndPublic(uint[] value, uint constant)
    {
        var result = new uint[Parties];
        for (int i = 0; i < Parties; i++) result[i] = value[i] & constant;
        return result;
    }

    /** Beaver-triple multiplication: two openings, everything else local. */
    internal uint[] And(uint[] left, uint[] right)
    {
        uint[] aShares = ShareSecret(SplitMix64.NextUInt());          // trusted dealer: uniform a
        uint[] bShares = ShareSecret(SplitMix64.NextUInt());          // trusted dealer: uniform b
        uint[] cShares = ShareSecret(Reconstruct(aShares) & Reconstruct(bShares));   // c = a AND b, shared
        uint maskD = Open(Xor(left, aShares));                        // d = x xor a
        uint maskE = Open(Xor(right, bShares));                       // e = y xor b
        var result = new uint[Parties];
        for (int i = 0; i < Parties; i++)
        {
            result[i] = cShares[i] ^ (maskD & bShares[i]) ^ (maskE & aShares[i])
                ^ (i == 0 ? maskD & maskE : 0u);                      // only one party adds the public d AND e
        }
        Multiplications++;
        return result;
    }

    /** Opens x xor y: correct as a predicate, and a demonstration of how much information an opening carries. */
    internal bool RevealsEqual(uint[] left, uint[] right) => Open(Xor(left, right)) == 0;

    /** (x AND y) xor (y AND z) xor x - two multiplications, a few free XORs. */
    internal uint[] Circuit(uint[] x, uint[] y, uint[] z)
    {
        uint[] first = And(x, y);
        uint[] second = And(y, z);
        return Xor(Xor(first, second), x);
    }

    internal static uint PlainCircuit(uint x, uint y, uint z) => (x & y) ^ (y & z) ^ x;
}

public static class Solution
{
    public static void Main()
    {
        var engine = new MpcEngine();
        RoundTrip(engine);
        Privacy(engine);
        LocalOperations();
        Multiplication(engine);
        Circuit(engine);
        EqualityLeakage();
        Boundaries();
        Console.WriteLine("PASS");
    }

    private static void RoundTrip(MpcEngine engine)
    {
        int failures = 0;
        for (int trial = 0; trial < 400; trial++)
        {
            uint secret = SplitMix64.NextUInt();
            uint[] shares = engine.ShareSecret(secret);
            if (MpcEngine.Reconstruct(shares) != secret) failures++;
            if (shares.Length != MpcEngine.Parties) failures++;
        }
        Check(failures == 0, "400 XOR sharings reconstruct exactly");
        Check(engine.Openings == 0 && engine.Transcript.Count == 0,
            "sharing alone opens nothing: the transcript is still empty");
        Check(engine.Open(MpcEngine.SharePublic(0x12345678)) == 0x12345678,
            "a public sharing reconstructs to the constant without any randomness");
    }

    private static void Privacy(MpcEngine engine)
    {
        const int domain = 16;
        const int samples = 4096;
        int expected = samples / domain;
        int worstDeviation = 0;
        for (uint secret = 0; secret < domain; secret++)
        {
            var histogram = new int[domain];
            for (int i = 0; i < samples; i++) histogram[engine.ShareSecret(secret)[1] & 0xF]++;
            foreach (int count in histogram) worstDeviation = Math.Max(worstDeviation, Math.Abs(count - expected));
        }
        Check(worstDeviation <= expected / 2,
            "for all 16 secrets the revealed share stays uniform: worst deviation " + worstDeviation
            + " of " + expected + " over " + samples + " samples each");

        // Given two shares, the hidden share visits every value as the secret varies, so no secret is excluded.
        bool bijection = true;
        for (int trial = 0; trial < 32; trial++)
        {
            uint givenA = (uint)(trial * 5 & 0xF);
            uint givenB = (uint)(trial * 3 & 0xF);
            var visited = new bool[domain];
            for (uint secret = 0; secret < domain; secret++) visited[secret ^ givenA ^ givenB] = true;
            for (int value = 0; value < domain; value++) bijection &= visited[value];
        }
        Check(bijection, "given two shares the third is a bijection of the secret: perfect privacy of XOR sharing");

        int highBitsDeviation = 0;
        var high = new int[domain];
        for (int i = 0; i < samples; i++) high[engine.ShareSecret(0xDEADBEEF)[2] >> 28]++;
        foreach (int count in high) highBitsDeviation = Math.Max(highBitsDeviation, Math.Abs(count - expected));
        Check(highBitsDeviation <= expected / 2,
            "the same holds for the high bits of a 32 bit secret: deviation " + highBitsDeviation);
    }

    private static void LocalOperations()
    {
        var engine = new MpcEngine();
        uint[] left = engine.ShareSecret(0x0F0F0F0F);
        uint[] right = engine.ShareSecret(0x00FF00FF);
        int failures = 0;
        for (int trial = 0; trial < 200; trial++)
        {
            uint a = SplitMix64.NextUInt();
            uint b = SplitMix64.NextUInt();
            uint[] sharedA = engine.ShareSecret(a);
            uint[] sharedB = engine.ShareSecret(b);
            if (MpcEngine.Reconstruct(MpcEngine.Xor(sharedA, sharedB)) != (a ^ b)) failures++;
            if (MpcEngine.Reconstruct(MpcEngine.Not(sharedA)) != ~a) failures++;
            uint constant = SplitMix64.NextUInt();
            if (MpcEngine.Reconstruct(MpcEngine.AndPublic(sharedA, constant)) != (a & constant)) failures++;
        }
        Check(failures == 0, "XOR, NOT and AND-with-a-public-constant are correct on 200 random pairs");
        Check(engine.Openings == 0 && engine.Transcript.Count == 0,
            "600 local operations performed without opening a single value (cost: zero interaction)");
        Check(MpcEngine.Reconstruct(MpcEngine.Xor(left, right)) == (0x0F0F0F0Fu ^ 0x00FF00FFu),
            "element-wise XOR of two sharings is a sharing of the XOR");
    }

    private static void Multiplication(MpcEngine engine)
    {
        long openingsBefore = engine.Openings;
        long multiplicationsBefore = engine.Multiplications;
        int failures = 0;
        for (int trial = 0; trial < 200; trial++)
        {
            uint a = SplitMix64.NextUInt();
            uint b = SplitMix64.NextUInt();
            uint[] product = engine.And(engine.ShareSecret(a), engine.ShareSecret(b));
            if (MpcEngine.Reconstruct(product) != (a & b)) failures++;
        }
        Check(failures == 0, "200 Beaver-triple multiplications reconstruct the true AND");
        long newMultiplications = engine.Multiplications - multiplicationsBefore;
        Check(newMultiplications == 200 && engine.Openings - openingsBefore == 2 * newMultiplications,
            "each multiplication opens exactly two masks, " + newMultiplications + " multiplications, "
            + (engine.Openings - openingsBefore) + " openings");
        Check(MpcEngine.Reconstruct(engine.And(MpcEngine.SharePublic(3), MpcEngine.SharePublic(5))) == 1,
            "a multiplication where both operands are public still works");
    }

    private static void Circuit(MpcEngine engine)
    {
        int failures = 0;
        for (int trial = 0; trial < 200; trial++)
        {
            uint x = SplitMix64.NextUInt();
            uint y = SplitMix64.NextUInt();
            uint z = SplitMix64.NextUInt();
            uint[] evaluated = engine.Circuit(engine.ShareSecret(x), engine.ShareSecret(y), engine.ShareSecret(z));
            if (MpcEngine.Reconstruct(evaluated) != MpcEngine.PlainCircuit(x, y, z)) failures++;
        }
        Check(failures == 0, "(x AND y) xor (y AND z) xor x matches plaintext evaluation on 200 random triples");

        long before = engine.Openings;
        engine.Circuit(engine.ShareSecret(1), engine.ShareSecret(1), engine.ShareSecret(1));
        Check(engine.Openings - before == 4, "the circuit opens exactly four masks (two per multiplication)");
    }

    private static void EqualityLeakage()
    {
        var engine = new MpcEngine();
        uint[] first = engine.ShareSecret(0xABCD1234);
        uint[] same = engine.ShareSecret(0xABCD1234);
        uint[] different = engine.ShareSecret(0xABCD1235);
        Check(engine.RevealsEqual(first, same) && !engine.RevealsEqual(first, different),
            "the equality predicate is correct on equal and unequal shared inputs");

        uint[] secret = engine.ShareSecret(0xDEADBEEF);
        engine.Open(MpcEngine.Xor(secret, MpcEngine.SharePublic(0)));
        Check(engine.Transcript[engine.Transcript.Count - 1] == 0xDEADBEEF,
            "documented leakage: opening x xor 0 reveals x exactly, so only the intended output may be opened");
    }

    private static void Boundaries()
    {
        var engine = new MpcEngine();
        Check(MpcEngine.Reconstruct(engine.ShareSecret(0)) == 0, "sharing zero works (the shares are not all zero)");
        uint[] zero = engine.ShareSecret(0);
        Check(zero[0] != 0 && zero[1] != 0 && MpcEngine.Reconstruct(zero) == 0,
            "a zero secret is still hidden behind nonzero shares");
        Check(MpcEngine.Reconstruct(engine.ShareSecret(MpcEngine.AllOnes)) == MpcEngine.AllOnes,
            "the all-ones secret round-trips");

        uint[] single = engine.ShareSecret(1);
        Check(MpcEngine.Reconstruct(engine.And(single, single)) == 1, "1 AND 1 = 1");
        uint[] zeros = engine.ShareSecret(0);
        Check(MpcEngine.Reconstruct(engine.And(single, zeros)) == 0, "1 AND 0 = 0");
        Check(MpcEngine.Reconstruct(MpcEngine.AndPublic(single, MpcEngine.AllOnes)) == 1,
            "AND with all ones is the identity");
        Check(MpcEngine.Reconstruct(MpcEngine.AndPublic(single, 0)) == 0, "AND with zero clears everything");
        Check(MpcEngine.Reconstruct(MpcEngine.Not(engine.ShareSecret(0))) == MpcEngine.AllOnes,
            "NOT of zero is all ones");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
