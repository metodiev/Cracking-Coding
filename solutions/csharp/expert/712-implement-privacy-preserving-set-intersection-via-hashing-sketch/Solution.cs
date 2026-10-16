// Task 712 - Implement privacy-preserving set intersection via hashing sketch.
// (Tier: Expert, CTCI Beyond — Security & Privacy)
//
// Two protocols for two-party private set intersection, over sets A (Alice) and B (Bob).
//
// 1. Double-blinded hashing (the Diffie-Hellman style PSI of Meadows). Pick a safe prime p = 2q + 1
//    and hash every element into the multiplicative group, H(x) in [1, p-1]. Alice blinds her set
//    with a secret exponent a, Bob raises the received values to his own secret b and sends the
//    result back shuffled; Bob also sends his set blinded with b. Alice raises those to a and now
//    holds {H(y)^{ab}} for y in B, which she compares with the shuffled {H(x)^{ab}} she got back.
//    The sets match exactly on A n B, and the transcript never contains a raw element: the
//    security rests on the group's one-wayness, not on the set sizes.
//
// 2. A hashing sketch (Bloom filter). Both sides hash their elements into a bit vector of the same
//    geometry; the intersection estimate is the set of Bob's elements whose bits are all set in
//    Alice's filter. It is approximate (false positives, never false negatives), one round, and it
//    leaks only membership bits, which is why the file measures its false positive rate against the
//    (1 - e^{-kn/m})^k prediction instead of pretending it is exact.
//
// Verified below: the prime really is a safe prime, the group laws hold (commutation, the exponent
// inverse, Fermat), H is injective on the test universe, the exact protocol reproduces A n B on
// random, empty, disjoint, nested and single-element inputs, the transcript contains no raw hash,
// the shuffled reply really is permuted, the sketch protocol has no false negatives and a false
// positive rate close to the theoretical value, and malformed exponents or a non-invertible hash
// value are rejected.
using System;
using System.Collections.Generic;
using System.Numerics;

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

internal static class PrimeMath
{
    internal static bool IsPrime(long value)
    {
        if (value < 2) return false;
        if (value % 2 == 0) return value == 2;
        for (long d = 3; d * d <= value; d += 2) if (value % d == 0) return false;
        return true;
    }

    /// <summary>Smallest safe prime p = 2q + 1 above the given lower bound (both p and q prime).</summary>
    internal static long SafePrime(long lowerBound)
    {
        for (long q = lowerBound / 2 | 1; ; q += 2)
        {
            if (!IsPrime(q)) continue;
            long p = 2 * q + 1;
            if (IsPrime(p)) return p;
        }
    }

    internal static long Inverse(long value, long modulus)
    {
        long a = ((value % modulus) + modulus) % modulus;
        if (a == 0) throw new ArgumentException("zero has no inverse");
        long t = 0, newT = 1, r = modulus, newR = a;
        while (newR != 0)
        {
            long quotient = r / newR;
            (t, newT) = (newT, t - quotient * newT);
            (r, newR) = (newR, r - quotient * newR);
        }
        if (r != 1) throw new ArgumentException("the value is not invertible");
        return ((t % modulus) + modulus) % modulus;
    }

    internal static long Gcd(long a, long b)
    {
        while (b != 0) { (a, b) = (b, a % b); }
        return Math.Abs(a);
    }
}

internal sealed class HashingGroup
{
    internal long Prime { get; }
    private readonly ulong _salt;

    internal HashingGroup(long lowerBound, ulong salt)
    {
        Prime = PrimeMath.SafePrime(lowerBound);
        _salt = salt;
    }

    /// <summary>Deterministic hash of an element into the multiplicative group [1, p-1].</summary>
    internal long Hash(long element)
    {
        ulong state = (ulong)element * 0x9E3779B97F4A7C15UL ^ _salt;
        ulong z = Rng.Next(ref state) ^ Rng.Next(ref state);
        return 1 + (long)(z % (ulong)(Prime - 1));
    }

    internal long Blind(long value, long exponent) => (long)BigInteger.ModPow(value, exponent, Prime);

    /// <summary>Validates that an exponent is invertible modulo p-1, as the protocol requires.</summary>
    internal long CheckExponent(long exponent)
    {
        if (exponent <= 1 || exponent >= Prime - 1)
            throw new ArgumentOutOfRangeException(nameof(exponent), "exponents live in [2, p-2]");
        if (PrimeMath.Gcd(exponent, Prime - 1) != 1)
            throw new ArgumentException("the exponent must be coprime with p-1 to be invertible");
        return exponent;
    }
}

internal static class PrivateIntersection
{
    internal sealed class Transcript
    {
        internal List<long> AliceToBob = new List<long>();
        internal List<long> BobReplyBlinded = new List<long>();      // {H(x)^{ab}}, shuffled
        internal List<long> BobValuesBlinded = new List<long>();     // {H(y)^b}
        internal bool ReplyWasPermuted;
    }

    /// <summary>Double-blinded hashing PSI; returns A n B (in Bob's element order, de-duplicated).</summary>
    internal static List<long> DiffeHellmanIntersection(HashingGroup group, IReadOnlyList<long> alice,
        IReadOnlyList<long> bob, long a, long b, ulong shuffleSeed, out Transcript transcript)
    {
        group.CheckExponent(a);
        group.CheckExponent(b);
        transcript = new Transcript();

        // Round 1: Alice blinds each of her elements with a.
        foreach (long x in alice) transcript.AliceToBob.Add(group.Blind(group.Hash(x), a));
        // Round 2: Bob doubles the blinded values and blinds his own set with b.
        var doubledByBob = new List<long>();
        foreach (long v in transcript.AliceToBob) doubledByBob.Add(group.Blind(v, b));
        foreach (long y in bob) transcript.BobValuesBlinded.Add(group.Blind(group.Hash(y), b));

        // Bob shuffles his reply so Alice cannot line it up with the order she sent.
        var order = new List<int>();
        for (int i = 0; i < doubledByBob.Count; i++) order.Add(i);
        ulong state = shuffleSeed;
        for (int i = order.Count - 1; i > 0; i--)
        {
            int j = Rng.NextInt(ref state, i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        foreach (int i in order) transcript.BobReplyBlinded.Add(doubledByBob[i]);
        transcript.ReplyWasPermuted = !SameOrder(doubledByBob, transcript.BobReplyBlinded);

        // Alice doubles Bob's blinded values; the two doubly blinded collections then agree exactly
        // on A n B, and she can only identify elements she already owns.
        var reply = new HashSet<long>(transcript.BobReplyBlinded);
        var aliceSet = new HashSet<long>(alice);
        var result = new List<long>();
        var emitted = new HashSet<long>();
        for (int i = 0; i < bob.Count; i++)
        {
            long doubled = group.Blind(transcript.BobValuesBlinded[i], a);      // H(y)^{ab}
            if (aliceSet.Contains(bob[i]) && reply.Contains(doubled) && emitted.Add(bob[i])) result.Add(bob[i]);
        }
        return result;
    }

    private static bool SameOrder(List<long> a, List<long> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>Bloom-filter based approximate intersection: Bob tests his elements against Alice's sketch.</summary>
    internal static (List<long> matches, HashSet<int> bits) SketchIntersection(IReadOnlyList<long> alice,
        IReadOnlyList<long> bob, int width, int hashes)
    {
        if (width <= 0 || hashes <= 0) throw new ArgumentOutOfRangeException(nameof(width), "the sketch needs positive geometry");
        var bits = new HashSet<int>();
        foreach (long x in alice)
        {
            foreach (int index in Sketch.Intervals(x, width, hashes)) bits.Add(index);
        }
        var matches = new List<long>();
        foreach (long y in bob)
        {
            bool present = true;
            foreach (int index in Sketch.Intervals(y, width, hashes)) if (!bits.Contains(index)) present = false;
            if (present) matches.Add(y);
        }
        return (matches, bits);
    }
}

internal static class Sketch
{
    /// <summary>Independent hash indices, one full splitmix64 round per position.</summary>
    internal static IEnumerable<int> Intervals(long element, int width, int hashes)
    {
        for (int i = 0; i < hashes; i++)
        {
            ulong state = (ulong)element * 0x9E3779B97F4A7C15UL + (ulong)i * 0xD1B54A32D192ED03UL + 0x1234567UL;
            ulong z = Rng.Next(ref state);
            yield return (int)(z % (ulong)width);
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static List<long> Sorted(IEnumerable<long> values)
    {
        var copy = new List<long>(values);
        copy.Sort();
        return copy;
    }

    private static bool SameSet(IEnumerable<long> a, IEnumerable<long> b)
    {
        List<long> left = Sorted(a), right = Sorted(b);
        if (left.Count != right.Count) return false;
        for (int i = 0; i < left.Count; i++) if (left[i] != right[i]) return false;
        return true;
    }

    public static void Main()
    {
        // ---- the group -------------------------------------------------------------------------
        {
            long prime = PrimeMath.SafePrime(500_000_000L);
            Check(PrimeMath.IsPrime(prime), "the generated modulus is prime");
            Check(PrimeMath.IsPrime((prime - 1) / 2), "and so is (p-1)/2, so p is a safe prime");
            var group = new HashingGroup(500_000_000L, 0x5EEDUL);
            Check(group.Prime == prime, "the group uses that safe prime");

            long a = 123456789L, b = 98765431L;
            while (PrimeMath.Gcd(a, prime - 1) != 1) a++;
            while (PrimeMath.Gcd(b, prime - 1) != 1) b++;
            group.CheckExponent(a);
            group.CheckExponent(b);

            bool inRange = true, commutes = true, fermat = true, injective = true;
            var hashes = new Dictionary<long, long>();
            for (long x = 1; x <= 200; x++)
            {
                long h = group.Hash(x);
                if (h < 1 || h >= prime) inRange = false;
                if (hashes.ContainsKey(h) && hashes[h] != x) injective = false;
                hashes[h] = x;
                if (group.Blind(group.Blind(h, a), b) != group.Blind(group.Blind(h, b), a)) commutes = false;
                if (group.Blind(h, prime - 1) != 1) fermat = false;               // Fermat's little theorem
            }
            Check(inRange, "every hash lands in [1, p-1]");
            Check(injective, "the hash is injective on the test universe");
            Check(commutes, "blinding commutes: (H^a)^b = (H^b)^a");
            Check(fermat, "H^{p-1} = 1, as Fermat's little theorem requires");

            long inverse = PrimeMath.Inverse(a, prime - 1);
            Check(a * inverse % (prime - 1) == 1, "the exponent inverse is correct modulo p-1");
            Check(group.Blind(group.Blind(group.Hash(42), a), inverse) == group.Hash(42),
                "blinding with a then a^-1 restores the hash");
            Console.WriteLine("safePrime=" + prime + " universeHashRange=[1," + (prime - 1) + "]");

            bool thrown = false;
            try { group.CheckExponent(prime - 1); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "an out-of-range exponent is rejected");
            thrown = false;
            try { group.CheckExponent((prime - 1) / 2); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "an exponent sharing a factor with p-1 is rejected");
            thrown = false;
            try { PrimeMath.Inverse(0, prime); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "zero has no inverse");
        }

        // ---- the exact protocol on random sets -------------------------------------------------
        {
            var group = new HashingGroup(500_000_000L, 0xABCDEFUL);
            ulong state = 0x1234UL;
            const int Rounds = 25;
            bool everyRoundExact = true;
            string firstFailure = "none";
            for (int round = 0; round < Rounds; round++)
            {
                var alice = new List<long>();
                var bob = new List<long>();
                for (int i = 0; i < 30; i++) alice.Add(Rng.NextInt(ref state, 400));
                for (int i = 0; i < 30; i++) bob.Add(Rng.NextInt(ref state, 400));
                var expected = new List<long>();
                foreach (long x in alice) if (bob.Contains(x) && !expected.Contains(x)) expected.Add(x);

                long a = 3 + Rng.NextInt(ref state, 1000), b = 5 + Rng.NextInt(ref state, 1000);
                while (PrimeMath.Gcd(a, group.Prime - 1) != 1) a++;
                while (PrimeMath.Gcd(b, group.Prime - 1) != 1) b++;
                var got = PrivateIntersection.DiffeHellmanIntersection(group, alice, bob, a, b,
                    0xDEAD0000UL + (ulong)round, out _);
                if (!SameSet(got, expected))
                {
                    everyRoundExact = false;
                    if (firstFailure == "none") firstFailure = "round " + round + ": expected " + expected.Count
                        + " matches, got " + got.Count;
                }
            }
            Check(everyRoundExact, "25 random rounds reproduce A n B exactly (" + firstFailure + ")");
            Console.WriteLine("psiRounds=" + Rounds);
        }

        // ---- the transcript leaks no raw element ----------------------------------------------
        {
            var group = new HashingGroup(500_000_000L, 0xF00DUL);
            var alice = new List<long> { 11, 22, 33, 44, 55 };
            var bob = new List<long> { 22, 44, 66, 88 };
            var intersection = PrivateIntersection.DiffeHellmanIntersection(group, alice, bob, 7, 11, 0x5150UL,
                out var transcript);
            Check(SameSet(intersection, new List<long> { 22, 44 }), "the tiny fixture intersects in {22, 44}");
            Check(transcript.ReplyWasPermuted, "Bob really shuffles the reply");

            var raw = new HashSet<long>();
            for (long x = 1; x <= 200; x++) raw.Add(group.Hash(x));
            bool noRawValue = true;
            foreach (long v in transcript.AliceToBob)
                if (v != 1 && v != group.Prime - 1 && raw.Contains(v)) noRawValue = false;
            foreach (long v in transcript.BobReplyBlinded)
                if (v != 1 && v != group.Prime - 1 && raw.Contains(v)) noRawValue = false;
            foreach (long v in transcript.BobValuesBlinded)
                if (v != 1 && v != group.Prime - 1 && raw.Contains(v)) noRawValue = false;
            Check(noRawValue, "no non-trivial transcript entry equals the unblinded hash of any universe element");
            Check(transcript.AliceToBob.Count == alice.Count && transcript.BobValuesBlinded.Count == bob.Count,
                "the transcript has exactly the expected number of protocol values");
            Console.WriteLine("transcriptEntries=" + (transcript.AliceToBob.Count + transcript.BobReplyBlinded.Count
                + transcript.BobValuesBlinded.Count));
        }

        // ---- boundary shapes -------------------------------------------------------------------
        {
            var group = new HashingGroup(500_000_000L, 0x99UL);
            var none = new List<long>();
            var some = new List<long> { 1, 2, 3 };
            Check(SameSet(PrivateIntersection.DiffeHellmanIntersection(group, none, some, 3, 5, 1UL, out _), none),
                "an empty set intersects in nothing");
            Check(SameSet(PrivateIntersection.DiffeHellmanIntersection(group, some, none, 3, 5, 1UL, out _), none),
                "and so does an empty second set");
            var disjointLeft = new List<long> { 1, 3, 5 };
            var disjointRight = new List<long> { 2, 4, 6 };
            Check(PrivateIntersection.DiffeHellmanIntersection(group, disjointLeft, disjointRight, 3, 5, 1UL, out _).Count == 0,
                "disjoint sets intersect in nothing");
            var identical = new List<long> { 7, 8, 9 };
            Check(SameSet(PrivateIntersection.DiffeHellmanIntersection(group, identical, identical, 3, 5, 1UL, out _), identical),
                "identical sets intersect in themselves");
            var nested = new List<long> { 7, 8 };
            Check(SameSet(PrivateIntersection.DiffeHellmanIntersection(group, identical, nested, 3, 5, 1UL, out _), nested),
                "a nested set intersects in the smaller set");
            Check(SameSet(PrivateIntersection.DiffeHellmanIntersection(group, new List<long> { 5 }, new List<long> { 5 },
                3, 5, 1UL, out _), new List<long> { 5 }), "a single shared element is found");
            var duplicated = new List<long> { 5, 5, 5 };
            Check(PrivateIntersection.DiffeHellmanIntersection(group, duplicated, new List<long> { 5 }, 3, 5, 1UL, out _).Count == 1,
                "duplicates are reported once");
        }

        // ---- the hashing sketch protocol -------------------------------------------------------
        {
            const int Width = 1024, Hashes = 4, Universe = 2000;
            ulong state = 0xBEEFUL;
            var alice = new List<long>();
            var bob = new List<long>();
            for (int i = 0; i < 300; i++) alice.Add(Rng.NextInt(ref state, Universe));
            for (int i = 0; i < 300; i++) bob.Add(Rng.NextInt(ref state, Universe));

            (List<long> matches, HashSet<int> bits) = PrivateIntersection.SketchIntersection(alice, bob, Width, Hashes);
            bool noFalseNegatives = true;
            var trueIntersection = new HashSet<long>();
            foreach (long x in alice) if (bob.Contains(x)) trueIntersection.Add(x);
            foreach (long y in trueIntersection) if (!matches.Contains(y)) noFalseNegatives = false;
            Check(noFalseNegatives, "the sketch protocol never misses a true intersection element");

            var falsePositives = new HashSet<long>();
            foreach (long y in matches) if (!trueIntersection.Contains(y)) falsePositives.Add(y);
            var testedDistinct = new HashSet<long>(bob);
            int nonIntersecting = 0;
            foreach (long y in testedDistinct) if (!trueIntersection.Contains(y)) nonIntersecting++;
            double observedRate = nonIntersecting == 0 ? 0 : (double)falsePositives.Count / nonIntersecting;
            double predicted = Math.Pow(1 - Math.Exp(-(double)Hashes * alice.Count / Width), Hashes);
            Check(matches.Count >= 40, "the sketch returns enough candidates to judge its accuracy (" + matches.Count + ")");
            Check(observedRate <= predicted * 1.35 + 0.02 && observedRate >= predicted * 0.65 - 0.02,
                "the false positive rate stays near the prediction over " + nonIntersecting
                + " non-shared elements (observed " + observedRate.ToString("F4")
                + ", predicted " + predicted.ToString("F4") + ")");

            double bitDensity = (double)bits.Count / Width;
            double predictedDensity = 1 - Math.Exp(-(double)Hashes * alice.Count / Width);
            Check(Math.Abs(bitDensity - predictedDensity) < 0.05,
                "the sketch density matches 1 - e^{-kn/m} (" + bitDensity.ToString("F4") + ")");

            (List<long> empty, _) = PrivateIntersection.SketchIntersection(new List<long>(), bob, Width, Hashes);
            Check(empty.Count == 0, "an empty sketch matches nothing");
            (List<long> self, _) = PrivateIntersection.SketchIntersection(alice, alice, Width, Hashes);
            Check(self.Count >= alice.Count, "the sketch matches every element of its own set");
            bool thrown = false;
            try { PrivateIntersection.SketchIntersection(alice, bob, 0, Hashes); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a zero-width sketch is rejected instead of dividing by zero");
            thrown = false;
            try { PrivateIntersection.SketchIntersection(alice, bob, Width, 0); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a sketch without hash functions is rejected");
            Console.WriteLine("sketchWidth=" + Width + " hashes=" + Hashes + " matches=" + matches.Count
                + " falsePositiveRate=" + observedRate.ToString("F4") + " predicted=" + predicted.ToString("F4"));
        }

        Console.WriteLine("PASS");
    }
}
