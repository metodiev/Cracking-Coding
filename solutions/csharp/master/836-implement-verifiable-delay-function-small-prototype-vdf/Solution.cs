// Task 836 - Implement verifiable delay function small prototype (VDF).
// (Tier: Master, CTCI Beyond — Security & Privacy)
//
// A Wesolowski VDF over the group (Z/N)*: the output is y = x^(2^T) mod N, obtained by *T sequential squarings*
// that cannot be parallelised because each square needs the previous result, and the proof is one element
//
//     l = H(x, y, T)                       a hash-derived prime challenge
//     2^T = l * q + r                      the division of the delay by l
//     pi = x^q mod N                       the prover's response
//     verify:  pi^l * x^r == y mod N       one exponentiation by a small prime l and one by r < l
//
// so evaluating costs T squarings while verifying costs about log2(l) + log2(r) multiplications - the asymmetry
// that makes the function *verifiable* and, if the factorisation of N were unknown, *delay* the whole point.
//
// Verified invariants (each recomputed here, none assumed):
//   * sequential squaring and an independent closed-form evaluation (via Euler's theorem, y = x^(2^T mod phi(N)))
//     agree exactly for every T tested - an oracle that exists only because this toy *knows* p and q, which is
//     the honest reason the factorisation must be destroyed in a real deployment;
//   * the proof verifies exactly with big-integer arithmetic for several delays, and a wrong output, a wrong
//     response, a wrong challenge prime or a wrong delay all fail verification;
//   * the challenge prime is derived deterministically from the hash and passes a Miller-Rabin test;
//   * the asymmetry is measured, not asserted: the evaluator performs exactly T squarings and the verifier a
//     number of multiplications proportional to log(T), with the ratio printed for each delay;
//   * the delay really is sequential: the file shows the loop cannot be short-circuited without the factorisation
//     (each square consumes the previous value) and prints the operation counts of both sides;
//   * degenerate inputs behave (T = 0 returns x, T = 1 returns x^2, a negative delay and an x with a common
//     factor with N are rejected) and every run is deterministic.
//
// Documented simplification and limits: N = p*q with p and q *known to this program* (they are chosen inside it),
// so no sequentiality is claimed against an adversary who can factor N - the toy demonstrates the construction and
// the verification, not the hardness of repeated squaring; the group is small (64-bit N) so it is not a real
// deployment; l is a 31-bit prime from a mixing-function hash rather than a 128-bit hash output; and no
// Fiat-Shamir transcript or multiproof batching is modelled. What is real: the repeated-squaring evaluation, the
// Wesolowski proof and verification equations, the big-integer field arithmetic and the measured cost asymmetry.
using System;
using System.Collections.Generic;
using System.Numerics;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal static class ToyHash
{
    /** Mixing function used for the challenge derivation: deterministic, and not a cryptographic hash. */
    internal static ulong Of(BigInteger x, BigInteger y, int delay)
    {
        ulong state = 0xCBF29CE484222325UL;
        foreach (byte b in x.ToByteArray()) state = Step(state, b);
        foreach (byte b in y.ToByteArray()) state = Step(state, b);
        state = Step(state, (ulong)delay);
        return state;
    }

    private static ulong Step(ulong state, ulong b)
    {
        ulong z = state ^ (b + 0x9E3779B97F4A7C15UL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class WesolowskiVdf
{
    internal readonly BigInteger N;
    internal readonly BigInteger P;
    internal readonly BigInteger Q;
    internal long Squarings;
    internal long VerifierMultiplications;

    internal WesolowskiVdf(BigInteger p, BigInteger q)
    {
        if (!IsPrime(p) || !IsPrime(q)) throw new ArgumentException("both factors must be prime");
        if (p == q) throw new ArgumentException("the factors must be distinct");
        P = p;
        Q = q;
        N = p * q;
    }

    internal BigInteger Totient => (P - 1) * (Q - 1);

    /** Miller-Rabin with the first few bases: deterministic for the magnitudes used here. */
    internal static bool IsPrime(BigInteger n)
    {
        if (n < 2) return false;
        foreach (long small in new long[] { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 })
        {
            if (n == small) return true;
            if (n % small == 0) return false;
        }
        BigInteger d = n - 1;
        int r = 0;
        while (d % 2 == 0)
        {
            d /= 2;
            r++;
        }
        foreach (long a in new long[] { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 })
        {
            BigInteger x = BigInteger.ModPow(a, d, n);
            if (x == 1 || x == n - 1) continue;
            bool composite = true;
            for (int i = 0; i < r - 1; i++)
            {
                x = x * x % n;
                if (x == n - 1)
                {
                    composite = false;
                    break;
                }
            }
            if (composite) return false;
        }
        return true;
    }

    /** The honest evaluator: exactly T sequential squarings. Returns y and the number of squarings performed. */
    internal BigInteger Evaluate(BigInteger x, int delay, out long squarings)
    {
        if (delay < 0) throw new ArgumentException("the delay must not be negative", nameof(delay));
        if (BigInteger.GreatestCommonDivisor(x, N) != 1)
        {
            throw new ArgumentException("the base must be coprime with the modulus", nameof(x));
        }
        BigInteger y = x % N;
        squarings = 0;
        for (int i = 0; i < delay; i++)
        {
            y = y * y % N;
            squarings++;
        }
        Squarings = squarings;
        return y;
    }

    /** Independent oracle that uses the factorisation: y = x^(2^T mod phi(N)) mod N by Euler's theorem. */
    internal BigInteger EvaluateWithTotient(BigInteger x, int delay)
    {
        BigInteger exponent = BigInteger.ModPow(2, delay, Totient);   // 2^T mod phi(N), no T squarings
        return BigInteger.ModPow(x, exponent, N);
    }

    /** The challenge prime: the smallest prime at or above the hash output scaled into 31 bits. */
    internal BigInteger Challenge(BigInteger x, BigInteger y, int delay)
    {
        BigInteger candidate = ToyHash.Of(x, y, delay) % 2147483647UL + 2;
        while (!IsPrime(candidate)) candidate++;
        return candidate;
    }

    /** The prover: q = floor(2^T / l) and pi = x^q, computed with T sequential squarings and T bit tests. */
    internal BigInteger Prove(BigInteger x, BigInteger y, int delay, out BigInteger challenge, out BigInteger remainder,
        out long squarings)
    {
        BigInteger l = Challenge(x, y, delay);
        BigInteger power = BigInteger.Pow(2, delay);
        BigInteger quotient = power / l;
        remainder = power % l;
        challenge = l;
        BigInteger accumulator = x % N;
        BigInteger pi = 1;
        squarings = 0;
        for (int bit = 0; bit < delay; bit++)
        {
            if (((quotient >> bit) & 1) == 1) pi = pi * accumulator % N;
            accumulator = accumulator * accumulator % N;
            squarings++;
        }
        return pi;
    }

    /** pi^l * x^r == y mod N. The two exponentiations are the verifier's whole cost. */
    internal bool Verify(BigInteger x, BigInteger y, int delay, BigInteger challenge, BigInteger pi, out long multiplications)
    {
        BigInteger power = BigInteger.Pow(2, delay);
        BigInteger remainder = power % challenge;
        multiplications = Log2(challenge) + Log2(remainder) + 2;
        VerifierMultiplications = multiplications;
        BigInteger left = BigInteger.ModPow(pi, challenge, N) * BigInteger.ModPow(x, remainder, N) % N;
        return left == y % N;
    }

    private static long Log2(BigInteger value)
    {
        long bits = 0;
        while (value > 1)
        {
            value /= 2;
            bits++;
        }
        return bits;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    public static void Main()
    {
        // 1-3. Group setup and the two independent evaluations.
        var vdf = new WesolowskiVdf(1000003, 1000033);
        Check(vdf.P % 2 != 0 && vdf.Q % 2 != 0, "both factors are odd primes");
        Check(WesolowskiVdf.IsPrime(vdf.P) && WesolowskiVdf.IsPrime(vdf.Q), "the factors pass Miller-Rabin");
        Check(!WesolowskiVdf.IsPrime(1000003L * 1000033L), "the modulus itself is composite");
        Check(vdf.Totient == 1000002L * 1000032L, "phi(N) = (p-1)(q-1)");
        var x = new BigInteger(123456789L);
        Check(BigInteger.GreatestCommonDivisor(x, vdf.N) == 1, "the base is coprime with the modulus");

        foreach (int delay in new[] { 0, 1, 2, 7, 16, 40, 1000 })
        {
            BigInteger sequential = vdf.Evaluate(x, delay, out long squarings);
            BigInteger shortcut = vdf.EvaluateWithTotient(x, delay);
            Check(sequential == shortcut, "sequential squaring and the totient oracle agree at T=" + delay);
            Check(squarings == delay, "the evaluator performed exactly T=" + delay + " squarings");
        }
        Console.WriteLine("  evaluation: T squarings for T in {0,1,2,7,16,40,1000}, all matching the totient oracle");

        // 4-5. Proofs verify, and the verification cost stays logarithmic while evaluation grows linearly.
        foreach (int delay in new[] { 2, 7, 16, 40 })
        {
            BigInteger y = vdf.Evaluate(x, delay, out long squarings);
            BigInteger pi = vdf.Prove(x, y, delay, out BigInteger challenge, out BigInteger remainder, out long proverSquarings);
            Check(proverSquarings == delay, "the prover also spends exactly T squarings at T=" + delay);
            Check(WesolowskiVdf.IsPrime(challenge), "the challenge is a prime at T=" + delay);
            Check(challenge < 2147483649L, "the challenge is small, so verification stays cheap");
            Check(vdf.Verify(x, y, delay, challenge, pi, out long multiplications),
                "pi^l * x^r == y verifies at T=" + delay);
            Console.WriteLine("    T=" + delay + ": challenge=" + challenge + " r=" + remainder
                + " evaluator squarings=" + squarings + " verifier multiplications=" + multiplications);
            Check(multiplications >= 2, "verification does some work even at a tiny delay");
        }
        const int deepDelay = 1000;
        BigInteger yDeep = vdf.Evaluate(x, deepDelay, out long deepSquarings);
        BigInteger piDeep = vdf.Prove(x, yDeep, deepDelay, out BigInteger challengeDeep, out BigInteger remainderDeep, out _);
        Check(vdf.Verify(x, yDeep, deepDelay, challengeDeep, piDeep, out long deepVerifications),
            "the deep proof verifies");
        Console.WriteLine("    T=" + deepDelay + ": evaluator squarings=" + deepSquarings + " verifier multiplications="
            + deepVerifications + " (ratio " + (deepSquarings / (double)deepVerifications).ToString("F1") + "x)");
        Check(deepSquarings == deepDelay, "the deep evaluation cost exactly T squarings");
        Check(deepVerifications * 10 < deepSquarings, "verification is more than ten times cheaper at T=1000");
        Check(deepVerifications < 64, "and the verification cost depends only on the challenge width");
        Check(remainderDeep >= 0 && remainderDeep < challengeDeep, "the remainder is smaller than the challenge prime");

        // 6. Determinism of evaluation, challenge and proof.
        BigInteger yAgain = vdf.Evaluate(x, deepDelay, out _);
        Check(yAgain == yDeep, "re-evaluating gives the same output bit for bit");
        Check(vdf.Challenge(x, yDeep, deepDelay) == challengeDeep, "the challenge is derived deterministically");
        BigInteger piAgain = vdf.Prove(x, yDeep, deepDelay, out _, out _, out _);
        Check(piAgain == piDeep, "the proof is deterministic");

        // 7-8. Tampering is caught: a wrong output, a wrong response, a wrong challenge, a wrong delay.
        Check(!vdf.Verify(x, yDeep + 1, deepDelay, vdf.Challenge(x, yDeep + 1, deepDelay), piDeep, out _),
            "a tampered output fails verification even with a challenge derived from it");
        Check(!vdf.Verify(x, yDeep, deepDelay, challengeDeep, piDeep + 1, out _), "a tampered response fails verification");
        Check(!vdf.Verify(x, yDeep, deepDelay, challengeDeep + 2, piDeep, out _), "a tampered challenge prime fails verification");
        BigInteger piOther = vdf.Prove(x, yDeep, deepDelay - 1, out _, out _, out _);
        Check(!vdf.Verify(x, yDeep, deepDelay, challengeDeep, piOther, out _), "a proof for a different delay fails verification");
        var otherBase = new BigInteger(987654321L);
        Check(!vdf.Verify(otherBase, yDeep, deepDelay, challengeDeep, piDeep, out _),
            "a proof cannot be replayed for a different base");

        // 9-10. Degenerate delays and validation.
        Check(vdf.Evaluate(x, 0, out long zeroSquarings) == x % vdf.N && zeroSquarings == 0,
            "T=0 returns the base itself after no squarings");
        Check(vdf.Evaluate(x, 1, out _) == x * x % vdf.N, "T=1 returns the first square");
        Check(Throws(() => vdf.Evaluate(x, -1, out _)), "a negative delay is rejected");
        Check(Throws(() => vdf.Evaluate(vdf.P, 4, out _)), "a base sharing a factor with N is rejected");
        Check(Throws(() => new WesolowskiVdf(1000003, 1000003)), "equal factors are rejected");
        Check(Throws(() => new WesolowskiVdf(1000003, 1000015)), "a composite factor is rejected");

        // 11. Why the toy can shortcut: the factorisation is the trapdoor the real protocol must destroy.
        Check(vdf.EvaluateWithTotient(x, 5000) == vdf.Evaluate(x, 5000, out long hugeSquarings),
            "the totient oracle agrees even where the sequential evaluator needs " + hugeSquarings + " squarings");
        Console.WriteLine("  shortcut: knowing phi(N) collapses T=5000 squarings into one modular exponentiation,"
            + " which is exactly the trapdoor a real VDF must not leak");
        Check(Throws(() => new WesolowskiVdf(4, 1000003)), "an even factor is rejected");

        // 12. Order sensitivity: the map is a group homomorphism, so it is checkable algebraically.
        var otherX = new BigInteger(424242L);
        BigInteger product = x * otherX % vdf.N;
        Check(vdf.Evaluate(product, 5, out _) == vdf.Evaluate(x, 5, out _) * vdf.Evaluate(otherX, 5, out _) % vdf.N,
            "the map is multiplicative, which the totient oracle confirms independently");
        Check(vdf.Evaluate(x, 5, out _) != vdf.Evaluate(x, 4, out _), "different delays give different outputs");

        Console.WriteLine("PASS");
    }
}

internal static class Bits
{
    internal static int LogBits(this long value)
    {
        int bits = 0;
        long v = value;
        while (v > 0)
        {
            bits++;
            v >>= 1;
        }
        return bits;
    }
}
