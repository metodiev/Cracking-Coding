// Task 840 - Implement fully homomorphic encryption demo simple arithmetic (toy). (Tier: Master, CTCI Beyond — Security & Privacy)
//
// The DGHV scheme (van Dijk - Gentry - Halevi - Vaikuntanathan, "Fully Homomorphic Encryption over the Integers"),
// which is the simplest construction that is genuinely fully homomorphic:
//
//   * the secret key is a 32-bit odd prime p; the public key is lambda = 32 integers x_i = q_i * p + 2 r_i with
//     q_i about 2^30 and a four-bit noise r_i. Each x_i is a near-multiple of p, and its distance to the nearest
//     multiple of p carries the noise that the scheme must keep small.
//   * Encrypt(m) picks a sparse random subset S of the x_i and a fresh r and returns c = m + 2r + sum_{i in S} x_i.
//     The noise of c - writes n(c) - is m + 2r + sum 2 r_i, and c = (multiple of p) + n(c).
//   * Decrypt(c) reduces c modulo p into the symmetric range [-p/2, p/2) and reads the low bit, which is m exactly
//     as long as |n(c)| < p/2.
//   * Addition and multiplication are performed on the ciphertexts themselves: n(c1 + c2) = n(c1) + n(c2) and
//     n(c1 * c2) = n(c1) * n(c2), so XOR is addition and AND is multiplication. Noise therefore grows additively
//     under XOR and multiplicatively under AND, which is exactly the depth limitation that bootstrapping removes.
//
// Verified, each by recomputation:
//   * the noise model is exact: an independent symbolic evaluation of n through the same circuit agrees with the
//     residue the decryptor measures, node by node, until the budget is exhausted;
//   * the truth tables of XOR (addition), AND (multiplication) and NOT (1 - c) hold on all input combinations,
//     and a mixed circuit matches its plaintext evaluation bit for bit;
//   * noise growth is measured: linear in the number of XORed ciphertexts, and multiplicative under AND. The file
//     *predicts* the largest AND depth that still decrypts correctly from |n| < p/2 and then checks that the next
//     depth fails - correctness and failure are both verified, not just the happy path;
//   * randomized encryption: repeated encryptions of the same bit differ, and a fresh ciphertext is never the bare
//     bit;
//   * a bounded brute-force attack recovers p: scanning 2049 odd candidate moduli near p, the known-noise bound
//     rejects all but the true modulus, which is the measured statement of how little security 32 bits of p
//     provide. This is a measurement, not a claim about a real parameter set.
//
// Documented simplifications: ciphertexts are not reduced modulo x_0 after every multiplication (that reduction
// bounds ciphertext size but pushes k * 2 r_0 of extra noise in, and the size/noise trade-off is the whole point of
// the real parameter choice), so the sizes here grow with the multiplicative depth and the evaluator keeps exact
// BigInteger arithmetic instead of the fixed-size integer arithmetic of a real implementation; there is no
// squashing, no relinearization and no bootstrapping, so the depth is bounded by the noise budget as measured;
// p is 32 bits, which no real instance would use, and the attack above shows why. Randomness is hand-written
// splitmix64 with explicit seeds, so every number in the output is reproducible. No real key material is used.
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

    internal static long NextBelow(ref ulong state, long bound) => (long)(Next(ref state) % (ulong)bound);

    /// <summary>A random BigInteger below 2^bits, from whole 64-bit PRNG words.</summary>
    internal static BigInteger NextBig(ref ulong state, int bits)
    {
        var bytes = new byte[(bits + 7) / 8];
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)(Next(ref state) & 0xFF);
        bytes[bytes.Length - 1] &= (byte)(0xFF >> (bytes.Length * 8 - bits));
        return new BigInteger(bytes, isUnsigned: true);
    }
}

/// <summary>A ciphertext together with the symbolic noise the evaluator carries for verification.</summary>
internal sealed class Ciphertext
{
    internal BigInteger Value;
    internal BigInteger Noise;
    internal int Depth;
    internal string Origins;

    internal Ciphertext(BigInteger value, BigInteger noise, int depth, string origins)
    {
        Value = value;
        Noise = noise;
        Depth = depth;
        Origins = origins;
    }
}

/// <summary>Explicit in-memory channel: it counts the bits that actually travel, so cost is measured.</summary>
internal sealed class Channel
{
    internal long Bits { get; private set; }
    internal int Transfers { get; private set; }

    internal void Send(BigInteger value)
    {
        Transfers++;
        Bits += value.IsZero ? 1 : value.GetBitLength();
    }
}

internal sealed class FheKey
{
    internal BigInteger P { get; }
    internal BigInteger[] PublicKey { get; }

    /// <summary>
    /// The per-element noise r_i of x_i = q_i p + 2 r_i. A real deployment would not publish these; the checks use
    /// them to build the exact symbolic noise n(c) = m + 2r + sum 2 r_i, which is what makes the noise model
    /// verifiable rather than asserted.
    /// </summary>
    internal BigInteger[] ElementNoise { get; }

    internal int NoiseBits { get; }

    /// <summary>Bits of p: the toy security parameter eta, printed so the attack below can be judged against it.</summary>
    internal int SecurityBits => (int)P.GetBitLength();

    internal FheKey(BigInteger p, BigInteger[] publicKey, BigInteger[] elementNoise, int noiseBits)
    {
        P = p;
        PublicKey = publicKey;
        ElementNoise = elementNoise;
        NoiseBits = noiseBits;
    }

    internal BigInteger HalfP => P >> 1;
}

internal static class Fhe
{
    /// <summary>Largest prime below 2^32: the toy secret modulus. Real instances use thousands of bits.</summary>
    internal static readonly BigInteger SecretPrime = 4294967291;

    internal static FheKey KeyGen(int lambda, int noiseBits, ulong seed)
    {
        ulong state = seed;
        var publicKey = new BigInteger[lambda];
        var elementNoise = new BigInteger[lambda];
        long noiseRange = 1L << noiseBits;
        for (int i = 0; i < lambda; i++)
        {
            BigInteger q = SplitMix64.NextBig(ref state, 30) + (BigInteger.One << 29);
            BigInteger r = SplitMix64.NextBelow(ref state, noiseRange);
            elementNoise[i] = r;
            publicKey[i] = q * SecretPrime + 2 * r;
        }
        return new FheKey(SecretPrime, publicKey, elementNoise, noiseBits);
    }

    /// <summary>
    /// Encryption: a sparse subset sum of the public key plus the message and a fresh noise. The returned
    /// ciphertext carries the symbolic noise m + 2r + sum 2 r_i, which the checks compare against the residue the
    /// decryptor measures.
    /// </summary>
    internal static Ciphertext Encrypt(FheKey key, int bit, ref ulong state, int subsetSize)
    {
        if (bit != 0 && bit != 1) throw new ArgumentException("a plaintext bit must be 0 or 1", nameof(bit));
        long noiseRange = 1L << key.NoiseBits;
        BigInteger noise = bit + 2 * SplitMix64.NextBelow(ref state, noiseRange);
        BigInteger value = noise;
        for (int i = 0; i < subsetSize; i++)
        {
            int index = 1 + (int)SplitMix64.NextBelow(ref state, key.PublicKey.Length - 1);
            value += key.PublicKey[index];
            noise += 2 * key.ElementNoise[index];
        }
        return new Ciphertext(value, noise, 0, "enc(" + bit + ")");
    }

    /// <summary>
    /// A genuine ciphertext of <paramref name="bit"/> whose noise is exactly <paramref name="targetNoise"/>: the
    /// subset of public elements is chosen (by exact subset sum over the key's own noise terms) so that the noise
    /// lands on the requested value. This is used to walk the noise budget boundary precisely; it only ever makes
    /// the noise smaller than a random encryption would, so the ciphertext stays valid.
    /// </summary>
    internal static Ciphertext EncryptWithNoise(FheKey key, int bit, int targetNoise, ref ulong state)
    {
        if (targetNoise % 2 != bit) throw new ArgumentException("noise parity must match the bit", nameof(targetNoise));
        int needed = (targetNoise - bit) / 2;                    // r + sum r_i
        int n = key.PublicKey.Length;
        var reachable = new bool[needed + 1, n + 1];
        reachable[0, 0] = true;
        for (int i = 0; i < n; i++)
        {
            int r = (int)key.ElementNoise[i];
            for (int s = 0; s <= needed; s++)
            {
                if (!reachable[s, i]) continue;
                reachable[s, i + 1] = true;                      // skip element i
                if (s + r <= needed) reachable[s + r, i + 1] = true;
            }
        }
        if (!reachable[needed, n]) throw new ArgumentException("target noise is not assembleable from this key");
        var chosen = new List<int>();
        int remaining = needed;
        for (int i = n - 1; i >= 0 && remaining > 0; i--)
        {
            if (reachable[remaining, i]) continue;               // element i was skipped
            chosen.Add(i);
            remaining -= (int)key.ElementNoise[i];
        }
        BigInteger value = bit;
        BigInteger noise = bit;
        foreach (int index in chosen)
        {
            value += key.PublicKey[index];
            noise += 2 * key.ElementNoise[index];
        }
        _ = state;
        return new Ciphertext(value, noise, 0, "enc" + bit + "(n=" + targetNoise + ")");
    }

    /// <summary>
    /// Decryption: reduce modulo p into the symmetric range and read the low bit. The residue is the noise the
    /// scheme depends on, so it is returned alongside the bit and checked against the symbolic noise.
    /// </summary>
    internal static (int Bit, BigInteger Residue) Decrypt(FheKey key, BigInteger ciphertext)
    {
        BigInteger residue = ciphertext % key.P;                 // BigInteger % yields a non-negative remainder
        if (residue > key.HalfP) residue -= key.P;               // symmetric range [-p/2, p/2)
        return (residue.IsEven ? 0 : 1, residue);
    }

    internal static Ciphertext Xor(Ciphertext a, Ciphertext b, Channel channel)
    {
        channel.Send(a.Value);
        channel.Send(b.Value);
        return new Ciphertext(a.Value + b.Value, a.Noise + b.Noise, Math.Max(a.Depth, b.Depth),
            "xor(" + a.Origins + "," + b.Origins + ")");
    }

    internal static Ciphertext And(Ciphertext a, Ciphertext b, Channel channel)
    {
        channel.Send(a.Value);
        channel.Send(b.Value);
        return new Ciphertext(a.Value * b.Value, a.Noise * b.Noise, Math.Max(a.Depth, b.Depth) + 1,
            "and(" + a.Origins + "," + b.Origins + ")");
    }

    /// <summary>NOT is 1 - c: it flips the bit and the parity of the noise without changing its size much.</summary>
    internal static Ciphertext Not(Ciphertext a)
    {
        return new Ciphertext(BigInteger.One - a.Value, BigInteger.One - a.Noise, a.Depth + 1, "not(" + a.Origins + ")");
    }

    internal static int Bits(Ciphertext c) => c.Value.IsZero ? 1 : (int)c.Value.GetBitLength();
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Plaintext evaluation of a boolean circuit, used as the oracle for the encrypted evaluations.</summary>
    private static int Xor(int a, int b) => a ^ b;

    private static int And(int a, int b) => a & b;

    private static int Not(int a) => 1 - a;

    public static void Main()
    {
        const int Lambda = 32, NoiseBits = 2, Subset = 8;
        var key = Fhe.KeyGen(Lambda, NoiseBits, 0x5EED840000000001UL);
        Check(key.SecurityBits == 32, "the toy modulus p is 32 bits (" + key.P + ")");
        Check(key.PublicKey.Length == Lambda, "the public key holds lambda = " + Lambda + " near-multiples of p");
        bool nearMultiples = true;
        foreach (BigInteger x in key.PublicKey)
        {
            BigInteger residue = x % key.P;
            if (residue > key.HalfP) residue -= key.P;
            if (!residue.IsEven || BigInteger.Abs(residue) >= (BigInteger.One << (NoiseBits + 1))) nearMultiples = false;
        }
        Check(nearMultiples, "every public element is a multiple of p plus an even noise of at most "
            + (1 << (NoiseBits + 1)) + " (the structure the approximate-GCD assumption hides)");

        {
            // Negative check: the decryption key must be the exact secret modulus. A modulus that differs from it
            // by 2 no longer cancels the multiples of p, so the bit is wrong and the "noise" reads far too large.
            var wrongKey = new FheKey(key.P - 2, key.PublicKey, key.ElementNoise, NoiseBits);
            ulong s = 0x5EED840000000002UL;
            int wrong = 0;
            BigInteger totalReading = 0;
            for (int t = 0; t < 40; t++)
            {
                Ciphertext c = Fhe.Encrypt(key, t % 2, ref s, Subset);
                (int bit, BigInteger residue) = Fhe.Decrypt(wrongKey, c.Value);
                if (bit != t % 2) wrong++;
                totalReading += BigInteger.Abs(residue);
            }
            BigInteger averageReading = totalReading / 40;
            Check(wrong >= 10, "a decryption modulus off by 2 gives wrong bits (" + wrong + " of 40), so the key must "
                + "match exactly");
            Check(averageReading > (BigInteger.One << 20), "and its noise reading averages " + averageReading
                + " instead of the true noise below 128, which is how a decryptor notices the wrong key");
        }

        // ---- round trip, noise model and randomised encryption ---------------------------------------
        {
            ulong state = 0x5EED840000000003UL;
            int wrong = 0, mismatchedResidue = 0, smallNoise = 0, bareBits = 0;
            var ciphertexts = new HashSet<string>();
            long maxNoise = 0;
            for (int t = 0; t < 200; t++)
            {
                int bit = t % 2;
                Ciphertext c = Fhe.Encrypt(key, bit, ref state, Subset);
                (int got, BigInteger measured) = Fhe.Decrypt(key, c.Value);
                if (got != bit) wrong++;
                if (measured != c.Noise) mismatchedResidue++;
                if (BigInteger.Abs(c.Noise) > key.HalfP) smallNoise++;
                if (BigInteger.Abs(c.Noise) > maxNoise) maxNoise = (long)BigInteger.Abs(c.Noise);
                if (c.Value == bit) bareBits++;
                ciphertexts.Add(c.Value.ToString());
            }
            Check(wrong == 0, "decryption recovers the plaintext bit in all 200 fresh ciphertexts");
            Check(mismatchedResidue == 0, "the measured residue equals the symbolic noise exactly, so the additive "
                + "noise model n(c) is verified rather than assumed");
            Check(smallNoise == 0 && maxNoise < 100, "fresh noise stays far inside the budget (worst |n| = " + maxNoise
                + " against p/2 = " + key.HalfP + ")");
            Check(bareBits == 0 && ciphertexts.Count > 190, "encryption is randomised: no ciphertext is the bare bit and "
                + ciphertexts.Count + " of 200 encryptions of one bit are distinct");
            Console.WriteLine("fresh ciphertexts: 200 round trips, worst |noise|=" + maxNoise + ", distinct=" + ciphertexts.Count);
        }

        // ---- truth tables of XOR, AND and NOT ---------------------------------------------------------
        {
            ulong state = 0x5EED840000000004UL;
            var channel = new Channel();
            bool xorOk = true, andOk = true, notOk = true, channelOk = true;
            for (int a = 0; a < 2; a++)
            {
                for (int b = 0; b < 2; b++)
                {
                    Ciphertext ca = Fhe.Encrypt(key, a, ref state, Subset);
                    Ciphertext cb = Fhe.Encrypt(key, b, ref state, Subset);
                    Ciphertext sum = Fhe.Xor(ca, cb, channel);
                    Ciphertext product = Fhe.And(ca, cb, channel);
                    if (Fhe.Decrypt(key, sum.Value).Bit != Xor(a, b)) xorOk = false;
                    if (Fhe.Decrypt(key, product.Value).Bit != And(a, b)) andOk = false;
                    if (sum.Noise != ca.Noise + cb.Noise) channelOk = false;
                    if (product.Noise != ca.Noise * cb.Noise) channelOk = false;
                }
                Ciphertext c = Fhe.Encrypt(key, a, ref state, Subset);
                if (Fhe.Decrypt(key, Fhe.Not(c).Value).Bit != Not(a)) notOk = false;
            }
            Check(xorOk, "XOR is ciphertext addition on all four input combinations");
            Check(andOk, "AND is ciphertext multiplication on all four input combinations");
            Check(notOk, "NOT is 1 - c and flips the bit");
            Check(channelOk, "the symbolic noise of a sum is n1+n2 and of a product is n1*n2, exactly");
            Check(channel.Bits > 0 && channel.Transfers == 16, "the evaluator exchanged " + channel.Transfers
                + " ciphertexts (" + channel.Bits + " bits) for 16 operations");
        }

        // ---- a mixed circuit matches its plaintext evaluation -----------------------------------------
        {
            ulong state = 0x5EED840000000005UL;
            var channel = new Channel();
            bool ok = true;
            for (int t = 0; t < 8; t++)
            {
                int a = t & 1, b = (t >> 1) & 1, c = (t >> 2) & 1;
                Ciphertext ca = Fhe.Encrypt(key, a, ref state, Subset);
                Ciphertext cb = Fhe.Encrypt(key, b, ref state, Subset);
                Ciphertext cc = Fhe.Encrypt(key, c, ref state, Subset);
                // (a AND b) XOR (NOT c) XOR (a AND c)
                Ciphertext left = Fhe.And(ca, cb, channel);
                Ciphertext right = Fhe.And(ca, cc, channel);
                Ciphertext together = Fhe.Xor(left, right, channel);
                Ciphertext result = Fhe.Xor(together, Fhe.Not(cc), channel);
                int expect = Xor(Xor(And(a, b), And(a, c)), Not(c));
                (int bit, BigInteger residue) = Fhe.Decrypt(key, result.Value);
                if (bit != expect || residue != result.Noise) ok = false;
            }
            Check(ok, "a mixed 4-gate circuit matches its plaintext evaluation on all 8 input combinations, and its "
                + "symbolic noise matches the residue too");
            Console.WriteLine("mixed circuit: 8 combinations exact, depth=2");
        }

        // ---- XOR is cheap: noise grows linearly ------------------------------------------------------
        {
            ulong state = 0x5EED840000000006UL;
            var channel = new Channel();
            Ciphertext accumulator = Fhe.Encrypt(key, 1, ref state, Subset);
            int parity = 1;
            BigInteger symbolic = accumulator.Noise;
            for (int i = 1; i < 64; i++)
            {
                Ciphertext next = Fhe.Encrypt(key, i % 2, ref state, Subset);
                symbolic += next.Noise;
                parity ^= i % 2;
                accumulator = Fhe.Xor(accumulator, next, channel);
            }
            (int bit, BigInteger measured) = Fhe.Decrypt(key, accumulator.Value);
            Check(measured == symbolic && accumulator.Noise == symbolic, "after 64 XORs the measured residue still equals "
                + "the symbolically summed noise (" + measured + ")");
            Check(bit == parity, "XORing 64 fresh ciphertexts yields the plaintext parity (bit " + bit + ", expected "
                + parity + ")");
            Check(BigInteger.Abs(measured) < key.HalfP, "the linear noise of " + BigInteger.Abs(measured)
                + " stays inside the budget of " + key.HalfP);
            Console.WriteLine("64-way XOR: noise=" + measured + " budget=" + key.HalfP + " depthFree=true");
        }

        // ---- AND is expensive: the depth limit is predicted and then observed ------------------------
        {
            ulong state = 0x5EED840000000007UL;
            var channel = new Channel();
            var level = new Ciphertext[32];
            BigInteger maxFresh = 1;
            for (int i = 0; i < level.Length; i++)
            {
                level[i] = Fhe.Encrypt(key, 1, ref state, Subset);
                if (BigInteger.Abs(level[i].Noise) > maxFresh) maxFresh = BigInteger.Abs(level[i].Noise);
            }

            // Closed-form prediction: level d has noise about maxFresh^(2^d), so it keeps decrypting while that
            // quantity stays below p/2. Computed from the fresh noises alone, without running the tree.
            int modelDepth = 0;
            BigInteger modelNoise = maxFresh * maxFresh;
            while (modelNoise < key.HalfP) { modelDepth++; modelNoise *= modelNoise; }
            Check(modelDepth >= 1, "the toy budget guarantees at least one level of AND for the worst fresh noise ("
                + maxFresh + "): " + modelDepth + " levels");

            int depth = 0, correctLevels = 0, overBudgetLevel = 0, wrongBits = 0;
            bool guaranteeHeld = true;
            while (level.Length > 1)
            {
                var next = new Ciphertext[level.Length / 2];
                BigInteger maxNoise = 0;
                for (int i = 0; i < next.Length; i++)
                {
                    next[i] = Fhe.And(level[i * 2], level[i * 2 + 1], channel);
                    if (next[i].Noise > maxNoise) maxNoise = next[i].Noise;
                }
                depth++;
                level = next;
                if (maxNoise < key.HalfP)
                {
                    correctLevels = depth;
                    foreach (Ciphertext node in level)                    // the guarantee: bit 1 and residue == noise
                    {
                        (int bit, BigInteger residue) = Fhe.Decrypt(key, node.Value);
                        if (bit != 1 || residue != node.Noise) guaranteeHeld = false;
                    }
                }
                else if (overBudgetLevel == 0)
                {
                    overBudgetLevel = depth;
                    foreach (Ciphertext node in level) if (Fhe.Decrypt(key, node.Value).Bit != 1) wrongBits++;
                }
            }
            Check(guaranteeHeld, "every AND node inside the budget decrypts correctly and its measured residue equals "
                + "the symbolic noise, so the budget rule is verified rather than assumed");
            Check(correctLevels >= modelDepth, "the closed-form prediction is a true lower bound: predicted " + modelDepth
                + " levels, observed " + correctLevels);
            Check(overBudgetLevel == correctLevels + 1 && wrongBits > 0, "the next level (noise above p/2) already "
                + "produces " + wrongBits + " wrong bits out of " + (32 >> overBudgetLevel)
                + ": the depth limit is real");

            // Now the boundary itself, walked exactly: a chain of ANDs whose noises multiply across p/2 = 2.1e9.
            var chain = new[] { 49, 49, 49, 49, 49, 9 };
            BigInteger product = 1;
            Ciphertext accumulator = Fhe.EncryptWithNoise(key, 1, chain[0], ref state);
            product = accumulator.Noise;
            Check(product == chain[0], "the boundary chain starts from a ciphertext whose noise is exactly "
                + chain[0] + " (" + product + ")");
            int boundaryHits = 0;
            bool boundaryExact = true;
            for (int i = 1; i < chain.Length; i++)
            {
                Ciphertext factor = Fhe.EncryptWithNoise(key, 1, chain[i], ref state);
                Ciphertext productCipher = Fhe.And(accumulator, factor, channel);
                BigInteger expected = product * factor.Noise;
                Check(productCipher.Noise == expected, "the AND of step " + i + " carries the symbolic product "
                    + expected);
                (int bit, BigInteger residue) = Fhe.Decrypt(key, productCipher.Value);
                if (expected < key.HalfP)
                {
                    if (bit != 1 || residue != expected) boundaryExact = false;
                }
                else
                {
                    boundaryHits++;
                    // Past the budget the decryptor measures noise - p (one wrap), which is even, so the bit flips.
                    if (bit != 0 || residue != expected - key.P) boundaryExact = false;
                }
                accumulator = productCipher;
                product = expected;
            }
            Check(boundaryHits == 1, "the engineered chain crosses the budget exactly once (" + boundaryHits
                + " crossing, product " + product + " against p/2 = " + key.HalfP + ")");
            Check(boundaryExact, "inside the budget the bit is 1 and the residue is the noise; on the crossing the "
                + "residue is exactly noise - p and the bit flips to 0, which is the depth limit derived rather than "
                + "observed");
            Console.WriteLine("AND tree: predicted=" + modelDepth + " correct=" + correctLevels + " maxFresh=" + maxFresh
                + " overBudgetLevel=" + overBudgetLevel + " wrongBits=" + wrongBits
                + " | boundary product=" + product + " p/2=" + key.HalfP);
        }

        // ---- ciphertext growth under the (omitted) modulus reduction ---------------------------------
        {
            ulong state = 0x5EED840000000009UL;
            var channel = new Channel();
            Ciphertext a = Fhe.Encrypt(key, 1, ref state, Subset);
            int size0 = Fhe.Bits(a);
            Ciphertext squared = Fhe.And(a, a, channel);
            Ciphertext cubed = Fhe.And(squared, a, channel);
            Check(size0 > 30 && Fhe.Bits(squared) > size0 && Fhe.Bits(cubed) > Fhe.Bits(squared),
                "ciphertext size grows with the multiplicative depth (" + size0 + " -> " + Fhe.Bits(squared) + " -> "
                + Fhe.Bits(cubed) + " bits), which is the size that a real instance controls by reducing modulo x_0");
            Check(BigInteger.Abs(cubed.Noise) > BigInteger.Abs(squared.Noise) && BigInteger.Abs(squared.Noise)
                > BigInteger.Abs(a.Noise), "the same growth shows up in the noise");
            Console.WriteLine("size growth: " + size0 + " -> " + Fhe.Bits(squared) + " -> " + Fhe.Bits(cubed)
                + " bits; noise " + a.Noise + " -> " + squared.Noise + " -> " + cubed.Noise);
        }

        // ---- the toy is insecure, by measurement -----------------------------------------------------
        {
            // The attacker knows the advertised noise bound rho: the true modulus is the only candidate for which
            // every known ciphertext reduces to a *small* residue of the right parity. A candidate an even offset
            // away from p preserves the parity of the residue but not its size, which is what separates them.
            const int Window = 2048, Samples = 16, KnownNoiseBound = 1024;
            ulong state = 0x5EED84000000000AUL;
            var samples = new Ciphertext[Samples];
            for (int i = 0; i < samples.Length; i++) samples[i] = Fhe.Encrypt(key, i % 2, ref state, Subset);
            BigInteger start = key.P - Window;
            if (start.IsEven) start += 1;                            // p is odd, so only odd candidates are tried
            int survivors = 0, rejectedByFirst = 0, tried = 0;
            BigInteger found = 0;
            for (BigInteger candidate = start; candidate <= key.P + Window; candidate += 2)
            {
                tried++;
                bool consistent = true;
                for (int i = 0; i < samples.Length; i++)
                {
                    BigInteger residue = samples[i].Value % candidate;
                    if (residue > candidate >> 1) residue -= candidate;
                    if ((residue.IsEven ? 0 : 1) != i % 2 || BigInteger.Abs(residue) >= KnownNoiseBound)
                    {
                        consistent = false;
                        if (i == 0) rejectedByFirst++;
                        break;
                    }
                }
                if (consistent)
                {
                    survivors++;
                    if (found.IsZero) found = candidate;
                }
            }
            Check(survivors == 1 && found == key.P, "a bounded scan of " + tried + " odd candidate moduli accepts "
                + "exactly one, the true p (" + found + "), from " + Samples + " known-plaintext ciphertexts");
            Check(rejectedByFirst >= tried - 2, "the noise bound alone rejects " + rejectedByFirst + " of " + tried
                + " candidates on the first ciphertext, which is the pruning this attack relies on");
            Check(key.SecurityBits == 32, "the search space of the toy modulus is 2^" + key.SecurityBits
                + ", which is the measured reason the instance above is not secure and a real p is thousands of bits wide");
            Console.WriteLine("bounded attack: candidates=" + tried + " survivors=" + survivors + " rejectedByFirstBit="
                + rejectedByFirst + " recovered=" + found + " (2^" + key.SecurityBits + " space)");
        }

        // ---- determinism -----------------------------------------------------------------------------
        {
            ulong stateA = 0x5EED84000000000BUL, stateB = stateA;
            Ciphertext first = Fhe.Encrypt(key, 1, ref stateA, Subset);
            Ciphertext second = Fhe.Encrypt(key, 1, ref stateB, Subset);
            Check(first.Value == second.Value && first.Noise == second.Noise,
                "two runs with the same seed produce identical ciphertexts and identical noise");
            var keyA = Fhe.KeyGen(Lambda, NoiseBits, 0x5EED84000000000CUL);
            var keyB = Fhe.KeyGen(Lambda, NoiseBits, 0x5EED84000000000CUL);
            bool sameKey = true;
            for (int i = 0; i < Lambda; i++) if (keyA.PublicKey[i] != keyB.PublicKey[i]) sameKey = false;
            Check(sameKey, "key generation is reproducible for a fixed seed");
            Check(keyA.PublicKey[0] != key.PublicKey[0], "a different seed gives a different public key");
            Console.WriteLine("determinism: ciphertext=" + Fhe.Bits(first) + " bits reproducible, keys reproducible");
        }

        Console.WriteLine("PASS");
    }
}
