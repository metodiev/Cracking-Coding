// Task 571 - Implement collision-resistant hash property tests (statistical). (Tier: Advanced, CTCI Beyond — Security & Privacy)
//
// Statistical property tests for a cryptographic hash (SHA-256 from System.Security.Cryptography,
// plus truncated variants used to make collisions observable). Four properties are tested, each
// against a closed form or a known value: (1) correctness - the published SHA-256 digests of "" and
// "abc" are reproduced, and the digest of a message is deterministic and equals the digest of its
// copy; (2) avalanche - flipping one bit of a 64-byte message changes roughly half of the 256 digest
// bits, whose mean over 256 single-bit flips is checked to lie within three standard errors
// (8 / sqrt(256) = 0.5) of 128; (3) collision resistance - no two of 5000 distinct messages collide in
// the full 256-bit digest, and the number of colliding pairs found by drawing N inputs into a 2^16
// bucket space follows the birthday expectation lambda = N (N - 1) / 2 / 2^16, which is checked
// against the observed mean over 32 trials and against the Poisson standard deviation sqrt(lambda),
// with the same check repeated on a 2^20 truncation where lambda = 190.725327; (4) uniformity - a
// chi-square test on 256 byte-value bins has an expectation of 255 with standard deviation
// sqrt(2 * 255) = 22.6, so the file asserts the statistic stays inside four standard deviations.
// A control family shows the tests have power: the identity hash on 16 bits (harmless-looking, and
// exactly uniform) fails the avalanche test with a mean Hamming distance of 1 instead of 8, and
// collides on 34464 of 100000 consecutive inputs by the pigeonhole principle, which SHA-256 does not.
// All randomness is a fixed-seed 64-bit splitmix64 stream (documented because the repo's 16-bit LCG
// has too short a cycle for the 20000-input draws of the 20-bit birthday test).
//
// Documented checks: the two published vectors; determinism and copy equality; the avalanche mean,
// minimum and maximum over all 512 single-bit flips; no collision among 5000 or 100000 full 256-bit
// digests; the 16-bit birthday mean within 2.0 of lambda = 7.9921875 and every one of the 32 trials below
// 30 collisions; the 20-bit birthday count within four standard deviations of 190.725327; the
// chi-square uniformity statistic within four sigma of its mean 255; the information-theoretic floor
// of 34464 colliding pairs for any 16-bit hash on 100000 distinct inputs, which the truncated digest
// meets; the control family of 2000 inputs that all share their low 16 bits, which the identity-style
// hash collapses onto a single bucket (1999000 colliding pairs) while SHA-256 keeps it at the
// birthday value (asserted below 100); the identity-style hash failing the
// avalanche test with a mean of 1 flipped bit instead of 8; and the statistics reappearing exactly on
// a second run.
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

internal static class Rng
{
    internal static ulong NextRaw(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal static class Hashes
{
    internal static byte[] Digest(byte[] message)
    {
        using var sha = SHA256.Create();
        return sha.ComputeHash(message);
    }

    internal static string Hex(byte[] bytes)
    {
        var sb = new StringBuilder(bytes.Length * 2);
        foreach (byte b in bytes) sb.Append(b.ToString("x2"));
        return sb.ToString();
    }

    internal static byte[] Utf8(string text)
    {
        return Encoding.UTF8.GetBytes(text);
    }

    /** First 16 bits of the SHA-256 digest, i.e. a deliberately weak 65536-bucket hash. */
    internal static int Truncate16(byte[] message)
    {
        byte[] digest = Digest(message);
        return (digest[0] << 8) | digest[1];
    }

    /** First 20 bits of the SHA-256 digest: 1048576 buckets, enough for observable collisions. */
    internal static int Truncate20(byte[] message)
    {
        byte[] digest = Digest(message);
        return (digest[0] << 12) | (digest[1] << 4) | (digest[2] >> 4);
    }

    internal static int HammingBits(byte[] a, byte[] b)
    {
        int distance = 0;
        for (int i = 0; i < a.Length; i++)
        {
            int diff = a[i] ^ b[i];
            while (diff != 0)
            {
                distance += diff & 1;
                diff >>= 1;
            }
        }
        return distance;
    }

    /** The control hash: the low 16 bits of the message, i.e. essentially the identity. */
    internal static int Identity16(byte[] message)
    {
        return (message[0] << 8) | message[1];
    }

    internal static byte[] Message(ulong value)
    {
        var bytes = new byte[8];
        for (int i = 0; i < 8; i++) bytes[i] = (byte)(value >> (8 * i));
        return bytes;
    }

    /** Counts colliding pairs among the bucket values by summing C(count, 2) over buckets. */
    internal static long CollidingPairs(IReadOnlyList<int> values)
    {
        var counts = new Dictionary<int, int>();
        foreach (int value in values)
        {
            counts.TryGetValue(value, out int seen);
            counts[value] = seen + 1;
        }
        long pairs = 0;
        foreach (KeyValuePair<int, int> entry in counts)
        {
            pairs += (long)entry.Value * (entry.Value - 1) / 2;
        }
        return pairs;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    internal static void Main()
    {
        // Published vectors.
        Check(Hashes.Hex(Hashes.Digest(Array.Empty<byte>()))
                == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "the SHA-256 digest of the empty message matches the published vector");
        Check(Hashes.Hex(Hashes.Digest(Hashes.Utf8("abc")))
                == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "the SHA-256 digest of \"abc\" matches the published vector");
        Check(Hashes.Digest(Hashes.Utf8("abc")).Length == 32, "a SHA-256 digest is 32 bytes");

        // Determinism and copy equality.
        var message = new byte[64];
        ulong state = 571001UL;
        for (int i = 0; i < message.Length; i++) message[i] = (byte)(Rng.NextRaw(ref state) >> 56);
        var copy = (byte[])message.Clone();
        byte[] digest = Hashes.Digest(message);
        Check(Hashes.Hex(Hashes.Digest(copy)) == Hashes.Hex(digest), "copying the message does not change the digest");
        Check(Hashes.Hex(Hashes.Digest(message)) == Hashes.Hex(digest), "hashing is deterministic");
        Check(Hashes.HammingBits(Hashes.Digest(Hashes.Utf8("abc")), Hashes.Digest(Hashes.Utf8("abd"))) > 40,
                "a one-character change produces a very different digest");

        // Avalanche: flips of single bits of the 512-bit message.
        double avalancheTotal = 0.0;
        int avalancheMin = int.MaxValue;
        int avalancheMax = 0;
        int flips = 0;
        for (int bit = 0; bit < message.Length * 8; bit++)
        {
            var flipped = (byte[])message.Clone();
            flipped[bit / 8] ^= (byte)(1 << (bit % 8));
            int distance = Hashes.HammingBits(digest, Hashes.Digest(flipped));
            avalancheTotal += distance;
            avalancheMin = Math.Min(avalancheMin, distance);
            avalancheMax = Math.Max(avalancheMax, distance);
            flips++;
        }
        Check(flips == 512, "all 512 single-bit flips were tested");
        double avalancheMean = avalancheTotal / flips;
        Check(Math.Abs(avalancheMean - 128.0) <= 3.0 * 8.0 / Math.Sqrt(flips),
                "the avalanche mean is within three standard errors of 128 (" + avalancheMean + ")");
        Check(avalancheMin > 90 && avalancheMax < 170, "every single-bit flip lands in a plausible band ("
                + avalancheMin + " to " + avalancheMax + ")");

        // Collision resistance on the full 256-bit digest.
        var fullDigests = new HashSet<string>();
        var inputs = new List<byte[]>();
        state = 571123UL;
        for (int i = 0; i < 5000; i++)
        {
            inputs.Add(Hashes.Message(Rng.NextRaw(ref state)));
        }
        foreach (byte[] input in inputs) fullDigests.Add(Hashes.Hex(Hashes.Digest(input)));
        Check(fullDigests.Count == 5000, "no collisions among 5000 distinct inputs (" + fullDigests.Count + ")");

        // Birthday behaviour on the 16-bit truncation.
        state = 571777UL;
        double birthdayTotal = 0.0;
        int birthdayMax = 0;
        int trials = 32;
        int sampleSize = 1024;
        double lambda16 = (double)sampleSize * (sampleSize - 1) / 2.0 / 65536.0;
        Check(Math.Abs(lambda16 - 7.9921875) <= 1e-9, "the 16-bit birthday expectation is 7.9921875");
        for (int trial = 0; trial < trials; trial++)
        {
            var values = new List<int>(sampleSize);
            for (int i = 0; i < sampleSize; i++) values.Add(Hashes.Truncate16(Hashes.Message(Rng.NextRaw(ref state))));
            long pairs = Hashes.CollidingPairs(values);
            birthdayTotal += pairs;
            birthdayMax = Math.Max(birthdayMax, (int)pairs);
        }
        double birthdayMean = birthdayTotal / trials;
        double sigma16 = Math.Sqrt(lambda16);
        Check(Math.Abs(birthdayMean - lambda16) <= 2.0, "the observed 16-bit collision mean is near the birthday "
                + "expectation (" + birthdayMean + " vs " + lambda16 + ")");
        Check(Math.Abs(birthdayMean - lambda16) <= 3.0 * sigma16 / Math.Sqrt(trials) + 1.0,
                "the mean is within the sampling error of the expectation (" + sigma16 / Math.Sqrt(trials) + ")");
        Check(birthdayMax < 30, "no single 16-bit trial exceeded 30 collisions (" + birthdayMax + ")");

        // Birthday behaviour on the 20-bit truncation with a much larger sample.
        state = 571999UL;
        var values20 = new List<int>(20000);
        for (int i = 0; i < 20000; i++) values20.Add(Hashes.Truncate20(Hashes.Message(Rng.NextRaw(ref state))));
        long pairs20 = Hashes.CollidingPairs(values20);
        double lambda20 = (double)20000 * 19999 / 2.0 / 1048576.0;
        double sigma20 = Math.Sqrt(lambda20);
        Check(Math.Abs(lambda20 - 190.725327) <= 1e-5, "the 20-bit birthday expectation is 190.725327");
        Check(Math.Abs(pairs20 - lambda20) <= 4.0 * sigma20, "the 20-bit collision count is within four sigmas ("
                + pairs20 + " vs " + lambda20 + ")");
        Check(pairs20 > 0, "the 20-bit truncation does produce collisions");

        // Uniformity: chi-square over the 256 byte values of the digests of 1000 messages.
        var histogram = new long[256];
        state = 572222UL;
        int hashCount = 1000;
        for (int i = 0; i < hashCount; i++)
        {
            byte[] d = Hashes.Digest(Hashes.Message(Rng.NextRaw(ref state)));
            foreach (byte value in d) histogram[value]++;
        }
        long total = hashCount * 32;
        double expected = total / 256.0;
        double chiSquare = 0.0;
        for (int value = 0; value < 256; value++)
        {
            double diff = histogram[value] - expected;
            chiSquare += diff * diff / expected;
        }
        Check(total == 32000, "32000 digest bytes were binned");
        Check(Math.Abs(expected - 125.0) <= 1e-9, "the expected bin count is 125");
        Check(Math.Abs(chiSquare - 255.0) <= 4.0 * Math.Sqrt(2.0 * 255.0),
                "the chi-square statistic is within four sigma of its 255 mean (" + chiSquare + ")");

        // Control: the identity-style hash fails avalanche and collides by the pigeonhole principle.
        double identityTotal = 0.0;
        int identityFlips = 0;
        for (int bit = 0; bit < 16; bit++)
        {
            byte[] a16 = Hashes.Message(12345UL);
            byte[] b16 = (byte[])a16.Clone();
            b16[bit / 8] ^= (byte)(1 << (bit % 8));
            byte[] da = new byte[2];
            byte[] db = new byte[2];
            int ha = Hashes.Identity16(a16);
            int hb = Hashes.Identity16(b16);
            da[0] = (byte)(ha >> 8);
            da[1] = (byte)ha;
            db[0] = (byte)(hb >> 8);
            db[1] = (byte)hb;
            identityTotal += Hashes.HammingBits(da, db);
            identityFlips++;
        }
        double identityMean = identityTotal / identityFlips;
        Check(identityMean <= 5.0, "the identity-style hash fails the avalanche test (mean " + identityMean
                + " of 16 bits instead of 8)");
        Check(Math.Abs(identityMean - 8.0) > 2.5, "the identity mean is far from the ideal 8");

        // Pigeonhole: no 16-bit hash can be collision free on 100000 distinct inputs, so the
        // information-theoretic floor on the number of colliding pairs is 100000 - 65536 = 34464;
        // the full 256-bit digest of the same inputs has no collisions at all.
        var sequential = new List<int>(100000);
        for (int i = 0; i < 100000; i++) sequential.Add(Hashes.Truncate16(Hashes.Message((ulong)i)));
        long sequentialPairs = Hashes.CollidingPairs(sequential);
        Check(sequentialPairs >= 34464, "any 16-bit hash shows at least 34464 colliding pairs on 100000 inputs ("
                + sequentialPairs + ")");
        var sequentialFull = new HashSet<string>();
        for (int i = 0; i < 100000; i++)
        {
            sequentialFull.Add(Hashes.Hex(Hashes.Digest(Hashes.Message((ulong)i))));
        }
        Check(sequentialFull.Count == 100000, "the full 256-bit digest is collision free on those 100000 inputs");

        // A structured family that defeats the identity hash but not SHA-256: every input is a multiple
        // of 65536, so the identity maps all of them onto one bucket while SHA-256 spreads them like a
        // random function, with the observed pair count inside four sigmas of the birthday value 30.5.
        var identityFamily = new List<int>();
        var hashedFamily = new List<int>();
        for (int i = 0; i < 2000; i++)
        {
            byte[] structured = Hashes.Message((ulong)i * 65536UL);
            identityFamily.Add(Hashes.Identity16(structured));
            hashedFamily.Add(Hashes.Truncate16(structured));
        }
        Check(new HashSet<int>(identityFamily).Count == 1, "the identity hash maps the whole family to one bucket");
        Check(Hashes.CollidingPairs(identityFamily) == 1999000L,
                "the identity hash produces C(2000, 2) = 1999000 colliding pairs on the family");
        long hashedPairs = Hashes.CollidingPairs(hashedFamily);
        double lambdaFamily = 2000.0 * 1999.0 / 2.0 / 65536.0;
        Check(Math.Abs(lambdaFamily - 30.5023193) <= 1e-6, "the family birthday expectation is 30.5023193");
        Check(Math.Abs(hashedPairs - lambdaFamily) <= 4.0 * Math.Sqrt(lambdaFamily),
                "SHA-256 spreads the structured family like a random function (" + hashedPairs + " vs " + lambdaFamily + ")");
        Check(hashedPairs < 100, "SHA-256 does not collapse the structured family");

        // Reproducibility of the measured statistics.
        var repeat = new List<int>();
        for (int i = 0; i < sampleSize; i++) repeat.Add(Hashes.Truncate16(Hashes.Message((ulong)(571777 + i))));
        long repeatPairs = Hashes.CollidingPairs(repeat);
        var again = new List<int>();
        for (int i = 0; i < sampleSize; i++) again.Add(Hashes.Truncate16(Hashes.Message((ulong)(571777 + i))));
        Check(Hashes.CollidingPairs(again) == repeatPairs, "the collision count is reproducible for a fixed input list");

        Console.WriteLine("PASS");
    }
}
