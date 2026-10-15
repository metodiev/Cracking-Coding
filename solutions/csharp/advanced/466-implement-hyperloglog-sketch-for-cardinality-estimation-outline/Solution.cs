// Task 466 - Implement hyperloglog sketch for cardinality estimation (outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Outline HyperLogLog with m = 2^9 = 512 registers. Items are hashed with the FIXED
// 32-bit mixer mix32 (0x45d9f3b xorshift-multiply; identical constants in every
// language port). The top p = 9 bits pick the register; the rank is the 1-based
// position of the leftmost set bit of the remaining 23 bits (all-zero tail counts
// as 24). Estimate = alpha_m * m^2 / sum(2^-M[j]) with alpha_m = 0.7213/(1+1.079/m),
// plus linear counting m*ln(m/V) while raw <= 2.5m and V registers are still zero.
//
// Checks: for each digest size n in {100, 1000, 5000, 20000}, 16 independent digests
// are built from keys hashed as (pool << 17) | value so the pools are disjoint (a
// shared 16-bit key space would correlate the digests). Documented bounds:
// |mean estimate - n| / n <= 1.04/sqrt(m) ~ 0.046 (measured max 0.008); every single
// digest within the 3-sigma envelope 3*1.04/sqrt(m) ~ 0.138 (measured max 0.098);
// mean estimates increase with cardinality; a register-wise max merge of two
// complementary digests equals the digest of the union exactly.
using System;
using System.Collections.Generic;

internal class HyperLogLog
{
    public const int PBits = 9;
    public const int M = 1 << PBits;
    public const int RankWidth = 32 - PBits;
    public const double Alpha = 0.7213 / (1 + 1.079 / M);

    public readonly int[] Registers = new int[M];

    public static uint Mix32(uint x)
    {
        x ^= x >> 16;
        x *= 0x45D9F3Bu;
        x ^= x >> 16;
        x *= 0x45D9F3Bu;
        x ^= x >> 16;
        return x;
    }

    private static int BitLength(uint value)
    {
        int bl = 0;
        while (value > 0)
        {
            bl++;
            value >>= 1;
        }
        return bl;
    }

    public static int RankOf(uint rest)
    {
        if (rest == 0) return RankWidth + 1;
        return RankWidth - BitLength(rest) + 1;
    }

    public void Add(uint key)
    {
        uint h = Mix32(key);
        int index = (int)(h >> RankWidth);
        uint rest = h & ((1u << RankWidth) - 1);
        int r = RankOf(rest);
        if (r > Registers[index]) Registers[index] = r;
    }

    public void Merge(HyperLogLog other)
    {
        for (int i = 0; i < M; i++) Registers[i] = Math.Max(Registers[i], other.Registers[i]);
    }

    public double Estimate()
    {
        double z = 0.0;
        int zeros = 0;
        foreach (int m in Registers)
        {
            z += Math.Pow(2.0, -m);
            if (m == 0) zeros++;
        }
        double raw = Alpha * M * M / z;
        if (raw <= 2.5 * M && zeros > 0) return M * Math.Log((double)M / zeros);
        return raw;
    }
}

public static class Solution
{
    private static int seed;

    private static int Lcg() => seed = (seed * 75 + 74) % 65537;

    private static uint[] PoolKeys(int count, int pool, int startSeed)
    {
        seed = startSeed;
        var keys = new uint[count];
        var seen = new HashSet<int>();
        int n = 0;
        while (n < count)
        {
            int value = Lcg();
            if (seen.Add(value)) keys[n++] = (uint)((pool << 17) | value);
        }
        return keys;
    }

    private static HyperLogLog Digest(uint[] keys)
    {
        var sketch = new HyperLogLog();
        foreach (uint key in keys) sketch.Add(key);
        return sketch;
    }

    public static void Main()
    {
        Check(HyperLogLog.Mix32(0) == 0, "mix32(0)");
        Check(HyperLogLog.RankOf(0) == HyperLogLog.RankWidth + 1, "rank(0)");
        Check(HyperLogLog.RankOf(1u << (HyperLogLog.RankWidth - 1)) == 1, "rank msb");
        Check(HyperLogLog.RankOf(1) == HyperLogLog.RankWidth, "rank lsb");
        Check(HyperLogLog.RankOf(0b101) == HyperLogLog.RankWidth - 2, "rank 101");

        double sigma = 1.04 / Math.Sqrt(HyperLogLog.M);
        var sizes = new[] { 100, 1000, 5000, 20000 };
        const int pools = 16;
        var meanEstimates = new double[sizes.Length];
        double maxMeanError = 0, maxSingleError = 0;
        for (int si = 0; si < sizes.Length; si++)
        {
            int size = sizes[si];
            double sum = 0;
            for (int pool = 0; pool < pools; pool++)
            {
                double est = Digest(PoolKeys(size, pool, (pool + 1) * 131)).Estimate();
                sum += est;
                double singleError = Math.Abs(est - size) / size;
                maxSingleError = Math.Max(maxSingleError, singleError);
                Check(singleError <= 3 * sigma, "3-sigma size " + size + " pool " + pool);
            }
            double mean = sum / pools;
            meanEstimates[si] = mean;
            double meanError = Math.Abs(mean - size) / size;
            maxMeanError = Math.Max(maxMeanError, meanError);
            Check(meanError <= sigma, "mean error size " + size); // documented 1.04/sqrt(m)
        }
        for (int i = 1; i < meanEstimates.Length; i++)
            Check(meanEstimates[i - 1] < meanEstimates[i], "estimates grow");

        var keys = PoolKeys(3000, 7, 4242);
        var merged = Digest(keys[..1500]);
        var second = Digest(keys[1500..]);
        merged.Merge(second);
        var whole = Digest(keys);
        bool sameRegisters = true;
        for (int i = 0; i < HyperLogLog.M; i++)
            if (merged.Registers[i] != whole.Registers[i]) sameRegisters = false;
        Check(sameRegisters, "merge registers");
        Check(merged.Estimate() == whole.Estimate(), "merge estimate");
        Check(maxMeanError <= 0.02 && maxSingleError <= 0.12, "measured margins");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
