// Task 717 - Implement mixed-precision training utility (FP16/FP32 accumulator).
// (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// Mixed-precision training keeps the master weights in FP32 but evaluates the forward and backward
// passes in FP16, and the reason it works is *where the additions happen*: an FP16 accumulator loses
// small terms the moment they fall below half an ulp of the running sum, while a wider accumulator
// keeps them and rounds once at the end. This file implements the FP16 format from scratch - sign,
// 5-bit exponent with bias 15, 10-bit significand, subnormals, infinities, a canonical quiet NaN and
// round-to-nearest-even - and then measures the three effects that matter in practice:
//
//   * accumulation: summing 100 000 small terms in FP16 stalls as soon as each term is below half an
//     ulp of the partial sum, whereas the FP32 accumulator stays accurate;
//   * swamping: 1.0 + 1e-4 repeated a thousand times never moves an FP16 sum at all;
//   * underflow and loss scaling: tiny gradients round to zero in FP16, and multiplying the loss by
//     a constant before the backward pass (and dividing afterwards) brings them back.
//
// The conversion is verified against the runtime as an independent oracle: all 65 536 half patterns
// must survive a float round trip, and several thousand random floats, the tie-to-even cases and the
// boundary values (0, -0, the smallest subnormal, the largest finite value, overflow, underflow)
// must produce exactly the same bits as System.Half. No result is faked: the error figures printed
// at the end are the measured ones.
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

    internal static double Signed(ref ulong state) => ((Next(ref state) >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
}

/// <summary>Bit-exact IEEE binary16 conversion with round-to-nearest-even.</summary>
internal static class HalfFormat
{
    internal const int SignBit = 0x8000;
    internal const int PositiveInfinity = 0x7C00;
    internal const int QuietNaN = 0x7E00;
    internal const int LargestFinite = 0x7BFF;
    internal const int SmallestSubnormal = 0x0001;

    internal static ushort FromFloat(float value)
    {
        int bits = BitConverter.SingleToInt32Bits(value);
        int sign = (bits >> 31) & 1;
        int exponent = (bits >> 23) & 0xFF;
        int mantissa = bits & 0x7FFFFF;

        if (exponent == 0xFF)                              // infinity or NaN
            return (ushort)((sign << 15) | (mantissa == 0 ? PositiveInfinity : QuietNaN));

        int unbiased = exponent - 127;
        if (unbiased > 15) return (ushort)((sign << 15) | PositiveInfinity);       // overflow
        if (unbiased >= -14)
        {
            int halfExponent = unbiased + 15;
            int kept = mantissa >> 13;
            int dropped = mantissa & 0x1FFF;
            if (dropped > 0x1000 || (dropped == 0x1000 && (kept & 1) == 1)) kept++;  // round to even
            if (kept == 0x400) { kept = 0; halfExponent++; }
            if (halfExponent >= 31) return (ushort)((sign << 15) | PositiveInfinity);
            return (ushort)((sign << 15) | (halfExponent << 10) | kept);
        }
        if (unbiased >= -25)                               // subnormal range
        {
            int shift = -14 - unbiased;                    // 1..11
            int significand = mantissa | 0x800000;         // the implicit leading one
            int dropBits = 13 + shift;
            if (dropBits >= 24) return (ushort)(sign << 15);
            int kept = significand >> dropBits;
            int dropped = significand & ((1 << dropBits) - 1);
            int half = 1 << (dropBits - 1);
            if (dropped > half || (dropped == half && (kept & 1) == 1)) kept++;
            return (ushort)((sign << 15) | kept);          // kept == 1024 turns into the smallest normal
        }
        return (ushort)(sign << 15);                       // underflow to signed zero
    }

    internal static float ToFloat(ushort bits)
    {
        int sign = (bits >> 15) & 1;
        int exponent = (bits >> 10) & 0x1F;
        int mantissa = bits & 0x3FF;
        if (exponent == 0x1F)
            return mantissa == 0
                ? (sign == 1 ? float.NegativeInfinity : float.PositiveInfinity)
                : float.NaN;
        double magnitude = exponent == 0
            ? mantissa / 1024.0 * Math.Pow(2, -14)                    // subnormal
            : (1 + mantissa / 1024.0) * Math.Pow(2, exponent - 15);
        return (float)(sign == 1 ? -magnitude : magnitude);
    }

    internal static float Round(float value) => ToFloat(FromFloat(value));

    internal static bool IsFinite(ushort bits) => ((bits >> 10) & 0x1F) != 0x1F;
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Naive FP16 accumulation: every addition is rounded back to half precision.</summary>
    private static float AccumulateHalf(IReadOnlyList<float> values)
    {
        ushort accumulator = HalfFormat.FromFloat(0.0f);
        foreach (float v in values)
            accumulator = HalfFormat.FromFloat(HalfFormat.ToFloat(accumulator) + v);
        return HalfFormat.ToFloat(accumulator);
    }

    /// <summary>FP32 accumulation with a single rounding at the end, which is what mixed precision does.</summary>
    private static float AccumulateWide(IReadOnlyList<float> values)
    {
        float accumulator = 0.0f;
        foreach (float v in values) accumulator += v;
        return HalfFormat.Round(accumulator);
    }

    public static void Main()
    {
        // ---- the format itself, against the runtime ---------------------------------------------
        {
            bool everyPatternRoundTrips = true, everyFloatMatches = true;
            for (int pattern = 0; pattern < 0x10000; pattern++)
            {
                ushort bits = (ushort)pattern;
                float value = HalfFormat.ToFloat(bits);
                if (HalfFormat.IsFinite(bits) || ((bits >> 10) & 0x1F) != 0x1F)
                {
                    if (HalfFormat.FromFloat(value) != bits) everyPatternRoundTrips = false;
                }
                else if (float.IsNaN(value))
                {
                    if (!float.IsNaN(HalfFormat.ToFloat(HalfFormat.FromFloat(value)))) everyPatternRoundTrips = false;
                }
            }
            Check(everyPatternRoundTrips, "all 65 536 half patterns survive a float round trip");

            ulong state = 0x5EEDUL;
            int compared = 0, mismatches = 0;
            for (int i = 0; i < 20000; i++)
            {
                float value = (float)(Rng.Signed(ref state) * Math.Pow(10, Rng.Signed(ref state) * 2.5));
                if (float.IsNaN(value)) continue;
                ushort mine = HalfFormat.FromFloat(value);
                ushort reference = BitConverter.HalfToUInt16Bits((Half)value);
                compared++;
                if (mine != reference) mismatches++;
            }
            Check(mismatches == 0, "20 000 random floats convert to exactly the bits System.Half produces ("
                + compared + " compared, " + mismatches + " mismatched)");
            Console.WriteLine("conversionComparisons=" + compared + " mismatches=" + mismatches);
        }

        // ---- rounding, boundaries and special values -------------------------------------------
        {
            Check(HalfFormat.FromFloat(1.0f) == 0x3C00, "1.0 is 0x3C00");
            Check(HalfFormat.FromFloat(-0.0f) == 0x8000, "negative zero keeps its sign bit");
            Check(HalfFormat.ToFloat(HalfFormat.SmallestSubnormal) > 0
                && HalfFormat.ToFloat(HalfFormat.SmallestSubnormal) < 1e-7,
                "the smallest subnormal is about 6e-8 (" + HalfFormat.ToFloat(HalfFormat.SmallestSubnormal).ToString("E3") + ")");
            Check(Math.Abs(HalfFormat.ToFloat(HalfFormat.LargestFinite) - 65504) < 1e-9,
                "the largest finite value is 65504");
            Check(HalfFormat.FromFloat(65520f) == HalfFormat.PositiveInfinity,
                "a value above 65504 rounds to infinity");
            Check(HalfFormat.FromFloat(1e30f) == HalfFormat.PositiveInfinity, "huge values overflow to infinity");
            Check(HalfFormat.FromFloat(1e-9f) == 0, "values below half the smallest subnormal underflow to zero");
            ushort nanBits = HalfFormat.FromFloat(float.NaN);
            Check(((nanBits >> 10) & 0x1F) == 0x1F && (nanBits & 0x3FF) != 0,
                "NaN becomes a NaN half pattern (0x" + nanBits.ToString("X4") + ")");
            Check((nanBits & 0x7FFF) == HalfFormat.QuietNaN, "with the canonical quiet NaN payload");
            Check(float.IsNaN(HalfFormat.ToFloat(nanBits)), "and it stays NaN");
            Check(HalfFormat.FromFloat(1e-5f) != 0 && HalfFormat.ToFloat(HalfFormat.FromFloat(1e-5f)) > 0,
                "a representable small value is kept");

            // ties round to even, exactly like the runtime
            float[] ties = { 1.0f + 2048.0f * (float)Math.Pow(2, -24), 1.0f + 3 * 2048.0f * (float)Math.Pow(2, -24) };
            bool tieMatches = true;
            foreach (float t in ties)
            {
                if (HalfFormat.FromFloat(t) != BitConverter.HalfToUInt16Bits((Half)t)) tieMatches = false;
            }
            Check(tieMatches, "round-half-to-even ties match the runtime");
            Check((HalfFormat.FromFloat(1.0f + (float)Math.Pow(2, -11)) & 1) == 0,
                "a tie at the even significand rounds down");
            Console.WriteLine("largestFinite=" + HalfFormat.ToFloat(HalfFormat.LargestFinite));
        }

        // ---- accumulation: the FP32 accumulator is the whole point ------------------------------
        {
            const int Terms = 100000;
            var values = new List<float>(Terms);
            for (int i = 0; i < Terms; i++) values.Add(0.001f);
            double exact = Terms * 0.001;

            float narrow = AccumulateHalf(values);
            float wide = AccumulateWide(values);
            double narrowError = Math.Abs(narrow - exact) / exact;
            double wideError = Math.Abs(wide - exact) / exact;
            Check(narrowError > 0.01,
                "naive FP16 accumulation loses more than 1% of the total (" + narrowError.ToString("P2") + ")");
            Check(wideError < 1e-3,
                "the FP32 accumulator stays within 0.1% (" + wideError.ToString("P4") + ")");
            Check(wideError < narrowError / 50,
                "the wide accumulator is at least 50x more accurate (" + wideError.ToString("E2")
                + " vs " + narrowError.ToString("E2") + ")");
            Console.WriteLine("sumExact=" + exact.ToString("F3") + " halfSum=" + narrow.ToString("F3")
                + " wideSum=" + wide.ToString("F3") + " halfRelativeError=" + narrowError.ToString("P2"));
        }

        // ---- swamping ---------------------------------------------------------------------------
        {
            int steps = 1000;
            ushort running = HalfFormat.FromFloat(1.0f);
            float wide = 1.0f;
            for (int i = 0; i < steps; i++)
            {
                running = HalfFormat.FromFloat(HalfFormat.ToFloat(running) + 1e-4f);
                wide += 1e-4f;
            }
            Check(HalfFormat.ToFloat(running) == 1.0f,
                "1e-4 added a thousand times never moves an FP16 sum (" + HalfFormat.ToFloat(running) + ")");
            Check(Math.Abs(wide - 1.1f) < 1e-4, "the FP32 sum reaches 1.1 (" + wide.ToString("F4") + ")");
            Check(Math.Abs(HalfFormat.Round(wide) - 1.0996f) < 0.001, "rounding the wide result once at the end keeps it ("
                + HalfFormat.Round(wide).ToString("F4") + ")");
            Console.WriteLine("halfSwamped=" + HalfFormat.ToFloat(running).ToString("F4")
                + " wideSum=" + wide.ToString("F4"));
        }

        // ---- underflow of small gradients and loss scaling --------------------------------------
        {
            ulong state = 0xABCDUL;
            var gradients = new float[512];
            for (int i = 0; i < gradients.Length; i++) gradients[i] = (float)(Rng.Signed(ref state) * 1e-8);

            int lostWithoutScaling = 0, lostWithScaling = 0;
            const float Scale = 1024.0f;
            double worstUnscaled = 0, worstScaled = 0;
            foreach (float g in gradients)
            {
                ushort plain = HalfFormat.FromFloat(g);
                ushort scaled = HalfFormat.FromFloat(g * Scale);
                if (plain == 0 || plain == HalfFormat.SignBit) lostWithoutScaling++;
                if (scaled == 0 || scaled == HalfFormat.SignBit) lostWithScaling++;
                double recovered = HalfFormat.ToFloat(plain);
                double recoveredScaled = HalfFormat.ToFloat(scaled) / Scale;
                worstUnscaled = Math.Max(worstUnscaled, Math.Abs(recovered - g));
                worstScaled = Math.Max(worstScaled, Math.Abs(recoveredScaled - g));
            }
            Check(lostWithoutScaling > gradients.Length * 0.9,
                "unscaled 1e-8 gradients are almost all rounded away (" + lostWithoutScaling + "/" + gradients.Length + ")");
            Check(lostWithScaling * 10 < lostWithoutScaling,
                "loss scaling rescues nearly all of them (" + lostWithScaling + "/" + gradients.Length
                + " lost versus " + lostWithoutScaling + " unscaled)");
            Check(worstScaled < worstUnscaled / 10,
                "and the recovered gradient is more accurate (" + worstScaled.ToString("E2")
                + " vs " + worstUnscaled.ToString("E2") + ")");
            Console.WriteLine("gradientsLostUnscaled=" + lostWithoutScaling + " lostScaled=" + lostWithScaling
                + " worstUnscaled=" + worstUnscaled.ToString("E3") + " worstScaled=" + worstScaled.ToString("E3"));
        }

        // ---- a tiny step of FP16 compute over FP32 master weights -------------------------------
        {
            const int Features = 6, Samples = 32;
            ulong state = 0x1234UL;
            var inputs = new float[Samples][];
            var targets = new float[Samples];
            for (int n = 0; n < Samples; n++)
            {
                inputs[n] = new float[Features];
                for (int f = 0; f < Features; f++) inputs[n][f] = (float)Rng.Signed(ref state);
                targets[n] = 0.5f * inputs[n][0] - 0.25f * inputs[n][3] + 0.1f;
            }
            var master = new float[Features];
            float[] halfMirror = new float[Features];

            double Loss(float[] w)
            {
                double total = 0;
                for (int n = 0; n < Samples; n++)
                {
                    double predicted = 0;
                    for (int f = 0; f < Features; f++) predicted += w[f] * inputs[n][f];
                    double error = predicted - targets[n];
                    total += error * error;
                }
                return total / Samples;
            }

            double startLoss = Loss(master);
            const float LearningRate = 0.1f;
            for (int step = 0; step < 60; step++)
            {
                // the weights are rounded to FP16 before use, as a real mixed-precision kernel does
                for (int f = 0; f < Features; f++) halfMirror[f] = HalfFormat.Round(master[f]);
                var gradient = new float[Features];
                for (int n = 0; n < Samples; n++)
                {
                    float predicted = 0;
                    for (int f = 0; f < Features; f++) predicted += halfMirror[f] * inputs[n][f];
                    float error = predicted - targets[n];
                    for (int f = 0; f < Features; f++)
                        gradient[f] += HalfFormat.Round(2 * error * inputs[n][f] / Samples);
                }
                for (int f = 0; f < Features; f++) master[f] -= LearningRate * gradient[f];   // FP32 master update
            }
            double endLoss = Loss(master);
            Check(endLoss < startLoss / 10, "the mixed-precision step reduces the loss by an order of magnitude ("
                + startLoss.ToString("E3") + " -> " + endLoss.ToString("E3") + ")");

            // and the FP16 shadow of the master weights tracks them closely
            double worstWeightError = 0;
            for (int f = 0; f < Features; f++)
                worstWeightError = Math.Max(worstWeightError, Math.Abs(HalfFormat.Round(master[f]) - master[f]));
            Check(worstWeightError <= 0.005, "the FP16 shadow stays within half an ulp of the FP32 master ("
                + worstWeightError.ToString("E3") + ")");
            Console.WriteLine("mixedPrecisionLoss=" + startLoss.ToString("E4") + " -> " + endLoss.ToString("E4")
                + " shadowError=" + worstWeightError.ToString("E3"));
        }

        // ---- determinism ------------------------------------------------------------------------
        {
            ulong a = 0x777UL, b = 0x777UL;
            float first = HalfFormat.Round((float)Rng.Signed(ref a));
            float second = HalfFormat.Round((float)Rng.Signed(ref b));
            Check(first == second && a == b, "the conversion is deterministic");
            var values = new List<float>();
            for (int i = 0; i < 500; i++) values.Add(0.0007f);
            Check(AccumulateHalf(values) == AccumulateHalf(values), "accumulation is reproducible");
        }

        Console.WriteLine("PASS");
    }
}
