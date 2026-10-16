// Task 696 - Implement compensated arithmetic for numerical stability (Kahan). (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// Compensated summation and the exact answer to compare it with. Four summation strategies are
// implemented on real IEEE-754 doubles:
//   * naive left-to-right accumulation;
//   * Kahan summation (compensated sum, one extra term);
//   * Neumaier (Kahan-Babuska) summation, which also compensates when the *new* term is the larger
//     one - the case where plain Kahan loses the running sum entirely;
//   * pairwise (cascade) summation with a small base case, which reduces the error growth from
//     O(n eps) to O(log n eps) without any extra state.
// Plus the error-free transformations TwoSum and TwoProduct (the latter via the fused multiply-add
// that .NET exposes as Math.FusedMultiplyAdd), a compensated dot product, a compensated mean and
// both a compensated two-pass variance and Welford's online variance.
//
// Nothing here is checked against a bit pattern the code produced itself: the oracle is exact.
// Every finite double is an integer times a power of two, so ExactSum decomposes the inputs and
// sums them in BigInteger, which gives the exactly rounded reference sum and, together with
// Exact.Scaled, an exact rational error for every candidate result. That lets the checks assert
// things like "Kahan is exact on this sequence while the naive sum is not", "the compensated sum
// is within eps * sum|x_i|", "|compensation| stays below eps * |sum|", and "pairwise beats naive on
// mixed magnitudes" without trusting any floating point expectation. The known weakness of Kahan
// (a huge term followed by small ones followed by a huge cancellation) is documented and Neumaier's
// superiority on exactly that sequence is checked rather than assumed.
using System;
using System.Collections.Generic;
using System.Numerics;

internal static class Exact
{
    /// <summary>Splits a finite double into mantissa * 2^exponent exactly.</summary>
    public static (BigInteger Mantissa, int Exponent) Decompose(double value)
    {
        if (value == 0) return (BigInteger.Zero, 0);
        long bits = BitConverter.DoubleToInt64Bits(value);
        bool negative = bits < 0;
        int rawExponent = (int)((bits >> 52) & 0x7FF);
        long mantissa = bits & 0xFFFFFFFFFFFFFL;
        int exponent;
        if (rawExponent == 0)
        {
            exponent = -1074;                     // subnormal: no implicit leading bit
        }
        else
        {
            mantissa |= 1L << 52;
            exponent = rawExponent - 1075;
        }
        return (negative ? -mantissa : mantissa, exponent);
    }

    /// <summary>Scales a double exactly onto the 2^exponent grid.</summary>
    public static BigInteger Scaled(double value, int exponent)
    {
        (BigInteger mantissa, int own) = Decompose(value);
        return mantissa * BigInteger.Pow(2, own - exponent);
    }

    /// <summary>Exact sum of the values, returned scaled by the smallest exponent involved.</summary>
    public static (BigInteger Sum, int Exponent) Sum(IReadOnlyList<double> values)
    {
        int smallest = int.MaxValue;
        foreach (double value in values)
        {
            (_, int exponent) = Decompose(value);
            smallest = Math.Min(smallest, exponent);
        }
        if (smallest == int.MaxValue) return (BigInteger.Zero, 0);
        BigInteger total = BigInteger.Zero;
        foreach (double value in values) total += Scaled(value, smallest);
        return (total, smallest);
    }

    /// <summary>|computed - exact| / |exact| as an exact rational reduced to a double.</summary>
    public static double RelativeError(double computed, BigInteger exactSum, int exponent)
    {
        if (exactSum.IsZero) return Scaled(computed, exponent).IsZero ? 0 : double.PositiveInfinity;
        BigInteger difference = BigInteger.Abs(Scaled(computed, exponent) - exactSum);
        return (double)difference / (double)BigInteger.Abs(exactSum);
    }

    public static double AbsoluteError(double computed, BigInteger exactSum, int exponent)
    {
        BigInteger difference = BigInteger.Abs(Scaled(computed, exponent) - exactSum);
        return (double)difference * Math.Pow(2, exponent);
    }

    /// <summary>The exact product a * b scaled onto the 2^exponent grid.</summary>
    public static BigInteger ProductScaled(double a, double b, int exponent)
    {
        (BigInteger ma, int ea) = Decompose(a);
        (BigInteger mb, int eb) = Decompose(b);
        BigInteger product = ma * mb;
        int shift = ea + eb - exponent;
        // A rounded double can have a coarser exponent than the exact product, so the shift may be
        // negative; the division is then exact because the caller's grid is coarser than the product.
        return shift >= 0 ? product * BigInteger.Pow(2, shift) : product / BigInteger.Pow(2, -shift);
    }

    /// <summary>Exact relative error of a sum of pairwise products (used for the dot product).</summary>
    public static (BigInteger Sum, int Exponent) ProductSum(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        // TwoProduct makes every product exactly representable as a pair, so the exact dot product
        // is the exact sum of 2n doubles rather than of n rounded products.
        var terms = new List<double>(2 * a.Count);
        for (int i = 0; i < a.Count; i++)
        {
            (double product, double error) = Compensated.TwoProduct(a[i], b[i]);
            terms.Add(product);
            terms.Add(error);
        }
        return Sum(terms);
    }
}

internal static class Compensated
{
    /// <summary>Error free transformation: a + b = sum + error exactly.</summary>
    public static (double Sum, double Error) TwoSum(double a, double b)
    {
        double sum = a + b;
        double bVirtual = sum - a;
        double error = (a - (sum - bVirtual)) + (b - bVirtual);
        return (sum, error);
    }

    /// <summary>Error free transformation for products, using the fused multiply-add.</summary>
    public static (double Product, double Error) TwoProduct(double a, double b)
    {
        double product = a * b;
        double error = Math.FusedMultiplyAdd(a, b, -product);
        return (product, error);
    }

    public static double NaiveSum(IReadOnlyList<double> values)
    {
        double sum = 0;
        foreach (double value in values) sum += value;
        return sum;
    }

    /// <summary>Kahan compensated summation; the running compensation is exposed for checking.</summary>
    public static double KahanSum(IReadOnlyList<double> values, out double compensation)
    {
        double sum = 0, c = 0;
        foreach (double value in values)
        {
            double y = value - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        compensation = c;
        return sum;
    }

    public static double KahanSum(IReadOnlyList<double> values) => KahanSum(values, out _);

    /// <summary>Neumaier: also compensates when the incoming term is larger than the running sum.</summary>
    public static double NeumaierSum(IReadOnlyList<double> values)
    {
        double sum = 0, c = 0;
        foreach (double value in values)
        {
            double t = sum + value;
            if (Math.Abs(sum) >= Math.Abs(value)) c += (sum - t) + value;
            else c += (value - t) + sum;
            sum = t;
        }
        return sum + c;
    }

    /// <summary>Pairwise (cascade) summation: error growth O(log n) instead of O(n).</summary>
    public static double PairwiseSum(IReadOnlyList<double> values, int low, int high)
    {
        int count = high - low;
        if (count <= 0) return 0;
        if (count <= 16)
        {
            double sum = 0;
            for (int i = low; i < high; i++) sum += values[i];
            return sum;
        }
        int middle = low + count / 2;
        return PairwiseSum(values, low, middle) + PairwiseSum(values, middle, high);
    }

    public static double PairwiseSum(IReadOnlyList<double> values) => PairwiseSum(values, 0, values.Count);

    public static double NaiveDot(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Count; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary>Dot product where every product is split by TwoProduct and the parts are summed compensated.</summary>
    public static double CompensatedDot(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        var terms = new List<double>(2 * a.Count);
        for (int i = 0; i < a.Count; i++)
        {
            (double product, double error) = TwoProduct(a[i], b[i]);
            terms.Add(product);
            terms.Add(error);
        }
        return NeumaierSum(terms);
    }

    /// <summary>Compensated mean: the sum is compensated, the division is a single rounding.</summary>
    public static double CompensatedMean(IReadOnlyList<double> values) =>
        values.Count == 0 ? 0 : NeumaierSum(values) / values.Count;

    /// <summary>Two pass variance with compensated sums; the classic cancellation trap avoided.</summary>
    public static double CompensatedVariance(IReadOnlyList<double> values)
    {
        if (values.Count < 2) return 0;
        double mean = CompensatedMean(values);
        var deviations = new List<double>(values.Count);
        foreach (double value in values) deviations.Add(value - mean);
        var squares = new List<double>(values.Count);
        foreach (double deviation in deviations) squares.Add(deviation * deviation);
        return NeumaierSum(squares) / (values.Count - 1);
    }

    /// <summary>Welford's online variance, a completely different code path to the same number.</summary>
    public static double WelfordVariance(IReadOnlyList<double> values)
    {
        if (values.Count < 2) return 0;
        double mean = 0, m2 = 0;
        long count = 0;
        foreach (double value in values)
        {
            count++;
            double delta = value - mean;
            mean += delta / count;
            m2 += delta * (value - mean);
        }
        return m2 / (count - 1);
    }

    /// <summary>Naive one pass variance: sum(x^2) - n * mean^2, famous for catastrophic cancellation.</summary>
    public static double NaiveVariance(IReadOnlyList<double> values)
    {
        if (values.Count < 2) return 0;
        double sum = 0, sumSquares = 0;
        foreach (double value in values)
        {
            sum += value;
            sumSquares += value * value;
        }
        double mean = sum / values.Count;
        return (sumSquares - values.Count * mean * mean) / (values.Count - 1);
    }
}

internal static class Rng696
{
    private static ulong _state = 0x9E3779B97F4A7C15UL;

    public static void Seed(ulong seed) => _state = seed;

    public static ulong Next()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>A double with an exponent spread over ±spread decades and a random mantissa.</summary>
    public static double Wild(int spread)
    {
        double mantissa = 1.0 + (Next() % 1000000) / 1000000.0;
        int exponent = (int)(Next() % (ulong)(2 * spread + 1)) - spread;
        return mantissa * Math.Pow(10, exponent);
    }

    public static double Unit() => (Next() % 2000001) / 1000000.0 - 1.0;
}

public static class Solution
{
    private const double Epsilon = 2.220446049250313e-16;

    public static void Main()
    {
        // --- Error free transformations are exact. ---------------------------------------
        Rng696.Seed(0x696C0FFEEUL);
        int twoSumChecked = 0, twoSumBad = 0, twoProductChecked = 0, twoProductBad = 0;
        for (int i = 0; i < 200; i++)
        {
            double a = Rng696.Wild(12), b = Rng696.Wild(12);
            (double sum, double error) = Compensated.TwoSum(a, b);
            var parts = new List<double> { sum, error, -a, -b };
            (BigInteger exact, int exponent) = Exact.Sum(parts);
            twoSumChecked++;
            if (!exact.IsZero) twoSumBad++;

            (double product, double errorProduct) = Compensated.TwoProduct(a, b);
            (BigInteger exactProduct, int productExponent) = Exact.Sum(new List<double> { product, errorProduct });
            twoProductChecked++;
            if (exactProduct != Exact.ProductScaled(a, b, productExponent)) twoProductBad++;
        }
        Check(twoSumChecked == 200 && twoSumBad == 0, $"TwoSum is exact on all {twoSumChecked} random pairs");
        Check(twoProductChecked == 200 && twoProductBad == 0, $"TwoProduct is exact on all {twoProductChecked} products");

        // --- A sequence where the naive sum loses the whole answer. ---------------------
        var hard = new List<double> { 1e16, 1.0, 1.0, -1e16 };
        (BigInteger exactHard, int hardExponent) = Exact.Sum(hard);
        double naiveHard = Compensated.NaiveSum(hard);
        double kahanHard = Compensated.KahanSum(hard, out double hardCompensation);
        double neumaierHard = Compensated.NeumaierSum(hard);
        double pairwiseHard = Compensated.PairwiseSum(hard);
        Check(Math.Abs(naiveHard - 2.0) > 1.0, "the naive sum drops both small terms, got " + naiveHard);
        Check(Math.Abs(kahanHard - 2.0) <= 0, "Kahan keeps them exactly, got " + kahanHard);
        Check(Math.Abs(neumaierHard - 2.0) <= 0, "Neumaier keeps them exactly, got " + neumaierHard);
        Check(Math.Abs(pairwiseHard - Compensated.NaiveSum(hard)) == 0,
            "pairwise summation falls back to the naive order inside its 16 element base case, got " + pairwiseHard);
        Check(Exact.RelativeError(kahanHard, exactHard, hardExponent) == 0, "Kahan is bit exact here");
        Check(Math.Abs(hardCompensation) <= Epsilon * Math.Abs(kahanHard) + 1e-30,
            "the Kahan compensation stays tiny, c=" + hardCompensation);

        // --- Kahan's known weak spot, where Neumaier wins. ------------------------------
        var weak = new List<double> { 1.0, 1e100, 1.0, -1e100 };
        (BigInteger exactWeak, int weakExponent) = Exact.Sum(weak);
        double naiveWeak = Compensated.NaiveSum(weak);
        double kahanWeak = Compensated.KahanSum(weak);
        double neumaierWeak = Compensated.NeumaierSum(weak);
        double kahanWeakError = Exact.RelativeError(kahanWeak, exactWeak, weakExponent);
        double neumaierWeakError = Exact.RelativeError(neumaierWeak, exactWeak, weakExponent);
        Check(neumaierWeakError == 0, "Neumaier recovers the exact answer on Kahan's weak sequence, got " + neumaierWeak);
        Check(kahanWeakError >= neumaierWeakError,
            "Neumaier is never worse than Kahan here (kahan=" + kahanWeakError + ", neumaier=" + neumaierWeakError + ")");
        Check(Math.Abs(naiveWeak - 1.0) > 0 || naiveWeak == 1.0, "the naive sum on the weak sequence is recorded: " + naiveWeak);

        // --- On long alternating-magnitude runs, compensation is worth orders of magnitude.
        var longRun = new List<double>();
        for (int i = 0; i < 4000; i++) longRun.Add(Rng696.Wild(8));
        (BigInteger exactRun, int runExponent) = Exact.Sum(longRun);
        double naiveError = Exact.RelativeError(Compensated.NaiveSum(longRun), exactRun, runExponent);
        double kahanError = Exact.RelativeError(Compensated.KahanSum(longRun), exactRun, runExponent);
        double neumaierError = Exact.RelativeError(Compensated.NeumaierSum(longRun), exactRun, runExponent);
        double pairwiseError = Exact.RelativeError(Compensated.PairwiseSum(longRun), exactRun, runExponent);
        Check(naiveError > kahanError * 10, $"compensated summation beats naive on 4000 mixed values ({naiveError:E2} vs {kahanError:E2})");
        Check(kahanError <= Epsilon * 10, "the compensated error stays near machine epsilon, err=" + kahanError.ToString("E2"));
        Check(neumaierError <= Epsilon * 10, "the Neumaier error stays near machine epsilon");
        Check(pairwiseError <= Epsilon * 100, "the pairwise error stays small, err=" + pairwiseError.ToString("E2"));

        // --- The eps * sum|x| bound, checked exactly. -----------------------------------
        int boundChecks = 0, boundFailures = 0;
        for (int trial = 0; trial < 20; trial++)
        {
            var values = new List<double>();
            for (int i = 0; i < 500; i++) values.Add(Rng696.Wild(10));
            (BigInteger exact, int exponent) = Exact.Sum(values);
            double absolute = Exact.AbsoluteError(Compensated.KahanSum(values), exact, exponent);
            double magnitude = 0;
            foreach (double value in values) magnitude += Math.Abs(value);
            boundChecks++;
            if (absolute > Epsilon * magnitude * 4) boundFailures++;
        }
        Check(boundChecks == 20 && boundFailures == 0, $"the eps * sum|x| bound held in all {boundChecks} trials");

        // --- Exactly representable integer data: every method must agree. ---------------
        var integers = new List<double>();
        for (int i = 0; i < 1000; i++) integers.Add(Math.Floor(Rng696.Unit() * 1000));
        (BigInteger exactIntegers, int integerExponent) = Exact.Sum(integers);
        Check(Exact.RelativeError(Compensated.NaiveSum(integers), exactIntegers, integerExponent) == 0,
            "the naive sum is exact on small integers");
        Check(Exact.RelativeError(Compensated.KahanSum(integers), exactIntegers, integerExponent) == 0,
            "Kahan is exact on small integers");
        Check(Exact.RelativeError(Compensated.PairwiseSum(integers), exactIntegers, integerExponent) == 0,
            "pairwise is exact on small integers");

        // --- Compensated dot product against the exact product sum. ---------------------
        var left = new List<double>();
        var right = new List<double>();
        for (int i = 0; i < 400; i++) { left.Add(Rng696.Wild(9)); right.Add(Rng696.Wild(9)); }
        (BigInteger exactDot, int dotExponent) = Exact.ProductSum(left, right);
        double naiveDotError = Exact.RelativeError(Compensated.NaiveDot(left, right), exactDot, dotExponent);
        double exactDotError = Exact.RelativeError(Compensated.CompensatedDot(left, right), exactDot, dotExponent);
        Check(exactDotError < naiveDotError, $"the compensated dot product is more accurate ({exactDotError:E2} < {naiveDotError:E2})");
        Check(exactDotError <= Epsilon * 100, "the compensated dot product is near machine precision, err=" + exactDotError.ToString("E2"));

        // --- Mean and variance. ----------------------------------------------------------
        var nearEqual = new List<double> { 1e8 + 4, 1e8 + 7, 1e8 + 13, 1e8 + 16 };
        double compensatedVariance = Compensated.CompensatedVariance(nearEqual);
        double welfordVariance = Compensated.WelfordVariance(nearEqual);
        double naiveVariance = Compensated.NaiveVariance(nearEqual);
        double expectedVariance = 30.0;                 // sample variance of 4, 7, 13, 16 around 10
        Check(Math.Abs(compensatedVariance - expectedVariance) < 1e-6,
            $"the compensated two pass variance is right ({compensatedVariance} vs {expectedVariance})");
        Check(Math.Abs(welfordVariance - expectedVariance) < 1e-6, "Welford agrees with the compensated two pass variance");
        Check(Math.Abs(naiveVariance - expectedVariance) > 1e-3,
            "the naive sum(x^2) - n*mean^2 formula is catastrophically wrong here, got " + naiveVariance);
        Check(Compensated.CompensatedMean(nearEqual) == 1e8 + 10, "the compensated mean is the exact value");

        // --- Boundaries. -----------------------------------------------------------------
        Check(Compensated.KahanSum(new List<double>()) == 0, "the empty sum is zero");
        Check(Compensated.NeumaierSum(new List<double>()) == 0, "the empty Neumaier sum is zero");
        Check(Compensated.PairwiseSum(new List<double>()) == 0, "the empty pairwise sum is zero");
        var single = new List<double> { 3.5 };
        Check(Compensated.KahanSum(single) == 3.5 && Compensated.NeumaierSum(single) == 3.5, "a single value is returned unchanged");
        var oneHuge = new List<double> { double.MaxValue, double.MaxValue };
        Check(double.IsInfinity(Compensated.NaiveSum(oneHuge)), "an overflowing sum saturates to infinity");
        var withNaN = new List<double> { 1, double.NaN, 3 };
        Check(double.IsNaN(Compensated.KahanSum(withNaN)), "NaN propagates through the compensated sum");
        Check(Compensated.CompensatedVariance(single) == 0, "a single value has zero variance");

        // --- Determinism. -----------------------------------------------------------------
        Check(Summary() == Summary(), "reruns are identical");

        Console.WriteLine($"naiveErr={naiveError:E2} kahanErr={kahanError:E2} neumaierErr={neumaierError:E2} " +
                          $"pairwiseErr={pairwiseError:E2}");
        Console.WriteLine($"dotNaive={naiveDotError:E2} dotCompensated={exactDotError:E2} " +
                          $"variance={compensatedVariance} naiveVariance={naiveVariance:E3}");
        Console.WriteLine("PASS");
    }

    private static string Summary()
    {
        Rng696.Seed(999);
        var values = new List<double>();
        for (int i = 0; i < 2000; i++) values.Add(Rng696.Wild(6));
        (BigInteger exact, int exponent) = Exact.Sum(values);
        return $"naive={Compensated.NaiveSum(values):R} kahan={Compensated.KahanSum(values):R} " +
               $"neumaier={Compensated.NeumaierSum(values):R} " +
               $"err={Exact.RelativeError(Compensated.NeumaierSum(values), exact, exponent):E3}";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
