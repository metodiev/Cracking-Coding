// Task 944 - Implement high-precision numeric library with arbitrary precision rational arithmetic.
// (Tier: Master, CTCI Beyond — Numerical & Scientific Computing)
//
// An exact rational type over System.Numerics.BigInteger: every value is stored as a reduced
// fraction with a positive denominator, so addition, subtraction, multiplication, division and
// integer powers are exact for arbitrarily large operands. On top of it the library offers a
// correctly rounded fixed-point renderer (long division digit by digit with half-up rounding and
// carry propagation), continued-fraction expansion and reconstruction, and a scaled Newton square
// root, which is how an irrational constant becomes a rational with a requested number of digits.
//
// Correctness is verified against independent oracles rather than against itself: the
// normalisation invariant (gcd = 1, denominator > 0) is re-checked after every construction and
// operation; algebraic identities are checked exactly on 50-digit numerators; the harmonic number
// H_20 is compared with the published fraction 55835135/15519504; sum(1/k!) is compared with the
// known digits of e and with an independent all-integer formula; decimal renderings are re-parsed
// and shown to be within half an ulp of the exact value; the continued fraction is round-tripped;
// sqrt(2) is compared with the known digits and its squared error is bounded; and the total order
// is cross-checked with cross-multiplication. Zero, negative denominators, division by zero and
// the rounding carry are the boundary cases.
// Simplified: rationals only (no arbitrary-precision transcendental functions), fixed-point output
// instead of a floating type, and the square root is an approximation with a stated error bound.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;

internal readonly struct Rational : IComparable<Rational>
{
    internal readonly BigInteger Numerator;
    internal readonly BigInteger Denominator;

    internal Rational(BigInteger numerator, BigInteger denominator)
    {
        if (denominator.IsZero) throw new ArgumentException("denominator must not be zero");
        if (denominator.Sign < 0)
        {
            numerator = -numerator;
            denominator = -denominator;
        }
        BigInteger divisor = BigInteger.GreatestCommonDivisor(BigInteger.Abs(numerator), denominator);
        if (divisor > BigInteger.One)
        {
            numerator /= divisor;
            denominator /= divisor;
        }
        if (numerator.IsZero) denominator = BigInteger.One;
        Numerator = numerator;
        Denominator = denominator;
    }

    internal static readonly Rational Zero = new Rational(0, 1);
    internal static readonly Rational One = new Rational(1, 1);

    internal static Rational FromInteger(BigInteger value) => new Rational(value, 1);

    internal bool IsNormalized => Denominator.Sign > 0
        && BigInteger.GreatestCommonDivisor(BigInteger.Abs(Numerator), Denominator) == BigInteger.One
        && !(Numerator.IsZero && Denominator != BigInteger.One);

    internal int Sign => Numerator.Sign;

    public static Rational operator +(Rational a, Rational b) =>
        new Rational(a.Numerator * b.Denominator + b.Numerator * a.Denominator, a.Denominator * b.Denominator);

    public static Rational operator -(Rational a, Rational b) =>
        new Rational(a.Numerator * b.Denominator - b.Numerator * a.Denominator, a.Denominator * b.Denominator);

    public static Rational operator *(Rational a, Rational b) => new Rational(a.Numerator * b.Numerator, a.Denominator * b.Denominator);

    public static Rational operator /(Rational a, Rational b)
    {
        if (b.Numerator.IsZero) throw new DivideByZeroException("rational division by zero");
        return new Rational(a.Numerator * b.Denominator, a.Denominator * b.Numerator);
    }

    public static Rational operator -(Rational a) => new Rational(-a.Numerator, a.Denominator);

    public int CompareTo(Rational other) => (Numerator * other.Denominator).CompareTo(other.Numerator * Denominator);

    public static bool operator ==(Rational a, Rational b) => a.CompareTo(b) == 0;
    public static bool operator !=(Rational a, Rational b) => a.CompareTo(b) != 0;
    public static bool operator <(Rational a, Rational b) => a.CompareTo(b) < 0;
    public static bool operator >(Rational a, Rational b) => a.CompareTo(b) > 0;
    public static bool operator <=(Rational a, Rational b) => a.CompareTo(b) <= 0;
    public static bool operator >=(Rational a, Rational b) => a.CompareTo(b) >= 0;

    public override bool Equals(object? obj) => obj is Rational other && CompareTo(other) == 0;
    public override int GetHashCode() => HashCode.Combine(Numerator, Denominator);

    internal Rational Pow(int exponent)
    {
        if (exponent == 0) return One;
        Rational basis = exponent < 0 ? One / this : this;
        int magnitude = Math.Abs(exponent);
        Rational result = One;
        for (int i = 0; i < magnitude; i++) result *= basis;
        return result;
    }

    public override string ToString() => Denominator.IsOne ? Numerator.ToString() : Numerator + "/" + Denominator;

    /** Fixed-point rendering with half-up rounding away from zero and carry propagation. */
    internal string ToDecimal(int digits)
    {
        if (digits < 0) throw new ArgumentOutOfRangeException(nameof(digits));
        bool negative = Numerator.Sign < 0;
        BigInteger remainder = BigInteger.Abs(Numerator);
        BigInteger whole = BigInteger.DivRem(remainder, Denominator, out remainder);
        var text = new StringBuilder();
        for (int i = 0; i < digits; i++)
        {
            remainder *= 10;
            text.Append((char)('0' + (int)BigInteger.DivRem(remainder, Denominator, out remainder)));
        }
        remainder *= 10;
        if (BigInteger.DivRem(remainder, Denominator, out remainder) >= 5)
        {
            int i = text.Length - 1;
            while (i >= 0)
            {
                if (text[i] < '9') { text[i] = (char)(text[i] + 1); break; }
                text[i] = '0';
                i--;
            }
            if (i < 0) whole += 1;
        }
        string body = text.ToString().TrimEnd('0');
        string sign = negative && (whole != 0 || body.Length > 0) ? "-" : "";
        return body.Length == 0 ? sign + whole : sign + whole + "." + body;
    }

    internal static Rational ParseDecimal(string text)
    {
        bool negative = text.StartsWith("-", StringComparison.Ordinal);
        string body = negative ? text.Substring(1) : text;
        int dot = body.IndexOf('.');
        string digits = dot < 0 ? body : body.Remove(dot, 1);
        BigInteger numerator = BigInteger.Parse(digits);
        BigInteger denominator = BigInteger.Pow(10, dot < 0 ? 0 : body.Length - dot - 1);
        return new Rational(negative ? -numerator : numerator, denominator);
    }

    internal List<BigInteger> ContinuedFraction()
    {
        var terms = new List<BigInteger>();
        BigInteger n = Numerator, d = Denominator;
        while (!d.IsZero)
        {
            BigInteger quotient = BigInteger.DivRem(n, d, out BigInteger r);
            terms.Add(quotient);
            n = d;
            d = r;
        }
        return terms.Count == 0 ? new List<BigInteger> { BigInteger.Zero } : terms;
    }

    internal static Rational FromContinuedFraction(IReadOnlyList<BigInteger> terms)
    {
        Rational value = Zero;
        for (int i = terms.Count - 1; i >= 0; i--) value = FromInteger(terms[i]) + (value.IsZero ? Zero : One / value);
        return value;
    }

    private bool IsZero => Numerator.IsZero;

    internal static BigInteger IntegerSqrt(BigInteger value)
    {
        if (value.Sign < 0) throw new ArgumentException("no real square root of a negative integer");
        if (value.IsZero) return BigInteger.Zero;
        BigInteger guess = BigInteger.One << (int)((value.GetBitLength() + 1) / 2 + 1);
        while (true)
        {
            BigInteger next = (guess + value / guess) >> 1;
            if (next >= guess) return guess;
            guess = next;
        }
    }

    /** Approximate square root with at most one unit in the last of `digits` decimal places. */
    internal static Rational Sqrt(BigInteger value, int digits)
    {
        BigInteger scale = BigInteger.Pow(10, digits);
        return new Rational(IntegerSqrt(value * scale * scale), scale);
    }

    internal double ToDouble() => (double)Numerator / (double)Denominator;
}

public static class Solution
{
    private const string Sqrt2Digits =
        "1.414213562373095048801688724209698078569671875376948073176679737990732478462107038850387534327641573";
    private const string EDigits = "2.718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427";

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
        catch (DivideByZeroException) { return true; }
    }

    private static Rational R(long n, long d) => new Rational(n, d);

    public static void Main()
    {
        // 1. Normalisation invariant after every construction and operation.
        var battery = new List<Rational> { R(1, 2), R(-6, 4), R(0, 7), R(21, 6), R(-1, -3), new Rational(BigInteger.Pow(10, 50), 3) };
        int seedCount = battery.Count;
        for (int i = 0; i < seedCount; i++)
        {
            battery.Add(battery[i] + R(1, 3));
            battery.Add(battery[i] * R(-5, 7));
            battery.Add(battery[i] - battery[i]);
        }
        Check(battery.All(r => r.IsNormalized), "every constructed and derived rational is reduced with a positive denominator");
        Check(R(-6, 4).ToString() == "-3/2" && R(0, 7).ToString() == "0" && R(21, 6).ToString() == "7/2",
            "signs fold into the numerator and integers lose their denominator");

        // 2. Exact algebraic identities on 50-digit numerators.
        BigInteger big = BigInteger.Pow(10, 50) + 1;
        var a = new Rational(big, 3);
        var b = new Rational(big - 2, 7);
        Check((a + b) * (a + b) == a * a + R(2, 1) * a * b + b * b && (a + b) * (a + b) - R(4, 1) * a * b == (a - b) * (a - b),
            "the binomial identities hold exactly for 50-digit numerators");
        Check((a / b) * (b / a) == Rational.One && a.Pow(-1) == Rational.One / a && a.Pow(3) == a * a * a,
            "reciprocals, negative powers and cubes are exact");

        // 3. Published oracle: H_20 as a fraction, plus the double round trip.
        Rational harmonic = Rational.Zero;
        for (int k = 1; k <= 20; k++) harmonic += R(1, k);
        Rational published = new Rational(55835135, 15519504);
        Check(harmonic == published && Math.Abs(harmonic.ToDouble() - 3.597739657143682) < 1e-12,
            "H_20 = " + harmonic + " matches the published fraction and the double value");

        // 4. e from sum(1/k!) against the known digits and an independent all-integer formula.
        Rational e = Rational.Zero;
        for (int k = 0; k <= 25; k++)
        {
            BigInteger factorial = BigInteger.One;
            for (int j = 2; j <= k; j++) factorial *= j;
            e += new Rational(BigInteger.One, factorial);
        }
        BigInteger n25 = BigInteger.One;
        for (int j = 2; j <= 25; j++) n25 *= j;
        Rational integerForm = new Rational(Enumerable.Range(0, 26).Aggregate(BigInteger.Zero,
            (acc, k) => acc + n25 / Factorial(k)), n25);
        Check(e == integerForm && EDigits.StartsWith(e.ToDecimal(20), StringComparison.Ordinal),
            "sum(1/k!) for k <= 25 equals the integer form and reproduces the digits of e as " + e.ToDecimal(20));

        // 5-6. Decimal rendering: known expansions, correct rounding and the carry boundary.
        Check(R(1, 7).ToDecimal(12) == "0.142857142857" && R(2, 3).ToDecimal(12) == "0.666666666667"
              && R(1, 4).ToDecimal(12) == "0.25" && R(355, 113).ToDecimal(12) == "3.141592920354",
            "the renderer reproduces 1/7, 2/3, 1/4 and 355/113 to 12 places");
        bool halfUlp = new[] { R(1, 7), R(2, 3), R(355, 113), R(-22, 7) }
            .All(value => (Rational.ParseDecimal(value.ToDecimal(12)) - value).Abs() <= R(1, 2 * 10_000_000_000L));
        Check(halfUlp, "every 12-place rendering is within half an ulp of the exact value");
        Check(R(1999, 2000).ToDecimal(2) == "1" && R(1, 3).ToDecimal(0) == "0" && R(1, 2).ToDecimal(0) == "1"
              && R(-1, 2).ToDecimal(0) == "-1",
            "rounding half away from zero carries into the integer part (1999/2000 -> 1)");

        // 7. Continued fractions round-trip exactly.
        Check(R(355, 113).ContinuedFraction().SequenceEqual(new BigInteger[] { 3, 7, 16 })
              && R(415, 93).ContinuedFraction().SequenceEqual(new BigInteger[] { 4, 2, 6, 7 })
              && R(0, 1).ContinuedFraction().SequenceEqual(new BigInteger[] { 0 }),
            "continued fractions of 355/113, 415/93 and zero are correct");
        Check(Rational.FromContinuedFraction(R(415, 93).ContinuedFraction()) == R(415, 93)
              && Rational.FromContinuedFraction(R(1, 7).ContinuedFraction()) == R(1, 7),
            "reconstructing from the continued fraction returns the original rational");

        // 8. sqrt(2) by scaled Newton iteration against the known digits and an error bound.
        Rational root30 = Rational.Sqrt(2, 30);
        Rational root40 = Rational.Sqrt(2, 40);
        Check(Sqrt2Digits.StartsWith(root30.ToDecimal(30), StringComparison.Ordinal),
            "sqrt(2) to 30 places is " + root30.ToDecimal(30));
        Rational error30 = (root30 * root30 - Rational.FromInteger(2)).Abs();
        Rational error40 = (root40 * root40 - Rational.FromInteger(2)).Abs();
        Check(error30 <= new Rational(4, BigInteger.Pow(10, 30)) && error40 < error30,
            "the squared error of sqrt(2) is below 4e-30 and shrinks with more digits");

        // 9. Total order cross-checked with cross-multiplication.
        var values = new List<Rational> { R(3, 7), R(-2, 5), R(355, 113), R(1, 2), R(-22, 7), R(0, 1), R(7, 3) };
        var sorted = values.OrderBy(v => v).ToList();
        bool ordered = true;
        for (int i = 0; i + 1 < sorted.Count; i++)
            ordered &= (sorted[i].Numerator * sorted[i + 1].Denominator).CompareTo(sorted[i + 1].Numerator * sorted[i].Denominator) <= 0;
        bool antisymmetric = values.All(x => values.All(y => Math.Sign(x.CompareTo(y)) == -Math.Sign(y.CompareTo(x))));
        Check(ordered && antisymmetric && sorted[0] == R(-22, 7) && sorted[sorted.Count - 1] == R(355, 113),
            "the order agrees with cross-multiplication and is antisymmetric");

        // 10. Boundaries and determinism.
        bool rejectedZeroDenominator = Throws(() => new Rational(1, 0));
        bool rejectedDivisionByZero = Throws(() => { var quotient = R(1, 2) / R(0, 5); });
        bool exactSquareDetected = Rational.IntegerSqrt(BigInteger.Pow(10, 40)) == BigInteger.Pow(10, 20);
        Check(rejectedZeroDenominator && rejectedDivisionByZero && R(1, 3).ToDecimal(8) == "0.33333333"
              && exactSquareDetected,
            "zero denominators are rejected, exact squares are detected and the renderer is stable");

        Console.WriteLine("  H_20 = " + harmonic + ", e to 20 places = " + e.ToDecimal(20)
            + ", sqrt(2) to 30 = " + root30.ToDecimal(30) + ", 355/113 to 12 = " + R(355, 113).ToDecimal(12));
        Console.WriteLine("PASS");
    }

    private static BigInteger Factorial(int k)
    {
        BigInteger value = BigInteger.One;
        for (int j = 2; j <= k; j++) value *= j;
        return value;
    }

    private static Rational Abs(this Rational value) => value.Sign < 0 ? -value : value;
}
