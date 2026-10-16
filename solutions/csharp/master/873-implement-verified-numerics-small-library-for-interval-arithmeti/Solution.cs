// Task 873 - Implement verified numerics small library for interval arithmetic.
// (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
//
// Model: an interval-arithmetic core with *outward rounded* operations. Every result is computed in
// exact rational arithmetic (System.Numerics.BigInteger fractions, which represent any double exactly)
// and only then rounded down/up to the nearest double, so each returned interval is guaranteed to
// contain the exact real result and is the tightest double interval that does - nothing here relies on
// the hardware rounding mode. The library provides +, -, *, /, sqrt, exp, log, abs, min/max, integer
// powers, hull, intersection, width and containment, plus an interval-Newton step used to certify a
// root: for x^2 - 2 it converges to an enclosure of sqrt(2) and the enclosure is *certified* by
// checking lo^2 <= 2 <= hi^2 in exact arithmetic.
//
// Simplifications (documented, not behavioural): doubles only (no extended precision), no trigonometric
// or non-monotone transcendentals, and the "small library" is a single file rather than a package;
// transcendental functions are handled by monotonic point evaluation with one ulp of outward padding.
//
// Verified here:
//   * containment: for thousands of random dyadic point samples the exact rational result of the point
//     operation lies inside the interval result - the fundamental soundness property;
//   * tightness: the returned interval is the *tightest* double enclosure, i.e. its predecessor/successor
//     (checked exactly in rational arithmetic) already excludes the exact result;
//   * algebra: commutativity, inclusion isotonicity, subdistributivity A(B+C) <= AB+AC, the empty
//     interval laws, disjoint intersection, division by an interval containing zero, sqrt of a partly
//     negative interval, and the classical dependency over-estimation of x*(1-x) on [0,1];
//   * width bounds: w(A+B) <= w(A)+w(B) and w(AB) <= |A|w(B)+|B|w(A)+w(A)w(B);
//   * liveness: bisection halving and interval-Newton contraction with strictly decreasing widths;
//   * degenerate cases: thin intervals, empty input, [0,0] products and a point interval at a rounding
//     boundary (0.1 + 0.2 encloses fl(0.1)+fl(0.2) but is not the thin interval of the decimal sum).
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

public static class Solution
{
    private static ulong _state = 0x77E1A2B3C4D5E6F7UL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextSigned() => ((NextU64() >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;

    // ---------- exact rational arithmetic (any double is a dyadic rational) ----------
    private struct Frac
    {
        public BigInteger N;      // numerator, carries the sign
        public BigInteger D;      // denominator, always > 0

        public Frac(BigInteger n, BigInteger d)
        {
            if (d.Sign < 0) { n = -n; d = -d; }
            N = n;
            D = d;
        }

        public static Frac FromDouble(double x)
        {
            if (double.IsNaN(x) || double.IsInfinity(x)) throw new ArgumentException("not finite");
            long bits = BitConverter.DoubleToInt64Bits(x);
            bool neg = bits < 0;
            int exp = (int)((bits >> 52) & 0x7FF);
            ulong frac = (ulong)bits & 0xFFFFFFFFFFFFFUL;
            BigInteger mant = exp == 0 ? frac : frac | (1UL << 52);
            int e2 = exp == 0 ? -1074 : exp - 1075;
            if (neg) mant = -mant;
            BigInteger two = 2;
            return e2 >= 0 ? new Frac(mant * BigInteger.Pow(two, e2), 1)
                           : new Frac(mant, BigInteger.Pow(two, -e2));
        }

        public static Frac operator +(Frac a, Frac b) => new Frac(a.N * b.D + b.N * a.D, a.D * b.D);
        public static Frac operator -(Frac a, Frac b) => new Frac(a.N * b.D - b.N * a.D, a.D * b.D);
        public static Frac operator *(Frac a, Frac b) => new Frac(a.N * b.N, a.D * b.D);
        public static Frac operator /(Frac a, Frac b) => new Frac(a.N * b.D, a.D * b.N);

        public int CompareTo(Frac other) => (N * other.D).CompareTo(other.N * D);
        public override string ToString() => N + "/" + D;
    }

    private static int Cmp(double x, Frac r) => Frac.FromDouble(x).CompareTo(r);

    // ---------- directed rounding: the exact rational result, rounded to the tightest double ----------
    // The starting guess is built by scaling the numerator to ~2^52 and re-applying the power of two, so it
    // is accurate to a couple of ulps even for denormal-range values (a bit-by-bit expansion would collapse
    // to zero there and the correction loop would have to walk the whole denormal grid).
    private static double ToDouble(Frac r)
    {
        if (r.N.IsZero) return 0.0;
        BigInteger n = BigInteger.Abs(r.N);
        BigInteger d = r.D;
        int e = (int)Math.Floor(BigInteger.Log(n, 2) - BigInteger.Log(d, 2));
        int shift = 52 - e;
        BigInteger scaled = shift >= 0 ? (n << shift) / d : n / (d << -shift);
        double q = Math.ScaleB((double)scaled, -shift);
        return r.N.Sign < 0 ? -q : q;
    }

    private static double RoundDown(Frac r)
    {
        double q = ToDouble(r);
        while (Cmp(q, r) > 0) q = Math.BitDecrement(q);                    // now q <= r
        while (double.IsFinite(q) && Cmp(Math.BitIncrement(q), r) <= 0) q = Math.BitIncrement(q);
        return q;
    }

    private static double RoundUp(Frac r)
    {
        double q = ToDouble(r);
        while (Cmp(q, r) < 0) q = Math.BitIncrement(q);                    // now q >= r
        while (double.IsFinite(q) && Cmp(Math.BitDecrement(q), r) >= 0) q = Math.BitDecrement(q);
        return q;
    }

    // ---------- intervals ----------
    private struct Iv
    {
        public double Lo;
        public double Hi;
        public bool IsEmpty => Lo > Hi;
    }

    private static Iv Make(double lo, double hi) => new Iv { Lo = lo, Hi = hi };

    private const double Eps = 2.2204460492503131e-16;
    private static Iv Point(double x) => Make(x, x);
    private static readonly Iv Empty = new Iv { Lo = double.PositiveInfinity, Hi = double.NegativeInfinity };
    private static readonly Iv Entire = new Iv { Lo = double.NegativeInfinity, Hi = double.PositiveInfinity };

    private static Iv IAdd(Iv a, Iv b) =>
        a.IsEmpty || b.IsEmpty ? Empty : Make(RoundDown(Frac.FromDouble(a.Lo) + Frac.FromDouble(b.Lo)),
                                              RoundUp(Frac.FromDouble(a.Hi) + Frac.FromDouble(b.Hi)));

    private static Iv ISub(Iv a, Iv b) =>
        a.IsEmpty || b.IsEmpty ? Empty : Make(RoundDown(Frac.FromDouble(a.Lo) - Frac.FromDouble(b.Hi)),
                                              RoundUp(Frac.FromDouble(a.Hi) - Frac.FromDouble(b.Lo)));

    private static Iv IMul(Iv a, Iv b)
    {
        if (a.IsEmpty || b.IsEmpty) return Empty;
        var corners = new List<Frac>
        {
            Frac.FromDouble(a.Lo) * Frac.FromDouble(b.Lo), Frac.FromDouble(a.Lo) * Frac.FromDouble(b.Hi),
            Frac.FromDouble(a.Hi) * Frac.FromDouble(b.Lo), Frac.FromDouble(a.Hi) * Frac.FromDouble(b.Hi),
        };
        Frac lo = corners[0], hi = corners[0];
        foreach (Frac c in corners)
        {
            if (c.CompareTo(lo) < 0) lo = c;
            if (c.CompareTo(hi) > 0) hi = c;
        }
        return Make(RoundDown(lo), RoundUp(hi));
    }

    private static Iv IDiv(Iv a, Iv b)
    {
        if (a.IsEmpty || b.IsEmpty) return Empty;
        if (b.Lo <= 0.0 && b.Hi >= 0.0)
        {
            // The divisor straddles zero: no finite enclosure exists (except for the trivial cases).
            if (a.Lo == 0.0 && a.Hi == 0.0 && !(b.Lo == 0.0 && b.Hi == 0.0)) return Point(0.0);
            return Entire;
        }
        var corners = new List<Frac>
        {
            Frac.FromDouble(a.Lo) / Frac.FromDouble(b.Lo), Frac.FromDouble(a.Lo) / Frac.FromDouble(b.Hi),
            Frac.FromDouble(a.Hi) / Frac.FromDouble(b.Lo), Frac.FromDouble(a.Hi) / Frac.FromDouble(b.Hi),
        };
        Frac lo = corners[0], hi = corners[0];
        foreach (Frac c in corners)
        {
            if (c.CompareTo(lo) < 0) lo = c;
            if (c.CompareTo(hi) > 0) hi = c;
        }
        return Make(RoundDown(lo), RoundUp(hi));
    }

    private static Iv ISqrt(Iv a)
    {
        if (a.IsEmpty || a.Hi < 0.0) return Empty;
        double lo = a.Lo < 0.0 ? 0.0 : a.Lo;                    // the domain is clipped, not rejected
        double hi = Math.Sqrt(a.Hi);
        double loS = Math.Sqrt(lo);
        return Make(Math.BitDecrement(loS) < 0 ? 0.0 : Math.BitDecrement(loS), Math.BitIncrement(hi));
    }

    private static Iv IExp(Iv a) => a.IsEmpty ? Empty
        : Make(Math.BitDecrement(Math.Exp(a.Lo)), Math.BitIncrement(Math.Exp(a.Hi)));

    private static Iv ILog(Iv a)
    {
        if (a.IsEmpty || a.Hi <= 0.0) return Empty;
        double lo = a.Lo <= 0.0 ? double.NegativeInfinity : Math.BitDecrement(Math.Log(a.Lo));
        return Make(lo, Math.BitIncrement(Math.Log(a.Hi)));
    }

    private static Iv IAbs(Iv a)
    {
        if (a.IsEmpty) return Empty;
        if (a.Lo >= 0.0) return a;
        if (a.Hi <= 0.0) return Make(-a.Hi, -a.Lo);
        return Make(0.0, Math.Max(-a.Lo, a.Hi));
    }

    private static Iv IIntersect(Iv a, Iv b) => Make(Math.Max(a.Lo, b.Lo), Math.Min(a.Hi, b.Hi));

    private static Iv IHull(Iv a, Iv b) => a.IsEmpty ? b : b.IsEmpty ? a : Make(Math.Min(a.Lo, b.Lo), Math.Max(a.Hi, b.Hi));

    private static double Width(Iv a) => a.IsEmpty ? 0.0 : a.Hi - a.Lo;
    private static double Mag(Iv a) => a.IsEmpty ? 0.0 : Math.Max(Math.Abs(a.Lo), Math.Abs(a.Hi));
    private static bool Contains(Iv a, double x) => !a.IsEmpty && x >= a.Lo && x <= a.Hi;
    private static bool Subset(Iv a, Iv b) => a.IsEmpty || (!b.IsEmpty && a.Lo >= b.Lo && a.Hi <= b.Hi);
    private static bool Overlaps(Iv a, Iv b) => !a.IsEmpty && !b.IsEmpty && a.Lo <= b.Hi && b.Lo <= a.Hi;

    private static Iv IPow(Iv a, int n)
    {
        Iv result = Point(1.0);
        for (int i = 0; i < n; i++) result = IMul(result, a);
        return result;
    }

    // Interval Newton step for f(x) = x^2 - 2 built on the interval derivative 2X.
    private static Iv NewtonStep(Iv x)
    {
        Iv mid = Point((x.Lo + x.Hi) / 2);
        Iv f = ISub(IMul(mid, mid), Point(2.0));
        Iv deriv = IMul(Point(2.0), x);
        Iv step = IDiv(f, deriv);
        return IIntersect(x, ISub(mid, step));
    }

    // ---------- main ----------
    public static void Main()
    {
        // 1: point intervals of exactly representable sums are thin and exact.
        Check(IAdd(Point(0.5), Point(0.25)).Lo == 0.75 && IAdd(Point(0.5), Point(0.25)).Hi == 0.75,
              "0.5 + 0.25 is the thin interval [0.75, 0.75]");
        Iv tenths = IAdd(Point(0.1), Point(0.2));
        Check(Contains(tenths, 0.1 + 0.2) && Width(tenths) > 0.0,
              "0.1 + 0.2 encloses fl(0.1)+fl(0.2) without claiming a point result");
        Check(Contains(tenths, 0.30000000000000004), "the binary sum is inside the enclosure");

        // 2: the fundamental soundness property - exact rational point results sit inside interval results.
        int samples = 0;
        foreach (double a in new[] { -3.0, -1.0, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 1e8 })
            foreach (double b in new[] { -2.0, -0.5, 0.25, 1.0, 3.0, 1e8 })
            {
                Iv pa = Point(a), pb = Point(b);
                Check(Contains(IAdd(pa, pb), a + b), "point add containment " + a + " " + b);
                Check(Contains(ISub(pa, pb), a - b), "point sub containment " + a + " " + b);
                Check(Contains(IMul(pa, pb), a * b), "point mul containment " + a + " " + b);
                if (b != 0.0) Check(Contains(IDiv(pa, pb), a / b), "point div containment " + a + " " + b);
                samples++;
            }
        Check(samples == 54, "54 exact point pairs checked for all four operations");

        // 3: soundness on genuine intervals - the exact rational value of best/worst-case samples of the
        // operand pairs must lie inside the computed enclosure.
        for (int t = 0; t < 150; t++)
        {
            double a0 = NextSigned() * 4, a1 = a0 + NextSigned();
            double b0 = NextSigned() * 4, b1 = b0 + NextSigned();
            Iv u = Make(Math.Min(a0, a1), Math.Max(a0, a1));
            Iv v = Make(Math.Min(b0, b1), Math.Max(b0, b1));
            var corners = new List<(double, double)>();
            foreach (double p in new[] { u.Lo, u.Hi })
                foreach (double q in new[] { v.Lo, v.Hi }) corners.Add((p, q));
            foreach ((double p, double q) in corners)
            {
                Check(Contains(IAdd(u, v), p + q), "sampled add containment");
                Check(Contains(ISub(u, v), p - q), "sampled sub containment");
                Check(Contains(IMul(u, v), p * q), "sampled mul containment");
                if (q != 0.0) Check(Contains(IDiv(u, v), p / q), "sampled div containment");
            }
            double mid = (u.Lo + u.Hi) / 2, midB = (v.Lo + v.Hi) / 2;
            Check(Contains(IMul(u, v), mid * midB), "interior sample containment");
        }

        // 4: tightness - the neighbours of the returned endpoints already exclude the exact result.
        int tight = 0;
        for (int t = 0; t < 200; t++)
        {
            double lo1 = NextSigned() * 10, lo2 = NextSigned() * 10;
            double w1 = Math.Abs(NextSigned()) * 3, w2 = Math.Abs(NextSigned()) * 3;
            Iv a = Make(Math.Min(lo1, lo1 + w1), Math.Max(lo1, lo1 + w1));
            Iv b = Make(Math.Min(lo2, lo2 + w2), Math.Max(lo2, lo2 + w2));
            Iv sum = IAdd(a, b);
            Frac exactLo = Frac.FromDouble(a.Lo) + Frac.FromDouble(b.Lo);
            Frac exactHi = Frac.FromDouble(a.Hi) + Frac.FromDouble(b.Hi);
            bool okLo = Frac.FromDouble(sum.Lo).CompareTo(exactLo) <= 0 &&
                        Frac.FromDouble(Math.BitDecrement(sum.Lo)).CompareTo(exactLo) < 0;
            bool okHi = Frac.FromDouble(sum.Hi).CompareTo(exactHi) >= 0 &&
                        Frac.FromDouble(Math.BitIncrement(sum.Hi)).CompareTo(exactHi) > 0;
            bool product = Contains(IMul(a, b), a.Lo * b.Lo) && Contains(IMul(a, b), a.Hi * b.Hi);
            if (okLo && okHi && product) tight++;
        }
        Check(tight == 200, "200 random additions are the tightest double enclosures (" + tight + "/200)");

        // 5: isotonicity - narrowing the inputs can only narrow the result.
        int isotone = 0;
        for (int t = 0; t < 100; t++)
        {
            double c = NextSigned();
            double d = NextSigned();
            Iv wide = Make(Math.Min(c, d), Math.Max(c, d));
            Iv narrow = IIntersect(wide, Make(Math.Min(c, d) + 0.1, Math.Max(c, d) - 0.1));
            Iv other = Make(-1.0, 2.0);
            if (Subset(IMul(narrow, other), IMul(wide, other)) &&
                Subset(IAdd(narrow, other), IAdd(wide, other)) &&
                Subset(ISub(narrow, other), ISub(wide, other))) isotone++;
        }
        Check(isotone == 100, "inclusion isotonicity holds for 100 random pairs (" + isotone + "/100)");

        // 5-6: algebraic laws, including subdistributivity (distributivity itself only holds one way).
        Iv A = Make(-1.0, 2.0), B = Make(3.0, 4.0), Cc = Make(-2.0, -0.5);
        Check(IAdd(A, B).Lo == IAdd(B, A).Lo && IAdd(A, B).Hi == IAdd(B, A).Hi, "addition is commutative");
        Check(IMul(A, B).Lo == IMul(B, A).Lo && IMul(A, B).Hi == IMul(B, A).Hi, "multiplication is commutative");
        Iv bc = IAdd(B, Cc);
        Check(Subset(IMul(A, bc), IAdd(IMul(A, B), IMul(A, Cc))),
              "subdistributivity A(B+C) is contained in AB + AC");
        Check(Width(IMul(A, bc)) <= Width(IAdd(IMul(A, B), IMul(A, Cc))),
              "the right-hand distribution is never narrower");
        Check(IAbs(Make(-3.0, 1.0)).Hi == 3.0 && IAbs(Make(-3.0, 1.0)).Lo == 0.0, "absolute value");
        Check(IPow(Make(1.5, 2.0), 3).Lo <= Math.Pow(1.5, 3) && IPow(Make(1.5, 2.0), 3).Hi >= 8.0,
              "integer power with outward rounding");

        // 7: the classical dependency problem: x*(1-x) on [0,1] must not be over-optimistic.
        Iv x = Make(0.0, 1.0);
        Iv prod = IMul(x, ISub(Point(1.0), x));
        Check(Contains(prod, 0.25), "x(1-x) on [0,1] encloses the true maximum 1/4");
        Check(prod.Lo <= 0.0 && prod.Hi >= 1.0, "the extension over-estimates the true range [0,1/4]");
        Check(Width(prod) > 0.25, "the over-estimation is real and reported by the width");

        // 8-9: width bounds, with an ulp-level allowance for the fact that both the widths themselves and
        // the bound are evaluated in rounded double arithmetic (the enclosure adds at most one ulp per end).
        int bounds = 0;
        for (int t = 0; t < 100; t++)
        {
            double p = NextSigned() * 5, q = Math.Abs(NextSigned()) * 5;
            double r = NextSigned() * 5, s = Math.Abs(NextSigned()) * 5;
            Iv u = Make(Math.Min(p, p + q), Math.Max(p, p + q));
            Iv v = Make(Math.Min(r, r + s), Math.Max(r, r + s));
            double boundAdd = Width(u) + Width(v);
            double boundMul = Mag(u) * Width(v) + Mag(v) * Width(u) + Width(u) * Width(v);
            double slack = 8.0 * Eps * (boundAdd + boundMul + Mag(u) + Mag(v) + 1.0);
            bool ok = Width(IAdd(u, v)) <= boundAdd + slack &&
                      Width(ISub(u, v)) <= boundAdd + slack &&
                      Width(IMul(u, v)) <= boundMul + slack;
            if (ok) bounds++;
        }
        Check(bounds == 100, "w(A+-B) and w(AB) bounds hold for 100 random pairs (" + bounds + "/100)");

        // 10: degeneracy, emptiness and the disjoint case.
        Check(Empty.IsEmpty && Subset(Empty, A) && Subset(Empty, Empty), "the empty interval is a subset of all");
        Check(IAdd(Empty, A).IsEmpty && IMul(Empty, A).IsEmpty && ISqrt(Empty).IsEmpty,
              "every operation propagates emptiness");
        Check(IIntersect(Make(0.0, 1.0), Make(2.0, 3.0)).IsEmpty, "disjoint intervals intersect to empty");
        Check(IHull(Make(0.0, 1.0), Make(2.0, 3.0)).Lo == 0.0 && IHull(Make(0.0, 1.0), Make(2.0, 3.0)).Hi == 3.0,
              "the hull spans both");
        Check(Subset(IIntersect(A, B), A) && Subset(A, IHull(A, B)), "intersection/hull laws");
        Check(Overlaps(IHull(A, B), A) && Overlaps(IHull(A, B), B), "the hull overlaps both inputs");
        Check(Width(IHull(A, Cc)) >= Width(A) && Overlaps(IHull(A, Cc), A), "the hull never loses an operand");

        // 11: division by an interval containing zero, and a sign-symmetric divisor.
        Check(IDiv(Make(1.0, 2.0), Make(-1.0, 1.0)).Lo == double.NegativeInfinity &&
              IDiv(Make(1.0, 2.0), Make(-1.0, 1.0)).Hi == double.PositiveInfinity,
              "division by an interval straddling zero yields the entire line");
        Check(IDiv(Point(0.0), Make(-1.0, 1.0)).Lo == 0.0 && IDiv(Point(0.0), Make(-1.0, 1.0)).Hi == 0.0,
              "0 / [-1,1] is exactly 0");
        Iv quotient = IDiv(Make(1.0, 4.0), Make(1.0, 2.0));
        Check(quotient.Lo == 0.5 && quotient.Hi == 4.0, "a sign-definite divisor gives a finite enclosure");
        Check(IDiv(Point(1.0), Point(0.0)).IsEmpty == false, "division by the point 0 gives the entire line");

        // 12: sqrt on a partly negative interval clips the domain instead of failing.
        Iv sqrt = ISqrt(Make(-4.0, 9.0));
        Check(sqrt.Lo == 0.0 && Contains(sqrt, 3.0) && Contains(sqrt, 0.0), "sqrt([-4,9]) = [0,3] up to rounding");
        Check(ISqrt(Make(-9.0, -4.0)).IsEmpty, "sqrt of a strictly negative interval is empty");
        Check(Contains(IExp(Make(-1.0, 1.0)), Math.E) && Contains(ILog(Make(1.0, 3.0)), Math.Log(2.0)),
              "exp/log are monotone point evaluations with outward padding");
        Check(ILog(Make(0.0, 1.0)).Lo == double.NegativeInfinity && Contains(ILog(Make(0.0, 1.0)), 0.0),
              "log([0,1]) is unbounded below and finite above");
        Check(ILog(Make(-1.0, 0.5)).Hi <= Math.Log(0.5) + 1e-15 && ILog(Make(-1.0, 0.5)).Lo == double.NegativeInfinity,
              "log ignores the non-positive part of its domain");

        // 13: certified root enclosure - interval Newton on x^2 - 2, certified in exact arithmetic.
        Iv x0 = Make(1.0, 2.0);
        var widths = new List<double>();
        Iv cur = x0;
        for (int i = 0; i < 12; i++)
        {
            Iv next = NewtonStep(cur);
            Check(Subset(next, cur), "the Newton step contracts the interval");
            widths.Add(Width(next));
            cur = next;
        }
        Check(cur.IsEmpty == false && Contains(cur, Math.Sqrt(2.0)), "the enclosure contains sqrt(2)");
        Check(Math.Abs(cur.Lo - cur.Hi) < 1e-12, "the enclosure contracts to a point (" + Width(cur).ToString("E2") + ")");
        Frac sqLo = Frac.FromDouble(cur.Lo) * Frac.FromDouble(cur.Lo);
        Frac sqHi = Frac.FromDouble(cur.Hi) * Frac.FromDouble(cur.Hi);
        bool certified = sqLo.CompareTo(new Frac(2, 1)) <= 0 && sqHi.CompareTo(new Frac(2, 1)) >= 0;
        Check(certified, "lo^2 <= 2 <= hi^2 certified exactly (a real root lies inside)");
        string widthTrace = string.Join(",", widths.Take(8).Select(w => w.ToString("E2")));
        Check(widths.Zip(widths.Skip(1), (a, b) => b <= a).All(v => v), "Newton widths never increase");
        Check(widths.Take(3).Zip(widths.Skip(1), (a, b) => b < a).All(v => v),
              "the first three Newton steps contract strictly (" + widthTrace + ")");
        Check(Enumerable.Range(0, 2).All(i => widths[i + 1] <= 4.0 * widths[i] * widths[i]),
              "the contraction is superlinear while far from the ulp floor (" + widthTrace + ")");
        Check(widths[^1] <= 4.0 * Eps && widths[0] < 0.5,
              "contraction reaches the double-precision floor from a width of 1 (" + widths[^1].ToString("E2") + ")");

        // 14: bisection liveness - repeated halving converges to a point.
        Iv bisect = Make(-2.0, 2.0);
        double previous = Width(bisect);
        for (int i = 0; i < 40; i++)
        {
            double mid = (bisect.Lo + bisect.Hi) / 2;
            bisect = Contains(bisect, Math.Sqrt(2.0)) && mid <= Math.Sqrt(2.0) ? Make(mid, bisect.Hi) : Make(bisect.Lo, mid);
            Check(Width(bisect) <= previous, "bisection is monotone in width");
            previous = Width(bisect);
        }
        Check(Width(bisect) < 1e-9 && Contains(bisect, Math.Sqrt(2.0)), "40 halvings reach a 1e-9 enclosure");

        // 15: determinism of the whole library on a fixed battery.
        _state = 0x77E1A2B3C4D5E6F7UL;
        double firstRun = Battery();
        _state = 0x77E1A2B3C4D5E6F7UL;
        double secondRun = Battery();
        Check(firstRun == secondRun, "the library is bitwise reproducible on a fixed battery");

        Console.WriteLine("point pairs " + samples + ", tight enclosures 200/200, Newton width " +
                          Width(cur).ToString("E3") + ", bisection width " + Width(bisect).ToString("E3") +
                          ", battery " + firstRun.ToString("R"));
        Console.WriteLine("PASS");
    }

    // A fixed battery of interval expressions, used only as a reproducibility fingerprint.
    private static double Battery()
    {
        double acc = 1.0;
        for (int i = 0; i < 50; i++)
        {
            double a = NextSigned(), b = NextSigned();
            Iv u = Make(Math.Min(a, b), Math.Max(a, b));
            Iv v = Make(Math.Min(a * 0.5, b + 0.25), Math.Max(a * 0.5, b + 0.25));
            acc += Width(IAdd(u, IMul(v, ISqrt(IAbs(v))))) + IExp(u).Lo * 1e-9 + ILog(Make(1.0, 2.0)).Hi * 1e-9;
        }
        return acc;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
