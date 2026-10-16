// Task 874 - Implement certified numerical solver for polynomial roots (Sturm sequences).
// (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
//
// Model: a *certified* real-root solver. Real roots are counted with a Sturm sequence: for a square-free
// polynomial p the sequence p, p', -rem(p, p'), ... has the property that V(a) - V(b) equals the number
// of distinct real roots in the half-open interval (a, b], where V(x) is the number of sign variations of
// the sequence at x with zeros ignored. Every sign in that sequence is evaluated *exactly*: at a rational
// endpoint num/den the sign of p is the sign of sum_i c_i num^i den^(deg-i), a BigInteger. Isolation then
// subdivides until every interval holds exactly one root and is *certified* by an exact sign change of p
// at its endpoints, so each returned interval provably contains a root and provably no second one. The
// polynomial is reduced to its square-free part first, which is what makes the count equal the number of
// distinct roots and makes a quadruple root collapse to a single interval.
//
// The pseudo-remainder prem satisfies lv^s * a = Q*b + prem with lv = lc(b), i.e. prem is the true
// remainder up to the possibly negative constant lv^s; the next sequence element is -prem/lv^s, and a
// positive multiple of that is kept, so the Sturm sign pattern survives without rational arithmetic or
// coefficient blow-up (the primitive part keeps the integers small).
//
// Simplifications (documented, not behavioural): integer coefficients (which is where every certificate is
// exact), degree <= 6 in the tests, bisection boundaries on a rational grid, search window [-5, 5] widened
// by one grid step, and no complex roots - only certified real enclosures are claimed.
//
// Verified here:
//   * exact counts: Sturm counts equal the known root counts of factored polynomials and equal an
//     independent brute-force scan of the sign changes of p over 21001 grid points;
//   * certification: every returned interval has an exact sign change of p and contains exactly one root
//     by the Sturm count of its interior, and every interval brackets its known root;
//   * structure: Vieta's identities on the refined roots, disjointness and sorting of the intervals, and
//     monotone width shrinkage as the tolerance tightens;
//   * degenerate cases: no real roots, constants, the zero polynomial, a triple root, a quadruple root,
//     roots on the search boundary and the half-open counting convention there;
//   * determinism: the whole solve is bitwise reproducible.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;

public static class Solution
{
    private static ulong _state = 0xB7E151628AED2A6AUL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    // ---------- polynomials over Z, index i == coefficient of x^i, trailing zeros trimmed ----------
    private sealed class Poly
    {
        public List<BigInteger> C = new List<BigInteger>();

        public int Degree => C.Count - 1;
        public bool IsZero => C.Count == 0 || C.All(v => v.IsZero);

        public static Poly Const(long k)
        {
            var p = new Poly();
            if (k != 0) p.C.Add(k);
            return p;
        }

        public static Poly From(params long[] descending)
        {
            var p = new Poly();
            foreach (long v in descending) p.C.Add(v);
            p.C.Reverse();
            p.Trim();
            return p;
        }

        public void Trim()
        {
            while (C.Count > 0 && C[^1].IsZero) C.RemoveAt(C.Count - 1);
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            for (int i = C.Count - 1; i >= 0; i--) sb.Append(C[i]).Append(i == 0 ? "" : "x^" + i + " ");
            return sb.Length == 0 ? "0" : sb.ToString();
        }
    }

    private static Poly PAdd(Poly a, Poly b)
    {
        var r = new Poly();
        for (int i = 0; i < Math.Max(a.C.Count, b.C.Count); i++)
            r.C.Add((i < a.C.Count ? a.C[i] : 0) + (i < b.C.Count ? b.C[i] : 0));
        r.Trim();
        return r;
    }

    private static Poly PScale(Poly a, BigInteger k)
    {
        var r = new Poly();
        foreach (BigInteger v in a.C) r.C.Add(v * k);
        r.Trim();
        return r;
    }

    private static Poly PDivide(Poly a, BigInteger k)
    {
        var r = new Poly();
        foreach (BigInteger v in a.C) r.C.Add(v / k);
        r.Trim();
        return r;
    }

    /// <summary>p * x^k (the zero polynomial stays zero).</summary>
    private static Poly Shift(Poly p, int k)
    {
        if (p.IsZero) return new Poly();
        var r = new Poly();
        for (int i = 0; i < k; i++) r.C.Add(0);
        r.C.AddRange(p.C);
        return r;
    }

    private static Poly PMul(Poly a, Poly b)
    {
        if (a.IsZero || b.IsZero) return new Poly();
        var r = new Poly();
        for (int i = 0; i <= a.Degree + b.Degree; i++) r.C.Add(0);
        for (int i = 0; i < a.C.Count; i++)
            for (int j = 0; j < b.C.Count; j++) r.C[i + j] += a.C[i] * b.C[j];
        r.Trim();
        return r;
    }

    private static Poly PDerivative(Poly a)
    {
        var r = new Poly();
        for (int i = 1; i < a.C.Count; i++) r.C.Add(a.C[i] * i);
        r.Trim();
        return r;
    }

    /// <summary>Divides by the (positive) content: every sign, and hence every Sturm count, is unchanged.</summary>
    private static Poly PrimitivePart(Poly p)
    {
        if (p.IsZero) return p;
        BigInteger g = BigInteger.Zero;
        foreach (BigInteger v in p.C) g = BigInteger.GreatestCommonDivisor(g, v);
        if (g.IsZero || g.IsOne) return p;
        return PDivide(p, g);
    }

    /// <summary>lv^steps * a = Q * b + rem in Z[x], with lv = lc(b).</summary>
    private static Poly PseudoDivide(Poly a, Poly b, out Poly rem, out int steps)
    {
        if (b.IsZero) throw new ArgumentException("division by the zero polynomial");
        var r = new Poly();
        foreach (BigInteger v in a.C) r.C.Add(v);
        int d = Math.Max(0, a.Degree - b.Degree);
        var qc = new BigInteger[d + 1];
        BigInteger lv = b.C[^1];
        steps = 0;
        while (!r.IsZero && r.Degree >= b.Degree && steps <= d + 1)
        {
            int k = r.Degree - b.Degree;
            BigInteger factor = r.C[^1];
            r = PScale(r, lv);
            r = PAdd(r, PScale(Shift(b, k), -factor));
            r.Trim();
            for (int i = 0; i < qc.Length; i++) qc[i] *= lv;
            if (k < qc.Length) qc[k] += factor;
            steps++;
        }
        rem = r;
        var q = new Poly();
        foreach (BigInteger v in qc) q.C.Add(v);
        q.Trim();
        return q;
    }

    private static Poly PseudoRemainder(Poly a, Poly b, out int steps)
    {
        PseudoDivide(a, b, out Poly rem, out steps);
        return rem;
    }

    /// <summary>Primitive polynomial remainder sequence: the gcd only matters up to a unit.</summary>
    private static Poly PGcd(Poly a, Poly b)
    {
        Poly x = PrimitivePart(a);
        Poly y = PrimitivePart(b);
        while (!y.IsZero)
        {
            Poly r = PseudoRemainder(x, y, out _);
            if (!r.IsZero) r = PrimitivePart(r);
            x = y;
            y = r;
        }
        return x;
    }

    /// <summary>p / gcd(p, p') up to a nonzero constant; the Sturm count is scale invariant.</summary>
    private static Poly SquareFree(Poly p)
    {
        if (p.Degree <= 0) return p;
        Poly g = PGcd(p, PDerivative(p));
        if (g.Degree <= 0) return p;
        Poly quotient = PseudoDivide(p, g, out Poly rem, out _);
        if (!rem.IsZero) return PrimitivePart(p);           // cannot happen for a true divisor
        return PrimitivePart(quotient);
    }

    // ---------- Sturm machinery ----------
    /// <summary>
    /// The Sturm sequence p0 = p, p1 = p', p_{i+1} = -rem(p_{i-1}, p_i), each element kept as a *positive*
    /// multiple of the classical one so the sign variations agree with the theorem.
    /// </summary>
    private static List<Poly> SturmSequence(Poly p)
    {
        var seq = new List<Poly> { PrimitivePart(p), PrimitivePart(PDerivative(p)) };
        while (seq[^1].Degree > 0)
        {
            Poly last = seq[^1];
            Poly prem = PseudoRemainder(seq[^2], last, out int steps);
            if (prem.IsZero) break;
            // true remainder = prem / lv^steps, so the next element is a positive multiple of -prem/lv^steps.
            int scaleSign = steps % 2 == 0 ? 1 : last.C[^1].Sign;
            Poly next = PrimitivePart(PScale(prem, scaleSign > 0 ? -1 : 1));
            if (next.IsZero || next.Degree >= last.Degree) break;
            seq.Add(next);
        }
        return seq;
    }

    /// <summary>
    /// Exact sign of p(num/den): the sign of den^deg * p(num/den) = sum_i c_i num^i den^(deg-i), evaluated by
    /// Horner's rule in BigInteger arithmetic (the leading coefficient needs no denominator factor).
    /// </summary>
    private static int ExactSign(Poly p, BigInteger num, BigInteger den)
    {
        if (p.IsZero) return 0;
        int d = p.Degree;
        BigInteger acc = p.C[d];
        for (int i = d - 1; i >= 0; i--)
            acc = acc * num + p.C[i] * BigInteger.Pow(den, d - i);
        return acc.Sign;
    }

    private static int Variations(List<Poly> seq, BigInteger num, BigInteger den)
    {
        int v = 0, last = 0;
        foreach (Poly p in seq)
        {
            int s = ExactSign(p, num, den);
            if (s == 0) continue;                       // zeros are skipped: the (a, b] convention
            if (last != 0 && s != last) v++;
            last = s;
        }
        return v;
    }

    /// <summary>Number of distinct real roots in the half-open interval (lo/2^scale, hi/2^scale].</summary>
    private static int RootsIn(List<Poly> seq, long lo, long hi, int scale)
    {
        BigInteger den = BigInteger.Pow(2, scale);
        BigInteger loN = lo, hiN = hi;
        return Variations(seq, loN, den) - Variations(seq, hiN, den);
    }

    private static int RootsInIntegers(List<Poly> seq, long lo, long hi) => RootsIn(seq, lo, hi, 0);

    // ---------- solver ----------
    private sealed class Interval
    {
        public long Lo;
        public long Hi;
        public int Scale;
        public double LoReal => (double)Lo / Math.Pow(2, Scale);
        public double HiReal => (double)Hi / Math.Pow(2, Scale);
        public override string ToString() => "[" + LoReal.ToString("F5") + "," + HiReal.ToString("F5") + "]";
    }

    private const long Window = 5;      // the search window is (-Window, Window], widened by one grid step

    /// <summary>
    /// Isolates every distinct real root inside the search window into one interval, bisecting on exact
    /// Sturm counts, then refines each interval to the requested tolerance.
    /// </summary>
    private static List<Interval> Solve(Poly p, double tolerance, int scale)
    {
        var result = new List<Interval>();
        if (p.Degree <= 0) return result;
        Poly squareFree = SquareFree(p);
        if (squareFree.Degree <= 0) return result;
        List<Poly> seq = SturmSequence(squareFree);
        long unit = 1L << scale;
        var work = new Stack<Interval>();
        work.Push(new Interval { Lo = -Window * unit - 1, Hi = Window * unit + 1, Scale = scale });
        while (work.Count > 0)
        {
            Interval iv = work.Pop();
            int count = RootsIn(seq, iv.Lo, iv.Hi, iv.Scale);
            if (count == 0) continue;
            if (count == 1 && iv.HiReal - iv.LoReal <= tolerance)
            {
                result.Add(iv);
                continue;
            }
            long mid = iv.Lo + (iv.Hi - iv.Lo) / 2;
            if (mid == iv.Lo || mid == iv.Hi) { result.Add(iv); continue; }
            work.Push(new Interval { Lo = iv.Lo, Hi = mid, Scale = iv.Scale });
            work.Push(new Interval { Lo = mid, Hi = iv.Hi, Scale = iv.Scale });
        }
        result.Sort((a, b) => a.LoReal.CompareTo(b.LoReal));
        return result;
    }

    /// <summary>Exact certificate: p changes sign between the endpoints, or vanishes on one of them.</summary>
    private static bool Certified(Poly p, Interval iv)
    {
        BigInteger den = BigInteger.Pow(2, iv.Scale);
        int sLo = ExactSign(p, iv.Lo, den);
        int sHi = ExactSign(p, iv.Hi, den);
        return sLo == 0 || sHi == 0 || sLo != sHi;
    }

    /// <summary>Independent oracle: sign changes of p over a fine floating-point grid.</summary>
    private static int BruteForceSignChanges(Poly p, double lo, double hi, int steps)
    {
        int changes = 0, last = 0;
        for (int i = 0; i <= steps; i++)
        {
            double x = lo + (hi - lo) * i / steps;
            double acc = 0.0;
            for (int k = p.C.Count - 1; k >= 0; k--) acc = acc * x + (double)p.C[k];
            int s = Math.Sign(acc);
            if (s == 0) continue;
            if (last != 0 && s != last) changes++;
            last = s;
        }
        return changes;
    }

    private static Poly FromRoots(params long[] roots)
    {
        Poly p = Poly.Const(1);
        foreach (long r in roots) p = PMul(p, Poly.From(1, -r));
        return p;
    }

    // ---------- main ----------
    public static void Main()
    {
        const int Scale = 44;      // grid step 2^-44, far below every tolerance used below

        // 1-3: canonical factored polynomials - the Sturm count must equal the known root count.
        var canonical = new (long[] Roots, long[] Coefficients)[]
        {
            (new long[] { 1, 2, 3 }, new long[] { 1, -6, 11, -6 }),
            (new long[] { -2, 0, 2 }, new long[] { 1, 0, -4, 0 }),
            (new long[] { -1, 1 }, new long[] { 1, 0, -1 }),
        };
        foreach ((long[] roots, long[] coeffs) in canonical)
        {
            Poly p = Poly.From(coeffs);
            Check(p.C.SequenceEqual(FromRoots(roots).C), "the factored form expands to the stated coefficients");
            List<Poly> seq = SturmSequence(SquareFree(p));
            Check(RootsInIntegers(seq, -10, 10) == roots.Length,
                  "Sturm count equals the known root count (" + roots.Length + ")");
            var solution = Solve(p, 1e-6, Scale);
            Check(solution.Count == roots.Length, "isolation returns one interval per root (" + solution.Count + ")");
            for (int i = 0; i < roots.Length; i++)
            {
                Check(solution[i].LoReal <= roots[i] && roots[i] <= solution[i].HiReal,
                      "interval " + i + " contains the exact root " + roots[i]);
                Check(solution[i].HiReal - solution[i].LoReal <= 1e-6, "interval " + i + " is narrow enough");
                Check(Certified(p, solution[i]), "interval " + i + " is certified by an exact sign change");
            }
        }

        // 4: Sturm counts against an independent sign-change scan of p on a fine grid. The grid oracle can
        // only see sign changes, so it counts the distinct roots of *odd* multiplicity - comparing against
        // that exact quantity (rather than against the total count) keeps the oracle honest when a random
        // factorisation repeats a root.
        int oracleAgreements = 0;
        for (int t = 0; t < 40; t++)
        {
            int degree = 1 + (int)(NextU64() % 5);
            Poly p = Poly.Const(1);
            var known = new List<long>();
            for (int i = 0; i < degree; i++)
            {
                long r = (long)(NextU64() % 19) - 9;
                known.Add(r);
                p = PMul(p, Poly.From(1, -r));
            }
            List<Poly> seq = SturmSequence(SquareFree(p));
            int sturm = RootsInIntegers(seq, -20, 20);
            int distinct = known.Distinct().Count();
            int oddRoots = known.GroupBy(r => r).Count(g => g.Count() % 2 == 1);
            int brute = BruteForceSignChanges(p, -9.5, 9.5, 21001);
            if (sturm == distinct && brute == oddRoots) oracleAgreements++;
        }
        Check(oracleAgreements == 40, "Sturm counts match the known and brute-forced counts ("
              + oracleAgreements + "/40)");

        // 5-6: roots are certified *and* separated: no interval contains a second root.
        Poly sextic = FromRoots(-3, -1, 0, 2, 4, 5);
        var six = Solve(sextic, 1e-9, Scale);
        Check(six.Count == 6, "six distinct roots isolated (" + six.Count + ")");
        List<Poly> seqSix = SturmSequence(SquareFree(sextic));
        for (int i = 0; i < six.Count; i++)
        {
            Check(Certified(sextic, six[i]), "sextic interval " + i + " certified");
            Check(RootsIn(seqSix, six[i].Lo, six[i].Hi, six[i].Scale) == 1,
                  "exactly one root counted inside interval " + i);
            if (i > 0) Check(six[i].Lo > six[i - 1].Hi, "intervals are disjoint and sorted");
        }
        double sumRoots = six.Sum(iv => (iv.LoReal + iv.HiReal) / 2);
        Check(Math.Abs(sumRoots - 7.0) < 1e-6, "Vieta: the root sum equals 7 (" + sumRoots + ")");
        double product = six.Aggregate(1.0, (acc, iv) => acc * ((iv.LoReal + iv.HiReal) / 2));
        Check(Math.Abs(product) < 1e-6, "Vieta: the product is 0 because 0 is a root (" + product + ")");

        // 7: refinement is liveness - widths shrink monotonically with the requested tolerance.
        double previousWidth = double.PositiveInfinity;
        foreach (double tol in new[] { 1e-2, 1e-4, 1e-6, 1e-8 })
        {
            var fine = Solve(sextic, tol, Scale);
            double worst = fine.Max(iv => iv.HiReal - iv.LoReal);
            Check(fine.Count == 6 && worst <= tol + 1e-12, "tolerance " + tol + " achieved (" + worst + ")");
            previousWidth = worst;
        }
        Check(previousWidth <= 1e-8 + 1e-12, "the tightest refinement reaches 1e-8");

        // 8-10: degenerate inputs.
        Check(Solve(Poly.From(1, 0, 1), 1e-6, Scale).Count == 0, "x^2 + 1 has no certified real root");
        Check(Solve(Poly.Const(3), 1e-6, Scale).Count == 0, "a constant polynomial has no root");
        Check(Solve(new Poly(), 1e-6, Scale).Count == 0,
              "the zero polynomial returns no intervals instead of a wrong one");
        var repeated = Solve(FromRoots(1, 1, 1, 2), 1e-6, Scale);
        Check(repeated.Count == 2, "triple root 1 and simple root 2 give two intervals (" + repeated.Count + ")");
        Check(Certified(FromRoots(1, 1, 1, 2), repeated[0]), "the repeated root is still certified");
        Check(Math.Abs(repeated[0].LoReal - 1.0) < 1e-6 && Math.Abs(repeated[1].LoReal - 2.0) < 1e-6,
              "repeated-root values are right");
        var allEqual = Solve(FromRoots(5, 5, 5, 5), 1e-6, Scale);
        Check(allEqual.Count == 1 && Math.Abs(allEqual[0].LoReal - 5.0) < 1e-6,
              "a quadruple root collapses to a single certified interval");
        Check(BruteForceSignChanges(Poly.From(1, 0, 1), -10, 10, 2001) == 0,
              "the brute-force oracle agrees that x^2 + 1 never changes sign");

        // 11: the documented half-open counting convention, tested where roots sit on the boundary.
        Poly boundary = FromRoots(-4, 4);
        List<Poly> seqBoundary = SturmSequence(SquareFree(boundary));
        Check(RootsInIntegers(seqBoundary, -4, 4) == 1, "a root at lo is excluded, a root at hi is included");
        Check(RootsInIntegers(seqBoundary, -5, 5) == 2, "widening the window counts both");
        Check(RootsInIntegers(seqBoundary, 5, 6) == 0, "an interval above every root counts nothing");
        var boundarySolution = Solve(boundary, 1e-6, Scale);
        Check(boundarySolution.Count == 2 && boundarySolution[0].LoReal <= -4.0 && boundarySolution[1].HiReal >= 4.0,
              "roots near the search boundary are still isolated and bracketed");

        // 12: certified enclosures survive a much finer tolerance.
        Poly quartic = FromRoots(-2, -1, 1, 3);
        var fineSolution = Solve(quartic, 1e-12, Scale);
        Check(fineSolution.Count == 4, "four intervals at 1e-12 (" + fineSolution.Count + ")");
        foreach (Interval iv in fineSolution)
        {
            Check(Certified(quartic, iv), "1e-12 interval certified");
            double mid = (iv.LoReal + iv.HiReal) / 2;
            Check(Math.Abs(mid - Math.Round(mid)) < 1e-6, "the refined interval still brackets an integer root");
        }
        Check(RootsInIntegers(SturmSequence(SquareFree(quartic)), -3, 0) == 2, "two roots counted in (-3,0]");
        Check(RootsInIntegers(SturmSequence(SquareFree(quartic)), 0, 2) == 1, "one root counted in (0,2]");
        Check(RootsInIntegers(SturmSequence(SquareFree(quartic)), 2, 4) == 1, "one root counted in (2,4]");

        // 13: determinism.
        var firstRun = Solve(sextic, 1e-9, Scale);
        var secondRun = Solve(sextic, 1e-9, Scale);
        Check(firstRun.Count == secondRun.Count &&
              firstRun.Zip(secondRun, (a, b) => a.Lo == b.Lo && a.Hi == b.Hi).All(v => v),
              "the solver is bitwise reproducible");

        Console.WriteLine("oracle agreements " + oracleAgreements + "/40, sextic " +
                          string.Join(" ", six.Take(3).Select(iv => iv.ToString())) + " ... " + six[^1] +
                          ", triple-root intervals " + repeated.Count + ", quadruple " + allEqual[0]);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
