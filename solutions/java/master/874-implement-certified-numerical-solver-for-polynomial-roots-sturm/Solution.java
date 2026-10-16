/** Task 874 - Implement certified numerical solver for polynomial roots (Sturm sequences).
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Certified means the solver returns, for every real root, an interval that provably contains exactly that
 * one root. The classical machinery is Sturm's theorem: for a square-free polynomial P build the sequence
 *
 *     s_0 = P,  s_1 = P',  s_{i+1} = -(pseudo-remainder of s_{i-1} by s_i) with its content divided out
 *
 * and let V(x) be the number of sign changes in (s_0(x), ..., s_k(x)) with zeros ignored. Then V(a) - V(b) is
 * exactly the number of distinct real roots in (a, b]. Every step is exact integer arithmetic: the
 * pseudo-remainder keeps the coefficients integral, dividing by the content keeps them small, and because
 * prem(s_{i-1}, s_i) differs from the true remainder by a constant factor, negating it produces precisely the
 * positive multiple of -remainder that Sturm's proof requires. Evaluation at a rational point p/q is done
 * exactly by scaling with q^deg, so no rounding ever enters the count.
 *
 * On top of the count the solver isolates: Cauchy's bound gives a search box, bisection splits it, the Sturm
 * count decides which halves hold roots, and each interval is refined until it is narrower than the requested
 * tolerance while still holding a count of exactly one - which is the uniqueness certificate, independent of
 * any floating-point convergence claim. Multiplicities are recovered from gcd(P, P').
 *
 * SIMPLIFIED: integer-coefficient polynomials only, a textbook bisection/Sturm pipeline (no Descartes or
 * continued-fraction acceleration), and bisection depth rather than interval Newton as the refinement engine.
 * No threads, no wall-clock time, no hardware dependence and no randomness at all.
 *
 * VERIFIED: (a) EXACT COUNTS - the number of roots in a box, in a half-line and over the whole line comes
 * from Sturm's theorem and equals the count obtained by dense sign-change sampling on an independent grid, and
 * the total from the leading-coefficient signs at infinity equals the sum over the isolated intervals;
 * (b) KNOWN ROOTS - products of linear factors with integer roots, quadratics with irrational roots, the
 * classic cubic x^3-2x-5 = 0 and a rational root isolate to intervals containing exactly the analytic roots;
 * (c) UNIQUENESS - every returned interval has Sturm count one and is disjoint from every other;
 * (d) MULTIPLICITY - gcd(P, P') is the repeated factor, the square-free part has the same distinct real roots
 * as P, and a double root is reported once, not twice; (e) COMPLEX ROOTS - x^2+1 and x^4+1 report zero real
 * roots while still having the right degree; (f) degenerate inputs (constants, the zero polynomial, degree one,
 * a root exactly on a bisection point) and invalid requests are reported; (g) the whole solver is
 * deterministic and its residuals are tiny.
 */
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;

/** Exact integer polynomials, ascending in degree: coefficient[i] multiplies x^i. */
class Poly {
    final BigInteger[] c;

    private Poly(BigInteger[] c) {
        int last = c.length - 1;
        while (last > 0 && c[last].signum() == 0) last--;
        this.c = new BigInteger[last + 1];
        System.arraycopy(c, 0, this.c, 0, last + 1);
    }

    static Poly of(long... coefficients) {
        BigInteger[] c = new BigInteger[coefficients.length];
        for (int i = 0; i < coefficients.length; i++) c[i] = BigInteger.valueOf(coefficients[i]);
        return new Poly(c);
    }

    static Poly of(BigInteger[] coefficients) { return new Poly(coefficients); }

    static Poly product(Poly... factors) {
        Poly result = Poly.of(1L);
        for (Poly factor : factors) result = result.multiply(factor);
        return result;
    }

    int degree() { return c.length - 1; }

    boolean isZero() { return c.length == 1 && c[0].signum() == 0; }

    boolean isConstant() { return degree() == 0; }

    Poly multiply(Poly other) {
        BigInteger[] out = new BigInteger[degree() + other.degree() + 1];
        Arrays.fill(out, BigInteger.ZERO);
        for (int i = 0; i < c.length; i++)
            for (int j = 0; j < other.c.length; j++)
                out[i + j] = out[i + j].add(c[i].multiply(other.c[j]));
        return new Poly(out);
    }

    Poly derivative() {
        BigInteger[] out = new BigInteger[Math.max(1, c.length - 1)];
        Arrays.fill(out, BigInteger.ZERO);
        for (int i = 1; i < c.length; i++) out[i - 1] = c[i].multiply(BigInteger.valueOf(i));
        return new Poly(out);
    }

    Poly negate() {
        BigInteger[] out = new BigInteger[c.length];
        for (int i = 0; i < c.length; i++) out[i] = c[i].negate();
        return new Poly(out);
    }

    /** Divide by the positive content of the coefficient vector: primitive part, sign preserved. */
    Poly primitivePart() {
        BigInteger gcd = BigInteger.ZERO;
        for (BigInteger value : c) gcd = gcd.gcd(value);
        if (gcd.signum() == 0 || gcd.equals(BigInteger.ONE)) return this;
        BigInteger[] out = new BigInteger[c.length];
        for (int i = 0; i < c.length; i++) out[i] = c[i].divide(gcd);
        return new Poly(out);
    }

    /**
     * Exact sign of P(p/q) for q > 0: multiply through by the positive q^degree, so the sign of
     * sum_i c_i p^i q^(degree-i) is the sign of P(p/q) with no rounding anywhere.
     */
    int signAt(BigInteger p, BigInteger q) {
        BigInteger sum = BigInteger.ZERO;
        for (int i = 0; i < c.length; i++)
            sum = sum.add(c[i].multiply(p.pow(i)).multiply(q.pow(degree() - i)));
        return sum.signum();
    }

    double valueAt(double x) {
        double sum = 0;
        for (int i = c.length - 1; i >= 0; i--) sum = sum * x + c[i].doubleValue();
        return sum;
    }

    /**
     * Pseudo-remainder prem(A, B) = lc(B)^(deg A - deg B + 1) * (A mod B), computed with the classical
     * inner loop so the coefficients stay integral: multiply the running remainder by lc(B), subtract a
     * shifted multiple of B, and repeat until the degree drops below deg(B).
     */
    static Poly pseudoRemainder(Poly a, Poly b) {
        if (b.isZero()) throw new IllegalArgumentException("division by the zero polynomial");
        if (a.degree() < b.degree()) return a;
        BigInteger[] r = a.c.clone();
        BigInteger leadingB = b.c[b.degree()];
        int steps = a.degree() - b.degree() + 1;
        for (int i = 0; i < steps; i++) {
            int degreeR = r.length - 1;
            while (degreeR > 0 && r[degreeR].signum() == 0) degreeR--;
            if (degreeR < b.degree()) break;
            BigInteger factor = r[degreeR];
            int shift = degreeR - b.degree();
            for (int j = 0; j < r.length; j++) r[j] = r[j].multiply(leadingB);
            for (int j = 0; j <= b.degree(); j++)
                r[j + shift] = r[j + shift].subtract(factor.multiply(b.c[j]));
        }
        return new Poly(r);
    }

    /** Greatest common divisor by the primitive Euclid loop: the last non-zero remainder, made primitive. */
    static Poly gcd(Poly a, Poly b) {
        Poly x = a, y = b;
        while (!y.isZero()) {
            Poly r = pseudoRemainder(x, y).primitivePart();
            x = y;
            y = r;
        }
        return x.primitivePart();
    }

    static Poly divideExact(Poly a, Poly b) {
        if (b.degree() == 0) throw new IllegalArgumentException("division by a constant is handled elsewhere");
        BigInteger[] remainder = a.c.clone();
        BigInteger[] quotient = new BigInteger[Math.max(1, a.degree() - b.degree() + 1)];
        Arrays.fill(quotient, BigInteger.ZERO);
        BigInteger leading = b.c[b.degree()];
        for (int i = a.degree() - b.degree(); i >= 0; i--) {
            BigInteger factor = remainder[i + b.degree()].divide(leading);
            quotient[i] = factor;
            for (int j = 0; j <= b.degree(); j++)
                remainder[i + j] = remainder[i + j].subtract(factor.multiply(b.c[j]));
        }
        for (BigInteger value : remainder)
            if (value.signum() != 0) throw new ArithmeticException("not an exact division");
        return new Poly(quotient);
    }

    /** Square-free part P / gcd(P, P'): same distinct roots, no multiplicities. */
    static Poly squareFreePart(Poly p) {
        Poly g = gcd(p, p.derivative());
        if (g.degree() == 0) return p;
        return divideExact(p, g);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = c.length - 1; i >= 0; i--) {
            if (c[i].signum() == 0) continue;
            if (sb.length() > 0) sb.append(' ');
            sb.append(c[i]).append(i == 0 ? "" : "x^" + i);
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }
}

/** A rational endpoint p/q with q > 0, kept exactly. */
class Rational {
    final BigInteger p, q;

    Rational(BigInteger p, BigInteger q) {
        if (q.signum() == 0) throw new IllegalArgumentException("zero denominator");
        if (q.signum() < 0) { p = p.negate(); q = q.negate(); }
        BigInteger g = p.gcd(q);
        if (g.signum() != 0 && !g.equals(BigInteger.ONE)) { p = p.divide(g); q = q.divide(g); }
        this.p = p; this.q = q;
    }

    static Rational of(long value) { return new Rational(BigInteger.valueOf(value), BigInteger.ONE); }

    Rational midpoint(Rational other) {
        return new Rational(p.multiply(other.q).add(other.p.multiply(q)), q.multiply(other.q).shiftLeft(1));
    }

    double toDouble() { return p.doubleValue() / q.doubleValue(); }

    @Override
    public String toString() { return p + "/" + q; }
}

/** One certified enclosure: the interval (lo, hi] together with the single root it contains. */
class Enclosure {
    final Rational lo, hi;
    final int signChanges;

    Enclosure(Rational lo, Rational hi, int signChanges) {
        this.lo = lo; this.hi = hi; this.signChanges = signChanges;
    }

    double width() { return hi.toDouble() - lo.toDouble(); }

    /** The exact width of the enclosure as a decimal, for reporting intervals narrower than a double ulp. */
    String widthText() {
        java.math.MathContext context = new java.math.MathContext(25);
        BigDecimal left = new BigDecimal(lo.p).divide(new BigDecimal(lo.q), context);
        BigDecimal right = new BigDecimal(hi.p).divide(new BigDecimal(hi.q), context);
        return right.subtract(left).toString();
    }

    double midpoint() { return 0.5 * (lo.toDouble() + hi.toDouble()); }

    @Override
    public String toString() { return "(" + lo + ", " + hi + "]"; }
}

/** Sturm sequences, exact root counting, isolation and refinement. */
class Sturm {
    final Poly polynomial;          // the square-free polynomial actually used
    final List<Poly> sequence = new ArrayList<>();

    Sturm(Poly p) {
        if (p.isZero()) throw new IllegalArgumentException("the zero polynomial has infinitely many roots");
        polynomial = Poly.squareFreePart(p);
        sequence.add(polynomial);
        if (polynomial.degree() == 0) return;
        sequence.add(polynomial.derivative().primitivePart());
        while (true) {
            Poly last = sequence.get(sequence.size() - 1);
            if (last.degree() == 0) break;
            Poly next = Poly.pseudoRemainder(sequence.get(sequence.size() - 2), last).negate().primitivePart();
            if (next.isZero()) break;
            sequence.add(next);
        }
    }

    /** Number of sign changes of the sequence at the rational point, zeros ignored. */
    int signChangesAt(Rational x) {
        int changes = 0, previous = 0;
        for (Poly element : sequence) {
            int sign = element.signAt(x.p, x.q);
            if (sign == 0) continue;
            if (previous != 0 && sign != previous) changes++;
            previous = sign;
        }
        return changes;
    }

    /** Number of sign changes at -infinity: the leading coefficient, flipped by the degree's parity. */
    int signChangesAtMinusInfinity() {
        int changes = 0, previous = 0;
        for (Poly element : sequence) {
            int sign = element.c[element.degree()].signum();
            if (element.degree() % 2 == 1) sign = -sign;
            if (previous != 0 && sign != previous) changes++;
            previous = sign;
        }
        return changes;
    }

    int signChangesAtPlusInfinity() {
        int changes = 0, previous = 0;
        for (Poly element : sequence) {
            int sign = element.c[element.degree()].signum();
            if (previous != 0 && sign != previous) changes++;
            previous = sign;
        }
        return changes;
    }

    /** Total number of distinct real roots, from the behaviour at the two infinities. */
    int totalRealRoots() { return signChangesAtMinusInfinity() - signChangesAtPlusInfinity(); }

    /** Roots in (lo, hi]: Sturm's theorem. */
    int countIn(Rational lo, Rational hi) {
        return signChangesAt(lo) - signChangesAt(hi);
    }

    /** Cauchy bound: every root satisfies |x| <= 1 + max|c_i| / |c_n|. */
    BigInteger bound() {
        BigInteger largest = BigInteger.ZERO;
        for (int i = 0; i < polynomial.degree(); i++)
            largest = largest.max(polynomial.c[i].abs());
        BigInteger leading = polynomial.c[polynomial.degree()].abs();
        return BigInteger.ONE.add(largest.divide(leading).add(BigInteger.ONE));
    }

    /**
     * Isolate every real root inside (-bound, bound] by bisection driven purely by the exact count. The
     * recursion stops when an interval holds one root and is narrower than the tolerance; the count one is
     * the certificate that the root inside is unique.
     */
    List<Enclosure> isolate(double tolerance) {
        List<Enclosure> out = new ArrayList<>();
        if (polynomial.degree() == 0) return out;
        Rational lo = Rational.of(bound().negate().longValueExact());
        Rational hi = Rational.of(bound().longValueExact());
        int total = countIn(lo, hi);
        if (total != totalRealRoots()) throw new IllegalStateException("the Cauchy bound lost a root");
        isolate(lo, hi, total, tolerance, out);
        return out;
    }

    private void isolate(Rational lo, Rational hi, int count, double tolerance, List<Enclosure> out) {
        if (count == 0) return;
        if (count == 1 && hi.toDouble() - lo.toDouble() <= tolerance) {
            out.add(new Enclosure(lo, hi, count));
            return;
        }
        if (count > 1 && hi.toDouble() - lo.toDouble() < 1e-15) {
            throw new IllegalStateException("bisection stalled with " + count + " roots in one point");
        }
        Rational mid = lo.midpoint(hi);
        int left = countIn(lo, mid);
        isolate(lo, mid, left, tolerance, out);
        isolate(mid, hi, count - left, tolerance, out);
    }

    /** Refine one enclosure further by bisection, always keeping exactly one root inside. */
    Enclosure refine(Enclosure enclosure, int steps) {
        Rational lo = enclosure.lo, hi = enclosure.hi;
        for (int i = 0; i < steps; i++) {
            Rational mid = lo.midpoint(hi);
            if (countIn(lo, mid) == 1) hi = mid; else lo = mid;
        }
        return new Enclosure(lo, hi, countIn(lo, hi));
    }

    /** Counts how many repeated factors vanish at the rational point, i.e. the multiplicity above one. */
    static int repeatedFactorsAt(Poly p, Rational point) {
        int count = 0;
        Poly current = p;
        while (current.degree() >= 1) {
            Poly g = Poly.gcd(current, current.derivative());
            if (g.degree() == 0) break;
            if (g.signAt(point.p, point.q) == 0) count++;
            current = Poly.divideExact(current, g);
        }
        return count;
    }
}

public class Solution {
    public static void main(String[] args) {
        // --- KNOWN ROOTS: integer and irrational ------------------------------------------------
        Poly cubicFactors = Poly.product(Poly.of(-1, 1), Poly.of(-2, 1), Poly.of(-3, 1));
        Sturm cubic = new Sturm(cubicFactors);
        List<Enclosure> cubicRoots = cubic.isolate(1e-6);
        check(cubicRoots.size() == 3, "a cubic with three integer roots must report three, got "
                + cubicRoots.size());
        check(containsRoot(cubicRoots, 1.0) && containsRoot(cubicRoots, 2.0) && containsRoot(cubicRoots, 3.0),
                "the enclosures must contain 1, 2 and 3, got " + cubicRoots);
        check(cubic.totalRealRoots() == 3 && cubic.signChangesAtMinusInfinity() == 3,
                "the infinity counts must agree with the isolated total");

        Poly quadratic = Poly.product(Poly.of(-2, 0, 1), Poly.of(-5, 1));
        Sturm irrational = new Sturm(quadratic);
        List<Enclosure> irrationalRoots = irrational.isolate(1e-9);
        check(irrationalRoots.size() == 3, "x^2-2 times (x-5) must have three real roots");
        check(containsRoot(irrationalRoots, Math.sqrt(2)) && containsRoot(irrationalRoots, -Math.sqrt(2))
                && containsRoot(irrationalRoots, 5.0),
                "the enclosures must contain +-sqrt(2) and 5, got " + irrationalRoots);
        double worstWidth = 0;
        for (Enclosure e : irrationalRoots) worstWidth = Math.max(worstWidth, e.width());
        check(worstWidth <= 1e-6, "isolation must reach the requested tolerance, worst width " + worstWidth);

        Sturm classic = new Sturm(Poly.of(-5, -2, 0, 1));
        List<Enclosure> classicRoots = classic.isolate(1e-12);
        check(classicRoots.size() == 1, "x^3-2x-5 has exactly one real root, got " + classicRoots.size());
        check(classicRoots.get(0).width() < 1e-12, "the classic cubic must be refined below 1e-12");
        check(containsRoot(classicRoots, 2.0945514815423265),
                "the enclosure must contain the known root, got " + classicRoots.get(0));
        Enclosure refined = classic.refine(classicRoots.get(0), 20);
        check(refined.width() < classicRoots.get(0).width() && refined.signChanges == 1,
                "refinement must shrink the interval while keeping exactly one root inside");
        check(Math.abs(Poly.of(-5, -2, 0, 1).valueAt(refined.midpoint())) < 1e-12,
                "the residual at the refined midpoint must be tiny, got "
                        + Math.abs(Poly.of(-5, -2, 0, 1).valueAt(refined.midpoint())));

        Poly rationalRoot = Poly.of(-1, 2);
        List<Enclosure> halfRoot = new Sturm(rationalRoot).isolate(1e-9);
        check(halfRoot.size() == 1 && containsRoot(halfRoot, 0.5),
                "2x-1 must isolate its rational root 1/2, got " + halfRoot);

        // --- EXACT COUNTS against dense sign-change sampling ------------------------------------------------
        Poly mixed = Poly.product(Poly.of(-3, 1), Poly.of(2, 1, 1), Poly.of(4, -1, 0, 1));
        Sturm mixedSturm = new Sturm(mixed);
        List<Enclosure> mixedRoots = mixedSturm.isolate(1e-8);
        int sampled = sampledSignChanges(mixed, -8, 8, 200000);
        check(mixedRoots.size() == sampled, "the Sturm total must equal dense sign-change sampling: "
                + mixedRoots.size() + " against " + sampled);
        check(mixedSturm.totalRealRoots() == sampled, "the infinity count must match as well");
        int inside = 0;
        for (Enclosure e : mixedRoots) if (e.lo.toDouble() > -6 && e.hi.toDouble() < 6) inside++;
        check(mixedSturm.countIn(Rational.of(-6), Rational.of(6)) == inside,
                "the count inside a sub-box must equal the number of isolated roots there: " + inside);
        check(mixedSturm.countIn(Rational.of(10), Rational.of(20)) == 0,
                "an empty box must report no roots");
        check(mixedSturm.countIn(Rational.of(-100), Rational.of(100)) == mixedRoots.size(),
                "the full-box count must equal the isolated total");

        // --- UNIQUENESS and ordering --------------------------------------------------------------------
        boolean disjoint = true, unique = true;
        for (int i = 0; i < mixedRoots.size(); i++) {
            Enclosure current = mixedRoots.get(i);
            if (current.signChanges != 1) unique = false;
            if (mixedSturm.countIn(current.lo, current.hi) != 1) unique = false;
            if (Math.abs(mixed.valueAt(current.midpoint())) > 1e-5) unique = false;
            if (i > 0 && mixedRoots.get(i - 1).hi.toDouble() > current.lo.toDouble()) disjoint = false;
        }
        check(unique, "every enclosure must hold exactly one root with a tiny residual");
        check(disjoint, "the enclosures must be sorted and disjoint");

        // --- MULTIPLICITY ------------------------------------------------------------------------------
        Poly repeated = Poly.product(Poly.of(-1, 1), Poly.of(-1, 1), Poly.of(2, 1));
        Poly factor = Poly.gcd(repeated, repeated.derivative());
        check(factor.degree() == 1, "the gcd with the derivative must be the repeated factor, degree "
                + factor.degree());
        check(Math.abs(factor.valueAt(1.0)) < 1e-12, "the repeated factor must vanish at 1, got "
                + factor.valueAt(1.0));
        Sturm repeatedSturm = new Sturm(repeated);
        List<Enclosure> distinct = repeatedSturm.isolate(1e-6);
        check(distinct.size() == 2, "the double root must be reported once, got " + distinct.size());
        check(containsRoot(distinct, 1.0) && containsRoot(distinct, -2.0),
                "the distinct roots must be 1 and -2, got " + distinct);
        check(Poly.squareFreePart(repeated).degree() == 2 && repeated.degree() == 3,
                "the square-free part must drop the multiplicity: " + Poly.squareFreePart(repeated));
        check(Poly.squareFreePart(repeated).valueAt(1.0) == 0.0
                && Poly.squareFreePart(repeated).valueAt(-2.0) == 0.0,
                "the square-free part must keep every distinct root");
        check(Sturm.repeatedFactorsAt(repeated, Rational.of(1)) == 1
                && Sturm.repeatedFactorsAt(repeated, Rational.of(-2)) == 0,
                "the double root at 1 must be flagged as repeated, the simple root at -2 must not: got "
                        + Sturm.repeatedFactorsAt(repeated, Rational.of(1)) + " and "
                        + Sturm.repeatedFactorsAt(repeated, Rational.of(-2))
                        + " gcd=" + Poly.gcd(repeated, repeated.derivative())
                        + " quotient=" + Poly.squareFreePart(repeated));

        // --- COMPLEX ROOTS ------------------------------------------------------------------------------
        Sturm complex = new Sturm(Poly.of(1, 0, 1));
        check(complex.totalRealRoots() == 0 && complex.isolate(1e-6).isEmpty(),
                "x^2+1 has no real roots");
        Sturm quartic = new Sturm(Poly.of(1, 0, 0, 0, 1));
        check(quartic.isolate(1e-6).isEmpty() && Poly.of(1, 0, 0, 0, 1).degree() == 4,
                "x^4+1 has no real roots even though its degree is four");
        Sturm almost = new Sturm(Poly.of(6, 0, -5, 0, 1));
        check(almost.isolate(1e-9).size() == 4, "(x^2-2)(x^2-3) has four distinct real roots, got "
                + almost.isolate(1e-9).size());

        // --- roots exactly on bisection points and near neighbours ----------------------------------------
        Sturm boundary = new Sturm(Poly.of(0, -1, 1));
        List<Enclosure> boundaryRoots = boundary.isolate(1e-6);
        check(boundaryRoots.size() == 2, "x^2-x has roots at 0 and 1");
        check(boundary.countIn(Rational.of(0), Rational.of(1)) == 1,
                "the half-open convention must count the root at the right endpoint");
        check(boundary.countIn(Rational.of(-1), Rational.of(0)) == 1,
                "the root at zero must fall into the interval ending there");
        check(boundary.countIn(Rational.of(1), Rational.of(2)) == 0,
                "no root may be counted twice by the half-open convention");
        Poly neighbours = Poly.product(Poly.of(-100, 100), Poly.of(-101, 100));
        List<Enclosure> closeRoots = new Sturm(neighbours).isolate(1e-9);
        check(closeRoots.size() == 2 && containsRoot(closeRoots, 1.0) && containsRoot(closeRoots, 1.01),
                "roots one hundredth apart must still be separated, got " + closeRoots);
        check(closeRoots.get(0).signChanges == 1 && closeRoots.get(1).signChanges == 1,
                "each of the close roots must carry its own uniqueness certificate");

        // --- degenerate and invalid inputs ---------------------------------------------------------------
        check(new Sturm(Poly.of(7)).isolate(1e-6).isEmpty(),
                "a non-zero constant polynomial has no roots");
        boolean zeroRejected = false;
        try { new Sturm(Poly.of(0L)); } catch (IllegalArgumentException ex) { zeroRejected = true; }
        check(zeroRejected, "the zero polynomial has infinitely many roots and must be rejected");
        boolean divisionRejected = false;
        try { Poly.pseudoRemainder(Poly.of(1, 1), Poly.of(0, 0)); }
        catch (IllegalArgumentException ex) { divisionRejected = true; }
        check(divisionRejected, "division by the zero polynomial must be rejected");
        check(Poly.of(-2, 1).degree() == 1 && Poly.of(0, 0, 3).degree() == 2,
                "leading zero coefficients must be normalised away");
        check(Poly.divideExact(mixed, Poly.of(4, -1, 0, 1)).multiply(Poly.of(4, -1, 0, 1)).toString()
                .equals(mixed.toString()), "exact division must invert multiplication");

        // --- determinism --------------------------------------------------------------------------------
        List<Enclosure> replay = new Sturm(mixed).isolate(1e-8);
        boolean same = replay.size() == mixedRoots.size();
        for (int i = 0; same && i < replay.size(); i++)
            same = replay.get(i).toString().equals(mixedRoots.get(i).toString());
        check(same, "the solver must be deterministic");
        check(new Sturm(Poly.of(-5, -2, 0, 1)).isolate(1e-12).get(0).toString()
                .equals(classicRoots.get(0).toString()), "a repeated solve must give identical enclosures");
        check(cubic.bound().signum() > 0 && new Sturm(Poly.of(-5, -2, 0, 1)).bound().intValue() >= 6,
                "the Cauchy bound must cover every root");

        System.out.println("Classic cubic: " + classicRoots.get(0) + " contains 2.0945514815423, refined to "
                + refined + " width " + refined.widthText());
        System.out.println("x^2-2 times (x-5): " + irrationalRoots.size() + " roots with worst width "
                + String.format(Locale.ROOT, "%.3e", worstWidth) + "; mixed degree-6 polynomial: "
                + mixedRoots.size() + " roots matching dense sampling");
        System.out.println("Multiplicity: gcd degree " + factor.degree() + " removed, square-free degree "
                + Poly.squareFreePart(repeated).degree() + "; x^2+1 and x^4+1 report no real roots");
        System.out.println("PASS");
    }

    static boolean containsRoot(List<Enclosure> enclosures, double root) {
        for (Enclosure e : enclosures) if (e.hi.toDouble() >= root && root > e.lo.toDouble()) return true;
        return false;
    }

    /** Independent oracle: count sign changes of P on a dense grid, one root per sign change. */
    static int sampledSignChanges(Poly p, double lo, double hi, int steps) {
        int changes = 0;
        double previous = p.valueAt(lo);
        for (int i = 1; i <= steps; i++) {
            double x = lo + (hi - lo) * i / steps;
            double current = p.valueAt(x);
            if (previous != 0 && current != 0 && (previous > 0) != (current > 0)) changes++;
            if (current != 0) previous = current;
        }
        return changes;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
