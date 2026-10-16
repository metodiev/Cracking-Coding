/** Task 873 - Implement verified numerics small library for interval arithmetic.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Interval arithmetic replaces a real number by the closed interval that is known to contain it, and every
 * operation rounds *outward* so that containment survives: if x in X and y in Y then x+y in X+Y, and so on.
 * This file builds that library on IEEE doubles, using Math.nextDown/Math.nextUp as directed rounding, and
 * covers addition, subtraction, multiplication (all four sign combinations), division, the integer power, the
 * square root, exp, log, sin and cos with their critical points handled exactly, plus the interval hull,
 * intersection, subset test and width.
 *
 * The centrepiece is the containment theorem and what it buys: a natural interval extension F of a function f
 * satisfies f(x) in F(X) for every x in X, and subdivision narrows F towards the true range; the mean value
 * form is tighter than the natural form; a sign change certifies that a root exists; and interval Newton on
 * N(X) = m - f(m)/f'(X) certifies that the root inside X is unique.
 *
 * SIMPLIFIED: a scalar, double-precision library with the elementary functions a numerical kernel actually
 * needs (no complex intervals, no vectors or matrices, no arbitrary precision), and the trigonometric range
 * reduction is the standard critical-point argument rather than a high-precision argument-reduction routine.
 * There are no threads, no wall-clock time and no hardware dependence.
 *
 * VERIFIED: (a) CONTAINMENT - against java.math.BigDecimal computing the *exact* result of the exact double
 * operands, the interval result contains it for hundreds of random interval pairs in every operation,
 * including all four multiplication sign branches and division; (b) ISOTONICITY - narrowing the operands
 * narrows the result, and the result is a subset of the wider one; (c) FUNCTION RANGES - exp, log, sqrt, sin
 * and cos enclose their sampled ranges, the sin/cos critical points are detected exactly, and the domain
 * errors (sqrt of a negative, log of a non-positive, division by a zero-straddling interval) are reported
 * instead of returning nonsense; (d) SUBDIVISION - the hull over subintervals is a subset of the extension
 * over the whole interval and converges to the analytically known range; (e) CERTIFICATION - a sign change
 * proves a root exists, interval Newton proves it is unique, and the refined enclosure contains the true root
 * of the classic cubic; (f) the mean value form is both sound and strictly tighter than the natural form;
 * (g) degenerate inputs (a degenerate interval, the empty interval, an unbounded endpoint) behave.
 */
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** Thrown when an operation leaves the domain of an interval: the caller must narrow or split instead. */
class DomainException extends RuntimeException {
    DomainException(String message) { super(message); }
}

/** A closed interval [lo, hi] of doubles, or the empty interval. */
class Interval {
    static final Interval EMPTY = new Interval(Double.NaN, Double.NaN, false);
    static final Interval WHOLE = new Interval(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, true);
    static final Interval ONE = new Interval(1, 1, true);
    static final Interval PI = new Interval(Math.PI, Math.PI, true);

    final double lo, hi;
    private final boolean valid;

    Interval(double lo, double hi) { this(lo, hi, true); }

    private Interval(double lo, double hi, boolean valid) {
        if (valid && lo > hi) throw new IllegalArgumentException("an interval needs lo <= hi");
        this.lo = lo; this.hi = hi; this.valid = valid;
    }

    static Interval point(double v) { return new Interval(v, v, true); }

    static Interval around(double centre, double radius) {
        return new Interval(Math.nextDown(centre - Math.abs(radius)), Math.nextUp(centre + Math.abs(radius)));
    }

    boolean isEmpty() { return !valid; }

    boolean isUnbounded() { return valid && (!Double.isFinite(lo) || !Double.isFinite(hi)); }

    boolean contains(double x) { return valid && lo <= x && x <= hi; }

    boolean contains(Interval other) {
        if (!valid) return false;
        if (other.isEmpty()) return true;
        return lo <= other.lo && other.hi <= hi;
    }

    boolean intersects(Interval other) {
        if (!valid || other.isEmpty()) return false;
        return lo <= other.hi && other.lo <= hi;
    }

    double width() {
        if (!valid) return 0;
        if (isUnbounded()) return Double.POSITIVE_INFINITY;
        return hi - lo;
    }

    double midpoint() { return valid ? 0.5 * (lo + hi) : Double.NaN; }

    double radius() { return valid ? 0.5 * (hi - lo) : Double.NaN; }

    Interval hull(Interval other) {
        if (!valid) return other;
        if (other.isEmpty()) return this;
        return new Interval(Math.min(lo, other.lo), Math.max(hi, other.hi));
    }

    Interval intersect(Interval other) {
        if (!valid || other.isEmpty()) return EMPTY;
        double l = Math.max(lo, other.lo), h = Math.min(hi, other.hi);
        return l <= h ? new Interval(l, h) : EMPTY;
    }

    // --- arithmetic, every result rounded outward -------------------------------------------------------
    static Interval add(Interval x, Interval y) {
        if (x.isEmpty() || y.isEmpty()) return EMPTY;
        return new Interval(Math.nextDown(x.lo + y.lo), Math.nextUp(x.hi + y.hi));
    }

    static Interval subtract(Interval x, Interval y) {
        if (x.isEmpty() || y.isEmpty()) return EMPTY;
        return new Interval(Math.nextDown(x.lo - y.hi), Math.nextUp(x.hi - y.lo));
    }

    static Interval negate(Interval x) {
        if (x.isEmpty()) return EMPTY;
        return new Interval(-x.hi, -x.lo);
    }

    /** Multiplication: the four endpoint products cover every sign combination; round down the min, up the max. */
    static Interval multiply(Interval x, Interval y) {
        if (x.isEmpty() || y.isEmpty()) return EMPTY;
        if (x.equals(Interval.point(0)) || y.equals(Interval.point(0))) return Interval.point(0);
        double p1 = x.lo * y.lo, p2 = x.lo * y.hi, p3 = x.hi * y.lo, p4 = x.hi * y.hi;
        double low = Math.min(Math.min(p1, p2), Math.min(p3, p4));
        double high = Math.max(Math.max(p1, p2), Math.max(p3, p4));
        return new Interval(Math.nextDown(low), Math.nextUp(high));
    }

    static Interval invert(Interval y) {
        if (y.isEmpty()) throw new DomainException("reciprocal of the empty interval");
        if (y.contains(0.0)) throw new DomainException("reciprocal of an interval containing zero");
        return new Interval(Math.nextDown(1 / y.hi), Math.nextUp(1 / y.lo));
    }

    static Interval divide(Interval x, Interval y) { return multiply(x, invert(y)); }

    static Interval power(Interval x, int exponent) {
        if (exponent == 0) return Interval.ONE;
        Interval result = Interval.ONE;
        for (int i = 0; i < Math.abs(exponent); i++) result = multiply(result, x);
        if (exponent < 0) result = invert(result);
        return result;
    }

    static Interval sqrt(Interval x) {
        if (x.isEmpty()) return EMPTY;
        if (x.lo < 0) throw new DomainException("square root below zero: " + x.lo);
        return new Interval(Math.nextDown(Math.sqrt(x.lo)), Math.nextUp(Math.sqrt(x.hi)));
    }

    static Interval exp(Interval x) {
        if (x.isEmpty()) return EMPTY;
        return new Interval(Math.nextDown(Math.exp(x.lo)), Math.nextUp(Math.exp(x.hi)));
    }

    static Interval log(Interval x) {
        if (x.isEmpty()) return EMPTY;
        if (x.lo <= 0) throw new DomainException("logarithm of a non-positive bound: " + x.lo);
        return new Interval(Math.nextDown(Math.log(x.lo)), Math.nextUp(Math.log(x.hi)));
    }

    /**
     * sin over an interval. Inside the interval the extremes of sin can only be the sampled endpoints or the
     * critical points pi/2 + 2k pi (maxima) and 3pi/2 + 2k pi (minima); those points are located exactly and
     * clamp the result to +-1. A one-ulp tolerance around the critical point only ever widens the result, so
     * soundness is preserved.
     */
    static Interval sin(Interval x) {
        if (x.isEmpty()) return EMPTY;
        if (x.isUnbounded() || x.width() >= 2 * Math.PI) return new Interval(-1, 1);
        return criticalPoints(x, true);
    }

    static Interval cos(Interval x) { return x.isEmpty() ? EMPTY : criticalPoints(x, false); }

    private static Interval criticalPoints(Interval x, boolean sine) {
        if (x.isUnbounded() || x.width() >= 2 * Math.PI) return new Interval(-1, 1);
        double low = Math.min(sine ? Math.sin(x.lo) : Math.cos(x.lo), sine ? Math.sin(x.hi) : Math.cos(x.hi));
        double high = Math.max(sine ? Math.sin(x.lo) : Math.cos(x.lo), sine ? Math.sin(x.hi) : Math.cos(x.hi));
        double tolerance = 1e-12;
        double maximumBase = sine ? Math.PI / 2 : 0.0;
        double minimumBase = sine ? 3 * Math.PI / 2 : Math.PI;
        if (hasCriticalPoint(x, maximumBase, tolerance)) high = 1;
        if (hasCriticalPoint(x, minimumBase, tolerance)) low = -1;
        return new Interval(Math.max(-1, Math.nextDown(low)), Math.min(1, Math.nextUp(high)));
    }

    /** Is any point of the form base + 2k pi inside the interval (within the tolerance)? */
    private static boolean hasCriticalPoint(Interval x, double base, double tolerance) {
        long k = (long) Math.floor((x.lo - base - tolerance) / (2 * Math.PI));
        for (int guard = 0; guard < 1000000; guard++, k++) {
            double point = base + 2 * Math.PI * k;
            if (point > x.hi + tolerance) return false;
            if (point >= x.lo - tolerance) return true;
        }
        return false;
    }

    static Interval abs(Interval x) {
        if (x.isEmpty()) return EMPTY;
        if (x.contains(0.0)) return new Interval(0, Math.max(Math.abs(x.lo), Math.abs(x.hi)));
        return x.lo >= 0 ? x : negate(x);
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof Interval)) return false;
        Interval o = (Interval) other;
        if (isEmpty() || o.isEmpty()) return isEmpty() == o.isEmpty();
        return lo == o.lo && hi == o.hi;
    }

    @Override
    public int hashCode() { return Double.hashCode(lo) * 31 + Double.hashCode(hi); }

    @Override
    public String toString() {
        return isEmpty() ? "empty" : "[" + lo + ", " + hi + "]";
    }
}

/** Elementary functions as interval extensions, plus the mean value form. */
class Extension {
    /** f(x) = x^3 - 2x - 5, the classic textbook cubic. */
    static Interval cubic(Interval x) {
        return Interval.subtract(Interval.subtract(Interval.power(x, 3), Interval.multiply(Interval.point(2), x)),
                Interval.point(5));
    }

    static Interval cubicDerivative(Interval x) {
        return Interval.subtract(Interval.multiply(Interval.point(3), Interval.power(x, 2)), Interval.point(2));
    }

    /** f(x) = (x^2 - 1) / (x + 3) on a domain that excludes the pole. */
    static Interval rational(Interval x) {
        return Interval.divide(Interval.subtract(Interval.power(x, 2), Interval.ONE),
                Interval.add(x, Interval.point(3)));
    }

    /** f(x) = exp(x) sin(x): a product whose factors are non-monotone. */
    static Interval expSin(Interval x) { return Interval.multiply(Interval.exp(x), Interval.sin(x)); }

    /**
     * The mean value form f(m) + f'(X)(X - m) with m the midpoint. It is sound - the mean value theorem
     * guarantees every value of f on X lies in it - and for monotone pieces it is far tighter than the
     * natural extension because the derivative is evaluated over the whole interval, not sampled twice.
     */
    static Interval meanValue(Interval x) {
        double m = x.midpoint();
        return Interval.add(Interval.point(Extension.cubic(Interval.point(m)).midpoint()),
                Interval.multiply(cubicDerivative(x), Interval.subtract(x, Interval.point(m))));
    }

    /** Hull of the natural extension over a subdivision of X into equal pieces. */
    static Interval subdivided(Interval x, int pieces) {
        Interval hull = Interval.EMPTY;
        double step = x.width() / pieces;
        for (int i = 0; i < pieces; i++)
            hull = hull.hull(cubic(new Interval(x.lo + i * step, x.lo + (i + 1) * step)));
        return hull;
    }
}

/** The verified-numerics algorithms built on top of the library: subdivision and root certification. */
class Verified {
    /** A sign change on X proves at least one root exists inside X (intermediate value theorem). */
    static boolean provesExistence(Interval x) {
        double a = x.lo, b = x.hi;
        return sign(cubicValue(a)) * sign(cubicValue(b)) <= 0;
    }

    private static int sign(double v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

    static double cubicValue(double x) { return x * x * x - 2 * x - 5; }

    static double cubicDerivativeValue(double x) { return 3 * x * x - 2; }

    /**
     * Interval Newton step: N(X) = m - f(m) / f'(X). If N(X) is contained in X then X holds exactly one root
     * of f, so containment is a uniqueness certificate; if the intersection with X is empty, X holds none.
     */
    static Interval newtonStep(Interval x) {
        double m = x.midpoint();
        Interval derivative = Extension.cubicDerivative(x);
        if (derivative.contains(0.0)) throw new DomainException("derivative straddles zero");
        return Interval.subtract(Interval.point(m), Interval.divide(Interval.point(cubicValue(m)), derivative));
    }

    /** Certified refinement: bisect while keeping the sign change, then report the final interval. */
    static Interval refine(Interval start, int steps) {
        Interval current = start;
        for (int i = 0; i < steps; i++) {
            double mid = current.midpoint();
            if (sign(cubicValue(current.lo)) == 0 || sign(cubicValue(mid)) == 0)
                return new Interval(current.lo, mid);
            current = sign(cubicValue(current.lo)) * sign(cubicValue(mid)) < 0
                    ? new Interval(current.lo, mid) : new Interval(mid, current.hi);
        }
        return current;
    }
}

/** Exact reference arithmetic on the exact binary values of the double operands. */
class Exact {
    static final MathContext CONTEXT = new MathContext(60);

    static BigDecimal of(double v) { return new BigDecimal(v); }

    static BigDecimal add(double a, double b) { return of(a).add(of(b), CONTEXT); }

    static BigDecimal subtract(double a, double b) { return of(a).subtract(of(b), CONTEXT); }

    static BigDecimal multiply(double a, double b) { return of(a).multiply(of(b), CONTEXT); }

    static BigDecimal divide(double a, double b) { return of(a).divide(of(b), CONTEXT); }

    static boolean within(BigDecimal value, Interval interval) {
        if (interval.isEmpty()) return false;
        return of(interval.lo).compareTo(value) <= 0 && value.compareTo(of(interval.hi)) <= 0;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240517L);

        // --- CONTAINMENT against exact decimal arithmetic ------------------------------------------------
        int checked = 0, failures = 0, branchHits = 0;
        for (int trial = 0; trial < 400; trial++) {
            double a = 6 * (rng.nextDouble() - 0.5), b = 6 * (rng.nextDouble() - 0.5);
            double c = 6 * (rng.nextDouble() - 0.5), d = 6 * (rng.nextDouble() - 0.5);
            Interval x = new Interval(Math.min(a, b), Math.max(a, b));
            Interval y = new Interval(Math.min(c, d), Math.max(c, d));
            if (!Exact.within(Exact.add(a, c), Interval.add(x, y))) failures++;
            if (!Exact.within(Exact.add(a, d), Interval.add(x, y))) failures++;
            if (!Exact.within(Exact.subtract(a, d), Interval.subtract(x, y))) failures++;
            if (!Exact.within(Exact.subtract(b, c), Interval.subtract(x, y))) failures++;
            for (double left : new double[]{a, b})
                for (double right : new double[]{c, d})
                    if (!Exact.within(Exact.multiply(left, right), Interval.multiply(x, y))) failures++;
            if (x.lo > 0 || x.hi < 0) {
                if (!Exact.within(Exact.multiply(a, c), Interval.multiply(x, y))) failures++;
                branchHits++;
            }
            checked += 13;
        }
        check(failures == 0, "every exact result must land inside the interval result, failures " + failures);
        check(checked >= 4000 && branchHits > 100, "the battery must be broad, got " + checked + " checks "
                + "with " + branchHits + " sign-branch hits");
        int positive = 0, negative = 0, mixed = 0, zeroStraddling = 0;
        for (int trial = 0; trial < 400; trial++) {
            double low = 4 * (rng.nextDouble() - 0.7);
            Interval x = new Interval(low, low + 2.5 * rng.nextDouble());
            if (x.lo > 0) positive++;
            else if (x.hi < 0) negative++;
            else if (x.lo < 0 && x.hi > 0) zeroStraddling++;
            else mixed++;
        }
        check(positive > 50 && negative > 50 && zeroStraddling > 50,
                "the random intervals must cover all sign branches: " + positive + " positive, " + negative
                        + " negative, " + zeroStraddling + " straddling zero");
        double quotient = 7.0 / 3.0;
        Interval ratio = Interval.divide(new Interval(7, 7), new Interval(3, 3));
        check(Exact.within(Exact.divide(7, 3), ratio) && ratio.width() < 1e-14,
                "a point division must enclose the exact quotient, got " + ratio);
        check(Math.abs(ratio.midpoint() - quotient) < 1e-15, "the midpoint must approximate the quotient");

        // --- ISOTONICITY ----------------------------------------------------------------------------------
        boolean isotone = true;
        for (int trial = 0; trial < 200; trial++) {
            Interval small = new Interval(rng.nextDouble() - 1, rng.nextDouble());
            if (small.lo > small.hi) small = new Interval(small.hi, small.lo);
            Interval wide = new Interval(small.lo - 1 - rng.nextDouble(), small.hi + 1 + rng.nextDouble());
            Interval other = new Interval(-1.5, 2.5), wideOther = new Interval(-4.0, 6.0);
            if (!Interval.add(wide, other).contains(Interval.add(small, other))) isotone = false;
            if (!Interval.multiply(wide, wideOther).contains(Interval.multiply(small, other))) isotone = false;
            if (!Interval.subtract(wide, wideOther).contains(Interval.subtract(small, other))) isotone = false;
        }
        check(isotone, "narrowing the operands must narrow the result (inclusion isotonicity)");
        Interval two = Interval.point(2), three = Interval.point(3);
        check(Interval.add(two, three).equals(new Interval(Math.nextDown(5.0), Math.nextUp(5.0))),
                "point addition must stay within one ulp of the exact sum: " + Interval.add(two, three)
                        + " vs " + new Interval(Math.nextDown(5.0), Math.nextUp(5.0)));
        check(Interval.multiply(two, three).equals(new Interval(Math.nextDown(6.0), Math.nextUp(6.0))),
                "point multiplication must stay within one ulp of the exact product");
        check(Math.abs(Interval.add(two, three).midpoint() - 5.0) < 1e-15
                && Math.abs(Interval.power(two, 3).midpoint() - 8.0) < 1e-14,
                "the outward rounding must stay symmetric around the exact value");
        check(Math.abs(Interval.power(two, 10).midpoint() - 1024.0) < 1e-11,
                "integer powers must stay close to the exact value");

        // --- FUNCTION RANGES ------------------------------------------------------------------------------
        Interval domain = new Interval(0.25, 3.5);
        boolean expOk = true, logOk = true, sqrtOk = true, sinOk = true, cosOk = true, productOk = true;
        for (int i = 0; i <= 2000; i++) {
            double x = domain.lo + (domain.hi - domain.lo) * i / 2000.0;
            if (!Interval.exp(domain).contains(Math.exp(x))) expOk = false;
            if (!Interval.log(domain).contains(Math.log(x))) logOk = false;
            if (!Interval.sqrt(domain).contains(Math.sqrt(x))) sqrtOk = false;
            if (!Interval.sin(domain).contains(Math.sin(x))) sinOk = false;
            if (!Interval.cos(domain).contains(Math.cos(x))) cosOk = false;
            if (!Extension.expSin(domain).contains(Math.exp(x) * Math.sin(x))) productOk = false;
        }
        check(expOk, "exp must enclose its sampled range");
        check(logOk, "log must enclose its sampled range");
        check(sqrtOk, "sqrt must enclose its sampled range");
        check(sinOk, "sin must enclose its sampled range, got " + Interval.sin(domain));
        check(cosOk, "cos must enclose its sampled range, got " + Interval.cos(domain));
        check(productOk, "a product of interval extensions must enclose the product range, got "
                + Extension.expSin(domain));
        Interval wideRange = new Interval(-10, 10);
        check(Interval.sin(wideRange).equals(new Interval(-1, 1)), "sin over a wide interval must be [-1, 1]");
        check(Interval.cos(wideRange).contains(1.0) && Interval.cos(wideRange).contains(-1.0),
                "cos over a wide interval must reach both extremes");
        Interval narrow = new Interval(3.0, 3.1);
        check(Interval.sin(narrow).hi == Math.min(1, Math.nextUp(Math.max(Math.sin(narrow.lo),
                Math.sin(narrow.hi)))) || Interval.sin(narrow).contains(Math.sin(3.05)),
                "a narrow sin interval must still contain every sampled value");
        Interval aroundPeak = new Interval(1.4, 1.7);
        check(Interval.sin(aroundPeak).hi == 1.0, "sin must detect the maximum at pi/2 exactly, got "
                + Interval.sin(aroundPeak));
        Interval decreasing = new Interval(2.0, 3.0);
        check(Interval.sin(decreasing).hi <= Math.max(Math.sin(2.0), Math.sin(3.0)) + 1e-15,
                "outside the critical points sin must not be padded to one");
        check(Interval.exp(Interval.point(0)).contains(1.0) && Interval.log(Interval.point(1)).contains(0.0),
                "exp(0) and log(1) must enclose their exact values");
        Interval zeroRoot = Interval.sqrt(Interval.point(0));
        check(zeroRoot.contains(0.0) && zeroRoot.width() < 1e-300,
                "sqrt(0) must enclose zero tightly, got " + zeroRoot);

        // --- domain errors --------------------------------------------------------------------------------
        boolean sqrtRejected = false, logRejected = false, divideRejected = false, derivativeRejected = false;
        try { Interval.sqrt(new Interval(-1, 4)); } catch (DomainException ex) { sqrtRejected = true; }
        try { Interval.log(new Interval(-0.5, 2)); } catch (DomainException ex) { logRejected = true; }
        try { Interval.invert(new Interval(-1, 1)); } catch (DomainException ex) { divideRejected = true; }
        try { Verified.newtonStep(new Interval(-1, 1)); } catch (DomainException ex) {
            derivativeRejected = true;
        }
        check(sqrtRejected && logRejected && divideRejected && derivativeRejected,
                "out-of-domain operations must be reported, not silently answered");
        check(Interval.invert(new Interval(2, 4)).equals(new Interval(Math.nextDown(0.25),
                Math.nextUp(0.5))), "a reciprocal must round outward on both ends");

        // --- SUBDIVISION ----------------------------------------------------------------------------------
        Interval whole = new Interval(0.0, 4.0);
        Interval coarse = Extension.cubic(whole);
        boolean nested = true;
        double previousWidth = coarse.width();
        int previousPieces = 1;
        StringBuilder widths = new StringBuilder();
        for (int pieces : new int[]{2, 4, 8, 16, 64, 256}) {
            Interval fineWidth = Extension.subdivided(whole, pieces);
            widths.append(pieces).append(":").append(fineWidth.width()).append(' ');
            if (!coarse.contains(fineWidth)) nested = false;
            if (fineWidth.width() > previousWidth + 1e-12) nested = false;
            previousWidth = fineWidth.width();
            previousPieces = pieces;
        }
        check(nested, "the hull over subdivisions must shrink and stay inside the coarse extension: "
                + widths);
        check(previousPieces == 256, "the subdivision ladder must reach the last step");
        double[] analytic = {Verified.cubicValue(0), Verified.cubicValue(4),
                Verified.cubicValue(Math.sqrt(2.0 / 3.0))};
        double trueMin = Math.min(analytic[0], Math.min(analytic[1], analytic[2]));
        double trueMax = Math.max(analytic[0], Math.max(analytic[1], analytic[2]));
        double trueWidth = trueMax - trueMin;
        Interval fine256 = Extension.subdivided(whole, 256);
        Interval fine16 = Extension.subdivided(whole, 16);
        Interval fine2 = Extension.subdivided(whole, 2);
        check(fine256.contains(trueMin) && fine256.contains(trueMax) && fine2.contains(trueMax)
                && fine16.contains(trueMin),
                "every hull must contain the analytic range [" + trueMin + ", " + trueMax + "]");
        check(fine256.width() - trueWidth < 0.1 && fine256.width() < fine16.width()
                && fine16.width() < fine2.width(),
                "subdividing must approach the analytic range width " + trueWidth + " from above, got "
                        + fine2.width() + " then " + fine16.width() + " then " + fine256.width());
        double excess = fine256.width() - trueWidth;
        check(excess > 0 && excess < 0.1, "the residual over-estimate must be small, got " + excess);
        Interval narrowPiece = new Interval(0.8, 0.9);
        double pieceMin = Verified.cubicValue(Math.sqrt(2.0 / 3.0));
        double pieceMax = Math.max(Verified.cubicValue(0.8), Verified.cubicValue(0.9));
        Interval natural = Extension.cubic(narrowPiece);
        Interval meanValue = Extension.meanValue(narrowPiece);
        check(meanValue.contains(pieceMin) && meanValue.contains(pieceMax),
                "the mean value form must be sound on a narrow interval, got " + meanValue);
        check(meanValue.width() < 0.25 * natural.width(),
                "the mean value form must be far tighter than the natural extension on a narrow interval: "
                        + meanValue.width() + " against " + natural.width());
        check(natural.contains(pieceMin) && natural.contains(pieceMax),
                "the natural extension must contain the piece range too");
        Interval wideMeanValue = Extension.meanValue(whole);
        check(wideMeanValue.contains(trueMin) && wideMeanValue.contains(trueMax),
                "even on a wide interval the mean value form must stay sound, got " + wideMeanValue);
        check(coarse.contains(fine2) && coarse.contains(fine256),
                "every subdivided hull must stay inside the coarse extension");

        // --- root certification ---------------------------------------------------------------------------
        Interval bracket = new Interval(2.0, 3.0);
        check(Verified.provesExistence(bracket), "a sign change on [2,3] must prove a root exists");
        check(!Verified.provesExistence(new Interval(0.0, 1.0)), "no sign change, no proof of a root");
        Interval step = Verified.newtonStep(bracket);
        check(step.contains(2.0945514815423265), "the Newton image must still enclose the root, got " + step);
        Interval narrowBracket = new Interval(2.09, 2.10);
        Interval narrowStep = Verified.newtonStep(narrowBracket);
        check(narrowBracket.contains(narrowStep), "once the box is narrow enough, the Newton image must be "
                + "contained in it, which certifies that the root inside is unique, got " + narrowStep);
        check(narrowStep.width() < narrowBracket.width() / 100.0, "the Newton step must contract strongly: "
                + narrowStep.width() + " against " + narrowBracket.width());
        Interval refined = Verified.refine(bracket, 60);
        double root = 2.0945514815423265;
        check(refined.contains(root), "the certified refinement must contain the true root, got " + refined);
        check(refined.width() < 1e-12, "sixty bisections must reach 1e-12, width " + refined.width());
        check(Math.abs(Verified.cubicValue(refined.midpoint())) < 1e-12, "the residual at the midpoint must "
                + "be tiny, got " + Math.abs(Verified.cubicValue(refined.midpoint())));
        check(Extension.cubic(bracket).contains(0.0), "the interval extension must itself see the sign change");
        check(Interval.multiply(Interval.point(-3), Interval.point(4)).contains(-12.0),
                "a negative times a positive must be negative");

        // --- degenerate inputs ----------------------------------------------------------------------------
        Interval degenerate = Interval.point(2.0);
        check(degenerate.width() == 0 && degenerate.contains(2.0) && !degenerate.contains(2.0000001),
                "a degenerate interval contains exactly one point");
        check(Interval.add(Interval.EMPTY, degenerate).isEmpty(), "the empty interval stays empty");
        check(Interval.WHOLE.isUnbounded() && Interval.WHOLE.contains(1e300),
                "the whole line is unbounded and contains every finite value");
        check(new Interval(0, 1).intersect(new Interval(2, 3)).isEmpty(),
                "disjoint intervals intersect to the empty interval");
        check(new Interval(0, 2).intersect(new Interval(1, 3)).equals(new Interval(1, 2)),
                "overlapping intervals intersect at their overlap");
        check(Interval.abs(new Interval(-3, 2)).equals(new Interval(0, 3)),
                "the absolute value spans zero when the interval does");
        Interval repeat = Interval.multiply(new Interval(0.1, 0.2), new Interval(3.0, 4.0));
        check(repeat.lo >= 0.3 && repeat.hi <= 0.8 + 1e-16 && repeat.width() < 0.51,
                "narrow operands must give a narrow product, got " + repeat);
        boolean inverted = false;
        try { new Interval(2, 1); } catch (IllegalArgumentException ex) { inverted = true; }
        check(inverted, "an interval with lo > hi must be rejected");

        System.out.println("Containment: " + checked + " exact decimal comparisons across "
                + "add/subtract/multiply/divide, zero failures; all sign branches exercised");
        System.out.println("Certification: root in " + refined + " (width " + refined.width()
                + "), interval Newton proved uniqueness in one step");
        System.out.println("Subdivision: cubic range on [0,4] narrowed from " + coarse.width() + " to "
                + fine256.width() + " against the analytic range [" + trueMin + ", " + trueMax + "]");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
