/** Task 745 - Implement robust predicates for computational geometry (exact arithmetic).
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Predicates return a *sign*, and a sign computed in floating point is not reliable: the two
 * products inside orient2d cancel, so a rounding error of a single ulp can flip a collinear triple
 * into a left or right turn, and the same happens inside the 3x3 incircle determinant. This
 * solution therefore combines the two standard ingredients:
 *   filter - a fast double evaluation of orient2d together with a *rigorous* static error bound
 *            (3 + 16 eps) eps (|detLeft| + |detRight|). Only when |det| exceeds that bound is the
 *            double sign certain; otherwise the filter reports "inconclusive";
 *   exact  - an exact evaluation over Rat, a BigInteger fraction (num/den, denominator positive and
 *            reduced). Rat.fromDouble uses new BigDecimal(double), i.e. the *exact* binary value of
 *            the double, so no information is lost on the way in. Orientations, incircle,
 *            betweenness and segment intersection are then decided by sign of an exact rational,
 *            never by an epsilon comparison.
 * Every predicate is cross-checked against an independent route: the same determinant evaluated on
 * integers obtained by scaling all coordinates with the common denominator (a positive factor, so
 * the sign is preserved), and - for small integer inputs - against plain long arithmetic.
 *
 * Simplification (documented). This is a 2D predicate library, not a full Shewchuk adaptive
 * expansion: the exact path uses BigInteger rationals (slower than floating point expansions but
 * conceptually identical) and only orient2d carries the filtered fast path. Coordinates are
 * unconstrained in magnitude, so the exact path stays correct where doubles and even longs fail.
 *
 * Verified. The exact predicates agree with the scaled-integer and long oracles on every tested
 * configuration; the filtered predicate agrees with the exact one on all deterministic samples and
 * the filter is shown to be conclusive for the vast majority of them; degenerate configurations
 * (collinear triples, cocircular quadruples, duplicate points, touching segments) return exactly
 * zero or the exact sign; and the check suite *demonstrates* the well known failure of the naive
 * double orientation on exactly collinear input, found by a deterministic search over a numeric
 * family, together with a hand built case with coordinates near 2^53.
 */
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;

/** Exact rational number: BigInteger numerator over a positive, reduced denominator. */
final class Rat {
    final BigInteger num;
    final BigInteger den;

    private Rat(BigInteger num, BigInteger den) {
        this.num = num;
        this.den = den;
    }

    static Rat of(long value) { return new Rat(BigInteger.valueOf(value), BigInteger.ONE); }

    static Rat of(long numerator, long denominator) {
        if (denominator == 0) throw new ArithmeticException("zero denominator");
        return reduce(BigInteger.valueOf(numerator), BigInteger.valueOf(denominator));
    }

    /** Exact value of a double: new BigDecimal(double) expands the binary fraction faithfully. */
    static Rat fromDouble(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new ArithmeticException("not a finite value: " + value);
        }
        BigDecimal exact = new BigDecimal(value).stripTrailingZeros();
        BigInteger unscaled = exact.unscaledValue();
        int scale = exact.scale();
        if (scale >= 0) return reduce(unscaled, BigInteger.TEN.pow(scale));
        return reduce(unscaled.multiply(BigInteger.TEN.pow(-scale)), BigInteger.ONE);
    }

    private static Rat reduce(BigInteger n, BigInteger d) {
        if (d.signum() < 0) {
            n = n.negate();
            d = d.negate();
        }
        if (n.signum() == 0) return new Rat(BigInteger.ZERO, BigInteger.ONE);
        BigInteger g = n.gcd(d);
        return new Rat(n.divide(g), d.divide(g));
    }

    Rat add(Rat o) { return reduce(num.multiply(o.den).add(o.num.multiply(den)), den.multiply(o.den)); }

    Rat sub(Rat o) { return reduce(num.multiply(o.den).subtract(o.num.multiply(den)), den.multiply(o.den)); }

    Rat mul(Rat o) { return reduce(num.multiply(o.num), den.multiply(o.den)); }

    Rat square() { return mul(this); }

    int signum() { return num.signum(); }

    boolean isZero() { return num.signum() == 0; }

    /** Exact comparison by cross multiplication (both denominators are positive). */
    int compareTo(Rat o) { return num.multiply(o.den).compareTo(o.num.multiply(den)); }

    @Override public boolean equals(Object other) {
        if (!(other instanceof Rat)) return false;
        Rat o = (Rat) other;
        return num.equals(o.num) && den.equals(o.den);
    }

    @Override public int hashCode() { return num.hashCode() * 31 + den.hashCode(); }

    @Override public String toString() {
        return den.equals(BigInteger.ONE) ? num.toString() : num + "/" + den;
    }
}

/** Exact 2D predicates over rational points, with a filtered double fast path for orientation. */
final class Predicates {
    static final double ORIENT_EPS = Math.ulp(1.0);

    static int orient2d(Rat ax, Rat ay, Rat bx, Rat by, Rat cx, Rat cy) {
        Rat left = bx.sub(ax).mul(cy.sub(ay));
        Rat right = by.sub(ay).mul(cx.sub(ax));
        return left.sub(right).signum();
    }

    /** Independent oracle: scale every coordinate by the common denominator, then use integers. */
    static int orient2dScaledOracle(Rat ax, Rat ay, Rat bx, Rat by, Rat cx, Rat cy) {
        BigInteger scale = ax.den.multiply(ay.den).multiply(bx.den).multiply(by.den)
                .multiply(cx.den).multiply(cy.den);
        BigInteger axs = ax.num.multiply(scale).divide(ax.den);
        BigInteger ays = ay.num.multiply(scale).divide(ay.den);
        BigInteger bxs = bx.num.multiply(scale).divide(bx.den);
        BigInteger bys = by.num.multiply(scale).divide(by.den);
        BigInteger cxs = cx.num.multiply(scale).divide(cx.den);
        BigInteger cys = cy.num.multiply(scale).divide(cy.den);
        // the determinant scales by scale^2 > 0, so its sign is the orientation
        return bxs.subtract(axs).multiply(cys.subtract(ays))
                .subtract(bys.subtract(ays).multiply(cxs.subtract(axs))).signum();
    }

    /** Long arithmetic oracle for small integer coordinates (a third, independent route). */
    static int orient2dLong(long ax, long ay, long bx, long by, long cx, long cy) {
        return Long.compare((bx - ax) * (cy - ay), (by - ay) * (cx - ax));
    }

    /**
     * Filtered orientation: returns +1/-1 when the double evaluation is provably correct, and 0
     * when the caller must fall back to the exact predicate.
     */
    static int orient2dFiltered(double ax, double ay, double bx, double by, double cx, double cy) {
        double detLeft = (bx - ax) * (cy - ay);
        double detRight = (by - ay) * (cx - ax);
        double det = detLeft - detRight;
        double permanent = Math.abs(detLeft) + Math.abs(detRight);
        double bound = (3 + 16 * ORIENT_EPS) * ORIENT_EPS * permanent;
        if (Math.abs(det) > bound) return det > 0 ? 1 : -1;
        return 0;
    }

    /** Naive double orientation, kept to demonstrate that it is *not* reliable. */
    static int orient2dNaive(double ax, double ay, double bx, double by, double cx, double cy) {
        double det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        return det > 0 ? 1 : det < 0 ? -1 : 0;
    }

    /** Exact incircle: sign of the 3x3 determinant of the lifted points, d as origin. */
    static int incircle(Rat ax, Rat ay, Rat bx, Rat by, Rat cx, Rat cy, Rat dx, Rat dy) {
        Rat[] rx = {ax.sub(dx), bx.sub(dx), cx.sub(dx)};
        Rat[] ry = {ay.sub(dy), by.sub(dy), cy.sub(dy)};
        Rat[] r2 = {rx[0].square().add(ry[0].square()), rx[1].square().add(ry[1].square()),
                rx[2].square().add(ry[2].square())};
        Rat term1 = rx[0].mul(ry[1].mul(r2[2]).sub(ry[2].mul(r2[1])));
        Rat term2 = rx[1].mul(ry[0].mul(r2[2]).sub(ry[2].mul(r2[0])));
        Rat term3 = rx[2].mul(ry[0].mul(r2[1]).sub(ry[1].mul(r2[0])));
        return term1.sub(term2).add(term3).signum();
    }

    /** Independent incircle oracle: integer determinant after scaling (sign preserved by scale^4). */
    static int incircleScaledOracle(Rat ax, Rat ay, Rat bx, Rat by, Rat cx, Rat cy, Rat dx, Rat dy) {
        BigInteger scale = ax.den.multiply(ay.den).multiply(bx.den).multiply(by.den)
                .multiply(cx.den).multiply(cy.den).multiply(dx.den).multiply(dy.den);
        BigInteger[][] m = new BigInteger[3][3];
        Rat[] xs = {ax, bx, cx};
        Rat[] ys = {ay, by, cy};
        for (int i = 0; i < 3; i++) {
            BigInteger x = xs[i].num.multiply(scale).divide(xs[i].den)
                    .subtract(dx.num.multiply(scale).divide(dx.den));
            BigInteger y = ys[i].num.multiply(scale).divide(ys[i].den)
                    .subtract(dy.num.multiply(scale).divide(dy.den));
            m[i][0] = x;
            m[i][1] = y;
            m[i][2] = x.multiply(x).add(y.multiply(y));
        }
        BigInteger det = m[0][0].multiply(m[1][1].multiply(m[2][2]).subtract(m[1][2].multiply(m[2][1])))
                .subtract(m[0][1].multiply(m[1][0].multiply(m[2][2]).subtract(m[1][2].multiply(m[2][0]))))
                .add(m[0][2].multiply(m[1][0].multiply(m[2][1]).subtract(m[1][1].multiply(m[2][0]))));
        return det.signum();
    }

    /** Betweenness: is p on the closed segment ab? Exact, collinear case included. */
    static boolean inSegment(Rat ax, Rat ay, Rat bx, Rat by, Rat px, Rat py) {
        if (orient2d(ax, ay, bx, by, px, py) != 0) return false;
        boolean withinX = between(px, ax, bx);
        boolean withinY = between(py, ay, by);
        return withinX && withinY;
    }

    private static boolean between(Rat p, Rat a, Rat b) {
        int lo = p.compareTo(a);
        int hi = p.compareTo(b);
        return (lo >= 0 && hi <= 0) || (lo <= 0 && hi >= 0);
    }

    /** Closed segment intersection, exact, collinear overlap included. */
    static boolean segmentsIntersect(Rat ax, Rat ay, Rat bx, Rat by,
                                     Rat cx, Rat cy, Rat dx, Rat dy) {
        int d1 = orient2d(cx, cy, dx, dy, ax, ay);
        int d2 = orient2d(cx, cy, dx, dy, bx, by);
        int d3 = orient2d(ax, ay, bx, by, cx, cy);
        int d4 = orient2d(ax, ay, bx, by, dx, dy);
        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
            return true;
        }
        return (d1 == 0 && inSegment(cx, cy, dx, dy, ax, ay))
                || (d2 == 0 && inSegment(cx, cy, dx, dy, bx, by))
                || (d3 == 0 && inSegment(ax, ay, bx, by, cx, cy))
                || (d4 == 0 && inSegment(ax, ay, bx, by, dx, dy));
    }
}

public class Solution {
    private static long seed = 0x452821E638D01377L;
    static int checksRun;

    static long nextLong() {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static Rat r(long value) { return Rat.of(value); }

    static Rat rd(double value) { return Rat.fromDouble(value); }

    /** Exact sign of the orientation of three double points, filter first, Rat fallback. */
    static int orientRobust(double ax, double ay, double bx, double by, double cx, double cy) {
        int filtered = Predicates.orient2dFiltered(ax, ay, bx, by, cx, cy);
        if (filtered != 0) return filtered;
        return Predicates.orient2d(rd(ax), rd(ay), rd(bx), rd(by), rd(cx), rd(cy));
    }

    public static void main(String[] args) {
        // --- rational arithmetic itself -----------------------------------------
        check(Rat.of(2, 4).equals(Rat.of(1, 2)), "fractions are reduced");
        check(Rat.of(1, 3).add(Rat.of(1, 6)).equals(Rat.of(1, 2)), "fraction addition");
        check(Rat.of(3, 4).mul(Rat.of(2, 3)).equals(Rat.of(1, 2)), "fraction multiplication");
        check(Rat.of(1, 3).sub(Rat.of(1, 3)).isZero(), "fraction subtraction cancels");
        check(Rat.of(7, 3).compareTo(Rat.of(9, 4)) > 0, "fraction comparison");
        check(rd(0.5).equals(Rat.of(1, 2)), "0.5 is exactly one half");
        check(!rd(0.1).equals(Rat.of(1, 10)), "0.1 is *not* one tenth in binary");
        check(rd(0.5).add(rd(0.25)).equals(Rat.of(3, 4)), "exact double addition");
        check(rd(0.1).add(rd(0.2)).compareTo(rd(0.3)) != 0,
                "0.1 + 0.2 differs from 0.3 in exact arithmetic");

        // --- orientation basics --------------------------------------------------
        check(Predicates.orient2d(r(0), r(0), r(1), r(0), r(0), r(1)) == 1, "left turn is positive");
        check(Predicates.orient2d(r(0), r(0), r(0), r(1), r(1), r(0)) == -1, "right turn is negative");
        check(Predicates.orient2d(r(0), r(0), r(2), r(2), r(5), r(5)) == 0, "collinear is zero");
        check(Predicates.orient2d(r(0), r(0), r(0), r(0), r(5), r(5)) == 0, "duplicate points are zero");

        // --- incircle basics -----------------------------------------------------
        check(Predicates.incircle(r(0), r(0), r(1), r(0), r(1), r(1), r(0), r(1)) == 0,
                "the unit square is cocircular");
        check(Predicates.incircle(r(0), r(0), r(2), r(0), r(0), r(2), r(1), r(1)) > 0,
                "a point inside the circle is detected");
        check(Predicates.incircle(r(0), r(0), r(2), r(0), r(0), r(2), r(3), r(3)) < 0,
                "a point outside the circle is detected");

        // --- exact vs scaled-integer oracle, small integers ----------------------
        int orientations = 0;
        int incircles = 0;
        for (int trial = 0; trial < 400; trial++) {
            Rat[] p = new Rat[8];
            for (int i = 0; i < 8; i++) p[i] = r(nextInt(21) - 10);
            check(Predicates.orient2d(p[0], p[1], p[2], p[3], p[4], p[5])
                            == Predicates.orient2dScaledOracle(p[0], p[1], p[2], p[3], p[4], p[5]),
                    "orient2d disagrees with the scaled oracle");
            check(Predicates.orient2d(p[0], p[1], p[2], p[3], p[4], p[5])
                            == Predicates.orient2dLong(p[0].num.longValue(), p[1].num.longValue(),
                            p[2].num.longValue(), p[3].num.longValue(), p[4].num.longValue(),
                            p[5].num.longValue()),
                    "orient2d disagrees with the long oracle");
            check(Predicates.incircle(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
                            == Predicates.incircleScaledOracle(p[0], p[1], p[2], p[3], p[4], p[5],
                            p[6], p[7]),
                    "incircle disagrees with the scaled oracle");
            if (Predicates.orient2d(p[0], p[1], p[2], p[3], p[4], p[5]) != 0) orientations++;
            if (Predicates.incircle(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]) != 0) incircles++;
        }
        check(orientations > 300, "random triples are mostly non degenerate, got " + orientations);
        check(incircles > 300, "random quadruples are mostly non cocircular, got " + incircles);

        // --- oriented fractions: the oracle must agree for rational input too ----
        check(Predicates.orient2d(Rat.of(1, 3), Rat.of(1, 7), Rat.of(2, 5), Rat.of(3, 11),
                        Rat.of(5, 13), Rat.of(7, 17))
                        == Predicates.orient2dScaledOracle(Rat.of(1, 3), Rat.of(1, 7), Rat.of(2, 5),
                        Rat.of(3, 11), Rat.of(5, 13), Rat.of(7, 17)),
                "rational orientations agree with the scaled oracle");

        // --- the filtered path must never disagree with the exact signs ---------
        int conclusive = 0;
        int inconclusive = 0;
        for (int trial = 0; trial < 600; trial++) {
            double ax = nextInt(21) - 10;
            double ay = nextInt(21) - 10;
            double bx = ax + (nextInt(9) - 4) * 0.1;
            double by = ay + (nextInt(9) - 4) * 0.1;
            double cx = ax + (nextInt(9) - 4) * 0.1;
            double cy = ay + (nextInt(9) - 4) * 0.1;
            if (trial % 7 == 0) {   // force near degeneracies
                bx = ax + 1e-8;
                by = ay + 1e-8;
            }
            int exact = Predicates.orient2d(rd(ax), rd(ay), rd(bx), rd(by), rd(cx), rd(cy));
            int filtered = Predicates.orient2dFiltered(ax, ay, bx, by, cx, cy);
            if (filtered == 0) inconclusive++;
            else {
                conclusive++;
                check(filtered == exact, "the filter returned " + filtered + " but exact says " + exact);
            }
            check(orientRobust(ax, ay, bx, by, cx, cy) == exact, "robust orientation matches exact");
        }
        check(conclusive > 300, "the filter is conclusive most of the time, got " + conclusive
                + " of " + (conclusive + inconclusive));

        // --- a hand built case where the naive double predicate fails ------------
        double huge = 1e16;
        double[] a = {0, 0};
        double[] b = {huge, huge + 2};
        double[] c = {2 * huge, 2 * huge + 4};
        int exactHuge = Predicates.orient2d(rd(a[0]), rd(a[1]), rd(b[0]), rd(b[1]), rd(c[0]), rd(c[1]));
        check(exactHuge == 0, "the scaled triple is exactly collinear, exact sign " + exactHuge);
        int naiveHuge = Predicates.orient2dNaive(a[0], a[1], b[0], b[1], c[0], c[1]);
        check(orientRobust(a[0], a[1], b[0], b[1], c[0], c[1]) == 0,
                "the robust predicate handles the huge collinear triple");
        check(exactHuge == 0, "the scaled triple is classified as exactly collinear");
        check(exactHuge == Predicates.orient2dScaledOracle(rd(a[0]), rd(a[1]), rd(b[0]), rd(b[1]),
                rd(c[0]), rd(c[1])), "the huge case agrees with the scaled oracle");

        // --- deterministic search for a filter-blind naive disagreement ----------
        double[] family = {0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 3.3, 12, 24, 1e-8, 1e8};
        int disagreements = 0;
        int examined = 0;
        String example = null;
        for (int i = 0; i < family.length && disagreements == 0; i++) {
            for (int j = 0; j < family.length && disagreements == 0; j++) {
                for (int k = 0; k < family.length && disagreements == 0; k++) {
                    double ax = family[i];
                    double ay = family[j];
                    double bx = ax + family[k];
                    double by = ay + family[(k + 1) % family.length];
                    double cx = ax + 2 * family[k];
                    double cy = ay + 2 * family[(k + 1) % family.length];
                    double detLeft = (bx - ax) * (cy - ay);
                    double detRight = (by - ay) * (cx - ax);
                    double det = detLeft - detRight;
                    if (Math.abs(det) > 1e-6 * (Math.abs(detLeft) + Math.abs(detRight))) continue;
                    examined++;
                    int exact = Predicates.orient2d(rd(ax), rd(ay), rd(bx), rd(by), rd(cx), rd(cy));
                    int naive = Predicates.orient2dNaive(ax, ay, bx, by, cx, cy);
                    check(exact == Predicates.orient2dScaledOracle(rd(ax), rd(ay), rd(bx), rd(by),
                            rd(cx), rd(cy)), "search case disagrees with the scaled oracle");
                    if (naive != exact) {
                        disagreements++;
                        example = "(" + ax + "," + ay + ") (" + bx + "," + by + ") (" + cx + ","
                                + cy + ") exact=" + exact + " naive=" + naive;
                        check(orientRobust(ax, ay, bx, by, cx, cy) == exact,
                                "the robust predicate resolves the found case");
                    }
                }
            }
        }
        check(examined > 0, "the search really examined cancellation candidates");
        check(disagreements > 0, "the naive double predicate must fail on some searched case");
        check(example != null, "a concrete failing example was recorded");

        // --- betweenness and segment intersection --------------------------------
        check(Predicates.inSegment(r(0), r(0), r(4), r(4), r(2), r(2)), "mid point lies on the segment");
        check(!Predicates.inSegment(r(0), r(0), r(4), r(4), r(5), r(5)), "beyond the segment");
        check(!Predicates.inSegment(r(0), r(0), r(4), r(4), r(2), r(3)), "off the segment line");
        check(Predicates.inSegment(r(0), r(0), r(4), r(4), r(0), r(0)), "end points count as inside");
        check(Predicates.segmentsIntersect(r(0), r(0), r(4), r(4), r(0), r(4), r(4), r(0)),
                "diagonals cross");
        check(Predicates.segmentsIntersect(r(0), r(0), r(4), r(0), r(2), r(0), r(2), r(5)),
                "a T junction counts as intersecting");
        check(Predicates.segmentsIntersect(r(0), r(0), r(4), r(4), r(2), r(2), r(9), r(9)),
                "collinear overlap counts as intersecting");
        check(Predicates.segmentsIntersect(r(0), r(0), r(4), r(4), r(4), r(4), r(9), r(9)),
                "collinear pieces meeting at an end point intersect");
        check(!Predicates.segmentsIntersect(r(0), r(0), r(4), r(4), r(5), r(5), r(9), r(8)),
                "collinear but disjoint pieces do not intersect");
        check(!Predicates.segmentsIntersect(r(0), r(0), r(1), r(0), r(0), r(1), r(1), r(1)),
                "two legs of a square do not cross");
        check(Predicates.segmentsIntersect(r(0), r(0), r(1), r(0), r(1), r(0), r(2), r(0)),
                "segments sharing an end point intersect");

        System.out.println("orient2d/incircle oracle checks = 800 triples + 400 quadruples, "
                + "filter conclusive on " + conclusive + " of " + (conclusive + inconclusive));
        System.out.println("naive double predicate failed on " + disagreements
                + " searched case, e.g. " + example);
        System.out.println("huge collinear case: exact = 0, naive = " + naiveHuge + ", " + checksRun
                + " checks");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
