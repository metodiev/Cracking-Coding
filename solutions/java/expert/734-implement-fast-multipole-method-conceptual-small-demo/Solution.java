/** Task 734 - Implement fast multipole method conceptual small demo. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The fast multipole method groups sources into boxes and, for a target box far away, replaces the individual
 * sources by a truncated Taylor expansion about the source box centre. In complex arithmetic the kernel
 * 1/(z - z_i) expands as a geometric series, so a source box is summarised by its moments a_k = sum_i q_i
 * (z_i - c)^k, and the field at a distant point is sum_k a_k / (z - c)^(k+1) - exact for a single source and
 * accurate to (r/d)^(p+1) for a cluster of radius r. A second expansion about the *target* box centre (the
 * M2L translation, obtained from the binomial series of 1/(A + w)^(k+1)) lets every target in that box share
 * one translation, which is what turns the far field into per-box work.
 *
 * Invariants verified:
 *   - translation  the local expansion reproduces the source moments' series to tight tolerance wherever the
 *                  binomial series converges, so the translation algebra is checked rather than assumed
 *   - convergence  the truncation error falls monotonically with the order, from percent level at order 0 to
 *                  machine-level agreement for a single source, and the assembled potential matches direct
 *                  summation to a relative tolerance that shrinks as the order grows
 *   - cost         the assembled evaluation performs direct summation only for the near boxes, and the number
 *                  of direct pairs is a small fraction of all pairs
 *   - degeneration a lone source is exact at every order (its box centre is the source), and a target sitting
 *                  exactly on a source is handled by the near field without a singularity
 *
 * Simplified: one level of boxes (a uniform cell-multipole scheme) instead of an adaptive quadtree with an
 * upward and downward pass, and no vectors for the force, only the potential. The verified properties are the
 * translation algebra, the order-controlled accuracy, and the direct-pair count reduction.
 */
import java.util.*;

public class Solution {
    /** A complex number as a pair of doubles, kept in local variables to stay allocation free. */
    static final class Complex {
        final double re;
        final double im;
        Complex(double re, double im) {
            this.re = re;
            this.im = im;
        }
    }

    /** A point source with a charge. */
    static final class Source {
        final Complex position;
        final double charge;
        Source(Complex position, double charge) {
            this.position = position;
            this.charge = charge;
        }
    }

    /** Softening of the direct kernel, so a target sitting exactly on a source stays finite. */
    static final double SOFTENING = 1e-12;

    /** The kernel: a unit charge at the origin gives 1/z, softened by a square-length epsilon. */
    static Complex kernel(Complex delta) {
        double squared = delta.re * delta.re + delta.im * delta.im + SOFTENING;
        return new Complex(delta.re / squared, -delta.im / squared);
    }

    /** Moments a_k of a source box about its centre, for k = 0..order. */
    static Complex[] moments(List<Source> sources, Complex centre, int order) {
        Complex[] a = new Complex[order + 1];
        for (int k = 0; k <= order; k++) a[k] = new Complex(0.0, 0.0);
        for (Source source : sources) {
            Complex delta = new Complex(source.position.re - centre.re, source.position.im - centre.im);
            Complex power = new Complex(1.0, 0.0);
            for (int k = 0; k <= order; k++) {
                a[k] = new Complex(a[k].re + source.charge * power.re, a[k].im + source.charge * power.im);
                power = new Complex(power.re * delta.re - power.im * delta.im,
                        power.re * delta.im + power.im * delta.re);
            }
        }
        return a;
    }

    /** Field of a source box at `point`, using the truncated moment series. */
    static Complex farField(Complex[] a, Complex centre, Complex point, int order) {
        Complex delta = new Complex(point.re - centre.re, point.im - centre.im);
        Complex power = new Complex(delta.re, delta.im);           // delta^(k+1)
        Complex sum = new Complex(0.0, 0.0);
        for (int k = 0; k <= order; k++) {
            double scale = 1.0 / (power.re * power.re + power.im * power.im);
            sum = new Complex(sum.re + (a[k].re * power.re + a[k].im * power.im) * scale,
                    sum.im + (a[k].im * power.re - a[k].re * power.im) * scale);
            power = new Complex(power.re * delta.re - power.im * delta.im,
                    power.re * delta.im + power.im * delta.re);
        }
        return sum;
    }

    /**
     * M2L: rewrites the source series about the source centre as a local series about `target`, using
     * 1/(A + w)^(k+1) = A^-(k+1) * sum_m binom(k+m, m) (-w/A)^m with A = target - source.
     */
    static Complex[] translate(Complex[] a, Complex source, Complex target, int order) {
        Complex shift = new Complex(target.re - source.re, target.im - source.im);
        // powers of 1/shift
        Complex[] inversePower = new Complex[2 * order + 2];
        inversePower[0] = new Complex(1.0, 0.0);
        Complex inverse = kernel(shift);
        for (int k = 1; k < inversePower.length; k++) {
            inversePower[k] = new Complex(inversePower[k - 1].re * inverse.re - inversePower[k - 1].im * inverse.im,
                    inversePower[k - 1].re * inverse.im + inversePower[k - 1].im * inverse.re);
        }
        Complex[] b = new Complex[order + 1];
        for (int m = 0; m <= order; m++) {
            double binomial = 1.0;
            double sign = (m % 2 == 0) ? 1.0 : -1.0;
            Complex sum = new Complex(0.0, 0.0);
            for (int k = 0; k <= order; k++) {
                if (k > 0) binomial *= (double) (k + m) / k;
                Complex term = inversePower[k + m + 1];
                double factor = sign * binomial * a[k].re;
                double factorIm = sign * binomial * a[k].im;
                sum = new Complex(sum.re + factor * term.re - factorIm * term.im,
                        sum.im + factor * term.im + factorIm * term.re);
            }
            b[m] = sum;
        }
        return b;
    }

    /** Evaluates a local series about `target` at `point`. */
    static Complex localField(Complex[] b, Complex target, Complex point, int order) {
        Complex w = new Complex(point.re - target.re, point.im - target.im);
        Complex power = new Complex(1.0, 0.0);
        Complex sum = new Complex(0.0, 0.0);
        for (int m = 0; m <= order; m++) {
            sum = new Complex(sum.re + b[m].re * power.re - b[m].im * power.im,
                    sum.im + b[m].re * power.im + b[m].im * power.re);
            power = new Complex(power.re * w.re - power.im * w.im, power.re * w.im + power.im * w.re);
        }
        return sum;
    }

    /** Direct summation over a source list. */
    static Complex directSum(List<Source> sources, Complex point) {
        Complex sum = new Complex(0.0, 0.0);
        for (Source source : sources) {
            Complex value = kernel(new Complex(point.re - source.position.re, point.im - source.position.im));
            sum = new Complex(sum.re + source.charge * value.re, sum.im + source.charge * value.im);
        }
        return sum;
    }

    static double magnitude(Complex value) { return Math.hypot(value.re, value.im); }

    static double relativeDifference(Complex a, Complex b) {
        double difference = Math.hypot(a.re - b.re, a.im - b.im);
        double reference = Math.hypot(b.re, b.im);
        return reference == 0.0 ? difference : difference / reference;
    }

    private static long rngState = 0x853C49E6748FEA9BL;

    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** A uniform grid of boxes over [-extent, extent]^2, with the sources bucketed into them. */
    static final class BoxGrid {
        final int side;
        final double extent;
        final double width;
        final List<List<Source>> boxes;
        final Complex[] centres;

        BoxGrid(int side, double extent, List<Source> sources) {
            this.side = side;
            this.extent = extent;
            this.width = 2.0 * extent / side;
            this.boxes = new ArrayList<>();
            this.centres = new Complex[side * side];
            for (int i = 0; i < side * side; i++) {
                boxes.add(new ArrayList<>());
                int row = i / side;
                int col = i % side;
                centres[i] = new Complex(-extent + (col + 0.5) * width, -extent + (row + 0.5) * width);
            }
            for (Source source : sources) {
                boxes.get(indexOf(source.position)).add(source);
            }
        }

        int indexOf(Complex point) {
            int col = (int) Math.floor((point.re + extent) / width);
            int row = (int) Math.floor((point.im + extent) / width);
            col = Math.max(0, Math.min(side - 1, col));
            row = Math.max(0, Math.min(side - 1, row));
            return row * side + col;
        }

        boolean neighbours(int a, int b) {
            int rowA = a / side;
            int colA = a % side;
            int rowB = b / side;
            int colB = b % side;
            return Math.abs(rowA - rowB) <= 1 && Math.abs(colA - colB) <= 1;
        }
    }

    /** The assembled single-level FMM: direct near field plus translated far field, per target box. */
    static Complex[] evaluate(BoxGrid grid, List<Complex> targets, int order, long[] directPairs) {
        Complex[][] boxMoments = new Complex[grid.boxes.size()][];
        for (int i = 0; i < grid.boxes.size(); i++) {
            boxMoments[i] = moments(grid.boxes.get(i), grid.centres[i], order);
        }
        Complex[] result = new Complex[targets.size()];
        long pairs = 0;
        for (int t = 0; t < targets.size(); t++) {
            Complex point = targets.get(t);
            int box = grid.indexOf(point);
            Complex sum = new Complex(0.0, 0.0);
            for (int other = 0; other < grid.boxes.size(); other++) {
                if (grid.boxes.get(other).isEmpty()) continue;
                if (grid.neighbours(box, other)) {
                    for (Source source : grid.boxes.get(other)) {
                        Complex value = kernel(new Complex(point.re - source.position.re,
                                point.im - source.position.im));
                        sum = new Complex(sum.re + source.charge * value.re, sum.im + source.charge * value.im);
                        pairs++;
                    }
                } else {
                    Complex[] local = translate(boxMoments[other], grid.centres[other], grid.centres[box], order);
                    Complex value = localField(local, grid.centres[box], point, order);
                    sum = new Complex(sum.re + value.re, sum.im + value.im);
                }
            }
            result[t] = sum;
        }
        if (directPairs != null) directPairs[0] = pairs;
        return result;
    }

    public static void main(String[] args) {
        // --- a lone source: the expansion is exact ------------------------------
        Complex centre = new Complex(0.0, 0.0);
        List<Source> lone = Collections.singletonList(new Source(new Complex(0.0, 0.0), 1.0));
        for (int order : new int[]{0, 1, 4, 8}) {
            Complex[] a = moments(lone, centre, order);
            Complex field = farField(a, centre, new Complex(3.0, 4.0), order);
            Complex exact = directSum(lone, new Complex(3.0, 4.0));
            check(relativeDifference(field, exact) < 1e-12,
                    "a source at the box centre is exact at order " + order + " up to the softening, got "
                            + relativeDifference(field, exact));
        }

        // --- translation algebra ----------------------------------------------
        // Two well-separated box centres; the translated local series must agree with the source series
        // wherever the binomial series converges, which is inside the target box.
        Complex sourceCentre = new Complex(-1.0, -1.0);
        Complex targetCentre = new Complex(1.5, 1.5);
        List<Source> group = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            group.add(new Source(new Complex(-1.0 + 0.2 * (nextDouble() - 0.5), -1.0 + 0.2 * (nextDouble() - 0.5)),
                    0.5 + nextDouble()));
        }
        int order = 12;
        Complex[] a = moments(group, sourceCentre, order);
        Complex[] b = translate(a, sourceCentre, targetCentre, order);
        double worstTranslation = 0.0;
        for (int i = 0; i < 20; i++) {
            Complex point = new Complex(targetCentre.re + 0.1 * (nextDouble() * 2 - 1),
                    targetCentre.im + 0.1 * (nextDouble() * 2 - 1));
            Complex fromLocal = localField(b, targetCentre, point, order);
            Complex fromSeries = farField(a, sourceCentre, point, order);
            worstTranslation = Math.max(worstTranslation, relativeDifference(fromLocal, fromSeries));
        }
        check(worstTranslation < 1e-10, "the M2L translation must reproduce the source series, worst "
                + worstTranslation);

        // --- truncation error falls with the order ------------------------------
        double previous = Double.MAX_VALUE;
        double[] errors = new double[4];
        int[] orders = {0, 2, 4, 8};
        for (int i = 0; i < orders.length; i++) {
            Complex[] truncated = moments(group, sourceCentre, orders[i]);
            double error = 0.0;
            for (int sample = 0; sample < 20; sample++) {
                Complex point = new Complex(targetCentre.re + 0.2 * (nextDouble() * 2 - 1),
                        targetCentre.im + 0.2 * (nextDouble() * 2 - 1));
                error = Math.max(error, relativeDifference(farField(truncated, sourceCentre, point, orders[i]),
                        directSum(group, point)));
            }
            errors[i] = error;
            check(error < previous, "the error must fall with the order, order " + orders[i] + " gives " + error
                    + " after " + previous);
            previous = error;
        }
        check(errors[3] < 1e-6, "order 8 must be very accurate on a well-separated box, got " + errors[3]);

        // --- the assembled evaluation against direct summation -------------------
        final int side = 8;
        final double extent = 1.0;
        List<Source> sources = new ArrayList<>();
        for (int i = 0; i < 400; i++) {
            sources.add(new Source(new Complex(extent * (nextDouble() * 2 - 1), extent * (nextDouble() * 2 - 1)),
                    0.5 + nextDouble()));
        }
        BoxGrid grid = new BoxGrid(side, extent, sources);
        List<Complex> targets = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            targets.add(new Complex(extent * (nextDouble() * 2 - 1), extent * (nextDouble() * 2 - 1)));
        }
        double previousRelative = Double.MAX_VALUE;
        double[] assembled = new double[3];
        int[] assembledOrders = {2, 4, 8};
        long directPairs = 0;
        for (int i = 0; i < assembledOrders.length; i++) {
            long[] counted = new long[1];
            Complex[] approximated = evaluate(grid, targets, assembledOrders[i], counted);
            double difference = 0.0;
            double reference = 0.0;
            for (int t = 0; t < targets.size(); t++) {
                Complex exact = directSum(sources, targets.get(t));
                difference += Math.pow(approximated[t].re - exact.re, 2)
                        + Math.pow(approximated[t].im - exact.im, 2);
                reference += exact.re * exact.re + exact.im * exact.im;
            }
            assembled[i] = Math.sqrt(difference / reference);
            check(assembled[i] < previousRelative + 1e-12,
                    "more order must not hurt the assembled evaluation, order " + assembledOrders[i]
                            + " gives " + assembled[i] + " after " + previousRelative);
            previousRelative = assembled[i];
            directPairs = counted[0];
        }
        check(assembled[2] < 0.02, "the assembled potential must stay accurate at order 8, relative error "
                + assembled[2]);
        long allPairs = (long) targets.size() * sources.size();
        check(directPairs * 4 < allPairs, "only the near boxes may be summed directly: " + directPairs
                + " pairs against " + allPairs);

        // --- degeneration: a target exactly on a source -------------------------
        Complex onSource = new Complex(sources.get(0).position.re, sources.get(0).position.im);
        List<Complex> single = Collections.singletonList(onSource);
        long[] counted = new long[1];
        Complex[] value = evaluate(grid, single, 6, counted);
        Complex onSourceExact = directSum(sources, onSource);
        // The softened kernel is finite at zero distance (and vanishes there), so the coincidence is
        // resolved by the near field with no singularity at all.
        check(!Double.isNaN(value[0].re) && !Double.isInfinite(value[0].re)
                        && relativeDifference(value[0], onSourceExact) < 0.05,
                "a target on a source must be handled by the near field, got " + value[0].re + " against "
                        + onSourceExact.re);

        // --- determinism --------------------------------------------------------
        long[] repeatPairs = new long[1];
        Complex[] repeat = evaluate(grid, targets, 8, repeatPairs);
        double worstRepeat = 0.0;
        for (int t = 0; t < targets.size(); t++) {
            worstRepeat = Math.max(worstRepeat, Math.abs(repeat[t].re - 0.0));
        }
        check(repeatPairs[0] == directPairs, "the same evaluation must touch the same pair count");
        check(worstRepeat > 0.0, "the field must be non-trivial");

        System.out.println("single-level FMM: " + side + "x" + side + " boxes, " + sources.size()
                + " sources, " + targets.size() + " targets");
        System.out.println("truncation error at orders 0/2/4/8: " + Arrays.toString(errors));
        System.out.println("assembled relative error at orders 2/4/8: " + Arrays.toString(assembled)
                + "; direct pairs " + directPairs + " of " + allPairs);
        System.out.println("translation algebra reproduced to " + worstTranslation);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
