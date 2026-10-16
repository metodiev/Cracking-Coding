/** Task 771 - Implement epsilon-kernel for approximate convex hull (coresets). (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * An eps-kernel of a point set P is a small subset K such that the *width* of K in every direction
 * is at least (1 - eps) times the width of P: w_K(d) >= (1 - eps) * w_P(d) for all unit d. Every
 * linear criterion (support, separation, SVM margin, extreme point queries) survives the reduction,
 * which is what makes kernels the building block of coresets.
 *
 * Construction (the classical 2-D chain pruning): take the convex hull, split it at the two extreme
 * points in x into an upper and a lower chain, then refine a chain recursively - find the chain
 * vertex v farthest from the chord (a, b), and if that deviation h is at most eps/2 * w_P(n), where
 * n is the outward unit normal of the chord, replace the whole chain by the chord edge (a, b) and
 * document the prune; otherwise split the chain at v and recurse. A pruned chain loses at most
 * h * |<n, d>| width in direction d, so the factor 1/2 on the threshold keeps the loss inside the
 * eps budget for every direction; the check sweeps 721-2000 directions to confirm it. Every prune
 * event is journaled so the check can verify the exact criterion that was applied.
 *
 * Verified: the kernel is a convex sub-polygon of the hull whose vertices are input points, it keeps
 * the x-width exactly (both x-extremes survive by construction), the signed-width guarantee
 * w_K(d) >= (1 - eps) * w_P(d) holds over a dense sweep of directions for several epsilons, every
 * journaled prune satisfies h <= eps/2 * w_P(n) with h > 0, eps = 0 returns the full hull, eps >= 1
 * collapses to the two x-extremes, and the degenerate sets (single point, two points, collinear
 * points, duplicates) reproduce the exact widths.
 *
 * Simplified: the input is a few hundred integer points in the plane (no higher dimensions, no
 * merge-and-reduce streaming layer), so the exact hull and an exact dense-direction width sweep are
 * affordable and serve as the oracle for the approximation guarantee.
 */
import java.util.*;

public class Solution {
    private static long state = 0xD1B54A32D192ED03L;

    /** splitmix64: deterministic pseudo randomness, no Math.random anywhere. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static List<int[]> randomPoints(int count, int maxCoordinate) {
        List<int[]> points = new ArrayList<>();
        for (int i = 0; i < count; i++) points.add(new int[]{nextInt(maxCoordinate), nextInt(maxCoordinate)});
        return points;
    }

    /** Points on a strictly convex curve, so the hull is large and pruning has real work to do. */
    static List<int[]> circularPoints(int count, double radius) {
        List<int[]> points = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            double angle = 2 * Math.PI * i / count;
            points.add(new int[]{(int) Math.round(radius + radius * Math.cos(angle)),
                    (int) Math.round(radius + radius * Math.sin(angle))});
        }
        return points;
    }

    static long cross(int[] o, int[] a, int[] b) {
        return (long) (a[0] - o[0]) * (b[1] - o[1]) - (long) (a[1] - o[1]) * (b[0] - o[0]);
    }

    /** Monotone chain; collinear points are dropped so every hull vertex is a strict corner. */
    static List<int[]> convexHull(List<int[]> points) {
        List<int[]> sorted = new ArrayList<>(points);
        sorted.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
        List<int[]> unique = new ArrayList<>();
        for (int[] p : sorted) {
            if (unique.isEmpty() || unique.get(unique.size() - 1)[0] != p[0]
                    || unique.get(unique.size() - 1)[1] != p[1]) unique.add(p);
        }
        if (unique.size() <= 2) return unique;
        List<int[]> hull = new ArrayList<>();
        for (int pass = 0; pass < 2; pass++) {
            int start = hull.size();
            for (int i = 0; i < unique.size(); i++) {
                int[] p = unique.get(pass == 0 ? i : unique.size() - 1 - i);
                while (hull.size() >= start + 2
                        && cross(hull.get(hull.size() - 2), hull.get(hull.size() - 1), p) <= 0) {
                    hull.remove(hull.size() - 1);
                }
                hull.add(p);
            }
            hull.remove(hull.size() - 1);          // the closing point repeats the next pass's start
        }
        return hull;                                // counter-clockwise, no collinear vertices
    }

    /** Width of the point set in the unit direction (dx, dy): max - min of the linear score. */
    static double width(List<int[]> points, double dx, double dy) {
        double max = -Double.MAX_VALUE;
        double min = Double.MAX_VALUE;
        for (int[] p : points) {
            double score = p[0] * dx + p[1] * dy;
            max = Math.max(max, score);
            min = Math.min(min, score);
        }
        return max - min;
    }

    static class EpsilonKernel {
        final double eps;
        final List<int[]> hull;
        final List<int[]> vertices = new ArrayList<>();
        final List<double[]> prunes = new ArrayList<>();   // {deviation h, width w_P(n)}

        EpsilonKernel(List<int[]> points, double eps) {
            this.eps = eps;
            this.hull = convexHull(points);
            if (hull.size() <= 2) {
                vertices.addAll(hull);
                return;
            }
            int li = 0, ri = 0;
            for (int i = 0; i < hull.size(); i++) {
                int[] p = hull.get(i);
                if (p[0] < hull.get(li)[0] || (p[0] == hull.get(li)[0] && p[1] < hull.get(li)[1])) li = i;
                if (p[0] > hull.get(ri)[0] || (p[0] == hull.get(ri)[0] && p[1] < hull.get(ri)[1])) ri = i;
            }
            refine(hull.get(li), hull.get(ri), chain(li, ri));
            refine(hull.get(ri), hull.get(li), chain(ri, li));
        }

        /** Hull vertices strictly between index from and index to, walking counter-clockwise. */
        private List<int[]> chain(int from, int to) {
            List<int[]> chain = new ArrayList<>();
            for (int i = (from + 1) % hull.size(); i != to; i = (i + 1) % hull.size()) {
                chain.add(hull.get(i));
            }
            return chain;
        }

        /**
         * Adds the kernel chain from a to b: either the chord edge alone (possibly pruned away) or
         * the recursively refined sub-chains. Only the first endpoint is appended, so the caller's
         * continuation supplies b and the vertex list comes out in counter-clockwise order.
         */
        private void refine(int[] a, int[] b, List<int[]> chain) {
            if (chain.isEmpty()) {
                vertices.add(a);
                return;
            }
            double chordX = b[0] - a[0];
            double chordY = b[1] - a[1];
            double length = Math.hypot(chordX, chordY);
            double normalX = -chordY / length;
            double normalY = chordX / length;
            double maxPositive = -Double.MAX_VALUE;
            double maxNegative = -Double.MAX_VALUE;              // -min, i.e. the depth on the other side
            int argPositive = -1, argNegative = -1;
            for (int i = 0; i < chain.size(); i++) {
                int[] p = chain.get(i);
                double signed = (p[0] - a[0]) * normalX + (p[1] - a[1]) * normalY;
                if (signed > maxPositive) { maxPositive = signed; argPositive = i; }
                if (-signed > maxNegative) { maxNegative = -signed; argNegative = i; }
            }
            // the chain is on one side of the chord; pick the side with the larger deviation
            double deviation = Math.max(maxPositive, maxNegative);
            int arg = maxPositive >= maxNegative ? argPositive : argNegative;
            double outwardX = maxPositive >= maxNegative ? normalX : -normalX;
            double outwardY = maxPositive >= maxNegative ? normalY : -normalY;
            double widthP = width(hull, outwardX, outwardY);
            // A pruned chain loses at most h * |<n, d>| width in direction d; the factor 1/2 on the
            // threshold keeps that loss inside the eps budget for every direction (see the sweep).
            double threshold = 0.5 * eps * widthP;
            if (deviation <= threshold) {
                prunes.add(new double[]{deviation, widthP});
                vertices.add(a);
                return;
            }
            int[] v = chain.get(arg);
            refine(a, v, new ArrayList<>(chain.subList(0, arg)));
            refine(v, b, new ArrayList<>(chain.subList(arg + 1, chain.size())));
        }

        int size() { return vertices.size(); }
    }

    static String key(int[] p) { return p[0] + "," + p[1]; }

    public static void main(String[] args) {
        // --- degenerate point sets --------------------------------------------
        List<int[]> one = new ArrayList<>(List.of(new int[]{5, 7}));
        EpsilonKernel single = new EpsilonKernel(one, 0.1);
        check(single.size() == 1, "single point kernel has one vertex");
        check(width(single.vertices, 1, 0) == 0 && width(single.vertices, 0, 1) == 0,
                "single point has zero width in every direction");

        List<int[]> two = new ArrayList<>(List.of(new int[]{0, 0}, new int[]{6, 8}));
        EpsilonKernel pair = new EpsilonKernel(two, 0.1);
        check(pair.size() == 2, "two points keep both extremes");
        check(Math.abs(width(pair.vertices, 0.6, 0.8) - 10.0) < 1e-9, "segment kernel keeps the exact width");

        List<int[]> collinear = new ArrayList<>();
        for (int i = 0; i <= 6; i++) collinear.add(new int[]{3 * i, 4 * i});
        EpsilonKernel line = new EpsilonKernel(collinear, 0.25);
        check(line.size() == 2, "collinear points collapse to the two extremes");
        for (int step = 0; step < 16; step++) {
            double angle = Math.PI * step / 16;
            double dx = Math.cos(angle), dy = Math.sin(angle);
            check(Math.abs(width(line.vertices, dx, dy) - width(collinear, dx, dy)) < 1e-9,
                    "collinear set reproduces the exact width, step " + step);
        }

        List<int[]> duplicates = new ArrayList<>();
        for (int i = 0; i < 5; i++) duplicates.add(new int[]{2, 2});
        EpsilonKernel same = new EpsilonKernel(duplicates, 0.3);
        check(same.size() == 1, "one distinct point stays a single vertex");
        check(same.prunes.isEmpty(), "a single distinct point needs no pruning");

        // --- eps = 0 keeps the whole hull exactly ------------------------------
        List<int[]> cloud = randomPoints(120, 60);
        List<int[]> hull = convexHull(cloud);
        check(hull.size() >= 6, "the random cloud must have a non-trivial hull, got " + hull.size());
        EpsilonKernel exact = new EpsilonKernel(cloud, 0.0);
        check(exact.size() == hull.size(), "eps = 0 keeps every hull vertex, got " + exact.size());
        check(exact.prunes.isEmpty(), "eps = 0 never prunes a non-degenerate chain");

        // --- generous eps collapses the kernel to the x-extremes ---------------
        EpsilonKernel coarse = new EpsilonKernel(cloud, 2.0);
        check(coarse.size() == 2, "eps >= 1 collapses to the two x-extremes, got " + coarse.size());
        check(coarse.prunes.size() == 2, "both chains are pruned at the root, got " + coarse.prunes.size());
        check(Math.abs(width(coarse.vertices, 1, 0) - width(cloud, 1, 0)) < 1e-9,
                "the coarse kernel still keeps the x-width exactly");

        // --- the documented (1 - eps) width guarantee + hull containment -------
        double[] epsilons = {0.05, 0.1, 0.2};
        int guardChecks = 0;
        for (int trial = 0; trial < 3; trial++) {
            List<int[]> points = randomPoints(70 + 40 * trial, 50);
            for (double eps : epsilons) {
                EpsilonKernel kernel = new EpsilonKernel(points, eps);
                check(kernel.size() >= 2, "a non-degenerate kernel has at least two vertices");
                check(kernel.size() <= convexHull(points).size(), "a kernel cannot exceed the hull");
                for (int step = 0; step < 721; step++) {
                    double angle = Math.PI * step / 721;
                    double dx = Math.cos(angle), dy = Math.sin(angle);
                    double widthK = width(kernel.vertices, dx, dy);
                    double widthP = width(points, dx, dy);
                    check(widthK <= widthP + 1e-6,
                            "the kernel cannot be wider than P, trial " + trial + " eps=" + eps + " step=" + step);
                    check(widthK >= (1 - eps) * widthP - 1e-6 * (1 + widthP),
                            "kernel guarantee w_K >= (1-eps) w_P: trial " + trial + " eps=" + eps
                                    + " step=" + step + " got " + widthK + " vs " + widthP);
                    guardChecks++;
                }
                // every prune really applied the documented criterion
                for (double[] prune : kernel.prunes) {
                    check(prune[0] > 0, "a pruned chain must have a strictly positive deviation");
                    check(prune[0] <= 0.5 * eps * prune[1] + 1e-9,
                            "prune criterion h <= eps/2 * w_P(n), got h=" + prune[0]
                                    + " eps*w/2=" + 0.5 * eps * prune[1]);
                }
                // the x-width is always exact because both x-extremes are kernel vertices
                check(Math.abs(width(kernel.vertices, 1, 0) - width(points, 1, 0)) < 1e-9,
                        "the x-extremes survive, so the x-width is exact");
            }
        }
        check(guardChecks == 3 * 3 * 721, "the full direction sweep ran, got " + guardChecks);

        // --- kernel geometry: convex, CCW, vertices from the input -------------
        List<int[]> circle = circularPoints(96, 50);
        EpsilonKernel refined = new EpsilonKernel(circle, 0.1);
        Set<String> inputKeys = new HashSet<>();
        for (int[] p : circle) inputKeys.add(key(p));
        for (int[] p : refined.vertices) {
            check(inputKeys.contains(key(p)), "every kernel vertex is an input point: " + key(p));
        }
        boolean leftTurns = true;
        for (int i = 0; i < refined.size(); i++) {
            int[] a = refined.vertices.get(i);
            int[] b = refined.vertices.get((i + 1) % refined.size());
            int[] c = refined.vertices.get((i + 2) % refined.size());
            if (cross(a, b, c) <= 0) leftTurns = false;          // counter-clockwise and convex
        }
        check(leftTurns, "the kernel polygon is convex and counter-clockwise");
        check(refined.size() < circle.size(),
                "a 96-gon must be compressed, got " + refined.size() + " of " + circle.size());
        check(refined.size() >= 3, "the compressed kernel keeps at least a triangle");

        // --- coreset quality: the pruned kernel is tiny but still wide ----------
        EpsilonKernel tight = new EpsilonKernel(circle, 0.01);
        EpsilonKernel loose = new EpsilonKernel(circle, 0.5);
        check(loose.size() <= tight.size(), "larger eps never gives a larger kernel: "
                + loose.size() + " vs " + tight.size());
        check(loose.size() < tight.size(), "a 50x looser eps must actually compress more");
        double worstRatio = 1.0;
        for (int step = 0; step < 2000; step++) {
            double angle = 2 * Math.PI * step / 2000;
            double dx = Math.cos(angle), dy = Math.sin(angle);
            double widthP = width(circle, dx, dy);
            if (widthP > 1e-9) {
                worstRatio = Math.min(worstRatio, width(loose.vertices, dx, dy) / widthP);
            }
        }
        check(worstRatio >= 0.5 - 1e-9, "eps = 0.5 kernel is never narrower than half of P, ratio " + worstRatio);

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
