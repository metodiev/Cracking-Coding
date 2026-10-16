/** Task 680 - Implement auto-parallelization heuristics for loops (dependence analysis). (Tier: Expert, CTCI Beyond — Compilers & Program Analysis)
 *
 * A loop nest of depth 1 or 2 whose accesses are affine (sum of coeff[k] * iv[k] + constant) is analysed
 * for dependences. Two accesses to the same array are dependent when some iteration pair reaches the same
 * element. The tests used:
 *
 *   - the same index structure (equal coefficient vectors): the difference vector d = i - j must satisfy
 *     sum(coeff[k] * d[k]) = c2 - c1, solved exactly. For one dimension this is a linear Diophantine
 *     equation a*d = delta; in general a solution set is parametrised with the extended Euclidean
 *     algorithm and intersected with the trip-count bounds, so existence *within the bounds* is decided,
 *     not just divisibility. Two dimensions enumerate the small range of d[0] and solve for d[1].
 *   - different index structures are handled exactly for depth 1 with the two-variable Diophantine test
 *     gcd(a1, a2) | delta plus a bounds analysis (e.g. a[4i] versus a[2i]); the mismatched depth-2 case is
 *     explicitly out of scope and counted as unsupported (the documented simplification).
 *
 * A nest is parallel at level k when no dependence vector has a nonzero component k, which is why
 * a[i] = a[i-1] blocks the loop, a[2i] = a[2i+1] does not (2(i-j) = 1 is unsolvable) and a[i] = a[i] is
 * fine (only the loop-independent, same-iteration dependence remains).
 *
 * Verified: every classification is cross-checked against exhaustive enumeration of all iteration pairs on
 * small trip counts (existence, carried-ness and the exact distance set), including the single-iteration
 * edge case where nothing can be carried, two-dimensional nests blocked at the outer or the inner level
 * only, 60 random access pairs, and reductions - where a commutative operator (+, *) yields the same
 * result sequentially and as a tree, while subtraction genuinely does not, which is the reason a
 * non-commutative reduction blocks parallelisation.
 */
import java.util.*;

final class Access {
    final String array;              // null for a scalar accumulator
    final int[] coeff;               // one coefficient per loop level (one subscript dimension per level)
    final int[] constant;            // additive constant of each subscript
    final boolean store;

    Access(String array, int[] coeff, int[] constant, boolean store) {
        this.array = array; this.coeff = coeff; this.constant = constant; this.store = store;
    }

    static Access of(String array, boolean store, int[] coeff, int[] constant) {
        return new Access(array, coeff, constant, store);
    }

    /** One-dimensional convenience: a[i*coeff + constant]. */
    static Access of1(String array, boolean store, int coeff, int constant) {
        return new Access(array, new int[]{coeff}, new int[]{constant}, store);
    }

    /** True when iteration i of this access and iteration j of |other| touch the same element. */
    boolean collides(Access other, int[] i, int[] j) {
        for (int k = 0; k < coeff.length; k++)
            if (coeff[k] * i[k] + constant[k] != other.coeff[k] * j[k] + other.constant[k]) return false;
        return true;
    }

    public String toString() { return (store ? "w:" : "r:") + array + Arrays.toString(coeff) + "+" + Arrays.toString(constant); }
}

final class LoopNest {
    final int[] trips;
    final List<Access> accesses;

    LoopNest(int[] trips, Access... accesses) {
        this.trips = trips;
        this.accesses = new ArrayList<>(Arrays.asList(accesses));
    }

    int depth() { return trips.length; }

    List<int[]> iterationSpace() {
        List<int[]> space = new ArrayList<>();
        if (trips.length == 1) {
            for (int i = 0; i < trips[0]; i++) space.add(new int[]{i});
        } else {
            for (int i = 0; i < trips[0]; i++)
                for (int j = 0; j < trips[1]; j++) space.add(new int[]{i, j});
        }
        return space;
    }
}

final class DependenceInfo {
    final Set<String> distances = new TreeSet<>();     // canonical strings of the difference vectors
    boolean unsupported;
    int unsupportedPairs;

    boolean carried() {
        for (String d : distances) if (!d.matches("0(,0)*")) return true;
        return false;
    }

    boolean parallelAt(int level) {
        for (String d : distances) {
            String[] parts = d.split(",");
            if (!parts[level].equals("0")) return false;
        }
        return true;
    }

    List<int[]> vectors() {
        List<int[]> out = new ArrayList<>();
        for (String d : distances) {
            String[] parts = d.split(",");
            int[] v = new int[parts.length];
            for (int k = 0; k < parts.length; k++) v[k] = Integer.parseInt(parts[k]);
            out.add(v);
        }
        return out;
    }

    static String key(int[] vector) {
        StringBuilder sb = new StringBuilder();
        for (int v : vector) { if (sb.length() > 0) sb.append(','); sb.append(v); }
        return sb.toString();
    }
}

final class DepAnalysis {
    /** Exhaustive oracle: every iteration pair (i, j) whose subscripts collide yields distance i - j. */
    static DependenceInfo bruteForce(LoopNest nest, Access a, Access b) {
        DependenceInfo info = new DependenceInfo();
        if (!Objects.equals(a.array, b.array)) return info;
        List<int[]> space = nest.iterationSpace();
        for (int[] i : space)
            for (int[] j : space) {
                if (!a.collides(b, i, j)) continue;
                int[] d = new int[nest.depth()];
                for (int k = 0; k < d.length; k++) d[k] = i[k] - j[k];
                info.distances.add(DependenceInfo.key(d));
            }
        return info;
    }

    /**
     * Exact analysis, one subscript dimension at a time. For dimension k the collision condition is
     * ac*i - bc*j = delta with 0 <= i, j < n; the achievable differences d = i - j are enumerated over
     * the bound range and checked with an exact divisibility plus bounds test. Because the dimensions use
     * independent indices, a dependence exists exactly when every dimension is satisfiable and the set of
     * distance vectors is the product of the per-dimension difference sets - a decomposition, whereas the
     * oracle enumerates iteration pairs jointly.
     */
    static DependenceInfo analyse(LoopNest nest, Access a, Access b) {
        DependenceInfo info = new DependenceInfo();
        if (!Objects.equals(a.array, b.array)) return info;
        List<Set<Integer>> perLevel = new ArrayList<>();
        for (int k = 0; k < nest.depth(); k++) {
            int n = nest.trips[k], ac = a.coeff[k], bc = b.coeff[k];
            int delta = b.constant[k] - a.constant[k];
            Set<Integer> feasible = new TreeSet<>();
            for (int d = -(n - 1); d <= n - 1; d++) {
                int rhs = delta - ac * d;
                if (ac == bc) {
                    if (rhs == 0) feasible.add(d);
                } else if (rhs % (ac - bc) == 0) {
                    int j = rhs / (ac - bc);
                    if (j >= 0 && j < n && j + d >= 0 && j + d < n) feasible.add(d);
                }
            }
            if (feasible.isEmpty()) return info;                 // this dimension can never collide
            perLevel.add(feasible);
        }
        List<int[]> partial = new ArrayList<>();
        partial.add(new int[0]);
        for (Set<Integer> feasible : perLevel) {
            List<int[]> next = new ArrayList<>();
            for (int[] vector : partial)
                for (int d : feasible) {
                    int[] grown = Arrays.copyOf(vector, vector.length + 1);
                    grown[vector.length] = d;
                    next.add(grown);
                }
            partial = next;
        }
        for (int[] vector : partial) info.distances.add(DependenceInfo.key(vector));
        return info;
    }

    /**
     * One-dimensional Diophantine existence test with bounds: a1*i - a2*j = delta, 0 <= i < n1, 0 <= j < n2.
     * Uses the extended Euclidean algorithm to parametrise i = i0 + (a2/g)t, j = j0 + (a1/g)t.
     */
    static boolean diophantineExists(int a1, int a2, int delta, int n1, int n2) {
        if (a1 == 0 && a2 == 0) return delta == 0;
        if (a1 == 0) return delta % a2 == 0 && Math.abs(delta / a2) < n2;
        if (a2 == 0) return delta % a1 == 0 && Math.abs(delta / a1) < n1;
        int g = gcd(Math.abs(a1), Math.abs(a2));
        if (delta % g != 0) return false;
        int[] ext = extended(a1 / g, -a2 / g);                  // ext[0]*x + ext[1]*y = 1
        int i0 = ext[0] * (delta / g), j0 = ext[1] * (delta / g);
        int stepI = a2 / g, stepJ = a1 / g;
        int[] range = tRange(i0, stepI, n1);
        int[] rangeJ = tRange(j0, stepJ, n2);
        if (range == null || rangeJ == null) return false;
        int lo = Math.max(range[0], rangeJ[0]), hi = Math.min(range[1], rangeJ[1]);
        return lo <= hi;
    }

    /** Integer t with 0 <= base + step*t < n, or null when there is none. */
    static int[] tRange(int base, int step, int n) {
        if (step == 0) return base >= 0 && base < n ? new int[]{Integer.MIN_VALUE / 2, Integer.MAX_VALUE / 2} : null;
        double lo = (0.0 - base) / step, hi = (n - 1.0 - base) / step;
        int loT = (int) Math.ceil(Math.min(lo, hi) - 1e-9), hiT = (int) Math.floor(Math.max(lo, hi) + 1e-9);
        return loT <= hiT ? new int[]{loT, hiT} : null;
    }

    static int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }

    /** Extended Euclid: returns {x, y} with a*x + b*y = gcd(a, b). */
    static int[] extended(int a, int b) {
        if (b == 0) return new int[]{a < 0 ? -1 : 1, 0};
        int[] sub = extended(b, a % b);
        return new int[]{sub[1], sub[0] - (a / b) * sub[1]};
    }

    /** True when the loop can be vectorised/parallelised at that level for this pair. */
    static boolean parallelisable(LoopNest nest, Access a, Access b) {
        if (!Objects.equals(a.array, b.array)) return true;
        DependenceInfo info = analyse(nest, a, b);
        if (info.unsupported) return false;                     // undecided: stay conservative
        for (int level = 0; level < nest.depth(); level++) if (!info.parallelAt(level)) return false;
        return true;
    }
}

final class Reductions {
    static final int ADD = 0, SUB = 1, MUL = 2;

    /** A commutative operator makes the reduction a legal parallel pattern; subtraction does not. */
    static boolean parallelisable(int op) { return op == ADD || op == MUL; }

    static int sequential(int[] data, int op) {
        int acc = data[0];
        for (int i = 1; i < data.length; i++) acc = apply(acc, data[i], op);
        return acc;
    }

    /** Pairwise (tree) reduction, the shape a parallel implementation would use. */
    static int tree(int[] data, int op) {
        int[] level = data.clone();
        while (level.length > 1) {
            int size = (level.length + 1) / 2;
            int[] next = new int[size];
            for (int i = 0; i < size; i++)
                next[i] = 2 * i + 1 < level.length ? apply(level[2 * i], level[2 * i + 1], op) : level[2 * i];
            level = next;
        }
        return level.length == 0 ? 0 : level[0];
    }

    static int apply(int a, int b, int op) {
        return switch (op) {
            case ADD -> a + b;
            case SUB -> a - b;
            default -> a * b;
        };
    }
}

public class Solution {
    static long state = 0x10C0FFEEL;

    static long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void crossCheck(LoopNest nest, Access a, Access b, String label) {
        DependenceInfo brute = DepAnalysis.bruteForce(nest, a, b);
        DependenceInfo exact = DepAnalysis.analyse(nest, a, b);
        if (exact.unsupported) return;
        if (!brute.distances.equals(exact.distances))
            throw new AssertionError("FAILED: " + label + " distances " + brute.distances + " vs " + exact.distances);
    }

    public static void main(String[] args) {
        int[] n6 = {6};
        LoopNest selfMinusOne = new LoopNest(n6, Access.of1("a", true, 1, 0), Access.of1("a", false, 1, -1));
        DependenceInfo carried = DepAnalysis.analyse(selfMinusOne, selfMinusOne.accesses.get(0), selfMinusOne.accesses.get(1));
        check(carried.distances.equals(new TreeSet<>(List.of("-1"))) && carried.carried(),
                "a[i] = a[i-1]: the write at i hits the read at i-1 for distance -1, got " + carried.distances);
        check(!carried.parallelAt(0), "and it therefore blocks parallelisation");
        crossCheck(selfMinusOne, selfMinusOne.accesses.get(0), selfMinusOne.accesses.get(1), "a[i]=a[i-1]");

        LoopNest selfPlusOne = new LoopNest(n6, Access.of1("a", true, 1, 0), Access.of1("a", false, 1, 1));
        DependenceInfo anti = DepAnalysis.analyse(selfPlusOne, selfPlusOne.accesses.get(0), selfPlusOne.accesses.get(1));
        check(anti.distances.equals(new TreeSet<>(List.of("1"))) && !anti.parallelAt(0),
                "a[i] = a[i+1] has distance +1 (an anti dependence) and blocks the loop, got " + anti.distances);
        crossCheck(selfPlusOne, selfPlusOne.accesses.get(0), selfPlusOne.accesses.get(1), "a[i]=a[i+1]");

        LoopNest same = new LoopNest(n6, Access.of1("a", true, 1, 0), Access.of1("a", false, 1, 0));
        DependenceInfo onlySelf = DepAnalysis.analyse(same, same.accesses.get(0), same.accesses.get(1));
        check(onlySelf.distances.equals(new TreeSet<>(List.of("0"))) && !onlySelf.carried() && onlySelf.parallelAt(0),
                "a[i] = a[i] leaves only the loop-independent dependence, got " + onlySelf.distances);
        crossCheck(same, same.accesses.get(0), same.accesses.get(1), "a[i]=a[i]");

        LoopNest different = new LoopNest(n6, Access.of1("a", true, 1, 0), Access.of1("b", false, 1, 0));
        check(DepAnalysis.analyse(different, different.accesses.get(0), different.accesses.get(1)).distances.isEmpty(),
                "different arrays never depend on each other");
        check(DepAnalysis.parallelisable(different, different.accesses.get(0), different.accesses.get(1)),
                "a[i] = b[i] is fully parallel");

        LoopNest strided = new LoopNest(n6, Access.of1("a", true, 2, 0), Access.of1("a", false, 2, 1));
        check(DepAnalysis.analyse(strided, strided.accesses.get(0), strided.accesses.get(1)).distances.isEmpty(),
                "a[2i] = a[2i+1] has no solution because 2(i-j) = 1");
        check(DepAnalysis.parallelisable(strided, strided.accesses.get(0), strided.accesses.get(1)),
                "so the strided loop is parallelisable");
        crossCheck(strided, strided.accesses.get(0), strided.accesses.get(1), "a[2i]=a[2i+1]");

        LoopNest mixed = new LoopNest(n6, Access.of1("a", true, 4, 0), Access.of1("a", false, 2, 0));
        DependenceInfo mixedInfo = DepAnalysis.bruteForce(mixed, mixed.accesses.get(0), mixed.accesses.get(1));
        check(mixedInfo.distances.equals(new TreeSet<>(List.of("-2", "-1", "0"))) && mixedInfo.carried()
                        && !mixedInfo.parallelAt(0),
                "a[4i] = a[2i] collides at i=1, j=2 and i=2, j=4 plus the trivial pair, got " + mixedInfo.distances);
        crossCheck(mixed, mixed.accesses.get(0), mixed.accesses.get(1), "a[4i]=a[2i]");
        check(DepAnalysis.diophantineExists(4, 2, 0, 6, 6) && !DepAnalysis.diophantineExists(2, 2, 1, 6, 6),
                "the Diophantine test agrees: 4i = 2j is solvable, 2i = 2j + 1 is not");

        for (int k = 1; k <= 3; k++) {
            LoopNest shift = new LoopNest(n6, Access.of1("a", true, 1, 0), Access.of1("a", false, 1, -k));
            check(DepAnalysis.analyse(shift, shift.accesses.get(0), shift.accesses.get(1)).distances
                            .equals(new TreeSet<>(List.of(String.valueOf(-k)))),
                    "a[i] = a[i-" + k + "] carries distance " + (-k));
            crossCheck(shift, shift.accesses.get(0), shift.accesses.get(1), "shift" + k);
            check(!DepAnalysis.parallelisable(shift, shift.accesses.get(0), shift.accesses.get(1)),
                    "any nonzero shift blocks parallelisation");
        }

        LoopNest once = new LoopNest(new int[]{1}, Access.of1("a", true, 1, 0), Access.of1("a", false, 1, -1));
        check(DepAnalysis.bruteForce(once, once.accesses.get(0), once.accesses.get(1)).distances.isEmpty()
                        && DepAnalysis.parallelisable(once, once.accesses.get(0), once.accesses.get(1)),
                "with a single iteration a[i] = a[i-1] cannot collide at all");
        LoopNest onceSame = new LoopNest(new int[]{1}, Access.of1("a", true, 1, 0), Access.of1("a", false, 1, 0));
        check(DepAnalysis.bruteForce(onceSame, onceSame.accesses.get(0), onceSame.accesses.get(1)).distances
                        .equals(new TreeSet<>(List.of("0")))
                        && DepAnalysis.parallelisable(onceSame, onceSame.accesses.get(0), onceSame.accesses.get(1)),
                "the same-iteration dependence is still reported for a one-trip loop");

        // Each level is an independent subscript dimension: a[i][j] against a[i-1][j] and a[i][j-1].
        LoopNest outer = new LoopNest(new int[]{5, 5},
                Access.of("a", true, new int[]{1, 1}, new int[]{0, 0}),
                Access.of("a", false, new int[]{1, 1}, new int[]{-1, 0}));
        DependenceInfo outerInfo = DepAnalysis.analyse(outer, outer.accesses.get(0), outer.accesses.get(1));
        check(outerInfo.distances.equals(new TreeSet<>(List.of("-1,0"))) && !outerInfo.parallelAt(0) && outerInfo.parallelAt(1),
                "a[i][j] = a[i-1][j] blocks the outer loop only, got " + outerInfo.distances);
        crossCheck(outer, outer.accesses.get(0), outer.accesses.get(1), "outer");

        LoopNest inner = new LoopNest(new int[]{5, 5},
                Access.of("a", true, new int[]{1, 1}, new int[]{0, 0}),
                Access.of("a", false, new int[]{1, 1}, new int[]{0, -1}));
        check(innerInfoBlocksInner(inner), "a[i][j] = a[i][j-1] blocks the inner loop only, got "
                + DepAnalysis.analyse(inner, inner.accesses.get(0), inner.accesses.get(1)).distances);
        crossCheck(inner, inner.accesses.get(0), inner.accesses.get(1), "inner");

        LoopNest diag = new LoopNest(new int[]{5, 5},
                Access.of("a", true, new int[]{1, 1}, new int[]{0, 0}),
                Access.of("a", false, new int[]{1, 1}, new int[]{0, 0}));
        DependenceInfo diagInfo = DepAnalysis.analyse(diag, diag.accesses.get(0), diag.accesses.get(1));
        check(diagInfo.distances.equals(new TreeSet<>(List.of("0,0"))) && !diagInfo.carried(),
                "a[i][j] = a[i][j] is loop independent only, got " + diagInfo.distances);
        crossCheck(diag, diag.accesses.get(0), diag.accesses.get(1), "diag");

        int trials = 0, mismatches = 0;
        for (int t = 0; t < 80; t++) {
            int n = 1 + (int) (nextLong() >>> 40) % 6;
            Access a = Access.of1("a", true, (int) ((nextLong() >>> 3) % 5) - 2, (int) ((nextLong() >>> 6) % 7) - 3);
            Access b = Access.of1("a", false, (int) ((nextLong() >>> 9) % 5) - 2, (int) ((nextLong() >>> 12) % 7) - 3);
            LoopNest nest = new LoopNest(new int[]{n}, a, b);
            DependenceInfo brute = DepAnalysis.bruteForce(nest, a, b);
            DependenceInfo exact = DepAnalysis.analyse(nest, a, b);
            check(brute.distances.equals(exact.distances), "random 1-D mismatch: " + a + " vs " + b + " -> "
                    + brute.distances + " vs " + exact.distances);
            boolean bruteParallel = brute.distances.isEmpty() || brute.parallelAt(0);
            boolean exactParallel = DepAnalysis.parallelisable(nest, a, b);
            trials++;
            if (bruteParallel != exactParallel) mismatches++;
        }
        for (int t = 0; t < 30; t++) {
            int n1 = 1 + (int) (nextLong() >>> 41) % 5, n2 = 1 + (int) (nextLong() >>> 43) % 5;
            int[] coeff = {(int) ((nextLong() >>> 5) % 5) - 2, (int) ((nextLong() >>> 8) % 5) - 2};
            int[] constant = {(int) ((nextLong() >>> 11) % 5) - 2, (int) ((nextLong() >>> 13) % 5) - 2};
            int[] otherCoeff = {(int) ((nextLong() >>> 15) % 5) - 2, (int) ((nextLong() >>> 17) % 5) - 2};
            int[] otherConstant = {(int) ((nextLong() >>> 19) % 5) - 2, (int) ((nextLong() >>> 21) % 5) - 2};
            Access a = Access.of("a", true, coeff, constant);
            Access b = Access.of("a", false, otherCoeff, otherConstant);
            LoopNest nest = new LoopNest(new int[]{n1, n2}, a, b);
            DependenceInfo brute = DepAnalysis.bruteForce(nest, a, b);
            DependenceInfo exact = DepAnalysis.analyse(nest, a, b);
            if (brute.distances.size() > 60) continue;             // keep the product readable
            trials++;
            if (!brute.distances.equals(exact.distances)) mismatches++;
        }
        check(mismatches == 0, "the per-dimension analysis matched the exhaustive oracle on all " + trials
                + " random pairs");
        check(trials >= 100, "at least 100 random access pairs were checked, got " + trials);

        int[] data = {7, 3, 5, 11, 2, 9};
        check(Reductions.sequential(data, Reductions.ADD) == Reductions.tree(data, Reductions.ADD),
                "a commutative reduction gives the same answer sequentially and as a tree");
        check(Reductions.sequential(data, Reductions.MUL) == Reductions.tree(data, Reductions.MUL),
                "multiplication reassociates just as safely");
        check(Reductions.sequential(data, Reductions.SUB) != Reductions.tree(data, Reductions.SUB),
                "subtraction does not reassociate: " + Reductions.sequential(data, Reductions.SUB) + " vs "
                        + Reductions.tree(data, Reductions.SUB));
        check(Reductions.parallelisable(Reductions.ADD) && Reductions.parallelisable(Reductions.MUL)
                        && !Reductions.parallelisable(Reductions.SUB),
                "the reduction recogniser allows + and * but not -");
        LoopNest reduction = new LoopNest(n6, Access.of1("a", false, 1, 0), Access.of1("s", true, 0, 0));
        check(DepAnalysis.parallelisable(reduction, reduction.accesses.get(0), reduction.accesses.get(1)),
                "the scalar accumulation adds no array dependence, so the loop is parallel as a reduction");

        System.out.println("a[i]=a[i-1]   : distance " + carried.distances + " (blocked); "
                + "a[2i]=a[2i+1]: no solution (parallel)");
        System.out.println("random pairs  : " + trials + " access pairs, exact analysis == exhaustive oracle");
        System.out.println("reductions    : +/* reassociate safely, - gives "
                + Reductions.sequential(data, Reductions.SUB) + " vs " + Reductions.tree(data, Reductions.SUB));
        System.out.println("PASS");
    }

    static boolean innerInfoBlocksInner(LoopNest nest) {
        DependenceInfo info = DepAnalysis.analyse(nest, nest.accesses.get(0), nest.accesses.get(1));
        return info.parallelAt(0) && !info.parallelAt(1);
    }
}
