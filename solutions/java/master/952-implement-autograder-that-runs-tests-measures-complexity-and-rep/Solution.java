/** Task 952 - Implement autograder that runs tests, measures complexity, and reports regression.
 *  (Tier: Master, CTCI Ch 11 - Testing)
 *
 *  In-memory autograder: a submission is a lambda from int[] to long that reports how many elementary
 *  operations it performed, so "runtime" is a deterministic counter, not wall clock. The grader runs
 *  hidden cases stopping at the first mismatch, estimates the growth class by least-squares fitting the
 *  counts of doubling inputs against {1, log n, n, n log n, n^2}, and flags a regression when the largest
 *  count exceeds the baseline by 15%. Simplified: submissions are in-process lambdas and the fit assumes
 *  smooth work, so a constant factor alone can never change the class.
 *  Verified: correct submissions pass (n=0 and n=1 included); the wrong one reports its first failing
 *  case; fitted classes match the intended shapes; counts reproduce; the padded revision is flagged
 *  while an identical rerun is not; Kadane agrees with a brute-force oracle on 20 arrays.
 */
import java.util.*;

public class Solution {
    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }
    static long splitmix64(long x) {
        long z = x + 0x9E3779B97F4A7C15L; z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL; return z ^ (z >>> 31);
    }
    static int[] randomArray(int size, long seed, int bound) {
        int[] a = new int[size]; long s = seed;
        for (int i = 0; i < size; i++) { s = splitmix64(s + i); a[i] = (int) Math.floorMod(s, 2L * bound + 1) - bound; }
        return a;
    }
    static int[] maxSubarrayInput(int size, long seed) { return randomArray(size, seed, 40); }
    static int[] lookupInput(int size, long seed) { int[] a = randomArray(size, seed, 200); Arrays.sort(a); return a; }
    interface Solve { long solve(int[] a, long[] ops); }
    interface Oracle { long expected(int[] a); }
    interface Gen { int[] make(int size, long seed); }
    static long bruteMaxSubarray(int[] a) {
        long best = 0;                                  // empty subarray allowed, like Kadane's reset
        for (int i = 0; i < a.length; i++) {
            long sum = 0;
            for (int j = i; j < a.length; j++) { sum += a[j]; best = Math.max(best, sum); }
        }
        return best;
    }
    static int lowerBound(int[] a, int key) {
        int lo = 0, hi = a.length;
        while (lo < hi) { int mid = (lo + hi) >>> 1; if (a[mid] < key) lo = mid + 1; else hi = mid; }
        return lo;
    }
    static int lookupKey(int[] a) { return a.length == 0 ? 0 : a[a.length / 2]; }
    static final Solve KADANE = (a, ops) -> {
        long best = 0, run = 0;
        for (int i = 0; i < a.length; i++) { run = Math.max(0, run + a[i]); best = Math.max(best, run); ops[0]++; }
        return best;
    };
    /** Same answer as KADANE with three passes per element: the "regressed" revision. */
    static final Solve PADDED_KADANE = (a, ops) -> {
        long best = 0;
        for (int pass = 0; pass < 3; pass++) {
            long run = 0;
            for (int i = 0; i < a.length; i++) { run = Math.max(0, run + a[i]); best = Math.max(best, run); ops[0]++; }
        }
        return best;
    };
    static final Solve BRUTE = (a, ops) -> {
        long best = 0;
        for (int i = 0; i < a.length; i++) {
            long sum = 0;
            for (int j = i; j < a.length; j++) { sum += a[j]; best = Math.max(best, sum); ops[0]++; }
        }
        return best;
    };
    static final Solve DIVIDE = (a, ops) -> maxSubarray(a, 0, a.length, ops);
    /** 2T(n/2) + n counted work, i.e. exactly n log2 n for power-of-two sizes. */
    static long maxSubarray(int[] a, int lo, int hi, long[] ops) {
        if (lo >= hi) return 0;
        if (hi - lo == 1) { ops[0]++; return Math.max(0, a[lo]); }
        int mid = (lo + hi) >>> 1; long left = maxSubarray(a, lo, mid, ops), right = maxSubarray(a, mid, hi, ops);
        long bestLeft = Long.MIN_VALUE, bestRight = Long.MIN_VALUE, sum = 0;
        for (int i = mid - 1; i >= lo; i--) { sum += a[i]; bestLeft = Math.max(bestLeft, sum); ops[0]++; }
        sum = 0;
        for (int i = mid; i < hi; i++) { sum += a[i]; bestRight = Math.max(bestRight, sum); ops[0]++; }
        return Math.max(Math.max(left, right), Math.max(0, bestLeft + bestRight));
    }
    /** Deliberately wrong: sums only the first run of positive values. */
    static final Solve FIRST_POSITIVE_RUN = (a, ops) -> {
        long sum = 0; int i = 0;
        while (i < a.length && a[i] <= 0) i++;
        for (; i < a.length && a[i] > 0; i++) { sum += a[i]; ops[0]++; }
        return sum;
    };
    static final Solve BINARY_LOOKUP = (a, ops) -> {
        if (a.length == 0) return 0;
        int key = lookupKey(a), lo = 0, hi = a.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1; ops[0]++;
            if (a[mid] < key) lo = mid + 1; else hi = mid;
        }
        return lo;
    };
    static final Solve LINEAR_LOOKUP = (a, ops) -> {
        if (a.length == 0) return 0;
        int key = lookupKey(a), i = 0;
        while (i < a.length && a[i] < key) { i++; ops[0]++; }
        return i;
    };
    static final String[] MODELS = { "O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)" };
    static double model(int m, int n) {
        switch (m) {
            case 0: return 1.0;
            case 1: return Math.log(n) / Math.log(2);
            case 2: return n;
            case 3: return n * (Math.log(n) / Math.log(2));
            default: return (double) n * n;
        }
    }
    /** Least-squares amplitude c per model; the smallest relative residual wins. */
    static String fitClass(int[] sizes, long[] ops) {
        int best = 0; double bestResidual = Double.MAX_VALUE;
        for (int m = 0; m < MODELS.length; m++) {
            double sff = 0, sof = 0, soo = 0;
            for (int i = 0; i < sizes.length; i++) {
                double f = model(m, sizes[i]); sff += f * f; sof += ops[i] * f; soo += (double) ops[i] * ops[i];
            }
            double c = sof / sff, sse = soo - 2 * c * sof + c * c * sff;
            double residual = sse / (soo <= 0 ? 1 : soo);
            if (residual < bestResidual) { bestResidual = residual; best = m; }
        }
        return MODELS[best];
    }
    record TestCase(String label, int[] input, long expected) {}
    record Revision(String name, String expectedClass, Solve solve, Gen gen) {}
    record Grade(String problem, String revision, boolean passed, String failure, TestCase failingCase, long[] ops, String fitted, String expected, boolean regression) {}
    static final class Problem {
        final String name; final Oracle oracle; final Gen gen; final int[] caseSizes, fitSizes;
        final List<Revision> revisions = new ArrayList<>();
        Problem(String name, Oracle oracle, Gen gen, int[] caseSizes, int[] fitSizes) {
            this.name = name; this.oracle = oracle; this.gen = gen;
            this.caseSizes = caseSizes; this.fitSizes = fitSizes;
        }
        Problem add(Revision r) { revisions.add(r); return this; }
        List<TestCase> cases(long seed) {
            List<TestCase> out = new ArrayList<>();
            for (int size : caseSizes)
                for (int variant = 0; variant < 2; variant++) {
                    int[] input = gen.make(size, seed + 1000L * variant + size);
                    out.add(new TestCase(name + ":n=" + size + "#" + variant, input, oracle.expected(input)));
                }
            return out;
        }
    }
    static final class Grader {
        static final long REGRESSION_PERCENT = 15; final Map<String, Long> baseline = new TreeMap<>();
        List<Grade> grade(Problem p, long seed) {
            List<TestCase> cases = p.cases(seed); List<Grade> out = new ArrayList<>();
            for (Revision r : p.revisions) {
                boolean passed = true; String failure = "none"; TestCase failing = null;
                for (TestCase c : cases) {
                    long got = r.solve().solve(c.input().clone(), new long[1]);
                    if (got != c.expected()) {
                        passed = false; failing = c;
                        failure = c.label() + " expected=" + c.expected() + " got=" + got; break;
                    }
                }
                long[] ops = new long[p.fitSizes.length];
                for (int i = 0; i < p.fitSizes.length; i++) {
                    long[] counter = new long[1];
                    r.solve().solve(r.gen().make(p.fitSizes[i], seed + 17L * p.fitSizes[i]), counter);
                    ops[i] = counter[0];
                }
                Long base = baseline.get(p.name + "/" + r.name());
                boolean regression = base != null && ops[ops.length - 1] * 100 > base * (100 + REGRESSION_PERCENT);
                out.add(new Grade(p.name, r.name(), passed, failure, failing, ops, fitClass(p.fitSizes, ops), r.expectedClass(), regression));
            }
            return out;
        }
        void captureBaseline(List<Grade> grades) {
            baseline.clear();
            for (Grade g : grades) baseline.put(g.problem() + "/" + g.revision(), g.ops()[g.ops().length - 1]);
        }
    }
    static final int[] CASE_SIZES = { 0, 1, 2, 3, 7, 16, 31 };
    static final int[] FIT_SIZES = { 16, 32, 64, 128, 256 };
    static Problem maxSubarrayProblem(Solve kadaneVariant) {
        return new Problem("max-subarray", Solution::bruteMaxSubarray, Solution::maxSubarrayInput, CASE_SIZES, FIT_SIZES)
                .add(new Revision("kadane", "O(n)", kadaneVariant, Solution::maxSubarrayInput))
                .add(new Revision("brute-force", "O(n^2)", BRUTE, Solution::maxSubarrayInput))
                .add(new Revision("divide-and-conquer", "O(n log n)", DIVIDE, Solution::maxSubarrayInput))
                .add(new Revision("first-positive-run", "O(n)", FIRST_POSITIVE_RUN, Solution::maxSubarrayInput));
    }
    static Problem lookupProblem() {
        return new Problem("sorted-lookup", a -> lowerBound(a, lookupKey(a)), Solution::lookupInput, CASE_SIZES, FIT_SIZES)
                .add(new Revision("binary-search", "O(log n)", BINARY_LOOKUP, Solution::lookupInput))
                .add(new Revision("linear-scan", "O(n)", LINEAR_LOOKUP, Solution::lookupInput));
    }
    static Grade find(List<Grade> grades, String problem, String revision) {
        for (Grade g : grades) if (g.problem().equals(problem) && g.revision().equals(revision)) return g;
        return null;
    }
    static int regressions(List<Grade> grades) {
        int n = 0;
        for (Grade g : grades) if (g.regression()) n++;
        return n;
    }
    public static void main(String[] args) {
        Grader grader = new Grader(); List<Grade> maxSub = grader.grade(maxSubarrayProblem(KADANE), 4242L);
        List<Grade> lookup = grader.grade(lookupProblem(), 4242L);
        // 1. every correct submission passes, n = 0 and n = 1 included
        boolean correctPass = true;
        for (Grade g : maxSub) if (!g.revision().equals("first-positive-run")) correctPass &= g.passed();
        for (Grade g : lookup) correctPass &= g.passed();
        check(correctPass, "all correct submissions pass the hidden battery including boundary sizes");
        // 2. the wrong submission is rejected at its first mismatching case
        Grade wrong = find(maxSub, "max-subarray", "first-positive-run");
        long wrongOnFailingCase = wrong == null ? 0 : FIRST_POSITIVE_RUN.solve(wrong.failingCase().input().clone(), new long[1]);
        check(wrong != null && !wrong.passed() && wrong.failingCase() != null && wrong.failingCase().expected() == bruteMaxSubarray(wrong.failingCase().input()) && wrong.failingCase().expected() != wrongOnFailingCase, "wrong submission reports its first failing case and the oracle confirms it");
        // 3. complexity inference matches the intended shape of each submission
        boolean classesOk = true;
        for (Grade g : maxSub) if (!g.revision().equals("first-positive-run")) classesOk &= g.fitted().equals(g.expected());
        for (Grade g : lookup) classesOk &= g.fitted().equals(g.expected());
        check(classesOk, "fitted complexity class equals the expected class for every submission");
        // 4. determinism: grading twice yields identical counts and classes
        List<Grade> again = new Grader().grade(maxSubarrayProblem(KADANE), 4242L);
        boolean reproducible = maxSub.size() == again.size();
        for (int i = 0; reproducible && i < maxSub.size(); i++)
            reproducible &= maxSub.get(i).revision().equals(again.get(i).revision())
                    && Arrays.equals(maxSub.get(i).ops(), again.get(i).ops())
                    && maxSub.get(i).fitted().equals(again.get(i).fitted());
        check(reproducible, "operation counts and fitted classes reproduce on a second grading run");
        // 5. regression: the padded revision trips the tolerance, nothing else does
        Grader regrade = new Grader(); regrade.captureBaseline(maxSub);
        List<Grade> padded = regrade.grade(maxSubarrayProblem(PADDED_KADANE), 4242L);
        Grade paddedKadane = find(padded, "max-subarray", "kadane"); boolean othersStable = true;
        for (Grade g : padded) if (!g.revision().equals("kadane")) othersStable &= !g.regression();
        check(paddedKadane != null && paddedKadane.regression() && paddedKadane.passed() && paddedKadane.ops()[4] == 3 * find(maxSub, "max-subarray", "kadane").ops()[4] && othersStable, "padded revision is flagged as a regression although it is still correct");
        // 6. an unchanged rerun against the same baseline reports nothing
        Grader clean = new Grader(); clean.captureBaseline(maxSub);
        check(regressions(clean.grade(maxSubarrayProblem(KADANE), 4242L)) == 0, "identical rerun produces zero regressions (no false positives)");
        // 7. boundary: empty and single-element inputs, negative singleton included
        check(bruteMaxSubarray(new int[0]) == 0 && KADANE.solve(new int[0], new long[1]) == 0 && bruteMaxSubarray(new int[] { -5 }) == 0 && DIVIDE.solve(new int[] { -5 }, new long[1]) == 0 && lowerBound(new int[0], 7) == 0 && BINARY_LOOKUP.solve(new int[0], new long[1]) == 0 && new Grader().grade(maxSubarrayProblem(KADANE), 1L).get(0).passed(), "empty array grades 0, a negative singleton grades 0 and empty lookup returns 0");
        // 8. independent oracle: the three correct max-subarray revisions agree on 20 arrays
        boolean oracleAgrees = true;
        for (int i = 0; i < 20; i++) {
            int[] a = randomArray(1 + i * 3, 7000L + i, 25); long expected = bruteMaxSubarray(a);
            oracleAgrees &= KADANE.solve(a.clone(), new long[1]) == expected
                    && DIVIDE.solve(a.clone(), new long[1]) == expected
                    && BRUTE.solve(a.clone(), new long[1]) == expected;
        }
        check(oracleAgrees, "Kadane, divide-and-conquer and brute force agree with the oracle on 20 arrays");
        System.out.println("submissions graded: " + (maxSub.size() + lookup.size()));
        System.out.println("kadane ops by n: " + Arrays.toString(find(maxSub, "max-subarray", "kadane").ops()) + " -> " + find(maxSub, "max-subarray", "kadane").fitted());
        System.out.println("brute-force ops by n: " + Arrays.toString(find(maxSub, "max-subarray", "brute-force").ops()) + " -> " + find(maxSub, "max-subarray", "brute-force").fitted());
        System.out.println("divide-and-conquer -> " + find(maxSub, "max-subarray", "divide-and-conquer").fitted() + ", binary-search -> " + find(lookup, "sorted-lookup", "binary-search").fitted() + ", linear-scan -> " + find(lookup, "sorted-lookup", "linear-scan").fitted());
        System.out.println("first failing case: " + wrong.failure());
        System.out.println("regressions in padded run: " + regressions(padded)); System.out.println("PASS");
    }
}
