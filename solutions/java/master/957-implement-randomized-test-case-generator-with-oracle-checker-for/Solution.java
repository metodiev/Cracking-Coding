/** Task 957 - Implement randomized test-case generator with oracle checker for each problem.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory test-case factory for the "count inversions" problem. A fixed-seed splitmix64 generator
 *  produces random arrays with sizes and values drawn from explicit bounds, a curated set covers the
 *  boundary shapes (empty, single, sorted, reversed, all-equal, extreme values), and every case is
 *  checked against a quadratic brute-force oracle before the candidate solution is trusted. When the
 *  oracle disagrees, a delta-debugging shrinker removes elements and halves values until the failure is
 *  irreducible, so the reported reproducer is minimal.
 *  Simplified: cases live in memory instead of a JSON corpus and only one problem is wired up, but the
 *  generator bounds, oracle checking and shrinking are the real ones.
 *  Verified: generated shapes stay inside the declared bounds; the fast merge count agrees with the
 *  quadratic and the insertion-sort oracles; the seed determines the corpus; the boundary set covers
 *  degenerate shapes; the planted bug is caught and shrunk to [0, 0]; the shrunk case is irreducible and
 *  still fails while the correct solution passes; shrinking is deterministic.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static long splitmix64(long x) {
        long z = x + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final long FNV_OFFSET = 0xcbf29ce484222325L, FNV_PRIME = 0x100000001b3L;

    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }

    static String hex(long v) { return String.format("%016x", v); }

    record Case(String label, int[] values) {}

    interface InversionCounter { long count(int[] a); }
    interface CasePredicate { boolean test(int[] a); }

    record Mismatch(String label, int[] input, long expected, long got) {}

    static final class CaseGenerator {
        final long seed;
        CaseGenerator(long seed) { this.seed = seed; }

        List<Case> random(int count, int sizeBound, int valueBound) {
            List<Case> cases = new ArrayList<>();
            long state = seed;
            for (int i = 0; i < count; i++) {
                state = splitmix64(state + i);
                int size = (int) Math.floorMod(state, sizeBound + 1);
                int[] values = new int[size];
                long local = state;
                for (int j = 0; j < size; j++) {
                    local = splitmix64(local + j);
                    values[j] = (int) Math.floorMod(local, 2L * valueBound + 1) - valueBound;
                }
                cases.add(new Case("random#" + i + ":n=" + size, values));
            }
            return cases;
        }

        String digest(List<Case> cases) {
            long h = fnvText(FNV_OFFSET, "corpus");
            for (Case c : cases) {
                h = fnvText(h, c.label());
                for (int v : c.values()) h = fnvText(h, v + ",");
            }
            return hex(h);
        }
    }

    static List<Case> curated() {
        return List.of(
                new Case("empty", new int[0]),
                new Case("single", new int[] { 7 }),
                new Case("sorted", new int[] { 1, 2, 3, 4, 5 }),
                new Case("reversed", new int[] { 5, 4, 3, 2, 1 }),
                new Case("all-equal", new int[] { 3, 3, 3, 3 }),
                new Case("extremes", new int[] { -1000000, 1000000, 0 }));
    }

    /** Oracle 1: quadratic double loop over strictly decreasing pairs. */
    static long oracleInversions(int[] a) {
        long count = 0;
        for (int i = 0; i < a.length; i++)
            for (int j = i + 1; j < a.length; j++) if (a[i] > a[j]) count++;
        return count;
    }

    /** Oracle 2: insertion sort swap count, a completely different mechanism. */
    static long insertionSwaps(int[] a) {
        int[] b = a.clone();
        long swaps = 0;
        for (int i = 1; i < b.length; i++) {
            int value = b[i], j = i - 1;
            while (j >= 0 && b[j] > value) { b[j + 1] = b[j]; j--; swaps++; }
            b[j + 1] = value;
        }
        return swaps;
    }

    /** Candidate solution: O(n log n) merge count. */
    static long fastInversions(int[] a) { return mergeCount(a.clone(), new int[a.length], 0, a.length); }

    static long mergeCount(int[] a, int[] tmp, int lo, int hi) {
        if (hi - lo <= 1) return 0;
        int mid = (lo + hi) >>> 1;
        long count = mergeCount(a, tmp, lo, mid) + mergeCount(a, tmp, mid, hi);
        int i = lo, j = mid, k = lo;
        while (i < mid && j < hi) {
            if (a[i] <= a[j]) tmp[k++] = a[i++];
            else { count += mid - i; tmp[k++] = a[j++]; }
        }
        while (i < mid) tmp[k++] = a[i++];
        while (j < hi) tmp[k++] = a[j++];
        System.arraycopy(tmp, lo, a, lo, hi - lo);
        return count;
    }

    /** Planted bug: counts equal pairs too, because the comparison lost its strictness. */
    static long buggyInversions(int[] a) { return mergeCountBuggy(a.clone(), new int[a.length], 0, a.length); }

    static long mergeCountBuggy(int[] a, int[] tmp, int lo, int hi) {
        if (hi - lo <= 1) return 0;
        int mid = (lo + hi) >>> 1;
        long count = mergeCountBuggy(a, tmp, lo, mid) + mergeCountBuggy(a, tmp, mid, hi);
        int i = lo, j = mid, k = lo;
        while (i < mid && j < hi) {
            if (a[i] < a[j]) tmp[k++] = a[i++];
            else { count += mid - i; tmp[k++] = a[j++]; }
        }
        while (i < mid) tmp[k++] = a[i++];
        while (j < hi) tmp[k++] = a[j++];
        System.arraycopy(tmp, lo, a, lo, hi - lo);
        return count;
    }

    static Mismatch firstMismatch(List<Case> cases, InversionCounter candidate) {
        for (Case c : cases) {
            long expected = oracleInversions(c.values());
            long got = candidate.count(c.values().clone());
            if (expected != got) return new Mismatch(c.label(), c.values().clone(), expected, got);
        }
        return null;
    }

    static int[] removeAt(int[] a, int index) {
        int[] out = new int[a.length - 1];
        System.arraycopy(a, 0, out, 0, index);
        System.arraycopy(a, index + 1, out, index, out.length - index);
        return out;
    }

    /** Delta debugging: drop elements, halve single values, then halve the whole vector, keeping failures. */
    static int[] shrink(int[] input, CasePredicate stillFails) {
        int[] current = input.clone();
        boolean improved = true;
        while (improved && current.length > 1) {
            improved = false;
            for (int i = 0; i < current.length && !improved; i++) {
                int[] candidate = removeAt(current, i);
                if (stillFails.test(candidate)) { current = candidate; improved = true; }
            }
        }
        improved = true;
        while (improved) {
            improved = false;
            for (int i = 0; i < current.length && !improved; i++) {
                if (current[i] == 0) continue;
                int[] candidate = current.clone();
                candidate[i] = candidate[i] / 2;
                if (stillFails.test(candidate)) { current = candidate; improved = true; }
            }
        }
        improved = true;
        while (improved) {
            improved = false;
            int[] candidate = new int[current.length];
            boolean anyNonZero = false;
            for (int i = 0; i < current.length; i++) {
                candidate[i] = current[i] / 2;
                anyNonZero |= current[i] != 0;
            }
            if (anyNonZero && stillFails.test(candidate)) { current = candidate; improved = true; }
        }
        return current;
    }

    static boolean fails(int[] a) {
        return firstMismatch(List.of(new Case("probe", a)), Solution::buggyInversions) != null;
    }

    public static void main(String[] args) {
        CaseGenerator generator = new CaseGenerator(20241016L);
        List<Case> random = generator.random(25, 40, 30);
        List<Case> all = new ArrayList<>(curated());
        all.addAll(random);

        // 1. generated shapes stay inside the declared bounds
        boolean shapesOk = random.size() == 25;
        for (Case c : random) {
            shapesOk &= c.values().length <= 40;
            for (int v : c.values()) shapesOk &= v >= -30 && v <= 30;
        }
        check(shapesOk, "25 generated cases respect the size bound 40 and the value bound 30");

        // 2. the candidate agrees with the oracle on every case, curated and random
        check(firstMismatch(all, Solution::fastInversions) == null,
                "merge-count candidate matches the quadratic oracle on all " + all.size() + " cases");

        // 3. the seed determines the corpus
        CaseGenerator same = new CaseGenerator(20241016L);
        CaseGenerator other = new CaseGenerator(20241017L);
        check(generator.digest(random).equals(same.digest(same.random(25, 40, 30)))
                        && !generator.digest(random).equals(other.digest(other.random(25, 40, 30))),
                "same seed reproduces the corpus digest, a different seed does not");

        // 4. boundary coverage: degenerate shapes are present and answered
        Set<String> labels = new TreeSet<>();
        for (Case c : curated()) labels.add(c.label());
        check(labels.size() == 6 && labels.contains("empty") && labels.contains("all-equal")
                        && oracleInversions(new int[0]) == 0 && oracleInversions(new int[] { 5 }) == 0
                        && fastInversions(new int[] { 5, 4, 3, 2, 1 }) == 10
                        && fastInversions(new int[] { 3, 3, 3, 3 }) == 0,
                "curated set covers empty, single, sorted, reversed, all-equal and extremes");

        // 5. the planted bug is caught on a meaningful case
        Mismatch mismatch = firstMismatch(all, Solution::buggyInversions);
        check(mismatch != null && mismatch.got() > mismatch.expected() && hasEqualPair(mismatch.input()),
                "buggy candidate is caught and the mismatch is caused by counting equal pairs");

        // 6. shrinking reaches a minimal reproducer that still fails
        int[] shrunk = shrink(mismatch.input(), Solution::fails);
        boolean irreducible = true;
        for (int i = 0; i < shrunk.length; i++) irreducible &= !fails(removeAt(shrunk, i));
        check(shrunk.length == 2 && Arrays.equals(shrunk, new int[] { 0, 0 }) && irreducible
                        && fails(shrunk) && fastInversions(shrunk) == oracleInversions(shrunk),
                "shrinker reaches the minimal reproducer [0, 0] with no removable element left");

        // 7. shrinking is deterministic and portable across failing corpora
        int[] again = shrink(mismatch.input(), Solution::fails);
        Mismatch otherMismatch = null;
        for (long seed = 700L; seed < 800L && otherMismatch == null; seed++)
            otherMismatch = firstMismatch(new CaseGenerator(seed).random(25, 60, 9), Solution::buggyInversions);
        int[] shrunkOther = shrink(otherMismatch.input(), Solution::fails);
        check(Arrays.equals(shrunk, again) && otherMismatch != null && Arrays.equals(shrunkOther, new int[] { 0, 0 })
                        && shrunk.length <= Math.max(2, mismatch.input().length / 2),
                "shrinking is deterministic and reaches the same minimal reproducer from another corpus");

        // 8. independent oracles: insertion-sort swaps and long arithmetic on a larger case
        boolean oraclesAgree = true;
        for (int i = 0; i < 30; i++) {
            int[] a = new CaseGenerator(31000L + i).random(1, 24, 12).get(0).values();
            oraclesAgree &= insertionSwaps(a) == oracleInversions(a) && fastInversions(a) == insertionSwaps(a);
        }
        int[] reversedLarge = new int[400];
        for (int i = 0; i < reversedLarge.length; i++) reversedLarge[i] = reversedLarge.length - i;
        int[] equalLarge = new int[400];
        long expectedReversed = 400L * 399L / 2;
        check(oraclesAgree && fastInversions(reversedLarge) == expectedReversed
                        && fastInversions(equalLarge) == 0L && oracleInversions(reversedLarge) == expectedReversed,
                "three independent counters agree, including a 400-element reversed case");

        System.out.println("cases generated: " + all.size() + " (curated " + curated().size() + ")");
        System.out.println("corpus digest: " + generator.digest(random));
        System.out.println("first buggy mismatch: " + mismatch.label() + " expected=" + mismatch.expected()
                + " got=" + mismatch.got());
        System.out.println("minimal reproducer: " + Arrays.toString(shrunk));
        System.out.println("PASS");
    }

    /** True when the case contains an equal pair, which is exactly what the buggy merge miscounts. */
    static boolean hasEqualPair(int[] a) {
        for (int i = 0; i < a.length; i++)
            for (int j = i + 1; j < a.length; j++) if (a[i] == a[j]) return true;
        return false;
    }
}
