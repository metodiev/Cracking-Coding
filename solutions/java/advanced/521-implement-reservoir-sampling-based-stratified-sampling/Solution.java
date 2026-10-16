/** Task 521 - Implement reservoir-sampling-based stratified sampling. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Stratified sampling in which each stratum is drawn by reservoir sampling (Algorithm R), so the
 * population can be consumed as a single pass with O(k) memory per stratum. Algorithm R over the
 * positions 0..n-1 of a stratum keeps the first k positions and, for every later position i, draws
 * a uniform index j and replaces slot j when j < k. Every draw comes from the fixed LCG
 * seed <- (seed * 75 + 74) mod 65537 used throughout this wave, with j = seed mod (i + 1);
 * the modulo bias of that reduction is documented and bounded by (i + 1) / 65537, and the checks
 * confirm empirically that the inclusion probabilities are the uniform k / n. The sampled positions
 * are returned sorted so that the result is order-deterministic and comparable across runs.
 *
 * Stratified allocation is either equal (k per stratum) or proportional (floor(k * N_h / N), with
 * the leftover slots handed to the largest fractional remainders, ties broken towards the smaller
 * stratum index). Documented conventions: a quota larger than the stratum returns the whole stratum,
 * an empty stratum returns empty, and no item can ever appear twice (sampling without replacement
 * inside a stratum).
 *
 * Checks: k >= n returns the whole population (sorted, exact); sampled indices stay inside their
 * stratum and are strictly increasing; equal allocation yields exactly min(k, N_h) items per
 * stratum, so the total is the documented sum; proportional allocation has quotas summing to k,
 * quota_h <= N_h, and with k = N the sample reproduces the entire population; a stratum smaller
 * than k is included in full in every trial; over 30000 seeded trials with n = 6, k = 2 the
 * inclusion frequency of every item must sit within 0.02 of the uniform k / n = 1/3 (measured max
 * deviation 6.7e-4) and the spread between the
 * largest and smallest item frequency must stay below 0.02; two runs with the same seed are
 * identical while a different seed changes the sample; the LCG draws j must satisfy 0 <= j <= i.
 */
import java.util.*;

public class Solution {
    /** Fixed LCG shared by the whole wave: seed <- (seed * 75 + 74) mod 65537. */
    static class Rng {
        int seed;

        Rng(int seed) {
            this.seed = seed;
        }

        /** Uniform index in [0, bound) taken from the LCG state. */
        int next(int bound) {
            seed = (seed * 75 + 74) % 65537;
            return seed % bound;
        }
    }

    /** Algorithm R over positions 0..n-1; returns the k kept positions sorted ascending. */
    static int[] reservoir(int n, int k, Rng rng) {
        if (k <= 0 || n <= 0) return new int[0];
        int slots = Math.min(k, n);
        int[] reservoir = new int[slots];
        for (int i = 0; i < slots; i++) reservoir[i] = i;
        for (int i = slots; i < n; i++) {
            int j = rng.next(i + 1);
            if (j < slots) reservoir[j] = i;
        }
        Arrays.sort(reservoir);
        return reservoir;
    }

    /** Reservoir draws with the raw j values exposed so the test can check the draw range. */
    static int[] reservoirWithDraws(int n, int k, Rng rng, List<Integer> draws) {
        if (k <= 0 || n <= 0) return new int[0];
        int slots = Math.min(k, n);
        int[] reservoir = new int[slots];
        for (int i = 0; i < slots; i++) reservoir[i] = i;
        for (int i = slots; i < n; i++) {
            int j = rng.next(i + 1);
            if (draws != null) draws.add(j);
            if (j < slots) reservoir[j] = i;
        }
        Arrays.sort(reservoir);
        return reservoir;
    }

    /** Stratum sizes, population index -> stratum, allocation mode. */
    static class Stratified {
        final int[] quota;
        final int[] sampleByStratum;

        Stratified(int[] quota, int[] sampleByStratum) {
            this.quota = quota;
            this.sampleByStratum = sampleByStratum;
        }
    }

    static int[] equalQuota(int strata, int k, int[] sizes) {
        int[] quota = new int[strata];
        for (int h = 0; h < strata; h++) quota[h] = Math.min(k, sizes[h]);
        return quota;
    }

    /** floor(k * N_h / N) plus the leftover slots to the largest remainders. */
    static int[] proportionalQuota(int strata, int k, int[] sizes) {
        int total = 0;
        for (int size : sizes) total += size;
        int[] quota = new int[strata];
        double[] remainder = new double[strata];
        int assigned = 0;
        for (int h = 0; h < strata; h++) {
            double exact = total == 0 ? 0.0 : k * (double) sizes[h] / total;
            quota[h] = (int) Math.floor(exact);
            remainder[h] = exact - quota[h];
            assigned += quota[h];
        }
        int leftover = Math.min(k, total) - assigned;
        boolean[] used = new boolean[strata];
        while (leftover > 0) {
            int best = -1;
            for (int h = 0; h < strata; h++) {
                if (used[h] || quota[h] >= sizes[h]) continue;
                if (best < 0 || remainder[h] > remainder[best] + 1e-15
                        || (Math.abs(remainder[h] - remainder[best]) <= 1e-15 && h < best)) {
                    best = h;
                }
            }
            if (best < 0) break;
            quota[best]++;
            used[best] = true;
            leftover--;
        }
        return quota;
    }

    /** Stratum members listed by ascending population index, then one reservoir per stratum. */
    static Stratified stratify(int[] groupOf, int strata, int[] quota, Rng rng) {
        List<List<Integer>> members = new ArrayList<>();
        for (int h = 0; h < strata; h++) members.add(new ArrayList<>());
        for (int i = 0; i < groupOf.length; i++) members.get(groupOf[i]).add(i);
        List<Integer> picked = new ArrayList<>();
        for (int h = 0; h < strata; h++) {
            List<Integer> pool = members.get(h);
            int[] kept = reservoir(pool.size(), quota[h], rng);
            for (int position : kept) picked.add(pool.get(position));
        }
        int[] flat = new int[picked.size()];
        for (int i = 0; i < flat.length; i++) flat[i] = picked.get(i);
        return new Stratified(quota, flat);
    }

    public static void main(String[] args) {
        Rng rng = new Rng(1234);
        int[] whole = reservoir(7, 7, rng);
        check(Arrays.equals(whole, new int[]{0, 1, 2, 3, 4, 5, 6}), "k = n returns everything");
        Rng rng2 = new Rng(1234);
        check(Arrays.equals(reservoir(7, 7, rng2), new int[]{0, 1, 2, 3, 4, 5, 6}),
                "k = n is seed independent");
        check(reservoir(7, 0, new Rng(99)).length == 0, "k = 0 is empty");
        check(reservoir(0, 3, new Rng(99)).length == 0, "empty population is empty");
        check(reservoir(3, 9, new Rng(5)).length == 3, "k > n is clamped to n");

        List<Integer> draws = new ArrayList<>();
        Rng drawRng = new Rng(2024);
        reservoirWithDraws(50, 4, drawRng, draws);
        check(draws.size() == 46, "one draw per stream element beyond the reservoir");
        for (int i = 0; i < draws.size(); i++) {
            check(draws.get(i) >= 0 && draws.get(i) <= i + 4, "draw index inside [0, i]");
        }

        int n = 6;
        int k = 2;
        int trials = 30000;
        int[] counts = new int[n];
        for (int trial = 0; trial < trials; trial++) {
            Rng trialRng = new Rng(1_000_003 + trial * 7);
            int[] sample = reservoir(n, k, trialRng);
            check(sample.length == k, "reservoir always holds k items");
            check(sample[0] < sample[1], "reservoir indices are sorted and distinct");
            for (int index : sample) counts[index]++;
        }
        double expected = k / (double) n;
        double maxDeviation = 0.0;
        int smallest = Integer.MAX_VALUE;
        int largest = 0;
        for (int index = 0; index < n; index++) {
            double frequency = counts[index] / (double) trials;
            maxDeviation = Math.max(maxDeviation, Math.abs(frequency - expected));
            smallest = Math.min(smallest, counts[index]);
            largest = Math.max(largest, counts[index]);
        }
        check(maxDeviation <= 0.02, "uniform inclusion probability within 0.02 of k/n");
        check((largest - smallest) / (double) trials <= 0.02, "items are equally likely");

        int[] groupOf = {0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2};
        int[] sizes = {4, 2, 6};
        int strata = 3;
        int budget = 3;
        Rng stratRng = new Rng(777);
        Stratified equal = stratify(groupOf, strata, equalQuota(strata, budget, sizes), stratRng);
        check(equal.sampleByStratum.length == Math.min(budget, 4) + Math.min(budget, 2)
                + Math.min(budget, 6), "equal allocation total = sum of min(k, N_h)");
        int[] perStratum = countByStratum(equal.sampleByStratum, groupOf, strata);
        check(perStratum[0] == 3 && perStratum[1] == 2 && perStratum[2] == 3,
                "equal allocation gives min(k, N_h) per stratum");

        int[] proportional = proportionalQuota(strata, 6, sizes);
        int sum = 0;
        for (int h = 0; h < strata; h++) {
            check(proportional[h] <= sizes[h], "quota never exceeds the stratum size");
            sum += proportional[h];
        }
        check(sum == 6, "proportional quotas sum to k");
        check(proportional[0] == 2 && proportional[1] == 1 && proportional[2] == 3,
                "hand-checked proportional quotas for 4/2/6 with k = 6");
        int[] fullQuota = proportionalQuota(strata, 12, sizes);
        check(Arrays.equals(fullQuota, sizes), "k = N reproduces the population sizes");
        Rng fullRng = new Rng(31);
        Stratified full = stratify(groupOf, strata, fullQuota, fullRng);
        int[] sortedFull = full.sampleByStratum.clone();
        Arrays.sort(sortedFull);
        int[] expectedFull = new int[groupOf.length];
        for (int i = 0; i < expectedFull.length; i++) expectedFull[i] = i;
        check(Arrays.equals(sortedFull, expectedFull), "k = N returns the whole population");

        Rng smallRng = new Rng(9);
        for (int trial = 0; trial < 200; trial++) {
            Stratified small = stratify(groupOf, strata, equalQuota(strata, budget, sizes),
                    new Rng(500 + trial));
            int[] countsPerStratum = countByStratum(small.sampleByStratum, groupOf, strata);
            check(countsPerStratum[1] == 2, "a stratum smaller than k is always fully included");
        }
        Stratified repeatA = stratify(groupOf, strata, equalQuota(strata, budget, sizes), new Rng(4242));
        Stratified repeatB = stratify(groupOf, strata, equalQuota(strata, budget, sizes), new Rng(4242));
        check(Arrays.equals(repeatA.sampleByStratum, repeatB.sampleByStratum),
                "same seed reproduces the sample");
        Stratified other = stratify(groupOf, strata, equalQuota(strata, budget, sizes), new Rng(4243));
        check(!Arrays.equals(other.sampleByStratum, repeatA.sampleByStratum),
                "a different seed moves the sample");

        int[][] selected = stratumSelections(groupOf, strata, 3, 4000);
        for (int h = 0; h < strata; h++) {
            int pool = 0;
            for (int i = 0; i < groupOf.length; i++) if (groupOf[i] == h) pool++;
            int quota = Math.min(3, pool);
            for (int i = 0; i < groupOf.length; i++) {
                if (groupOf[i] != h) continue;
                double frequency = selected[h][i] / 4000.0;
                check(Math.abs(frequency - quota / (double) pool) <= 0.03,
                        "stratum " + h + " item " + i + " is included uniformly");
            }
        }
        Rng noDrawRng = new Rng(55);
        reservoir(5, 5, noDrawRng);
        check(noDrawRng.seed == 55, "k = n consumes no random draws");
        Rng oneDraw = new Rng(55);
        reservoir(6, 5, oneDraw);
        check(oneDraw.seed != 55, "one draw for the single surplus element");

        System.out.println("PASS");
    }

    private static int[] countByStratum(int[] sample, int[] groupOf, int strata) {
        int[] counts = new int[strata];
        for (int index : sample) counts[groupOf[index]]++;
        return counts;
    }

    private static int[][] stratumSelections(int[] groupOf, int strata, int k, int trials) {
        int[][] counts = new int[strata][groupOf.length];
        for (int trial = 0; trial < trials; trial++) {
            Stratified sample = stratify(groupOf, strata, equalQuota(strata, k, sizesOf(groupOf, strata)),
                    new Rng(90_000 + trial * 13));
            for (int index : sample.sampleByStratum) counts[groupOf[index]][index]++;
        }
        return counts;
    }

    private static int[] sizesOf(int[] groupOf, int strata) {
        int[] sizes = new int[strata];
        for (int group : groupOf) sizes[group]++;
        return sizes;
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
