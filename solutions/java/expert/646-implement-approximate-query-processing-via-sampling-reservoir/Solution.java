/** Task 646 - Implement approximate query processing via sampling reservoir. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * Two real reservoir samplers over a deterministic stream of items, plus the AQP estimators built on
 * them:
 *
 *   algorithm R  - the classic single-pass uniform reservoir: the first k items fill the reservoir and
 *                  after i items seen, item i replaces a uniformly chosen slot with probability k/i;
 *   A-Res        - Efraimidis-Spirakis weighted sampling: each item gets the key u^(1/w) with u uniform
 *                  and the k largest keys win, which makes the inclusion probability proportional to
 *                  the weight.
 *
 * Estimators (Horvitz-Thompson style, with the sampling fraction known because the full scan reports
 * the stream length): SUM = (n/k) * sum of the sample, AVG = mean of the sample, COUNT(predicate) =
 * n * (matching sample items)/k. Standard errors are computed in closed form from the sampling
 * fraction (sqrt(1 - k/n) finite-population correction), which is what makes the accuracy checks
 * quantitative instead of hand waving.
 *
 * Simplified (deliberately): a single reservoir per query, no stratified or correlated sampling, no
 * progress reporting and no incremental aggregates beyond the estimators above.
 *
 * Verified: the reservoir never contains duplicates or items outside the stream, and it is the whole
 * stream whenever k >= n (exactness); over 400 independent trials the per-item inclusion frequency of
 * algorithm R stays within four standard deviations of k/n and many different index sets appear (so it
 * is a sample, not the first k rows); the estimated average and the estimated selective count stay
 * inside three standard errors of the exact answer; the average absolute error falls as the reservoir
 * grows; A-Res really favours the heavy items while uniform sampling does not; and the degenerate
 * cases (k = 0, k = 1, empty stream, k > n, single item) are handled or rejected explicitly.
 */
import java.util.*;
import java.util.function.IntPredicate;

class ReservoirSampling {

    static final class Item {
        final int index, value, weight;

        Item(int index, int value, int weight) { this.index = index; this.value = value; this.weight = weight; }
    }

    /** Algorithm R: uniform over all subsets of size k of the stream. */
    static List<Item> uniformReservoir(List<Item> stream, int k, SplitMix64 rng) {
        if (k < 0) throw new IllegalArgumentException("reservoir size must not be negative");
        List<Item> reservoir = new ArrayList<>();
        for (int i = 0; i < stream.size(); i++) {
            Item item = stream.get(i);
            if (i < k) {
                reservoir.add(item);
            } else {
                int slot = rng.nextInt(i + 1);
                if (slot < k) reservoir.set(slot, item);
            }
        }
        return reservoir;
    }

    /** A-Res: key = u^(1/weight); keep the k largest keys (a min-heap of size k). */
    static List<Item> weightedReservoir(List<Item> stream, int k, SplitMix64 rng) {
        if (k < 0) throw new IllegalArgumentException("reservoir size must not be negative");
        PriorityQueue<double[]> heap = new PriorityQueue<>(Comparator.comparingDouble(entry -> entry[0]));
        for (int i = 0; i < stream.size(); i++) {
            Item item = stream.get(i);
            double key = Math.pow(rng.nextDouble(), 1.0 / item.weight);
            if (heap.size() < k) {
                heap.add(new double[]{key, i});
            } else if (k > 0 && key > heap.peek()[0]) {
                heap.poll();
                heap.add(new double[]{key, i});
            }
        }
        List<double[]> kept = new ArrayList<>(heap);
        kept.sort(Comparator.comparingDouble(entry -> entry[1]));
        List<Item> reservoir = new ArrayList<>();
        for (double[] entry : kept) reservoir.add(stream.get((int) entry[1]));
        return reservoir;
    }

    static double exactAverage(List<Item> stream) {
        if (stream.isEmpty()) return Double.NaN;
        double total = 0;
        for (Item item : stream) total += item.value;
        return total / stream.size();
    }

    /** Horvitz-Thompson mean estimate; with a uniform reservoir this is just the sample mean. */
    static double estimatedAverage(List<Item> stream, List<Item> sample) {
        if (stream.isEmpty() || sample.isEmpty()) return Double.NaN;
        double total = 0;
        for (Item item : sample) total += item.value;
        return total / sample.size();
    }

    /** Horvitz-Thompson total estimate: n/k times the sample sum. */
    static double estimatedTotal(List<Item> stream, List<Item> sample) {
        if (stream.isEmpty() || sample.isEmpty()) return Double.NaN;
        double total = 0;
        for (Item item : sample) total += item.value;
        return (stream.size() / (double) sample.size()) * total;
    }

    static double exactCount(List<Item> stream, IntPredicate predicate) {
        long matches = 0;
        for (Item item : stream) if (predicate.test(item.value)) matches++;
        return matches;
    }

    static double estimatedCount(List<Item> stream, List<Item> sample, IntPredicate predicate) {
        if (sample.isEmpty()) return Double.NaN;
        long matches = 0;
        for (Item item : sample) if (predicate.test(item.value)) matches++;
        return stream.size() * (matches / (double) sample.size());
    }

    /** Standard error of the sample mean with the finite population correction. */
    static double standardErrorOfMean(List<Item> stream, int k) {
        int n = stream.size();
        if (n < 2 || k < 2) return Double.NaN;
        double mean = exactAverage(stream);
        double variance = 0;
        for (Item item : stream) variance += (item.value - mean) * (item.value - mean);
        variance /= (n - 1);
        return Math.sqrt(variance / k) * Math.sqrt(1.0 - (double) k / n);
    }

    /** Standard error of the estimated selective count, from the hypergeometric variance. */
    static double standardErrorOfCount(List<Item> stream, int k, IntPredicate predicate) {
        int n = stream.size();
        double p = exactCount(stream, predicate) / n;
        return n * Math.sqrt(p * (1 - p) / k) * Math.sqrt(1.0 - (double) k / n);
    }

    static List<Item> uniqueItems(List<Item> stream, int count) {
        List<Item> out = new ArrayList<>();
        for (int i = 0; i < count; i++) out.add(new Item(i, i, 1));
        return out;
    }
}

class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

public class Solution {

    static List<ReservoirSampling.Item> table(long seed, int rows) {
        SplitMix64 rng = new SplitMix64(seed);
        List<ReservoirSampling.Item> out = new ArrayList<>();
        for (int i = 0; i < rows; i++) out.add(new ReservoirSampling.Item(i, rng.nextInt(1000), 1));
        return out;
    }

    static List<ReservoirSampling.Item> skewed(int rows) {
        List<ReservoirSampling.Item> out = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            boolean heavy = i % 10 == 0;
            out.add(new ReservoirSampling.Item(i, heavy ? 100 : 1, heavy ? 10 : 1));
        }
        return out;
    }

    public static void main(String[] args) {
        List<ReservoirSampling.Item> stream = ReservoirSampling.uniqueItems(null, 40);
        SplitMix64 rng = new SplitMix64(31337L);

        List<ReservoirSampling.Item> reservoir = ReservoirSampling.uniformReservoir(stream, 8, rng);
        check(reservoir.size() == 8, "reservoir holds k items: " + reservoir.size());
        Set<Integer> indices = new TreeSet<>();
        boolean insideStream = true;
        for (ReservoirSampling.Item item : reservoir) {
            indices.add(item.index);
            if (item.index < 0 || item.index >= stream.size()) insideStream = false;
        }
        check(indices.size() == 8, "no duplicate rows in the reservoir");
        check(insideStream, "every sampled row comes from the stream");

        List<ReservoirSampling.Item> whole = ReservoirSampling.uniformReservoir(stream, 40, new SplitMix64(1L));
        check(whole.size() == 40, "k == n returns everything");
        List<ReservoirSampling.Item> over = ReservoirSampling.uniformReservoir(stream, 100, new SplitMix64(1L));
        check(over.size() == 40, "k > n still returns the whole stream (exactness)");

        List<ReservoirSampling.Item> repeat = ReservoirSampling.uniformReservoir(stream, 8, new SplitMix64(31337L));
        check(repeat.equals(reservoir), "the same seed reproduces the same reservoir");

        int trials = 400;
        int k = 8;
        int[] frequency = new int[stream.size()];
        Set<String> distinctSamples = new TreeSet<>();
        for (int trial = 0; trial < trials; trial++) {
            List<ReservoirSampling.Item> sample =
                    ReservoirSampling.uniformReservoir(stream, k, new SplitMix64(9000L + trial));
            StringBuilder key = new StringBuilder();
            for (ReservoirSampling.Item item : sample) {
                frequency[item.index]++;
                key.append(item.index).append(',');
            }
            distinctSamples.add(key.toString());
        }
        double expected = k / (double) stream.size();
        double tolerance = 4.0 * Math.sqrt(expected * (1 - expected) / trials);
        int outside = 0;
        double maxDeviation = 0;
        for (int f : frequency) {
            double observed = f / (double) trials;
            maxDeviation = Math.max(maxDeviation, Math.abs(observed - expected));
            if (Math.abs(observed - expected) > tolerance) outside++;
        }
        check(outside == 0, "inclusion frequency stays within four sigma of k/n: " + outside + " items outside");
        check(maxDeviation < 0.1, "maximum deviation " + maxDeviation);
        check(distinctSamples.size() > 200, "the sampler produces many different subsets: "
                + distinctSamples.size() + " of " + trials);
        check(!distinctSamples.contains("0,1,2,3,4,5,6,7,"), "it is not simply the first k rows");

        List<ReservoirSampling.Item> data = table(424242L, 4000);
        double exactMean = ReservoirSampling.exactAverage(data);
        double meanError = 0, countError = 0;
        for (int seed = 0; seed < 40; seed++) {
            List<ReservoirSampling.Item> small =
                    ReservoirSampling.uniformReservoir(data, 10, new SplitMix64(100L + seed));
            meanError += Math.abs(ReservoirSampling.estimatedAverage(data, small) - exactMean);
        }
        double smallMeanError = meanError / 40;
        meanError = 0;
        for (int seed = 0; seed < 40; seed++) {
            List<ReservoirSampling.Item> large =
                    ReservoirSampling.uniformReservoir(data, 200, new SplitMix64(100L + seed));
            meanError += Math.abs(ReservoirSampling.estimatedAverage(data, large) - exactMean);
        }
        double largeMeanError = meanError / 40;
        check(largeMeanError < smallMeanError / 3,
                "a larger reservoir is much more accurate: " + largeMeanError + " vs " + smallMeanError);
        check(largeMeanError < 3 * ReservoirSampling.standardErrorOfMean(data, 200),
                "the average estimate is within three standard errors: " + largeMeanError);

        IntPredicate selective = value -> value < 50;
        long exactSelective = (long) ReservoirSampling.exactCount(data, selective);
        for (int seed = 0; seed < 20; seed++) {
            List<ReservoirSampling.Item> sample =
                    ReservoirSampling.uniformReservoir(data, 300, new SplitMix64(700L + seed));
            double estimate = ReservoirSampling.estimatedCount(data, sample, selective);
            double error = Math.abs(estimate - exactSelective);
            countError += error;
            check(error <= 3 * ReservoirSampling.standardErrorOfCount(data, 300, selective) + 1,
                    "selective count within three standard errors (seed " + seed + "): " + error);
        }
        double selectiveError = ReservoirSampling.standardErrorOfCount(data, 300, selective);
        check(countError / 20 < 1.1 * selectiveError,
                "the mean absolute selective-count error stays within the estimator's standard error: "
                        + (countError / 20) + " vs " + selectiveError);

        List<ReservoirSampling.Item> skewed = skewed(1000);
        int heavyInUniform = 0, heavyInWeighted = 0;
        double uniformValueSum = 0, weightedValueSum = 0;
        for (int trial = 0; trial < 200; trial++) {
            List<ReservoirSampling.Item> uniform =
                    ReservoirSampling.uniformReservoir(skewed, 20, new SplitMix64(5000L + trial));
            List<ReservoirSampling.Item> weighted =
                    ReservoirSampling.weightedReservoir(skewed, 20, new SplitMix64(5000L + trial));
            check(weighted.size() == 20, "A-Res returns exactly k items");
            for (ReservoirSampling.Item item : uniform) {
                if (item.weight > 1) heavyInUniform++;
                uniformValueSum += item.value;
            }
            for (ReservoirSampling.Item item : weighted) {
                if (item.weight > 1) heavyInWeighted++;
                weightedValueSum += item.value;
            }
        }
        check(heavyInWeighted > 3 * heavyInUniform, "A-Res over-samples heavy rows: " + heavyInWeighted
                + " vs " + heavyInUniform);
        check(weightedValueSum > 3 * uniformValueSum, "weighted sampling finds the high-value rows: "
                + Math.round(weightedValueSum / 200) + " vs " + Math.round(uniformValueSum / 200));
        double exactSkewMean = ReservoirSampling.exactAverage(skewed);
        check(Math.abs(ReservoirSampling.estimatedAverage(skewed,
                        ReservoirSampling.uniformReservoir(skewed, 200, new SplitMix64(9L))) - exactSkewMean)
                        < 3 * ReservoirSampling.standardErrorOfMean(skewed, 200),
                "the skew estimator stays within three standard errors");
        check(ReservoirSampling.estimatedTotal(data,
                        ReservoirSampling.uniformReservoir(data, 400, new SplitMix64(11L))) > 0,
                "the total estimator is defined and positive");

        List<ReservoirSampling.Item> one = ReservoirSampling.uniformReservoir(stream, 1, new SplitMix64(77L));
        check(one.size() == 1, "a reservoir of one keeps a single row");
        double singleEstimate = ReservoirSampling.estimatedAverage(
                ReservoirSampling.uniqueItems(null, 10),
                ReservoirSampling.uniformReservoir(ReservoirSampling.uniqueItems(null, 10), 1, new SplitMix64(2L)));
        check(Double.isFinite(singleEstimate) && singleEstimate >= 0 && singleEstimate <= 9,
                "k = 1 still yields a usable estimate: " + singleEstimate);

        List<ReservoirSampling.Item> empty = List.of();
        check(ReservoirSampling.uniformReservoir(empty, 5, new SplitMix64(1L)).isEmpty(), "empty stream");
        check(Double.isNaN(ReservoirSampling.estimatedAverage(empty, empty)), "an empty stream has no average estimate");
        List<ReservoirSampling.Item> single = ReservoirSampling.uniqueItems(null, 1);
        check(ReservoirSampling.uniformReservoir(single, 5, new SplitMix64(1L)).size() == 1, "single item stream");
        check(ReservoirSampling.estimatedAverage(single,
                        ReservoirSampling.uniformReservoir(single, 1, new SplitMix64(1L))) == 0.0,
                "a one-row table estimates its own value");
        boolean rejected = false;
        try {
            ReservoirSampling.uniformReservoir(stream, -1, new SplitMix64(1L));
        } catch (IllegalArgumentException expected2) {
            rejected = true;
        }
        check(rejected, "a negative reservoir size is rejected");

        System.out.println("uniform reservoir: k=" + k + " n=" + stream.size()
                + " inclusion frequency within " + String.format("%.4f", maxDeviation) + " of " + expected
                + ", distinct subsets " + distinctSamples.size() + "/" + trials);
        System.out.println("mean absolute error k=10 " + smallMeanError + " vs k=200 " + largeMeanError
                + "; selective count error " + (countError / 20) + " of " + exactSelective);
        System.out.println("A-Res heavy rows " + heavyInWeighted + " vs uniform " + heavyInUniform
                + " over 200 trials of k=20");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
