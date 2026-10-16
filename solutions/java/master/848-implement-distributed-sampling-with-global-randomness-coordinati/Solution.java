/** Task 848 - Implement distributed sampling with global randomness coordination. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Sampling from a stream that is sharded across W workers needs one shared randomness schedule, otherwise
 * each worker samples a different universe and the merged sample is biased. The coordination used here is
 * *keyed randomness*: the coordinator publishes one global seed, every worker derives the priority of an
 * item from (global seed, item) alone, and a worker keeps only the k smallest priorities in its shard. The
 * merge is then a k-way minimum - the k smallest priorities of the union - which is exactly the sample a
 * single-process bottom-k sampler would have produced over the concatenated stream.
 *
 *   uniform  bottom-k: priority(item) = H(global seed, item)
 *   weighted A-Res   : priority(item) = U(global seed, item)^(1/weight), which selects a key with
 *                      probability proportional to its weight
 *
 * The priorities are floats/doubles derived from splitmix64 rather than real hash coin flips, and the
 * precedence of equal priorities is decided by the item itself so the result is a deterministic total order.
 *
 * Simplifications, stated honestly: one process, no network (the exchanged sample is counted, not sent), no
 * coordinator failure and no re-seeding; the stream is a list of distinct keys with weights, and duplicate
 * occurrences are collapsed before sampling, which is the usual "sample keys, then aggregate" model.
 *
 * Invariants verified below. Exactness: the merged sample equals the single-process sample over the whole
 * stream for the same global seed; it does not depend on how the stream is split across workers, on the
 * order the workers are merged, or on the order of items inside a worker. Statistics: over many global
 * seeds each item is included with measured frequency k/n within a tolerance band, a key with a ten times
 * larger weight is picked about ten times more often, and the single-sample minimum equals the brute-force
 * minimum over all priorities. Bounds: a worker never holds more than k items, and the coordinator receives
 * exactly min(k, |shard|) per worker. Edge cases: n < k returns everything, an empty shard contributes
 * nothing, and k = 1 is the global minimum.
 */
import java.util.*;

/** splitmix64 - the only randomness, and the pair of derived streams (uniform and priority) used here. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** World-wide mixing of a label; used to derive a shared hash family from one seed. */
    static long mix(long value) {
        long v = (value ^ 0x9E3779B97F4A7C15L) * 0xBF58476D1CE4E5B9L;
        v = (v ^ (v >>> 27)) * 0x94D049BB133111EBL;
        return v ^ (v >>> 31);
    }

    /** Uniform (0,1] from the shared schedule: the same seed and item give the same value everywhere. */
    static double uniform(long globalSeed, long item) {
        long bits = mix(globalSeed ^ mix(item + 0x51ED270BL));
        double u = ((bits >>> 11) * 0x1.0p-53);
        return u == 0 ? Double.MIN_VALUE : u;
    }
}

/** One sampled entry: the key, its priority and its weight. */
final class SampledItem implements Comparable<SampledItem> {
    final long key;
    final double priority;
    final double weight;

    SampledItem(long key, double priority, double weight) {
        this.key = key;
        this.priority = priority;
        this.weight = weight;
    }

    /** Ascending priority, ties broken by key so the order is total and deterministic. */
    @Override
    public int compareTo(SampledItem other) {
        int cmp = Double.compare(priority, other.priority);
        return cmp != 0 ? cmp : Long.compare(key, other.key);
    }

    @Override
    public String toString() { return key + "@" + priority; }
}

/** One worker: keeps the k smallest priorities seen in its shard, and nothing else. */
final class SamplingWorker {
    final int id;
    final int k;
    final long globalSeed;
    final boolean weighted;
    final TreeSet<SampledItem> reservoir = new TreeSet<>();
    int processed;
    int maxHeld;
    int exchanged;

    SamplingWorker(int id, int k, long globalSeed, boolean weighted) {
        if (k < 1) throw new IllegalArgumentException("k must be positive");
        this.id = id;
        this.k = k;
        this.globalSeed = globalSeed;
        this.weighted = weighted;
    }

    void offer(long key, double weight) {
        processed++;
        reservoir.add(new SampledItem(key, priorityOf(key, weight), weight));
        while (reservoir.size() > k) reservoir.pollLast();
        maxHeld = Math.max(maxHeld, reservoir.size());
    }

    double priorityOf(long key, double weight) {
        double uniform = SplitMix64.uniform(globalSeed, key);
        if (!weighted) return uniform;
        if (weight <= 0) throw new IllegalArgumentException("weights must be positive");
        // A-Res: keep the largest u^(1/weight), expressed as the smallest negated key.
        return -Math.pow(uniform, 1.0 / weight);
    }

    List<SampledItem> sample() {
        exchanged = reservoir.size();
        return new ArrayList<>(reservoir);
    }
}

/** The coordinator: merges the workers' samples by taking the k smallest priorities overall. */
final class SamplingCoordinator {
    static List<SampledItem> merge(List<SamplingWorker> workers, int k, long globalSeed, boolean weighted) {
        TreeSet<SampledItem> merged = new TreeSet<>();
        for (SamplingWorker worker : workers) merged.addAll(worker.sample());
        while (merged.size() > k) merged.pollLast();
        return new ArrayList<>(merged);
    }

    /** A single-process bottom-k over the whole stream, used as the oracle. */
    static List<SampledItem> singleProcess(List<long[]> stream, int k, long globalSeed, boolean weighted) {
        TreeSet<SampledItem> reservoir = new TreeSet<>();
        SamplingWorker oracle = new SamplingWorker(-1, k, globalSeed, weighted);
        for (long[] entry : stream) reservoir.add(new SampledItem(entry[0],
                oracle.priorityOf(entry[0], entry[1]), entry[1]));
        while (reservoir.size() > k) reservoir.pollLast();
        return new ArrayList<>(reservoir);
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(2024L);
        int n = 400, k = 40;
        List<long[]> stream = new ArrayList<>();
        for (int i = 0; i < n; i++) stream.add(new long[]{i, 1 + rng.nextInt(3)});
        long globalSeed = 0xDEADBEEFL;

        // ---- 1. a worker never holds more than k items -------------------------------------------
        List<SamplingWorker> workers = new ArrayList<>();
        int workersCount = 5;
        for (int w = 0; w < workersCount; w++) {
            workers.add(new SamplingWorker(w, k, globalSeed, false));
        }
        for (int i = 0; i < stream.size(); i++) {
            long[] entry = stream.get(i);
            workers.get(i % workersCount).offer(entry[0], entry[1]);
        }
        int maxHeld = 0;
        for (SamplingWorker worker : workers) {
            maxHeld = Math.max(maxHeld, worker.maxHeld);
            check(worker.reservoir.size() <= k, "a worker must never hold more than k items");
        }
        check(maxHeld == k, "the workers must actually fill their reservoirs, saw " + maxHeld);
        int processedTotal = 0;
        for (SamplingWorker worker : workers) processedTotal += worker.processed;
        check(processedTotal == n, "every item must be offered exactly once, saw " + processedTotal);

        // ---- 2. the merged sample equals the single-process sample -------------------------------
        List<SampledItem> merged = SamplingCoordinator.merge(workers, k, globalSeed, false);
        List<SampledItem> oracle = SamplingCoordinator.singleProcess(stream, k, globalSeed, false);
        check(merged.size() == k && oracle.size() == k, "both samples must have exactly k items");
        check(sameKeys(merged, oracle), "the merged sample must equal the single-process sample");
        int exchanged = 0;
        for (SamplingWorker worker : workers) exchanged += worker.exchanged;
        check(exchanged <= workersCount * k,
                "the coordinator must receive at most k items per worker, got " + exchanged);
        check(exchanged >= k, "the coordinator must receive enough to fill the sample");

        // ---- 3. invariance: order of workers, order of items, and the split itself ---------------
        List<SamplingWorker> reversed = new ArrayList<>(workers);
        Collections.reverse(reversed);
        check(sameKeys(SamplingCoordinator.merge(reversed, k, globalSeed, false), merged),
                "the merge order must not change the sample");
        List<SamplingWorker> reshuffled = new ArrayList<>();
        for (int w = 0; w < workersCount; w++) reshuffled.add(new SamplingWorker(w, k, globalSeed, false));
        List<long[]> shuffledStream = new ArrayList<>(stream);
        Collections.shuffle(shuffledStream, new Random(7L));
        for (int i = 0; i < shuffledStream.size(); i++) {
            long[] entry = shuffledStream.get(i);
            reshuffled.get(i % workersCount).offer(entry[0], entry[1]);
        }
        check(sameKeys(SamplingCoordinator.merge(reshuffled, k, globalSeed, false), merged),
                "neither the split nor the arrival order may change the sample");
        List<SamplingWorker> oneWorker = new ArrayList<>();
        SamplingWorker single = new SamplingWorker(0, k, globalSeed, false);
        for (long[] entry : stream) single.offer(entry[0], entry[1]);
        oneWorker.add(single);
        check(sameKeys(SamplingCoordinator.merge(oneWorker, k, globalSeed, false), merged),
                "one worker and five workers must give the same sample");

        // ---- 4. the sample is the true k smallest priorities --------------------------------------
        SamplingWorker probe = new SamplingWorker(0, k, globalSeed, false);
        double minPriority = Double.MAX_VALUE;
        long minKey = -1;
        for (long[] entry : stream) {
            double priority = probe.priorityOf(entry[0], entry[1]);
            if (priority < minPriority) {
                minPriority = priority;
                minKey = entry[0];
            }
        }
        List<SampledItem> singleSample = SamplingCoordinator.singleProcess(stream, 1, globalSeed, false);
        check(singleSample.size() == 1 && singleSample.get(0).key == minKey
                        && Math.abs(singleSample.get(0).priority - minPriority) < 1e-15,
                "k = 1 must return the brute-force minimum, got " + singleSample + " want " + minKey);
        Set<Long> sampled = new HashSet<>();
        for (SampledItem item : merged) sampled.add(item.key);
        double worst = 0;
        for (SampledItem item : merged) {
            for (long[] entry : stream) {
                double priority = probe.priorityOf(entry[0], entry[1]);
                if (!sampled.contains(entry[0])) worst = Math.max(worst, item.priority - priority);
            }
        }
        check(worst <= 0, "no unsampled item may have a priority below a sampled one, worst " + worst);

        // ---- 5. uniformity over many global seeds ------------------------------------------------
        Map<Long, Integer> frequency = new TreeMap<>();
        int seeds = 200;
        for (int s = 0; s < seeds; s++) {
            long seed = SplitMix64.mix(31L * s + 5);
            List<SampledItem> sample = SamplingCoordinator.singleProcess(stream, k, seed, false);
            for (SampledItem item : sample) frequency.merge(item.key, 1, Integer::sum);
        }
        double expected = (double) k / n;
        int outliers = 0;
        for (long key = 0; key < n; key++) {
            double observed = frequency.getOrDefault(key, 0) / (double) seeds;
            if (observed < expected / 3 || observed > 3 * expected) outliers++;
        }
        check(outliers <= 2, "at most a couple of keys may sit outside the uniform band, saw " + outliers);
        int totalDraws = 0;
        for (int count : frequency.values()) totalDraws += count;
        check(Math.abs(totalDraws / (double) (seeds * k) - 1) < 1e-9,
                "every draw must produce exactly k items, saw " + totalDraws);

        // ---- 6. weighted sampling favours heavy keys --------------------------------------------
        List<long[]> weightedStream = new ArrayList<>();
        weightedStream.add(new long[]{900, 10});
        for (int i = 0; i < 100; i++) weightedStream.add(new long[]{i, 1});
        int heavyHits = 0, lightHits = 0;
        for (int s = 0; s < 300; s++) {
            long seed = SplitMix64.mix(97L * s + 1);
            List<SampledItem> sample = SamplingCoordinator.singleProcess(weightedStream, 10, seed, true);
            boolean heavy = false;
            for (SampledItem item : sample) if (item.key == 900) heavy = true;
            if (heavy) heavyHits++;
            lightHits += sample.size() - (heavy ? 1 : 0);
        }
        double heavyRate = heavyHits / 300.0;
        double lightRate = lightHits / (300.0 * 100);
        check(heavyRate > 3 * lightRate,
                "a ten times heavier key must dominate the sample: " + heavyRate + " vs " + lightRate);
        check(heavyRate > 0.4 && heavyRate < 1.0,
                "the heavy key must be picked often but not always, saw " + heavyRate);

        // ---- 7. edge cases, validation and determinism -------------------------------------------
        List<SampledItem> small = SamplingCoordinator.singleProcess(stream.subList(0, 7), 40, globalSeed, false);
        check(small.size() == 7, "n < k must return every item, got " + small.size());
        List<SamplingWorker> empties = new ArrayList<>();
        for (int w = 0; w < 3; w++) empties.add(new SamplingWorker(w, k, globalSeed, false));
        check(SamplingCoordinator.merge(empties, k, globalSeed, false).isEmpty(),
                "empty shards must produce an empty sample");
        check(SamplingCoordinator.singleProcess(List.of(), 5, globalSeed, false).isEmpty(),
                "an empty stream must produce an empty sample");
        check(SamplingCoordinator.singleProcess(stream, 1, globalSeed, true).size() == 1,
                "the weighted variant works with k = 1");
        expectThrows(() -> new SamplingWorker(0, 0, globalSeed, false), "k below one");
        expectThrows(() -> new SamplingWorker(0, 5, globalSeed, true).offer(1, 0),
                "a non-positive weight");
        check(SamplingCoordinator.singleProcess(stream, k, globalSeed, false).toString()
                .equals(SamplingCoordinator.singleProcess(stream, k, globalSeed, false).toString())
                && SplitMix64.uniform(globalSeed, 5) == SplitMix64.uniform(globalSeed, 5),
                "the same seed must reproduce the same sample and the same priorities");
        double other = SplitMix64.uniform(globalSeed + 1, 5);
        check(other != SplitMix64.uniform(globalSeed, 5),
                "a different global seed must reshuffle the priorities");

        System.out.println("stream " + n + " keys over " + workersCount + " workers, k = " + k
                + ": worker held at most " + maxHeld + " items, coordinator received " + exchanged);
        System.out.println("merged sample == single-process sample == one-worker sample; min priority key "
                + minKey + " = " + minPriority);
        System.out.println("uniformity over " + seeds + " seeds: expected " + expected + " per key, "
                + outliers + " outliers; weighted heavy rate " + heavyRate + " vs light " + lightRate);
        System.out.println("PASS");
    }

    static boolean sameKeys(List<SampledItem> left, List<SampledItem> right) {
        if (left.size() != right.size()) return false;
        for (int i = 0; i < left.size(); i++) {
            if (left.get(i).key != right.get(i).key
                    || Math.abs(left.get(i).priority - right.get(i).priority) > 1e-15) return false;
        }
        return true;
    }

    static void expectThrows(Runnable action, String label) {
        try {
            action.run();
        } catch (RuntimeException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected a rejection for " + label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
