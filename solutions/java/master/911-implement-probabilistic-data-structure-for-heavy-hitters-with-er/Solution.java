/** Task 911 - Implement probabilistic data structure for heavy hitters with error guarantees and mergeability.
 *  (Tier: Master, CTCI Beyond - Advanced Data Structures & Algorithms)
 *
 * Misra-Gries keeps m = k-1 counters over a stream. A repeated item increments its counter; a new item with
 * a free counter gets one; otherwise every counter is decremented by one and the zeros are dropped. Two facts
 * follow from that bookkeeping and are what the checks actually test:
 *
 *   error bound   each item that still has a counter satisfies true(i) - est(i) <= D, where D is the number
 *                 of decrement rounds. Counting the updates shows sum(est) = N - (m+1)D, hence D <= N/k, so
 *                 every estimate is a lower bound within N/k of the truth (the sketch never overestimates);
 *   recall        an item with true(i) > N/k must still be tracked, because an item that loses its counter
 *                 was decremented at least D times and therefore cannot be that frequent.
 *
 * Mergeability comes for free from that argument: adding two sketches counter-wise and pruning to m entries
 * is the same operator as running one sketch over the concatenation, so the merged sketch inherits the same
 * N/k guarantee with its own N and D. That is what makes the structure usable in a distributed aggregator.
 *
 * Simplification: string keys, a 5000-item synthetic Zipf stream and an exact HashMap oracle. Invariants
 * verified: the counter count never exceeds m, estimates never overestimate and stay within N/k below the
 * truth, every true heavy hitter is recalled, merging commutes and is the identity on an empty sketch, short
 * streams are exact, and a lone counter is rejected as unusable.
 */
import java.util.*;

/** splitmix64 - the only source of randomness. */
final class Rng911 {
    private long state;

    Rng911(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** Misra-Gries sketch with m = k - 1 counters. */
final class MisraGries {
    final int counters;
    final NavigableMap<String, Integer> estimates = new TreeMap<>();
    long processed;
    long decrements;

    MisraGries(int k) {
        if (k < 2) throw new IllegalArgumentException("a Misra-Gries sketch needs k >= 2, got " + k);
        this.counters = k - 1;
    }

    void add(String item) {
        processed++;
        Integer current = estimates.get(item);
        if (current != null) {
            estimates.put(item, current + 1);
            return;
        }
        if (estimates.size() < counters) {
            estimates.put(item, 1);
            return;
        }
        decrements++;
        List<String> dropped = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : estimates.entrySet()) {
            int next = entry.getValue() - 1;
            entry.setValue(next);
            if (next == 0) dropped.add(entry.getKey());
        }
        for (String key : dropped) estimates.remove(key);
    }

    int estimate(String item) { return estimates.getOrDefault(item, 0); }

    int size() { return estimates.size(); }

    /** Upper bound on the overestimation error for every tracked item. */
    long errorBound() { return processed / (counters + 1L); }

    /** Counter-wise addition followed by pruning to m entries - the merge operator. */
    static MisraGries merge(MisraGries a, MisraGries b) {
        if (a.counters != b.counters) throw new IllegalArgumentException("merging sketches of different sizes");
        MisraGries merged = new MisraGries(a.counters + 1);
        merged.processed = a.processed + b.processed;
        merged.decrements = a.decrements + b.decrements;
        NavigableMap<String, Integer> combined = new TreeMap<>(a.estimates);
        for (Map.Entry<String, Integer> entry : b.estimates.entrySet()) {
            combined.merge(entry.getKey(), entry.getValue(), Integer::sum);
        }
        while (combined.size() > merged.counters) {
            // Drop the smallest counter; ties break on the key so the result stays deterministic.
            Map.Entry<String, Integer> smallest = null;
            for (Map.Entry<String, Integer> entry : combined.entrySet()) {
                if (smallest == null || entry.getValue() < smallest.getValue()) smallest = entry;
            }
            combined.remove(smallest.getKey());
        }
        merged.estimates.putAll(combined);
        return merged;
    }
}

/** Exact counts, used as the oracle. */
final class ExactCounts {
    final Map<String, Long> counts = new TreeMap<>();

    void add(String item) { counts.merge(item, 1L, Long::sum); }

    long get(String item) { return counts.getOrDefault(item, 0L); }

    long total() {
        long sum = 0;
        for (long value : counts.values()) sum += value;
        return sum;
    }
}

public class Solution {
    static final int STREAM = 5000;
    static final int K = 12;
    static final int SHARDS = 4;

    public static void main(String[] args) {
        List<String> stream = generate(0x911ABCDEL, STREAM);
        ExactCounts exact = new ExactCounts();
        MisraGries sketch = new MisraGries(K);
        int worstSize = 0;
        for (String item : stream) {
            exact.add(item);
            sketch.add(item);
            worstSize = Math.max(worstSize, sketch.size());
        }

        // ---- 1. space bound ---------------------------------------------------------------------------
        check(worstSize <= K - 1, "the sketch held " + worstSize + " counters, more than k-1");
        check(sketch.processed == STREAM, "the sketch must account for every update: " + sketch.processed);
        check(exact.total() == STREAM, "the oracle must account for every update");

        // ---- 2. the error bound, per item and from first principles -----------------------------------
        long bound = sketch.errorBound();
        check(bound > 0, "the error bound must be positive for a long stream");
        for (Map.Entry<String, Integer> entry : sketch.estimates.entrySet()) {
            long truth = exact.get(entry.getKey());
            check(entry.getValue() <= truth, "the estimate for " + entry.getKey() + " overestimates: "
                    + entry.getValue() + " > " + truth);
            check(truth - entry.getValue() <= bound, "the estimate for " + entry.getKey() + " is off by "
                    + (truth - entry.getValue()) + ", above the N/k bound " + bound);
        }
        check(sketch.decrements <= bound, "decrement rounds " + sketch.decrements + " must respect D <= N/k");
        long sumEstimates = 0;
        for (int value : sketch.estimates.values()) sumEstimates += value;
        long m = K - 1;
        check(sumEstimates == STREAM - (m + 1) * sketch.decrements,
                "the identity sum(est) = N - (m+1)D must hold exactly: " + sumEstimates + " vs "
                        + (STREAM - (m + 1) * sketch.decrements));

        // ---- 3. heavy hitter recall and the exact oracle ----------------------------------------------
        List<String> trueHeavy = new ArrayList<>();
        for (Map.Entry<String, Long> entry : exact.counts.entrySet()) {
            if (entry.getValue() > bound) trueHeavy.add(entry.getKey());
        }
        check(trueHeavy.size() > 2, "the fixture must contain real heavy hitters, got " + trueHeavy.size());
        check(trueHeavy.size() <= m, "with m counters at most m items can exceed N/k, got " + trueHeavy.size());
        check(exact.counts.size() > K - 1, "the distinct-key count must exceed the counter budget, got "
                + exact.counts.size());
        check(sketch.decrements > 0, "compression must actually force decrement rounds");
        for (String item : trueHeavy) {
            check(sketch.estimate(item) > 0, "heavy hitter " + item + " was not recalled");
        }
        String oracleTop = null;
        for (Map.Entry<String, Long> entry : exact.counts.entrySet()) {
            if (oracleTop == null || entry.getValue() > exact.get(oracleTop)) oracleTop = entry.getKey();
        }
        String sketchTop = sketch.estimates.firstEntry().getKey();
        for (String key : sketch.estimates.keySet()) {
            if (sketch.estimate(key) > sketch.estimate(sketchTop)) sketchTop = key;
        }
        check(sketchTop.equals(oracleTop), "the sketch must find the true most frequent item: " + sketchTop
                + " vs " + oracleTop);
        check(sketch.estimate(oracleTop) <= exact.get(oracleTop), "the top estimate must never overshoot");
        check(exact.get(oracleTop) - sketch.estimate(oracleTop) <= bound,
                "even the top estimate must stay within the bound");

        // ---- 4. mergeability --------------------------------------------------------------------------
        List<ExactCounts> shardTruth = new ArrayList<>();
        List<MisraGries> shards = new ArrayList<>();
        int perShard = STREAM / SHARDS;
        for (int s = 0; s < SHARDS; s++) {
            ExactCounts truth = new ExactCounts();
            MisraGries shard = new MisraGries(K);
            for (int i = 0; i < perShard; i++) {
                String item = stream.get(s * perShard + i);
                truth.add(item);
                shard.add(item);
            }
            shardTruth.add(truth);
            shards.add(shard);
        }
        MisraGries merged = shards.get(0);
        ExactCounts mergedTruth = shardTruth.get(0);
        for (int s = 1; s < SHARDS; s++) {
            merged = MisraGries.merge(merged, shards.get(s));
            for (Map.Entry<String, Long> entry : shardTruth.get(s).counts.entrySet()) {
                for (long i = 0; i < entry.getValue(); i++) mergedTruth.add(entry.getKey());
            }
        }
        check(merged.processed == perShard * SHARDS, "the merged sketch must see every update: "
                + merged.processed);
        check(merged.size() <= K - 1, "the merged sketch must respect the space bound: " + merged.size());
        long mergedBound = merged.processed / (K - 1L + 1);
        for (Map.Entry<String, Integer> entry : merged.estimates.entrySet()) {
            long truth = mergedTruth.get(entry.getKey());
            check(entry.getValue() <= truth, "merged estimate overestimates " + entry.getKey());
            check(truth - entry.getValue() <= mergedBound, "merged estimate for " + entry.getKey() + " is off by "
                    + (truth - entry.getValue()) + ", above " + mergedBound);
        }
        for (Map.Entry<String, Long> entry : mergedTruth.counts.entrySet()) {
            if (entry.getValue() > mergedBound) {
                check(merged.estimate(entry.getKey()) > 0, "merged sketch missed the heavy hitter "
                        + entry.getKey());
            }
        }
        MisraGries commutated = MisraGries.merge(shards.get(1), shards.get(0));
        MisraGries direct = MisraGries.merge(shards.get(0), shards.get(1));
        check(commutated.estimates.equals(direct.estimates), "merging two shards must commute");

        // ---- 5. merge identities and boundaries --------------------------------------------------------
        MisraGries empty = new MisraGries(K);
        check(MisraGries.merge(empty, empty).size() == 0, "merging two empty sketches stays empty");
        check(empty.processed == 0 && empty.estimates.isEmpty(), "a fresh sketch is empty");
        check(MisraGries.merge(empty, shards.get(0)).estimates.equals(shards.get(0).estimates),
                "an empty sketch must be a merge identity");
        check(merged.estimates.equals(MisraGries.merge(merged, empty).estimates),
                "merging with nothing changes nothing");
        MisraGries shortStream = new MisraGries(K);
        ExactCounts shortTruth = new ExactCounts();
        for (int i = 0; i < K - 2; i++) {
            shortStream.add("solo-" + i);
            shortTruth.add("solo-" + i);
        }
        check(shortStream.decrements == 0 && shortStream.size() == K - 2,
                "a stream shorter than the counter budget must be stored exactly");
        for (Map.Entry<String, Integer> entry : shortStream.estimates.entrySet()) {
            check(entry.getValue() == shortTruth.get(entry.getKey()),
                    "short streams must have zero error for " + entry.getKey());
        }
        MisraGries allDistinct = new MisraGries(K);
        for (int i = 0; i < 200; i++) allDistinct.add("unique-" + i);
        check(allDistinct.size() <= K - 1, "an all-distinct stream must stay within the counter budget");
        for (int value : allDistinct.estimates.values()) check(value == 1, "no counter can exceed 1 when "
                + "every item is new, got " + value);
        check(allDistinct.decrements == 200 / K, "with all-distinct items one decrement round fires every k "
                + "updates, got " + allDistinct.decrements);
        check(allDistinct.size() == 200 % K, "the leftover counters must match the stream length modulo k, got "
                + allDistinct.size());
        boolean rejected = false;
        try {
            new MisraGries(1);
        } catch (IllegalArgumentException ex) {
            rejected = true;
        }
        check(rejected, "k = 1 leaves no usable counter and must be rejected");

        // ---- 6. determinism -----------------------------------------------------------------------------
        MisraGries replay = new MisraGries(K);
        for (String item : stream) replay.add(item);
        check(replay.estimates.equals(sketch.estimates), "the same stream must give the same sketch");
        check(replay.decrements == sketch.decrements, "the decrement count must be reproducible");
        check(generate(0x911ABCDEL, 100).equals(generate(0x911ABCDEL, 100)),
                "stream generation must be reproducible");

        System.out.println("stream " + STREAM + " items, k=" + K + " (" + (K - 1) + " counters), decrement "
                + "rounds " + sketch.decrements + ", error bound N/k = " + bound);
        System.out.println("heavy hitters recalled " + trueHeavy.size() + "/" + trueHeavy.size() + ", top item '"
                + oracleTop + "' true " + exact.get(oracleTop) + " est " + sketch.estimate(oracleTop));
        System.out.println("merged " + SHARDS + " shards: " + merged.size() + " counters, bound "
                + mergedBound + ", order independent " + commutated.estimates.equals(direct.estimates));
        System.out.println("PASS");
    }

    /** Zipf-ish stream: rank r gets weight 1/(r+1) over 20 distinct keys, so 11 counters must compress. */
    static List<String> generate(long seed, int length) {
        Rng911 rng = new Rng911(seed);
        int keys = 20;
        double[] cumulative = new double[keys];
        double running = 0;
        for (int r = 0; r < keys; r++) {
            running += 1.0 / (r + 1);
            cumulative[r] = running;
        }
        List<String> stream = new ArrayList<>(length);
        for (int i = 0; i < length; i++) {
            double u = rng.nextDouble() * running;
            int rank = 0;
            while (rank < keys - 1 && cumulative[rank] < u) rank++;
            stream.add("key-" + rank);
        }
        return stream;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
