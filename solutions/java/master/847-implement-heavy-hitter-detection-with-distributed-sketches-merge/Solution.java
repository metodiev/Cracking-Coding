/** Task 847 - Implement heavy-hitter detection with distributed sketches mergeable. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A heavy hitter is a key whose frequency passes a threshold phi * N. On a stream that is sharded across W
 * workers, the classic streaming answer is a mergeable sketch:
 *
 *   Count-Min sketch  rows x cols counters over a hash family shared by every shard (which is what makes
 *                     the sketches mergeable); item x updates counter (row, h_row(x)) in every row, and the
 *                     estimate is the minimum over the rows, so a counter over-estimates but never
 *                     under-estimates. Two sketches merge by adding counters cell by cell, so the merge is
 *                     associative and commutative and the merged sketch equals the sketch of the
 *                     concatenated stream exactly.
 *   Space-Saving     a bounded candidate table of the most promising keys; merged by taking the union of
 *                     the candidates and re-estimating each with the merged Count-Min sketch.
 *
 * The detector reports every key whose merged estimate passes phi * N; because the estimate never
 * under-estimates, no true heavy hitter can be missed, and the over-estimate is bounded by the additive
 * Count-Min error N * e / cols (measured, not assumed).
 *
 * Simplifications, stated honestly: a single process with in-memory shards, no network and no coordinator;
 * the two workers' sketches are merged by a driver. The hashes are splitmix64 mixes, not pairwise
 * independent families, so the error bound is verified empirically rather than proven. Space-Saving uses
 * the union-then-re-estimate merge instead of the streaming decrement variant.
 *
 * Invariants verified below. Mergeability: merging in any order gives byte-identical sketches and equal
 * estimates, the identity (empty) shard is neutral, and the merged sketch equals the sketch built over the
 * concatenated stream. Recall and error: every key above phi * N is reported, every estimate is at least
 * the true count, the additive error never exceeds N * e / cols, and the reported set contains no key whose
 * true count is far below the threshold. Cost: memory (rows x cols counters) and the per-item update cost
 * (rows hashes) are counted.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, and the hash family used per row. */
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

    static long mix(long value) {
        long v = (value ^ 0x9E3779B97F4A7C15L) * 0xBF58476D1CE4E5B9L;
        v = (v ^ (v >>> 27)) * 0x94D049BB133111EBL;
        return v ^ (v >>> 31);
    }
}

/** Count-Min sketch: mergeable by cell-wise addition. */
final class CountMinSketch {
    final int rows;
    final int cols;
    final long[] seeds;
    final long[][] counters;
    long total;

    CountMinSketch(int rows, int cols, long seed) {
        if (rows < 1 || cols < 1) throw new IllegalArgumentException("sketch needs at least one cell");
        this.rows = rows;
        this.cols = cols;
        this.seeds = new long[rows];
        for (int r = 0; r < rows; r++) seeds[r] = SplitMix64.mix(seed + r * 0x9E3779B9L);
        this.counters = new long[rows][cols];
    }

    void add(long item) {
        for (int r = 0; r < rows; r++) {
            int column = (int) Math.floorMod(SplitMix64.mix(item ^ seeds[r]), cols);
            counters[r][column]++;
        }
        total++;
    }

    long estimate(long item) {
        long best = Long.MAX_VALUE;
        for (int r = 0; r < rows; r++) {
            int column = (int) Math.floorMod(SplitMix64.mix(item ^ seeds[r]), cols);
            best = Math.min(best, counters[r][column]);
        }
        return best;
    }

    /** Merges another sketch of the same shape into this one; returns this. */
    CountMinSketch merge(CountMinSketch other) {
        if (other.rows != rows || other.cols != cols) throw new IllegalArgumentException("shape mismatch");
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) counters[r][c] += other.counters[r][c];
        }
        total += other.total;
        return this;
    }

    CountMinSketch copy() {
        CountMinSketch clone = new CountMinSketch(rows, cols, 0);
        System.arraycopy(seeds, 0, clone.seeds, 0, rows);
        for (int r = 0; r < rows; r++) {
            System.arraycopy(counters[r], 0, clone.counters[r], 0, cols);
        }
        clone.total = total;
        return clone;
    }

    int memoryCells() { return rows * cols; }
}

/** Space-Saving style candidate table, merged by union plus re-estimation. */
final class CandidateTable {
    final int capacity;
    final Map<Long, Long> counters = new TreeMap<>();

    CandidateTable(int capacity) {
        if (capacity < 1) throw new IllegalArgumentException("capacity must be positive");
        this.capacity = capacity;
    }

    void add(long item) {
        counters.merge(item, 1L, Long::sum);
        if (counters.size() > capacity) evictSmallest();
    }

    private void evictSmallest() {
        long worstKey = 0;
        long worstValue = Long.MAX_VALUE;
        for (Map.Entry<Long, Long> entry : counters.entrySet()) {
            if (entry.getValue() < worstValue) {
                worstValue = entry.getValue();
                worstKey = entry.getKey();
            }
        }
        counters.remove(worstKey);
    }

    /** Union with another table, then keep the `capacity` keys with the best merged estimate. */
    void mergeInto(CandidateTable other, CountMinSketch merged) {
        Set<Long> unions = new TreeSet<>(counters.keySet());
        unions.addAll(other.counters.keySet());
        List<long[]> ranked = new ArrayList<>();
        for (long key : unions) ranked.add(new long[]{key, merged.estimate(key)});
        ranked.sort((a, b) -> a[1] != b[1] ? Long.compare(b[1], a[1]) : Long.compare(a[0], b[0]));
        counters.clear();
        for (int i = 0; i < ranked.size() && i < capacity; i++) {
            counters.put(ranked.get(i)[0], ranked.get(i)[1]);
        }
    }
}

/** One worker's local state: a sketch plus a candidate table. */
final class Shard {
    final int id;
    final CountMinSketch sketch;
    final CandidateTable candidates;

    Shard(int id, int rows, int cols, int capacity, long seed) {
        this.id = id;
        this.sketch = new CountMinSketch(rows, cols, seed);
        this.candidates = new CandidateTable(capacity);
    }

    void process(int[] stream) {
        for (int item : stream) {
            sketch.add(item);
            candidates.add(item);
        }
    }
}

/** The driver: merges every shard and reports the heavy hitters. */
final class HeavyHitterDetector {
    final int rows;
    final int cols;
    final int capacity;
    final double phi;
    long mergedCells;

    HeavyHitterDetector(int rows, int cols, int capacity, double phi) {
        if (phi <= 0 || phi >= 1) throw new IllegalArgumentException("phi must be in (0,1)");
        this.rows = rows;
        this.cols = cols;
        this.capacity = capacity;
        this.phi = phi;
    }

    /** Merges the shards in the given order and returns the heavy hitters sorted by estimate. */
    List<long[]> detect(List<Shard> shards, int[] order) {
        CountMinSketch merged = new CountMinSketch(rows, cols, 900);
        for (int index : order) merged.merge(shards.get(index).sketch.copy());
        mergedCells = merged.memoryCells();
        CandidateTable table = new CandidateTable(capacity);
        for (int index : order) table.mergeInto(shards.get(index).candidates, merged);
        long threshold = (long) Math.ceil(phi * merged.total);
        List<long[]> hits = new ArrayList<>();
        for (Map.Entry<Long, Long> entry : table.counters.entrySet()) {
            if (entry.getValue() >= threshold) hits.add(new long[]{entry.getKey(), entry.getValue()});
        }
        hits.sort((a, b) -> a[1] != b[1] ? Long.compare(b[1], a[1]) : Long.compare(a[0], b[0]));
        return hits;
    }

    /** The merged sketch itself, for the mergeability checks. */
    CountMinSketch mergedSketch(List<Shard> shards, int[] order) {
        CountMinSketch merged = new CountMinSketch(rows, cols, 900);
        for (int index : order) merged.merge(shards.get(index).sketch.copy());
        return merged;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(777L);
        int shardsCount = 5;
        int[] heavy = {1000, 2000, 3000};
        long[] heavyCount = new long[heavy.length];
        Map<Long, Long> truth = new TreeMap<>();
        List<List<Integer>> streams = new ArrayList<>();
        for (int w = 0; w < shardsCount; w++) streams.add(new ArrayList<>());
        List<Integer> concatenated = new ArrayList<>();
        for (int i = 0; i < 3000; i++) {
            int item;
            if (i % 4 == 0) item = heavy[(i / 4) % heavy.length];
            else item = 100 + rng.nextInt(60);
            streams.get(i % shardsCount).add(item);
            truth.merge((long) item, 1L, Long::sum);
            concatenated.add(item);
        }
        long total = concatenated.size();
        for (int i = 0; i < heavy.length; i++) heavyCount[i] = truth.getOrDefault((long) heavy[i], 0L);

        int rows = 5, cols = 512, capacity = 32;
        double phi = 0.05;
        List<Shard> shards = new ArrayList<>();
        for (int w = 0; w < shardsCount; w++) {
            Shard shard = new Shard(w, rows, cols, capacity, 900);
            shard.process(streams.get(w).stream().mapToInt(Integer::intValue).toArray());
            shards.add(shard);
        }
        long exactTotal = 0;
        for (Shard shard : shards) exactTotal += shard.sketch.total;
        check(exactTotal == total, "the shards must cover the whole stream: " + exactTotal + " vs " + total);

        HeavyHitterDetector detector = new HeavyHitterDetector(rows, cols, capacity, phi);
        int[] naturalOrder = {0, 1, 2, 3, 4};
        List<long[]> hits = detector.detect(shards, naturalOrder);

        // ---- 1. no false negatives among the true heavy hitters ----------------------------------
        check(heavyCount[0] > phi * total && heavyCount[1] > phi * total && heavyCount[2] > phi * total,
                "the planted heavy hitters must pass the threshold");
        for (int i = 0; i < heavy.length; i++) {
            boolean found = false;
            for (long[] hit : hits) if (hit[0] == heavy[i]) found = true;
            check(found, "heavy hitter " + heavy[i] + " (count " + heavyCount[i] + ") must be reported");
        }
        check(hits.size() > 0, "the detector must report something");

        // ---- 2. the estimate never under-estimates and the error respects the bound --------------
        double epsilon = Math.E / cols;
        long worstError = 0;
        for (long[] hit : hits) {
            long real = truth.getOrDefault(hit[0], 0L);
            check(hit[1] >= real, "the estimate must never under-estimate: " + hit[0] + " " + hit[1]
                    + " < " + real);
            worstError = Math.max(worstError, hit[1] - real);
        }
        check(worstError <= total * epsilon,
                "the additive error must respect N * e / cols = " + (long) (total * epsilon) + ", saw "
                        + worstError);
        CountMinSketch all = detector.mergedSketch(shards, naturalOrder);
        for (Map.Entry<Long, Long> entry : truth.entrySet()) {
            long value = all.estimate(entry.getKey());
            check(value >= entry.getValue(), "every key must satisfy estimate >= truth for " + entry.getKey());
            check(value - entry.getValue() <= total * epsilon,
                    "every key's error must respect the bound, " + entry.getKey() + " error "
                            + (value - entry.getValue()));
        }

        // ---- 3. merge order invariance and the neutral element -----------------------------------
        int[] reversed = {4, 3, 2, 1, 0};
        int[] shuffled = {2, 0, 4, 1, 3};
        List<long[]> hitsReversed = detector.detect(shards, reversed);
        List<long[]> hitsShuffled = detector.detect(shards, shuffled);
        check(sameHits(hits, hitsReversed) && sameHits(hits, hitsShuffled),
                "the merge order must not change the reported heavy hitters");
        CountMinSketch forward = detector.mergedSketch(shards, naturalOrder);
        CountMinSketch backward = detector.mergedSketch(shards, reversed);
        boolean identicalCells = true;
        for (int r = 0; r < rows; r++) {
            if (!Arrays.equals(forward.counters[r], backward.counters[r])) identicalCells = false;
        }
        check(identicalCells, "merges in different orders must give identical counters");
        CountMinSketch empty = new CountMinSketch(rows, cols, 0);
        CountMinSketch mergedWithEmpty = forward.copy().merge(empty);
        check(mergedWithEmpty.total == forward.total
                        && mergedWithEmpty.estimate(heavy[0]) == forward.estimate(heavy[0]),
                "an empty shard must be a neutral element");

        // ---- 4. a true single-pass sketch over the whole stream ----------------------------------
        Shard monolithic = new Shard(0, rows, cols, 4096, 900);
        monolithic.process(concatenated.stream().mapToInt(Integer::intValue).toArray());
        boolean sameAsMonolithic = true;
        for (int r = 0; r < rows; r++) {
            if (!Arrays.equals(forward.counters[r], monolithic.sketch.counters[r])) {
                sameAsMonolithic = false;
            }
        }
        check(sameAsMonolithic,
                "the merged sketch must be cell-for-cell identical to a single-pass sketch of the whole "
                        + "stream");
        List<Shard> oneShard = new ArrayList<>();
        oneShard.add(monolithic);
        List<long[]> monolithicHits = detector.detect(oneShard, new int[]{0});
        for (long[] hit : hits) {
            boolean found = false;
            for (long[] other : monolithicHits) if (other[0] == hit[0]) found = true;
            check(found, "a single-shard run must find every hitter the distributed run found");
        }
        check(monolithicHits.size() <= capacity,
                "the candidate table bounds the reported set, saw " + monolithicHits.size());

        // ---- 5. counts, memory, validation and determinism ---------------------------------------
        check(detector.mergedCells == rows * cols,
                "the merged sketch must use rows * cols cells, saw " + detector.mergedCells);
        check(capacity < 60, "the candidate table must be much smaller than the key space");
        long reported = 0;
        for (long[] hit : hits) reported += hit[1];
        check(reported <= 2 * total, "the reported mass must not explode: " + reported);
        List<long[]> again = detector.detect(shards, naturalOrder);
        check(sameHits(hits, again), "two identical runs must report the same hitters");
        expectThrows(() -> new HeavyHitterDetector(rows, cols, capacity, 0), "phi below zero");
        expectThrows(() -> new HeavyHitterDetector(rows, cols, capacity, 1), "phi at one");
        expectThrows(() -> new CountMinSketch(0, 8, 1), "zero rows");
        expectThrows(() -> new CandidateTable(0), "zero capacity");
        List<Shard> noShards = List.of();
        check(detector.detect(noShards, new int[0]).isEmpty(),
                "no shards means no heavy hitters");
        Shard emptyShard = new Shard(0, rows, cols, capacity, 900);
        emptyShard.process(new int[0]);
        check(emptyShard.sketch.total == 0 && emptyShard.candidates.counters.isEmpty(),
                "an empty stream leaves an empty sketch");
        List<Shard> tinyShards = new ArrayList<>();
        Shard tiny = new Shard(0, rows, cols, capacity, 900);
        tiny.process(new int[]{7, 7, 7, 9});
        tinyShards.add(tiny);
        List<long[]> tinyHits = detector.detect(tinyShards, new int[]{0});
        check(tinyHits.size() == 2 && tinyHits.get(0)[0] == 7 && tinyHits.get(0)[1] == 3,
                "a tiny stream must report both keys with their exact counts, got "
                        + tinyHits.size());
        check(truth.getOrDefault(7L, 0L) == 0, "the tiny stream is independent of the big one");

        System.out.println("stream " + total + " items over " + shardsCount + " shards; heavy hitters "
                + hits.size() + " reported, planted " + Arrays.toString(heavy) + " counts "
                + Arrays.toString(heavyCount));
        System.out.println("sketch " + rows + "x" + cols + " = " + (rows * cols) + " cells, epsilon "
                + epsilon + ", worst additive error " + worstError);
        System.out.println("merge order invariance verified; neutral element verified; single-shard run "
                + "found all " + monolithicHits.size() + " hitters");
        System.out.println("PASS");
    }

    static boolean sameHits(List<long[]> left, List<long[]> right) {
        if (left.size() != right.size()) return false;
        for (int i = 0; i < left.size(); i++) {
            if (left.get(i)[0] != right.get(i)[0] || left.get(i)[1] != right.get(i)[1]) return false;
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
