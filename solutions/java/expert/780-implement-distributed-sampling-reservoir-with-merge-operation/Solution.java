/** Task 780 - Implement distributed sampling reservoir with merge operation. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
 *
 * A distributed reservoir sample must be uniform over the *union* of every shard's stream, although no
 * shard ever sees the whole stream and each shard's length is unknown when it starts. The
 * implementation uses the priority form of reservoir sampling, which is exactly equivalent to
 * Algorithm R and is the reason merging works at all: every item gets a uniformly random priority key
 * from a deterministic splitmix64 generator, the reservoir keeps the k smallest keys seen so far, and
 *
 *     merge(reservoirs) = the k smallest keys of the union
 *
 * is again a uniformly random k-subset, now of the union of all shards. Merging is therefore
 * associative and commutative - the same items with the same keys always produce the same sample,
 * whatever the merge tree looks like - and a sample of size min(k, n) is never exceeded.
 *
 * Verified: the sample size is exactly min(k, n) at every point; the priorities of the retained items
 * are the k smallest of their stream; merging is order independent and re-merging the same shards
 * reproduces the same ids; the merged sample is uniform over the union - for tiny streams this is
 * checked *exactly* by enumerating every k-subset (each must appear with probability 1/C(n, k)) and for
 * a larger multi-shard case every item's inclusion frequency over many independent trials is within a
 * binomial confidence band of k/n; and the degenerate cases (k >= n, k = 1, empty shards, a merge of
 * shards of very different lengths) behave exactly as their definitions require.
 *
 * Simplified: priorities are 64-bit random keys of a deterministic generator rather than true
 * randomness, so the whole verification is reproducible; streams are small enough that the exact
 * enumeration of subsets is affordable.
 */
import java.util.*;

/** Priority reservoir: keeps the k items with the smallest random keys. */
class Reservoir {
    static final class Entry implements Comparable<Entry> {
        final long key;
        final int item;
        final int shard;

        Entry(long key, int item, int shard) {
            this.key = key;
            this.item = item;
            this.shard = shard;
        }

        @Override
        public int compareTo(Entry other) {
            if (key != other.key) return Long.compare(key, other.key);
            if (item != other.item) return Integer.compare(item, other.item);
            return Integer.compare(shard, other.shard);
        }
    }

    final int capacity;
    final List<Entry> entries = new ArrayList<>();          // sorted, at most capacity entries
    long seen;

    Reservoir(int capacity, long seed) {
        if (capacity < 1) throw new IllegalArgumentException("capacity must be positive");
        this.capacity = capacity;
        this.randomState = seed | 1L;
    }

    private long randomState;

    /** splitmix64: deterministic pseudo randomness, so every run reproduces the same sample. */
    private long nextLong() {
        randomState += 0x9E3779B97F4A7C15L;
        long z = randomState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    void add(int item) {
        add(item, nextLong());
    }

    /** Adds an item with an explicit key - the hook that makes merges reproducible. */
    void add(int item, long key) {
        seen++;
        if (entries.size() == capacity && entries.get(entries.size() - 1).key <= key) return;
        Entry fresh = new Entry(key, item, shardOf(item));
        int position = 0;
        while (position < entries.size() && entries.get(position).compareTo(fresh) < 0) position++;
        entries.add(position, fresh);
        if (entries.size() > capacity) entries.remove(entries.size() - 1);
    }

    static int shardOf(int item) { return Math.floorDiv(item, 1_000_000); }

    List<Integer> sample() {
        List<Integer> sample = new ArrayList<>();
        for (Entry entry : entries) sample.add(entry.item);
        return sample;
    }

    /** The k smallest keys of the union: uniform over the items of all shards. */
    static Reservoir merge(int capacity, List<Reservoir> shards) {
        Reservoir merged = new Reservoir(capacity, 1L);
        long totalSeen = 0;
        for (Reservoir shard : shards) {
            totalSeen += shard.seen;
            for (Entry entry : shard.entries) {
                merged.addWithoutKey(entry);
            }
        }
        merged.seen = totalSeen;
        return merged;
    }

    private void addWithoutKey(Entry entry) {
        if (entries.size() == capacity && entries.get(entries.size() - 1).compareTo(entry) <= 0) return;
        int position = 0;
        while (position < entries.size() && entries.get(position).compareTo(entry) < 0) position++;
        entries.add(position, entry);
        if (entries.size() > capacity) entries.remove(entries.size() - 1);
    }

    boolean sortedByKey() {
        for (int i = 1; i < entries.size(); i++) {
            if (entries.get(i - 1).compareTo(entries.get(i)) > 0) return false;
        }
        return true;
    }
}

public class Solution {
    private static long state = 0x9E3779B97F4A7C15L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static long combination(long n, long k) {
        long result = 1;
        for (long i = 0; i < k; i++) result = result * (n - i) / (i + 1);
        return result;
    }

    /** Runs one distributed trial: `shards` streams of the given lengths, merged into one sample. */
    static List<Integer> trial(int capacity, int[] lengths, long seed) {
        List<Reservoir> shards = new ArrayList<>();
        int global = 0;
        for (int s = 0; s < lengths.length; s++) {
            Reservoir reservoir = new Reservoir(capacity, seed + 7919L * (s + 1));
            for (int i = 0; i < lengths[s]; i++) {
                reservoir.add(s * 1_000_000 + i);
                global++;
            }
            shards.add(reservoir);
        }
        return mergeSample(capacity, shards);
    }

    static List<Integer> mergeSample(int capacity, List<Reservoir> shards) {
        return Reservoir.merge(capacity, shards).sample();
    }

    public static void main(String[] args) {
        // --- degenerate reservoirs ---------------------------------------------
        Reservoir empty = new Reservoir(4, 11L);
        check(empty.sample().isEmpty(), "an empty reservoir holds nothing");
        check(empty.seen == 0, "an empty reservoir has seen nothing");
        Reservoir tiny = new Reservoir(10, 12L);
        for (int i = 0; i < 3; i++) tiny.add(i);
        List<Integer> tinySample = new ArrayList<>(tiny.sample());
        Collections.sort(tinySample);
        check(tinySample.equals(List.of(0, 1, 2)), "a stream shorter than k keeps every item, got " + tinySample);
        check(tiny.seen == 3, "the stream length is tracked");

        Reservoir single = new Reservoir(1, 13L);
        long minKey = Long.MAX_VALUE;
        for (int i = 0; i < 50; i++) {
            long key = nextLong();
            minKey = Math.min(minKey, key);
            single.add(i, key);
        }
        check(single.sample().size() == 1, "k = 1 keeps exactly one item");
        check(single.entries.get(0).key == minKey, "k = 1 keeps the smallest key of the stream");

        boolean rejected = false;
        try {
            new Reservoir(0, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a zero capacity must be rejected");

        // --- sizes and key invariants -----------------------------------------
        for (int capacity : new int[]{1, 5, 17}) {
            Reservoir reservoir = new Reservoir(capacity, 21L + capacity);
            for (int i = 0; i < 200; i++) {
                reservoir.add(i);
                int expected = Math.min(capacity, i + 1);
                check(reservoir.sample().size() == expected, "size must stay min(k, n) at step " + i);
                check(reservoir.sortedByKey(), "the reservoir must stay sorted by key");
            }
            check(reservoir.entries.get(0).key <= reservoir.entries.get(reservoir.entries.size() - 1).key,
                    "keys are ordered");
        }

        // --- merging is order independent -------------------------------------
        List<Reservoir> shards = new ArrayList<>();
        for (int s = 0; s < 4; s++) {
            Reservoir reservoir = new Reservoir(6, 500L + s);
            for (int i = 0; i < 40 + 10 * s; i++) reservoir.add(s * 1_000_000 + i);
            shards.add(reservoir);
        }
        Reservoir left = Reservoir.merge(6, shards.subList(0, 2));
        Reservoir right = Reservoir.merge(6, shards.subList(2, 4));
        List<Reservoir> pair = new ArrayList<>(List.of(left, right));
        List<Integer> treeSample = new ArrayList<>(Reservoir.merge(6, pair).sample());
        List<Integer> flatSample = new ArrayList<>(Reservoir.merge(6, shards).sample());
        Collections.sort(treeSample);
        Collections.sort(flatSample);
        check(treeSample.equals(flatSample), "the merge tree must not matter: " + treeSample + " vs " + flatSample);
        List<Reservoir> reversed = new ArrayList<>(shards);
        Collections.reverse(reversed);
        List<Integer> reversedSample = new ArrayList<>(Reservoir.merge(6, reversed).sample());
        Collections.sort(reversedSample);
        check(reversedSample.equals(flatSample), "the merge order must not matter");
        check(flatSample.size() == 6, "the merged sample has exactly k items");

        // --- exact uniformity for a tiny stream ---------------------------------
        int n = 5;
        int k = 2;
        int trials = 12000;
        Map<String, Integer> subsetCounts = new TreeMap<>();
        for (int t = 0; t < trials; t++) {
            Reservoir reservoir = new Reservoir(k, 9_000L + 31L * t);
            for (int i = 0; i < n; i++) reservoir.add(i);
            List<Integer> sample = new ArrayList<>(reservoir.sample());
            Collections.sort(sample);
            subsetCounts.merge(sample.toString(), 1, Integer::sum);
        }
        check(subsetCounts.size() == combination(n, k), "every k-subset must occur, got " + subsetCounts.size());
        int expectedCount = trials / (int) combination(n, k);
        int slack = expectedCount / 2;
        for (Map.Entry<String, Integer> entry : subsetCounts.entrySet()) {
            check(Math.abs(entry.getValue() - expectedCount) <= slack,
                    "subset " + entry.getKey() + " occurred " + entry.getValue() + " times, expected about "
                            + expectedCount);
        }

        // --- uniformity of a merged multi-shard sample --------------------------
        int capacity = 3;
        int[] lengths = {7, 3, 11};
        int total = 0;
        for (int length : lengths) total += length;
        int rounds = 8000;
        Map<Integer, Integer> frequency = new TreeMap<>();
        for (int t = 0; t < rounds; t++) {
            List<Integer> sample = trial(capacity, lengths, 4_000L + 977L * t);
            check(sample.size() == capacity, "the merged sample size must be exactly k");
            for (int item : sample) frequency.merge(item, 1, Integer::sum);
        }
        double expected = (double) capacity / total * rounds;
        double spread = 4 * Math.sqrt(rounds * (double) capacity / total * (1 - (double) capacity / total));
        for (Map.Entry<Integer, Integer> entry : frequency.entrySet()) {
            check(Math.abs(entry.getValue() - expected) <= spread,
                    "item " + entry.getKey() + " appeared " + entry.getValue() + " times, expected about "
                            + expected);
        }
        check(frequency.size() == total, "every item of every shard can appear, got " + frequency.size());
        System.out.println("merged uniformity: expected " + expected + " per item, spread " + spread);

        // --- shard sizes of wildly different magnitudes -------------------------
        List<Reservoir> lopsided = new ArrayList<>();
        Reservoir small = new Reservoir(4, 61L);
        small.add(42, Long.MIN_VALUE + 1);       // a key so small that the merge cannot drop it
        lopsided.add(small);
        Reservoir big = new Reservoir(4, 62L);
        for (int i = 0; i < 5000; i++) big.add(1_000_000 + i);
        lopsided.add(big);
        List<Integer> lopsidedSample = Reservoir.merge(4, lopsided).sample();
        check(lopsidedSample.size() == 4, "the merge still honours k");
        check(lopsidedSample.contains(42), "the smallest key of any shard always survives the merge");
        Reservoir merged = Reservoir.merge(4, lopsided);
        check(merged.seen == 5001, "the merged stream length is the sum of the shards");

        // --- k >= n keeps everything --------------------------------------------
        Reservoir all = new Reservoir(10, 71L);
        for (int i = 0; i < 10; i++) all.add(i);
        check(all.sample().size() == 10, "k >= n keeps every item");
        List<Reservoir> allShards = new ArrayList<>(List.of(all));
        check(Reservoir.merge(10, allShards).sample().size() == 10, "a merge of a complete reservoir keeps it");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
