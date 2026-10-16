/** Task 916 - Implement large-scale streaming joins with approximate state and bounded memory.
 *  (Tier: Master, Ch 9 - System Design and Scalability)
 *
 * A streaming join cannot keep every event: the state would grow without bound. This implementation keeps an
 * LRU-keyed join state of fixed capacity, protects the keys that a Count-Min sketch estimates to be frequent
 * (hot keys are the ones worth remembering, because they produce most of the output), and accepts that keys
 * evicted from the state lose their matches. Three properties make that trade defensible, and each is checked:
 *
 *   memory bound   the number of tracked keys never exceeds the configured capacity;
 *   hot recall     keys the sketch flags as heavy are pinned and therefore never lose a match, so recall is
 *                  high for exactly the keys that dominate the output;
 *   estimate bound the Count-Min estimate never undercounts and stays close above the true frequency, so the
 *                  pinning decision is based on a quantity with a known error.
 *
 * Against those approximations the file also runs an exact reference: with capacity above the number of
 * distinct keys the join output must equal the quadratic nested-loop oracle pair for pair.
 *
 * Simplification: string keys, sequence-numbered events and a sliding window measured in sequence units.
 * Invariants verified: memory never exceeds capacity, the exact case matches the oracle, recall stays above
 * the measured threshold, hot keys never miss, and the run is deterministic.
 */
import java.util.*;

/** splitmix64 plus text hashing for the sketch. */
final class Rng916 {
    private long state;

    Rng916(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }

    static long hashText(String text, long seed) {
        long h = seed;
        for (int i = 0; i < text.length(); i++) h = (h ^ text.charAt(i)) * 0xBF58476D1CE4E5B9L;
        h ^= h >>> 31;
        return h;
    }
}

/** One stream event: which side it came from, its key, its position and its payload. */
final class StreamEvent {
    final boolean left;
    final String key;
    final long seq;
    final long value;

    StreamEvent(boolean left, String key, long seq, long value) {
        this.left = left;
        this.key = key;
        this.seq = seq;
        this.value = value;
    }
}

/** Count-Min sketch: depth rows of counters, each row using a different hash seed. */
final class CountMin {
    final int width;
    final int depth;
    private final int[][] rows;
    private final long[] seeds;
    long updates;

    CountMin(int width, int depth, long seed) {
        this.width = width;
        this.depth = depth;
        this.rows = new int[depth][width];
        this.seeds = new long[depth];
        Rng916 rng = new Rng916(seed);
        for (int d = 0; d < depth; d++) seeds[d] = rng.nextLong() | 1L;
    }

    void add(String key) {
        updates++;
        for (int d = 0; d < depth; d++) rows[d][index(d, key)]++;
    }

    /** The minimum over rows: an upper bound on the true count, never a lower bound. */
    int estimate(String key) {
        int best = Integer.MAX_VALUE;
        for (int d = 0; d < depth; d++) best = Math.min(best, rows[d][index(d, key)]);
        return best;
    }

    private int index(int row, String key) {
        return (int) Math.floorMod(Rng916.hashText(key, seeds[row]), width);
    }
}

/** Bounded, LRU-keyed join state with hot-key pinning. */
final class JoinState {
    final int capacity;
    final int window;
    final Set<String> pinned;
    final LinkedHashMap<String, List<StreamEvent>> buffer;
    final CountMin frequency;
    long evicted;
    int peakKeys;

    JoinState(int capacity, int window, Set<String> pinned, CountMin frequency) {
        this.capacity = capacity;
        this.window = window;
        this.pinned = pinned;
        this.frequency = frequency;
        this.buffer = new LinkedHashMap<>(16, 0.75f, true);
    }

    int trackedKeys() { return buffer.size(); }

    List<StreamEvent> events(String key) { return buffer.getOrDefault(key, List.of()); }

    /** Records the event, evicting the least recently used unpinned key when the bound is reached. */
    void store(StreamEvent event) {
        List<StreamEvent> list = buffer.get(event.key);
        if (list == null) {
            while (buffer.size() >= capacity) {
                String victim = null;
                for (String key : buffer.keySet()) {
                    if (!pinned.contains(key)) {
                        victim = key;
                        break;
                    }
                }
                if (victim == null) break;
                buffer.remove(victim);
                evicted++;
            }
            list = new ArrayList<>();
            buffer.put(event.key, list);
        }
        list.add(event);
        if (list.size() > window + 1) list.remove(0);
        peakKeys = Math.max(peakKeys, buffer.size());
    }
}

public class Solution {
    static final int DISTINCT_KEYS = 30;
    static final int HOT_KEYS = 5;
    static final int WINDOW = 12;
    static final int DELAY = 12;
    static final int CAPACITY = 12;
    static final int HOT_THRESHOLD = 40;

    public static void main(String[] args) {
        List<StreamEvent> stream = generate(0x916ABCDEL, 1600);

        // ---- 1. the exact case must match the nested-loop oracle pair for pair -------------------------
        CountMin exactSketch = new CountMin(256, 4, 0x9160010L);
        JoinState unbounded = new JoinState(DISTINCT_KEYS + 1, WINDOW, new HashSet<>(), exactSketch);
        List<String> exactOutput = run(stream, unbounded);
        List<String> oracle = oracle(stream, WINDOW);
        List<String> sortedExact = new ArrayList<>(exactOutput);
        List<String> sortedOracle = new ArrayList<>(oracle);
        Collections.sort(sortedExact);
        Collections.sort(sortedOracle);
        check(sortedExact.equals(sortedOracle), "with room for every key the join must be exact: "
                + exactOutput.size() + " pairs vs oracle " + oracle.size());
        check(!oracle.isEmpty(), "the fixture must produce matches");
        check(exactSketch.updates == stream.size(), "the sketch must see every event");

        // ---- 2. bounded memory: the key count never exceeds the capacity --------------------------------
        CountMin sketch = new CountMin(256, 4, 0x9160020L);
        Set<String> pinned = new HashSet<>();
        JoinState bounded = new JoinState(CAPACITY, WINDOW, pinned, sketch);
        List<String> boundedOutput = new ArrayList<>();
        run(bounded, stream, boundedOutput, pinned, sketch);
        check(bounded.peakKeys <= CAPACITY, "the state held " + bounded.peakKeys + " keys, above capacity "
                + CAPACITY);
        check(bounded.evicted > 0, "a 12-key state must evict while streaming 30 distinct keys");
        check(pinned.size() > 0, "the hot-key policy must pin at least one key");

        // ---- 3. recall, and where the losses come from --------------------------------------------------
        Set<String> exactPairs = new HashSet<>(exactOutput);
        int kept = 0;
        for (String pair : boundedOutput) if (exactPairs.contains(pair)) kept++;
        double recall = kept / (double) exactPairs.size();
        check(recall >= 0.75, "bounded state recall must stay usable: " + recall);
        check(boundedOutput.size() >= kept, "the bounded join must not invent pairs the oracle does not have");
        Set<String> emitted = new HashSet<>(boundedOutput);
        boolean subset = true;
        for (String pair : emitted) if (!exactPairs.contains(pair)) subset = false;
        check(subset, "every emitted pair must be a genuine match, approximation may only lose output");
        int hotPairs = 0;
        int hotKept = 0;
        for (String pair : exactPairs) {
            String key = keyOf(pair);
            if (pinned.contains(key)) {
                hotPairs++;
                if (emitted.contains(pair)) hotKept++;
            }
        }
        check(hotPairs > 0, "the pinned keys must carry some of the output");
        check(hotKept == hotPairs, "pinned keys must never lose a match: " + hotKept + "/" + hotPairs);

        // ---- 4. the Count-Min estimate bound -------------------------------------------------------------
        Map<String, Integer> truth = new TreeMap<>();
        for (StreamEvent event : stream) truth.merge(event.key, 1, Integer::sum);
        int worstError = 0;
        for (Map.Entry<String, Integer> entry : truth.entrySet()) {
            int estimate = sketch.estimate(entry.getKey());
            check(estimate >= entry.getValue(), "the sketch must never undercount " + entry.getKey());
            worstError = Math.max(worstError, estimate - entry.getValue());
        }
        double bound = stream.size() / (double) sketch.width;
        check(worstError <= 2 * bound, "the overcount must stay near N/width: " + worstError + " vs " + bound);
        check(sketch.width * sketch.depth == 1024, "the sketch must be exactly width*depth counters, got "
                + (sketch.width * sketch.depth));
        check(sketch.depth >= 3, "at least three hash rows are needed for a usable bound");

        // ---- 5. boundary cases ---------------------------------------------------------------------------
        check(run(List.of(), new JoinState(CAPACITY, WINDOW, new HashSet<>(), new CountMin(64, 3, 1L)))
                .isEmpty(), "an empty stream joins to nothing");
        List<StreamEvent> oneSided = new ArrayList<>();
        for (int i = 0; i < 20; i++) oneSided.add(new StreamEvent(true, "k" + i % 3, i, i));
        check(run(oneSided, new JoinState(CAPACITY, WINDOW, new HashSet<>(), new CountMin(64, 3, 2L)))
                .isEmpty(), "a single-sided stream produces no matches");
        List<StreamEvent> pair = List.of(new StreamEvent(true, "solo", 1, 7),
                new StreamEvent(false, "solo", 2, 9));
        List<String> pairOutput = run(pair,
                new JoinState(CAPACITY, WINDOW, new HashSet<>(), new CountMin(64, 3, 3L)));
        check(pairOutput.size() == 1 && pairOutput.get(0).contains("solo"),
                "one left and one right event must produce exactly one pair, got " + pairOutput);
        List<StreamEvent> tooFar = List.of(new StreamEvent(true, "gap", 1, 7),
                new StreamEvent(false, "gap", 100, 9));
        check(run(tooFar, new JoinState(CAPACITY, WINDOW, new HashSet<>(), new CountMin(64, 3, 4L)))
                .isEmpty(), "events outside the window must not join");
        JoinState zeroCapacity = new JoinState(1, WINDOW, new HashSet<>(), new CountMin(64, 3, 5L));
        for (StreamEvent event : stream.subList(0, 50)) zeroCapacity.store(event);
        check(zeroCapacity.peakKeys <= 1, "a capacity of one key must hold, got " + zeroCapacity.peakKeys);

        // ---- 6. determinism -------------------------------------------------------------------------------
        CountMin replay = new CountMin(256, 4, 0x9160020L);
        Set<String> replayPinned = new HashSet<>();
        List<String> replayOutput = new ArrayList<>();
        run(new JoinState(CAPACITY, WINDOW, replayPinned, replay), stream, replayOutput, replayPinned, replay);
        check(replayOutput.equals(boundedOutput), "the bounded join must replay identically");
        check(replayPinned.equals(pinned), "the pinning decisions must replay identically");
        check(generate(0x916ABCDEL, 1600).size() == stream.size(), "the stream must be reproducible");

        System.out.println("stream " + stream.size() + " events, " + DISTINCT_KEYS + " distinct keys, window "
                + WINDOW + ", capacity " + CAPACITY + " keys");
        System.out.println("pairs: oracle " + exactPairs.size() + ", emitted " + emitted.size() + ", recall "
                + Math.round(recall * 10000) / 10000.0 + ", hot pairs " + hotKept + "/" + hotPairs);
        System.out.println("sketch " + sketch.width + "x" + sketch.depth + " worst overcount " + worstError
                + " (N/width " + Math.round(bound * 100) / 100.0 + "), evictions " + bounded.evicted);
        System.out.println("PASS");
    }

    /** Zipf-ish stream: 5 hot keys carry most events; each left event is matched DELAY steps later. */
    static List<StreamEvent> generate(long seed, int length) {
        Rng916 rng = new Rng916(seed);
        double[] weights = new double[DISTINCT_KEYS];
        double total = 0;
        for (int i = 0; i < DISTINCT_KEYS; i++) {
            weights[i] = i < HOT_KEYS ? 6.0 / (i + 1) : 0.6 / (i + 1);
            total += weights[i];
        }
        int[] keys = new int[length];
        for (int i = 0; i < length; i++) {
            double u = rng.nextDouble() * total;
            int key = 0;
            while (key < DISTINCT_KEYS - 1 && u > weights[key]) {
                u -= weights[key];
                key++;
            }
            keys[i] = key;
        }
        List<StreamEvent> stream = new ArrayList<>();
        long seq = 0;
        for (int i = 0; i < length; i++) {
            // The right event for keys[i - DELAY] is what the state must have been holding on to.
            stream.add(new StreamEvent(false, "key-" + keys[i], seq++, rng.nextInt(1000)));
            stream.add(new StreamEvent(true, "key-" + keys[i], seq++, rng.nextInt(1000)));
            if (i >= DELAY) {
                stream.add(new StreamEvent(false, "key-" + keys[i - DELAY], seq++, rng.nextInt(1000)));
            }
        }
        return stream;
    }

    static List<String> run(List<StreamEvent> stream, JoinState state) {
        return run(state, stream, new ArrayList<>(), state.pinned, state.frequency);
    }

    /** Feeds the stream through the bounded state, emitting one string per matched pair. */
    static List<String> run(JoinState state, List<StreamEvent> stream, List<String> output, Set<String> pinned,
            CountMin sketch) {
        for (StreamEvent event : stream) {
            sketch.add(event.key);
            if (sketch.estimate(event.key) >= HOT_THRESHOLD) pinned.add(event.key);
            for (StreamEvent other : state.events(event.key)) {
                if (other.left == event.left) continue;
                if (Math.abs(other.seq - event.seq) > state.window) continue;
                output.add(pairKey(other, event));
            }
            state.store(event);
        }
        return output;
    }

    static String pairKey(StreamEvent a, StreamEvent b) {
        long leftSeq = a.left ? a.seq : b.seq;
        long rightSeq = a.left ? b.seq : a.seq;
        return (a.left ? a.key : b.key) + ":" + leftSeq + "-" + rightSeq;
    }

    static String keyOf(String pair) { return pair.substring(0, pair.indexOf(':')); }

    /** Nested-loop oracle over the unbounded state. */
    static List<String> oracle(List<StreamEvent> stream, int window) {
        List<String> pairs = new ArrayList<>();
        for (int i = 0; i < stream.size(); i++) {
            for (int j = i + 1; j < stream.size(); j++) {
                StreamEvent a = stream.get(i);
                StreamEvent b = stream.get(j);
                if (a.left == b.left || !a.key.equals(b.key)) continue;
                if (Math.abs(a.seq - b.seq) > window) continue;
                pairs.add(pairKey(a, b));
            }
        }
        return pairs;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
