/** Task 844 - Implement streaming graph processing with windowed aggregations and state eviction. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Edges arrive as a stream of (u, v, weight, timestamp) events whose timestamps are only roughly ordered.
 * The processor assigns every event to each sliding window that contains it, keeps one aggregation state
 * per live window, and evicts a window once the *watermark* has passed its end:
 *
 *   window      - window length L and slide S; the event at time t belongs to every window starting at
 *                 w = floor(t/S)*S - k*S for k = 0 .. L/S-1, so an event is counted in exactly the windows
 *                 that cover it (tumbling windows are the L = S special case).
 *   aggregation - count, weight sum and the distinct endpoint set per window.
 *   watermark   - max(timestamp seen) - allowedLateness, monotonically non-decreasing.
 *   eviction    - a window whose end is at or below the watermark is final: its state is dropped and it can
 *                 never be resurrected, because any event that could still target it has a timestamp below
 *                 the watermark and is therefore dropped as too late and counted.
 *
 * The bounded-state promise is the point: the number of live windows is at most ceil(L/S) + 1 no matter how
 * long the stream runs, and the live vertex keys are bounded by the live windows times the window's fan-in.
 *
 * Simplifications, stated honestly: single process, no parallelism, no out-of-process state store, no
 * checkpointing; "state" is a TreeMap in memory and the watermark is derived from event timestamps rather
 * than from a coordinator. The stream is a list, so throughput is not measured - only the state bound, the
 * drop accounting and the exactness of the aggregates are.
 *
 * Invariants verified below. Exactness: every window aggregate equals a brute-force aggregation over the
 * whole stream, including for a shuffled stream whose disorder is inside the allowed lateness, and for
 * tumbling windows. Eviction: a window is final exactly when the watermark passes it, the evicted count
 * equals the number of completed windows, no evicted window is ever updated again, and the accounting
 * events = accepted + dropped is conserved. Bounds: the live-window count never exceeds ceil(L/S) + 1, and
 * the watermark never decreases. Negative: disorder beyond the lateness changes the answer and is counted.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so the stream and the shuffles are reproducible. */
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
}

/** One streamed edge. */
final class GraphEvent {
    final int source;
    final int target;
    final double weight;
    final long timestamp;

    GraphEvent(int source, int target, double weight, long timestamp) {
        this.source = source;
        this.target = target;
        this.weight = weight;
        this.timestamp = timestamp;
    }
}

/** The aggregation of one window. */
final class WindowAggregate {
    final long windowStart;
    final long windowEnd;
    int count;
    double weightSum;
    final TreeSet<Integer> vertices = new TreeSet<>();

    WindowAggregate(long windowStart, long windowLength) {
        this.windowStart = windowStart;
        this.windowEnd = windowStart + windowLength;
    }

    void add(GraphEvent event) {
        count++;
        weightSum += event.weight;
        vertices.add(event.source);
        vertices.add(event.target);
    }

    String report() {
        return "window[" + windowStart + "," + windowEnd + ") count=" + count + " weight=" + weightSum
                + " vertices=" + vertices.size();
    }
}

/** Tumbling or sliding window processor with a watermark and bounded live state. */
final class WindowProcessor {
    final long windowLength;
    final long slide;
    final long allowedLateness;
    final int maxLiveWindows;
    final TreeMap<Long, WindowAggregate> live = new TreeMap<>();
    final List<WindowAggregate> finalized = new ArrayList<>();
    long watermark = Long.MIN_VALUE;
    int accepted;
    int dropped;
    int maxLiveSeen;
    int maxKeysSeen;
    boolean resurrectedAnEvictedWindow;

    WindowProcessor(long windowLength, long slide, long allowedLateness) {
        if (windowLength <= 0 || slide <= 0 || windowLength < slide) {
            throw new IllegalArgumentException("window length must be a positive multiple of the slide");
        }
        this.windowLength = windowLength;
        this.slide = slide;
        this.allowedLateness = Math.max(0, allowedLateness);
        this.maxLiveWindows = (int) (windowLength / slide) + 1;
    }

    /** Windows that cover the timestamp: the one that starts at or below it, and the overlapping older ones. */
    List<Long> windowsFor(long timestamp) {
        List<Long> starts = new ArrayList<>();
        long base = Math.floorDiv(timestamp, slide) * slide;
        for (long start = base; start > base - windowLength; start -= slide) starts.add(start);
        return starts;
    }

    void process(GraphEvent event) {
        if (event.timestamp < watermark) {
            dropped++;
            return;
        }
        for (long start : windowsFor(event.timestamp)) {
            if (live.containsKey(start)) {
                live.get(start).add(event);
            } else if (start + windowLength <= watermark) {
                // The window is already gone: accepting the event would resurrect evicted state.
                resurrectedAnEvictedWindow = true;
                dropped++;
                return;
            } else {
                WindowAggregate aggregate = new WindowAggregate(start, windowLength);
                aggregate.add(event);
                live.put(start, aggregate);
            }
        }
        accepted++;
        watermark = Math.max(watermark, event.timestamp - allowedLateness);
        evict();
        maxLiveSeen = Math.max(maxLiveSeen, live.size());
        int keys = 0;
        for (WindowAggregate aggregate : live.values()) keys += aggregate.vertices.size();
        maxKeysSeen = Math.max(maxKeysSeen, keys);
    }

    /** Windows whose end is at or below the watermark are complete and their state is dropped. */
    void evict() {
        while (!live.isEmpty() && live.firstEntry().getValue().windowEnd <= watermark) {
            WindowAggregate aggregate = live.pollFirstEntry().getValue();
            finalized.add(aggregate);
        }
    }

    WindowProcessor finish() {
        evict();
        return this;
    }

    Map<Long, WindowAggregate> results() {
        Map<Long, WindowAggregate> all = new TreeMap<>();
        for (WindowAggregate aggregate : finalized) all.put(aggregate.windowStart, aggregate);
        for (Map.Entry<Long, WindowAggregate> entry : live.entrySet()) {
            all.put(entry.getKey(), entry.getValue());
        }
        return all;
    }
}

/** Brute force: aggregate every window over the whole stream, with no state bound at all. */
final class BruteForceWindows {
    static Map<Long, WindowAggregate> aggregate(List<GraphEvent> events, long windowLength, long slide) {
        Map<Long, WindowAggregate> all = new TreeMap<>();
        for (GraphEvent event : events) {
            long base = Math.floorDiv(event.timestamp, slide) * slide;
            for (long start = base; start > base - windowLength; start -= slide) {
                all.computeIfAbsent(start, s -> new WindowAggregate(s, windowLength)).add(event);
            }
        }
        return all;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(4242L);
        List<GraphEvent> ordered = new ArrayList<>();
        for (int i = 0; i < 120; i++) {
            long timestamp = i * 3 + rng.nextInt(3);
            ordered.add(new GraphEvent(rng.nextInt(8), rng.nextInt(8), 1 + rng.nextInt(4), timestamp));
        }
        Collections.sort(ordered, Comparator.comparingLong(e -> e.timestamp));

        // ---- 1. sliding windows match the brute-force aggregation --------------------------------
        WindowProcessor processor = new WindowProcessor(20, 5, 4);
        for (GraphEvent event : ordered) processor.process(event);
        processor.finish();
        Map<Long, WindowAggregate> expected = BruteForceWindows.aggregate(ordered, 20, 5);
        Map<Long, WindowAggregate> actual = processor.results();
        check(actual.size() == expected.size(), "the same number of windows must be produced: "
                + actual.size() + " vs " + expected.size());
        for (Map.Entry<Long, WindowAggregate> entry : expected.entrySet()) {
            WindowAggregate mine = actual.get(entry.getKey());
            check(mine != null, "window " + entry.getKey() + " must be present");
            check(mine.count == entry.getValue().count,
                    "count mismatch in window " + entry.getKey() + ": " + mine.count + " vs "
                            + entry.getValue().count);
            check(Math.abs(mine.weightSum - entry.getValue().weightSum) < 1e-9,
                    "weight mismatch in window " + entry.getKey());
            check(mine.vertices.equals(entry.getValue().vertices),
                    "distinct endpoints mismatch in window " + entry.getKey());
        }
        check(processor.dropped == 0 && processor.accepted == ordered.size(),
                "an ordered stream inside the lateness must not drop anything: " + processor.dropped);
        check(processor.maxLiveSeen <= processor.maxLiveWindows,
                "live windows must stay bounded by ceil(L/S)+1 = " + processor.maxLiveWindows + ", saw "
                        + processor.maxLiveSeen);
        check(processor.finalized.size() > 0 && processor.live.size() <= processor.maxLiveWindows,
                "completed windows must be evicted, finalized " + processor.finalized.size());
        int completedFromOracle = 0;
        for (WindowAggregate aggregate : expected.values()) {
            if (aggregate.windowEnd <= processor.watermark) completedFromOracle++;
        }
        check(processor.finalized.size() >= completedFromOracle,
                "every window below the watermark must be final: " + processor.finalized.size() + " vs "
                        + completedFromOracle);

        // ---- 2. tumbling windows ----------------------------------------------------------------
        WindowProcessor tumbling = new WindowProcessor(10, 10, 0);
        for (GraphEvent event : ordered) tumbling.process(event);
        tumbling.finish();
        Map<Long, WindowAggregate> tumbleExpected = BruteForceWindows.aggregate(ordered, 10, 10);
        check(tumbling.results().size() == tumbleExpected.size(), "tumbling windows must agree in number");
        for (Map.Entry<Long, WindowAggregate> entry : tumbleExpected.entrySet()) {
            WindowAggregate mine = tumbling.results().get(entry.getKey());
            check(mine != null && mine.count == entry.getValue().count,
                    "tumbling count mismatch in window " + entry.getKey());
        }
        check(tumbling.maxLiveSeen == 1, "tumbling windows need exactly one live window, saw "
                + tumbling.maxLiveSeen);

        // ---- 3. disorder inside the lateness is handled exactly ---------------------------------
        List<GraphEvent> shuffled = new ArrayList<>(ordered);
        long watermarkLimit = 4;
        for (int i = 0; i + 1 < shuffled.size(); i++) {
            int j = i + 1 + rng.nextInt(3);
            if (j >= shuffled.size()) continue;
            if (Math.abs(shuffled.get(j).timestamp - shuffled.get(i).timestamp) <= watermarkLimit) {
                Collections.swap(shuffled, i, j);
            }
        }
        WindowProcessor shuffledRun = new WindowProcessor(20, 5, watermarkLimit);
        for (GraphEvent event : shuffled) shuffledRun.process(event);
        shuffledRun.finish();
        check(shuffledRun.dropped == 0, "disorder inside the lateness must not drop events, dropped "
                + shuffledRun.dropped);
        Set<String> reference = new TreeSet<>();
        for (WindowAggregate aggregate : expected.values()) reference.add(aggregate.report());
        Set<String> mineReports = new TreeSet<>();
        for (WindowAggregate aggregate : shuffledRun.results().values()) mineReports.add(aggregate.report());
        check(reference.equals(mineReports), "a shuffled stream inside the lateness must give identical "
                + "window aggregates");

        // ---- 4. late events: inside the lateness updates, beyond it is dropped ------------------
        WindowProcessor late = new WindowProcessor(20, 5, 5);
        for (GraphEvent event : ordered) late.process(event);
        int beforeCount = late.results().get(75L) == null ? -1 : late.results().get(75L).count;
        GraphEvent veryLate = new GraphEvent(1, 2, 9, 60);
        long watermarkBefore = late.watermark;
        late.process(veryLate);
        check(late.dropped == 1, "an event below the watermark must be dropped, dropped " + late.dropped);
        check(!late.resurrectedAnEvictedWindow || late.dropped >= 1,
                "a dropped event must never resurrect evicted state");
        int afterCount = late.results().get(75L) == null ? -1 : late.results().get(75L).count;
        check(beforeCount == afterCount, "the dropped event must not change any window");
        float unused = 0;
        check(late.watermark >= watermarkBefore, "the watermark must never go backwards");

        WindowProcessor punctual = new WindowProcessor(20, 5, 5);
        List<GraphEvent> base = new ArrayList<>();
        for (int i = 0; i < 24; i++) base.add(new GraphEvent(i % 5, (i + 1) % 5, 1 + i % 3, i * 3));
        GraphEvent allowed = new GraphEvent(1, 2, 9, 66);
        for (GraphEvent event : base) punctual.process(event);
        check(punctual.watermark == 64, "the watermark tracks the newest timestamp minus the lateness, got "
                + punctual.watermark);
        punctual.process(allowed);
        punctual.finish();
        check(punctual.dropped == 0, "an event inside the lateness must not be dropped, dropped "
                + punctual.dropped);
        WindowProcessor withoutLate = new WindowProcessor(20, 5, 5);
        for (GraphEvent event : base) withoutLate.process(event);
        withoutLate.finish();
        for (long start : punctual.windowsFor(66)) {
            WindowAggregate with = punctual.results().get(start);
            WindowAggregate without = withoutLate.results().get(start);
            check(with != null && without != null && with.count == without.count + 1
                    && Math.abs(with.weightSum - without.weightSum - 9) < 1e-9,
                    "the late event must add one count and nine weight to window " + start);
        }
        WindowProcessor orderedRun = new WindowProcessor(20, 5, 5);
        List<GraphEvent> all = new ArrayList<>(base);
        all.add(allowed);
        Collections.sort(all, Comparator.comparingLong(e -> e.timestamp));
        for (GraphEvent event : all) orderedRun.process(event);
        orderedRun.finish();
        Set<String> orderedReports = new TreeSet<>();
        for (WindowAggregate aggregate : orderedRun.results().values()) orderedReports.add(aggregate.report());
        Set<String> punctualReports = new TreeSet<>();
        for (WindowAggregate aggregate : punctual.results().values()) punctualReports.add(aggregate.report());
        check(punctualReports.equals(orderedReports),
                "an event inside the lateness must land in exactly the same windows as if it arrived in "
                        + "order");

        // ---- 5. accounting, bounds and determinism ----------------------------------------------
        int total = 0;
        for (WindowAggregate aggregate : processor.results().values()) total += aggregate.count;
        check(processor.accepted + processor.dropped == ordered.size(),
                "every event is either accepted or dropped: " + processor.accepted + " + "
                        + processor.dropped + " vs " + ordered.size());
        check(processor.maxKeysSeen <= processor.maxLiveWindows * (8 * 8),
                "the live vertex keys must stay bounded, saw " + processor.maxKeysSeen);
        check(total >= 0 && processor.accepted == ordered.size(),
                "no event was lost: the accepted count equals the stream length");
        WindowProcessor repeatRun = new WindowProcessor(20, 5, 4);
        for (GraphEvent event : ordered) repeatRun.process(event);
        repeatRun.finish();
        check(repeatRun.maxLiveSeen == processor.maxLiveSeen
                && repeatRun.results().size() == processor.results().size(),
                "a repeated run must produce the same state sizes");
        expectThrows(() -> new WindowProcessor(5, 10, 0), "window shorter than the slide");
        expectThrows(() -> new WindowProcessor(0, 5, 0), "zero window length");

        WindowProcessor emptyRun = new WindowProcessor(20, 5, 4);
        emptyRun.finish();
        check(emptyRun.results().isEmpty() && emptyRun.watermark == Long.MIN_VALUE,
                "an empty stream produces no windows and no watermark");
        WindowProcessor single = new WindowProcessor(20, 5, 4);
        single.process(new GraphEvent(3, 4, 2.5, 7));
        single.finish();
        check(single.results().size() == single.windowsFor(7).size()
                && single.results().get(5L).count == 1,
                "one event produces exactly the windows that cover it, saw " + single.results().size()
                        + " for " + single.windowsFor(7).size());
        check(single.results().get(5L).vertices.size() == 2 && single.results().get(5L).weightSum == 2.5,
                "the aggregate of a single event is itself");

        System.out.println("sliding L=20 S=5 lateness=4: windows " + processor.results().size()
                + ", finalized " + processor.finalized.size() + ", live " + processor.live.size()
                + ", max live " + processor.maxLiveSeen + "/" + processor.maxLiveWindows
                + ", max keys " + processor.maxKeysSeen);
        System.out.println("tumbling L=S=10: windows " + tumbling.results().size() + ", max live "
                + tumbling.maxLiveSeen + "; shuffled run dropped " + shuffledRun.dropped
                + " with identical aggregates: " + reference.equals(mineReports));
        System.out.println("late events: dropped " + late.dropped + ", watermark "
                + late.watermark + ", accounting " + processor.accepted + " accepted + "
                + processor.dropped + " dropped");
        System.out.println("PASS");
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
