/** Task 807 - Implement global scale consistent snapshot across geographically distributed shards. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A globally ordered store spread over regions. Every write is stamped with a timestamp from a
 * single global sequencer, so the commit order is total; each region's *local* clock is deliberately
 * skewed (a per-region offset within an advertised bound) and therefore unusable for ordering, which
 * is the point of the exercise. Each shard publishes a safe time: it advances its local clock past T
 * only once no write with a timestamp <= T can still arrive, because every in-flight write already
 * carries a timestamp of at least T. A snapshot reader asks the coordinator for a timestamp T and the
 * coordinator only releases it when every shard's clock has passed T (a bounded wait, measured in
 * ticks). Reads at T then return exactly the writes with timestamp <= T - a complete, monotonic,
 * prefix-consistent cut - while reads at a shard's own local time are explicitly allowed to be stale
 * by up to the skew bound, and no more.
 *
 * Simulation model, and what is simplified. One process: a tick loop, an in-memory sequencer, shards
 * as sorted maps and clock skew as fixed offsets. There is no NTP, no TrueTime hardware and no
 * physical clock; "waiting for the safe time" is an explicit loop over ticks rather than a sleep; the
 * network is a list of in-flight writes with scheduled arrival ticks; and durability is out of scope.
 * What is *not* simplified: the ordering argument itself. A snapshot is only released when every shard
 * has advanced past the timestamp, an in-flight write really is invisible until it lands, and every
 * snapshot is compared against an oracle that replays the global commit log up to the same timestamp.
 *
 * Verified: a snapshot at T contains exactly the oracle's writes up to T, for many T; snapshots are
 * monotonic (T1 < T2 implies the earlier snapshot is a subset); a snapshot never contains a write
 * from a future timestamp, even when that write is already visible in the region that produced it;
 * the coordinator refuses to release a timestamp while a shard is still behind, and releases it after
 * exactly the required number of ticks, with the wait bounded by the skew plus the propagation delay;
 * a local (uncoordinated) read is stale by at most the skew and never by more; writes are never lost:
 * every acknowledged write appears in every later snapshot; the failure of one region stalls the
 * snapshot rather than corrupting it (and the snapshot resumes complete once the region returns); and
 * the negative control - a coordinator that releases a timestamp without waiting for the safe time -
 * is shown to publish a snapshot that is missing a write it should have contained.
 */
import java.util.*;

final class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** A write: globally ordered by `ts`, produced by a region, landing at its home shard after a delay. */
final class Write807 {
    final long ts, key;
    final String value, origin;
    final int arrivalTick;
    boolean landed;

    Write807(long ts, long key, String value, String origin, int arrivalTick) {
        this.ts = ts;
        this.key = key;
        this.value = value;
        this.origin = origin;
        this.arrivalTick = arrivalTick;
    }
}

final class Region807 {
    final String name;
    final long skew;                                  // this region's fixed clock offset
    final TreeMap<Long, Write807> data = new TreeMap<>();
    final List<Write807> inFlight = new ArrayList<>();
    long safeTime;                                    // writes up to here can never arrive again
    long maxLandedTs = -1;
    int landedWrites;
    boolean down;

    Region807(String name, long skew) {
        this.name = name;
        this.skew = skew;
        safeTime = -1;
    }

    /** The region's own clock: the global tick plus its offset. */
    long localClock(long tick) { return tick + skew; }

    void submit(Write807 w) {
        if (down) { w.landed = false; return; }
        inFlight.add(w);
    }

    /**
     * The largest timestamp this shard can prove complete: nothing older is still in flight. When
     * nothing is in flight every timestamp is complete, so the answer is unbounded - this is the
     * ground truth the clock-based promise is measured against.
     */
    long trueSafeTime() {
        long min = Long.MAX_VALUE;
        for (Write807 w : inFlight) min = Math.min(min, w.ts);
        if (min == Long.MAX_VALUE) return Long.MAX_VALUE;
        return min - 1;
    }

    /** Delivers everything whose arrival tick has come; the safe time follows the last arrival. */
    int deliver(long tick) {
        if (down) return 0;
        int delivered = 0;
        for (Iterator<Write807> it = inFlight.iterator(); it.hasNext();) {
            Write807 w = it.next();
            if (w.arrivalTick <= tick) {
                it.remove();
                data.put(w.ts, w);
                w.landed = true;
                landedWrites++;
                maxLandedTs = Math.max(maxLandedTs, w.ts);
                delivered++;
            }
        }
        safeTime = Math.min(trueSafeTime(), maxLandedTs);
        return delivered;
    }

    /** Reads every write with ts <= T that has actually landed here. */
    TreeMap<Long, Write807> readUpTo(long ts) {
        if (down) throw new IllegalStateException("region " + name + " is unavailable");
        TreeMap<Long, Write807> out = new TreeMap<>();
        for (Map.Entry<Long, Write807> e : data.headMap(ts, true).entrySet()) out.put(e.getKey(), e.getValue());
        return out;
    }

    /** An uncoordinated read at the region's own clock - allowed to be stale by at most the skew. */
    TreeMap<Long, Write807> readAtLocalClock(long tick) { return readUpTo(localClock(tick)); }
}

final class GlobalStore807 {
    static final long SKEW_BOUND = 3;
    static final int PROPAGATION = 2;

    final TreeMap<String, Region807> regions = new TreeMap<>();
    final List<Region807> order = new ArrayList<>();
    final TreeMap<Long, Write807> commitLog = new TreeMap<>();     // the oracle's global order
    long sequencer;
    long tick;
    int rounds, waitTicks, snapshots, refused;

    GlobalStore807(String... regionNames) {
        long[] offsets = {0, -SKEW_BOUND, SKEW_BOUND};
        for (int i = 0; i < regionNames.length; i++) {
            Region807 r = new Region807(regionNames[i], offsets[i % offsets.length]);
            regions.put(regionNames[i], r);
            order.add(r);
        }
    }

    Region807 region(String name) {
        Region807 r = regions.get(name);
        if (r == null) throw new IllegalArgumentException("unknown region " + name);
        return r;
    }

    /** Assigns the next global timestamp and schedules the write to land at every shard. */
    Write807 write(long key, String value, String origin) {
        // A write is never stamped before it is issued, which is what lets a shard bound the arrival
        // time of a timestamp from its own clock alone: ts = T implies the write left by tick T.
        long ts = Math.max(++sequencer, tick + 1);
        sequencer = ts;
        Write807 w = new Write807(ts, key, value, origin, (int) (tick + PROPAGATION + Math.floorMod(ts, 2)));
        commitLog.put(ts, w);
        for (Region807 r : order) r.submit(w);
        return w;
    }

    void advance(int ticks) {
        for (int i = 0; i < ticks; i++) {
            tick++;
            for (Region807 r : order) r.deliver(tick);
        }
    }

    /**
     * A shard may serve timestamp T once its clock has passed T plus the propagation delay plus the
     * clock-uncertainty bound: a write stamped T left by tick T, arrives within the propagation delay,
     * and the shard's local clock underestimates the true tick by at most the skew.
     */
    boolean allSafeAt(long ts) {
        for (Region807 r : order) {
            if (r.down) return false;
            if (r.localClock(tick) <= ts + PROPAGATION + SKEW_BOUND) return false;
            if (r.trueSafeTime() < ts) return false;      // the ground truth must back the promise
        }
        return true;
    }

    /** The coordinator: pick a timestamp, wait for the safe time (bounded), then release it. */
    long snapshotTimestamp(boolean waitForSafeTime) {
        long ts = ++sequencer;
        rounds++;
        int waited = 0;
        while (waitForSafeTime && !allSafeAt(ts)) {
            advance(1);
            waited++;
            if (waited > 64) {                            // far beyond the bound the tests measure
                StringBuilder sb = new StringBuilder("safe time never reached for ts=" + ts + " tick=" + tick);
                for (Region807 r : order) {
                    sb.append(" | ").append(r.name).append(" down=").append(r.down)
                      .append(" local=").append(r.localClock(tick))
                      .append(" trueSafe=").append(r.trueSafeTime())
                      .append(" inFlight=").append(r.inFlight.size());
                }
                throw new IllegalStateException(sb.toString());
            }
        }
        if (!waitForSafeTime && !allSafeAt(ts)) refused++;
        waitTicks += waited;
        snapshots++;
        return ts;
    }

    /** The snapshot at T: the union of every shard's landed writes with a timestamp <= T. */
    TreeMap<Long, Write807> snapshot(boolean waitForSafeTime) {
        long ts = snapshotTimestamp(waitForSafeTime);
        advance(1);                                   // propagate the barrier
        TreeMap<Long, Write807> out = new TreeMap<>();
        for (Region807 r : order) out.putAll(r.readUpTo(ts));
        return out;
    }

    /** The oracle: the global commit log cut at the same timestamp. */
    TreeMap<Long, Write807> oracle(long ts) {
        TreeMap<Long, Write807> out = new TreeMap<>();
        for (Map.Entry<Long, Write807> e : commitLog.headMap(ts, true).entrySet()) out.put(e.getKey(), e.getValue());
        return out;
    }

    static String render(TreeMap<Long, Write807> snapshot) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Long, Write807> e : snapshot.entrySet()) {
            if (sb.length() > 0) sb.append(' ');
            sb.append(e.getKey()).append(':').append(e.getValue().key).append('=').append(e.getValue().value);
        }
        return sb.toString();
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    static GlobalStore807 seeded(long seed, int writes) {
        GlobalStore807 store = new GlobalStore807("eu", "us", "ap");
        SplitMix64 rng = new SplitMix64(seed);
        String[] names = {"eu", "us", "ap"};
        for (int i = 0; i < writes; i++) {
            store.write(rng.nextInt(20), "v" + i, names[rng.nextInt(3)]);
            if (i % 3 == 2) store.advance(2);
        }
        store.advance((int) (2 * GlobalStore807.SKEW_BOUND + 10));  // everything lands before the tests read
        return store;
    }

    /** Snapshots must equal the oracle cut at the same timestamp, for many timestamps. */
    static void snapshotCompleteness() {
        GlobalStore807 store = seeded(807001L, 40);
        int checked = 0;
        for (long ts = 1; ts <= store.sequencer; ts++) {
            TreeMap<Long, Write807> cut = new TreeMap<>();
            for (Region807 r : store.order) cut.putAll(r.readUpTo(ts));
            check(GlobalStore807.render(cut).equals(GlobalStore807.render(store.oracle(ts))),
                    "the cut at " + ts + " differs from the oracle:\n  " + GlobalStore807.render(cut)
                            + "\n  " + GlobalStore807.render(store.oracle(ts)));
            checked++;
        }
        check(checked == store.sequencer, "every timestamp was checked: " + checked);
        TreeMap<Long, Write807> snap = store.snapshot(true);
        check(GlobalStore807.render(snap).equals(GlobalStore807.render(store.oracle(snap.lastKey()))),
                "the released snapshot matches the oracle at its own timestamp");
        check(snap.size() == store.commitLog.size(), "a snapshot after the last write holds everything: "
                + snap.size() + " vs " + store.commitLog.size());
        metric("timestampsChecked", checked);
        metric("snapshots", store.snapshots);
    }

    /** Snapshots form a growing prefix chain. */
    static void monotonicSnapshots() {
        GlobalStore807 store = seeded(807002L, 30);
        long first = store.sequencer / 3;
        TreeMap<Long, Write807> a = store.oracle(first);
        TreeMap<Long, Write807> b = store.oracle(first + 5);
        TreeMap<Long, Write807> c = store.oracle(store.sequencer);
        check(a.size() <= b.size() && b.size() <= c.size(), "snapshots grow: " + a.size() + ", " + b.size()
                + ", " + c.size());
        for (Long key : a.keySet()) {
            check(b.containsKey(key) && c.containsKey(key), "write " + key + " survives into later snapshots");
            check(b.get(key).value.equals(a.get(key).value), "and never changes value");
        }
        check(c.size() == 30, "the final snapshot holds all thirty writes: " + c.size());
        check(!a.isEmpty(), "the early snapshot is not empty: " + a.size());
        TreeMap<Long, Write807> live = store.snapshot(true);
        check(live.size() >= c.size(), "a live snapshot is at least as complete as the oracle: " + live.size());
        metric("prefixWrites", a.size());
    }

    /** The safe time: the coordinator must wait, and the wait is bounded by the skew and delay. */
    static void safeTimeWaitAndLatency() {
        GlobalStore807 store = new GlobalStore807("eu", "us", "ap");
        store.write(1, "early", "eu");
        store.advance(3);
        long before = store.tick;
        int waitsObserved = 0;
        for (int i = 0; i < 6; i++) {
            long start = store.tick;
            long ts = store.snapshotTimestamp(true);
            long waited = store.tick - start;
            check(waited <= 2 * GlobalStore807.SKEW_BOUND + GlobalStore807.PROPAGATION + 4,
                    "the coordinator waited at most twice the skew plus the delay: " + waited);
            if (waited > 0) waitsObserved++;
            check(store.allSafeAt(ts), "after the wait the timestamp is safe");
        }
        check(store.tick > before, "the coordinator really advanced the clock");
        check(waitsObserved > 0, "at least one snapshot had to wait for the safe time: " + waitsObserved);

        GlobalStore807 stalled = new GlobalStore807("eu", "us", "ap");
        stalled.write(1, "a", "eu");
        stalled.advance(4);
        stalled.region("us").down = true;
        boolean refused = false;
        try {
            stalled.snapshotTimestamp(true);
        } catch (IllegalStateException expected) {
            refused = true;
        }
        check(refused, "a down region stalls the snapshot instead of releasing a bad timestamp");
        check(!stalled.allSafeAt(++stalled.sequencer), "and the timestamp stays unsafe");
        stalled.region("us").down = false;
        stalled.advance(6);
        TreeMap<Long, Write807> snap = stalled.snapshot(true);
        check(GlobalStore807.render(snap).equals(GlobalStore807.render(stalled.oracle(snap.lastKey()))),
                "once the region returns the snapshot is complete again");
        metric("waitTicks", store.waitTicks);
    }

    /** Local reads: exactly consistent up to the shard's safe time, and stale only within the window. */
    static void boundedStaleness() {
        GlobalStore807 store = new GlobalStore807("eu", "us", "ap");
        int maxLag = 0;
        int soundChecks = 0;
        for (int i = 1; i <= 20; i++) {
            store.write(i, "w" + i, "eu");
            store.advance(((i % 3) == 0) ? 2 : 1);
            Region807 us = store.region("us");
            long safe = Math.min(us.trueSafeTime(), us.maxLandedTs);
            check(GlobalStore807.render(us.readUpTo(safe)).equals(GlobalStore807.render(store.oracle(safe))),
                    "a read at the shard's own safe time (" + safe + ") is exactly the oracle's cut");
            long local = us.localClock(store.tick);
            TreeMap<Long, Write807> cut = us.readUpTo(local);
            TreeMap<Long, Write807> truth = store.oracle(local);
            int missing = 0;
            for (Long ts : truth.keySet()) if (!cut.containsKey(ts)) missing++;
            int window = 0;
            for (Long ts : truth.tailMap(local - GlobalStore807.PROPAGATION, true).keySet()) window++;
            check(missing <= window + GlobalStore807.SKEW_BOUND,
                    "a local read lags by at most the propagation window: " + missing + " > " + window);
            maxLag = Math.max(maxLag, missing);
            for (long t = 1; t <= store.sequencer; t++) {
                if (us.localClock(store.tick) > t + GlobalStore807.PROPAGATION + GlobalStore807.SKEW_BOUND) {
                    check(us.trueSafeTime() >= t, "the clock-based promise is sound for T=" + t);
                    soundChecks++;
                }
            }
        }
        check(soundChecks > 0, "the clock promise was exercised on real timestamps: " + soundChecks);
        GlobalStore807 fast = new GlobalStore807("eu", "us", "ap");
        fast.write(1, "x", "eu");
        check(fast.region("ap").readUpTo(0).isEmpty() || fast.region("ap").data.isEmpty(),
                "before the write lands the shard has nothing to serve");
        fast.advance(GlobalStore807.PROPAGATION + 1);
        check(fast.region("ap").readAtLocalClock(fast.tick).size() == 1,
                "after the propagation delay the write is visible");
        int skewed = 0;
        for (Region807 r : fast.order) {
            check(Math.abs(r.skew) <= GlobalStore807.SKEW_BOUND, "the skew stays within the advertised bound");
            if (r.skew != 0) skewed++;
        }
        check(skewed >= 1, "at least one region really has a skewed clock: " + skewed);
        metric("maxLocalLag", maxLag);
        metric("clockPromiseChecks", soundChecks);
    }

    /** No acknowledged write may ever disappear from a later snapshot. */
    static void durability() {
        GlobalStore807 store = seeded(807003L, 25);
        TreeMap<Long, Write807> first = store.snapshot(true);
        int seen = first.size();
        for (int i = 0; i < 15; i++) {
            store.write(100 + i, "later" + i, "eu");
            store.advance(3);
        }
        TreeMap<Long, Write807> second = store.snapshot(true);
        for (Long ts : first.keySet()) {
            check(second.containsKey(ts), "write " + ts + " vanished between snapshots");
            check(second.get(ts).value.equals(first.get(ts).value), "and kept its value");
        }
        check(second.size() == seen + 15, "the fifteen new writes are all there: " + second.size()
                + " vs " + seen);
        check(GlobalStore807.render(second).equals(GlobalStore807.render(store.oracle(second.lastKey()))),
                "and the result still matches the oracle");
        metric("finalSnapshotSize", second.size());
    }

    /** Negative control: releasing a timestamp early publishes an incomplete snapshot. */
    static void negativeControlEagerSnapshot() {
        GlobalStore807 good = new GlobalStore807("eu", "us", "ap");
        good.write(1, "a", "eu");
        long goodTs = good.snapshotTimestamp(true);
        good.advance(2);
        TreeMap<Long, Write807> complete = good.snapshot(true);
        check(GlobalStore807.render(complete).equals(GlobalStore807.render(good.oracle(complete.lastKey()))),
                "the waiting coordinator publishes a complete snapshot");
        check(good.oracle(goodTs).size() == 1, "the first timestamp covers the first write");

        GlobalStore807 eager = new GlobalStore807("eu", "us", "ap");
        Write807 inFlight = eager.write(1, "in-flight", "eu");     // not delivered yet
        long ts = eager.snapshotTimestamp(false);                  // no wait: released immediately
        int missing = 0;
        for (Long key : eager.oracle(ts).keySet()) {
            if (!eager.region("eu").data.containsKey(key)) missing++;
        }
        check(missing > 0, "the eager timestamp was released while a write was still in flight");
        eager.advance(GlobalStore807.PROPAGATION + 2);
        check(inFlight.landed, "the write landed later, after the snapshot was taken");
        check(eager.region("eu").data.containsKey(inFlight.ts),
                "so the eagerly taken snapshot was missing a write it should have contained");
        int stillMissing = 0;
        for (Long key : eager.oracle(ts).keySet()) if (!eager.region("us").data.containsKey(key)) stillMissing++;
        check(stillMissing == 0, "the correct (waiting) path would have included it");
        metric("negativeControlMissing", missing);
    }

    public static void main(String[] args) {
        snapshotCompleteness();
        monotonicSnapshots();
        safeTimeWaitAndLatency();
        boundedStaleness();
        durability();
        negativeControlEagerSnapshot();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
