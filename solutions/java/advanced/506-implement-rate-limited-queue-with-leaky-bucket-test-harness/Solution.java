/** Task 506 - Implement rate-limited queue with leaky-bucket test harness. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * A leaky-bucket rate limiter as a discrete-time test harness: the bucket holds at most `capacity`
 * queued items and drains exactly `drain` items per tick (an integer drain, so no fractional
 * bookkeeping is hidden). Every tick is processed in the documented order - (1) leak min(level,
 * drain) items out in FIFO order, (2) admit this tick's arrivals in arrival order while the bucket
 * has room, (3) reject the rest - which makes the harness fully deterministic: time is the integer
 * tick counter, never the wall clock, and no threads are involved. The harness logs one record per
 * tick (drained, accepted, rejected, level) and keeps the global drain sequence, so the invariants
 * can be checked against a second, independently written reference simulation that stores the queue
 * in an ArrayList, keeps no level field at all (it recomputes the level from the list) and drives
 * its drain with a while-loop instead of the min() closed form.
 *
 * Documented checks: the level invariant 0 <= level <= capacity holds at every tick; the drained
 * count equals min(level_before, drain) and the accepted count equals max(0, min(capacity -
 * level_after_leak, arrivals_this_tick)) tick by tick; every arrival is either accepted or rejected
 * and the conservation identities arrivals == accepted + rejected and accepted == drained + level
 * hold; drained items leave in strict FIFO order; the reference simulation reproduces every
 * per-tick triple (drained, accepted, rejected) over a 400-tick LCG traffic pattern (mean 2.5
 * arrivals/tick against drain 1, so the bucket overflows); with drain = 0 exactly `capacity` items
 * are accepted, the level sits at capacity and nothing is ever processed; with capacity = 0 all 40
 * arrivals are rejected; with drain >= capacity a burst is fully drained during idle ticks and an
 * empty bucket processes nothing further; a saturated steady load fed nine arrivals per tick
 * accepts exactly drain * (ticks - 1) + capacity = 208 of the 900 arrivals, drains exactly drain *
 * (ticks - 1) = 198 items (the first tick starts empty, so it can leak nothing) and ends exactly at
 * capacity; and a re-run reproduces every tick record and the whole drained sequence bit for bit.
 */
import java.util.*;

public class Solution {
    /** 75/74 LCG modulo 65537 - the deterministic generator used by waves 1-2. */
    static class Lcg {
        private int state;

        Lcg(int seed) {
            this.state = ((seed % 65537) + 65537) % 65537;
        }

        double next() {
            state = (state * 75 + 74) % 65537;
            return state / 65537.0;
        }

        int nextInt(int bound) {
            return (int) (next() * bound);
        }
    }

    static class TickRecord {
        final int tick;
        final int drained;
        final int accepted;
        final int rejected;
        final int level;

        TickRecord(int tick, int drained, int accepted, int rejected, int level) {
            this.tick = tick;
            this.drained = drained;
            this.accepted = accepted;
            this.rejected = rejected;
            this.level = level;
        }
    }

    static class LeakyBucket {
        final int capacity;
        final int drain;
        final List<TickRecord> log = new ArrayList<>();
        final List<Integer> processed = new ArrayList<>();
        final List<Integer> arrivals = new ArrayList<>();
        final Deque<Integer> queue = new ArrayDeque<>();
        int level;
        int tick;
        int accepted;
        int rejected;
        int drainedTotal;

        LeakyBucket(int capacity, int drain) {
            this.capacity = capacity;
            this.drain = drain;
        }

        /** One tick: leak first, then admit arrivals in order while the bucket has room. */
        TickRecord step(int[] tickArrivals) {
            tick++;
            int leaked = Math.min(level, drain);
            int emitted = 0;
            for (int i = 0; i < leaked; i++) {
                Integer item = queue.pollFirst();
                if (item == null) break;
                processed.add(item);
                emitted++;
            }
            int remaining = level - leaked;
            drainedTotal += emitted;
            int admitted = 0;
            int refused = 0;
            for (int item : tickArrivals) {
                arrivals.add(item);
                if (remaining < capacity) {
                    queue.addLast(item);
                    remaining++;
                    admitted++;
                } else {
                    refused++;
                }
            }
            level = remaining;
            check(level == queue.size(), "level tracks the queue length");
            accepted += admitted;
            rejected += refused;
            TickRecord record = new TickRecord(tick, emitted, admitted, refused, level);
            log.add(record);
            return record;
        }

        int totalAccepted() {
            return accepted;
        }

        int totalRejected() {
            return rejected;
        }
    }

    /** Independent reference: an ArrayList queue, no level field, a while-loop drain. */
    static class ReferenceBucket {
        final int capacity;
        final int drain;
        final List<Integer> queue = new ArrayList<>();
        final List<Integer> processed = new ArrayList<>();
        final List<int[]> perTick = new ArrayList<>();
        int accepted;
        int rejected;

        ReferenceBucket(int capacity, int drain) {
            this.capacity = capacity;
            this.drain = drain;
        }

        int[] step(int[] tickArrivals) {
            int emitted = 0;
            while (emitted < drain && !queue.isEmpty()) {
                processed.add(queue.remove(0));
                emitted++;
            }
            int admitted = 0;
            int refused = 0;
            for (int item : tickArrivals) {
                if (queue.size() + 1 <= capacity) {
                    queue.add(item);
                    admitted++;
                } else {
                    refused++;
                }
            }
            accepted += admitted;
            rejected += refused;
            int[] record = {emitted, admitted, refused};
            perTick.add(record);
            return record;
        }
    }

    static int[][] traffic(int ticks, int seed, int maxPerTick) {
        Lcg lcg = new Lcg(seed);
        int[][] out = new int[ticks][];
        for (int t = 0; t < ticks; t++) {
            int size = lcg.nextInt(maxPerTick + 1);
            int[] arrivals = new int[size];
            for (int i = 0; i < size; i++) arrivals[i] = t * 1000 + i;
            out[t] = arrivals;
        }
        return out;
    }

    static int[] burst(int start, int count) {
        int[] out = new int[count];
        for (int i = 0; i < count; i++) out[i] = start + i;
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        int ticks = 400;
        int capacity = 7;
        int drain = 1;
        int[][] schedule = traffic(ticks, 60606, 5);
        LeakyBucket bucket = new LeakyBucket(capacity, drain);
        for (int[] arrivals : schedule) {
            int before = bucket.level;
            TickRecord record = bucket.step(arrivals);
            check(record.level >= 0 && record.level <= capacity,
                    "level invariant at tick " + record.tick);
            check(record.drained == Math.min(before, drain),
                    "drain equals min(level_before, drain) at tick " + record.tick);
            int room = capacity - (before - record.drained);
            check(record.accepted == Math.max(0, Math.min(room, arrivals.length)),
                    "acceptance equals the closed form at tick " + record.tick);
            check(record.accepted + record.rejected == arrivals.length,
                    "every arrival is either accepted or rejected at tick " + record.tick);
        }

        // 1. Conservation and FIFO order.
        int totalArrivals = 0;
        for (int[] arrivals : schedule) totalArrivals += arrivals.length;
        check(bucket.arrivals.size() == totalArrivals, "the harness saw every arrival");
        check(totalArrivals == bucket.totalAccepted() + bucket.totalRejected(),
                "arrivals == accepted + rejected");
        check(bucket.totalAccepted() == bucket.processed.size() + bucket.level,
                "accepted == processed + queue level");
        for (int i = 1; i < bucket.processed.size(); i++) {
            check(bucket.processed.get(i) > bucket.processed.get(i - 1),
                    "drained items leave in strict FIFO order at index " + i);
        }
        check(bucket.processed.size() > 100, "the trace actually processes items");
        check(bucket.totalRejected() > 0, "the trace (mean 2.5 arrivals/tick vs drain 1) overflows");
        check(bucket.drainedTotal == bucket.processed.size(),
                "the drain counter matches the sequence");

        // 2. The independent reference simulation, tick by tick.
        ReferenceBucket reference = new ReferenceBucket(capacity, drain);
        for (int[] arrivals : schedule) reference.step(arrivals);
        check(reference.perTick.size() == bucket.log.size(),
                "the reference ran the same number of ticks");
        for (int t = 0; t < bucket.log.size(); t++) {
            TickRecord record = bucket.log.get(t);
            int[] expected = reference.perTick.get(t);
            check(record.drained == expected[0] && record.accepted == expected[1]
                            && record.rejected == expected[2],
                    "reference agrees at tick " + t);
        }
        check(reference.processed.equals(bucket.processed), "identical drained sequences");
        check(reference.accepted == bucket.totalAccepted() && reference.rejected == bucket.totalRejected(),
                "identical counters");

        // 3. drain = 0: the bucket fills to capacity and then rejects everything.
        LeakyBucket blocked = new LeakyBucket(5, 0);
        for (int t = 0; t < 20; t++) blocked.step(burst(t * 10, 3));
        check(blocked.totalAccepted() == 5, "with drain 0 exactly capacity items are accepted");
        check(blocked.processed.isEmpty(), "with drain 0 nothing is ever processed");
        check(blocked.level == 5, "the bucket sits exactly at capacity");
        check(blocked.totalRejected() == 60 - 5,
                "everything after the first five arrivals is rejected");

        // 4. capacity = 0 rejects every arrival.
        LeakyBucket zero = new LeakyBucket(0, 2);
        for (int t = 0; t < 10; t++) zero.step(burst(t * 10, 4));
        check(zero.totalAccepted() == 0 && zero.totalRejected() == 40,
                "capacity 0 rejects everything");

        // 5. drain >= capacity empties the bucket; an idle stretch drains it to zero.
        LeakyBucket fast = new LeakyBucket(3, 3);
        fast.step(burst(0, 3));
        check(fast.level == 3, "the burst fills the bucket");
        for (int t = 0; t < 5; t++) fast.step(new int[0]);
        check(fast.level == 0, "an idle stretch drains the bucket to zero");
        check(fast.processed.size() == 3,
                "all three burst items were processed during the idle ticks");
        int before = fast.processed.size();
        fast.step(new int[0]);
        check(fast.processed.size() == before, "an empty bucket processes nothing");

        // 6. Long-run throughput: once saturated, accepted == drain * ticks + capacity.
        LeakyBucket steady = new LeakyBucket(10, 2);
        int steadyTicks = 100;
        int offered = 0;
        for (int t = 0; t < steadyTicks; t++) {
            int[] arrivals = burst(t * 1000, 9);
            offered += arrivals.length;
            steady.step(arrivals);
        }
        check(steady.totalAccepted() == 2 * (steadyTicks - 1) + 10,
                "saturated throughput is drain * (ticks - 1) + capacity");
        check(offered == 9 * steadyTicks && steady.totalAccepted() + steady.totalRejected() == offered,
                "conservation on the saturated run");
        check(steady.processed.size() == 2 * (steadyTicks - 1),
                "processed equals drain * (ticks - 1) when saturated");
        check(steady.log.get(steadyTicks - 1).level == 10, "the saturated bucket ends at capacity");

        // 7. Determinism of the whole run.
        LeakyBucket rerun = new LeakyBucket(capacity, drain);
        for (int[] arrivals : schedule) rerun.step(arrivals);
        check(rerun.log.size() == bucket.log.size(), "deterministic re-run: log length");
        for (int t = 0; t < bucket.log.size(); t++) {
            TickRecord a = bucket.log.get(t);
            TickRecord b = rerun.log.get(t);
            check(a.drained == b.drained && a.accepted == b.accepted && a.rejected == b.rejected
                            && a.level == b.level,
                    "deterministic re-run at tick " + t);
        }
        check(rerun.processed.equals(bucket.processed), "deterministic re-run: drained sequence");

        System.out.println("PASS");
    }
}
