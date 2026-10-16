/** Task 625 - Implement wait-free queue algorithm (Kogan-Petrank style outline). (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * A wait-free queue guarantees that every operation finishes after a *bounded* number of its
 * own steps, even if other participants are descheduled forever. Kogan-Petrank get this from
 * two ingredients: (1) a *fetch-and-increment* cursor that hands out unique slot indices, so a
 * stalled thread can never block the cursors, and (2) an *announcement array* that holds an
 * operation's (index, value) before it is published, so any participant can finish a peer's
 * half-done operation ("helping"). A dequeue that finds slot i not yet published looks for the
 * announcement of index i and publishes it on the owner's behalf; index i is only really
 * "missing" when the cursor never handed it out, which means the queue is empty.
 *
 * What is simplified, and what is still verified:
 *   * The real algorithm keeps a fast two-slot queue in front of an unbounded slow queue, uses
 *     the phase/version of `top` to tell an empty slow queue from a stalled announcement, and
 *     retries a bounded number of times (OP_TRIES) before announcing. Here the two queues
 *     collapse into one bounded ring, and "empty versus stalled" is decided by comparing the
 *     slot index with the allocated-index cursor - the same test KP makes with phases.
 *   * The announcement lives in the slot itself: allocate writes (value, announced) to slot i
 *     in one indivisible simulation step, and publish flips `ready`. A peer that helps only
 *     sets `ready`, so it can never invent data. In the real algorithm the announcement is a
 *     separate array entry written before the index becomes visible.
 *   * The ring is bounded (CAP), so an enqueue past the bound reports FULL instead of growing
 *     the slow queue.
 * The invariants genuinely verified: with helping, an operation whose owner is descheduled
 * forever after announcing is still completed by a peer within a bounded number of that peer's
 * own steps; values come out exactly once in FIFO order under a deterministic interleaving;
 * and with helping disabled the very same schedule stalls and cannot retrieve the announced
 * value - the negative control.
 */
import java.util.*;

/** Bounded ring that keeps every allocated index announced until it is published. */
class WfQueue {
    static final int EMPTY = Integer.MIN_VALUE;
    static final int FULL = Integer.MIN_VALUE + 1;

    final int cap;
    final int[] value;
    final boolean[] announced;     // index handed out and value visible, not yet published
    final boolean[] ready;         // published: a dequeuer may take it
    int enqIndex;                  // fetch-and-increment cursor for allocation
    int deqIndex;                  // fetch-and-increment cursor for removal
    int enqueued;
    int dequeued;
    int helped;
    int stallDetected;             // an allocated index was neither announced nor published
    int lostWithoutHelping;        // a dequeue gave up on an announcement it could have published
    final List<Integer> log = new ArrayList<>();   // values in true dequeue order
    boolean helpingEnabled = true;

    WfQueue(int cap) {
        this.cap = cap;
        this.value = new int[cap];
        this.announced = new boolean[cap];
        this.ready = new boolean[cap];
    }

    int outstanding() {
        return enqIndex - deqIndex;
    }

    /** Hand out a unique index and make the value visible for helpers. */
    int allocate(int v) {
        if (outstanding() >= cap) return FULL;
        int i = enqIndex++;
        value[i % cap] = v;
        announced[i % cap] = true;
        return i;
    }

    /** Publish index i; idempotent and safe to run on behalf of the owner. */
    boolean publish(int i) {
        if (!announced[i % cap]) return false;
        ready[i % cap] = true;
        announced[i % cap] = false;
        enqueued++;
        return true;
    }

    int enqueue(int v) {
        int i = allocate(v);
        if (i == FULL) return FULL;
        publish(i);
        return i;
    }

    int dequeue() {
        int i = deqIndex;
        if (!ready[i % cap] && i < enqIndex) {
            if (announced[i % cap]) {
                if (!helpingEnabled) {
                    // Nobody is willing to help: the value stays announced, the queue stops
                    // making progress although data is outstanding.
                    lostWithoutHelping++;
                    return EMPTY;
                }
                publish(i);                       // help: finish the stalled owner's operation
                helped++;
            } else {
                stallDetected++;                  // allocated but invisible: a real bug
                return EMPTY;
            }
        }
        if (!ready[i % cap]) {
            if (i < enqIndex) stallDetected++;
            return EMPTY;                         // cursor never handed out index i: empty
        }
        int v = value[i % cap];
        ready[i % cap] = false;
        deqIndex++;
        dequeued++;
        log.add(v);
        return v;
    }
}

/** One simulated operation: one micro-step per run() call, counted for the wait-free bound. */
abstract class WfOp {
    final int thread;
    boolean done;
    boolean frozen;                // descheduled forever: never runs again
    boolean abandoned;
    int steps;

    WfOp(int thread) {
        this.thread = thread;
    }

    abstract void run();
}

class EnqOp extends WfOp {
    final WfQueue queue;
    final int v;
    int index = -1;
    int stage;

    EnqOp(int thread, WfQueue queue, int v) {
        super(thread);
        this.queue = queue;
        this.v = v;
    }

    void run() {
        steps++;
        if (stage == 0) {
            index = queue.allocate(v);
            stage = 1;
            if (index == WfQueue.FULL) {
                abandoned = true;
                done = true;
            }
        } else {
            queue.publish(index);                 // no-op when a helper already published it
            done = true;
        }
    }
}

class DeqOp extends WfOp {
    final WfQueue queue;
    final int maxTries;
    int result = WfQueue.EMPTY;
    int tries;

    DeqOp(int thread, WfQueue queue, int maxTries) {
        super(thread);
        this.queue = queue;
        this.maxTries = maxTries;
    }

    void run() {
        steps++;
        int v = queue.dequeue();
        if (v != WfQueue.EMPTY) {
            result = v;
            done = true;
            return;
        }
        if (++tries >= maxTries) {                // bounded retry, then give up
            done = true;
            abandoned = true;
        }
    }
}

public class Solution {
    static long seed = 0xD1342543DE82EF95L;

    static int nextInt(int bound) {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Math.floorMod(z, bound);
    }

    /** Interleave every live operation once per tick, in PRNG order. */
    static int runOps(List<WfOp> ops, int maxTicks) {
        int ticks = 0;
        while (ticks < maxTicks) {
            List<WfOp> live = new ArrayList<>();
            for (WfOp op : ops) if (!op.done && !op.frozen) live.add(op);
            if (live.isEmpty()) return ticks;
            for (int i = live.size() - 1; i > 0; i--) {
                int j = nextInt(i + 1);
                WfOp tmp = live.get(i);
                live.set(i, live.get(j));
                live.set(j, tmp);
            }
            for (WfOp op : live) op.run();
            ticks++;
        }
        return -1;
    }

    static int maxSteps(List<WfOp> ops, boolean enqueue) {
        int worst = 0;
        for (WfOp op : ops) {
            if ((op instanceof EnqOp) == enqueue) worst = Math.max(worst, op.steps);
        }
        return worst;
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    public static void main(String[] args) {
        singleThreadedFifo();
        emptyAndDegenerate();
        ringWraparound();
        stalledPublisherIsHelped();
        waitFreeBound();
        noHelpingLosesData();
        interleavedSchedule();
        System.out.println("PASS");
    }

    /** Sequential use: FIFO order and exactly-once removal. */
    static void singleThreadedFifo() {
        WfQueue q = new WfQueue(8);
        for (int v = 1; v <= 5; v++) check(q.enqueue(v) == v - 1, "enqueue " + v);
        check(q.enqueued == 5 && q.outstanding() == 5, "five items outstanding");
        for (int v = 1; v <= 5; v++) check(q.dequeue() == v, "FIFO dequeue expects " + v);
        check(q.dequeue() == WfQueue.EMPTY, "empty after draining");
        check(q.helped == 0 && q.stallDetected == 0, "no helping needed on the fast path");
        check(q.enqueued == q.dequeued && q.outstanding() == 0, "accounting balances");
    }

    /** Degenerate cases: empty queue, single element, full ring. */
    static void emptyAndDegenerate() {
        WfQueue q = new WfQueue(4);
        check(q.dequeue() == WfQueue.EMPTY, "empty dequeue reports EMPTY");
        check(q.deqIndex == 0 && q.outstanding() == 0, "empty dequeue consumes nothing");
        check(q.enqueue(7) == 0, "the first index handed out is 0");
        check(q.dequeue() == 7, "single element round trip");
        check(q.dequeue() == WfQueue.EMPTY, "empty again");
        check(q.stallDetected == 0 && q.lostWithoutHelping == 0, "empty is not a stall");

        WfQueue full = new WfQueue(2);
        check(full.enqueue(1) >= 0 && full.enqueue(2) >= 0, "ring filled to capacity");
        check(full.enqueue(3) == WfQueue.FULL, "bounded ring rejects overflow");
        check(full.dequeue() == 1, "oldest item first");
        check(full.enqueue(4) >= 0, "space becomes available after a removal");
        check(full.dequeue() == 2 && full.dequeue() == 4, "FIFO preserved across the wrap");
    }

    /** Many wraps around the ring keep FIFO order and the outstanding bound. */
    static void ringWraparound() {
        final int cap = 8;
        WfQueue q = new WfQueue(cap);
        List<Integer> model = new ArrayList<>();
        int next = 0;
        for (int round = 0; round < 30; round++) {
            for (int k = 0; k < 3; k++) {
                check(q.enqueue(next) == next, "enqueue " + next);
                model.add(next);
                next++;
            }
            check(q.outstanding() <= cap, "ring never overflows");
            for (int k = 0; k < 3; k++) {
                int got = q.dequeue();
                check(got == model.get(0), "FIFO dequeue in round " + round + " was " + got);
                model.remove(0);
            }
            check(q.outstanding() == model.size(), "cursors agree with the model");
        }
        check(q.helped == 0 && q.stallDetected == 0, "no helping when publishes never stall");
        check(q.enqueued == next && q.dequeued == next, "all items enqueued and dequeued");
    }

    /** A thread announces an index and is never scheduled again; a peer must finish it. */
    static void stalledPublisherIsHelped() {
        WfQueue q = new WfQueue(8);
        check(q.allocate(42) == 0, "the owner announces index 0 and then stalls");
        check(!q.ready[0] && q.announced[0] && q.outstanding() == 1, "slot 0 unpublished");
        check(q.dequeue() == 42, "a peer publishes the announced value on its behalf");
        check(q.helped == 1 && q.stallDetected == 0, "helping happened exactly once");
        check(q.dequeued == 1 && q.outstanding() == 0 && q.enqueued == 1,
                "the helped item is accounted for exactly once");

        check(q.allocate(43) == 1 && q.allocate(44) == 2, "two more announcements");
        check(q.dequeue() == 43 && q.dequeue() == 44, "helped items keep FIFO order");
        check(q.helped == 3, "each stalled publish was helped exactly once");
        check(q.dequeue() == WfQueue.EMPTY, "drained");
    }

    /** Wait-free bound: a permanently stalled peer adds nothing to anyone else's step count. */
    static void waitFreeBound() {
        WfQueue q = new WfQueue(16);
        List<WfOp> ops = new ArrayList<>();
        EnqOp stalled = new EnqOp(1, q, 100);
        stalled.run();                            // announces index 0, then is descheduled
        stalled.frozen = true;
        ops.add(stalled);
        check(stalled.steps == 1 && !stalled.done, "the stalled enqueue stopped after one step");

        for (int i = 0; i < 4; i++) ops.add(new EnqOp(2, q, 200 + i));
        List<DeqOp> deqs = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            DeqOp d = new DeqOp(3, q, 4);
            deqs.add(d);
            ops.add(d);
        }
        check(runOps(ops, 200) >= 0, "the schedule terminated");
        check(stalled.steps == 1 && !stalled.done, "the stalled operation never ran again");

        List<Integer> got = new ArrayList<>();
        for (DeqOp d : deqs) {
            check(!d.abandoned && d.result != WfQueue.EMPTY, "every dequeue got a value");
        }
        got.addAll(q.log);
        check(got.equals(allocationOrderOf(ops)), "FIFO order follows the allocation order: " + got);
        check(new HashSet<>(got).size() == 5, "no value lost or duplicated");
        check(q.dequeued == 5 && q.outstanding() == 0, "everything was dequeued exactly once");
        check(q.helped >= 1 && q.stallDetected == 0, "the stalled announcement was helped");
        check(maxSteps(ops, true) <= 2 && maxSteps(ops, false) <= 4,
                "bounded own-steps: enq " + maxSteps(ops, true) + ", deq " + maxSteps(ops, false));
    }

    /** Values in the order their queue indices were handed out - the FIFO ground truth. */
    static List<Integer> allocationOrderOf(List<WfOp> ops) {
        List<int[]> pairs = new ArrayList<>();
        for (WfOp op : ops) {
            if (op instanceof EnqOp) {
                EnqOp e = (EnqOp) op;
                check(e.steps >= 1, "an enqueue in the list must have allocated");
                pairs.add(new int[]{e.index, e.v});
            }
        }
        pairs.sort(Comparator.comparingInt(p -> p[0]));
        List<Integer> order = new ArrayList<>();
        for (int[] p : pairs) order.add(p[1]);
        return order;
    }

    /** The negative control: with helping disabled the same schedule stalls and loses data. */
    static void noHelpingLosesData() {
        WfQueue q = new WfQueue(4);
        q.helpingEnabled = false;
        check(q.allocate(9) == 0, "announce without publishing");
        check(q.dequeue() == WfQueue.EMPTY, "without helping the dequeuer cannot proceed");
        check(q.lostWithoutHelping == 1 && q.dequeued == 0, "the value is stuck in the announcement");
        check(q.outstanding() == 1 && q.enqueued == 0, "nothing was published");
        q.helpingEnabled = true;
        check(q.dequeue() == 9, "enabling helping recovers the value");
        check(q.helped == 1 && q.stallDetected == 0, "helping recovers what waiting cannot");
    }

    /** Random-but-reproducible interleaving with a permanently descheduled participant. */
    static void interleavedSchedule() {
        WfQueue q = new WfQueue(12);
        List<WfOp> ops = new ArrayList<>();

        EnqOp stalled = new EnqOp(1, q, 5000);
        stalled.run();                            // index 0 announced, owner descheduled forever
        stalled.frozen = true;
        ops.add(stalled);

        for (int i = 0; i < 10; i++) {
            for (int k = 0; k < 2; k++) {
                ops.add(new EnqOp(2 + k, q, 100 * (i + 1) + k));
            }
        }

        List<DeqOp> deqs = new ArrayList<>();
        for (int i = 0; i < 21; i++) {
            DeqOp d = new DeqOp(3, q, 8);
            deqs.add(d);
            ops.add(d);
        }
        check(runOps(ops, 5000) >= 0, "schedule terminated with a stalled peer");

        List<Integer> got = new ArrayList<>();
        for (DeqOp d : deqs) {
            check(!d.abandoned && d.result != WfQueue.EMPTY, "no dequeue gave up while data remained");
        }
        got.addAll(q.log);
        List<Integer> model = allocationOrderOf(ops);
        check(model.size() == 21 && model.get(0) == 5000, "the stalled announcement owns index 0");
        check(got.equals(model), "FIFO order matches the allocation order");
        check(new HashSet<>(got).size() == got.size(), "no value was handed out twice");
        check(q.stallDetected == 0, "no dequeue found an index that was invisible");
        check(q.helped >= 1, "the descheduled announcement was helped: " + q.helped);
        check(q.enqueued == 21 && q.dequeued == 21 && q.outstanding() == 0,
                "all 21 items accounted for: " + q.enqueued + "/" + q.dequeued);
        check(stalled.steps == 1 && maxSteps(ops, true) <= 2 && maxSteps(ops, false) <= 8,
                "every operation finished in bounded own-steps: " + maxSteps(ops, true) + "/"
                        + maxSteps(ops, false));
    }
}
