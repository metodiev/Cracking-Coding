/** Task 626 - Implement work-stealing deque (Chase-Lev) for task scheduler. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * A Chase-Lev deque has two ends: the owner pushes and pops at the *bottom* (so its own work
 * is LIFO - depth-first, cache friendly), while thieves steal at the *top* (so they take the
 * oldest task, which keeps the stolen subtree big enough to be worth the transfer). Only one
 * index ever needs a CAS: `top`, written by the owner only when it pops the very last element
 * (racing a thief) and by thieves on every steal. `bottom` is written by the owner alone.
 *
 * Deliberate simplifications, documented because they are not the interesting part:
 *   * The array is indexed by absolute position modulo the current capacity, so growing only
 *     doubles the capacity and re-scatters the live entries - no re-centering dance.
 *   * Shrinking happens only when the owner has just taken the last element, i.e. when the
 *     deque is empty. The absolute indices are left alone (only the modulus changes), which
 *     keeps stale thieves safe: their CAS on `top` still fails.
 *   * There are no threads: the owner and thief operations are micro-steps of a step machine,
 *     and the one truly racy interleaving (owner popping the last element while a thief has
 *     already read it) is scripted by hand.
 *
 * Verified: owner pops are LIFO and steals are FIFO; a twenty-times-larger deque grows and
 * shrinks without losing order; a PRNG-driven push/pop/steal schedule matches an ArrayDeque
 * model element for element; the last-element race hands the element to exactly one of the two
 * contenders in both interleavings; and a three-worker scheduler drains a lopsided workload
 * with every task executed exactly once.
 */
import java.util.*;

/** Chase-Lev deque over a resizable circular array indexed by absolute position. */
class ChaseLevDeque {
    static final int EMPTY = Integer.MIN_VALUE;
    static final int MIN_CAPACITY = 4;

    int[] array;
    int top;          // thieves' end: the oldest element lives here
    int bottom;       // owner's end: the newest element lives here
    int capacity;
    int grows;
    int shrinks;
    int steals;
    int pops;

    ChaseLevDeque() {
        capacity = MIN_CAPACITY;
        array = new int[capacity];
    }

    int size() {
        return bottom - top;
    }

    void push(int v) {
        int b = bottom;
        if (b - top > capacity - 1) grow();
        array[b % capacity] = v;
        bottom = b + 1;                       // publish
    }

    /** Owner: takes the newest element (LIFO); races a thief only for the last one. */
    int pop() {
        int b = bottom - 1;
        bottom = b;
        int t = top;
        if (t <= b) {
            int v = array[b % capacity];
            if (t == b) {                     // last element: thieves may be racing for it
                if (!casTop(t, t + 1)) {
                    bottom = t + 1;
                    return EMPTY;
                }
                bottom = t + 1;
                shrinkIfIdle();
            }
            pops++;
            return v;
        }
        bottom = t;
        return EMPTY;
    }

    /** Thief: takes the oldest element (FIFO) by CASing `top` forward. */
    int steal() {
        while (true) {
            int t = top;
            int b = bottom;
            if (t >= b) return EMPTY;
            int v = array[t % capacity];
            if (!casTop(t, t + 1)) continue;  // lost the race, re-read and retry
            steals++;
            return v;
        }
    }

    boolean casTop(int expected, int update) {
        if (top != expected) return false;
        top = update;
        return true;
    }

    void grow() {
        int newCapacity = capacity * 2;
        int[] bigger = new int[newCapacity];
        for (int i = top; i < bottom; i++) bigger[i % newCapacity] = array[i % capacity];
        array = bigger;
        capacity = newCapacity;
        grows++;
    }

    /** Only called on an empty deque, so no live index has to move. */
    void shrinkIfIdle() {
        if (capacity <= MIN_CAPACITY) return;
        while (capacity > MIN_CAPACITY && (bottom - top) * 4 < capacity) {
            capacity /= 2;
            shrinks++;
        }
        array = new int[capacity];
    }

    /** A thief's read split into two steps so the race can be scripted. */
    static class Spy {
        int seenTop;
        int seenBottom;
        int value;
    }

    Spy spy() {
        Spy s = new Spy();
        s.seenTop = top;
        s.seenBottom = bottom;
        if (top >= bottom) return null;
        s.value = array[top % capacity];
        return s;
    }

    int commit(Spy s) {
        if (s == null) return EMPTY;
        if (!casTop(s.seenTop, s.seenTop + 1)) return EMPTY;
        steals++;
        return s.value;
    }
}

/** A worker of the simulated task scheduler: it owns one deque. */
class Worker {
    final int id;
    final ChaseLevDeque deque = new ChaseLevDeque();
    final List<Integer> ranLocally = new ArrayList<>();
    int stolen = 0;

    Worker(int id) {
        this.id = id;
    }
}

public class Solution {
    static long seed = 0x123456789ABCDEF0L;

    static int nextInt(int bound) {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Math.floorMod(z, bound);
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    public static void main(String[] args) {
        ownerIsLifo();
        thiefIsFifo();
        emptyAndDegenerate();
        growthAndShrink();
        drainShrinks();
        modelCrossCheck();
        lastElementRace();
        schedulerDemo();
        System.out.println("PASS");
    }

    /** The owner drains its own deque newest-first. */
    static void ownerIsLifo() {
        ChaseLevDeque d = new ChaseLevDeque();
        for (int v = 1; v <= 5; v++) d.push(v);
        check(d.size() == 5, "size after five pushes");
        check(d.grows == 1 && d.capacity == 8, "five elements forced exactly one doubling");
        for (int v = 5; v >= 1; v--) check(d.pop() == v, "LIFO pop expects " + v);
        check(d.pop() == ChaseLevDeque.EMPTY, "owner pop on an empty deque");
        check(d.size() == 0 && d.bottom == d.top, "indices collapse when drained");
        check(d.shrinks == 1 && d.capacity == ChaseLevDeque.MIN_CAPACITY,
                "an emptied deque gives its array back");
    }

    /** Thieves drain the same deque oldest-first. */
    static void thiefIsFifo() {
        ChaseLevDeque d = new ChaseLevDeque();
        for (int v = 1; v <= 5; v++) d.push(v);
        for (int v = 1; v <= 5; v++) check(d.steal() == v, "FIFO steal expects " + v);
        check(d.steal() == ChaseLevDeque.EMPTY, "steal on an empty deque");
        check(d.steals == 5 && d.size() == 0, "five steals recorded");
    }

    /** Both ends on an empty deque, and the two ends interleaved. */
    static void emptyAndDegenerate() {
        ChaseLevDeque d = new ChaseLevDeque();
        check(d.pop() == ChaseLevDeque.EMPTY && d.steal() == ChaseLevDeque.EMPTY, "empty both ends");
        check(d.size() == 0 && d.top == 0 && d.bottom == 0, "an empty deque stays empty");

        d.push(7);
        check(d.steal() == 7 && d.steal() == ChaseLevDeque.EMPTY, "the single element goes to a thief");
        d.push(8);
        check(d.pop() == 8 && d.pop() == ChaseLevDeque.EMPTY, "the single element goes to the owner");

        for (int v = 1; v <= 10; v++) d.push(v);
        check(d.steal() == 1 && d.steal() == 2, "thieves take the oldest first");
        check(d.pop() == 10 && d.pop() == 9, "the owner takes the newest first");
        check(d.steal() == 3 && d.size() == 5, "ends meet in the middle");
    }

    /** Growth and shrink keep every element and its order. */
    static void growthAndShrink() {
        ChaseLevDeque d = new ChaseLevDeque();
        final int n = 300;
        for (int v = 0; v < n; v++) d.push(v);
        check(d.size() == n, "all pushes counted");
        check(d.grows >= 6, "capacity grew repeatedly: " + d.grows);
        check(d.capacity >= d.size(), "capacity always leaves room for one more: " + d.capacity);
        List<Integer> stolen = new ArrayList<>();
        for (int k = 0; k < 150; k++) stolen.add(d.steal());
        check(stolen.get(0) == 0 && stolen.get(149) == 149, "thieves walked the oldest half in order");
        List<Integer> popped = new ArrayList<>();
        while (d.size() > 0) popped.add(d.pop());
        check(popped.get(0) == n - 1 && popped.get(popped.size() - 1) == 150, "owner drained newest-first");
        check(d.size() == 0 && d.pop() == ChaseLevDeque.EMPTY, "drained");
        d.push(999);
        check(d.pop() == 999, "the deque is reusable after draining");
    }

    /** Draining with the owner only: the array must shrink again but stay usable. */
    static void drainShrinks() {
        ChaseLevDeque d = new ChaseLevDeque();
        for (int v = 0; v < 200; v++) d.push(v);
        int peak = d.capacity;
        for (int v = 199; v >= 0; v--) check(d.pop() == v, "owner drain expects " + v);
        check(d.shrinks > 0 && d.capacity < peak, "array shrank after the drain");
        check(d.capacity >= ChaseLevDeque.MIN_CAPACITY, "capacity never falls below the floor");
        d.push(1);
        d.push(2);
        check(d.pop() == 2 && d.steal() == 1, "still usable after shrinking");
    }

    /** A PRNG-driven schedule cross-checked against an ArrayDeque model. */
    static void modelCrossCheck() {
        ChaseLevDeque d = new ChaseLevDeque();
        ArrayDeque<Integer> model = new ArrayDeque<>();
        int next = 0;
        int emptyPops = 0;
        for (int step = 0; step < 4000; step++) {
            int op = nextInt(10);
            if (op < 5 || model.isEmpty()) {
                d.push(next);
                model.addLast(next);
                next++;
            } else if (op < 8) {
                Integer expected = model.pollLast();
                int got = d.pop();
                if (expected == null) {
                    check(got == ChaseLevDeque.EMPTY, "owner pop on empty model");
                    emptyPops++;
                } else {
                    check(got == expected, "owner pop step " + step + ": " + got + " vs " + expected);
                }
            } else {
                Integer expected = model.pollFirst();
                int got = d.steal();
                check(got == expected, "steal step " + step + ": " + got + " vs " + expected);
            }
            check(d.size() == model.size(), "size invariant at step " + step);
            check(d.capacity >= d.size(), "room for one more at step " + step);
        }
        check(next == d.pops + d.steals + model.size(), "pushes == pops + steals + leftovers");
        check(d.steals > 100 && d.pops > 100, "both ends were exercised: " + d.steals + "/" + d.pops);
        check(emptyPops == 0, "the schedule never pops an empty deque");
    }

    /** Owner popping the last element while a thief already read it: exactly one winner. */
    static void lastElementRace() {
        ChaseLevDeque a = new ChaseLevDeque();
        a.push(42);
        ChaseLevDeque.Spy spy = a.spy();
        check(spy != null && spy.value == 42, "the thief reads the last element");
        check(a.pop() == 42, "the owner wins the CAS");
        check(a.steal() == ChaseLevDeque.EMPTY, "the thief is rejected afterwards");
        check(a.commit(spy) == ChaseLevDeque.EMPTY, "the stale thief CAS fails");
        check(a.size() == 0, "the element is handed out exactly once");

        ChaseLevDeque b = new ChaseLevDeque();
        b.push(42);
        ChaseLevDeque.Spy spy2 = b.spy();
        check(b.commit(spy2) == 42, "the thief wins the CAS");
        check(b.pop() == ChaseLevDeque.EMPTY, "the owner gets nothing");
        check(b.size() == 0 && b.pop() == ChaseLevDeque.EMPTY, "the element is handed out once");
        check(a.pops + b.steals == 2, "both races produced exactly one winner");
    }

    /** Three workers, one lopsided workload, every task executed exactly once. */
    static void schedulerDemo() {
        Worker[] workers = {new Worker(0), new Worker(1), new Worker(2)};
        boolean[] executed = new boolean[24];
        int[] perWorker = {12, 8, 4};
        int task = 0;
        for (int w = 0; w < workers.length; w++) {
            for (int k = 0; k < perWorker[w]; k++) workers[w].deque.push(task++);
        }
        check(task == 24, "24 tasks queued");

        int executedCount = 0;
        int guard = 0;
        while (executedCount < 24 && guard++ < 200) {
            boolean progress = false;
            for (int w = 0; w < workers.length; w++) {
                Worker worker = workers[w];
                int t = worker.deque.pop();
                if (t == ChaseLevDeque.EMPTY) {
                    for (int k = 1; k <= workers.length; k++) {   // look for a victim
                        Worker victim = workers[(w + k) % workers.length];
                        int stolenTask = victim.deque.steal();
                        if (stolenTask == ChaseLevDeque.EMPTY) continue;
                        check(!executed[stolenTask], "stolen task " + stolenTask + " ran twice");
                        executed[stolenTask] = true;
                        worker.ranLocally.add(stolenTask);
                        worker.stolen++;
                        executedCount++;
                        progress = true;
                        break;
                    }
                    continue;
                }
                check(!executed[t], "task " + t + " executed twice");
                executed[t] = true;
                worker.ranLocally.add(t);
                executedCount++;
                progress = true;
            }
            if (!progress) break;
        }
        check(executedCount == 24, "every task ran: " + executedCount);
        for (int i = 0; i < 24; i++) check(executed[i], "task " + i + " ran exactly once");

        int steals = workers[0].stolen + workers[1].stolen + workers[2].stolen;
        check(steals >= 1, "load balancing happened: " + steals + " steals");
        check(workers[0].deque.size() == 0 && workers[1].deque.size() == 0
                && workers[2].deque.size() == 0, "all deques drained");

        int[] lo = {0, 12, 20};
        int[] hi = {12, 20, 24};
        List<List<Integer>> own = new ArrayList<>();
        List<List<Integer>> taken = new ArrayList<>();
        for (int w = 0; w < 3; w++) {
            own.add(new ArrayList<>());
            taken.add(new ArrayList<>());
        }
        for (int w = 0; w < 3; w++) {
            for (int t : workers[w].ranLocally) {
                if (t >= lo[w] && t < hi[w]) own.get(w).add(t);
                else taken.get(t < 12 ? 0 : (t < 20 ? 1 : 2)).add(t);
            }
        }
        for (int w = 0; w < 3; w++) {
            List<Integer> local = own.get(w);
            check(!local.isEmpty() && local.get(0) == hi[w] - 1,
                    "worker " + w + " started with its newest task: " + local);
            for (int i = 1; i < local.size(); i++) {
                check(local.get(i) == local.get(0) - i, "worker " + w + " drained its own work LIFO");
            }
        }
        for (int v = 0; v < 3; v++) {
            List<Integer> stolenIds = taken.get(v);
            Collections.sort(stolenIds);
            for (int i = 0; i < stolenIds.size(); i++) {
                check(stolenIds.get(i) == lo[v] + i,
                        "thieves took victim " + v + "'s oldest tasks first: " + stolenIds);
            }
        }
    }
}
