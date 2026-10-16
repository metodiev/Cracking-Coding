/** Task 623 - Implement hazard pointers memory reclamation in concurrent structures. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * Hazard pointers let a lock-free structure reclaim removed nodes without epochs: each
 * participant owns a couple of *hazard registers* and publishes a pointer there before it
 * dereferences the node. `retire` only deposits the node on a per-thread retire list; a scan
 * frees the deposited nodes that no hazard register protects, once a deposit threshold is hit.
 *
 * There are no real threads: an in-memory scheduler interleaves *micro-steps* of simulated
 * participants (protect -> read -> publish -> retire). Every micro-step boundary is a place
 * where a real thread could be descheduled, which is exactly where an unsound reclamation
 * scheme breaks, so the interleavings are scripted or drawn from a fixed PRNG.
 *
 * Sound formulation used here (documented deviation): the walker hazards its successor
 * *before* releasing its predecessor and then validates that the node it read from is still
 * linked, using the node's sticky `unlinked` marker set by the unlinker (the same role the
 * "marked" bit plays in Harris/Michael-Scott lists). Once that validation passes and the
 * successor is hazarded, no scan can free the successor: removing it requires its predecessor
 * to have been unlinked first, and the sticky marker makes that ordering observable.
 *
 * Verified: (1) a threshold-triggered scan frees unprotected deposits but keeps a hazarded
 * node; (2) under a deterministic random schedule every walk observes a contiguous
 * descending run of the chain and never a freed node; (3) accounting - allocated ==
 * live + freed, no double free, drain reclaims everything; (4) with an *eager* (free-on-
 * retire) reclaimer the very same scripted interleaving makes a walker observe a freed node -
 * the negative control proving the checks have teeth.
 */
import java.util.*;

class HpNode {
    final int id;
    final int value;
    HpNode next;
    boolean unlinked;   // sticky marker set when the node leaves the chain
    boolean freed;

    HpNode(int id, int value) {
        this.id = id;
        this.value = value;
    }
}

/** Simulated arena: hazard registers, per-thread retire lists, threshold scans. */
class HpArena {
    static final int SLOTS = 2;          // predecessor + successor (hand over hand)
    final int threads;
    final HpNode[] registers;
    final List<List<HpNode>> retired = new ArrayList<>();
    final int threshold;
    final boolean eager;
    int allocated;
    int freedCount;
    int scans;
    int freesOfHazardedNode;
    boolean doubleFree;
    final List<Integer> freeOrder = new ArrayList<>();

    HpArena(int threads, int threshold, boolean eager) {
        this.threads = threads;
        this.threshold = threshold;
        this.eager = eager;
        this.registers = new HpNode[threads * SLOTS];
        for (int t = 0; t < threads; t++) retired.add(new ArrayList<>());
    }

    HpNode allocate(int value) {
        return new HpNode(allocated++, value);
    }

    HpNode protect(int thread, int slot, HpNode node) {
        registers[thread * SLOTS + slot] = node;
        return node;
    }

    void release(int thread, int slot) {
        registers[thread * SLOTS + slot] = null;
    }

    /** Advance the hand-over-hand window: slot 1 becomes slot 0, slot 1 is cleared. */
    void slide(int thread) {
        registers[thread * SLOTS] = registers[thread * SLOTS + 1];
        registers[thread * SLOTS + 1] = null;
    }

    boolean isHazarded(HpNode node) {
        for (HpNode h : registers) if (h == node) return true;
        return false;
    }

    void retire(int thread, HpNode node) {
        if (node == null) return;
        if (eager) {
            free(node);
            return;
        }
        retired.get(thread).add(node);
        if (retired.get(thread).size() >= threshold) scan(thread);
    }

    /** Free every deposit of `thread` that no hazard register protects. */
    void scan(int thread) {
        scans++;
        List<HpNode> survivors = new ArrayList<>();
        for (HpNode node : retired.get(thread)) {
            if (isHazarded(node)) survivors.add(node);
            else free(node);
        }
        retired.set(thread, survivors);
    }

    void free(HpNode node) {
        if (node.freed) {
            doubleFree = true;
            return;
        }
        if (isHazarded(node)) freesOfHazardedNode++;
        node.freed = true;
        freedCount++;
        freeOrder.add(node.id);
    }

    int retiredCount() {
        int total = 0;
        for (List<HpNode> list : retired) total += list.size();
        return total;
    }

    /** Quiescent point: nobody references anything any more, so every deposit can go. */
    void drain() {
        for (int i = 0; i < registers.length; i++) registers[i] = null;
        for (int t = 0; t < threads; t++) {
            for (HpNode node : retired.get(t)) free(node);
            retired.get(t).clear();
        }
    }

    String freedJoin() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < freeOrder.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(freeOrder.get(i));
        }
        return sb.toString();
    }
}

/** Shared LIFO chain of nodes; the only structure under reclamation. */
class HpStack {
    final HpArena arena;
    HpNode head;
    int pushed;
    int popped;
    int useAfterFree;
    final List<Integer> popOrder = new ArrayList<>();

    HpStack(HpArena arena) {
        this.arena = arena;
    }

    HpNode push(int value) {
        HpNode node = arena.allocate(value);
        node.next = head;
        head = node;
        pushed++;
        return node;
    }
}

abstract class HpFrame {
    final int thread;
    boolean done;

    HpFrame(int thread) {
        this.thread = thread;
    }

    abstract void step();
}

/** pop(): hazard head, hazard head.next, validate, unlink (set marker), retire. */
class PopFrame extends HpFrame {
    final HpStack stack;
    HpNode head;
    HpNode next;
    int stage;

    PopFrame(int thread, HpStack stack) {
        super(thread);
        this.stack = stack;
    }

    void step() {
        switch (stage) {
            case 0:
                head = stack.arena.protect(thread, 0, stack.head);
                stage = head == null ? 9 : 1;
                break;
            case 1:
                next = head.next;
                stage = 2;
                break;
            case 2:
                stack.arena.protect(thread, 1, next);
                stage = 3;
                break;
            case 3:
                if (!head.unlinked && stack.head == head && head.next == next) {
                    head.unlinked = true;      // sticky marker: readers can observe the unlink
                    stack.head = next;
                    stack.popped++;
                    stack.popOrder.add(head.value);
                    stage = 4;
                } else {
                    stack.arena.release(thread, 0);
                    stack.arena.release(thread, 1);
                    stage = 0;
                }
                break;
            case 4:
                stack.arena.retire(thread, head);   // last use of `head` under protection
                stack.arena.release(thread, 0);
                stage = 5;
                break;
            case 5:
                stack.arena.release(thread, 1);
                stage = 9;
                break;
            default:
                done = true;
        }
    }
}

/** walk(): hazard cur, read cur.next, hazard it, validate cur is still linked, visit cur. */
class WalkFrame extends HpFrame {
    final HpStack stack;
    final List<Integer> seen = new ArrayList<>();
    HpNode cur;
    HpNode next;
    int stage;
    int restarts;

    WalkFrame(int thread, HpStack stack) {
        super(thread);
        this.stack = stack;
    }

    void step() {
        switch (stage) {
            case 0:
                cur = stack.arena.protect(thread, 0, stack.head);
                seen.clear();
                stage = 1;
                break;
            case 1:
                if (cur == null) {
                    stage = 9;
                    break;
                }
                next = cur.next;                       // read before protecting the successor
                stack.arena.protect(thread, 1, next);
                stage = 2;
                break;
            case 2:
                if (!cur.unlinked) {
                    stage = 3;
                } else {
                    restarts++;
                    stack.arena.release(thread, 0);
                    stack.arena.release(thread, 1);
                    stage = 0;
                }
                break;
            case 3:
                if (cur.freed) stack.useAfterFree++;   // models a use-after-free read
                seen.add(cur.value);
                stack.arena.release(thread, 0);
                stack.arena.slide(thread);
                cur = next;
                stage = 1;
                break;
            default:
                done = true;
        }
    }
}

public class Solution {
    static long seed = 0x2545F4914F6CDD1DL;

    /** splitmix64: fixed-seed PRNG, so every interleaving is reproducible. */
    static int nextInt(int bound) {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Math.floorMod(z, bound);
    }

    static int runSchedule(HpStack stack, List<HpFrame> frames) {
        int ticks = 0;
        while (true) {
            List<HpFrame> live = new ArrayList<>();
            for (HpFrame f : frames) if (!f.done) live.add(f);
            if (live.isEmpty()) return ticks;
            for (int i = live.size() - 1; i > 0; i--) {     // Fisher-Yates with the PRNG
                int j = nextInt(i + 1);
                HpFrame tmp = live.get(i);
                live.set(i, live.get(j));
                live.set(j, tmp);
            }
            for (HpFrame f : live) f.step();
            ticks++;
            check(ticks < 5000, "schedule terminates");
        }
    }

    static boolean descendingRunToZero(List<Integer> seen) {
        for (int i = 0; i < seen.size(); i++) {
            if (seen.get(i) != seen.get(0) - i) return false;
        }
        return seen.isEmpty() || seen.get(seen.size() - 1) == 0;
    }

    public static void main(String[] args) {
        thresholdBoundary();
        scriptedEagerContrast();
        deterministicSchedule();
        System.out.println("PASS");
    }

    /** Retire below the threshold frees nothing; a hazarded deposit survives the scan. */
    static void thresholdBoundary() {
        HpArena arena = new HpArena(1, 3, false);
        HpNode a = arena.allocate(1);
        HpNode b = arena.allocate(2);
        HpNode c = arena.allocate(3);
        arena.retire(0, a);
        arena.retire(0, b);
        check(arena.scans == 0 && arena.freedCount == 0, "no scan below threshold");
        arena.protect(0, 0, b);
        arena.retire(0, c);                     // third deposit reaches the threshold
        check(arena.scans == 1, "threshold triggers exactly one scan");
        check(arena.freedCount == 2, "unprotected deposits reclaimed");
        check(arena.retiredCount() == 1, "hazarded deposit survives the scan");
        check(arena.freedJoin().equals("0,2"), "scan frees a and c, not b");
        check(!b.freed && arena.isHazarded(b), "hazard register still protects b");
        arena.drain();
        check(arena.freedCount == 3 && !arena.doubleFree, "drain reclaims the survivor once");
        check(arena.freesOfHazardedNode == 0, "no node freed while hazarded");
    }

    /**
     * One scripted interleaving, run twice: with hazard pointers and with an eager
     * free-on-retire reclaimer. The walker validates while still protected, then visits one
     * micro-step later - the popper gets the whole pop in between.
     */
    static void scriptedEagerContrast() {
        check(contrast(false) == 0, "hazard pointers hide the freed node");
        check(contrast(true) == 1, "eager reclaimer exposes the freed node");
    }

    static int contrast(boolean eager) {
        HpArena arena = new HpArena(2, 8, eager);
        HpStack stack = new HpStack(arena);
        for (int v = 0; v < 3; v++) stack.push(v);          // head is 2
        WalkFrame walk = new WalkFrame(0, stack);
        PopFrame pop = new PopFrame(1, stack);
        walk.step();                                        // hazard head
        walk.step();                                        // read + hazard successor
        walk.step();                                        // validate
        for (int i = 0; i < 6; i++) {                       // full pop while the walker waits
            pop.step();
            if (pop.done) break;
        }
        check(stack.popped == 1, "pop completed during the walk");
        check(arena.freedCount == (eager ? 1 : 0), "reclaim happened only when eager");
        walk.step();                                        // visit the node
        return stack.useAfterFree;
    }

    /** Random-but-reproducible interleaving of pushes, pops and two walkers. */
    static void deterministicSchedule() {
        final int n = 30;
        final int pops = 8;
        HpArena arena = new HpArena(4, 3, false);
        HpStack stack = new HpStack(arena);
        for (int v = 0; v < n; v++) stack.push(v);
        check(stack.head.value == n - 1 && stack.head.next.value == n - 2, "push order");

        List<HpFrame> frames = new ArrayList<>();
        for (int i = 0; i < pops; i++) frames.add(new PopFrame(i % 3, stack));
        WalkFrame w1 = new WalkFrame(3, stack);
        WalkFrame w2 = new WalkFrame(3, stack);
        frames.add(w1);
        frames.add(w2);

        int ticks = runSchedule(stack, frames);
        check(ticks > 5 && stack.popped == pops, "schedule ran and every pop completed");

        check(stack.popOrder.size() == pops, "pop results recorded");
        boolean lifo = true;
        for (int i = 0; i < pops; i++) lifo &= stack.popOrder.get(i) == n - 1 - i;
        check(lifo, "LIFO pop order under interleaving: " + stack.popOrder);

        check(stack.useAfterFree == 0, "no walk observed a freed node");
        check(arena.freesOfHazardedNode == 0, "no scan freed a hazarded node");
        check(!arena.doubleFree, "no double free");
        check(descendingRunToZero(w1.seen) && descendingRunToZero(w2.seen),
                "each walk saw a contiguous descending run ending at the tail: "
                        + w1.seen + " / " + w2.seen);
        check(w1.seen.size() >= n - pops && w2.seen.size() >= n - pops,
                "a walk covered the whole remaining chain: " + w1.seen.size());

        for (HpNode cur = stack.head; cur != null; cur = cur.next) {
            check(!cur.freed, "no node reachable from head was reclaimed (node " + cur.id + ")");
        }
        check(arena.scans > 0 && arena.retiredCount() > 0, "scans ran, deposits still pending");

        arena.drain();
        int liveNodes = 0;
        for (HpNode cur = stack.head; cur != null; cur = cur.next) liveNodes++;
        check(liveNodes == n - pops, "chain still holds the un-popped nodes: " + liveNodes);
        check(arena.freedCount == pops, "drain reclaims every popped node exactly once: " + arena.freedCount);
        check(arena.freedCount + liveNodes == arena.allocated, "allocated == live + freed");
        check(!arena.doubleFree && arena.freesOfHazardedNode == 0, "drain is exactly-once");
        check(arena.retiredCount() == 0, "retire lists emptied by the drain");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
