/** Task 621 - Implement scalable concurrent queue (Michael-Scott queue). (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * The Michael-Scott queue is simulated on one thread by decomposing every operation into its atomic
 * actions (a pointer read, a pointer compare-and-swap) and driving them from an explicit schedule: a
 * schedule step executes exactly one action of one thread's operation, so the interleavings that make
 * the algorithm interesting - a thread stalled between reading the tail and linking its node, a tail
 * pointer that lags behind the last node, a dequeue that has to help advance the tail - are reproduced
 * exactly instead of being left to chance. compareAndSwap is the real primitive, evaluated against the
 * simulated shared pointer it names, so a stale read really does fail the CAS.
 *
 * Verified: the chain reachable from head never contains a cycle, the tail is always reachable, and
 * exactly as many nodes are linked as there are successful linking CASes; the value sequence of the
 * chain equals the tail of the linking order, and the returned values form the matching prefix, which
 * is the FIFO property together with "no element is lost or duplicated"; the stalled-producer and
 * lagging-tail schedules exercise the helping paths (counted) and still keep every element; empty
 * dequeues report EMPTY; and a hand-corrupted chain is reported by the same invariant checker, so the
 * checker is not vacuous.
 */
import java.util.*;

class MsNode {
    final int value;                   // -1 marks the dummy head node, which stores no element
    MsNode next;

    MsNode(int value) {
        this.value = value;
    }
}

class MsQueueSimulator {
    static final int EMPTY = -1;
    static final int ENQUEUE = 0, DEQUEUE = 1;

    MsNode head;
    MsNode tail;
    int linkedCount;
    int headAdvanceCount;
    final List<Integer> linkedOrder = new ArrayList<>();      // the linearization order of enqueue
    final List<Integer> returnedOrder = new ArrayList<>();    // the completed dequeues, in order
    long helperTailAdvances;                                  // successful tail CASes that only help
    long restartsOnStaleTail;

    MsQueueSimulator() {
        MsNode dummy = new MsNode(EMPTY);
        head = dummy;
        tail = dummy;
    }

    /** The real compare-and-swap semantics on the simulated shared pointer. */
    boolean casNext(MsNode node, MsNode expected, MsNode update) {
        if (node.next == expected) {
            node.next = update;
            return true;
        }
        return false;
    }

    boolean casHead(MsNode expected, MsNode update) {
        if (head == expected) {
            head = update;
            return true;
        }
        return false;
    }

    boolean casTail(MsNode expected, MsNode update) {
        if (tail == expected) {
            tail = update;
            return true;
        }
        return false;
    }

    /** The queue contents: the values of the nodes after head, in chain order. */
    List<Integer> contents() {
        List<Integer> values = new ArrayList<>();
        for (MsNode node = head.next; node != null; node = node.next) values.add(node.value);
        return values;
    }

    int nodeCount() {
        int count = 0;
        for (MsNode node = head; node != null; node = node.next) count++;
        return count;
    }

    /** Returns null when the structure is consistent, or a message describing the first problem. */
    String verifyInvariants() {
        Set<MsNode> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        int nodes = 0;
        for (MsNode node = head; node != null; node = node.next) {
            if (!seen.add(node)) return "the chain from head contains a cycle";
            nodes++;
        }
        if (!seen.contains(tail)) return "the tail is not reachable from head";
        int expectedNodes = linkedCount - headAdvanceCount + 1;
        if (nodes != expectedNodes) {
            return "the chain has " + nodes + " nodes but " + expectedNodes + " were linked and not dequeued";
        }
        List<Integer> values = contents();
        int offset = linkedOrder.size() - values.size();
        if (offset < 0) return "the chain holds more values than were linked";
        for (int i = 0; i < values.size(); i++) {
            if (linkedOrder.get(offset + i).intValue() != values.get(i).intValue()) {
                return "the chain is not the tail of the linking order";
            }
        }
        return null;
    }
}

class MsThread {
    final int id;
    final int operation;
    final int argument;
    int pc;
    MsNode localTail, localNext, localHead, localNode;
    int result = -2;
    boolean done;
    long links, headAdvances, helped;

    MsThread(int id, int operation, int argument) {
        this.id = id;
        this.operation = operation;
        this.argument = argument;
        if (operation == MsQueueSimulator.ENQUEUE) localNode = new MsNode(argument);
    }

    /** Executes one atomic action; returns true once the operation has completed. */
    boolean step(MsQueueSimulator sim) {
        if (done) return true;
        if (operation == MsQueueSimulator.ENQUEUE) return stepEnqueue(sim);
        return stepDequeue(sim);
    }

    private boolean stepEnqueue(MsQueueSimulator sim) {
        switch (pc) {
            case 0:
                localTail = sim.tail;
                pc = 1;
                break;
            case 1:
                localNext = localTail.next;
                pc = 2;
                break;
            case 2:
                if (localTail == sim.tail) {
                    pc = 3;
                } else {
                    sim.restartsOnStaleTail++;
                    pc = 0;
                }
                break;
            case 3:
                pc = localNext == null ? 4 : 5;
                break;
            case 4:
                if (sim.casNext(localTail, null, localNode)) {
                    sim.linkedCount++;
                    sim.linkedOrder.add(argument);
                    links++;
                    pc = 6;
                } else {
                    pc = 0;
                }
                break;
            case 5:
                if (sim.casTail(localTail, localNext)) {
                    sim.helperTailAdvances++;
                    helped++;
                }
                pc = 0;
                break;
            default:
                if (sim.casTail(localTail, localNode)) {
                    sim.helperTailAdvances++;
                    helped++;
                }
                done = true;
                break;
        }
        return done;
    }

    private boolean stepDequeue(MsQueueSimulator sim) {
        switch (pc) {
            case 0:
                localHead = sim.head;
                pc = 1;
                break;
            case 1:
                localTail = sim.tail;
                pc = 2;
                break;
            case 2:
                localNext = localHead.next;
                pc = 3;
                break;
            case 3:
                if (localHead == sim.head) {
                    pc = 4;
                } else {
                    sim.restartsOnStaleTail++;
                    pc = 0;
                }
                break;
            case 4:
                pc = localNext == null ? 8 : 5;
                break;
            case 5:
                pc = localHead == localTail ? 6 : 7;
                break;
            case 6:
                if (sim.casTail(localTail, localNext)) {
                    sim.helperTailAdvances++;
                    helped++;
                }
                pc = 0;
                break;
            default:
                result = localNext.value;
                if (sim.casHead(localHead, localNext)) {
                    headAdvances++;
                    sim.headAdvanceCount++;
                    sim.returnedOrder.add(result);
                    done = true;
                } else {
                    pc = 0;
                }
                break;
        }
        if (pc == 8) {
            result = MsQueueSimulator.EMPTY;
            done = true;
        }
        return done;
    }
}

public class Solution {
    private static long state = 0x7F4A7C15BF58476DL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Runs the schedule (repeatedly, for stalls and retries) and checks the invariants every step. */
    static void runSchedule(MsQueueSimulator sim, List<MsThread> threads, int[] schedule, int rounds,
                            String label) {
        MsThread[] byId = new MsThread[threads.size()];
        for (MsThread thread : threads) byId[thread.id] = thread;
        for (int round = 0; round < rounds; round++) {
            for (int id : schedule) {
                MsThread thread = byId[id];
                if (thread == null || thread.done) continue;
                thread.step(sim);
                String problem = sim.verifyInvariants();
                check(problem == null, label + ": " + problem);
            }
        }
    }

    static int completions(List<MsThread> threads) {
        int done = 0;
        for (MsThread thread : threads) if (thread.done) done++;
        return done;
    }

    static void checkFinalState(MsQueueSimulator sim, List<MsThread> threads, String label) {
        check(completions(threads) == threads.size(), label + ": every operation completed");
        check(sim.verifyInvariants() == null, label + ": " + sim.verifyInvariants());
        for (int i = 0; i < sim.returnedOrder.size(); i++) {
            check(sim.returnedOrder.get(i).intValue() == sim.linkedOrder.get(i).intValue(),
                    label + ": the dequeued values are a prefix of the linking order");
        }
        List<Integer> expected = new ArrayList<>(sim.linkedOrder.subList(sim.returnedOrder.size(),
                sim.linkedOrder.size()));
        check(sim.contents().equals(expected), label + ": the remaining chain continues the order");
    }

    public static void main(String[] args) {
        // 1. a purely sequential run: enqueue 1..5, then dequeue them in FIFO order
        MsQueueSimulator sequential = new MsQueueSimulator();
        List<MsThread> threads = new ArrayList<>();
        for (int i = 0; i < 5; i++) threads.add(new MsThread(i, MsQueueSimulator.ENQUEUE, i + 1));
        for (int i = 5; i < 10; i++) threads.add(new MsThread(i, MsQueueSimulator.DEQUEUE, 0));
        for (MsThread thread : threads) {
            int[] schedule = new int[]{thread.id};
            runSchedule(sequential, threads, schedule, 40, "sequential " + thread.id);
        }
        checkFinalState(sequential, threads, "sequential");
        check(sequential.returnedOrder.equals(Arrays.asList(1, 2, 3, 4, 5)), "the FIFO order");
        check(sequential.contents().isEmpty() && sequential.nodeCount() == 1,
                "an empty queue whose chain is just the last dequeued node");

        // 2. a producer stalled between reading the tail and linking its node
        MsQueueSimulator stalled = new MsQueueSimulator();
        List<MsThread> stallThreads = new ArrayList<>();
        MsThread producer = new MsThread(0, MsQueueSimulator.ENQUEUE, 7);
        MsThread fast = new MsThread(1, MsQueueSimulator.ENQUEUE, 8);
        stallThreads.add(producer);
        stallThreads.add(fast);
        runSchedule(stalled, stallThreads, new int[]{0}, 1, "the producer reads the tail");
        check(producer.pc == 1 && !producer.done, "the producer has read the tail and nothing else");
        runSchedule(stalled, stallThreads, new int[]{1}, 40, "the fast producer finishes");
        check(fast.done && stalled.contents().equals(Arrays.asList(8)), "the fast producer is in the queue");
        runSchedule(stalled, stallThreads, new int[]{0}, 40, "the stalled producer resumes");
        checkFinalState(stalled, stallThreads, "stalled producer");
        check(stalled.contents().equals(Arrays.asList(8, 7)), "both elements are queued in linking order");
        check(stalled.restartsOnStaleTail > 0, "the stale tail was detected and the operation restarted");

        // 3. a dequeue that finds head == tail and helps advance the tail
        MsQueueSimulator lagging = new MsQueueSimulator();
        List<MsThread> lagThreads = new ArrayList<>();
        MsThread lazy = new MsThread(0, MsQueueSimulator.ENQUEUE, 6);
        MsThread consumer = new MsThread(1, MsQueueSimulator.DEQUEUE, 0);
        MsThread later = new MsThread(2, MsQueueSimulator.ENQUEUE, 7);
        lagThreads.add(lazy);
        lagThreads.add(consumer);
        lagThreads.add(later);
        for (int i = 0; i < 5; i++) runSchedule(lagging, lagThreads, new int[]{0}, 1, "lazy link");
        check(!lazy.done && lazy.pc == 6, "the lazy producer linked its node but did not move the tail");
        check(lagging.head == lagging.tail, "the tail pointer still points at the head");
        runSchedule(lagging, lagThreads, new int[]{1}, 40, "the consumer");
        check(consumer.done && consumer.result == 6, "the consumer took the only element");
        runSchedule(lagging, lagThreads, new int[]{0}, 40, "the lazy producer finishes");
        runSchedule(lagging, lagThreads, new int[]{2}, 40, "a later producer");
        checkFinalState(lagging, lagThreads, "lagging tail");
        check(lagging.helperTailAdvances > 0, "some CAS only helped move the tail");
        check(lagging.contents().equals(Arrays.asList(7)), "the later element is queued");

        // 4. an empty dequeue reports EMPTY
        MsQueueSimulator empty = new MsQueueSimulator();
        List<MsThread> onlyReader = new ArrayList<>();
        MsThread reader = new MsThread(0, MsQueueSimulator.DEQUEUE, 0);
        onlyReader.add(reader);
        runSchedule(empty, onlyReader, new int[]{0}, 20, "empty dequeue");
        check(reader.done && reader.result == MsQueueSimulator.EMPTY, "an empty queue gives EMPTY");
        check(empty.returnedOrder.isEmpty() && empty.nodeCount() == 1, "nothing was returned");

        // 5. randomized aggressive schedules over several threads
        long totalHelpers = 0;
        long totalRestarts = 0;
        for (int trial = 0; trial < 20; trial++) {
            MsQueueSimulator sim = new MsQueueSimulator();
            List<MsThread> mixed = new ArrayList<>();
            int count = 3 + (int) ((nextLong() >>> 33) % 4);
            for (int i = 0; i < count; i++) {
                boolean enqueue = ((nextLong() >>> 33) & 1L) == 0 || i == 0;
                mixed.add(new MsThread(i, enqueue ? MsQueueSimulator.ENQUEUE : MsQueueSimulator.DEQUEUE,
                        i + 1));
            }
            int[] schedule = new int[40];
            for (int i = 0; i < schedule.length; i++) schedule[i] = (int) ((nextLong() >>> 33) % mixed.size());
            for (int round = 0; round < 30 && completions(mixed) < mixed.size(); round++) {
                runSchedule(sim, mixed, schedule, 1, "random trial " + trial);
            }
            check(completions(mixed) == mixed.size(), "trial " + trial + ": every operation completed");
            checkFinalState(sim, mixed, "random trial " + trial);
            check(sim.returnedOrder.size() + sim.contents().size() == sim.linkedCount,
                    "trial " + trial + ": no element is lost or duplicated");
            check(sim.returnedOrder.size() == sim.headAdvanceCount,
                    "trial " + trial + ": one return per head advance");
            check(sim.nodeCount() == sim.linkedCount - sim.headAdvanceCount + 1,
                    "trial " + trial + ": one node per link plus the dummy");
            totalHelpers += sim.helperTailAdvances;
            totalRestarts += sim.restartsOnStaleTail;
        }
        check(totalHelpers > 0, "the helping paths were exercised across the trials");
        check(totalRestarts > 0, "stale reads were detected across the trials");

        // 6. the invariant checker is not vacuous: a hand-corrupted chain is reported
        MsQueueSimulator hand = new MsQueueSimulator();
        MsNode first = new MsNode(1);
        MsNode second = new MsNode(2);
        hand.casNext(hand.head, null, first);
        hand.casNext(first, null, second);
        hand.casTail(hand.head, second);
        hand.linkedCount = 2;
        hand.linkedOrder.add(1);
        hand.linkedOrder.add(2);
        check(hand.verifyInvariants() == null, "a hand built queue of two elements is consistent");
        check(hand.contents().equals(Arrays.asList(1, 2)), "the hand built queue holds both elements");
        hand.casNext(hand.head, first, second);                      // bypass the first node
        check(hand.verifyInvariants() != null, "bypassing a node is detected");
        check(hand.verifyInvariants().contains("nodes"), "the reported problem is about the node count");
        System.out.println("random schedules: helper tail advances " + totalHelpers
                + ", stale-tail restarts " + totalRestarts);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
