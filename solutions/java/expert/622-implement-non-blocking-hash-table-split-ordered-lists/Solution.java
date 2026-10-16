/** Task 622 - Implement non-blocking hash table (split-ordered lists). (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * A split-ordered list is a lock-free hash table whose buckets are sublists of one sorted list. The order
 * key of a table key is the bit reversal of a bijective mixing of it, and the list is sorted by that
 * "split order"; bucket b of the current table owns exactly the keys whose split-order prefix is b, and
 * its first node is a dummy sitting on the bucket boundary. A bucket is populated by recursively
 * splitting it: the parent bucket is split first and the new boundary dummy is spliced into the parent's
 * sublist, so buckets hang off one another and a bucket can never be missing while a key below it exists.
 * Removal marks a node deleted and then unlinks it; an insert that meets a deleted node unlinks it and
 * retries.
 *
 * Simplified: operations are decomposed into atomic actions - a pointer read, one traversal step, a
 * compare-and-swap, a table resize - and driven by an explicit schedule instead of real threads, and the
 * table is resized only at the start of an insert, with the other operations restarting when the logged
 * table size changes (a real implementation would use the same optimistic table-size validation).
 * Verified against a HashMap oracle on randomized schedules: the list is strictly increasing in split
 * order, every bucket pointer names its own boundary dummy and is reachable from the head, the live node
 * count equals the maintained size, the size counter survives deletions and re-insertions, resizing keeps
 * every key findable, two producers racing to split the same fresh bucket both land (with the failing CAS
 * attempts counted), the whole thing is deterministic, and the invariant checker rejects a hand-broken
 * list so it is not vacuous.
 */
import java.util.*;

class SoNode {
    final long splitKey;
    final int key;                     // -1 for a boundary dummy
    SoNode next;
    boolean deleted;

    SoNode(long splitKey, int key) {
        this.splitKey = splitKey;
        this.key = key;
    }
}

class SoTable {
    static final int MAX_LOG = 6;

    int logSize;                        // the number of buckets is 1 << logSize
    SoNode[] buckets;
    final SoNode head;                  // the dummy of bucket 0, always the first node of the list
    int size;                           // live (not deleted) elements
    long splitCasFailures, linkCasFailures, resizeRestarts, createdDummies;

    SoTable(int logSize) {
        this.logSize = logSize;
        buckets = new SoNode[1 << logSize];
        head = new SoNode(0L, -1);
        buckets[0] = head;
    }

    static long mix(long key) {
        long z = key * 0xBF58476D1CE4E5B9L + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Reversing the mixed key puts the table prefix in the high bits, which is the split order. */
    static long splitOrder(int key) { return Long.reverse(mix(key & 0xFFFFFFFFL)); }

    /** Split keys are unsigned, so every order comparison has to be unsigned too. */
    static boolean less(long a, long b) { return Long.compareUnsigned(a, b) < 0; }

    int bucketOf(long splitKey) { return (int) (splitKey >>> (64 - logSize)); }

    long boundaryOf(int bucket) { return ((long) bucket) << (64 - logSize); }

    boolean casNext(SoNode node, SoNode expected, SoNode update) {
        if (node.next == expected) {
            node.next = update;
            return true;
        }
        return false;
    }

    void ensureCapacity() {
        if (logSize >= MAX_LOG || size + 1 <= buckets.length * 2) return;
        SoNode[] bigger = new SoNode[buckets.length * 2];
        for (int b = 0; b < buckets.length; b++) {
            if (buckets[b] != null) bigger[2 * b] = buckets[b];      // an old boundary stays a boundary
        }
        buckets = bigger;
        logSize++;
    }

    /** The nearest bucket at or above `bucket` that has a dummy; the region starts there. */
    SoNode bucketStart(int bucket) {
        int b = bucket;
        while (b > 0 && buckets[b] == null) b >>= 1;
        return buckets[b];
    }

    boolean lookup(int key) {
        long splitKey = splitOrder(key);
        SoNode start = bucketStart(bucketOf(splitKey));
        if (start == null) return false;
        SoNode node = start.next;
        while (node != null && less(node.splitKey, splitKey)) node = node.next;
        return node != null && node.splitKey == splitKey && !node.deleted;
    }

    int liveCount() {
        int count = 0;
        for (SoNode node = head.next; node != null; node = node.next) {
            if (node.key != -1 && !node.deleted) count++;
        }
        return count;
    }

    /** null when the list is consistent, otherwise the first problem found. */
    String verifyInvariants() {
        Set<SoNode> reachable = Collections.newSetFromMap(new IdentityHashMap<>());
        long previous = head.splitKey;
        for (SoNode node = head; node != null; node = node.next) {
            if (!reachable.add(node)) return "the list contains a cycle";
            if (less(node.splitKey, previous)) return "the list is not sorted by split order";
            previous = node.splitKey;
        }
        if (liveCount() != size) {
            return "the list holds " + liveCount() + " live nodes but the size counter says " + size;
        }
        for (int b = 0; b < buckets.length; b++) {
            SoNode dummy = buckets[b];
            if (dummy == null) continue;
            if (b == 0) {
                if (dummy != head) return "bucket 0 does not start at the head dummy";
                continue;
            }
            if (!reachable.contains(dummy)) return "bucket " + b + " points outside the list";
            if (dummy.key != -1) return "bucket " + b + " points at an element instead of a dummy";
            if (dummy.splitKey != boundaryOf(b)) return "bucket " + b + " points at a foreign boundary";
        }
        for (int b = 1; b < buckets.length; b++) {
            if (buckets[b] == null) continue;
            int ancestor = b >> 1;
            while (ancestor > 0 && buckets[ancestor] == null) ancestor >>= 1;
            boolean reached = false;
            for (SoNode node = buckets[ancestor]; node != null; node = node.next) {
                if (node == buckets[b]) {
                    reached = true;
                    break;
                }
            }
            if (!reached) return "bucket " + b + " is not reachable from its nearest ancestor";
        }
        return null;
    }
}

class SoThread {
    static final int INSERT = 0, REMOVE = 1;

    final int id;
    final int operation;
    final int key;
    int pc;
    int result = -1;                    // 1 = changed the table, 0 = no change, -1 = not finished
    boolean done;
    boolean applied;                    // set by the driver once the effect reached the oracle
    int logSizeSeen = -1;
    long splitKey;
    int bucket, splitBucket;
    SoNode walk, next, newNode;

    SoThread(int id, int operation, int key) {
        this.id = id;
        this.operation = operation;
        this.key = key;
    }

    /** Executes exactly one atomic action; returns true once the operation has completed. */
    boolean step(SoTable table) {
        if (done) return true;
        if (pc > 0 && logSizeSeen != table.logSize) {                  // the table was resized meanwhile
            table.resizeRestarts++;
            pc = 0;
            logSizeSeen = -1;
            return false;
        }
        return operation == INSERT ? stepInsert(table) : stepRemove(table);
    }

    private boolean stepInsert(SoTable table) {
        switch (pc) {
            case 0:
                table.ensureCapacity();
                logSizeSeen = table.logSize;
                splitKey = SoTable.splitOrder(key);
                bucket = table.bucketOf(splitKey);
                splitBucket = bucket;
                newNode = new SoNode(splitKey, key);
                pc = 1;
                break;
            case 1:
                if (table.buckets[bucket] != null) {
                    pc = 5;
                } else {
                    splitBucket = bucket;                                  // split the chain from the top down
                    pc = 2;
                }
                break;
            case 2:
                if (table.buckets[splitBucket >> 1] == null) {
                    splitBucket >>= 1;                                     // descend to an existing ancestor
                    pc = 2;
                } else {
                    walk = table.buckets[splitBucket >> 1];
                    pc = 3;
                }
                break;
            case 3:
                next = walk.next;
                pc = 4;
                break;
            case 4:
                if (next != null && SoTable.less(next.splitKey, table.boundaryOf(splitBucket))) {
                    walk = next;
                    pc = 3;
                } else {
                    SoNode dummy = new SoNode(table.boundaryOf(splitBucket), -1);
                    dummy.next = next;                                 // splice, never truncate
                    if (table.casNext(walk, next, dummy)) {
                        table.buckets[splitBucket] = dummy;
                        table.createdDummies++;
                        pc = 1;
                    } else {
                        table.splitCasFailures++;
                        pc = 1;                                            // somebody else split it first
                    }
                }
                break;
            case 5:
                walk = table.buckets[bucket];
                pc = walk == null ? 1 : 6;
                break;
            case 6:
                next = walk.next;
                pc = 8;
                break;
            case 8:
                if (next != null && SoTable.less(next.splitKey, splitKey)) {
                    walk = next;
                    pc = 6;
                } else {
                    pc = 9;
                }
                break;
            case 9:
                if (next != null && next.splitKey == splitKey) {
                    if (next.deleted) {
                        if (!table.casNext(walk, next, next.next)) table.linkCasFailures++;
                        pc = 5;                                            // unlink the corpse and retry
                    } else {
                        result = 0;                                        // already present
                        done = true;
                    }
                } else {
                    pc = 10;
                }
                break;
            default:
                newNode.next = next;                                   // splice, never truncate
                if (table.casNext(walk, next, newNode)) {
                    table.size++;
                    result = 1;
                    done = true;
                } else {
                    table.linkCasFailures++;
                    pc = 5;
                }
                break;
        }
        return done;
    }

    private boolean stepRemove(SoTable table) {
        switch (pc) {
            case 0:
                logSizeSeen = table.logSize;
                splitKey = SoTable.splitOrder(key);
                bucket = table.bucketOf(splitKey);
                pc = 1;
                break;
            case 1:
                walk = table.bucketStart(bucket);
                if (walk == null) {
                    result = 0;
                    pc = 6;
                } else {
                    pc = 2;
                }
                break;
            case 2:
                next = walk.next;
                pc = 3;
                break;
            case 3:
                if (next != null && SoTable.less(next.splitKey, splitKey)) {
                    walk = next;
                    pc = 2;
                } else {
                    pc = 4;
                }
                break;
            case 4:
                if (next == null || next.splitKey != splitKey || next.deleted) {
                    result = 0;
                    pc = 6;
                } else {
                    next.deleted = true;                                   // the linearization point
                    table.size--;
                    result = 1;
                    pc = 5;
                }
                break;
            case 5:
                if (!table.casNext(walk, next, next.next)) {
                    table.linkCasFailures++;                               // the corpse stays; inserts sweep it up
                }
                pc = 6;
                break;
            default:
                done = true;
                break;
        }
        return done;
    }
}

public class Solution {
    private static long state = 0xC13FA5B17D4E2F69L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Runs schedule steps, checking the invariants after each step and feeding the oracle on completion. */
    static void runSchedule(SoTable table, List<SoThread> threads, int[] schedule, int rounds, String label,
                            Set<Integer> oracle) {
        SoThread[] byId = new SoThread[threads.size()];
        for (SoThread thread : threads) byId[thread.id] = thread;
        for (int round = 0; round < rounds; round++) {
            for (int id : schedule) {
                SoThread thread = byId[id];
                if (thread == null || thread.done) continue;
                thread.step(table);
                check(table.verifyInvariants() == null, label + ": " + table.verifyInvariants());
                if (thread.done && !thread.applied && oracle != null) {
                    thread.applied = true;
                    if (thread.result == 1) {
                        if (thread.operation == SoThread.INSERT) oracle.add(thread.key);
                        else oracle.remove(thread.key);
                    }
                }
            }
        }
    }

    static int completions(List<SoThread> threads) {
        int done = 0;
        for (SoThread thread : threads) if (thread.done) done++;
        return done;
    }

    /** Drives the schedule until every operation has completed, applying the oracle effects. */
    static void drain(SoTable table, List<SoThread> threads, int[] schedule, Set<Integer> oracle,
                      String label) {
        for (int round = 0; round < 400 && completions(threads) < threads.size(); round++) {
            runSchedule(table, threads, schedule, 1, label, oracle);
        }
        check(completions(threads) == threads.size(), label + ": every operation completed");
    }

    static void checkAgreesWithOracle(SoTable table, Set<Integer> oracle, int domain, String label) {
        for (int key = 0; key <= domain; key++) {
            check(table.lookup(key) == oracle.contains(key),
                    label + ": key " + key + " should be " + (oracle.contains(key) ? "present" : "absent"));
        }
        check(table.liveCount() == oracle.size(), label + ": the live count matches the oracle");
        check(table.verifyInvariants() == null, label + ": " + table.verifyInvariants());
    }

    static SoThread single(int operation, int key) {
        return new SoThread(0, operation, key);
    }

    public static void main(String[] args) {
        // 1. the split order is injective and the bucket is the table prefix of the split key
        Set<Long> orders = new HashSet<>();
        for (int key = 0; key < 2000; key++) orders.add(SoTable.splitOrder(key));
        check(orders.size() == 2000, "the split order is injective on the test domain");
        SoTable probe = new SoTable(3);
        for (int key = 0; key < 512; key++) {
            long splitKey = SoTable.splitOrder(key);
            check(probe.bucketOf(splitKey) == (int) (splitKey >>> (64 - probe.logSize)),
                    "the bucket is the high bits of the split key");
        }

        // 2. a sequential run: insert a spread of keys, delete some, re-insert one of them
        SoTable table = new SoTable(2);
        Set<Integer> oracle = new TreeSet<>();
        for (int key = 0; key < 24; key += 2) {
            SoThread insert = single(SoThread.INSERT, key);
            drain(table, Arrays.asList(insert), new int[]{0}, oracle, "sequential insert " + key);
            check(insert.result == 1, "the insert of " + key + " changed the table");
        }
        checkAgreesWithOracle(table, oracle, 30, "after the sequential inserts");
        check(table.logSize > 2, "the table grew while inserting");
        for (int key = 0; key < 24; key += 4) {
            SoThread remove = single(SoThread.REMOVE, key);
            drain(table, Arrays.asList(remove), new int[]{0}, oracle, "sequential remove " + key);
            check(remove.result == 1, "the removal of " + key + " changed the table");
        }
        checkAgreesWithOracle(table, oracle, 30, "after the sequential removals");
        SoThread again = single(SoThread.INSERT, 8);
        drain(table, Arrays.asList(again), new int[]{0}, oracle, "re-insert 8");
        check(again.result == 1, "the deleted key can be inserted again");
        SoThread duplicate = single(SoThread.INSERT, 8);
        drain(table, Arrays.asList(duplicate), new int[]{0}, oracle, "duplicate insert");
        check(duplicate.result == 0, "a duplicate insert reports no change");
        SoThread missing = single(SoThread.REMOVE, 7);
        drain(table, Arrays.asList(missing), new int[]{0}, oracle, "remove of an absent key");
        check(missing.result == 0, "removing an absent key reports no change");
        checkAgreesWithOracle(table, oracle, 30, "after the duplicate and the absent removal");
        check(table.lookup(8) && !table.lookup(7), "the direct lookups agree");

        // 3. two producers racing to split the same fresh bucket
        SoTable picker = new SoTable(1);
        List<Integer> candidates = new ArrayList<>();
        int targetBucket = -1;
        for (int key = 0; key < 400 && candidates.size() < 2; key++) {
            int bucket = picker.bucketOf(SoTable.splitOrder(key));
            if (bucket == 0) continue;
            if (targetBucket < 0) {
                targetBucket = bucket;
                candidates.add(key);
            } else if (bucket == targetBucket) {
                candidates.add(key);
            }
        }
        check(candidates.size() == 2 && targetBucket != 0, "two keys share one fresh bucket");
        SoTable raced = new SoTable(1);
        Set<Integer> racedOracle = new TreeSet<>();
        List<SoThread> racers = new ArrayList<>();
        racers.add(single(SoThread.INSERT, candidates.get(0)));
        racers.add(new SoThread(1, SoThread.INSERT, candidates.get(1)));
        drain(raced, racers, new int[]{0, 1}, racedOracle, "racing split");
        check(racedOracle.equals(new TreeSet<>(candidates)), "both racing keys landed");
        check(raced.createdDummies > 0, "the racing split created boundary dummies");
        checkAgreesWithOracle(raced, racedOracle, 31, "after the racing split");

        // 4. randomized mixed schedules against an oracle, including a resizing table
        long totalSplitFailures = 0, totalLinkFailures = 0, totalResizeRestarts = 0;
        for (int trial = 0; trial < 30; trial++) {
            SoTable random = new SoTable(1);
            Set<Integer> model = new TreeSet<>();
            List<SoThread> mixed = new ArrayList<>();
            int count = 3 + (int) ((nextLong() >>> 33) % 3);
            for (int i = 0; i < count; i++) {
                int key = (int) ((nextLong() >>> 33) % 20);
                boolean insert = ((nextLong() >>> 33) & 1L) == 0;
                mixed.add(new SoThread(i, insert ? SoThread.INSERT : SoThread.REMOVE, key));
            }
            int[] schedule = new int[24];
            for (int i = 0; i < schedule.length; i++) {
                schedule[i] = (int) ((nextLong() >>> 33) % mixed.size());
            }
            drain(random, mixed, schedule, model, "random trial " + trial);
            checkAgreesWithOracle(random, model, 25, "random trial " + trial);
            totalSplitFailures += random.splitCasFailures;
            totalLinkFailures += random.linkCasFailures;
            totalResizeRestarts += random.resizeRestarts;
        }

        // 5. one bigger load in a table that starts small and resizes several times
        SoTable load = new SoTable(1);
        Set<Integer> loadOracle = new TreeSet<>();
        for (int key = 0; key < 60; key++) {
            SoThread insert = single(SoThread.INSERT, key);
            drain(load, Arrays.asList(insert), new int[]{0}, loadOracle, "load insert " + key);
        }
        checkAgreesWithOracle(load, loadOracle, 70, "after the load");
        check(load.logSize >= 5, "the table grew to hold the load");
        for (int key = 0; key < 60; key += 3) {
            SoThread remove = single(SoThread.REMOVE, key);
            drain(load, Arrays.asList(remove), new int[]{0}, loadOracle, "load remove " + key);
        }
        checkAgreesWithOracle(load, loadOracle, 70, "after the load removals");
        for (int key = 58; key < 60; key++) {
            SoThread reinsert = single(SoThread.INSERT, key - 40);
            drain(load, Arrays.asList(reinsert), new int[]{0}, loadOracle, "load reinsert");
        }
        checkAgreesWithOracle(load, loadOracle, 70, "after the load re-insertions");

        // 6. determinism: the same mixed schedule twice reaches the same table
        SoTable firstRun = new SoTable(2), secondRun = new SoTable(2);
        List<SoThread> runA = new ArrayList<>(), runB = new ArrayList<>();
        int[] mix = new int[]{0, 1, 2, 1, 0, 2, 2, 1, 0, 1, 2, 0};
        for (int i = 0; i < 4; i++) {
            int operation = i % 2 == 0 ? SoThread.INSERT : SoThread.REMOVE;
            runA.add(new SoThread(i, operation, i * 3));
            runB.add(new SoThread(i, operation, i * 3));
        }
        Set<Integer> oracleA = new TreeSet<>(), oracleB = new TreeSet<>();
        runSchedule(firstRun, runA, mix, 12, "determinism A", oracleA);
        runSchedule(secondRun, runB, mix, 12, "determinism B", oracleB);
        for (int key = 0; key <= 12; key++) {
            check(firstRun.lookup(key) == secondRun.lookup(key), "determinism for key " + key);
        }
        check(firstRun.createdDummies == secondRun.createdDummies && firstRun.size == secondRun.size
                && oracleA.equals(oracleB), "the same schedule produces the same table");

        // 7. the invariant checker is not vacuous
        SoTable broken = new SoTable(2);
        SoNode lower = new SoNode(16L, 1);
        SoNode higher = new SoNode(32L, 2);
        broken.casNext(broken.head, null, lower);
        broken.casNext(lower, null, higher);
        broken.size = 2;
        check(broken.verifyInvariants() == null, "the hand built list of two keys is consistent");
        broken.casNext(broken.head, lower, higher);                    // drop the lower key from the list
        String problem = broken.verifyInvariants();
        check(problem != null && problem.contains("live nodes"), "the size mismatch is reported");

        System.out.println("random schedules: split CAS failures " + totalSplitFailures
                + ", link CAS failures " + totalLinkFailures + ", resize restarts " + totalResizeRestarts);
        System.out.println("load table: log size " + load.logSize + ", size " + load.size
                + ", live nodes " + load.liveCount() + ", dummies " + load.createdDummies);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
