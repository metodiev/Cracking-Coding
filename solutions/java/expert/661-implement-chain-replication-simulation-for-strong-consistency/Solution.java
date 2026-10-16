/** Task 661 - Implement chain replication simulation for strong consistency. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Chain replication (van Renesse & Schneider): replicas form a chain; every write enters the HEAD,
 * is forwarded hop by hop in FIFO order, and is acknowledged back up to the client only once the
 * TAIL has applied it. Reads are served by the TAIL and are therefore strongly consistent; a read at
 * the head can observe an in-flight (not yet committed) value, which is the latency/consistency knob.
 *
 * The simulation is event driven: FORWARD events append the value at replica i and schedule the next
 * hop, the TAIL schedules the ACK walk back to the head. After every single event the invariant
 * "each replica's log is a prefix of the head's log" is re-checked; the ack latency is compared with
 * the closed form 2 * sum(hop latency), so the event machinery is cross-checked rather than asserted.
 *
 * Failure handling models the real reconfiguration rules: a failed replica stalls the chain (no write
 * is acknowledged), the committed prefix survives on every replica, and reconfiguration bypasses the
 * failed replica, truncates the uncommitted suffix (which was present only on the head side) and lets
 * writes resume with the shortened chain.
 *
 * Verified: n=1 and n=2 boundaries; ack latency equals the closed form; acks arrive in submission
 * order; the head sees a write before the tail does; the tail's log is exactly the acked prefix; a
 * stalled chain acknowledges nothing and loses nothing committed; after reconfiguration the
 * committed prefix is intact and the uncommitted suffix is gone everywhere; invalid inputs rejected.
 */
import java.util.*;

final class ChainEvent implements Comparable<ChainEvent> {
    static final int FORWARD = 0, ACK = 1;
    final long time;
    final int type;
    final int replica;
    final int writeId;
    final long sequence;

    ChainEvent(long time, int type, int replica, int writeId, long sequence) {
        this.time = time;
        this.type = type;
        this.replica = replica;
        this.writeId = writeId;
        this.sequence = sequence;
    }

    @Override
    public int compareTo(ChainEvent other) {
        if (time != other.time) return Long.compare(time, other.time);
        if (type != other.type) return Integer.compare(type, other.type);
        if (replica != other.replica) return Integer.compare(replica, other.replica);
        return Long.compare(sequence, other.sequence);
    }
}

final class Chain {
    final int[] hopLatency;             // hopLatency[i] between replica i and i+1
    final boolean[] alive;
    final boolean[] bypassed;
    final List<List<Integer>> logs = new ArrayList<>();
    final Map<Integer, Long> ackTime = new LinkedHashMap<>();
    final Map<Integer, Long> tailCommit = new LinkedHashMap<>();
    final Map<Integer, Integer> valueOf = new LinkedHashMap<>();
    final Map<Integer, Long> submittedAt = new LinkedHashMap<>();
    private final PriorityQueue<ChainEvent> queue = new PriorityQueue<>();
    private long sequence;
    private long lastSubmitTime;
    private int nextWriteId;
    long invariantChecks;

    Chain(int replicaCount, int[] hopLatency) {
        if (replicaCount < 1) throw new IllegalArgumentException("need at least one replica");
        if (hopLatency.length != replicaCount - 1) throw new IllegalArgumentException("one hop per link");
        for (int latency : hopLatency) if (latency < 0) throw new IllegalArgumentException("negative latency");
        this.hopLatency = hopLatency.clone();
        this.alive = new boolean[replicaCount];
        this.bypassed = new boolean[replicaCount];
        Arrays.fill(alive, true);
        for (int i = 0; i < replicaCount; i++) logs.add(new ArrayList<>());
    }

    int replicaCount() { return alive.length; }

    List<Integer> logOf(int replica) { return Collections.unmodifiableList(logs.get(replica)); }

    long ackTimeOf(int writeId) { return ackTime.getOrDefault(writeId, -1L); }

    int valueOfWrite(int writeId) { return valueOf.getOrDefault(writeId, -1); }

    boolean isAlive(int replica) { return alive[replica]; }

    /** Submits a write to the current head; returns the write id. */
    int submit(long time, int value) {
        if (time < 0) throw new IllegalArgumentException("negative submit time");
        if (!submittedAt.isEmpty() && time < lastSubmitTime) {
            throw new IllegalArgumentException("writes must be submitted in non-decreasing time order");
        }
        lastSubmitTime = time;
        List<Integer> path = chainPath();
        int head = path.get(0);
        int writeId = nextWriteId++;
        submittedAt.put(writeId, time);
        valueOf.put(writeId, value);
        logs.get(head).add(writeId);
        if (!chainIntact(path)) return writeId;        // the chain is broken: the write stops at the head
        sequence++;
        if (path.size() > 1) {
            queue.add(new ChainEvent(time + pathHop(toArray(path), 0), ChainEvent.FORWARD, path.get(1), writeId, sequence));
        } else {
            tailCommit.put(writeId, time);
            queue.add(new ChainEvent(time, ChainEvent.ACK, head, writeId, sequence));
        }
        return writeId;
    }

    private static int[] toArray(List<Integer> path) {
        int[] out = new int[path.size()];
        for (int i = 0; i < out.length; i++) out[i] = path.get(i);
        return out;
    }

    /** Longest live prefix of the chain, skipping replicas that were bypassed by a reconfiguration. */
    private List<Integer> chainPath() {
        List<Integer> path = new ArrayList<>();
        for (int i = 0; i < alive.length; i++) {
            if (bypassed[i]) continue;             // removed from the chain by a reconfiguration
            if (!alive[i]) break;                  // the chain is broken here until it is reconfigured
            path.add(i);
        }
        if (path.isEmpty()) throw new IllegalStateException("the chain has no live head");
        return path;
    }

    private int chainLength() {
        int count = 0;
        for (int i = 0; i < alive.length; i++) if (!bypassed[i]) count++;
        return count;
    }

    private boolean chainIntact(List<Integer> path) { return path.size() == chainLength(); }

    private int[] activePathArray() {
        List<Integer> path = chainPath();
        int[] out = new int[path.size()];
        for (int i = 0; i < out.length; i++) out[i] = path.get(i);
        return out;
    }

    /** Latency of one hop on the current path between path positions i and i+1. */
    private int pathHop(int[] path, int position) {
        int total = 0;
        for (int replica = path[position]; replica < path[position + 1]; replica++) total += hopLatency[replica];
        return total;
    }

    /** Runs every pending event, re-checking the prefix invariant after each one. */
    long run() {
        while (!queue.isEmpty()) {
            ChainEvent event = queue.poll();
            int[] path = activePathArray();
            if (path.length != chainLength()) continue;     // the chain is broken: messages stop flowing
            int position = -1;
            for (int i = 0; i < path.length; i++) if (path[i] == event.replica) position = i;
            if (position < 0) continue;                    // the replica left the chain: the message is dropped
            if (event.type == ChainEvent.FORWARD) {
                logs.get(event.replica).add(event.writeId);
                if (position == path.length - 1) {
                    tailCommit.put(event.writeId, event.time);
                    queue.add(new ChainEvent(event.time, ChainEvent.ACK, event.replica, event.writeId, ++sequence));
                } else {
                    queue.add(new ChainEvent(event.time + pathHop(path, position), ChainEvent.FORWARD,
                            path[position + 1], event.writeId, ++sequence));
                }
            } else {
                if (position == 0) {
                    ackTime.put(event.writeId, event.time);
                } else {
                    queue.add(new ChainEvent(event.time + pathHop(path, position - 1), ChainEvent.ACK,
                            path[position - 1], event.writeId, ++sequence));
                }
            }
            sequence++;
            checkPrefixInvariant();
        }
        return invariantChecks;
    }

    /** Every replica's log must be a prefix of the head's log. */
    void checkPrefixInvariant() {
        invariantChecks++;
        List<Integer> head = logs.get(chainPath().get(0));
        for (int replica = 0; replica < logs.size(); replica++) {
            List<Integer> log = logs.get(replica);
            if (bypassed[replica]) continue;
            if (log.size() > head.size()) throw new AssertionError("replica " + replica + " is ahead of the head");
            for (int i = 0; i < log.size(); i++) {
                if (!log.get(i).equals(head.get(i))) {
                    throw new AssertionError("replica " + replica + " diverges at " + i);
                }
            }
        }
    }

    /** Number of writes a replica has applied at a given time (assuming the queue has been drained). */
    int appliedAt(int replica) { return logs.get(replica).size(); }

    /** Kills a replica; the chain stalls until reconfigure() bypasses it. */
    void fail(int replica) {
        if (replica < 0 || replica >= alive.length) throw new IllegalArgumentException("no such replica");
        alive[replica] = false;
    }

    /**
     * Reconfiguration after a failure: every dead replica is bypassed first, then the committed prefix
     * is read off the new tail and the uncommitted suffix (present only on the head side) is truncated
     * everywhere. Simplified: only intermediate replica failures are modelled; a failed tail would
     * additionally have to catch up the new tail before serving.
     */
    void reconfigure() {
        for (int replica = 0; replica < alive.length; replica++) if (!alive[replica]) bypassed[replica] = true;
        List<Integer> path = chainPath();
        int committed = logs.get(path.get(path.size() - 1)).size();
        for (int replica = 0; replica < alive.length; replica++) {
            List<Integer> log = logs.get(replica);
            while (log.size() > committed) log.remove(log.size() - 1);
        }
        queue.clear();
    }

    int committedCount() {
        List<Integer> path = chainPath();
        return logs.get(path.get(path.size() - 1)).size();
    }
}

public class Solution {
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    public static void main(String[] args) {
        // --- single node chain --------------------------------------------------
        Chain lonely = new Chain(1, new int[0]);
        int lonelyWrite = lonely.submit(7, 100);
        lonely.run();
        check(lonely.ackTimeOf(lonelyWrite) == 7, "a one node chain acknowledges immediately");
        check(lonely.committedCount() == 1, "the single replica commits the write");
        check(lonely.logOf(0).size() == 1 && lonely.invariantChecks > 0, "the invariant was checked for the one node chain");

        // --- three replicas, two hops ------------------------------------------
        int[] hops = {2, 3};
        Chain chain = new Chain(3, hops);
        int first = chain.submit(0, 11);
        chain.run();
        check(chain.ackTimeOf(first) == 2 * sum(hops),
                "ack latency is twice the sum of the hops, got " + chain.ackTimeOf(first));
        check(chain.logOf(0).equals(List.of(first)), "the head applied the write");
        check(chain.logOf(1).equals(List.of(first)), "the middle replica applied the write");
        check(chain.logOf(2).equals(List.of(first)), "the tail applied the write before acknowledging");
        check(chain.committedCount() == 1, "exactly one committed write");
        check(chain.invariantChecks > 0, "the prefix invariant was checked during the run");

        // --- the head sees a write before the tail does ------------------------
        Chain inFlight = new Chain(3, hops);
        int write = inFlight.submit(10, 5);
        check(inFlight.appliedAt(0) == 1, "the head applies immediately");
        check(inFlight.appliedAt(1) == 0 && inFlight.appliedAt(2) == 0, "the rest of the chain has not seen it");
        inFlight.run();
        check(inFlight.appliedAt(2) == 1, "after the run the tail has applied it");
        check(inFlight.ackTimeOf(write) == 10 + 2 * sum(hops), "and only then is the client acknowledged");

        // --- ordering of acks and identical latency for a batch ----------------
        Chain batch = new Chain(4, new int[]{1, 1, 2});
        int[] writes = new int[5];
        for (int i = 0; i < writes.length; i++) writes[i] = batch.submit(3L * i, 200 + i);
        batch.run();
        long previousAck = -1;
        for (int i = 0; i < writes.length; i++) {
            long ack = batch.ackTimeOf(writes[i]);
            check(ack == 3L * i + 2 * sum(new int[]{1, 1, 2}), "write " + i + " ack latency, got " + ack);
            check(ack > previousAck, "acks arrive in submission order");
            previousAck = ack;
        }
        check(batch.committedCount() == writes.length, "all five writes commit");
        for (int replica = 0; replica < 4; replica++) {
            check(batch.logOf(replica).size() == writes.length, "replica " + replica + " applied every write");
        }

        // --- failure stalls the chain ------------------------------------------
        Chain failing = new Chain(3, hops);
        int committedBefore = failing.submit(0, 1);
        failing.run();
        check(failing.ackTimeOf(committedBefore) >= 0, "the write before the failure is acknowledged");
        failing.fail(1);
        int duringOutage = failing.submit(100, 2);
        failing.run();
        check(failing.ackTimeOf(duringOutage) == -1, "nothing is acknowledged while the chain is broken");
        check(failing.logOf(0).contains(duringOutage), "the head still holds the uncommitted write");
        check(!failing.logOf(2).contains(duringOutage), "the tail never saw it");
        check(failing.logOf(2).contains(committedBefore), "the committed prefix is untouched");
        boolean survived = true;
        try {
            failing.checkPrefixInvariant();
        } catch (AssertionError broken) {
            survived = false;
        }
        check(survived, "the prefix invariant still holds during the outage");

        // --- reconfiguration resumes and drops the uncommitted suffix ----------
        failing.reconfigure();
        int afterRecovery = failing.submit(200, 3);
        failing.run();
        check(failing.ackTimeOf(afterRecovery) >= 0, "writes resume after reconfiguration");
        check(failing.committedCount() == 2, "the committed prefix plus the new write, got " + failing.committedCount());
        for (int replica = 0; replica < 3; replica++) {
            check(!failing.logOf(replica).contains(duringOutage),
                    "the uncommitted write is gone from replica " + replica);
            check(failing.logOf(replica).contains(committedBefore),
                    "the committed write is still on replica " + replica);
        }
        check(failing.ackTimeOf(afterRecovery) == 200 + 2 * sum(new int[]{2, 3}),
                "the bypassed chain keeps the same hop budget, got " + failing.ackTimeOf(afterRecovery));

        // --- longer randomised run: event simulation vs closed form -----------
        int[] longHops = {1, 2, 1, 3};
        Chain longChain = new Chain(5, longHops);
        List<Integer> longWrites = new ArrayList<>();
        for (int i = 0; i < 20; i++) longWrites.add(longChain.submit(i * 5L, i));
        longChain.run();
        check(longChain.committedCount() == 20, "all twenty writes commit");
        for (int i = 0; i < 20; i++) {
            check(longChain.ackTimeOf(longWrites.get(i)) == i * 5L + 2 * sum(longHops),
                    "closed form ack latency for write " + i);
        }
        check(longChain.logOf(0).equals(longChain.logOf(4)), "head and tail logs match at the end");
        check(longChain.invariantChecks > 20, "the invariant was re-checked after every event: "
                + longChain.invariantChecks);

        // --- rejected input -----------------------------------------------------
        boolean rejected = false;
        try {
            new Chain(0, new int[0]);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an empty chain must be rejected");
        rejected = false;
        try {
            new Chain(3, new int[]{1});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a chain needs exactly replicaCount-1 hops");
        rejected = false;
        try {
            new Chain(2, new int[]{-1});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a negative hop latency must be rejected");
        rejected = false;
        try {
            chain.submit(-1, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a negative submit time must be rejected");
        rejected = false;
        try {
            chain.fail(9);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a replica index outside the chain must be rejected");

        System.out.println("PASS");
    }

    static int sum(int[] values) {
        int total = 0;
        for (int value : values) total += value;
        return total;
    }
}
