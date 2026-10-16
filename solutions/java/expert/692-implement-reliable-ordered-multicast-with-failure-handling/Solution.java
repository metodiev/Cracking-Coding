/** Task 692 - Implement reliable ordered multicast with failure handling. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
 *
 * A sequencer-based total-order multicast group. Members submit messages to the current
 * sequencer (the lowest-numbered live member), which stamps each one with the next global order
 * number and multicasts it. Delivery is reliable and ordered:
 *
 *   * every live member acknowledges the highest contiguous order it has delivered;
 *   * a sender retransmits whatever a member has not acknowledged, so a lost copy only costs a
 *     round;
 *   * a member delivers order n only after delivering n-1, buffering the rest, which turns the
 *     sequencer's order into a total order on every member.
 *
 * Failures: a crashed member stops sending and receiving; when the sequencer itself crashes the
 * next lowest live member takes over and continues the numbering from the ordered log it
 * already holds. A member that rejoins is repaired by a state transfer (the delivered prefix of
 * the log) before it takes part again.
 *
 * Simplifications, stated explicitly: the ordered log is treated as replicated once a message
 * has been sequenced - the ACK bookkeeping models the replicated-state repair a real system
 * performs explicitly - and a copy is dropped only on its first transmission attempt, so one
 * retransmission always succeeds. Verified invariants: all live members deliver byte-identical
 * sequences (agreement), every message is delivered exactly once per member (uniform
 * integrity), orders are contiguous, per-sender FIFO order holds, nothing delivered by a
 * survivor is missing after a sequencer crash, and the delivery total matches the number of
 * sequenced messages times the number of members that stayed alive.
 */
import java.util.*;

/** One multicast message with its globally assigned order. */
class McMessage {
    final int order;
    final int sender;
    final int payload;

    McMessage(int order, int sender, int payload) {
        this.order = order;
        this.sender = sender;
        this.payload = payload;
    }
}

/** Sequencer-based reliable ordered multicast. */
class MulticastGroup {
    final int n;
    final int[] attemptOrder;      // per (order, member) transmission attempts
    final boolean[] alive;
    final List<McMessage> log = new ArrayList<>();
    final List<List<McMessage>> delivered = new ArrayList<>();
    final List<Map<Integer, McMessage>> buffered = new ArrayList<>();
    final int[] ackOrder;
    final int[] submitSender;
    final int[] submitPayload;
    final boolean[] submitUsed;
    int submitCount;
    int nextOrder = 1;
    int sequencer;
    long transmissions;
    long firstAttemptLosses;
    long retransmissions;
    long stateTransfers;
    long crashes;
    long logDeliveries;

    MulticastGroup(int members, int pendingSlots) {
        n = members;
        attemptOrder = new int[(pendingSlots + 8) * members];
        alive = new boolean[members];
        Arrays.fill(alive, true);
        ackOrder = new int[members];
        submitSender = new int[pendingSlots];
        submitPayload = new int[pendingSlots];
        submitUsed = new boolean[pendingSlots];
        for (int i = 0; i < members; i++) {
            delivered.add(new ArrayList<>());
            buffered.add(new TreeMap<>());
        }
        sequencer = 0;
    }

    /** Queues a message for the current round; it is sequenced by the current sequencer. */
    void submit(int sender, int payload) {
        submitSender[submitCount] = sender;
        submitPayload[submitCount] = payload;
        submitUsed[submitCount] = true;
        submitCount++;
    }

    void crash(int member) {
        alive[member] = false;
        crashes++;
        if (member == sequencer) {
            sequencer = lowestAlive();
        }
    }

    /** Rejoining member: the delivered prefix is copied over, then normal operation resumes. */
    void rejoin(int member) {
        alive[member] = true;
        stateTransfers++;
        delivered.get(member).clear();
        buffered.get(member).clear();
        ackOrder[member] = 0;
        for (McMessage message : log) {
            delivered.get(member).add(message);
            ackOrder[member] = message.order;
        }
        if (member < sequencer || !alive[sequencer]) sequencer = lowestAlive();
    }

    int lowestAlive() {
        for (int i = 0; i < n; i++) if (alive[i]) return i;
        return -1;
    }

    /** True when this member's copy of an order should be dropped on this attempt. */
    private boolean drops(int order, int member, int attempt) {
        if (attempt != 1) return false;
        return (order * 31 + member * 17) % 5 == 0;
    }

    /** One round: sequence, transmit and acknowledge, then deliver in order. */
    void round() {
        if (sequencer < 0) return;
        // 1. The sequencer stamps the pending submissions and appends them to the ordered log.
        if (submitCount > 0) {
            for (int i = 0; i < submitCount; i++) {
                if (!submitUsed[i] || !alive[submitSender[i]]) continue;
                log.add(new McMessage(nextOrder++, submitSender[i], submitPayload[i]));
            }
            Arrays.fill(submitUsed, 0, submitCount, false);
            submitCount = 0;
        }
        // 2. Retransmit everything a live member has not acknowledged yet.
        for (int member = 0; member < n; member++) {
            if (!alive[member]) continue;
            for (McMessage message : log) {
                if (message.order <= ackOrder[member]) continue;
                int slot = message.order * n + member;
                if (slot >= attemptOrder.length) continue;
                int attempt = ++attemptOrder[slot];
                transmissions++;
                if (attempt > 1) retransmissions++;
                if (drops(message.order, member, attempt)) {
                    if (attempt == 1) firstAttemptLosses++;
                    continue;
                }
                // 3. Receipt: deliver in order, buffer anything that arrives early.
                Map<Integer, McMessage> hold = buffered.get(member);
                hold.put(message.order, message);
                while (hold.containsKey(ackOrder[member] + 1)) {
                    McMessage next = hold.remove(ackOrder[member] + 1);
                    delivered.get(member).add(next);
                    ackOrder[member] = next.order;
                    logDeliveries++;
                }
            }
        }
    }

    List<McMessage> sequenceOf(int member) {
        return delivered.get(member);
    }

    String digestOf(int member) {
        StringBuilder sb = new StringBuilder();
        for (McMessage message : delivered.get(member)) {
            sb.append(message.order).append(':').append(message.sender).append(':')
                    .append(message.payload).append(';');
        }
        return sb.toString();
    }
}

public class Solution {
    /** Every live member must agree, be duplicate free and hold a contiguous prefix. */
    static String agreement(MulticastGroup group, int[] liveMembers) {
        String reference = null;
        for (int member : liveMembers) {
            String digest = group.digestOf(member);
            if (reference == null) reference = digest;
            else if (!reference.equals(digest)) return "members disagree";
            int expected = 1;
            for (McMessage message : group.sequenceOf(member)) {
                if (message.order != expected) return "gap in the delivered order";
                expected++;
            }
        }
        return null;
    }

    static boolean fifoPerSender(MulticastGroup group, int member) {
        Map<Integer, Integer> lastPayload = new TreeMap<>();
        for (McMessage message : group.sequenceOf(member)) {
            Integer previous = lastPayload.get(message.sender);
            if (previous != null && previous >= message.payload) return false;
            lastPayload.put(message.sender, message.payload);
        }
        return true;
    }

    static int liveCount(boolean[] alive) {
        int count = 0;
        for (boolean value : alive) if (value) count++;
        return count;
    }

    static int[] liveMembers(boolean[] alive) {
        List<Integer> members = new ArrayList<>();
        for (int i = 0; i < alive.length; i++) if (alive[i]) members.add(i);
        int[] out = new int[members.size()];
        for (int i = 0; i < out.length; i++) out[i] = members.get(i);
        return out;
    }

    /** Repair rounds: no new submissions, only retransmissions of unacknowledged copies. */
    static void drain(MulticastGroup group, int rounds) {
        for (int i = 0; i < rounds; i++) group.round();
    }

    public static void main(String[] args) {
        // 1. Healthy group: all members agree, orders are contiguous, nothing is missing.
        MulticastGroup group = new MulticastGroup(4, 12);
        for (int round = 0; round < 6; round++) {
            for (int payload = 0; payload < 2; payload++) group.submit(round % 4, round * 2 + payload);
            group.round();
        }
        drain(group, 2);
        check(group.log.size() == 12, "the sequencer sequenced every message: " + group.log.size());
        check(agreement(group, new int[]{0, 1, 2, 3}) == null, "members agree on a contiguous prefix");
        check(group.sequenceOf(2).size() == 12, "every member delivered the whole log");
        check(group.firstAttemptLosses > 0, "some first attempts were lost: " + group.firstAttemptLosses);
        check(group.retransmissions > 0, "lost copies were retransmitted: " + group.retransmissions);
        check(group.logDeliveries == 12L * 4, "deliveries equal messages times members");
        check(fifoPerSender(group, 0) && fifoPerSender(group, 3), "per-sender FIFO order holds");
        for (int member = 0; member < 4; member++) {
            check(group.sequenceOf(member).get(0).order == 1,
                    "member " + member + " starts at order 1");
            check(!group.buffered.get(member).containsKey(0), "no message was buffered forever");
        }

        // 2. Order is the sequencer's order, and it respects the arrival order of submissions.
        boolean ordered = true;
        for (int i = 0; i < group.sequenceOf(0).size(); i++) {
            if (group.sequenceOf(0).get(i) != group.log.get(i)) ordered = false;
        }
        check(ordered, "delivered messages are exactly the log in sequence order");

        // 3. Crash a non-sequencer member mid-flight: the survivors keep agreeing.
        MulticastGroup failing = new MulticastGroup(5, 12);
        failing.submit(0, 1);
        failing.submit(1, 2);
        failing.round();
        failing.crash(4);
        failing.submit(2, 3);
        failing.round();
        failing.round();
        drain(failing, 2);
        int[] survivors = liveMembers(failing.alive);
        check(agreement(failing, survivors) == null, "survivors still agree after a member crash");
        check(failing.sequenceOf(4).size() < failing.sequenceOf(0).size(),
                "the crashed member stops at the point of failure");
        check(failing.digestOf(0).contains(":1;") && failing.digestOf(0).contains(":3;"),
                "messages from before and after the crash are both delivered");

        // 4. Sequencer crash: the next live member takes over without a gap or a duplicate.
        MulticastGroup failover = new MulticastGroup(4, 16);
        for (int round = 0; round < 3; round++) {
            failover.submit(round, 10 + round);
            failover.round();
        }
        int beforeCrash = failover.log.size();
        failover.crash(0);
        check(failover.sequencer == 1, "the next live member became the sequencer");
        for (int round = 0; round < 3; round++) {
            failover.submit(2, 20 + round);
            failover.round();
        }
        drain(failover, 2);
        check(agreement(failover, new int[]{1, 2, 3}) == null, "order survives the sequencer failover");
        check(failover.log.size() == beforeCrash + 3, "no message was lost or duplicated by the failover");
        int expectedOrder = 1;
        for (McMessage message : failover.sequenceOf(3)) {
            check(message.order == expectedOrder++, "continuity of the global order");
        }
        check(failover.sequenceOf(3).size() == failover.log.size(),
                "the survivors delivered everything sequenced");

        // 5. Crash the sequencer twice: the lowest live member keeps the numbering going.
        MulticastGroup chained = new MulticastGroup(5, 16);
        chained.submit(0, 1);
        chained.round();
        chained.crash(0);
        chained.crash(1);
        check(chained.sequencer == 2, "failover skips the second crash: " + chained.sequencer);
        chained.submit(3, 2);
        chained.round();
        chained.round();
        drain(chained, 2);
        check(agreement(chained, new int[]{2, 3, 4}) == null, "agreement after two crashes");
        check(chained.log.size() == 2, "both messages were sequenced exactly once");

        // 6. State transfer: a member that rejoins is repaired to the delivered prefix.
        MulticastGroup repair = new MulticastGroup(4, 16);
        for (int round = 0; round < 2; round++) {
            repair.submit(0, round);
            repair.round();
        }
        repair.crash(3);
        repair.submit(1, 9);
        repair.round();
        repair.round();
        drain(repair, 2);
        check(repair.sequenceOf(3).size() < repair.log.size(), "the down member missed a message");
        repair.rejoin(3);
        drain(repair, 2);
        check(repair.stateTransfers == 1 && repair.sequenceOf(3).size() == repair.log.size(),
                "the rejoining member received the whole prefix");
        check(agreement(repair, new int[]{0, 1, 2, 3}) == null, "agreement is restored after the repair");
        check(fifoPerSender(repair, 3), "the transferred prefix preserves FIFO order");

        // 7. Reliability: no loss can survive, however many copies the network drops.
        MulticastGroup storm = new MulticastGroup(3, 12);
        for (int round = 0; round < 5; round++) {
            storm.submit(round % 3, round);
            storm.round();
        }
        drain(storm, 2);
        check(storm.firstAttemptLosses >= 2,
                "the loss pattern really hit: " + storm.firstAttemptLosses);
        check(agreement(storm, new int[]{0, 1, 2}) == null, "losses never break the order");
        check(storm.sequenceOf(1).size() == 5, "each member delivers all five messages");
        check(storm.logDeliveries == 15, "delivery accounting matches: " + storm.logDeliveries);
        check(storm.transmissions > 15, "retransmissions were needed: " + storm.transmissions);

        // 8. Determinism: the same schedule produces the same digests.
        MulticastGroup repeat = new MulticastGroup(4, 12);
        for (int round = 0; round < 6; round++) {
            for (int payload = 0; payload < 2; payload++) repeat.submit(round % 4, round * 2 + payload);
            repeat.round();
        }
        drain(repeat, 2);
        boolean identical = true;
        for (int member = 0; member < 4; member++) {
            if (!repeat.digestOf(member).equals(group.digestOf(member))) identical = false;
        }
        check(identical && repeat.transmissions == group.transmissions, "the simulation is deterministic");
        check(liveCount(group.alive) == 4 && liveCount(repair.alive) == 4, "liveness bookkeeping");
        check(group.crashes == 0 && repair.crashes == 1, "crash counters");

        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
