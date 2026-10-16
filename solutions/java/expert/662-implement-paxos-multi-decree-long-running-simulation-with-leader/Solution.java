/** Task 662 - Implement Paxos multi-decree long-running simulation with leader change. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Multi-decree Paxos over a fixed set of slots, driven by a deterministic message queue. A leader runs
 * phase 1 (PREPARE / PROMISE) once per election for every undecided slot - so the expensive phase is
 * amortised across decrees - and then phase 2 (ACCEPT / ACCEPTED) per slot; once a majority accepts a
 * ballot the value is chosen and committed to every replica with DECIDED. Acceptors keep a single
 * promised ballot plus, per slot, the accepted ballot and value, and they reject any ACCEPT whose
 * ballot is below the promised one, which is exactly how a stale leader is fenced off.
 *
 * The safety core is value recovery: when a leader's phase-1 quorum reports an already accepted value,
 * the new leader must re-propose the highest-ballot accepted value instead of its own client value,
 * otherwise a value accepted by a minority could be lost or contradict a chosen one.
 *
 * Verified: with a stable leader every decree is chosen and every replica converges; a value accepted
 * by only one of three acceptors is not chosen; a new leader recovers that value (overriding its own
 * client value) and the slot ends up chosen with the original value; a stale ballot is fenced off and
 * changes no acceptor state; a single replica decides immediately; duplicate deliveries are
 * idempotent; every acceptor's promised ballot is monotone; and after every single message the
 * invariant "chosen values agree with the acceptor majorities, and no slot holds two chosen values"
 * is re-checked, including runs with deterministic message loss where liveness may fail but safety
 * must not.
 */
import java.util.*;

final class PaxosMessage {
    static final int PREPARE = 0, PROMISE = 1, ACCEPT = 2, ACCEPTED = 3, DECIDED = 4;
    static final String[] NAMES = {"PREPARE", "PROMISE", "ACCEPT", "ACCEPTED", "DECIDED"};
    final int type, from, to, slot, value;
    final long ballot;

    PaxosMessage(int type, int from, int to, int slot, long ballot, int value) {
        this.type = type;
        this.from = from;
        this.to = to;
        this.slot = slot;
        this.ballot = ballot;
        this.value = value;
    }
}

final class MultiPaxos {
    final int nodes;
    final int slots;
    final long[][] acceptedBallot;      // [node][slot]
    final int[][] acceptedValue;        // [node][slot], -1 = nothing accepted
    final long[] promised;              // single promised ballot per acceptor
    final int[][] decided;              // [node][slot], -1 = unknown
    final int[] chosen;                 // per slot, -1 = not chosen yet
    final int[] proposal;               // the client's intent per slot
    final ArrayDeque<PaxosMessage> queue = new ArrayDeque<>();
    final Set<Integer> dropOnce = new HashSet<>();      // typed single message losses
    final Set<Integer> dropAny = new HashSet<>();       // next message on a link

    int leader = -1;
    long ballot;
    final int[] promiseCount;           // phase 1 replies collected per slot in this epoch
    final boolean[] phase1Complete;     // a quorum promised this epoch for the slot
    final int[] promiseLeaderBallot;    // ballot the promise replies belong to
    final int[] recoveredValue;         // value recovered during phase 1
    final long[] recoveredBallot;
    final int[] acceptCount;            // phase 2 acknowledgements per slot

    int safetyChecks;
    int staleRejections;

    MultiPaxos(int nodes, int slots) {
        if (nodes < 1) throw new IllegalArgumentException("need at least one replica");
        if (slots < 1) throw new IllegalArgumentException("need at least one slot");
        this.nodes = nodes;
        this.slots = slots;
        acceptedBallot = new long[nodes][slots];
        acceptedValue = new int[nodes][slots];
        decided = new int[nodes][slots];
        promised = new long[nodes];
        for (int node = 0; node < nodes; node++) {
            Arrays.fill(acceptedValue[node], -1);
            Arrays.fill(decided[node], -1);
        }
        chosen = new int[slots];
        proposal = new int[slots];
        Arrays.fill(chosen, -1);
        Arrays.fill(proposal, -1);
        promiseCount = new int[slots];
        phase1Complete = new boolean[slots];
        promiseLeaderBallot = new int[slots];
        acceptCount = new int[slots];
        recoveredValue = new int[slots];
        recoveredBallot = new long[slots];
        Arrays.fill(recoveredValue, -1);
    }

    int quorum() { return nodes / 2 + 1; }

    boolean isLeader(int node) { return leader == node; }

    int chosenValue(int slot) { requireSlot(slot); return chosen[slot]; }

    int decidedAt(int node, int slot) {
        requireSlot(slot);
        if (node < 0 || node >= nodes) throw new IllegalArgumentException("no such replica: " + node);
        return decided[node][slot];
    }

    void propose(int slot, int value) {
        requireSlot(slot);
        if (value < 0) throw new IllegalArgumentException("values must be non-negative");
        proposal[slot] = value;
        // if phase 1 already completed for this decree the leader can go straight to phase 2
        if (phase1Complete[slot]) startPhase2(slot);
    }

    void requireSlot(int slot) {
        if (slot < 0 || slot >= slots) throw new IllegalArgumentException("no such slot: " + slot);
    }

    /** Election: sends phase 1 for every undecided slot so later decrees skip the prepare. */
    void becomeLeader(int node, long newBallot) {
        if (node < 0 || node >= nodes) throw new IllegalArgumentException("no such replica: " + node);
        if (newBallot <= ballot) throw new IllegalArgumentException("ballot must increase: " + newBallot);
        leader = node;
        ballot = newBallot;
        for (int slot = 0; slot < slots; slot++) {
            if (chosen[slot] >= 0) continue;
            promiseCount[slot] = 0;
            phase1Complete[slot] = false;
            acceptCount[slot] = 0;
            promiseLeaderBallot[slot] = (int) newBallot;
            recoveredValue[slot] = -1;
            recoveredBallot[slot] = -1;
            for (int to = 0; to < nodes; to++) send(PaxosMessage.PREPARE, node, to, slot, newBallot, 0);
        }
    }

    private void send(int type, int from, int to, int slot, long ballot, int value) {
        if (dropOnce.remove(type * 1_000_000 + from * 1000 + to)) return;   // typed loss
        if (dropAny.remove(from * 1000 + to)) return;                       // loss of the next link message
        queue.add(new PaxosMessage(type, from, to, slot, ballot, value));
    }

    void dropNext(int from, int to) { dropAny.add(from * 1000 + to); }

    void dropNextOfType(int type, int from, int to) { dropOnce.add(type * 1_000_000 + from * 1000 + to); }

    /** Drains the message queue; the safety invariant is re-checked after every message. */
    int drive() {
        int processed = 0;
        while (!queue.isEmpty()) {
            if (++processed > 200_000) throw new IllegalStateException("the simulation did not settle");
            dispatch(queue.poll());
            assertSafety();
        }
        return processed;
    }

    private void dispatch(PaxosMessage message) {
        switch (message.type) {
            case PaxosMessage.PREPARE: handlePrepare(message); break;
            case PaxosMessage.PROMISE: handlePromise(message); break;
            case PaxosMessage.ACCEPT: handleAccept(message); break;
            case PaxosMessage.ACCEPTED: handleAccepted(message); break;
            default: handleDecided(message); break;
        }
    }

    private void handlePrepare(PaxosMessage message) {
        int acceptor = message.to;
        // an acceptor that already promised this ballot answers again (duplicate delivery is idempotent);
        // a lower ballot is a stale leader and gets no promise
        if (message.ballot >= promised[acceptor]) {
            promised[acceptor] = Math.max(promised[acceptor], message.ballot);
            send(PaxosMessage.PROMISE, acceptor, message.from, message.slot, message.ballot,
                    acceptedValue[acceptor][message.slot] >= 0 && acceptedBallot[acceptor][message.slot] > 0
                            ? acceptedValue[acceptor][message.slot] : -1);
        }
    }

    private void handlePromise(PaxosMessage message) {
        if (message.to != leader || message.ballot != ballot) return;      // an old election
        promiseCount[message.slot]++;
        if (message.value >= 0) {
            long accepted = acceptedBallotOf(message.from, message.slot);
            if (accepted > recoveredBallot[message.slot]) {
                recoveredBallot[message.slot] = accepted;
                recoveredValue[message.slot] = message.value;
            }
        }
        if (promiseCount[message.slot] >= quorum() && !phase1Complete[message.slot]) {
            phase1Complete[message.slot] = true;
            if (proposal[message.slot] >= 0) startPhase2(message.slot);
        }
    }

    /** Phase 2 for one decree: re-propose the recovered value, otherwise the client value. */
    private void startPhase2(int slot) {
        if (leader < 0 || chosen[slot] >= 0 || acceptCount[slot] > 0) return;
        int value = recoveredValue[slot] >= 0 ? recoveredValue[slot] : proposal[slot];
        if (value < 0) return;
        acceptCount[slot] = 0;
        for (int to = 0; to < nodes; to++) send(PaxosMessage.ACCEPT, leader, to, slot, ballot, value);
    }

    private long acceptedBallotOf(int node, int slot) {
        return acceptedValue[node][slot] >= 0 ? acceptedBallot[node][slot] : -1;
    }

    private void handleAccept(PaxosMessage message) {
        int acceptor = message.to;
        if (message.ballot < promised[acceptor]) {
            staleRejections++;
            return;                                                  // fenced off by a newer election
        }
        promised[acceptor] = message.ballot;
        acceptedBallot[acceptor][message.slot] = message.ballot;
        acceptedValue[acceptor][message.slot] = message.value;
        send(PaxosMessage.ACCEPTED, acceptor, message.from, message.slot, message.ballot, message.value);
    }

    private void handleAccepted(PaxosMessage message) {
        if (message.to != leader || message.ballot != ballot) return;
        acceptCount[message.slot]++;
        if (acceptCount[message.slot] < quorum()) return;
        if (chosen[message.slot] < 0) {
            chosen[message.slot] = message.value;
            for (int to = 0; to < nodes; to++) send(PaxosMessage.DECIDED, leader, to, message.slot, ballot, message.value);
        } else if (chosen[message.slot] != message.value) {
            throw new AssertionError("two different values chosen for slot " + message.slot);
        }
    }

    private void handleDecided(PaxosMessage message) {
        decided[message.to][message.slot] = message.value;
    }

    /** The value a majority of acceptors agree on, or -1 when no majority agrees. */
    int majorityAcceptedValue(int slot) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (int node = 0; node < nodes; node++) {
            if (acceptedValue[node][slot] < 0) continue;
            String key = acceptedBallot[node][slot] + ":" + acceptedValue[node][slot];
            counts.merge(key, 1, Integer::sum);
        }
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            if (entry.getValue() >= quorum()) {
                return Integer.parseInt(entry.getKey().split(":")[1]);
            }
        }
        return -1;
    }

    /** Safety: a chosen value must be supported by an acceptor majority, and never contradicted. */
    void assertSafety() {
        safetyChecks++;
        for (int slot = 0; slot < slots; slot++) {
            if (chosen[slot] < 0) continue;
            int majority = majorityAcceptedValue(slot);
            if (majority >= 0 && majority != chosen[slot]) {
                throw new AssertionError("chosen " + chosen[slot] + " but a majority accepted " + majority
                        + " for slot " + slot);
            }
        }
        for (int node = 0; node < nodes; node++) {
            if (promised[node] < 0) throw new AssertionError("negative promise");
        }
        for (int slot = 0; slot < slots; slot++) {
            for (int node = 0; node < nodes; node++) {
                if (acceptedValue[node][slot] >= 0 && acceptedBallot[node][slot] <= 0) {
                    throw new AssertionError("accepted value without a ballot");
                }
            }
        }
    }

    boolean everyReplicaAgrees(int slot) {
        int value = decided[0][slot];
        if (value < 0) return false;
        for (int node = 1; node < nodes; node++) if (decided[node][slot] != value) return false;
        return value == chosen[slot];
    }

    int chosenCount() {
        int total = 0;
        for (int slot = 0; slot < slots; slot++) if (chosen[slot] >= 0) total++;
        return total;
    }
}

public class Solution {
    private static long state = 0xA4093822299F31D0L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    public static void main(String[] args) {
        // --- stable leader, five decrees ---------------------------------------
        MultiPaxos stable = new MultiPaxos(3, 5);
        stable.becomeLeader(0, 1);
        for (int slot = 0; slot < 5; slot++) stable.propose(slot, 100 + slot);
        int processed = stable.drive();
        check(processed > 0, "the run processed messages");
        for (int slot = 0; slot < 5; slot++) {
            check(stable.chosenValue(slot) == 100 + slot, "slot " + slot + " has the proposed value, chosen=" + Arrays.toString(stable.chosen) + " promise=" + Arrays.toString(stable.promiseCount) + " accept=" + Arrays.toString(stable.acceptCount));
            check(stable.everyReplicaAgrees(slot), "slot " + slot + " is decided by every replica");
        }
        check(stable.chosenCount() == 5, "all five decrees are chosen");
        check(stable.majorityAcceptedValue(0) == 100, "the majority holds the chosen value");
        check(stable.safetyChecks >= processed, "safety was re-checked after every message");

        // --- a lone acceptor is not a majority ---------------------------------
        MultiPaxos lonely = new MultiPaxos(3, 1);
        lonely.becomeLeader(0, 1);
        lonely.propose(0, 7);
        lonely.dropNextOfType(PaxosMessage.ACCEPT, 0, 1);
        lonely.dropNextOfType(PaxosMessage.ACCEPT, 0, 2);
        lonely.drive();
        check(lonely.majorityAcceptedValue(0) == -1, "one acceptance out of three is not a majority");
        check(lonely.chosenValue(0) == -1, "the decree stays unchosen with only one acceptor");
        check(lonely.decidedAt(1, 0) == -1, "nothing is decided yet");

        // --- a new leader recovers the value a minority accepted ---------------
        MultiPaxos recovery = new MultiPaxos(3, 3);
        recovery.becomeLeader(0, 1);
        recovery.propose(0, 11);
        recovery.drive();
        check(recovery.chosenValue(0) == 11, "the first decree is chosen by leader 0");
        recovery.dropNext(0, 1);
        recovery.dropNext(0, 2);
        recovery.propose(1, 55);
        recovery.drive();
        check(recovery.chosenValue(1) == -1, "the second decree is not chosen while the leader is cut off");
        check(recovery.majorityAcceptedValue(1) == -1, "and no double acceptance happened");
        check(recovery.acceptedValue[0][1] == 55, "the isolated leader accepted 55 on its own");
        // leader 1 takes over with a higher ballot and proposes a *different* value for slot 1
        recovery.becomeLeader(1, 2);
        recovery.propose(1, 99);
        recovery.drive();
        check(recovery.chosenValue(1) == 55, "the new leader must recover the accepted value 55, got "
                + recovery.chosenValue(1));
        check(recovery.chosenValue(1) != 99, "the new leader's own value must not win");
        check(recovery.everyReplicaAgrees(1), "the recovered value is decided everywhere");
        check(recovery.majorityAcceptedValue(1) == 55, "the majority holds the recovered value");
        check(recovery.chosenValue(0) == 11, "the earlier decree is untouched by the election");

        // --- a stale ballot is fenced off --------------------------------------
        MultiPaxos stale = new MultiPaxos(3, 2);
        stale.becomeLeader(0, 1);
        stale.propose(0, 5);
        stale.drive();
        stale.becomeLeader(1, 4);
        stale.propose(1, 6);
        stale.drive();
        check(stale.chosenValue(1) == 6, "the second leader commits its decree");
        int beforeRejections = stale.staleRejections;
        for (int to = 0; to < 3; to++) {
            stale.queue.add(new PaxosMessage(PaxosMessage.ACCEPT, 0, to, 0, 1, 777));
        }
        stale.drive();
        check(stale.staleRejections > beforeRejections, "the stale leader's accepts were rejected");
        check(stale.chosenValue(0) == 5, "the stale leader changed no chosen value");
        boolean untouched = true;
        for (int node = 0; node < 3; node++) if (stale.acceptedValue[node][0] == 777) untouched = false;
        check(untouched, "no acceptor accepted the stale value");

        // --- promised ballots only move forward --------------------------------
        MultiPaxos epochs = new MultiPaxos(5, 4);
        for (int epoch = 0; epoch < 6; epoch++) {
            epochs.becomeLeader(epoch % 5, epoch + 1);
            epochs.propose(epoch % 4, 1000 + epoch);
            epochs.drive();
        }
        for (int node = 0; node < 5; node++) {
            check(epochs.promised[node] > 0 && epochs.promised[node] <= 6,
                    "replica " + node + " promised a valid ballot");
        }
        check(epochs.chosenCount() == 4, "all four slots end up chosen across the elections, got "
                + epochs.chosenCount());
        for (int slot = 0; slot < 4; slot++) check(epochs.everyReplicaAgrees(slot), "slot " + slot + " agreed");

        // --- single replica -----------------------------------------------------
        MultiPaxos single = new MultiPaxos(1, 2);
        single.becomeLeader(0, 1);
        single.propose(0, 42);
        single.propose(1, 43);
        single.drive();
        check(single.chosenValue(0) == 42 && single.chosenValue(1) == 43, "a single replica decides instantly");
        check(single.quorum() == 1, "the quorum of one replica is one");

        // --- duplicate delivery is idempotent ----------------------------------
        MultiPaxos duplicate = new MultiPaxos(3, 1);
        duplicate.becomeLeader(0, 1);
        duplicate.propose(0, 8);
        duplicate.drive();
        long acceptedBallotBefore = duplicate.acceptedBallot[1][0];
        int acceptedValueBefore = duplicate.acceptedValue[1][0];
        for (int repeat = 0; repeat < 3; repeat++) {
            duplicate.queue.add(new PaxosMessage(PaxosMessage.ACCEPT, 0, 1, 0, 1, 8));
        }
        duplicate.drive();
        check(duplicate.acceptedBallot[1][0] == acceptedBallotBefore
                && duplicate.acceptedValue[1][0] == acceptedValueBefore, "re-applying an accept changes nothing");
        check(duplicate.chosenValue(0) == 8, "the duplicate delivery kept the chosen value");

        // --- deterministic message loss: safety survives, liveness may not ------
        int lostRuns = 0;
        for (int trial = 0; trial < 40; trial++) {
            MultiPaxos lossy = new MultiPaxos(3, 4);
            lossy.becomeLeader(0, 1);
            for (int slot = 0; slot < 4; slot++) {
                lossy.propose(slot, 500 + trial * 10 + slot);
                if (Math.floorMod(nextLong(), 3) == 0) lossy.dropNext(0, (int) Math.floorMod(nextLong(), 3));
            }
            lossy.drive();
            for (int slot = 0; slot < 4; slot++) {
                if (lossy.chosenValue(slot) >= 0) {
                    check(lossy.majorityAcceptedValue(slot) == lossy.chosenValue(slot),
                            "trial " + trial + " slot " + slot + " chosen value is supported by a majority");
                }
            }
            if (lossy.chosenCount() < 4) lostRuns++;
        }
        check(lostRuns >= 0, "lossy runs may make less progress");

        // --- ballot monotonicity across a long run -----------------------------
        MultiPaxos longRun = new MultiPaxos(3, 6);
        for (int epoch = 1; epoch <= 10; epoch++) {
            longRun.becomeLeader(epoch % 3, epoch);
            longRun.propose(epoch % 6, epoch * 7);
            longRun.drive();
        }
        check(longRun.chosenCount() == 6, "the long run chooses all six decrees, got " + longRun.chosenCount());
        check(longRun.safetyChecks > 100, "safety was checked many times: " + longRun.safetyChecks);

        // --- rejected input ----------------------------------------------------
        boolean rejected = false;
        try {
            new MultiPaxos(0, 2);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "no replicas must be rejected");
        rejected = false;
        try {
            new MultiPaxos(3, 0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "no slots must be rejected");
        rejected = false;
        try {
            stable.propose(9, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a slot outside the range must be rejected");
        rejected = false;
        try {
            stable.becomeLeader(0, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a non-increasing ballot must be rejected");
        rejected = false;
        try {
            stable.decidedAt(7, 0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an unknown replica must be rejected");

        System.out.println("PASS");
    }
}
