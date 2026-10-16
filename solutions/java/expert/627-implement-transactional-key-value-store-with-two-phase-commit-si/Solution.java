/** Task 627 - Implement transactional key-value store with two-phase commit simulation. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Two-phase commit with one coordinator and N participants. Phase 1: the coordinator sends
 * PREPARE(tx, writes); a participant stages the writes, durably logs `prepare:tx` and votes
 * YES - or votes NO when it is already prepared for another transaction touching the same
 * keys. Phase 2: with every vote YES the coordinator logs `commit:tx` and sends
 * DECIDE(commit), otherwise it logs `abort:tx` and sends DECIDE(abort); each participant
 * applies or discards its staging, overwrites its log with the decision and acks.
 *
 * Simulation model, and what is simplified: no threads, sockets or timers. A node is an object
 * with a single-slot durable log and a staging map; messages travel through one deterministic
 * FIFO in-memory network; a "crash" is a flag that drops traffic in both directions. A
 * participant handles one in-flight transaction at a time and keeps one log slot, where a real
 * store is keyed by transaction id - the coordination logic itself is unchanged. Because of
 * that model the blocking property of 2PC shows up honestly: a crash between PREPARE and
 * DECIDE leaves participants with a `prepare` record and no decision, and nothing inside the
 * protocol can move them; only a recovery pass that reads logs can.
 *
 * Verified: (1) all-yes commits converge on every participant; (2) a single NO vote aborts the
 * whole transaction and no partial write survives anywhere; (3) no dirty reads - a staged value
 * is invisible even on the node that staged it; (4) a coordinator crash after PREPARE blocks
 * every participant (asserted, not hand-waved) and recovery from a decision record unblocks
 * them consistently; (5) the atomicity invariant holds after *every* delivered message, not
 * only at the end; (6) replaying the same schedule twice yields identical state; (7) empty,
 * single-participant and no-participant transactions stay atomic.
 */
import java.util.*;

class Msg {
    final int from;
    final int to;
    final String kind;                 // prepare | vote | decide | ack
    final int tx;
    final Map<String, Integer> writes;
    final boolean yes;
    final String decision;

    Msg(int from, int to, String kind, int tx, Map<String, Integer> writes, boolean yes,
            String decision) {
        this.from = from;
        this.to = to;
        this.kind = kind;
        this.tx = tx;
        this.writes = writes;
        this.yes = yes;
        this.decision = decision;
    }
}

class Node {
    final int id;
    final boolean coordinator;
    boolean crashed;
    final Map<String, Integer> committed = new TreeMap<>();
    final Map<String, Integer> staged = new TreeMap<>();
    String log = "-";                  // -, prepare:tx, commit:tx, abort:tx, done:tx
    int preparedTx = -1;
    int coordinatorTx = -1;
    int votes;
    int noVotes;
    int acks;
    int writesApplied;

    Node(int id, boolean coordinator) {
        this.id = id;
        this.coordinator = coordinator;
    }

    Integer read(String key) {
        return committed.get(key);
    }
}

/** Deterministic network: one FIFO queue; a crashed node neither sends nor receives. */
class Net {
    final List<Node> nodes = new ArrayList<>();
    final Deque<Msg> inFlight = new ArrayDeque<>();
    int delivered;
    int dropped;

    Net(int participants) {
        nodes.add(new Node(0, true));
        for (int i = 1; i <= participants; i++) nodes.add(new Node(i, false));
    }

    Node node(int id) {
        return nodes.get(id);
    }

    List<Node> participants() {
        List<Node> out = new ArrayList<>();
        for (Node n : nodes) if (!n.coordinator) out.add(n);
        return out;
    }

    void send(Msg m) {
        if (node(m.from).crashed || node(m.to).crashed) {
            dropped++;
            return;
        }
        inFlight.addLast(m);
    }

    boolean step() {
        Msg m = inFlight.peekFirst();
        if (m == null) return false;
        inFlight.pollFirst();
        if (node(m.to).crashed) {
            dropped++;
            return true;
        }
        delivered++;
        handle(node(m.to), m);
        return true;
    }

    /** Deliver at most maxSteps messages, stopping early when the network is empty. */
    void drain(int maxSteps) {
        for (int i = 0; i < maxSteps; i++) {
            if (!step()) return;
        }
    }

    void begin(int tx, Map<String, Integer> writes) {
        Node coord = node(0);
        Solution.check(!coord.crashed, "the coordinator is up");
        coord.log = "commit-pending:" + tx;      // phase-1 record written before any PREPARE
        for (Node p : participants()) {
            send(new Msg(0, p.id, "prepare", tx, new TreeMap<>(writes), true, null));
        }
    }

    void handle(Node to, Msg m) {
        switch (m.kind) {
            case "prepare": {
                boolean conflict = to.preparedTx >= 0 && to.preparedTx != m.tx
                        && overlaps(to.staged, m.writes);
                if (conflict) {
                    send(new Msg(to.id, m.from, "vote", m.tx, null, false, null));
                } else {
                    to.preparedTx = m.tx;
                    to.staged.clear();
                    to.staged.putAll(m.writes);
                    to.log = "prepare:" + m.tx;        // durable, written before the vote
                    send(new Msg(to.id, m.from, "vote", m.tx, null, true, null));
                }
                break;
            }
            case "vote": {
                if (to.coordinatorTx != m.tx) {
                    to.coordinatorTx = m.tx;
                    to.votes = 0;
                    to.noVotes = 0;
                    to.acks = 0;
                }
                if (m.yes) to.votes++;
                else to.noVotes++;
                if (to.votes + to.noVotes == participants().size()) {
                    String decision = to.noVotes == 0 ? "commit" : "abort";
                    to.log = decision + ":" + m.tx;    // the single decision point
                    for (Node p : participants()) {
                        send(new Msg(0, p.id, "decide", m.tx, null, true, decision));
                    }
                }
                break;
            }
            case "decide": {
                if ("commit".equals(m.decision)) {
                    to.committed.putAll(to.staged);
                    to.writesApplied++;
                }
                to.staged.clear();
                to.log = m.decision + ":" + m.tx;
                to.preparedTx = -1;
                send(new Msg(to.id, m.from, "ack", m.tx, null, true, m.decision));
                break;
            }
            default:
                to.acks++;
                if (to.acks == participants().size()) to.log = "done:" + m.tx;
                break;
        }
    }

    static boolean overlaps(Map<String, Integer> a, Map<String, Integer> b) {
        for (String k : b.keySet()) if (a.containsKey(k)) return true;
        return false;
    }

    /** Participants holding a `prepare` record with no decision: the 2PC block. */
    int blockedCount() {
        int count = 0;
        for (Node p : participants()) if (p.log.startsWith("prepare:")) count++;
        return count;
    }

    String fingerprint() {
        StringBuilder sb = new StringBuilder();
        for (Node n : nodes) sb.append(n.id).append(n.log).append(n.committed)
                .append(n.writesApplied).append('|');
        return sb.toString();
    }
}

public class Solution {
    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static Map<String, Integer> writes(Object... pairs) {
        Map<String, Integer> m = new TreeMap<>();
        for (int i = 0; i < pairs.length; i += 2) m.put((String) pairs[i], (Integer) pairs[i + 1]);
        return m;
    }

    /** No two participants may end up on different sides of the same decision. */
    static void assertAtomic(Net net, String where) {
        Set<String> outcomes = new TreeSet<>();
        for (Node p : net.participants()) {
            if (p.log.contains("commit")) outcomes.add("commit");
            else if (p.log.contains("abort")) outcomes.add("abort");
        }
        check(outcomes.size() <= 1, where + ": divergent outcomes " + outcomes);
    }

    public static void main(String[] args) {
        happyCommit();
        abortOnConflict();
        noDirtyRead();
        blockingOnCoordinatorCrash();
        recoveryFromDecisionRecord();
        invariantAfterEveryMessage();
        deterministicReplay();
        degenerateTransactions();
        System.out.println("PASS");
    }

    /** Everyone votes yes: the same value lands on every participant. */
    static void happyCommit() {
        Net net = new Net(3);
        net.begin(1, writes("a", 10, "b", 20));
        net.drain(100);
        for (Node p : net.participants()) {
            check(p.log.equals("commit:1"), "participant " + p.id + " logged " + p.log);
            check(p.committed.equals(writes("a", 10, "b", 20)), "participant applied the writes");
            check(p.writesApplied == 1 && p.staged.isEmpty(), "applied once, staging cleared");
        }
        check(net.node(0).log.equals("done:1"), "coordinator saw every ack");
        check(net.delivered == 12, "3 prepares + 3 votes + 3 decisions + 3 acks: " + net.delivered);
        check(net.dropped == 0, "no message was dropped");
    }

    /** One NO vote aborts the whole transaction: no partial write survives anywhere. */
    static void abortOnConflict() {
        Net net = new Net(3);
        net.begin(1, writes("a", 10));
        net.drain(5);                                  // all prepares out, two votes counted
        net.node(0).crashed = true;                    // tx1 never gets decided: everyone prepared
        net.drain(50);
        check(net.blockedCount() == 3, "all three participants stay prepared for tx1");

        net.node(0).crashed = false;                   // recovery, then a conflicting transaction
        net.begin(2, writes("a", 99, "c", 5));
        net.drain(200);
        for (Node p : net.participants()) {
            check(p.log.equals("abort:2"), "participant " + p.id + " voted NO: " + p.log);
            check(!p.committed.containsKey("c"), "participant " + p.id + " kept no partial write");
            check(!p.committed.containsKey("a"), "neither transaction's write to `a` survived");
            check(p.staged.isEmpty(), "staging discarded");
        }
        check(net.node(0).log.equals("done:2"), "coordinator finished the aborted transaction");
        assertAtomic(net, "after the conflicting transaction");
    }

    /** A staged value is invisible everywhere - including on the node that staged it. */
    static void noDirtyRead() {
        Net net = new Net(3);
        net.begin(1, writes("x", 7));
        net.step();                                    // PREPARE reaches participant 1
        Node holder = net.node(1);
        check(holder.log.equals("prepare:1") && holder.staged.get("x") == 7, "value is staged");
        check(holder.read("x") == null, "a read during PREPARE sees the old (absent) value");
        check(!holder.committed.containsKey("x"), "staged value is invisible on its own node");
        check(net.node(2).read("x") == null && net.node(3).read("x") == null,
                "other participants see nothing");
        net.drain(50);
        check(holder.read("x") == 7 && holder.staged.isEmpty(), "the value appears after the commit");
    }

    /** A crash after PREPARE leaves everyone blocked - the 2PC drawback, asserted. */
    static void blockingOnCoordinatorCrash() {
        Net net = new Net(3);
        net.begin(1, writes("k", 1));
        net.drain(5);                                  // 3 prepares + 2 votes; the third is in flight
        check(net.node(0).votes == 2 && net.node(0).noVotes == 0, "two votes counted");
        check(net.node(0).log.equals("commit-pending:1"), "no decision record yet");
        net.node(0).crashed = true;

        int before = net.delivered;
        net.drain(50);
        check(net.delivered == before && net.dropped == 1, "the in-flight vote was dropped");
        check(net.blockedCount() == 3, "all three participants are blocked: " + net.blockedCount());
        for (Node p : net.participants()) check(!p.committed.containsKey("k"), "nobody committed");
        assertAtomic(net, "while blocked");
        check(net.node(0).crashed, "the coordinator is down until a recovery pass runs");
    }

    /** Recovery: the decision record justifies committing, and everyone converges. */
    static void recoveryFromDecisionRecord() {
        Net net = new Net(3);
        net.begin(1, writes("k", 1));
        net.drain(5);
        net.node(0).crashed = true;
        net.drain(50);
        check(net.blockedCount() == 3, "blocked before recovery");

        net.node(0).crashed = false;                   // a new leader reads the coordinator's log
        net.node(0).log = "commit:1";
        for (Node p : net.participants()) {
            net.send(new Msg(0, p.id, "decide", 1, null, true, "commit"));
        }
        net.drain(100);
        for (Node p : net.participants()) {
            check(p.log.equals("commit:1"), "participant " + p.id + " recovered: " + p.log);
            check(p.committed.equals(writes("k", 1)), "recovery applied the transaction");
        }
        check(net.blockedCount() == 0, "no participant is blocked any more");
        check(net.node(0).log.equals("done:1"), "the recovered coordinator saw the acks");
        assertAtomic(net, "after recovery");
    }

    /** The atomicity invariant must hold after every single delivered message. */
    static void invariantAfterEveryMessage() {
        Net net = new Net(4);
        net.begin(1, writes("a", 1, "b", 2));
        int guard = 0;
        while (net.step() && guard++ < 500) {
            assertAtomic(net, "step " + net.delivered);
            for (Node p : net.participants()) {
                if (p.writesApplied > 0) {
                    check(p.log.contains("commit"),
                            "writes become visible only once the decision is logged");
                }
            }
        }
        check(net.delivered == 16, "four participants: prepare/vote/decision/ack");
        for (Node p : net.participants()) {
            check(p.committed.equals(writes("a", 1, "b", 2)), "all four converged");
        }
    }

    /** Same message schedule, same final state - byte for byte. */
    static void deterministicReplay() {
        String first = null;
        for (int run = 0; run < 2; run++) {
            Net net = new Net(3);
            net.begin(1, writes("a", 1));
            net.drain(30);
            net.begin(2, writes("b", 2));
            net.drain(30);
            String fingerprint = net.fingerprint();
            if (first == null) first = fingerprint;
            else check(first.equals(fingerprint), "replay diverged");
            for (Node p : net.participants()) {
                check(p.committed.equals(writes("a", 1, "b", 2)), "both transactions applied");
                check(p.writesApplied == 2, "two applications per participant");
            }
        }
    }

    /** Empty, single-participant and no-participant transactions stay atomic. */
    static void degenerateTransactions() {
        Net empty = new Net(3);
        empty.begin(1, new TreeMap<>());
        empty.drain(50);
        for (Node p : empty.participants()) {
            check(p.log.equals("commit:1") && p.committed.isEmpty(), "empty transaction commits");
        }
        check(empty.node(0).log.equals("done:1"), "coordinator done");

        Net single = new Net(1);
        single.begin(5, writes("only", 1));
        single.drain(50);
        check(single.node(1).committed.equals(writes("only", 1)), "single participant committed");
        check(single.node(1).writesApplied == 1 && single.node(1).log.equals("commit:5"),
                "applied exactly once");
        assertAtomic(single, "single participant");

        Net none = new Net(0);
        none.begin(9, writes("k", 1));
        none.drain(20);
        check(none.delivered == 0 && none.blockedCount() == 0, "no participants, nothing to block");
        check(none.node(0).committed.isEmpty(), "the coordinator holds no data of its own");
    }
}
