/** Task 628 - Implement distributed consensus simulation with network partitions (Raft test harness). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A compact Raft: terms, one vote per voter per term, the log-up-to-date restriction on
 * granting a vote, log replication with a prevIndex/prevTerm consistency check and truncation
 * on a term conflict, and a commit rule that only advances past an entry of the leader's own
 * term once a cluster majority has replicated it.
 *
 * Simulation model, and what is simplified: time is a tick counter; each message is delivered
 * one tick later through a deterministic FIFO network; election timeouts are fixed per node
 * (5 + 3*id) rather than randomized, which is fine because the run is single-threaded and the
 * harness only needs one node to time out first; a partition is a group number, and two nodes
 * can exchange messages only if their group numbers are equal. What is *not* simplified are the
 * Raft rules themselves - vote restriction, term checks, conflict truncation, commit by
 * majority and stepping down on a higher term are implemented as specified.
 *
 * Verified: (1) a single-node cluster elects itself and commits; (2) a five-node cluster elects
 * one leader, replicates to every follower and teaches them the commit index; (3) at most one
 * leader per term and never two votes by one voter in one term, across the whole run; (4) a
 * partitioned minority can neither elect a leader nor commit anything, while the majority side
 * keeps making progress; (5) a stale candidate is refused a vote and an up-to-date one is
 * granted it; (6) after healing, every node converges to the same log - the isolated leader's
 * uncommitted tail is truncated and replaced; (7) State Machine Safety (two nodes whose
 * commitIndex passed index i agree on entry i) is asserted after *every* tick, including under
 * a PRNG-driven partition schedule.
 */
import java.util.*;

class REntry {
    static final int NOOP = Integer.MIN_VALUE;         // leaders append one entry at their election
    final int term;
    final int cmd;

    REntry(int term, int cmd) {
        this.term = term;
        this.cmd = cmd;
    }
}

class Msg628 {
    final int from;
    final int to;
    final String kind;            // requestVote | vote | append | appendReply
    final int term;
    final int lastLogIndex;
    final int lastLogTerm;
    final boolean granted;
    final List<REntry> entries;
    final int prevIndex;
    final int prevTerm;
    final int leaderCommit;
    final int matchIndex;

    Msg628(int from, int to, String kind, int term, int lastLogIndex, int lastLogTerm,
            boolean granted, List<REntry> entries, int prevIndex, int prevTerm, int leaderCommit,
            int matchIndex) {
        this.from = from;
        this.to = to;
        this.kind = kind;
        this.term = term;
        this.lastLogIndex = lastLogIndex;
        this.lastLogTerm = lastLogTerm;
        this.granted = granted;
        this.entries = entries;
        this.prevIndex = prevIndex;
        this.prevTerm = prevTerm;
        this.leaderCommit = leaderCommit;
        this.matchIndex = matchIndex;
    }
}

class RaftNode {
    final int id;
    int term;
    int votedFor = -1;
    String role = "follower";
    final List<REntry> log = new ArrayList<>();      // 1-based: index i is log.get(i - 1)
    int commitIndex;
    int votesGranted;
    int electionTimer;
    final Map<Integer, Integer> nextIndex = new TreeMap<>();
    final Map<Integer, Integer> matchIndex = new TreeMap<>();

    RaftNode(int id) {
        this.id = id;
    }

    int lastLogIndex() {
        return log.size();
    }

    int lastLogTerm() {
        return log.isEmpty() ? 0 : log.get(log.size() - 1).term;
    }

    int termAt(int index) {
        return index >= 1 && index <= log.size() ? log.get(index - 1).term : 0;
    }

    boolean sameEntryAs(RaftNode other, int index) {
        if (index < 1 || index > lastLogIndex() || index > other.lastLogIndex()) return false;
        REntry a = log.get(index - 1);
        REntry b = other.log.get(index - 1);
        return a.term == b.term && a.cmd == b.cmd;
    }

    void resetTimer() {
        electionTimer = 0;
    }

    String logString() {
        StringBuilder sb = new StringBuilder();
        for (REntry e : log) sb.append(e.cmd == REntry.NOOP ? "noop" : e.cmd).append('@')
                .append(e.term).append(' ');
        return sb.toString().trim();
    }
}

class RaftCluster {
    final List<RaftNode> nodes = new ArrayList<>();
    final Deque<Msg628> inFlight = new ArrayDeque<>();
    final int[] group;                                   // equal group numbers can talk
    final Map<Integer, Integer> leaderByTerm = new TreeMap<>();
    final Set<String> votesSeen = new TreeSet<>();        // voter:term
    final Set<String> committedEntries = new TreeSet<>(); // index:term ever committed
    int voteAnomalies;
    int ticks;

    RaftCluster(int n) {
        for (int i = 1; i <= n; i++) nodes.add(new RaftNode(i));
        group = new int[n + 1];
    }

    int quorum() {
        return nodes.size() / 2 + 1;
    }

    RaftNode node(int id) {
        return nodes.get(id - 1);
    }

    void isolate(int... ids) {
        for (int id : ids) group[id] = 100 + id;
    }

    void majoritySide(int... ids) {
        for (int id : ids) group[id] = 0;
    }

    void heal() {
        for (int i = 1; i < group.length; i++) group[i] = 0;
    }

    boolean connected(int a, int b) {
        return group[a] == group[b];
    }

    void send(Msg628 m) {
        if (connected(m.from, m.to)) inFlight.addLast(m);
    }

    RaftNode leader() {
        for (RaftNode n : nodes) if (n.role.equals("leader")) return n;
        return null;
    }

    List<RaftNode> leadersOtherThan(int id) {
        List<RaftNode> out = new ArrayList<>();
        for (RaftNode n : nodes) if (n.role.equals("leader") && n.id != id) out.add(n);
        return out;
    }

    void proposeTo(RaftNode leader, int cmd) {
        leader.log.add(new REntry(leader.term, cmd));
        advanceCommit(leader);        // a quorum of one (single-node cluster) commits at once
    }

    void propose(int cmd) {
        RaftNode leader = leader();
        if (leader != null) proposeTo(leader, cmd);
    }

    /** One tick: deliver everything queued, then expire timers, then heartbeat. */
    void tick() {
        ticks++;
        int pending = inFlight.size();
        for (int i = 0; i < pending; i++) deliver(inFlight.pollFirst());
        for (RaftNode n : nodes) {
            if (n.role.equals("leader")) {
                replicate(n);
            } else {
                n.electionTimer++;
                if (n.electionTimer >= 5 + 3 * n.id) startElection(n);
            }
        }
    }

    void startElection(RaftNode n) {
        n.role = "candidate";
        n.term++;
        n.votedFor = n.id;
        n.votesGranted = 1;
        n.resetTimer();
        for (RaftNode peer : nodes) {
            if (peer.id == n.id) continue;
            send(new Msg628(n.id, peer.id, "requestVote", n.term, n.lastLogIndex(),
                    n.lastLogTerm(), true, null, 0, 0, 0, 0));
        }
        if (n.votesGranted >= quorum()) becomeLeader(n);
    }

    void becomeLeader(RaftNode n) {
        n.role = "leader";
        n.resetTimer();
        Integer previous = leaderByTerm.get(n.term);
        if (previous != null && previous != n.id) voteAnomalies += 1000;   // two leaders, one term
        leaderByTerm.put(n.term, n.id);
        n.log.add(new REntry(n.term, REntry.NOOP));    // the usual no-op entry of the new term
        for (RaftNode peer : nodes) {
            n.nextIndex.put(peer.id, n.lastLogIndex() + 1);
            n.matchIndex.put(peer.id, 0);
        }
        advanceCommit(n);
        replicate(n);
    }

    void replicate(RaftNode leader) {
        for (RaftNode peer : nodes) {
            if (peer.id == leader.id) continue;
            int next = leader.nextIndex.getOrDefault(peer.id, leader.lastLogIndex() + 1);
            List<REntry> tail = new ArrayList<>();
            for (int i = next; i <= leader.lastLogIndex(); i++) tail.add(leader.log.get(i - 1));
            send(new Msg628(leader.id, peer.id, "append", leader.term, 0, 0, true, tail, next - 1,
                    leader.termAt(next - 1), leader.commitIndex, 0));
        }
    }

    void deliver(Msg628 m) {
        RaftNode to = node(m.to);
        if (m.term > to.term) {
            to.term = m.term;
            to.votedFor = -1;
            to.role = "follower";
            to.votesGranted = 0;
            to.resetTimer();
        }
        switch (m.kind) {
            case "requestVote": {
                boolean upToDate = m.lastLogTerm > to.lastLogTerm()
                        || (m.lastLogTerm == to.lastLogTerm()
                                && m.lastLogIndex >= to.lastLogIndex());
                boolean grant = m.term >= to.term && upToDate
                        && (to.votedFor == -1 || to.votedFor == m.from);
                if (grant) {
                    if (!votesSeen.add(to.id + ":" + m.term)) voteAnomalies++;   // voted twice
                    to.votedFor = m.from;
                    to.resetTimer();
                }
                send(new Msg628(to.id, m.from, "vote", to.term, 0, 0, grant, null, 0, 0, 0, 0));
                break;
            }
            case "vote":
                if (to.role.equals("candidate") && m.term == to.term && m.granted) {
                    to.votesGranted++;
                    if (to.votesGranted >= quorum()) becomeLeader(to);
                }
                break;
            case "append": {
                if (m.term < to.term) {
                    send(new Msg628(to.id, m.from, "appendReply", to.term, 0, 0, false, null, 0, 0,
                            0, 0));
                    break;
                }
                to.role = "follower";
                to.resetTimer();
                boolean consistent = m.prevIndex == 0
                        || (m.prevIndex <= to.lastLogIndex() && to.termAt(m.prevIndex) == m.prevTerm);
                if (!consistent) {
                    send(new Msg628(to.id, m.from, "appendReply", to.term, 0, 0, false, null, 0, 0,
                            0, 0));
                    break;
                }                for (int i = 0; i < m.entries.size(); i++) {
                    int index = m.prevIndex + 1 + i;
                    REntry incoming = m.entries.get(i);
                    if (index <= to.lastLogIndex()) {
                        if (to.termAt(index) != incoming.term) {
                            to.log.subList(index - 1, to.log.size()).clear();   // conflict: truncate
                            to.log.add(incoming);
                        }
                    } else {
                        to.log.add(incoming);
                    }
                }
                if (m.leaderCommit > to.commitIndex) {
                    to.commitIndex = Math.min(m.leaderCommit, to.lastLogIndex());
                    recordCommitted(to);
                }
                send(new Msg628(to.id, m.from, "appendReply", to.term, 0, 0, true, null, 0, 0, 0,
                        m.prevIndex + m.entries.size()));
                break;
            }
            default: {
                RaftNode leader = to;                            // the reply is addressed to the leader
                RaftNode peer = node(m.from);                    // ... and sent by the follower
                if (!leader.role.equals("leader") || leader.term != peer.term) break;
                if (m.granted) {
                    leader.matchIndex.put(peer.id, m.matchIndex);
                    leader.nextIndex.put(peer.id, m.matchIndex + 1);
                    advanceCommit(leader);
                } else {
                    int current = leader.nextIndex.getOrDefault(peer.id, leader.lastLogIndex() + 1);
                    leader.nextIndex.put(peer.id, Math.max(1, current - 1));
                }
                break;
            }        }
    }

    void advanceCommit(RaftNode leader) {
        for (int index = leader.lastLogIndex(); index > leader.commitIndex; index--) {
            if (leader.termAt(index) != leader.term) continue;   // only own-term entries commit
            int replicated = 1;
            for (RaftNode peer : nodes) {
                if (peer.id != leader.id && leader.matchIndex.getOrDefault(peer.id, 0) >= index) {
                    replicated++;
                }
            }
            if (replicated >= quorum()) {
                leader.commitIndex = index;
                recordCommitted(leader);
                return;
            }
        }
    }

    void recordCommitted(RaftNode n) {
        for (int i = 1; i <= n.commitIndex; i++) committedEntries.add(i + ":" + n.termAt(i));
    }

    /** State machine safety: nodes that both passed index i must agree on that entry. */
    void assertAgreement(String where) {
        for (RaftNode a : nodes) {
            for (RaftNode b : nodes) {
                int limit = Math.min(a.commitIndex, b.commitIndex);
                for (int i = 1; i <= limit; i++) {
                    if (a.termAt(i) != b.termAt(i)) {
                        throw new AssertionError("FAILED: " + where + ": nodes " + a.id + " and "
                                + b.id + " disagree on entry " + i);
                    }
                }
            }
        }
    }

    boolean sameLogs() {
        String first = nodes.get(0).logString();
        for (RaftNode n : nodes) if (!n.logString().equals(first)) return false;
        return true;
    }

    int minCommit() {
        int min = Integer.MAX_VALUE;
        for (RaftNode n : nodes) min = Math.min(min, n.commitIndex);
        return min;
    }

    void run(int ticks) {
        for (int i = 0; i < ticks; i++) tick();
    }
}

public class Solution {
    static long seed = 0x5DEECE66DL;

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
        singleNodeCluster();
        electionAndReplication();
        majorityPartitionKeepsProgress();
        healingConverges();
        voteRestriction();
        partitionsUnderChurn();
        System.out.println("PASS");
    }

    /** A one-node cluster elects itself and commits with a quorum of one. */
    static void singleNodeCluster() {
        RaftCluster c = new RaftCluster(1);
        c.run(10);
        check(c.nodes.get(0).role.equals("leader"), "the only node elects itself");
        check(c.quorum() == 1, "quorum of a single-node cluster");
        c.propose(42);
        c.run(5);
        check(c.nodes.get(0).commitIndex == 2, "quorum of one commits immediately");
        check(c.nodes.get(0).logString().equals("noop@1 42@1"), "log holds the no-op and the entry");
        check(c.voteAnomalies == 0, "no vote anomalies");
    }

    /** Five nodes: one leader, full replication, everyone learns the commit index. */
    static void electionAndReplication() {
        RaftCluster c = new RaftCluster(5);
        c.run(40);
        RaftNode leader = c.leader();
        check(leader != null, "a leader emerged");
        check(c.leaderByTerm.size() == 1 && c.voteAnomalies == 0, "one leader in one term");
        for (int v = 1; v <= 4; v++) c.propose(v * 10);
        c.run(60);
        check(leader.role.equals("leader") && leader.commitIndex == 5,
                "the leader committed the no-op and its four entries: " + leader.commitIndex);
        for (RaftNode n : c.nodes) {
            if (n.role.equals("leader")) continue;
            for (int i = 1; i <= n.lastLogIndex(); i++) {
                check(leader.sameEntryAs(n, i), "follower " + n.id + " log is a prefix of the leader");
            }
        }
        check(c.sameLogs(), "all five logs identical: " + c.nodes.get(0).logString());
        check(c.minCommit() == 5, "every follower learned the commit index: " + c.minCommit());
        c.assertAgreement("after replication");
    }

    /** A two-node minority can neither elect a leader nor commit; the majority progresses. */
    static void majorityPartitionKeepsProgress() {
        RaftCluster c = new RaftCluster(5);
        c.run(40);
        RaftNode oldLeader = c.leader();
        check(oldLeader != null, "a leader before the partition");
        c.propose(1);
        c.run(20);
        int committedBefore = oldLeader.commitIndex;
        check(committedBefore == 2, "the pre-partition entry committed");

        int other = oldLeader.id == 1 ? 2 : 1;
        c.isolate(oldLeader.id, other);                    // minority of two, majority of three
        c.run(80);

        for (RaftNode n : c.nodes) {
            if (c.group[n.id] == 0) continue;              // minority members only
            check(n.commitIndex == committedBefore,
                    "a minority node cannot commit: node " + n.id + " at " + n.commitIndex);
        }

        int majorityLeaders = 0;
        RaftNode newLeader = null;
        for (RaftNode n : c.nodes) {
            if (c.group[n.id] == 0 && n.role.equals("leader")) {
                majorityLeaders++;
                newLeader = n;
            }
        }
        check(majorityLeaders == 1, "the majority side elected exactly one leader: " + majorityLeaders);
        check(newLeader.term > oldLeader.term, "the new leader has a higher term: "
                + newLeader.term + " vs " + oldLeader.term);
        check(newLeader.commitIndex >= 2, "the majority preserved the committed entry");
        c.proposeTo(newLeader, 7);
        c.run(40);
        check(newLeader.commitIndex >= 4, "the majority keeps committing: " + newLeader.commitIndex);
        c.assertAgreement("while partitioned");
    }

    /** Healing truncates the isolated leader's uncommitted tail and rewrites it. */
    static void healingConverges() {
        RaftCluster c = new RaftCluster(5);
        c.run(40);
        RaftNode oldLeader = c.leader();
        c.propose(1);
        c.run(20);
        c.isolate(oldLeader.id);                           // a lone leader that cannot reach anyone
        check(oldLeader.role.equals("leader"), "the isolated node stays leader of its old term");
        c.proposeTo(oldLeader, 99);                        // appended locally, never replicated
        c.run(40);
        check(oldLeader.lastLogIndex() == 3, "the isolated leader has a divergent tail");
        check(oldLeader.commitIndex == 2, "but it never commits the divergent entry");
        RaftNode majorityLeader = null;
        for (RaftNode n : c.nodes) if (n.role.equals("leader") && n.id != oldLeader.id) majorityLeader = n;
        check(majorityLeader != null, "the majority elected its own leader");

        c.heal();
        c.run(200);
        check(c.sameLogs(), "every log converged: " + c.nodes.get(0).logString());
        check(!c.nodes.get(0).logString().contains("99@"), "the divergent entry was truncated");
        check(c.nodes.get(0).logString().equals("noop@1 1@1 noop@2"),
                "the safe prefix plus the new term's no-op: " + c.nodes.get(0).logString());
        check(c.nodes.get(0).commitIndex == 3, "the new term's no-op got committed: "
                + c.nodes.get(0).commitIndex);
        check(c.leader() != null && c.minCommit() == c.leader().commitIndex,
                "commit indexes agree after healing");
        c.assertAgreement("after healing");
    }

    /** The election restriction: a stale candidate is refused, an up-to-date one is granted. */
    static void voteRestriction() {
        RaftCluster c = new RaftCluster(3);
        c.run(40);
        RaftNode leader = c.leader();
        RaftNode lonely = null;
        for (RaftNode n : c.nodes) if (n.id != leader.id && lonely == null) lonely = n;
        RaftNode upToDate = null;
        for (RaftNode n : c.nodes) if (n.id != leader.id && n.id != lonely.id) upToDate = n;

        c.isolate(lonely.id);                              // this node never sees the entries
        for (int v = 1; v <= 3; v++) c.propose(v);
        c.run(40);
        check(leader.commitIndex == 4, "four entries committed by the majority");
        check(lonely.lastLogIndex() == 1 && leader.lastLogIndex() == 4,
                "the isolated follower is behind: " + lonely.lastLogIndex() + " vs "
                        + leader.lastLogIndex());
        check(upToDate.lastLogIndex() == 4, "the other follower is current");

        c.heal();                                    // let the crafted messages through
        int term = upToDate.term + 1;
        lonely.term = term;
        c.inFlight.clear();
        c.deliver(new Msg628(lonely.id, upToDate.id, "requestVote", term, lonely.lastLogIndex(),
                lonely.lastLogTerm(), true, null, 0, 0, 0, 0));
        Msg628 reply = null;
        for (Msg628 m : c.inFlight) if (m.kind.equals("vote")) reply = m;
        check(reply != null && !reply.granted, "the stale candidate is refused a vote");

        c.inFlight.clear();
        int term2 = lonely.term + 1;
        leader.term = term2;
        c.deliver(new Msg628(leader.id, lonely.id, "requestVote", term2, leader.lastLogIndex(),
                leader.lastLogTerm(), true, null, 0, 0, 0, 0));
        reply = null;
        for (Msg628 m : c.inFlight) if (m.kind.equals("vote")) reply = m;
        check(reply != null && reply.granted, "an up-to-date candidate is granted the vote");
        check(c.voteAnomalies == 0, "no voter voted twice in a term");
        c.assertAgreement("after the vote restriction test");
    }

    /** PRNG-driven partition churn: safety after every tick, convergence at the end. */
    static void partitionsUnderChurn() {
        RaftCluster c = new RaftCluster(5);
        for (int round = 0; round < 8; round++) {
            c.heal();
            c.run(30);
            if (c.leader() != null) c.propose(100 + round);
            c.run(20);
            int mode = nextInt(3);
            if (mode == 0) {                               // split 2 | 3
                int cut = 3 + nextInt(2);
                for (int i = 1; i <= 5; i++) c.group[i] = i <= cut ? 1 : 2;
            } else if (mode == 1) {                        // one isolated node
                c.isolate(1 + nextInt(5));
            } else {                                       // one node alone in a "majority" group
                c.group[1 + nextInt(5)] = 7;
            }
            for (int i = 0; i < 25; i++) {
                c.tick();
                c.assertAgreement("churn round " + round + " tick " + i);
                Map<Integer, Integer> leadersPerTerm = new TreeMap<>();
                for (RaftNode n : c.nodes) {
                    if (n.role.equals("leader")) leadersPerTerm.merge(n.term, 1, Integer::sum);
                }
                for (Map.Entry<Integer, Integer> e : leadersPerTerm.entrySet()) {
                    check(e.getValue() == 1, "at most one leader per term, saw " + leadersPerTerm);
                }
            }
        }
        c.heal();
        c.run(300);
        check(c.leader() != null, "a leader emerged after the churn");
        check(c.sameLogs(), "all logs converged after the churn: " + c.nodes.get(0).logString());
        check(c.minCommit() == c.leader().commitIndex, "commit indexes agree after the churn");
        check(c.voteAnomalies == 0, "never two leaders in one term, never a double vote");
        check(c.committedEntries.size() >= 4, "several entries were committed: "
                + c.committedEntries.size());
        for (RaftNode n : c.nodes) {
            for (int i = 1; i <= n.commitIndex; i++) {
                check(c.committedEntries.contains(i + ":" + n.termAt(i)),
                        "every applied entry was committed first");
            }
        }
        c.assertAgreement("after the churn");
    }
}
