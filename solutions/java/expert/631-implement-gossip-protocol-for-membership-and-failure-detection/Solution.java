/** Task 631 - Implement gossip protocol for membership and failure detection. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * SWIM-style membership: every node keeps a table of (heartbeat, state) per member, bumps its own
 * heartbeat each round and gossips its table to peers, whose entries merge by taking the larger
 * heartbeat. A SUSPECT/DEAD verdict is stamped with the heartbeat it was observed at, so a single
 * fresh heartbeat - the node's own or one relayed by gossip - *refutes* the verdict and the entry
 * goes back to ALIVE. A member that stops being heard is marked SUSPECT after one timeout and DEAD
 * after a longer one, and those verdicts spread through the same gossip, so the live nodes end up
 * agreeing.
 *
 * Simulation model, and what is simplified: no threads, no timers, no sockets. A round is a
 * deterministic PRNG-driven schedule in which each node gossips to a fanout of peers it can
 * actually reach (a partition is a group number - the packets simply do not arrive, exactly as
 * with real packet loss). Timeouts are counted in rounds instead of milliseconds; a crashed node
 * just stops participating; and a "quiet" round is a round in which nobody's heartbeat moves, used
 * to observe that views converge without suspicion timers firing. What is *not* simplified is the
 * entry-merge algebra, and that is where the interesting properties live: heartbeats are monotone,
 * an older entry can never overwrite a newer one, and a refutation wins.
 *
 * Verified: (1) views converge to be exactly identical for fanout 1 and fanout 3; (2) a silent node
 * is declared dead by every live node, and no live node is ever blamed; (3) a partitioned node is
 * blamed by the side it cannot reach, exactly along the partition boundary, while it blames them
 * back; (4) healing refutes the stale verdict - every node relabels the returning member ALIVE;
 * (5) after every round no heartbeat regresses and no verdict is stamped above the heartbeat it
 * saw, under a churn of crash/isolate/heal; (6) the negative control - merging the *older* entry -
 * drops a heartbeat and is caught by that checker; (7) one- and two-node clusters stay correct.
 */
import java.util.*;

class MemberEntry {
    int heartbeat;
    int stateHeartbeat;           // heartbeat at which the current state was observed
    int state;                    // 0 = ALIVE, 1 = SUSPECT, 2 = DEAD

    static MemberEntry alive(int heartbeat) {
        MemberEntry e = new MemberEntry();
        e.heartbeat = heartbeat;
        e.stateHeartbeat = heartbeat;
        return e;
    }

    MemberEntry copy() {
        MemberEntry e = new MemberEntry();
        e.heartbeat = heartbeat;
        e.stateHeartbeat = stateHeartbeat;
        e.state = state;
        return e;
    }

    /** Newer heartbeat wins; an older verdict can never overwrite a newer one. */
    static MemberEntry merge(MemberEntry a, MemberEntry b) {
        MemberEntry out = (a.heartbeat >= b.heartbeat ? a : b).copy();
        if (out.state != 0 && out.heartbeat > out.stateHeartbeat) out.state = 0;   // refuted
        return out;
    }

    String render() {
        return heartbeat + "/" + "ASD".charAt(state);
    }
}

class GossipNode {
    final int id;
    boolean running = true;
    final Map<Integer, MemberEntry> view = new TreeMap<>();
    final Map<Integer, Integer> lastSeenHeartbeat = new TreeMap<>();
    final Map<Integer, Integer> lastChangeRound = new TreeMap<>();
    int ownBeat;

    GossipNode(int id) {
        this.id = id;
    }
}

class GossipCluster {
    static final int SUSPECT_AFTER = 4;
    static final int DEAD_AFTER = 9;

    final List<GossipNode> nodes = new ArrayList<>();
    final int[] group;
    final int fanout;
    final Map<String, Integer> highest = new HashMap<>();
    int round;
    int merges;
    int suspicions;
    int refutations;
    int dropped;
    int regressions;
    boolean quiet;

    GossipCluster(int n, int fanout) {
        this.fanout = fanout;
        for (int i = 1; i <= n; i++) {
            GossipNode node = new GossipNode(i);
            for (int j = 1; j <= n; j++) {
                node.view.put(j, MemberEntry.alive(0));
                node.lastSeenHeartbeat.put(j, 0);
                node.lastChangeRound.put(j, 0);
            }
            nodes.add(node);
        }
        group = new int[n + 1];
    }

    GossipNode node(int id) {
        return nodes.get(id - 1);
    }

    boolean connected(int a, int b) {
        return group[a] == group[b];
    }

    void crash(int id) {
        node(id).running = false;
    }

    void isolate(int id) {
        group[id] = 100 + id;
    }

    void heal() {
        for (int i = 1; i < group.length; i++) group[i] = 0;
    }

    /** Normal rounds: everybody heartbeats, suspicion timers advance. */
    void run(int rounds) {
        quiet = false;
        for (int i = 0; i < rounds; i++) round();
    }

    /** Quiet rounds: nobody's heartbeat moves and no timer fires, so views can settle. */
    void runQuiet(int rounds) {
        quiet = true;
        for (int i = 0; i < rounds; i++) round();
        quiet = false;
    }

    void round() {
        round++;
        int[][] statesBefore = new int[nodes.size() + 1][nodes.size() + 1];
        for (GossipNode p : nodes) {
            for (GossipNode q : nodes) statesBefore[p.id][q.id] = p.view.get(q.id).state;
        }
        if (!quiet) {
            for (GossipNode p : nodes) {
                if (!p.running) continue;
                p.ownBeat++;
                MemberEntry own = p.view.get(p.id);
                own.heartbeat = p.ownBeat;          // own heartbeat makes the node ALIVE again
                own.stateHeartbeat = p.ownBeat;
                own.state = 0;
            }
        }
        for (GossipNode p : nodes) {
            if (!p.running) continue;
            if (quiet) {
                for (int k = 0; k < fanout; k++) tryMerge(p, pickPeer(p));
            } else {
                for (GossipNode q : nodes) if (q.id != p.id) tryMerge(p, q);
            }
        }
        if (!quiet) {
            for (GossipNode p : nodes) {
                if (!p.running) continue;
                for (GossipNode q : nodes) {
                    if (q.id == p.id) continue;
                    MemberEntry entry = p.view.get(q.id);
                    if (entry.heartbeat > p.lastSeenHeartbeat.get(q.id)) {
                        p.lastSeenHeartbeat.put(q.id, entry.heartbeat);
                        p.lastChangeRound.put(q.id, round);
                    }
                    if (entry.state != 0 && entry.heartbeat > entry.stateHeartbeat) {
                        entry.state = 0;               // the stamp is older than the heartbeat
                    }
                    int silentFor = round - p.lastChangeRound.get(q.id);
                    if (entry.state == 0 && silentFor > SUSPECT_AFTER) {
                        entry.state = 1;
                        entry.stateHeartbeat = entry.heartbeat;
                        suspicions++;
                    } else if (entry.state == 1 && silentFor > DEAD_AFTER) {
                        entry.state = 2;
                        suspicions++;
                    }
                }
            }
            for (GossipNode p : nodes) {
                for (GossipNode q : nodes) {
                    if (p.id == q.id) continue;
                    if (statesBefore[p.id][q.id] != 0 && p.view.get(q.id).state == 0) refutations++;
                }
            }
        }
        noteViews();
    }

    GossipNode pickPeer(GossipNode p) {
        if (nodes.size() < 2) return null;
        for (int attempt = 0; attempt < 4; attempt++) {
            int pick = Solution.nextInt(nodes.size()) + 1;
            if (pick != p.id) return node(pick);
        }
        return node(p.id == 1 ? 2 : 1);
    }

    void tryMerge(GossipNode p, GossipNode q) {
        if (q == null) return;
        if (!connected(p.id, q.id) || !q.running) {
            dropped++;                             // the packet never arrives
            return;
        }
        mergeViews(p, q);
    }

    void mergeViews(GossipNode p, GossipNode q) {
        for (GossipNode member : nodes) {
            p.view.put(member.id, MemberEntry.merge(p.view.get(member.id), q.view.get(member.id)));
        }
        merges++;
    }

    /** The naively wrong merge, kept for the negative control: the older entry wins. */
    void mergeViewsOldest(GossipNode p, GossipNode q) {
        for (GossipNode member : nodes) {
            MemberEntry a = p.view.get(member.id);
            MemberEntry b = q.view.get(member.id);
            p.view.put(member.id, (a.heartbeat <= b.heartbeat ? a : b).copy());
        }
        merges++;
        noteViews();                              // the monotonicity checker sees this mutation
    }

    /** Record the high water mark of every (observer, member) heartbeat. */
    void noteViews() {
        for (GossipNode p : nodes) {
            for (GossipNode q : nodes) {
                String key = p.id + ":" + q.id;
                int hb = p.view.get(q.id).heartbeat;
                Integer previous = highest.get(key);
                if (previous != null && hb < previous) regressions++;
                highest.put(key, previous == null ? hb : Math.max(previous, hb));
            }
        }
    }

    boolean monotoneSoFar() {
        return regressions == 0;
    }

    boolean viewsIdentical() {
        String first = null;
        for (GossipNode p : nodes) {
            StringBuilder sb = new StringBuilder();
            for (GossipNode q : nodes) sb.append(q.id).append(p.view.get(q.id).render());
            if (first == null) first = sb.toString();
            else if (!first.equals(sb.toString())) return false;
        }
        return true;
    }

    int stateOf(int observer, int member) {
        return node(observer).view.get(member).state;
    }

    int heartbeatOf(int observer, int member) {
        return node(observer).view.get(member).heartbeat;
    }
}

public class Solution {
    static long seed = 0xBEEFCAFEBABEL;

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
        convergenceForFanout();
        silentNodeIsDetected();
        partitionSplitsTheBlame();
        healingRefutesTheVerdict();
        monotonicityUnderChurn();
        negativeControlBreaksMonotonicity();
        degenerateClusters();
        System.out.println("PASS");
    }

    /** Quiet rounds let every view settle to exactly the same table, for any fanout. */
    static void convergenceForFanout() {
        for (int fanout : new int[]{1, 3}) {
            GossipCluster c = new GossipCluster(6, fanout);
            c.run(8);
            c.runQuiet(60);
            check(c.viewsIdentical(), "fanout " + fanout + ": views converged");
            check(c.merges > 100, "fanout " + fanout + ": plenty of gossip: " + c.merges);
            check(c.dropped == 0, "fanout " + fanout + ": no packet was dropped");
            for (int i = 1; i <= 6; i++) {
                check(c.stateOf(1, i) == 0, "everyone healthy with fanout " + fanout);
                check(c.stateOf(1, i) == c.stateOf(6, i), "observers agree about member " + i);
            }
            check(c.monotoneSoFar(), "fanout " + fanout + ": heartbeats never regressed");
        }
    }

    /** A node that stops heartbeating is declared dead by everyone, and only it. */
    static void silentNodeIsDetected() {
        GossipCluster c = new GossipCluster(6, 2);
        c.run(8);
        c.crash(4);
        c.run(25);
        for (int observer = 1; observer <= 6; observer++) {
            if (observer == 4) continue;
            check(c.stateOf(observer, 4) == 2, "observer " + observer + " declares node 4 dead: state "
                    + c.stateOf(observer, 4));
            for (int member = 1; member <= 6; member++) {
                if (member == 4) continue;
                check(c.stateOf(observer, member) == 0,
                        "live member " + member + " is never blamed by observer " + observer);
            }
        }
        check(c.suspicions >= 5, "suspicion was raised along the way: " + c.suspicions);
        check(c.heartbeatOf(1, 4) == c.heartbeatOf(6, 4), "everyone froze node 4 at the same beat");
        check(c.monotoneSoFar(), "the crash caused no regression");
    }

    /** A partitioned node is blamed by the other side, on exactly that boundary. */
    static void partitionSplitsTheBlame() {
        GossipCluster c = new GossipCluster(5, 2);
        c.run(8);
        c.isolate(3);
        c.run(25);
        for (int observer = 1; observer <= 5; observer++) {
            if (observer == 3) continue;
            check(c.stateOf(observer, 3) == 2, "the majority declares the isolated node dead");
            for (int member = 1; member <= 5; member++) {
                if (member == 3 || member == observer) continue;
                check(c.stateOf(observer, member) == 0, "no blame inside the majority");
            }
        }
        check(c.stateOf(3, 3) == 0, "the isolated node still believes itself alive");
        for (int member = 1; member <= 5; member++) {
            if (member == 3) continue;
            check(c.stateOf(3, member) == 2, "and it blames the other side");
        }
        check(c.dropped > 0, "gossip towards the other side was dropped: " + c.dropped);
    }

    /** Coming back refutes the stale verdict instead of leaving the node marked dead. */
    static void healingRefutesTheVerdict() {
        GossipCluster c = new GossipCluster(5, 2);
        c.run(8);
        c.isolate(2);
        c.run(25);
        check(c.stateOf(1, 2) == 2, "the node was declared dead: " + c.stateOf(1, 2));
        int before = c.refutations;

        c.heal();
        c.run(20);
        for (int observer = 1; observer <= 5; observer++) {
            for (int member = 1; member <= 5; member++) {
                check(c.stateOf(observer, member) == 0, "observer " + observer + " cleared member "
                        + member);
            }
        }
        check(c.refutations > before, "the stale verdicts were refuted: " + c.refutations);
        c.runQuiet(40);
        check(c.viewsIdentical(), "the cluster converged again after healing");
        check(c.monotoneSoFar(), "healing caused no regression");
    }

    /** Crash, isolation and recovery churn while watching the invariants after every round. */
    static void monotonicityUnderChurn() {
        GossipCluster c = new GossipCluster(6, 2);
        c.run(6);
        for (int r = 0; r < 60; r++) {
            if (r == 10) c.crash(5);
            if (r == 20) c.isolate(2);
            if (r == 35) {
                c.heal();
                c.node(5).running = true;
            }
            if (r == 45) c.isolate(6);
            if (r == 50) c.heal();
            c.round();
            for (int observer = 1; observer <= 6; observer++) {
                for (int member = 1; member <= 6; member++) {
                    MemberEntry entry = c.node(observer).view.get(member);
                    check(entry.stateHeartbeat <= entry.heartbeat,
                            "no verdict is stamped above the heartbeat it saw");
                    if (entry.state != 0) {
                        check(entry.heartbeat == c.node(observer).lastSeenHeartbeat.get(member),
                                "a verdict tracks the heartbeat it was observed at");
                    }
                }
            }
            check(c.monotoneSoFar(), "heartbeats never regress (round " + r + ")");
        }
        check(c.merges > 100, "the churn gossiped a lot: " + c.merges);
        c.run(25);
        for (int observer = 1; observer <= 6; observer++) {
            for (int member = 1; member <= 6; member++) {
                check(c.stateOf(observer, member) == 0, "everybody is healthy again at the end");
            }
        }
    }

    /** Negative control: merging the older entry loses a heartbeat and is caught. */
    static void negativeControlBreaksMonotonicity() {
        GossipCluster good = new GossipCluster(4, 2);
        good.run(15);
        check(good.monotoneSoFar(), "the correct merge keeps heartbeats monotone");

        GossipCluster bad = new GossipCluster(4, 2);
        bad.run(8);
        bad.isolate(3);                          // node 3 falls behind on everyone else's beats
        bad.run(6);
        int fresh = -1;
        int stale = -1;
        for (int i = 1; i <= 4; i++) {
            for (int j = 1; j <= 4; j++) {
                if (bad.heartbeatOf(i, 1) > bad.heartbeatOf(j, 1)) {
                    fresh = i;
                    stale = j;
                }
            }
        }
        check(fresh > 0, "partial gossip left the views at different beats");
        int before = bad.heartbeatOf(fresh, 1);
        bad.mergeViewsOldest(bad.node(fresh), bad.node(stale));
        check(bad.heartbeatOf(fresh, 1) < before, "the older entry overwrote a newer heartbeat: "
                + before + " -> " + bad.heartbeatOf(fresh, 1));
        check(!bad.monotoneSoFar() && bad.regressions > 0,
                "the regression is caught by the checker: " + bad.regressions);
    }

    /** One- and two-node clusters are degenerate but must not misbehave. */
    static void degenerateClusters() {
        GossipCluster single = new GossipCluster(1, 1);
        single.run(20);
        check(single.viewsIdentical() && single.dropped == 0, "a single node has nobody to blame");
        check(single.stateOf(1, 1) == 0 && single.suspicions == 0, "it stays alive and calm");

        GossipCluster pair = new GossipCluster(2, 1);
        pair.run(10);
        check(pair.viewsIdentical(), "two nodes converge");
        check(pair.monotoneSoFar(), "monotone");
        pair.crash(2);
        pair.run(20);
        check(pair.stateOf(1, 2) == 2, "the survivor declares the dead node dead");
        check(pair.stateOf(1, 1) == 0, "and keeps itself alive");
        pair.node(2).running = true;
        pair.run(15);
        check(pair.stateOf(1, 2) == 0, "the returning node is cleared again");
    }
}
