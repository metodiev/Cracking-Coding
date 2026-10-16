/** Task 629 - Implement dynamic reconfiguration in Raft (joint consensus). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Reconfiguring a Raft cluster without losing the safety argument needs *joint consensus*:
 * the leader first appends C_old,new, an entry that carries both the old and the new
 * membership. While that configuration is the one in effect, every quorum decision - elections
 * and committing any entry at all - must be a majority of C_old *and* a majority of C_new. Only
 * once C_old,new is committed may the leader append C_new, and only once C_new is committed
 * does C_old disappear. Two more rules keep it sound: at most one configuration change may be in
 * flight, and the leader must be a member of C_new when it commits the new configuration.
 * The safety lever is purely combinatorial: any quorum of C_old and any quorum of C_new are
 * *sealed together* by the joint requirement, because a joint quorum already contains an old
 * majority and a new majority, and two majorities of the same set always intersect.
 *
 * Simulation model, and what is simplified: there are no threads and no real log replication.
 * A configuration change is a small state machine over four phases (old / joint-pending /
 * joint-committed-C_new-pending / new) with an explicitly supplied set of acknowledging nodes,
 * which is what makes the quorum algebra directly checkable. Vote requests, log truncation and
 * the rest of Raft are left out; the property that is *not* simplified is the quorum rule, and
 * it is verified by exhaustive enumeration of every majority of every configuration involved.
 *
 * Verified: (1) any two quorums of the same configuration intersect; (2) any joint quorum
 * intersects any old majority, any new majority and any other joint quorum; (3) an ack set that
 * is only an old majority, or only a new majority, is *not* a joint quorum and cannot commit;
 * (4) a second configuration change while one is in flight is rejected; (5) after C_new is
 * committed the old-only members no longer count for quorums; (6) a leader elected under the
 * joint rule needs majorities in both configurations; and (7) the negative control - switching
 * directly from C_old to a disjoint C_new with no joint phase - admits two disjoint quorums, so
 * two leaders can commit conflicting entries, which the joint rule forbids.
 */
import java.util.*;

/** Quorum algebra over node-id sets, including exhaustive enumeration for the proofs. */
class Quorums {
    static boolean majority(Set<Integer> config, Set<Integer> acked) {
        int count = 0;
        for (int member : config) if (acked.contains(member)) count++;
        return count > config.size() / 2;
    }

    static Set<Integer> union(Set<Integer> a, Set<Integer> b) {
        Set<Integer> out = new TreeSet<>(a);
        out.addAll(b);
        return out;
    }

    /** Every subset of `universe` that is a majority of `config`. */
    static List<Set<Integer>> allMajorities(Set<Integer> config, Set<Integer> universe) {
        List<Integer> members = new ArrayList<>(universe);
        int n = members.size();
        List<Set<Integer>> out = new ArrayList<>();
        for (int mask = 0; mask < (1 << n); mask++) {
            Set<Integer> acked = new TreeSet<>();
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) acked.add(members.get(i));
            if (majority(config, acked)) out.add(acked);
        }
        return out;
    }

    static boolean intersects(Set<Integer> a, Set<Integer> b) {
        for (int x : a) if (b.contains(x)) return true;
        return false;
    }

    static int choose(int n, int k) {
        int result = 1;
        for (int i = 0; i < k; i++) result = result * (n - i) / (i + 1);
        return result;
    }

    /** Number of majorities of a configuration, computed independently of the enumerator. */
    static int majorityCount(int size) {
        int total = 0;
        for (int k = size / 2 + 1; k <= size; k++) total += choose(size, k);
        return total;
    }
}

/** The four phases of a configuration change, with the quorum rule of each. */
class Reconfig {
    final Set<Integer> universe;
    Set<Integer> current;                     // configuration in effect
    Set<Integer> incoming;                    // C_new while a change is in flight, else null
    String phase = "C_old";
    boolean jointCommitted;
    Integer leader;
    int term;
    int rejectedChanges;
    final List<String> committed = new ArrayList<>();

    Reconfig(Set<Integer> oldConfig, Set<Integer> universe) {
        this.current = new TreeSet<>(oldConfig);
        this.universe = new TreeSet<>(universe);
    }

    boolean quorum(Set<Integer> acked) {
        if ("C_old".equals(phase)) return Quorums.majority(current, acked);
        if ("joint".equals(phase)) {
            return Quorums.majority(current, acked) && Quorums.majority(incoming, acked);
        }
        return Quorums.majority(current, acked);            // C_new is the configuration in effect
    }

    /** A candidate needs a quorum of the configuration currently in effect. */
    boolean campaign(int candidate, Set<Integer> voters) {
        Set<Integer> acked = new TreeSet<>(voters);
        acked.add(candidate);
        if (quorum(acked)) {
            leader = candidate;
            term++;
            return true;
        }
        return false;
    }

    boolean commit(String entry, Set<Integer> acked) {
        if (!quorum(acked)) return false;
        committed.add(entry);
        return true;
    }

    /** Phase 1: the leader appends C_old,new; no other change may be in flight. */
    boolean startChange(Set<Integer> newConfig) {
        if (incoming != null) {
            rejectedChanges++;
            return false;
        }
        incoming = new TreeSet<>(newConfig);
        phase = "joint";
        jointCommitted = false;
        return true;
    }

    /** Phase 2: commit the joint entry - needs majorities of both configurations. */
    boolean commitJoint(Set<Integer> acked) {
        if (!"joint".equals(phase)) return false;
        if (!commit("C_old,new", acked)) return false;
        jointCommitted = true;
        return true;
    }

    /** Phase 3: append C_new. Raft commits it with a majority of C_new itself. */
    boolean appendFinal(Set<Integer> acked) {
        if (!jointCommitted) return false;
        if (!Quorums.majority(incoming, acked)) return false;
        committed.add("C_new");
        phase = "C_new";
        current = new TreeSet<>(incoming);
        if (leader != null && !current.contains(leader)) leader = null;   // a leader outside C_new steps down
        return true;
    }

    /** Phase 4: C_new is committed, so the old-only members are gone. */
    void retireOldMembers() {
        incoming = null;
    }

    Set<Integer> configInEffect() {
        return "joint".equals(phase) ? incoming : current;
    }
}

public class Solution {
    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static Set<Integer> cfg(int... ids) {
        Set<Integer> s = new TreeSet<>();
        for (int id : ids) s.add(id);
        return s;
    }

    public static void main(String[] args) {
        sameConfigQuorumsIntersect();
        jointQuorumsIntersectEverything();
        partialAcksCannotCommit();
        oneChangeAtATime();
        electionNeedsBothMajorities();
        oldMembersRetired();
        naiveSwitchIsUnsafe();
        System.out.println("PASS");
    }

    /** Two majorities of one configuration always share a node - the base Raft argument. */
    static void sameConfigQuorumsIntersect() {
        int[][] raw = {{1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6, 7}};
        for (int[] ids : raw) {
            Set<Integer> config = cfg(ids);
            List<Set<Integer>> quorums = Quorums.allMajorities(config, config);
            check(quorums.size() == Quorums.majorityCount(config.size()),
                    "majority count of " + config + ": " + quorums.size());
            for (Set<Integer> a : quorums) {
                for (Set<Integer> b : quorums) {
                    check(Quorums.intersects(a, b), "two majorities of " + config + " must intersect: "
                            + a + " vs " + b);
                }
            }
        }
    }

    /** Every joint quorum overlaps every old quorum, every new quorum and every other joint one. */
    static void jointQuorumsIntersectEverything() {
        Set<Integer> universe = cfg(1, 2, 3, 4, 5, 6);
        int[][][] transitions = {
            {{1, 2, 3}, {3, 4, 5}},
            {{1, 2, 3}, {4, 5, 6}},              // disjoint: the hard case
            {{1, 2, 3, 4, 5}, {2, 3, 4}},
        };
        for (int[][] transition : transitions) {
            Set<Integer> oldCfg = cfg(transition[0]);
            Set<Integer> newCfg = cfg(transition[1]);
            List<Set<Integer>> jointQuorums = new ArrayList<>();
            for (Set<Integer> q : Quorums.allMajorities(oldCfg, universe)) {
                if (Quorums.majority(newCfg, q)) jointQuorums.add(q);
            }
            check(!jointQuorums.isEmpty(), "a joint quorum exists for " + oldCfg + " -> " + newCfg);
            for (Set<Integer> joint : jointQuorums) {
                for (Set<Integer> oldQ : Quorums.allMajorities(oldCfg, universe)) {
                    check(Quorums.intersects(joint, oldQ), "joint quorum meets every old quorum");
                }
                for (Set<Integer> newQ : Quorums.allMajorities(newCfg, universe)) {
                    check(Quorums.intersects(joint, newQ), "joint quorum meets every new quorum");
                }
                for (Set<Integer> other : jointQuorums) {
                    check(Quorums.intersects(joint, other), "two joint quorums intersect");
                }
            }
        }
    }

    /** An old-only or new-only ack set is not a joint quorum. */
    static void partialAcksCannotCommit() {
        Reconfig r = new Reconfig(cfg(1, 2, 3), cfg(1, 2, 3, 4, 5));
        r.campaign(1, cfg(2, 3));
        check(r.leader == 1 && r.term == 1, "a leader under C_old");
        check(r.commit("e1", cfg(1, 2)), "an old majority commits under C_old");
        check(!r.commit("e2", cfg(1)), "a single node cannot commit");

        check(r.startChange(cfg(3, 4, 5)), "the change starts");
        check("joint".equals(r.phase), "the joint configuration is in effect");
        check(!r.commit("e3", cfg(1, 2)), "an old-only majority cannot commit in the joint phase");
        check(!r.commit("e4", cfg(4, 5)), "a new-only majority cannot commit in the joint phase");
        check(!r.commitJoint(cfg(1, 2)), "the joint entry is not committed by an old majority alone");
        check(!r.commitJoint(cfg(3, 4)), "nor by a new majority alone");
        check(r.commitJoint(cfg(2, 3, 4)), "a joint quorum commits the joint entry");
        check(r.jointCommitted && r.quorum(cfg(2, 3, 4, 5)), "the joint quorum rule is still in force");
        check(!r.commit("e5", cfg(1, 2)), "still no old-only commits until C_new is committed");
        check(r.appendFinal(cfg(3, 4, 5)), "C_new is committed by a majority of C_new");
        check("C_new".equals(r.phase) && r.current.equals(cfg(3, 4, 5)), "C_new is now in effect");
        check(r.leader == null, "the old leader was not in C_new and stepped down");
        check(r.committed.equals(Arrays.asList("e1", "C_old,new", "C_new")),
                "the committed log: " + r.committed);
    }

    /** At most one configuration change may be in flight, and the leader must be in C_new. */
    static void oneChangeAtATime() {
        Reconfig r = new Reconfig(cfg(1, 2, 3), cfg(1, 2, 3, 4, 5, 6));
        r.campaign(1, cfg(2, 3));
        check(r.startChange(cfg(1, 3, 4)), "first change accepted");
        check(!r.startChange(cfg(4, 5, 6)), "the second change is rejected while one is in flight");
        check(r.rejectedChanges == 1, "the rejection is recorded");
        check(r.commitJoint(cfg(2, 3, 4)), "the joint entry commits");
        check(r.appendFinal(cfg(1, 3, 4)), "C_new commits with the leader inside it");
        check(r.leader == 1, "a leader inside C_new keeps leading");
        check(!r.startChange(cfg(2, 5)), "no further change until the previous one is retired");
        r.retireOldMembers();
        check(r.startChange(cfg(1, 2, 3)), "a new change can start once the old one is retired");
        check(r.rejectedChanges == 2, "the earlier rejections still stand");
    }

    /** A candidate needs a majority in both configurations during the joint phase. */
    static void electionNeedsBothMajorities() {
        Reconfig r = new Reconfig(cfg(1, 2, 3), cfg(1, 2, 3, 4, 5));
        r.startChange(cfg(3, 4, 5));
        check(!r.campaign(1, cfg(2, 3)), "old majority only: not elected");
        check(!r.campaign(4, cfg(5)), "new majority only (4 and 5): not elected");
        check(!r.campaign(4, cfg(3, 5)), "4 with 3 and 5 is 3 of 5 in C_new but 1 of 3 in C_old");
        check(r.campaign(3, cfg(1, 2, 4)), "a candidate with majorities in both wins");
        check(r.leader == 3 && r.term == 1, "the joint-phase leader is the one with both majorities");
        check(r.quorum(cfg(1, 2, 4, 5)), "the winning vote set is a joint quorum");
    }

    /** Once C_new is committed the old-only members stop counting. */
    static void oldMembersRetired() {
        Set<Integer> universe = cfg(1, 2, 3, 4, 5, 6);
        Reconfig r = new Reconfig(cfg(1, 2, 3), universe);
        r.campaign(1, cfg(2, 3));
        r.startChange(cfg(3, 4, 5));
        r.commitJoint(cfg(1, 3, 4));
        r.appendFinal(cfg(3, 4, 5));
        r.retireOldMembers();
        check(r.configInEffect().equals(cfg(3, 4, 5)), "C_new is the configuration in effect");
        check(r.quorum(cfg(3, 4)), "two of C_new commit");
        check(!r.quorum(cfg(1, 2)), "the old members alone cannot commit any more");
        check(!r.quorum(cfg(1, 2, 3)), "even the old members plus one are not a C_new majority");
        check(r.quorum(cfg(3, 4, 5)), "all of C_new commits");
        for (Set<Integer> q : Quorums.allMajorities(cfg(3, 4, 5), universe)) {
            check(Quorums.majority(cfg(3, 4, 5), q), "enumerated C_new quorums check out");
        }
    }

    /** Negative control: no joint phase and disjoint configurations admit disjoint quorums. */
    static void naiveSwitchIsUnsafe() {
        Set<Integer> oldCfg = cfg(1, 2, 3);
        Set<Integer> newCfg = cfg(4, 5, 6);
        List<Set<Integer>> oldQuorums = Quorums.allMajorities(oldCfg, cfg(1, 2, 3, 4, 5, 6));
        List<Set<Integer>> newQuorums = Quorums.allMajorities(newCfg, cfg(1, 2, 3, 4, 5, 6));
        boolean disjoint = false;
        for (Set<Integer> a : oldQuorums) {
            for (Set<Integer> b : newQuorums) {
                if (!Quorums.intersects(a, b)) disjoint = true;
            }
        }
        check(disjoint, "a direct switch to a disjoint configuration admits disjoint quorums");

        // Two leaders, elected at overlapping times by the two disjoint quorums, both commit.
        Reconfig naive = new Reconfig(oldCfg, cfg(1, 2, 3, 4, 5, 6));
        check(naive.campaign(1, cfg(2)) && naive.commit("A", cfg(1, 2)),
                "the old configuration commits entry A");
        Reconfig switched = new Reconfig(newCfg, cfg(1, 2, 3, 4, 5, 6));
        check(switched.campaign(4, cfg(5)) && switched.commit("B", cfg(4, 5)),
                "the new configuration commits entry B with a disjoint quorum");
        check(!Quorums.intersects(cfg(1, 2), cfg(4, 5)),
                "the two committing quorums share no node at all");
        check(naive.committed.equals(Arrays.asList("A")) && switched.committed.equals(Arrays.asList("B")),
                "both conflicting entries are committed - the safety violation");

        // With the joint phase the very same ack sets are rejected.
        Reconfig joint = new Reconfig(oldCfg, cfg(1, 2, 3, 4, 5, 6));
        joint.campaign(1, cfg(2));
        joint.startChange(newCfg);
        check(!joint.commit("B", cfg(4, 5)), "a new-only majority cannot commit during the joint phase");
        check(!joint.commit("A", cfg(1, 2)), "an old-only majority cannot either");
        check(joint.commit("joint", cfg(1, 2, 4, 5)), "only a joint quorum can");
    }
}
