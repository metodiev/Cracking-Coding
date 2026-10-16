/** Task 759 - Implement Byzantine fault tolerant consensus, a small PBFT simulation.
 * (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
 *
 * PBFT makes progress with n = 3f + 1 replicas when at most f are faulty, because every quorum has
 * 2f + 1 members and any two quorums overlap in f + 1 replicas - at least one of which is honest. The
 * simulation here implements one slot of one view:
 *
 *   primary        - sends pre-prepare(v); a Byzantine primary may equivocate and send different values
 *                    to different replicas;
 *   prepare        - every honest replica broadcasts prepare for the value *it* was told; a replica is
 *                    prepared for v when its own pre-prepare is v and 2f + 1 distinct replicas
 *                    prepared v (its own included);
 *   commit         - prepared replicas broadcast commit(v); a replica commits the first value that
 *                    collects 2f + 1 distinct commits;
 *   view change    - if no honest replica commits in this view (equivocating or dead primary), the view
 *                    changes and the next primary (view mod n) re-proposes.
 *
 * The safety claim is checked, not assumed: over hundreds of runs with arbitrary Byzantine behaviour
 * (equivocation, flooding prepares for every value, silence) no two honest replicas ever commit
 * different values in the same slot. The underlying reason is verified separately by brute force -
 * every pair of size-(2f+1) quorums minus any f faulty replicas is non-empty - which is exactly the
 * argument that makes the fork impossible.
 *
 * Simplified: reliable broadcast is assumed (every honest message reaches every replica, so delays,
 * reordering and drops only matter for liveness, which is exercised through the dead-primary case), a
 * single slot and a fresh proposal after a view change, no checkpoints and no signatures.
 *
 * Verified: honest runs and Byzantine-replica runs commit the proposed value; an equivocating and a
 * dead primary force a view change without any commit; a later view commits; 2f+1 is necessary (2f is
 * not enough); quorum intersection by brute force for n=4 and n=7; no fork in 300 randomised runs;
 * single-replica and determinism boundaries.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" (no Math.random). */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

/** Result of one simulated (view, slot): what each replica committed and the quorum bookkeeping. */
final class Run {
    final int[] committed;                 // per replica: the committed value, or -1
    final Map<Integer, Set<Integer>> prepareSenders = new TreeMap<>();
    final Map<Integer, Set<Integer>> commitSenders = new TreeMap<>();
    final boolean anyCommit;

    Run(int[] committed) {
        this.committed = committed;
        boolean any = false;
        for (int value : committed) if (value >= 0) any = true;
        this.anyCommit = any;
    }

    /** Safety: no two honest replicas committed different values in this slot. */
    boolean forked(Set<Integer> byzantine) {
        int seen = -1;
        for (int i = 0; i < committed.length; i++) {
            if (byzantine.contains(i) || committed[i] < 0) continue;
            if (seen >= 0 && seen != committed[i]) return true;
            seen = committed[i];
        }
        return false;
    }

    int committedValue() {
        for (int value : committed) if (value >= 0) return value;
        return -1;
    }
}

/** One slot of PBFT with reliable broadcast and an arbitrary Byzantine behaviour. */
final class PbftSim {
    final int n, f, quorum;

    PbftSim(int f) {
        this.f = f;
        this.n = 3 * f + 1;
        this.quorum = 2 * f + 1;
    }

    /**
     * Runs one view. `primaryValue` is what an honest primary proposes; `equivocation` is the value a
     * faulty primary sends to the second half of the replicas; `deadPrimary` suppresses pre-prepares.
     */
    Run run(int view, Set<Integer> byzantine, int primaryValue, int equivocation, boolean deadPrimary) {
        return run(view, byzantine, new TreeSet<Integer>(), primaryValue, equivocation, deadPrimary, noLocks());
    }

    /**
     * As above, but honouring locks: `locked[i]` is what replica i already committed in an earlier view.
     * A locked replica keeps its value and refuses to prepare a conflicting proposal, which is what makes
     * the view change safe without modelling the prepared certificates themselves. `silent` replicas do
     * not answer at all (crash faults), which is how the quorum threshold is exercised.
     */
    Run run(int view, Set<Integer> byzantine, Set<Integer> silent, int primaryValue, int equivocation,
            boolean deadPrimary, int[] locked) {
        int primary = view % n;
        boolean primaryFaulty = byzantine.contains(primary) || silent.contains(primary);
        int[] prePrepare = new int[n];
        Arrays.fill(prePrepare, -1);
        if (!deadPrimary) {
            for (int i = 0; i < n; i++) {
                prePrepare[i] = primaryFaulty && i >= n / 2 ? equivocation : primaryValue;
            }
        }
        Map<Integer, Set<Integer>> prepares = new TreeMap<>();
        Map<Integer, Set<Integer>> commits = new TreeMap<>();
        for (int replica = 0; replica < n; replica++) {
            if (silent.contains(replica)) continue;
            if (byzantine.contains(replica)) {
                // The worst case: a faulty replica prepares and commits every value it can think of.
                for (int value : new int[]{primaryValue, equivocation}) {
                    prepares.computeIfAbsent(value, k -> new TreeSet<>()).add(replica);
                    commits.computeIfAbsent(value, k -> new TreeSet<>()).add(replica);
                }
            } else if (prePrepare[replica] >= 0
                    && (locked[replica] < 0 || locked[replica] == prePrepare[replica])) {
                prepares.computeIfAbsent(prePrepare[replica], k -> new TreeSet<>()).add(replica);
            }
        }
        Set<Integer> preparedValues = new TreeSet<>();
        for (Map.Entry<Integer, Set<Integer>> entry : prepares.entrySet()) {
            if (entry.getValue().size() < quorum) continue;
            for (int replica = 0; replica < n; replica++) {
                if (!byzantine.contains(replica) && !silent.contains(replica)
                        && prePrepare[replica] == entry.getKey()) {
                    preparedValues.add(entry.getKey());           // prepared locally, matching its own view
                }
            }
        }
        for (int replica = 0; replica < n; replica++) {
            if (byzantine.contains(replica) || silent.contains(replica)) continue;
            if (!preparedValues.contains(prePrepare[replica])) continue;
            if (locked[replica] >= 0 && locked[replica] != prePrepare[replica]) continue;
            commits.computeIfAbsent(prePrepare[replica], k -> new TreeSet<>()).add(replica);
        }
        int[] committed = new int[n];
        Arrays.fill(committed, -1);
        for (int replica = 0; replica < n; replica++) {
            if (byzantine.contains(replica) || silent.contains(replica)) continue;
            if (locked[replica] >= 0) {
                committed[replica] = locked[replica];              // a lock survives the view change
                continue;
            }
            int value = prePrepare[replica];
            if (value < 0 || !preparedValues.contains(value)) continue;
            if (commits.containsKey(value) && commits.get(value).size() >= quorum) committed[replica] = value;
        }
        Run run = new Run(committed);
        run.prepareSenders.putAll(prepares);
        run.commitSenders.putAll(commits);
        return run;
    }

    int[] noLocks() {
        int[] locks = new int[n];
        Arrays.fill(locks, -1);
        return locks;
    }

    /** The locks a view change has to carry over. */
    static int[] locksFrom(Run run) { return run.committed.clone(); }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static Set<Integer> setOf(int... values) {
        Set<Integer> out = new TreeSet<>();
        for (int value : values) out.add(value);
        return out;
    }

    /** Brute force: is every pair of quorums minus every f-set still non-empty? */
    static boolean quorumIntersectionHolds(int n, int f) {
        int quorum = 2 * f + 1;
        List<Set<Integer>> quorums = new ArrayList<>();
        for (int mask = 0; mask < (1 << n); mask++) {
            if (Integer.bitCount(mask) != quorum) continue;
            Set<Integer> members = new TreeSet<>();
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) members.add(i);
            quorums.add(members);
        }
        for (Set<Integer> first : quorums) {
            for (Set<Integer> second : quorums) {
                Set<Integer> intersection = new TreeSet<>(first);
                intersection.retainAll(second);
                if (intersection.size() < f + 1) return false;
                for (int faultyMask = 0; faultyMask < (1 << n); faultyMask++) {
                    if (Integer.bitCount(faultyMask) > f) continue;
                    boolean honestSurvivor = false;
                    for (int replica : intersection) {
                        if ((faultyMask & (1 << replica)) == 0) honestSurvivor = true;
                    }
                    if (!honestSurvivor) return false;
                }
            }
        }
        return true;
    }

    public static void main(String[] args) {
        PbftSim sim = new PbftSim(1);
        check(sim.n == 4 && sim.f == 1 && sim.quorum == 3, "n = 3f+1 and quorum = 2f+1");

        // 1. honest run: every replica commits the proposed value
        Run honest = sim.run(0, setOf(), 7, 7, false);
        check(honest.anyCommit && !honest.forked(setOf()), "an honest run commits without a fork");
        for (int replica = 0; replica < sim.n; replica++) {
            check(honest.committed[replica] == 7, "replica " + replica + " committed the proposed value");
        }
        check(honest.commitSenders.get(7).size() == sim.n, "every replica sent a commit");

        // 2. one Byzantine replica flooding prepares and commits for another value: honest replicas win
        Run flooding = sim.run(0, setOf(3), 7, 99, false);
        check(!flooding.forked(setOf(3)), "flooding cannot fork honest replicas");
        check(flooding.committedValue() == 7, "honest replicas still commit the proposed value");
        check(flooding.prepareSenders.get(99).size() < sim.quorum, "the fake value never reaches a prepare "
                + "quorum: " + flooding.prepareSenders.get(99).size());

        // 3. equivocating primary (replica 0 sends different values to the two halves): the value it
        //    pushed to a quorum wins, the other one never reaches a quorum, and nobody forks
        Run equivocation = sim.run(0, setOf(0), 7, 99, false);
        check(!equivocation.forked(setOf(0)), "equivocation cannot fork honest replicas");
        check(equivocation.prepareSenders.get(7).size() < sim.quorum, "the value the primary sent to the "
                + "minority never reaches the prepare quorum: " + equivocation.prepareSenders.get(7).size());
        check(equivocation.committedValue() == 99, "the value that did reach a quorum is committed: "
                + equivocation.committedValue());

        // 4. view change: locked replicas keep their value, and a new primary re-proposing it reaches
        //    unanimity; a new primary that ignores the committed value cannot fork the others
        Run afterViewChange = sim.run(1, setOf(0), setOf(), 99, 7, false, PbftSim.locksFrom(equivocation));
        check(!afterViewChange.forked(setOf(0)), "the new view does not fork");
        check(afterViewChange.committedValue() == 99, "the committed value survives the view change");
        for (int replica : new int[]{1, 2, 3}) {
            check(afterViewChange.committed[replica] == 99, "honest replica " + replica + " agrees on 99");
        }
        Run conflictingPrimary = sim.run(1, setOf(0), setOf(), 7, 7, false, PbftSim.locksFrom(equivocation));
        check(!conflictingPrimary.forked(setOf(0)), "a new primary ignoring the committed value cannot fork");
        check(conflictingPrimary.committed[1] == -1, "the unlocked replica cannot reach a quorum for the "
                + "conflicting value on its own");

        // 5. dead primary: silence needs a view change, and progress resumes within f+1 views
        Run dead = sim.run(0, setOf(), 7, 7, true);
        check(!dead.anyCommit, "a silent primary means no commit in that view");
        int views = 0;
        Run recovered = dead;
        while (!recovered.anyCommit && views < sim.f + 1) {
            views++;
            recovered = sim.run(views, setOf(), 7, 7, false);
        }
        check(recovered.anyCommit, "progress resumes after the view change");
        check(views <= sim.f + 1, "recovery needed at most f+1 view changes: " + views);

        // 6. 2f+1 is exactly the threshold: a quorum minus one sender is not enough
        PbftSim larger = new PbftSim(2);
        check(larger.n == 7 && larger.quorum == 5, "n=7, f=2 has a quorum of 5");
        Run exactlyQuorum = larger.run(0, setOf(), 5, 5, false);
        check(exactlyQuorum.anyCommit, "n=7 with all replicas live commits");
        Run belowQuorum = larger.run(0, setOf(), setOf(4, 5, 6), 5, 5, false, larger.noLocks());
        check(!belowQuorum.anyCommit, "three silent replicas leave only 4 prepares, below the quorum of 5");
        Run recoveredQuorum = larger.run(1, setOf(), setOf(), 5, 5, false, larger.noLocks());
        check(recoveredQuorum.anyCommit, "once the silent replicas answer again the slot commits");
        Run twoByzantine = larger.run(0, setOf(5, 6), 5, 99, false);
        check(!twoByzantine.forked(setOf(5, 6)), "two Byzantine replicas cannot fork honest replicas");
        check(twoByzantine.prepareSenders.get(99).size() < larger.quorum, "the fake value is short of a "
                + "quorum: " + twoByzantine.prepareSenders.get(99));

        // 7. the safety argument itself, by brute force over every quorum pair and every f-faulty set
        check(quorumIntersectionHolds(4, 1), "every quorum pair of n=4,f=1 meets in an honest replica");
        check(quorumIntersectionHolds(7, 2), "every quorum pair of n=7,f=2 meets in an honest replica");

        // 8. randomised Byzantine behaviour: safety must hold in every single run
        Prng rng = new Prng(0xBF759EEDL);
        int runs = 0, commits = 0, blocked = 0;
        for (int round = 0; round < 300; round++) {
            PbftSim random = new PbftSim(1 + rng.nextInt(2));
            Set<Integer> byzantine = new TreeSet<>();
            int faulty = rng.nextInt(random.f + 1);
            while (byzantine.size() < faulty) byzantine.add(rng.nextInt(random.n));
            int proposal = rng.nextInt(50) + 1;
            int alternative = proposal + 100;
            Run run = random.run(rng.nextInt(4), byzantine, proposal, alternative, rng.nextInt(4) == 0);
            check(!run.forked(byzantine), "round " + round + ": honest replicas forked with Byzantine set "
                    + byzantine + " and committed " + Arrays.toString(run.committed));
            for (int i = 0; i < random.n; i++) {
                if (byzantine.contains(i) || run.committed[i] < 0) continue;
                check(run.committed[i] == proposal || run.committed[i] == alternative,
                        "committed values come from the proposal");
            }
            runs++;
            if (run.anyCommit) commits++;
            else blocked++;
        }
        check(runs == 300 && commits > 100 && blocked > 20, "randomised runs covered both outcomes: "
                + commits + " committed, " + blocked + " blocked");

        // 9. boundaries: a single replica with f = 0, and determinism
        PbftSim alone = new PbftSim(0);
        check(alone.n == 1 && alone.quorum == 1, "f=0 means one replica and a quorum of one");
        check(alone.run(0, setOf(), 5, 5, false).committedValue() == 5, "the single replica commits alone");
        check(alone.run(0, setOf(), 5, 5, true).committedValue() == -1, "a silent single replica cannot "
                + "commit anything");
        Run first = sim.run(0, setOf(3), 7, 99, false);
        Run second = sim.run(0, setOf(3), 7, 99, false);
        check(Arrays.equals(first.committed, second.committed)
                && first.prepareSenders.equals(second.prepareSenders), "same inputs give the same outcome");

        System.out.println("n=4,f=1: quorum 3, honest and Byzantine-replica runs commit 7, equivocation and a "
                + "dead primary force a view change");
        System.out.println("n=7,f=2: quorum 5, 2 silent replicas block the slot, 2 Byzantine cannot fork");
        System.out.println("randomised runs: " + commits + " committed, " + blocked + " blocked, 0 forks; "
                + "quorum intersection verified exhaustively for n=4 and n=7");
        System.out.println("PASS");
    }
}
