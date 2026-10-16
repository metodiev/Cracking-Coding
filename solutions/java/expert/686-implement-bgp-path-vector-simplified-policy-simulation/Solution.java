/** Task 686 - Implement BGP path-vector simplified policy simulation. (Tier: Expert, CTCI Beyond - Networking, OS & Real-Time)
 *
 * Each AS holds a routing information base of (prefix, AS-path) entries. Every round, in
 * synchronous Bellman-Ford style, an AS rebuilds its RIB by asking what each neighbour chooses
 * to advertise, rejecting any advertisement whose AS-path already contains itself (loop
 * prevention) and rejecting advertisements its neighbour's export policy blocks. It then keeps
 * the best candidate under the classic BGP decision key:
 *
 *     highest LOCAL_PREF, then shortest AS-path, then lexicographically smallest AS-path
 *     (the last rule is only there to make the outcome unique - real routers break the tie on
 *     MED / IGP cost / router-id, which behaves the same way: pick one deterministically).
 *
 * LOCAL_PREF is derived from the relationship the route arrived over, the Gao-Rexford ranking
 * customer > peer > provider. Export policy is valley-free: an AS may hand a route to a
 * customer freely, but to a provider or peer it may only export its own prefix or a route it
 * learned from one of its customers. Without that rule routes leak across peer links and traffic
 * rides a path nobody is paid for - the hand-built topology below shows a strictly shorter path
 * being suppressed by exactly that rule.
 *
 * Invariants verified: the fixed point is reached in at most n rounds; every RIB path starts at
 * the origin, ends at the AS, visits adjacent ASes, repeats no AS and is valley-free; and no
 * admissible neighbour advertisement would improve any entry (the fixpoint condition).
 * Cross-checked against two independent oracles: BFS for hop counts and DFS enumeration of all
 * simple paths for the optimal path, plus a policy-free run of the same topology.
 *
 * What is simplified: routers here exchange whole RIB snapshots synchronously and there is no
 * MRAI timer, no message loss, no route flap damping and only one destination prefix.
 */
import java.util.*;

/**
 * Path-vector simulator. `relation[a][b]` records what AS b is to AS a:
 * 'c' b is a customer of a, 'p' b is a provider of a, 's' a peer, '-' no adjacency.
 */
class BgpSim {
    static final int LP_CUSTOMER = 200;
    static final int LP_PEER = 100;
    static final int LP_PROVIDER = 50;

    final int n;
    final int origin;
    final boolean policy;
    final boolean useLocalPref;
    final char[][] relation;
    final int[][] rib;
    int[] ribLocalPref;
    int rounds;
    long loopRejections;
    long policyRejections;

    BgpSim(int n, int origin, boolean policy, boolean useLocalPref) {
        this.n = n;
        this.origin = origin;
        this.policy = policy;
        this.useLocalPref = useLocalPref;
        this.relation = new char[n][n];
        this.rib = new int[n][];
        this.ribLocalPref = new int[n];
        for (char[] row : relation) Arrays.fill(row, '-');
    }

    static char opposite(char relationship) {
        if (relationship == 'c') return 'p';
        if (relationship == 'p') return 'c';
        return 's';
    }

    /** Declares that `b` is, from `a`'s point of view, a customer/provider/peer of `a`. */
    void link(int a, int b, char bFromA) {
        relation[a][b] = bFromA;
        relation[b][a] = opposite(bFromA);
    }

    void unlink(int a, int b) {
        relation[a][b] = '-';
        relation[b][a] = '-';
    }

    boolean adjacent(int a, int b) {
        return relation[a][b] != '-';
    }

    /** Valley-free export: to a customer always, otherwise only own or customer-learned routes. */
    boolean exportAllowed(int exporter, int receiver, int[] path) {
        if (!policy) return true;
        if (relation[exporter][receiver] == 'c') return true;
        if (path.length == 1) return true;
        int learnedFrom = path[path.length - 2];
        return relation[exporter][learnedFrom] == 'c';
    }

    int localPrefAt(int receiver, int from) {
        if (!useLocalPref) return 0;
        switch (relation[receiver][from]) {
            case 'c': return LP_CUSTOMER;
            case 's': return LP_PEER;
            default: return LP_PROVIDER;
        }
    }

    static boolean contains(int[] path, int as) {
        for (int v : path) if (v == as) return true;
        return false;
    }

    static boolean better(int candidatePref, int[] candidate, int incumbentPref, int[] incumbent) {
        if (candidatePref != incumbentPref) return candidatePref > incumbentPref;
        if (candidate.length != incumbent.length) return candidate.length < incumbent.length;
        for (int i = 0; i < candidate.length; i++) {
            if (candidate[i] != incumbent[i]) return candidate[i] < incumbent[i];
        }
        return false;
    }

    static int[] extend(int[] path, int as) {
        int[] out = Arrays.copyOf(path, path.length + 1);
        out[path.length] = as;
        return out;
    }

    void run() {
        int[][] current = new int[n][];
        int[] currentPref = new int[n];
        current[origin] = new int[]{origin};
        currentPref[origin] = Integer.MAX_VALUE;
        for (rounds = 1; rounds <= n + 1; rounds++) {
            int[][] next = new int[n][];
            int[] nextPref = new int[n];
            next[origin] = new int[]{origin};
            nextPref[origin] = Integer.MAX_VALUE;
            boolean changed = false;
            for (int as = 0; as < n; as++) {
                if (as == origin) continue;
                int[] bestPath = null;
                int bestPref = Integer.MIN_VALUE;
                for (int neighbour = 0; neighbour < n; neighbour++) {
                    if (!adjacent(as, neighbour)) continue;
                    int[] advertised = current[neighbour];
                    if (advertised == null) continue;
                    if (contains(advertised, as)) {
                        loopRejections++;
                        continue;
                    }
                    if (!exportAllowed(neighbour, as, advertised)) {
                        policyRejections++;
                        continue;
                    }
                    int pref = localPrefAt(as, neighbour);
                    if (bestPath == null || better(pref, advertised, bestPref, bestPath)) {
                        bestPath = advertised;
                        bestPref = pref;
                    }
                }
                next[as] = bestPath == null ? null : extend(bestPath, as);
                nextPref[as] = bestPath == null ? 0 : bestPref;
                if (!Arrays.equals(next[as], current[as])) changed = true;
            }
            current = next;
            currentPref = nextPref;
            if (!changed) break;
        }
        for (int i = 0; i < n; i++) {
            rib[i] = current[i] == null ? null : current[i].clone();
            ribLocalPref[i] = currentPref[i];
        }
    }

    /** Checks the fixpoint condition and all structural invariants; returns the violations. */
    List<String> verify() {
        List<String> problems = new ArrayList<>();
        for (int as = 0; as < n; as++) {
            int[] path = rib[as];
            if (path == null) continue;
            if (path[0] != origin) problems.add("AS " + as + " path does not start at the origin");
            if (path[path.length - 1] != as) problems.add("AS " + as + " path does not end at itself");
            for (int i = 1; i < path.length; i++) {
                if (!adjacent(path[i - 1], path[i])) problems.add("AS " + as + " path uses a missing link");
                if (!exportAllowed(path[i - 1], path[i], Arrays.copyOf(path, i))) {
                    problems.add("AS " + as + " path violates valley-free policy");
                }
            }
            Set<Integer> seen = new TreeSet<>();
            for (int v : path) seen.add(v);
            if (seen.size() != path.length) problems.add("AS " + as + " path repeats an AS");
            if (as == origin && path.length != 1) problems.add("origin has a non-trivial path");
            for (int neighbour = 0; neighbour < n; neighbour++) {
                if (!adjacent(as, neighbour)) continue;
                int[] advertised = rib[neighbour];
                if (advertised == null || contains(advertised, as)) continue;
                if (!exportAllowed(neighbour, as, advertised)) continue;
                int[] candidate = extend(advertised, as);
                int pref = localPrefAt(as, neighbour);
                if (path == null || better(pref, candidate, ribLocalPref[as], path)) {
                    problems.add("AS " + as + " could improve via AS " + neighbour);
                }
            }
        }
        return problems;
    }
}

public class Solution {
    private static long prngState = 0x5DEECE66DL;

    static long nextLong() {
        prngState += 0x9E3779B97F4A7C15L;
        long z = prngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), (long) bound);
    }

    static String pathText(int[] path) {
        if (path == null) return "unreachable";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < path.length; i++) {
            if (i > 0) sb.append('-');
            sb.append(path[i]);
        }
        return sb.toString();
    }

    /** Oracle 1: unweighted BFS hop counts. */
    static int[] bfsHops(BgpSim sim, int source) {
        int[] hops = new int[sim.n];
        Arrays.fill(hops, -1);
        hops[source] = 0;
        Deque<Integer> queue = new ArrayDeque<>();
        queue.add(source);
        while (!queue.isEmpty()) {
            int as = queue.poll();
            for (int next = 0; next < sim.n; next++) {
                if (sim.adjacent(as, next) && hops[next] < 0) {
                    hops[next] = hops[as] + 1;
                    queue.add(next);
                }
            }
        }
        return hops;
    }

    /** Oracle 2: enumerate every simple path and keep the shortest, ties broken lexicographically. */
    static int[] enumerateBest(BgpSim sim, int target, boolean[] used, Deque<Integer> trail) {
        if (trail.getLast() == target) {
            int[] path = new int[trail.size()];
            int i = 0;
            for (int v : trail) path[i++] = v;
            return path;
        }
        int[] best = null;
        for (int next = 0; next < sim.n; next++) {
            if (!sim.adjacent(trail.getLast(), next) || used[next]) continue;
            used[next] = true;
            trail.addLast(next);
            int[] candidate = enumerateBest(sim, target, used, trail);
            trail.removeLast();
            used[next] = false;
            if (candidate == null) continue;
            if (best == null || BgpSim.better(0, candidate, 0, best)) best = candidate;
        }
        return best;
    }

    static int[] enumerateBest(BgpSim sim, int target) {
        boolean[] used = new boolean[sim.n];
        used[sim.origin] = true;
        Deque<Integer> trail = new ArrayDeque<>();
        trail.add(sim.origin);
        return enumerateBest(sim, target, used, trail);
    }

    /** Random connected topology used for the cross-checks. */
    static BgpSim randomTopology(int n, int origin, boolean policy) {
        BgpSim sim = new BgpSim(n, origin, policy, policy);
        for (int as = 1; as < n; as++) {
            int parent = nextInt(as);
            sim.link(parent, as, 'c');
        }
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                if (sim.adjacent(a, b) || nextInt(3) != 0) continue;
                char[] kinds = {'c', 'p', 's'};
                sim.link(a, b, kinds[nextInt(3)]);
            }
        }
        return sim;
    }

    public static void main(String[] args) {
        // 1. Line topology, policy disabled: RIB paths must be the unique shortest paths.
        BgpSim line = new BgpSim(5, 0, false, false);
        line.link(0, 1, 'c');
        line.link(1, 2, 'c');
        line.link(2, 3, 'c');
        line.link(3, 4, 'c');
        line.run();
        check(line.rounds <= 5, "line network converges in at most n rounds: " + line.rounds);
        for (int as = 0; as < 5; as++) {
            check(pathText(line.rib[as]).equals(expectedLinePath(as)), "line path for AS " + as);
        }
        check(line.verify().isEmpty(), "line fixpoint has no violations");

        // 2. Random topologies, policy disabled: BFS and exhaustive enumeration must both agree.
        for (int trial = 0; trial < 4; trial++) {
            BgpSim sim = randomTopology(6, 0, false);
            sim.run();
            int[] hops = bfsHops(sim, 0);
            for (int target = 0; target < 6; target++) {
                int[] expected = enumerateBest(sim, target);
                check(pathText(sim.rib[target]).equals(pathText(expected)),
                        "trial " + trial + " exhaustive path for AS " + target);
                check(sim.rib[target].length - 1 == hops[target],
                        "trial " + trial + " hop count from BFS for AS " + target);
            }
            check(sim.verify().isEmpty(), "trial " + trial + " fixpoint invariants");
        }

        // 3. Valley-free policy: a lexicographically smaller 3-hop path is suppressed, so the
        //    receiver must fall back to the other 3-hop path. Without the policy the first wins.
        BgpSim valleyOff = valleyTopology(false, true);
        valleyOff.run();
        check(pathText(valleyOff.rib[3]).equals("0-1-3"), "policy off: shortest-lex path taken");
        BgpSim valleyOn = valleyTopology(true, true);
        valleyOn.run();
        check(pathText(valleyOn.rib[3]).equals("0-4-3"), "policy on: peer route not exported to a provider");
        check(valleyOn.policyRejections > valleyOff.policyRejections,
                "policy on counts the suppressed advertisement");
        check(valleyOn.verify().isEmpty(), "policy topology stays valley-free");

        // 4. LOCAL_PREF: with local preference disabled the shorter peer path wins; enabling it
        //    makes the router prefer the longer but customer-learned path.
        BgpSim noPref = preferenceTopology(false);
        noPref.run();
        check(pathText(noPref.rib[4]).equals("0-3-4"), "local pref off: AS-path length decides");
        BgpSim withPref = preferenceTopology(true);
        withPref.run();
        check(pathText(withPref.rib[4]).equals("0-1-2-4"), "local pref on: customer route preferred");
        check(withPref.ribLocalPref[4] == BgpSim.LP_CUSTOMER, "winning route carries customer preference");
        check(noPref.ribLocalPref[4] == 0, "local pref disabled means the key has no preference part");

        // 5. Random topologies with the full policy: invariants plus a stable fixpoint.
        for (int trial = 0; trial < 4; trial++) {
            BgpSim sim = randomTopology(7, 0, true);
            sim.run();
            check(sim.rounds <= 7, "policy run converges quickly");
            check(sim.verify().isEmpty(), "policy trial " + trial + " invariants (loop/valley/fixpoint)");
            check(sim.rib[0] != null && sim.rib[0].length == 1, "origin always keeps its own prefix");
            for (int as = 0; as < 7; as++) {
                int[] path = sim.rib[as];
                if (path == null || path.length == 1) continue;
                int[] parentPath = sim.rib[path[path.length - 2]];
                check(parentPath != null && parentPath.length == path.length - 1,
                        "policy trial " + trial + " AS " + as + " extends its parent's route");
                for (int i = 0; i < parentPath.length; i++) {
                    check(parentPath[i] == path[i], "policy trial " + trial + " route prefix consistent");
                }
            }
        }

        // 6. Withdrawal by link failure: ASes cut off from the origin lose the prefix entirely.
        BgpSim outage = new BgpSim(4, 0, false, false);
        outage.link(0, 1, 'c');
        outage.link(1, 2, 'c');
        outage.link(2, 3, 'c');
        outage.run();
        check(pathText(outage.rib[3]).equals("0-1-2-3"), "before the outage AS 3 has a path");
        outage.unlink(1, 2);
        BgpSim recovered = new BgpSim(4, 0, false, false);
        for (int a = 0; a < 4; a++) {
            for (int b = 0; b < 4; b++) {
                if (outage.relation[a][b] != '-') recovered.link(a, b, outage.relation[a][b]);
            }
        }
        recovered.run();
        check(recovered.rib[3] == null && recovered.rib[2] == null,
                "ASes beyond the failed link withdraw the route");
        check(pathText(recovered.rib[1]).equals("0-1"), "AS 1 keeps its path");
        int[] reachable = bfsHops(recovered, 0);
        for (int as = 0; as < 4; as++) {
            check((recovered.rib[as] != null) == (reachable[as] >= 0), "withdrawal matches reachability");
        }

        // 7. Determinism: a second run of the same topology reproduces the RIB byte for byte.
        BgpSim again = valleyTopology(true, true);
        again.run();
        boolean identical = true;
        for (int as = 0; as < 4; as++) {
            if (!pathText(again.rib[as]).equals(pathText(valleyOn.rib[as]))) identical = false;
        }
        check(identical && again.rounds == valleyOn.rounds, "simulation is deterministic");

        System.out.println("PASS");
    }

    static String expectedLinePath(int as) {
        StringBuilder sb = new StringBuilder("0");
        for (int i = 1; i <= as; i++) sb.append('-').append(i);
        return sb.toString();
    }

    /** AS 3 can hear 0-1-3 (1 would send a peer-learned route to its provider) or 0-4-3. */
    static BgpSim valleyTopology(boolean policy, boolean useLocalPref) {
        BgpSim sim = new BgpSim(5, 0, policy, useLocalPref);
        sim.link(0, 1, 's');   // 0 and 1 are peers
        sim.link(1, 3, 'p');   // 3 is a provider of 1, so 1 must not hand it a peer-learned route
        sim.link(0, 4, 'c');   // 4 is a customer of 0
        sim.link(4, 3, 'c');   // 3 is a customer of 4, so 4 forwards freely
        return sim;
    }

    /** AS 4 hears 0-3-4 over a peer link (3 hops) and 0-1-2-4 over a customer link (4 hops). */
    static BgpSim preferenceTopology(boolean useLocalPref) {
        BgpSim sim = new BgpSim(5, 0, true, useLocalPref);
        sim.link(3, 0, 'c');   // 0 is a customer of 3, so 3 may forward it upwards
        sim.link(3, 4, 's');   // 3 and 4 are peers
        sim.link(1, 0, 'c');   // 0 is a customer of 1
        sim.link(2, 1, 'c');   // 1 is a customer of 2
        sim.link(4, 2, 'c');   // 2 is a customer of 4
        return sim;
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
