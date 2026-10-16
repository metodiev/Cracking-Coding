/** Task 688 - Implement distance-vector routing (RIP) with split horizon. (Tier: Expert, CTCI Beyond - Networking, OS & Real-Time)
 *
 * A RIP routing domain: every router stores, for one destination prefix, its metric (hop count)
 * and the neighbour it learned the route from. Each round every router re-evaluates its
 * neighbours from their current advertisement; a router whose next hop is `v` *follows* v's
 * advertisement (so it inherits an increase, which is what makes the loop visible), while a
 * route over any other neighbour is only adopted when it is strictly better. Metric 16 means
 * "unreachable", so nothing beyond 15 hops is ever usable, and a route whose next hop is no
 * longer a neighbour is dropped at once because the interface went down.
 *
 * Advertisement policies:
 *   NONE              - advertise every known route to every neighbour;
 *   SPLIT_HORIZON     - stay silent towards the neighbour the route was learned from (that
 *                       neighbour keeps whatever it had);
 *   POISONED_REVERSE  - same, but advertise the route as 16 so the neighbour actively
 *                       invalidates it.
 *
 * The distinction is demonstrated on the classic count-to-infinity: after the destination's
 * attachment fails, the two neighbours of the failure exchange their stale promises and the
 * metric climbs two per round until it hits 16 (~8 rounds). Both horizon variants stop that
 * immediately, so no bogus route is ever installed. The climb needs the neighbour to
 * re-advertise before it learns of the failure, which the in-place (Gauss-Seidel) update order
 * used here reproduces deterministically.
 *
 * Verified: convergence to the true hop counts on random topologies for all three policies
 * (cross-checked against BFS), the 15-hop usability ceiling, the route-consistency invariant
 * dist[u] = 1 + dist[next[u]], loop-free forwarding, and the count-to-infinity trace itself.
 */
import java.util.*;

/** One destination prefix in a distance-vector domain. */
class RipDomain {
    static final int INFINITY = 16;

    enum Mode { NONE, SPLIT_HORIZON, POISONED_REVERSE }

    final int n;
    final Mode mode;
    private final boolean[][] linked;
    final int[] dist;
    final int[] nextHop;
    int destination;
    int rounds;
    private boolean attached = true;

    RipDomain(int n, Mode mode) {
        this.n = n;
        this.mode = mode;
        this.linked = new boolean[n][n];
        this.dist = new int[n];
        this.nextHop = new int[n];
    }

    void link(int a, int b) {
        linked[a][b] = true;
        linked[b][a] = true;
    }

    void unlink(int a, int b) {
        linked[a][b] = false;
        linked[b][a] = false;
    }

    boolean linked(int a, int b) {
        return linked[a][b];
    }

    /** Starts over with the prefix attached to `dest` (metric 0, no next hop). */
    void reset(int dest) {
        destination = dest;
        attached = true;
        Arrays.fill(dist, INFINITY);
        Arrays.fill(nextHop, -1);
        dist[dest] = 0;
        rounds = 0;
    }

    /** The prefix is no longer attached locally: the router has to learn a (bogus) route. */
    void detach(int router) {
        dist[router] = INFINITY;
        nextHop[router] = -1;
        if (router == destination) attached = false;
    }

    int advertisement(int from, int to) {
        boolean suppressed = mode != Mode.NONE && nextHop[from] == to;
        if (suppressed && mode == Mode.SPLIT_HORIZON) return -1; // -1 means "says nothing"
        return suppressed ? INFINITY : dist[from];
    }

    /** One in-place (Gauss-Seidel) update round; returns true when something changed. */
    boolean step() {
        boolean changed = false;
        // A next hop that is no longer a neighbour means the interface went down: RIP drops
        // routes pointing out of a failed interface immediately.
        for (int u = 0; u < n; u++) {
            if (nextHop[u] >= 0 && !linked[u][nextHop[u]]) {
                dist[u] = INFINITY;
                nextHop[u] = -1;
                changed = true;
            }
        }
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (v == u || !linked[u][v]) continue;
                int advertised = advertisement(v, u);
                if (advertised < 0) continue;
                int candidate = Math.min(INFINITY, 1 + advertised);
                if (nextHop[u] == v) {
                    // A router follows its own next hop, so it inherits an increase; once the
                    // metric reaches 16 the route is gone and the next hop is cleared.
                    int newHop = candidate >= INFINITY ? -1 : v;
                    if (candidate != dist[u] || newHop != nextHop[u]) {
                        dist[u] = candidate;
                        nextHop[u] = newHop;
                        changed = true;
                    }
                } else if (candidate < dist[u]) {
                    dist[u] = candidate;
                    nextHop[u] = v;
                    changed = true;
                }
            }
        }
        return changed;
    }

    int converge(int maxRounds) {
        while (rounds < maxRounds && step()) rounds++;
        return rounds;
    }

    /** Route-consistency problems, empty when the tables are coherent. */
    List<String> verify() {
        List<String> problems = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            if (u == destination && attached) {
                if (dist[u] != 0 || nextHop[u] != -1) problems.add("destination is not local");
                continue;
            }
            if (dist[u] >= INFINITY) {
                if (nextHop[u] != -1) problems.add("unreachable router keeps a next hop");
                continue;
            }
            if (nextHop[u] < 0 || !linked[u][nextHop[u]]) {
                problems.add("router " + u + " has no valid next hop");
                continue;
            }
            if (dist[u] != Math.min(INFINITY, 1 + dist[nextHop[u]])) {
                problems.add("router " + u + " metric is inconsistent with its next hop");
            }
        }
        return problems;
    }
}

public class Solution {
    private static long prngState = 0xA5A5A5A55A5A5A5AL;

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

    static RipDomain line(int length, RipDomain.Mode mode) {
        RipDomain net = new RipDomain(length, mode);
        for (int i = 0; i + 1 < length; i++) net.link(i, i + 1);
        return net;
    }

    /** Oracle: unweighted BFS hop counts, capped at RIP's unusable metric. */
    static int[] bfsHops(RipDomain net, int source) {
        int[] hops = new int[net.n];
        Arrays.fill(hops, RipDomain.INFINITY);
        hops[source] = 0;
        Deque<Integer> queue = new ArrayDeque<>();
        queue.add(source);
        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int next = 0; next < net.n; next++) {
                if (net.linked(node, next) && hops[next] == RipDomain.INFINITY) {
                    hops[next] = hops[node] + 1;
                    queue.add(next);
                }
            }
        }
        for (int i = 0; i < hops.length; i++) {
            if (hops[i] > 15) hops[i] = RipDomain.INFINITY;
        }
        return hops;
    }

    static String tableText(RipDomain net) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < net.n; i++) sb.append(net.dist[i]).append(i + 1 < net.n ? "," : "");
        return sb.toString();
    }

    public static void main(String[] args) {
        // 1. Healthy three-router line: metrics and next hops settle on the true hop counts.
        RipDomain healthy = line(3, RipDomain.Mode.NONE);
        healthy.reset(0);
        check(healthy.converge(50) <= 3, "healthy line converges in at most n rounds");
        check(tableText(healthy).equals("0,1,2"), "line metrics are hop counts");
        check(healthy.nextHop[1] == 0 && healthy.nextHop[2] == 1, "line next hops point at the prefix");
        check(healthy.verify().isEmpty(), "healthy line has no consistency problem");

        // 2. Count-to-infinity: the destination loses its local attachment.
        RipDomain none = line(3, RipDomain.Mode.NONE);
        none.reset(0);
        check(none.converge(10) <= 3 && tableText(none).equals("0,1,2"), "healthy state is 0,1,2");
        none.detach(0);
        List<Integer> trace = new ArrayList<>();
        for (int round = 0; round < 40; round++) {
            trace.add(none.dist[0]);
            if (!none.step()) break;
        }
        none.rounds = trace.size();
        int lowest = Integer.MAX_VALUE;
        for (int value : trace) lowest = Math.min(lowest, value);
        check(lowest < RipDomain.INFINITY, "a bogus route to the dead prefix really appeared");
        check(trace.get(0) == RipDomain.INFINITY, "the failed attachment starts unreachable");
        check(none.dist[0] == RipDomain.INFINITY && none.dist[1] == RipDomain.INFINITY
                && none.dist[2] == RipDomain.INFINITY, "count-to-infinity ends unreachable");
        check(trace.size() >= 6, "climbing to 16 takes several rounds: " + trace.size());
        check(none.nextHop[0] == -1 && none.nextHop[2] == -1, "routes are dropped once the metric is 16");

        // 3. Split horizon and poisoned reverse both stop it immediately and never install a route.
        RipDomain splitHorizon = line(3, RipDomain.Mode.SPLIT_HORIZON);
        splitHorizon.reset(0);
        splitHorizon.converge(10);
        splitHorizon.detach(0);
        int splitRounds = splitHorizon.converge(40);
        RipDomain poison = line(3, RipDomain.Mode.POISONED_REVERSE);
        poison.reset(0);
        poison.converge(10);
        poison.detach(0);
        int poisonRounds = poison.converge(40);
        check(splitRounds <= 2 && poisonRounds <= 2,
                "horizon variants converge at once: " + splitRounds + "/" + poisonRounds);
        check(splitRounds < trace.size() && poisonRounds < trace.size(),
                "horizon variants beat plain distance vector (" + trace.size() + " rounds)");
        check(tableText(splitHorizon).equals("16,16,16") && tableText(poison).equals("16,16,16"),
                "no router keeps a false route under a horizon policy");
        check(splitHorizon.verify().isEmpty() && poison.verify().isEmpty(),
                "horizon tables are consistent");

        // 4. Split horizon withholds precisely towards the next hop, nothing else.
        RipDomain advertise = line(3, RipDomain.Mode.SPLIT_HORIZON);
        advertise.reset(0);
        advertise.converge(10);
        check(advertise.advertisement(1, 0) == -1, "route is withheld from its own next hop");
        check(advertise.advertisement(1, 2) == 1, "other neighbours still receive the metric");
        RipDomain reverse = line(3, RipDomain.Mode.POISONED_REVERSE);
        reverse.reset(0);
        reverse.converge(10);
        check(reverse.advertisement(1, 0) == RipDomain.INFINITY,
                "poisoned reverse sends an explicit 16");

        // 5. The 15-hop ceiling: usable up to 15, unreachable beyond.
        RipDomain longLine = line(20, RipDomain.Mode.SPLIT_HORIZON);
        longLine.reset(0);
        longLine.converge(60);
        int[] hopOracle = bfsHops(longLine, 0);
        for (int node = 0; node < 20; node++) {
            check(longLine.dist[node] == hopOracle[node], "chain metric at node " + node);
        }
        check(longLine.dist[15] == 15, "hop 15 is still usable");
        check(longLine.dist[16] == RipDomain.INFINITY, "hop 16 is not usable in RIP");
        check(longLine.verify().isEmpty(), "chain tables are consistent");

        // 6. Random topologies: all three policies must agree with BFS, per destination.
        int mismatches = 0;
        for (int trial = 0; trial < 6; trial++) {
            RipDomain shape = new RipDomain(9, RipDomain.Mode.NONE);
            for (int node = 1; node < 9; node++) shape.link(node, nextInt(node));
            for (int a = 0; a < 9; a++) {
                for (int b = a + 1; b < 9; b++) {
                    if (!shape.linked(a, b) && nextInt(4) == 0) shape.link(a, b);
                }
            }
            for (int dest = 0; dest < 9; dest++) {
                int[] oracle = bfsHops(shape, dest);
                for (RipDomain.Mode mode : RipDomain.Mode.values()) {
                    RipDomain net = new RipDomain(9, mode);
                    for (int a = 0; a < 9; a++) {
                        for (int b = a + 1; b < 9; b++) if (shape.linked(a, b)) net.link(a, b);
                    }
                    net.reset(dest);
                    net.converge(200);
                    for (int node = 0; node < 9; node++) {
                        if (net.dist[node] != oracle[node]) mismatches++;
                    }
                    if (!net.verify().isEmpty()) mismatches++;
                }
            }
        }
        check(mismatches == 0, "6 random topologies x 9 destinations x 3 policies match BFS");

        // 7. Forwarding loop-freeness: following next hops reaches the destination exactly once.
        RipDomain routed = new RipDomain(10, RipDomain.Mode.SPLIT_HORIZON);
        for (int node = 1; node < 10; node++) routed.link(node, nextInt(node));
        for (int a = 0; a < 10; a++) {
            for (int b = a + 1; b < 10; b++) if (!routed.linked(a, b) && nextInt(5) == 0) routed.link(a, b);
        }
        for (int dest = 0; dest < 10; dest++) {
            routed.reset(dest);
            routed.converge(200);
            for (int start = 0; start < 10; start++) {
                if (routed.dist[start] >= RipDomain.INFINITY) continue;
                int current = start;
                int hops = 0;
                while (current != dest) {
                    current = routed.nextHop[current];
                    hops++;
                    check(hops <= 10, "forwarding walk from " + start + " to " + dest + " loops");
                }
                check(hops == routed.dist[start], "forwarding walk uses exactly metric hops");
            }
        }

        // 8. A dead end: the router whose only path to the prefix is gone stays unreachable.
        RipDomain island = new RipDomain(4, RipDomain.Mode.SPLIT_HORIZON);
        island.link(0, 1);
        island.link(1, 2);
        island.link(2, 3);
        island.reset(0);
        island.converge(20);
        check(tableText(island).equals("0,1,2,3"), "island test starts from a healthy chain");
        island.unlink(1, 2);
        island.converge(60);
        check(island.dist[2] == RipDomain.INFINITY && island.dist[3] == RipDomain.INFINITY,
                "the cut-off side of the network forgets the prefix");
        check(island.dist[1] == 1 && island.nextHop[1] == 0, "the attached side is unaffected");
        check(island.verify().isEmpty(), "island tables stay consistent");

        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
