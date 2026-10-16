/** Task 687 - Implement routing algorithm link-state (OSPF) simulation with LSAs. (Tier: Expert, CTCI Beyond - Networking, OS & Real-Time)
 *
 * Every router originates a Router-LSA: its own links with their costs and a sequence number.
 * LSAs are flooded with reliable hop-by-hop propagation - a transmission is acknowledged, and an
 * unacknowledged (lost) transmission is retransmitted in the next round - while a higher
 * sequence number replaces an older LSA and a lower one is discarded. Flooding stops when every
 * router has acknowledged every LSA it received, which is what makes the link-state database
 * synchronous.
 *
 * Once the LSDBs are synchronised each router runs Dijkstra on the graph described by its own
 * copy of the database and derives a next-hop table. Because every router holds the same links,
 * the resulting shortest-path trees agree, which is the defining property of link-state routing
 * (and the reason OSPF needs no split horizon).
 *
 * Verified invariants: all LSDBs contain exactly the same (router, sequence) pairs; each router's
 * distance vector equals an independent Floyd-Warshall reference on the flat topology; next hops
 * satisfy dist[u] = cost(u, nh) + dist[nh]; packet walks follow next hops without looping and use
 * exactly dist[destination] hops; lossy links really were retransmitted; stale LSAs are rejected
 * and equal-sequence LSAs are not re-flooded; after a cost change and after a link failure every
 * router reconverges to the reference answer.
 */
import java.util.*;

/** Router-LSA: one router's links and the sequence number of that advertisement. */
class Lsa {
    final int routerId;
    final int seq;
    final int[] peer;
    final int[] cost;

    Lsa(int routerId, int seq, int[] peer, int[] cost) {
        this.routerId = routerId;
        this.seq = seq;
        this.peer = peer;
        this.cost = cost;
    }

    String signature() {
        StringBuilder sb = new StringBuilder(routerId + "@" + seq + ":");
        for (int i = 0; i < peer.length; i++) sb.append(peer[i]).append('/').append(cost[i]).append(' ');
        return sb.toString();
    }
}

/** Deterministic link-state routing domain with reliable LSA flooding. */
class OspfNetwork {
    static final int INF = 1 << 20;

    final int n;
    final boolean lossyLinks;
    private final int[][] cost;
    private final int[] sequence;
    private final List<Map<Integer, Lsa>> lsdb = new ArrayList<>();
    private final List<Map<Integer, Set<Integer>>> pending = new ArrayList<>();
    private final Map<Long, Integer> attempts = new HashMap<>();
    long transmissions;
    long losses;
    long staleDrops;
    long duplicateDrops;
    int floodRounds;

    OspfNetwork(int n, boolean lossyLinks) {
        this.n = n;
        this.lossyLinks = lossyLinks;
        this.cost = new int[n][n];
        this.sequence = new int[n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(cost[i], INF);
            cost[i][i] = 0;
            lsdb.add(new TreeMap<>());
            pending.add(new TreeMap<>());
        }
    }

    void setLink(int a, int b, int weight) {
        cost[a][b] = weight;
        cost[b][a] = weight;
    }

    void removeLink(int a, int b) {
        cost[a][b] = INF;
        cost[b][a] = INF;
    }

    int linkCost(int a, int b) {
        return cost[a][b];
    }

    List<Integer> peersOf(int router) {
        List<Integer> peers = new ArrayList<>();
        for (int i = 0; i < n; i++) if (i != router && cost[router][i] < INF) peers.add(i);
        return peers;
    }

    private Lsa buildLsa(int router) {
        List<Integer> peers = peersOf(router);
        int[] peer = new int[peers.size()];
        int[] weight = new int[peers.size()];
        for (int i = 0; i < peers.size(); i++) {
            peer[i] = peers.get(i);
            weight[i] = cost[router][peers.get(i)];
        }
        return new Lsa(router, ++sequence[router], peer, weight);
    }

    /** First origination: every router floods its own LSA and starts with an empty database. */
    void originateAll() {
        for (int r = 0; r < n; r++) {
            lsdb.get(r).clear();
            pending.get(r).clear();
        }
        attempts.clear();
        for (int r = 0; r < n; r++) install(r, buildLsa(r), -1);
    }

    /** Re-origination after a local change: one router bumps its sequence number. */
    void originate(int router) {
        install(router, buildLsa(router), -1);
    }

    /**
     * Installs an LSA into `router`'s database, flooding it onwards when it is newer. Returns
     * true when it was accepted, false when the sequence number was not an improvement.
     */
    boolean install(int router, Lsa lsa, int receivedFrom) {
        Lsa stored = lsdb.get(router).get(lsa.routerId);
        if (stored != null && lsa.seq <= stored.seq) {
            if (lsa.seq == stored.seq) duplicateDrops++;
            else staleDrops++;
            return false;
        }
        lsdb.get(router).put(lsa.routerId, lsa);
        for (int peer : peersOf(router)) {
            if (peer == receivedFrom) continue;
            addPending(router, lsa.routerId, peer);
        }
        return true;
    }

    private void addPending(int router, int owner, int peer) {
        pending.get(router).computeIfAbsent(owner, k -> new TreeSet<>()).add(peer);
    }

    private void dropPending(int router, int owner, int peer) {
        Set<Integer> targets = pending.get(router).get(owner);
        if (targets == null) return;
        targets.remove(peer);
        if (targets.isEmpty()) pending.get(router).remove(owner);
    }

    private boolean hasPending() {
        for (Map<Integer, Set<Integer>> perRouter : pending) if (!perRouter.isEmpty()) return true;
        return false;
    }

    private boolean lossy(int lsdbOwner, int sender, int peer) {
        if (!lossyLinks) return false;
        long key = ((long) sender * 1024 + lsdbOwner) * 1024 + peer;
        int attempt = attempts.merge(key, 1, Integer::sum);
        return (sender * 31 + peer * 17 + lsdbOwner * 7 + attempt * 5) % 11 == 3;
    }

    /** Runs rounds of flooding until every transmission has been acknowledged. */
    int flood() {
        floodRounds = 0;
        while (hasPending()) {
            if (++floodRounds > 200) throw new IllegalStateException("flooding did not converge");
            List<int[]> work = new ArrayList<>();
            for (int r = 0; r < n; r++) {
                for (Map.Entry<Integer, Set<Integer>> entry : pending.get(r).entrySet()) {
                    for (int peer : entry.getValue()) work.add(new int[]{r, entry.getKey(), peer});
                }
            }
            work.sort((x, y) -> {
                if (x[0] != y[0]) return Integer.compare(x[0], y[0]);
                if (x[1] != y[1]) return Integer.compare(x[1], y[1]);
                return Integer.compare(x[2], y[2]);
            });
            for (int[] item : work) {
                int sender = item[0];
                int owner = item[1];
                int peer = item[2];
                Lsa lsa = lsdb.get(sender).get(owner);
                if (lsa == null) {
                    dropPending(sender, owner, peer);
                    continue;
                }
                transmissions++;
                if (lossy(owner, sender, peer)) {
                    losses++;
                    continue;
                }
                dropPending(sender, owner, peer);
                install(peer, lsa, sender);
            }
        }
        return floodRounds;
    }

    /** Dijkstra on the graph described by one router's own copy of the database. */
    int[] distances(int source) {
        int[] dist = new int[n];
        boolean[] done = new boolean[n];
        Arrays.fill(dist, INF);
        dist[source] = 0;
        for (int step = 0; step < n; step++) {
            int best = -1;
            for (int v = 0; v < n; v++) if (!done[v] && (best < 0 || dist[v] < dist[best])) best = v;
            if (best < 0 || dist[best] >= INF) break;
            done[best] = true;
            for (int[] edge : edges(best, source)) {
                int to = edge[0];
                int candidate = dist[best] + edge[1];
                if (candidate < dist[to]) dist[to] = candidate;
            }
        }
        return dist;
    }

    /** Next hop for every destination, taken from the same shortest-path tree. */
    int[] nextHops(int source) {
        int[] dist = new int[n];
        int[] via = new int[n];
        boolean[] done = new boolean[n];
        Arrays.fill(dist, INF);
        Arrays.fill(via, -1);
        dist[source] = 0;
        for (int step = 0; step < n; step++) {
            int best = -1;
            for (int v = 0; v < n; v++) if (!done[v] && (best < 0 || dist[v] < dist[best])) best = v;
            if (best < 0 || dist[best] >= INF) break;
            done[best] = true;
            for (int[] edge : edges(best, source)) {
                int to = edge[0];
                if (dist[best] + edge[1] < dist[to]) {
                    dist[to] = dist[best] + edge[1];
                    via[to] = best;
                }
            }
        }
        int[] next = new int[n];
        for (int dest = 0; dest < n; dest++) {
            if (dest == source || dist[dest] >= INF) {
                next[dest] = -1;
                continue;
            }
            int hop = dest;
            while (via[hop] != source && via[hop] != -1) hop = via[hop];
            next[dest] = hop;
        }
        return next;
    }

    /** Links advertised for `router` by the LSDB of `viewer`. */
    private List<int[]> edges(int router, int viewer) {
        List<int[]> out = new ArrayList<>();
        for (Lsa lsa : lsdb.get(viewer).values()) {
            if (lsa.routerId != router) continue;
            for (int i = 0; i < lsa.peer.length; i++) out.add(new int[]{lsa.peer[i], lsa.cost[i]});
        }
        return out;
    }

    String databaseSignature(int router) {
        StringBuilder sb = new StringBuilder();
        for (Lsa lsa : lsdb.get(router).values()) sb.append(lsa.signature());
        return sb.toString();
    }

    /** Reference oracle: Floyd-Warshall on the flat adjacency matrix. */
    int[][] allPairsReference() {
        int[][] dist = new int[n][n];
        for (int i = 0; i < n; i++) dist[i] = cost[i].clone();
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
        return dist;
    }

    long transmissions() {
        return transmissions;
    }

    long losses() {
        return losses;
    }

    long staleDrops() {
        return staleDrops;
    }

    long duplicateDrops() {
        return duplicateDrops;
    }

    Lsa stored(int router, int owner) {
        return lsdb.get(router).get(owner);
    }
}

public class Solution {
    /** A labelled six-router domain; every link is bidirectional at equal cost. */
    static OspfNetwork pentagon(boolean lossy) {
        OspfNetwork net = new OspfNetwork(6, lossy);
        net.setLink(0, 1, 4);
        net.setLink(1, 2, 3);
        net.setLink(2, 3, 5);
        net.setLink(3, 4, 2);
        net.setLink(4, 5, 6);
        net.setLink(5, 0, 7);
        net.setLink(1, 4, 1);
        return net;
    }

    public static void main(String[] args) {
        OspfNetwork net = pentagon(true);
        net.originateAll();
        int rounds = net.flood();

        check(rounds > 1 && rounds <= 40, "flooding takes several rounds: " + rounds);
        check(net.losses() > 0, "lossy links forced retransmissions: " + net.losses());
        check(net.transmissions() > 6 * 5, "flooding cost exceeds one transmission per router");

        String reference = net.databaseSignature(0);
        for (int r = 1; r < 6; r++) {
            check(net.databaseSignature(r).equals(reference), "router " + r + " has the same LSDB");
        }
        for (int r = 0; r < 6; r++) {
            for (int owner = 0; owner < 6; owner++) {
                check(net.stored(r, owner) != null, "router " + r + " knows LSA " + owner);
            }
        }

        int[][] oracle = net.allPairsReference();
        for (int r = 0; r < 6; r++) {
            int[] dist = net.distances(r);
            for (int d = 0; d < 6; d++) {
                check(dist[d] == oracle[r][d], "router " + r + " distance to " + d);
            }
        }

        int[][] distTable = new int[6][];
        int[][] nextTable = new int[6][];
        for (int r = 0; r < 6; r++) {
            distTable[r] = net.distances(r);
            nextTable[r] = net.nextHops(r);
        }
        for (int r = 0; r < 6; r++) {
            int[] dist = distTable[r];
            int[] next = nextTable[r];
            for (int d = 0; d < 6; d++) {
                if (d == r) continue;
                int hop = next[d];
                check(hop >= 0, "router " + r + " has a next hop for " + d);
                check(net.linkCost(r, hop) + distTable[hop][d] == dist[d],
                        "next hop of " + r + "->" + d + " is on a shortest path");
            }
            for (int d = 0; d < 6; d++) {
                if (d == r) continue;
                // Each router forwards using its own table, indexed by the destination.
                int cur = r;
                int hops = 0;
                int walked = 0;
                while (cur != d) {
                    int nxt = nextTable[cur][d];
                    check(nxt >= 0, "forwarding table entry exists at AS " + cur + " for " + d);
                    walked += net.linkCost(cur, nxt);
                    cur = nxt;
                    hops++;
                    check(hops <= 6, "packet walk cannot loop");
                }
                check(walked == dist[d], "cost of the walk equals dist " + r + "->" + d);
            }
        }
        check(nextTable[1][4] == 4 && nextTable[1][0] == 0, "AS 1 prefers the cheap direct link to 4");

        // A cost change is a new LSA with a higher sequence number and must reconverge everywhere.
        // Both endpoints re-originate, as they do in OSPF when an interface metric is changed.
        long before = net.transmissions();
        check(net.stored(0, 1).seq == 1, "first LSA of router 1 is sequence 1");
        net.setLink(1, 4, 20);
        net.originate(1);
        net.originate(4);
        int changeRounds = net.flood();
        check(net.transmissions() > before, "the update was flooded again");
        check(changeRounds <= 40, "update converges");
        check(net.stored(5, 1).seq == 2, "sequence number was bumped");
        int[][] updatedOracle = net.allPairsReference();
        check(oracle[0][4] == 5 && updatedOracle[0][4] == 13,
                "reference reflects the new cost: " + oracle[0][4] + " -> " + updatedOracle[0][4]);
        for (int r = 0; r < 6; r++) {
            int[] dist = net.distances(r);
            for (int d = 0; d < 6; d++) {
                check(dist[d] == updatedOracle[r][d], "router " + r + " reconverged to " + d);
            }
        }
        check(net.distances(1)[4] == 10 && net.nextHops(1)[4] == 2,
                "router 1 detours via router 2 now that the direct link costs 20");

        // Stale and duplicate LSAs must not be installed or re-flooded.
        long staleBefore = net.staleDrops();
        long dupBefore = net.duplicateDrops();
        Lsa stale = new Lsa(1, 1, new int[]{0}, new int[]{1});
        check(!net.install(2, stale, 0), "stale LSA rejected");
        check(net.staleDrops() == staleBefore + 1, "stale rejection counted");
        Lsa current = net.stored(3, 3);
        check(!net.install(2, current, 0), "equal-sequence LSA is a duplicate");
        check(net.duplicateDrops() == dupBefore + 1, "duplicate counted");
        check(net.stored(2, 3) == current, "duplicate did not replace the stored LSA");

        // Link failure: both endpoints re-originate; routes must detour and the LSDBs stay equal.
        net.removeLink(2, 3);
        net.originate(2);
        net.originate(3);
        net.flood();
        int[][] cut = net.allPairsReference();
        check(cut[0][3] == 15, "reference detours around the failed link: " + cut[0][3]);
        for (int r = 0; r < 6; r++) {
            int[] dist = net.distances(r);
            for (int d = 0; d < 6; d++) {
                check(dist[d] == cut[r][d], "router " + r + " after the failure, destination " + d);
            }
        }
        check(net.nextHops(0)[3] == 5, "router 0 now reaches 3 through 5");
        check(net.stored(0, 3).peer.length == 1 && net.stored(0, 3).peer[0] == 4,
                "router 3 advertises only its remaining link");
        check(net.databaseSignature(0).equals(net.databaseSignature(5)),
                "LSDBs still agree after the failure");

        // A fresh, loss-free domain with a different shape: invariants on another instance.
        OspfNetwork full = new OspfNetwork(7, false);
        for (int a = 0; a < 7; a++) {
            for (int b = a + 1; b < 7; b++) {
                full.setLink(a, b, 1 + (a * 7 + b * 13) % 9);
            }
        }
        full.removeLink(2, 5);
        full.originateAll();
        full.flood();
        check(full.losses() == 0, "loss-free domain performs no retransmissions");
        int[][] dense = full.allPairsReference();
        for (int r = 0; r < 7; r++) {
            int[] dist = full.distances(r);
            for (int d = 0; d < 7; d++) {
                check(dist[d] == dense[r][d], "dense domain distance " + r + "->" + d);
            }
        }
        check(full.databaseSignature(0).equals(full.databaseSignature(6)), "dense LSDBs agree");
        long denseTransmissions = full.transmissions();
        check(full.flood() == 0 && full.transmissions() == denseTransmissions,
                "a converged domain needs no further flooding");

        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
