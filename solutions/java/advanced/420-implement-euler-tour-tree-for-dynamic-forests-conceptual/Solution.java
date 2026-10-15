/** Task 420 - Implement Euler tour tree for dynamic forests (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * An Euler tour tree stores one Euler circuit of each tree as a sequence of DIRECTED EDGE
 * events (u -> v): travelling the circuit enters u and leaves towards v, so the sequence
 * chains (each event's target is the next event's source, cyclically) and every directed
 * edge of the tree appears exactly once. A real ETT keeps these sequences in balanced
 * search trees and links/cuts by splitting/merging them in O(log n); this conceptual
 * implementation uses plain lists, so each operation costs O(size) — the operations are
 * identical, only the balance is simulated away (documented deviation).
 *
 *   * connect(u, v): rotate u's tour to start leaving u and v's tour to start leaving v,
 *     then splice:  tour(u) + [u->v] + tour(v) + [v->u]  — a valid circuit of the merged tree.
 *   * disconnect(u, v) (edge exists): rotate to start with [u->v], find the matching
 *     [v->u]; the events strictly between them form the v-side circuit, the rest (after
 *     [v->u]) forms the u-side circuit.
 *   * reroot(v): rotate v's circuit so it starts with an event leaving v.
 *   * connected / componentVertices: answered from the owner map (a list simulation of
 *     the tree lookup a real ETT answers via the roots of its balanced trees).
 */
import java.util.*;

class EulerTourForest {
    final int n;
    final Map<Integer, List<int[]>> tours = new HashMap<>();
    final int[] owner;
    int nextId;

    EulerTourForest(int n) {
        this.n = n;
        owner = new int[n + 1];
        for (int v = 1; v <= n; v++) {
            owner[v] = v;
            tours.put(v, new ArrayList<>());
        }
        nextId = n + 1;
    }

    List<int[]> tourOf(int x) {
        return tours.get(owner[x]);
    }

    /** Rotate so the sequence starts with an event leaving x (no-op if isolated). */
    static List<int[]> rotate(List<int[]> tour, int x) {
        for (int i = 0; i < tour.size(); i++) {
            if (tour.get(i)[0] == x) {
                List<int[]> out = new ArrayList<>(tour.subList(i, tour.size()));
                out.addAll(tour.subList(0, i));
                return out;
            }
        }
        return new ArrayList<>(tour);
    }

    static Set<Integer> endpoints(List<int[]> tour) {
        Set<Integer> verts = new TreeSet<>();
        for (int[] e : tour) {
            verts.add(e[0]);
            verts.add(e[1]);
        }
        return verts;
    }

    void reassign(int id, Set<Integer> members) {
        for (int vert : members) owner[vert] = id;
    }

    /** Link u and v (must be in different components); returns true when linked. */
    boolean connect(int u, int v) {
        if (owner[u] == owner[v]) return false;
        List<int[]> merged = new ArrayList<>(rotate(tourOf(u), u));
        merged.add(new int[]{u, v});
        merged.addAll(rotate(tourOf(v), v));
        merged.add(new int[]{v, u});
        tours.remove(owner[u]);
        tours.remove(owner[v]);
        int id = nextId++;
        tours.put(id, merged);
        Set<Integer> members = endpoints(merged);
        members.add(u);
        members.add(v);
        reassign(id, members);
        return true;
    }

    /** Cut the edge {u, v} when it exists; returns true when cut. */
    boolean disconnect(int u, int v) {
        if (owner[u] != owner[v]) return false;
        List<int[]> tour = tourOf(u);
        int forward = -1, reverse = -1;
        for (int i = 0; i < tour.size(); i++) {
            int[] e = tour.get(i);
            if (e[0] == u && e[1] == v) forward = i;
            if (e[0] == v && e[1] == u) reverse = i;
        }
        if (forward < 0 || reverse < 0) return false;
        List<int[]> rot = new ArrayList<>(tour.subList(forward, tour.size()));
        rot.addAll(tour.subList(0, forward));               // starts with u -> v
        int q = -1;
        for (int i = 0; i < rot.size(); i++) {
            int[] e = rot.get(i);
            if (e[0] == v && e[1] == u) { q = i; break; }
        }
        List<int[]> sideV = new ArrayList<>(rot.subList(1, q));           // closed walk at v
        List<int[]> sideU = new ArrayList<>(rot.subList(q + 1, rot.size())); // closed walk at u
        tours.remove(owner[u]);
        int idU = nextId++;
        int idV = nextId++;
        tours.put(idU, sideU);
        tours.put(idV, sideV);
        Set<Integer> mu = endpoints(sideU);
        mu.add(u);
        reassign(idU, mu);
        Set<Integer> mv = endpoints(sideV);
        mv.add(v);
        reassign(idV, mv);
        return true;
    }

    void reroot(int x) {
        int id = owner[x];
        tours.put(id, rotate(tours.get(id), x));
    }

    boolean connected(int u, int v) {
        return owner[u] == owner[v];
    }

    int[] componentVertices(int x) {
        Set<Integer> members = endpoints(tours.get(owner[x]));
        members.add(x);
        int[] out = new int[members.size()];
        int k = 0;
        for (int vert : members) out[k++] = vert;
        return out;
    }

    int componentSize(int x) {
        return componentVertices(x).length;
    }
}

/** Adjacency oracle with BFS components. */
class OracleForest {
    final List<Set<Integer>> adj;

    OracleForest(int n) {
        adj = new ArrayList<>();
        for (int i = 0; i <= n; i++) adj.add(new TreeSet<>());
    }

    void link(int u, int v) {
        adj.get(u).add(v);
        adj.get(v).add(u);
    }

    void cut(int u, int v) {
        adj.get(u).remove(v);
        adj.get(v).remove(u);
    }

    boolean hasEdge(int u, int v) {
        return adj.get(u).contains(v);
    }

    Set<Integer> component(int x) {
        Set<Integer> seen = new TreeSet<>();
        seen.add(x);
        Deque<Integer> queue = new ArrayDeque<>();
        queue.add(x);
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            for (int nxt : adj.get(cur)) {
                if (!seen.contains(nxt)) {
                    seen.add(nxt);
                    queue.add(nxt);
                }
            }
        }
        return seen;
    }

    boolean connected(int u, int v) {
        return component(u).contains(v);
    }
}

public class Solution {
    private static int seed;

    private static int nextRand() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    /** Every component's tour must be a valid Euler circuit of the oracle tree. */
    static void validate(EulerTourForest forest, OracleForest oracle, String label) {
        boolean[] seen = new boolean[forest.n + 1];
        for (int start = 1; start <= forest.n; start++) {
            if (seen[start]) continue;
            Set<Integer> members = oracle.component(start);
            for (int v : members) seen[v] = true;
            List<int[]> tour = forest.tourOf(start);
            if (members.size() == 1) {
                if (!tour.isEmpty()) throw new AssertionError("FAILED: singleton tour " + label + " " + start);
                continue;
            }
            if (tour.size() != 2 * (members.size() - 1)) throw new AssertionError("FAILED: tour size " + label);
            for (int i = 0; i < tour.size(); i++) {
                if (tour.get(i)[1] != tour.get((i + 1) % tour.size())[0]) {
                    throw new AssertionError("FAILED: chaining " + label + " " + i);
                }
            }
            Set<String> seenEvents = new HashSet<>();
            Set<Integer> covered = new TreeSet<>();
            for (int[] e : tour) {
                String key = e[0] + "->" + e[1];
                if (!seenEvents.add(key)) throw new AssertionError("FAILED: duplicate event " + label);
                if (!oracle.hasEdge(e[0], e[1])) throw new AssertionError("FAILED: bogus edge " + label);
                covered.add(e[0]);
                covered.add(e[1]);
            }
            if (!covered.equals(members)) throw new AssertionError("FAILED: coverage " + label + " " + start);
            int[] cv = forest.componentVertices(start);
            if (cv.length != members.size()) throw new AssertionError("FAILED: vertices " + label);
            for (int v : cv) {
                if (!members.contains(v)) throw new AssertionError("FAILED: stray vertex " + label);
            }
            if (forest.componentSize(start) != members.size()) throw new AssertionError("FAILED: size " + label);
        }
    }

    public static void main(String[] args) {
        // hand trace on three vertices
        EulerTourForest forest = new EulerTourForest(3);
        check(forest.connected(1, 1) && !forest.connected(1, 2), "isolated");
        check(forest.componentVertices(1).length == 1 && forest.componentVertices(1)[0] == 1, "single");
        check(forest.connect(1, 2), "connect 1 2");
        check(Arrays.equals(forest.tourOf(1).get(0), new int[]{1, 2})
                && Arrays.equals(forest.tourOf(1).get(1), new int[]{2, 1}), "pair tour");
        check(forest.connect(2, 3), "connect 2 3");
        // tour of the walk 1 -> 2 -> 3 -> 2 -> 1, rotated to start leaving 2
        List<int[]> joined = forest.tourOf(2);
        int[][] want = {{2, 1}, {1, 2}, {2, 3}, {3, 2}};
        check(joined.size() == 4, "joined size");
        for (int i = 0; i < 4; i++) {
            check(joined.get(i)[0] == want[i][0] && joined.get(i)[1] == want[i][1], "joined " + i);
        }
        check(forest.connected(1, 3) && forest.componentSize(1) == 3, "chain");
        int[] all = forest.componentVertices(1);
        check(all.length == 3 && all[0] == 1 && all[1] == 2 && all[2] == 3, "chain vertices");

        // reroot rotates the circuit so that it starts leaving the new root
        forest.reroot(3);
        List<int[]> rerooted = forest.tourOf(3);
        int[][] wantRerooted = {{3, 2}, {2, 1}, {1, 2}, {2, 3}};
        check(rerooted.size() == 4, "reroot size");
        for (int i = 0; i < 4; i++) {
            check(rerooted.get(i)[0] == wantRerooted[i][0] && rerooted.get(i)[1] == wantRerooted[i][1],
                    "reroot " + i);
        }

        // cut edge 2-3: the v-side is the singleton {3}, the rest keeps {1, 2}
        check(forest.disconnect(2, 3), "disconnect 2 3");
        check(!forest.connected(1, 3), "1 3 split");
        check(forest.connected(1, 2), "1 2 kept");
        check(forest.tourOf(3).isEmpty(), "3 singleton");
        List<int[]> side = forest.tourOf(1);
        check(side.size() == 2, "side size");
        for (int i = 0; i < 2; i++) {
            int[] e = side.get(i);
            check((e[0] == 1 && e[1] == 2) || (e[0] == 2 && e[1] == 1), "side event " + i);
        }
        check(forest.componentVertices(3).length == 1, "3 alone");

        // cutting again without the edge is a no-op
        check(!forest.disconnect(2, 3), "disconnect gone");
        check(forest.connected(1, 2), "1 2 still");

        // a non-adjacent pair cannot be disconnected even when connected
        forest.connect(2, 3);
        check(forest.connected(1, 3), "reconnected");
        check(!forest.disconnect(1, 3), "no 1 3 edge");
        check(forest.connected(1, 3), "still connected");
        check(forest.disconnect(1, 2), "cut 1 2");
        check(!forest.connected(1, 2) && forest.connected(2, 3), "final shape");

        // scripted operations against the union-find + tree oracle
        int n = 25;
        seed = 4200;
        forest = new EulerTourForest(n);
        OracleForest oracle = new OracleForest(n);
        for (int step = 0; step < 320; step++) {
            int kind = nextRand() % 4;
            int u = 1 + nextRand() % n;
            int v = 1 + nextRand() % n;
            if (kind == 0) {
                if (u != v && !oracle.connected(u, v)) {
                    check(forest.connect(u, v), "script connect " + step);
                    oracle.link(u, v);
                    check(forest.connected(u, v), "script connect check " + step);
                }
            } else if (kind == 1) {
                if (u != v && oracle.hasEdge(u, v)) {
                    check(forest.disconnect(u, v), "script cut " + step);
                    oracle.cut(u, v);
                    check(!forest.connected(u, v), "script cut check " + step);
                }
            } else if (kind == 2) {
                check(forest.connected(u, v) == oracle.connected(u, v), "script conn " + step);
                int[] cv = forest.componentVertices(u);
                Set<Integer> members = oracle.component(u);
                check(cv.length == members.size(), "script vertices " + step);
                for (int w : cv) check(members.contains(w), "script member " + step);
                check(forest.componentSize(u) == members.size(), "script size " + step);
            } else {
                forest.reroot(u);
                check(forest.connected(u, v) == oracle.connected(u, v), "script reroot " + step);
            }
            // structural validity after every single operation
            validate(forest, oracle, "step " + step);
            // incremental spot checks on fresh random pairs
            for (int q = 0; q < 3; q++) {
                int a = 1 + nextRand() % n;
                int b = 1 + nextRand() % n;
                check(forest.connected(a, b) == oracle.connected(a, b), "spot " + step);
            }
        }

        // final full audit: every pair, every component
        for (int u = 1; u <= n; u++) {
            for (int v = u + 1; v <= n; v++) {
                check(forest.connected(u, v) == oracle.connected(u, v), "final conn " + u + " " + v);
            }
        }
        validate(forest, oracle, "final");
        // final shape: the deterministic script leaves one giant component plus a few
        // cut-off singletons; assert the giant exists (exact matching is oracle-checked)
        int largest = 0;
        for (int v = 1; v <= n; v++) largest = Math.max(largest, oracle.component(v).size());
        check(largest >= n - 5, "giant component");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
