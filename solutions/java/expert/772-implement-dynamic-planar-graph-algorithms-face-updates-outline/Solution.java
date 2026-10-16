/** Task 772 - Implement dynamic planar graph algorithms (face updates) outline. (Tier: Expert, CTCI Ch 4 - Trees and Graphs)
 *
 * A dynamic planar graph keeps an explicit *embedding*: every directed edge (half-edge) points at the
 * next half-edge on the face it borders, which is exactly the rotation system a DCEL stores. The two
 * structural updates are
 *
 *   insert (u, v)   splice a new twin pair into the face structure: if the endpoints already lie on a
 *                   common face the face is split in two (F grows by one), otherwise the edge is a
 *                   bridge that merges two face walks (F unchanged, C drops by one).
 *   remove (u, v)   the inverse splice; the two faces that met at the edge are merged again.
 *
 * Both cases are served by the same four-pointer splice (a = an arrival at u, t = an arrival at v):
 * a.next = e1, e1.next = t.next, t.next = e2, e2.next = old a.next. Faces are then recovered by
 * walking the next pointers.
 *
 * Invariants verified after *every* mutation: the half-edge structure is consistent (next/prev/twin
 * are mutual inverses and the destination agrees with the next origin); every live half-edge belongs
 * to exactly one face walk; sum of face degrees == 2E; the cycle count equals E - V + 2C; and Euler's
 * formula V - E + F = 1 + C holds with F = cycles - C + 1 and C from a breadth-first search. The
 * checks also confirm the derived update rule: inserting an edge inside a face changes F by +1 and
 * inserting a bridge changes it by 0, the exact opposite of deleting it.
 *
 * Simplified (this is the outline variant): faces are re-derived by a full O(E) traversal after each
 * update instead of being spliced locally in O(1); a new vertex enters only as a pendant of an
 * existing vertex, isolated vertices are pruned on deletion, and no vertex splits or edge flips are
 * provided. The embedding, its face walks and the Euler invariants are real, not simulated.
 */
import java.util.*;

/** Face-tracking embedding of a planar graph on half-edges (the DCEL core of the outline). */
class PlanarGraph {
    private int[] origin;
    private int[] next;
    private int[] prev;
    private int[] twin;
    private boolean[] alive;
    private int allocated;
    private final Map<Long, Integer> edgeToHalf = new HashMap<>();
    final Set<Integer> vertices = new TreeSet<>();

    PlanarGraph() {
        origin = new int[16];
        next = new int[16];
        prev = new int[16];
        twin = new int[16];
        alive = new boolean[16];
    }

    static long edgeKey(int u, int v) {
        return u < v ? ((long) u << 32) | (v & 0xFFFFFFFFL) : ((long) v << 32) | (u & 0xFFFFFFFFL);
    }

    private void grow() {
        if (allocated < origin.length) return;
        int size = origin.length * 2;
        origin = Arrays.copyOf(origin, size);
        next = Arrays.copyOf(next, size);
        prev = Arrays.copyOf(prev, size);
        twin = Arrays.copyOf(twin, size);
        alive = Arrays.copyOf(alive, size);
    }

    private int allocate(int u) {
        grow();
        int id = allocated++;
        origin[id] = u;
        alive[id] = true;
        return id;
    }

    boolean hasEdge(int u, int v) { return edgeToHalf.containsKey(edgeKey(u, v)); }

    int edgeCount() { return edgeToHalf.size(); }

    int vertexCount() { return vertices.size(); }

    /** Any live half-edge leaving u; used to reach an arrival at u through prev. */
    private int anyOutgoing(int u) {
        for (int h = 0; h < allocated; h++) {
            if (alive[h] && origin[h] == u) return h;
        }
        throw new IllegalStateException("vertex " + u + " has no incident edge");
    }

    void insertEdge(int u, int v) {
        if (u == v) throw new IllegalArgumentException("self loops are not supported: " + u);
        if (hasEdge(u, v)) throw new IllegalArgumentException("edge already present: " + u + "-" + v);
        boolean freshU = !vertices.contains(u);
        boolean freshV = !vertices.contains(v);
        if (freshU && freshV && edgeCount() > 0) {
            throw new IllegalArgumentException("a second component may only be grown one vertex at a time");
        }
        int e1, e2;
        if (edgeCount() == 0) {
            e1 = allocate(u);
            e2 = allocate(v);
            next[e1] = e2;
            next[e2] = e1;
            prev[e1] = e2;
            prev[e2] = e1;
        } else if (freshU || freshV) {
            // pendant vertex: it is placed inside the face the host vertex borders, so F never changes
            int host = freshU ? v : u;
            int t = prev[anyOutgoing(host)];              // an arrival at host
            int z = next[t];                              // the walk continues after host
            e1 = allocate(u);
            e2 = allocate(v);
            int outgoing = freshU ? e2 : e1;              // host -> newcomer, then back
            int incoming = freshU ? e1 : e2;
            next[t] = outgoing;
            prev[outgoing] = t;
            next[outgoing] = incoming;
            prev[incoming] = outgoing;
            next[incoming] = z;
            prev[z] = incoming;
        } else {
            int[] splice = findSplice(u, v);
            int a = splice[0], t = splice[1];
            e1 = allocate(u);
            e2 = allocate(v);
            int b = next[a];
            int z = next[t];
            next[a] = e1;
            prev[e1] = a;
            next[e1] = z;
            prev[z] = e1;
            next[t] = e2;
            prev[e2] = t;
            next[e2] = b;
            prev[b] = e2;
        }
        twin[e1] = e2;
        twin[e2] = e1;
        edgeToHalf.put(edgeKey(u, v), e1);
        vertices.add(u);
        vertices.add(v);
    }

    /**
     * Picks one arrival at u and one arrival at v to splice the new edge between. Endpoints in the
     * same component must share a face (a chord splits it); endpoints in different components give a
     * bridge that merges the two face walks.
     */
    private int[] findSplice(int u, int v) {
        boolean sameComponent = componentOf(u) == componentOf(v);
        List<int[]> cycles = cycles();
        int arrivalU = -1, arrivalV = -1;
        for (int[] cycle : cycles) {
            int candidateU = -1, candidateV = -1;
            for (int h : cycle) {
                if (origin[h] == u) candidateU = prev[h];
                if (origin[h] == v) candidateV = prev[h];
            }
            if (candidateU >= 0 && candidateV >= 0) {
                arrivalU = candidateU;
                arrivalV = candidateV;
                break;
            }
        }
        if (sameComponent) {
            if (arrivalU < 0 || arrivalV < 0) {
                throw new IllegalArgumentException("no common face for " + u + " and " + v);
            }
            return new int[]{arrivalU, arrivalV};
        }
        // different components: merge the face walks through the new bridge
        return new int[]{prev[anyOutgoing(u)], prev[anyOutgoing(v)]};
    }

    void removeEdge(int u, int v) {
        Integer id = edgeToHalf.remove(edgeKey(u, v));
        if (id == null) throw new IllegalArgumentException("edge not present: " + u + "-" + v);
        int e1 = id;
        int e2 = twin[e1];
        if (prev[e1] == e2) {
            // pendant edge whose u end was the newcomer: t -> e2 -> e1 -> z becomes t -> z
            int t = prev[e2];
            int z = next[e1];
            next[t] = z;
            prev[z] = t;
        } else if (prev[e2] == e1) {
            int t = prev[e1];
            int z = next[e2];
            next[t] = z;
            prev[z] = t;
        } else {
            // chord or bridge: undo a -> e1 -> next[t] and t -> e2 -> a.next
            int a = prev[e1];
            int t = prev[e2];
            int b = next[e2];
            int z = next[e1];
            next[a] = b;
            prev[b] = a;
            next[t] = z;
            prev[z] = t;
        }
        alive[e1] = false;
        alive[e2] = false;
        // isolated vertices are pruned so that V counts only vertices that carry an edge
        for (int vertex : new int[]{u, v}) {
            if (!hasIncident(vertex)) vertices.remove(vertex);
        }
    }

    private boolean hasIncident(int vertex) {
        for (int h = 0; h < allocated; h++) {
            if (alive[h] && origin[h] == vertex) return true;
        }
        return false;
    }

    int degree(int vertex) {
        int degree = 0;
        for (int h = 0; h < allocated; h++) {
            if (alive[h] && origin[h] == vertex) degree++;
        }
        return degree;
    }

    /** Face walks: every live half-edge is followed through next until it returns to the start. */
    List<int[]> cycles() {
        boolean[] seen = new boolean[allocated];
        List<int[]> result = new ArrayList<>();
        for (int h = 0; h < allocated; h++) {
            if (!alive[h] || seen[h]) continue;
            List<Integer> walk = new ArrayList<>();
            int cursor = h;
            while (!seen[cursor]) {
                seen[cursor] = true;
                walk.add(cursor);
                cursor = next[cursor];
            }
            int[] cycle = new int[walk.size()];
            for (int i = 0; i < walk.size(); i++) cycle[i] = walk.get(i);
            result.add(cycle);
        }
        return result;
    }

    /** The vertices of each face, in walk order. */
    List<List<Integer>> faces() {
        List<List<Integer>> faces = new ArrayList<>();
        for (int[] cycle : cycles()) {
            List<Integer> face = new ArrayList<>();
            for (int h : cycle) face.add(origin[h]);
            faces.add(face);
        }
        return faces;
    }

    int componentOf(int start) {
        Set<Integer> seen = new HashSet<>();
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(start);
        seen.add(start);
        while (!stack.isEmpty()) {
            int u = stack.pop();
            for (int h = 0; h < allocated; h++) {
                if (!alive[h] || origin[h] != u) continue;
                int w = origin[twin[h]];
                if (seen.add(w)) stack.push(w);
            }
        }
        int lowest = Integer.MAX_VALUE;
        for (int vertex : seen) lowest = Math.min(lowest, vertex);
        return lowest;
    }

    int componentCount() {
        Set<Integer> labels = new HashSet<>();
        for (int vertex : vertices) labels.add(componentOf(vertex));
        return labels.size();
    }

    int faceCount() { return edgeCount() == 0 ? 1 : cycles().size() - componentCount() + 1; }

    List<int[]> edges() {
        List<int[]> edges = new ArrayList<>();
        for (Map.Entry<Long, Integer> entry : edgeToHalf.entrySet()) {
            int u = (int) (entry.getKey() >> 32);
            int v = (int) (long) entry.getKey();
            edges.add(new int[]{u, v});
        }
        edges.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
        return edges;
    }

    /** Returns null when every structural and Euler invariant holds, else a description. */
    String verify() {
        for (int h = 0; h < allocated; h++) {
            if (!alive[h]) continue;
            if (twin[twin[h]] != h) return "twin not involutive at " + h;
            if (next[prev[h]] != h) return "prev/next mismatch at " + h;
            if (origin[twin[h]] == origin[h]) return "self loop half-edge " + h;
            if (!alive[twin[h]]) return "dead twin at " + h;
            if (origin[next[h]] != origin[twin[h]]) {
                return "destination of " + h + " does not continue the walk";
            }
        }
        int liveUndirected = 0;
        for (int h = 0; h < allocated; h++) if (alive[h]) liveUndirected++;
        if (liveUndirected != 2 * edgeCount()) return "half-edge count " + liveUndirected + " != 2E";
        if (vertices.size() == 0 && edgeCount() > 0) return "edges without vertices";
        for (int vertex : vertices) {
            if (degree(vertex) == 0) return "isolated vertex " + vertex + " was not pruned";
        }
        List<int[]> cycles = cycles();
        int covered = 0;
        for (int[] cycle : cycles) covered += cycle.length;
        if (covered != 2 * edgeCount()) return "faces cover " + covered + " half-edges, expected 2E";
        int c = componentCount();
        if (edgeCount() > 0 && cycles.size() != edgeCount() - vertexCount() + 2 * c) {
            return "cycle count " + cycles.size() + " != E - V + 2C = "
                    + (edgeCount() - vertexCount() + 2 * c);
        }
        int f = faceCount();
        if (vertexCount() - edgeCount() + f != 1 + c) {
            return "Euler broken: V-E+F = " + (vertexCount() - edgeCount() + f) + " but 1+C = " + (1 + c);
        }
        return null;
    }
}

public class Solution {
    private static long state = 0x9E3779B97F4A7C15L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static List<Integer> sortedDegrees(PlanarGraph graph) {
        List<Integer> degrees = new ArrayList<>();
        for (List<Integer> face : graph.faces()) degrees.add(face.size());
        Collections.sort(degrees);
        return degrees;
    }

    public static void main(String[] args) {
        // --- degenerate graph: nothing at all ---------------------------------
        PlanarGraph empty = new PlanarGraph();
        check(empty.edgeCount() == 0 && empty.vertexCount() == 0, "empty graph");
        check(empty.faceCount() == 1, "the empty graph has the single outer face");
        check(empty.verify() == null, "empty graph invariants: " + empty.verify());

        // --- single edge, then a triangle -------------------------------------
        PlanarGraph triangle = new PlanarGraph();
        triangle.insertEdge(0, 1);
        check(triangle.faceCount() == 1, "one edge has a single face");
        check(triangle.verify() == null, "single edge invariants: " + triangle.verify());
        int beforeFaces = triangle.faceCount();
        triangle.insertEdge(1, 2);
        check(triangle.faceCount() == beforeFaces, "a pendant vertex does not split a face");
        check(triangle.verify() == null, "path invariants: " + triangle.verify());
        triangle.insertEdge(2, 0);
        check(triangle.faceCount() == 2, "closing the triangle creates the inner face");
        check(triangle.verify() == null, "triangle invariants: " + triangle.verify());
        check(sortedDegrees(triangle).equals(List.of(3, 3)), "triangle faces have degree 3 and 3");
        check(triangle.componentCount() == 1, "triangle is connected");

        // --- rejected updates (negative checks) -------------------------------
        boolean rejected = false;
        try {
            triangle.insertEdge(1, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "self loops must be rejected");
        rejected = false;
        try {
            triangle.insertEdge(0, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "duplicate edges must be rejected");
        rejected = false;
        try {
            triangle.removeEdge(1, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "removing a missing edge must be rejected");

        // --- square plus a diagonal: the classic face update -------------------
        PlanarGraph square = new PlanarGraph();
        square.insertEdge(0, 1);
        square.insertEdge(1, 2);
        square.insertEdge(2, 3);
        square.insertEdge(3, 0);
        check(square.faceCount() == 2, "the square has inner and outer faces");
        square.insertEdge(0, 2);
        check(square.faceCount() == 3, "a diagonal splits the inner face");
        check(square.verify() == null, "square with diagonal invariants: " + square.verify());
        check(sortedDegrees(square).equals(List.of(3, 3, 4)), "faces of the diagonalised square");
        int facesBefore = square.faceCount();
        square.removeEdge(0, 2);
        check(square.faceCount() == facesBefore - 1, "removing a chord merges the two faces: before="
                + facesBefore + " after=" + square.faceCount() + " cycles=" + square.cycles().size()
                + " verify=" + square.verify());
        check(square.verify() == null, "square after removing the diagonal: " + square.verify());
        check(sortedDegrees(square).equals(List.of(4, 4)), "each square face has degree 4");

        // --- the wheel W4: a bridge-free planar graph --------------------------
        PlanarGraph wheel = new PlanarGraph();
        wheel.insertEdge(0, 1);
        wheel.insertEdge(1, 2);
        wheel.insertEdge(2, 3);
        wheel.insertEdge(3, 4);
        wheel.insertEdge(4, 1);
        wheel.insertEdge(0, 2);
        wheel.insertEdge(0, 3);
        wheel.insertEdge(0, 4);
        check(wheel.verify() == null, "wheel invariants: " + wheel.verify());
        check(wheel.faceCount() == 5, "the wheel has four triangles and the outer face, got " + wheel.faceCount());
        check(sortedDegrees(wheel).equals(List.of(3, 3, 3, 3, 4)), "wheel face degrees");
        check(wheel.vertexCount() == 5 && wheel.edgeCount() == 8 && wheel.componentCount() == 1,
                "wheel counts V, E and C");

        // --- a bridge between two components changes F by zero -----------------
        PlanarGraph split = new PlanarGraph();
        split.insertEdge(10, 11);
        split.insertEdge(11, 12);
        split.insertEdge(12, 10);
        check(split.faceCount() == 2, "the first triangle has two faces");
        split.insertEdge(12, 30);
        check(split.faceCount() == 2, "a bridge adds no face");
        split.insertEdge(30, 31);
        split.insertEdge(31, 32);
        split.insertEdge(32, 30);
        check(split.faceCount() == 3, "the second triangle adds its inner face");
        check(split.componentCount() == 1, "the bridged graph is connected");
        check(split.verify() == null, "bridged graph invariants: " + split.verify());
        int bridgedFaces = split.faceCount();
        int bridgedCycles = split.cycles().size();
        split.removeEdge(12, 30);
        check(split.componentCount() == 2, "removing the bridge splits the graph");
        check(split.faceCount() == bridgedFaces, "removing a bridge leaves F unchanged");
        check(split.cycles().size() == bridgedCycles + 1,
                "the two pieces contribute one extra outer face walk");
        check(split.verify() == null, "split graph invariants: " + split.verify());
        split.insertEdge(12, 30);
        check(split.faceCount() == bridgedFaces, "re-inserting the bridge restores the same F");
        check(split.cycles().size() == bridgedCycles, "the face walks merge again");
        check(split.verify() == null, "re-bridged graph invariants: " + split.verify());

        // --- incremental updates on a triangulated fan -------------------------
        // vertices 0..n sit on an outer cycle; every vertex is joined to 0 and to its successor,
        // which keeps the graph planar and the updates legal.
        PlanarGraph fan = new PlanarGraph();
        fan.insertEdge(0, 1);
        fan.insertEdge(0, 2);
        int previousF = fan.faceCount();
        for (int v = 2; v < 12; v++) {
            fan.insertEdge(v, v + 1);
            check(fan.faceCount() == previousF, "growing the fan has no face change yet");
            check(fan.verify() == null, "fan invariants while growing: " + fan.verify());
            fan.insertEdge(0, v + 1);
            check(fan.faceCount() == previousF + 1, "each closing edge splits one more face");
            check(fan.verify() == null, "fan invariants after closing: " + fan.verify());
            previousF = fan.faceCount();
        }
        check(fan.componentCount() == 1, "the fan is connected");
        check(fan.vertexCount() == 13, "fan has 13 vertices, got " + fan.vertexCount());
        check(fan.edgeCount() == 22, "fan has 22 edges, got " + fan.edgeCount());
        check(fan.faceCount() == 11, "fan has 11 faces, got " + fan.faceCount());
        check(fan.faceCount() == 1 + fan.componentCount() - fan.vertexCount() + fan.edgeCount(),
                "Euler for the fan");

        // --- randomised workload with the invariants checked after each step ---
        PlanarGraph random = new PlanarGraph();
        random.insertEdge(0, 1);
        random.insertEdge(0, 2);
        random.insertEdge(2, 0 + 3);
        random.insertEdge(0, 3);
        List<Integer> known = new ArrayList<>(List.of(0, 1, 2, 3));
        int mutations = 0;
        int deletions = 0;
        for (int step = 0; step < 200; step++) {
            int existing = known.get(nextInt(known.size()));
            int label = nextInt(3) == 0 ? 1000 + step : known.get(nextInt(known.size()));
            if (label == existing || random.hasEdge(existing, label)) continue;
            try {
                random.insertEdge(existing, label);
                mutations++;
                if (label >= 1000) known.add(label);
            } catch (IllegalArgumentException legalRejection) {
                // no common face for the two endpoints: a legal rejection, not a bug
            }
            check(random.verify() == null, "random invariants at step " + step + ": " + random.verify());
            if (step % 4 == 0 && random.edgeCount() > 6) {
                List<int[]> edges = random.edges();
                int[] victim = edges.get(nextInt(edges.size()));
                random.removeEdge(victim[0], victim[1]);
                deletions++;
                check(random.verify() == null, "random invariants after deletion at step " + step + ": "
                        + random.verify());
            }
        }
        check(mutations > 40, "the random workload must actually insert, got " + mutations);
        check(deletions > 10, "the random workload must actually delete, got " + deletions);
        check(random.faceCount() >= 1, "the random graph keeps at least the outer face");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
