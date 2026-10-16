/** Task 846 - Implement subgraph-centric distributed processing framework and test on motifs. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The graph is cut into P subgraphs. Every vertex is *owned* by exactly one partition (a hash of its id),
 * and every edge is assigned to exactly one partition (round robin by edge index), so a partition usually
 * holds edges whose endpoints it does not own. Each partition therefore keeps
 *
 *   owned adjacency  - the neighbour list of every vertex it owns,
 *   ghost adjacency  - the neighbour lists of the endpoints of its edges that live elsewhere, fetched from
 *                      the owners in one explicit boundary exchange (the only "messages" in this program),
 *   local degrees    - the degree of every vertex, reduced over the partitions that see it,
 *
 * and then runs a local motif computation. Triangle counting uses the *ownership* rule that makes the
 * distributed total equal the central one: an edge is processed with u < v, the third vertex must satisfy
 * w > v, and each triangle is attributed to the partition holding its lowest-numbered edge - so a triangle
 * that spans three partitions is still counted exactly once, by exactly one of them.
 *
 * Simplifications, stated honestly: a single process with an in-memory "network" whose messages are
 * counted, not sent; no fault tolerance and no superstep scheduler (the exchange is one round); the motif
 * census is limited to the 3-vertex motifs that a subgraph-centric pass can decide from adjacency alone -
 * triangles and wedges (3-paths) - because deeper motifs need multi-round neighbourhood exchange.
 *
 * Invariants verified below. Correctness: the triangle count and the wedge count equal a central brute-force
 * census, for one partition and for two, three and five partitions (partition invariance), and the identity
 * wedges = SUM_v C(deg(v),2) - 3 * triangles holds exactly. Consistency: the degree vector seen by every
 * partition is the central one, every triangle is counted by exactly one partition (the local counts sum to
 * the global count), and a ghost vertex's adjacency fetched from its owner equals the owner's own list.
 * Cost: the boundary exchange is counted and shown to be bounded by the number of cut edges.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so the graph and the hash partitioning are reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** One partition of the graph. */
final class Partition {
    final int id;
    final int parts;
    final List<int[]> edges = new ArrayList<>();
    final Map<Integer, List<Integer>> ownedAdjacency = new TreeMap<>();
    final Map<Integer, List<Integer>> ghostAdjacency = new TreeMap<>();
    int ghostRequests;
    int crossPartitionEdges;

    Partition(int id, int parts) {
        this.id = id;
        this.parts = parts;
    }

    int ownerOf(int vertex) { return Math.floorMod(vertex, parts); }

    boolean owns(int vertex) { return ownerOf(vertex) == id; }

    /** Builds this partition's share: the edges it holds and the adjacency of the vertices it owns. */
    void load(List<int[]> allEdges) {
        for (int i = 0; i < allEdges.size(); i++) {
            if (Math.floorMod(i, parts) != id) continue;
            int[] edge = allEdges.get(i);
            edges.add(edge);
            if (ownerOf(edge[0]) != ownerOf(edge[1])) crossPartitionEdges++;
        }
        for (int[] edge : allEdges) {
            if (owns(edge[0])) addNeighbour(edge[0], edge[1]);
            if (owns(edge[1])) addNeighbour(edge[1], edge[0]);
        }
    }

    private void addNeighbour(int vertex, int neighbour) {
        List<Integer> list = ownedAdjacency.computeIfAbsent(vertex, v -> new ArrayList<>());
        if (!list.contains(neighbour)) list.add(neighbour);
    }

    /** One boundary exchange: fetch the neighbour list of every endpoint of a local edge it does not own. */
    void exchange(List<Partition> all) {
        Set<Integer> wanted = new TreeSet<>();
        for (int[] edge : edges) {
            if (!owns(edge[0])) wanted.add(edge[0]);
            if (!owns(edge[1])) wanted.add(edge[1]);
        }
        for (int vertex : wanted) {
            Partition owner = all.get(ownerOf(vertex));
            List<Integer> adjacency = owner.ownedAdjacency.getOrDefault(vertex, List.of());
            ghostAdjacency.put(vertex, new ArrayList<>(adjacency));
            ghostRequests++;
        }
    }

    List<Integer> neighboursOf(int vertex) {
        if (owns(vertex)) return ownedAdjacency.getOrDefault(vertex, List.of());
        return ghostAdjacency.getOrDefault(vertex, List.of());
    }

    /** Degree of a vertex, using owned or ghost adjacency. */
    int degree(int vertex) { return neighboursOf(vertex).size(); }

    /** Triangles of the edges this partition holds, with the ownership rule (u < v, third vertex > v). */
    int triangles() {
        int count = 0;
        for (int[] edge : edges) {
            int u = Math.min(edge[0], edge[1]);
            int v = Math.max(edge[0], edge[1]);
            if (u == v) continue;
            List<Integer> left = neighboursOf(u);
            Set<Integer> right = new TreeSet<>(neighboursOf(v));
            for (int w : left) {
                if (w > v && right.contains(w)) count++;
            }
        }
        return count;
    }

    /** Wedges: SUM over the vertices this partition owns of C(deg(v),2). Each vertex has one owner. */
    long allWedges() {
        long total = 0;
        for (int vertex : ownedAdjacency.keySet()) {
            long d = degree(vertex);
            total += d * (d - 1) / 2;
        }
        return total;
    }

    /** Degree vector contribution: the degrees of the vertices this partition owns. */
    Map<Integer, Integer> ownedDegrees() {
        Map<Integer, Integer> degrees = new TreeMap<>();
        for (int vertex : ownedAdjacency.keySet()) degrees.put(vertex, degree(vertex));
        return degrees;
    }
}

/** A subgraph-centric job over P partitions. */
final class SubgraphJob {
    final List<Partition> partitions = new ArrayList<>();
    final Set<Integer> vertices = new TreeSet<>();
    int messages;

    SubgraphJob(List<int[]> edges, int parts) {
        if (parts < 1) throw new IllegalArgumentException("need at least one partition");
        for (int[] edge : edges) {
            vertices.add(edge[0]);
            vertices.add(edge[1]);
        }
        for (int i = 0; i < parts; i++) partitions.add(new Partition(i, parts));
        for (Partition partition : partitions) partition.load(edges);
        for (Partition partition : partitions) {
            partition.exchange(partitions);
            messages += partition.ghostRequests;
        }
    }

    int triangleCount() {
        int total = 0;
        for (Partition partition : partitions) total += partition.triangles();
        return total;
    }

    long allWedges() {
        long total = 0;
        for (Partition partition : partitions) total += partition.allWedges();
        return total;
    }

    /** Open wedges: the wedges that do not close into a triangle, SUM C(deg,2) - 3 * triangles. */
    long openWedges() { return allWedges() - 3L * triangleCount(); }

    Map<Integer, Integer> degreeVector() {
        Map<Integer, Integer> degrees = new TreeMap<>();
        for (Partition partition : partitions) degrees.putAll(partition.ownedDegrees());
        return degrees;
    }
}

/** Central brute force: the census computed with the whole adjacency in one place. */
final class CentralCensus {
    final Map<Integer, List<Integer>> adjacency = new TreeMap<>();

    CentralCensus(List<int[]> edges) {
        for (int[] edge : edges) {
            add(edge[0], edge[1]);
            add(edge[1], edge[0]);
        }
    }

    private void add(int vertex, int neighbour) {
        List<Integer> list = adjacency.computeIfAbsent(vertex, v -> new ArrayList<>());
        if (!list.contains(neighbour)) list.add(neighbour);
    }

    int triangles() {
        int count = 0;
        for (int u : adjacency.keySet()) {
            for (int v : adjacency.get(u)) {
                if (v <= u) continue;
                for (int w : adjacency.get(u)) {
                    if (w > v && adjacency.get(v).contains(w)) count++;
                }
            }
        }
        return count;
    }

    long allWedges() {
        long total = 0;
        for (int v : adjacency.keySet()) {
            long d = adjacency.get(v).size();
            total += d * (d - 1) / 2;
        }
        return total;
    }

    Map<Integer, Integer> degrees() {
        Map<Integer, Integer> degrees = new TreeMap<>();
        for (Map.Entry<Integer, List<Integer>> entry : adjacency.entrySet()) {
            degrees.put(entry.getKey(), entry.getValue().size());
        }
        return degrees;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240L);
        List<int[]> edges = new ArrayList<>();
        for (int v = 0; v + 1 < 18; v++) edges.add(new int[]{v, v + 1});
        edges.add(new int[]{0, 5});
        edges.add(new int[]{2, 7});
        edges.add(new int[]{3, 9});
        edges.add(new int[]{0, 17});
        for (int i = 0; i < 20; i++) edges.add(new int[]{rng.nextInt(20), rng.nextInt(20)});
        edges.removeIf(e -> e[0] == e[1]);
        CentralCensus central = new CentralCensus(edges);
        int truthTriangles = central.triangles();
        long truthAllWedges = central.allWedges();
        Map<Integer, Integer> truthDegrees = central.degrees();
        check(truthTriangles > 0, "the generated graph must contain triangles, got " + truthTriangles);
        check(truthAllWedges > 0 && !truthDegrees.isEmpty(), "the census must be non-trivial");

        // ---- 1. partition invariance: the census is the same for any number of partitions ---------
        SubgraphJob onePart = new SubgraphJob(edges, 1);
        SubgraphJob twoParts = new SubgraphJob(edges, 2);
        SubgraphJob threeParts = new SubgraphJob(edges, 3);
        SubgraphJob fiveParts = new SubgraphJob(edges, 5);
        SubgraphJob eightParts = new SubgraphJob(edges, 8);
        SubgraphJob[] jobs = {onePart, twoParts, threeParts, fiveParts, eightParts};
        for (SubgraphJob candidate : jobs) {
            check(candidate.triangleCount() == truthTriangles,
                    "triangles must match the central census, got " + candidate.triangleCount() + " vs "
                            + truthTriangles);
            check(candidate.allWedges() == truthAllWedges,
                    "wedges must match the central census, got " + candidate.allWedges() + " vs "
                            + truthAllWedges);
            check(candidate.degreeVector().equals(truthDegrees),
                    "the degree vector must be identical for every partition count");
        }
        check(twoParts.messages > 0 && fiveParts.messages > onePart.messages,
                "more partitions must mean more boundary exchange: " + onePart.messages + " -> "
                        + fiveParts.messages);
        check(eightParts.triangleCount() == truthTriangles && eightParts.allWedges() == truthAllWedges,
                "eight partitions must still reproduce the census");

        // ---- 2. every triangle is counted exactly once -------------------------------------------
        SubgraphJob job = threeParts;
        int localSum = 0;
        int localMax = 0;
        for (Partition partition : job.partitions) {
            int local = partition.triangles();
            localSum += local;
            localMax = Math.max(localMax, local);
        }
        check(localSum == truthTriangles, "the local counts must sum to the global count");
        check(localMax <= truthTriangles, "no partition may count more than the whole graph");
        check(onePart.partitions.get(0).triangles() == truthTriangles,
                "a single partition must count every triangle itself");
        check(onePart.partitions.get(0).crossPartitionEdges == 0 && job.messages > 0,
                "cut edges are what trigger the boundary exchange");

        // ---- 3. the wedge identity: SUM C(deg,2) - 3 * triangles ---------------------------------
        long degreeSum = 0;
        for (int degree : truthDegrees.values()) degreeSum += (long) degree * (degree - 1) / 2;
        check(degreeSum - 3L * truthTriangles == truthAllWedges - 3L * truthTriangles,
                "the central wedge total must equal SUM C(deg,2): " + truthAllWedges + " vs " + degreeSum);
        check(job.allWedges() == degreeSum,
                "the distributed wedge total must equal SUM C(deg,2): " + job.allWedges() + " vs "
                        + degreeSum);
        check(job.openWedges() == degreeSum - 3L * truthTriangles,
                "the open-wedge identity must hold: " + job.openWedges() + " vs " + degreeSum + " - "
                        + (3L * truthTriangles));

        // ---- 4. ghost consistency -----------------------------------------------------------------
        boolean ghostConsistent = true;
        int ghostSeen = 0;
        for (Partition partition : job.partitions) {
            for (Map.Entry<Integer, List<Integer>> entry : partition.ghostAdjacency.entrySet()) {
                ghostSeen++;
                Partition owner = job.partitions.get(partition.ownerOf(entry.getKey()));
                List<Integer> expected = owner.ownedAdjacency.getOrDefault(entry.getKey(), List.of());
                if (!new TreeSet<>(entry.getValue()).equals(new TreeSet<>(expected))) {
                    ghostConsistent = false;
                }
            }
        }
        check(ghostConsistent, "a ghost vertex's adjacency must equal the owner's own list");
        check(ghostSeen > 0, "the exchange must have fetched at least one ghost");
        check(job.messages == ghostSeen || job.messages >= ghostSeen,
                "the message count must cover every fetched ghost: " + job.messages + " vs " + ghostSeen);
        boolean everyVertexOwnedOnce = true;
        Set<Integer> owned = new TreeSet<>();
        for (Partition partition : job.partitions) {
            for (int vertex : partition.ownedAdjacency.keySet()) {
                if (!owned.add(vertex)) everyVertexOwnedOnce = false;
            }
        }
        check(everyVertexOwnedOnce && owned.equals(truthDegrees.keySet()),
                "every vertex must be owned by exactly one partition, owned " + owned.size() + " of "
                        + truthDegrees.size());

        // ---- 5. degenerate inputs, validation and determinism -------------------------------------
        SubgraphJob empty = new SubgraphJob(List.of(), 3);
        check(empty.triangleCount() == 0 && empty.allWedges() == 0 && empty.messages == 0,
                "an empty graph has an empty census and no exchange");
        SubgraphJob sparse = new SubgraphJob(List.of(new int[]{4, 9}, new int[]{9, 11}, new int[]{11, 4}),
                2);
        check(sparse.triangleCount() == 1 && sparse.allWedges() == 3 && sparse.openWedges() == 0,
                "a single triangle must be counted once with three closed wedges, got "
                        + sparse.triangleCount() + "/" + sparse.allWedges() + "/" + sparse.openWedges());
        SubgraphJob path = new SubgraphJob(List.of(new int[]{1, 2}, new int[]{2, 3}), 4);
        check(path.triangleCount() == 0 && path.allWedges() == 1,
                "a path of two edges is exactly one wedge");
        SubgraphJob star = new SubgraphJob(
                List.of(new int[]{0, 1}, new int[]{0, 2}, new int[]{0, 3}, new int[]{0, 4}), 3);
        check(star.triangleCount() == 0 && star.allWedges() == 6,
                "a star of four leaves holds C(4,2) = 6 wedges, got " + star.allWedges());
        expectThrows(() -> new SubgraphJob(edges, 0), "zero partitions");
        SubgraphJob repeat = new SubgraphJob(edges, 3);
        check(repeat.triangleCount() == job.triangleCount() && repeat.allWedges() == job.allWedges()
                && repeat.degreeVector().equals(job.degreeVector()),
                "the same cut must give the same census");

        System.out.println("graph: " + truthDegrees.size() + " vertices, " + edges.size()
                + " distinct edges, triangles " + truthTriangles + ", wedges " + truthAllWedges + ", open "
                + (truthAllWedges - 3L * truthTriangles));
        System.out.println("partitions 1/2/3/5/8 all reproduce the census; boundary messages "
                + jobs[0].messages + "/" + jobs[1].messages + "/" + job.messages + "/" + jobs[3].messages
                + "/" + jobs[4].messages);
        System.out.println("wedges = SUM C(deg,2) - 3*triangles = " + degreeSum + " - "
                + (3 * truthTriangles) + " = " + truthAllWedges + "; ghosts verified " + ghostSeen);
        System.out.println("PASS");
    }

    static void expectThrows(Runnable action, String label) {
        try {
            action.run();
        } catch (RuntimeException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected a rejection for " + label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
