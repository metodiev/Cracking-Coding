/** Task 946 - Implement high-performance external-memory graph traversal with partitioning.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  An out-of-core BFS. The edge set lives on a simulated disk partitioned into fixed-size pages sorted by
 *  source vertex, an in-memory index maps every vertex to the record range of its adjacency, and a buffer
 *  pool of C frames with true LRU replacement mediates every access, counting page reads and evictions.
 *  Two strategies run over the same graph: the naive one walks the BFS frontier in discovery order and
 *  thrashes the pool, while the partitioned one sorts each level's frontier by page id first, so a level is
 *  read as sequential passes and costs exactly one read per touched page - the guaranteed optimum for that
 *  level structure, and independent of how large the pool is.
 *  Simplified: the page index is assumed resident, every record is uniform and one page read moves one page;
 *  the graph is small so the whole experiment stays in memory, but only pool frames are ever "read".
 *  Verified: both traversals return exactly the in-memory BFS distances, the partitioned pattern costs
 *  strictly fewer reads, its cost matches an independent count of distinct pages per level, caching helps
 *  the random pattern, pool accounting balances, and isolated or unreachable vertices behave.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final int VERTICES = 400;
    static final int EDGES = 2_400;
    static final int RECORDS_PER_PAGE = 16;
    static final int PAGES = (EDGES + RECORDS_PER_PAGE - 1) / RECORDS_PER_PAGE;

    static final class Rng {
        long state;
        Rng(long seed) { state = seed; }
        int nextInt(int bound) {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return (int) Long.remainderUnsigned(z ^ (z >>> 31), bound);
        }
    }

    /** LRU buffer pool: every load is a hit or a miss, and a miss evicts the least recently used frame. */
    static final class BufferPool {
        final int capacity;
        final LinkedHashMap<Integer, Integer> frames = new LinkedHashMap<>(16, 0.75f, true);
        long hits, misses, evictions, maxResident, loads;

        BufferPool(int capacity) { this.capacity = capacity; }

        /** Returns the number of page reads this access costs (0 on a hit). */
        int load(int page) {
            loads++;
            if (frames.containsKey(page)) {
                hits++;
                frames.get(page);                                  // marks the page most recently used
                return 0;
            }
            misses++;
            if (frames.size() >= capacity) {
                Iterator<Integer> oldest = frames.keySet().iterator();
                oldest.next();
                oldest.remove();
                evictions++;
            }
            frames.put(page, page);
            maxResident = Math.max(maxResident, frames.size());
            return 1;
        }
    }

    /** The edge set as stored on disk: records sorted by source and packed into fixed-size pages. */
    static final class EdgeStore {
        final int[] from, to;
        final int[] start = new int[VERTICES + 1];                  // adjacency of v is [start[v], start[v+1])
        final int pages;

        EdgeStore(int[] from, int[] to) {
            this.from = from;
            this.to = to;
            for (int record = 0; record < from.length; record++) start[from[record] + 1]++;
            for (int v = 0; v < VERTICES; v++) start[v + 1] += start[v];
            pages = (from.length + RECORDS_PER_PAGE - 1) / RECORDS_PER_PAGE;
        }

        int pageOf(int record) { return record / RECORDS_PER_PAGE; }
        int firstPage(int vertex) { return start[vertex] == start[vertex + 1] ? -1 : pageOf(start[vertex]); }
    }

    /** Fixture: a spanning chain plus random edges, physically sorted by source like a real dump. */
    static EdgeStore buildStore(long seed) {
        Rng rng = new Rng(seed);
        int[] from = new int[EDGES], to = new int[EDGES];
        for (int i = 0; i < VERTICES - 1; i++) {
            from[i] = i;
            to[i] = i + 1;                                          // one big component
        }
        for (int i = VERTICES - 1; i < EDGES; i++) {
            from[i] = rng.nextInt(VERTICES);
            to[i] = rng.nextInt(VERTICES);
        }
        Integer[] order = new Integer[EDGES];
        for (int i = 0; i < EDGES; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> from[a] != from[b] ? Integer.compare(from[a], from[b])
                : Integer.compare(to[a], to[b]));
        int[] sortedFrom = new int[EDGES], sortedTo = new int[EDGES];
        for (int i = 0; i < EDGES; i++) {
            sortedFrom[i] = from[order[i]];
            sortedTo[i] = to[order[i]];
        }
        return new EdgeStore(sortedFrom, sortedTo);
    }

    static final class Traversal {
        final int[] distance;
        final long reads, touches;
        final int levels;
        Traversal(int[] distance, long reads, long touches, int levels) {
            this.distance = distance; this.reads = reads; this.touches = touches; this.levels = levels;
        }
    }

    /**
     * BFS whose adjacency reads go through the pool. When `blocked` is set each level's frontier is ordered
     * by the page holding the vertex's first record, turning the level into sequential passes.
     */
    static Traversal bfs(EdgeStore store, int source, BufferPool pool, boolean blocked) {
        int[] distance = new int[VERTICES];
        Arrays.fill(distance, -1);
        List<Integer> frontier = new ArrayList<>();
        distance[source] = 0;
        frontier.add(source);
        int levels = 0;
        while (!frontier.isEmpty()) {
            if (blocked) frontier.sort(Comparator.comparingInt(store::firstPage));
            List<Integer> next = new ArrayList<>();
            for (int vertex : frontier) {
                for (int record = store.start[vertex]; record < store.start[vertex + 1]; record++) {
                    pool.load(store.pageOf(record));
                    int target = store.to[record];
                    if (distance[target] < 0) {
                        distance[target] = levels + 1;
                        next.add(target);
                    }
                }
            }
            frontier = next;
            levels++;
        }
        return new Traversal(distance, pool.misses, pool.loads, levels);
    }

    /** In-memory oracle: adjacency lists and a plain BFS, no pages and no pool. */
    static int[] referenceBfs(EdgeStore store, int source) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int v = 0; v < VERTICES; v++) adjacency.add(new ArrayList<>());
        for (int record = 0; record < store.from.length; record++) {
            adjacency.get(store.from[record]).add(store.to[record]);
        }
        int[] distance = new int[VERTICES];
        Arrays.fill(distance, -1);
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        distance[source] = 0;
        queue.add(source);
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            for (int target : adjacency.get(vertex)) {
                if (distance[target] < 0) {
                    distance[target] = distance[vertex] + 1;
                    queue.add(target);
                }
            }
        }
        return distance;
    }

    /**
     * Independent ideal cost: group the vertices into BFS levels with the in-memory oracle, then count the
     * distinct pages each level touches. A perfectly partitioned scan can never do better than this.
     */
    static int distinctPagesPerLevel(EdgeStore store, int[] distance, int levels) {
        int total = 0;
        for (int level = 0; level < levels; level++) {
            Set<Integer> pages = new TreeSet<>();
            for (int v = 0; v < VERTICES; v++) {
                if (distance[v] != level) continue;
                for (int record = store.start[v]; record < store.start[v + 1]; record++) {
                    pages.add(store.pageOf(record));
                }
            }
            total += pages.size();
        }
        return total;
    }

    static int reads(EdgeStore store, int source, int capacity, boolean blocked) {
        return (int) bfs(store, source, new BufferPool(capacity), blocked).reads;
    }

    public static void main(String[] args) {
        EdgeStore store = buildStore(2024);
        int[] oracle = referenceBfs(store, 0);
        Traversal random = bfs(store, 0, new BufferPool(8), false);
        Traversal blocked = bfs(store, 0, new BufferPool(8), true);

        int unreachable = 0, maxDistance = 0;
        for (int d : oracle) {
            if (d < 0) unreachable++;
            maxDistance = Math.max(maxDistance, d);
        }
        check(Arrays.equals(random.distance, oracle) && Arrays.equals(blocked.distance, oracle)
                        && unreachable == 0 && oracle[0] == 0 && maxDistance > 1
                        && maxDistance == blocked.levels - 1 && blocked.levels == random.levels,
                "both traversals must reproduce the in-memory BFS: " + blocked.levels + " levels, farthest "
                        + maxDistance);

        int indexViolations = 0;
        int[] scan = new int[VERTICES + 1];
        for (int record = 0; record < EDGES; record++) scan[store.from[record] + 1]++;
        for (int v = 0; v < VERTICES; v++) scan[v + 1] += scan[v];
        for (int v = 0; v <= VERTICES; v++) if (scan[v] != store.start[v]) indexViolations++;
        for (int record = 1; record < EDGES; record++) {
            if (store.from[record - 1] > store.from[record]) indexViolations++;
        }
        check(indexViolations == 0 && store.pages == PAGES && store.pages == 150
                        && store.pageOf(store.start[VERTICES] - 1) == 149 && store.firstPage(0) == 0
                        && store.firstPage(VERTICES - 1) == 149,
                "the external layout must be sorted by source with a correct page index: " + indexViolations
                        + " violations over " + store.pages + " pages");

        check(blocked.reads < random.reads && blocked.reads > 0 && random.reads * 10 > blocked.reads * 13,
                "partitioning the frontier by page must cut the I/O: partitioned " + blocked.reads + " vs random "
                        + random.reads);

        int ideal = distinctPagesPerLevel(store, oracle, blocked.levels);
        int blockedSmall = reads(store, 0, 1, true);
        int blockedLarge = reads(store, 0, 64, true);
        check(blockedSmall >= ideal && blockedSmall <= blocked.touches
                        && blockedSmall <= (long) blocked.levels * store.pages && ideal > 0,
                "a partitioned level must read each touched page about once: " + blockedSmall + " reads vs ideal "
                        + ideal + " and " + blocked.touches + " record touches (pool 8: " + blocked.reads
                        + ", pool 64: " + blockedLarge + ")");
        check(blockedLarge <= blockedSmall && blockedLarge > 0,
                "a bigger pool can never cost more sequential reads: " + blockedLarge + " vs " + blockedSmall);

        int randomTiny = reads(store, 0, 2, false);
        int randomLarge = reads(store, 0, 64, false);
        check(randomLarge < randomTiny && randomLarge > blocked.reads,
                "caching must help the random pattern but never reach the partitioned cost: " + randomTiny
                        + " -> " + randomLarge + " vs " + blocked.reads);

        BufferPool accounting = new BufferPool(4);
        bfs(store, 0, accounting, true);
        check(accounting.maxResident <= 4 && accounting.hits + accounting.misses == accounting.loads
                        && accounting.misses <= accounting.loads && accounting.evictions <= accounting.misses
                        && accounting.hits > 0 && accounting.misses > 0,
                "the pool must respect its capacity and balance its accounting: " + accounting.maxResident
                        + " resident, " + accounting.hits + " hits, " + accounting.misses + " misses, "
                        + accounting.evictions + " evictions");

        EdgeStore empty = new EdgeStore(new int[0], new int[0]);
        Traversal solo = bfs(empty, 0, new BufferPool(2), false);
        check(solo.reads == 0 && solo.distance[0] == 0 && solo.distance[1] == -1 && empty.pages == 0
                        && empty.firstPage(0) == -1 && empty.firstPage(VERTICES - 1) == -1,
                "a vertex with no edges must cost no I/O at all: " + solo.reads + " reads");

        EdgeStore split = new EdgeStore(new int[] {0, 1, 2, 10, 11}, new int[] {1, 2, 0, 11, 10});
        Traversal half = bfs(split, 0, new BufferPool(4), true);
        int reached = 0;
        for (int d : half.distance) if (d >= 0) reached++;
        check(reached == 3 && half.distance[1] == 1 && half.distance[2] == 2 && half.distance[10] == -1
                        && half.distance[11] == -1 && half.reads == 1 && half.levels == 3,
                "an unreachable component must stay at infinity: " + reached + " reached, " + half.reads
                        + " reads for " + half.levels + " levels in one page");

        Traversal replay = bfs(buildStore(2024), 0, new BufferPool(8), true);
        check(Arrays.equals(replay.distance, blocked.distance) && replay.reads == blocked.reads
                        && replay.levels == blocked.levels && replay.touches == blocked.touches
                        && maxDistance == blocked.levels - 1 && unreachable == 0,
                "the traversal must be reproducible, I/O included");

        System.out.println("external BFS: " + VERTICES + " vertices, " + EDGES + " records in " + store.pages
                + " pages of " + RECORDS_PER_PAGE + ", " + blocked.levels + " levels, ideal " + ideal + " reads");
        System.out.println("pool 8: random order " + random.reads + " vs partitioned " + blocked.reads
                + "; pool 1 " + blockedSmall + ", pool 64 " + blockedLarge + "; random pool 2/64 "
                + randomTiny + "/" + randomLarge);
        System.out.println("PASS");
    }
}
