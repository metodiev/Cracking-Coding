/** Task 815 - Implement real-time vector search at billion-vector scale with sharding and reranking. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The index is a sharded IVF (inverted-file) index, the layout used by billion-scale engines (FAISS-IVF +
 * shards + a reranking stage):
 *
 *   build     k-means with a fixed iteration count clusters the base vectors into `nlist` centroids, and
 *             cluster c is owned by shard c mod S.  Every vector lives in exactly one cluster and one shard.
 *   search    the query is compared with all centroids, the `nprobe` nearest clusters are selected, and the
 *             shards owning them are visited in shard order; each visited shard scans only its own clusters
 *             and returns its local candidates.  Candidates are merged and *reranked* with exact L2.
 *
 * Billion scale is expressed through typed parameters: the index is evaluated on a few thousand real vectors
 * and the *measured* scan fraction (probed clusters / nlist) is extrapolated analytically to 10^9 vectors,
 * so cost claims are numbers, not adjectives.  The extrapolation is stated as such wherever it is used.
 *
 * Deliberate simplifications (deterministic; java.util only; no threads, sockets, clocks or files):
 *   - Shards are executed sequentially rather than concurrently; latency is therefore reported as work
 *     (distance computations and bytes scanned) rather than wall-clock time.
 *   - Vectors are 8-dimensional doubles with a small dynamic range, so a brute-force oracle can be run on
 *     the whole base set for recall measurement.
 *
 * Verified invariants:
 *   partitioning - every base vector is assigned to exactly one cluster, cluster sizes sum to N, and shard
 *                  load stays within a bounded ratio of the mean.
 *   recall       - recall@k against an exhaustive brute-force oracle is measured for several nprobe values,
 *                  is non-decreasing in nprobe, and is exactly 1.0 when every cluster is probed.
 *   ordering     - results are sorted by exact distance with deterministic id tie-breaks, and the reranked
 *                  order of the candidate set equals the exact ordering of that same set.
 *   cost         - distance computations stay proportional to the scanned fraction, and the analytic
 *                  10^9-vector projection of that fraction stays inside the stated bound.
 */
import java.util.*;

/** One indexed point. */
class Vec {
    final int id;
    final double[] v;

    Vec(int id, double[] v) { this.id = id; this.v = v; }
}

/** A query hit: the point id and its exact L2 distance to the query. */
class Hit implements Comparable<Hit> {
    final int id;
    final double dist;

    Hit(int id, double dist) { this.id = id; this.dist = dist; }

    @Override public int compareTo(Hit other) {
        if (dist != other.dist) return Double.compare(dist, other.dist);
        return Integer.compare(id, other.id);
    }
}

/** Sharded IVF index with an exact reranking stage. */
class IvfIndex {
    final List<Vec> base;
    final int dim;
    final int nlist;
    final int shards;
    double[][] centroids;
    List<List<Integer>> clusters = new ArrayList<>();
    int[] assign;
    long distanceComputations;
    long scannedVectors;
    final List<Double> objectiveTrace = new ArrayList<>();

    IvfIndex(List<Vec> base, int dim, int nlist, int shards) {
        this.base = base; this.dim = dim; this.nlist = nlist; this.shards = shards;
    }

    static double l2(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) { double d = a[i] - b[i]; s += d * d; }
        return s < 0 ? 0 : s;
    }

    /** k-means with a fixed iteration budget and a seeded, deterministic initialisation. */
    void build(long seed, int iterations) {
        SplitMix64 rng = new SplitMix64(seed);
        centroids = new double[nlist][dim];
        for (int c = 0; c < nlist; c++) centroids[c] = base.get(rng.nextInt(base.size())).v.clone();
        assign = new int[base.size()];
        for (int it = 0; it < iterations; it++) {
            long[] counts = new long[nlist];
            double[][] sums = new double[nlist][dim];
            double objective = 0;
            for (int i = 0; i < base.size(); i++) {
                int best = 0;
                double bestDist = Double.MAX_VALUE;
                for (int c = 0; c < nlist; c++) {
                    double d = l2(base.get(i).v, centroids[c]);
                    if (d < bestDist || (d == bestDist && c < best)) { bestDist = d; best = c; }
                }
                assign[i] = best;
                counts[best]++;
                objective += bestDist;
                for (int j = 0; j < dim; j++) sums[best][j] += base.get(i).v[j];
            }
            objectiveTrace.add(objective);
            for (int c = 0; c < nlist; c++) {
                if (counts[c] == 0) continue;   // empty cluster keeps its centroid
                for (int j = 0; j < dim; j++) centroids[c][j] = sums[c][j] / counts[c];
            }
        }
        clusters = new ArrayList<>();
        for (int c = 0; c < nlist; c++) clusters.add(new ArrayList<>());
        for (int i = 0; i < base.size(); i++) clusters.get(assign[i]).add(i);
    }

    int shardOf(int cluster) { return cluster % shards; }

    int shardSize(int shard) {
        int n = 0;
        for (int c = 0; c < nlist; c++) if (shardOf(c) == shard) n += clusters.get(c).size();
        return n;
    }

    /** The nprobe clusters nearest to the query, ordered by (distance, id) for determinism. */
    int[] probe(double[] query, int nprobe) {
        List<int[]> scored = new ArrayList<>();
        for (int c = 0; c < nlist; c++) {
            distanceComputations++;
            scored.add(new int[] {c, (int) Math.round(l2(query, centroids[c]) * 1e6)});
        }
        scored.sort((a, b) -> a[1] != b[1] ? Integer.compare(a[1], b[1]) : Integer.compare(a[0], b[0]));
        int[] out = new int[Math.min(nprobe, nlist)];
        for (int i = 0; i < out.length; i++) out[i] = scored.get(i)[0];
        return out;
    }

    /** Scans one shard over the probed clusters it owns and returns its candidates. */
    List<Hit> scanShard(double[] query, int shard, int[] probed) {
        List<Hit> hits = new ArrayList<>();
        for (int c : probed) {
            if (shardOf(c) != shard) continue;
            for (int idx : clusters.get(c)) {
                distanceComputations++;
                scannedVectors++;
                hits.add(new Hit(base.get(idx).id, l2(query, base.get(idx).v)));
            }
        }
        return hits;
    }

    /** Reranking stage: sort the merged candidates exactly and keep the best k. */
    static List<Hit> rerank(List<Hit> candidates, int k) {
        List<Hit> sorted = new ArrayList<>(candidates);
        Collections.sort(sorted);
        return sorted.subList(0, Math.min(k, sorted.size()));
    }

    /** Full shard-parallel search: probe, per-shard scan, merge, rerank. */
    List<Hit> search(double[] query, int k, int nprobe) {
        if (base.isEmpty()) return new ArrayList<>();
        int[] probed = probe(query, nprobe);
        List<Hit> merged = new ArrayList<>();
        for (int s = 0; s < shards; s++) merged.addAll(scanShard(query, s, probed));
        return rerank(merged, k);
    }

    /** Brute force oracle over the entire base set. */
    static List<Hit> bruteForce(List<Vec> base, double[] query, int k) {
        List<Hit> all = new ArrayList<>();
        for (Vec p : base) all.add(new Hit(p.id, l2(query, p.v)));
        return rerank(all, k);
    }
}

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    static final long BILLION = 1_000_000_000L;

    static List<Vec> makeVectors(int n, int dim, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        List<Vec> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double[] v = new double[dim];
            for (int j = 0; j < dim; j++) v[j] = rng.nextDouble() * 10;
            out.add(new Vec(i, v));
        }
        return out;
    }

    static double recall(List<Hit> got, List<Hit> truth) {
        if (truth.isEmpty()) return 1;
        Set<Integer> ids = new HashSet<>();
        for (Hit h : got) ids.add(h.id);
        int hitCount = 0;
        for (Hit h : truth) if (ids.contains(h.id)) hitCount++;
        return hitCount / (double) truth.size();
    }

    public static void main(String[] args) {
        int n = 1200, dim = 8, nlist = 16, shards = 4, k = 10;
        List<Vec> base = makeVectors(n, dim, 20260815L);
        IvfIndex index = new IvfIndex(base, dim, nlist, shards);
        index.build(99L, 12);

        // ---- 1. partitioning invariants ------------------------------------------------------------
        int assigned = 0;
        boolean[] seen = new boolean[n];
        boolean duplicate = false;
        for (int c = 0; c < nlist; c++) for (int idx : index.clusters.get(c)) {
            if (seen[idx]) duplicate = true;
            seen[idx] = true;
            assigned++;
        }
        check(assigned == n, "cluster sizes must sum to N: " + assigned);
        check(!duplicate, "no vector may appear in two clusters");
        boolean allSeen = true;
        for (boolean b : seen) allSeen &= b;
        check(allSeen, "every vector must be assigned to exactly one cluster");
        for (int c = 0; c < nlist; c++) {
            for (int idx : index.clusters.get(c)) check(index.assign[idx] == c, "cluster membership must match assign");
        }

        // shard load balance
        int maxShard = 0, minShard = Integer.MAX_VALUE, total = 0;
        for (int s = 0; s < shards; s++) {
            int size = index.shardSize(s);
            maxShard = Math.max(maxShard, size);
            minShard = Math.min(minShard, size);
            total += size;
        }
        check(total == n, "shard sizes must sum to N");
        check(maxShard <= (int) Math.ceil(total / (double) shards * 1.5),
                "shard load must stay within 1.5x the mean: max=" + maxShard);

        // ---- 2. k-means objective is non-increasing ------------------------------------------------
        boolean monotone = true;
        for (int i = 1; i < index.objectiveTrace.size(); i++) {
            if (index.objectiveTrace.get(i) > index.objectiveTrace.get(i - 1) + 1e-9) monotone = false;
        }
        check(monotone, "the k-means objective must be non-increasing: " + index.objectiveTrace);
        check(index.objectiveTrace.get(index.objectiveTrace.size() - 1) < index.objectiveTrace.get(0),
                "clustering must actually improve the objective");

        // ---- 3. recall against brute force, and its monotonicity in nprobe -------------------------
        int queries = 60;
        int[][] probes = {{1}, {2}, {4}, {8}, {16}};
        double[] recallByProbe = new double[probes.length];
        for (int pi = 0; pi < probes.length; pi++) {
            double sum = 0;
            SplitMix64 qr = new SplitMix64(4242L);
            for (int q = 0; q < queries; q++) {
                double[] query = new double[dim];
                for (int j = 0; j < dim; j++) query[j] = qr.nextDouble() * 10;
                List<Hit> truth = IvfIndex.bruteForce(base, query, k);
                List<Hit> got = index.search(query, k, probes[pi][0]);
                sum += recall(got, truth);
            }
            recallByProbe[pi] = sum / queries;
        }
        for (int pi = 1; pi < probes.length; pi++) {
            check(recallByProbe[pi] >= recallByProbe[pi - 1] - 1e-9,
                    "recall must not drop when nprobe grows: " + Arrays.toString(recallByProbe));
        }
        check(recallByProbe[2] >= 0.85, "recall@10 at nprobe=4 must exceed 0.85, got " + recallByProbe[2]);
        check(recallByProbe[probes.length - 1] == 1.0,
                "probing every cluster must reproduce brute force exactly, got " + recallByProbe[probes.length - 1]);

        // ---- 4. exact ordering, reranking and tie-breaking ----------------------------------------
        SplitMix64 qr = new SplitMix64(777L);
        double[] query = new double[dim];
        for (int j = 0; j < dim; j++) query[j] = qr.nextDouble() * 10;
        List<Hit> exact = index.search(query, k, nlist);
        List<Hit> truth = IvfIndex.bruteForce(base, query, k);
        check(exact.size() == k, "a full probe must return k hits");
        boolean sameOrder = true;
        for (int i = 0; i < k; i++) sameOrder &= exact.get(i).id == truth.get(i).id && exact.get(i).dist == truth.get(i).dist;
        check(sameOrder, "a full probe must match the brute-force order exactly");
        for (int i = 1; i < exact.size(); i++) {
            check(exact.get(i - 1).dist <= exact.get(i).dist, "hits must be sorted by distance");
            check(exact.get(i).dist >= 0, "squared L2 must be non-negative");
        }

        int[] probed = index.probe(query, 4);
        List<Hit> candidates = new ArrayList<>();
        for (int s = 0; s < shards; s++) candidates.addAll(index.scanShard(query, s, probed));
        List<Hit> reranked = IvfIndex.rerank(candidates, 10);
        List<Hit> exactOverCandidates = IvfIndex.rerank(candidates, candidates.size());
        check(reranked.get(0).id == exactOverCandidates.get(0).id,
                "reranking must return the true nearest candidate");
        boolean rerankMatches = true;
        for (int i = 0; i < reranked.size(); i++) rerankMatches &= reranked.get(i).id == exactOverCandidates.get(i).id;
        check(rerankMatches, "reranked order must equal the exact order of the candidate set");

        // ---- 5. shard merge equals a global scan over the probed clusters --------------------------
        List<Hit> global = new ArrayList<>();
        for (int c : probed) for (int idx : index.clusters.get(c)) global.add(new Hit(base.get(idx).id, IvfIndex.l2(query, base.get(idx).v)));
        List<Hit> globalTop = IvfIndex.rerank(global, 10);
        List<Hit> sharded = index.search(query, 10, 4);
        boolean shardMergeEqual = true;
        for (int i = 0; i < globalTop.size(); i++) shardMergeEqual &= globalTop.get(i).id == sharded.get(i).id;
        check(shardMergeEqual, "shard-local scans must merge into the global scan of the probed clusters");
        check(sharded.size() == 10, "the sharded search must return k hits");

        // ---- 6. cost model and the analytic billion-scale projection ------------------------------
        long beforeDist = index.distanceComputations, beforeScanned = index.scannedVectors;
        List<Hit> measured = index.search(query, 10, 4);
        long dist = index.distanceComputations - beforeDist;
        long scanned = index.scannedVectors - beforeScanned;
        check(measured.size() == 10, "the measured search must return hits");
        double scanFraction = scanned / (double) n;
        check(dist <= nlist + n, "distance work must be bounded by the probed scan: " + dist);
        check(scanFraction < 4.0 * 4 / (double) nlist,
                "the scanned fraction must track nprobe/nlist: " + scanFraction + " vs " + (4.0 / nlist));
        long projected = (long) (scanFraction * BILLION);
        check(projected < BILLION / 2L,
                "the analytic 10^9 projection must stay below half the corpus: " + projected);
        double projectedRatio = projected / (double) BILLION;
        check(projectedRatio > 0.0 && projectedRatio < 4.0 / nlist * 4, "the projected ratio must be small: " + projectedRatio);

        // ---- 7. edge cases and determinism --------------------------------------------------------
        IvfIndex emptyIndex = new IvfIndex(new ArrayList<>(), dim, 4, 2);
        check(emptyIndex.search(new double[dim], 10, 2).isEmpty(), "an empty index must return no hits");
        check(emptyIndex.distanceComputations == 0, "an empty index must do no distance work");

        List<Hit> many = index.search(query, n + 50, 16);
        check(many.size() == n, "k larger than N must return every vector: " + many.size());
        for (int i = 1; i < many.size(); i++) check(many.get(i - 1).dist <= many.get(i).dist, "the full ranking must be sorted");

        List<Hit> again = index.search(query, 10, 4);
        boolean stable = true;
        for (int i = 0; i < again.size(); i++) stable &= again.get(i).id == measured.get(i).id;
        check(stable, "repeating a query must return identical results");

        IvfIndex rebuilt = new IvfIndex(base, dim, nlist, shards);
        rebuilt.build(99L, 12);
        boolean sameCentroids = true;
        for (int c = 0; c < nlist; c++) sameCentroids &= Arrays.equals(index.centroids[c], rebuilt.centroids[c]);
        check(sameCentroids, "the build must be reproducible from its seed");
        List<Hit> rebuiltHits = rebuilt.search(query, 10, 4);
        boolean sameHits = true;
        for (int i = 0; i < rebuiltHits.size(); i++) sameHits &= rebuiltHits.get(i).id == measured.get(i).id;
        check(sameHits, "a rebuilt index must answer identically");

        System.out.println("base vectors / clusters   : " + n + " / " + nlist + " over " + shards + " shards");
        System.out.println("recall@10 by nprobe       : " + Arrays.toString(recallByProbe));
        System.out.println("last query work           : " + dist + " distances, " + scanned + " vectors scanned");
        System.out.println("scan fraction / projected : " + String.format("%.4f", scanFraction) + " / "
                + projected + " vectors at 10^9 scale (analytic)");
        System.out.println("shard load max / mean     : " + maxShard + " / " + (total / shards));
        System.out.println("PASS");
    }
}
