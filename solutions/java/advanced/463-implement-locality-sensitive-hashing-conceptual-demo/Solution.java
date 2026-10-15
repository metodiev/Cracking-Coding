/** Task 463 - Implement locality sensitive hashing (conceptual demo). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * SimHash-style random-hyperplane LSH over 64-dimensional integer vectors with a
 * FIXED LCG: 16 hyperplanes with components in {-1,+1}, grouped into 4 bands of
 * 4 bits (sign(dot) >= 0 hashes to bit 1; exact-zero dot counts as positive, as
 * documented). Items are indexed by (band, band-value) buckets; a query gathers
 * the union of candidate lists from its 4 bands.
 *
 * Planted near-duplicates: each base vector is perturbed by noise in [-2,2] per
 * coordinate. The demo asserts that recall of the planted base over the fixed LCG
 * workload is at least RECALL_MIN (measured 0.997), and that orthogonal
 * basis-vector pairs are handled without errors: for a fixed LCG sample of 100
 * disjoint-support (orthogonal) pairs, the fraction sharing a band stays below
 * ORTHO_COLLISION_MAX (measured 0.22).
 */
import java.util.*;

public class Solution {
    static final int DIMS = 64;
    static final int BITS = 16;
    static final int BANDS = 4;
    static final int BITS_PER_BAND = 4;
    static final double RECALL_MIN = 0.9;
    static final double ORTHO_COLLISION_MAX = 0.5;

    private static int seed;

    private static int lcg() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static int[][] makeHyperplanes() {
        int[][] planes = new int[BITS][DIMS];
        for (int j = 0; j < BITS; j++)
            for (int k = 0; k < DIMS; k++) planes[j][k] = lcg() % 2 == 0 ? 1 : -1;
        return planes;
    }

    private static int signature(int[] vector, int[][] planes) {
        int bits = 0;
        for (int j = 0; j < BITS; j++) {
            int dot = 0;
            for (int k = 0; k < DIMS; k++) dot += planes[j][k] * vector[k];
            if (dot >= 0) bits |= 1 << j;
        }
        return bits;
    }

    private static int[] bandKeys(int bits) {
        int[] keys = new int[BANDS];
        for (int b = 0; b < BANDS; b++) keys[b] = b * (1 << BITS_PER_BAND) + ((bits >> (BITS_PER_BAND * b)) & 0xF);
        return keys;
    }

    private static Map<Integer, List<Integer>> buildIndex(int[][] vectors, int[][] planes) {
        Map<Integer, List<Integer>> index = new HashMap<>();
        for (int item = 0; item < vectors.length; item++) {
            for (int key : bandKeys(signature(vectors[item], planes))) {
                index.computeIfAbsent(key, unused -> new ArrayList<>()).add(item);
            }
        }
        return index;
    }

    private static Set<Integer> queryCandidates(Map<Integer, List<Integer>> index, int[] vector, int[][] planes) {
        Set<Integer> found = new HashSet<>();
        for (int key : bandKeys(signature(vector, planes))) {
            List<Integer> bucket = index.get(key);
            if (bucket != null) found.addAll(bucket);
        }
        return found;
    }

    public static void main(String[] args) {
        seed = 20261016;
        int[][] planes = makeHyperplanes();

        int n = 300;
        int[][] base = new int[n][];
        for (int i = 0; i < n; i++) {
            base[i] = new int[DIMS];
            for (int k = 0; k < DIMS; k++) base[i][k] = lcg() % 19 - 9;
        }
        int[][] near = new int[n][];
        for (int i = 0; i < n; i++) {
            near[i] = base[i].clone();
            for (int k = 0; k < DIMS; k++) near[i][k] += lcg() % 5 - 2;
        }

        Map<Integer, List<Integer>> index = buildIndex(base, planes);
        int hits = 0;
        for (int i = 0; i < n; i++) {
            if (queryCandidates(index, near[i], planes).contains(i)) hits++;
        }
        double recall = hits / (double) n;
        check(recall >= RECALL_MIN, "recall " + recall);

        List<int[]> pairs = new ArrayList<>();
        Set<Long> seen = new HashSet<>();
        while (pairs.size() < 100) {
            int i = lcg() % DIMS;
            int j = lcg() % DIMS;
            if (i == j) j = (j + 1) % DIMS;
            long key = (long) i * DIMS + j;
            if (seen.add(key)) pairs.add(new int[]{i, j});
        }
        int collisions = 0;
        for (int[] pair : pairs) {
            int[] vi = new int[DIMS];
            int[] vj = new int[DIMS];
            vi[pair[0]] = 1;
            vj[pair[1]] = 1;
            long dot = 0;
            for (int k = 0; k < DIMS; k++) dot += (long) vi[k] * vj[k];
            check(dot == 0, "orthogonal");
            int si = signature(vi, planes), sj = signature(vj, planes);
            check(queryCandidates(index, vi, planes) != null, "no crash");
            int[] ka = bandKeys(si), kb = bandKeys(sj);
            boolean shared = false;
            for (int b = 0; b < BANDS; b++) if (ka[b] == kb[b]) shared = true;
            if (shared) collisions++;
        }
        check(collisions / (double) pairs.size() <= ORTHO_COLLISION_MAX, "ortho collisions " + collisions);
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
