/** Task 950 - Implement scalable nearest neighbor with multi-index hashing and rerank pipeline.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A multi-index-hashing (MIH) search pipeline for binary codes. The b-bit code is split into m banded
 *  substrings, each band gets its own bucket table, and a radius-r query probes the buckets whose key lies
 *  within floor(r/m) bits of the query's own slice, unions the bands and reranks the candidates by the exact
 *  Hamming distance. Two codes at distance at most r must agree within floor(r/m) bits on some band, so the
 *  candidate set is provably complete for radius r whatever m is - a guarantee this file checks against a
 *  brute-force scan rather than assuming - while m = r+1 needs only the exact buckets and is therefore the
 *  cheapest complete configuration. An adaptive search widens the radius until enough candidates exist.
 *  Simplified: an exhaustive scan of one bucket space per band instead of a recursive enumeration of the
 *  neighbouring keys, uniform band splits with the remainder in the last band, and no early termination in
 *  the rerank stage.
 *  Verified: radius recall is exactly 1 for every band count, the adaptive top-k search recovers the true
 *  neighbours, fewer bands cost more probing at equal recall, the reranked distances agree with brute force
 *  and come out sorted, candidates are deduplicated, and uneven splits, duplicates, exact queries and empty
 *  indexes all behave.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final int BITS = 32;
    static final int ITEMS = 600;
    static final int TOP_K = 5;
    static final int QUERIES = 200;

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

    /** Multi-index hash: one bucket table per band, plus the probe accounting the cost model needs. */
    static final class Index {
        final int[] codes;
        final int bits, bands;
        final List<Map<Integer, List<Integer>>> tables = new ArrayList<>();
        long probes;

        Index(int[] codes, int bits, int bands) {
            this.codes = codes;
            this.bits = bits;
            this.bands = bands;
            for (int band = 0; band < bands; band++) {
                Map<Integer, List<Integer>> table = new TreeMap<>();
                for (int item = 0; item < codes.length; item++) {
                    table.computeIfAbsent(slice(codes[item], band), key -> new ArrayList<>()).add(item);
                }
                tables.add(table);
            }
        }

        int sliceBits(int band) {
            return band == bands - 1 ? bits / bands + bits % bands : bits / bands;   // the remainder rides along
        }

        int shift(int band) {
            int shift = bits;
            for (int k = 0; k <= band; k++) shift -= sliceBits(k);
            return shift;
        }

        int slice(int code, int band) {
            return (code >>> shift(band)) & ((1 << sliceBits(band)) - 1);
        }

        /** Candidates for a whole radius: every bucket within floor(radius / bands) bits of the query slice. */
        List<Integer> candidates(int query, int radius) {
            int width = radius / bands;
            boolean[] seen = new boolean[codes.length];
            List<Integer> found = new ArrayList<>();
            for (int band = 0; band < bands; band++) {
                int wanted = slice(query, band);
                int space = 1 << sliceBits(band);
                for (int key = 0; key < space; key++) {
                    if (Integer.bitCount(key ^ wanted) > width) continue;
                    probes++;                                        // one relevant bucket lookup per band
                    List<Integer> bucket = tables.get(band).get(key);
                    if (bucket == null) continue;
                    for (int item : bucket) {
                        if (seen[item]) continue;
                        seen[item] = true;
                        found.add(item);
                    }
                }
            }
            return found;
        }
    }

    /** Rerank stage: exact Hamming distances, sorted, ties broken by id so results are reproducible. */
    static List<int[]> rerank(int[] codes, int query, List<Integer> candidates, int k) {
        List<int[]> scored = new ArrayList<>();
        for (int item : candidates) scored.add(new int[] {Integer.bitCount(codes[item] ^ query), item});
        scored.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
        return new ArrayList<>(scored.subList(0, Math.min(k, scored.size())));
    }

    static List<int[]> bruteForce(int[] codes, int query, int k) {
        List<Integer> all = new ArrayList<>();
        for (int item = 0; item < codes.length; item++) all.add(item);
        return rerank(codes, query, all, k);
    }

    /** Fraction of the true neighbours inside `radius` that the candidate set actually contains. */
    static double radiusRecall(Index index, int query, int radius) {
        List<Integer> candidates = index.candidates(query, radius);
        boolean[] present = new boolean[index.codes.length];
        for (int item : candidates) present[item] = true;
        int within = 0, covered = 0;
        for (int item = 0; item < index.codes.length; item++) {
            if (Integer.bitCount(index.codes[item] ^ query) > radius) continue;
            within++;
            if (present[item]) covered++;
        }
        return within == 0 ? 1.0 : (double) covered / within;
    }

    /** The full pipeline on a fixed radius: candidates from the index, then the exact rerank. */
    static List<int[]> search(Index index, int query, int radius, int k) {
        return rerank(index.codes, query, index.candidates(query, radius), k);
    }

    /** Adaptive pipeline: widen the radius until enough candidates exist, then rerank exactly. */
    static List<int[]> adaptiveSearch(Index index, int query, int k, int minCandidates, int maxRadius) {
        for (int radius = 0; radius <= maxRadius; radius++) {
            List<Integer> candidates = index.candidates(query, radius);
            if (radius == maxRadius || candidates.size() >= Math.max(k, minCandidates)) {
                return rerank(index.codes, query, candidates, k);
            }
        }
        return new ArrayList<>();
    }

    /** Fixture: random codes plus planted near-duplicates, so true neighbours exist inside radius 2. */
    static int[] fixture(int items, long seed) {
        Rng rng = new Rng(seed);
        int[] codes = new int[items];
        for (int i = 0; i < items; i++) codes[i] = rng.nextInt(1 << 30);
        for (int i = items / 2; i < items; i++) {
            if (i % 2 == 0) codes[i] = codes[i - items / 2] ^ (1 << rng.nextInt(BITS));   // distance exactly 1
        }
        return codes;
    }

    static int flip(int code, int bit) { return code ^ (1 << bit); }

    public static void main(String[] args) {
        int[] codes = fixture(ITEMS, 20240);
        Index index = new Index(codes, BITS, 4);

        double exactRecall = 0;
        for (int q = 0; q < QUERIES; q++) exactRecall += radiusRecall(index, codes[(q * 7) % ITEMS], 2);
        check(exactRecall / QUERIES == 1.0 && index.sliceBits(0) == 8 && index.sliceBits(3) == 8
                        && index.shift(0) == 24,
                "with m = r + 1 bands the radius-r candidate set must be complete: recall " + exactRecall / QUERIES);

        int distanceViolations = 0, duplicateViolations = 0, sortViolations = 0;
        double topRecall = 0;
        for (int q = 0; q < QUERIES; q++) {
            int query = flip(codes[(q * 11) % ITEMS], q % BITS);
            List<Integer> candidates = index.candidates(query, 3);
            List<int[]> ranked = adaptiveSearch(index, query, TOP_K, 64, 16);
            List<int[]> truth = bruteForce(codes, query, TOP_K);
            Set<Integer> unique = new TreeSet<>(candidates);
            if (unique.size() != candidates.size()) duplicateViolations++;
            for (int i = 1; i < ranked.size(); i++) if (ranked.get(i - 1)[0] > ranked.get(i)[0]) sortViolations++;
            for (int[] row : ranked) {
                if (Integer.bitCount(codes[row[1]] ^ query) != row[0]) distanceViolations++;
            }
            int hits = 0;
            for (int[] want : truth) for (int[] got : ranked) if (got[1] == want[1]) hits++;
            topRecall += (double) hits / truth.size();
        }
        check(distanceViolations == 0 && duplicateViolations == 0 && sortViolations == 0
                        && topRecall / QUERIES >= 0.9,
                "the rerank stage must be exact and recover almost every true neighbour: " + topRecall / QUERIES
                        + ", " + duplicateViolations + " duplicate lists, " + sortViolations + " mis-sorts");

        Index fourBand = new Index(codes, BITS, 4);
        Index twoBand = new Index(codes, BITS, 2);
        double fourRecall = 0, twoRecall = 0;
        for (int q = 0; q < QUERIES; q++) {
            int query = codes[(q * 13) % ITEMS];
            fourRecall += radiusRecall(fourBand, query, 2);
            twoRecall += radiusRecall(twoBand, query, 2);
        }
        check(fourRecall == twoRecall && twoRecall / QUERIES == 1.0 && twoBand.probes >= 4 * fourBand.probes
                        && fourBand.probes == QUERIES * 4,
                "fewer bands stay complete but cost more probing: " + twoBand.probes + " vs " + fourBand.probes
                        + " bucket lookups at equal recall " + twoRecall / QUERIES);

        Index accounting = new Index(codes, BITS, 4);
        long candidateTotal = 0;
        for (int q = 0; q < QUERIES; q++) candidateTotal += accounting.candidates(codes[q], 2).size();
        double averageCandidates = (double) candidateTotal / QUERIES;
        check(accounting.probes == (long) QUERIES * 4 && averageCandidates < ITEMS / 2
                        && accounting.probes + candidateTotal < (long) QUERIES * ITEMS,
                "probing must stay sublinear in the index: " + accounting.probes + " bucket lookups and "
                        + String.format("%.1f", averageCandidates) + " candidates per query");

        int[] duplicated = codes.clone();
        duplicated[10] = codes[3];
        duplicated[11] = codes[3];
        Index withDuplicates = new Index(duplicated, BITS, 4);
        List<Integer> exactHits = withDuplicates.candidates(codes[3], 0);
        List<int[]> rankedDuplicates = rerank(duplicated, codes[3], exactHits, TOP_K);
        check(exactHits.contains(3) && exactHits.contains(10) && exactHits.contains(11) && exactHits.size() < 40
                        && rankedDuplicates.size() == TOP_K && rankedDuplicates.get(0)[0] == 0
                        && rankedDuplicates.get(1)[0] == 0 && rankedDuplicates.get(2)[0] == 0
                        && rankedDuplicates.get(0)[1] == 3 && rankedDuplicates.get(3)[0] > 0,
                "a radius-zero query must keep every duplicate: " + exactHits.size() + " candidates, top "
                        + Arrays.toString(rankedDuplicates.get(0)));

        Index uneven = new Index(codes, BITS, 3);
        int unevenMisses = 0;
        for (int q = 0; q < 40; q++) {
            int query = codes[q * 5];
            if (radiusRecall(uneven, query, 2) != 1.0) unevenMisses++;
        }
        check(unevenMisses == 0 && uneven.sliceBits(0) + uneven.sliceBits(1) + uneven.sliceBits(2) == BITS
                        && uneven.sliceBits(2) == 12,
                "an uneven band split must keep the completeness guarantee: " + unevenMisses + " misses, bands "
                        + uneven.sliceBits(0) + "/" + uneven.sliceBits(1) + "/" + uneven.sliceBits(2));

        Index emptyIndex = new Index(new int[0], BITS, 4);
        List<Integer> emptyCandidates = emptyIndex.candidates(12345, 2);
        check(emptyCandidates.isEmpty() && rerank(new int[0], 12345, emptyCandidates, TOP_K).isEmpty()
                        && bruteForce(new int[0], 12345, TOP_K).isEmpty() && emptyIndex.tables.size() == 4
                        && emptyIndex.probes == 4,
                "an empty index must answer with an empty result: " + emptyCandidates.size());

        double nearRecall = 0, farRecall = 0;
        for (int q = 0; q < QUERIES; q++) {
            int query = flip(codes[(q * 17) % ITEMS], q % 13);
            nearRecall += radiusRecall(index, query, 1);
            farRecall += radiusRecall(index, query, 4);
        }
        check(farRecall / QUERIES >= nearRecall / QUERIES && farRecall / QUERIES >= 0.95,
                "a wider radius must recall at least as much: " + nearRecall / QUERIES + " at r = 1 vs "
                        + farRecall / QUERIES + " at r = 4");

        Index replayIndex = new Index(fixture(ITEMS, 20240), BITS, 4);
        List<Integer> firstRun = index.candidates(codes[77], 3);
        List<Integer> secondRun = replayIndex.candidates(codes[77], 3);
        List<int[]> rankedTop = search(index, codes[77], 3, TOP_K);
        check(firstRun.equals(secondRun) && rankedTop.get(0)[0] == 0 && rankedTop.get(0)[1] == 77
                        && bruteForce(codes, codes[77], TOP_K).get(0)[1] == 77,
                "the pipeline must be reproducible and rank an exact hit first: " + firstRun.size());

        check(exactRecall / QUERIES == 1.0 && topRecall / QUERIES >= 0.9
                        && twoBand.probes >= 4 * fourBand.probes && unevenMisses == 0 && exactHits.size() < 40
                        && accounting.probes + candidateTotal < (long) QUERIES * ITEMS,
                "the whole battery must hold together: exact " + exactRecall / QUERIES + ", top-k "
                        + topRecall / QUERIES + ", bands " + fourBand.probes + " vs " + twoBand.probes);

        System.out.println("MIH index: " + ITEMS + " codes of " + BITS + " bits in 4 bands of "
                + index.sliceBits(0) + " bits, radius-2 recall " + exactRecall / QUERIES + ", top-" + TOP_K
                + " recall " + String.format("%.3f", topRecall / QUERIES) + ", two-band probes "
                + twoBand.probes + " vs four-band " + fourBand.probes);
        System.out.println("cost: " + accounting.probes + " bucket lookups and "
                + String.format("%.1f", averageCandidates) + " candidates per query against " + ITEMS
                + " codes scanned by brute force");
        System.out.println("PASS");
    }
}
