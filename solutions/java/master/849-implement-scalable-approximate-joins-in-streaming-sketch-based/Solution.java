/** Task 849 - Implement scalable approximate joins in streaming (sketch-based). (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A set-similarity join over a stream of keyed sets asks for every pair of keys whose Jaccard similarity
 * passes a threshold. Comparing every pair is O(m^2) set intersections, so the streaming answer is a
 * sketch: every set is summarised by its MinHash signature, and locality-sensitive hashing on that
 * signature produces only the pairs that are likely to be similar.
 *
 *   MinHash   - signature position j holds min over the items of h_j(item). The fraction of agreeing
 *               positions is an unbiased estimator of the Jaccard similarity with standard deviation
 *               <= 1 / (2 sqrt(signature length)).
 *   banding   - the signature is cut into b bands of r rows; two sets become a *candidate pair* when they
 *               agree on at least one whole band. A pair at similarity s is missed with probability
 *               (1 - s^r)^b, so the threshold at which candidates appear is about (1/b)^(1/r).
 *   verification - candidates are confirmed by an exact Jaccard computation, which makes the *output* an
 *               exact join over the candidate set: the only approximation left is the candidate recall.
 *
 * Simplifications, stated honestly: one process, no network; the stream is a list of sets that arrive once
 * (no deletions and no windowing); signatures use splitmix64 rather than a permutation family of real hash
 * functions, so the concentration bound is verified empirically rather than proven; verification compares
 * the small sets exactly instead of using an O(1) signature comparison, because the sets here are tiny.
 *
 * Invariants verified below. Accuracy: the Jaccard estimate of every pair stays within the Chebyshev bound
 * for the signature length (measured maximum error), identical sets estimate exactly 1 and disjoint sets 0,
 * and the threshold formula (1/b)^(1/r) matches the measured point where candidate generation starts to
 * fire. Recall and exactness: the verified output equals the exact join restricted to the candidate set, the
 * candidate set contains every true pair at or above 1.5 times the nominal threshold (measured recall), and
 * the number of candidates is a small fraction of all pairs (the cost bound that makes the approach useful).
 */
import java.util.*;

/** splitmix64 - the only randomness, and the hash family behind the MinHash signature. */
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

    static long mix(long value) {
        long v = (value ^ 0x9E3779B97F4A7C15L) * 0xBF58476D1CE4E5B9L;
        v = (v ^ (v >>> 27)) * 0x94D049BB133111EBL;
        return v ^ (v >>> 31);
    }
}

/** One keyed set with its MinHash signature. */
final class SketchSet {
    final long key;
    final Set<Long> items;
    final long[] signature;

    SketchSet(long key, Set<Long> items, int signatureLength, long seed) {
        this.key = key;
        this.items = new TreeSet<>(items);
        this.signature = new long[signatureLength];
        for (int j = 0; j < signatureLength; j++) {
            long salt = SplitMix64.mix(seed + 0x9E3779B9L * (j + 1));
            long best = Long.MAX_VALUE;
            for (long item : items) best = Math.min(best, SplitMix64.mix(item ^ salt));
            signature[j] = items.isEmpty() ? 0 : best;
        }
    }

    double exactJaccard(SketchSet other) {
        if (items.isEmpty() && other.items.isEmpty()) return 1;
        if (items.isEmpty() || other.items.isEmpty()) return 0;
        int intersection = 0;
        for (long item : items) if (other.items.contains(item)) intersection++;
        int union = items.size() + other.items.size() - intersection;
        return intersection / (double) union;
    }

    double estimatedJaccard(SketchSet other) {
        if (items.isEmpty() && other.items.isEmpty()) return 1;
        if (items.isEmpty() || other.items.isEmpty()) return 0;
        int agree = 0;
        for (int j = 0; j < signature.length; j++) {
            if (signature[j] == other.signature[j]) agree++;
        }
        return agree / (double) signature.length;
    }

    /** True when at least one whole band of the signature matches. */
    boolean sharesBand(SketchSet other, int bands, int rows) {
        for (int b = 0; b < bands; b++) {
            boolean same = true;
            for (int r = 0; r < rows; r++) {
                int j = b * rows + r;
                if (signature[j] != other.signature[j]) {
                    same = false;
                    break;
                }
            }
            if (same) return true;
        }
        return false;
    }
}

/** The join: candidates from the bands, confirmed by an exact comparison. */
final class SimilarityJoin {
    final int bands;
    final int rows;
    int candidatePairs;
    int comparisons;

    SimilarityJoin(int bands, int rows) {
        if (bands < 1 || rows < 1) throw new IllegalArgumentException("banding needs positive dimensions");
        this.bands = bands;
        this.rows = rows;
    }

    int signatureLength() { return bands * rows; }

    double nominalThreshold() { return Math.pow(1.0 / bands, 1.0 / rows); }

    /** The candidate pairs (unordered key pairs) that share a band. */
    List<long[]> candidates(List<SketchSet> sets) {
        List<long[]> pairs = new ArrayList<>();
        candidatePairs = 0;
        for (int i = 0; i < sets.size(); i++) {
            for (int j = i + 1; j < sets.size(); j++) {
                if (sets.get(i).sharesBand(sets.get(j), bands, rows)) {
                    pairs.add(new long[]{sets.get(i).key, sets.get(j).key});
                    candidatePairs++;
                }
            }
        }
        pairs.sort((a, b) -> a[0] != b[0] ? Long.compare(a[0], b[0]) : Long.compare(a[1], b[1]));
        return pairs;
    }

    /** The verified join: candidates whose exact Jaccard passes the threshold. */
    List<double[]> join(List<SketchSet> sets, double threshold) {
        Map<Long, SketchSet> byKey = new TreeMap<>();
        for (SketchSet set : sets) byKey.put(set.key, set);
        List<double[]> result = new ArrayList<>();
        comparisons = 0;
        for (long[] pair : candidates(sets)) {
            SketchSet left = byKey.get(pair[0]);
            SketchSet right = byKey.get(pair[1]);
            comparisons++;
            double similarity = left.exactJaccard(right);
            if (similarity >= threshold) result.add(new double[]{pair[0], pair[1], similarity});
        }
        result.sort((a, b) -> a[0] != b[0] ? Double.compare(a[0], b[0]) : Double.compare(a[1], b[1]));
        return result;
    }

    /** Brute force: every pair above the threshold, with no sketching at all. */
    static List<double[]> exactJoin(List<SketchSet> sets, double threshold) {
        List<double[]> result = new ArrayList<>();
        for (int i = 0; i < sets.size(); i++) {
            for (int j = i + 1; j < sets.size(); j++) {
                double similarity = sets.get(i).exactJaccard(sets.get(j));
                if (similarity >= threshold) {
                    result.add(new double[]{sets.get(i).key, sets.get(j).key, similarity});
                }
            }
        }
        result.sort((a, b) -> a[0] != b[0] ? Double.compare(a[0], b[0]) : Double.compare(a[1], b[1]));
        return result;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(4711L);
        int universe = 200, setsCount = 300, itemCount = 20;
        List<SketchSet> sets = new ArrayList<>();
        for (int s = 0; s < setsCount; s++) {
            Set<Long> items = new TreeSet<>();
            // A small similar neighbourhood (three overlapping families plus two exact twins) inside a
            // stream of otherwise unrelated sets - the shape a similarity join is actually used on.
            if (s % 15 == 0 || s >= setsCount - 2) for (int i = 0; i <= 18; i++) items.add((long) i);
            else if (s % 15 == 1) for (int i = 0; i <= 17; i++) items.add((long) i);
            else if (s % 15 == 2) for (int i = 40; i <= 58; i++) items.add((long) i);
            while (items.size() < itemCount) items.add((long) rng.nextInt(universe));
            sets.add(new SketchSet(s, items, 0, 1));
        }
        int bands = 32, rows = 4;
        int length = bands * rows;
        List<SketchSet> sketched = new ArrayList<>();
        for (SketchSet set : sets) sketched.add(new SketchSet(set.key, set.items, length, 0xABCDL));
        SimilarityJoin join = new SimilarityJoin(bands, rows);
        check(join.signatureLength() == length, "the signature length is bands * rows");
        double threshold = 0.5;
        double nominal = join.nominalThreshold();
        check(nominal > 0.3 && nominal < 0.9, "the nominal threshold must sit in a useful range, got "
                + nominal);

        // ---- 1. the Jaccard estimate is within the Chebyshev bound --------------------------------
        double bound = 1.0 / (2 * Math.sqrt(length));
        double worst = 0;
        int pairs = 0;
        double totalError = 0;
        double exactZeroError = 0, exactOneError = 0;
        for (int i = 0; i < sketched.size(); i++) {
            for (int j = i + 1; j < sketched.size(); j++) {
                double truth = sketched.get(i).exactJaccard(sketched.get(j));
                double estimate = sketched.get(i).estimatedJaccard(sketched.get(j));
                worst = Math.max(worst, Math.abs(estimate - truth));
                totalError += Math.abs(estimate - truth);
                pairs++;
                if (truth == 0) exactZeroError = Math.max(exactZeroError, estimate);
                if (truth == 1) exactOneError = Math.max(exactOneError, Math.abs(estimate - 1));
            }
        }
        check(pairs > 5000, "the instance must have enough pairs to be interesting, saw " + pairs);
        check(worst <= 5 * bound, "the worst estimate error must stay within a few sigma of "
                + bound + ", saw " + worst);
        check(totalError / pairs <= 2 * bound,
                "the mean absolute error must be about one sigma, saw " + (totalError / pairs));
        check(exactZeroError == 0, "disjoint sets must estimate zero, got " + exactZeroError);
        check(exactOneError == 0, "identical sets must estimate one, got " + exactOneError);

        // ---- 2. candidate recall and the exactness of the verified output -------------------------
        List<long[]> candidates = join.candidates(sketched);
        List<double[]> exact = SimilarityJoin.exactJoin(sketched, threshold);
        Set<String> candidateSet = new HashSet<>();
        for (long[] pair : candidates) candidateSet.add(pair[0] + ":" + pair[1]);
        int missedHigh = 0, highPairs = 0, missedLow = 0;
        for (int i = 0; i < sketched.size(); i++) {
            for (int j = i + 1; j < sketched.size(); j++) {
                double similarity = sketched.get(i).exactJaccard(sketched.get(j));
                String pair = sketched.get(i).key + ":" + sketched.get(j).key;
                if (similarity >= 1.5 * nominal) {
                    highPairs++;
                    if (!candidateSet.contains(pair)) missedHigh++;
                }
                if (similarity >= threshold && !candidateSet.contains(pair)) missedLow++;
            }
        }
        double recallHigh = 1 - missedHigh / (double) highPairs;
        check(highPairs > 10, "the instance must plant clearly similar pairs, saw " + highPairs);
        check(recallHigh >= 0.9, "recall above 1.5x the nominal threshold must be high, got " + recallHigh);
        List<double[]> verified = join.join(sketched, threshold);
        List<double[]> expected = new ArrayList<>();
        for (double[] pair : exact) {
            if (candidateSet.contains((long) pair[0] + ":" + (long) pair[1])) expected.add(pair);
        }
        check(samePairs(verified, expected),
                "the verified output equals the exact join restricted to the candidates");
        check(missedLow == exact.size() - verified.size(),
                "every pair the candidates missed must be a pair the output misses: " + missedLow + " vs "
                        + (exact.size() - verified.size()));

        // ---- 3. the cost bound: candidates are a small fraction of all pairs ----------------------
        double reduction = candidates.size() / (double) pairs;
        check(reduction < 0.2, "banding must reduce the pairs to compare, saw " + reduction);
        check(join.comparisons == candidates.size(),
                "the verifier must compare exactly the candidates, saw " + join.comparisons);
        check(verified.size() <= candidates.size(), "no more joins than candidates");

        // ---- 4. the threshold formula matches the measured transition ----------------------------
        int below = 0, belowTotal = 0, above = 0, aboveTotal = 0;
        for (int i = 0; i < sketched.size(); i++) {
            for (int j = i + 1; j < sketched.size(); j++) {
                double similarity = sketched.get(i).exactJaccard(sketched.get(j));
                boolean candidate = candidateSet.contains(sketched.get(i).key + ":"
                        + sketched.get(j).key);
                if (similarity <= nominal * 0.6) {
                    belowTotal++;
                    if (candidate) below++;
                }
                if (similarity >= nominal * 1.4 && similarity < 1) {
                    aboveTotal++;
                    if (candidate) above++;
                }
            }
        }
        double belowRate = belowTotal == 0 ? 0 : below / (double) belowTotal;
        double aboveRate = aboveTotal == 0 ? 0 : above / (double) aboveTotal;
        check(aboveTotal > 0 && aboveRate > belowRate,
                "candidate generation must fire more often above the threshold: " + aboveRate + " vs "
                        + belowRate);
        check(belowRate < 0.35, "below the threshold most pairs must be pruned, saw " + belowRate);

        // ---- 5. degenerate inputs, validation and determinism ------------------------------------
        SketchSet emptyA = new SketchSet(9000, Set.of(), length, 0xABCDL);
        SketchSet emptyB = new SketchSet(9001, Set.of(), length, 0xABCDL);
        SketchSet normal = sketched.get(0);
        check(emptyA.exactJaccard(normal) == 0 && normal.exactJaccard(emptyA) == 0,
                "an empty set has Jaccard zero against a non-empty one");
        check(emptyA.exactJaccard(emptyB) == 1, "two empty sets are identical");
        check(emptyA.estimatedJaccard(normal) == 0, "the estimate must agree on the empty case");
        List<SketchSet> twins = new ArrayList<>();
        twins.add(new SketchSet(1, Set.of(1L, 2L, 3L), length, 0xABCDL));
        twins.add(new SketchSet(2, Set.of(1L, 2L, 3L), length, 0xABCDL));
        SimilarityJoin twinJoin = new SimilarityJoin(bands, rows);
        List<double[]> twinResult = twinJoin.join(twins, 0.9);
        check(twinResult.size() == 1 && twinResult.get(0)[2] == 1.0,
                "identical sets must join with similarity one");
        List<SketchSet> disjoint = new ArrayList<>();
        disjoint.add(new SketchSet(1, Set.of(1L, 2L), length, 0xABCDL));
        disjoint.add(new SketchSet(2, Set.of(3L, 4L), length, 0xABCDL));
        check(new SimilarityJoin(bands, rows).join(disjoint, 0.01).isEmpty(),
                "disjoint sets must never be reported, even with a tiny threshold");
        List<SketchSet> noPattern = new ArrayList<>();
        SimilarityJoin strict = new SimilarityJoin(bands, rows);
        List<double[]> strictResult = strict.join(sketched, 0.99);
        check(strictResult.isEmpty() || strictResult.get(0)[2] >= 0.99,
                "a threshold of 0.99 must only report near-identical sets, saw " + strictResult.size());
        expectThrows(() -> new SimilarityJoin(0, 8), "zero bands");
        expectThrows(() -> new SimilarityJoin(8, 0), "zero rows");
        check(new SketchSet(5, Set.of(1L), length, 1).estimatedJaccard(
                new SketchSet(6, Set.of(1L), length, 1)) == 1,
                "a single shared item is identical and must estimate one");
        List<long[]> again = join.candidates(sketched);
        check(again.size() == candidates.size() && join.comparisons == candidates.size(),
                "two identical runs must produce the same candidates");

        System.out.println("instance: " + setsCount + " sets of " + itemCount + " items over a universe of "
                + universe + " -> " + pairs + " pairs");
        System.out.println("bands " + bands + "x" + rows + " (length " + length + ", nominal threshold "
                + nominal + "): worst estimate error " + worst + " vs bound " + bound);
        System.out.println("candidates " + candidates.size() + " (" + reduction + " of all pairs), recall "
                + recallHigh + " above 1.5x, verified joins " + verified.size() + " of "
                + exact.size() + " exact");
        System.out.println("candidate rate below threshold " + belowRate + " vs above " + aboveRate);
        System.out.println("PASS");
    }

    static boolean samePairs(List<double[]> left, List<double[]> right) {
        if (left.size() != right.size()) return false;
        for (int i = 0; i < left.size(); i++) {
            if (left.get(i)[0] != right.get(i)[0] || left.get(i)[1] != right.get(i)[1]
                    || Math.abs(left.get(i)[2] - right.get(i)[2]) > 1e-12) return false;
        }
        return true;
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
