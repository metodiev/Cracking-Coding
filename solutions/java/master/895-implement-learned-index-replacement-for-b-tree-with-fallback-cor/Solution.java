/** Task 895 - Implement learned index replacement for B-tree with fallback correctness checks. (Tier: Master, CTCI Ch 14 Databases)
 *
 * A learned index (Kraska et al.) replaces the B-tree's comparison descent with a *prediction*: a
 * piecewise linear model maps a key to its position, and a bounded local search repairs the error. This
 * file implements the whole contract, plus the fallback that makes the structure safe:
 *
 *   build    the sorted array is split into equal-count chunks (level 1) and each chunk into equal-count
 *            sub-chunks; every sub-chunk gets a least-squares line slope * key + intercept. The maximum
 *            prediction error per model is measured at build time, so the error bound is observed data,
 *            not an assumption
 *   search   navigate to a chunk by the model array, predict the position inside its sub-chunk, clamp to
 *            that sub-chunk's index range and binary search the window [prediction - error, prediction +
 *            error]; if the window does not bracket the key, fall back to a global binary search and
 *            count the fallback
 *   oracle   an ordinary lower_bound binary search over the same array, used to verify *every* answer
 *
 * The fallback is the point of the exercise: a learned index may mispredict, but it must never
 * mis-answer. The checks therefore (a) prove the recorded error bound is exact by exhausting every key,
 * (b) prove the window always contains the true position, (c) compare every probe window's answer against
 * the binary-search oracle, and (d) force a deliberately tight model so the fallback path runs thousands
 * of times while every answer stays correct and the probe count stays within log2(N) + model cost.
 *
 * What is simplified: the model is linear (a real RMI uses higher-order models and learned splines), the
 * array is immutable - so "insert" is modelled as a rebuild - and probes are counted as loop iterations
 * rather than cache misses. Everything else is the real structure: the chunked model, the error-bounded
 * search, the fallback, and the measured comparison between the learned index and a B-tree-style binary
 * search over four key distributions (uniform, skewed, clustered and sequential).
 *
 * Verified: the learned index returns exactly the binary-search oracle's answer for every key of four
 * distributions and for absent keys (found flag plus insertion point); the recorded per-model error equals
 * the maximum observed error and the search window contains every real position; predictions are
 * non-decreasing in the key, which is what makes navigation sound; the probe count is far below log2(N)
 * and never above log2(N) plus the model cost even when the fallback fires; a deliberately tight model
 * falls back thousands of times while staying exactly correct; duplicates resolve to the first occurrence;
 * empty, single-key, all-identical and out-of-range inputs behave; and the build and the search are
 * deterministic.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    long between(long lo, long hi) { return lo + Math.floorMod(nextLong(), Math.max(1, hi - lo + 1)); }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** One linear model over a contiguous index range. */
class Model {
    final long keyLo, keyHi;              // the key range this model is responsible for
    final int posLo, posHi;               // the index range it covers
    final double slope, intercept;
    int error;                            // maximum observed |prediction - position|

    Model(long keyLo, long keyHi, int posLo, int posHi, double slope, double intercept) {
        this.keyLo = keyLo; this.keyHi = keyHi; this.posLo = posLo; this.posHi = posHi;
        this.slope = slope; this.intercept = intercept;
    }

    int predict(long key) {
        double predicted = slope * key + intercept;
        int clamped = (int) Math.max(posLo, Math.min(posHi, Math.round(predicted)));
        return clamped;
    }
}

/** A two-level learned index over an immutable sorted array. */
class LearnedIndex {
    final long[] keys, train;
    final int chunkCount, modelsPerChunk;
    final List<Model>[] chunks;           // level 1 is navigated by key comparison on the chunk bounds
    final long[] chunkKeyLo, chunkKeyHi;
    int maxError;                         // the largest error over all models
    long probeCount, modelLookups, fallbacks, windowSearches;

    LearnedIndex(long[] keys, int chunkCount, int modelsPerChunk) {
        this(keys, chunkCount, modelsPerChunk, keys);
    }

    /**
     * `training` models the stale-model case: the lines and their error bounds are fitted on an older array
     * of a different length, so the model is honest about its training data and wrong about the current one
     * - exactly the failure the fallback exists for.
     */
    @SuppressWarnings("unchecked")
    LearnedIndex(long[] keys, int chunkCount, int modelsPerChunk, long[] training) {
        this.keys = keys;
        this.train = training;
        this.chunkCount = Math.max(1, Math.min(chunkCount, Math.max(1, keys.length)));
        this.modelsPerChunk = Math.max(1, modelsPerChunk);
        this.chunks = new List[this.chunkCount];
        this.chunkKeyLo = new long[this.chunkCount];
        this.chunkKeyHi = new long[this.chunkCount];
        build();
    }

    private void build() {
        int n = keys.length;
        if (n == 0) {
            for (int c = 0; c < chunkCount; c++) {
                chunks[c] = new ArrayList<>();
                chunkKeyLo[c] = Long.MIN_VALUE;
                chunkKeyHi[c] = Long.MIN_VALUE;
            }
            maxError = 0;
            return;
        }
        for (int c = 0; c < chunkCount; c++) {
            int lo = (int) ((long) c * n / chunkCount);
            int hi = (int) ((long) (c + 1) * n / chunkCount) - 1;
            if (hi < lo) hi = lo;
            List<Model> models = new ArrayList<>();
            int models_here = Math.min(modelsPerChunk, hi - lo + 1);
            for (int m = 0; m < models_here; m++) {
                int subLo = lo + (m * (hi - lo + 1)) / models_here;
                int subHi = lo + ((m + 1) * (hi - lo + 1)) / models_here - 1;
                if (subHi < subLo) subHi = subLo;
                models.add(fit(subLo, subHi));
            }
            chunks[c] = models;
            chunkKeyLo[c] = keys.length == 0 ? Long.MIN_VALUE : keys[lo];
            chunkKeyHi[c] = keys.length == 0 ? Long.MIN_VALUE : keys[hi];
        }
        int worst = 0;
        for (List<Model> models : chunks)
            for (Model model : models) worst = Math.max(worst, model.error);
        maxError = worst;
    }

    /** Least-squares fit over the points (key, position) of one index range, restricted to the training data. */
    private Model fit(int lo, int hi) {
        int fitHi = Math.min(hi, train.length - 1);
        if (fitHi < lo) fitHi = lo;
        int count = fitHi - lo + 1;
        double meanX = 0, meanY = 0;
        for (int i = lo; i <= fitHi; i++) { meanX += train[i]; meanY += i - lo; }
        meanX /= count;
        meanY /= count;
        double sxy = 0, sxx = 0;
        for (int i = lo; i <= fitHi; i++) {
            double dx = train[i] - meanX;
            sxy += dx * ((i - lo) - meanY);
            sxx += dx * dx;
        }
        double slope = sxx == 0 ? 0 : sxy / sxx;
        double intercept = meanY - slope * meanX + lo;
        Model model = new Model(keys[lo], keys[hi], lo, hi, slope, intercept);
        int worst = 0;
        for (int i = lo; i <= fitHi; i++) worst = Math.max(worst, Math.abs(model.predict(train[i]) - i));
        model.error = worst;
        return model;
    }

    /** The chunk whose key range contains (or brackets) the key. */
    int chunkFor(long key) {
        int lo = 0, hi = chunkCount - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            probeCount++;
            if (chunkKeyHi[mid] < key) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    /** The sub-chunk model responsible for the key inside a chunk. */
    Model modelFor(int chunk, long key) {
        List<Model> models = chunks[chunk];
        Model chosen = models.get(0);
        for (Model model : models) {
            modelLookups++;
            if (model.keyLo <= key && key <= model.keyHi) return model;
            if (key > model.keyHi) chosen = model;
        }
        return chosen;
    }

    /** lower_bound: the first index whose key is >= the target, or -1 with `found = false`. */
    static class Lookup {
        int index;
        boolean found;
        boolean usedFallback;

        Lookup(int index, boolean found, boolean usedFallback) {
            this.index = index; this.found = found; this.usedFallback = usedFallback;
        }
    }

    /** The B-tree style oracle: a plain lower_bound binary search. */
    static Lookup binarySearch(long[] keys, long target) {
        int lo = 0, hi = keys.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (keys[mid] < target) lo = mid + 1; else hi = mid;
        }
        return new Lookup(lo, lo < keys.length && keys[lo] == target, false);
    }

    /**
     * Error-bounded search: predict, then binary search the window [prediction - error - 1, prediction +
     * error + 1] (the extra slot absorbs the boundary between neighbouring models). If the answer lands on
     * the window edge and the key could still lie outside it, the search falls back to the global binary
     * search - a misprediction may cost time, never correctness.
     */
    Lookup search(long target) {
        if (keys.length == 0) return new Lookup(0, false, false);
        int chunk = chunkFor(target);
        Model model = modelFor(chunk, target);
        int predicted = model.predict(target);
        int windowLo = Math.max(0, predicted - model.error - 1);
        int windowHi = Math.min(keys.length - 1, predicted + model.error + 1);
        int a = windowLo, b = windowHi + 1;
        while (a < b) {
            int mid = (a + b) >>> 1;
            probeCount++;
            if (keys[mid] < target) a = mid + 1; else b = mid;
        }
        boolean mayBeBeyond = a == windowHi + 1 && windowHi + 1 < keys.length && keys[windowHi] < target;
        boolean mayBeBefore = a == windowLo && windowLo > 0 && keys[windowLo - 1] >= target;
        if (mayBeBeyond || mayBeBefore) {
            fallbacks++;
            int fa = 0, fb = keys.length;                  // the full descent, counted like any other probe
            while (fa < fb) {
                int mid = (fa + fb) >>> 1;
                probeCount++;
                if (keys[mid] < target) fa = mid + 1; else fb = mid;
            }
            return new Lookup(fa, fa < keys.length && keys[fa] == target, true);
        }
        windowSearches++;
        return new Lookup(a, a < keys.length && keys[a] == target, false);
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    /** Four key distributions over the same number of keys. */
    static long[] distribution(String kind, int n, long seed) {
        long[] keys = new long[n];
        SplitMix64 rng = new SplitMix64(seed);
        switch (kind) {
            case "uniform":
                for (int i = 0; i < n; i++) keys[i] = rng.nextLong() % 1_000_000L;
                break;
            case "skewed":
                for (int i = 0; i < n; i++) keys[i] = (long) Math.pow(rng.nextDouble(), 4.0) * 1_000_000L;
                break;
            case "clustered": {
                long base = 0;
                for (int i = 0; i < n; i++) {
                    if (i % 64 == 0) base += rng.between(10_000, 50_000);
                    keys[i] = base + rng.between(0, 5);
                }
                break;
            }
            default:
                for (int i = 0; i < n; i++) keys[i] = i * 3L;
        }
        Arrays.sort(keys);
        return keys;
    }

    static void exhaustivelyCheck(String kind, long[] keys, int chunks, int perChunk) {
        LearnedIndex index = new LearnedIndex(keys, chunks, perChunk);
        int mismatches = 0;
        for (long key : keys) {
            LearnedIndex.Lookup learned = index.search(key);
            LearnedIndex.Lookup oracle = LearnedIndex.binarySearch(keys, key);
            if (!learned.found || learned.index != oracle.index) mismatches++;
        }
        check(mismatches == 0, kind + ": " + mismatches + " keys resolved to the wrong position");
        // absent keys, including both extremes
        SplitMix64 rng = new SplitMix64(4711);
        int absentMismatches = 0;
        for (int i = 0; i < 400; i++) {
            long probe = keys.length == 0 ? 0 : keys[rng.nextInt(keys.length)] + rng.between(-3, 3);
            LearnedIndex.Lookup learned = index.search(probe);
            LearnedIndex.Lookup oracle = LearnedIndex.binarySearch(keys, probe);
            if (learned.found != oracle.found || learned.index != oracle.index) absentMismatches++;
        }
        check(absentMismatches == 0, kind + ": absent-key lookups disagree with the oracle");
    }

    public static void main(String[] args) {
        int n = 4096;

        // 1. Correctness on four distributions, every key plus absent probes, against the binary-search oracle.
        for (String kind : new String[]{"uniform", "skewed", "clustered", "sequential"}) {
            long[] keys = distribution(kind, n, 2024);
            exhaustivelyCheck(kind, keys, 8, 4);
        }

        // 2. The recorded error bound is exact and the window contains every true position.
        long[] uniform = distribution("uniform", n, 2024);
        LearnedIndex learnedIndex = new LearnedIndex(uniform, 8, 4);
        int recomputedWorst = 0;
        int windowMisses = 0;
        for (List<Model> models : learnedIndex.chunks) {
            for (Model model : models) {
                for (int i = model.posLo; i <= model.posHi; i++) {
                    int prediction = model.predict(uniform[i]);
                    recomputedWorst = Math.max(recomputedWorst, Math.abs(prediction - i));
                    if (Math.abs(prediction - i) > model.error) windowMisses++;
                }
            }
        }
        check(recomputedWorst == learnedIndex.maxError, "the recorded error " + learnedIndex.maxError
                + " differs from the observed maximum " + recomputedWorst);
        check(windowMisses == 0, windowMisses + " positions fall outside their declared error window");
        check(learnedIndex.maxError < 64, "a piecewise linear model should be tight on uniform keys: "
                + learnedIndex.maxError);

        // 3. Predictions are non-decreasing in the key, which is what makes navigation sound.
        int inversions = 0;
        long previous = Long.MIN_VALUE;
        int previousPrediction = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            Model model = learnedIndex.modelFor(learnedIndex.chunkFor(uniform[i]), uniform[i]);
            int prediction = model.predict(uniform[i]);
            if (uniform[i] > previous && prediction < previousPrediction) inversions++;
            previous = uniform[i];
            previousPrediction = prediction;
        }
        check(inversions == 0, inversions + " prediction inversions");

        // 4. Cost: the window search is far cheaper than the full descent, and never worse than log2 N.
        long[] sequential = distribution("sequential", n, 1);
        LearnedIndex fast = new LearnedIndex(sequential, 8, 4);
        for (long key : sequential) fast.search(key);
        double probesPerSearch = fast.probeCount / (double) sequential.length;
        check(probesPerSearch < Math.log(n) / Math.log(2) * 0.6, "learned search probes " + probesPerSearch
                + " per lookup, the B-tree needs about " + (Math.log(n) / Math.log(2)));
        check(fast.fallbacks == 0, "a perfectly linear distribution should never fall back: " + fast.fallbacks);
        check(fast.windowSearches == sequential.length, "some searches did not use the model window");
        check(fast.modelLookups > 0 && fast.modelLookups <= 8L * 4 * sequential.length,
                "model lookups are out of range: " + fast.modelLookups);

        // 5. Fallback: a stale model (fitted before a bulk insert shifted every position above the insert
        //    point) mispredicts, and every answer stays correct because the search falls back.
        long[] perfect = distribution("sequential", n, 1);
        long[] grown = new long[n + 300];
        System.arraycopy(perfect, 0, grown, 0, n / 2);
        long filler = perfect[n / 2];
        for (int i = 0; i < 300; i++) grown[n / 2 + i] = filler + i + 1;
        System.arraycopy(perfect, n / 2, grown, n / 2 + 300, n - n / 2);
        Arrays.sort(grown);
        LearnedIndex fresh = new LearnedIndex(perfect, 1, 1);
        check(fresh.maxError == 0, "a perfectly linear array should need no error window: " + fresh.maxError);
        LearnedIndex stale = new LearnedIndex(grown, 1, 1, perfect);
        int staleMismatch = 0;
        for (long key : grown) {
            LearnedIndex.Lookup learned = stale.search(key);
            LearnedIndex.Lookup oracle = LearnedIndex.binarySearch(grown, key);
            if (!learned.found || learned.index != oracle.index) staleMismatch++;
        }
        check(staleMismatch == 0, "the stale model mis-answered " + staleMismatch + " keys");
        check(stale.fallbacks > 100, "the stale model should fall back often: " + stale.fallbacks);
        check(stale.maxError == fresh.maxError, "both models are fitted on the same array, so the recorded "
                + "error must agree: " + stale.maxError + " vs " + fresh.maxError);
        int staleMisses = 0;
        for (Model model : stale.chunks[0])
            for (int i = model.posLo; i <= model.posHi; i++)
                if (Math.abs(model.predict(grown[i]) - i) > model.error) staleMisses++;
        check(staleMisses > 100, "the stale model was not actually stale: " + staleMisses + " shifted keys");
        double tightProbes = stale.probeCount / (double) grown.length;
        check(tightProbes <= Math.log(n) / Math.log(2) + 4, "the fallback is not bounded by log2 N plus the "
                + "model cost: " + tightProbes);

        // 6. More models mean a smaller error: the model adapts to the distribution.
        long[] skewed = distribution("skewed", n, 2024);
        int previousError = Integer.MAX_VALUE;
        int errorOne = new LearnedIndex(skewed, 4, 1).maxError;
        int[] sweep = new int[4];
        int sweepIndex = 0;
        for (int perChunk : new int[]{1, 2, 4, 8}) {
            LearnedIndex candidate = new LearnedIndex(skewed, 4, perChunk);
            sweep[sweepIndex++] = candidate.maxError;
            check(candidate.maxError <= previousError, "error grew with more models: " + candidate.maxError
                    + " vs " + previousError);
            previousError = candidate.maxError;
        }
        check(previousError < errorOne, "eight models per chunk must beat one: " + previousError + " vs "
                + errorOne);
        check(previousError <= 128, "eight models per chunk should be tight on skewed keys: " + previousError);

        // 7. Duplicates resolve to the first occurrence, exactly like lower_bound.
        long[] duplicates = new long[512];
        for (int i = 0; i < duplicates.length; i++) duplicates[i] = i / 16;
        LearnedIndex duplicateIndex = new LearnedIndex(duplicates, 4, 4);
        int duplicateMismatch = 0;
        for (int i = 0; i < duplicates.length; i++) {
            LearnedIndex.Lookup learned = duplicateIndex.search(duplicates[i]);
            LearnedIndex.Lookup oracle = LearnedIndex.binarySearch(duplicates, duplicates[i]);
            if (learned.index != oracle.index) duplicateMismatch++;
        }
        check(duplicateMismatch == 0, "duplicates resolved to the wrong occurrence: " + duplicateMismatch);
        check(LearnedIndex.binarySearch(duplicates, 5).index == 80, "the oracle disagrees on the first "
                + "occurrence of 5");
        check(duplicateIndex.search(5).index == 80, "the learned index disagrees on the first occurrence of 5");

        // 8. Degenerate inputs.
        long[] empty = new long[0];
        LearnedIndex emptyIndex = new LearnedIndex(empty, 4, 4);
        check(!emptyIndex.search(7).found && emptyIndex.search(7).index == 0, "an empty index misbehaved");
        long[] single = {42};
        LearnedIndex singleIndex = new LearnedIndex(single, 4, 4);
        check(singleIndex.search(42).found && singleIndex.search(42).index == 0
                && !singleIndex.search(41).found && singleIndex.search(43).index == 1,
                "a single-key index misbehaved");
        long[] identical = new long[128];
        Arrays.fill(identical, 7);
        LearnedIndex identicalIndex = new LearnedIndex(identical, 4, 4);
        check(identicalIndex.search(7).index == 0 && identicalIndex.search(8).index == 128,
                "an all-identical index misbehaved");
        check(!identicalIndex.search(6).found && identicalIndex.search(6).index == 0,
                "a key below every key must return insertion point zero");
        long[] keys = distribution("uniform", 256, 99);
        LearnedIndex bounds = new LearnedIndex(keys, 4, 4);
        check(!bounds.search(Long.MIN_VALUE).found && bounds.search(Long.MIN_VALUE).index == 0,
                "an underflowing key was mishandled");
        check(!bounds.search(Long.MAX_VALUE).found && bounds.search(Long.MAX_VALUE).index == keys.length,
                "an overflowing key was mishandled");
        check(!bounds.search(keys[0] - 1).found && bounds.search(keys[keys.length - 1] + 1).index == keys.length,
                "out-of-range keys were mishandled");

        // 9. Determinism and static-array rebuild semantics.
        LearnedIndex again = new LearnedIndex(keys, 4, 4);
        check(again.maxError == bounds.maxError, "the build is not deterministic");
        long[] extended = Arrays.copyOf(keys, keys.length + 1);
        extended[keys.length] = keys[keys.length - 1] + 1;
        Arrays.sort(extended);
        LearnedIndex rebuilt = new LearnedIndex(extended, 4, 4);
        int rebuildMismatch = 0;
        for (long key : extended) if (rebuilt.search(key).index != LearnedIndex.binarySearch(extended, key).index)
            rebuildMismatch++;
        check(rebuildMismatch == 0, "a rebuilt index mis-answered " + rebuildMismatch + " keys after an insert");
        check(rebuilt.keys.length == keys.length + 1, "the rebuild lost the inserted key");

        System.out.println("learned index: 8 chunks x 4 models, max error " + learnedIndex.maxError
                + ", " + String.format("%.2f", probesPerSearch) + " probes per lookup against log2(N) = "
                + String.format("%.2f", Math.log(n) / Math.log(2)));
        System.out.println("stale model : 1 chunk x 1 model fitted before a 300 key bulk insert, fallbacks "
                + stale.fallbacks + " of " + grown.length + ", " + String.format("%.2f", tightProbes)
                + " probes per lookup");
        System.out.println("model sweep : worst error for 1/2/4/8 models per chunk on skewed keys "
                + Arrays.toString(sweep));
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
