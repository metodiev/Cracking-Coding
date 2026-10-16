/** Task 951 - Implement end-to-end reproducible benchmarking harness for all exercises in repo.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory benchmarking harness. Wall-clock timing is replaced by a deterministic operation count
 *  (each workload returns how many elementary steps it executed) because that is what makes a whole
 *  run reproducible inside one process; everything else is faithful to a timing harness: repetitions,
 *  min/median/max aggregation, an environment fingerprint over the toolchain tags + seed, and a single
 *  FNV-1a digest over the sorted result rows so two runs compare with one number.
 *  Simplified: no filesystem or JSON output, no JIT warm-up, no subprocess per exercise - the registry
 *  of exercises is built in memory and the "cost" of an exercise is the counter its workload returns.
 *  Verified: exactly one row per registered exercise; identical (tags, seed) reproduce the digest
 *  bit-for-bit while changing the seed or a tag changes it; aggregates are recomputed from the raw
 *  samples by an independent insertion sort; a deliberately slowed workload is reported as a regression
 *  against the captured baseline while untouched exercises are not; an empty registry yields zero rows.
 */
import java.util.*;

public class Solution {
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
    static final long FNV_OFFSET = 0xcbf29ce484222325L; static final long FNV_PRIME = 0x100000001b3L;
    /** Sink for workload results so the counted work cannot be optimised away. */
    static long sink;
    static long splitmix64(long x) {
        long z = x + 0x9E3779B97F4A7C15L; z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL; return z ^ (z >>> 31);
    }
    static long fnvByte(long h, long v) {
        for (int i = 0; i < 8; i++) { h ^= (v >>> (i * 8)) & 0xFFL; h *= FNV_PRIME; }
        return h;
    }
    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }
    static String hex(long v) { return String.format("%016x", v); }
    interface Workload { long steps(int size, long seed); }
    /** Single pass over `size` pseudo-random values: O(n) steps. */
    static final Workload LINEAR = (size, seed) -> {
        long acc = seed, steps = 0;
        for (int i = 0; i < size; i++) { acc = splitmix64(acc + i); steps++; }
        sink = acc; return steps;
    };
    /** Every unordered pair: O(n^2) steps. */
    static final Workload PAIRWISE = (size, seed) -> {
        long acc = seed, steps = 0;
        for (int i = 0; i < size; i++)
            for (int j = i + 1; j < size; j++) { acc = splitmix64(acc + i * 31L + j); steps++; }
        sink = acc; return steps;
    };
    /** Four bucket passes: same asymptotic shape as LINEAR with a constant factor of four. */
    static final Workload BUCKETS = (size, seed) -> {
        long acc = seed, steps = 0;
        for (int pass = 0; pass < 4; pass++)
            for (int i = 0; i < size; i++) { acc = splitmix64(acc + pass * 1000003L + i); steps++; }
        sink = acc; return steps;
    };
    /** Regressed twin of LINEAR: three times the work, so it must trip the tolerance. */
    static final Workload SLOW_LINEAR = (size, seed) -> {
        long acc = seed, steps = 0;
        for (int pass = 0; pass < 3; pass++)
            for (int i = 0; i < size; i++) { acc = splitmix64(acc + pass * 7919L + i); steps++; }
        sink = acc; return steps;
    };
    static final class Spec {
        final String exerciseId; final int size; final int reps; final Workload workload;
        Spec(String exerciseId, int size, int reps, Workload workload) {
            this.exerciseId = exerciseId; this.size = size; this.reps = reps; this.workload = workload;
        }
    }
    /** Insertion sort, used by rows and by the independent digest oracle instead of Arrays.sort. */
    static void sortInPlace(long[] a) {
        for (int i = 1; i < a.length; i++) {
            long v = a[i]; int j = i - 1;
            while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
            a[j + 1] = v;
        }
    }
    static final class Row {
        final String exerciseId; final int size; final long[] samples; final long min, median, max;
        Row(String exerciseId, int size, long[] samples) {
            this.exerciseId = exerciseId; this.size = size; this.samples = samples;
            long[] sorted = samples.clone(); sortInPlace(sorted); this.min = sorted[0];
            this.max = sorted[sorted.length - 1];
            this.median = sorted[sorted.length / 2];   // upper median for even repetition counts
        }
        long contribution() {
            long h = fnvText(FNV_OFFSET, exerciseId); h = fnvByte(h, size); h = fnvByte(h, min);
            h = fnvByte(h, median); h = fnvByte(h, max); return h;
        }
    }
    static final class Report {
        final String envFingerprint; final long seed; final List<Row> rows; final long digest;
        final int regressions;
        Report(String envFingerprint, long seed, List<Row> rows, long digest, int regressions) {
            this.envFingerprint = envFingerprint; this.seed = seed; this.rows = rows;
            this.digest = digest; this.regressions = regressions;
        }
        String digestHex() { return hex(digest); }
        Row row(String exerciseId) {
            for (Row r : rows) if (r.exerciseId.equals(exerciseId)) return r;
            return null;
        }
    }
    static final class Harness {
        static final long TOLERANCE_PERCENT = 25; final long seed; final List<String> toolchain;
        final String envFingerprint; final List<Spec> specs = new ArrayList<>();
        final Map<String, Long> baseline = new TreeMap<>();
        Harness(long seed, List<String> toolchain) {
            this.seed = seed; this.toolchain = new ArrayList<>(toolchain); long h = fnvText(FNV_OFFSET, "env");
            for (String tag : this.toolchain) h = fnvText(h, tag);
            h = fnvByte(h, seed); this.envFingerprint = hex(h);
        }
        Harness register(Spec spec) { specs.add(spec); return this; }
        void captureBaseline(Report report) {
            baseline.clear();
            for (Row r : report.rows) baseline.put(r.exerciseId, r.median);
        }
        Report run() {
            List<Row> rows = new ArrayList<>();
            for (Spec s : specs) {
                long[] samples = new long[s.reps];
                for (int i = 0; i < s.reps; i++) samples[i] = s.workload.steps(s.size, seed + i * 0x9E3779B97F4A7C15L);
                rows.add(new Row(s.exerciseId, s.size, samples));
            }
            rows.sort((a, b) -> a.exerciseId.compareTo(b.exerciseId));
            long digest = fnvText(FNV_OFFSET, envFingerprint); int regressions = 0;
            for (Row r : rows) {
                digest = fnvByte(digest, r.contribution()); Long base = baseline.get(r.exerciseId);
                if (base != null && r.median * 100 > base * (100 + TOLERANCE_PERCENT)) regressions++;
            }
            return new Report(envFingerprint, seed, rows, digest, regressions);
        }
    }
    static final String SLOWED = "ex-007-windowed-sum";
    /** The repository's exercise registry; `slowFlatten` swaps one exercise for its regressed twin. */
    static List<Spec> registry(boolean slowFlatten) {
        List<Spec> out = new ArrayList<>(); out.add(new Spec("ex-001-max-subarray", 60, 5, LINEAR));
        out.add(new Spec("ex-002-pair-count", 24, 5, PAIRWISE));
        out.add(new Spec("ex-003-bucket-histogram", 40, 5, BUCKETS));
        out.add(new Spec("ex-004-word-ladder", 12, 5, PAIRWISE));
        out.add(new Spec("ex-005-lru-cache", 50, 5, LINEAR));
        out.add(new Spec("ex-006-top-k-streams", 30, 5, BUCKETS));
        out.add(new Spec(SLOWED, 45, 5, slowFlatten ? SLOW_LINEAR : LINEAR));
        out.add(new Spec("ex-008-inverted-index", 36, 5, LINEAR));
        out.add(new Spec("ex-009-skyline", 20, 5, PAIRWISE)); out.add(new Spec("ex-010-tokenizer", 64, 5, LINEAR));
        return out;
    }
    static Harness harnessFor(long seed, List<String> tags, boolean slowFlatten) {
        Harness h = new Harness(seed, tags);
        for (Spec s : registry(slowFlatten)) h.register(s);
        return h;
    }
    public static void main(String[] args) {
        List<String> tags = List.of("jdk-21.0.6", "windows-x64", "harness-1.0");
        List<String> otherOs = List.of("jdk-21.0.6", "linux-x64", "harness-1.0");
        Report first = harnessFor(20241016L, tags, false).run();
        Report rerun = harnessFor(20241016L, tags, false).run();
        Report reseeded = harnessFor(20241017L, tags, false).run();
        Report relocated = harnessFor(20241016L, otherOs, false).run();
        // 1. coverage: every registered exercise exactly once
        Set<String> ids = new TreeSet<>();
        for (Row r : first.rows) ids.add(r.exerciseId);
        check(first.rows.size() == registry(false).size() && ids.size() == first.rows.size(), "one result row per registered exercise, no duplicates");
        // 2. aggregates recomputed independently from the raw samples
        boolean aggregatesOk = true;
        for (Row r : first.rows) {
            long[] s = r.samples.clone(); sortInPlace(s);
            aggregatesOk &= r.min == s[0] && r.max == s[s.length - 1] && r.median == s[s.length / 2]
                    && r.min <= r.median && r.median <= r.max;
        }
        check(aggregatesOk, "min <= median <= max and equal to an independent recomputation");
        // 3. reproducibility: same tags and seed give a bit-identical report
        boolean rowsEqual = first.rows.size() == rerun.rows.size();
        for (int i = 0; rowsEqual && i < first.rows.size(); i++) {
            Row a = first.rows.get(i), b = rerun.rows.get(i);
            rowsEqual &= a.exerciseId.equals(b.exerciseId) && a.min == b.min && a.median == b.median && a.max == b.max;
        }
        check(rowsEqual && first.digestHex().equals(rerun.digestHex()), "identical toolchain tags and seed reproduce the run digest bit-for-bit");
        // 4. seed sensitivity
        check(!first.digestHex().equals(reseeded.digestHex()) && !first.envFingerprint.equals(reseeded.envFingerprint), "changing the seed changes the environment fingerprint and the digest");
        // 5. toolchain sensitivity
        check(!first.digestHex().equals(relocated.digestHex()), "changing an OS tag changes the digest even though the workloads are identical");
        // 6. boundary: empty registry and a single-element workload
        Report empty = new Harness(7L, tags).run(); long emptyOracle = fnvText(FNV_OFFSET, empty.envFingerprint);
        check(empty.rows.isEmpty() && empty.digest == emptyOracle && empty.regressions == 0, "empty registry produces zero rows and the digest of the bare environment");
        Harness single = new Harness(99L, tags); single.register(new Spec("ex-bounds-1", 1, 5, LINEAR));
        Row singleRow = single.run().row("ex-bounds-1");
        check(singleRow != null && singleRow.min == 1 && singleRow.median == 1 && singleRow.max == 1, "size-1 workload records exactly one step per repetition");
        // 7. regression: only the swapped exercise is flagged
        Harness slow = harnessFor(20241016L, tags, true); slow.captureBaseline(first);
        Report slowReport = slow.run(); Row slowRow = slowReport.row(SLOWED); boolean othersUnchanged = true;
        for (Row r : slowReport.rows) {
            if (r.exerciseId.equals(SLOWED)) continue;
            Row base = first.row(r.exerciseId); othersUnchanged &= base != null && base.median == r.median;
        }
        check(slowReport.regressions == 1 && slowRow != null && slowRow.median == 3 * first.row(SLOWED).median && othersUnchanged, "slowed exercise is reported as the only regression, the rest are untouched");
        // 8. independent oracle: refold the digest from the samples with a separate code path
        long oracle = fnvText(FNV_OFFSET, first.envFingerprint);
        for (Row r : first.rows) {
            long[] s = r.samples.clone(); sortInPlace(s); long h = fnvText(FNV_OFFSET, r.exerciseId);
            h = fnvByte(h, r.size); h = fnvByte(h, s[0]); h = fnvByte(h, s[s.length / 2]);
            h = fnvByte(h, s[s.length - 1]); oracle = fnvByte(oracle, h);
        }
        check(oracle == first.digest, "digest recomputed from raw samples matches the harness report");
        System.out.println("exercises benchmarked: " + first.rows.size());
        System.out.println("environment fingerprint: " + first.envFingerprint);
        System.out.println("run digest: " + first.digestHex() + " (rerun " + rerun.digestHex() + ")");
        System.out.println("regressions detected in slowed registry: " + slowReport.regressions);
        System.out.println("PASS");
    }
}
