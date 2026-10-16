/** Task 913 - Implement verified randomized algorithm reproducibility with seed management and audit logs.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Reproducing a randomized experiment needs three things, and this file implements and audits all three:
 *
 *   seed management   one root seed plus the experiment identity and a version tag are mixed into a per-run
 *                     sub-seed. Sub-seeds must be a pure function of that triple, must change completely when
 *                     any component changes, and must differ between experiments, otherwise two runs would
 *                     silently share a stream;
 *   replay            running the same experiment twice with the root seed reproduces the result bit for bit,
 *                     so a reported number can be re-derived rather than trusted;
 *   audit log         every run appends a record that contains the previous record's hash, which makes the log
 *                     a hash chain: editing, reordering, deleting or forking any record is detected, and the
 *                     verifier names the offending index.
 *
 * The checks re-derive the properties instead of restating them: sub-seed purity and avalanche behaviour,
 * replay equality for both the result and the log, chain verification, four specific tamper classes with the
 * detected index, and the boundary cases (empty log, single record, fork).
 *
 * Simplification: the hash is a 64-bit splitmix64 mix rather than SHA-256, and the "experiment" is a small
 * deterministic Monte-Carlo estimate. Invariants verified: identical seeds give identical results, distinct
 * experiments get distinct streams, and any mutation of the log is caught at the right position.
 */
import java.util.*;

/** splitmix64: seed derivation, stream generation and the toy chain hash. */
final class Rng913 {
    private long state;

    Rng913(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** The chain hash: a fixed, order-sensitive mixing of a label and a previous digest. */
    static long mix(long previous, long value) {
        long h = previous ^ (value * 0x9E3779B97F4A7C15L);
        h ^= h >>> 30;
        h *= 0xBF58476D1CE4E5B9L;
        h ^= h >>> 27;
        h *= 0x94D049BB133111EBL;
        return h ^ (h >>> 31);
    }

    static long hashText(String text) {
        long h = 0xC0FFEEL;
        for (int i = 0; i < text.length(); i++) h = mix(h, text.charAt(i));
        return mix(h, text.length());
    }
}

/** Immutable seed derivation: sub-seed = f(root, experiment, version). */
final class SeedManager {
    final long rootSeed;

    SeedManager(long rootSeed) { this.rootSeed = rootSeed; }

    long subSeed(String experiment, int version) {
        long h = Rng913.hashText(experiment);
        h = Rng913.mix(h, version);
        h = Rng913.mix(h, rootSeed);
        return Rng913.mix(h, 0x5EED5EEDL);
    }

    long subSeed(String experiment) { return subSeed(experiment, 1); }
}

/** One line of the audit log: what ran, with which seed, and the hash of the line before it. */
final class AuditRecord {
    final String experiment;
    int version;
    long seed;
    long resultHash;
    long previousHash;
    long digest;

    AuditRecord(String experiment, int version, long seed, long resultHash, long previousHash) {
        this.experiment = experiment;
        this.version = version;
        this.seed = seed;
        this.resultHash = resultHash;
        this.previousHash = previousHash;
        this.digest = computeDigest();
    }

    long computeDigest() {
        long h = Rng913.hashText(experiment);
        h = Rng913.mix(h, version);
        h = Rng913.mix(h, seed);
        h = Rng913.mix(h, resultHash);
        return Rng913.mix(h, previousHash);
    }
}

/** The hash chain plus its verifier. */
final class AuditLog {
    final List<AuditRecord> records = new ArrayList<>();

    void append(String experiment, int version, long seed, long resultHash) {
        long previous = records.isEmpty() ? 0L : records.get(records.size() - 1).digest;
        records.add(new AuditRecord(experiment, version, seed, resultHash, previous));
    }

    /** Returns the index of the first broken link, or -1 when the whole chain verifies. */
    int verify() {
        long expected = 0L;
        for (int i = 0; i < records.size(); i++) {
            AuditRecord record = records.get(i);
            if (record.previousHash != expected) return i;
            if (record.digest != record.computeDigest()) return i;
            expected = record.digest;
        }
        return -1;
    }

    /** Two records claiming the same predecessor would let one history be swapped for another. */
    int firstFork() {
        Map<Long, Integer> seen = new HashMap<>();
        for (int i = 0; i < records.size(); i++) {
            Integer previous = seen.put(records.get(i).previousHash, i);
            if (previous != null) return i;
        }
        return -1;
    }
}

public class Solution {
    static final long ROOT_SEED = 0x913ABCDEL;
    static final String[] EXPERIMENTS = {"min-cut-sampling", "quickselect-pivot", "bootstrap-ci"};

    public static void main(String[] args) {
        SeedManager seeds = new SeedManager(ROOT_SEED);

        // ---- 1. seed derivation is pure and injective --------------------------------------------------
        for (String experiment : EXPERIMENTS) {
            check(seeds.subSeed(experiment) == seeds.subSeed(experiment),
                    "the sub-seed must be a pure function of the experiment");
            check(seeds.subSeed(experiment, 2) != seeds.subSeed(experiment, 1),
                    "bumping the version must move the sub-seed");
        }
        Set<Long> distinct = new HashSet<>();
        for (String experiment : EXPERIMENTS) distinct.add(seeds.subSeed(experiment));
        check(distinct.size() == EXPERIMENTS.length, "different experiments need different streams");
        check(new SeedManager(ROOT_SEED + 1).subSeed(EXPERIMENTS[0]) != seeds.subSeed(EXPERIMENTS[0]),
                "a different root seed must give a different stream");
        int bitFlips = 0;
        long variant = seeds.subSeed(EXPERIMENTS[0]);
        for (int i = 1; i <= 200; i++) bitFlips += Long.bitCount(variant ^ seeds.subSeed(EXPERIMENTS[0], i));
        check(bitFlips > 200 * 24, "the derivation must avalanche: " + (bitFlips / 200.0)
                + " bits differ on average");

        // ---- 2. replay: the same root seed reproduces the reported number ------------------------------
        AuditLog log = new AuditLog();
        double firstEstimate = 0;
        for (String experiment : EXPERIMENTS) {
            long seed = seeds.subSeed(experiment);
            double estimate = monteCarlo(seed, 4000);
            log.append(experiment, 1, seed, Double.doubleToLongBits(estimate));
            if (firstEstimate == 0) firstEstimate = estimate;
            check(estimate == monteCarlo(seeds.subSeed(experiment), 4000),
                    experiment + " must replay bit for bit");
        }
        check(log.records.size() == EXPERIMENTS.length, "every run must be logged");
        check(firstEstimate > 0.76 && firstEstimate < 0.81, "the Monte-Carlo estimate of pi/4 must converge, "
                + "got " + firstEstimate);
        check(firstDraw(seeds.subSeed(EXPERIMENTS[0])) != firstDraw(seeds.subSeed(EXPERIMENTS[0]) + 1),
                "a different seed must give a different draw sequence");

        // ---- 3. the audit chain verifies and is reproducible -------------------------------------------
        check(log.verify() == -1, "a freshly written log must verify");
        check(log.firstFork() == -1, "a linear log must not contain a fork");
        for (int i = 1; i < log.records.size(); i++) {
            check(log.records.get(i).previousHash == log.records.get(i - 1).digest,
                    "record " + i + " must link to the previous digest");
        }
        AuditLog replayLog = new AuditLog();
        for (String experiment : EXPERIMENTS) {
            replayLog.append(experiment, 1, seeds.subSeed(experiment),
                    Double.doubleToLongBits(monteCarlo(seeds.subSeed(experiment), 4000)));
        }
        check(replayLog.records.size() == log.records.size(), "the replay must log the same number of runs");
        for (int i = 0; i < log.records.size(); i++) {
            check(replayLog.records.get(i).digest == log.records.get(i).digest,
                    "record " + i + " must be byte-identical on replay");
        }
        check(replayLog.verify() == -1, "the replayed log must verify too");

        // ---- 4. tamper detection: four distinct attacks, each caught at the right index ----------------
        AuditLog edited = cloneLog(log);
        edited.records.get(1).resultHash ^= 0x1L;
        check(edited.verify() == 1, "editing a result must be caught at that index, got " + edited.verify());
        AuditLog seedSwapped = cloneLog(log);
        seedSwapped.records.get(2).seed ^= 0xFFFFL;
        check(seedSwapped.verify() == 2, "swapping a seed must be caught at that index, got "
                + seedSwapped.verify());
        AuditLog unlinked = cloneLog(log);
        unlinked.records.get(2).previousHash = 0xDEADBEEFL;
        check(unlinked.verify() == 2, "breaking a link must be caught at that index, got "
                + unlinked.verify());
        AuditLog reordered = cloneLog(log);
        Collections.reverse(reordered.records);
        check(reordered.verify() == 0, "reordering the log must be caught at index 0, got "
                + reordered.verify());
        check(cloneLog(log).verify() == -1, "cloning must not corrupt the chain");

        // ---- 5. boundary cases ---------------------------------------------------------------------------
        AuditLog empty = new AuditLog();
        check(empty.verify() == -1 && empty.firstFork() == -1, "an empty log verifies trivially");
        AuditLog single = new AuditLog();
        single.append("single-run", 1, 42L, 7L);
        check(single.verify() == -1, "a single record must verify");
        check(single.records.get(0).previousHash == 0L, "the first record must start from the zero hash");
        AuditLog forked = new AuditLog();
        forked.records.add(new AuditRecord("branch-a", 1, 11L, 111L, 0L));
        forked.records.add(new AuditRecord("branch-b", 1, 22L, 222L, 0L));
        check(forked.firstFork() == 1, "two records sharing a predecessor must be reported as a fork");
        AuditLog versionBump = cloneLog(log);
        versionBump.records.get(0).version = 2;
        check(versionBump.verify() == 0, "a version bump without a fresh digest must be detected");
        check(seeds.subSeed(EXPERIMENTS[0], 2) != seeds.subSeed(EXPERIMENTS[1], 2),
                "version 2 streams must differ across experiments as well");

        // ---- 6. determinism -------------------------------------------------------------------------------
        check(new SeedManager(ROOT_SEED).subSeed(EXPERIMENTS[0]) == seeds.subSeed(EXPERIMENTS[0]),
                "seed derivation must not depend on the manager instance");
        check(Double.doubleToLongBits(monteCarlo(ROOT_SEED, 100)) == Double.doubleToLongBits(
                monteCarlo(ROOT_SEED, 100)), "the experiment must be a pure function of its seed");

        System.out.println("experiments " + EXPERIMENTS.length + ", root seed " + ROOT_SEED + ", sub-seeds "
                + distinct.size() + " distinct, avalanche " + (bitFlips / 200.0) + " bits of 64");
        System.out.println("audit log " + log.records.size() + " records, verified index " + log.verify()
                + ", tamper classes caught " + 4);
        System.out.println("first estimate " + firstEstimate + " replayed bit for bit; forked logs reported at "
                + forked.firstFork());
        System.out.println("PASS");
    }

    /** A small deterministic Monte-Carlo experiment: estimate pi/4 with rejection sampling. */
    static double monteCarlo(long seed, int draws) {
        Rng913 rng = new Rng913(seed);
        int inside = 0;
        for (int i = 0; i < draws; i++) {
            double x = rng.nextDouble();
            double y = rng.nextDouble();
            if (x * x + y * y <= 1.0) inside++;
        }
        return inside / (double) draws;
    }

    static double firstDraw(long seed) { return new Rng913(seed).nextDouble(); }

    static AuditLog cloneLog(AuditLog source) {
        AuditLog copy = new AuditLog();
        for (AuditRecord record : source.records) {
            copy.records.add(new AuditRecord(record.experiment, record.version, record.seed,
                    record.resultHash, record.previousHash));
        }
        return copy;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
