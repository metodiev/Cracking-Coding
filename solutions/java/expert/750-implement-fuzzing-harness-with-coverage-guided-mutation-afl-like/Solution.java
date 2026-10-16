/** Task 750 - Implement fuzzing harness with coverage-guided mutation (AFL-like).
 * (Tier: Expert, CTCI Ch 11 — Testing)
 *
 * The harness has the three parts an AFL-like fuzzer needs:
 *   target      - a byte-string parser with a ladder of guarded branches ("CRASH" prefix, then a
 *                 derived checksum byte) that finally trips the crash flag. Every branch edge is
 *                 recorded into a hit-count map keyed by hash(previous edge, edge id), i.e. the
 *                 classic edge bitmap, and the first failing comparison of a run is logged as a
 *                 compare hint (the Redqueen / AFL `cmplog` idea);
 *   mutations   - bit flips, random byte writes, byte insert/delete (so lengths vary), splicing with
 *                 a second corpus entry, and a hint-guided write that plants the exact byte a
 *                 comparison is still waiting for;
 *   scheduler   - a deterministic queue: an input enters the corpus only when it reaches *new*
 *                 coverage, and half of the mutations go to the most recently discovered entry
 *                 (AFL's favoured-entry heuristic), the rest round-robin the queue. Everything is
 *                 driven by a fixed-seed splitmix64 generator, so a run is exactly reproducible.
 *
 * Simplification (documented). Single-process and single-threaded (no forkserver, no shared memory),
 * the map is a plain array with a simple edge hash instead of a shared 64K bitmap, the queue keeps
 * every interesting input instead of applying AFL's culling, and timings/exec-speed based power
 * schedules are replaced by the two-way favoured/round-robin choice.
 *
 * Verified. The crash is replayed on a fresh target instance; the found input is checked to satisfy
 * the parser's own precondition (the "CRASH" prefix and the derived checksum) rather than just
 * "the flag is set"; the corpus only ever grows when coverage strictly increases, coverage is
 * monotone, and every accepted entry is re-run to confirm it does not crash; the same seed
 * reproduces the identical crash input, corpus size and execution count; edge coverage grows
 * strictly with each prefix of the magic string (the ladder property that makes the search work);
 * an uninformed random fuzzer with the same budget is shown *not* to find the crash, which is the
 * controlled experiment the technique is about; and empty, one-byte and oversized inputs are
 * exercised as boundary cases.
 */
import java.util.*;

/** Instrumented target: a byte parser whose last branch trips the crash flag. */
final class Target {
    static final int MAP_SIZE = 512;

    final int[] map = new int[MAP_SIZE];
    private final List<Integer> touched = new ArrayList<>();
    int status;                 // 0 = ok, 1 = crash
    int hintIndex = -1;         // first failing comparison of the last run
    int hintValue;

    void reset() {
        for (int index : touched) map[index] = 0;
        touched.clear();
        status = 0;
        hintIndex = -1;
        hintValue = 0;
    }

    private int hit(int previous, int edge) {
        int bucket = (previous * 31 + edge) & (MAP_SIZE - 1);
        if (map[bucket] == 0) touched.add(bucket);
        map[bucket]++;
        return edge;
    }

    private void hint(int index, int value) {
        if (hintIndex < 0) {
            hintIndex = index;
            hintValue = value & 0xFF;
        }
    }

    int coverageCount() {
        int count = 0;
        for (int hits : map) if (hits > 0) count++;
        return count;
    }

    /** Runs the parser; the crash flag is set only when every stage passes. */
    void run(byte[] input) {
        int previous = 0;
        previous = hit(previous, 1);
        if (input.length < 6) {
            hit(previous, 2);
            return;
        }
        previous = hit(previous, 3);
        if (input[0] != 'C') {
            hit(previous, 4);
            hint(0, 'C');
            return;
        }
        previous = hit(previous, 5);
        if (input[1] != 'R') {
            hit(previous, 6);
            hint(1, 'R');
            return;
        }
        previous = hit(previous, 7);
        if (input[2] != 'A') {
            hit(previous, 8);
            hint(2, 'A');
            return;
        }
        previous = hit(previous, 9);
        if (input[3] != 'S') {
            hit(previous, 10);
            hint(3, 'S');
            return;
        }
        previous = hit(previous, 11);
        if (input[4] != 'H') {
            hit(previous, 12);
            hint(4, 'H');
            return;
        }
        previous = hit(previous, 13);
        int expected = (input[0] ^ input[1] ^ input[2] ^ input[3] ^ input[4] ^ 0x2A) & 0xFF;
        if ((input[5] & 0xFF) != expected) {
            hit(previous, 14);
            hint(5, expected);
            return;
        }
        hit(previous, 15);
        status = 1;                                  // the crash
    }
}

/** The fuzzing engine: mutations, coverage-based queue, and a random baseline. */
final class Fuzzer {
    static final int MAP_SIZE = Target.MAP_SIZE;

    static final class Outcome {
        final byte[] crashInput;
        final int executions;
        final int corpusSize;
        final int coverage;
        final int coverageGrowthSteps;

        Outcome(byte[] crashInput, int executions, int corpusSize, int coverage,
                int coverageGrowthSteps) {
            this.crashInput = crashInput;
            this.executions = executions;
            this.corpusSize = corpusSize;
            this.coverage = coverage;
            this.coverageGrowthSteps = coverageGrowthSteps;
        }
    }

    private final Target target = new Target();
    private final List<byte[]> corpus = new ArrayList<>();
    private final List<Integer> corpusCoverage = new ArrayList<>();
    private long state;
    private int executions;
    private final int[] coverageHistory = new int[64];
    private int historyLength;

    Fuzzer(long seed) {
        this.state = seed;
    }

    private long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    List<byte[]> corpusEntries() { return corpus; }

    List<Integer> corpusCoverage() { return corpusCoverage; }

    int[] coverageHistory() { return Arrays.copyOf(coverageHistory, historyLength); }

    /** One mutation step: a hinted single-byte repair, or a small stack of havoc operations. */
    byte[] mutate(byte[] input, byte[] spliceSource, int hintIndex, int hintValue, boolean useHint) {
        if (useHint && hintIndex >= 0 && nextInt(3) == 0) {
            byte[] repaired = hintIndex < input.length ? input.clone()
                    : Arrays.copyOf(input, hintIndex + 1);
            repaired[hintIndex] = (byte) hintValue;
            return repaired;
        }
        byte[] result = input.clone();
        int rounds = 1 + nextInt(4);
        for (int round = 0; round < rounds; round++) {
            switch (nextInt(7)) {
                case 0:                                   // flip a random bit
                    if (result.length > 0) {
                        int index = nextInt(result.length);
                        result[index] ^= (byte) (1 << nextInt(8));
                    }
                    break;
                case 1:                                   // random byte
                    if (result.length > 0) result[nextInt(result.length)] = (byte) nextInt(256);
                    break;
                case 2:                                   // insert a byte
                    if (result.length < 32) {
                        byte[] grown = new byte[result.length + 1];
                        int at = result.length == 0 ? 0 : nextInt(result.length + 1);
                        System.arraycopy(result, 0, grown, 0, at);
                        grown[at] = (byte) nextInt(256);
                        System.arraycopy(result, at, grown, at + 1, result.length - at);
                        result = grown;
                    }
                    break;
                case 3:                                   // delete a byte
                    if (result.length > 1) {
                        int at = nextInt(result.length);
                        byte[] shrunk = new byte[result.length - 1];
                        System.arraycopy(result, 0, shrunk, 0, at);
                        System.arraycopy(result, at + 1, shrunk, at, result.length - at - 1);
                        result = shrunk;
                    }
                    break;
                case 4:                                   // splice with another corpus entry
                    if (spliceSource != null && spliceSource.length > 0 && result.length > 0) {
                        int from = nextInt(spliceSource.length);
                        int to = nextInt(result.length);
                        int span = Math.min(spliceSource.length - from, result.length - to);
                        for (int i = 0; i < span; i++) result[to + i] = spliceSource[from + i];
                    }
                    break;
                case 5:                                   // grow towards the expected length
                    if (result.length < 8) {
                        byte[] grown = Arrays.copyOf(result, result.length + 1);
                        grown[grown.length - 1] = (byte) nextInt(256);
                        result = grown;
                    }
                    break;
                default:                                  // plant the byte a comparison waits for
                    if (useHint && hintIndex >= 0) {
                        if (hintIndex >= result.length) {
                            result = Arrays.copyOf(result, hintIndex + 1);
                        }
                        result[hintIndex] = (byte) hintValue;
                    }
                    break;
            }
        }
        return result;
    }

    /** Coverage-guided search; returns the crashing input or null when the budget is exhausted. */
    Outcome run(int budget) {
        corpus.clear();
        corpusCoverage.clear();
        historyLength = 0;
        executions = 0;
        target.reset();
        byte[] seed = {(byte) 's', (byte) 'e', (byte) 'e', (byte) 'd'};
        target.run(seed);
        corpus.add(seed);
        corpusCoverage.add(target.coverageCount());
        while (executions < budget) {
            boolean favoured = !corpus.isEmpty() && nextInt(2) == 0;
            int pick = favoured ? corpus.size() - 1 : nextInt(corpus.size());
            byte[] parent = corpus.get(pick);
            target.reset();
            target.run(parent);                      // re-run the parent to obtain its compare hint
            executions++;
            int hintIndex = target.hintIndex;
            int hintValue = target.hintValue;
            byte[] splice = corpus.size() > 1 ? corpus.get(nextInt(corpus.size())) : null;
            byte[] child = mutate(parent, splice, hintIndex, hintValue, true);
            executions++;
            target.reset();
            target.run(child);
            if (target.status == 1) {
                return new Outcome(child, executions, corpus.size(), target.coverageCount(),
                        historyLength);
            }
            int coverage = target.coverageCount();
            if (coverage > corpusCoverage.get(corpusCoverage.size() - 1)) {
                corpus.add(child);
                corpusCoverage.add(coverage);
                if (historyLength < coverageHistory.length) coverageHistory[historyLength++] = coverage;
            }
        }
        return null;
    }

    /** Control experiment: the same engine without coverage feedback and without compare hints. */
    Outcome runRandom(int budget) {
        executions = 0;
        historyLength = 0;
        byte[] seed = {(byte) 's', (byte) 'e', (byte) 'e', (byte) 'd'};
        while (executions < budget) {
            byte[] child = mutate(seed, null, -1, 0, false);
            executions++;
            target.reset();
            target.run(child);
            if (target.status == 1) {
                return new Outcome(child, executions, 1, target.coverageCount(), 0);
            }
        }
        return null;
    }
}

public class Solution {
    static int checksRun;

    static String escape(byte[] input) {
        StringBuilder sb = new StringBuilder();
        for (byte b : input) {
            char c = (char) (b & 0xFF);
            if (c >= 32 && c < 127) sb.append(c);
            else sb.append('\\').append(String.format("%02x", b & 0xFF));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // --- target semantics and boundary inputs --------------------------------
        Target target = new Target();
        target.reset();
        target.run(new byte[0]);
        check(target.status == 0, "an empty input cannot crash");
        check(target.coverageCount() >= 1, "the entry edge is always recorded");
        int earlyCoverage = target.coverageCount();
        target.reset();
        target.run(new byte[]{'C', 'R', 'A', 'S', 'H', 0});
        check(target.status == 0, "a wrong checksum does not crash");
        check(target.coverageCount() > earlyCoverage, "a deeper prefix reaches more edges");

        target.reset();
        target.run(new byte[]{(byte) 'C', (byte) 'R', (byte) 'A', (byte) 'S', (byte) 'H', 1});
        check(target.hintIndex == 5, "the checksum comparison is reported as a hint");
        int expectedChecksum = ('C' ^ 'R' ^ 'A' ^ 'S' ^ 'H' ^ 0x2A) & 0xFF;
        check(target.hintValue == expectedChecksum, "the hint carries the expected byte, got "
                + target.hintValue + " vs " + expectedChecksum);
        target.reset();
        byte[] crashing = {(byte) 'C', (byte) 'R', (byte) 'A', (byte) 'S', (byte) 'H',
                (byte) expectedChecksum};
        target.run(crashing);
        check(target.status == 1, "the full magic and checksum crash the target");

        target.reset();
        target.run(new byte[]{(byte) 'X', (byte) 'R', (byte) 'A', (byte) 'S', (byte) 'H', 0});
        check(target.status == 0 && target.hintIndex == 0, "the first stage hint is reported");
        check(target.hintValue == 'C', "the first stage hint carries the expected byte");
        byte[] oversized = new byte[4096];
        Arrays.fill(oversized, (byte) 0xFF);
        target.reset();
        target.run(oversized);
        check(target.status == 0, "an oversized input is handled without crashing");

        // --- the ladder property: coverage grows with each correct byte -----------
        byte[] template = {(byte) 'C', (byte) 'R', (byte) 'A', (byte) 'S', (byte) 'H', 0};
        byte[] wrongFirst = {(byte) 'X', (byte) 'X', (byte) 'X', (byte) 'X', (byte) 'X', 0};
        int ladderTop = -1;
        for (int correct = 0; correct <= 5; correct++) {
            byte[] probe = wrongFirst.clone();
            for (int i = 0; i < correct; i++) probe[i] = template[i];
            target.reset();
            target.run(probe);
            int coverage = target.coverageCount();
            check(coverage > ladderTop, "coverage must grow with each correct leading byte ("
                    + correct + "), got " + coverage + " after " + ladderTop);
            ladderTop = coverage;
        }
        byte[] failedChecksum = {(byte) 'C', (byte) 'R', (byte) 'A', (byte) 'S', (byte) 'H', 0};
        target.reset();
        target.run(failedChecksum);
        int checksumFailureCoverage = target.coverageCount();
        int[] checksumFailureMap = target.map.clone();
        byte[] full = {(byte) 'C', (byte) 'R', (byte) 'A', (byte) 'S', (byte) 'H',
                (byte) expectedChecksum};
        target.reset();
        target.run(full);
        check(target.status == 1, "the complete input crashes");
        check(target.coverageCount() == checksumFailureCoverage,
                "the crash replaces the rejection edge one for one");
        check(!Arrays.equals(target.map, checksumFailureMap),
                "the crash edge is a different edge than the rejection edge");

        // --- determinism of the target ------------------------------------------
        target.reset();
        target.run(new byte[]{(byte) 'C', (byte) 'R', 7, 7, 7, 7});
        int firstCoverage = target.coverageCount();
        target.reset();
        target.run(new byte[]{(byte) 'C', (byte) 'R', 7, 7, 7, 7});
        check(target.coverageCount() == firstCoverage, "the same input gives the same coverage");

        // --- coverage-guided search ---------------------------------------------
        Fuzzer fuzzer = new Fuzzer(0x1234ABCDL);
        Fuzzer.Outcome guided = fuzzer.run(20000);
        check(guided != null, "the coverage guided fuzzer must find the crash");
        check(guided.executions < 5000, "the guided search should be quick, took "
                + guided.executions + " executions");
        check(guided.corpusSize > 1, "the corpus grew while climbing, size " + guided.corpusSize);
        check(guided.coverage >= ladderTop, "the corpus reached deep coverage, got " + guided.coverage);
        check(guided.coverageGrowthSteps > 1, "coverage grew in several steps, got "
                + guided.coverageGrowthSteps);

        // the found input must satisfy the parser's own precondition, not just the flag
        byte[] found = guided.crashInput;
        check(found.length >= 6, "the crash input is long enough");
        check(found[0] == 'C' && found[1] == 'R' && found[2] == 'A' && found[3] == 'S'
                        && found[4] == 'H',
                "the crash input starts with the magic string, got " + escape(found));
        int recomputed = (found[0] ^ found[1] ^ found[2] ^ found[3] ^ found[4] ^ 0x2A) & 0xFF;
        check((found[5] & 0xFF) == recomputed, "the checksum byte satisfies the derived equation");
        Target replay = new Target();
        replay.reset();
        replay.run(found);
        check(replay.status == 1, "a fresh target instance crashes on the reported input");

        // --- the corpus only contains crash-free inputs and coverage is monotone -
        for (byte[] entry : fuzzer.corpusEntries()) {
            Target probe = new Target();
            probe.reset();
            probe.run(entry);
            check(probe.status == 0, "a corpus entry must not crash the target");
        }
        List<Integer> history = fuzzer.corpusCoverage();
        for (int i = 1; i < history.size(); i++) {
            check(history.get(i) > history.get(i - 1),
                    "corpus coverage must strictly increase at step " + i);
        }
        int[] growth = fuzzer.coverageHistory();
        for (int i = 1; i < growth.length; i++) {
            check(growth[i] > growth[i - 1], "coverage history must be monotone");
        }

        // --- control experiment: no feedback, no hints --------------------------
        Fuzzer randomFuzzer = new Fuzzer(0x77777777L);
        Fuzzer.Outcome blind = randomFuzzer.runRandom(20000);
        check(blind == null, "the uninformed fuzzer must not find the crash within the budget");
        check(guided.executions < 20000,
                "the guided search must beat the blind budget it is compared against");

        // --- determinism of the whole harness ------------------------------------
        Fuzzer repeatFuzzer = new Fuzzer(0x1234ABCDL);
        Fuzzer.Outcome repeat = repeatFuzzer.run(20000);
        check(repeat != null && repeat.executions == guided.executions,
                "the same seed gives the same execution count");
        check(repeat.corpusSize == guided.corpusSize, "the same seed gives the same corpus size");
        check(escape(repeat.crashInput).equals(escape(guided.crashInput)),
                "the same seed gives the same crash input");

        System.out.println("guided search: " + guided.executions + " executions, corpus "
                + guided.corpusSize + " entries, coverage " + guided.coverage
                + ", crash input \"" + escape(guided.crashInput) + "\"");
        System.out.println("blind search: " + 20000 + " executions without a crash; "
                + checksRun + " checks");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
