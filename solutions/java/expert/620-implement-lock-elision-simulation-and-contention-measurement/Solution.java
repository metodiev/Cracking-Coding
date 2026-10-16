/** Task 620 - Implement lock elision simulation and contention measurement. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * Lock elision is simulated on a logical clock. The workload is a list of critical sections, each with a
 * start tick, a duration, the locations it increments and the locations it reads. One scheduler drives
 * three modes:
 *   - coarse locking: a single global lock taken in arrival order, so the sections run one after another;
 *   - elision: a section that finds no conflicting section running executes without the lock and may
 *     overlap everything it does not conflict with; a section that finds a conflict aborts, is deferred
 *     until the conflicting sections are gone and then runs under the lock (which is exclusive);
 *   - unsafe elision: the same scheduler with conflict detection switched off, kept as the counterexample
 *     that shows why the conflict check is not optional.
 * Conflicts are detected on exact locations or on cache lines, which produces the classic false aborts:
 * sections touching different keys that happen to share a line must serialize.
 *
 * Verified: the counter invariant (each location ends at its initial value plus the number of sections
 * that incremented it) holds in the coarse and the eliding runs although they take different time; no
 * two conflicting sections ever overlap in a safe run; the unsafe mode loses updates; elision never takes
 * longer than coarse locking and reaches the ideal parallel makespan on a conflict-free workload;
 * line-granular detection aborts strictly more often than exact detection on adjacent keys; contention is
 * measured and grows with the conflicts; and the whole simulation is deterministic.
 */
import java.util.*;

class CriticalSection {
    final int id;
    final int start;
    final int duration;
    final int[] writes;
    final int[] reads;
    int begin = -1, end = -1;
    int[] startValues;
    boolean elided;
    boolean aborted;

    CriticalSection(int id, int start, int duration, int[] writes, int[] reads) {
        this.id = id;
        this.start = start;
        this.duration = duration;
        this.writes = writes;
        this.reads = reads;
    }

    CriticalSection copy() {
        return new CriticalSection(id, start, duration, writes, reads);
    }

    String label() {
        return "#" + id + "(start " + start + ", dur " + duration + ")";
    }
}

class ElisionSimulator {
    static final int LINE_SIZE = 4;

    final int[] memory;
    final List<CriticalSection> sections;
    final boolean coarse;
    final boolean detectConflicts;
    final boolean lineGranularity;

    final List<CriticalSection> running = new ArrayList<>();
    CriticalSection lockOwner;
    int now;
    int elided;
    int locked;
    int fallbacks;
    int lockWaitTicks;
    int makespan;

    ElisionSimulator(int[] memory, List<CriticalSection> sections, boolean coarse,
                     boolean detectConflicts, boolean lineGranularity) {
        this.memory = memory.clone();
        this.sections = new ArrayList<>();                  // every run gets its own fresh schedule
        for (CriticalSection section : sections) this.sections.add(section.copy());
        this.coarse = coarse;
        this.detectConflicts = detectConflicts;
        this.lineGranularity = lineGranularity;
    }

    boolean sameLocation(int a, int b) {
        return lineGranularity ? a / LINE_SIZE == b / LINE_SIZE : a == b;
    }

    /** A write conflicts with any other access to the same location; two reads never conflict. */
    boolean conflicts(CriticalSection a, CriticalSection b) {
        for (int x : a.writes) {
            for (int y : b.writes) if (sameLocation(x, y)) return true;
            for (int y : b.reads) if (sameLocation(x, y)) return true;
        }
        for (int x : a.reads) {
            for (int y : b.writes) if (sameLocation(x, y)) return true;
        }
        return false;
    }

    private boolean conflictsWithRunning(CriticalSection candidate) {
        for (CriticalSection other : running) if (conflicts(candidate, other)) return true;
        return false;
    }

    private void start(CriticalSection section, boolean useLock) {
        section.begin = now;
        section.end = now + section.duration;
        section.elided = !useLock;
        section.startValues = new int[section.writes.length];
        for (int i = 0; i < section.writes.length; i++) section.startValues[i] = memory[section.writes[i]];
        running.add(section);
        if (useLock) {
            locked++;
            lockOwner = section;
        } else {
            elided++;
        }
    }

    private void finish(CriticalSection section) {
        for (int i = 0; i < section.writes.length; i++) {
            memory[section.writes[i]] = section.startValues[i] + 1;      // a plain read-modify-write
        }
    }

    void run() {
        int limit = 1;
        for (CriticalSection section : sections) limit += section.start + section.duration + 2;
        int finished = 0;
        for (now = 0; now <= limit && finished < sections.size(); now++) {
            for (int i = running.size() - 1; i >= 0; i--) {
                if (running.get(i).end == now) {
                    finish(running.get(i));
                    running.remove(i);
                    finished++;
                }
            }
            if (lockOwner != null && lockOwner.end <= now) lockOwner = null;
            if (lockOwner != null) {
                lockWaitTicks++;
                continue;
            }
            for (CriticalSection section : sections) {
                if (section.begin >= 0 || section.start > now) continue;
                if (coarse) {
                    start(section, true);
                    break;
                }
                if (!detectConflicts) {
                    start(section, false);
                    continue;
                }
                if (conflictsWithRunning(section)) {
                    if (!section.aborted) {
                        section.aborted = true;
                        fallbacks++;
                    }
                    continue;                                    // abort, retry when the lock is free
                }
                start(section, section.aborted);                  // a retried section takes the lock
                if (section.aborted) break;                       // and it is then exclusive
            }
        }
        for (CriticalSection section : sections) {
            if (section.end < 0) throw new IllegalStateException("section " + section.id + " never ran");
            makespan = Math.max(makespan, section.end);
        }
    }

    boolean allFinished() {
        for (CriticalSection section : sections) if (section.end < 0) return false;
        return true;
    }

    /** Fallbacks plus sections that had to wait for the lock: the contention of the workload. */
    int contention() {
        int serialised = 0;
        for (CriticalSection section : sections) if (!section.elided) serialised++;
        return fallbacks + serialised;
    }

    int[] snapshot() { return memory.clone(); }
}

public class Solution {
    private static long state = 0x2B8E1D4C6F5A0937L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static List<CriticalSection> disjointWorkload() {
        List<CriticalSection> sections = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            sections.add(new CriticalSection(i, 0, 5, new int[]{40 + i * 8}, new int[]{}));
        }
        return sections;
    }

    static List<CriticalSection> contendedWorkload() {
        List<CriticalSection> sections = new ArrayList<>();
        for (int i = 0; i < 4; i++) sections.add(new CriticalSection(i, i, 5, new int[]{0}, new int[]{}));
        return sections;
    }

    /** Four different keys, but they all share one cache line with the first write. */
    static List<CriticalSection> adjacentKeysWorkload() {
        List<CriticalSection> sections = new ArrayList<>();
        sections.add(new CriticalSection(0, 0, 5, new int[]{4}, new int[]{}));
        sections.add(new CriticalSection(1, 0, 5, new int[]{6}, new int[]{}));
        sections.add(new CriticalSection(2, 0, 5, new int[]{5}, new int[]{}));
        sections.add(new CriticalSection(3, 0, 5, new int[]{}, new int[]{7}));
        return sections;
    }

    static int[] expectedCounters(int[] initial, List<CriticalSection> sections) {
        int[] expected = initial.clone();
        for (CriticalSection section : sections) {
            for (int location : section.writes) expected[location]++;
        }
        return expected;
    }

    static void verifyCounters(int[] initial, int[] result, List<CriticalSection> sections, String label) {
        int[] expected = expectedCounters(initial, sections);
        for (int i = 0; i < initial.length; i++) {
            check(result[i] == expected[i], label + ": counter " + i + " is " + result[i]
                    + " but " + expected[i] + " is expected");
        }
    }

    /** No two conflicting sections may overlap in a safe run, which is what makes the counters exact. */
    static void verifyAtomicity(ElisionSimulator sim, String label) {
        for (CriticalSection a : sim.sections) {
            check(a.begin >= 0, label + ": " + a.label() + " ran");
            check(a.end - a.begin == a.duration, label + ": " + a.label() + " ran for its duration");
            for (CriticalSection b : sim.sections) {
                if (a.id >= b.id || !sim.conflicts(a, b)) continue;
                boolean overlap = a.begin < b.end && b.begin < a.end;
                check(!overlap, label + ": conflicting sections " + a.label() + " and " + b.label()
                        + " ran at the same time");
            }
        }
    }

    public static void main(String[] args) {
        int[] initial = new int[120];
        for (int i = 0; i < initial.length; i++) initial[i] = i % 7;

        // 1. a conflict-free workload: everything elides and runs fully in parallel
        List<CriticalSection> disjoint = disjointWorkload();
        ElisionSimulator elide = new ElisionSimulator(initial, disjoint, false, true, false);
        elide.run();
        ElisionSimulator coarse = new ElisionSimulator(initial, disjoint, true, true, false);
        coarse.run();
        check(elide.elided == disjoint.size() && elide.locked == 0, "the disjoint workload elides entirely");
        check(elide.makespan == 5, "the disjoint workload finishes in one duration");
        check(coarse.makespan == 30, "coarse locking serialises it");
        check(elide.makespan < coarse.makespan, "elision beats coarse locking");
        check(elide.contention() == 0, "no contention without conflicts");
        verifyAtomicity(elide, "disjoint");
        verifyCounters(initial, elide.snapshot(), elide.sections, "disjoint eliding");
        verifyCounters(initial, coarse.snapshot(), coarse.sections, "disjoint coarse");

        // 2. a contended workload: the later sections abort and fall back to the lock
        List<CriticalSection> contended = contendedWorkload();
        ElisionSimulator contendedElide = new ElisionSimulator(initial, contended, false, true, false);
        contendedElide.run();
        ElisionSimulator contendedCoarse = new ElisionSimulator(initial, contended, true, true, false);
        contendedCoarse.run();
        check(contendedElide.elided >= 1, "the first section elides");
        check(contendedElide.locked == 3, "the three later sections take the lock");
        check(contendedElide.fallbacks >= 1, "the contended workload measures fallbacks");
        check(contendedElide.makespan <= contendedCoarse.makespan, "elision is never slower than coarse");
        check(contendedElide.makespan == 20 && contendedCoarse.makespan == 20, "the contended makespans");
        verifyAtomicity(contendedElide, "contended");
        verifyCounters(initial, contendedElide.snapshot(), contendedElide.sections, "contended eliding");
        verifyCounters(initial, contendedCoarse.snapshot(), contendedCoarse.sections, "contended coarse");
        check(contendedElide.snapshot()[0] == initial[0] + 4, "all four increments land");
        check(contendedElide.contention() > elide.contention(), "contention grows with the conflicts");

        // 3. the counterexample: without conflict detection the increments are lost
        ElisionSimulator unsafe = new ElisionSimulator(initial, contended, false, false, false);
        unsafe.run();
        check(unsafe.elided == contended.size() && unsafe.fallbacks == 0, "the unsafe mode elides everything");
        check(unsafe.snapshot()[0] == initial[0] + 1, "the unsafe mode loses three increments");
        check(unsafe.makespan == 8, "the unsafe mode is fast because it is wrong");

        // 4. adjacent keys on one cache line: exact detection elides, line detection aborts
        List<CriticalSection> adjacent = adjacentKeysWorkload();
        ElisionSimulator exact = new ElisionSimulator(initial, adjacent, false, true, false);
        exact.run();
        ElisionSimulator byLine = new ElisionSimulator(initial, adjacent, false, true, true);
        byLine.run();
        check(exact.fallbacks == 0 && exact.makespan == 5, "distinct keys do not conflict exactly");
        check(byLine.fallbacks == 3 && byLine.makespan == 20, "sharing a line forces serialisation");
        int falseAborts = byLine.fallbacks - exact.fallbacks;
        check(falseAborts > 0, "the false aborts are measured");
        verifyAtomicity(exact, "exact granularity");
        verifyAtomicity(byLine, "line granularity");
        verifyCounters(initial, exact.snapshot(), exact.sections, "exact granularity");
        verifyCounters(initial, byLine.snapshot(), byLine.sections, "line granularity");

        // 5. randomized workloads: determinism, atomicity, the counter invariant and the makespan order
        long largestSaving = 0;
        for (int trial = 0; trial < 25; trial++) {
            List<CriticalSection> random = new ArrayList<>();
            int count = 4 + (int) ((nextLong() >>> 33) % 6);
            for (int i = 0; i < count; i++) {
                int start = (int) ((nextLong() >>> 33) % 8);
                int duration = 1 + (int) ((nextLong() >>> 33) % 5);
                int writes = (int) ((nextLong() >>> 33) % 3);
                int reads = (int) ((nextLong() >>> 33) % 2);
                Set<Integer> writeSet = new TreeSet<>();
                for (int k = 0; k < writes; k++) writeSet.add((int) ((nextLong() >>> 33) % 6));
                Set<Integer> readSet = new TreeSet<>();
                for (int k = 0; k < reads; k++) readSet.add((int) ((nextLong() >>> 33) % 6));
                int[] writeKeys = new int[writeSet.size()];
                int[] readKeys = new int[readSet.size()];
                int index = 0;
                for (int key : writeSet) writeKeys[index++] = key;
                index = 0;
                for (int key : readSet) readKeys[index++] = key;
                random.add(new CriticalSection(i, start, duration, writeKeys, readKeys));
            }
            ElisionSimulator first = new ElisionSimulator(initial, random, false, true, false);
            ElisionSimulator again = new ElisionSimulator(initial, random, false, true, false);
            first.run();
            again.run();
            check(first.makespan == again.makespan && first.fallbacks == again.fallbacks,
                    "trial " + trial + " is deterministic");
            check(first.contention() == again.contention(), "trial " + trial + " contention is deterministic");
            verifyAtomicity(first, "random trial " + trial);
            verifyCounters(initial, first.snapshot(), first.sections, "random trial " + trial);
            ElisionSimulator coarseRandom = new ElisionSimulator(initial, random, true, true, false);
            coarseRandom.run();
            check(first.makespan <= coarseRandom.makespan, "trial " + trial + ": elision is never slower");
            verifyCounters(initial, coarseRandom.snapshot(), coarseRandom.sections, "random trial " + trial + " coarse");
            largestSaving = Math.max(largestSaving, coarseRandom.makespan - first.makespan);
        }
        check(largestSaving > 0, "some random workload profits from elision");
        System.out.println("disjoint: elided " + elide.elided + ", makespan " + elide.makespan + " vs coarse "
                + coarse.makespan + "; contended: elided " + contendedElide.elided + ", makespan "
                + contendedElide.makespan + " vs coarse " + contendedCoarse.makespan + ", contention "
                + contendedElide.contention() + " vs " + elide.contention());
        System.out.println("cache-line false aborts " + falseAborts + " (makespan " + exact.makespan + " -> "
                + byLine.makespan + "), unsafe mode loses " + (contendedElide.snapshot()[0]
                - unsafe.snapshot()[0]) + " increments, best random saving " + largestSaving
                + " ticks, lock wait ticks " + contendedElide.lockWaitTicks);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
