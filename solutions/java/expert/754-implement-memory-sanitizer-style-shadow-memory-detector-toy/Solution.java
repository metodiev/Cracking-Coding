/** Task 754 - Implement a memory sanitizer style shadow memory detector (toy).
 * (Tier: Expert, CTCI Beyond - Security & Privacy)
 *
 * MemorySanitizer's trick is a *shadow* array with one shadow byte per application byte; the shadow
 * records only whether that byte is initialized, so an uninitialized read can be reported without ever
 * looking at a value. This toy adds the addressability half of AddressSanitizer by storing a state byte
 * per heap byte:
 *
 *   UNALLOCATED (redzone / never handed out)   UNINIT (handed out, never written)
 *   INIT (written at least once)               FREED (returned to the allocator)
 *
 * Every access checks the shadow *before* it mutates it, so a store that touches a poisoned byte is
 * reported and writes nothing - the "check first, then apply" discipline that makes sanitizers
 * trustworthy. Reports name the first offending byte plus the origin (the allocation or the explicit
 * poison that left the byte uninitialized), which is what turns "uninitialized read" into an
 * actionable finding.
 *
 * The independent oracle is a completely different layout: per-block byte arrays in a map, no flat heap
 * at all, and the same policy re-derived from scratch. It must produce exactly the same report list
 * (op index + report kind) as the shadow-memory detector on every generated program.
 *
 * Simplified: one byte of shadow per byte (real MSan tracks one *bit*), a bump allocator that never
 * reuses freed space, no stack of nested frames, no threads, and no data values are modelled at all -
 * only the shadow state, which is the point.
 *
 * Verified: fresh memory read, full and partial initialization, use-after-free on load and store,
 * double free, out-of-bounds both past the end and before the start, explicit poisoning with a fresh
 * origin, a fully valid program with zero reports, and detector == oracle on 200 generated programs.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" (no Math.random). */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

/** One operation of the scripted program: allocation id plus an offset/size inside that block. */
final class Op {
    final String kind;              // alloc | free | store | load | poison
    final int id;                   // allocation id
    final int offset;
    final int size;
    final int allocSize;            // only for alloc

    Op(String kind, int id, int offset, int size, int allocSize) {
        this.kind = kind;
        this.id = id;
        this.offset = offset;
        this.size = size;
        this.allocSize = allocSize;
    }

    static Op alloc(int id, int allocSize) { return new Op("alloc", id, 0, allocSize, allocSize); }
    static Op free(int id) { return new Op("free", id, 0, 0, 0); }
    static Op store(int id, int offset, int size) { return new Op("store", id, offset, size, 0); }
    static Op load(int id, int offset, int size) { return new Op("load", id, offset, size, 0); }
    static Op poison(int id, int offset, int size) { return new Op("poison", id, offset, size, 0); }

    String show() {
        return kind.equals("alloc") ? "alloc #" + id + " size " + allocSize
                : kind.equals("free") ? "free #" + id
                : kind + " #" + id + " [" + offset + ", " + (offset + size) + ")";
    }
}

/** The detector: one flat shadow array, one byte per heap byte, checked before every mutation. */
final class ShadowMemory {
    static final byte UNALLOCATED = 0;
    static final byte UNINIT = 1;
    static final byte INIT = 2;
    static final byte FREED = 3;

    static final int HEAP = 512;
    static final int REDZONE = 4;

    final byte[] state = new byte[HEAP];
    final int[] origin = new int[HEAP];
    final Map<Integer, Integer> base = new LinkedHashMap<>();
    final Map<Integer, Integer> blockSize = new LinkedHashMap<>();
    final Set<Integer> freed = new LinkedHashSet<>();
    final List<String> reports = new ArrayList<>();
    int bump;
    long shadowBytesRead;

    /** Runs the program on the shadow array and returns one report per offending operation. */
    List<String> run(List<Op> program) {
        for (int i = 0; i < program.size(); i++) {
            Op op = program.get(i);
            String report;
            switch (op.kind) {
                case "alloc":
                    report = allocate(op);
                    break;
                case "free":
                    report = release(op);
                    break;
                case "poison":
                    report = poison(op);
                    break;
                default:
                    report = access(op);
            }
            if (report != null) reports.add(i + " " + report);
        }
        return reports;
    }

    private String allocate(Op op) {
        int at = bump + REDZONE;
        if (at + op.allocSize + REDZONE > HEAP) return "OUT_OF_MEMORY #" + op.id;
        bump = at + op.allocSize + REDZONE;
        base.put(op.id, at);
        blockSize.put(op.id, op.allocSize);
        for (int byteIndex = 0; byteIndex < op.allocSize; byteIndex++) {
            state[at + byteIndex] = UNINIT;             // a fresh allocation is uninitialized
            origin[at + byteIndex] = op.id;
        }
        return null;
    }

    private String release(Op op) {
        Integer at = base.get(op.id);
        if (at == null) return "BAD_FREE unknown allocation #" + op.id;
        if (freed.contains(op.id)) return "BAD_FREE double free of #" + op.id;
        freed.add(op.id);
        for (int byteIndex = 0; byteIndex < blockSize.get(op.id); byteIndex++) {
            state[at + byteIndex] = FREED;
            origin[at + byteIndex] = op.id;
        }
        return null;
    }

    /** An API that returns a buffer it never wrote: the bytes become uninitialized again, new origin. */
    private String poison(Op op) {
        Integer at = base.get(op.id);
        if (at == null) return "OUT_OF_BOUNDS unknown allocation #" + op.id;
        String bounds = boundsProblem(op);
        if (bounds != null) return bounds;
        if (freed.contains(op.id)) return "USE_AFTER_FREE #" + op.id;
        int freshOrigin = 1000 + op.id;
        for (int byteIndex = 0; byteIndex < op.size; byteIndex++) {
            int index = at + op.offset + byteIndex;
            state[index] = UNINIT;
            origin[index] = freshOrigin;
        }
        return null;
    }

    private String access(Op op) {
        Integer at = base.get(op.id);
        if (at == null) return "OUT_OF_BOUNDS unknown allocation #" + op.id;
        String bounds = boundsProblem(op);
        if (bounds != null) return bounds;
        boolean load = op.kind.equals("load");
        for (int byteIndex = 0; byteIndex < op.size; byteIndex++) {
            int index = at + op.offset + byteIndex;
            shadowBytesRead++;
            if (state[index] == FREED) return "USE_AFTER_FREE #" + op.id;
            if (state[index] == UNALLOCATED) return "OUT_OF_BOUNDS redzone byte " + index;
            if (state[index] == UNINIT && load) {
                return "UNINITIALIZED_READ #" + op.id + " at offset " + (op.offset + byteIndex)
                        + " origin " + origin[index];
            }
        }
        if (!load) {
            for (int byteIndex = 0; byteIndex < op.size; byteIndex++) state[at + op.offset + byteIndex] = INIT;
        }
        return null;
    }

    private String boundsProblem(Op op) {
        int size = blockSize.get(op.id);
        if (op.offset < 0) return "OUT_OF_BOUNDS negative offset " + op.offset + " on #" + op.id;
        if (op.offset + op.size > size) {
            return "OUT_OF_BOUNDS range [" + op.offset + ", " + (op.offset + op.size) + ") on #" + op.id
                    + " of size " + size;
        }
        return null;
    }

    long uninitializedReports() {
        return reports.stream().filter(r -> r.contains("UNINITIALIZED_READ")).count();
    }
}

/** The oracle: per-block byte arrays in a map, same policy, independently written. */
final class ShadowOracle {
    static List<String> run(List<Op> program) {
        Map<Integer, Integer> sizes = new HashMap<>();
        Map<Integer, byte[]> blocks = new HashMap<>();
        Set<Integer> freed = new LinkedHashSet<>();
        List<String> reports = new ArrayList<>();
        for (int i = 0; i < program.size(); i++) {
            Op op = program.get(i);
            String report = null;
            if (op.kind.equals("alloc")) {
                sizes.put(op.id, op.allocSize);
                byte[] block = new byte[op.allocSize];
                Arrays.fill(block, (byte) 1);
                blocks.put(op.id, block);
            } else if (op.kind.equals("free")) {
                byte[] block = blocks.get(op.id);
                if (block == null) report = "BAD_FREE unknown allocation #" + op.id;
                else if (!freed.add(op.id)) report = "BAD_FREE double free of #" + op.id;
                else Arrays.fill(block, (byte) 3);
            } else {
                byte[] block = blocks.get(op.id);
                if (block == null) {
                    report = "OUT_OF_BOUNDS unknown allocation #" + op.id;
                } else if (op.offset < 0) {
                    report = "OUT_OF_BOUNDS negative offset " + op.offset + " on #" + op.id;
                } else if (op.offset + op.size > sizes.get(op.id)) {
                    report = "OUT_OF_BOUNDS range [" + op.offset + ", " + (op.offset + op.size) + ") on #"
                            + op.id + " of size " + sizes.get(op.id);
                } else if (freed.contains(op.id)) {
                    report = "USE_AFTER_FREE #" + op.id;
                } else if (op.kind.equals("poison")) {
                    for (int b = 0; b < op.size; b++) block[op.offset + b] = 1;
                } else {
                    int firstUninit = -1;
                    for (int b = 0; b < op.size; b++) if (block[op.offset + b] == 1) { firstUninit = op.offset + b; break; }
                    if (op.kind.equals("load") && firstUninit >= 0) {
                        report = "UNINITIALIZED_READ #" + op.id + " at offset " + firstUninit;
                    } else if (op.kind.equals("store")) {
                        for (int b = 0; b < op.size; b++) block[op.offset + b] = 2;
                    }
                }
            }
            if (report != null) reports.add(i + " " + report);
        }
        return reports;
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    /** Reports compared by op index and kind, so the two engines only have to agree on the verdict. */
    static List<String> verdicts(List<String> reports) {
        List<String> out = new ArrayList<>();
        for (String report : reports) {
            String[] parts = report.split(" ");
            out.add(parts[0] + " " + parts[1]);
        }
        return out;
    }

    static List<String> runDetector(List<Op> program) { return verdicts(new ShadowMemory().run(program)); }

    public static void main(String[] args) {
        // 1. a fresh allocation is uninitialized: reading it is reported with the allocation as origin
        ShadowMemory fresh = new ShadowMemory();
        fresh.run(List.of(Op.alloc(1, 8), Op.load(1, 0, 8)));
        check(fresh.reports.size() == 1 && fresh.reports.get(0).equals("1 UNINITIALIZED_READ #1 at offset 0 origin 1"),
                "fresh memory read reports the allocation as origin: " + fresh.reports);

        // 2. partial initialization is precisely tracked: storing 3 of 8 bytes leaves 5 uninitialized
        ShadowMemory partial = new ShadowMemory();
        partial.run(List.of(Op.alloc(1, 8), Op.store(1, 0, 3), Op.load(1, 0, 8)));
        check(partial.reports.size() == 1 && partial.reports.get(0).contains("at offset 3"),
                "a partially written block reports the first unwritten byte: " + partial.reports);
        ShadowMemory partialOk = new ShadowMemory();
        partialOk.run(List.of(Op.alloc(1, 8), Op.store(1, 0, 3), Op.load(1, 0, 3), Op.store(1, 3, 5),
                Op.load(1, 0, 8)));
        check(partialOk.reports.isEmpty(), "covering the whole block clears the shadow: " + partialOk.reports);

        // 3. a fully written block is readable - the detector must not cry wolf
        ShadowMemory clean = new ShadowMemory();
        clean.run(List.of(Op.alloc(1, 16), Op.store(1, 0, 16), Op.load(1, 0, 16), Op.free(1)));
        check(clean.reports.isEmpty(), "a valid alloc/store/load/free program is silent: " + clean.reports);
        check(verdicts(new ShadowMemory().run(List.of(Op.alloc(1, 16), Op.store(1, 0, 16), Op.load(1, 0, 16))))
                .isEmpty(), "writes are not mistaken for reads");

        // 4. use-after-free, on both a load and a store
        ShadowMemory afterFree = new ShadowMemory();
        afterFree.run(List.of(Op.alloc(1, 8), Op.store(1, 0, 8), Op.free(1), Op.load(1, 0, 8), Op.store(1, 0, 8)));
        check(afterFree.reports.size() == 2
                && afterFree.reports.get(0).equals("3 USE_AFTER_FREE #1")
                && afterFree.reports.get(1).equals("4 USE_AFTER_FREE #1"),
                "load and store after free are both caught: " + afterFree.reports);

        // 5. double free
        ShadowMemory doubleFree = new ShadowMemory();
        doubleFree.run(List.of(Op.alloc(1, 4), Op.free(1), Op.free(1)));
        check(doubleFree.reports.size() == 1 && doubleFree.reports.get(0).equals("2 BAD_FREE double free of #1"),
                "double free is caught: " + doubleFree.reports);

        // 6. out of bounds, past the end and before the start
        ShadowMemory overflow = new ShadowMemory();
        overflow.run(List.of(Op.alloc(1, 8), Op.store(1, 0, 8), Op.store(1, 6, 4)));
        check(overflow.reports.size() == 1 && overflow.reports.get(0).contains("OUT_OF_BOUNDS range [6, 10)"),
                "a store past the end is out of bounds: " + overflow.reports);
        ShadowMemory underflow = new ShadowMemory();
        underflow.run(List.of(Op.alloc(1, 8), Op.load(1, -1, 2)));
        check(underflow.reports.size() == 1 && underflow.reports.get(0).contains("negative offset"),
                "a negative offset is out of bounds: " + underflow.reports);
        ShadowMemory redzone = new ShadowMemory();
        redzone.run(List.of(Op.alloc(1, 8), Op.alloc(2, 8), Op.load(1, 0, 12)));
        check(redzone.reports.get(0).contains("OUT_OF_BOUNDS range"), "the range check fires before the bytes: "
                + redzone.reports);

        // 7. explicit poisoning gives uninitialized bytes a fresh origin
        ShadowMemory poisoned = new ShadowMemory();
        poisoned.run(List.of(Op.alloc(1, 8), Op.store(1, 0, 8), Op.poison(1, 2, 2), Op.load(1, 0, 8)));
        check(poisoned.reports.size() == 1 && poisoned.reports.get(0).contains("origin 1001"),
                "poisoned bytes carry the poisoning origin: " + poisoned.reports);

        // 8. a rejected store writes nothing: after an out-of-bounds store the bytes are still poisoned
        ShadowMemory rejected = new ShadowMemory();
        rejected.run(List.of(Op.alloc(1, 4), Op.store(1, 0, 4), Op.poison(1, 0, 4), Op.store(1, 3, 4),
                Op.load(1, 2, 2)));
        check(rejected.reports.size() == 2 && rejected.reports.get(0).contains("OUT_OF_BOUNDS")
                && rejected.reports.get(1).contains("UNINITIALIZED_READ"),
                "a rejected store leaves the shadow untouched: " + rejected.reports);

        // 9. differential fuzzing: identical reports over 200 generated programs
        int programs = 200, reports = 0, uninitialized = 0;
        long shadowReads = 0;
        for (int p = 0; p < programs; p++) {
            List<Op> program = generate(p, 24);
            ShadowMemory detector = new ShadowMemory();
            List<String> byDetector = verdicts(detector.run(program));
            List<String> byOracle = verdicts(ShadowOracle.run(program));
            check(byDetector.equals(byOracle), "program " + p + ": detector " + byDetector + " != oracle "
                    + byOracle + " for " + describe(program));
            reports += byDetector.size();
            uninitialized += detector.uninitializedReports();
            shadowReads += detector.shadowBytesRead;
        }
        check(reports > 200, "the corpus produces plenty of findings: " + reports);
        check(uninitialized > 50, "uninitialized reads are exercised: " + uninitialized);

        // 10. determinism of the generated corpus
        check(describe(generate(7, 24)).equals(describe(generate(7, 24))), "same seed reproduces a program");
        check(!describe(generate(7, 24)).equals(describe(generate(8, 24))), "a different seed differs");

        System.out.println("programs: " + programs + " generated, " + reports + " reports, " + uninitialized
                + " uninitialized reads");
        System.out.println("shadow bytes examined: " + shadowReads + " (1 byte of shadow per heap byte)");
        System.out.println("PASS");
    }

    static String describe(List<Op> program) {
        StringBuilder out = new StringBuilder();
        for (Op op : program) out.append(op.show()).append(';');
        return out.toString();
    }

    /**
     * Random program: allocations and frees of declared ids, and store/load operations that may
     * deliberately address freed blocks or step outside their bounds.
     */
    static List<Op> generate(long seed, int length) {
        Prng rng = new Prng(0x5EED0000L + seed);
        List<Op> program = new ArrayList<>();
        List<Integer> ids = new ArrayList<>();
        Map<Integer, Integer> sizes = new LinkedHashMap<>();
        int nextId = 1;
        for (int i = 0; i < length; i++) {
            int roll = rng.nextInt(10);
            if (roll < 3 || ids.isEmpty()) {
                int size = 4 + rng.nextInt(12);
                sizes.put(nextId, size);
                program.add(Op.alloc(nextId, size));
                ids.add(nextId);
                nextId++;
            } else if (roll == 3) {
                program.add(Op.free(ids.get(rng.nextInt(ids.size()))));
            } else {
                int id = ids.get(rng.nextInt(ids.size()));
                int size = sizes.get(id);
                int span = 1 + rng.nextInt(4);
                int offset = rng.nextInt(size + 2) - 1;              // may go out of bounds
                program.add(rng.nextInt(4) == 0 ? Op.poison(id, offset, span)
                        : (rng.nextInt(2) == 0 ? Op.store(id, offset, span) : Op.load(id, offset, span)));
            }
        }
        return program;
    }
}
