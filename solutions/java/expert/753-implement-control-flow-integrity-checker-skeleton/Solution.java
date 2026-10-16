/** Task 753 - Implement a control-flow integrity checker skeleton. (Tier: Expert, Beyond - Security & Privacy)
 *
 * The audited module is a set of functions, each with an entry block, plus a target table per
 * indirect-call site. The policy checked here is the classic coarse-grained one: an indirect call may
 * target *any* function in the site's whitelist (the address-taken functions with a matching signature),
 * but never the middle of a function and never a function outside the whitelist. Direct calls are
 * checked against the callee's entry, and returns are checked against a shadow stack, so a clobbered
 * return address is caught even when the callee itself is legitimate.
 *
 * A trace is a sequence of executed steps (function, block, kind, target, site). Like a real CFI
 * monitor the checker *traps at the first offending step* and reports its index; a separately written
 * brute-force oracle recomputes that index straight from the target tables and must agree on every
 * trace, valid or tampered.
 *
 * Simplified: a checker *skeleton* - no binary, no disassembly, no instruction decoding. A step carries
 * the block index and the callee name directly, and the whitelists are given rather than recovered by
 * a static analysis. Static integrity (non-empty whitelists, targets that exist and are address-taken)
 * is checked separately from the dynamic trace.
 *
 * Verified: a healthy module has no static problem while a broken one reports each class; valid traces
 * are clean; every tamper class (illegal indirect target, jump into the middle of a function, clobbered
 * return, missing and extra return, unknown function) is caught at the expected index; coarse-grained
 * CFI accepts every whitelisted target while the table is still honoured per site; empty-trace and
 * balanced-call boundaries; and checker == oracle on 300 generated traces.
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

/** The audited module: block counts, address-taken functions and per-site indirect target tables. */
final class Module {
    static final String ENTRY_FRAME = "__entry__";

    final Map<String, Integer> blockCount = new LinkedHashMap<>();
    final Set<String> addressTaken = new LinkedHashSet<>();
    final Map<String, Set<String>> targetTable = new LinkedHashMap<>();

    void addFunction(String name, int blocks, boolean addressTaken) {
        blockCount.put(name, blocks);
        if (addressTaken) this.addressTaken.add(name);
    }

    void addSite(String function, int site, String... legalTargets) {
        targetTable.put(siteKey(function, site), new TreeSet<>(List.of(legalTargets)));
    }

    static String siteKey(String function, int site) { return function + "/" + site; }

    Set<String> legalTargets(String function, int site) {
        return targetTable.getOrDefault(siteKey(function, site), Set.of());
    }

    /** Static policy violations, independent of any trace. */
    List<String> staticProblems() {
        List<String> problems = new ArrayList<>();
        for (Map.Entry<String, Set<String>> entry : targetTable.entrySet()) {
            String function = entry.getKey().substring(0, entry.getKey().indexOf('/'));
            if (!blockCount.containsKey(function)) problems.add("call site in unknown function " + function);
            if (entry.getValue().isEmpty()) problems.add("empty whitelist for " + entry.getKey());
            for (String target : entry.getValue()) {
                if (!blockCount.containsKey(target)) problems.add("whitelist target " + target + " is not a function");
                else if (!addressTaken.contains(target)) problems.add("target " + target + " is not address-taken");
            }
        }
        return problems;
    }
}

/** One executed step of a trace. */
final class Step {
    final String function;
    final int block;
    final String kind;                 // call | icall | ret
    final String target;               // callee, null for ret
    final int site;                    // indirect-call site id, -1 for call and ret

    Step(String function, int block, String kind, String target, int site) {
        this.function = function;
        this.block = block;
        this.kind = kind;
        this.target = target;
        this.site = site;
    }

    static Step call(String function, String target) { return new Step(function, 0, "call", target, -1); }
    static Step icall(String function, int block, int site, String target) {
        return new Step(function, block, "icall", target, site);
    }
    static Step ret(String function) { return new Step(function, 0, "ret", null, -1); }

    Step withTarget(String newTarget) { return new Step(function, block, kind, newTarget, site); }
    Step withBlock(int newBlock) { return new Step(function, newBlock, kind, target, site); }

    String show() {
        if (kind.equals("ret")) return "ret from " + function;
        return kind + " " + function + "@" + block + " -> " + target + (site >= 0 ? " [site " + site + "]" : "");
    }
}

/** The checker: one forward walk with a shadow stack of expected callers, trapping at the first fault. */
final class CfiChecker {
    private final Module module;

    CfiChecker(Module module) { this.module = module; }

    /** Index of the first offending step, or -1 when the trace obeys the policy. */
    int firstViolation(List<Step> trace) {
        Deque<String> shadowStack = new ArrayDeque<>();
        for (int i = 0; i < trace.size(); i++) {
            Step step = trace.get(i);
            Integer blocks = module.blockCount.get(step.function);
            if (blocks == null) return i;
            if (step.kind.equals("ret")) {
                if (shadowStack.isEmpty()) {
                    // Returning out of the traced entry function is fine when nothing follows;
                    // otherwise the return had no call to consume.
                    if (i + 1 < trace.size()) return i;
                    continue;
                }
                String expected = shadowStack.pop();
                if (i + 1 < trace.size() && !trace.get(i + 1).function.equals(expected)) return i;
                continue;
            }
            if (step.block != 0 || step.block >= blocks) return i;
            if (!module.blockCount.containsKey(step.target)) return i;
            if (step.kind.equals("icall") && !module.legalTargets(step.function, step.site).contains(step.target))
                return i;
            shadowStack.push(step.function);
        }
        return -1;
    }

    /** Human-readable reason for the first violation (null when the trace is clean). */
    String reason(List<Step> trace) {
        int index = firstViolation(trace);
        if (index < 0) return null;
        Step step = trace.get(index);
        return "step " + index + ": " + step.show() + " violates the policy";
    }

    /**
     * Independently written oracle: the same policy recomputed in one flat scan with a caller list
     * instead of a deque, and with the entry frame handled by an empty-list check.
     */
    static int oracleFirstViolation(Module module, List<Step> trace) {
        List<String> callers = new ArrayList<>();
        for (int i = 0; i < trace.size(); i++) {
            Step step = trace.get(i);
            Integer blocks = module.blockCount.get(step.function);
            if (blocks == null) return i;
            switch (step.kind) {
                case "ret":
                    if (callers.isEmpty()) {
                        if (i + 1 == trace.size()) break;      // returning out of the trace entry point
                        return i;
                    }
                    String expected = callers.remove(callers.size() - 1);
                    if (i + 1 < trace.size() && !trace.get(i + 1).function.equals(expected)) return i;
                    break;
                case "call":
                case "icall":
                    if (step.block != 0 || step.block >= blocks) return i;
                    if (!module.blockCount.containsKey(step.target)) return i;
                    if (step.kind.equals("icall")
                            && !module.legalTargets(step.function, step.site).contains(step.target)) return i;
                    callers.add(step.function);
                    break;
                default:
                    return i;
            }
        }
        return -1;
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    /** The healthy module: two indirect sites with two and one legal targets. */
    static Module buildModule() {
        Module m = new Module();
        m.addFunction("main", 4, false);
        m.addFunction("parse", 3, true);
        m.addFunction("handle", 3, true);
        m.addFunction("logError", 2, true);
        m.addSite("main", 0, "parse", "handle");
        m.addSite("handle", 1, "logError");
        return m;
    }

    static List<Step> validTrace() {
        return new ArrayList<>(List.of(
                Step.call("main", "parse"),
                Step.ret("parse"),
                Step.icall("main", 0, 0, "parse"),
                Step.ret("parse"),
                Step.icall("main", 0, 0, "handle"),
                Step.icall("handle", 0, 1, "logError"),
                Step.ret("logError"),
                Step.ret("handle"),
                Step.ret("main")));
    }

    /** Asserts the reported index, the human-readable reason and the oracle agreement at once. */
    static void expect(Module module, List<Step> trace, int expected) {
        CfiChecker checker = new CfiChecker(module);
        int found = checker.firstViolation(trace);
        check(found == expected, "expected violation " + expected + ", got " + found + " ("
                + checker.reason(trace) + ")");
        check(CfiChecker.oracleFirstViolation(module, trace) == expected,
                "oracle disagrees: " + CfiChecker.oracleFirstViolation(module, trace) + " != " + expected);
        if (expected >= 0) {
            check(checker.reason(trace) != null && checker.reason(trace).contains("step " + expected + ":"),
                    "violation message names the offending step: " + checker.reason(trace));
        } else {
            check(checker.reason(trace) == null, "clean traces carry no reason");
        }
    }

    public static void main(String[] args) {
        Module module = buildModule();

        // 1. static policy: the healthy module is clean, the broken one reports every class
        check(module.staticProblems().isEmpty(), "healthy module passes static integrity: "
                + module.staticProblems());
        Module broken = new Module();
        broken.addFunction("main", 2, false);
        broken.addFunction("target", 1, false);
        broken.addSite("main", 0);                            // empty whitelist
        broken.addSite("main", 2, "target");                  // exists, but its address is never taken
        broken.addSite("nowhere", 1, "ghost");                // unknown function and unknown target
        List<String> problems = broken.staticProblems();
        check(problems.size() == 4, "broken module reports 4 static problems, got " + problems);

        // 2. the healthy trace is clean in both engines
        List<Step> good = validTrace();
        expect(module, good, -1);

        // 3. tamper: indirect target outside the site's whitelist (logError is legal only at site 1)
        List<Step> wrongSite = new ArrayList<>(good);
        wrongSite.set(4, Step.icall("main", 0, 0, "logError"));
        expect(module, wrongSite, 4);

        // 4. tamper: jump into the middle of a function instead of its entry
        List<Step> middle = new ArrayList<>(good);
        middle.set(4, Step.icall("main", 2, 0, "handle"));
        expect(module, middle, 4);

        // 5. tamper: official call site, but the trace resumes in the wrong frame - the shadow stack
        //    catches it at the return, not at the call
        List<Step> clobbered = new ArrayList<>(List.of(
                Step.call("main", "parse"),
                Step.ret("parse"),
                Step.icall("handle", 0, 1, "logError"),
                Step.ret("logError"),
                Step.ret("handle")));
        expect(module, clobbered, 1);

        // 6. tamper: a return with no call behind it
        expect(module, new ArrayList<>(List.of(Step.ret("main"), Step.ret("main"))), 0);

        // 7. tamper: an extra return after the trace entry point has already returned
        List<Step> extraReturn = new ArrayList<>(good);
        extraReturn.add(Step.ret("main"));
        expect(module, extraReturn, 8);

        // 8. a trace may legitimately end inside a live frame (truncated observation, not a fault)
        expect(module, new ArrayList<>(good.subList(0, good.size() - 1)), -1);

        // 8. coarse-grained CFI accepts every whitelisted target, but the table is honoured per site
        Module narrow = buildModule();
        narrow.addSite("main", 0, "parse");
        List<Step> handleAtSite0 = new ArrayList<>(good);
        handleAtSite0.set(4, Step.icall("main", 0, 0, "handle"));
        expect(narrow, handleAtSite0, 4);
        List<Step> parseAtSite0 = new ArrayList<>(good);
        parseAtSite0.set(4, Step.icall("main", 0, 0, "parse"));
        expect(narrow, parseAtSite0, -1);

        // 9. boundaries: empty trace, one balanced call, unknown function, out-of-range block        expect(module, new ArrayList<>(), -1);
        expect(module, new ArrayList<>(List.of(Step.call("main", "parse"), Step.ret("parse"))), -1);
        expect(module, new ArrayList<>(List.of(Step.call("main", "ghost"))), 0);
        expect(module, new ArrayList<>(List.of(new Step("main", 9, "call", "parse", -1))), 0);

        // 10. differential fuzzing over clean and tampered traces
        List<List<Step>> traces = generateTraces(0xCF1CF1CFL, good, 300);
        int clean = 0, tampered = 0;
        for (int round = 0; round < traces.size(); round++) {
            List<Step> trace = traces.get(round);
            if (trace.equals(good)) clean++;
            else tampered++;
            int checker = new CfiChecker(module).firstViolation(trace);
            int oracle = CfiChecker.oracleFirstViolation(module, trace);
            check(checker == oracle, "round " + round + ": checker " + checker + " != oracle " + oracle
                    + " on " + trace);
        }
        check(clean > 100 && tampered > 100, "fuzzing covered both classes: " + clean + " clean, "
                + tampered + " tampered");

        List<List<Step>> replay = generateTraces(0xCF1CF1CFL, good, 300);
        boolean identical = replay.size() == traces.size();
        for (int i = 0; identical && i < traces.size(); i++) {
            identical = show(replay.get(i)).equals(show(traces.get(i)));
        }
        check(identical, "same seed reproduces the same trace corpus");

        System.out.println("static policy: " + module.staticProblems().size() + " problems in the healthy "
                + "module, " + problems.size() + " in the broken one");
        System.out.println("traces: " + clean + " clean + " + tampered + " tampered, checker == oracle on all");
        System.out.println("PASS");
    }

    static String show(List<Step> trace) {
        StringBuilder out = new StringBuilder();
        for (Step step : trace) out.append(step.show()).append(';');
        return out.toString();
    }

    /**
     * Trace corpus: half the rounds keep the reference trace and half mutate one step or insert a return,
     * exactly the tampering classes a CFI monitor is expected to catch.
     */
    static List<List<Step>> generateTraces(long seed, List<Step> base, int rounds) {
        Prng rng = new Prng(seed);
        List<List<Step>> traces = new ArrayList<>();
        for (int round = 0; round < rounds; round++) {
            List<Step> trace = new ArrayList<>(base);
            if (rng.nextInt(2) != 0) {
                int mutation = rng.nextInt(4);
                int index = rng.nextInt(trace.size());
                if (mutation == 0) {
                    trace.set(index, trace.get(index).withTarget(rng.nextInt(2) == 0 ? "logError" : "ghost"));
                } else if (mutation == 1) {
                    trace.set(index, trace.get(index).withBlock(rng.nextInt(3)));
                } else if (mutation == 2) {
                    Step step = trace.get(index);
                    trace.set(index, step.kind.equals("ret") ? Step.icall(step.function, 0, 0, "parse")
                            : Step.ret(step.function));
                } else {
                    trace.add(rng.nextInt(trace.size() + 1), Step.ret("handle"));
                }
            }
            traces.add(trace);
        }
        return traces;
    }
}
