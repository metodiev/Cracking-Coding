/** Task 752 - Implement taint analysis with simple propagation for security checks.
 * (Tier: Expert, CTCI Beyond - Security & Privacy)
 *
 * The toy IR is a straight-line list of statements, each one of
 *   source(v)   v takes the attacker-controlled secret
 *   assign(v, x..)   v is computed from the listed variables
 *   sanitize(v, x)   v is the *escaped* copy of x, so the taint stops here
 *   sink(i, x..)     the value reaches a dangerous operation
 *
 * Three engines, all deterministic (splitmix64 PRNG for corpus generation, no Math.random):
 *  1. FlowSensitive  - one forward pass, so a later clean assignment kills the taint of a variable,
 *     keeping a witness chain (statement ids) for every reaching flow. Precision is exact for this IR.
 *  2. FlowInsensitive - the naive "iterate the whole rule set to a fixpoint" analysis: a variable stays
 *     tainted once any definition is tainted, and sanitizers are ignored. It is the classic
 *     over-approximation, and it is kept in order to *measure* the false alarms it costs.
 *  3. DynamicOracle   - an independent concrete interpreter that propagates a runtime tag per value and
 *     runs the program with secret = 0 and secret = 1. Because the IR has no branches, the runtime tag
 *     of every sink is a fixed property of the program, which is what makes exact agreement between
 *     static and dynamic taint checkable rather than merely hoped for.
 *
 * Simplified: linear control flow (no branches, loops or inter-procedural calls), an unknown call is
 * conservatively treated as taint-preserving, and sanitizers are assumed correct.
 *
 * Verified: hand-built positive/negative cases; every flagged sink carries a witness chain that is
 * replayed step by step; static (flow-sensitive) verdicts equal the dynamic oracle's verdicts on the
 * whole generated corpus; flow-insensitive >= flow-sensitive alarms with a strict gap; boundary cases
 * (empty program, no sink, sink without source, unused source) and ill-formed programs are rejected.
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

/** One statement of the toy IR. */
final class Stmt {
    final String kind;              // source | assign | sanitize | sink
    final String target;            // variable written, or the sink/source label
    final List<String> args;

    Stmt(String kind, String target, String... args) {
        this.kind = kind;
        this.target = target;
        this.args = List.of(args);
    }

    String show() {
        return kind + "(" + target + (args.isEmpty() ? "" : ", " + String.join(", ", args)) + ")";
    }
}

/** Flow-sensitive taint analysis with witness chains, plus the naive flow-insensitive baseline. */
final class TaintAnalysis {
    final List<Stmt> program;
    final Map<Integer, List<String>> witnesses = new LinkedHashMap<>();
    final Set<Integer> flagged = new TreeSet<>();
    long transfers;                                  // statement visits, i.e. the analysis' work

    TaintAnalysis(List<Stmt> program) { this.program = program; }

    /** Ill-formed programs are rejected instead of analysed: every operand must exist, sinks must exist. */
    static void validate(List<Stmt> program) {
        Set<String> defined = new LinkedHashSet<>();
        for (Stmt s : program) {
            for (String arg : s.args) {
                if (!defined.contains(arg))
                    throw new IllegalArgumentException("undefined variable '" + arg + "' in " + s.show());
            }
            if (s.kind.equals("source") || s.kind.equals("assign") || s.kind.equals("sanitize")) {
                defined.add(s.target);
            } else if (!s.kind.equals("sink")) {
                throw new IllegalArgumentException("unknown statement kind: " + s.show());
            }
        }
    }

    /** Forward pass: taint(v) = OR over the taint of the last definitions of v's operands. */
    Map<Integer, List<String>> analyse() {
        Map<String, Boolean> tainted = new HashMap<>();
        Map<String, List<String>> chain = new HashMap<>();
        for (int i = 0; i < program.size(); i++) {
            Stmt s = program.get(i);
            transfers++;
            if (s.kind.equals("source")) {
                tainted.put(s.target, true);
                chain.put(s.target, List.of("stmt#" + i + ": " + s.show()));
            } else if (s.kind.equals("sanitize")) {
                tainted.put(s.target, false);                       // trust the sanitizer
                chain.remove(s.target);
            } else if (s.kind.equals("assign")) {
                boolean dirty = false;
                List<String> path = new ArrayList<>();
                for (String arg : s.args) {
                    if (Boolean.TRUE.equals(tainted.get(arg))) {
                        dirty = true;
                        path = chain.getOrDefault(arg, new ArrayList<>());
                    }
                }
                tainted.put(s.target, dirty);
                if (dirty) {
                    List<String> extended = new ArrayList<>(path);
                    extended.add("stmt#" + i + ": " + s.show());
                    chain.put(s.target, extended);
                } else {
                    chain.remove(s.target);
                }
            } else {                                                // sink
                for (String arg : s.args) {
                    if (Boolean.TRUE.equals(tainted.get(arg))) {
                        flagged.add(i);
                        List<String> path = new ArrayList<>(chain.getOrDefault(arg, new ArrayList<>()));
                        path.add("stmt#" + i + ": " + s.show());
                        witnesses.putIfAbsent(i, path);
                    }
                }
            }
        }
        return witnesses;
    }

    /**
     * The naive baseline: repeat the transfer rules over the whole program until nothing changes. A
     * variable is tainted if it ever gets a tainted definition; untainting is not modelled, and neither
     * are sanitizers, so the result is a monotone over-approximation.
     */
    Set<Integer> analyseFlowInsensitive() {
        Set<String> dirty = new TreeSet<>();
        boolean changed = true;
        int rounds = 0;
        while (changed) {
            changed = false;
            rounds++;
            for (Stmt s : program) {
                transfers++;
                if (s.kind.equals("source") && dirty.add(s.target)) changed = true;
                else if (s.kind.equals("assign") && s.args.stream().anyMatch(dirty::contains)
                        && dirty.add(s.target)) changed = true;
            }
        }
        checkRounds(rounds);
        Set<Integer> sinks = new TreeSet<>();
        for (int i = 0; i < program.size(); i++) {
            Stmt s = program.get(i);
            if (s.kind.equals("sink") && s.args.stream().anyMatch(dirty::contains)) sinks.add(i);
        }
        return sinks;
    }

    private void checkRounds(int rounds) {
        if (rounds > program.size() + 1) throw new IllegalStateException("no fixpoint after " + rounds);
    }

    /** Replays a witness chain: it must start at a source, walk through real statements, end at a sink. */
    void validateWitness(int sinkIndex, List<String> path) {
        List<Stmt> chain = new ArrayList<>();
        for (String step : path) {
            int id = Integer.parseInt(step.substring(5, step.indexOf(':')));
            chain.add(program.get(id));
        }
        if (chain.isEmpty() || chain.get(chain.size() - 1) != program.get(sinkIndex))
            throw new AssertionError("FAILED: witness does not end at the sink it explains");
        if (!chain.get(0).kind.equals("source"))
            throw new AssertionError("FAILED: witness does not start at a source: " + chain.get(0).show());
        Set<String> carried = new LinkedHashSet<>();
        carried.add(chain.get(0).target);
        for (int i = 1; i < chain.size(); i++) {
            Stmt step = chain.get(i);
            boolean linked = i == chain.size() - 1
                    ? step.args.stream().anyMatch(carried::contains)
                    : step.args.stream().anyMatch(carried::contains);
            if (!linked) throw new AssertionError("FAILED: witness step " + step.show() + " breaks the chain");
            carried.add(step.target);
        }
    }
}

/** Independent oracle: run the program concretely and propagate a runtime tag per value. */
final class DynamicOracle {
    static Map<Integer, Boolean> sinks(List<Stmt> program, int secret) {
        Map<String, Integer> value = new HashMap<>();
        Map<String, Boolean> tag = new HashMap<>();
        Map<Integer, Boolean> taintedSinks = new LinkedHashMap<>();
        for (int i = 0; i < program.size(); i++) {
            Stmt s = program.get(i);
            switch (s.kind) {
                case "source":
                    value.put(s.target, secret);
                    tag.put(s.target, true);
                    break;
                case "assign": {
                    int v = 1;
                    boolean dirty = false;
                    for (String arg : s.args) {
                        v = v * 31 + value.getOrDefault(arg, 0);
                        dirty |= Boolean.TRUE.equals(tag.get(arg));
                    }
                    value.put(s.target, v);
                    tag.put(s.target, dirty);
                    break;
                }
                case "sanitize":
                    value.put(s.target, value.getOrDefault(s.args.get(0), 0) & 0x7F);
                    tag.put(s.target, false);
                    break;
                default: {
                    boolean dirty = false;
                    for (String arg : s.args) dirty |= Boolean.TRUE.equals(tag.get(arg));
                    taintedSinks.put(i, dirty);
                }
            }
        }
        return taintedSinks;
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static List<Stmt> program(Stmt... statements) { return new ArrayList<>(List.of(statements)); }

    static Set<Integer> flowSensitive(List<Stmt> p) {
        TaintAnalysis analysis = new TaintAnalysis(p);
        Map<Integer, List<String>> witnesses = analysis.analyse();
        for (Map.Entry<Integer, List<String>> entry : witnesses.entrySet()) {
            analysis.validateWitness(entry.getKey(), entry.getValue());
        }
        return analysis.flagged;
    }

    static Set<Integer> dynamic(List<Stmt> p) {
        Set<Integer> zero = new TreeSet<>(), one = new TreeSet<>();
        DynamicOracle.sinks(p, 0).forEach((index, dirty) -> { if (dirty) zero.add(index); });
        DynamicOracle.sinks(p, 1).forEach((index, dirty) -> { if (dirty) one.add(index); });
        check(zero.equals(one), "the runtime tag of a sink does not depend on the secret's value");
        return zero;
    }

    /** Deterministic corpus of random straight-line programs over four variables. */
    static List<List<Stmt>> corpus(int count, long seed) {
        Prng rng = new Prng(seed);
        List<List<Stmt>> programs = new ArrayList<>();
        for (int c = 0; c < count; c++) {
            int length = 4 + rng.nextInt(8);
            List<Stmt> p = new ArrayList<>();
            for (int i = 0; i < 4; i++) p.add(new Stmt("assign", "v" + i));   // constants, so operands exist
            for (int i = 0; i < length; i++) {
                String target = "v" + rng.nextInt(4);
                int kind = rng.nextInt(4);
                if (kind == 0) {
                    p.add(new Stmt("source", target));
                } else if (kind == 1 && !p.isEmpty()) {
                    String arg = "v" + rng.nextInt(4);
                    p.add(new Stmt("assign", target, arg, "v" + rng.nextInt(4)));
                } else if (kind == 2) {
                    p.add(new Stmt("sanitize", target, "v" + rng.nextInt(4)));
                } else {
                    p.add(new Stmt("sink", "log", target));
                }
            }
            TaintAnalysis.validate(p);
            programs.add(p);
        }
        return programs;
    }

    public static void main(String[] args) {
        // 1. hand-built positive and negative cases
        List<Stmt> direct = program(
                new Stmt("source", "a"),
                new Stmt("assign", "b", "a"),
                new Stmt("sink", "exec", "b"));
        check(flowSensitive(direct).equals(new TreeSet<>(List.of(2))), "source -> copy -> sink is flagged");
        check(flowSensitive(direct).equals(dynamic(direct)), "static and dynamic agree on the direct flow");

        List<Stmt> sanitized = program(
                new Stmt("source", "a"),
                new Stmt("sanitize", "b", "a"),
                new Stmt("sink", "exec", "b"));
        check(flowSensitive(sanitized).isEmpty(), "a sanitized value is not flagged");
        check(dynamic(sanitized).isEmpty(), "the sanitized flow is also clean at runtime");

        List<Stmt> overwritten = program(
                new Stmt("source", "a"),
                new Stmt("assign", "a", "c", "c"),
                new Stmt("sink", "exec", "a"));
        check(flowSensitive(overwritten).isEmpty(), "a later clean definition kills the taint");
        check(new TaintAnalysis(overwritten).analyseFlowInsensitive().equals(new TreeSet<>(List.of(2))),
                "the flow-insensitive baseline still warns about the overwritten value");

        List<Stmt> partial = program(
                new Stmt("source", "a"),
                new Stmt("sanitize", "b", "a"),
                new Stmt("sink", "sql", "a", "b"));
        check(flowSensitive(partial).equals(new TreeSet<>(List.of(2))),
                "one tainted operand is enough to taint the call");

        List<Stmt> unknownCall = program(
                new Stmt("source", "a"),
                new Stmt("assign", "b", "a"),
                new Stmt("assign", "c", "b"),
                new Stmt("assign", "d", "c"),
                new Stmt("sink", "net", "d"));
        check(flowSensitive(unknownCall).equals(new TreeSet<>(List.of(4))), "long chains keep the taint");

        // 2. boundary cases
        check(flowSensitive(program()).isEmpty(), "empty program has no flow");
        check(flowSensitive(program(new Stmt("source", "a"), new Stmt("sink", "exec", "a")))
                .equals(new TreeSet<>(List.of(1))), "a direct source-to-sink statement is flagged");
        check(flowSensitive(program(new Stmt("assign", "a", "b"), new Stmt("sink", "exec", "a"))).isEmpty(),
                "a sink with no source behind it is clean");
        List<Stmt> deadSource = program(new Stmt("source", "a"), new Stmt("sink", "exec", "b"));
        check(flowSensitive(deadSource).isEmpty(), "an unused source never reaches a sink");
        List<Stmt> noSink = program(new Stmt("source", "a"), new Stmt("assign", "b", "a"));
        check(flowSensitive(noSink).isEmpty() && dynamic(noSink).isEmpty(),
                "an untriggered source is not reported");

        // 3. rejected ill-formed programs
        int rejected = 0;
        try {
            TaintAnalysis.validate(program(new Stmt("sink", "exec", "ghost")));
        } catch (IllegalArgumentException e) {
            rejected++;
        }
        try {
            TaintAnalysis.validate(program(new Stmt("teleport", "a")));
        } catch (IllegalArgumentException e) {
            rejected++;
        }
        check(rejected == 2, "undefined operands and unknown kinds are rejected (" + rejected + "/2)");

        // 4. exact agreement with the independent oracle on the whole corpus
        List<List<Stmt>> programs = corpus(400, 0xBADC0DE5EEDL);
        int sinks = 0, falseNegatives = 0;
        long work = 0;
        for (List<Stmt> p : programs) {
            TaintAnalysis analysis = new TaintAnalysis(p);
            analysis.analyse();
            work += analysis.transfers;
            Set<Integer> staticVerdict = analysis.flagged;
            Set<Integer> dynamicVerdict = dynamic(p);
            sinks += dynamicVerdict.size();
            for (int i = 0; i < p.size(); i++) {
                if (!staticVerdict.contains(i) && dynamicVerdict.contains(i)) falseNegatives++;
            }
            check(staticVerdict.equals(dynamicVerdict),
                    "flow-sensitive analysis is exact on " + p.size() + "-statement program");
        }
        check(falseNegatives == 0, "no false negatives over " + programs.size() + " programs");
        check(sinks > 100, "corpus exercises tainted sinks: " + sinks);

        // 5. flow-insensitive is a strict over-approximation
        int baselineWarnings = 0, flowSensitiveWarnings = 0, extra = 0;
        for (List<Stmt> p : programs) {
            TaintAnalysis analysis = new TaintAnalysis(p);
            Set<Integer> exact = new TreeSet<>(analysis.analyse().keySet());
            Set<Integer> naive = analysis.analyseFlowInsensitive();
            flowSensitiveWarnings += exact.size();
            baselineWarnings += naive.size();
            for (int i : naive) if (!exact.contains(i)) extra++;
            check(naive.containsAll(exact), "the baseline never misses what the exact analysis finds");
            for (int i : extra(naive, exact)) {
                check(!dynamic(p).contains(i), "the extra baseline warning is a false alarm at runtime");
            }
        }
        check(baselineWarnings > flowSensitiveWarnings,
                "baseline warns more often: " + baselineWarnings + " vs " + flowSensitiveWarnings);
        check(extra > 20, "the gap is material: " + extra + " extra alarms");

        // 6. determinism of the corpus and of the analysis work
        List<List<Stmt>> replay = corpus(400, 0xBADC0DE5EEDL);
        check(replay.get(0).size() == programs.get(0).size()
                && replay.get(0).get(0).show().equals(programs.get(0).get(0).show()),
                "same seed reproduces the corpus");
        long replayWork = 0;
        for (List<Stmt> p : replay) {
            TaintAnalysis analysis = new TaintAnalysis(p);
            analysis.analyse();
            replayWork += analysis.transfers;
        }
        check(replayWork == work, "analysis work is deterministic (" + work + " transfers)");

        System.out.println("corpus: " + programs.size() + " programs, " + sinks + " runtime-tainted sinks");
        System.out.println("flow-sensitive alarms: " + flowSensitiveWarnings + ", flow-insensitive baseline: "
                + baselineWarnings + " (" + extra + " false alarms removed)");
        System.out.println("statement transfers: " + work + ", false negatives: " + falseNegatives);
        System.out.println("PASS");
    }

    static Set<Integer> extra(Set<Integer> naive, Set<Integer> exact) {
        Set<Integer> out = new TreeSet<>(naive);
        out.removeAll(exact);
        return out;
    }
}
