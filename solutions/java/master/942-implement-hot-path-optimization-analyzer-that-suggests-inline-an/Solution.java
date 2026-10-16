/** Task 942 - Implement hot-path optimization analyzer that suggests inline and caching opportunities.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A profiler-driven optimizer for a call graph. Every function carries its code size, call count, call
 *  sites, per-call CPU cost, observed distinct argument tuples and a purity flag. The analyzer prices the
 *  two classic hot-path rewrites - inlining a small hot callee (saving the per-call overhead, paying code
 *  growth) and memoising a pure hot callee with repetitive arguments (saving the recomputation, paying
 *  cache entries) - keeps the better rewrite per function, ranks all rewrites by benefit and accepts them
 *  greedily while the code-growth and cache budgets allow. A plan is then applied to a copy of the cost
 *  model and the total recomputed from scratch, so the "predicted" saving is never taken on faith.
 *  Simplified: a flat call graph with one level of inlining, constant per-byte code-growth penalties and
 *  no cache-eviction modelling.
 *  Verified: inlining never fires on recursive, oversized or cold functions, memoisation never fires on
 *  impure or non-repetitive ones, budgets are never exceeded, the plan ranking matches a brute-force sort,
 *  and applying a plan lowers the recomputed total by exactly the predicted amount.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final double CALL_OVERHEAD = 12.0;       // ns paid per invocation, saved by inlining
    static final double HASH_COST = 40.0;           // ns to hash the arguments on a memoisation miss
    static final double LOOKUP_COST = 8.0;          // ns to hit the memoisation cache
    static final double GROWTH_PENALTY = 0.5;       // ns of instruction-cache pressure per added byte
    static final double ENTRY_BYTES = 32.0;         // cache footprint of one memoised argument tuple
    static final int MIN_CALLS = 50_000;
    static final int MAX_SITES = 4;
    static final int INLINE_LIMIT = 64;             // bytes of code worth duplicating

    static final class Function {
        final String name;
        final int size, callSites;
        final long calls, uniqueArgs;
        final double selfCost;
        final boolean pure, recursive;

        Function(String name, int size, int callSites, long calls, long uniqueArgs, double selfCost,
                 boolean pure, boolean recursive) {
            this.name = name; this.size = size; this.callSites = callSites; this.calls = calls;
            this.uniqueArgs = uniqueArgs; this.selfCost = selfCost; this.pure = pure; this.recursive = recursive;
        }

        double baseCost() { return calls * (CALL_OVERHEAD + selfCost); }
    }

    static final class Suggestion {
        final String function, action, reason;
        final double benefit, resource;
        Suggestion(String function, String action, double benefit, double resource, String reason) {
            this.function = function; this.action = action; this.benefit = benefit; this.resource = resource;
            this.reason = reason;
        }
    }

    static double inlineBenefit(Function f) {
        double growth = (double) f.size * f.callSites;
        return f.calls * CALL_OVERHEAD - growth * GROWTH_PENALTY;
    }

    static double memoBenefit(Function f) {
        double memo = f.uniqueArgs * (CALL_OVERHEAD + f.selfCost + HASH_COST)
                + (f.calls - f.uniqueArgs) * LOOKUP_COST;
        return f.baseCost() - memo;
    }

    /** Why a rewrite is not allowed, independent of how profitable it looks. */
    static String inlineBlocker(Function f) {
        if (f.recursive) return "recursive";
        if (f.size > INLINE_LIMIT) return "too-large";
        if (f.callSites > MAX_SITES) return "too-many-sites";
        if (f.calls < MIN_CALLS) return "cold";
        return "";
    }

    static String memoBlocker(Function f) {
        if (!f.pure) return "impure";
        if (f.calls < MIN_CALLS) return "cold";
        if (f.uniqueArgs >= f.calls) return "no-repetition";
        if (memoBenefit(f) <= 0) return "unprofitable";
        return "";
    }

    static final class Analyzer {
        final List<Function> functions;
        final double growthBudget, cacheBudget;
        final List<Suggestion> ranked = new ArrayList<>();
        final List<Suggestion> accepted = new ArrayList<>();
        final List<String> rejected = new ArrayList<>();

        Analyzer(List<Function> functions, double growthBudget, double cacheBudget) {
            this.functions = functions;
            this.growthBudget = growthBudget;
            this.cacheBudget = cacheBudget;
            for (Function f : functions) {
                String inlineBlock = inlineBlocker(f);
                String memoBlock = memoBlocker(f);
                double inline = inlineBlock.isEmpty() ? inlineBenefit(f) : Double.NEGATIVE_INFINITY;
                double memo = memoBlock.isEmpty() ? memoBenefit(f) : Double.NEGATIVE_INFINITY;
                if (inline <= 0 && memo <= 0) {
                    rejected.add(f.name + ": " + (inlineBlock.isEmpty() ? memoBlock : inlineBlock));
                    continue;
                }
                if (inline >= memo) {
                    ranked.add(new Suggestion(f.name, "inline", inline, (double) f.size * f.callSites,
                            inlineBlock.isEmpty() ? "" : inlineBlock));
                } else {
                    ranked.add(new Suggestion(f.name, "memoize", memo, f.uniqueArgs * ENTRY_BYTES, memoBlock));
                }
            }
            ranked.sort((a, b) -> {
                int byBenefit = Double.compare(b.benefit, a.benefit);
                return byBenefit != 0 ? byBenefit : a.function.compareTo(b.function);
            });
            double growth = 0, cache = 0;
            for (Suggestion s : ranked) {
                boolean inline = s.action.equals("inline");
                double nextGrowth = growth + (inline ? s.resource : 0);
                double nextCache = cache + (inline ? 0 : s.resource);
                if (nextGrowth > growthBudget || nextCache > cacheBudget) {
                    rejected.add(s.function + ": budget");
                    continue;
                }
                growth = nextGrowth;
                cache = nextCache;
                accepted.add(s);
            }
        }

        double predictedTotal() {
            double total = 0;
            for (Function f : functions) total += f.baseCost();
            for (Suggestion s : accepted) total -= s.benefit;
            return total;
        }

        String rejectedLog() { return String.join("; ", rejected); }
    }

    /** The same cost model, mutated by a plan - the independent recomputation of the predicted total. */
    static double appliedTotal(List<Function> functions, List<Suggestion> plan) {
        Map<String, Suggestion> byName = new TreeMap<>();
        for (Suggestion s : plan) byName.put(s.function, s);
        double total = 0;
        for (Function f : functions) {
            Suggestion s = byName.get(f.name);
            if (s == null) total += f.baseCost();
            else if (s.action.equals("inline")) {
                // the per-call overhead disappears, the duplicated code still pressures the i-cache
                total += f.calls * f.selfCost + (double) f.size * f.callSites * GROWTH_PENALTY;
            } else {
                total += f.uniqueArgs * (CALL_OVERHEAD + f.selfCost + HASH_COST)
                        + (f.calls - f.uniqueArgs) * LOOKUP_COST;
            }
        }
        return total;
    }

    static List<Function> fixture() {
        List<Function> functions = new ArrayList<>();
        functions.add(new Function("json.decode", 220, 3, 120_000, 500, 900.0, true, false));
        functions.add(new Function("is_valid", 6, 2, 400_000, 1, 30.0, true, false));
        functions.add(new Function("render.template", 800, 5, 20_000, 20_000, 4_000.0, false, false));
        functions.add(new Function("traverse_graph", 30, 1, 90_000, 90_000, 60.0, false, true));
        functions.add(new Function("checksum", 10, 1, 5_000, 5_000, 20.0, true, false));
        functions.add(new Function("hash.key", 12, 4, 300_000, 300_000, 45.0, true, false));
        return functions;
    }

    static Suggestion find(List<Suggestion> plan, String name) {
        for (Suggestion s : plan) if (s.function.equals(name)) return s;
        return null;
    }

    static boolean rejected(String log, String prefix) {
        for (String entry : log.split("; ")) if (entry.startsWith(prefix)) return true;
        return false;
    }

    public static void main(String[] args) {
        List<Function> functions = fixture();
        Analyzer analyzer = new Analyzer(functions, 100, 32_768);

        double baseline = 0;
        for (Function f : functions) baseline += f.calls * (CALL_OVERHEAD + f.selfCost);
        check(baseline > 200e6 && analyzer.predictedTotal() < baseline
                        && Math.abs(analyzer.predictedTotal() - appliedTotal(functions, analyzer.accepted)) < 1e-6,
                "the predicted total must equal a from-scratch recomputation: " + analyzer.predictedTotal()
                        + " vs " + appliedTotal(functions, analyzer.accepted));

        check(analyzer.accepted.size() == 3 && analyzer.accepted.get(0).function.equals("json.decode")
                        && analyzer.accepted.get(0).action.equals("memoize")
                        && analyzer.accepted.get(1).function.equals("is_valid")
                        && analyzer.accepted.get(2).function.equals("hash.key")
                        && analyzer.accepted.get(2).action.equals("inline"),
                "the plan must hold the three profitable rewrites in benefit order: " + analyzer.accepted.size()
                        + " of " + analyzer.ranked.size());

        int orderViolations = 0;
        for (int i = 1; i < analyzer.ranked.size(); i++) {
            if (analyzer.ranked.get(i).benefit > analyzer.ranked.get(i - 1).benefit) orderViolations++;
        }
        List<Suggestion> reference = new ArrayList<>(analyzer.ranked);
        reference.sort((a, b) -> {
            int byBenefit = Double.compare(b.benefit, a.benefit);
            return byBenefit != 0 ? byBenefit : a.function.compareTo(b.function);
        });
        check(orderViolations == 0 && reference.equals(new ArrayList<>(analyzer.ranked)),
                "the ranking must match a brute-force sort by benefit");

        int predicateViolations = 0;
        for (Function f : functions) {
            boolean predictedInline = inlineBlocker(f).isEmpty() && inlineBenefit(f) > 0
                    && inlineBenefit(f) >= memoBenefit(f);
            boolean predictedMemo = !predictedInline && memoBlocker(f).isEmpty() && memoBenefit(f) > 0;
            Suggestion accepted = find(analyzer.accepted, f.name);
            if (predictedInline || predictedMemo) {
                String expected = predictedInline ? "inline" : "memoize";
                if (accepted == null || !accepted.action.equals(expected)) predicateViolations++;
            } else if (accepted != null || !rejected(analyzer.rejectedLog(), f.name)) {
                predicateViolations++;
            }
        }
        check(predicateViolations == 0,
                predicateViolations + " rewrites disagree with an independent re-derivation of the rules");

        check(rejected(analyzer.rejectedLog(), "render.template") && rejected(analyzer.rejectedLog(), "traverse_graph")
                        && rejected(analyzer.rejectedLog(), "checksum")
                        && find(analyzer.ranked, "render.template") == null,
                "impure, recursive and cold functions must be rejected with a reason: " + analyzer.rejectedLog());

        double growth = 0, cache = 0;
        for (Suggestion s : analyzer.accepted) {
            if (s.action.equals("inline")) growth += s.resource;
            else cache += s.resource;
        }
        check(growth <= 100 && cache <= 32_768 && growth == 48 && Math.abs(cache - 16_032) < 1e-6,
                "the plan must stay inside both budgets: growth " + growth + ", cache " + cache);

        double applied = appliedTotal(functions, analyzer.accepted);
        check(applied < baseline && Math.abs((baseline - applied) - analyzer.accepted.stream()
                        .mapToDouble(s -> s.benefit).sum()) < 1e-6,
                "applying the plan must lower the recomputed total by the predicted saving: " + baseline + " -> "
                        + applied);

        Analyzer tight = new Analyzer(functions, 0, 32_768);
        check(tight.accepted.size() == 2 && exhausted(tight, "inline") == 0
                        && appliedTotal(functions, tight.accepted) < baseline
                        && rejected(tight.rejectedLog(), "hash.key"),
                "a zero code-growth budget must drop the inlining rewrites only: " + tight.rejectedLog());

        Analyzer starved = new Analyzer(functions, 100, 0);
        check(exhausted(starved, "memoize") == 0 && starved.accepted.size() == 1
                        && starved.accepted.get(0).function.equals("hash.key"),
                "a zero cache budget must drop the memoisation rewrites only: " + starved.rejectedLog());

        Analyzer empty = new Analyzer(List.of(), 100, 1024);
        check(empty.ranked.isEmpty() && empty.accepted.isEmpty() && empty.predictedTotal() == 0
                        && empty.rejectedLog().isEmpty(),
                "an empty call graph must yield an empty plan");

        List<Function> solo = List.of(new Function("only", 200, 1, 10, 10, 5.0, false, false));
        Analyzer soloPlan = new Analyzer(solo, 100, 1024);
        check(soloPlan.accepted.isEmpty() && soloPlan.predictedTotal() == solo.get(0).baseCost(),
                "a single large cold function must be left alone");

        Analyzer replay = new Analyzer(fixture(), 100, 32_768);
        check(replay.accepted.size() == analyzer.accepted.size()
                        && replay.predictedTotal() == analyzer.predictedTotal()
                        && replay.rejectedLog().equals(analyzer.rejectedLog()),
                "the analyzer must be reproducible");

        System.out.println("hot paths: " + functions.size() + " functions, baseline "
                + String.format("%.1f", baseline / 1e6) + " ms, optimised "
                + String.format("%.1f", applied / 1e6) + " ms (-"
                + String.format("%.1f", 100 * (baseline - applied) / baseline) + "%)");
        System.out.println("plan: " + analyzer.accepted.size() + " rewrites (" + growth + " bytes, " + cache
                + " cache bytes), rejected " + analyzer.rejectedLog());
        System.out.println("PASS");
    }

    static int exhausted(Analyzer analyzer, String action) {
        int count = 0;
        for (Suggestion s : analyzer.accepted) if (s.action.equals(action)) count++;
        return count;
    }
}
