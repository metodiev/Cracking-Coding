/** Task 973 - Implement automated hint generator using solved-solution decomposition.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  Hints are generated from a decomposition of the canonical solution: each step carries a category
 *  (init/loop/branch/update/return), the identifiers it touches, the operation kind and the canonical
 *  body line. Three levels are produced: L1 is one strategy hint (technique, step count, invariant),
 *  L2 is one hint per step naming the category and the identifiers, and L3 extends each L2 hint with the
 *  operation verb, so L2's tokens are always a strict subset of L3's. A validator rejects steps with an
 *  unknown category, and a leak filter guarantees that no hint ever contains a declared secret
 *  (final/intermediate values) or a canonical body line verbatim.
 *  Simplified: hints are produced from a hand-written decomposition table rather than from real AST
 *  slicing or from mining learner failures; the three-level specificity ordering, the coverage over step
 *  indices and the leak filter are the real properties being checked.
 *  Verified: L1 content, per-step L2/L3 coverage and wording against the operation table, the strict token
 *  subset property, position-aware next-step hints, end-of-solution behaviour, leak-freedom on both
 *  decompositions, rejection of an unknown category, the empty-decomposition boundary and determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final Set<String> CATEGORIES = Set.of("init", "loop", "branch", "update", "return");
    static final Map<String, String> ACTIONS = Map.of(
            "index", "record where the current value was last seen",
            "compare", "check the stored complement before consuming the value",
            "accumulate", "extend the best answer found so far",
            "iterate", "advance to the next element",
            "return", "hand back the accumulated answer");

    static final class Step {
        final String category, subject, operation, body;
        Step(String category, String subject, String operation, String body) {
            if (!CATEGORIES.contains(category)) throw new IllegalStateException("unknown category " + category);
            if (!ACTIONS.containsKey(operation)) throw new IllegalStateException("unknown operation " + operation);
            this.category = category; this.subject = subject; this.operation = operation; this.body = body;
        }
    }

    static final class Decomposition {
        final String problem, technique, invariant;
        final List<Step> steps; final List<String> hidden;
        Decomposition(String problem, String technique, String invariant, List<Step> steps, List<String> hidden) {
            this.problem = problem; this.technique = technique; this.invariant = invariant;
            this.steps = steps; this.hidden = hidden;
        }
    }

    /** L1: one strategy hint; L2: structure only; L3: structure plus the operation verb. */
    static List<String> hints(Decomposition d, int level) {
        List<String> out = new ArrayList<>();
        if (d.steps.isEmpty()) return out;
        if (level == 1) {
            out.add("Technique: " + d.technique + "; " + d.steps.size() + " steps; invariant: " + d.invariant + ".");
            return out;
        }
        for (int i = 0; i < d.steps.size(); i++) {
            Step s = d.steps.get(i);
            String base = "Step " + (i + 1) + "/" + d.steps.size() + " (" + s.category + ") on `" + s.subject + "`";
            out.add(level == 2 ? base : base + ": " + ACTIONS.get(s.operation));
        }
        return out;
    }

    /** Hint for the step the learner is about to take; a finished solution gets a review nudge. */
    static String nextHint(Decomposition d, int progress) {
        if (d.steps.isEmpty()) return "no canonical decomposition available";
        if (progress >= d.steps.size()) return "you have completed every step; re-derive the invariant: " + d.invariant;
        return hints(d, 3).get(Math.max(0, progress));
    }

    /** Any hint mentioning a declared secret or a canonical body verbatim is a leak. */
    static List<String> leaks(Decomposition d) {
        List<String> bad = new ArrayList<>();
        for (int level = 1; level <= 3; level++) {
            for (String hint : hints(d, level)) {
                for (String secret : d.hidden) if (hint.contains(secret)) bad.add("secret " + secret + " in " + hint);
                for (Step s : d.steps) if (hint.contains(s.body)) bad.add("body in " + hint);
            }
        }
        for (String hint : List.of(nextHint(d, 0), nextHint(d, d.steps.size()))) {
            for (String secret : d.hidden) if (hint.contains(secret)) bad.add("secret " + secret + " in " + hint);
        }
        return bad;
    }

    /** Independent tokenizer used to compare two hint wordings as sets. */
    static Set<String> tokens(String text) {
        Set<String> out = new TreeSet<>();
        for (String token : text.split("[^A-Za-z]+")) if (!token.isEmpty()) out.add(token);
        return out;
    }

    static Decomposition twoSumDecomposition() {
        return new Decomposition("longest window with all-distinct values", "hash-map scan",
                "seen maps each value to its latest index",
                List.of(new Step("init", "best", "index", "best = 0"),
                        new Step("init", "seen", "index", "seen = {}"),
                        new Step("loop", "i", "iterate", "for i in range(n)"),
                        new Step("branch", "seen", "compare", "if a[i] in seen"),
                        new Step("update", "best", "accumulate", "best = max(best, i - seen[a[i]])"),
                        new Step("update", "seen", "index", "seen[a[i]] = i"),
                        new Step("return", "best", "return", "return best")),
                List.of("19", "4,7,2,9,6,1"));
    }

    static Decomposition binarySearchDecomposition() {
        return new Decomposition("count pairs summing to k", "two-pointer sweep",
                "lo and hi bracket the unseen suffix",
                List.of(new Step("init", "lo", "index", "lo = 0"),
                        new Step("loop", "hi", "iterate", "while lo < hi"),
                        new Step("update", "hi", "accumulate", "hi = mid - 1"),
                        new Step("return", "lo", "return", "return lo")),
                List.of("k = 9", "-1"));
    }

    static void checkInvariants(Decomposition d) {
        List<String> l1 = hints(d, 1), l2 = hints(d, 2), l3 = hints(d, 3);
        check(l1.size() == 1 && l1.get(0).contains(d.technique) && l1.get(0).contains(d.invariant)
                        && l1.get(0).contains(d.steps.size() + " steps"),
                "L1 must state technique, step count and invariant: " + l1);
        check(l2.size() == d.steps.size() && l3.size() == d.steps.size(),
                "L2 and L3 must cover every step index exactly once");
        boolean wording = true;
        for (int i = 0; i < d.steps.size(); i++) {
            Step s = d.steps.get(i);
            wording &= l2.get(i).contains("(" + s.category + ")") && l2.get(i).contains(s.subject)
                    && l3.get(i).equals(l2.get(i) + ": " + ACTIONS.get(s.operation))
                    && l3.get(i).startsWith(l2.get(i)) && tokens(l2.get(i)).containsAll(tokens(s.subject));
        }
        check(wording, "L3 must extend L2 with the operation verb for every step: " + l3.get(0));
        boolean specificity = true;
        for (int i = 0; i < d.steps.size(); i++) {
            Set<String> small = tokens(l2.get(i)), big = tokens(l3.get(i));
            specificity &= big.containsAll(small) && big.size() > small.size();
        }
        check(specificity, "L2 tokens must be a strict subset of L3 tokens for every step");
        check(leaks(d).isEmpty(), "no hint may leak a secret or a canonical body line: " + leaks(d));
    }

    public static void main(String[] args) {
        Decomposition d = twoSumDecomposition();
        checkInvariants(d);

        check(nextHint(d, 0).equals(hints(d, 3).get(0)) && nextHint(d, 3).contains("Step 4/7")
                        && nextHint(d, 3).contains("check the stored complement") && nextHint(d, 6).contains("return"),
                "nextHint must point at the step the learner is about to take: " + nextHint(d, 3));
        check(nextHint(d, 7).contains("completed every step") && nextHint(d, 99).equals(nextHint(d, 7))
                        && nextHint(d, -5).equals(hints(d, 3).get(0)),
                "a finished solution must get a review nudge and an out-of-range cursor must clamp: "
                        + nextHint(d, 7));
        check(hints(d, 1).get(0).equals("Technique: hash-map scan; 7 steps; invariant: seen maps each value"
                        + " to its latest index."),
                "the strategy hint must be reproducible word for word");

        boolean sorted = true;
        for (int i = 1; i < hints(d, 3).size(); i++) sorted &= hints(d, 3).get(i - 1).compareTo(hints(d, 3).get(i)) < 0;
        check(sorted && new HashSet<>(hints(d, 3)).size() == d.steps.size()
                        && hints(twoSumDecomposition(), 3).equals(hints(d, 3))
                        && hints(twoSumDecomposition(), 2).equals(hints(d, 2)),
                "generated hints must be index-ordered, unique and reproducible across runs");

        checkInvariants(binarySearchDecomposition());

        boolean rejected = false;
        try {
            new Step("guess", "x", "iterate", "x = 1");
        } catch (IllegalStateException e) {
            rejected = e.getMessage().contains("unknown category");
        }
        check(rejected, "a step with an unknown category must be rejected by the validator");

        Decomposition empty = new Decomposition("none", "none", "none", List.of(), List.of());
        check(hints(empty, 1).isEmpty() && hints(empty, 2).isEmpty() && hints(empty, 3).isEmpty()
                        && nextHint(empty, 0).equals("no canonical decomposition available"),
                "an empty decomposition must produce no hints and a graceful message");

        System.out.println("steps=" + d.steps.size() + " hints=" + hints(d, 3).size() + " leaks=" + leaks(d).size());
        System.out.println("L3[1]=" + hints(d, 3).get(1));
        System.out.println("PASS");
    }
}
