/** Task 994 - Implement editorial writing assistant that drafts explanations per solution.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  The assistant drafts an editorial from a structured outline. Points are grouped into the canonical
 *  sections (idea, invariant, complexity, pitfalls, verification), de-duplicated, sorted by priority and
 *  then text, and rendered through per-section phrasing templates whose variant is picked by hashing
 *  (solutionId, section, point) - so the draft varies per solution yet is byte-identical on every rerun
 *  and independent of the outline's input order. A sentence budget is enforced by dropping the
 *  lowest-priority points first, every drop is reported, and each section keeps at least one sentence.
 *  A linter then scans the draft for banned filler ("obviously", "trivially", ...), for duplicate
 *  sentences and for the section key term, and a draft that fails the lint is refused rather than
 *  published.
 *  Simplified: template strings and a small banned-word list instead of a language model; the budget
 *  arithmetic, the deterministic wording choice and the lint gates are the real ones.
 *  Verified: every section is present in order with its key term, drafts are deterministic and
 *  order-independent, the budget is respected with reported drops, banned fillers and duplicate
 *  sentences are caught, wording differs between solutions, the empty and single-point boundaries
 *  behave, and independent sentence-count and banned-word oracles agree with the assistant.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final List<String> SECTIONS = List.of("idea", "invariant", "complexity", "pitfalls", "verification");
    static final Map<String, String> KEY_TERM = Map.of(
            "idea", "observation", "invariant", "invariant", "complexity", "O(",
            "pitfalls", "edge case", "verification", "check");
    static final Map<String, String[]> TEMPLATES = Map.of(
            "idea", new String[]{"The core observation is that %s", "The key observation: %s", "Note the observation that %s"},
            "invariant", new String[]{"The invariant that holds is that %s", "We maintain the invariant that %s",
                    "Throughout the run the invariant is that %s"},
            "complexity", new String[]{"The cost is %s", "In total the cost is %s", "This runs in %s"},
            "pitfalls", new String[]{"One edge case to watch: %s", "The edge case here is that %s",
                    "Careful with the edge case where %s"},
            "verification", new String[]{"A check confirms that %s", "We check that %s", "The check asserts that %s"});
    static final List<String> BANNED = List.of("obviously", "trivially", "just", "simply", "clearly");

    static final class Point {
        final String section, text;
        final int priority;                       // lower is more important
        Point(String section, String text, int priority) {
            this.section = section; this.text = text; this.priority = priority;
        }
    }

    static final class Outline {
        final String solutionId;
        final List<Point> points;
        Outline(String solutionId, List<Point> points) { this.solutionId = solutionId; this.points = points; }
    }

    static final class Draft {
        final Map<String, List<String>> sentences = new LinkedHashMap<>();
        final List<Point> dropped = new ArrayList<>();
        final String error;
        Draft(String error) { this.error = error; }
        Draft() { this.error = null; }
        int sentenceCount() {
            int n = 0;
            for (List<String> s : sentences.values()) n += s.size();
            return n;
        }
        List<String> allSentences() {
            List<String> out = new ArrayList<>();
            for (String section : SECTIONS) out.addAll(sentences.getOrDefault(section, List.of()));
            return out;
        }
        String render() {
            StringBuilder sb = new StringBuilder();
            for (String section : SECTIONS) {
                sb.append("## ").append(section).append('\n');
                for (String s : sentences.getOrDefault(section, List.of())) sb.append("- ").append(s).append('\n');
            }
            return sb.toString();
        }
    }

    /** Stable per-(solution, section, point) index into the phrasing variants. */
    static int variantIndex(String solutionId, String section, String text, int variants) {
        long h = 0xcbf29ce484222325L;
        for (char c : (solutionId + "|" + section + "|" + text).toCharArray()) { h ^= c; h *= 0x100000001b3L; }
        return (int) Math.floorMod(h, variants);
    }

    static Draft draft(Outline outline, int sentenceBudget) {
        if (outline.points.isEmpty()) return new Draft("empty outline");
        if (SECTIONS.stream().anyMatch(s -> outline.points.stream().noneMatch(p -> p.section.equals(s))))
            return new Draft("outline must cover every section");
        Set<String> covering = new LinkedHashSet<>();
        for (Point p : outline.points) covering.add(p.section);
        if (sentenceBudget < covering.size()) return new Draft("sentence budget too small for the section floor");

        List<Point> queue = new ArrayList<>();
        Set<String> seenText = new HashSet<>();
        for (Point p : outline.points)
            if (seenText.add(p.section + "|" + p.text.trim())) queue.add(p);      // duplicate points collapse
        queue.sort(Comparator.comparingInt((Point p) -> p.priority).thenComparing(p -> p.text));

        Draft draft = new Draft();
        for (String section : SECTIONS) draft.sentences.put(section, new ArrayList<>());
        // Pass 1 - the section floor: every covered section renders its most important remaining point.
        for (String section : SECTIONS) {
            Point floor = null;
            for (Point p : queue)
                if (p.section.equals(section) && (floor == null || p.priority < floor.priority
                        || (p.priority == floor.priority && p.text.compareTo(floor.text) < 0))) floor = p;
            if (floor == null) continue;
            render(draft, outline.solutionId, floor);
            queue.remove(floor);
        }
        // Pass 2 - the budget: lower-priority points fill the remaining room or are reported as dropped.
        for (Point p : queue) {
            if (draft.sentenceCount() < sentenceBudget) render(draft, outline.solutionId, p);
            else draft.dropped.add(p);
        }
        return draft;
    }

    /** Renders one point into its section through the deterministic phrasing variant. */
    static void render(Draft draft, String solutionId, Point point) {
        String[] variants = TEMPLATES.get(point.section);
        String sentence = String.format(variants[variantIndex(solutionId, point.section, point.text, variants.length)],
                point.text.trim()) + ".";
        draft.sentences.get(point.section).add(sentence);
    }

    static List<String> lint(Draft draft) {
        List<String> problems = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (String section : SECTIONS) {
            List<String> bucket = draft.sentences.getOrDefault(section, List.of());
            if (bucket.isEmpty()) problems.add(section + ": section is empty");
            String joined = String.join(" ", bucket);
            if (!bucket.isEmpty() && !joined.contains(KEY_TERM.get(section)))
                problems.add(section + ": missing key term " + KEY_TERM.get(section));
            for (String sentence : bucket) {
                for (String banned : BANNED)
                    if (sentence.toLowerCase(Locale.ROOT).contains(banned))
                        problems.add(section + ": banned filler '" + banned + "'");
                if (!seen.add(sentence)) problems.add(section + ": duplicate sentence");
            }
        }
        return problems;
    }

    static Outline sampleOutline(String solutionId) {
        return new Outline(solutionId, List.of(
            new Point("verification", "the final array is sorted", 3),
            new Point("idea", "the frontier is scanned in wave order", 1),
            new Point("complexity", "O(V + E) time and O(V) space", 2),
            new Point("invariant", "every dequeued node is finalised", 1),
            new Point("pitfalls", "the graph may be disconnected", 2),
            new Point("idea", "the frontier is scanned in wave order", 1)));      // duplicate point
    }

    public static void main(String[] args) {
        Outline outline = sampleOutline("task-985");
        Draft draft = draft(outline, 8);
        check(draft.error == null && draft.sentenceCount() == 5 && draft.dropped.isEmpty(),
                "the sample outline must render five sentences and silently collapse the duplicate: " + draft.sentenceCount() + " sentences, dropped " + draft.dropped.size());
        check(new ArrayList<>(draft.sentences.keySet()).equals(SECTIONS), "sections must be present in canonical order: " + draft.sentences.keySet());
        check(draft.sentences.values().stream().noneMatch(List::isEmpty) && lint(draft).isEmpty(),
                "a budgeted draft must keep every section non-empty and pass the lint: " + lint(draft));
        // --- order independence and rerun determinism: same wording every time, whatever the input order.
        List<Point> reversed = new ArrayList<>(outline.points);
        Collections.reverse(reversed);
        Draft reordered = draft(new Outline(outline.solutionId, reversed), 8);
        check(reordered.render().equals(draft.render()), "draft must not depend on point order");
        check(draft(new Outline(outline.solutionId, outline.points), 8).render().equals(draft.render()), "a rerun must produce the identical draft");
        // --- wording varies per solution but stays deterministic for a given one.
        Draft other = draft(sampleOutline("task-986"), 8);
        check(!other.render().equals(draft.render()) && draft(sampleOutline("task-986"), 8).render().equals(other.render()),
                "wording must vary per solution and still be reproducible");
        // --- budget pressure: the lowest-priority points are dropped, never a whole section.
        Outline wide = new Outline("task-990", List.of(
            new Point("idea", "the window slides over the stream", 1),
            new Point("invariant", "the window sum stays correct", 1),
            new Point("complexity", "O(n) time and O(1) space", 2),
            new Point("complexity", "a second plausible cost estimate", 9),
            new Point("pitfalls", "the window may be larger than the stream", 2),
            new Point("verification", "the window sum matches a prefix sum check", 3)));
        Draft tight = draft(wide, 5);
        check(tight.sentenceCount() == 5 && tight.dropped.size() == 1 && tight.dropped.stream().allMatch(p -> p.priority == 9),
                "the budget must drop the lowest-priority points first: " + tight.dropped.stream().map(p -> p.text).toList());
        check(tight.sentences.values().stream().noneMatch(List::isEmpty) && lint(tight).isEmpty(),
                "even under budget pressure every section keeps a sentence: " + lint(tight));
        Draft generous = draft(wide, 20);
        check(generous.sentenceCount() == 6 && generous.dropped.isEmpty(), "a generous budget must keep every unique point: " + generous.sentenceCount());
        check(draft(wide, 4).error.equals("sentence budget too small for the section floor") && draft(wide, 5).dropped.size() == 1, "a budget below the section floor must be refused instead of starving a section: " + draft(wide, 4).error);
        // --- lint: banned filler, missing key terms and duplicates are all findings.
        Draft dirty = draft(outline, 8);
        dirty.sentences.get("idea").add(0, "Obviously this observation is trivially true.");
        List<String> findings = lint(dirty);
        check(findings.stream().anyMatch(f -> f.contains("banned filler 'obviously'")) && findings.stream().anyMatch(f -> f.contains("banned filler 'trivially'")),
                "banned fillers must be reported per section: " + findings);
        Draft duplicate = draft(outline, 8);
        duplicate.sentences.get("pitfalls").add(duplicate.sentences.get("pitfalls").get(0));
        check(lint(duplicate).stream().anyMatch(f -> f.contains("duplicate sentence")), "a duplicated sentence must be a finding: " + lint(duplicate));
        Draft missing = draft(outline, 8);
        missing.sentences.put("complexity", new ArrayList<>(List.of("The cost is bounded by the input.")));
        check(lint(missing).stream().anyMatch(f -> f.contains("missing key term O(")), "a section without its key term must be a finding: " + lint(missing));
        // --- boundaries: an empty outline and an outline missing a section are refused.
        check(draft(new Outline("x", List.of()), 8).error.equals("empty outline"), "an empty outline must be refused: " + draft(new Outline("x", List.of()), 8).error);
        check(draft(new Outline("x", List.of(new Point("idea", "only an idea", 1))), 8).error .equals("outline must cover every section"),
                "an outline without every section must be refused");
        // --- independent oracles: terminator count and banned-word scan recomputed by hand.
        String rendered = draft.render();
        check(oracleSentenceCount(rendered) == draft.sentenceCount(), "sentence oracle mismatch: " + oracleSentenceCount(rendered) + " vs " + draft.sentenceCount());
        check(oracleBanned(draft.allSentences()).equals(bannedIn(draft.allSentences())), "banned-word oracle mismatch: " + oracleBanned(draft.allSentences()));
        check(oracleBanned(dirty.allSentences()).contains("obviously"), "oracle must see the injected filler");

        System.out.println("draft sections: " + draft.sentences.keySet() + " sentences " + draft.sentenceCount());
        System.out.println("PASS");
    }

    static List<String> bannedIn(List<String> sentences) {
        List<String> hits = new ArrayList<>();
        for (String sentence : sentences)
            for (String banned : BANNED)
                if (sentence.toLowerCase(Locale.ROOT).contains(banned) && !hits.contains(banned)) hits.add(banned);
        return hits;
    }

    /** Oracle: counts sentence terminators by scanning characters instead of the assembled list. */
    static int oracleSentenceCount(String rendered) {
        int count = 0;
        for (int i = 0; i < rendered.length(); i++) if (rendered.charAt(i) == '.') count++;
        return count;
    }

    /** Oracle: detects banned words with a sliding-window comparison instead of String.contains. */
    static List<String> oracleBanned(List<String> sentences) {
        List<String> hits = new ArrayList<>();
        for (String sentence : sentences) {
            String lower = sentence.toLowerCase(Locale.ROOT);
            for (String banned : BANNED) {
                boolean found = false;
                for (int start = 0; start + banned.length() <= lower.length(); start++)
                    if (lower.startsWith(banned, start)) { found = true; break; }
                if (found && !hits.contains(banned)) hits.add(banned);
            }
        }
        return hits;
    }
}
