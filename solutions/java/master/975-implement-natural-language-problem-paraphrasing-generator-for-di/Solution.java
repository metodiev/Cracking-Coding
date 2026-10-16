/** Task 975 - Implement natural-language problem paraphrasing generator for diverse practice.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  A template-and-lexicon paraphraser. A spec holds the numeric bound, the input format and the canonical
 *  target/constraint phrases. Sentence templates use {slot} placeholders filled from synonym banks, and the
 *  synonym variant per template comes from a fixed-seed splitmix64 stream, so practice sets vary in wording
 *  but never in meaning. Each paraphrase is validated by a round trip: a normalizer strips the wording back
 *  to `bound|target|constraint|format` and must reproduce the spec exactly, which makes the paraphrase
 *  semantically faithful rather than merely different.
 *  Simplified: a hand-written template/lexicon bank with four sentence styles instead of a seq2seq model,
 *  and canonical-normalization validation instead of entailment checking; bound preservation, round-trip
 *  equivalence and the template-slot validation gate are the real ones.
 *  Verified: one distinct paraphrase per template, exact bound/format preservation, round trip against the
 *  spec, wording diversity across paraphrases, agreement with an independent word-set validator, unknown
 *  slot rejection, a bound change flowing through, seed determinism and the count boundaries.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final List<String> TARGETS = List.of("longest substring",
            "longest run of characters", "longest stretch of text");
    static final List<String> CONSTRAINTS = List.of("contains at most %d distinct characters",
            "uses no more than %d different letters", "has at most %d unique characters");
    static final List<String> FORMATS = List.of("string");
    static final List<String> TEMPLATES = List.of(
            "Given a {format}, return the length of the {target} that {constraint}.",
            "A platform analyzes a {format} and must report the {target} that {constraint}.",
            "Interview round: from a {format}, produce the {target} that {constraint}.",
            "Devise the {target} of a {format} such that it {constraint}.");

    static final class Spec {
        final int bound; final String format;
        Spec(int bound, String format) {
            if (bound < 1) throw new IllegalArgumentException("bound must be positive");
            if (!FORMATS.contains(format)) throw new IllegalArgumentException("unsupported format " + format);
            this.bound = bound; this.format = format;
        }
        /** Canonical reading of the problem: what every paraphrase must normalize back to. */
        String canonical() {
            return bound + "|" + TARGETS.get(0) + "|" + String.format(Locale.ROOT, CONSTRAINTS.get(0), bound)
                    + "|" + format;
        }
    }

    /** Deterministic splitmix64 stream: fixes which synonym variant each template uses. */
    static final class Rng {
        long state;
        Rng(long seed) { state = seed; }
        long next() {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }
        int pick(int bound) { return (int) Math.floorMod(next(), bound); }
    }

    /** Substitutes {slot} placeholders; an unknown or unfilled slot is a hard error. */
    static String fill(String template, Map<String, String> values) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < template.length(); i++) {
            if (template.charAt(i) != '{') { out.append(template.charAt(i)); continue; }
            int close = template.indexOf('}', i);
            if (close < 0) throw new IllegalArgumentException("unbalanced template: " + template);
            String slot = template.substring(i + 1, close).trim();
            if (!values.containsKey(slot)) throw new IllegalArgumentException("unknown slot {" + slot + "}");
            out.append(values.get(slot));
            i = close;
        }
        return out.toString();
    }

    /** Deterministic Fisher-Yates over a synonym bank; the stream makes the order seed-dependent. */
    static List<String> shuffled(List<String> bank, Rng rng) {
        List<String> out = new ArrayList<>(bank);
        for (int i = out.size() - 1; i > 0; i--) Collections.swap(out, i, rng.pick(i + 1));
        return out;
    }

    static List<String> generate(Spec spec, int count, long seed) {
        Rng rng = new Rng(seed);
        List<String> targetBank = shuffled(TARGETS, rng);
        List<String> constraintBank = shuffled(CONSTRAINTS, rng);
        List<String> out = new ArrayList<>();
        for (int t = 0; t < TEMPLATES.size() && out.size() < count; t++) {
            Map<String, String> values = new TreeMap<>();
            values.put("target", targetBank.get(t % targetBank.size()));
            values.put("constraint", String.format(Locale.ROOT, constraintBank.get(t % constraintBank.size()), spec.bound));
            values.put("format", spec.format);
            out.add(fill(TEMPLATES.get(t), values));
        }
        return out;
    }

    /** Canonical key whose synonym occurs in `text`, or "?" when that content is missing entirely. */
    static String canonicalOf(List<String> synonyms, String text) {
        for (String synonym : synonyms) if (text.contains(synonym)) return synonyms.get(0);
        return "?";
    }

    /** Index of the synonym bank entry occurring in `text`, or -1 when none does. */
    static int synonymIndex(List<String> synonyms, String text) {
        for (int i = 0; i < synonyms.size(); i++) if (text.contains(synonyms.get(i))) return i;
        return -1;
    }

    static int constraintIndex(String text, int bound) {
        for (int i = 0; i < CONSTRAINTS.size(); i++) {
            if (text.contains(String.format(Locale.ROOT, CONSTRAINTS.get(i), bound))) return i;
        }
        return -1;
    }

    static String canonicalConstraint(String text, String bound) {
        int index = constraintIndex(text, Integer.parseInt(bound));
        return index < 0 ? "?" : String.format(Locale.ROOT, CONSTRAINTS.get(0), Integer.parseInt(bound));
    }

    static String firstNumber(String text) {
        for (String token : text.split("[^0-9]+")) if (!token.isEmpty()) return token;
        return "?";
    }

    static int digitGroups(String text) { return text.split("[0-9]+").length - 1; }

    static String normalize(String paraphrase) {
        String bound = firstNumber(paraphrase);
        if (bound.equals("?")) return "?";
        return bound + "|" + canonicalOf(TARGETS, paraphrase) + "|" + canonicalConstraint(paraphrase, bound)
                + "|" + canonicalOf(FORMATS, paraphrase);
    }

    /** Independent validator: splits into words and requires every content piece as a separate token. */
    static boolean wordSetValidator(String paraphrase, Spec spec) {
        Set<String> words = new HashSet<>(List.of(paraphrase.split("[^A-Za-z0-9]+")));
        boolean targetOk = false;
        for (String synonym : TARGETS) {
            boolean all = true;
            for (String word : synonym.split(" ")) all &= words.contains(word);
            targetOk |= all;
        }
        boolean constraintOk = false;
        for (String synonym : CONSTRAINTS) {
            boolean all = true;
            for (String word : String.format(Locale.ROOT, synonym, spec.bound).split(" ")) all &= words.contains(word);
            constraintOk |= all;
        }
        return targetOk && constraintOk && words.contains(spec.format) && words.contains(Integer.toString(spec.bound));
    }

    static Set<String> wordings(List<String> paraphrases) {
        Set<String> out = new TreeSet<>();
        for (String p : paraphrases) {
            out.add(synonymIndex(TARGETS, p) + "-" + constraintIndex(p, Integer.parseInt(firstNumber(p))));
        }
        return out;
    }

    public static void main(String[] args) {
        Spec spec = new Spec(2, "string");
        List<String> paras = generate(spec, TEMPLATES.size(), 42L);

        check(paras.size() == TEMPLATES.size() && new HashSet<>(paras).size() == TEMPLATES.size(),
                "the generator must emit one distinct paraphrase per template: " + paras.size());
        check(paras.stream().allMatch(p -> firstNumber(p).equals("2") && p.contains("string") && digitGroups(p) == 1),
                "every paraphrase must carry exactly the bound and the input format: " + paras.get(0));

        boolean roundTrip = true;
        for (String p : paras) roundTrip &= normalize(p).equals(spec.canonical());
        check(roundTrip, "every paraphrase must normalize back to the canonical spec: " + normalize(paras.get(0))
                + " vs " + spec.canonical());

        check(paras.stream().allMatch(p -> wordSetValidator(p, spec))
                        && paras.stream().allMatch(p -> normalize(p).split("\\|").length == 4),
                "an independent word-set validator must accept every paraphrase: " + paras.get(2));

        Set<Integer> targetVariants = new TreeSet<>();
        Set<Integer> constraintVariants = new TreeSet<>();
        for (String p : paras) {
            targetVariants.add(synonymIndex(TARGETS, p));
            constraintVariants.add(constraintIndex(p, spec.bound));
        }
        check(targetVariants.size() == TARGETS.size() && constraintVariants.size() == CONSTRAINTS.size()
                        && !targetVariants.contains(-1) && wordings(paras).size() >= TARGETS.size(),
                "the lexicon must vary the wording so paraphrases are not mere template swaps: " + targetVariants);
        List<String> otherSeed = generate(spec, TEMPLATES.size(), 7L);
        check(!otherSeed.equals(paras) && otherSeed.stream().allMatch(p -> normalize(p).equals(spec.canonical()))
                        && otherSeed.stream().allMatch(p -> wordSetValidator(p, spec)),
                "a differently seeded practice set must stay semantically identical: " + otherSeed.get(0));

        boolean rejectedSlot = false;
        try {
            fill("Solve the {puzzle} for a {format}.", new TreeMap<>(Map.of("format", "string")));
        } catch (IllegalArgumentException e) {
            rejectedSlot = e.getMessage().contains("unknown slot");
        }
        check(rejectedSlot, "a template referencing an unknown slot must be rejected");

        Spec stricter = new Spec(3, "string");
        List<String> strict = generate(stricter, 2, 42L);
        check(generate(stricter, 2, 42L).equals(strict) && strict.stream().allMatch(p -> firstNumber(p).equals("3"))
                        && normalize(strict.get(0)).equals(stricter.canonical())
                        && !normalize(strict.get(0)).equals(spec.canonical())
                        && !stricter.canonical().equals(spec.canonical()),
                "changing the bound must flow through the paraphrase and the round trip: " + strict.get(0));

        boolean rejectedSpec = false;
        try {
            new Spec(0, "string");
        } catch (IllegalArgumentException e) {
            rejectedSpec = e.getMessage().contains("positive");
        }
        check(rejectedSpec && generate(spec, 0, 7L).isEmpty() && generate(spec, 99, 7L).size() == TEMPLATES.size()
                        && generate(spec, -3, 7L).isEmpty(),
                "the requested count must be honoured, clamped and never negative");

        check(generate(spec, TEMPLATES.size(), 42L).equals(paras)
                        && !generate(spec, TEMPLATES.size(), 43L).equals(paras),
                "the same seed must reproduce the set while another seed varies the wording");

        System.out.println("templates=" + TEMPLATES.size() + " paraphrases=" + paras.size() + " bound=" + spec.bound);
        System.out.println("sample=" + paras.get(2));
        System.out.println("PASS");
    }
}
