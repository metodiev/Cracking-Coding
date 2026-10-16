/** Task 953 - Implement contributor guide and templates for new problems, tests, and solutions.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory documentation scaffold: a tiny {{placeholder}} template engine, a metadata validator for
 *  exercise specs, a deterministic file planner, and the contributor guide itself. Rendering a spec
 *  produces the task page, the test plan and one solution stub per declared language, all under the
 *  repository's canonical paths.
 *  Simplified: files are produced in a TreeMap instead of being written to disk, and the "build" of the
 *  guide is a string assembly; the naming, validation and placeholder rules are the real ones.
 *  Verified: planned paths and generated paths agree exactly; every placeholder is substituted; the four
 *  language stubs carry the canonical file names; five malformed specs are rejected while valid ones
 *  pass; unknown or unterminated placeholders raise; rendering is idempotent and independent of the
 *  order in which languages are listed; every placeholder used by any template is documented in the
 *  guide, checked with an independent scanner.
 */
import java.util.*;

public class Solution {
    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }
    static final class TemplateError extends RuntimeException {
        TemplateError(String message) { super(message); }
    }
    static final Set<String> TIERS = Set.of("beginner", "intermediate", "advanced", "expert", "master");
    static final Map<String, String> FILE_NAMES = new LinkedHashMap<>();
    static {
        FILE_NAMES.put("java", "Solution.java");
        FILE_NAMES.put("python", "solution.py");
        FILE_NAMES.put("javascript", "solution.js");
        FILE_NAMES.put("csharp", "Solution.cs");
    }
    static final String PROBLEM_TEMPLATE =
            "# {{number}} - {{title}}\n\n"
          + "- **Number:** {{number}}\n- **Tier:** {{tier}}\n- **CTCI chapter:** {{chapter}}\n\n"
          + "## Statement\n\n{{statement}}\n\n## Constraints\n\n{{constraints}}\n\n## Solutions\n\n{{solutionIndex}}\n";
    static final String TEST_TEMPLATE =
            "# Test plan for {{number}} - {{title}}\n\n"
          + "- tier: {{tier}}\n- languages: {{languages}}\n"
          + "- cases: {{caseCount}} deterministic cases plus one boundary case\n"
          + "- oracle: independent brute-force reference implementation\n";
    static String solutionTemplate(String language) {
        return "/* Task {{number}} - {{title}}. Tier: {{tier}} */\n"
             + "// " + language + " entry point: {{fileName}}\n"
             + "// verified against tests/{{tier}}/{{folder}}_test.py\n";
    }
    static final Map<String, String> PLACEHOLDERS = new LinkedHashMap<>();
    static {
        PLACEHOLDERS.put("number", "three digit zero padded task number");
        PLACEHOLDERS.put("title", "task title copied from the manifest");
        PLACEHOLDERS.put("tier", "one of beginner, intermediate, advanced, expert, master");
        PLACEHOLDERS.put("slug", "lower-case dash separated slug from the manifest");
        PLACEHOLDERS.put("chapter", "CTCI chapter or Beyond topic");
        PLACEHOLDERS.put("folder", "NNN-slug folder shared by all language solutions");
        PLACEHOLDERS.put("statement", "one paragraph statement of the problem");
        PLACEHOLDERS.put("constraints", "stdlib only, deterministic, self-checking");
        PLACEHOLDERS.put("solutionIndex", "bulleted list linking every language solution");
        PLACEHOLDERS.put("languages", "comma separated language list");
        PLACEHOLDERS.put("caseCount", "number of deterministic test cases");
        PLACEHOLDERS.put("fileName", "canonical solution file name for the language");
    }
    static String render(String template, Map<String, String> values) {
        StringBuilder out = new StringBuilder();
        int i = 0;
        while (i < template.length()) {
            int open = template.indexOf("{{", i);
            if (open < 0) { out.append(template, i, template.length()); break; }
            int close = template.indexOf("}}", open + 2);
            if (close < 0) throw new TemplateError("unterminated placeholder at index " + open);
            String key = template.substring(open + 2, close).trim();
            if (!values.containsKey(key)) throw new TemplateError("missing value for placeholder " + key);
            out.append(template, i, open).append(values.get(key));
            i = close + 2;
        }
        return out.toString();
    }
    /** Independent scanner used by the documentation check, not by render(). */
    static Set<String> placeholdersIn(String template) {
        Set<String> found = new TreeSet<>();
        int i = 0;
        while (true) {
            int open = template.indexOf("{{", i);
            if (open < 0) return found;
            int close = template.indexOf("}}", open + 2);
            if (close < 0) return found;
            found.add(template.substring(open + 2, close).trim());
            i = close + 2;
        }
    }
    record ExerciseSpec(int number, String tier, String slug, String title, String chapter, List<String> languages) {}
    static String folder(ExerciseSpec spec) { return String.format("%03d", spec.number()) + "-" + spec.slug(); }
    static boolean isSlug(String slug) {
        if (slug.isEmpty() || slug.length() > 64 || slug.charAt(0) == '-' || slug.charAt(slug.length() - 1) == '-') return false;
        boolean previousDash = false;
        for (int i = 0; i < slug.length(); i++) {
            char c = slug.charAt(i);
            boolean allowed = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-';
            if (!allowed) return false;
            if (c == '-') { if (previousDash) return false; previousDash = true; } else previousDash = false;
        }
        return true;
    }
    static List<String> validate(ExerciseSpec spec, Set<Integer> seenNumbers) {
        List<String> errors = new ArrayList<>();
        if (spec.number() < 1 || spec.number() > 998) errors.add("number out of range: " + spec.number());
        if (seenNumbers.contains(spec.number())) errors.add("duplicate number: " + spec.number());
        if (!TIERS.contains(spec.tier())) errors.add("unknown tier: " + spec.tier());
        if (!isSlug(spec.slug())) errors.add("invalid slug: " + spec.slug());
        if (spec.title().isBlank() || spec.title().length() > 120 || spec.title().endsWith(".")) errors.add("invalid title");
        if (spec.chapter().isBlank()) errors.add("missing CTCI chapter");
        if (spec.languages().isEmpty()) errors.add("at least one language is required");
        for (String language : spec.languages()) if (!FILE_NAMES.containsKey(language)) errors.add("unsupported language: " + language);
        return errors;
    }
    /** Canonical java, python, javascript, csharp order so the generated text never depends on input order. */
    static List<String> orderedLanguages(ExerciseSpec spec) {
        List<String> out = new ArrayList<>();
        for (String language : FILE_NAMES.keySet()) if (spec.languages().contains(language)) out.add(language);
        return out;
    }
    static List<String> planFiles(ExerciseSpec spec) {
        List<String> planned = new ArrayList<>();
        String folder = folder(spec);
        planned.add("tasks/" + spec.tier() + "/" + folder + ".md");
        planned.add("tests/" + spec.tier() + "/" + folder + "_test.py");
        for (String language : orderedLanguages(spec))
            planned.add("solutions/" + language + "/" + spec.tier() + "/" + folder + "/" + FILE_NAMES.get(language));
        return planned;
    }
    static Map<String, String> generate(ExerciseSpec spec) {
        String folder = folder(spec);
        StringBuilder index = new StringBuilder();
        for (String language : orderedLanguages(spec))
            index.append("- ").append(language).append(": solutions/").append(language).append('/').append(spec.tier())
                 .append('/').append(folder).append('/').append(FILE_NAMES.get(language)).append('\n');
        Map<String, String> values = new LinkedHashMap<>();
        values.put("number", String.valueOf(spec.number()));
        values.put("title", spec.title());
        values.put("tier", spec.tier());
        values.put("slug", spec.slug());
        values.put("chapter", spec.chapter());
        values.put("folder", folder);
        values.put("languages", String.join(", ", orderedLanguages(spec)));
        values.put("caseCount", "8");
        values.put("statement", "Implement " + spec.title() + " and prove it with the shipped checks.");
        values.put("constraints", "standard library only, deterministic, self-checking, prints PASS");
        values.put("solutionIndex", index.toString());
        Map<String, String> files = new TreeMap<>();
        files.put("tasks/" + spec.tier() + "/" + folder + ".md", render(PROBLEM_TEMPLATE, values));
        files.put("tests/" + spec.tier() + "/" + folder + "_test.py", render(TEST_TEMPLATE, values));
        for (String language : orderedLanguages(spec)) {
            Map<String, String> solutionValues = new LinkedHashMap<>(values);
            solutionValues.put("fileName", FILE_NAMES.get(language));
            files.put("solutions/" + language + "/" + spec.tier() + "/" + folder + "/" + FILE_NAMES.get(language), render(solutionTemplate(language), solutionValues));
        }
        return files;
    }
    static String contributorGuide() {
        StringBuilder out = new StringBuilder("# Contributor guide\n\n## Checklist for a new problem\n\n");
        for (String step : new String[] { "pick the next free number from tools/tasks-manifest.tsv", "create tasks/<tier>/<NNN-slug>.md from the problem template", "create tests/<tier>/<NNN-slug>_test.py from the test plan template", "add one solution file per language from the solution templates", "run python tools/run_solutions.py and paste the PASS line into the pull request" })
            out.append("- ").append(step).append('\n');
        out.append("\n## Placeholders\n\n");
        for (Map.Entry<String, String> entry : PLACEHOLDERS.entrySet())
            out.append("- `{{").append(entry.getKey()).append("}}`: ").append(entry.getValue()).append('\n');
        out.append("\n## Review checklist\n\n")
           .append("- [ ] the slug is lower-case, dash separated and at most 64 characters\n")
           .append("- [ ] the tier is one of the five tiers\n")
           .append("- [ ] the solution file name matches the language convention\n")
           .append("- [ ] the solution prints exactly PASS and leaves no scratch artifact\n");
        return out.toString();
    }
    public static void main(String[] args) {
        ExerciseSpec sample = new ExerciseSpec(812, "master", "sample-problem-two-sum", "Sample problem two sum", "Ch 1 - Arrays and Strings", List.of("java", "python", "javascript", "csharp"));
        Map<String, String> generated = generate(sample);
        List<String> planned = planFiles(sample);
        String taskPage = generated.get("tasks/master/812-sample-problem-two-sum.md");
        // 1. planned and generated paths agree, the page has all headings and no placeholder survives
        boolean noPlaceholders = true;
        for (String text : generated.values()) noPlaceholders &= !text.contains("{{") && !text.contains("}}");
        check(generated.size() == planned.size() && new TreeSet<>(generated.keySet()).equals(new TreeSet<>(planned)) && taskPage.contains("## Statement") && taskPage.contains("## Constraints") && taskPage.contains("## Solutions") && noPlaceholders, "plan matches generation, sections present, every placeholder substituted");
        // 2. one solution file per language with the canonical file name
        boolean matrixOk = true;
        for (String language : FILE_NAMES.keySet()) {
            String path = "solutions/" + language + "/master/812-sample-problem-two-sum/" + FILE_NAMES.get(language);
            matrixOk &= generated.containsKey(path) && generated.get(path).contains(FILE_NAMES.get(language))
                    && generated.get(path).contains(language);
        }
        check(matrixOk, "four language solutions rendered with their canonical file names");
        // 3. malformed specs are rejected, the sample is accepted
        Set<Integer> seen = new TreeSet<>(Set.of(812));
        List<ExerciseSpec> bad = List.of( new ExerciseSpec(813, "master", "Bad-Slug", "Bad", "Beyond", List.of("java")), new ExerciseSpec(0, "master", "zero-number", "Bad", "Beyond", List.of("java")), new ExerciseSpec(999, "master", "too-large", "Bad", "Beyond", List.of("java")), new ExerciseSpec(814, "wizard", "unknown-tier", "Bad", "Beyond", List.of("java")), new ExerciseSpec(815, "master", "no-language", "Bad", "Beyond", List.of()), new ExerciseSpec(812, "master", "duplicate-number", "Bad", "Beyond", List.of("java")), new ExerciseSpec(816, "master", "bad-language", "Bad", "Beyond", List.of("cobol")));
        int rejected = 0;
        for (ExerciseSpec spec : bad) if (!validate(spec, seen).isEmpty()) rejected++;
        check(rejected == bad.size() && validate(sample, Set.of()).isEmpty(), "all " + bad.size() + " malformed specs rejected, the sample spec validates clean");
        // 4. template engine failures
        boolean unknownThrew = false, unterminatedThrew = false;
        try { render("a {{nope}} b", new TreeMap<>()); } catch (TemplateError e) { unknownThrew = true; }
        try { render("a {{unterminated", new TreeMap<>()); } catch (TemplateError e) { unterminatedThrew = true; }
        check(unknownThrew && unterminatedThrew && render("x{{number}}y", Map.of("number", "7")).equals("x7y"), "unknown and unterminated placeholders raise, a valid one renders");
        // 5. idempotent and independent of the language-list order
        ExerciseSpec shuffled = new ExerciseSpec(812, "master", "sample-problem-two-sum", "Sample problem two sum", "Ch 1 - Arrays and Strings", List.of("csharp", "javascript", "python", "java"));
        check(generate(sample).equals(generate(sample)) && generate(shuffled).equals(generated), "rendering twice, or with the languages listed in another order, is byte identical");
        // 6. boundary: smallest valid spec, padded folder name, three planned files
        ExerciseSpec minimal = new ExerciseSpec(1, "advanced", "x", "Tiny task", "Beyond", List.of("java"));
        check(validate(minimal, Set.of(812)).isEmpty() && folder(minimal).equals("001-x") && planFiles(minimal).size() == 3 && generate(minimal).size() == 3, "minimal spec validates, folder is zero padded and only three files are planned");
        // 7. documentation completeness, checked with an independent placeholder scanner
        Set<String> used = new TreeSet<>();
        used.addAll(placeholdersIn(PROBLEM_TEMPLATE));
        used.addAll(placeholdersIn(TEST_TEMPLATE));
        for (String language : FILE_NAMES.keySet()) used.addAll(placeholdersIn(solutionTemplate(language)));
        String guide = contributorGuide();
        boolean documented = true;
        for (String key : used) documented &= guide.contains("{{" + key + "}}");
        check(documented && guide.contains("## Review checklist") && guide.contains("## Placeholders"), "every placeholder used by a template is documented in the guide");
        // 8. independent oracle: rebuild the expected plan with plain string concatenation for 30 specs
        boolean planOracle = true;
        for (int i = 0; i < 30; i++) {
            ExerciseSpec spec = new ExerciseSpec(100 + i, i % 2 == 0 ? "expert" : "master", "task-" + i, "Task " + i, "Beyond", List.of("python", "java"));
            List<String> rebuilt = new ArrayList<>();
            String f = String.format("%03d", 100 + i) + "-task-" + i;
            rebuilt.add("tasks/" + spec.tier() + "/" + f + ".md");
            rebuilt.add("tests/" + spec.tier() + "/" + f + "_test.py");
            rebuilt.add("solutions/java/" + spec.tier() + "/" + f + "/Solution.java");
            rebuilt.add("solutions/python/" + spec.tier() + "/" + f + "/solution.py");
            planOracle &= new TreeSet<>(planFiles(spec)).equals(new TreeSet<>(rebuilt))
                    && generate(spec).keySet().size() == rebuilt.size();
        }
        check(planOracle, "independently rebuilt paths match the planner for 30 generated specs");
        System.out.println("generated files for the sample spec: " + generated.size());
        System.out.println("documented placeholders: " + PLACEHOLDERS.size() + ", used by templates: " + used.size());
        System.out.println("rejected specs: " + rejected + " of " + bad.size());
        System.out.println("PASS");
    }
}
