/** Task 983 - Implement end-to-end security review checklist and automated scans for repository code.
 *  (Tier: Master, CTCI Beyond -- Security & Privacy)
 *
 *  A static scanner over an in-memory repository of source files. Six hand-rolled detectors (no regex, only
 *  token/case-insensitive substring analysis) look for hardcoded secrets, SQL string concatenation, command
 *  injection, weak hashes, route handlers without an authorisation call in the following line, and security
 *  TODOs. Findings carry the rule severity, are deduplicated per (file, line, rule), can be suppressed with
 *  an inline `scan-ignore: <rule>` marker and are gated: the release is blocked by any critical/high finding
 *  or by an unchecked mandatory checklist item, and the reason names the worst rule by weight.
 *  Simplified: in-memory files, six substring detectors and a static checklist instead of a real parser,
 *  SAST engine and dependency scanner; the dedup, suppression, severity aggregation and fail-closed gate are
 *  the real properties being checked.
 *  Verified: exact findings against a hand-derived expectation, duplicate suppression, inline scan-ignore
 *  handling with unknown-rule rejection, the authorisation window boundary, severity aggregation, the gate
 *  blocking on the worst rule, the checklist gate on an otherwise clean tree, the empty-repository boundary
 *  and determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final Map<String, Integer> WEIGHTS = Map.of(
            "hardcoded-secret", 9, "command-injection", 9, "sql-injection", 8,
            "missing-auth", 7, "weak-hash", 5, "todo-security", 2);
    static final Set<String> MANDATORY_CHECKS = Set.of("dependency-scan", "threat-model", "secret-scan", "pen-test");
    static final List<String> SECRET_NAMES = List.of("password", "apikey", "api_key", "secret", "token");

    static final class SourceFile {
        final String path; final List<String> lines;
        SourceFile(String path, List<String> lines) { this.path = path; this.lines = List.copyOf(lines); }
    }

    static final class Finding {
        final String file, rule, snippet; final int line, weight;
        Finding(String file, int line, String rule, String snippet) {
            this.file = file; this.line = line; this.rule = rule; this.snippet = snippet;
            this.weight = WEIGHTS.get(rule);
        }
        String key() { return file + ":" + line + ":" + rule; }
    }

    static boolean mentions(String lower, String token) { return lower.contains(token); }

    /** Rules are evaluated per line; a null result means the rule does not apply. */
    static List<String> rulesOn(SourceFile file, int index) {
        String line = file.lines.get(index);
        String lower = line.toLowerCase(Locale.ROOT);
        List<String> hits = new ArrayList<>();
        boolean quotedAssignment = lower.indexOf('=') >= 0 && (line.indexOf('"') >= 0 || line.indexOf('\'') >= 0);
        boolean fromEnvironment = mentions(lower, "getenv(") || mentions(lower, "process.env");
        for (String name : SECRET_NAMES) {
            if (mentions(lower, name) && quotedAssignment && !fromEnvironment) { hits.add("hardcoded-secret"); break; }
        }
        if (mentions(lower, "select") && line.indexOf('+') >= 0) hits.add("sql-injection");
        if (mentions(lower, "runtime.getruntime().exec(") || mentions(lower, "os.system(") || mentions(lower, "eval(")) {
            hits.add("command-injection");
        }
        if (mentions(lower, "md5") || mentions(lower, "sha1")) hits.add("weak-hash");
        if (lower.startsWith("route ") && !authorised(file, index)) hits.add("missing-auth");
        if (mentions(lower, "todo(security)") || mentions(lower, "fixme(auth)")) hits.add("todo-security");
        return hits;
    }

    /** A route is authorised when the same or the immediately following line calls an authorisation helper. */
    static boolean authorised(SourceFile file, int index) {
        for (int i = index; i <= Math.min(index + 1, file.lines.size() - 1); i++) {
            String lower = file.lines.get(i).toLowerCase(Locale.ROOT);
            if (mentions(lower, "authorize(") || mentions(lower, "requireauth(")) return true;
        }
        return false;
    }

    /** Suppressions declared inline: `scan-ignore: <rule>` disables that rule for that line only. */
    static Set<String> suppressions(String line) {
        Set<String> ignored = new TreeSet<>();
        int marker = line.indexOf("scan-ignore:");
        if (marker < 0) return ignored;
        String rest = line.substring(marker + "scan-ignore:".length()).trim();
        for (String token : rest.split("[^A-Za-z0-9-]+")) {
            if (token.isEmpty()) continue;
            if (!WEIGHTS.containsKey(token)) throw new IllegalArgumentException("unknown rule in scan-ignore: " + token);
            ignored.add(token);
        }
        return ignored;
    }

    static List<Finding> scan(List<SourceFile> repo) {
        Map<String, Finding> unique = new TreeMap<>();
        for (SourceFile file : repo) {
            for (int i = 0; i < file.lines.size(); i++) {
                Set<String> ignored = suppressions(file.lines.get(i));
                for (String rule : rulesOn(file, i)) {
                    if (ignored.contains(rule)) continue;
                    Finding finding = new Finding(file.path, i + 1, rule, file.lines.get(i).trim());
                    unique.putIfAbsent(finding.key(), finding);
                }
            }
        }
        return new ArrayList<>(unique.values());
    }

    static final class Report {
        final List<Finding> findings; final List<String> blockers; final String worstRule; final int worstWeight;
        Report(List<Finding> findings, List<String> blockers, String worstRule, int worstWeight) {
            this.findings = findings; this.blockers = blockers; this.worstRule = worstRule; this.worstWeight = worstWeight;
        }
        boolean blocked() { return !blockers.isEmpty(); }
    }

    static Report review(List<SourceFile> repo, Set<String> completedChecks) {
        List<Finding> findings = scan(repo);
        List<String> blockers = new ArrayList<>();
        String worstRule = "-";
        int worstWeight = 0;
        Map<String, Integer> perRule = new TreeMap<>();
        for (Finding finding : findings) {
            perRule.merge(finding.rule, 1, Integer::sum);
            if (finding.weight > worstWeight) { worstWeight = finding.weight; worstRule = finding.rule; }
        }
        if (worstWeight >= 7) {
            blockers.add("blocking findings: " + findings.size() + " (" + perRule + "), worst rule " + worstRule);
        }
        for (String required : new TreeSet<>(MANDATORY_CHECKS)) {
            if (!completedChecks.contains(required)) blockers.add("checklist item incomplete: " + required);
        }
        return new Report(findings, blockers, worstRule, worstWeight);
    }

    static List<SourceFile> vulnerableRepo() {
        return List.of(new SourceFile("src/Auth.java", List.of(
                "token = \"abc123\"",
                "query = \"SELECT * FROM users WHERE id = \" + id",
                "md5 = MessageDigest.getInstance(\"MD5\"); md5.update(x)",
                "Runtime.getRuntime().exec(cmd)",
                "route GET /admin -> adminHandler()",
                "route GET /profile -> profileHandler()",
                "authorize(user)",
                "hash = sha1(data) // scan-ignore: weak-hash",
                "// TODO(security) rotate the key")),
                new SourceFile("src/Repo.java", List.of(
                        "query = \"SELECT * FROM users WHERE id = ?\"",
                        "token = System.getenv(\"TOKEN\")")));
    }

    static List<SourceFile> cleanRepo() {
        return List.of(new SourceFile("src/Safe.java", List.of(
                "token = System.getenv(\"TOKEN\")",
                "query = \"SELECT * FROM users WHERE id = ?\"",
                "digest = MessageDigest.getInstance(\"SHA-256\")",
                "route GET /health -> healthHandler()",
                "requireAuth(request)",
                "// rotate keys every 90 days")));
    }

    public static void main(String[] args) {
        Set<String> allChecks = new TreeSet<>(MANDATORY_CHECKS);
        Report report = review(vulnerableRepo(), allChecks);

        Set<String> expected = new TreeSet<>(List.of("src/Auth.java:1:hardcoded-secret",
                "src/Auth.java:2:sql-injection", "src/Auth.java:3:weak-hash",
                "src/Auth.java:4:command-injection", "src/Auth.java:5:missing-auth",
                "src/Auth.java:9:todo-security"));
        Set<String> actual = new TreeSet<>();
        for (Finding finding : report.findings) actual.add(finding.key());
        check(actual.equals(expected) && report.findings.size() == 6,
                "the scan must reproduce the hand-derived findings: " + actual);

        long weakHashes = report.findings.stream().filter(f -> f.rule.equals("weak-hash")).count();
        check(weakHashes == 1 && report.findings.stream().noneMatch(f -> f.line == 8),
                "duplicates on one line must collapse and an inline scan-ignore must suppress that rule: " + actual);

        boolean rejected = false;
        try {
            suppressions("hash = sha1(x) // scan-ignore: weak-hash, mystery-rule");
        } catch (IllegalArgumentException e) {
            rejected = e.getMessage().contains("mystery-rule");
        }
        check(rejected && suppressions("x // scan-ignore: sql-injection").equals(Set.of("sql-injection")),
                "an unknown rule in a scan-ignore marker must be rejected");

        check(actual.contains("src/Auth.java:5:missing-auth") && !actual.contains("src/Auth.java:6:missing-auth")
                        && authorised(vulnerableRepo().get(0), 6),
                "only a route without authorisation on the next line may be flagged: " + actual);

        Map<String, Integer> perRule = new TreeMap<>();
        for (Finding finding : report.findings) perRule.merge(finding.rule, 1, Integer::sum);
        check(perRule.values().stream().mapToInt(Integer::intValue).sum() == 6 && report.worstWeight == 9
                        && (report.worstRule.equals("hardcoded-secret") || report.worstRule.equals("command-injection"))
                        && report.findings.stream().allMatch(f -> f.weight == WEIGHTS.get(f.rule)),
                "severity aggregation must pick the heaviest rule: " + report.worstRule + " weight " + report.worstWeight);

        check(report.blocked() && report.blockers.size() == 1 && report.blockers.get(0).contains("worst rule"),
                "critical findings must block the release and name the reason: " + report.blockers);

        Report clean = review(cleanRepo(), allChecks);
        check(clean.findings.isEmpty() && !clean.blocked() && clean.worstWeight == 0,
                "a clean tree with a complete checklist must pass: " + clean.findings);

        Set<String> partial = new TreeSet<>(MANDATORY_CHECKS);
        partial.remove("pen-test");
        Report missingCheck = review(cleanRepo(), partial);
        check(missingCheck.blocked() && missingCheck.findings.isEmpty()
                        && missingCheck.blockers.get(0).contains("checklist item incomplete: pen-test")
                        && missingCheck.blockers.size() == 1,
                "an unchecked mandatory checklist item must block even without findings: " + missingCheck.blockers);

        Report emptyRepo = review(List.of(), Set.of());
        check(emptyRepo.findings.isEmpty() && emptyRepo.blocked() && emptyRepo.blockers.size() == MANDATORY_CHECKS.size()
                        && review(List.of(new SourceFile("src/Empty.java", List.of())), allChecks).findings.isEmpty()
                        && !review(List.of(new SourceFile("src/Empty.java", List.of())), allChecks).blocked(),
                "an empty repository must block only on the checklist and an empty file must yield nothing");

        List<String> first = new ArrayList<>();
        for (Finding finding : scan(vulnerableRepo())) first.add(finding.key() + " " + finding.snippet);
        List<String> second = new ArrayList<>();
        for (Finding finding : scan(vulnerableRepo())) second.add(finding.key() + " " + finding.snippet);
        check(first.equals(second) && first.size() == 6 && report.findings.stream().allMatch(f -> !f.snippet.isEmpty()),
                "scanning the same repository twice must be identical");

        System.out.println("findings=" + report.findings.size() + " worst=" + report.worstRule + "/" + report.worstWeight
                + " blocked=" + report.blocked());
        System.out.println("rules=" + perRule + " cleanfindings=" + clean.findings.size());
        System.out.println("PASS");
    }
}
