/** Task 997 - Implement ethics section generator for problems with privacy/security implications.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  Every editorial needs an ethics section, and the generator derives it from policy rules instead of
 *  from the author's memory. A rule pairs a disclosure clause with the indicators that trigger it
 *  (email/identifier -> privacy, password/credential -> security, exploit/vulnerability -> dual use,
 *  consent/opt-out -> consent). The statement is scanned for those indicators, triggered rules are
 *  ordered by severity then id, and each clause is emitted with its rule id and with redacted evidence
 *  tokens: digit runs become '#' and any token containing '@' becomes "<redacted-email>", so the section
 *  never republishes the very data it warns about. The generator fails closed - suppressing a
 *  high-severity clause refuses publication, while suppressing a medium clause only raises the review
 *  flag - and it always emits a baseline clause when nothing is detected.
 *  Simplified: substring indicator lists instead of a PII classifier; the severity gating, the redaction
 *  and the coverage accounting are the real ones.
 *  Verified: risky statements trigger exactly the expected rules in severity order, coverage matches the
 *  trigger count, benign statements produce only the baseline clause, both fail-closed paths fire,
 *  redaction removes the raw email and digit runs, output is deterministic and rule-order independent,
 *  the empty-statement and repeated-indicator boundaries behave, and independent char-scan oracles agree.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final Map<String, Integer> SEVERITY = Map.of("high", 3, "medium", 2, "low", 1);

    static final class Policy {
        final String id, clause, severity;
        final List<String> indicators;
        Policy(String id, String clause, String severity, List<String> indicators) {
            this.id = id; this.clause = clause; this.severity = severity; this.indicators = indicators;
        }
    }

    static List<Policy> policies() {
        return new ArrayList<>(List.of(
            new Policy("pii", "Do not publish personal data: the statement names identifiers such as email "
                    + "addresses or account numbers.", "high", List.of("email", "@", "identifier")),
            new Policy("credential", "Never store or log credentials: passwords, tokens and keys used in the "
                    + "exercise must be synthetic.", "high", List.of("password", "token", "credential")),
            new Policy("dual-use", "Assess dual use: offensive techniques shown here (exploit, vulnerability, "
                    + "payload) must stay inside a sandbox.", "high", List.of("exploit", "vulnerability", "payload")),
            new Policy("consent", "Obtain consent and offer an opt-out when user activity is recorded.", "medium",
                    List.of("consent", "opt-out", "recording")),
            new Policy("records", "Aggregate or mask record level data before sharing it.", "medium",
                    List.of("records", "log file", "dataset"))));
    }

    static final class Finding {
        final Policy policy;
        final int hits;
        final List<String> evidence;
        Finding(Policy policy, int hits, List<String> evidence) {
            this.policy = policy; this.hits = hits; this.evidence = evidence;
        }
    }

    static final class EthicsSection {
        final String body;
        final List<Finding> findings;
        final boolean reviewRequired;
        final String error;
        EthicsSection(String body, List<Finding> findings, boolean reviewRequired, String error) {
            this.body = body; this.findings = findings; this.reviewRequired = reviewRequired; this.error = error;
        }
        boolean ok() { return error == null; }
    }

    static String redact(String token) {
        if (token.contains("@")) return "<redacted-email>";
        StringBuilder sb = new StringBuilder();
        for (char c : token.toCharArray()) sb.append(Character.isDigit(c) ? '#' : c);
        return sb.toString();
    }

    /** Scans the statement for a policy's indicators and collects the redacted evidence tokens. */
    static Finding detect(String statement, Policy policy) {
        String lower = statement.toLowerCase(Locale.ROOT);
        int hits = 0;
        for (String indicator : policy.indicators) {
            for (int at = lower.indexOf(indicator); at >= 0; at = lower.indexOf(indicator, at + 1)) hits++;
        }
        Set<String> evidence = new TreeSet<>();
        for (String token : statement.split("\\s+")) {
            String tokenLower = token.toLowerCase(Locale.ROOT);
            for (String indicator : policy.indicators)
                if (tokenLower.contains(indicator)) { evidence.add(redact(token)); break; }
        }
        return new Finding(policy, hits, new ArrayList<>(evidence));
    }

    /** Builds the ethics section; `suppressed` simulates an author dropping required disclosures. */
    static EthicsSection generate(String statement, Set<String> suppressed, List<Policy> policies) {
        List<Finding> findings = new ArrayList<>();
        for (Policy policy : policies) {
            Finding finding = detect(statement, policy);
            if (finding.hits > 0) findings.add(finding);
        }
        findings.sort((a, b) -> SEVERITY.get(b.policy.severity) != SEVERITY.get(a.policy.severity)
                ? Integer.compare(SEVERITY.get(b.policy.severity), SEVERITY.get(a.policy.severity))
                : a.policy.id.compareTo(b.policy.id));
        for (Finding finding : findings)
            if (suppressed.contains(finding.policy.id) && finding.policy.severity.equals("high"))
                return new EthicsSection(null, findings, true, "missing required disclosure: " + finding.policy.id);
        boolean reviewRequired = false;
        for (Finding finding : findings)
            if (suppressed.contains(finding.policy.id) || !finding.policy.severity.equals("high")) reviewRequired = true;

        StringBuilder sb = new StringBuilder("## Ethics\n");
        if (findings.isEmpty()) {
            sb.append("No privacy or security implications detected in the problem statement.\n");
            sb.append("Detected indicators: 0\n");
        } else {
            for (Finding finding : findings) {
                if (suppressed.contains(finding.policy.id)) continue;      // a suppressed clause is never rendered
                sb.append("- [").append(finding.policy.id).append("/").append(finding.policy.severity).append("] ")
                        .append(finding.policy.clause).append(" Evidence: ")
                        .append(finding.evidence.isEmpty() ? "none" : String.join(", ", finding.evidence)).append('\n');
            }
            int total = 0;
            for (Finding finding : findings) total += finding.hits;
            sb.append("Detected indicators: ").append(total).append(" across ").append(findings.size())
                    .append(" policies; review flag: ").append(reviewRequired).append('\n');
        }
        return new EthicsSection(sb.toString(), findings, reviewRequired, null);
    }

    static String riskyStatement() {
        return "Students submit an email address (student@example.edu) and an identifier records-9988776655, "
                + "then use an exploit against a vulnerable service protected by a password.";
    }

    public static void main(String[] args) {
        List<Policy> rules = policies();
        EthicsSection risky = generate(riskyStatement(), Set.of(), rules);
        check(risky.ok(), "a fully disclosed risky problem must generate a section: " + risky.error);
        check(risky.findings.stream().map(f -> f.policy.id).toList().equals(List.of("credential", "dual-use", "pii", "records")),
                "triggered rules must be ordered by severity then id: " + risky.findings.stream().map(f -> f.policy.id).toList());
        check(risky.findings.get(0).policy.severity.equals("high") && risky.findings.get(3).policy.severity.equals("medium"),
                "high severity clauses must precede medium ones");
        check(risky.body.contains("[pii/high]") && risky.body.contains("[credential/high]") && risky.body.contains("[dual-use/high]")
                        && risky.body.contains("[records/medium]"),
                "every triggered rule must contribute its clause: " + risky.body);

        // --- coverage: the section mentions every triggered policy, verified by an independent scan.
        check(oracleCoverage(risky) == risky.findings.size(),
                "coverage oracle mismatch: " + oracleCoverage(risky) + " vs " + risky.findings.size());
        check(risky.body.contains("Detected indicators: 6 across 4 policies"),
                "the coverage line must count hits and policies: " + risky.body.lines().filter(l -> l.startsWith("Detected")).toList());

        // --- redaction: the section must never echo the raw email or the raw digit run.
        check(risky.body.contains("<redacted-email>") && risky.body.contains("records-##########"),
                "evidence tokens must be redacted: " + risky.body);
        check(!risky.body.contains("student@example.edu") && !risky.body.contains("9988776655"),
                "raw personal data must never be republished: " + risky.body);
        check(redact("id-12345") .equals("id-#####") && redact("a@b.c").equals("<redacted-email>")
                        && redact("plain").equals("plain"),
                "redaction must mask digit runs and addresses only: " + redact("id-12345"));

        // --- benign statements: no findings, only the baseline clause, no false positives.
        EthicsSection benign = generate("Sort an array of integers and report the inversion count.", Set.of(), rules);
        check(benign.ok() && benign.findings.isEmpty() && benign.body.contains("No privacy or security implications")
                        && benign.body.contains("Detected indicators: 0") && !benign.reviewRequired,
                "a benign problem must produce only the baseline clause: " + benign.body);

        // --- fail-closed gating: a suppressed high clause refuses publication, a medium one only flags.
        EthicsSection refused = generate(riskyStatement(), Set.of("dual-use"), rules);
        check(!refused.ok() && refused.error.equals("missing required disclosure: dual-use"),
                "suppressing a high severity clause must refuse the section: " + refused.error);
        EthicsSection flagged = generate(riskyStatement(), Set.of("records"), rules);
        check(flagged.ok() && flagged.reviewRequired && !flagged.body.contains("[records/medium]"),
                "suppressing a medium clause must keep the section but raise the review flag: " + flagged.body);
        EthicsSection mediumOnly = generate("We record consent and provide an opt-out for the dataset.", Set.of(), rules);
        check(mediumOnly.findings.stream().anyMatch(f -> f.policy.id.equals("consent")) && mediumOnly.reviewRequired,
                "a consent finding alone must mark the section for review: " + mediumOnly.findings.stream().map(f -> f.policy.id).toList());

        // --- determinism and rule-order independence.
        check(generate(riskyStatement(), Set.of(), rules).body.equals(risky.body),
                "regenerating the same statement must be byte identical");
        List<Policy> shuffled = policies();
        Collections.rotate(shuffled, 3);
        check(generate(riskyStatement(), Set.of(), shuffled).body.equals(risky.body),
                "the clause order must not depend on the rule insertion order");

        // --- boundaries: empty statement, and repeated indicators counted once per token.
        EthicsSection empty = generate("", Set.of(), rules);
        check(empty.ok() && empty.findings.isEmpty() && !empty.reviewRequired,
                "an empty statement must yield the baseline section: " + empty.body);
        EthicsSection repeated = generate("password password token credential", Set.of(), rules);
        check(repeated.findings.size() == 1 && repeated.body.contains("Detected indicators: 4")
                        && repeated.body.contains("credential, password, token"),
                "repeated indicators must be counted but evidence tokens de-duplicated: " + repeated.body);

        // --- independent oracles: char-scan indicator detection and a clause recount.
        check(oracleTriggered(riskyStatement(), rules).equals(risky.findings.stream().map(f -> f.policy.id).toList()),
                "indicator oracle mismatch: " + oracleTriggered(riskyStatement(), rules));
        check(oracleTriggered("", rules).isEmpty() && oracleTriggered("plain sorting task", rules).isEmpty(),
                "the oracle must not invent findings for clean statements");

        System.out.println("triggered: " + risky.findings.stream().map(f -> f.policy.id).toList()
                + " (" + risky.body.lines().filter(l -> l.startsWith("Detected")).toList() + ")");
        System.out.println("redacted evidence sample: " + risky.findings.get(2).evidence);
        System.out.println("benign section: " + benign.body.lines().skip(1).toList());
        System.out.println("PASS");
    }

    /** Oracle: counts the policy markers in the rendered section without using the findings list. */
    static int oracleCoverage(EthicsSection section) {
        int count = 0;
        for (Policy policy : policies()) if (section.body.contains("[" + policy.id + "/")) count++;
        return count;
    }

    /** Oracle: detects indicators with a manual character window scan instead of String.indexOf. */
    static List<String> oracleTriggered(String statement, List<Policy> rules) {
        String lower = statement.toLowerCase(Locale.ROOT);
        List<String> triggered = new ArrayList<>();
        for (Policy policy : rules) {
            boolean hit = false;
            for (String indicator : policy.indicators)
                for (int start = 0; start + indicator.length() <= lower.length() && !hit; start++)
                    if (lower.startsWith(indicator, start)) hit = true;
            if (hit) triggered.add(policy.id);
        }
        triggered.sort((a, b) -> {
            Policy pa = null, pb = null;
            for (Policy p : rules) { if (p.id.equals(a)) pa = p; if (p.id.equals(b)) pb = p; }
            int cmp = Integer.compare(SEVERITY.get(pb.severity), SEVERITY.get(pa.severity));
            return cmp != 0 ? cmp : a.compareTo(b);
        });
        return triggered;
    }
}
