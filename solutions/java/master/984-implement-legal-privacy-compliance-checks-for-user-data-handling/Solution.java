/** Task 984 - Implement legal/privacy compliance checks for user data handling in platform.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  A data-protection audit over a field inventory plus subject-rights tooling. Each field declares its
 *  classification, purpose, storage region, retention, consent, encryption and transfer adequacy, and five
 *  rules are evaluated: marketing fields need consent, retention must stay inside the classification limit,
 *  cross-border storage needs an adequacy decision, sensitive/financial data must be encrypted at rest and a
 *  field without a purpose violates data minimisation. Subjects can export everything stored about them
 *  (data-subject access request) and be erased across every table with an idempotent tombstone.
 *  Simplified: an in-memory inventory and row store with five static rules instead of a real data catalogue,
 *  DSR pipeline and regulator-specific rule packs; the retention/consent boundaries, the export completeness
 *  and the idempotent erasure properties are the real ones.
 *  Verified: a compliant inventory passing, one crafted violation per rule against an exact expected set, the
 *  retention and consent boundaries, DSAR export completeness with no other subject leaking, idempotent
 *  erasure that preserves other subjects, the empty-inventory boundary, unknown-classification rejection and
 *  determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final Map<String, Integer> RETENTION_DAYS = Map.of(
            "PUBLIC", 3650, "PII", 365, "SENSITIVE", 180, "FINANCIAL", 2555);
    static final Set<String> HARD_CLASSES = Set.of("SENSITIVE", "FINANCIAL");
    static final String HOME_REGION = "eu-west";

    static final class Field {
        final String table, name, classification, purpose, region;
        final int retentionDays; final boolean consent, encrypted, adequacy;
        Field(String table, String name, String classification, String purpose, String region, int retentionDays,
              boolean consent, boolean encrypted, boolean adequacy) {
            if (!RETENTION_DAYS.containsKey(classification)) throw new IllegalArgumentException(
                    "unknown classification " + classification);
            this.table = table; this.name = name; this.classification = classification; this.purpose = purpose;
            this.region = region; this.retentionDays = retentionDays; this.consent = consent;
            this.encrypted = encrypted; this.adequacy = adequacy;
        }
        String key() { return table + "." + name; }
    }

    /** All rule violations as "rule@table.field", ordered for reproducible reports. */
    static List<String> audit(List<Field> inventory) {
        List<String> violations = new ArrayList<>();
        for (Field field : inventory) {
            if (field.purpose.equals("marketing") && !field.consent) violations.add("consent-required@" + field.key());
            if (field.retentionDays > RETENTION_DAYS.get(field.classification)) {
                violations.add("retention-limit@" + field.key());
            }
            if (!field.region.equals(HOME_REGION) && !field.adequacy) violations.add("cross-border-transfer@" + field.key());
            if (HARD_CLASSES.contains(field.classification) && !field.encrypted) {
                violations.add("encryption-required@" + field.key());
            }
            if (field.purpose.equals("none")) violations.add("purpose-limitation@" + field.key());
        }
        Collections.sort(violations);
        return violations;
    }

    static String report(List<Field> inventory, Set<String> completedChecks) {
        List<String> violations = audit(inventory);
        if (!violations.isEmpty()) return "FAIL: " + violations.size() + " violations, first " + violations.get(0);
        List<String> missing = new ArrayList<>();
        for (String required : new TreeSet<>(Set.of("consent-register", "retention-policy", "transfer-register"))) {
            if (!completedChecks.contains(required)) missing.add(required);
        }
        return missing.isEmpty() ? "PASS" : "FAIL: missing checklist " + missing;
    }

    static final class Table {
        final String name; final List<Map<String, String>> rows;
        Table(String name, List<Map<String, String>> rows) { this.name = name; this.rows = rows; }
    }

    static Map<String, String> row(String... kv) {
        Map<String, String> out = new TreeMap<>();
        for (int i = 0; i < kv.length; i += 2) out.put(kv[i], kv[i + 1]);
        return out;
    }

    /** Data-subject access request: every row the subject has, table by table, and nothing else. */
    static Map<String, List<Map<String, String>>> exportSubject(List<Table> database, String subjectId) {
        Map<String, List<Map<String, String>>> out = new TreeMap<>();
        for (Table table : database) {
            List<Map<String, String>> rows = new ArrayList<>();
            for (Map<String, String> candidate : table.rows) {
                if (subjectId.equals(candidate.get("user_id"))) rows.add(new TreeMap<>(candidate));
            }
            if (!rows.isEmpty()) out.put(table.name, rows);
        }
        return out;
    }

    /** Right to erasure: drops every row of the subject and records an idempotent tombstone. */
    static List<Table> erase(List<Table> database, String subjectId, List<String> tombstones) {
        List<Table> out = new ArrayList<>();
        int removed = 0;
        for (Table table : database) {
            List<Map<String, String>> rows = new ArrayList<>();
            for (Map<String, String> candidate : table.rows) {
                if (subjectId.equals(candidate.get("user_id"))) removed++;
                else rows.add(candidate);
            }
            out.add(new Table(table.name, rows));
        }
        String tombstone = "erased " + subjectId;
        if (!tombstones.contains(tombstone)) tombstones.add(tombstone);
        return out;
    }

    static List<Field> compliant() {
        return List.of(new Field("users", "email", "PII", "service", HOME_REGION, 365, false, true, true),
                new Field("users", "name", "PII", "service", HOME_REGION, 200, false, true, true),
                new Field("payments", "card", "FINANCIAL", "payments", HOME_REGION, 2555, false, true, true),
                new Field("users", "device", "PUBLIC", "analytics", HOME_REGION, 900, false, false, true));
    }

    static List<Field> violating() {
        return List.of(new Field("users", "email", "PII", "marketing", HOME_REGION, 30, false, true, true),
                new Field("users", "phone", "PII", "service", HOME_REGION, 400, false, true, true),
                new Field("users", "ip", "PII", "service", "us-east", 90, false, true, false),
                new Field("payments", "iban", "FINANCIAL", "payments", HOME_REGION, 100, false, false, true),
                new Field("users", "orphan", "PII", "none", HOME_REGION, 30, false, true, true));
    }

    static List<Table> database() {
        return new ArrayList<>(List.of(
                new Table("users", new ArrayList<>(List.of(row("user_id", "u1", "email", "a@x"),
                        row("user_id", "u2", "email", "b@x")))),
                new Table("events", new ArrayList<>(List.of(row("user_id", "u1", "kind", "login"),
                        row("user_id", "u2", "kind", "purchase"), row("user_id", "u1", "kind", "logout"))))));
    }

    public static void main(String[] args) {
        Set<String> allChecks = Set.of("consent-register", "retention-policy", "transfer-register");
        check(audit(compliant()).isEmpty() && report(compliant(), allChecks).equals("PASS"),
                "a compliant inventory must pass with a complete checklist: " + audit(compliant()));

        List<String> expected = List.of("consent-required@users.email", "cross-border-transfer@users.ip",
                "encryption-required@payments.iban", "purpose-limitation@users.orphan",
                "retention-limit@users.phone");
        check(audit(violating()).equals(expected) && report(violating(), allChecks).startsWith("FAIL: 5 violations"),
                "each rule must fire exactly once on the crafted inventory: " + audit(violating()));

        List<Field> boundaries = List.of(
                new Field("a", "at-limit", "PII", "service", HOME_REGION, 365, false, true, true),
                new Field("a", "over-limit", "SENSITIVE", "service", HOME_REGION, 181, false, true, true),
                new Field("a", "exact-hard-limit", "SENSITIVE", "service", HOME_REGION, 180, false, true, true),
                new Field("a", "opted-in", "PII", "marketing", HOME_REGION, 30, true, true, true),
                new Field("a", "service-no-consent", "PII", "service", HOME_REGION, 30, false, true, true),
                new Field("a", "abroad-clear", "PII", "service", "us-east", 30, false, true, true));
        check(audit(boundaries).equals(List.of("retention-limit@a.over-limit")),
                "retention and consent boundaries must be inclusive: " + audit(boundaries));

        Map<String, List<Map<String, String>>> export = exportSubject(database(), "u1");
        int exportedRows = export.values().stream().mapToInt(List::size).sum();
        boolean leaked = export.values().stream().flatMap(List::stream)
                .anyMatch(r -> r.containsValue("u2") || r.containsValue("b@x") || r.containsValue("purchase"));
        check(export.keySet().equals(Set.of("users", "events")) && exportedRows == 3
                        && export.get("events").size() == 2 && !leaked,
                "a DSAR export must contain exactly the subject's rows: " + export);
        check(exportSubject(database(), "u9").isEmpty() && exportSubject(List.of(), "u1").isEmpty(),
                "an unknown subject must export nothing");

        List<String> tombstones = new ArrayList<>();
        List<Table> erased = erase(database(), "u1", tombstones);
        List<Table> twice = erase(erased, "u1", tombstones);
        int remainingU1 = 0, remainingOthers = 0;
        for (Table table : twice) {
            for (Map<String, String> r : table.rows) {
                if ("u1".equals(r.get("user_id"))) remainingU1++;
                if ("u2".equals(r.get("user_id"))) remainingOthers++;
            }
        }
        check(remainingU1 == 0 && remainingOthers == 2 && tombstones.size() == 1
                        && tombstones.get(0).equals("erased u1") && exportSubject(twice, "u1").isEmpty(),
                "erasure must clear the subject, keep other subjects and stay idempotent: " + tombstones);
        check(erase(twice, "u1", tombstones).size() == twice.size() && tombstones.size() == 1
                        && exportSubject(database(), "u2").values().stream().mapToInt(List::size).sum() == 2,
                "erasing twice must change nothing and other subjects must keep their data");

        boolean rejected = false;
        try {
            new Field("a", "bad", "PHI", "service", HOME_REGION, 10, false, true, true);
        } catch (IllegalArgumentException e) {
            rejected = e.getMessage().contains("unknown classification");
        }
        check(rejected && audit(List.of()).isEmpty() && report(List.of(), allChecks).equals("PASS")
                        && report(List.of(), Set.of()).startsWith("FAIL: missing checklist"),
                "an unknown classification must be rejected and an empty inventory audited cleanly");

        check(audit(violating()).equals(audit(violating())) && exportSubject(database(), "u1").equals(export)
                        && audit(compliant()).equals(audit(compliant())),
                "re-running the audit and the export must be deterministic");

        System.out.println("fields=" + compliant().size() + " violations=" + audit(violating()).size()
                + " exportedRows=" + exportedRows);
        System.out.println("violations=" + audit(violating()));
        System.out.println("PASS");
    }
}
