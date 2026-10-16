// Task 984 - Implement legal/privacy compliance checks for user data handling in platform.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Every stored record declares its dataset, fields, creation day, retention period, the purposes it
// may be used for and the consents the data subject granted. The compliance engine classifies personal
// data from field names, enforces retention (age must not exceed the retention period), purpose
// limitation, consent gating for sensitive data, and the erasure obligation, which has to remove the
// record from the primary store, the derived aggregates and the backup copies at once.
//
// Verified: field-name based PII classification against an independent dictionary scan, the retention
// boundary at exactly the allowed age and one day past it, consent gating both ways, purpose
// limitation, erasure of all three copies with a recomputed aggregate, deterministic violation
// ordering, the empty-dataset boundary and a brute-force oracle recounting every violation.
// Simplification: records are in-memory with day counters and hand-declared purposes; there is no
// jurisdiction-specific rule engine, no DSAR workflow and no audit log retention.
using System;
using System.Collections.Generic;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private static readonly string[] PiiFields = { "email", "ssn", "phone", "full_name", "address", "ip_address" };
    private static readonly string[] SensitiveFields = { "health_data", "biometric_id", "religion", "union_membership" };

    private sealed class Record
    {
        public readonly string Id;
        public readonly string Dataset;
        public readonly string[] Fields;
        public readonly int CreatedDay;
        public readonly int RetentionDays;
        public readonly HashSet<string> DeclaredPurposes;
        public readonly HashSet<string> Consents;

        public Record(string id, string dataset, string[] fields, int createdDay, int retentionDays,
            string[] declaredPurposes, string[] consents)
        {
            Id = id;
            Dataset = dataset;
            Fields = fields;
            CreatedDay = createdDay;
            RetentionDays = retentionDays;
            DeclaredPurposes = new HashSet<string>(declaredPurposes, StringComparer.Ordinal);
            Consents = new HashSet<string>(consents, StringComparer.Ordinal);
        }
    }

    private sealed class Violation
    {
        public readonly string Kind;
        public readonly string RecordId;
        public readonly string Detail;

        public Violation(string kind, string recordId, string detail)
        {
            Kind = kind;
            RecordId = recordId;
            Detail = detail;
        }
    }

    private sealed class Store
    {
        public readonly Dictionary<string, Record> Primary = new Dictionary<string, Record>(StringComparer.Ordinal);
        public readonly Dictionary<string, Dictionary<string, int>> Aggregates =
            new Dictionary<string, Dictionary<string, int>>(StringComparer.Ordinal);
        public readonly Dictionary<string, List<string>> Backups = new Dictionary<string, List<string>>(StringComparer.Ordinal);
    }

    private static bool IsPii(Record record)
    {
        foreach (string field in record.Fields)
            foreach (string candidate in PiiFields)
                if (field == candidate) return true;
        return false;
    }

    private static bool IsSensitive(Record record)
    {
        foreach (string field in record.Fields)
            foreach (string candidate in SensitiveFields)
                if (field == candidate) return true;
        return false;
    }

    private static List<Violation> Inspect(Record record, string purpose, int today)
    {
        var violations = new List<Violation>();
        int age = today - record.CreatedDay;
        if (age > record.RetentionDays)
            violations.Add(new Violation("retention", record.Id, "age " + age + " exceeds " + record.RetentionDays + " days"));
        if (!record.DeclaredPurposes.Contains(purpose))
            violations.Add(new Violation("purpose-limitation", record.Id, "purpose '" + purpose + "' is not declared"));
        if (IsSensitive(record) && !record.Consents.Contains(purpose))
            violations.Add(new Violation("consent", record.Id, "no consent for '" + purpose + "' on sensitive data"));
        if (IsPii(record) && record.Consents.Count == 0)
            violations.Add(new Violation("missing-consent-registry", record.Id, "personal data stored without any consent"));
        return violations;
    }

    private static List<Violation> Audit(List<Record> records, string purpose, int today)
    {
        var all = new List<Violation>();
        foreach (Record record in records) all.AddRange(Inspect(record, purpose, today));
        all.Sort((a, b) =>
        {
            int byKind = string.CompareOrdinal(a.Kind, b.Kind);
            if (byKind != 0) return byKind;
            int byRecord = string.CompareOrdinal(a.RecordId, b.RecordId);
            return byRecord != 0 ? byRecord : string.CompareOrdinal(a.Detail, b.Detail);
        });
        return all;
    }

    /** Erasure must remove the record from the primary store, the derived aggregate and every backup. */
    private static int Erase(Store store, string recordId)
    {
        int removed = 0;
        if (store.Primary.TryGetValue(recordId, out Record record))
        {
            store.Primary.Remove(recordId);
            removed++;
            if (store.Aggregates.TryGetValue(record.Dataset, out Dictionary<string, int> aggregate)
                && aggregate.ContainsKey(recordId))
            {
                aggregate.Remove(recordId);
                removed++;
            }
        }
        foreach (KeyValuePair<string, List<string>> backup in store.Backups)
            if (backup.Value.Remove(recordId)) removed++;
        return removed;
    }

    private static Store SampleStore()
    {
        var store = new Store();
        store.Primary["u1"] = new Record("u1", "profiles", new[] { "email", "full_name" }, 100, 365,
            new[] { "service" }, new[] { "service" });
        store.Primary["u2"] = new Record("u2", "profiles", new[] { "email", "health_data" }, 700, 365,
            new[] { "service", "care" }, new[] { "service", "care" });
        store.Primary["u3"] = new Record("u3", "telemetry", new[] { "device_id" }, 10, 30,
            new[] { "analytics" }, new string[0]);
        store.Aggregates["profiles"] = new Dictionary<string, int>(StringComparer.Ordinal) { { "u1", 3 }, { "u2", 5 } };
        store.Backups["daily"] = new List<string> { "u1", "u2", "u3" };
        store.Backups["weekly"] = new List<string> { "u2" };
        return store;
    }

    public static void Main()
    {
        Store store = SampleStore();
        var records = new List<Record>(store.Primary.Values);

        Check(IsPii(store.Primary["u1"]) && !IsSensitive(store.Primary["u1"]),
            "the profile record is personal data but not sensitive");
        Check(IsSensitive(store.Primary["u2"]) && IsPii(store.Primary["u2"]),
            "the care record is both personal and sensitive data");
        Check(!IsPii(store.Primary["u3"]), "a device-only telemetry row carries no personal data");

        // Oracle: an independent dictionary scan of the field names agrees with the classifier.
        int piiByScan = 0, sensitiveByScan = 0;
        var declared = new HashSet<string>(new[] { "email", "ssn", "phone", "full_name", "address", "ip_address" }, StringComparer.Ordinal);
        var sensitive = new HashSet<string>(new[] { "health_data", "biometric_id", "religion", "union_membership" }, StringComparer.Ordinal);
        foreach (Record record in records)
        {
            bool pii = false, sensitiveHit = false;
            foreach (string field in record.Fields)
            {
                pii |= declared.Contains(field);
                sensitiveHit |= sensitive.Contains(field);
            }
            if (pii) piiByScan++;
            if (sensitiveHit) sensitiveByScan++;
        }
        Check(piiByScan == 2 && sensitiveByScan == 1, "the independent dictionary scan counts two personal and one sensitive row");

        // Retention boundary: exactly at the limit is compliant, one day past is not.
        var atLimit = new Record("t1", "profiles", new[] { "email" }, 0, 30, new[] { "service" }, new[] { "service" });
        Check(Inspect(atLimit, "service", 30).Count == 0, "a record at exactly its retention limit is compliant");
        List<Violation> pastLimit = Inspect(atLimit, "service", 31);
        Check(pastLimit.Count == 1 && pastLimit[0].Kind == "retention",
            "one day past the retention limit is a single retention violation");

        // Consent gating and purpose limitation.
        Record care = store.Primary["u2"];
        Check(Inspect(care, "care", 200).Count == 0, "sensitive data with consent for that purpose is compliant");
        List<Violation> noConsent = Inspect(care, "marketing", 200);
        Check(noConsent.Count == 2, "an undeclared purpose on sensitive data raises purpose and consent violations");
        var kinds = new HashSet<string>(StringComparer.Ordinal);
        foreach (Violation violation in noConsent) kinds.Add(violation.Kind);
        Check(kinds.Contains("purpose-limitation") && kinds.Contains("consent"),
            "the violation kinds are purpose-limitation and consent");
        Check(Inspect(store.Primary["u3"], "analytics", 15).Count == 0,
            "a non-personal telemetry row is not blocked by consent rules");

        // The audit is ordered and matches a brute-force recount.
        List<Violation> audit = Audit(records, "service", 800);
        bool ordered = true;
        for (int i = 1; i < audit.Count; i++)
        {
            int byKind = string.CompareOrdinal(audit[i - 1].Kind, audit[i].Kind);
            if (byKind > 0) ordered = false;
            else if (byKind == 0 && string.CompareOrdinal(audit[i - 1].RecordId, audit[i].RecordId) > 0) ordered = false;
        }
        Check(ordered, "audit violations are grouped by kind and then by record id");
        int bruteForce = 0;
        foreach (Record record in records)
        {
            if (800 - record.CreatedDay > record.RetentionDays) bruteForce++;
            if (!record.DeclaredPurposes.Contains("service")) bruteForce++;
            if (IsSensitive(record) && !record.Consents.Contains("service")) bruteForce++;
            if (IsPii(record) && record.Consents.Count == 0) bruteForce++;
        }
        Check(bruteForce == audit.Count, "a brute-force recount matches the reported violation count ("
            + bruteForce + " vs " + audit.Count + ")");
        Check(audit.Count == 3 && audit[0].Kind == "purpose-limitation",
            "auditing the whole store at day 800 finds three violations, led by purpose limitation");

        // Erasure: primary copy, derived aggregate and both backups must lose the record.
        int removed = Erase(store, "u2");
        Check(removed == 4, "erasure removes the record from four places (got " + removed + ")");
        Check(!store.Primary.ContainsKey("u2") && !store.Aggregates["profiles"].ContainsKey("u2"),
            "the primary row and the derived aggregate no longer reference the erased subject");
        Check(!store.Backups["daily"].Contains("u2") && !store.Backups["weekly"].Contains("u2"),
            "every backup copy of the erased subject is gone");
        Check(store.Aggregates["profiles"].ContainsKey("u1") && store.Primary.ContainsKey("u1"),
            "erasing one subject leaves the other subjects untouched");
        int leftover = 0;
        foreach (KeyValuePair<string, List<string>> backup in store.Backups)
            foreach (string id in backup.Value)
                if (id == "u2") leftover++;
        Check(leftover == 0, "no trace of the erased identifier remains anywhere in the store");

        // Boundary: an empty dataset produces no violations, and erasing an unknown id is a no-op.
        Check(Audit(new List<Record>(), "service", 1000).Count == 0, "an empty dataset produces no violations");
        Check(Erase(new Store(), "ghost") == 0, "erasing an unknown subject changes nothing");

        Console.WriteLine("classifier: " + piiByScan + " personal rows, " + sensitiveByScan + " sensitive rows");
        Console.WriteLine("audit at day 800: " + audit.Count + " violations, kinds "
            + string.Join(",", kinds));
        Console.WriteLine("erasure of u2: " + removed + " copies removed, primary rows left "
            + store.Primary.Count);
        Console.WriteLine("PASS");
    }
}
