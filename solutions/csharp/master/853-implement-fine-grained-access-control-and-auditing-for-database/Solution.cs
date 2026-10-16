// Task 853 - Implement fine-grained access control and auditing for database operations.
// (Tier: Master, CTCI Ch 14 — Databases)
//
// A deterministic model of an authorisation engine that guards every row-level database operation and
// writes a tamper-evident audit record for each decision.
//
// Policy model: subjects own roles, roles form a DAG (role inheritance is resolved transitively), and a
// rule binds a role to an action, a table, an optional ownership predicate (owner column == subject) and
// an optional column mask. Evaluation uses the standard enterprise semantics:
//
//     matching = rules whose role is in the subject's closure, whose action and table match and whose
//                ownership predicate holds for the target row
//     decision = allow iff some matching rule allows and no matching rule denies (deny overrides), with
//                default deny when nothing matches
//     masking  = an allow rule that names the requested column masks the value that is returned
//
// The engine answers a request from an index keyed by (action, table) over the subject's rules, while the
// independent oracle scans every rule of the policy for every request: the differential test compares the
// two on 4000 requests, and the evaluations counters prove the index really does less work.
//
// Auditing: every decision is appended to a hash chain entry
//     hash = mix(prevHash, seq, subject, action, table, row, column, allowed)
// (a hand-written 64-bit mixing function standing in for a keyed MAC). The chain is re-verified by
// recomputation, so a flipped decision, a swapped pair of entries or a forged hash is localised, and a
// signed (head, count) checkpoint catches a log that was cut short.
//
// What is simplified (documented, not behavioural): the "database" is an in-memory row store, values are
// synthetic integers, the mask is a fixed token instead of redaction or format-preserving encryption, the
// MAC is a non-keyed mixing function, and there is no persistence, network or thread. Relation, rule and
// request counts are typed parameters, so the same engine runs at 200 rows / 4000 requests here and
// larger by changing three constants.
//
// Invariants verified: the transitive role closure equals an independent DFS closure; cycles and dangling
// parents in the role graph are rejected; allow/deny/mask semantics match the brute-force oracle on every
// one of the 4000 random requests (allowed flag, mask flag, reason and returned value); deny overrides
// allow, both for a role-inherited grant and for a global rule; an ownership predicate cannot be
// bypassed through another role's grant; a masked column never leaks its raw value in any response; the
// index evaluates fewer than half the rules of the oracle; the audit chain has exactly one record per
// decision and verifies; tampering, reordering, forging and truncation are all detected at the right
// position; the log stores identifiers only; the run is reproducible from the seed.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng853
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound)
    {
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        return (int)(Next(ref state) % (ulong)bound);
    }
}

internal sealed class Role853
{
    internal int Id;
    internal string Name = "";
    internal List<int> Parents = new List<int>();
}

internal sealed class Row853
{
    internal int Id;
    internal int Table;
    internal int Owner;
    internal int[] Values = Array.Empty<int>();
}

internal sealed class Rule853
{
    internal int Id;
    internal int Role;          // -1 = any role
    internal string Action = "";
    internal int Table;         // -1 = any table
    internal int OwnerOnly;     // 1 = only rows owned by the subject
    internal int MaskColumn;    // -1 = no masking, otherwise the column whose value is masked
    internal bool Allow;
}

internal sealed class Policy853
{
    internal readonly List<Role853> Roles = new List<Role853>();
    internal readonly List<Rule853> Rules = new List<Rule853>();

    internal Role853 AddRole(string name, params int[] parents)
    {
        var role = new Role853 { Id = Roles.Count, Name = name };
        role.Parents.AddRange(parents);
        Roles.Add(role);
        return role;
    }

    internal Rule853 AddRule(int role, string action, int table, int ownerOnly, int maskColumn, bool allow)
    {
        var rule = new Rule853
        {
            Id = Rules.Count, Role = role, Action = action, Table = table,
            OwnerOnly = ownerOnly, MaskColumn = maskColumn, Allow = allow,
        };
        Rules.Add(rule);
        return rule;
    }

    // Rejects a role graph with a cycle or a dangling parent: inheritance would be ill-defined.
    internal void Validate()
    {
        for (int start = 0; start < Roles.Count; start++)
        {
            var stack = new Stack<int>(Roles[start].Parents);
            var seen = new HashSet<int>();
            while (stack.Count > 0)
            {
                int current = stack.Pop();
                if (current == start) throw new ArgumentException("the role graph contains a cycle at role " + start);
                if (current < 0 || current >= Roles.Count)
                    throw new ArgumentException("a role references an unknown parent " + current);
                if (!seen.Add(current)) continue;
                foreach (int parent in Roles[current].Parents) stack.Push(parent);
            }
        }
    }
}

internal sealed class Decision853
{
    internal bool Allowed;
    internal bool Masked;
    internal int Value;
    internal string Reason = "";
}

internal sealed class AuditEntry853
{
    internal long Seq;
    internal int Subject;
    internal string Action = "";
    internal int Table;
    internal int Row;
    internal int Column;
    internal bool Allowed;
    internal long PrevHash;
    internal long Hash;
}

internal static class Audit853
{
    // 64-bit mixing with avalanche: a deterministic stand-in for a keyed MAC.
    internal static long Mix(long hash, long value)
    {
        ulong h = (ulong)hash ^ ((ulong)value + 0x9E3779B97F4A7C15UL + ((ulong)hash << 6) + ((ulong)hash >> 2));
        h ^= h >> 33;
        h *= 0xFF51AFD7ED558CCDUL;
        h ^= h >> 33;
        h *= 0xC4CEB9FE1A85EC53UL;
        h ^= h >> 33;
        return (long)h;
    }

    internal static long HashText(string text)
    {
        long h = 2166136261L;
        foreach (char c in text) h = Mix(h, c);
        return h;
    }

    internal static long EntryHash(long prevHash, AuditEntry853 entry)
    {
        long h = Mix(prevHash, entry.Seq);
        h = Mix(h, entry.Subject * 31 + entry.Table * 7 + entry.Row);
        h = Mix(h, entry.Column * 13 + (entry.Allowed ? 1 : 0));
        h = Mix(h, HashText(entry.Action));
        return h;
    }
}

internal sealed class AuditLog853
{
    internal readonly List<AuditEntry853> Entries = new List<AuditEntry853>();
    private long head;
    private long count;

    internal long Head => head;
    internal long Count => count;

    internal void Append(int subject, string action, int table, int row, int column, bool allowed)
    {
        var entry = new AuditEntry853
        {
            Seq = count, Subject = subject, Action = action, Table = table, Row = row, Column = column,
            Allowed = allowed, PrevHash = head,
        };
        entry.Hash = Audit853.EntryHash(head, entry);
        Entries.Add(entry);
        head = entry.Hash;
        count++;
    }

    internal void Restore(long restoredCount, long restoredHead)
    {
        count = restoredCount;
        head = restoredHead;
    }

    internal void RemoveLast(int howMany)
    {
        for (int i = 0; i < howMany && Entries.Count > 0; i++) Entries.RemoveAt(Entries.Count - 1);
        // The checkpoint is deliberately left untouched, which is what makes a truncation detectable.
    }

    // Returns the index of the first entry that does not match the chain, or -1 when the whole log
    // verifies. A mismatch with the (count, head) checkpoint is reported as -2.
    internal long Verify(long expectedCount, long expectedHead)
    {
        if (Entries.Count != expectedCount || count != expectedCount) return -2;
        long prev = 0;
        for (int i = 0; i < Entries.Count; i++)
        {
            AuditEntry853 entry = Entries[i];
            if (entry.Seq != i || entry.PrevHash != prev) return i;
            if (Audit853.EntryHash(prev, entry) != entry.Hash) return i;
            prev = entry.Hash;
        }
        if (Entries.Count > 0 && prev != expectedHead) return -2;
        return -1;
    }
}

internal sealed class Database853
{
    private readonly List<Row853> rows = new List<Row853>();

    internal Database853(int tables, int rowsPerTable, int columns, int subjects)
    {
        Tables = tables;
        Columns = columns;
        Subjects = subjects;
        for (int t = 0; t < tables; t++)
        {
            for (int r = 0; r < rowsPerTable; r++)
            {
                var values = new int[columns];
                for (int c = 0; c < columns; c++) values[c] = 100000 + t * 10000 + r * 100 + c;
                rows.Add(new Row853 { Id = rows.Count, Table = t, Owner = r % subjects, Values = values });
            }
        }
    }

    internal int Tables { get; }
    internal int Columns { get; }
    internal int Subjects { get; }
    internal int RowCount => rows.Count;
    internal Row853 RowAt(int index) => rows[index];
}

internal sealed class AccessEngine853
{
    private readonly Database853 db;
    private readonly int subjectCount;
    private readonly HashSet<int>[] closure;
    private readonly Dictionary<string, List<Rule853>>[] index;
    private readonly Dictionary<string, List<Rule853>> globals = new Dictionary<string, List<Rule853>>();
    private readonly List<Rule853> allRules;

    internal long RuleEvaluations;

    internal AccessEngine853(Policy853 policy, Database853 db)
    {
        this.db = db;
        subjectCount = db.Subjects;
        policy.Validate();
        allRules = new List<Rule853>(policy.Rules);
        closure = new HashSet<int>[subjectCount];
        for (int s = 0; s < subjectCount; s++) closure[s] = ClosureOf(policy, s);
        // One index per role: the exact bucket holds the rules that name an action and table, the wildcard
        // bucket holds the rules that do not constrain the table.
        index = new Dictionary<string, List<Rule853>>[policy.Roles.Count];
        for (int r = 0; r < policy.Roles.Count; r++) index[r] = new Dictionary<string, List<Rule853>>();
        foreach (Rule853 rule in policy.Rules)
        {
            if (rule.Role < 0)
            {
                Add(globals, rule.Action + "|" + rule.Table, rule);
                continue;
            }
            Add(index[rule.Role], rule.Action + "|" + rule.Table, rule);
        }
    }

    private static void Add(Dictionary<string, List<Rule853>> map, string key, Rule853 rule)
    {
        if (!map.TryGetValue(key, out List<Rule853> list))
        {
            list = new List<Rule853>();
            map[key] = list;
        }
        list.Add(rule);
    }

    private static HashSet<int> ClosureOf(Policy853 policy, int subject)
    {
        var seen = new HashSet<int>();
        if (policy.Roles.Count == 0) return seen;
        var stack = new Stack<int>();
        stack.Push(subject % policy.Roles.Count);
        while (stack.Count > 0)
        {
            int role = stack.Pop();
            if (!seen.Add(role)) continue;
            foreach (int parent in policy.Roles[role].Parents) stack.Push(parent);
        }
        return seen;
    }

    internal HashSet<int> Closure(int subject) => new HashSet<int>(closure[subject]);
    internal long TotalRules => allRules.Count;

    internal Decision853 Evaluate(int subject, string action, int table, int rowId, int column)
    {
        if (subject < 0 || subject >= subjectCount) return new Decision853 { Reason = "unknown-subject" };
        if (table < 0 || table >= db.Tables) return new Decision853 { Reason = "unknown-table" };
        if (rowId < 0 || rowId >= db.RowCount) return new Decision853 { Reason = "unknown-row" };
        if (column < 0 || column >= db.Columns) return new Decision853 { Reason = "unknown-column" };
        Row853 row = db.RowAt(rowId);
        if (row.Table != table) return new Decision853 { Reason = "row-not-in-table" };

        bool allow = false, deny = false, mask = false;
        if (globals.TryGetValue(action + "|" + table, out List<Rule853> globalExact))
        {
            RuleEvaluations += globalExact.Count;
            Apply(globalExact, subject, row, column, ref allow, ref deny, ref mask);
        }
        if (globals.TryGetValue(action + "|-1", out List<Rule853> globalAny))
        {
            RuleEvaluations += globalAny.Count;
            Apply(globalAny, subject, row, column, ref allow, ref deny, ref mask);
        }
        foreach (int role in closure[subject])
        {
            if (index[role].TryGetValue(action + "|" + table, out List<Rule853> exact))
            {
                RuleEvaluations += exact.Count;
                Apply(exact, subject, row, column, ref allow, ref deny, ref mask);
            }
            if (index[role].TryGetValue(action + "|-1", out List<Rule853> wildcard))
            {
                RuleEvaluations += wildcard.Count;
                Apply(wildcard, subject, row, column, ref allow, ref deny, ref mask);
            }
        }
        if (!allow) return new Decision853 { Reason = deny ? "explicit-deny" : "default-deny" };
        if (deny) return new Decision853 { Reason = "explicit-deny" };
        if (mask) return new Decision853 { Allowed = true, Masked = true, Value = 0, Reason = "allow-masked" };
        return new Decision853 { Allowed = true, Value = row.Values[column], Reason = "allow" };
    }

    private static void Apply(List<Rule853> rules, int subject, Row853 row, int column,
        ref bool allow, ref bool deny, ref bool mask)
    {
        foreach (Rule853 rule in rules)
        {
            if (rule.Table >= 0 && rule.Table != row.Table) continue;
            if (rule.OwnerOnly == 1 && row.Owner != subject) continue;
            if (rule.Allow)
            {
                allow = true;
                if (rule.MaskColumn == column) mask = true;
            }
            else
            {
                deny = true;
            }
        }
    }
}

// The brute-force reference: it validates the same request shape, then walks every rule of the policy for
// every request and re-derives role reachability with its own BFS.
internal sealed class OracleEngine853
{
    private readonly Policy853 policy;
    private readonly Database853 db;
    internal long RuleEvaluations;

    internal OracleEngine853(Policy853 policy, Database853 db)
    {
        this.policy = policy;
        this.db = db;
    }

    internal Decision853 Evaluate(int subject, string action, int table, int rowId, int column)
    {
        if (subject < 0 || subject >= db.Subjects) return new Decision853 { Reason = "unknown-subject" };
        if (table < 0 || table >= db.Tables) return new Decision853 { Reason = "unknown-table" };
        if (rowId < 0 || rowId >= db.RowCount) return new Decision853 { Reason = "unknown-row" };
        if (column < 0 || column >= db.Columns) return new Decision853 { Reason = "unknown-column" };
        Row853 row = db.RowAt(rowId);
        if (row.Table != table) return new Decision853 { Reason = "row-not-in-table" };

        bool allow = false, deny = false, mask = false;
        foreach (Rule853 rule in policy.Rules)
        {
            RuleEvaluations++;
            if (rule.Role >= 0 && !Reachable(subject, rule.Role)) continue;
            if (!string.Equals(rule.Action, action, StringComparison.Ordinal)) continue;
            if (rule.Table >= 0 && rule.Table != table) continue;
            if (rule.OwnerOnly == 1 && row.Owner != subject) continue;
            if (rule.Allow)
            {
                allow = true;
                if (rule.MaskColumn == column) mask = true;
            }
            else
            {
                deny = true;
            }
        }
        if (!allow) return new Decision853 { Reason = deny ? "explicit-deny" : "default-deny" };
        if (deny) return new Decision853 { Reason = "explicit-deny" };
        if (mask) return new Decision853 { Allowed = true, Masked = true, Reason = "allow-masked" };
        return new Decision853 { Allowed = true, Value = row.Values[column], Reason = "allow" };
    }

    private bool Reachable(int subject, int role)
    {
        int direct = subject % policy.Roles.Count;
        if (direct == role) return true;
        var queue = new Queue<int>();
        queue.Enqueue(direct);
        var seen = new HashSet<int> { direct };
        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            foreach (int parent in policy.Roles[current].Parents)
            {
                if (parent == role) return true;
                if (seen.Add(parent)) queue.Enqueue(parent);
            }
        }
        return false;
    }
}

public static class Solution
{
    private const int Subjects = 12;
    private const int Tables = 4;
    private const int RowsPerTable = 50;
    private const int Columns = 8;
    private const int Requests = 4000;
    private static readonly string[] Actions = { "select", "update", "delete", "insert" };

    public static void Main()
    {
        var db = new Database853(Tables, RowsPerTable, Columns, Subjects);
        Policy853 policy = BuildPolicy();

        // 1-4. Role graph: transitive closure against an independent reachability, cycle and dangling
        // parent rejection, policy shape.
        var engine = new AccessEngine853(policy, db);
        var oracle = new OracleEngine853(policy, db);
        bool closureMatches = true;
        for (int s = 0; s < Subjects; s++)
        {
            HashSet<int> mine = engine.Closure(s);
            for (int r = 0; r < policy.Roles.Count; r++)
                if (mine.Contains(r) != BruteForceReachable(policy, s, r)) closureMatches = false;
        }
        Check(closureMatches, "the engine's transitive role closure equals the independent DFS closure");
        var cyclic = new Policy853();
        cyclic.AddRole("a", 1);
        cyclic.AddRole("b", 0);
        Check(Throws(() => cyclic.Validate()), "a cyclic role hierarchy is rejected");
        var dangling = new Policy853();
        dangling.AddRole("a", 7);
        Check(Throws(() => dangling.Validate()), "a role with an unknown parent is rejected");
        Check(policy.Roles.Count == 5 && policy.Rules.Count >= 15 && engine.TotalRules == policy.Rules.Count,
            "the policy has a real role ladder (" + policy.Roles.Count + " roles, " + policy.Rules.Count + " rules)");

        // 5-11. Point semantics: allow, default deny, ownership, masking, deny overrides.
        Check(engine.Evaluate(0, "select", 0, 0, 0).Allowed, "a viewer may read the first table");
        Check(!engine.Evaluate(0, "insert", 3, 5, 1).Allowed, "a viewer may not insert into the last table");
        Check(engine.Evaluate(1, "update", 0, RowOf(db, 0, 1), 2).Allowed, "an analyst may update a row it owns");
        Check(!engine.Evaluate(1, "update", 0, RowOf(db, 0, 3), 2).Allowed,
            "the same analyst may not update a row owned by somebody else");
        Decision853 masked = engine.Evaluate(2, "select", 3, RowOf(db, 3, 1), 5);
        Decision853 plain = engine.Evaluate(3, "select", 3, RowOf(db, 3, 1), 5);
        Console.WriteLine("sensitive column: auditor=" + (masked.Masked ? "masked" : "plain") + " admin="
            + (plain.Masked ? "masked" : "plain") + " raw-value=" + plain.Value);
        Check(masked.Allowed && masked.Masked && masked.Value == 0, "an auditor sees the sensitive column masked");
        Check(plain.Allowed && !plain.Masked && plain.Value != 0, "an admin sees the sensitive column unmasked");
        Check(!engine.Evaluate(4, "select", 0, RowOf(db, 0, 1), 0).Allowed,
            "an explicit deny overrides the allow inherited from the analyst role");
        Check(engine.Evaluate(0, "select", 0, RowOf(db, 0, 1), 0).Allowed,
            "the same request is allowed for a subject without the denying role");
        Check(!engine.Evaluate(3, "delete", 0, 0, 0).Allowed
            && engine.Evaluate(3, "delete", 0, 0, 0).Reason == "explicit-deny",
            "a global deny rule overrides even the administrator's own delete grant");
        Decision853 unknown = engine.Evaluate(Subjects + 3, "select", 0, 0, 0);
        Check(!unknown.Allowed && unknown.Reason == "unknown-subject",
            "an unknown subject falls back to default deny");
        var emptyEngine = new AccessEngine853(new Policy853(), db);
        Decision853 empty = emptyEngine.Evaluate(0, "select", 0, 0, 0);
        Check(!empty.Allowed && empty.Reason == "default-deny", "an empty policy denies everything by default");

        // 12. A masked response never carries the raw value, for any subject.
        int sensitiveRow = RowOf(db, 3, 2);
        int raw = db.RowAt(sensitiveRow).Values[5];
        bool leaks = false;
        for (int s = 0; s < Subjects; s++)
        {
            Decision853 d = engine.Evaluate(s, "select", 3, sensitiveRow, 5);
            if (d.Allowed && d.Masked && d.Value == raw) leaks = true;
        }
        Check(!leaks, "no masked response ever contains the raw value " + raw);

        // 13-16. Differential test against the brute-force oracle over 4000 requests.
        var audit = new AuditLog853();
        ulong seed = 0x853AUL;
        long agree = 0, allowed = 0, maskedCount = 0;
        bool decisionsAgree = true, reasonsAgree = true, valuesAgree = true;
        for (int i = 0; i < Requests; i++)
        {
            int subject = Rng853.NextInt(ref seed, Subjects + 2);   // two out-of-range subjects on purpose
            string action = Actions[Rng853.NextInt(ref seed, Actions.Length)];
            int table = Rng853.NextInt(ref seed, Tables + 1);
            int row = Rng853.NextInt(ref seed, db.RowCount + 1);
            int column = Rng853.NextInt(ref seed, Columns + 1);
            Decision853 mine = engine.Evaluate(subject, action, table, row, column);
            Decision853 reference = oracle.Evaluate(subject, action, table, row, column);
            if (mine.Allowed != reference.Allowed || mine.Masked != reference.Masked) decisionsAgree = false;
            if (mine.Reason != reference.Reason) reasonsAgree = false;
            if (mine.Allowed && !mine.Masked && mine.Value != reference.Value) valuesAgree = false;
            if (mine.Allowed == reference.Allowed && mine.Masked == reference.Masked
                && mine.Reason == reference.Reason) agree++;
            if (mine.Allowed) allowed++;
            if (mine.Masked) maskedCount++;
            audit.Append(subject, action, table, row, column, mine.Allowed);
        }
        Console.WriteLine(Requests + " requests: allow=" + allowed + " masked=" + maskedCount
            + " | rule evaluations engine=" + engine.RuleEvaluations + " oracle=" + oracle.RuleEvaluations);
        Check(agree == Requests && decisionsAgree && reasonsAgree && valuesAgree,
            "the indexed engine agrees with the brute-force oracle on all " + Requests + " requests");
        Check(engine.RuleEvaluations < oracle.RuleEvaluations / 2,
            "the (action, table) index evaluates fewer than half the rules of the oracle ("
            + engine.RuleEvaluations + " vs " + oracle.RuleEvaluations + ")");
        Check(allowed > 0 && allowed < Requests, "the differential workload contains both grants and refusals");

        // 17-20. The audit chain: one record per decision, linked hashes, consecutive numbering.
        Check(audit.Count == Requests && audit.Entries.Count == Requests,
            "the audit log holds exactly one record per decision");
        Check(audit.Verify(Requests, audit.Head) == -1, "the audit chain verifies end to end (head " + audit.Head + ")");
        bool numbered = true;
        for (int i = 0; i < audit.Entries.Count; i++) if (audit.Entries[i].Seq != i) numbered = false;
        Check(numbered, "the audit records are numbered consecutively from zero");
        Check(audit.Entries[0].PrevHash == 0 && audit.Entries[5].PrevHash == audit.Entries[4].Hash
            && audit.Entries[5].Hash != audit.Entries[4].Hash,
            "each audit record chains to the previous record's hash");

        // 21-24. Tamper evidence: a flipped decision, a swap, a forged hash and a truncation.
        AuditLog853 tampered = Clone(audit);
        tampered.Entries[17].Allowed = !tampered.Entries[17].Allowed;
        Check(tampered.Verify(Requests, audit.Head) == 17, "flipping one decision is detected at entry 17");
        AuditLog853 reordered = Clone(audit);
        (reordered.Entries[40], reordered.Entries[41]) = (reordered.Entries[41], reordered.Entries[40]);
        long swapIndex = reordered.Verify(Requests, audit.Head);
        Check(swapIndex == 40 || swapIndex == 41, "swapping two audit entries is detected at the swap");
        AuditLog853 forged = Clone(audit);
        forged.Entries[99].Hash = Audit853.EntryHash(forged.Entries[99].PrevHash, forged.Entries[99]) + 1;
        Check(forged.Verify(Requests, audit.Head) == 99, "a forged entry hash is detected at entry 99");
        AuditLog853 truncated = Clone(audit);
        truncated.RemoveLast(25);
        Check(truncated.Verify(Requests, audit.Head) == -2,
            "removing the last 25 records is caught by the signed (head, count) checkpoint");

        // 25. The log stores identifiers only: no data value appears anywhere in it.
        bool valueInLog = false;
        foreach (AuditEntry853 entry in audit.Entries)
        {
            if (entry.Row < 0 || entry.Row >= db.RowCount) continue;
            foreach (int value in db.RowAt(entry.Row).Values)
                if (value == entry.Hash || value == entry.PrevHash) valueInLog = true;
        }
        Check(!valueInLog, "no data value is recoverable from the audit records");

        // 26. Determinism: the same seed reproduces the decisions and the audit head.
        var again = new AuditLog853();
        var againEngine = new AccessEngine853(policy, db);
        ulong seed2 = 0x853AUL;
        for (int i = 0; i < Requests; i++)
        {
            int subject = Rng853.NextInt(ref seed2, Subjects + 2);
            string action = Actions[Rng853.NextInt(ref seed2, Actions.Length)];
            int table = Rng853.NextInt(ref seed2, Tables + 1);
            int row = Rng853.NextInt(ref seed2, db.RowCount + 1);
            int column = Rng853.NextInt(ref seed2, Columns + 1);
            again.Append(subject, action, table, row, column,
                againEngine.Evaluate(subject, action, table, row, column).Allowed);
        }
        Check(again.Head == audit.Head && again.Count == audit.Count,
            "the same seed reproduces the decisions and the audit head exactly");

        Console.WriteLine("PASS");
    }

    private static Policy853 BuildPolicy()
    {
        var policy = new Policy853();
        policy.AddRole("viewer");              // 0
        policy.AddRole("analyst", 0);          // 1 inherits viewer
        policy.AddRole("auditor", 0);          // 2 inherits viewer
        policy.AddRole("admin", 1);            // 3 inherits analyst
        policy.AddRole("restricted", 1);       // 4 inherits analyst
        policy.AddRule(0, "select", 0, 0, -1, true);
        policy.AddRule(0, "select", 1, 0, -1, true);
        policy.AddRule(1, "select", -1, 0, -1, true);
        policy.AddRule(1, "update", 0, 1, -1, true);
        policy.AddRule(1, "insert", 0, 0, -1, true);
        policy.AddRule(1, "update", 2, 1, -1, true);
        policy.AddRule(2, "select", 3, 0, -1, true);
        policy.AddRule(2, "select", 3, 0, 5, true);        // the auditor's read of column 5 is masked
        policy.AddRule(2, "select", 0, 0, 4, true);        // column 4 of table 0 is masked for auditors
        policy.AddRule(3, "select", -1, 0, -1, true);
        policy.AddRule(3, "update", -1, 0, -1, true);
        policy.AddRule(3, "insert", -1, 0, -1, true);
        policy.AddRule(3, "delete", -1, 0, -1, true);
        policy.AddRule(4, "select", 0, 0, -1, false);      // explicit deny: restricted may not read table 0
        policy.AddRule(4, "select", 2, 1, -1, true);       // restricted may read its own rows of table 2
        policy.AddRule(-1, "delete", -1, 0, -1, false);    // nobody deletes anything
        policy.AddRule(-1, "insert", 3, 0, -1, false);
        return policy;
    }

    private static int RowOf(Database853 db, int table, int offset)
    {
        int seen = 0;
        for (int i = 0; i < db.RowCount; i++)
        {
            if (db.RowAt(i).Table != table) continue;
            if (seen == offset) return i;
            seen++;
        }
        throw new ArgumentOutOfRangeException(nameof(offset));
    }

    private static bool BruteForceReachable(Policy853 policy, int subject, int role)
    {
        int direct = subject % policy.Roles.Count;
        if (direct == role) return true;
        var seen = new HashSet<int> { direct };
        var stack = new Stack<int>();
        stack.Push(direct);
        while (stack.Count > 0)
        {
            int current = stack.Pop();
            foreach (int parent in policy.Roles[current].Parents)
            {
                if (parent == role) return true;
                if (seen.Add(parent)) stack.Push(parent);
            }
        }
        return false;
    }

    private static AuditLog853 Clone(AuditLog853 source)
    {
        var copy = new AuditLog853();
        foreach (AuditEntry853 entry in source.Entries)
        {
            copy.Entries.Add(new AuditEntry853
            {
                Seq = entry.Seq, Subject = entry.Subject, Action = entry.Action, Table = entry.Table,
                Row = entry.Row, Column = entry.Column, Allowed = entry.Allowed,
                PrevHash = entry.PrevHash, Hash = entry.Hash,
            });
        }
        copy.Restore(source.Count, source.Head);
        return copy;
    }

    private static bool Throws(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (ArgumentException)
        {
            return true;
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
