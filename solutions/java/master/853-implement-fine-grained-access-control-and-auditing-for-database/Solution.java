/** Task 853 - Implement fine-grained access control and auditing for database operations.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A security engine for a small in-memory database. Subjects are users or roles, roles form a DAG of
 *   parents, and a GRANT names (subject, table, action set, optional column subset, optional row predicate,
 *   grant option, deny flag). A request is (user, action, table, column, row) and is resolved with
 *   deny-overrides semantics: any matching explicit DENY wins, otherwise a matching ALLOW whose row predicate
 *   accepts the row wins, otherwise the request is denied with a diagnostic reason code. Privileges are
 *   inherited through the role DAG (transitive closure) and revocation propagates immediately.
 *   Fine-grained access is granted per column and per row: a SELECT projects the columns the subject may read
 *   and substitutes a fixed mask for the others, and only rows accepted by the predicate are returned - the
 *   raw values of forbidden columns never reach the result set.
 *   Every decision (allow and deny) appends an entry to an append-only audit log whose entries are chained
 *   with a hand-written FNV-1a/64-bit mixer, so tampering or reordering is detectable.
 *
 * WHAT IS SIMPLIFIED
 *   No SQL parser, no storage engine, no durability: rows are in-memory string tuples and predicates are a
 *   tiny comparison tree (column op literal, AND/OR). Time is the audit sequence number, there is no
 *   concurrency (one decision at a time), no threads and no randomness - the whole file is deterministic.
 *
 * INVARIANTS VERIFIED
 *   Safety: the effective privilege set is exactly the union of the grants of the subject's roles and their
 *   ancestors (checked against an independent DFS reachability oracle over a full decision matrix), deny
 *   always overrides allow, role-graph cycles are rejected, privilege escalation (delegating more than the
 *   grantor holds, or with the grant option it does not have) is refused, forbidden column values are never
 *   emitted by a projection and the audit log cannot be silently edited.
 *   Liveness / accounting: every decision is logged exactly once, every reason code is reachable, revocation
 *   propagates (the denied matrix entry appears and the differential oracle agrees while the internal closure
 *   is recomputed), repeated identical requests are answered identically, and the audit hash chain verifies
 *   from genesis to head after every mutation.
 */
import java.util.*;

/** FNV-1a 64 plus a final avalanche mix - the deterministic, dependency-free hash used for the audit chain. */
final class Digests {
    static long fnv1a(String s) {
        long h = 0xCBF29CE484222325L;
        for (int i = 0; i < s.length(); i++) {
            h ^= s.charAt(i);
            h *= 0x100000001B3L;
        }
        return avalanche(h);
    }

    static long avalanche(long x) {
        x ^= x >>> 33;
        x *= 0xFF51AFD7ED558CCDL;
        x ^= x >>> 33;
        x *= 0xC4CEB9FE1A85EC53L;
        x ^= x >>> 33;
        return x;
    }

    static long combine(long a, long b) {
        return avalanche(a ^ (b + 0x9E3779B97F4A7C15L + (a << 6) + (a >>> 2)));
    }
}

enum Action { SELECT, INSERT, UPDATE, DELETE }

/** A row of a table plus its stable identifier, used by row-level security. */
final class Row {
    final int id;
    final String[] values;

    Row(int id, String[] values) {
        this.id = id;
        this.values = values;
    }
}

final class Table {
    final String name;
    final List<String> columns;
    final List<Row> rows = new ArrayList<>();

    Table(String name, List<String> columns) {
        this.name = name;
        this.columns = columns;
    }

    int columnIndex(String column) {
        for (int i = 0; i < columns.size(); i++) if (columns.get(i).equals(column)) return i;
        throw new IllegalArgumentException("unknown column " + column);
    }

    Row addRow(String... values) {
        if (values.length != columns.size()) throw new IllegalArgumentException("arity mismatch");
        Row row = new Row(rows.size(), values.clone());
        rows.add(row);
        return row;
    }
}

/** Tiny predicate tree: comparisons on a column, combined with AND/OR. `null` means "all rows". */
final class RowPred {
    final String column;
    final String op;
    final String operand;
    final String combinator;
    final RowPred left;
    final RowPred right;

    private RowPred(String column, String op, String operand, String combinator, RowPred left, RowPred right) {
        this.column = column;
        this.op = op;
        this.operand = operand;
        this.combinator = combinator;
        this.left = left;
        this.right = right;
    }

    static RowPred cmp(String column, String op, String operand) {
        return new RowPred(column, op, operand, null, null, null);
    }

    static RowPred and(RowPred a, RowPred b) { return new RowPred(null, null, null, "AND", a, b); }

    static RowPred or(RowPred a, RowPred b) { return new RowPred(null, null, null, "OR", a, b); }

    boolean eval(Table table, Row row) {
        if (combinator != null) {
            return combinator.equals("AND") ? left.eval(table, row) && right.eval(table, row)
                    : left.eval(table, row) || right.eval(table, row);
        }
        String value = row.values[table.columnIndex(column)];
        if (op.equals("=")) return value.equals(operand);
        if (op.equals("!=")) return !value.equals(operand);
        double a = Double.parseDouble(value);
        double b = Double.parseDouble(operand);
        if (op.equals("<")) return a < b;
        if (op.equals(">")) return a > b;
        throw new IllegalArgumentException("unknown operator " + op);
    }

    String describe() {
        if (combinator != null) return "(" + left.describe() + " " + combinator + " " + right.describe() + ")";
        return column + op + operand;
    }
}

/** A grant row of the privilege table. `columns == null` means every column; `filter == null` means all rows. */
final class Grant {
    final String subject;
    final String table;
    final EnumSet<Action> actions;
    final Set<String> columns;
    final RowPred filter;
    final boolean grantOption;
    final boolean deny;

    Grant(String subject, String table, EnumSet<Action> actions, Set<String> columns, RowPred filter,
            boolean grantOption, boolean deny) {
        this.subject = subject;
        this.table = table;
        this.actions = actions;
        this.columns = columns;
        this.filter = filter;
        this.grantOption = grantOption;
        this.deny = deny;
    }

    boolean matchesAction(Action action, String table, String column) {
        if (!this.table.equals(table) || !actions.contains(action)) return false;
        return columns == null || column == null || columns.contains(column);
    }

    @Override public String toString() {
        return (deny ? "DENY " : "GRANT ") + subject + " " + actions + " on " + table
                + (columns == null ? "" : columns.toString())
                + (filter == null ? "" : " where " + filter.describe());
    }
}

/** One immutable audit record; `hash` chains to the previous record. */
final class AuditEntry {
    final long seq;
    final String user;
    final String action;
    final String table;
    final String column;
    final int rowId;
    final String decision;
    final String reason;
    final long previousHash;
    final long hash;

    AuditEntry(long seq, String user, String action, String table, String column, int rowId, String decision,
            String reason, long previousHash) {
        this.seq = seq;
        this.user = user;
        this.action = action;
        this.table = table;
        this.column = column;
        this.rowId = rowId;
        this.decision = decision;
        this.reason = reason;
        this.previousHash = previousHash;
        this.hash = computeHash();
    }

    long computeHash() {
        long h = Digests.fnv1a(user + "|" + action + "|" + table + "|" + column + "|" + rowId + "|"
                + decision + "|" + reason);
        h = Digests.combine(h, seq);
        return Digests.combine(h, previousHash);
    }

    @Override public String toString() {
        return seq + " " + user + " " + action + " " + table + " " + column + " row" + rowId
                + " -> " + decision + "/" + reason;
    }
}

/** Outcome of one access decision, with the reason code used by the audit log. */
final class Verdict {
    static final String ALLOW = "ALLOW";
    static final String DENY_RULE = "DENY_RULE";
    static final String NO_GRANT = "NO_GRANT";
    static final String ROW_FILTERED = "ROW_FILTERED";
    static final String COLUMN_RESTRICTED = "COLUMN_RESTRICTED";

    final boolean allowed;
    final String reason;

    Verdict(boolean allowed, String reason) {
        this.allowed = allowed;
        this.reason = reason;
    }
}

/** The engine: privilege tables, role DAG, decision procedure, projections and the audit log. */
final class SecurityEngine {
    static final String MASK = "***";

    final Map<String, Table> tables = new LinkedHashMap<>();
    final Map<String, Set<String>> roleParents = new LinkedHashMap<>();
    final Map<String, Set<String>> userRoles = new LinkedHashMap<>();
    final Set<String> users = new LinkedHashSet<>();
    final List<Grant> grants = new ArrayList<>();
    final List<AuditEntry> audit = new ArrayList<>();
    private long head = Digests.fnv1a("genesis");

    Table createTable(String name, String... columns) {
        Table t = new Table(name, Arrays.asList(columns));
        tables.put(name, t);
        return t;
    }

    void createUser(String user) {
        users.add(user);
        userRoles.putIfAbsent(user, new LinkedHashSet<>());
    }

    void createRole(String role) {
        roleParents.putIfAbsent(role, new LinkedHashSet<>());
    }

    /** Adds `parent` to `role`'s parent set, rejecting anything that would make `role` reach itself again. */
    void addRoleParent(String role, String parent) {
        createRole(role);
        createRole(parent);
        roleParents.get(role).add(parent);
        if (parent.equals(role) || roleAncestors(parent).contains(role)) {
            roleParents.get(role).remove(parent);
            throw new SecurityException("cycle would be created by " + role + " -> " + parent);
        }
    }

    void assignRole(String user, String role) {
        createUser(user);
        createRole(role);
        userRoles.get(user).add(role);
    }

    void revokeRole(String user, String role) {
        Set<String> roles = userRoles.get(user);
        if (roles != null) roles.remove(role);
    }

    /** All roles reachable from `role` including itself (iterative worklist, cycle safe). */
    Set<String> roleAncestors(String role) {
        Set<String> seen = new LinkedHashSet<>();
        Deque<String> stack = new ArrayDeque<>();
        stack.push(role);
        while (!stack.isEmpty()) {
            String current = stack.pop();
            if (!seen.add(current)) continue;
            for (String parent : roleParents.getOrDefault(current, Set.of())) stack.push(parent);
        }
        return seen;
    }

    /** Effective roles of a subject: the subject itself, then every role it holds and their ancestors. */
    Set<String> effectiveRoles(String subject) {
        Set<String> result = new LinkedHashSet<>();
        result.add(subject);
        for (String role : userRoles.getOrDefault(subject, Set.of())) result.addAll(roleAncestors(role));
        return result;
    }

    Grant grant(String subject, String table, EnumSet<Action> actions, Set<String> columns, RowPred filter,
            boolean grantOption, boolean deny) {
        Grant g = new Grant(subject, table, actions, columns == null ? null : new LinkedHashSet<>(columns),
                filter, grantOption, deny);
        grants.add(g);
        return g;
    }

    void revoke(Grant grant) { grants.remove(grant); }

    /** Removes every matching privilege row; returns how many were revoked. */
    int revokeMatching(String subject, String table, Action action) {
        int removed = 0;
        Iterator<Grant> it = grants.iterator();
        while (it.hasNext()) {
            Grant g = it.next();
            if (g.subject.equals(subject) && g.table.equals(table) && g.actions.contains(action)) {
                it.remove();
                removed++;
            }
        }
        return removed;
    }

    /** Delegation: `from` may only hand out privileges it holds with the grant option. */
    void delegate(String from, String to, String table, EnumSet<Action> actions, Set<String> columns)
            throws SecurityException {
        for (Action action : actions) {
            Set<String> scope = columns == null ? new LinkedHashSet<>(tables.get(table).columns) : columns;
            for (String column : scope) {
                if (!heldWithGrantOption(from, action, table, column)) {
                    throw new SecurityException("privilege escalation: " + from + " cannot grant " + action
                            + " on " + table + "." + column);
                }
            }
        }
        grant(to, table, actions, columns, null, false, false);
    }

    boolean heldWithGrantOption(String subject, Action action, String table, String column) {
        Set<String> roles = effectiveRoles(subject);
        for (Grant g : grants) {
            if (g.deny || !g.grantOption) continue;
            if (!roles.contains(g.subject)) continue;
            if (g.matchesAction(action, table, column)) return true;
        }
        return false;
    }

    // ------------------------------------------------------------------ decision procedure

    Verdict decide(String user, Action action, Table table, String column, Row row) {
        // A request with column == null asks "may this subject touch any part of the row", so every grant that
        // covers the action on the table is considered; a request with a column asks about that column only.
        Set<String> roles = effectiveRoles(user);
        boolean sawAllowShape = false;
        boolean denied = false;
        boolean allowed = false;
        for (Grant g : grants) {
            if (!roles.contains(g.subject)) continue;
            if (!g.matchesAction(action, table.name, column)) continue;
            boolean rowOk = g.filter == null || row == null || g.filter.eval(table, row);
            if (g.deny) {
                if (rowOk) denied = true;
            } else {
                sawAllowShape = true;
                if (rowOk) allowed = true;
            }
        }
        Verdict verdict;
        if (denied) verdict = new Verdict(false, Verdict.DENY_RULE);
        else if (allowed) verdict = new Verdict(true, Verdict.ALLOW);
        else if (sawAllowShape) verdict = new Verdict(false, Verdict.ROW_FILTERED);
        else verdict = new Verdict(false, Verdict.NO_GRANT);
        record(user, action.name(), table.name, column, row == null ? -1 : row.id, verdict);
        return verdict;
    }

    void record(String user, String action, String table, String column, int rowId, Verdict verdict) {
        AuditEntry entry = new AuditEntry(audit.size(), user, action, table, column, rowId,
                verdict.allowed ? Verdict.ALLOW : "DENY", verdict.reason, head);
        audit.add(entry);
        head = entry.hash;
    }

    /**
     * Row-and-column level SELECT: rows filtered by the row predicates, columns the subject may not read
     * replaced by MASK. One audit decision per row plus one per projected column.
     */
    List<String[]> select(String user, String tableName, List<String> requestedColumns) {
        Table table = tables.get(tableName);
        List<String[]> out = new ArrayList<>();
        for (Row row : table.rows) {
            if (!decide(user, Action.SELECT, table, null, row).allowed) continue;
            String[] projected = new String[requestedColumns.size()];
            for (int i = 0; i < requestedColumns.size(); i++) {
                String column = requestedColumns.get(i);
                int index = table.columnIndex(column);
                projected[i] = decide(user, Action.SELECT, table, column, row).allowed
                        ? row.values[index] : MASK;
            }
            out.add(projected);
        }
        return out;
    }

    // ------------------------------------------------------------------ audit verification

    /** Recomputes the whole hash chain; returns the number of broken links. */
    int verifyAuditChain() {
        long previous = Digests.fnv1a("genesis");
        int broken = 0;
        for (AuditEntry e : audit) {
            if (e.previousHash != previous || e.hash != e.computeHash()) broken++;
            previous = e.hash;
        }
        return broken;
    }

    long auditHead() { return head; }

    Map<String, Integer> reasonHistogram() {
        Map<String, Integer> histogram = new TreeMap<>();
        for (AuditEntry e : audit) histogram.merge(e.reason, 1, Integer::sum);
        return histogram;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    // ------------------------------------------------------------------ independent oracle

    /** Brute-force reachability of one role from another, without using the engine's closure helper. */
    static boolean reachable(SecurityEngine engine, String from, String target) {
        if (from.equals(target)) return true;
        Deque<String> queue = new ArrayDeque<>();
        Set<String> seen = new HashSet<>();
        queue.add(from);
        while (!queue.isEmpty()) {
            String current = queue.poll();
            if (!seen.add(current)) continue;
            for (String parent : engine.roleParents.getOrDefault(current, Set.of())) {
                if (parent.equals(target)) return true;
                queue.add(parent);
            }
        }
        return false;
    }

    /** Does the user reach `target` through any of its assigned roles (own DFS, no cached closure)? */
    static boolean userReaches(SecurityEngine engine, String user, String target) {
        if (user.equals(target)) return true;
        for (String role : engine.userRoles.getOrDefault(user, Set.of())) {
            if (reachable(engine, role, target)) return true;
        }
        return false;
    }

    /**
     * Independent reference resolver: recomputes reachability from the user for every grant instead of using
     * the engine's cached effective-role set, and scans the privilege table linearly.
     */
    static Verdict oracleDecide(SecurityEngine engine, String user, Action action, Table table, String column,
            Row row) {
        boolean denied = false, allowed = false, sawAllowShape = false;
        for (Grant g : engine.grants) {
            if (!userReaches(engine, user, g.subject)) continue;
            if (g.table.equals(table.name) && g.actions.contains(action)
                    && (g.columns == null || column == null || g.columns.contains(column))) {
                boolean rowOk = g.filter == null || row == null || g.filter.eval(table, row);
                if (g.deny) {
                    if (rowOk) denied = true;
                } else {
                    sawAllowShape = true;
                    if (rowOk) allowed = true;
                }
            }
        }
        if (denied) return new Verdict(false, Verdict.DENY_RULE);
        if (allowed) return new Verdict(true, Verdict.ALLOW);
        if (sawAllowShape) return new Verdict(false, Verdict.ROW_FILTERED);
        return new Verdict(false, Verdict.NO_GRANT);
    }

    static String join(String[] values) { return String.join(",", values); }

    static SecurityEngine buildFixture() {
        SecurityEngine engine = new SecurityEngine();
        Table accounts = engine.createTable("accounts", "id", "owner", "region", "balance");
        accounts.addRow("1", "alice", "eu", "1200");
        accounts.addRow("2", "bob", "us", "500");
        accounts.addRow("3", "carol", "eu", "9500");
        accounts.addRow("4", "dave", "apac", "300");

        engine.createRole("reader");
        engine.createRole("analyst");
        engine.createRole("auditor");
        engine.createRole("admin");
        engine.addRoleParent("analyst", "reader");       // analyst inherits reader
        engine.addRoleParent("auditor", "reader");
        engine.addRoleParent("admin", "analyst");        // admin inherits analyst (and reader)

        engine.createUser("alice");
        engine.createUser("mallory");
        engine.createUser("root");
        engine.createUser("erin");
        engine.assignRole("alice", "analyst");
        engine.assignRole("mallory", "reader");
        engine.assignRole("root", "admin");

        engine.grant("reader", "accounts", EnumSet.of(Action.SELECT), Set.of("id", "owner"), null, false, false);
        engine.grant("analyst", "accounts", EnumSet.of(Action.SELECT), Set.of("region"), null, true, false);
        engine.grant("analyst", "accounts", EnumSet.of(Action.SELECT), Set.of("balance"),
                RowPred.cmp("region", "=", "eu"), false, false);
        engine.grant("auditor", "accounts", EnumSet.of(Action.SELECT), null, null, false, false);
        engine.grant("admin", "accounts", EnumSet.of(Action.SELECT), null, null, false, false);
        engine.grant("admin", "accounts", EnumSet.of(Action.UPDATE), null, null, false, false);
        engine.grant("erin", "accounts", EnumSet.of(Action.SELECT), Set.of("id", "owner"),
                RowPred.cmp("region", "=", "eu"), false, false);
        return engine;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        SecurityEngine engine = buildFixture();
        Table accounts = engine.tables.get("accounts");

        // --- 1. role hierarchy: transitive closure matches an independent DFS, and cycles are rejected.
        check(engine.roleAncestors("admin").equals(new LinkedHashSet<>(List.of("admin", "analyst", "reader"))),
                "admin inherits analyst and reader");
        check(engine.roleAncestors("reader").equals(new LinkedHashSet<>(List.of("reader"))),
                "reader has no parents");
        for (String role : engine.roleParents.keySet()) {
            for (String other : engine.roleParents.keySet()) {
                boolean viaClosure = engine.roleAncestors(role).contains(other);
                check(viaClosure == reachable(engine, role, other),
                        "closure matches DFS for " + role + " -> " + other);
            }
        }
        check(engine.effectiveRoles("root").equals(
                        new LinkedHashSet<>(List.of("root", "admin", "analyst", "reader"))),
                "root's effective roles come from the admin role");
        boolean cycleRejected = false;
        try {
            engine.addRoleParent("reader", "admin");
        } catch (SecurityException e) {
            cycleRejected = true;
        }
        check(cycleRejected, "a role cycle is rejected");
        check(engine.roleAncestors("reader").equals(new LinkedHashSet<>(List.of("reader"))),
                "the hierarchy is unchanged after the rejected cycle");

        // --- 2. differential test of the resolver against the independent oracle over the whole matrix.
        Set<String> probeUsers = new LinkedHashSet<>(List.of("alice", "mallory", "root", "erin", "nobody"));
        Action[] actions = Action.values();
        List<String> columns = new ArrayList<>(accounts.columns);
        int mismatches = 0, allows = 0, denies = 0, compared = 0;
        for (String user : probeUsers) {
            for (Action action : actions) {
                for (String column : columns) {
                    for (Row row : accounts.rows) {
                        // the oracle is called first so it does not see the engine's audit side effects
                        Verdict expected = oracleDecide(engine, user, action, accounts, column, row);
                        Verdict actual = engine.decide(user, action, accounts, column, row);
                        compared++;
                        if (expected.allowed != actual.allowed || !expected.reason.equals(actual.reason)) {
                            mismatches++;
                        }
                        if (actual.allowed) allows++;
                        else denies++;
                    }
                }
            }
        }
        System.out.printf("decision matrix: compared=%d allow=%d deny=%d mismatches=%d%n",
                compared, allows, denies, mismatches);
        check(compared == probeUsers.size() * actions.length * columns.size() * accounts.rows.size(),
                "the matrix covers every user/action/column/row combination");
        check(mismatches == 0, "the engine agrees with the independent oracle on every decision");
        check(allows > 0 && denies > 0, "the matrix contains both allows and denials");

        // --- 3. deny-overrides and column restriction behave exactly as documented.
        engine.grant("mallory", "accounts", EnumSet.of(Action.SELECT), Set.of("id"), null, false, true);
        Verdict deniedByRule = engine.decide("mallory", Action.SELECT, accounts, "id", accounts.rows.get(0));
        check(!deniedByRule.allowed && deniedByRule.reason.equals(Verdict.DENY_RULE),
                "an explicit deny for one column overrides the inherited allow");
        Verdict otherColumn = engine.decide("mallory", Action.SELECT, accounts, "owner", accounts.rows.get(0));
        check(otherColumn.allowed, "the deny is scoped to its column");
        engine.revokeMatching("mallory", "accounts", Action.SELECT);
        check(engine.decide("mallory", Action.SELECT, accounts, "id", accounts.rows.get(0)).allowed,
                "revoking the deny restores the inherited allow");

        // --- 4. row-level security: the predicate hides exactly the rows the oracle predicts.
        Verdict erinEu = engine.decide("erin", Action.SELECT, accounts, null, accounts.rows.get(0));
        Verdict erinUs = engine.decide("erin", Action.SELECT, accounts, null, accounts.rows.get(1));
        check(erinEu.allowed, "erin may read the eu row");
        check(!erinUs.allowed && erinUs.reason.equals(Verdict.ROW_FILTERED),
                "the us row is hidden from erin by the row predicate");
        Verdict euOnly = engine.decide("alice", Action.SELECT, accounts, "balance", accounts.rows.get(0));
        Verdict usRow = engine.decide("alice", Action.SELECT, accounts, "balance", accounts.rows.get(1));
        check(euOnly.allowed, "the eu row passes alice's balance predicate");
        check(!usRow.allowed && usRow.reason.equals(Verdict.ROW_FILTERED),
                "the us row is filtered out for alice");
        check(engine.decide("alice", Action.SELECT, accounts, "region", accounts.rows.get(1)).allowed,
                "the region column is not row filtered for alice");

        // --- 5. projection: masked columns never leak a raw value, permitted ones are returned verbatim.
        List<String> request = List.of("id", "owner", "region", "balance");
        List<String> aliceRows = new ArrayList<>();
        for (String[] row : engine.select("alice", "accounts", request)) aliceRows.add(join(row));
        System.out.println("alice projection: " + aliceRows);
        check(aliceRows.size() == 4, "the reader role is not row filtered, so alice sees every row");
        check(aliceRows.get(0).equals("1,alice,eu,1200"), "alice reads the eu balance of row 1");
        check(aliceRows.get(1).equals("2,bob,us," + SecurityEngine.MASK),
                "the us balance is masked for alice: " + aliceRows.get(1));
        check(aliceRows.get(2).equals("3,carol,eu,9500"), "alice reads carol's eu balance");
        check(aliceRows.get(3).equals("4,dave,apac," + SecurityEngine.MASK), "the apac balance is masked");
        check(!String.join(";", aliceRows).contains("2,bob,us,500")
                        && !String.join(";", aliceRows).contains("4,dave,apac,300"),
                "no filtered balance reaches alice's result set");
        List<String> erinRows = new ArrayList<>();
        for (String[] row : engine.select("erin", "accounts", request)) erinRows.add(join(row));
        System.out.println("erin projection: " + erinRows);
        check(erinRows.size() == 2 && erinRows.get(0).equals("1,alice," + SecurityEngine.MASK + ","
                        + SecurityEngine.MASK), "erin sees only the eu rows with the extra columns masked");
        List<String> malloryRows = new ArrayList<>();
        for (String[] row : engine.select("mallory", "accounts", request)) malloryRows.add(join(row));
        System.out.println("mallory projection: " + malloryRows);
        check(malloryRows.size() == 4, "mallory sees every row but only two columns");
        for (String row : malloryRows) {
            check(row.endsWith("," + SecurityEngine.MASK + "," + SecurityEngine.MASK),
                    "mallory's region and balance are masked: " + row);
        }
        check(!String.join(";", malloryRows).contains("1200")
                        && !String.join(";", malloryRows).contains("9500"),
                "no raw balance ever reaches mallory's result set");
        check(String.join(";", malloryRows).contains("dave"), "mallory can still read owner");
        Verdict unknownUser = engine.decide("nobody", Action.SELECT, accounts, "id", accounts.rows.get(0));
        check(!unknownUser.allowed && unknownUser.reason.equals(Verdict.NO_GRANT),
                "an unknown user is denied with NO_GRANT");

        // --- 6. privilege escalation through delegation is refused, legitimate delegation works.
        check(engine.heldWithGrantOption("alice", Action.SELECT, "accounts", "region"),
                "alice holds the region grant with the grant option");
        boolean escalated = false;
        try {
            engine.delegate("mallory", "dave", "accounts", EnumSet.of(Action.SELECT), Set.of("balance"));
        } catch (SecurityException e) {
            escalated = true;
        }
        check(escalated, "delegating a privilege the grantor does not hold is refused");
        boolean escalatedColumn = false;
        try {
            engine.delegate("alice", "dave", "accounts", EnumSet.of(Action.SELECT), null);
        } catch (SecurityException e) {
            escalatedColumn = true;
        }
        check(escalatedColumn, "delegating all columns is refused when only one is held");
        engine.delegate("alice", "dave", "accounts", EnumSet.of(Action.SELECT), Set.of("region"));
        check(engine.decide("dave", Action.SELECT, accounts, "region", accounts.rows.get(0)).allowed,
                "the legitimate delegation grants dave region access");
        check(!engine.decide("dave", Action.SELECT, accounts, "balance", accounts.rows.get(0)).allowed,
                "the delegation does not leak the balance column");
        Verdict first = engine.decide("dave", Action.SELECT, accounts, "region", accounts.rows.get(1));
        Verdict second = engine.decide("dave", Action.SELECT, accounts, "region", accounts.rows.get(1));
        check(first.allowed == second.allowed && first.reason.equals(second.reason),
                "repeated identical requests get the identical verdict");

        // --- 7. every decision is audited, exactly once, with a live hash chain.
        int before = engine.audit.size();
        Verdict update = engine.decide("root", Action.UPDATE, accounts, "balance", accounts.rows.get(0));
        check(update.allowed && update.reason.equals(Verdict.ALLOW), "root may update the balance column");
        engine.decide("root", Action.INSERT, accounts, "balance", accounts.rows.get(0));
        check(engine.audit.size() == before + 2, "one audit entry per decision");
        check(engine.audit.get(engine.audit.size() - 1).reason.equals(Verdict.NO_GRANT),
                "INSERT without a grant is audited as NO_GRANT");
        check(engine.verifyAuditChain() == 0, "the audit chain verifies, broken links=" + engine.verifyAuditChain());
        long maxSeq = -1;
        boolean seqOk = true;
        for (AuditEntry e : engine.audit) {
            seqOk &= e.seq == maxSeq + 1;
            maxSeq = e.seq;
        }
        check(seqOk && maxSeq == engine.audit.size() - 1, "audit sequence numbers are dense and ordered");
        Map<String, Integer> histogram = engine.reasonHistogram();
        System.out.println("audit reason histogram: " + histogram);
        for (String code : List.of(Verdict.ALLOW, Verdict.DENY_RULE, Verdict.NO_GRANT, Verdict.ROW_FILTERED)) {
            check(histogram.getOrDefault(code, 0) > 0, "reason code " + code + " occurs in the audit log");
        }

        // --- 8. tamper detection: editing, reordering or truncating the log breaks the chain.
        AuditEntry third = engine.audit.get(3);
        AuditEntry forged = new AuditEntry(third.seq, third.user, third.action, third.table, third.column,
                third.rowId, third.decision.equals("ALLOW") ? "DENY" : "ALLOW", third.reason, third.previousHash);
        check(forged.hash != third.hash, "flipping a decision changes the entry hash");
        AuditEntry forgedReason = new AuditEntry(third.seq, third.user, third.action, third.table, third.column,
                third.rowId, third.decision, "FORGED", third.previousHash);
        check(forgedReason.hash != third.hash, "flipping the reason changes the entry hash");
        AuditEntry forgedChain = new AuditEntry(third.seq, third.user, third.action, third.table, third.column,
                third.rowId, third.decision, third.reason, third.previousHash ^ 1L);
        check(forgedChain.hash != third.hash, "breaking the back-link changes the entry hash");
        List<AuditEntry> swapProbe = new ArrayList<>(engine.audit);
        Collections.swap(swapProbe, 3, 4);
        int swappedBroken = 0;
        long previous = Digests.fnv1a("genesis");
        for (AuditEntry e : swapProbe) {
            if (e.previousHash != previous || e.hash != e.computeHash()) swappedBroken++;
            previous = e.hash;
        }
        check(swappedBroken > 0, "reordering audit entries is detected");
        long headBefore = engine.auditHead();
        engine.record("tamperer", Action.DELETE.name(), "accounts", null, -1, new Verdict(false, Verdict.NO_GRANT));
        check(engine.auditHead() != headBefore, "the head hash advances with every appended entry");
        check(engine.verifyAuditChain() == 0, "the chain still verifies after appending");

        // --- 9. revocation propagates through the role hierarchy and the oracle still agrees.
        check(engine.decide("root", Action.UPDATE, accounts, "balance", accounts.rows.get(1)).allowed,
                "root may update before revocation");
        int removed = engine.revokeMatching("admin", "accounts", Action.UPDATE);
        check(removed == 1, "exactly one UPDATE grant was revoked");
        check(!engine.decide("root", Action.UPDATE, accounts, "balance", accounts.rows.get(1)).allowed,
                "after revocation root loses UPDATE");
        check(engine.decide("root", Action.SELECT, accounts, "region", accounts.rows.get(1)).allowed,
                "root keeps SELECT through the inherited analyst grant");
        int drift = 0;
        for (Action action : actions) {
            for (String column : columns) {
                Verdict expected = oracleDecide(engine, "root", action, accounts, column, accounts.rows.get(2));
                Verdict actual = engine.decide("root", action, accounts, column, accounts.rows.get(2));
                if (expected.allowed != actual.allowed || !expected.reason.equals(actual.reason)) drift++;
            }
        }
        check(drift == 0, "the oracle agrees after revocation");

        // --- 10. role revocation, empty grants and degenerate predicates.
        engine.assignRole("dave", "auditor");
        check(engine.decide("dave", Action.SELECT, accounts, "balance", accounts.rows.get(1)).allowed,
                "dave gains the auditor privileges");
        engine.revokeRole("dave", "auditor");
        check(!engine.decide("dave", Action.SELECT, accounts, "balance", accounts.rows.get(1)).allowed,
                "revoking the role removes the derived privilege");
        Table empty = engine.createTable("empty", "id");
        check(engine.decide("root", Action.SELECT, empty, "id", null).reason.equals(Verdict.NO_GRANT),
                "a table without grants yields NO_GRANT");
        check(engine.select("root", "empty", List.of("id")).isEmpty(), "an empty table projects nothing");
        RowPred tautology = RowPred.or(RowPred.cmp("region", "=", "eu"), RowPred.cmp("region", "!=", "eu"));
        engine.grant("mallory", "accounts", EnumSet.of(Action.SELECT), Set.of("id"), tautology, false, false);
        check(engine.decide("mallory", Action.SELECT, accounts, "id", accounts.rows.get(3)).allowed,
                "an OR predicate accepts every row");
        RowPred contradiction = RowPred.and(RowPred.cmp("region", "=", "eu"), RowPred.cmp("region", "!=", "eu"));
        check(!contradiction.eval(accounts, accounts.rows.get(0)), "a contradictory predicate accepts no row");
        check(RowPred.cmp("balance", ">", "1000").eval(accounts, accounts.rows.get(2)),
                "numeric comparison predicate works");

        // --- 11. the audit remains valid after the whole session and covers every decision taken.
        check(engine.verifyAuditChain() == 0, "the final audit chain verifies");
        check(engine.audit.size() > 200, "the session produced a substantial audit trail: " + engine.audit.size());
        int denialEntries = 0;
        for (AuditEntry e : engine.audit) if (!e.decision.equals("ALLOW")) denialEntries++;
        check(denialEntries > 0, "denials are audited too (" + denialEntries + ")");
        System.out.println("audit entries=" + engine.audit.size() + " head=" + Long.toHexString(engine.auditHead()));
        System.out.println("checks=" + checks);
        System.out.println("PASS");
    }
}
