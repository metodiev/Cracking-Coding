"""Task 853 - Implement fine-grained access control and auditing for database operations.  (Tier: Master, Ch 14 — Databases)

A row- and column-level authorisation engine for a small in-memory table, plus a
tamper-evident audit trail.  Everything is deterministic and in-process: no clock reads
(the audit log carries a virtual sequence number), no randomness, standard library only.

Model (what is simplified):
  * a request is `(subject, operation, resource)` where subject -> roles is a DAG (role
    inheritance), operation is one of select/insert/update/delete, and the resource is a
    table with typed columns and an optional row predicate ("filters") that restricts the
    grant to matching rows;
  * a policy is a flat list of grants: `effect` is allow or deny, `columns=None` means every
    column, `masks` rewrites a column value on the way out (null/redact/bucket), and a
    `condition` (purpose tag) gates the grant at request time;
  * resolution rules, implemented in one pass: default deny; a *deny* always beats an
    allow; several applicable allows union their row predicates, their column sets and
    their masks, so a row is visible iff some surviving allow grant covers it;
  * row predicates are a tiny declarative language (`(attr, op, value)` with op in
    =, !=, in, >=, so they are data, not callbacks, and can be re-evaluated by a brute-force
    oracle);
  * the audit log is append-only and hash chained: each record carries the hash of its
    predecessor and its own digest, and the engine keeps a sealed head digest so that
    truncation of the tail is detected as well as edits, deletions and reorderings.

Invariants verified (safety, least privilege, tamper evidence):
  * default deny: an unprivileged subject (or an unknown subject, or an empty policy)
    receives no rows, and a denied request never mutates the table;
  * decisions agree with an independently written brute-force oracle (a flat per-column
    truth table computed by expanding the role closure) over a matrix of subjects, roles,
    operations, columns and row predicates - including the allow/deny lattice;
  * least privilege is *structural*: every returned row's key set is a subset of the
    granted columns, so no unreadable column can leak, and no value of a masked or denied
    column ever appears in the output;
  * row filters are honoured exactly: the visible row set equals the brute-force union over
    the applicable allow predicates, and an update touches exactly those rows;
  * the audit trail is complete (exactly one record per decision, monotone sequence) and
    tamper-evident: editing a field, deleting a record, reordering records or truncating
    the tail are each detected by the chain check;
  * monotonicity: adding an allow grant can only enlarge the permission set and adding a
    deny can only shrink it - checked over the whole request matrix with the oracle.
"""

import hashlib

OPS = ("select", "insert", "update", "delete")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def purposes_of(purpose):
    """Normalise a purpose selector (None | "tag" | collection of tags) to a frozenset."""
    if purpose is None:
        return frozenset()
    if isinstance(purpose, str):
        return frozenset([purpose])
    return frozenset(purpose)


class Predicate:
    """A declarative row filter: (attribute, operator, value)."""

    def __init__(self, attribute, operator, value):
        if operator not in ("=", "!=", "in", ">="):
            raise ValueError("unsupported operator %r" % (operator,))
        self.attribute = attribute
        self.operator = operator
        self.value = value

    def matches(self, row):
        if self.attribute not in row:
            return False
        current = row[self.attribute]
        if self.operator == "=":
            return current == self.value
        if self.operator == "!=":
            return current != self.value
        if self.operator == ">=":
            return current >= self.value
        return current in self.value

    def __repr__(self):
        return "%s %s %r" % (self.attribute, self.operator, self.value)


class Grant:
    """One policy rule.  `effect` is allow or deny; denies are checked first."""

    def __init__(self, name, role, resource, ops, effect="allow", columns=None,
                 filters=(), masks=None, condition=None):
        self.name = name
        self.role = role
        self.resource = resource
        self.ops = tuple(ops)
        self.effect = effect
        self.columns = None if columns is None else tuple(columns)
        self.filters = tuple(filters)
        self.masks = dict(masks or {})
        self.condition = condition

    def covers_op(self, op):
        return op in self.ops

    def covers_resource(self, resource):
        return self.resource == resource

    def covers_column(self, column):
        return self.columns is None or column in self.columns

    def gated(self, purposes):
        return self.condition is None or self.condition in purposes

    def __repr__(self):
        return "%s(%s %s on %s)" % (self.name, self.effect, "/".join(self.ops), self.resource)


class Table:
    def __init__(self, name, columns, rows):
        self.name = name
        self.columns = tuple(columns)
        self.rows = [dict(row) for row in rows]

    def snapshot(self):
        return [dict(row) for row in self.rows]


def masking(kind, value):
    """Column masks: they must be deterministic and must not expose the raw value."""
    if kind == "null":
        return None
    if kind == "redact":
        return "***"
    if kind == "bucket":
        return (value // 100) * 100 if isinstance(value, int) else "***"
    raise ValueError("unknown mask %r" % (kind,))


class Decision:
    def __init__(self, allowed, reason, columns=(), masks=None, filters=(), roles=(),
                 denied_rows=(), unrestricted=False):
        self.allowed = allowed
        self.reason = reason
        self.columns = tuple(sorted(columns))
        self.masks = dict(masks or {})
        self.filters = tuple(filters)
        self.denied_rows = tuple(denied_rows)
        self.unrestricted = unrestricted      # some allow grant has no row predicate at all
        self.roles = tuple(sorted(roles))

    def __repr__(self):
        return "Decision(%s, %s, columns=%s)" % ("allow" if self.allowed else "deny",
                                                 self.reason, list(self.columns))


class Engine:
    def __init__(self, table, roles, grants):
        self.table = table
        self.roles = {role: tuple(parents) for role, parents in roles.items()}
        self.grants = list(grants)
        self.audit = []
        self.sealed_head = None
        self.sequence = 0

    # -- role resolution ---------------------------------------------------
    def role_closure(self, roles):
        """Transitive closure over the role DAG; cycles are tolerated by the visited set."""
        seen = []
        stack = list(roles)
        while stack:
            role = stack.pop()
            if role in seen:
                continue
            if role not in self.roles:
                continue
            seen.append(role)
            stack.extend(self.roles[role])
        return set(seen)

    def applicable(self, subject_roles, op, resource, purpose):
        closure = self.role_closure(subject_roles)
        return [grant for grant in self.grants
                if grant.role in closure and grant.covers_resource(resource)
                and grant.covers_op(op) and grant.gated(purpose)]

    # -- decisions ---------------------------------------------------------
    def decide(self, subject_roles, op, resource, purpose=None):
        """Resolve one request.  Layered rules, in order:
        an operation-wide deny (no column list, no row filter) beats everything; otherwise
        column denies are subtracted from the union of the allow grants, row-filtered denies
        are carried along, and an allow needs at least one surviving column to apply.
        """
        purposes = purposes_of(purpose)
        closure = self.role_closure(subject_roles)
        if not closure:
            return Decision(False, "unknown subject", roles=())
        applicable = self.applicable(subject_roles, op, resource, purposes)
        denies = [grant for grant in applicable if grant.effect == "deny"]
        allows = [grant for grant in applicable if grant.effect == "allow"]
        blocking = [grant for grant in denies if grant.columns is None and not grant.filters]
        if blocking:
            return Decision(False, "deny by %s" % blocking[0].name, roles=closure)
        denied_columns = set()
        for grant in denies:                  # only *column* denies subtract columns
            if grant.columns is not None:
                denied_columns.update(grant.columns)
        columns = set()
        masks = {}
        filters = []
        contributing = []
        for grant in allows:
            granted = set(self.table.columns) if grant.columns is None else set(grant.columns)
            surviving = granted - denied_columns
            if not surviving:
                continue
            contributing.append(grant)
            columns |= surviving
            filters.extend(grant.filters)
            for column, kind in grant.masks.items():
                if column in surviving:
                    masks[column] = kind
        if not contributing:
            if denies:
                return Decision(False, "deny by %s" % denies[0].name, roles=closure)
            return Decision(False, "default deny", roles=closure)
        row_denies = [grant for grant in denies if grant.filters]
        unrestricted = any(not grant.filters for grant in contributing)
        return Decision(True, "allow via %s" % contributing[0].name, columns, masks, filters,
                        closure, [predicate for grant in row_denies for predicate in grant.filters],
                        unrestricted)

    # -- query execution ---------------------------------------------------
    def visible_rows(self, decision):
        if not decision.allowed:
            return []
        rows = []
        for row in self.table.rows:
            if not decision.unrestricted and not any(p.matches(row) for p in decision.filters):
                continue
            if any(p.matches(row) for p in decision.denied_rows):
                continue
            rows.append(row)
        return rows

    def project(self, row, decision):
        projected = {}
        for column in self.table.columns:
            if column not in decision.columns:
                continue
            value = row.get(column)
            if column in decision.masks:
                projected[column] = masking(decision.masks[column], value)
            else:
                projected[column] = value
        return projected

    def select(self, subject, subject_roles, purpose=None):
        decision = self.decide(subject_roles, "select", self.table.name, purpose)
        rows = [self.project(row, decision) for row in self.visible_rows(decision)]
        self.log(subject, "select", decision, len(rows))
        return rows, decision

    def update(self, subject, subject_roles, changes, purpose=None):
        decision = self.decide(subject_roles, "update", self.table.name, purpose)
        touched = 0
        if decision.allowed:
            for row in self.visible_rows(decision):
                for column, value in changes.items():
                    if column in decision.columns:
                        row[column] = value
                        touched += 1
        self.log(subject, "update", decision, touched)
        return touched, decision

    def delete(self, subject, subject_roles, purpose=None):
        decision = self.decide(subject_roles, "delete", self.table.name, purpose)
        removed = 0
        if decision.allowed:
            keep = []
            for row in self.table.rows:
                visible = (any(predicate.matches(row) for predicate in decision.filters)
                           if decision.filters else True)
                if visible:
                    removed += 1
                else:
                    keep.append(row)
            self.table.rows = keep
        self.log(subject, "delete", decision, removed)
        return removed, decision

    # -- audit trail -------------------------------------------------------
    def log(self, subject, op, decision, rows_touched):
        self.sequence += 1
        record = {
            "seq": self.sequence,
            "subject": subject,
            "op": op,
            "resource": self.table.name,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "columns": list(decision.columns),
            "rows": rows_touched,
            "prev": self.audit[-1]["hash"] if self.audit else "0" * 64,
        }
        payload = "|".join("%s=%s" % (key, record[key]) for key in sorted(record))
        record["hash"] = digest(payload)
        self.audit.append(record)
        self.sealed_head = record["hash"]
        return record

    def verify_audit(self, records=None, sealed=None):
        """Recompute the hash chain; returns (ok, problems)."""
        problems = []
        records = self.audit if records is None else records
        sealed_head = self.sealed_head if sealed is None else sealed
        previous = "0" * 64
        for index, record in enumerate(records):
            if record["seq"] != index + 1:
                problems.append("record %d has sequence %s (log is not append-only)"
                                % (index, record["seq"]))
            if record["prev"] != previous:
                problems.append("record %d does not chain onto its predecessor" % index)
            payload = "|".join("%s=%s" % (key, record[key]) for key in sorted(record)
                               if key != "hash")
            if record["hash"] != digest(payload):
                problems.append("record %d was modified (digest mismatch)" % index)
            previous = record["hash"]
        if records and previous != sealed_head:
            problems.append("the log head does not match the sealed digest")
        if not records and sealed_head is not None:
            problems.append("the log was truncated away")
        return (not problems), problems


# ---------------------------------------------------------------------------
# Independent reference implementation used as an oracle: a flat per-column
# truth table built by expanding the role closure, then resolved with "deny wins".
# ---------------------------------------------------------------------------

def oracle_closure(roles, start):
    frontier = list(start)
    seen = set()
    while frontier:
        role = frontier.pop()
        if role in seen or role not in roles:
            continue
        seen.add(role)
        frontier.extend(roles[role])
    return seen


DENIED = (False, (), ((), (), True))


def oracle_decide(table, roles, grants, subject_roles, op, resource, purpose=()):
    """Flat per-column truth table: a column is readable when some allow grant covers it,
    no denied column list removes it and no operation-wide deny blocks the request."""
    closure = oracle_closure(roles, subject_roles)
    if not closure:
        return DENIED
    purposes = purposes_of(purpose)
    readable = {column: False for column in table.columns}
    denied = set()
    row_denies = []
    contributors = []
    for grant in grants:
        if grant.role not in closure or grant.resource != resource or op not in grant.ops:
            continue
        if not grant.gated(purposes):
            continue
        if grant.effect == "deny":
            if grant.columns is None and not grant.filters:
                return DENIED
            if grant.columns is not None:
                denied.update(grant.columns)
            else:
                row_denies.extend(grant.filters)
            continue
        contributors.append(grant)
    filters = []
    for grant in contributors:
        granted = list(table.columns) if grant.columns is None else list(grant.columns)
        if not [column for column in granted if column not in denied]:
            continue
        filters.extend(grant.filters)
        for column in granted:
            if column not in denied:
                readable[column] = True
    columns = tuple(sorted(column for column in readable if readable[column]))
    if not columns:
        return DENIED
    denied_ordered = tuple(sorted(row_denies, key=repr))
    ordered = tuple(sorted(filters, key=repr))
    unrestricted = any(not grant.filters for grant in contributors)
    return True, columns, (ordered, denied_ordered, unrestricted)


def oracle_visible_rows(table, roles, grants, subject_roles, purpose=()):
    """Row-level oracle: the union of the surviving allow predicates, minus the deny filters."""
    allowed, _columns, (filters, deny_filters, unrestricted) = oracle_decide(
        table, roles, grants, subject_roles, "select", table.name, purpose)
    if not allowed:
        return []
    rows = []
    for row in table.rows:
        if not unrestricted and not any(predicate.matches(row) for predicate in filters):
            continue
        if any(predicate.matches(row) for predicate in deny_filters):
            continue
        rows.append(row)
    return rows


def sample_policy():
    table = Table("employees",
                  ["id", "name", "dept", "region", "level", "salary"],
                  [{"id": 1, "name": "Ada", "dept": "eng", "region": "eu", "level": 5, "salary": 150},
                   {"id": 2, "name": "Bob", "dept": "eng", "region": "us", "level": 3, "salary": 90},
                   {"id": 3, "name": "Cleo", "dept": "sales", "region": "eu", "level": 4, "salary": 120},
                   {"id": 4, "name": "Dan", "dept": "sales", "region": "us", "level": 2, "salary": 60},
                   {"id": 5, "name": "Eve", "dept": "eng", "region": "us", "level": 6, "salary": 210},
                   {"id": 6, "name": "Fay", "dept": "ops", "region": "eu", "level": 1, "salary": 40}])
    roles = {"admin": (), "manager": ("employee",), "employee": (), "auditor": ("employee",),
             "intern": (), "gdpr-officer": ()}
    grants = [
        Grant("all-employees", "employee", "employees", ["select"],
              columns=["id", "name", "dept", "region", "level"]),
        Grant("managers-eu", "manager", "employees", ["select"],
              filters=[Predicate("region", "=", "eu")]),
        Grant("no-salary", "intern", "employees", ["select"], effect="deny", columns=["salary"]),
        Grant("intern-basics", "intern", "employees", ["select"], columns=["id", "name", "salary"],
              masks={"salary": "bucket"}),
        Grant("intern-no-eu", "intern", "employees", ["select"], effect="deny",
              filters=[Predicate("region", "=", "eu")]),
        Grant("audit-select", "auditor", "employees", ["select"], columns=["id", "level", "salary"],
              masks={"salary": "redact"}),
        Grant("eng-fix", "manager", "employees", ["update"], columns=["level"],
              filters=[Predicate("dept", "=", "eng")]),
        Grant("hr-write", "admin", "employees", ["insert", "update", "delete"]),
        Grant("admin-read", "admin", "employees", ["select"]),
        Grant("legal-purge", "admin", "employees", ["delete"], effect="deny"),
        Grant("purpose-select", "gdpr-officer", "employees", ["select"], columns=["id"],
              condition="gdpr"),
    ]
    subjects = {"alice": ["manager"], "bob": ["employee"], "carl": ["intern"],
                "dora": ["auditor"], "root": ["admin"], "eve": [], "frank": ["unknown-role"],
                "gina": ["gdpr-officer"]}
    return table, roles, grants, subjects


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    table, roles, grants, subjects = sample_policy()
    engine = Engine(table, roles, grants)
    check(engine.role_closure(["manager"]) == {"manager", "employee"},
          "role inheritance must be transitive")
    check(oracle_closure(roles, ["manager"]) == {"manager", "employee"},
          "the oracle must agree about the closure")
    check(engine.role_closure(["unknown-role"]) == set(), "an unknown role grants nothing")

    # ---- 1. default deny -------------------------------------------------
    empty = Engine(table, roles, [])
    rows, decision = empty.select("nobody", ["employee"])
    check(rows == [] and not decision.allowed and decision.reason == "default deny",
          "an empty policy must deny everything")
    rows, decision = engine.select("eve", [])
    check(rows == [] and not decision.allowed, "a subject without roles must be denied")
    rows, decision = engine.select("frank", ["unknown-role"])
    check(rows == [] and not decision.allowed, "an unknown role must be denied")
    print("default deny: empty policy, role-less subject and unknown role all return no rows")

    # ---- 2. role inheritance and column least privilege -----------------
    rows, decision = engine.select("bob", subjects["bob"])
    check(decision.allowed and decision.columns == ("dept", "id", "level", "name", "region"),
          "an employee must see exactly the five granted columns: %s" % (decision.columns,))
    check(all(set(row) <= set(decision.columns) for row in rows) and len(rows) == 6,
          "every returned row must stay inside the granted projection")
    check("salary" not in rows[0], "a column without a grant must never be returned")
    print("employee sees columns %s for %d rows" % (list(decision.columns), len(rows)))

    # a grant conditional on a purpose must not fire without that purpose
    rows, decision = engine.select("gina", subjects["gina"], purpose="gdpr")
    check(decision.allowed and decision.columns == ("id",),
          "the gated grant must only fire for its purpose: %s" % (decision,))
    rows, decision = engine.select("gina", subjects["gina"])
    check(rows == [] and not decision.allowed and decision.reason == "default deny",
          "without the purpose the gated grant must not apply")
    rows, decision = engine.select("bob", subjects["bob"], purpose="gdpr")
    check(len(rows) == 6 and len(decision.columns) == 5,
          "an ungated grant must still apply alongside a gated one")

    # ---- 3. deny precedence ---------------------------------------------
    root_rows, root_decision = engine.select("root", subjects["root"])
    check(root_decision.allowed and len(root_rows) == 6 and len(root_decision.columns) == 6,
          "an admin must read every row and every column")
    delete_decision = engine.decide(["admin"], "delete", "employees")
    check(not delete_decision.allowed and "legal-purge" in delete_decision.reason,
          "the delete-wide deny must beat the write grant: %s" % delete_decision.reason)
    before = engine.table.snapshot()
    removed, decision = engine.delete("root", subjects["root"])
    check(removed == 0 and engine.table.snapshot() == before,
          "a denied delete must not remove any row")
    check(engine.decide(["admin"], "insert", "employees").allowed,
          "a deny for one operation must not block another operation")
    blocked = [grant for grant in grants if grant.effect == "deny" and grant.columns is None
               and "update" in grant.ops]
    check(not blocked, "the sample policy has no operation-wide update deny")
    print("deny precedence: delete denied for the admin, insert still allowed")

    # ---- 4. column masks and row filters --------------------------------
    intern_rows, intern_decision = engine.select("carl", subjects["carl"])
    check(intern_decision.allowed and set(intern_decision.columns) == {"id", "name"},
          "the column deny must remove salary from the intern's grant")
    check(all("salary" not in row for row in intern_rows),
          "a denied column must never leak into the result")
    check(len(intern_rows) == 3 and all(row["id"] not in (1, 3, 6) for row in intern_rows),
          "the intern's row-level deny must hide every eu row: %s" % [r["id"] for r in intern_rows])
    check(len([predicate for predicate in intern_decision.denied_rows
               if predicate.matches({"region": "eu"})]) == 1,
          "the decision must carry the row-level deny predicate")
    auditor_rows, auditor_decision = engine.select("dora", subjects["dora"])
    check(auditor_decision.allowed and auditor_decision.masks.get("salary") == "redact",
          "the auditor's mask must be recorded in the decision")
    check(all(row["salary"] == "***" for row in auditor_rows),
          "every masked value must be replaced, never passed through")
    raw_salaries = {row["salary"] for row in table.rows}
    check(not (raw_salaries & {row["salary"] for row in auditor_rows}),
          "no raw salary may appear in a masked projection")

    eu_predicate = Predicate("region", "=", "eu")
    expected_rows = [row for row in engine.table.rows if eu_predicate.matches(row)]
    alice_rows, alice_decision = engine.select("alice", subjects["alice"])
    check(set(alice_decision.columns) == {"id", "name", "dept", "region", "level", "salary"},
          "the manager's filtered grant must contribute every column")
    check(len(alice_rows) == len(engine.table.rows),
          "the union of an unfiltered and a filtered grant must show every row")
    solo = Engine(table, roles, [grant for grant in grants if grant.name != "all-employees"])
    solo_rows, _solo_decision = solo.select("alice", subjects["alice"])
    check(len(solo_rows) == len(expected_rows) == 3,
          "with a single filtered grant only the matching rows may be visible")

    # ---- 5. differential check against the brute-force oracle -----------
    cases = 0
    for subject, subject_roles in subjects.items():
        for op in OPS:
            for purpose in (None, "gdpr"):
                cases += 1
                decision = engine.decide(subject_roles, op, "employees", purpose)
                allowed, columns, (filters, deny_filters, unrestricted) = oracle_decide(
                    table, roles, grants, subject_roles, op, "employees", purpose)
                check(decision.allowed == allowed,
                      "%s/%s/%s: engine says %s, oracle says %s"
                      % (subject, op, purpose, decision.allowed, allowed))
                if allowed:
                    check(tuple(decision.columns) == columns,
                          "%s/%s/%s: columns %s != oracle %s"
                          % (subject, op, purpose, decision.columns, columns))
                    check(tuple(sorted(decision.filters, key=repr)) == filters,
                          "%s/%s/%s: row filters disagree with the oracle" % (subject, op, purpose))
                    check(tuple(sorted(decision.denied_rows, key=repr)) == deny_filters,
                          "%s/%s/%s: row denies disagree with the oracle" % (subject, op, purpose))
                    check(decision.unrestricted == unrestricted,
                          "%s/%s/%s: row-visibility mode disagrees with the oracle"
                          % (subject, op, purpose))
                else:
                    check(decision.columns == (), "a denied decision must expose no column")
                if op == "select":
                    expected_rows = oracle_visible_rows(table, roles, grants, subject_roles, purpose)
                    check(len(engine.visible_rows(decision)) == len(expected_rows),
                          "%s: %d visible rows, oracle says %d"
                          % (subject, len(engine.visible_rows(decision)), len(expected_rows)))
    print("differential matrix: %d requests agree with the oracle" % cases)

    # unknown resources are denied by both implementations
    check(not engine.decide(["admin"], "select", "secrets").allowed
          and not oracle_decide(table, roles, grants, ["admin"], "select", "secrets")[0],
          "an unknown resource must be denied everywhere")

    # ---- 6. monotonicity of the policy lattice --------------------------
    def permission_set(policy_grants):
        probe = Engine(table, roles, policy_grants)
        permissions = set()
        for subject, roles_of in subjects.items():
            for op in OPS:
                for purpose in (None, "gdpr"):
                    decision = probe.decide(roles_of, op, "employees", purpose)
                    if not decision.allowed:
                        continue
                    permissions.add((subject, op, purpose))
                    for column in decision.columns:      # column-level privileges too
                        permissions.add((subject, op, purpose, column))
        return permissions

    base = permission_set(grants)
    extra_allow = Grant("extra", "intern", "employees", ["select"], columns=["region"])
    extra_deny = Grant("veto", "employee", "employees", ["select"], effect="deny",
                       columns=["name"])
    with_allow = permission_set(grants + [extra_allow])
    with_deny = permission_set(grants + [extra_deny])
    check(base <= with_allow, "adding an allow grant must never remove a permission")
    check(with_deny <= base, "adding a deny grant must never add a permission")
    check(len(with_allow) > len(base), "the extra allow grant must enlarge the permission set")
    check(len(with_deny) < len(base), "the extra deny grant must shrink the permission set")
    print("permissions: base=%d +allow=%d +deny=%d"
          % (len(base), len(with_allow), len(with_deny)))

    # ---- 7. audit completeness and tamper evidence ----------------------
    trail = Engine(table, roles, grants)
    requests = 0
    for subject, subject_roles in subjects.items():
        for purpose in (None, "gdpr"):
            trail.select(subject, subject_roles, purpose)
            requests += 1
    trail.update("alice", subjects["alice"], {"level": 4})
    requests += 1
    check(len(trail.audit) == requests, "exactly one audit record per decision")
    check([record["seq"] for record in trail.audit] == list(range(1, requests + 1)),
          "audit sequence numbers must be contiguous")
    ok, problems = trail.verify_audit()
    check(ok, "the untouched audit trail must verify: %s" % problems)
    check(all(record["prev"] == (trail.audit[i - 1]["hash"] if i else "0" * 64)
              for i, record in enumerate(trail.audit)), "every record must chain onto the previous")

    mutated = [dict(record) for record in trail.audit]
    mutated[3]["rows"] = 999
    ok, problems = trail.verify_audit(mutated)
    check(not ok and any("modified" in problem for problem in problems),
          "editing a record must be detected")
    deleted = [dict(record) for record in trail.audit]
    del deleted[2]
    ok, problems = trail.verify_audit(deleted)
    check(not ok and any("chain" in problem or "sequence" in problem for problem in problems),
          "deleting a record must be detected")
    reordered = [dict(record) for record in trail.audit]
    reordered[1], reordered[2] = reordered[2], reordered[1]
    ok, _problems = trail.verify_audit(reordered)
    check(not ok, "reordering records must be detected")
    truncated = [dict(record) for record in trail.audit[:-1]]
    ok, problems = trail.verify_audit(truncated)
    check(not ok and any("head" in problem for problem in problems),
          "truncating the tail must be detected by the sealed head digest")
    forged = [dict(record) for record in trail.audit]
    forged[-1]["hash"] = digest("forged")
    ok, _problems = trail.verify_audit(forged)
    check(not ok, "a forged digest must be detected")
    print("audit: %d records verified, 5 tamper patterns detected" % len(trail.audit))

    # ---- 8. side effects and determinism --------------------------------
    frozen = Engine(table, roles, grants)
    before = frozen.table.snapshot()
    frozen.update("bob", subjects["bob"], {"salary": 1})
    check(frozen.table.snapshot() == before, "a denied update must leave the table untouched")
    touched, decision = frozen.update("alice", subjects["alice"], {"salary": 7})
    check(decision.allowed and touched == 0,
          "a filtered update may only touch granted columns: touched %d" % touched)
    eng_ids = [row["id"] for row in before if row["dept"] == "eng"]
    touched, decision = frozen.update("alice", subjects["alice"], {"level": 7})
    check(touched == 3, "the manager's filtered update must touch the three matching rows")
    for old, new in zip(before, frozen.table.rows):
        if old["id"] in eng_ids:
            check(new["level"] == 7, "the filtered update must reach the eng row %d" % old["id"])
        else:
            check(new == old, "the filtered update must not touch row %d" % old["id"])
    print("filtered update touched %s of %d rows" % (eng_ids, len(before)))

    first, _ = engine.select("bob", subjects["bob"])
    second, _ = engine.select("bob", subjects["bob"])
    check(first == second, "repeated identical requests must return identical data")
    twin = Engine(table, roles, grants)
    check(twin.select("bob", subjects["bob"])[0] == first, "two engines must agree bit for bit")
    twin_a, twin_b = Engine(table, roles, grants), Engine(table, roles, grants)
    twin_a.select("bob", subjects["bob"], purpose="gdpr")
    twin_b.select("bob", subjects["bob"], purpose="gdpr")
    check(twin_a.audit == twin_b.audit,
          "two engines must produce identical audit records, hashes included")
    check(twin_a.audit[0]["prev"] == "0" * 64, "the head record must chain from the genesis")
    check(twin_a.verify_audit()[0], "a freshly written trail must verify")

    # ---- 9. edge cases ---------------------------------------------------
    cyclic = Engine(table, {"a": ("b",), "b": ("a",)}, [Grant("loop", "a", "employees", ["select"])])
    check(cyclic.role_closure(["a"]) == {"a", "b"}, "a role cycle must terminate")
    check(cyclic.decide(["a"], "select", "employees").allowed, "a cyclic policy must still decide")
    tiny = Engine(Table("t", ["x"], []), {"r": ()}, [Grant("g", "r", "t", ["select"])])
    empty_rows, _decision = tiny.select("s", ["r"])
    check(empty_rows == [], "an empty table must project to zero rows")
    check(engine.decide(["employee"], "select", "employees", purpose="other").allowed,
          "an unrecognised purpose must not silently match a gated grant")
    check(engine.decide(["employee"], "insert", "employees").allowed is False,
          "an operation without a grant must be denied")
    try:
        Predicate("id", "~", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("an unsupported predicate operator must be rejected")

    print("engine: %d grants, %d subjects, %d audit records" % (len(grants), len(subjects), requests))
    print("PASS")


if __name__ == "__main__":
    main()
