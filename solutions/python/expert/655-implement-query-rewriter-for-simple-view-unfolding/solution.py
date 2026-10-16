"""Task 655 - Implement query rewriter for simple view unfolding.  (Tier: Expert, Ch 14 - Databases)

Unfolding replaces every `ViewRef(name, alias)` by the view definition with its internal qualifiers
suffixed by the use-site alias (so one view used twice cannot collide) and its outputs renamed to
`alias.output`; nested views flatten recursively, leaving a plan with no view reference at all. The
optimiser then pushes selections down - through a `Project`'s column mapping, and inside an inner `Join`
to the side that holds all of an atom's columns, keeping atoms that span both sides above the join - and
prunes projection outputs no ancestor needs, iterating to a fixpoint.

Verified: structural soundness (every predicate atom only references columns its own child produces),
answers identical to the naive unfolded plan for a hand-checked ground truth plus generated queries,
idempotence, determinism, push-down where it is legal and its absence where it is illegal, and disjoint
qualifiers when one view is used twice. Simplified: inner joins only, no aggregates or grouping.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def qualify(alias, name):
    return "%s.%s" % (alias, name)


def suffix_move(suffix):
    """Column rename: `u.id` -> `u<suffix>.id` (qualifier-suffixing for one view use site)."""
    def move(column):
        if "." not in column:
            return column
        qualifier, name = column.split(".", 1)
        return "%s%s.%s" % (qualifier, suffix, name)
    return move


def identity_move(column):
    return column


# --------------------------------------------------------------------------- predicates
class Atom:
    """One predicate atom: a column compared with a literal, or two columns compared."""

    __slots__ = ("left", "op", "right")

    def __init__(self, left, op, right):
        self.left, self.op, self.right = left, op, right

    def columns(self):
        return [self.left, self.right] if self.op == "=col" else [self.left]

    def eval(self, env):
        if self.op == "=col":
            return env[self.left] == env[self.right]
        value = env[self.left]
        if self.op == "=":
            return value == self.right
        if self.op == "<":
            return value < self.right
        if self.op == "<=":
            return value <= self.right
        if self.op == ">":
            return value > self.right
        if self.op == ">=":
            return value >= self.right
        if self.op == "!=":
            return value != self.right
        raise ValueError(f"unknown operator {self.op}")

    def renamed(self, move):
        if self.op == "=col":
            return Atom(move(self.left), "=col", move(self.right))
        return Atom(move(self.left), self.op, self.right)

    def signature(self):
        return (self.left, self.op, self.right)


class Predicate:
    __slots__ = ("atoms",)

    def __init__(self, atoms):
        self.atoms = list(atoms)

    def eval(self, env):
        return all(atom.eval(env) for atom in self.atoms)

    def columns(self):
        found = set()
        for atom in self.atoms:
            found.update(atom.columns())
        return found

    def renamed(self, move):
        return Predicate([atom.renamed(move) for atom in self.atoms])

    def signature(self):
        return tuple(atom.signature() for atom in self.atoms)


def compare(column, op, value):
    return Predicate([Atom(column, op, value)])


def column_equals(left, right):
    return Predicate([Atom(left, "=col", right)])


# --------------------------------------------------------------------------- plan nodes
class Scan:
    __slots__ = ("table", "alias")

    def __init__(self, table, alias):
        self.table, self.alias = table, alias


class Select:
    __slots__ = ("predicate", "child")

    def __init__(self, predicate, child):
        self.predicate, self.child = predicate, child


class Project:
    __slots__ = ("items", "child")

    def __init__(self, items, child):
        self.items, self.child = list(items), child


class Join:
    __slots__ = ("predicate", "left", "right")

    def __init__(self, predicate, left, right):
        self.predicate, self.left, self.right = predicate, left, right


class ViewRef:
    __slots__ = ("name", "alias")

    def __init__(self, name, alias):
        self.name, self.alias = name, alias


class ViewDef:
    __slots__ = ("name", "items", "child")

    def __init__(self, name, items, child):
        self.name, self.items, self.child = name, list(items), child

    def outputs(self):
        return [out for _, out in self.items]


def children(node):
    if isinstance(node, (Select, Project)):
        return [node.child]
    if isinstance(node, Join):
        return [node.left, node.right]
    return []


def walk(node):
    yield node
    for child in children(node):
        yield from walk(child)


def plan_columns(node, catalog):
    if isinstance(node, Scan):
        columns, _ = catalog[node.table]
        return [qualify(node.alias, column) for column in columns]
    if isinstance(node, Select):
        return plan_columns(node.child, catalog)
    if isinstance(node, Project):
        return [out for _, out in node.items]
    if isinstance(node, Join):
        return plan_columns(node.left, catalog) + plan_columns(node.right, catalog)
    raise TypeError(f"cannot derive the schema of {type(node).__name__}")


def signature(node):
    if isinstance(node, Scan):
        return ("scan", node.table, node.alias)
    if isinstance(node, Select):
        return ("select", node.predicate.signature(), signature(node.child))
    if isinstance(node, Project):
        return ("project", tuple(node.items), signature(node.child))
    if isinstance(node, Join):
        return ("join", node.predicate.signature(), signature(node.left), signature(node.right))
    raise TypeError(f"cannot sign {type(node).__name__}")


# --------------------------------------------------------------------------- unfolding
def rename_qualifiers(node, move):
    if isinstance(node, Scan):
        return Scan(node.table, move(qualify(node.alias, "x")).split(".")[0])
    if isinstance(node, Select):
        return Select(node.predicate.renamed(move), rename_qualifiers(node.child, move))
    if isinstance(node, Project):
        items = [(move(source), move(out)) for source, out in node.items]
        return Project(items, rename_qualifiers(node.child, move))
    if isinstance(node, Join):
        return Join(node.predicate.renamed(move),
                    rename_qualifiers(node.left, move), rename_qualifiers(node.right, move))
    if isinstance(node, ViewRef):
        return ViewRef(node.name, move(qualify(node.alias, "x")).split(".")[0])
    raise TypeError(type(node).__name__)


def unfold(node, views):
    """Replace every view reference by its definition; each use site gets unique qualifiers."""
    if isinstance(node, Scan):
        return node
    if isinstance(node, Select):
        return Select(node.predicate, unfold(node.child, views))
    if isinstance(node, Project):
        return Project(node.items, unfold(node.child, views))
    if isinstance(node, Join):
        return Join(node.predicate, unfold(node.left, views), unfold(node.right, views))
    if isinstance(node, ViewRef):
        definition = views[node.name]
        move = suffix_move("_" + node.alias)
        inner = rename_qualifiers(unfold(definition.child, views), move)
        items = [(move(source), qualify(node.alias, out)) for source, out in definition.items]
        return Project(items, inner)
    raise TypeError(type(node).__name__)


# --------------------------------------------------------------------------- optimiser
def _split_atoms(atoms, left_columns, right_columns):
    left, right, rest = [], [], []
    for atom in atoms:
        columns = set(atom.columns())
        if columns <= left_columns:
            left.append(atom)
        elif columns <= right_columns:
            right.append(atom)
        else:
            rest.append(atom)
    return left, right, rest


def pushdown(node, catalog):
    if isinstance(node, Scan):
        return node
    if isinstance(node, ViewRef):
        raise ValueError("views must be unfolded before rewriting")
    if isinstance(node, Project):
        child = pushdown(node.child, catalog)
        if isinstance(child, Project):
            mapping = {out: source for source, out in child.items}
            items = [(mapping.get(source, source), out) for source, out in node.items]
            return Project(items, pushdown(child.child, catalog))
        return Project(node.items, child)
    if isinstance(node, Join):
        left = pushdown(node.left, catalog)
        right = pushdown(node.right, catalog)
        available = set(plan_columns(left, catalog)) | set(plan_columns(right, catalog))
        for atom in node.predicate.atoms:
            check(set(atom.columns()) <= available,
                  "an inner join predicate must reference columns of its two sides")
        return Join(node.predicate, left, right)
    if isinstance(node, Select):
        atoms = list(node.predicate.atoms)
        child = pushdown(node.child, catalog)
        while True:
            if isinstance(child, Project):
                mapping = {out: source for source, out in child.items}
                movable = [atom for atom in atoms
                           if all(column in mapping for column in atom.columns())]
                if not movable:
                    break
                atoms = [atom for atom in atoms if atom not in movable]
                inner = Predicate([atom.renamed(mapping.get) for atom in movable])
                child = Project(child.items, pushdown(Select(inner, child.child), catalog))
                continue
            if isinstance(child, Join):
                left_columns = set(plan_columns(child.left, catalog))
                right_columns = set(plan_columns(child.right, catalog))
                to_left, to_right, rest = _split_atoms(atoms, left_columns, right_columns)
                if not (to_left or to_right):
                    break
                atoms = rest
                left = (pushdown(Select(Predicate(to_left), child.left), catalog)
                        if to_left else child.left)
                right = (pushdown(Select(Predicate(to_right), child.right), catalog)
                         if to_right else child.right)
                child = Join(child.predicate, left, right)
                continue
            break
        if not atoms:
            return child
        return Select(Predicate(atoms), child)
    raise TypeError(type(node).__name__)


def prune(node, needed, catalog):
    """Drop projection outputs no ancestor uses. `needed` is None at the root (keep everything)."""
    if isinstance(node, Scan):
        return node
    if isinstance(node, Select):
        widened = None if needed is None else set(needed) | node.predicate.columns()
        return Select(node.predicate, prune(node.child, widened, catalog))
    if isinstance(node, Project):
        keep = [item for item in node.items if needed is None or item[1] in needed]
        if not keep:
            keep = node.items[:1]
        return Project(keep, prune(node.child, {source for source, _ in keep}, catalog))
    if isinstance(node, Join):
        if needed is None:
            return Join(node.predicate, prune(node.left, None, catalog), prune(node.right, None, catalog))
        wanted = set(needed) | node.predicate.columns()
        left_need = wanted & set(plan_columns(node.left, catalog))
        right_need = wanted & set(plan_columns(node.right, catalog))
        return Join(node.predicate, prune(node.left, left_need or None, catalog),
                    prune(node.right, right_need or None, catalog))
    raise TypeError(type(node).__name__)


def rewrite(query, views, catalog):
    """Unfold the views, then push selections down and prune projections until a fixpoint."""
    plan = unfold(query, views)
    for _ in range(4):
        before = signature(plan)
        plan = pushdown(plan, catalog)
        plan = prune(plan, None, catalog)
        if signature(plan) == before:
            break
    return plan


def verify_binding(node, catalog):
    """Every predicate must reference only columns produced by its own child."""
    if isinstance(node, Select):
        check(node.predicate.columns() <= set(plan_columns(node.child, catalog)),
              "a pushed predicate must reference columns of its own child")
    if isinstance(node, Join):
        available = set(plan_columns(node.left, catalog)) | set(plan_columns(node.right, catalog))
        check(node.predicate.columns() <= available, "a join predicate must reference available columns")
    for child in children(node):
        verify_binding(child, catalog)


# --------------------------------------------------------------------------- evaluation
def evaluate(node, catalog):
    if isinstance(node, Scan):
        columns, rows = catalog[node.table]
        schema = [qualify(node.alias, column) for column in columns]
        return schema, [tuple(row) for row in rows]
    if isinstance(node, ViewRef):
        raise ValueError("cannot evaluate an unfolded view reference")
    if isinstance(node, Select):
        schema, rows = evaluate(node.child, catalog)
        return schema, [row for row in rows if node.predicate.eval(dict(zip(schema, row)))]
    if isinstance(node, Project):
        schema, rows = evaluate(node.child, catalog)
        positions = {name: index for index, name in enumerate(schema)}
        for source, _ in node.items:
            check(source in positions, "a projected column must exist in the child schema")
        return ([out for _, out in node.items],
                [tuple(row[positions[source]] for source, _ in node.items) for row in rows])
    if isinstance(node, Join):
        left_schema, left_rows = evaluate(node.left, catalog)
        right_schema, right_rows = evaluate(node.right, catalog)
        out = []
        for left_row in left_rows:
            base = dict(zip(left_schema, left_row))
            for right_row in right_rows:
                env = dict(base)
                env.update(zip(right_schema, right_row))
                if node.predicate.eval(env):
                    out.append(tuple(left_row) + tuple(right_row))
        return left_schema + right_schema, out
    raise TypeError(type(node).__name__)


def canonical(schema, rows):
    """Order-insensitive comparison key: column names plus the bag of rows."""
    return (tuple(schema), sorted(tuple(str(value) for value in row) for row in rows))


def main():
    catalog = {
        "users": (["id", "name", "city"],
                  [(1, "ada", "paris"), (2, "bob", "rome"), (3, "cy", "paris"), (4, "dee", "oslo")]),
        "orders": (["oid", "user_id", "amount"],
                   [(10, 1, 100), (11, 1, 250), (12, 2, 50), (13, 3, 300), (14, 4, 20)]),
    }
    big_orders = ViewDef(
        "big_orders",
        [("u.id", "user_id"), ("u.name", "user_name"), ("u.city", "city"), ("o.amount", "amount")],
        Select(compare("o.amount", ">=", 100),
               Join(column_equals("u.id", "o.user_id"), Scan("users", "u"), Scan("orders", "o"))))
    views = {"big_orders": big_orders}

    # 1. hand-computed ground truth through the view
    query = Project([("v.user_name", "user_name"), ("v.amount", "amount")],
                    Select(compare("v.city", "=", "paris"), ViewRef("big_orders", "v")))
    lazy = unfold(query, views)
    check(not any(isinstance(node, ViewRef) for node in walk(lazy)), "unfolding must remove view refs")
    schema, rows = evaluate(lazy, catalog)
    check(schema == ["user_name", "amount"], "the projection must expose exactly the asked columns")
    check(sorted(rows) == sorted([("ada", 100), ("ada", 250), ("cy", 300)]),
          "hand-computed answer must be ada/100, ada/250, cy/300")

    # 2. rewritten plan answers identically and contains no view reference
    rewritten = rewrite(query, views, catalog)
    check(not any(isinstance(node, ViewRef) for node in walk(rewritten)),
          "the rewritten plan must be free of view references")
    check(canonical(*evaluate(rewritten, catalog)) == canonical(schema, rows),
          "rewriting must preserve the answer")
    verify_binding(rewritten, catalog)

    # 3. push-down really happened: the city filter sits directly above the users scan
    pushed = [node for node in walk(rewritten)
              if isinstance(node, Select) and isinstance(node.child, Scan)
              and node.child.table == "users"]
    check(bool(pushed), "the view predicate on the users column must reach the base table")
    check(pushed[0].predicate.columns() == {"u_v.city"},
          "the pushed predicate must be translated to the qualifier used by this view instance")
    check(not isinstance(rewritten, Select),
          "no predicate may linger above the whole join after rewriting")
    joins = [node for node in walk(rewritten) if isinstance(node, Join)]
    check(len(joins) == 1 and isinstance(joins[0].left, Select) and isinstance(joins[0].right, Select),
          "both sides of the view join must carry their own pushed selections")

    # 4. negative case: an atom binding both join sides must stay above the join
    cross = Project([("v.user_name", "user_name")],
                    Select(column_equals("v.user_id", "v.amount"), ViewRef("big_orders", "v")))
    rewritten_cross = rewrite(cross, views, catalog)
    stuck = [node for node in walk(rewritten_cross)
             if isinstance(node, Select) and isinstance(node.child, Join)]
    check(bool(stuck), "a predicate spanning both join sides must stay above the join")
    top_join = stuck[0].child
    left_columns = set(plan_columns(top_join.left, catalog))
    right_columns = set(plan_columns(top_join.right, catalog))
    check(any(set(atom.columns()) & left_columns and set(atom.columns()) & right_columns
              for atom in stuck[0].predicate.atoms),
          "the leftover atom must really bind columns of both sides")
    verify_binding(rewritten_cross, catalog)
    check(canonical(*evaluate(rewritten_cross, catalog)) == canonical(*evaluate(unfold(cross, views), catalog)),
          "the unpushable predicate must keep the answer")
    check(evaluate(rewritten_cross, catalog)[1] == [], "no row has id equal to an amount, so the answer is empty")

    # 5. one view used twice: qualifiers must be disambiguated and the self-join must work
    twice = Join(column_equals("a.user_id", "b.user_id"),
                 ViewRef("big_orders", "a"), ViewRef("big_orders", "b"))
    rewritten_twice = rewrite(twice, views, catalog)
    check(not any(isinstance(node, ViewRef) for node in walk(rewritten_twice)),
          "both uses of the view must unfold")
    columns = plan_columns(rewritten_twice, catalog)
    check(len(columns) == len(set(columns)) and len(columns) >= 8,
          "the two uses must expose disjoint qualified columns")
    verify_binding(rewritten_twice, catalog)
    check(canonical(*evaluate(rewritten_twice, catalog))
          == canonical(*evaluate(unfold(twice, views), catalog)),
          "the self-join through the view must keep the answer")
    base_aliases = sorted({node.alias for node in walk(rewritten_twice) if isinstance(node, Scan)})
    check(base_aliases == ["o_a", "o_b", "u_a", "u_b"],
          "each view use must alias its base tables apart from the other use")

    # 6. two-level nesting: a view over a view unfolds completely
    nested = ViewDef("paris_big", [("p.user_name", "user_name"), ("p.amount", "amount")],
                     Select(compare("p.city", "=", "paris"), ViewRef("big_orders", "p")))
    two_level = {"big_orders": big_orders, "paris_big": nested}
    query2 = Project([("w.user_name", "user_name")],
                     Select(compare("w.amount", ">", 200), ViewRef("paris_big", "w")))
    rewritten2 = rewrite(query2, two_level, catalog)
    check(not any(isinstance(node, ViewRef) for node in walk(rewritten2)),
          "a view over a view must unfold down to base tables")
    check(canonical(*evaluate(rewritten2, catalog))
          == canonical(*evaluate(unfold(query2, two_level), catalog)),
          "nested unfolding must preserve the answer")
    check(sorted(evaluate(rewritten2, catalog)[1]) == [("ada",), ("cy",)],
          "ada (250) and cy (300) are the paris customers above 200")
    verify_binding(rewritten2, catalog)

    # 7. idempotence and determinism
    check(signature(rewrite(rewritten, views, catalog)) == signature(rewritten),
          "rewriting a rewritten plan must change nothing")
    check(signature(rewrite(query, views, catalog)) == signature(rewritten),
          "rewriting must be deterministic")
    check(signature(rewrite(rewritten2, two_level, catalog)) == signature(rewritten2),
          "idempotence must hold for nested views too")

    # 8. projection pruning keeps the answer and never empties a projection
    amount_only = Project([("v.amount", "amount")], ViewRef("big_orders", "v"))
    narrow = rewrite(amount_only, views, catalog)
    check(canonical(*evaluate(narrow, catalog)) == canonical(*evaluate(unfold(amount_only, views), catalog)),
          "pruning must preserve the answer")
    check([node.items for node in walk(narrow) if isinstance(node, Project)]
          and all(len(node.items) >= 1 for node in walk(narrow) if isinstance(node, Project)),
          "no projection may be pruned to zero outputs")

    # 9. generated queries: rewritten == naive unfolded, well bound and a fixpoint
    for seed in range(6):
        threshold = 50 * (seed % 5) + 40
        candidate = Project([("v.user_name", "user_name"), ("v.amount", "amount")],
                            Select(Predicate([Atom("v.amount", ">", threshold),
                                              Atom("v.city", "!=", "oslo")]),
                                   ViewRef("big_orders", "v")))
        eager = rewrite(candidate, views, catalog)
        verify_binding(eager, catalog)
        check(canonical(*evaluate(eager, catalog))
              == canonical(*evaluate(unfold(candidate, views), catalog)),
              f"rewrite changed the answer for threshold {threshold}")
        check(signature(rewrite(eager, views, catalog)) == signature(eager),
              "every rewritten plan must be a fixpoint")

    print(f"655 unfolded nodes={sum(1 for _ in walk(rewritten_twice))} "
          f"pushed-base-selects={len(pushed)}")
    print(f"655 nested view output={sorted(evaluate(rewritten2, catalog)[1])}")
    print("PASS")


if __name__ == "__main__":
    main()
