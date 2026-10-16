"""Task 803 - Build a small SQL engine with parser, planner, and executor supporting joins and aggregates.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

A complete miniature query engine, end to end:

  * lexer    - hand written scanner producing (kind, value) tokens; rejects unterminated string
               literals and unknown characters with the offending position;
  * parser   - recursive-descent parser on the precedence ladder
               OR < AND < NOT < comparison < additive < multiplicative < primary, yielding an
               immutable AST (nested tuples) for SELECT [DISTINCT]/FROM/JOIN (inner, left,
               cross)/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT;
  * planner  - resolves every column reference against the aliases in scope (unknown column and
               ambiguous column are errors), splits WHERE into conjuncts, pushes every conjunct
               that mentions a single relation into that relation's scan, orders the joins
               smallest-relation-first and applies each remaining conjunct at the join where its
               last alias becomes available, then numbers the aggregate calls into slots;
  * executor - operators over dictionaries keyed "alias.column": Scan, Filter, Join (nested
               loop; inner/left/cross), Aggregate (hash groups; COUNT/SUM/AVG/MIN/MAX with SQL
               NULL semantics), Having, Sort, Project, Distinct, Limit.

Everything is checked against a second, deliberately different implementation: `naive_evaluate`
builds the whole cartesian product of the relations, applies the complete WHERE - join conditions
included - with a separately written three-valued evaluator, and then groups, sorts and projects
with plain Python.  A disagreement about pushdown, join ordering, NULL handling or operator order
therefore shows up as a difference between the two result sets.

Verified: lexer token kinds and the lexical errors; parser AST shape (aggregate call, join kinds,
star flag, precedence binding) and the parse errors for a missing FROM, an empty projection, a
dangling operator and trailing input; projection, star expansion and alias naming; WHERE with
AND/OR/NOT, arithmetic precedence and three-valued NULL logic; two- and three-way inner joins,
left join NULL padding, cross join cardinality; all five aggregates with and without GROUP BY and
HAVING; COUNT(*) versus COUNT(col) over NULLs; aggregates over an empty input (COUNT = 0,
SUM = NULL); DISTINCT; multi-key ORDER BY with a DESC key and NULLs last; LIMIT; the complete
query battery matching the oracle row for row; predicate pushdown placing exactly the
single-relation conjuncts below the joins while the un-pushed plan returns the same rows; the
smallest relation chosen as the join's outer side and the mirrored FROM order returning the same
rows; resolution errors (unknown column, ambiguous column, duplicate alias, unknown table,
non-grouped projection, arity); and determinism across repeated execution.

Simplified versus a real engine: no catalogs, indexes, statistics or cost model - the join order
is a deterministic smallest-first rule, not a cost-based decision; no transactions, DDL,
subqueries, window functions, UPDATE/DELETE, or type system beyond Python ints, strings and None;
GROUP BY keys may be expressions but HAVING/ORDER BY see only the select list, the group keys and
the underlying columns; rows are in-memory dictionaries so the data set is tiny.
"""
KEYWORDS = {"SELECT", "DISTINCT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT",
            "JOIN", "INNER", "LEFT", "CROSS", "ON", "AS", "ASC", "DESC", "AND", "OR", "NOT",
            "IS", "NULL", "TRUE", "FALSE", "COUNT", "SUM", "AVG", "MIN", "MAX"}
TWO_CHAR_OPS = ("<=", ">=", "<>", "!=")
ONE_CHAR_OPS = "=<>+-*/(),."
AGGREGATES = ("COUNT", "SUM", "AVG", "MIN", "MAX")


class SqlError(Exception):
    pass


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------- lexer
def tokenize(text):
    tokens = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char.isalpha() or char == "_":
            start = index
            while index < len(text) and (text[index].isalnum() or text[index] == "_"):
                index += 1
            word = text[start:index]
            tokens.append(("keyword", word.upper()) if word.upper() in KEYWORDS
                          else ("ident", word))
            continue
        if char.isdigit():
            start = index
            while index < len(text) and text[index].isdigit():
                index += 1
            tokens.append(("number", int(text[start:index])))
            continue
        if char == "'":
            index += 1
            start = index
            while index < len(text) and text[index] != "'":
                index += 1
            if index >= len(text):
                raise SqlError("unterminated string literal at position %d" % start)
            tokens.append(("string", text[start:index]))
            index += 1
            continue
        if text[index:index + 2] in TWO_CHAR_OPS:
            tokens.append(("op", text[index:index + 2]))
            index += 2
            continue
        if char in ONE_CHAR_OPS:
            tokens.append(("op", char))
            index += 1
            continue
        raise SqlError("unexpected character %r at position %d" % (char, index))
    tokens.append(("eof", None))
    return tokens


# -------------------------------------------------------------------------------- parser
class Parser:
    def __init__(self, text):
        self.tokens = tokenize(text)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def at_keyword(self, word):
        kind, value = self.peek()
        return kind == "keyword" and value == word

    def at_op(self, op):
        kind, value = self.peek()
        return kind == "op" and value == op

    def expect_keyword(self, word):
        if not self.at_keyword(word):
            raise SqlError("expected %s but found %r" % (word, self.peek()[1]))
        self.advance()

    def expect_op(self, op):
        if not self.at_op(op):
            raise SqlError("expected %r but found %r" % (op, self.peek()[1]))
        self.advance()

    # -------------------------------------------------------------------- statement
    def parse_statement(self):
        select = self.parse_select()
        if self.peek()[0] != "eof":
            raise SqlError("unexpected trailing input %r" % (self.peek()[1],))
        return select

    def parse_select(self):
        self.expect_keyword("SELECT")
        distinct = False
        if self.at_keyword("DISTINCT"):
            self.advance()
            distinct = True
        items = [self.parse_select_item()]
        while self.at_op(","):
            self.advance()
            items.append(self.parse_select_item())
        self.expect_keyword("FROM")
        table = self.parse_table_ref()
        tables = [(table, None)]
        while True:
            if self.at_keyword("INNER") or self.at_keyword("JOIN"):
                if self.at_keyword("INNER"):
                    self.advance()
                self.advance()
                target = self.parse_table_ref()
                self.expect_keyword("ON")
                tables.append((target, ("inner", self.parse_expression())))
            elif self.at_keyword("LEFT"):
                self.advance()
                self.expect_keyword("JOIN")
                target = self.parse_table_ref()
                self.expect_keyword("ON")
                tables.append((target, ("left", self.parse_expression())))
            elif self.at_keyword("CROSS"):
                self.advance()
                self.expect_keyword("JOIN")
                tables.append((self.parse_table_ref(), ("cross", None)))
            else:
                break
        where = None
        if self.at_keyword("WHERE"):
            self.advance()
            where = self.parse_expression()
        group_by = []
        if self.at_keyword("GROUP"):
            self.advance()
            self.expect_keyword("BY")
            group_by.append(self.parse_expression())
            while self.at_op(","):
                self.advance()
                group_by.append(self.parse_expression())
        having = None
        if self.at_keyword("HAVING"):
            self.advance()
            having = self.parse_expression()
        order_by = []
        if self.at_keyword("ORDER"):
            self.advance()
            self.expect_keyword("BY")
            order_by.append(self.parse_order_item())
            while self.at_op(","):
                self.advance()
                order_by.append(self.parse_order_item())
        limit = None
        if self.at_keyword("LIMIT"):
            self.advance()
            kind, value = self.advance()
            if kind != "number":
                raise SqlError("LIMIT expects an integer, found %r" % (value,))
            limit = value
        return {"kind": "select", "items": items, "tables": tables, "where": where,
                "group_by": group_by, "having": having, "order_by": order_by, "limit": limit,
                "distinct": distinct, "star": any(item[0] == "star" for item in items)}

    def parse_select_item(self):
        if self.at_op("*"):
            self.advance()
            return ("star", None, None)
        expression = self.parse_expression()
        alias = None
        if self.at_keyword("AS"):
            self.advance()
            kind, value = self.advance()
            if kind != "ident":
                raise SqlError("a column alias must be an identifier, found %r" % (value,))
            alias = value
        return ("item", expression, alias)

    def parse_order_item(self):
        expression = self.parse_expression()
        descending = False
        if self.at_keyword("ASC"):
            self.advance()
        elif self.at_keyword("DESC"):
            self.advance()
            descending = True
        return ("order", expression, descending)

    def parse_table_ref(self):
        kind, value = self.advance()
        if kind != "ident":
            raise SqlError("expected a table name, found %r" % (value,))
        alias = value
        if self.at_keyword("AS"):
            self.advance()
            alias_kind, alias = self.advance()
            if alias_kind != "ident":
                raise SqlError("a table alias must be an identifier, found %r" % (alias,))
        elif self.peek()[0] == "ident":
            alias = self.advance()[1]
        return (value, alias)

    # ------------------------------------------------------------------ expressions
    def parse_expression(self):
        return self.parse_or()

    def parse_or(self):
        node = self.parse_and()
        while self.at_keyword("OR"):
            self.advance()
            node = ("or", node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_not()
        while self.at_keyword("AND"):
            self.advance()
            node = ("and", node, self.parse_not())
        return node

    def parse_not(self):
        if self.at_keyword("NOT"):
            self.advance()
            return ("not", self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self):
        node = self.parse_additive()
        if self.at_keyword("IS"):
            self.advance()
            negated = False
            if self.at_keyword("NOT"):
                self.advance()
                negated = True
            self.expect_keyword("NULL")
            return ("isnull", node, negated)
        kind, value = self.peek()
        if kind == "op" and value in ("=", "<>", "!=", "<", "<=", ">", ">="):
            self.advance()
            return ("cmp", value, node, self.parse_additive())
        return node

    def parse_additive(self):
        node = self.parse_multiplicative()
        while self.at_op("+") or self.at_op("-"):
            operator = self.advance()[1]
            node = ("arith", operator, node, self.parse_multiplicative())
        return node

    def parse_multiplicative(self):
        node = self.parse_primary()
        while self.at_op("*") or self.at_op("/"):
            operator = self.advance()[1]
            node = ("arith", operator, node, self.parse_primary())
        return node

    def parse_primary(self):
        kind, value = self.advance()
        if kind in ("number", "string"):
            return ("lit", value)
        if kind == "op" and value == "(":
            node = self.parse_expression()
            self.expect_op(")")
            return node
        if kind == "op" and value == "-":
            return ("arith", "-", ("lit", 0), self.parse_primary())
        if kind == "keyword" and value in AGGREGATES:
            self.expect_op("(")
            if self.at_op("*"):
                self.advance()
                self.expect_op(")")
                return ("call", value, [("star", None, None)])
            arguments = [self.parse_expression()]
            while self.at_op(","):
                self.advance()
                arguments.append(self.parse_expression())
            self.expect_op(")")
            if len(arguments) != 1:
                raise SqlError("%s takes exactly one argument" % value)
            return ("call", value, arguments)
        if kind == "keyword" and value == "NULL":
            return ("lit", None)
        if kind == "keyword" and value in ("TRUE", "FALSE"):
            return ("lit", value == "TRUE")
        if kind == "ident":
            if self.at_op("."):
                self.advance()
                column_kind, column = self.advance()
                if column_kind != "ident":
                    raise SqlError("expected a column name after '.', found %r" % (column,))
                return ("col", value, column)
            return ("col", None, value)
        raise SqlError("unexpected token %r" % (value,))


def parse(text):
    return Parser(text).parse_statement()


# ------------------------------------------------------------------------ plan operators
class Operator:
    def rows(self):
        raise NotImplementedError

    def describe(self):
        return (type(self).__name__,)


class Scan(Operator):
    def __init__(self, table, alias, rows, columns):
        self.table, self.alias = table, alias
        self.source, self.columns = rows, columns

    def rows(self):
        return [{("%s.%s" % (self.alias, column)): row.get(column) for column in self.columns}
                for row in self.source]

    def describe(self):
        return ("Scan", self.table, self.alias)


class Filter(Operator):
    def __init__(self, child, predicate):
        self.child, self.predicate = child, predicate

    def rows(self):
        return [row for row in self.child.rows() if eval_expr(self.predicate, row) is True]

    def describe(self):
        return ("Filter", describe_expr(self.predicate), self.child.describe())


class Join(Operator):
    def __init__(self, left, right, predicate, kind, right_columns):
        self.left, self.right, self.predicate = left, right, predicate
        self.kind, self.right_columns = kind, right_columns

    def rows(self):
        left_rows, right_rows = self.left.rows(), self.right.rows()
        output = []
        for left_row in left_rows:
            matched = False
            for right_row in right_rows:
                merged = dict(left_row)
                merged.update(right_row)
                if self.predicate is None or eval_expr(self.predicate, merged) is True:
                    output.append(merged)
                    matched = True
            if not matched and self.kind == "left":
                padded = dict(left_row)
                for column in self.right_columns:
                    padded[column] = None
                output.append(padded)
        return output

    def describe(self):
        return ("Join", self.kind, self.left.describe(), self.right.describe())


class Aggregate(Operator):
    def __init__(self, child, group_exprs, calls):
        self.child, self.group_exprs, self.calls = child, group_exprs, calls

    def rows(self):
        groups, order = {}, []
        for row in self.child.rows():
            key = tuple(eval_expr(expr, row) for expr in self.group_exprs)
            if key not in groups:
                groups[key] = (row, [[] for _ in self.calls])
                order.append(key)
            accumulator = groups[key][1]
            for slot, call in enumerate(self.calls):
                if call[1] == "COUNT" and call[2][0][0] == "star":
                    accumulator[slot].append(1)
                else:
                    accumulator[slot].append(eval_expr(call[2][0], row))
        output = []
        if not order and not self.group_exprs:
            order, groups[(None,)] = [(None,)], ({}, [[] for _ in self.calls])
        for key in order:
            sample, accumulator = groups[key]
            row = dict(sample) if self.group_exprs else {}
            for position, expression in enumerate(self.group_exprs):
                row["group%d" % position] = key[position]
                if expression[0] == "col":
                    row["%s.%s" % (expression[1], expression[2])] = key[position]
            for slot, call in enumerate(self.calls):
                row["agg%d" % slot] = finalize(call[1], accumulator[slot])
            output.append(row)
        return output

    def describe(self):
        return ("Aggregate", len(self.calls), self.child.describe())


class Sort(Operator):
    def __init__(self, child, keys):
        self.child, self.keys = child, keys

    def rows(self):
        output = self.child.rows()
        for expression, descending in reversed(self.keys):   # stable passes, least significant first
            output.sort(key=lambda row: sort_key(eval_expr(expression, row), descending),
                        reverse=descending)
        return output

    def describe(self):
        return ("Sort", len(self.keys), self.child.describe())


class Project(Operator):
    def __init__(self, child, items, names):
        self.child, self.items, self.names = child, items, names

    def rows(self):
        return [{name: eval_expr(expr, row) for name, expr in zip(self.names, self.items)}
                for row in self.child.rows()]

    def describe(self):
        return ("Project", tuple(self.names), self.child.describe())


class Distinct(Operator):
    def __init__(self, child, names):
        self.child, self.names = child, names

    def rows(self):
        seen, output = set(), []
        for row in self.child.rows():
            key = tuple(row[name] for name in self.names)
            if key not in seen:
                seen.add(key)
                output.append(row)
        return output

    def describe(self):
        return ("Distinct", self.child.describe())


class Limit(Operator):
    def __init__(self, child, count):
        self.child, self.count = child, count

    def rows(self):
        return self.child.rows()[:self.count]

    def describe(self):
        return ("Limit", self.count, self.child.describe())


def sort_key(value, descending=False):
    """NULLs last in both directions: their rank stays below every real value."""
    if value is None:
        return (0, 0) if descending else (1, 0)
    return (1, value) if descending else (0, value)


# -------------------------------------------------------------------------------- runtime
def finalize(name, values):
    present = [value for value in values if value is not None]
    if name == "COUNT":
        return len(present)
    if not present:
        return None
    if name == "SUM":
        return sum(present)
    if name == "AVG":
        return sum(present) / len(present)
    if name == "MIN":
        return min(present)
    return max(present)


def eval_expr(expr, row):
    kind = expr[0]
    if kind == "lit":
        return expr[1]
    if kind == "col":
        return row.get("%s.%s" % (expr[1], expr[2]))
    if kind == "slot":
        return row.get(expr[1])
    if kind == "not":
        value = eval_expr(expr[1], row)
        return None if value is None else not value
    if kind == "isnull":
        return (eval_expr(expr[1], row) is not None) if expr[2] else (eval_expr(expr[1], row)
                                                                     is None)
    if kind in ("and", "or"):
        left, right = eval_expr(expr[1], row), eval_expr(expr[2], row)
        values = (left, right)
        if kind == "and":
            if False in values:
                return False
            return None if None in values else True
        if True in values:
            return True
        return None if None in values else False
    if kind == "cmp":
        return compare(expr[1], eval_expr(expr[2], row), eval_expr(expr[3], row))
    if kind == "arith":
        left, right = eval_expr(expr[2], row), eval_expr(expr[3], row)
        if left is None or right is None:
            return None
        if expr[1] == "+":
            return left + right
        if expr[1] == "-":
            return left - right
        if expr[1] == "*":
            return left * right
        return left / right if right else None
    raise SqlError("cannot evaluate %r" % (expr,))


def compare(operator, left, right):
    if left is None or right is None:
        return None
    if operator == "=":
        return left == right
    if operator in ("<>", "!="):
        return left != right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    return left >= right


def describe_expr(expr):
    """Stable display name for an expression; also the default output column name."""
    tag = expr[0]
    if tag == "col":
        return expr[2] if expr[1] is None else "%s.%s" % (expr[1], expr[2])
    if tag == "lit":
        return repr(expr[1])
    if tag == "slot":
        return expr[1]
    if tag == "star":
        return "*"
    if tag in ("cmp", "arith"):
        return "(%s %s %s)" % (describe_expr(expr[2]), expr[1], describe_expr(expr[3]))
    if tag in ("and", "or"):
        return "(%s %s %s)" % (describe_expr(expr[1]), tag.upper(), describe_expr(expr[2]))
    if tag == "not":
        return "NOT %s" % describe_expr(expr[1])
    if tag == "isnull":
        return "(%s IS%s NULL)" % (describe_expr(expr[1]), " NOT" if expr[2] else "")
    if tag == "call":
        return "%s(%s)" % (expr[1], describe_expr(expr[2][0]))
    return "?"


# ------------------------------------------------------------------------------- planner
def relation_specs(select, catalog):
    specs = []
    for position, (table_ref, join) in enumerate(select["tables"]):
        table, alias = table_ref
        if table not in catalog:
            raise SqlError("unknown table %r" % table)
        columns = list(catalog[table][0].keys()) if catalog[table] else []
        if any(existing[0] == alias for existing in specs):
            raise SqlError("duplicate table alias %r" % alias)
        kind = "inner" if position == 0 else join[0]     # the base relation carries no join kind
        specs.append((alias, table, columns, kind, None if position == 0 else join[1]))
    return specs


def conjuncts(expr):
    if expr is None:
        return []
    if expr[0] == "and":
        return conjuncts(expr[1]) + conjuncts(expr[2])
    return [expr]


def aliases_in(expr, known):
    found = set()
    if isinstance(expr, tuple):
        if expr[0] == "col" and expr[1] is not None and expr[1] in known:
            found.add(expr[1])
        for item in expr[1:]:
            found |= aliases_in(item, known)
    elif isinstance(expr, list):
        for element in expr:
            found |= aliases_in(element, known)
    return found


def resolve(expr, scope):
    """Rewrite and validate column references; unknown or ambiguous references are errors."""
    if not isinstance(expr, tuple):
        return expr
    tag = expr[0]
    if tag in ("lit", "star", "slot"):
        return expr
    if tag == "col":
        _, alias, column = expr
        if alias is not None:
            if alias not in scope or column not in scope[alias]:
                raise SqlError("unknown column %s.%s" % (alias, column))
            return expr
        candidates = [candidate for candidate in sorted(scope) if column in scope[candidate]]
        if not candidates:
            raise SqlError("unknown column %r" % column)
        if len(candidates) > 1:
            raise SqlError("ambiguous column %r: matches %s" % (column, ", ".join(candidates)))
        return ("col", candidates[0], column)
    if tag == "call":
        return ("call", expr[1], [argument if argument[0] == "star" else resolve(argument, scope)
                                  for argument in expr[2]])
    if tag in ("and", "or"):
        return (tag, resolve(expr[1], scope), resolve(expr[2], scope))
    if tag == "not":
        return ("not", resolve(expr[1], scope))
    if tag == "isnull":
        return ("isnull", resolve(expr[1], scope), expr[2])
    if tag == "cmp":
        return ("cmp", expr[1], resolve(expr[2], scope), resolve(expr[3], scope))
    if tag == "arith":
        return ("arith", expr[1], resolve(expr[2], scope), resolve(expr[3], scope))
    raise SqlError("unsupported expression %r" % (expr,))


def collect_calls(expr, found):
    if isinstance(expr, tuple):
        if expr[0] == "call":
            if expr not in found:
                found.append(expr)
            return
        for item in expr[1:]:
            collect_calls(item, found)
    elif isinstance(expr, list):
        for element in expr:
            collect_calls(element, found)


def rewrite_calls(expr, calls):
    if isinstance(expr, tuple):
        if expr[0] == "call":
            return ("slot", "agg%d" % calls.index(expr))
        return tuple(rewrite_calls(item, calls) if isinstance(item, tuple) else item
                     for item in expr)
    return expr


def plan(select, catalog, pushdown=True):
    specs = relation_specs(select, catalog)
    scope = {alias: columns for alias, _, columns, _, _ in specs}
    all_aliases = set(scope)
    left_aliases = {alias for alias, _, _, kind, _ in specs if kind == "left"}
    where = resolve(select["where"], scope) if select["where"] else None
    pushed = {alias: [] for alias in scope}
    join_filters = []
    for conjunct in conjuncts(where):
        used = aliases_in(conjunct, all_aliases)
        if pushdown and len(used) == 1 and not (used & left_aliases):
            pushed[next(iter(used))].append(conjunct)
        else:
            join_filters.append(conjunct)

    scans = {}
    for alias, table, columns, _kind, _on in specs:
        node = Scan(table, alias, catalog[table], columns)
        if pushed[alias]:
            predicate = pushed[alias][0]
            for extra in pushed[alias][1:]:
                predicate = ("and", predicate, extra)
            node = Filter(node, predicate)
        scans[alias] = node

    kinds = {alias: kind for alias, _, _, kind, _ in specs}
    ons = {alias: on for alias, _, _, _, on in specs}
    left_aliases = {alias for alias, kind in kinds.items() if kind == "left"}
    fixed_ons, on_pool = {}, []
    for alias, _, _, kind, on in specs:
        if on is None:
            continue
        resolved_on = resolve(on, scope)
        if kind == "left":
            fixed_ons[alias] = resolved_on
        else:
            on_pool.append(resolved_on)      # reordering may move it to another join
    if left_aliases:
        order = list(specs)                  # a left join fixes the join order
    else:
        order = sorted(specs, key=lambda spec: (len(catalog[spec[1]]), spec[0]))
    node = scans[order[0][0]]
    joined = {order[0][0]}
    pending_conjuncts = join_filters + on_pool
    for alias, table, columns, kind, _on in order[1:]:
        ready, pending = [], []
        for conjunct in pending_conjuncts:
            (ready if aliases_in(conjunct, all_aliases) <= (joined | {alias}) else pending
             ).append(conjunct)
        pending_conjuncts = pending
        predicate = fixed_ons.get(alias)
        if kind == "left":
            node = Join(node, scans[alias], predicate, kind,
                        ["%s.%s" % (alias, column) for column in columns])
            for conjunct in ready:           # a left join keeps its padded rows, so filter above
                node = Filter(node, conjunct)
        else:
            for conjunct in ready:           # safe to fold into the join for inner and cross
                predicate = conjunct if predicate is None else ("and", predicate, conjunct)
            node = Join(node, scans[alias], predicate, kind,
                        ["%s.%s" % (alias, column) for column in columns])
        joined.add(alias)
    for conjunct in pending_conjuncts:
        node = Filter(node, conjunct)

    items, names = [], []
    if select["star"]:
        for alias, _, columns, _, _ in specs:
            for column in columns:
                items.append(("col", alias, column))
                names.append("%s.%s" % (alias, column))
    for kind, expression, alias_name in select["items"]:
        if kind == "star":
            continue
        items.append(resolve(expression, scope))
        names.append(alias_name or describe_expr(expression))

    group_exprs = [resolve(expression, scope) for expression in select["group_by"]]
    having_expr = resolve(select["having"], scope) if select["having"] else None
    alias_to_expression = dict(zip(names, items))
    keys = []
    for _kind, expression, descending in select["order_by"]:
        if (expression[0] == "col" and expression[1] is None
                and expression[2] in alias_to_expression):
            keys.append((alias_to_expression[expression[2]], descending))
        else:
            keys.append((resolve(expression, scope), descending))

    if group_exprs:
        for item in select["items"]:
            if item[0] == "star":
                continue
            resolved = resolve(item[1], scope)
            if resolved[0] == "call" or resolved in group_exprs:
                continue
            raise SqlError("column %s must appear in GROUP BY or be aggregated"
                           % describe_expr(resolved))

    calls = []
    for expression in list(items) + ([having_expr] if having_expr is not None else []) \
            + [key for key, _descending in keys]:
        collect_calls(expression, calls)

    if calls or group_exprs:
        node = Aggregate(node, group_exprs, calls)
        if having_expr is not None:
            node = Filter(node, rewrite_calls(having_expr, calls))
        items = [rewrite_calls(item, calls) for item in items]
        keys = [(rewrite_calls(key, calls), descending) for key, descending in keys]
    if keys:
        node = Sort(node, keys)
    node = Project(node, items, names)
    if select["distinct"]:
        node = Distinct(node, names)
    if select["limit"] is not None:
        node = Limit(node, select["limit"])
    return node


def query(text, catalog, pushdown=True):
    return plan(parse(text), catalog, pushdown).rows()


# --------------------------------------------------- deliberately naive oracle engine
def naive_eval(expr, row):
    """Second, independent three-valued evaluator used by the cartesian-product oracle."""
    tag = expr[0]
    if tag == "lit":
        return expr[1]
    if tag == "col":
        return row.get("%s.%s" % (expr[1], expr[2]))
    if tag == "slot":
        return row.get(expr[1])
    if tag == "not":
        value = naive_eval(expr[1], row)
        return None if value is None else (not value)
    if tag == "isnull":
        value = naive_eval(expr[1], row)
        return (value is not None) if expr[2] else (value is None)
    if tag in ("and", "or"):
        values = [naive_eval(expr[1], row), naive_eval(expr[2], row)]
        if tag == "and":
            if False in values:
                return False
            return None if None in values else True
        if True in values:
            return True
        return None if None in values else False
    if tag == "cmp":
        left, right = naive_eval(expr[2], row), naive_eval(expr[3], row)
        if left is None or right is None:
            return None
        if expr[1] == "=":
            return left == right
        if expr[1] in ("<>", "!="):
            return left != right
        if expr[1] == "<":
            return left < right
        if expr[1] == "<=":
            return left <= right
        if expr[1] == ">":
            return left > right
        return left >= right
    if tag == "arith":
        left, right = naive_eval(expr[2], row), naive_eval(expr[3], row)
        if left is None or right is None:
            return None
        if expr[1] == "+":
            return left + right
        if expr[1] == "-":
            return left - right
        if expr[1] == "*":
            return left * right
        return left / right if right else None
    raise SqlError("oracle cannot evaluate %r" % (expr,))


def alias_columns(select, catalog):
    scope = {}
    for table_ref, _join in select["tables"]:
        table, alias = table_ref
        scope[alias] = list(catalog[table][0].keys()) if catalog[table] else []
    return scope


def naive_evaluate(text, catalog):
    """Brute force: full cartesian product, whole predicate at once, plain-Python grouping."""
    select = parse(text)
    scope = alias_columns(select, catalog)
    merged = [{}]
    for position, (table_ref, join) in enumerate(select["tables"]):
        table, alias = table_ref
        kind, on = ("cross", None) if position == 0 else join
        resolved = resolve(on, scope) if on is not None else None
        extension = []
        for base in merged:
            matches = []
            for source in catalog[table]:
                row = dict(base)
                for column in catalog[table][0]:
                    row["%s.%s" % (alias, column)] = source.get(column)
                if resolved is None or naive_eval(resolved, row) is True:
                    matches.append(row)
            if matches:
                extension.extend(matches)
            elif kind == "left":
                padded = dict(base)
                for column in catalog[table][0]:
                    padded["%s.%s" % (alias, column)] = None
                extension.append(padded)
        merged = extension
    conditions = [resolve(condition, scope) for condition in conjuncts(select["where"])]
    rows = [row for row in merged
            if all(naive_eval(condition, row) is True for condition in conditions)]

    items, names = [], []
    if select["star"]:
        for table_ref, _join in select["tables"]:
            table, alias = table_ref
            for column in catalog[table][0]:
                items.append(("col", alias, column))
                names.append("%s.%s" % (alias, column))
    for kind, expression, alias_name in select["items"]:
        if kind == "star":
            continue
        items.append(resolve(expression, scope))
        names.append(alias_name or describe_expr(expression))

    calls = []
    having_expr = resolve(select["having"], scope) if select["having"] else None
    alias_to_expression = dict(zip(names, items))
    keys = []
    for _kind, expression, descending in select["order_by"]:
        if (expression[0] == "col" and expression[1] is None
                and expression[2] in alias_to_expression):
            keys.append((alias_to_expression[expression[2]], descending))
        else:
            keys.append((resolve(expression, scope), descending))
    for expression in list(items) + ([having_expr] if having_expr is not None else []) \
            + [key for key, _descending in keys]:
        collect_calls(expression, calls)

    if calls or select["group_by"]:
        group_exprs = [resolve(expression, scope) for expression in select["group_by"]]
        buckets, order = {}, []
        for row in rows:
            key = tuple(naive_eval(expr, row) for expr in group_exprs)
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(row)
        if not order and not group_exprs:
            order = [(None,)]
            buckets[(None,)] = []
        materialized = []
        for key in order:
            bucket = buckets[key]
            row = {}
            for position, expr in enumerate(group_exprs):
                row["group%d" % position] = key[position]
                if expr[0] == "col":
                    row["%s.%s" % (expr[1], expr[2])] = key[position]
            for slot, call in enumerate(calls):
                if call[1] == "COUNT" and call[2][0][0] == "star":
                    values = [1 for _ in bucket]
                else:
                    values = [naive_eval(call[2][0], item) for item in bucket]
                row["agg%d" % slot] = finalize(call[1], values)
            materialized.append(row)
        rows = materialized
        if having_expr is not None:
            having = rewrite_calls(having_expr, calls)
            rows = [row for row in rows if naive_eval(having, row) is True]
        items = [rewrite_calls(item, calls) for item in items]
        keys = [(rewrite_calls(key, calls), descending) for key, descending in keys]

    for key_expr, descending in reversed(keys):
        rows.sort(key=lambda row: sort_key(naive_eval(key_expr, row), descending),
                  reverse=descending)
    output = [{name: naive_eval(expr, row) for name, expr in zip(names, items)} for row in rows]
    if select["distinct"]:
        seen, unique = set(), []
        for row in output:
            key = tuple(row[name] for name in names)
            if key not in seen:
                seen.add(key)
                unique.append(row)
        output = unique
    if select["limit"] is not None:
        output = output[:select["limit"]]
    return output


# ---------------------------------------------------------------------------------- data
CATALOG = {
    "dept": [
        {"id": 1, "name": "eng", "region": "eu"},
        {"id": 2, "name": "sales", "region": "us"},
        {"id": 3, "name": "ops", "region": "us"},
        {"id": 4, "name": "lab", "region": "apac"},
    ],
    "emp": [
        {"id": 1, "name": "ana", "dept_id": 1, "salary": 120, "mgr": None},
        {"id": 2, "name": "bo", "dept_id": 1, "salary": 90, "mgr": 1},
        {"id": 3, "name": "cy", "dept_id": 2, "salary": 110, "mgr": 1},
        {"id": 4, "name": "di", "dept_id": 2, "salary": 70, "mgr": 3},
        {"id": 5, "name": "ed", "dept_id": 3, "salary": 70, "mgr": None},
        {"id": 6, "name": "fi", "dept_id": None, "salary": 200, "mgr": 1},
    ],
    "project": [
        {"id": 10, "dept_id": 1, "budget": 500},
        {"id": 11, "dept_id": 2, "budget": 300},
        {"id": 12, "dept_id": 1, "budget": 150},
        {"id": 13, "dept_id": 5, "budget": 900},
    ],
}
QUERIES = [
    "SELECT name, salary FROM emp WHERE salary > 100",
    "SELECT * FROM dept WHERE region = 'us'",
    "SELECT e.name, d.name FROM emp AS e JOIN dept AS d ON e.dept_id = d.id",
    "SELECT e.name, d.name, p.id FROM emp AS e JOIN dept AS d ON e.dept_id = d.id "
    "JOIN project AS p ON p.dept_id = d.id",
    "SELECT e.name, d.region FROM emp AS e LEFT JOIN dept AS d ON e.dept_id = d.id",
    "SELECT COUNT(*), COUNT(mgr), SUM(salary), AVG(salary), MIN(salary), MAX(salary) FROM emp",
    "SELECT dept_id, COUNT(*), SUM(salary) FROM emp GROUP BY dept_id HAVING COUNT(*) > 1",
    "SELECT region, COUNT(*) FROM emp AS e JOIN dept AS d ON e.dept_id = d.id GROUP BY region",
    "SELECT DISTINCT region FROM dept",
    "SELECT name, salary FROM emp ORDER BY salary DESC, name ASC LIMIT 3",
    "SELECT dept_id, SUM(salary) AS total FROM emp WHERE salary IS NOT NULL GROUP BY dept_id "
    "ORDER BY total DESC",
    "SELECT e.name, d.name FROM emp AS e CROSS JOIN dept AS d WHERE e.dept_id = d.id",
    "SELECT name FROM emp WHERE dept_id = NULL",
    "SELECT name FROM emp WHERE mgr IS NULL",
    "SELECT region, COUNT(*) FROM dept GROUP BY region HAVING SUM(id) >= 3 ORDER BY region",
    "SELECT name FROM emp WHERE salary > 50 + 60 AND NOT (mgr IS NULL)",
]


def as_tuples(rows, names, ordered=False):
    """Tuples for comparison; unordered results are compared as multisets (SQL has no order)."""
    values = [tuple(row[name] for name in names) for row in rows]
    if ordered:
        return values
    return sorted(values, key=lambda item: [(value is None, 0 if value is None else value)
                                            for value in item])


def main():
    # --- 1. lexer ------------------------------------------------------------------
    tokens = tokenize("SELECT a.b, 12, 'xy' FROM t WHERE c >= -3")
    kinds = ["keyword", "ident", "op", "ident", "op", "number", "op", "string", "keyword",
             "ident", "keyword", "ident", "op", "op", "number", "eof"]
    check([token[0] for token in tokens] == kinds,
          "token kinds of a sample statement: %s" % [token[0] for token in tokens])
    check(tokens[0][1] == "SELECT" and tokens[2][1] == "." and tokens[7][1] == "xy"
          and tokens[13][1] == "-",
          "keyword folding, punctuation and string literals must be exact")
    for bad, expected in (("SELECT 'abc FROM t", "unterminated"),
                          ("SELECT # FROM t", "unexpected character"),
                          ("SELECT a FROM t LIMIT x", "LIMIT expects")):
        raised = None
        try:
            parse(bad)
        except SqlError as error:
            raised = str(error)
        check(raised is not None and expected in raised,
              "error for %r must mention %r, got %r" % (bad, expected, raised))

    # --- 2. parser -----------------------------------------------------------------
    ast = parse("SELECT dept_id, SUM(salary) AS total FROM emp WHERE salary > 10 "
                "GROUP BY dept_id HAVING SUM(salary) > 100 ORDER BY total DESC LIMIT 5")
    check(ast["kind"] == "select" and ast["tables"][0] == (("emp", "emp"), None),
          "the AST must record the FROM relation: %r" % (ast["tables"],))
    check(ast["items"][1] == ("item", ("call", "SUM", [("col", None, "salary")]), "total"),
          "an aliased aggregate must parse into a call node, got %r" % (ast["items"][1],))
    check(ast["where"] == ("cmp", ">", ("col", None, "salary"), ("lit", 10))
          and ast["order_by"] == [("order", ("col", None, "total"), True)]
          and ast["limit"] == 5 and len(ast["group_by"]) == 1 and ast["having"] is not None,
          "WHERE, GROUP BY, HAVING, ORDER BY and LIMIT must all be captured")
    joined = parse("SELECT * FROM a JOIN b ON a.x = b.y LEFT JOIN c ON b.z = c.w")
    check([spec[1][0] if spec[1] else None for spec in joined["tables"]] == [None, "inner", "left"]
          and joined["star"],
          "join kinds and the star flag must be recorded: %s" % (joined["tables"],))
    check(parse("SELECT a FROM t CROSS JOIN u")["tables"][1][1][0] == "cross",
          "CROSS JOIN takes no ON clause")
    precedence = parse("SELECT a FROM t WHERE 1 + 2 * 3 = 7 OR NOT x = 4 AND y = 5")["where"]
    check(precedence == ("or",
                         ("cmp", "=", ("arith", "+", ("lit", 1),
                                       ("arith", "*", ("lit", 2), ("lit", 3))), ("lit", 7)),
                         ("and", ("not", ("cmp", "=", ("col", None, "x"), ("lit", 4))),
                          ("cmp", "=", ("col", None, "y"), ("lit", 5)))),
          "the precedence ladder must bind as expected, got %r" % (precedence,))
    for bad, expected in (("SELECT a", "expected FROM"),
                          ("SELECT FROM t", "unexpected token"),
                          ("SELECT a FROM", "expected a table name"),
                          ("SELECT a FROM t WHERE", "unexpected token"),
                          ("SELECT a FROM t LIMIT 1 garbage", "trailing")):
        raised = None
        try:
            parse(bad)
        except SqlError as error:
            raised = str(error)
        check(raised is not None and expected.lower() in raised.lower(),
              "parse error for %r must mention %r, got %r" % (bad, expected, raised))

    # --- 3. executor versus the cartesian-product oracle ----------------------------
    mismatches = []
    for text in QUERIES:
        ordered = bool(parse(text)["order_by"])
        engine = plan(parse(text), CATALOG).rows()
        reference = naive_evaluate(text, CATALOG)
        names = list(engine[0].keys()) if engine else (list(reference[0].keys())
                                                       if reference else [])
        if as_tuples(engine, names, ordered) != as_tuples(reference, names, ordered):
            mismatches.append((text, engine, reference))
    check(not mismatches, "engine and oracle must agree on every query: %s"
          % [(item[0], len(item[1]), len(item[2])) for item in mismatches])

    # --- 4. hand-checked results ----------------------------------------------------
    rows = query("SELECT e.name, d.name FROM emp AS e JOIN dept AS d ON e.dept_id = d.id "
                 "WHERE d.region = 'us' ORDER BY e.name", CATALOG)
    check([tuple(row.values()) for row in rows]
          == [("cy", "sales"), ("di", "sales"), ("ed", "ops")],
          "the joined and filtered result must be exact: %s" % (rows,))
    cross = query("SELECT e.id, d.id FROM emp AS e CROSS JOIN dept AS d", CATALOG)
    check(len(cross) == len(CATALOG["emp"]) * len(CATALOG["dept"]),
          "a cross join must produce the full product, got %d" % len(cross))
    left = query("SELECT e.name, d.region FROM emp AS e LEFT JOIN dept AS d "
                 "ON e.dept_id = d.id ORDER BY e.name", CATALOG)
    check(len(left) == len(CATALOG["emp"]) and left[-1]["d.region"] is None,
          "a left join must keep the unmatched row with NULL padding: %s" % (left,))
    totals = query("SELECT dept_id, COUNT(*) AS n, SUM(salary) AS total FROM emp "
                   "GROUP BY dept_id ORDER BY dept_id", CATALOG)
    check([tuple(row.values()) for row in totals]
          == [(1, 2, 210), (2, 2, 180), (3, 1, 70), (None, 1, 200)],
          "grouped aggregates must be exact: %s" % (totals,))
    check(list(query("SELECT COUNT(*) FROM emp", CATALOG)[0].keys()) == ["COUNT(*)"],
          "an aggregate without an alias must be named after its call")
    check(query("SELECT COUNT(*) FROM emp WHERE salary > 1000", CATALOG) == [{"COUNT(*)": 0}],
          "COUNT over an empty input must be zero")
    check(query("SELECT SUM(salary) FROM emp WHERE salary > 1000", CATALOG)
          == [{"SUM(salary)": None}], "SUM over an empty input must be NULL, not zero")
    check(query("SELECT MIN(salary), MAX(salary) FROM emp WHERE salary > 1000", CATALOG)
          == [{"MIN(salary)": None, "MAX(salary)": None}],
          "MIN and MAX over an empty input must be NULL")
    check(query("SELECT name FROM emp WHERE dept_id = NULL", CATALOG) == []
          and len(query("SELECT name FROM emp WHERE dept_id IS NULL", CATALOG)) == 1,
          "a NULL comparison must not match while IS NULL must")
    check(query("SELECT COUNT(*) AS a, COUNT(mgr) AS b FROM emp", CATALOG)
          == [{"a": 6, "b": 4}], "COUNT(*) must count rows while COUNT(col) skips NULLs")
    check(query("SELECT AVG(salary) AS average FROM emp WHERE dept_id = 1", CATALOG)
          == [{"average": 105.0}], "AVG must divide by the number of non-NULL values")
    distinct = query("SELECT DISTINCT region FROM dept ORDER BY region", CATALOG)
    check([row["region"] for row in distinct] == ["apac", "eu", "us"],
          "DISTINCT must collapse duplicate regions: %s" % (distinct,))
    ordered = query("SELECT name FROM emp ORDER BY salary DESC, name ASC LIMIT 4", CATALOG)
    check([row["name"] for row in ordered] == ["fi", "ana", "cy", "bo"],
          "multi-key ORDER BY with a tie break must be exact: %s" % (ordered,))
    check(query("SELECT name, dept_id FROM emp ORDER BY dept_id", CATALOG)[-1]["dept_id"] is None,
          "NULLs must sort last in an ascending order")
    check(query("SELECT name, dept_id FROM emp ORDER BY dept_id DESC", CATALOG)[-1]["dept_id"]
          is None, "NULLs must sort last in a descending order too")
    check(query("SELECT name FROM emp WHERE salary > 50 + 60 AND NOT (mgr IS NULL)", CATALOG)
          == query("SELECT name FROM emp WHERE salary > 110 AND mgr IS NOT NULL", CATALOG),
          "arithmetic precedence and IS NOT NULL must agree with the explicit form")
    star = query("SELECT * FROM dept WHERE id = 1", CATALOG)
    check(list(star[0].keys()) == ["dept.id", "dept.name", "dept.region"]
          and star[0]["dept.name"] == "eng", "star expansion must expose every column: %s" % star)

    # --- 5. planner: pushdown and join order ---------------------------------------
    text = ("SELECT e.name, d.region FROM emp AS e JOIN dept AS d ON e.dept_id = d.id "
            "WHERE e.salary > 80 AND d.region = 'us' AND e.mgr < d.id")
    pushed = plan(parse(text), CATALOG, pushdown=True)
    flat = plan(parse(text), CATALOG, pushdown=False)
    check(repr(pushed.describe()).count("Filter") == 2,
          "pushdown must place exactly the two single-relation filters below the join: %s"
          % (pushed.describe(),))
    check("Filter" not in repr(flat.describe()),
          "without pushdown the whole predicate lives in the join: %s" % (flat.describe(),))
    check(len(pushed.describe()) == 3 and pushed.describe()[0] == "Project",
          "the top operator must be the projection: %s" % (pushed.describe(),))
    check(as_tuples(pushed.rows(), ["e.name", "d.region"])
          == as_tuples(flat.rows(), ["e.name", "d.region"]),
          "pushdown must not change the result")
    check(as_tuples(pushed.rows(), ["e.name", "d.region"])
          == as_tuples(naive_evaluate(text, CATALOG), ["e.name", "d.region"]),
          "the pushed plan must still match the oracle")
    inner = pushed.child
    check(inner.describe()[0] == "Join" and inner.describe()[1] == "inner"
          and "dept" in repr(inner.describe()[2]),
          "the four-row relation must be the outer side of the join: %s" % (inner.describe(),))
    mirrored = query("SELECT e.name, d.region FROM dept AS d JOIN emp AS e ON e.dept_id = d.id "
                     "WHERE e.salary > 80 AND d.region = 'us' AND e.mgr < d.id "
                     "ORDER BY e.name", CATALOG)
    check(as_tuples(mirrored, ["e.name", "d.region"])
          == as_tuples(pushed.rows(), ["e.name", "d.region"]),
          "mirroring the FROM order must return the same rows")
    three_way = plan(parse(QUERIES[3]), CATALOG)
    check(repr(three_way.describe()).count("Join") == 2,
          "a three-way join must produce two join operators: %s" % (three_way.describe(),))

    # --- 6. resolution errors -------------------------------------------------------
    for bad, expected in (("SELECT nope FROM emp", "unknown column"),
                          ("SELECT e.name FROM emp AS e JOIN dept AS d ON e.dept_id = d.id "
                           "WHERE id > 1", "ambiguous column"),
                          ("SELECT e.nope FROM emp AS e", "unknown column"),
                          ("SELECT name FROM emp AS e JOIN emp AS e ON e.id = e.id",
                           "duplicate table alias"),
                          ("SELECT salary, name FROM emp GROUP BY salary", "GROUP BY"),
                          ("SELECT x FROM missing", "unknown table"),
                          ("SELECT COUNT(salary, mgr) FROM emp", "exactly one argument")):
        raised = None
        try:
            query(bad, CATALOG)
        except SqlError as error:
            raised = str(error)
        check(raised is not None and expected.lower() in raised.lower(),
              "resolution error for %r must mention %r, got %r" % (bad, expected, raised))

    # --- 7. determinism -------------------------------------------------------------
    for text in QUERIES[:6]:
        check(plan(parse(text), CATALOG).rows() == plan(parse(text), CATALOG).rows(),
              "repeated execution must be identical for %r" % text)

    print("engine/oracle agreement on %d queries; pushed plan: %s"
          % (len(QUERIES), pushed.describe()))
    print("catalog: emp %d rows, dept %d rows, project %d rows; cross product %d"
          % (len(CATALOG["emp"]), len(CATALOG["dept"]), len(CATALOG["project"]),
             len(CATALOG["emp"]) * len(CATALOG["dept"])))
    print("grouped totals: %s" % ([(row["dept_id"], row["n"], row["total"]) for row in totals],))
    print("PASS")


if __name__ == "__main__":
    main()
