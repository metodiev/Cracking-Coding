/** Task 803 - Build a small SQL engine with parser, planner, and executor supporting joins and aggregates. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Three stages. THE PARSER: a hand-written lexer plus recursive-descent parser for
 * SELECT [DISTINCT] list FROM rel { [INNER|LEFT] JOIN rel ON pred } [WHERE pred] [GROUP BY ...]
 * [HAVING pred] [ORDER BY ...] [LIMIT n], building an expression AST with SQL three-valued logic
 * (TRUE / FALSE / UNKNOWN, only TRUE keeps a row), IS [NOT] NULL, integer and text literals, and the
 * aggregates COUNT(*), COUNT(x), SUM(x), MIN(x), MAX(x), AVG(x). THE PLANNER: the WHERE clause is
 * split into conjuncts and a conjunct referencing exactly one relation is pushed down to a Filter
 * directly above that relation - but only when every join after that relation is an inner join,
 * because a predicate on the null-supplying side of an outer join may not be pushed below it; an
 * equi-join predicate becomes a HashJoin, anything else a nested-loop join, and the planner costs
 * both and records the estimates it compared. THE EXECUTOR: materialized operators - hash join with
 * a probe table, hash aggregation, sort (NULLs last), distinct, limit - over rows of Object values
 * with SQL NULL semantics: NULL is never equal to anything, comparisons with NULL are UNKNOWN,
 * aggregate functions skip NULLs while COUNT(*) counts rows, and an empty input to a key-less group
 * still produces one row.
 *
 * Simulation model, and what is simplified. One process, in-memory row lists: no storage engine, no
 * index or B+tree, no transactions, no concurrent query execution, no cardinality statistics beyond
 * row counts, no subqueries, UNION, right/full outer joins or CROSS JOIN, and operators materialize
 * their input instead of streaming. Errors are exceptions rather than partial results. The parser,
 * planner and executor are real: the plan is a tree, pushdown and hash join are decided by the
 * planner and are *measurable* in the work the executor does, and grouping really is a hash table
 * keyed by the group keys.
 *
 * Verified: projection and column typing; WHERE with AND/OR/NOT and three-valued NULL logic; IS NULL
 * and IS NOT NULL; INNER, LEFT and three-way joins; GROUP BY with HAVING, COUNT(*), COUNT(x), SUM,
 * MIN, MAX and AVG including the empty-group result (COUNT(*) = 0, SUM = NULL); DISTINCT, ORDER BY
 * ASC/DESC with NULLs last, LIMIT; an empty result set. Every one of those queries is cross-checked
 * against a brute-force reference that ignores the planner completely (FROM-order nested loops over
 * the full cross product, no pushdown, sort-based grouping), and against the same plan with the
 * optimizer switched off. The planner itself is checked: a HashJoin must appear for an equi-join, a
 * pushdown Filter must sit on a bare Scan, the filter must NOT be pushed under a LEFT JOIN, the
 * pushdown must reduce the number of rows entering the join, and the estimated cost must explain the
 * algorithm choice. Invalid queries (unknown column, unknown table, malformed statement, non-boolean
 * predicate, incomparable comparison) must be rejected. Finally a 120 x 60 join checks that the
 * build side is scanned once per query rather than once per probe row.
 */
import java.util.*;

// ------------------------------------------------------------------ expressions

abstract class Expr { }

final class Lit extends Expr {
    final Object value;                          // null means SQL NULL
    Lit(Object value) { this.value = value; }
}

final class Col extends Expr {
    final String qualifier, name;
    Col(String qualifier, String name) { this.qualifier = qualifier; this.name = name; }
}

final class IsNull extends Expr {
    final Expr arg;
    final boolean negated;
    IsNull(Expr arg, boolean negated) { this.arg = arg; this.negated = negated; }
}

final class Op extends Expr {
    final String op;                             // AND OR NOT = <> < <= > >= + - *
    final Expr left, right;
    Op(String op, Expr left, Expr right) { this.op = op; this.left = left; this.right = right; }
}

final class Agg extends Expr {
    final String name;                           // COUNT, SUM, MIN, MAX, AVG
    final Expr arg;                              // null for COUNT(*)
    Agg(String name, Expr arg) { this.name = name; this.arg = arg; }
    String label() { return Text.of(this); }
}

final class OrderKey {
    final Expr expr;
    final boolean descending;
    OrderKey(Expr expr, boolean descending) { this.expr = expr; this.descending = descending; }
}

/** Prints an expression back to text; also gives every aggregate a unique column label. */
final class Text {
    static String of(Expr e) {
        if (e == null) return "*";
        if (e instanceof Lit) return ((Lit) e).value == null ? "NULL" : String.valueOf(((Lit) e).value);
        if (e instanceof Col) {
            Col c = (Col) e;
            return c.qualifier == null ? c.name : c.qualifier + "." + c.name;
        }
        if (e instanceof IsNull) {
            IsNull n = (IsNull) e;
            return of(n.arg) + (n.negated ? " IS NOT NULL" : " IS NULL");
        }
        if (e instanceof Op) {
            Op o = (Op) e;
            return o.right == null ? "NOT " + of(o.left) : "(" + of(o.left) + " " + o.op + " " + of(o.right) + ")";
        }
        Agg a = (Agg) e;
        return a.name.toLowerCase(Locale.ROOT) + "(" + of(a.arg) + ")";
    }
}

final class Query {
    boolean distinct;
    final List<Expr> select = new ArrayList<>();
    final List<String> tables = new ArrayList<>();
    final List<String> aliases = new ArrayList<>();
    final List<String> joinTypes = new ArrayList<>();      // "BASE", "INNER" or "LEFT"
    final List<Expr> joinPredicates = new ArrayList<>();
    Expr where, having;
    final List<Expr> groupBy = new ArrayList<>();
    final List<OrderKey> orderBy = new ArrayList<>();
    int limit = -1;
}

// ------------------------------------------------------------------ lexer and parser

final class Tok {
    final String kind;                                   // IDENT NUM STR PUNCT EOF
    final String text;
    Tok(String kind, String text) { this.kind = kind; this.text = text; }
    boolean is(String s) { return text.equalsIgnoreCase(s); }
}

final class Parser {
    private final List<Tok> toks = new ArrayList<>();
    private int pos;

    Parser(String sql) {
        int i = 0;
        while (i < sql.length()) {
            char c = sql.charAt(i);
            if (Character.isWhitespace(c)) { i++; continue; }
            if (c == '\'' || c == '"') {
                int j = i + 1;
                StringBuilder sb = new StringBuilder();
                while (j < sql.length() && sql.charAt(j) != c) sb.append(sql.charAt(j++));
                if (j >= sql.length()) throw new IllegalArgumentException("unterminated literal");
                toks.add(new Tok("STR", sb.toString()));
                i = j + 1;
            } else if (Character.isDigit(c)) {
                int j = i;
                while (j < sql.length() && Character.isDigit(sql.charAt(j))) j++;
                toks.add(new Tok("NUM", sql.substring(i, j)));
                i = j;
            } else if (Character.isLetter(c) || c == '_') {
                int j = i;
                while (j < sql.length() && (Character.isLetterOrDigit(sql.charAt(j)) || sql.charAt(j) == '_')) j++;
                toks.add(new Tok("IDENT", sql.substring(i, j)));
                i = j;
            } else if (i + 1 < sql.length() && (sql.startsWith("<>", i) || sql.startsWith("<=", i)
                    || sql.startsWith(">=", i) || sql.startsWith("!=", i))) {
                toks.add(new Tok("PUNCT", sql.charAt(i) == '!' ? "<>" : sql.substring(i, i + 2)));
                i += 2;
            } else {
                toks.add(new Tok("PUNCT", String.valueOf(c)));
                i++;
            }
        }
    }

    static Query parse(String sql) { return new Parser(sql).statement(); }

    private Tok peek() { return pos < toks.size() ? toks.get(pos) : new Tok("EOF", "<eof>"); }

    private Tok next() { return pos < toks.size() ? toks.get(pos++) : new Tok("EOF", "<eof>"); }

    private boolean accept(String text) {
        if (peek().is(text)) { pos++; return true; }
        return false;
    }

    private void expect(String text) {
        if (!accept(text)) throw new IllegalArgumentException("expected " + text + ", found " + peek().text);
    }

    private static final Set<String> CLAUSE = new HashSet<>(Arrays.asList(
            "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "JOIN", "INNER", "LEFT", "ON"));

    Query statement() {
        Query q = new Query();
        expect("SELECT");
        q.distinct = accept("DISTINCT");
        q.select.add(expression());
        while (accept(",")) q.select.add(expression());
        if (accept("FROM")) {
            relation(q, "BASE");
            while (peek().is("JOIN") || peek().is("INNER") || peek().is("LEFT")) {
                String type = "INNER";
                if (accept("LEFT")) { accept("OUTER"); expect("JOIN"); type = "LEFT"; }
                else if (accept("INNER")) expect("JOIN");
                else expect("JOIN");
                relation(q, type);
            }
        }
        if (accept("WHERE")) q.where = expression();
        if (accept("GROUP")) {
            expect("BY");
            q.groupBy.add(expression());
            while (accept(",")) q.groupBy.add(expression());
        }
        if (accept("HAVING")) q.having = expression();
        if (accept("ORDER")) {
            expect("BY");
            do {
                Expr e = expression();
                boolean desc = accept("DESC");
                if (!desc) accept("ASC");
                q.orderBy.add(new OrderKey(e, desc));
            } while (accept(","));
        }
        if (accept("LIMIT")) q.limit = Integer.parseInt(next().text);
        if (pos != toks.size()) throw new IllegalArgumentException("trailing input at " + peek().text);
        if (q.tables.isEmpty()) throw new IllegalArgumentException("a FROM clause is required");
        return q;
    }

    private void relation(Query q, String joinType) {
        String table = next().text;
        String alias = table;
        Tok ahead = peek();
        if (ahead.kind.equals("IDENT") && !CLAUSE.contains(ahead.text.toUpperCase(Locale.ROOT))) {
            alias = next().text;
        }
        for (String existing : q.aliases) {
            if (existing.equalsIgnoreCase(alias)) throw new IllegalArgumentException("duplicate alias " + alias);
        }
        q.tables.add(table);
        q.aliases.add(alias);
        q.joinTypes.add(joinType);
        q.joinPredicates.add(null);
        if (!joinType.equals("BASE")) {
            expect("ON");
            q.joinPredicates.set(q.joinPredicates.size() - 1, expression());
        }
    }

    private Expr expression() { return or(); }

    private Expr or() {
        Expr e = and();
        while (accept("OR")) e = new Op("OR", e, and());
        return e;
    }

    private Expr and() {
        Expr e = not();
        while (accept("AND")) e = new Op("AND", e, not());
        return e;
    }

    private Expr not() {
        if (accept("NOT")) return new Op("NOT", not(), null);
        return comparison();
    }

    private Expr comparison() {
        Expr e = additive();
        if (accept("IS")) {
            boolean negated = accept("NOT");
            expect("NULL");
            return new IsNull(e, negated);
        }
        for (String op : new String[] {"=", "<>", "<=", ">=", "<", ">"}) {
            if (accept(op)) return new Op(op, e, additive());
        }
        return e;
    }

    private Expr additive() {
        Expr e = multiplicative();
        while (peek().is("+") || peek().is("-")) e = new Op(next().text, e, multiplicative());
        return e;
    }

    private Expr multiplicative() {
        Expr e = primary();
        while (accept("*")) e = new Op("*", e, primary());
        return e;
    }

    private Expr primary() {
        if (accept("(")) {
            Expr inner = expression();
            expect(")");
            return inner;
        }
        Tok t = next();
        if (t.kind.equals("NUM")) return new Lit(Long.parseLong(t.text));
        if (t.kind.equals("STR")) return new Lit(t.text);
        if (t.kind.equals("EOF")) throw new IllegalArgumentException("unexpected end of statement");
        if (accept("(")) {
            String name = t.text.toUpperCase(Locale.ROOT);
            if (name.equals("COUNT") && accept("*")) {
                expect(")");
                return new Agg("COUNT", null);
            }
            Expr arg = expression();
            expect(")");
            if (!Arrays.asList("COUNT", "SUM", "MIN", "MAX", "AVG").contains(name)) {
                throw new IllegalArgumentException("unknown function " + t.text);
            }
            return new Agg(name, arg);
        }
        if (t.text.equalsIgnoreCase("NULL")) return new Lit(null);
        if (accept(".")) return new Col(t.text, next().text);
        return new Col(null, t.text);
    }
}

// ------------------------------------------------------------------ storage

final class Table {
    final String name;
    final List<String> columns;
    final List<List<Object>> rows = new ArrayList<>();

    Table(String name, List<String> columns) { this.name = name; this.columns = columns; }

    void insert(Object... values) { rows.add(new ArrayList<>(Arrays.asList(values))); }

    int indexOf(String column) {
        for (int i = 0; i < columns.size(); i++) if (columns.get(i).equalsIgnoreCase(column)) return i;
        return -1;
    }
}

final class Catalog {
    final Map<String, Table> tables = new LinkedHashMap<>();

    void add(Table t) { tables.put(t.name, t); }

    Table get(String name) {
        Table t = tables.get(name);
        if (t == null) throw new IllegalArgumentException("unknown table " + name);
        return t;
    }
}

// ------------------------------------------------------------------ plan tree

abstract class PlanNode {
    List<String> schema = new ArrayList<>();
    abstract String describe(String indent);
}

final class ScanNode extends PlanNode {
    final Table table;
    final String alias;
    ScanNode(Table table, String alias) {
        this.table = table;
        this.alias = alias;
        for (String c : table.columns) schema.add(alias + "." + c);
    }
    String describe(String indent) { return indent + "Scan(" + table.name + " as " + alias + ")\n"; }
}

final class FilterNode extends PlanNode {
    final PlanNode child;
    final Expr predicate;
    FilterNode(PlanNode child, Expr predicate) {
        this.child = child;
        this.predicate = predicate;
        this.schema = child.schema;
    }
    String describe(String indent) {
        return indent + "Filter " + Text.of(predicate) + "\n" + child.describe(indent + "  ");
    }
}

final class JoinNode extends PlanNode {
    final PlanNode left, right;
    final Expr predicate, leftKey, rightKey;
    final boolean leftOuter;
    final int estimatedCost, alternativeCost;

    JoinNode(PlanNode left, PlanNode right, Expr predicate, boolean leftOuter,
             Expr leftKey, Expr rightKey, int estimatedCost, int alternativeCost) {
        this.left = left;
        this.right = right;
        this.predicate = predicate;
        this.leftOuter = leftOuter;
        this.leftKey = leftKey;
        this.rightKey = rightKey;
        this.estimatedCost = estimatedCost;
        this.alternativeCost = alternativeCost;
        schema.addAll(left.schema);
        schema.addAll(right.schema);
    }

    boolean hashJoin() { return leftKey != null; }

    String describe(String indent) {
        return indent + (hashJoin() ? "HashJoin" : "NestedLoopJoin") + (leftOuter ? "(left outer)" : "")
                + " cost=" + estimatedCost + " alternative=" + alternativeCost + "\n"
                + left.describe(indent + "  ") + right.describe(indent + "  ");
    }
}

final class GroupNode extends PlanNode {
    final PlanNode child;
    final List<Expr> keys;
    final List<Agg> aggregates;
    GroupNode(PlanNode child, List<Expr> keys, List<Agg> aggregates) {
        this.child = child;
        this.keys = keys;
        this.aggregates = aggregates;
        for (int i = 0; i < keys.size(); i++) {
            Expr k = keys.get(i);
            schema.add(k instanceof Col ? Text.of(k) : "group" + i);
        }
        for (Agg a : aggregates) schema.add(a.label());
    }
    String describe(String indent) { return indent + "Aggregate\n" + child.describe(indent + "  "); }
}

final class ProjectNode extends PlanNode {
    final PlanNode child;
    final List<Expr> expressions;
    final boolean distinct;
    ProjectNode(PlanNode child, List<Expr> expressions, boolean distinct) {
        this.child = child;
        this.expressions = expressions;
        this.distinct = distinct;
        for (int i = 0; i < expressions.size(); i++) schema.add("out" + i);
    }
    String describe(String indent) {
        return indent + (distinct ? "Distinct\n" + indent + "  " : "") + "Project "
                + expressions.stream().map(Text::of).reduce((a, b) -> a + ", " + b).orElse("")
                + "\n" + child.describe(indent + "  ");
    }
}

final class SortNode extends PlanNode {
    final PlanNode child;
    final List<OrderKey> keys;
    SortNode(PlanNode child, List<OrderKey> keys) {
        this.child = child;
        this.keys = keys;
        this.schema = child.schema;
    }
    String describe(String indent) { return indent + "Sort\n" + child.describe(indent + "  "); }
}

final class LimitNode extends PlanNode {
    final PlanNode child;
    final int limit;
    LimitNode(PlanNode child, int limit) {
        this.child = child;
        this.limit = limit;
        this.schema = child.schema;
    }
    String describe(String indent) { return indent + "Limit(" + limit + ")\n" + child.describe(indent + "  "); }
}

// ------------------------------------------------------------------ planner

final class Planner {
    final Catalog catalog;
    boolean optimized = true;
    int lastEstimatedCost, lastAlternativeCost;

    Planner(Catalog catalog) { this.catalog = catalog; }

    PlanNode plan(Query q) {
        List<PlanNode> relations = new ArrayList<>();
        for (int i = 0; i < q.tables.size(); i++) {
            relations.add(new ScanNode(catalog.get(q.tables.get(i)), q.aliases.get(i)));
        }
        Expr where = q.where;
        if (optimized && where != null) {
            List<Expr> conjuncts = new ArrayList<>();
            splitAnd(where, conjuncts);
            for (int i = conjuncts.size() - 1; i >= 0; i--) {
                int target = soleRelation(conjuncts.get(i), relations);
                if (target >= 0 && pushdownSafe(q, target)) {
                    relations.set(target, new FilterNode(relations.get(target), conjuncts.get(i)));
                    conjuncts.remove(i);
                }
            }
            where = null;
            for (Expr e : conjuncts) where = where == null ? e : new Op("AND", where, e);
        }
        PlanNode node = relations.get(0);
        for (int i = 1; i < relations.size(); i++) {
            PlanNode right = relations.get(i);
            Expr on = q.joinPredicates.get(i);
            Expr[] equi = equiKeys(on, node, right);
            int leftRows = estimate(node), rightRows = estimate(right);
            int hashCost = leftRows + rightRows;
            int loopCost = leftRows * rightRows;
            lastEstimatedCost = equi == null ? loopCost : hashCost;
            lastAlternativeCost = equi == null ? hashCost : loopCost;
            node = new JoinNode(node, right, on, q.joinTypes.get(i).equals("LEFT"),
                    equi == null ? null : equi[0], equi == null ? null : equi[1],
                    lastEstimatedCost, lastAlternativeCost);
        }
        if (where != null) node = new FilterNode(node, where);
        List<Agg> aggs = new ArrayList<>();
        for (Expr e : q.select) collectAggregates(e, aggs);
        if (q.having != null) collectAggregates(q.having, aggs);
        if (!q.groupBy.isEmpty() || !aggs.isEmpty()) node = new GroupNode(node, q.groupBy, aggs);
        if (q.having != null) node = new FilterNode(node, q.having);
        if (!q.orderBy.isEmpty()) node = new SortNode(node, q.orderBy);
        node = new ProjectNode(node, q.select, q.distinct);
        if (q.limit >= 0) node = new LimitNode(node, q.limit);
        return node;
    }

    static void splitAnd(Expr e, List<Expr> out) {
        if (e instanceof Op && ((Op) e).op.equals("AND")) {
            splitAnd(((Op) e).left, out);
            splitAnd(((Op) e).right, out);
        } else {
            out.add(e);
        }
    }

    /**
     * Relation `target` is the null-supplying (right) side of the join that introduces it, so a
     * predicate on it cannot be evaluated before that join - filtered out right rows would turn into
     * NULL-padded rows instead of disappearing. For every later join the relation sits on the
     * preserved side, so pushing is safe there. The base relation is always safe.
     */
    static boolean pushdownSafe(Query q, int target) {
        if (target <= 0) return target == 0;
        return q.joinTypes.get(target).equals("INNER");
    }

    int soleRelation(Expr e, List<PlanNode> relations) {
        int found = -1;
        for (String name : columnsOf(e)) {
            int owner = -1;
            for (int i = 0; i < relations.size(); i++) {
                if (hasColumn(relations.get(i).schema, name)) { owner = i; break; }
            }
            if (owner < 0) throw new IllegalArgumentException("unknown column " + name);
            if (found < 0) found = owner;
            else if (found != owner) return -1;
        }
        return found;
    }

    /** True when the name resolves to exactly one column of this node's schema. */
    static boolean hasColumn(List<String> schema, String name) {
        int matches = 0;
        for (String column : schema) {
            String bare = column.substring(column.indexOf('.') + 1);
            if (column.equalsIgnoreCase(name) || bare.equalsIgnoreCase(name)) matches++;
        }
        return matches == 1;
    }

    static List<String> columnsOf(Expr e) {
        List<String> out = new ArrayList<>();
        if (e == null) return out;
        if (e instanceof Col) {
            Col c = (Col) e;
            out.add(c.qualifier == null ? c.name : c.qualifier + "." + c.name);
        } else if (e instanceof IsNull) {
            out.addAll(columnsOf(((IsNull) e).arg));
        } else if (e instanceof Op) {
            out.addAll(columnsOf(((Op) e).left));
            out.addAll(columnsOf(((Op) e).right));
        } else if (e instanceof Agg) {
            out.addAll(columnsOf(((Agg) e).arg));
        }
        return out;
    }

    static Expr[] equiKeys(Expr on, PlanNode left, PlanNode right) {
        if (on == null) return null;
        if (on instanceof Op && ((Op) on).op.equals("AND")) {
            Expr[] a = equiKeys(((Op) on).left, left, right);
            return a != null ? a : equiKeys(((Op) on).right, left, right);
        }
        if (!(on instanceof Op) || !((Op) on).op.equals("=")) return null;
        Expr l = ((Op) on).left, r = ((Op) on).right;
        if (!(l instanceof Col) || !(r instanceof Col)) return null;
        boolean lLeft = belongsTo((Col) l, left), rLeft = belongsTo((Col) r, left);
        boolean lRight = belongsTo((Col) l, right), rRight = belongsTo((Col) r, right);
        if (lLeft && rRight) return new Expr[] {l, r};
        if (rLeft && lRight) return new Expr[] {r, l};
        return null;
    }

    static boolean belongsTo(Col c, PlanNode node) {
        return hasColumn(node.schema, c.qualifier == null ? c.name : c.qualifier + "." + c.name);
    }

    /** Row-count estimate; a pushed-down filter is assumed to halve its input. */
    static int estimate(PlanNode node) {
        if (node instanceof FilterNode) return estimate(((FilterNode) node).child) / 2 + 1;
        if (node instanceof ScanNode) return ((ScanNode) node).table.rows.size();
        if (node instanceof JoinNode) {
            JoinNode j = (JoinNode) node;
            return j.estimatedCost;
        }
        return estimate(((GroupNode) node).child);
    }

    static void collectAggregates(Expr e, List<Agg> out) {
        if (e == null) return;
        if (e instanceof Agg) { out.add((Agg) e); return; }
        if (e instanceof Op) {
            collectAggregates(((Op) e).left, out);
            collectAggregates(((Op) e).right, out);
        } else if (e instanceof IsNull) {
            collectAggregates(((IsNull) e).arg, out);
        }
    }
}

// ------------------------------------------------------------------ executor

final class Executor {
    int rowsTouched, probes, hashTableRows, nestedLoopPairs, rowsSorted;

    Executor() { }

    /** TRUE keeps a row; FALSE and UNKNOWN (null) do not. A non-boolean predicate is an error. */
    static boolean isTrue(Object v) {
        if (v == null) return false;
        if (v instanceof Boolean) return (Boolean) v;
        throw new IllegalArgumentException("predicate is not boolean: " + v);
    }

    List<List<Object>> run(PlanNode node) {
        if (node instanceof ScanNode) {
            ScanNode s = (ScanNode) node;
            rowsTouched += s.table.rows.size();
            List<List<Object>> out = new ArrayList<>();
            for (List<Object> row : s.table.rows) out.add(new ArrayList<>(row));
            return out;
        }
        if (node instanceof FilterNode) {
            FilterNode f = (FilterNode) node;
            List<List<Object>> out = new ArrayList<>();
            List<List<Object>> input = run(f.child);
            for (List<Object> row : input) if (isTrue(eval(f.predicate, f.child.schema, row))) out.add(row);
            return out;
        }
        if (node instanceof JoinNode) return join((JoinNode) node);
        if (node instanceof GroupNode) return group((GroupNode) node);
        if (node instanceof SortNode) {
            SortNode s = (SortNode) node;
            List<List<Object>> rows = run(s.child);
            rowsSorted += rows.size();
            rows.sort((x, y) -> compareRows(x, y, s.keys, s.child.schema));
            return rows;
        }
        if (node instanceof ProjectNode) {
            ProjectNode p = (ProjectNode) node;
            List<List<Object>> out = new ArrayList<>();
            Set<List<Object>> seen = p.distinct ? new LinkedHashSet<>() : null;
            for (List<Object> row : run(p.child)) {
                List<Object> projected = new ArrayList<>();
                for (Expr e : p.expressions) projected.add(eval(e, p.child.schema, row));
                if (seen == null || seen.add(projected)) out.add(projected);
            }
            return out;
        }
        LimitNode l = (LimitNode) node;
        List<List<Object>> rows = run(l.child);
        return new ArrayList<>(rows.subList(0, Math.min(l.limit, rows.size())));
    }

    List<List<Object>> join(JoinNode j) {
        List<List<Object>> left = run(j.left);
        List<List<Object>> right = run(j.right);
        List<List<Object>> out = new ArrayList<>();
        if (j.hashJoin()) {
            Map<Object, List<List<Object>>> table = new LinkedHashMap<>();
            for (List<Object> r : right) {
                hashTableRows++;
                table.computeIfAbsent(eval(j.rightKey, j.right.schema, r), k -> new ArrayList<>()).add(r);
            }
            for (List<Object> l : left) {
                probes++;
                List<List<Object>> matches = table.get(eval(j.leftKey, j.left.schema, l));
                boolean matched = false;
                if (matches != null) {
                    for (List<Object> r : matches) {
                        List<Object> combined = new ArrayList<>(l);
                        combined.addAll(r);
                        if (j.predicate == null || isTrue(eval(j.predicate, j.schema, combined))) {
                            out.add(combined);
                            matched = true;
                        }
                    }
                }
                if (j.leftOuter && !matched) out.add(padLeft(l, j));
            }
            return out;
        }
        for (List<Object> l : left) {
            boolean matched = false;
            for (List<Object> r : right) {
                nestedLoopPairs++;
                List<Object> combined = new ArrayList<>(l);
                combined.addAll(r);
                if (j.predicate == null || isTrue(eval(j.predicate, j.schema, combined))) {
                    out.add(combined);
                    matched = true;
                }
            }
            if (j.leftOuter && !matched) out.add(padLeft(l, j));
        }
        return out;
    }

    static List<Object> padLeft(List<Object> left, JoinNode j) {
        List<Object> combined = new ArrayList<>(left);
        for (int i = 0; i < j.right.schema.size(); i++) combined.add(null);
        return combined;
    }

    /** Hash aggregation: one accumulator per group key, keys compared by value. */
    List<List<Object>> group(GroupNode g) {
        List<List<Object>> input = run(g.child);
        Map<List<Object>, List<Acc>> groups = new LinkedHashMap<>();
        for (List<Object> row : input) {
            List<Object> key = new ArrayList<>();
            for (Expr e : g.keys) key.add(eval(e, g.child.schema, row));
            List<Acc> accs = groups.get(key);
            if (accs == null) {
                accs = new ArrayList<>();
                for (int i = 0; i < g.aggregates.size(); i++) accs.add(new Acc());
                groups.put(key, accs);
            }
            for (int i = 0; i < g.aggregates.size(); i++) {
                Agg a = g.aggregates.get(i);
                Acc acc = accs.get(i);
                acc.rows++;
                Object value = a.arg == null ? Boolean.TRUE : eval(a.arg, g.child.schema, row);
                if (value == null) continue;
                acc.nonNull++;
                if (a.name.equals("COUNT")) continue;
                if (a.name.equals("SUM") || a.name.equals("AVG")) acc.value = plus(acc.value, asNumber(value));
                else if (a.name.equals("MIN")) {
                    acc.value = acc.value == null || compareValues(value, acc.value) < 0 ? value : acc.value;
                } else {
                    acc.value = acc.value == null || compareValues(value, acc.value) > 0 ? value : acc.value;
                }
            }
        }
        if (groups.isEmpty() && g.keys.isEmpty()) {                 // one empty group of aggregates
            List<Acc> accs = new ArrayList<>();
            for (int i = 0; i < g.aggregates.size(); i++) accs.add(new Acc());
            groups.put(new ArrayList<>(), accs);
        }
        List<List<Object>> out = new ArrayList<>();
        for (Map.Entry<List<Object>, List<Acc>> e : groups.entrySet()) {
            List<Object> row = new ArrayList<>(e.getKey());
            for (int i = 0; i < g.aggregates.size(); i++) {
                Agg a = g.aggregates.get(i);
                Acc acc = e.getValue().get(i);
                if (a.name.equals("COUNT")) row.add(a.arg == null ? acc.rows : acc.nonNull);
                else if (a.name.equals("AVG")) row.add(acc.value == null ? null : ((Number) acc.value).doubleValue() / acc.nonNull);
                else row.add(acc.value);
            }
            out.add(row);
        }
        return out;
    }

    static Object plus(Object acc, Number add) {
        double value = (acc == null ? 0 : ((Number) acc).doubleValue()) + add.doubleValue();
        if (add instanceof Long && (acc == null || acc instanceof Long)) return (long) value;
        return value;
    }

    static Number asNumber(Object v) {
        if (!(v instanceof Number)) throw new IllegalArgumentException("a numeric operand was expected but got " + v);
        return (Number) v;
    }

    Object eval(Expr e, List<String> schema, List<Object> row) {
        if (e == null) return null;
        if (e instanceof Lit) return ((Lit) e).value;
        if (e instanceof Col) return row.get(indexOf(schema, (Col) e));
        if (e instanceof IsNull) {
            Object v = eval(((IsNull) e).arg, schema, row);
            return Boolean.valueOf((v == null) != ((IsNull) e).negated);
        }
        if (e instanceof Agg) {
            int idx = schema.indexOf(((Agg) e).label());
            if (idx < 0) throw new IllegalArgumentException("aggregate outside a grouped query");
            return row.get(idx);
        }
        Op o = (Op) e;
        if (o.op.equals("NOT")) {
            Object v = eval(o.left, schema, row);
            return v == null ? null : Boolean.valueOf(!isTrue(v));   // NOT UNKNOWN stays UNKNOWN
        }
        Object l = eval(o.left, schema, row);
        Object r = eval(o.right, schema, row);
        if (o.op.equals("AND")) return and(l, r);
        if (o.op.equals("OR")) return or(l, r);
        if (o.op.equals("+") || o.op.equals("-") || o.op.equals("*")) {
            if (l == null || r == null) return null;
            return arithmetic(o.op, asNumber(l), asNumber(r));
        }
        if (l == null || r == null) return null;                  // UNKNOWN propagates
        return Boolean.valueOf(compareValues(l, r, o.op));
    }

    static Object and(Object l, Object r) {
        if (Boolean.FALSE.equals(l) || Boolean.FALSE.equals(r)) return Boolean.FALSE;
        if (l == null || r == null) return null;
        return Boolean.TRUE;
    }

    static Object or(Object l, Object r) {
        if (Boolean.TRUE.equals(l) || Boolean.TRUE.equals(r)) return Boolean.TRUE;
        if (l == null || r == null) return null;
        return Boolean.FALSE;
    }

    static Object arithmetic(String op, Number l, Number r) {
        if (op.equals("*")) return l.longValue() * r.longValue();
        if (op.equals("+")) return l.longValue() + r.longValue();
        return l.longValue() - r.longValue();
    }

    static int indexOf(List<String> schema, Col c) {
        int found = -1;
        for (int i = 0; i < schema.size(); i++) {
            String column = schema.get(i);
            String bare = column.substring(column.indexOf('.') + 1);
            if (c.qualifier == null) {
                if (bare.equalsIgnoreCase(c.name)) {
                    if (found >= 0) throw new IllegalArgumentException("ambiguous column " + c.name);
                    found = i;
                }
            } else if (column.equalsIgnoreCase(c.qualifier + "." + c.name)) {
                found = i;
            }
        }
        if (found < 0) {
            throw new IllegalArgumentException("unknown column "
                    + (c.qualifier == null ? c.name : c.qualifier + "." + c.name));
        }
        return found;
    }

    int compareRows(List<Object> x, List<Object> y, List<OrderKey> keys, List<String> schema) {
        for (OrderKey k : keys) {
            Object a = eval(k.expr, schema, x);
            Object b = eval(k.expr, schema, y);
            if (a == null && b == null) continue;
            if (a == null) return 1;                              // NULLs sort last in either direction
            if (b == null) return -1;
            int c = compareValues(a, b);
            if (c != 0) return k.descending ? -c : c;
        }
        return 0;
    }

    static int compareValues(Object a, Object b) {
        if (a instanceof Number && b instanceof Number) {
            return Double.compare(((Number) a).doubleValue(), ((Number) b).doubleValue());
        }
        if (a instanceof String && b instanceof String) return ((String) a).compareTo((String) b);
        throw new IllegalArgumentException("incomparable values " + a + " and " + b);
    }

    static boolean compareValues(Object a, Object b, String op) {
        int c = compareValues(a, b);
        switch (op) {
            case "=": return c == 0;
            case "<>": return c != 0;
            case "<": return c < 0;
            case "<=": return c <= 0;
            case ">": return c > 0;
            default: return c >= 0;
        }
    }
}

final class Acc {
    Object value;
    long rows, nonNull;
}

// ------------------------------------------------------------------ brute-force reference

/** Ignores the planner: FROM-order nested loops over the cross product, sort-based grouping. */
final class Reference {
    final Catalog catalog;
    int pairs;

    Reference(Catalog catalog) { this.catalog = catalog; }

    List<List<Object>> evaluate(Query q) {
        List<String> schema = new ArrayList<>();
        List<List<Object>> rows = new ArrayList<>();
        rows.add(new ArrayList<>());
        for (int i = 0; i < q.tables.size(); i++) {
            if (i == 0) {
                rows = new ArrayList<>();
                for (List<Object> r : catalog.get(q.tables.get(0)).rows) rows.add(new ArrayList<>(r));
                schema = new ArrayList<>();
                for (String c : catalog.get(q.tables.get(0)).columns) schema.add(q.aliases.get(0) + "." + c);
                continue;
            }
            Table t = catalog.get(q.tables.get(i));
            List<String> right = new ArrayList<>();
            for (String c : t.columns) right.add(q.aliases.get(i) + "." + c);
            List<String> combined = new ArrayList<>(schema);
            combined.addAll(right);
            Expr on = q.joinPredicates.get(i);
            boolean leftOuter = q.joinTypes.get(i).equals("LEFT");
            List<List<Object>> produced = new ArrayList<>();
            for (List<Object> leftRow : rows) {
                boolean matched = false;
                for (List<Object> r : t.rows) {
                    pairs++;
                    List<Object> row = new ArrayList<>(leftRow);
                    row.addAll(r);
                    if (on == null || truth(value(on, combined, row))) {
                        produced.add(row);
                        matched = true;
                    }
                }
                if (leftOuter && !matched) {
                    List<Object> padded = new ArrayList<>(leftRow);
                    for (int c = 0; c < right.size(); c++) padded.add(null);
                    produced.add(padded);
                }
            }
            schema = combined;
            rows = produced;
        }
        if (q.where != null) {
            List<List<Object>> kept = new ArrayList<>();
            for (List<Object> row : rows) if (truth(value(q.where, schema, row))) kept.add(row);
            rows = kept;
        }
        List<Agg> aggs = new ArrayList<>();
        for (Expr e : q.select) Planner.collectAggregates(e, aggs);
        if (q.having != null) Planner.collectAggregates(q.having, aggs);
        List<String> resultSchema;
        if (!q.groupBy.isEmpty() || !aggs.isEmpty()) {
            Map<List<Object>, List<List<Object>>> groups = new LinkedHashMap<>();
            for (List<Object> row : rows) {
                List<Object> key = new ArrayList<>();
                for (Expr e : q.groupBy) key.add(value(e, schema, row));
                groups.computeIfAbsent(key, k -> new ArrayList<>()).add(row);
            }
            if (rows.isEmpty() && q.groupBy.isEmpty()) groups.put(new ArrayList<>(), new ArrayList<>());
            resultSchema = new ArrayList<>();
            for (int i = 0; i < q.groupBy.size(); i++) {
                Expr k = q.groupBy.get(i);
                resultSchema.add(k instanceof Col ? Text.of(k) : "group" + i);
            }
            for (Agg a : aggs) resultSchema.add(a.label());
            List<List<Object>> grouped = new ArrayList<>();
            for (Map.Entry<List<Object>, List<List<Object>>> g : groups.entrySet()) {
                List<Object> row = new ArrayList<>(g.getKey());
                for (Agg a : aggs) row.add(aggregate(a, schema, g.getValue()));
                if (q.having == null || truth(value(q.having, resultSchema, row))) grouped.add(row);
            }
            rows = grouped;
        } else {
            resultSchema = schema;
        }
        if (!q.orderBy.isEmpty()) {
            List<OrderKey> keys = q.orderBy;
            List<String> orderSchema = resultSchema;
            rows.sort((x, y) -> {
                for (OrderKey k : keys) {
                    Object a = value(k.expr, orderSchema, x);
                    Object b = value(k.expr, orderSchema, y);
                    if (a == null && b == null) continue;
                    if (a == null) return 1;
                    if (b == null) return -1;
                    int c = Executor.compareValues(a, b);
                    if (c != 0) return k.descending ? -c : c;
                }
                return 0;
            });
        }
        List<List<Object>> projected = new ArrayList<>();
        Set<List<Object>> seen = q.distinct ? new LinkedHashSet<>() : null;
        for (List<Object> row : rows) {
            List<Object> out = new ArrayList<>();
            for (Expr e : q.select) out.add(value(e, resultSchema, row));
            if (seen == null || seen.add(out)) projected.add(out);
        }
        if (q.limit >= 0 && projected.size() > q.limit) {
            projected = new ArrayList<>(projected.subList(0, q.limit));
        }
        return projected;
    }

    static boolean truth(Object v) { return Boolean.TRUE.equals(v); }

    /** A deliberately separate evaluator: no shared dispatch, only the shared value comparison. */
    Object value(Expr e, List<String> schema, List<Object> row) {
        if (e instanceof Lit) return ((Lit) e).value;
        if (e instanceof Col) {
            Col c = (Col) e;
            for (int i = 0; i < schema.size(); i++) {
                String column = schema.get(i);
                String bare = column.substring(column.indexOf('.') + 1);
                if (c.qualifier == null ? bare.equalsIgnoreCase(c.name)
                        : column.equalsIgnoreCase(c.qualifier + "." + c.name)) {
                    return row.get(i);
                }
            }
            throw new IllegalArgumentException("unknown column " + c.name);
        }
        if (e instanceof IsNull) {
            Object v = value(((IsNull) e).arg, schema, row);
            return Boolean.valueOf((v == null) != ((IsNull) e).negated);
        }
        if (e instanceof Agg) return row.get(schema.indexOf(((Agg) e).label()));
        Op o = (Op) e;
        if (o.op.equals("NOT")) {
            Object v = value(o.left, schema, row);
            return v == null ? null : Boolean.valueOf(!truth(v));
        }
        Object l = value(o.left, schema, row);
        Object r = value(o.right, schema, row);
        if (o.op.equals("AND")) return Executor.and(l, r);
        if (o.op.equals("OR")) return Executor.or(l, r);
        if (o.op.equals("+") || o.op.equals("-") || o.op.equals("*")) {
            if (l == null || r == null) return null;
            return Executor.arithmetic(o.op, (Number) l, (Number) r);
        }
        if (l == null || r == null) return null;
        return Boolean.valueOf(Executor.compareValues(l, r, o.op));
    }

    /** Brute force over the whole group: no incremental accumulator, no hash table. */
    Object aggregate(Agg a, List<String> schema, List<List<Object>> rows) {
        if (a.name.equals("COUNT")) {
            if (a.arg == null) return (long) rows.size();
            long n = 0;
            for (List<Object> r : rows) if (value(a.arg, schema, r) != null) n++;
            return n;
        }
        List<Object> values = new ArrayList<>();
        for (List<Object> r : rows) {
            Object v = value(a.arg, schema, r);
            if (v != null) values.add(v);
        }
        if (values.isEmpty()) return null;
        if (a.name.equals("SUM") || a.name.equals("AVG")) {
            double total = 0;
            for (Object v : values) total += ((Number) v).doubleValue();
            return a.name.equals("SUM") ? (Object) (long) total : (Object) (total / values.size());
        }
        Object best = values.get(0);
        for (Object v : values) {
            int c = Executor.compareValues(v, best);
            if (a.name.equals("MIN") ? c < 0 : c > 0) best = v;
        }
        return best;
    }

    /** Orders the group keys by value, so the reference never relies on insertion order. */
    static boolean sameValues(List<Object> a, List<Object> b) {
        return a.equals(b);
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();
    static int comparisons;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    static Catalog sampleCatalog() {
        Catalog c = new Catalog();
        Table dept = new Table("departments", Arrays.asList("id", "name", "budget"));
        dept.insert(1L, "engineering", 900L);
        dept.insert(2L, "sales", 400L);
        dept.insert(3L, "support", 250L);
        dept.insert(4L, "legal", 300L);
        c.add(dept);
        Table emp = new Table("employees", Arrays.asList("id", "name", "dept_id", "salary"));
        emp.insert(1L, "ada", 1L, 200L);
        emp.insert(2L, "bob", 1L, 180L);
        emp.insert(3L, "cyd", 2L, 150L);
        emp.insert(4L, "dee", 2L, null);
        emp.insert(5L, "eve", 3L, 120L);
        emp.insert(6L, "fay", null, 160L);
        emp.insert(7L, "gil", 3L, 120L);
        emp.insert(8L, "hal", 1L, 260L);
        c.add(emp);
        Table proj = new Table("projects", Arrays.asList("emp_id", "code", "hours"));
        proj.insert(1L, "apollo", 40L);
        proj.insert(1L, "zephyr", 10L);
        proj.insert(2L, "apollo", 25L);
        proj.insert(3L, "zephyr", 5L);
        proj.insert(5L, "apollo", 15L);
        proj.insert(5L, "zephyr", 35L);
        proj.insert(8L, "apollo", 60L);
        proj.insert(9L, "apollo", 10L);
        c.add(proj);
        return c;
    }

    static String render(List<List<Object>> rows) {
        StringBuilder sb = new StringBuilder();
        for (List<Object> row : rows) {
            if (sb.length() > 0) sb.append('|');
            for (int i = 0; i < row.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(row.get(i) == null ? "NULL" : row.get(i).toString());
            }
        }
        return sb.toString();
    }

    /** Optimized plan vs optimizer-off plan vs brute-force reference: all three must agree. */
    static List<List<Object>> both(Catalog c, String sql) {
        Query forOptimized = Parser.parse(sql);
        Query forPlain = Parser.parse(sql);
        Query forReference = Parser.parse(sql);
        Executor optimizedExec = new Executor();
        List<List<Object>> optimized = optimizedExec.run(new Planner(c).plan(forOptimized));
        Planner plain = new Planner(c);
        plain.optimized = false;
        List<List<Object>> unoptimized = new Executor().run(plain.plan(forPlain));
        List<List<Object>> reference = new Reference(c).evaluate(forReference);
        comparisons++;
        check(render(optimized).equals(render(reference)), "brute-force reference disagrees for ["
                + sql + "]: " + render(optimized) + " vs " + render(reference));
        check(render(optimized).equals(render(unoptimized)), "the plan with the optimizer off disagrees for ["
                + sql + "]: " + render(optimized) + " vs " + render(unoptimized));
        return optimized;
    }

    /** Projection, filtering, three-valued NULL logic and the empty result. */
    static void projectionAndFiltering() {
        Catalog c = sampleCatalog();
        List<List<Object>> rows = both(c, "SELECT name, salary FROM employees WHERE salary > 150");
        check(render(rows).equals("ada,200|bob,180|fay,160|hal,260"),
                "greater-than keeps four rows in table order: " + render(rows));
        check(rows.get(0).get(0) instanceof String && rows.get(0).get(1) instanceof Long,
                "values keep their SQL types");
        rows = both(c, "SELECT name FROM employees WHERE salary IS NULL");
        check(render(rows).equals("dee"), "IS NULL finds the unknown salary: " + render(rows));
        rows = both(c, "SELECT name FROM employees WHERE salary IS NOT NULL AND dept_id = 1");
        check(render(rows).equals("ada|bob|hal"), "IS NOT NULL combined with AND: " + render(rows));
        rows = both(c, "SELECT name FROM employees WHERE salary > 150 OR (dept_id = 3 AND salary = 120)");
        check(render(rows).equals("ada|bob|eve|fay|gil|hal"), "OR with a parenthesised AND: " + render(rows));
        rows = both(c, "SELECT name FROM employees WHERE NOT salary > 150");
        check(render(rows).equals("cyd|eve|gil"),
                "a NULL salary is UNKNOWN, so the negation does not keep it: " + render(rows));
        rows = both(c, "SELECT name FROM employees WHERE salary > 1000");
        check(rows.isEmpty(), "an empty result set is empty, not null");
        rows = both(c, "SELECT name, salary * 2 - 5 FROM employees WHERE id = 1");
        check(render(rows).equals("ada,395"), "arithmetic in the projection: " + render(rows));
    }

    /** Joins: inner, left outer, three-way, and the outer-join pushdown rule. */
    static void joins() {
        Catalog c = sampleCatalog();
        List<List<Object>> rows = both(c, "SELECT d.name, e.name FROM employees e JOIN departments d "
                + "ON e.dept_id = d.id");
        check(rows.size() == 7, "the inner join drops the employee without a department: " + rows.size());
        check(render(rows).contains("engineering,ada"), "the join pairs rows: " + render(rows));
        rows = both(c, "SELECT e.name, d.name FROM employees e LEFT JOIN departments d ON e.dept_id = d.id");
        check(rows.size() == 8, "the left join keeps every employee: " + rows.size());
        check(render(rows).contains("fay,NULL"), "and pads the unmatched row with NULL: " + render(rows));
        rows = both(c, "SELECT e.name, p.code FROM employees e JOIN projects p ON e.id = p.emp_id "
                + "WHERE p.hours >= 35");
        check(render(rows).equals("ada,apollo|eve,zephyr|hal,apollo"), "join plus a pushed filter: "
                + render(rows));
        rows = both(c, "SELECT e.name, d.name, p.code FROM employees e JOIN departments d ON e.dept_id = d.id "
                + "JOIN projects p ON e.id = p.emp_id WHERE p.code = 'apollo'");
        check(rows.size() == 4, "the three-way join returns four rows: " + rows.size());
        check(render(rows).startsWith("ada,engineering,apollo"), "in FROM order: " + render(rows));
        rows = both(c, "SELECT e.name FROM employees e LEFT JOIN projects p ON e.id = p.emp_id "
                + "WHERE p.emp_id IS NULL");
        check(render(rows).equals("dee|fay|gil"), "antisemijoin through an outer join: " + render(rows));

        Planner planner = new Planner(c);
        PlanNode plan = planner.plan(Parser.parse("SELECT e.name FROM employees e JOIN departments d "
                + "ON e.dept_id = d.id WHERE e.salary > 150"));
        String text = plan.describe("");
        check(text.contains("HashJoin"), "an equi-join is planned as a hash join:\n" + text);
        check(text.indexOf("Filter") < text.indexOf("Scan(employees"),
                "the single-relation predicate is pushed onto the employees scan:\n" + text);
        check(planner.lastEstimatedCost < planner.lastAlternativeCost,
                "the planner preferred the hash join because it costs less: " + planner.lastEstimatedCost
                        + " vs " + planner.lastAlternativeCost);
        String outer = new Planner(c).plan(Parser.parse("SELECT e.name FROM employees e LEFT JOIN projects p "
                + "ON e.id = p.emp_id WHERE p.emp_id IS NULL")).describe("");
        check(outer.indexOf("Filter") >= 0 && outer.indexOf("Filter") < outer.indexOf("HashJoin"),
                "a predicate on the null-supplying side stays above the outer join:\n" + outer);

        Executor optimizedExec = new Executor();
        List<List<Object>> optimized = optimizedExec.run(new Planner(c).plan(Parser.parse(
                "SELECT e.name FROM employees e JOIN departments d ON e.dept_id = d.id WHERE e.salary > 150")));
        Planner plain = new Planner(c);
        plain.optimized = false;
        Executor plainExec = new Executor();
        List<List<Object>> unoptimized = plainExec.run(plain.plan(Parser.parse(
                "SELECT e.name FROM employees e JOIN departments d ON e.dept_id = d.id WHERE e.salary > 150")));
        check(render(optimized).equals(render(unoptimized)), "pushdown never changes the answer");
        check(optimizedExec.probes < plainExec.probes, "pushdown reduces the rows reaching the join: "
                + optimizedExec.probes + " vs " + plainExec.probes);
        check(optimizedExec.hashTableRows == 4, "the build side is the four-row department table: "
                + optimizedExec.hashTableRows);
        check(optimizedExec.nestedLoopPairs == 0, "the equi-join needed no nested-loop pairs");
    }

    /** Aggregates with GROUP BY and HAVING against the reference. */
    static void aggregates() {
        Catalog c = sampleCatalog();
        List<List<Object>> rows = both(c, "SELECT dept_id, count(*), sum(salary), min(salary), max(salary) "
                + "FROM employees WHERE dept_id IS NOT NULL GROUP BY dept_id");
        check(rows.size() == 3, "three department groups: " + render(rows));
        check(render(rows).contains("1,3,640,180,260"), "the engineering group: " + render(rows));
        check(render(rows).contains("2,2,150,150,150"), "SUM skips the NULL salary: " + render(rows));
        rows = both(c, "SELECT count(salary) FROM employees GROUP BY dept_id");
        check(render(rows).equals("3|1|2|1"), "COUNT(column) skips NULLs, COUNT(*) does not: " + render(rows));
        rows = both(c, "SELECT dept_id, avg(salary) FROM employees WHERE dept_id IS NOT NULL "
                + "GROUP BY dept_id HAVING sum(salary) > 300");
        check(render(rows).equals("1,213.33333333333334"), "AVG over a group kept by HAVING: " + render(rows));
        rows = both(c, "SELECT count(*) FROM employees WHERE id > 100");
        check(render(rows).equals("0"), "an empty input still produces one group: " + render(rows));
        rows = both(c, "SELECT sum(salary), min(salary) FROM employees WHERE id > 100");
        check(render(rows).equals("NULL,NULL"), "and SUM/MIN of nothing is NULL: " + render(rows));
        rows = both(c, "SELECT count(*) FROM employees");
        check(render(rows).equals("8"), "COUNT(*) counts rows including NULL columns: " + render(rows));
        rows = both(c, "SELECT d.name, count(*), sum(p.hours) FROM projects p JOIN employees e "
                + "ON p.emp_id = e.id JOIN departments d ON e.dept_id = d.id GROUP BY d.name HAVING count(*) > 1");
        check(render(rows).equals("engineering,4,135|support,2,50"), "grouping across three relations: "
                + render(rows));
    }

    /** DISTINCT, ordering, limits and the shape the planner produced. */
    static void orderingAndPlans() {
        Catalog c = sampleCatalog();
        List<List<Object>> rows = both(c, "SELECT DISTINCT salary FROM employees ORDER BY salary DESC");
        check(render(rows).equals("260|200|180|160|150|120|NULL"),
                "descending, NULLs last, duplicates removed: " + render(rows));
        rows = both(c, "SELECT name FROM employees ORDER BY dept_id, name LIMIT 3");
        check(render(rows).equals("ada|bob|hal"), "two sort keys and a limit: " + render(rows));
        rows = both(c, "SELECT d.name, count(*) FROM employees e JOIN departments d ON e.dept_id = d.id "
                + "GROUP BY d.name ORDER BY d.name");
        check(render(rows).equals("engineering,3|sales,2|support,2"), "ordered grouped join: " + render(rows));
        rows = both(c, "SELECT DISTINCT d.name FROM employees e JOIN departments d ON e.dept_id = d.id "
                + "ORDER BY d.name DESC LIMIT 2");
        check(render(rows).equals("support|sales"), "distinct after an ordered join: " + render(rows));
        List<List<Object>> bare = both(c, "SELECT name FROM departments WHERE id = 1");
        check(render(bare).equals("engineering"), "a single-relation query needs no join: " + render(bare));
    }

    /** Invalid queries are rejected instead of silently answered. */
    static void rejectedQueries() {
        Catalog c = sampleCatalog();
        int rejected = 0;
        for (String bad : new String[] {
                "SELECT nosuchcolumn FROM employees",
                "SELECT name FROM nosuchtable",
                "SELECT name employees",
                "SELECT name FROM employees WHERE name",
                "SELECT name FROM employees ORDER BY",
                "SELECT frobnicate(name) FROM employees"}) {
            try {
                Query q = Parser.parse(bad);
                new Executor().run(new Planner(c).plan(q));
            } catch (RuntimeException e) {
                rejected++;
            }
        }
        check(rejected == 6, "all six invalid queries were rejected: " + rejected);
        try {
            both(c, "SELECT name FROM employees WHERE name > 5");
            throw new AssertionError("FAILED: comparing text with a number was accepted");
        } catch (IllegalArgumentException expected) {
            check(expected.getMessage().contains("incomparable"),
                    "the error names the problem: " + expected.getMessage());
        }
        try {
            both(c, "SELECT name FROM employees WHERE name + 1 > 2");
            throw new AssertionError("FAILED: arithmetic over a text column was accepted");
        } catch (IllegalArgumentException expected) {
            check(expected.getMessage().contains("numeric operand"),
                    "arithmetic over text is rejected: " + expected.getMessage());
        }
    }

    /** The optimizer must not change results on a larger workload, and the build side is read once. */
    static void scaleCheck() {
        Catalog c = new Catalog();
        Table left = new Table("l", Arrays.asList("id", "g", "v"));
        for (int i = 0; i < 120; i++) left.insert((long) i, (long) (i % 7), (long) (i * 3 % 50));
        Table right = new Table("r", Arrays.asList("id", "label"));
        for (int i = 0; i < 60; i++) right.insert((long) (i % 40), "r" + i);
        c.add(left);
        c.add(right);
        String sql = "SELECT l.g, count(*), sum(l.v) FROM l JOIN r ON l.id = r.id WHERE l.v < 40 GROUP BY l.g";
        Executor exec = new Executor();
        List<List<Object>> rows = exec.run(new Planner(c).plan(Parser.parse(sql)));
        check(rows.size() == 7, "one row per group: " + rows.size());
        check(exec.hashTableRows == 60, "the build side was scanned exactly once: " + exec.hashTableRows);
        check(exec.probes < 120, "the pushed-down filter shrank the probe side: " + exec.probes);
        check(exec.nestedLoopPairs == 0, "no nested-loop pairs were needed");
        Planner plain = new Planner(c);
        plain.optimized = false;
        Executor plainExec = new Executor();
        List<List<Object>> unoptimized = plainExec.run(plain.plan(Parser.parse(sql)));
        check(render(rows).equals(render(unoptimized)), "the optimized plan agrees on 120 x 60 rows");
        check(exec.probes < plainExec.probes, "and it probes fewer rows: " + exec.probes
                + " vs " + plainExec.probes);
        Reference ref = new Reference(c);
        List<List<Object>> viaReference = ref.evaluate(Parser.parse(sql));
        check(render(rows).equals(render(viaReference)), "the brute-force reference agrees as well");
        check(ref.pairs > exec.hashTableRows, "the reference really did the brute-force work: "
                + ref.pairs + " pairs");
        metric("queriesCrossChecked", comparisons);
        metric("referencePairs", ref.pairs);
    }

    public static void main(String[] args) {
        projectionAndFiltering();
        joins();
        aggregates();
        orderingAndPlans();
        rejectedQueries();
        scaleCheck();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
