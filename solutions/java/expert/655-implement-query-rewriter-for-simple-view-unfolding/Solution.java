/** Task 655 - Implement query rewriter for simple view unfolding. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * A miniature relational rewriter over an immutable AST (Scan / Filter / Project / Join / ViewRef)
 * with named, already-qualified columns ("emp.salary"). Two rewrite families are implemented:
 *
 *   1. View unfolding    - every ViewRef is replaced by the view body (transitively), with a cycle
 *                          guard that rejects a self-referencing view.
 *   2. Predicate pushdown - a filter is moved as close to the scans as possible:
 *                          Filter(Project) -> Project(Filter) when the predicate only touches
 *                          pass-through projection outputs, Filter(Join) -> per-side filters when each
 *                          conjunct references only one side, Filter(Filter) -> a single conjunction,
 *                          and Project(Project) -> Project when the inner projection is pass-through.
 *                          A conjunct touching both sides (or a column the projection does not output)
 *                          is deliberately left where it is.
 *
 * Correctness is pinned by evaluation: the original and the rewritten plan are both executed over a
 * small deterministic database and their result *multisets* must be identical, on six query shapes
 * (join + conjunction, deep pushdown through two view bodies, non-pushable OR, non-pushable
 * cross-side conjunct, literal projection, empty result). Pushdown is additionally measured: it must
 * reduce the number of intermediate tuples produced, must reach a fixpoint, and must be idempotent.
 * Invalid plans (unknown view, recursive view, unknown column) are rejected.
 */
import java.util.*;

// ---------------------------------------------------------------- expressions
final class Expr {
    final String column;
    final Object literal;

    private Expr(String column, Object literal) {
        this.column = column;
        this.literal = literal;
    }

    static Expr col(String name) { return new Expr(name, null); }

    static Expr lit(int value) { return new Expr(null, value); }

    static Expr lit(String value) { return new Expr(null, value); }

    boolean isColumn() { return column != null; }

    Object value(Map<String, Object> row) {
        if (column == null) return literal;
        if (!row.containsKey(column)) throw new IllegalArgumentException("unknown column: " + column);
        return row.get(column);
    }

    void columns(Set<String> out) {
        if (column != null) out.add(column);
    }

    String describe() {
        if (column != null) return column;
        return literal instanceof String ? "'" + literal + "'" : String.valueOf(literal);
    }
}

final class Predicate {
    static final int EQ = 0, LT = 1, GT = 2, AND = 3, OR = 4;
    final int kind;
    final Expr left, right;
    final Predicate first, second;

    private Predicate(int kind, Expr left, Expr right, Predicate first, Predicate second) {
        this.kind = kind;
        this.left = left;
        this.right = right;
        this.first = first;
        this.second = second;
    }

    static Predicate compare(int kind, Expr left, Expr right) {
        return new Predicate(kind, left, right, null, null);
    }

    static Predicate and(Predicate a, Predicate b) { return new Predicate(AND, null, null, a, b); }

    static Predicate or(Predicate a, Predicate b) { return new Predicate(OR, null, null, a, b); }

    boolean eval(Map<String, Object> row) {
        if (kind == AND) return first.eval(row) && second.eval(row);
        if (kind == OR) return first.eval(row) || second.eval(row);
        Object a = left.value(row);
        Object b = right.value(row);
        if (kind == EQ) return a == null ? b == null : a.equals(b);
        int x = ((Number) a).intValue();
        int y = ((Number) b).intValue();
        return kind == LT ? x < y : x > y;
    }

    void columns(Set<String> out) {
        if (kind == AND || kind == OR) {
            first.columns(out);
            second.columns(out);
        } else {
            left.columns(out);
            right.columns(out);
        }
    }

    Set<String> columns() {
        Set<String> out = new LinkedHashSet<>();
        columns(out);
        return out;
    }

    /** Flattens a top level conjunction into its conjuncts (a non-AND predicate is its own conjunct). */
    List<Predicate> conjuncts() {
        List<Predicate> out = new ArrayList<>();
        if (kind == AND) {
            out.addAll(first.conjuncts());
            out.addAll(second.conjuncts());
        } else {
            out.add(this);
        }
        return out;
    }

    boolean isConjunction() { return kind == AND; }

    String describe() {
        switch (kind) {
            case AND: return "(" + first.describe() + " && " + second.describe() + ")";
            case OR: return "(" + first.describe() + " || " + second.describe() + ")";
            case EQ: return "(" + left.describe() + " == " + right.describe() + ")";
            case LT: return "(" + left.describe() + " < " + right.describe() + ")";
            default: return "(" + left.describe() + " > " + right.describe() + ")";
        }
    }
}

// ---------------------------------------------------------------- relational AST
abstract class Node {
    abstract List<String> schema();

    abstract String kind();
}

final class Catalog {
    static final Map<String, List<String>> TABLES = new LinkedHashMap<>();

    static void define(String table, String... columns) {
        TABLES.put(table, List.of(columns));
    }
}

final class Scan extends Node {
    final String table;

    Scan(String table) {
        if (!Catalog.TABLES.containsKey(table)) throw new IllegalArgumentException("unknown table: " + table);
        this.table = table;
    }

    List<String> schema() { return Catalog.TABLES.get(table); }

    String kind() { return "Scan"; }
}

final class Filter extends Node {
    final Node child;
    final Predicate predicate;

    Filter(Node child, Predicate predicate) {
        this.child = child;
        this.predicate = predicate;
    }

    List<String> schema() { return child.schema(); }

    String kind() { return "Filter"; }
}

final class Project extends Node {
    final Node child;
    final List<String> outputs;
    final List<Expr> expressions;

    Project(Node child, List<String> outputs, List<Expr> expressions) {
        if (outputs.size() != expressions.size()) throw new IllegalArgumentException("projection width");
        this.child = child;
        this.outputs = List.copyOf(outputs);
        this.expressions = List.copyOf(expressions);
    }

    List<String> schema() { return outputs; }

    String kind() { return "Project"; }
}

final class Join extends Node {
    final Node left, right;
    final Predicate on;

    Join(Node left, Node right, Predicate on) {
        this.left = left;
        this.right = right;
        this.on = on;
    }

    List<String> schema() {
        List<String> merged = new ArrayList<>(left.schema());
        merged.addAll(right.schema());
        return merged;
    }

    String kind() { return "Join"; }
}

final class ViewRef extends Node {
    final String view;

    ViewRef(String view) { this.view = view; }

    List<String> schema() { throw new IllegalStateException("view not unfolded: " + view); }

    String kind() { return "ViewRef"; }
}

// ---------------------------------------------------------------- evaluator
final class Plan {
    final Map<String, Node> views;
    final Map<String, List<Map<String, Object>>> data;
    long tuples;   // sum of the row counts produced by every operator, i.e. intermediate volume
    long joinInputRows;   // rows fed into join inputs - the volume predicate pushdown is meant to cut

    Plan(Map<String, Node> views, Map<String, List<Map<String, Object>>> data) {
        this.views = views;
        this.data = data;
    }

    List<Map<String, Object>> run(Node node) {
        List<Map<String, Object>> out = new ArrayList<>();
        if (node instanceof Scan) {
            out.addAll(data.get(((Scan) node).table));
        } else if (node instanceof Filter) {
            Filter f = (Filter) node;
            for (Map<String, Object> row : run(f.child)) if (f.predicate.eval(row)) out.add(row);
        } else if (node instanceof Project) {
            Project p = (Project) node;
            for (Map<String, Object> row : run(p.child)) {
                Map<String, Object> produced = new LinkedHashMap<>();
                for (int i = 0; i < p.outputs.size(); i++) produced.put(p.outputs.get(i), p.expressions.get(i).value(row));
                out.add(produced);
            }
        } else if (node instanceof Join) {
            Join j = (Join) node;
            List<Map<String, Object>> leftRows = run(j.left);
            List<Map<String, Object>> rightRows = run(j.right);
            joinInputRows += (long) leftRows.size() + rightRows.size();
            for (Map<String, Object> l : leftRows) {
                for (Map<String, Object> r : rightRows) {
                    Map<String, Object> merged = new LinkedHashMap<>(l);
                    merged.putAll(r);
                    if (j.on.eval(merged)) out.add(merged);
                }
            }
        } else {
            String name = ((ViewRef) node).view;
            Node body = views.get(name);
            if (body == null) throw new IllegalArgumentException("unknown view: " + name);
            out.addAll(run(body));
        }
        tuples += out.size();
        return out;
    }
}

// ---------------------------------------------------------------- rewriter
final class Rewriter {
    final Map<String, Node> views;

    Rewriter(Map<String, Node> views) { this.views = views; }

    Node unfold(Node node) { return unfold(node, new LinkedHashSet<>()); }

    private Node unfold(Node node, Set<String> active) {
        if (node instanceof Scan) return node;
        if (node instanceof ViewRef) {
            String name = ((ViewRef) node).view;
            Node body = views.get(name);
            if (body == null) throw new IllegalArgumentException("unknown view: " + name);
            if (!active.add(name)) throw new IllegalArgumentException("recursive view: " + name);
            Node expanded = unfold(body, active);
            active.remove(name);
            return expanded;
        }
        if (node instanceof Filter) {
            Filter f = (Filter) node;
            return new Filter(unfold(f.child, active), f.predicate);
        }
        if (node instanceof Project) {
            Project p = (Project) node;
            return new Project(unfold(p.child, active), p.outputs, p.expressions);
        }
        Join j = (Join) node;
        return new Join(unfold(j.left, active), unfold(j.right, active), j.on);
    }

    /** Repeatedly pushes filters down and merges projections until the plan stops changing. */
    Node rewrite(Node query) {
        Node current = unfold(query);
        String before = describe(current);
        for (int pass = 0; pass < 10; pass++) {
            Node next = pushdown(current);
            String after = describe(next);
            current = next;
            if (after.equals(before)) return current;
            before = after;
        }
        throw new IllegalStateException("pushdown did not reach a fixpoint");
    }

    Node pushdown(Node node) {
        if (node instanceof Scan) return node;
        if (node instanceof Filter) {
            Filter f = (Filter) node;
            return pushFilter(f.predicate, pushdown(f.child));
        }
        if (node instanceof Project) {
            Project p = (Project) node;
            Node child = pushdown(p.child);
            if (child instanceof Project) {
                Project inner = (Project) child;
                if (isPassThrough(inner)) {
                    List<Expr> composed = new ArrayList<>();
                    for (Expr out : p.expressions) composed.add(compose(out, inner));
                    return new Project(inner.child, p.outputs, composed);
                }
            }
            return new Project(child, p.outputs, p.expressions);
        }
        Join j = (Join) node;
        return new Join(pushdown(j.left), pushdown(j.right), j.on);
    }

    private Node pushFilter(Predicate predicate, Node child) {
        if (child instanceof Filter) {
            Filter inner = (Filter) child;
            return new Filter(inner.child, Predicate.and(predicate, inner.predicate));
        }
        if (child instanceof Project) {
            Project p = (Project) child;
            if (p.outputs.containsAll(predicate.columns())) {
                return new Project(pushFilter(predicate, p.child), p.outputs, p.expressions);
            }
            return new Filter(child, predicate);
        }
        if (child instanceof Join) {
            Join j = (Join) child;
            Set<String> leftColumns = new HashSet<>(j.left.schema());
            Set<String> rightColumns = new HashSet<>(j.right.schema());
            List<Predicate> leftPart = new ArrayList<>();
            List<Predicate> rightPart = new ArrayList<>();
            List<Predicate> stay = new ArrayList<>();
            for (Predicate conjunct : predicate.conjuncts()) {
                Set<String> columns = conjunct.columns();
                if (leftColumns.containsAll(columns)) leftPart.add(conjunct);
                else if (rightColumns.containsAll(columns)) rightPart.add(conjunct);
                else stay.add(conjunct);
            }
            Node newLeft = j.left, newRight = j.right;
            if (!leftPart.isEmpty()) newLeft = pushFilter(conjoin(leftPart), j.left);
            if (!rightPart.isEmpty()) newRight = pushFilter(conjoin(rightPart), j.right);
            Node joined = new Join(newLeft, newRight, j.on);
            return stay.isEmpty() ? joined : new Filter(joined, conjoin(stay));
        }
        return new Filter(child, predicate);
    }

    static Predicate conjoin(List<Predicate> parts) {
        Predicate out = parts.get(0);
        for (int i = 1; i < parts.size(); i++) out = Predicate.and(out, parts.get(i));
        return out;
    }

    static boolean isPassThrough(Project project) {
        for (int i = 0; i < project.outputs.size(); i++) {
            Expr expr = project.expressions.get(i);
            if (!expr.isColumn() || !expr.column.equals(project.outputs.get(i))) return false;
        }
        return true;
    }

    private static Expr compose(Expr outer, Project inner) {
        if (!outer.isColumn()) return outer;
        int index = inner.outputs.indexOf(outer.column);
        if (index < 0) throw new IllegalArgumentException("column not produced: " + outer.column);
        return inner.expressions.get(index);
    }

    static String describe(Node node) {
        if (node instanceof Scan) return "Scan(" + ((Scan) node).table + ")";
        if (node instanceof ViewRef) return "View(" + ((ViewRef) node).view + ")";
        if (node instanceof Filter) {
            Filter f = (Filter) node;
            return "Filter" + f.predicate.describe() + "[" + describe(f.child) + "]";
        }
        if (node instanceof Project) {
            Project p = (Project) node;
            return "Project" + p.outputs + "[" + describe(p.child) + "]";
        }
        Join j = (Join) node;
        return "Join" + j.on.describe() + "[" + describe(j.left) + ", " + describe(j.right) + "]";
    }

    static int count(Node node, String kind) {
        int total = node.kind().equals(kind) ? 1 : 0;
        if (node instanceof Filter) return total + count(((Filter) node).child, kind);
        if (node instanceof Project) return total + count(((Project) node).child, kind);
        if (node instanceof Join) {
            Join j = (Join) node;
            return total + count(j.left, kind) + count(j.right, kind);
        }
        return total;
    }

    /** Filters whose child is a scan: the deepest position a predicate can reach. */
    static int filtersOverScan(Node node) {
        if (node instanceof Filter && ((Filter) node).child instanceof Scan) return 1;
        if (node instanceof Filter) return filtersOverScan(((Filter) node).child);
        if (node instanceof Project) return filtersOverScan(((Project) node).child);
        if (node instanceof Join) {
            Join j = (Join) node;
            return filtersOverScan(j.left) + filtersOverScan(j.right);
        }
        return 0;
    }

    static int filtersOverJoin(Node node) {
        if (node instanceof Filter) {
            int here = ((Filter) node).child instanceof Join ? 1 : 0;
            return here + filtersOverJoin(((Filter) node).child);
        }
        if (node instanceof Project) return filtersOverJoin(((Project) node).child);
        if (node instanceof Join) {
            Join j = (Join) node;
            return filtersOverJoin(j.left) + filtersOverJoin(j.right);
        }
        return 0;
    }

    /** Validates that every predicate and projection only mentions columns its input produces. */
    static void validate(Node node) {
        if (node instanceof Filter) {
            Filter f = (Filter) node;
            validate(f.child);
            if (!new HashSet<>(f.child.schema()).containsAll(f.predicate.columns())) {
                throw new IllegalArgumentException("filter references unknown columns: " + f.predicate.describe());
            }
        } else if (node instanceof Project) {
            Project p = (Project) node;
            validate(p.child);
            Set<String> available = new HashSet<>(p.child.schema());
            for (Expr expr : p.expressions) {
                if (expr.isColumn() && !available.contains(expr.column)) {
                    throw new IllegalArgumentException("projection references unknown column: " + expr.column);
                }
            }
        } else if (node instanceof Join) {
            Join j = (Join) node;
            validate(j.left);
            validate(j.right);
            if (!j.schema().containsAll(j.on.columns())) {
                throw new IllegalArgumentException("join condition references unknown columns");
            }
        } else if (node instanceof ViewRef) {
            throw new IllegalStateException("view not unfolded");
        }
    }
}

public class Solution {
    static Map<String, Node> views() {
        Map<String, Node> views = new LinkedHashMap<>();
        views.put("big_dept", new Project(
                new Filter(new Scan("dept"), Predicate.compare(Predicate.GT, Expr.col("dept.size"), Expr.lit(50))),
                List.of("dept.id", "dept.name"),
                List.of(Expr.col("dept.id"), Expr.col("dept.name"))));
        views.put("rich_emp", new Project(
                new Filter(new Scan("emp"), Predicate.compare(Predicate.GT, Expr.col("emp.salary"), Expr.lit(100))),
                List.of("emp.id", "emp.dept_id", "emp.salary"),
                List.of(Expr.col("emp.id"), Expr.col("emp.dept_id"), Expr.col("emp.salary"))));
        views.put("big_rich", new Project(
                new Join(new ViewRef("big_dept"), new ViewRef("rich_emp"),
                        Predicate.compare(Predicate.EQ, Expr.col("dept.id"), Expr.col("emp.dept_id"))),
                List.of("dept.id", "dept.name", "emp.salary"),
                List.of(Expr.col("dept.id"), Expr.col("dept.name"), Expr.col("emp.salary"))));
        return views;
    }

    static Map<String, List<Map<String, Object>>> database() {
        Map<String, List<Map<String, Object>>> data = new LinkedHashMap<>();
        data.put("dept", List.of(
                row("dept.id", 1, "dept.name", "eng", "dept.size", 120),
                row("dept.id", 2, "dept.name", "ops", "dept.size", 40),
                row("dept.id", 3, "dept.name", "sales", "dept.size", 80),
                row("dept.id", 4, "dept.name", "hr", "dept.size", 10),
                row("dept.id", 5, "dept.name", "research", "dept.size", 200)));
        data.put("emp", List.of(
                row("emp.id", 10, "emp.dept_id", 1, "emp.salary", 900),
                row("emp.id", 11, "emp.dept_id", 1, "emp.salary", 1500),
                row("emp.id", 12, "emp.dept_id", 3, "emp.salary", 700),
                row("emp.id", 13, "emp.dept_id", 3, "emp.salary", 1100),
                row("emp.id", 14, "emp.dept_id", 5, "emp.salary", 2000),
                row("emp.id", 15, "emp.dept_id", 5, "emp.salary", 180),
                row("emp.id", 16, "emp.dept_id", 2, "emp.salary", 90),
                row("emp.id", 17, "emp.dept_id", 4, "emp.salary", 300),
                row("emp.id", 18, "emp.dept_id", 1, "emp.salary", 120),
                row("emp.id", 19, "emp.dept_id", 6, "emp.salary", 500)));
        return data;
    }

    static Map<String, Object> row(Object... pairs) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) out.put((String) pairs[i], pairs[i + 1]);
        return out;
    }

    /** Order independent multiset fingerprint of a result. */
    static List<String> fingerprint(List<Map<String, Object>> rows) {
        List<String> out = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            StringBuilder sb = new StringBuilder();
            List<String> keys = new ArrayList<>(row.keySet());
            Collections.sort(keys);
            for (String key : keys) sb.append(key).append('=').append(row.get(key)).append(';');
            out.add(sb.toString());
        }
        Collections.sort(out);
        return out;
    }

    /** Executes original and rewritten plan and requires the same result multiset. */
    static List<String> requireSameResult(Node query, Rewriter rewriter, Map<String, List<Map<String, Object>>> data,
                                          String label) {
        Plan originalPlan = new Plan(rewriter.views, data);
        List<String> original = fingerprint(originalPlan.run(query));
        Node rewritten = rewriter.rewrite(query);
        Plan rewrittenPlan = new Plan(rewriter.views, data);
        List<String> after = fingerprint(rewrittenPlan.run(rewritten));
        check(original.equals(after), label + ": rewritten plan must return the same rows");
        check(Rewriter.count(rewritten, "ViewRef") == 0, label + ": no view reference may survive");
        Rewriter.validate(rewritten);
        return original;
    }

    public static void main(String[] args) {
        Catalog.define("dept", "dept.id", "dept.name", "dept.size");
        Catalog.define("emp", "emp.id", "emp.dept_id", "emp.salary");
        Map<String, Node> views = views();
        Map<String, List<Map<String, Object>>> data = database();
        Rewriter rewriter = new Rewriter(views);

        // Q1: filter over a view whose projection is pass-through -> pushed into the view body
        Node q1 = new Filter(new ViewRef("big_dept"),
                Predicate.compare(Predicate.EQ, Expr.col("dept.name"), Expr.lit("eng")));
        List<String> q1Rows = requireSameResult(q1, rewriter, data, "Q1");
        check(q1Rows.size() == 1, "Q1 returns the single big engineering department");
        Node r1 = rewriter.rewrite(q1);
        check(Rewriter.count(r1, "Filter") == 1 && Rewriter.filtersOverScan(r1) == 1,
                "Q1 filter lands directly on the scan");
        Filter pushedQ1 = deepestFilter(r1);
        check(pushedQ1 != null && pushedQ1.predicate.isConjunction() && pushedQ1.predicate.conjuncts().size() == 2,
                "Q1 conjuncts are combined with the view's own filter");
        check(Rewriter.describe(r1).contains("dept.size > 50") && Rewriter.describe(r1).contains("'eng'"),
                "Q1 keeps both conditions in the pushed filter, got " + Rewriter.describe(r1));

        // Q2: one pushable conjunct plus a theta condition that references both sides
        Node q2 = new Project(
                new Filter(
                        new Join(new ViewRef("big_dept"), new Scan("emp"),
                                Predicate.compare(Predicate.EQ, Expr.col("dept.id"), Expr.col("emp.dept_id"))),
                        Predicate.and(
                                Predicate.compare(Predicate.GT, Expr.col("emp.salary"), Expr.lit(200)),
                                Predicate.compare(Predicate.LT, Expr.col("dept.id"), Expr.col("emp.id")))),
                List.of("emp.id", "dept.name"),
                List.of(Expr.col("emp.id"), Expr.col("dept.name")));
        List<String> q2Rows = requireSameResult(q2, rewriter, data, "Q2");
        check(q2Rows.size() == 5, "Q2 keeps the five employees above 200 in a big department, got " + q2Rows.size());
        Node r2 = rewriter.rewrite(q2);
        check(Rewriter.count(r2, "Filter") == 3, "Q2 has the view filter, the pushed filter and the retained one, got "
                + Rewriter.count(r2, "Filter"));
        check(Rewriter.filtersOverScan(r2) == 2, "Q2 pushes the salary conjunct onto the emp scan");
        check(Rewriter.filtersOverJoin(r2) == 1, "Q2 keeps the cross-side condition above the join");
        List<Predicate> aboveJoinQ2 = predicatesAboveJoins(r2);
        check(aboveJoinQ2.size() == 1 && aboveJoinQ2.get(0).describe().equals("(dept.id < emp.id)"),
                "Q2 leaves exactly the unsplit theta condition above the join");

        Plan unfoldedQ2 = new Plan(views, data);
        unfoldedQ2.run(rewriter.unfold(q2));
        Plan pushedQ2 = new Plan(views, data);
        pushedQ2.run(r2);
        check(pushedQ2.joinInputRows < unfoldedQ2.joinInputRows,
                "pushdown shrinks the join inputs: " + pushedQ2.joinInputRows + " < " + unfoldedQ2.joinInputRows);

        // Q3: deep pushdown through two view bodies, one conjunct per side of the join
        Node q3 = new Project(
                new Filter(
                        new Join(new ViewRef("big_dept"), new ViewRef("rich_emp"),
                                Predicate.compare(Predicate.EQ, Expr.col("dept.id"), Expr.col("emp.dept_id"))),
                        Predicate.and(
                                Predicate.compare(Predicate.GT, Expr.col("emp.salary"), Expr.lit(1200)),
                                Predicate.compare(Predicate.EQ, Expr.col("dept.name"), Expr.lit("eng")))),
                List.of("emp.id"),
                List.of(Expr.col("emp.id")));
        List<String> q3Rows = requireSameResult(q3, rewriter, data, "Q3");
        check(q3Rows.size() == 1 && q3Rows.get(0).contains("emp.id=11"), "Q3 keeps only emp 11, got " + q3Rows);
        Node r3 = rewriter.rewrite(q3);
        check(Rewriter.count(r3, "Filter") == 2, "Q3 pushes both conjuncts into the view bodies");
        check(Rewriter.filtersOverScan(r3) == 2, "both Q3 conjuncts reach scans");
        check(Rewriter.filtersOverJoin(r3) == 0, "Q3 leaves nothing above the join");

        // Q4: empty result
        Node q4 = new Project(new Filter(new Scan("emp"),
                Predicate.compare(Predicate.GT, Expr.col("emp.salary"), Expr.lit(10000))),
                List.of("emp.id"), List.of(Expr.col("emp.id")));
        List<String> q4Rows = requireSameResult(q4, rewriter, data, "Q4");
        check(q4Rows.isEmpty(), "Q4 has no rows");

        // Q5: nested projection merge plus a literal output
        Node q5 = new Project(new ViewRef("big_dept"), List.of("dept.id", "dept.name", "seven"),
                List.of(Expr.col("dept.id"), Expr.col("dept.name"), Expr.lit(7)));
        List<String> q5Rows = requireSameResult(q5, rewriter, data, "Q5");
        check(q5Rows.size() == 3, "Q5 sees the three big departments");
        check(q5Rows.get(0).contains("seven=7"), "Q5 evaluates the literal expression, got " + q5Rows);
        Node r5 = rewriter.rewrite(q5);
        check(Rewriter.count(r5, "Project") == 1 && Rewriter.count(r5, "Filter") == 1,
                "Q5 merges the nested projections");
        check(Rewriter.count(rewriter.unfold(q5), "Project") == 2, "before merging there are two projections");

        // Q6: a disjunction spanning both sides cannot be split - conservative behaviour
        Node q6 = new Filter(
                new Join(new ViewRef("big_dept"), new Scan("emp"),
                        Predicate.compare(Predicate.EQ, Expr.col("dept.id"), Expr.col("emp.dept_id"))),
                Predicate.or(
                        Predicate.compare(Predicate.GT, Expr.col("emp.salary"), Expr.lit(1200)),
                        Predicate.compare(Predicate.GT, Expr.col("dept.id"), Expr.col("emp.id"))));
        List<String> q6Rows = requireSameResult(q6, rewriter, data, "Q6");
        check(q6Rows.size() == 2, "Q6 keeps only the two top earners, got " + q6Rows);
        Node q6Unfolded = rewriter.unfold(q6);
        check(Rewriter.describe(rewriter.rewrite(q6)).equals(Rewriter.describe(q6Unfolded)),
                "Q6 is left untouched because the disjunction spans both join sides");
        List<Predicate> aboveJoinQ6 = predicatesAboveJoins(rewriter.rewrite(q6));
        check(aboveJoinQ6.size() == 1 && aboveJoinQ6.get(0).describe().contains("||")
                        && aboveJoinQ6.get(0).conjuncts().size() == 1,
                "the Q6 disjunction stays above the join as a single conjunct");

        // Q7: a conjunct referencing both sides of the join stays above the join
        Node q7 = new Filter(
                new Join(new ViewRef("big_dept"), new Scan("emp"),
                        Predicate.compare(Predicate.EQ, Expr.col("dept.id"), Expr.col("emp.dept_id"))),
                Predicate.compare(Predicate.EQ, Expr.col("dept.name"), Expr.col("emp.salary")));
        List<String> q7Rows = requireSameResult(q7, rewriter, data, "Q7");
        check(q7Rows.isEmpty(), "Q7 compares a name with a salary and matches nothing");
        Node r7 = rewriter.rewrite(q7);
        check(Rewriter.filtersOverJoin(r7) == 1, "the cross-side conjunct stays above the join");
        List<Predicate> aboveJoinQ7 = predicatesAboveJoins(r7);
        check(aboveJoinQ7.size() == 1 && aboveJoinQ7.get(0).describe().equals("(dept.name == emp.salary)"),
                "the Q7 filter above the join is exactly the cross-side condition");

        // fixpoint, idempotence and determinism
        Node r3again = rewriter.rewrite(q3);
        check(Rewriter.describe(r3again).equals(Rewriter.describe(r3)), "rewriting is deterministic");
        check(Rewriter.describe(rewriter.pushdown(r3)).equals(Rewriter.describe(r3)),
                "pushdown on a rewritten plan changes nothing (idempotence)");
        check(Rewriter.count(r3, "Scan") == 2 && Rewriter.count(r3, "Join") == 1, "no operator duplication");

        // rejected plans
        boolean rejected = false;
        Map<String, Node> cyclic = views();
        cyclic.put("loop", new Project(new ViewRef("loop"), List.of("dept.id"), List.of(Expr.col("dept.id"))));
        try {
            new Rewriter(cyclic).rewrite(new Filter(new ViewRef("loop"),
                    Predicate.compare(Predicate.EQ, Expr.col("dept.id"), Expr.lit(1))));
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a recursive view must be rejected");
        rejected = false;
        try {
            rewriter.rewrite(new Project(new ViewRef("missing_view"), List.of("dept.id"), List.of(Expr.col("dept.id"))));
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an unknown view must be rejected");
        rejected = false;
        try {
            Rewriter.validate(new Filter(new Scan("emp"),
                    Predicate.compare(Predicate.EQ, Expr.col("dept.name"), Expr.lit("eng"))));
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a filter on a column the scan does not produce must be rejected");
        rejected = false;
        try {
            new Scan("warehouse");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an unknown table must be rejected");

        // the fingerprint oracle itself is order independent
        Plan plan = new Plan(views, data);
        List<Map<String, Object>> rows = plan.run(rewriter.rewrite(q1));
        List<Map<String, Object>> shuffled = new ArrayList<>(rows);
        Collections.reverse(shuffled);
        check(fingerprint(rows).equals(fingerprint(shuffled)), "result fingerprint ignores row order");

        System.out.println("PASS");
    }

    /** First filter on a straight path down the left-most spine - used by the structural assertions. */
    static Filter deepestFilter(Node node) {
        if (node instanceof Filter) return (Filter) node;
        if (node instanceof Project) return deepestFilter(((Project) node).child);
        if (node instanceof Join) {
            Filter left = deepestFilter(((Join) node).left);
            return left != null ? left : deepestFilter(((Join) node).right);
        }
        return null;
    }

    /** Predicates of every filter that sits directly above a join. */
    static List<Predicate> predicatesAboveJoins(Node node) {
        List<Predicate> out = new ArrayList<>();
        collectPredicatesAboveJoins(node, out);
        return out;
    }

    private static void collectPredicatesAboveJoins(Node node, List<Predicate> out) {
        if (node instanceof Filter) {
            Filter f = (Filter) node;
            if (f.child instanceof Join) out.add(f.predicate);
            collectPredicatesAboveJoins(f.child, out);
        } else if (node instanceof Project) {
            collectPredicatesAboveJoins(((Project) node).child, out);
        } else if (node instanceof Join) {
            Join j = (Join) node;
            collectPredicatesAboveJoins(j.left, out);
            collectPredicatesAboveJoins(j.right, out);
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
