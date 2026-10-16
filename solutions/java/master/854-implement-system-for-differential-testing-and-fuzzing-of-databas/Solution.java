/** Task 854 - Implement system for differential testing and fuzzing of database engines.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   Three independent evaluators of a small relational algebra - SELECT ... FROM T1 JOIN T2 ON k1=k2
 *   WHERE p GROUP BY g with COUNT/SUM/MIN/MAX/AVG - plus a fourth, deliberately buggy one:
 *
 *       bruteForce  : materialises the full cross product, filters afterwards, aggregates by scanning a
 *                     fresh list per group (the reference of last resort);
 *       reference   : nested loops with the join predicate tested in the inner loop (same semantics, different
 *                     code path, aggregation written independently);
 *       optimized   : hash join over a per-key bucket index plus incremental hash aggregation;
 *       buggyHashJoin: the optimized engine with the classic NULL-key defect (NULL = NULL treated as equal).
 *
 *   A seeded fuzzer generates queries from a grammar over two populated tables (NULLs, duplicates and empty
 *   results are reachable), every pair of the three correct engines must agree byte-for-byte on the canonical
 *   result, and a mismatch against the buggy engine is delta-debugged (rows and WHERE conjuncts removed one
 *   at a time) until it is 1-minimal: dropping any single remaining row or conjunct makes the bug disappear.
 *
 * WHAT IS SIMPLIFIED
 *   No SQL parser, planner, buffer pool or disk: values are strings (null means SQL NULL), the schema is fixed
 *   and the "optimizer" is a hash join plus hash aggregation. Scale is a parameter (corpus size, table size)
 *   and everything is driven by a fixed-seed splitmix64 stream, so the corpus and the witnesses are stable.
 *
 * INVARIANTS VERIFIED
 *   Semantic: the three-valued logic tables are checked exhaustively against their definitions and NULL keys
 *   never join; the three correct engines agree on every generated query and on hand-built degenerate inputs
 *   (empty table, single row, all-NULL column, duplicated rows, zero-row aggregation); aggregation is
 *   cross-checked against an independent per-group enumeration.
 *   Fuzzing: the corpus is deterministic and seed-dependent, every grammar feature is covered, the buggy
 *   engine is detected, and the shrunk witness still fails while every one-element-smaller input passes
 *   (1-minimality) - so the checker itself is shown to have teeth. Cost: the hash join performs strictly
 *   fewer tuple comparisons than the cross product on the same query, and result comparison is invariant
 *   under permuting the stored rows.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so corpora and witnesses are reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }

    boolean nextBoolean() { return nextInt(2) == 0; }

    boolean chance(int percent) { return nextInt(100) < percent; }
}

/** FNV-1a 64 with an avalanche step: corpus digests. */
final class Digest {
    static long of(String text) {
        long h = 0xCBF29CE484222325L;
        for (int i = 0; i < text.length(); i++) {
            h ^= text.charAt(i);
            h *= 0x100000001B3L;
        }
        h ^= h >>> 33;
        h *= 0xFF51AFD7ED558CCDL;
        h ^= h >>> 33;
        return h;
    }
}

enum Tri { TRUE, FALSE, UNKNOWN }

/** SQL three-valued logic: the definitional tables AND/OR/NOT. */
final class Logic {
    static Tri and(Tri a, Tri b) {
        if (a == Tri.FALSE || b == Tri.FALSE) return Tri.FALSE;
        if (a == Tri.UNKNOWN || b == Tri.UNKNOWN) return Tri.UNKNOWN;
        return Tri.TRUE;
    }

    static Tri or(Tri a, Tri b) {
        if (a == Tri.TRUE || b == Tri.TRUE) return Tri.TRUE;
        if (a == Tri.UNKNOWN || b == Tri.UNKNOWN) return Tri.UNKNOWN;
        return Tri.FALSE;
    }

    static Tri not(Tri a) {
        if (a == Tri.TRUE) return Tri.FALSE;
        if (a == Tri.FALSE) return Tri.TRUE;
        return Tri.UNKNOWN;
    }
}

final class MiniTable {
    final String name;
    final String[] columns;
    final List<String[]> rows = new ArrayList<>();

    MiniTable(String name, String... columns) {
        this.name = name;
        this.columns = columns;
    }

    void add(String... values) { rows.add(values); }

    int indexOf(String column) {
        for (int i = 0; i < columns.length; i++) if (columns[i].equals(column)) return i;
        throw new IllegalArgumentException("unknown column " + column);
    }

    /** -1 when the column is absent, used by the generator to pick numeric columns safely. */
    int indexOrNull(String column) {
        for (int i = 0; i < columns.length; i++) if (columns[i].equals(column)) return i;
        return -1;
    }

    MiniTable copy() {
        MiniTable t = new MiniTable(name, columns);
        for (String[] row : rows) t.rows.add(row.clone());
        return t;
    }
}

/** Predicate over a joined tuple (T1 columns followed by T2 columns). */
final class Expr {
    final String kind;          // CMP, ISNULL, NOTNULL, AND, OR, NOT
    final int column;
    final String op;
    final String literal;
    final Expr left;
    final Expr right;

    private Expr(String kind, int column, String op, String literal, Expr left, Expr right) {
        this.kind = kind;
        this.column = column;
        this.op = op;
        this.literal = literal;
        this.left = left;
        this.right = right;
    }

    static Expr cmp(int column, String op, String literal) { return new Expr("CMP", column, op, literal, null, null); }

    static Expr isNull(int column) { return new Expr("ISNULL", column, null, null, null, null); }

    static Expr isNotNull(int column) { return new Expr("NOTNULL", column, null, null, null, null); }

    static Expr and(Expr a, Expr b) { return new Expr("AND", -1, null, null, a, b); }

    static Expr or(Expr a, Expr b) { return new Expr("OR", -1, null, null, a, b); }

    static Expr not(Expr a) { return new Expr("NOT", -1, null, null, a, null); }

    static Expr all() { return new Expr("TRUE", -1, null, null, null, null); }

    Tri eval(String[] tuple) {
        switch (kind) {
            case "TRUE":
                return Tri.TRUE;
            case "ISNULL":
                return tuple[column] == null ? Tri.TRUE : Tri.FALSE;
            case "NOTNULL":
                return tuple[column] == null ? Tri.FALSE : Tri.TRUE;
            case "AND":
                return Logic.and(left.eval(tuple), right.eval(tuple));
            case "OR":
                return Logic.or(left.eval(tuple), right.eval(tuple));
            case "NOT":
                return Logic.not(left.eval(tuple));
            default:
                if (tuple[column] == null) return Tri.UNKNOWN;
                int c = tuple[column].compareTo(literal);
                switch (op) {
                    case "=": return c == 0 ? Tri.TRUE : Tri.FALSE;
                    case "<>": return c != 0 ? Tri.TRUE : Tri.FALSE;
                    case "<": return c < 0 ? Tri.TRUE : Tri.FALSE;
                    case ">": return c > 0 ? Tri.TRUE : Tri.FALSE;
                    default: throw new IllegalArgumentException("unknown operator " + op);
                }
        }
    }

    String describe() {
        switch (kind) {
            case "TRUE": return "TRUE";
            case "ISNULL": return "col" + column + " IS NULL";
            case "NOTNULL": return "col" + column + " IS NOT NULL";
            case "AND": return "(" + left.describe() + " AND " + right.describe() + ")";
            case "OR": return "(" + left.describe() + " OR " + right.describe() + ")";
            case "NOT": return "NOT " + left.describe();
            default: return "col" + column + " " + op + " " + literal;
        }
    }
}

/** One aggregate term: COUNT(*) has column == -1. */
final class Agg {
    final String fn;
    final int column;

    Agg(String fn, int column) {
        this.fn = fn;
        this.column = column;
    }

    String describe() { return fn + "(" + (column < 0 ? "*" : "col" + column) + ")"; }
}

/** A generated query: join keys, WHERE conjuncts, optional grouping, aggregates. */
final class Query {
    final int leftKey;
    final int rightKey;
    final List<Expr> conjuncts;
    final int groupBy;
    final List<Agg> aggregates;

    Query(int leftKey, int rightKey, List<Expr> conjuncts, int groupBy, List<Agg> aggregates) {
        this.leftKey = leftKey;
        this.rightKey = rightKey;
        this.conjuncts = conjuncts;
        this.groupBy = groupBy;
        this.aggregates = aggregates;
    }

    Expr where() {
        Expr result = Expr.all();
        for (Expr e : conjuncts) result = Expr.and(result, e);
        return result;
    }

    Query withConjuncts(List<Expr> replacement) {
        return new Query(leftKey, rightKey, replacement, groupBy, aggregates);
    }

    String describe() {
        List<String> parts = new ArrayList<>();
        for (Expr e : conjuncts) parts.add(e.describe());
        List<String> aggText = new ArrayList<>();
        for (Agg a : aggregates) aggText.add(a.describe());
        return "join(" + leftKey + "=" + rightKey + ") where " + parts + " groupBy " + groupBy
                + " aggs " + aggText;
    }
}

/** How much work an evaluator did, so the hash join can be shown to be cheaper than the cross product. */
final class Work {
    long tupleComparisons;
    long rowsVisited;
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    // ------------------------------------------------------------------ schema and data

    static MiniTable[] buildData(long seed) {
        MiniTable users = new MiniTable("users", "id", "name", "region");
        users.add("1", "ada", "eu");
        users.add("2", "bob", "us");
        users.add("3", null, "eu");
        users.add("4", "dave", null);
        users.add("4", "dave", null);          // duplicate row on purpose
        MiniTable orders = new MiniTable("orders", "id", "user_id", "amount", "status");
        orders.add("10", "1", "100", "paid");
        orders.add("11", "1", "250", "open");
        orders.add("12", "2", null, "open");
        orders.add("13", null, "70", "paid");
        orders.add("14", "4", "15", "open");
        orders.add("15", "9", "500", "paid");
        orders.add("16", "2", "250", null);
        return new MiniTable[]{users, orders};
    }

    // ------------------------------------------------------------------ evaluators

    static String value(String v) { return v == null ? "NULL" : v; }

    /** Aggregation written the naive way: keep the group members in a list and scan for every aggregate. */
    static List<String> aggregateScan(List<String[]> groups, int groupColumn, List<Agg> aggregates) {
        Map<String, List<String[]>> byGroup = new TreeMap<>();
        for (String[] tuple : groups) {
            String key = groupColumn < 0 ? "" : value(tuple[groupColumn]);
            byGroup.computeIfAbsent(key, k -> new ArrayList<>()).add(tuple);
        }
        if (byGroup.isEmpty() && groupColumn < 0) byGroup.put("", new ArrayList<>());
        List<String> lines = new ArrayList<>();
        for (Map.Entry<String, List<String[]>> entry : byGroup.entrySet()) {
            StringBuilder sb = new StringBuilder(entry.getKey());
            for (Agg a : aggregates) sb.append("|").append(a.fn).append("=").append(computeAgg(a, entry.getValue()));
            lines.add(sb.toString());
        }
        Collections.sort(lines);
        return lines;
    }

    static String computeAgg(Agg a, List<String[]> group) {
        if (a.fn.equals("COUNT") && a.column < 0) return String.valueOf(group.size());
        int nonNull = 0;
        long sum = 0;
        long count = 0;
        String best = null;
        for (String[] tuple : group) {
            String v = tuple[a.column];
            if (v == null) continue;
            nonNull++;
            count++;
            if (a.fn.equals("SUM") || a.fn.equals("AVG")) sum += Long.parseLong(v);
            if (a.fn.equals("MIN") && (best == null || v.compareTo(best) < 0)) best = v;
            if (a.fn.equals("MAX") && (best == null || v.compareTo(best) > 0)) best = v;
        }
        switch (a.fn) {
            case "COUNT": return String.valueOf(nonNull);
            case "SUM": return count == 0 ? "NULL" : String.valueOf(sum);
            case "AVG": return count == 0 ? "NULL" : String.valueOf(sum / count);
            case "MIN":
            case "MAX": return best == null ? "NULL" : best;
            default: throw new IllegalArgumentException("unknown aggregate " + a.fn);
        }
    }

    /** Incremental hash aggregation - a genuinely different implementation of the same semantics. */
    static List<String> aggregateHash(List<String[]> joined, int groupColumn, List<Agg> aggregates) {
        Map<String, long[]> state = new HashMap<>();      // per aggregate: [sum|rows, non-null count]
        Map<String, String> best = new HashMap<>();       // MIN/MAX best value per (group, aggregate)
        Map<String, Integer> rowCounts = new HashMap<>();
        List<String> keys = new ArrayList<>();
        for (String[] tuple : joined) {
            String key = groupColumn < 0 ? "" : value(tuple[groupColumn]);
            long[] slot = state.get(key);
            if (slot == null) {
                slot = new long[aggregates.size() * 2];
                state.put(key, slot);
                keys.add(key);
            }
            rowCounts.merge(key, 1, Integer::sum);
            for (int i = 0; i < aggregates.size(); i++) {
                Agg a = aggregates.get(i);
                if (a.column < 0) { slot[2 * i]++; continue; }        // COUNT(*)
                String v = tuple[a.column];
                if (v == null) continue;                              // NULL never feeds an aggregate
                slot[2 * i + 1]++;
                if (a.fn.equals("SUM") || a.fn.equals("AVG")) slot[2 * i] += Long.parseLong(v);
                if (a.fn.equals("MIN") || a.fn.equals("MAX")) {
                    String current = best.get(key + "#" + i);
                    if (current == null || (a.fn.equals("MIN") ? v.compareTo(current) < 0
                            : v.compareTo(current) > 0)) {
                        best.put(key + "#" + i, v);
                    }
                }
            }
        }
        if (keys.isEmpty() && groupColumn < 0) {
            keys.add("");
            state.put("", new long[aggregates.size() * 2]);
        }
        List<String> lines = new ArrayList<>();
        for (String key : keys) {
            long[] slot = state.get(key);
            StringBuilder sb = new StringBuilder(key);
            for (int i = 0; i < aggregates.size(); i++) {
                Agg a = aggregates.get(i);
                long nonNull = slot[2 * i + 1];
                sb.append("|").append(a.fn).append("=");
                switch (a.fn) {
                    case "COUNT": sb.append(a.column < 0 ? rowCounts.getOrDefault(key, 0) : nonNull); break;
                    case "SUM": sb.append(nonNull == 0 ? "NULL" : String.valueOf(slot[2 * i])); break;
                    case "AVG": sb.append(nonNull == 0 ? "NULL" : String.valueOf(slot[2 * i] / nonNull)); break;
                    default: {
                        String b = best.get(key + "#" + i);
                        sb.append(b == null ? "NULL" : b);
                    }
                }
            }
            lines.add(sb.toString());
        }
        Collections.sort(lines);
        return lines;
    }

    static String[] merge(String[] a, String[] b) {
        String[] tuple = new String[a.length + b.length];
        System.arraycopy(a, 0, tuple, 0, a.length);
        System.arraycopy(b, 0, tuple, a.length, b.length);
        return tuple;
    }

    /** Full cross product, predicate evaluated on every combination. */
    static List<String> evaluateBrute(Query q, MiniTable left, MiniTable right, Work work) {
        List<String[]> joined = new ArrayList<>();
        for (String[] a : left.rows) {
            for (String[] b : right.rows) {
                work.rowsVisited++;
                if (a[q.leftKey] == null || b[q.rightKey] == null) continue;   // NULL never joins
                if (!a[q.leftKey].equals(b[q.rightKey])) continue;
                String[] tuple = merge(a, b);
                if (q.where().eval(tuple) == Tri.TRUE) joined.add(tuple);
            }
        }
        return aggregateScan(joined, q.groupBy, q.aggregates);
    }

    /** Nested loop join written separately (predicate tested in the inner loop, same semantics). */
    static List<String> evaluateReference(Query q, MiniTable left, MiniTable right, Work work) {
        List<String[]> joined = new ArrayList<>();
        int rightWidth = right.columns.length;
        for (String[] a : left.rows) {
            for (String[] b : right.rows) {
                work.rowsVisited++;
                if (!Objects.equals(a[q.leftKey], b[q.rightKey]) || a[q.leftKey] == null) continue;
                String[] tuple = new String[a.length + rightWidth];
                for (int i = 0; i < a.length; i++) tuple[i] = a[i];
                for (int i = 0; i < rightWidth; i++) tuple[a.length + i] = b[i];
                if (q.where().eval(tuple) != Tri.TRUE) continue;
                joined.add(tuple);
            }
        }
        return aggregateScan(joined, q.groupBy, q.aggregates);
    }

    /** Hash join over a per-key bucket index plus incremental hash aggregation. */
    static List<String> evaluateHash(Query q, MiniTable left, MiniTable right, Work work, boolean nullKeyBug) {
        Map<String, List<String[]>> index = new HashMap<>();
        for (String[] b : right.rows) {
            String key = b[q.rightKey] == null ? (nullKeyBug ? "\u0000NULL" : null) : b[q.rightKey];
            if (key == null) continue;
            index.computeIfAbsent(key, k -> new ArrayList<>()).add(b);
        }
        List<String[]> joined = new ArrayList<>();
        for (String[] a : left.rows) {
            String key = a[q.leftKey] == null ? (nullKeyBug ? "\u0000NULL" : null) : a[q.leftKey];
            if (key == null) continue;
            List<String[]> bucket = index.get(key);
            if (bucket == null) continue;
            for (String[] b : bucket) {
                work.rowsVisited++;
                work.tupleComparisons++;
                String[] tuple = merge(a, b);
                if (q.where().eval(tuple) != Tri.TRUE) continue;
                joined.add(tuple);
            }
        }
        return aggregateHash(joined, q.groupBy, q.aggregates);
    }

    static List<String> evaluateOptimized(Query q, MiniTable left, MiniTable right, Work work) {
        return evaluateHash(q, left, right, work, false);
    }

    static List<String> evaluateBuggy(Query q, MiniTable left, MiniTable right) {
        return evaluateHash(q, left, right, new Work(), true);
    }

    static boolean mismatch(Query q, MiniTable left, MiniTable right) {
        List<String> good = evaluateOptimized(q, left, right, new Work());
        List<String> bad = evaluateBuggy(q, left, right);
        return !good.equals(bad);
    }

    // ------------------------------------------------------------------ the fuzzer

    static final String[] REGIONS = {"eu", "us", "apac"};
    static final String[] STATUS = {"paid", "open", "cancelled"};

    /** Grammar-driven generation; every feature flag is recorded so coverage can be asserted. */
    static Query generate(SplitMix64 rng, MiniTable users, MiniTable orders, Set<String> features, int variant) {
        boolean selfJoin = variant % 4 == 3;
        MiniTable right = selfJoin ? users : orders;
        int leftKey;
        int rightKey;
        if (selfJoin) {
            // self joins use a nullable column so that NULL-key handling is actually exercised
            String column = variant % 8 == 3 ? "name" : "region";
            leftKey = users.indexOf(column);
            rightKey = users.indexOf(column);
            features.add("self-join");
            features.add("nullable-key");
        } else {
            leftKey = users.indexOf("id");
            rightKey = right.indexOf("user_id");
        }
        int width = users.columns.length + right.columns.length;
        int firstRight = users.columns.length;
        int amountIndex = right.indexOrNull("amount");
        int numericRight = amountIndex >= 0 ? amountIndex : right.indexOf("id");

        List<Expr> conjuncts = new ArrayList<>();
        int conjunctCount = rng.nextInt(4);
        for (int i = 0; i < conjunctCount; i++) {
            int pick = rng.nextInt(6);
            if (pick == 0) {
                conjuncts.add(Expr.isNull(rng.nextInt(width)));
                features.add("is-null");
            } else if (pick == 1) {
                conjuncts.add(Expr.isNotNull(rng.nextInt(width)));
                features.add("is-not-null");
            } else if (pick == 2 && !conjuncts.isEmpty()) {
                conjuncts.add(Expr.not(conjuncts.get(rng.nextInt(conjuncts.size()))));
                features.add("negation");
            } else if (pick == 3) {
                Expr a = Expr.cmp(rng.nextInt(width), rng.nextBoolean() ? "=" : "<>", REGIONS[rng.nextInt(3)]);
                Expr b = Expr.cmp(rng.nextInt(width), rng.nextBoolean() ? "=" : "<>", STATUS[rng.nextInt(3)]);
                conjuncts.add(rng.nextBoolean() ? Expr.and(a, b) : Expr.or(a, b));
                features.add("boolean-combinator");
            } else {
                int column = rng.nextInt(width);
                String op = new String[]{"=", "<>", "<", ">"}[rng.nextInt(4)];
                String literal = new String[]{"1", "2", "4", "10", "100", "eu", "us", "paid"}[rng.nextInt(8)];
                conjuncts.add(Expr.cmp(column, op, literal));
                features.add("comparison");
            }
        }
        int groupBy = -1;
        if (rng.chance(55)) {
            groupBy = rng.nextInt(width);
            features.add("group-by");
        } else {
            features.add("no-group-by");
        }
        List<Agg> aggregates = new ArrayList<>();
        int aggCount = 1 + rng.nextInt(3);
        for (int i = 0; i < aggCount; i++) {
            String fn = new String[]{"COUNT", "SUM", "MIN", "MAX", "AVG"}[rng.nextInt(5)];
            int column;
            if (fn.equals("COUNT") && rng.chance(40)) {
                column = -1;                       // COUNT(*)
                features.add("count-star");
            } else {
                column = firstRight + rng.nextInt(right.columns.length);
            }
            if (fn.equals("SUM") || fn.equals("AVG")) column = firstRight + numericRight;
            aggregates.add(new Agg(fn, column));
            features.add(fn.toLowerCase());
        }
        for (Expr e : conjuncts) if (e.kind.equals("CMP") && e.literal.matches("\\d+")) features.add("numeric-literal");
        return new Query(leftKey, rightKey, conjuncts, groupBy, aggregates);
    }

    static String corpusDigest(List<Query> corpus) {
        StringBuilder sb = new StringBuilder();
        for (Query q : corpus) sb.append(q.describe()).append('\n');
        return Long.toHexString(Digest.of(sb.toString()));
    }

    // ------------------------------------------------------------------ shrinking

    /** Delta debugging: drop rows and conjuncts while the buggy engine still misbehaves. */
    static Object[] shrink(Query q, MiniTable left, MiniTable right) {
        Query current = q;
        MiniTable bestLeft = left;
        MiniTable bestRight = right;
        boolean changed = true;
        while (changed) {
            changed = false;
            List<Expr> working = new ArrayList<>(current.conjuncts);
            for (int i = 0; i < working.size(); i++) {
                List<Expr> candidate = new ArrayList<>(working);
                candidate.remove(i);
                Query probe = current.withConjuncts(candidate);
                if (mismatch(probe, bestLeft, bestRight)) {
                    current = probe;
                    changed = true;
                    break;
                }
            }
            MiniTable rightMinus = new MiniTable(bestRight.name, bestRight.columns);
            for (int i = 0; i < bestRight.rows.size(); i++) {
                rightMinus.rows.clear();
                for (int j = 0; j < bestRight.rows.size(); j++) if (j != i) rightMinus.rows.add(bestRight.rows.get(j));
                if (mismatch(current, bestLeft, rightMinus)) {
                    bestRight = rightMinus;
                    changed = true;
                    break;
                }
            }
            MiniTable leftMinus = new MiniTable(bestLeft.name, bestLeft.columns);
            for (int i = 0; i < bestLeft.rows.size(); i++) {
                leftMinus.rows.clear();
                for (int j = 0; j < bestLeft.rows.size(); j++) if (j != i) leftMinus.rows.add(bestLeft.rows.get(j));
                if (mismatch(current, leftMinus, bestRight)) {
                    bestLeft = leftMinus;
                    changed = true;
                    break;
                }
            }
        }
        return new Object[]{current, bestLeft, bestRight};
    }

    /** 1-minimality: every single removal must make the mismatch disappear. */
    static boolean oneMinimal(Query q, MiniTable left, MiniTable right) {
        for (int i = 0; i < q.conjuncts.size(); i++) {
            List<Expr> candidate = new ArrayList<>(q.conjuncts);
            candidate.remove(i);
            if (mismatch(q.withConjuncts(candidate), left, right)) return false;
        }
        for (int i = 0; i < left.rows.size(); i++) {
            MiniTable smaller = new MiniTable(left.name, left.columns);
            for (int j = 0; j < left.rows.size(); j++) if (j != i) smaller.rows.add(left.rows.get(j));
            if (mismatch(q, smaller, right)) return false;
        }
        for (int i = 0; i < right.rows.size(); i++) {
            MiniTable smaller = new MiniTable(right.name, right.columns);
            for (int j = 0; j < right.rows.size(); j++) if (j != i) smaller.rows.add(right.rows.get(j));
            if (mismatch(q, left, smaller)) return false;
        }
        return true;
    }

    /** Independent aggregate oracle: enumerate every group member and recompute without shared code paths. */
    static List<String> oracleAggregate(Query q, MiniTable left, MiniTable right) {
        List<String[]> joined = new ArrayList<>();
        for (String[] a : left.rows) {
            for (String[] b : right.rows) {
                if (a[q.leftKey] == null || b[q.rightKey] == null) continue;
                if (!a[q.leftKey].equals(b[q.rightKey])) continue;
                String[] tuple = merge(a, b);
                if (q.where().eval(tuple) == Tri.TRUE) joined.add(tuple);
            }
        }
        Map<String, List<String[]>> groups = new TreeMap<>();
        for (String[] tuple : joined) {
            groups.computeIfAbsent(q.groupBy < 0 ? "" : value(tuple[q.groupBy]), k -> new ArrayList<>()).add(tuple);
        }
        if (groups.isEmpty() && q.groupBy < 0) groups.put("", new ArrayList<>());
        List<String> lines = new ArrayList<>();
        for (Map.Entry<String, List<String[]>> g : groups.entrySet()) {
            StringBuilder sb = new StringBuilder(g.getKey());
            for (Agg a : q.aggregates) {
                sb.append("|").append(a.fn).append("=");
                List<String> values = new ArrayList<>();
                for (String[] tuple : g.getValue()) if (a.column >= 0 && tuple[a.column] != null) values.add(tuple[a.column]);
                switch (a.fn) {
                    case "COUNT": sb.append(a.column < 0 ? g.getValue().size() : values.size()); break;
                    case "SUM": {
                        long sum = 0;
                        for (String v : values) sum += Long.parseLong(v);
                        sb.append(values.isEmpty() ? "NULL" : String.valueOf(sum));
                        break;
                    }
                    case "AVG": {
                        long sum = 0;
                        for (String v : values) sum += Long.parseLong(v);
                        sb.append(values.isEmpty() ? "NULL" : String.valueOf(sum / values.size()));
                        break;
                    }
                    case "MIN": {
                        String best = null;
                        for (String v : values) if (best == null || v.compareTo(best) < 0) best = v;
                        sb.append(best == null ? "NULL" : best);
                        break;
                    }
                    default: {
                        String best = null;
                        for (String v : values) if (best == null || v.compareTo(best) > 0) best = v;
                        sb.append(best == null ? "NULL" : best);
                    }
                }
            }
            lines.add(sb.toString());
        }
        Collections.sort(lines);
        return lines;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        MiniTable[] data = buildData(854);
        // split so that indices can be validated by the caller: users first, orders second.
        MiniTable users = data[0];
        MiniTable orders = data[1];
        check(users.rows.size() == 5 && orders.rows.size() == 7, "fixture sizes");

        // --- 1. three-valued logic tables against their definitions, exhaustively.
        for (Tri a : Tri.values()) {
            for (Tri b : Tri.values()) {
                Tri expectedAnd = (a == Tri.FALSE || b == Tri.FALSE) ? Tri.FALSE
                        : (a == Tri.TRUE && b == Tri.TRUE) ? Tri.TRUE : Tri.UNKNOWN;
                Tri expectedOr = (a == Tri.TRUE || b == Tri.TRUE) ? Tri.TRUE
                        : (a == Tri.FALSE && b == Tri.FALSE) ? Tri.FALSE : Tri.UNKNOWN;
                check(Logic.and(a, b) == expectedAnd, "AND(" + a + "," + b + ")");
                check(Logic.or(a, b) == expectedOr, "OR(" + a + "," + b + ")");
            }
            Tri expectedNot = a == Tri.TRUE ? Tri.FALSE : a == Tri.FALSE ? Tri.TRUE : Tri.UNKNOWN;
            check(Logic.not(a) == expectedNot, "NOT(" + a + ")");
        }
        check(Logic.and(Tri.UNKNOWN, Tri.FALSE) == Tri.FALSE, "FALSE dominates UNKNOWN in AND");
        check(Logic.or(Tri.UNKNOWN, Tri.TRUE) == Tri.TRUE, "TRUE dominates UNKNOWN in OR");

        // --- 2. NULL keys never join, on any evaluator (self join on the nullable name column).
        int nameColumn = users.indexOf("name");
        Query nullJoin = new Query(nameColumn, nameColumn, List.of(), -1, List.of(new Agg("COUNT", -1)));
        List<String> nullJoinExpected = List.of("|COUNT=6");
        check(evaluateBrute(nullJoin, users, users, new Work()).equals(nullJoinExpected),
                "the cross product counts six equal non-NULL name pairs");
        check(evaluateReference(nullJoin, users, users, new Work()).equals(nullJoinExpected),
                "the nested loop engine agrees");
        check(evaluateOptimized(nullJoin, users, users, new Work()).equals(nullJoinExpected),
                "the hash join engine agrees");
        check(!evaluateBuggy(nullJoin, users, users).equals(nullJoinExpected),
                "the buggy hash join matches NULL keys and disagrees: " + evaluateBuggy(nullJoin, users, users));

        // --- 3. coverage of the grammar, and determinism of the corpus.
        Set<String> features = new TreeSet<>();
        List<Query> corpus = new ArrayList<>();
        SplitMix64 rng = new SplitMix64(854_2024L);
        for (int i = 0; i < 300; i++) corpus.add(generate(rng, users, orders, features, i));
        System.out.println("corpus of " + corpus.size() + " queries, features=" + features);
        for (String required : List.of("comparison", "is-null", "is-not-null", "boolean-combinator", "negation",
                "group-by", "no-group-by", "count-star", "count", "sum", "min", "max", "avg", "numeric-literal",
                "self-join")) {
            check(features.contains(required), "corpus covers feature " + required);
        }
        String digest = corpusDigest(corpus);
        Set<String> featuresAgain = new TreeSet<>();
        List<Query> corpusAgain = new ArrayList<>();
        SplitMix64 rngAgain = new SplitMix64(854_2024L);
        for (int i = 0; i < 300; i++) corpusAgain.add(generate(rngAgain, users, orders, featuresAgain, i));
        check(digest.equals(corpusDigest(corpusAgain)), "the same seed reproduces the corpus digest");
        Set<String> otherFeatures = new TreeSet<>();
        List<Query> corpusOther = new ArrayList<>();
        SplitMix64 rngOther = new SplitMix64(854_9999L);
        for (int i = 0; i < 300; i++) corpusOther.add(generate(rngOther, users, orders, otherFeatures, i));
        check(!digest.equals(corpusDigest(corpusOther)), "a different seed produces a different corpus");

        // --- 4. differential agreement among the three correct engines over the whole corpus.
        int disagreements = 0, aggrementWithOracle = 0, detectedBugs = 0, emptyResults = 0;
        for (Query q : corpus) {
            List<String> brute = evaluateBrute(q, users, orders, new Work());
            List<String> reference = evaluateReference(q, users, orders, new Work());
            List<String> optimized = evaluateOptimized(q, users, orders, new Work());
            List<String> oracle = oracleAggregate(q, users, orders);
            if (!brute.equals(reference) || !brute.equals(optimized)) disagreements++;
            if (brute.equals(oracle)) aggrementWithOracle++;
            if (!optimized.equals(evaluateBuggy(q, users, orders))) detectedBugs++;
            if (brute.isEmpty() || brute.equals(List.of("|COUNT=0"))) emptyResults++;
        }
        System.out.printf("corpus results: disagreements=%d oracleAgreement=%d buggyDetections=%d emptyResults=%d%n",
                disagreements, aggrementWithOracle, detectedBugs, emptyResults);
        check(disagreements == 0, "the three correct engines agree on every query");
        check(aggrementWithOracle == corpus.size(), "the naive scan aggregation matches the oracle everywhere");
        check(detectedBugs > 0, "the fuzzer detects the NULL-key defect on " + detectedBugs + " queries");
        check(emptyResults > 0, "the corpus exercises empty results (" + emptyResults + ")");

        // --- 5. shrinking a failing case to a 1-minimal witness.
        Query failing = null;
        for (Query q : corpus) {
            if (mismatch(q, users, orders)) { failing = q; break; }
        }
        check(failing != null, "a failing query exists in the corpus");
        Object[] shrunk = shrink(failing, users.copy(), orders.copy());
        Query witness = (Query) shrunk[0];
        MiniTable witnessUsers = (MiniTable) shrunk[1];
        MiniTable witnessOrders = (MiniTable) shrunk[2];
        System.out.printf("witness: %s rows(users=%d, orders=%d) conjuncts=%d%n", witness.describe(),
                witnessUsers.rows.size(), witnessOrders.rows.size(), witness.conjuncts.size());
        check(mismatch(witness, witnessUsers, witnessOrders), "the minimised witness still fails");
        check(oneMinimal(witness, witnessUsers, witnessOrders),
                "the witness is 1-minimal (no single removal keeps the bug alive)");
        check(witnessUsers.rows.size() + witnessOrders.rows.size() <= 8,
                "the witness is small: " + (witnessUsers.rows.size() + witnessOrders.rows.size()) + " rows");
        check(witness.conjuncts.size() <= 4, "the witness keeps at most four conjuncts");

        // the witness must be a genuine NULL-key counterexample, and reproducible.
        boolean nullKeyPresent = false;
        for (String[] row : witnessUsers.rows) if (row[witness.leftKey] == null) nullKeyPresent = true;
        for (String[] row : witnessOrders.rows) if (row[witness.rightKey] == null) nullKeyPresent = true;
        check(nullKeyPresent, "the witness involves a NULL join key");
        List<String> reproducible = evaluateBuggy(witness, witnessUsers, witnessOrders);
        check(reproducible.equals(evaluateBuggy(witness, witnessUsers, witnessOrders)),
                "the witness reproduces deterministically");
        check(evaluateReference(witness, witnessUsers, witnessOrders, new Work())
                        .equals(evaluateOptimized(witness, witnessUsers, witnessOrders, new Work())),
                "the correct engines still agree on the witness");
        check(!evaluateBrute(witness, witnessUsers, witnessOrders, new Work()).equals(reproducible),
                "the witness is a genuine counterexample for the buggy engine");

        // --- 6. cost: the hash join does strictly less work than the cross product.
        Query heavy = new Query(users.indexOf("id"), orders.indexOf("user_id"),
                List.of(Expr.isNotNull(users.columns.length + orders.indexOf("amount"))), -1,
                List.of(new Agg("SUM", users.columns.length + orders.indexOf("amount"))));
        Work bruteWork = new Work();
        Work hashWork = new Work();
        List<String> bruteResult = evaluateBrute(heavy, users, orders, bruteWork);
        List<String> hashResult = evaluateOptimized(heavy, users, orders, hashWork);
        System.out.printf("work: crossProduct=%d hashJoin=%d%n", bruteWork.rowsVisited, hashWork.rowsVisited);
        check(bruteResult.equals(hashResult), "both engines agree on the aggregate query");
        check(hashWork.rowsVisited < bruteWork.rowsVisited, "the hash join visits fewer tuples");
        check(hashWork.tupleComparisons > 0, "the hash join still compares keys");

        // --- 7. degenerate inputs handled identically by all three correct engines.
        MiniTable emptyUsers = new MiniTable("users", "id", "name", "region");
        MiniTable emptyOrders = new MiniTable("orders", "id", "user_id", "amount", "status");
        for (Query q : List.of(
                new Query(0, 1, List.of(), -1, List.of(new Agg("COUNT", -1))),
                new Query(0, 1, List.of(), 1, List.of(new Agg("SUM", 2 + 2))),
                new Query(0, 1, List.of(Expr.isNull(0)), -1, List.of(new Agg("COUNT", -1), new Agg("MAX", 2))))) {
            List<String> brute = evaluateBrute(q, emptyUsers, emptyOrders, new Work());
            List<String> reference = evaluateReference(q, emptyUsers, emptyOrders, new Work());
            List<String> optimized = evaluateOptimized(q, emptyUsers, emptyOrders, new Work());
            check(brute.equals(reference) && brute.equals(optimized),
                    "empty tables agree across engines for " + q.describe());
        }
        Query noGroupAggregate = new Query(0, 1, List.of(), -1, List.of(new Agg("COUNT", -1)));
        check(evaluateBrute(noGroupAggregate, emptyUsers, emptyOrders, new Work()).equals(List.of("|COUNT=0")),
                "COUNT without GROUP BY over no rows returns one zero row");
        Query grouped = new Query(0, 1, List.of(), 1, List.of(new Agg("COUNT", -1)));
        check(evaluateBrute(grouped, emptyUsers, emptyOrders, new Work()).isEmpty(),
                "GROUP BY over no rows returns no rows");

        MiniTable single = new MiniTable("users", "id", "name", "region");
        single.add("7", "erin", "eu");
        MiniTable singleOrder = new MiniTable("orders", "id", "user_id", "amount", "status");
        singleOrder.add("70", "7", "5", "paid");
        Query singleQuery = new Query(single.indexOf("id"), singleOrder.indexOf("user_id"), List.of(), -1,
                List.of(new Agg("SUM", single.columns.length + singleOrder.indexOf("amount"))));
        check(evaluateBrute(singleQuery, single, singleOrder, new Work()).equals(List.of("|SUM=5")),
                "single-row tables aggregate correctly");
        MiniTable allNull = new MiniTable("orders", "id", "user_id", "amount", "status");
        allNull.add("80", "7", null, "paid");
        Query nullAmount = new Query(single.indexOf("id"), allNull.indexOf("user_id"), List.of(), -1,
                List.of(new Agg("SUM", single.columns.length + allNull.indexOf("amount")),
                        new Agg("COUNT", single.columns.length + allNull.indexOf("amount"))));
        check(evaluateBrute(nullAmount, single, allNull, new Work()).equals(List.of("|SUM=NULL|COUNT=0")),
                "SUM over an all-NULL column is NULL and COUNT(col) is zero");

        // --- 8. result comparison is invariant under the stored row order, and duplicates are kept.
        MiniTable shuffledUsers = new MiniTable("users", users.columns);
        for (int i = users.rows.size() - 1; i >= 0; i--) shuffledUsers.rows.add(users.rows.get(i).clone());
        MiniTable shuffledOrders = new MiniTable("orders", orders.columns);
        shuffledOrders.rows.add(orders.rows.get(3).clone());
        shuffledOrders.rows.add(orders.rows.get(0).clone());
        for (int i = 0; i < orders.rows.size(); i++) {
            if (i == 0 || i == 3) continue;
            shuffledOrders.rows.add(orders.rows.get(i).clone());
        }
        Query invariance = new Query(users.indexOf("id"), orders.indexOf("user_id"), List.of(), 0,
                List.of(new Agg("COUNT", -1)));
        check(evaluateBrute(invariance, users, orders, new Work())
                        .equals(evaluateBrute(invariance, shuffledUsers, shuffledOrders, new Work())),
                "the result is invariant under row order");
        check(users.rows.size() == 5 && users.copy().rows.size() == 5, "duplicate rows survive copying");
        List<String> duplicateResult = evaluateBrute(
                new Query(users.indexOf("id"), users.indexOf("id"), List.of(), 0, List.of(new Agg("COUNT", -1))),
                users, users, new Work());
        check(duplicateResult.equals(List.of("1|COUNT=1", "2|COUNT=1", "3|COUNT=1", "4|COUNT=4")),
                "the duplicated user row joins every copy: " + duplicateResult);
        check(oracleAggregate(invariance, users, orders).equals(evaluateBrute(invariance, users, orders, new Work())),
                "the aggregate oracle agrees on the grouped query");

        System.out.println("corpus digest=" + digest + " witness=" + witness.describe());
        System.out.println("checks=" + checks);
        System.out.println("PASS");
    }
}
