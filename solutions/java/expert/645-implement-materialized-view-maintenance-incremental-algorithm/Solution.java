/** Task 645 - Implement materialized view maintenance incremental algorithm. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * Two materialised views are kept incrementally and cross-checked against a full recompute after every
 * single delta:
 *
 *   V1 = SELECT dept, COUNT(*), SUM(salary) FROM emp WHERE active GROUP BY dept
 *        - an insert adds one to a group (creating it when absent), a delete subtracts one and, when
 *          the group's count reaches zero, removes the group. That last step is the classic "drift"
 *          case: a maintenance routine which only adjusts the SUM leaves a phantom group with
 *          count = 0 behind, and the check demonstrates exactly that by running a deliberately naive
 *          variant and showing it disagrees with the recompute while the real one does not.
 *
 *   V2 = emp (active) JOIN dept ON emp.dept = dept.id, materialised with duplicate multiplicities
 *        - a delete on the many side removes one derived tuple, while an insert or delete on the one
 *          side fans out to every matching employee (and duplicate department ids fan out
 *          independently), which is the delta rule that a one-table-at-a-time view cannot express.
 *
 * The invariant verified is the only one that matters for view maintenance: after *every* delta, the
 * incrementally maintained view is identical to the full recompute from the base tables - checked on
 * hand-worked cases, on a batch delta, on updates modelled as delete + insert, on 400 generated random
 * deltas, and on the empty/degenerate cases. Counters record how many view tuples each delta touched,
 * so the incremental cost can be compared with the recompute's full scan.
 *
 * Simplified (deliberately): the tables and the view are in-memory maps, there is no transaction log,
 * no compensation/undo of a delta, no recursive or nested views, and no concurrency.
 */
import java.util.*;

class GroupByView {

    static final class Row {
        final int id, dept, salary;
        final boolean active;

        Row(int id, int dept, int salary, boolean active) {
            this.id = id; this.dept = dept; this.salary = salary; this.active = active;
        }

        @Override
        public String toString() {
            return "emp(" + id + "," + dept + "," + salary + (active ? ",active" : ",inactive") + ")";
        }
    }

    static final class Group {
        final long count, sum;

        Group(long count, long sum) { this.count = count; this.sum = sum; }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof Group)) return false;
            Group that = (Group) other;
            return count == that.count && sum == that.sum;
        }

        @Override
        public int hashCode() { return (int) (count * 31 + sum); }

        @Override
        public String toString() { return "{count=" + count + ", sum=" + sum + "}"; }
    }

    final Map<Integer, Group> view = new TreeMap<>();
    final List<Row> base = new ArrayList<>();
    long viewTuplesTouched;
    long deltasApplied;

    void insert(Row row) {
        base.add(row);
        deltasApplied++;
        if (!row.active) return;
        Group current = view.get(row.dept);
        long count = current == null ? 1 : current.count + 1;
        long sum = current == null ? row.salary : current.sum + row.salary;
        view.put(row.dept, new Group(count, sum));
        viewTuplesTouched++;
    }

    /** Returns false when the row is not in the base table (a delta against nothing). */
    boolean delete(Row row) {
        if (!base.remove(row)) return false;
        deltasApplied++;
        if (!row.active) return true;
        Group current = view.get(row.dept);
        if (current == null) throw new IllegalStateException("view drift: group " + row.dept + " missing");
        viewTuplesTouched++;
        long count = current.count - 1;
        long sum = current.sum - row.salary;
        if (count == 0) {
            view.remove(row.dept);
        } else {
            view.put(row.dept, new Group(count, sum));
        }
        return true;
    }

    /** The maintenance a naive implementation would do: adjust the aggregate, never remove a group. */
    Map<Integer, Group> naiveDeleteKeepingEmptyGroups(Row row) {
        Map<Integer, Group> drifted = new TreeMap<>(view);
        Group current = drifted.get(row.dept);
        drifted.put(row.dept, new Group(current.count - 1, current.sum - row.salary));
        return drifted;
    }

    Map<Integer, Group> recompute() {
        Map<Integer, Group> out = new TreeMap<>();
        for (Row row : base) {
            if (!row.active) continue;
            Group current = out.get(row.dept);
            out.put(row.dept, new Group(current == null ? 1 : current.count + 1,
                    current == null ? row.salary : current.sum + row.salary));
        }
        return out;
    }

    boolean consistent() { return view.equals(recompute()); }
}

class JoinView {

    static final class Dept {
        final int id, name;

        Dept(int id, int name) { this.id = id; this.name = name; }
    }

    static final class Emp {
        final int id, dept, salary;
        final boolean active;

        Emp(int id, int dept, int salary, boolean active) {
            this.id = id; this.dept = dept; this.salary = salary; this.active = active;
        }
    }

    final List<Dept> depts = new ArrayList<>();
    final List<Emp> emps = new ArrayList<>();
    final TreeMap<String, Long> view = new TreeMap<>();
    long viewTuplesTouched;

    static String key(Dept dept, Emp emp) {
        return dept.id + ":" + dept.name + ":" + emp.id;
    }

    void add(String key) { view.merge(key, 1L, Long::sum); viewTuplesTouched++; }

    void remove(String key) {
        Long count = view.get(key);
        if (count == null) throw new IllegalStateException("view drift: missing " + key);
        viewTuplesTouched++;
        if (count == 1) view.remove(key);
        else view.put(key, count - 1);
    }

    void insertEmp(Emp emp) {
        emps.add(emp);
        if (!emp.active) return;
        for (Dept dept : depts) if (dept.id == emp.dept) add(key(dept, emp));
    }

    boolean deleteEmp(Emp emp) {
        if (!emps.remove(emp)) return false;
        if (!emp.active) return true;
        for (Dept dept : depts) if (dept.id == emp.dept) remove(key(dept, emp));
        return true;
    }

    /** Fan-out on the one side: every active employee of that department gains a view tuple. */
    void insertDept(Dept dept) {
        depts.add(dept);
        for (Emp emp : emps) if (emp.active && emp.dept == dept.id) add(key(dept, emp));
    }

    boolean deleteDept(Dept dept) {
        if (!depts.remove(dept)) return false;
        for (Emp emp : emps) if (emp.active && emp.dept == dept.id) remove(key(dept, emp));
        return true;
    }

    TreeMap<String, Long> recompute() {
        TreeMap<String, Long> out = new TreeMap<>();
        for (Emp emp : emps) {
            if (!emp.active) continue;
            for (Dept dept : depts) {
                if (dept.id == emp.dept) out.merge(key(dept, emp), 1L, Long::sum);
            }
        }
        return out;
    }

    boolean consistent() { return view.equals(recompute()); }
}

class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

public class Solution {

    public static void main(String[] args) {
        GroupByView view = new GroupByView();
        GroupByView.Row a1 = new GroupByView.Row(1, 10, 100, true);
        GroupByView.Row a2 = new GroupByView.Row(2, 10, 50, true);
        GroupByView.Row b1 = new GroupByView.Row(3, 20, 30, true);
        GroupByView.Row b2 = new GroupByView.Row(4, 20, 20, false);
        view.insert(a1);
        view.insert(a2);
        view.insert(b1);
        view.insert(b2);
        check(view.view.toString().equals("{10={count=2, sum=150}, 20={count=1, sum=30}}"),
                "initial view " + view.view);
        check(view.consistent(), "initial view equals the recompute");

        GroupByView.Row b3 = new GroupByView.Row(5, 20, 70, true);
        view.insert(b3);
        check(view.view.get(20).equals(new GroupByView.Group(2, 100)), "group 20 updated: " + view.view.get(20));
        check(view.consistent(), "insert equals the recompute");

        GroupByView.Row newGroup = new GroupByView.Row(6, 30, 999, true);
        view.insert(newGroup);
        check(view.view.containsKey(30) && view.view.get(30).equals(new GroupByView.Group(1, 999)),
                "a brand new group appears");
        check(view.consistent(), "new group equals the recompute");

        view.delete(b3);
        check(view.view.get(20).equals(new GroupByView.Group(1, 30)), "group 20 shrank: " + view.view.get(20));
        check(view.consistent(), "delete equals the recompute");

        view.delete(b1);
        check(!view.view.containsKey(20), "count reaching zero removes the group: " + view.view);
        check(view.consistent(), "no phantom group is left behind");
        GroupByView.Row resurrect = new GroupByView.Row(7, 20, 5, true);
        view.insert(resurrect);
        check(view.view.get(20).equals(new GroupByView.Group(1, 5)), "the group can come back cleanly");
        check(view.consistent(), "resurrected group equals the recompute");

        GroupByView.Row inactive = new GroupByView.Row(8, 10, 7777, false);
        long touchedBefore = view.viewTuplesTouched;
        view.insert(inactive);
        check(view.viewTuplesTouched == touchedBefore, "an inactive insert touches no view tuple");
        check(view.consistent(), "inactive insert leaves the view identical");
        check(view.delete(inactive), "the inactive row is deleted from the base table");
        check(view.consistent(), "inactive delete leaves the view identical");
        check(view.viewTuplesTouched == touchedBefore, "and still touches no view tuple");

        GroupByView.Row unknown = new GroupByView.Row(99, 10, 1, true);
        check(!view.delete(unknown), "deleting a row that is not there is rejected");
        check(view.consistent(), "a rejected delta changes nothing");

        Map<Integer, GroupByView.Group> drifted = view.naiveDeleteKeepingEmptyGroups(newGroup);
        view.delete(newGroup);
        check(view.consistent(), "the real implementation stayed consistent");
        check(!drifted.equals(view.recompute()),
                "the naive maintenance leaves drift: " + drifted + " vs " + view.recompute());
        view.insert(newGroup);
        check(view.consistent(), "re-inserting restores the state");

        GroupByView batch = new GroupByView();
        List<GroupByView.Row> rows = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            rows.add(new GroupByView.Row(i, i % 7, 10 + i, i % 3 != 0));
            batch.insert(rows.get(i));
        }
        check(batch.consistent(), "200 inserts keep the invariant");
        for (int i = 0; i < 200; i += 2) batch.delete(rows.get(i));
        check(batch.consistent(), "a mixed delete batch keeps the invariant");
        for (int i = 1; i < 200; i += 2) batch.delete(rows.get(i));
        check(batch.view.isEmpty() && batch.consistent(), "deleting everything empties the view");

        SplitMix64 rng = new SplitMix64(555111L);
        GroupByView randomView = new GroupByView();
        List<GroupByView.Row> live = new ArrayList<>();
        long violations = 0, activeDeltas = 0, cumulativeScan = 0;
        for (int step = 0; step < 400; step++) {
            boolean insert = live.isEmpty() || rng.nextInt(100) < 60;
            if (insert) {
                GroupByView.Row row = new GroupByView.Row(step, rng.nextInt(5), rng.nextInt(1000), rng.nextInt(10) > 0);
                randomView.insert(row);
                live.add(row);
                if (row.active) activeDeltas++;
            } else {
                GroupByView.Row row = live.remove(rng.nextInt(live.size()));
                randomView.delete(row);
                if (row.active) activeDeltas++;
            }
            if (!randomView.consistent()) violations++;
            cumulativeScan += randomView.base.size();
        }
        check(violations == 0, "random deltas never desynchronise the view: " + violations + " violations");
        check(randomView.consistent(), "the final random view equals the recompute");
        check(randomView.viewTuplesTouched == activeDeltas,
                "each active delta touched exactly one view tuple: " + randomView.viewTuplesTouched
                        + " vs " + activeDeltas);
        check(randomView.viewTuplesTouched * 20 < cumulativeScan,
                "incremental maintenance touches far fewer entries than repeated recomputes: "
                        + randomView.viewTuplesTouched + " vs " + cumulativeScan);

        JoinView joinView = new JoinView();
        JoinView.Dept sales = new JoinView.Dept(1, 100);
        JoinView.Dept support = new JoinView.Dept(2, 200);
        joinView.insertDept(sales);
        joinView.insertDept(support);
        JoinView.Emp e1 = new JoinView.Emp(1, 1, 50, true);
        JoinView.Emp e2 = new JoinView.Emp(2, 1, 60, true);
        JoinView.Emp e3 = new JoinView.Emp(3, 2, 70, true);
        JoinView.Emp e4 = new JoinView.Emp(4, 1, 80, false);
        joinView.insertEmp(e1);
        joinView.insertEmp(e2);
        joinView.insertEmp(e3);
        joinView.insertEmp(e4);
        check(joinView.view.size() == 3, "only active employees join: " + joinView.view);
        check(joinView.consistent(), "join view equals the recompute");

        long before = joinView.viewTuplesTouched;
        JoinView.Dept duplicate = new JoinView.Dept(1, 999);
        joinView.insertDept(duplicate);
        check(joinView.viewTuplesTouched - before == 2, "fan-out of two active employees: "
                + (joinView.viewTuplesTouched - before));
        check(joinView.consistent(), "duplicate department id fans out independently");
        joinView.deleteDept(duplicate);
        check(joinView.viewTuplesTouched - before == 4, "deleting the duplicate removes its own two tuples");
        check(joinView.consistent(), "and the view is back to three tuples: " + joinView.view.size());

        joinView.deleteEmp(e1);
        check(joinView.view.size() == 2, "deleting an employee removes one tuple");
        check(joinView.consistent(), "many-side delete equals the recompute");
        joinView.deleteDept(sales);
        check(joinView.view.size() == 1 && joinView.view.containsKey("2:200:3"),
                "one-side delete removes every derived tuple: " + joinView.view);
        check(joinView.consistent(), "fan-out delete equals the recompute");
        check(!joinView.deleteDept(sales), "deleting a missing department is rejected");
        joinView.insertDept(sales);
        check(joinView.view.size() == 2 && joinView.view.containsKey("1:100:2"),
                "re-inserting the department re-derives its surviving employee's tuple: " + joinView.view);
        check(joinView.consistent(), "re-inserted fan-out equals the recompute");

        SplitMix64 joinRng = new SplitMix64(24680L);
        JoinView randomJoin = new JoinView();
        List<JoinView.Dept> deptPool = new ArrayList<>();
        List<JoinView.Emp> empPool = new ArrayList<>();
        long joinViolations = 0, peakViewTuples = 0;
        for (int step = 0; step < 400; step++) {
            int choice = joinRng.nextInt(100);
            if (choice < 30 && deptPool.size() < 10) {
                JoinView.Dept dept = new JoinView.Dept(joinRng.nextInt(4), step);
                randomJoin.insertDept(dept);
                deptPool.add(dept);
            } else if (choice < 60 && empPool.size() < 40) {
                JoinView.Emp emp = new JoinView.Emp(step, joinRng.nextInt(4), joinRng.nextInt(100), joinRng.nextInt(5) > 0);
                randomJoin.insertEmp(emp);
                empPool.add(emp);
            } else if (!empPool.isEmpty()) {
                JoinView.Emp emp = empPool.remove(joinRng.nextInt(empPool.size()));
                randomJoin.deleteEmp(emp);
            } else if (!deptPool.isEmpty()) {
                JoinView.Dept dept = deptPool.remove(joinRng.nextInt(deptPool.size()));
                randomJoin.deleteDept(dept);
            }
            if (!randomJoin.consistent()) joinViolations++;
            peakViewTuples = Math.max(peakViewTuples, randomJoin.view.size());
        }
        check(joinViolations == 0, "random join deltas never desynchronise: " + joinViolations);
        check(randomJoin.consistent(), "the final join view equals the recompute");
        check(peakViewTuples > 20, "the random run really exercised the view, peak size " + peakViewTuples);

        JoinView empty = new JoinView();
        check(empty.recompute().isEmpty() && empty.view.isEmpty(), "empty base tables give an empty view");
        check(new GroupByView().recompute().isEmpty(), "empty employee table gives an empty group view");

        System.out.println("group view: " + view.view + " after " + view.deltasApplied
                + " deltas, " + view.viewTuplesTouched + " view tuples touched");
        System.out.println("random group deltas=400 violations=" + violations
                + " incrementalTouches=" + randomView.viewTuplesTouched + " cumulativeRecomputeScan="
                + cumulativeScan);
        System.out.println("random join deltas=400 violations=" + joinViolations
                + " finalViewTuples=" + randomJoin.view.size());
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
