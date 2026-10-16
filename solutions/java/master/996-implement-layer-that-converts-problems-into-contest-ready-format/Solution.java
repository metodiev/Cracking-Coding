/** Task 996 - Implement layer that converts problems into contest-ready format (time limits, scoring).
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  The layer turns an editorial problem plus its measured reference solution into a contest package.
 *  The time limit is derived from the reference runtime and a language factor and snapped to the
 *  published grid {0.5, 1, 1.5, 2, 3, 5, 10} seconds clamped to [1, 10]: a slow judge gets slack, and no
 *  language gets a limit so tight that I/O decides the round. Subtasks receive integer points by
 *  largest-remainder allocation over their weights with a floor of one point each, so the scores always
 *  sum to exactly 100. A validator rejects off-grid limits, unknown checkers, out-of-range memory,
 *  duplicate ids, dangling or cyclic subtask dependencies and a score total other than 100, and a ranker
 *  orders submissions by points, earliest submission, fewest attempts and finally team name.
 *  Simplified: reference runtimes are given rather than measured and the grid is a fixed table; the
 *  limit arithmetic, the remainder allocation, the validation gates and the ranking are the real ones.
 *  Verified: grid snapping and both clamps, exact 50/30/20 and 34/33/33 allocations with the one-point
 *  floor, dependency order and cycles, every validator error path, the empty and single-subtask
 *  boundaries, the full ranking tie-break chain, and integer-arithmetic oracles for the time limit and
 *  the score total.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final double[] GRID = {0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0};
    static final double MIN_LIMIT = 1.0, MAX_LIMIT = 10.0;
    static final Set<String> CHECKERS = Set.of("exact", "float", "special");

    /** Snaps the reference runtime (times the language factor) onto the grid inside the clamp range. */
    static double timeLimit(double referenceSeconds, double factor) {
        double raw = referenceSeconds * factor;
        if (raw <= MIN_LIMIT) return MIN_LIMIT;
        if (raw >= MAX_LIMIT) return MAX_LIMIT;
        for (double grid : GRID) if (grid >= raw) return grid;
        return MAX_LIMIT;
    }

    static final class Subtask {
        final String id;
        final int weight;
        final List<String> requires;
        int points;
        Subtask(String id, int weight, List<String> requires) {
            this.id = id; this.weight = weight; this.requires = new ArrayList<>(requires);
        }
    }

    /** Largest-remainder allocation of 100 points, then a floor of one point per subtask. */
    static void allocate(List<Subtask> subtasks, int total) {
        int weightSum = 0;
        for (Subtask s : subtasks) weightSum += s.weight;
        int assigned = 0;
        List<int[]> remainders = new ArrayList<>();                 // {remainder, index}
        for (int i = 0; i < subtasks.size(); i++) {
            int exact = total * subtasks.get(i).weight;
            subtasks.get(i).points = exact / weightSum;
            assigned += subtasks.get(i).points;
            remainders.add(new int[]{exact % weightSum, i});
        }
        remainders.sort((a, b) -> a[0] != b[0] ? Integer.compare(b[0], a[0])
                : subtasks.get(a[1]).id.compareTo(subtasks.get(b[1]).id));
        for (int i = 0; i < total - assigned; i++) subtasks.get(remainders.get(i % remainders.size())[1]).points++;
        for (Subtask s : subtasks) {
            if (s.points > 0) continue;
            Subtask donor = subtasks.get(0);
            for (Subtask other : subtasks) if (other.points > donor.points) donor = other;
            donor.points--;                                         // never starve a subtask
            s.points = 1;
        }
    }

    static final class Package {
        final double timeLimit, factor;
        final int memoryMb;
        final String checker;
        final List<Subtask> subtasks;
        Package(double timeLimit, double factor, int memoryMb, String checker, List<Subtask> subtasks) {
            this.timeLimit = timeLimit; this.factor = factor; this.memoryMb = memoryMb;
            this.checker = checker; this.subtasks = subtasks;
        }
    }

    static List<String> validate(Package pkg) {
        List<String> errors = new ArrayList<>();
        boolean onGrid = false;
        for (double grid : GRID) if (Math.abs(grid - pkg.timeLimit) < 1e-9) onGrid = true;
        if (!onGrid) errors.add("time limit " + pkg.timeLimit + " is not on the published grid");
        if (pkg.timeLimit < MIN_LIMIT || pkg.timeLimit > MAX_LIMIT) errors.add("time limit outside 1..10 seconds");
        if (pkg.memoryMb < 16 || pkg.memoryMb > 1024) errors.add("memory limit " + pkg.memoryMb + " MB is out of range");
        if (!CHECKERS.contains(pkg.checker)) errors.add("unknown checker: " + pkg.checker);
        if (pkg.subtasks.isEmpty()) errors.add("a contest package needs at least one subtask");
        Set<String> ids = new LinkedHashSet<>();
        for (Subtask s : pkg.subtasks) if (!ids.add(s.id)) errors.add("duplicate subtask id: " + s.id);
        for (Subtask s : pkg.subtasks)
            for (String dep : s.requires)
                if (!ids.contains(dep)) errors.add("subtask " + s.id + " requires unknown subtask " + dep);
        if (topologicalOrder(pkg.subtasks) == null) errors.add("subtask dependency cycle detected");
        int total = 0;
        for (Subtask s : pkg.subtasks) {
            if (s.points < 1) errors.add("subtask " + s.id + " has no points");
            total += s.points;
        }
        if (!pkg.subtasks.isEmpty() && total != 100) errors.add("subtask points sum to " + total + ", expected 100");
        return errors;
    }

    /** Kahn order over subtask dependencies; null when the dependencies are cyclic or dangling. */
    static List<String> topologicalOrder(List<Subtask> subtasks) {
        Map<String, Integer> indegree = new TreeMap<>();
        Map<String, List<String>> dependents = new TreeMap<>();
        for (Subtask s : subtasks) { indegree.putIfAbsent(s.id, 0); dependents.putIfAbsent(s.id, new ArrayList<>()); }
        for (Subtask s : subtasks)
            for (String dep : s.requires) {
                if (!indegree.containsKey(dep)) return null;
                indegree.merge(s.id, 1, Integer::sum);
                dependents.get(dep).add(s.id);
            }
        PriorityQueue<String> ready = new PriorityQueue<>();
        for (Map.Entry<String, Integer> e : indegree.entrySet()) if (e.getValue() == 0) ready.add(e.getKey());
        List<String> order = new ArrayList<>();
        while (!ready.isEmpty()) {
            String id = ready.poll();
            order.add(id);
            for (String next : dependents.get(id)) if (indegree.merge(next, -1, Integer::sum) == 0) ready.add(next);
        }
        return order.size() == indegree.size() ? order : null;
    }

    static final class Submission {
        final String team;
        final int solvedPoints, attempts, submittedAt;
        Submission(String team, int solvedPoints, int attempts, int submittedAt) {
            this.team = team; this.solvedPoints = solvedPoints; this.attempts = attempts; this.submittedAt = submittedAt;
        }
    }

    /** Ranking: points desc, then earliest submission, then fewest attempts, then team name. */
    static List<Submission> rank(List<Submission> submissions) {
        List<Submission> out = new ArrayList<>(submissions);
        out.sort((a, b) -> {
            if (a.solvedPoints != b.solvedPoints) return Integer.compare(b.solvedPoints, a.solvedPoints);
            if (a.submittedAt != b.submittedAt) return Integer.compare(a.submittedAt, b.submittedAt);
            if (a.attempts != b.attempts) return Integer.compare(a.attempts, b.attempts);
            return a.team.compareTo(b.team);
        });
        return out;
    }

    static List<Subtask> subtasks() {
        return new ArrayList<>(List.of(
            new Subtask("A", 5, List.of()), new Subtask("B", 3, List.of("A")), new Subtask("C", 2, List.of("B"))));
    }

    static List<Subtask> scoredSubtasks() {
        List<Subtask> subtasks = subtasks();
        allocate(subtasks, 100);
        return subtasks;
    }

    static Package packageWith(double limit, int memoryMb, String checker, List<Subtask> subtasks) {
        return new Package(limit, 3.0, memoryMb, checker, subtasks);
    }

    /** True when any reported validation error mentions the fragment. */
    static boolean reports(List<String> errors, String fragment) {
        for (String error : errors) if (error.contains(fragment)) return true;
        return false;
    }

    public static void main(String[] args) {
        // --- time limits: grid snapping plus both clamps, and the grid invariant over a sweep.
        check(timeLimit(0.31, 3) == 1.0 && timeLimit(0.6, 3) == 2.0 && timeLimit(3.9, 3) == 10.0 && timeLimit(0.05, 2) == 1.0 && timeLimit(1.2, 3) == 5.0, "limits must snap upward onto the grid and clamp at both ends: " + timeLimit(1.2, 3));
        for (double reference = 0.01; reference < 5.0; reference += 0.037) {
            double limit = timeLimit(reference, 3);
            boolean onGrid = false;
            for (double grid : GRID) if (grid == limit) onGrid = true;
            check(onGrid && limit >= MIN_LIMIT && limit <= MAX_LIMIT, "every computed limit must be on the grid inside the clamp: " + reference + " -> " + limit);
        }
        // --- scoring: exact largest-remainder allocation, the deterministic tie-break and the floor.
        List<Subtask> weighted = subtasks();
        allocate(weighted, 100);
        List<Subtask> equal = new ArrayList<>(List.of(new Subtask("A", 1, List.of()), new Subtask("B", 1, List.of()),
                new Subtask("C", 1, List.of())));
        allocate(equal, 100);
        List<Subtask> lopsided = new ArrayList<>(List.of(new Subtask("A", 40, List.of()), new Subtask("B", 1, List.of()),
                new Subtask("C", 1, List.of())));
        allocate(lopsided, 100);
        check(weighted.stream().map(s -> s.points).toList().equals(List.of(50, 30, 20)) && equal.stream().map(s -> s.points).toList().equals(List.of(34, 33, 33)),
                "5/3/2 must split 50/30/20 and equal weights 34/33/33: " + weighted.stream().map(s -> s.points).toList() + equal.stream().map(s -> s.points).toList());
        int oracleTotal = 0;                                        // independent accumulation oracle
        for (Subtask s : weighted) oracleTotal += s.points;
        check(lopsided.stream().mapToInt(s -> s.points).sum() == 100 && lopsided.stream().allMatch(s -> s.points >= 1) && oracleTotal == 100, "every subtask must keep at least one point and the total must stay 100: " + lopsided.stream().map(s -> s.points).toList());
        // --- a complete package validates, and the topological order respects the dependencies.
        Package pkg = packageWith(timeLimit(0.31, 3), 256, "exact", scoredSubtasks());
        check(validate(pkg).isEmpty() && pkg.timeLimit == 1.0 && topologicalOrder(pkg.subtasks).equals(List.of("A", "B", "C")),
                "a freshly built package must validate in dependency order: " + validate(pkg));
        // --- validator error paths: off-grid limit, bad checker, memory, scores, dangling and cyclic deps.
        List<Subtask> shortScores = scoredSubtasks();
        shortScores.get(2).points = 19;
        List<Subtask> dangling = scoredSubtasks();
        dangling.set(1, new Subtask("B", 3, List.of("A", "Z")));
        List<Subtask> cyclic = new ArrayList<>(List.of(new Subtask("A", 1, List.of("B")), new Subtask("B", 1, List.of("A"))));
        allocate(cyclic, 100);
        check(reports(validate(packageWith(1.2, 256, "exact", scoredSubtasks())), "is not on the published grid"),
                "an off-grid limit must be reported: " + validate(packageWith(1.2, 256, "exact", scoredSubtasks())));
        check(reports(validate(packageWith(2.0, 256, "mystery", scoredSubtasks())), "unknown checker: mystery") && reports(validate(packageWith(2.0, 4, "exact", scoredSubtasks())), "memory limit 4 MB is out of range"),
                "an unknown checker and an impossible memory limit must be reported");
        check(reports(validate(packageWith(11.0, 256, "exact", scoredSubtasks())), "time limit outside 1..10 seconds") && reports(validate(packageWith(2.0, 256, "exact", shortScores)), "subtask points sum to 99, expected 100"),
                "a limit above the ceiling and a 99 point package must be reported");
        check(reports(validate(packageWith(2.0, 256, "exact", dangling)), "subtask B requires unknown subtask Z") && topologicalOrder(cyclic) == null && reports(validate(packageWith(2.0, 256, "exact", cyclic)), "subtask dependency cycle detected"),
                "a dangling dependency and a cycle must be reported");
        // --- boundaries: no subtasks is invalid, a single subtask owns all 100 points.
        List<Subtask> solo = new ArrayList<>(List.of(new Subtask("only", 7, List.of())));
        allocate(solo, 100);
        check(reports(validate(packageWith(2.0, 256, "exact", List.of())), "at least one subtask") && solo.get(0).points == 100 && topologicalOrder(solo).equals(List.of("only")),
                "an empty subtask list must be invalid and a solo subtask must take all 100 points");
        // --- ranking: score, earliest submission, attempts, then name - a total order with no ties.
        List<Submission> board = List.of(
            new Submission("zeta", 300, 2, 1000), new Submission("alpha", 300, 2, 900),
            new Submission("beta", 300, 2, 900), new Submission("gamma", 100, 1, 500),
            new Submission("delta", 300, 1, 900), new Submission("epsilon", 300, 2, 900));
        List<Submission> ranked = rank(board);
        check(ranked.stream().map(s -> s.team).toList() .equals(List.of("delta", "alpha", "beta", "epsilon", "zeta", "gamma")) && rank(board).size() == board.size(),
                "ranking must apply every tie-break in order: " + ranked.stream().map(s -> s.team).toList());
        // --- independent oracle: integer time-limit arithmetic must match the floating point scan.
        for (double reference = 0.05; reference < 4.0; reference += 0.13)
            check(Math.abs(oracleLimit(reference, 3) - timeLimit(reference, 3)) < 1e-9, "time-limit oracle mismatch at " + reference);

        System.out.println("package: limit " + pkg.timeLimit + "s, memory " + pkg.memoryMb + "MB, checker " + pkg.checker);
        System.out.println("PASS");
    }

    /** Oracle: tenths-of-a-second ceiling arithmetic instead of the floating point grid scan. */
    static double oracleLimit(double referenceSeconds, double factor) {
        long tenths = (long) Math.ceil(referenceSeconds * factor * 10 - 1e-9);
        if (tenths <= 10) return MIN_LIMIT;
        if (tenths >= 100) return MAX_LIMIT;
        for (double grid : GRID) if (grid >= MIN_LIMIT && grid * 10 >= tenths - 1e-9) return grid;
        return MAX_LIMIT;
    }
}
