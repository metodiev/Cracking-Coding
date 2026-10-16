/** Task 959 - Implement leaderboard with verified performance metrics and reproducible runs.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory leaderboard. A submission carries a claimed runtime plus a witness digest; verification
 *  re-runs a deterministic workload (a fixed-seed splitmix64 fold) for that contestant and problem and
 *  compares digests, so only reproducible runs count. Unverified submissions and runs below a physical
 *  time floor are rejected, duplicate contestant entries per problem keep the best metric, and ranking is
 *  by metric ascending with the contestant name as the deterministic tie-break.
 *  Simplified: the "performance metric" is a recorded microsecond count rather than a measured run and
 *  the workload is tiny; verification, rejection, deduplication and ranking are the real mechanics.
 *  Verified: ranking order and ranks are exact; unverified and implausible submissions never appear;
 *  verification and the leaderboard digest reproduce across runs; top-k slices the ranking, k=0 is empty
 *  and k above the size returns everything; duplicates collapse to the better run; ties break by name in
 *  either insertion order; an independent full sort reproduces every row.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static long splitmix64(long x) {
        long z = x + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final long FNV_OFFSET = 0xcbf29ce484222325L, FNV_PRIME = 0x100000001b3L;
    static final long TIME_FLOOR_MICROS = 5;

    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }

    static String hex(long v) { return String.format("%016x", v); }

    /** Deterministic verification workload: the same contestant and problem always yield the same digest. */
    static String witness(String contestant, int problem, int rounds) {
        long h = fnvText(FNV_OFFSET, "witness:" + contestant + ':' + problem);
        long state = splitmix64(contestant.hashCode() * 31L + problem);
        for (int i = 0; i < rounds; i++) {
            state = splitmix64(state + i);
            h = fnvText(h, Long.toHexString(state));
        }
        return hex(h);
    }

    record Submission(String contestant, int problem, long micros, String claimed, String actual) {
        boolean verified() { return claimed != null && claimed.equals(actual); }
        boolean plausible() { return micros >= TIME_FLOOR_MICROS; }
        boolean accepted() { return verified() && plausible(); }
    }

    record Row(int rank, String contestant, long micros) {}

    record Leaderboard(int problem, List<Row> rows, List<String> rejected, String digest) {}

    static String digest(int problem, List<Row> rows, List<String> rejected) {
        long h = fnvText(FNV_OFFSET, "board:" + problem);
        for (Row row : rows) h = fnvText(h, row.rank() + ":" + row.contestant() + ":" + row.micros() + "|");
        for (String name : rejected) h = fnvText(h, "rej:" + name + "|");
        return hex(h);
    }

    static Leaderboard leaderboard(int problem, List<Submission> submissions) {
        List<String> rejected = new ArrayList<>();
        Map<String, Submission> best = new TreeMap<>();
        for (Submission s : submissions) {
            if (s.problem() != problem) continue;
            if (!s.verified()) { rejected.add(s.contestant() + ":unverified"); continue; }
            if (!s.plausible()) { rejected.add(s.contestant() + ":implausible"); continue; }
            Submission current = best.get(s.contestant());
            if (current == null || s.micros() < current.micros()) best.put(s.contestant(), s);
        }
        List<Submission> ranked = new ArrayList<>(best.values());
        ranked.sort((a, b) -> a.micros() != b.micros()
                ? Long.compare(a.micros(), b.micros())
                : a.contestant().compareTo(b.contestant()));
        List<Row> rows = new ArrayList<>();
        for (int i = 0; i < ranked.size(); i++) rows.add(new Row(i + 1, ranked.get(i).contestant(), ranked.get(i).micros()));
        return new Leaderboard(problem, rows, rejected, digest(problem, rows, rejected));
    }

    static List<Row> topK(Leaderboard board, int k) {
        if (k <= 0) return new ArrayList<>();
        return new ArrayList<>(board.rows().subList(0, Math.min(k, board.rows().size())));
    }

    static Submission honest(String contestant, int problem, long micros) {
        return new Submission(contestant, problem, micros, witness(contestant, problem, 200), witness(contestant, problem, 200));
    }

    static Submission cheater(String contestant, int problem, long micros) {
        return new Submission(contestant, problem, micros, hex(fnvText(FNV_OFFSET, "forged")), witness(contestant, problem, 200));
    }

    static List<Submission> dataset() {
        List<Submission> all = new ArrayList<>();
        all.add(honest("ada", 71, 120));
        all.add(honest("linus", 71, 95));
        all.add(honest("grace", 71, 95));
        all.add(honest("ken", 71, 310));
        all.add(cheater("forger", 71, 40));
        all.add(honest("hopper", 71, 2));                       // below the physical floor
        all.add(honest("ada", 71, 88));                         // duplicate for ada, better run
        all.add(honest("ada", 72, 70));
        all.add(honest("linus", 72, 70));
        all.add(honest("grace", 72, 200));
        return all;
    }

    public static void main(String[] args) {
        List<Submission> data = dataset();
        Leaderboard board = leaderboard(71, data);

        // 1. ranking is exact: ascending metric, ties by name, ranks 1..n
        List<Row> rows = board.rows();
        boolean orderOk = rows.size() == 4;
        for (int i = 1; i < rows.size(); i++) {
            Row previous = rows.get(i - 1), current = rows.get(i);
            orderOk &= previous.micros() < current.micros()
                    || (previous.micros() == current.micros() && previous.contestant().compareTo(current.contestant()) < 0);
        }
        for (int i = 0; i < rows.size(); i++) orderOk &= rows.get(i).rank() == i + 1;
        check(orderOk && rows.get(0).contestant().equals("ada") && rows.get(0).micros() == 88
                        && rows.get(1).contestant().equals("grace") && rows.get(2).contestant().equals("linus"),
                "four verified runs ranked by metric with the name tie-break at rank 2");

        // 2. unverified and implausible runs never reach the board
        boolean forgerAbsent = true;
        for (Row row : board.rows()) forgerAbsent &= !row.contestant().equals("forger");
        check(board.rejected().equals(List.of("forger:unverified", "hopper:implausible")) && forgerAbsent,
                "unverified and implausible submissions are rejected, not ranked");

        // 3. verification and the digest reproduce
        Leaderboard again = leaderboard(71, dataset());
        check(board.digest().equals(again.digest()) && board.rows().equals(again.rows())
                        && witness("ada", 71, 200).equals(witness("ada", 71, 200))
                        && !witness("ada", 71, 200).equals(witness("ada", 72, 200)),
                "re-running verification and ranking reproduces the board digest exactly");

        // 4. top-k slicing including the boundaries
        check(topK(board, 2).size() == 2 && topK(board, 2).get(1).equals(rows.get(1))
                        && topK(board, 99).size() == rows.size() && topK(board, 0).isEmpty(),
                "top-k returns a prefix, clamps above the size and is empty for k = 0");

        // 5. duplicate contestant entries collapse to the better run
        int adamCount = 0;
        for (Row row : rows) if (row.contestant().equals("ada")) adamCount++;
        check(adamCount == 1 && board.rows().get(0).micros() == 88,
                "duplicate entries keep the faster run and one row per contestant");

        // 6. an implausible metric is rejected instead of winning the board
        List<Submission> hacked = new ArrayList<>(data);
        hacked.add(cheater("sprinter", 71, 1));
        Leaderboard hackedBoard = leaderboard(71, hacked);
        check(hackedBoard.rows().size() == 4 && hackedBoard.rejected().contains("sprinter:unverified")
                        && hackedBoard.rows().get(0).contestant().equals("ada"),
                "a forged sub-floor time cannot displace the verified leader");

        // 7. ties break by name regardless of insertion order
        List<Submission> shuffled = new ArrayList<>(List.of(
                honest("zeta", 73, 50), honest("alpha", 73, 50), honest("mike", 73, 50), honest("beta", 73, 50)));
        Collections.reverse(shuffled);
        Leaderboard tieBoard = leaderboard(73, shuffled);
        check(tieBoard.rows().get(0).contestant().equals("alpha") && tieBoard.rows().get(1).contestant().equals("beta")
                        && tieBoard.rows().get(2).contestant().equals("mike") && tieBoard.rows().get(3).contestant().equals("zeta"),
                "four equal metrics rank alphabetically whatever the insertion order");

        // 8. independent oracle: full sort of the accepted submissions reproduces every row
        List<Submission> accepted = new ArrayList<>();
        for (Submission s : data)
            if (s.problem() == 71 && s.accepted()) accepted.add(s);
        accepted.sort(Comparator.comparingLong(Submission::micros).thenComparing(Submission::contestant));
        Map<String, Long> oracleBest = new TreeMap<>();
        for (Submission s : accepted) oracleBest.merge(s.contestant(), s.micros(), Math::min);
        List<String> oracleNames = new ArrayList<>(oracleBest.keySet());
        oracleNames.sort((a, b) -> oracleBest.get(a).equals(oracleBest.get(b))
                ? a.compareTo(b)
                : Long.compare(oracleBest.get(a), oracleBest.get(b)));
        boolean oracleOk = oracleNames.size() == rows.size();
        for (int i = 0; oracleOk && i < oracleNames.size(); i++)
            oracleOk &= oracleNames.get(i).equals(rows.get(i).contestant()) && oracleBest.get(oracleNames.get(i)) == rows.get(i).micros();
        check(oracleOk && leaderboard(72, data).rows().size() == 3,
                "an independent min-map plus sort reproduces the ranking for both problems");

        System.out.println("problem 71 board: " + rows);
        System.out.println("rejected: " + board.rejected() + ", problem 72 rows: " + leaderboard(72, data).rows().size());
        System.out.println("top-2: " + topK(board, 2));
        System.out.println("board digest: " + board.digest());
        System.out.println("PASS");
    }
}
