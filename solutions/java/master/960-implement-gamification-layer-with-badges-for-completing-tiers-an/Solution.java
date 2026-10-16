/** Task 960 - Implement gamification layer with badges for completing tiers and streaks.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory gamification engine. An event log records (user, day, tier, task) completions; from it the
 *  engine derives the current and longest daily streak over distinct day indices, the set of completed
 *  tiers, the number of distinct tasks, and the badges those facts earn. Day indices are abstract, so no
 *  time zone or wall clock can leak in, and repeated events for the same task on the same day are
 *  idempotent.
 *  Simplified: badges are a small fixed catalogue evaluated in memory, and a "day" is an integer the
 *  caller supplies instead of a timestamp; streak, idempotence, tier and ordering semantics are real.
 *  Verified: streaks follow consecutive day runs and break on any gap of two days; log replay is
 *  idempotent and order independent (identical digest); tier badges need every task of the tier; empty
 *  and single-event logs are handled; current and longest streak differ across two runs; users are
 *  isolated; an independent brute-force streak scan reproduces every streak over 30 generated logs.
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

    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }

    static String hex(long v) { return String.format("%016x", v); }

    record Completion(String user, int day, String tier, String task) {}

    static final Map<String, Integer> TIER_SIZES = Map.of("advanced", 3, "expert", 4, "master", 2);

    static List<Integer> daysOf(List<Completion> log, String user) {
        Set<Integer> days = new TreeSet<>();
        for (Completion c : log) if (c.user().equals(user)) days.add(c.day());
        return new ArrayList<>(days);
    }

    static int longestStreak(List<Integer> days) {
        int best = 0, run = 0, previous = Integer.MIN_VALUE;
        for (int day : days) {
            run = day == previous + 1 ? run + 1 : 1;
            best = Math.max(best, run);
            previous = day;
        }
        return best;
    }

    static int currentStreak(List<Integer> days) {
        if (days.isEmpty()) return 0;
        int run = 1;
        for (int i = days.size() - 1; i > 0; i--) {
            if (days.get(i) == days.get(i - 1) + 1) run++;
            else break;
        }
        return run;
    }

    /** Brute force streak over a boolean day axis, used as the independent oracle. */
    static int bruteForceLongest(Collection<Integer> days) {
        int best = 0;
        for (int day : days) {
            int run = 0;
            while (days.contains(day + run)) run++;
            best = Math.max(best, run);
        }
        return best;
    }

    static List<String> badges(List<Completion> log, String user) {
        Set<String> tasks = new TreeSet<>();
        Map<String, Set<String>> tasksByTier = new TreeMap<>();
        for (Completion c : log) {
            if (!c.user().equals(user)) continue;
            tasks.add(c.tier() + "#" + c.task());
            tasksByTier.computeIfAbsent(c.tier(), key -> new TreeSet<>()).add(c.task());
        }
        List<Integer> days = daysOf(log, user);
        List<String> earned = new ArrayList<>();
        if (!tasks.isEmpty()) earned.add("first-blood");
        for (Map.Entry<String, Set<String>> entry : tasksByTier.entrySet()) {
            Integer size = TIER_SIZES.get(entry.getKey());
            if (size != null && entry.getValue().size() >= size) earned.add("tier-complete:" + entry.getKey());
        }
        if (longestStreak(days) >= 3) earned.add("streak-3");
        if (longestStreak(days) >= 7) earned.add("streak-7");
        if (tasks.size() >= 10) earned.add("ten-tasks");
        Collections.sort(earned);
        return earned;
    }

    static String profileDigest(List<Completion> log, String user) {
        Set<String> events = new TreeSet<>();
        for (Completion c : log) if (c.user().equals(user)) events.add(c.day() + ":" + c.tier() + ":" + c.task());
        long h = fnvText(FNV_OFFSET, "profile:" + user);
        for (String event : events) h = fnvText(h, event + "|");
        h = fnvText(h, "days=" + daysOf(log, user));
        h = fnvText(h, "badges=" + badges(log, user));
        return hex(h);
    }

    static List<Completion> replay(List<Completion> log, boolean reversed, boolean withDuplicates) {
        List<Completion> copy = new ArrayList<>(log);
        if (reversed) Collections.reverse(copy);
        if (withDuplicates) copy.addAll(log);
        return copy;
    }

    public static void main(String[] args) {
        List<Completion> log = new ArrayList<>(List.of(
                new Completion("ada", 1, "advanced", "a1"),
                new Completion("ada", 1, "advanced", "a1"),          // duplicate event
                new Completion("ada", 2, "advanced", "a2"),
                new Completion("ada", 3, "advanced", "a3"),
                new Completion("ada", 3, "advanced", "a3"),          // duplicate event
                new Completion("ada", 9, "expert", "e1"),
                new Completion("ada", 10, "expert", "e2"),
                new Completion("linus", 1, "advanced", "a1"),
                new Completion("linus", 5, "master", "m1")));

        // 1. streak happy path: days 1,2,3 then 9,10
        List<Integer> adaDays = daysOf(log, "ada");
        check(adaDays.equals(List.of(1, 2, 3, 9, 10)) && currentStreak(adaDays) == 2 && longestStreak(adaDays) == 3,
                "distinct days are 1,2,3,9,10 so current streak is 2 and longest is 3");

        // 2. duplicate events are idempotent
        check(badges(replay(log, false, true), "ada").equals(badges(log, "ada"))
                        && profileDigest(replay(log, false, true), "ada").equals(profileDigest(log, "ada")),
                "replaying the same events changes neither the badges nor the digest");

        // 3. order independence
        check(profileDigest(replay(log, true, false), "ada").equals(profileDigest(log, "ada"))
                        && badges(replay(log, true, false), "ada").equals(badges(log, "ada")),
                "a reversed event log produces the same profile");

        // 4. tier badge needs every task of the tier
        List<Completion> almost = new ArrayList<>(log);
        almost.removeIf(c -> c.user().equals("ada") && c.task().equals("e2"));
        check(badges(log, "ada").contains("tier-complete:advanced") && !badges(almost, "ada").contains("tier-complete:expert")
                        && badges(almost, "ada").contains("tier-complete:advanced"),
                "advanced tier badge is earned, the expert one still needs all four tasks");

        // 5. boundary: empty log, single event
        List<Completion> single = List.of(new Completion("solo", 7, "master", "m1"));
        check(badges(List.of(), "ada").isEmpty() && currentStreak(daysOf(List.of(), "ada")) == 0
                        && badges(single, "solo").equals(List.of("first-blood")) && currentStreak(daysOf(single, "solo")) == 1,
                "empty log earns nothing, a single event earns first-blood with a streak of one");

        // 6. streak badges: a seven day run earns streak-3 and streak-7, the two-run log does not
        List<Completion> week = new ArrayList<>();
        for (int day = 20; day < 27; day++) week.add(new Completion("grace", day, "master", "m" + day));
        check(badges(week, "grace").contains("streak-3") && badges(week, "grace").contains("streak-7")
                        && !badges(log, "ada").contains("streak-7") && longestStreak(daysOf(week, "grace")) == 7,
                "seven consecutive days earn both streak badges while the two-run log only earns streak-3");

        // 7. users are isolated from each other
        check(badges(log, "linus").equals(List.of("first-blood")) && currentStreak(daysOf(log, "linus")) == 1
                        && longestStreak(daysOf(log, "linus")) == 1
                        && !profileDigest(log, "linus").equals(profileDigest(log, "ada")),
                "linus keeps his own profile, unaffected by ada's streak");

        // 8. independent oracle: brute-force streak scan over 30 generated logs
        boolean oracleOk = true;
        for (int i = 0; i < 30; i++) {
            List<Completion> generated = new ArrayList<>();
            long state = splitmix64(4000 + i);
            int day = 0;
            for (int j = 0; j < 12; j++) {
                state = splitmix64(state + j);
                day += (int) Math.floorMod(state, 3);                 // gaps of 0, 1 or 2 days
                generated.add(new Completion("u" + i, day, "master", "m" + j));
            }
            List<Integer> days = daysOf(generated, "u" + i);
            oracleOk &= longestStreak(days) == bruteForceLongest(days)
                    && currentStreak(days) >= 1 && currentStreak(days) <= longestStreak(days);
        }
        check(oracleOk && badges(List.of(), "nobody").isEmpty(), "brute-force streak oracle agrees on 30 logs");

        System.out.println("ada days: " + adaDays + ", current=" + currentStreak(adaDays) + ", longest=" + longestStreak(adaDays));
        System.out.println("ada badges: " + badges(log, "ada"));
        System.out.println("grace badges over a week: " + badges(week, "grace"));
        System.out.println("ada profile digest: " + profileDigest(log, "ada"));
        System.out.println("PASS");
    }
}
