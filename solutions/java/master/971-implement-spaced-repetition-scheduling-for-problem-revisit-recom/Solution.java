/** Task 971 - Implement spaced-repetition scheduling for problem revisit recommendations.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  An SM-2 spaced-repetition scheduler. A card stores its ease factor, current interval, repetition
 *  streak, next due day and lapse count. A review with grade q >= 3 advances the streak (interval 1, then
 *  6, then round(previous * ease)) and nudges the ease by 0.1 - (5-q)*(0.08 + (5-q)*0.02); a grade below 3
 *  is a lapse: the streak resets, the card relearns at interval 1 and the ease drops by 0.8, floored at
 *  1.3. The deck is driven by a deterministic day loop that reviews exactly the due cards, ordered by
 *  (due, id), and records a trace of day/grade/interval/ease/due.
 *  Simplified: one review per card per day, a 120-day horizon and a scripted grade stream instead of a
 *  learner model; the interval recurrence, the ease floor, lapse handling and due-date arithmetic are the
 *  real SM-2 rules.
 *  Verified: the 1/6/16/45 interval progression for perfect grades, a closed-form schedule oracle for the
 *  review days, per-mutation invariants (interval >= 1, ease >= 1.3, due == day + interval), queue
 *  discipline, lapse relearning, the ease floor, the grade 2 vs 3 boundary, empty/single-card decks and
 *  determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final double MIN_EF = 1.3;
    static final int DAYS = 120;

    static final class Card {
        final String id;
        double ef = 2.5; int interval = 0, reps = 0, due = 0, lapses = 0;
        Card(String id) { this.id = id; }
    }

    /** SM-2 update for one review on `today`; returns nothing, mutates `c` in place. */
    static void review(Card c, int grade, int today) {
        if (grade < 3) {                                                     // lapse: relearn from scratch
            c.lapses++;
            c.reps = 0;
            c.interval = 1;
            c.ef = Math.max(MIN_EF, c.ef - 0.8);
        } else {
            c.reps++;
            c.interval = c.reps == 1 ? 1 : c.reps == 2 ? 6 : (int) Math.round(c.interval * c.ef);
            c.ef = Math.max(MIN_EF, c.ef + 0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02));
        }
        c.due = today + c.interval;
    }

    static final class Record {
        final int day, grade, interval, due; final double ef; final String id;
        Record(String id, int day, int grade, Card c) {
            this.id = id; this.day = day; this.grade = grade;
            this.interval = c.interval; this.due = c.due; this.ef = c.ef;
        }
        String render() {
            return id + "@" + day + " g" + grade + " i" + interval + " ef" + String.format(Locale.ROOT, "%.2f", ef)
                    + " due" + due;
        }
    }

    /** Deterministic day loop: every day, review the due cards ordered by (due, id). */
    static Map<String, List<Record>> simulate(List<Card> deck, int days) {
        Map<String, Integer> seen = new TreeMap<>();
        Map<String, List<Record>> trace = new TreeMap<>();
        for (Card c : deck) trace.put(c.id, new ArrayList<>());
        for (int day = 0; day <= days; day++) {
            List<Card> due = new ArrayList<>();
            for (Card c : deck) if (c.due <= day) due.add(c);
            due.sort(Comparator.comparingInt((Card c) -> c.due).thenComparing(c -> c.id));
            for (Card c : due) {
                int index = seen.merge(c.id, 1, Integer::sum) - 1;
                int g = grade(c.id, index);
                review(c, g, day);
                trace.get(c.id).add(new Record(c.id, day, g, c));
            }
        }
        return trace;
    }

    /** Scripted, clamped grade stream per card: a perfect card, a mixed card and a lapse-prone card. */
    static int grade(String id, int index) {
        int[] perfect = {5, 5, 5, 5, 5, 5};
        int[] mixed = {4, 5, 3, 4, 2, 5, 5, 5, 5};
        int[] lapsed = {1, 1, 1, 5, 5, 5, 5, 5, 5};
        int[] grades = id.equals("two-sum") ? perfect : id.equals("bfs-maze") ? mixed : lapsed;
        return grades[Math.min(index, grades.length - 1)];
    }

    static List<Card> deck() {
        return List.of(new Card("two-sum"), new Card("bfs-maze"), new Card("coin-change"));
    }

    /** Independent oracle: a card always graded 5 is due on the running sum of a hand-derived schedule. */
    static int[] closedFormSchedule() {
        int[] intervals = {1, 6, 16, 45, 131};            // ef 2.6, 2.7, 2.8, 2.9 after each success
        int[] days = new int[intervals.length];
        for (int i = 1; i < intervals.length; i++) days[i] = days[i - 1] + intervals[i - 1];
        return days;
    }

    static int[] intervalsOf(List<Record> records) {
        int[] out = new int[records.size()];
        for (int i = 0; i < out.length; i++) out[i] = records.get(i).interval;
        return out;
    }

    static int[] daysOf(List<Record> records) {
        int[] out = new int[records.size()];
        for (int i = 0; i < out.length; i++) out[i] = records.get(i).day;
        return out;
    }

    static List<String> render(Map<String, List<Record>> trace) {
        List<String> out = new ArrayList<>();
        for (Map.Entry<String, List<Record>> e : trace.entrySet()) {
            for (Record r : e.getValue()) out.add(r.render());
        }
        return out;
    }

    public static void main(String[] args) {
        Map<String, List<Record>> trace = simulate(deck(), DAYS);
        List<Record> perfect = trace.get("two-sum");

        check(Arrays.equals(intervalsOf(perfect), new int[]{1, 6, 16, 45, 131})
                        && perfect.size() == 5 && perfect.stream().allMatch(r -> r.grade == 5),
                "perfect grades must give the 1/6/16/45/131 progression, got "
                        + Arrays.toString(intervalsOf(perfect)));
        check(Math.abs(perfect.get(0).ef - 2.6) < 1e-12 && Math.abs(perfect.get(2).ef - 2.8) < 1e-12
                        && Math.abs(perfect.get(4).ef - 3.0) < 1e-12,
                "each perfect grade must raise the ease factor by exactly 0.1");

        check(Arrays.equals(daysOf(perfect), closedFormSchedule()),
                "review days must equal the closed-form running sum of the intervals: "
                        + Arrays.toString(daysOf(perfect)));

        List<Record> all = new ArrayList<>();
        for (List<Record> rs : trace.values()) all.addAll(rs);
        boolean invariants = true;
        for (Record r : all) invariants &= r.interval >= 1 && r.ef >= MIN_EF - 1e-12 && r.due == r.day + r.interval;
        check(invariants && all.size() == 25,
                "after every mutation interval >= 1, ease >= 1.3 and due == day + interval: records=" + all.size());

        boolean queue = true;
        for (Map.Entry<String, List<Record>> e : trace.entrySet()) {
            int previousDue = 0;
            for (Record r : e.getValue()) {
                queue &= r.day >= previousDue && r.day >= 0;
                previousDue = r.due;
            }
        }
        check(queue, "a card may only be reviewed on or after its due day, once per sitting");

        List<Record> mixed = trace.get("bfs-maze");
        check(mixed.size() == 10 && mixed.get(4).grade == 2 && mixed.get(4).day == 62
                        && mixed.get(4).interval == 1 && mixed.get(5).interval == 1 && mixed.get(6).interval == 6
                        && mixed.get(4).ef < mixed.get(3).ef,
                "a lapse must relearn at interval 1 (twice) and lower the ease: " + mixed.get(4).render()
                        + " then " + mixed.get(5).render());

        List<Record> lapsed = trace.get("coin-change");
        check(lapsed.get(0).grade == 1 && Math.abs(lapsed.get(2).ef - MIN_EF) < 1e-12
                        && Math.abs(lapsed.get(3).ef - 1.4) < 1e-12 && lapsed.get(3).interval == 1
                        && lapsed.get(5).interval == 9,
                "repeated lapses must clamp the ease at 1.3 and recovery must resume growth: " + lapsed.get(2).render());

        Card borderline = new Card("edge");
        review(borderline, 3, 0);
        Card failed = new Card("edge-fail");
        review(failed, 2, 0);
        check(borderline.reps == 1 && borderline.lapses == 0 && Math.abs(borderline.ef - 2.36) < 1e-12
                        && failed.reps == 0 && failed.lapses == 1 && Math.abs(failed.ef - 1.7) < 1e-12
                        && failed.interval == 1,
                "grade 3 must pass and grade 2 must lapse: ef " + borderline.ef + " vs " + failed.ef);

        check(simulate(List.of(), DAYS).isEmpty()
                        && simulate(List.of(new Card("two-sum")), DAYS).get("two-sum").size() == 5
                        && Arrays.equals(daysOf(simulate(List.of(new Card("two-sum")), DAYS).get("two-sum")),
                        new int[]{0, 1, 7, 23, 68}),
                "an empty deck schedules nothing and a lone card follows the same review days");
        check(render(simulate(deck(), DAYS)).equals(render(trace)),
                "two simulations of the same deck must produce identical traces");

        System.out.println("cards=" + trace.size() + " reviews=" + all.size() + " horizon=" + DAYS);
        System.out.println("two-sum=" + Arrays.toString(daysOf(perfect)) + " " + Arrays.toString(intervalsOf(perfect)));
        System.out.println("PASS");
    }
}
