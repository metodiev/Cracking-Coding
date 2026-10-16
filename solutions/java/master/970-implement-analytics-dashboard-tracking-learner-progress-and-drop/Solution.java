/** Task 970 - Implement analytics dashboard tracking learner progress and drop-off.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  An analytics dashboard over a learner event stream. Events carry a learner id, a progress phase
 *  (0 viewed, 1 attempted, 2 solved) and a day. The dashboard reports (a) a phase funnel of distinct
 *  learners with per-step conversion and drop-off, (b) the same funnel per learner segment plus the
 *  roll-up check that segments partition the whole, and (c) a day-N retention curve over the cohort that
 *  started on day 0 (first-activity day == 0).
 *  Simplified: phases are treated as independent milestones rather than a strict ordered sequence, and the
 *  retention curve is computed on a fixed set of horizon days instead of a full survival model; the
 *  dedup-by-learner funnel, the drop-off arithmetic and the cohort/telescoping identities are the real
 *  ones.
 *  Verified: funnel counts against a brute-force oracle, conversion and drop-off arithmetic with the
 *  telescoping identity, segment roll-up against the overall funnel, retention on a scripted cohort,
 *  monotonicity of both funnel and retention, the no-data boundary (finite zero rates) and determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final class Event {
        final String learner; final int phase, day;
        Event(String learner, int phase, int day) { this.learner = learner; this.phase = phase; this.day = day; }
    }

    static final class Funnel {
        final int[] counts;                                                  // distinct learners per phase
        Funnel(int[] counts) { this.counts = counts; }
        double rate(int step) { return counts[step] == 0 ? 0.0 : (double) counts[step + 1] / counts[step]; }
        double drop(int step) { return 1.0 - rate(step); }
        String render() { return Arrays.toString(counts); }
    }

    /** Distinct-learner funnel; `cohort == null` means every learner in the stream counts. */
    static Funnel funnel(List<Event> events, Set<String> cohort) {
        Map<String, boolean[]> seen = new TreeMap<>();
        for (Event e : events) {
            if (cohort != null && !cohort.contains(e.learner)) continue;
            seen.computeIfAbsent(e.learner, k -> new boolean[3])[e.phase] = true;
        }
        int[] counts = new int[3];
        for (boolean[] s : seen.values()) for (int i = 0; i < 3; i++) if (s[i]) counts[i]++;
        return new Funnel(counts);
    }

    /** Independent oracle for the funnel counts: for every learner and phase, scan the whole stream. */
    static int[] funnelOracle(List<Event> events) {
        Set<String> learners = new TreeSet<>();
        for (Event e : events) learners.add(e.learner);
        int[] counts = new int[3];
        for (String learner : learners) {
            for (int phase = 0; phase < 3; phase++) {
                for (Event e : events) {
                    if (e.learner.equals(learner) && e.phase == phase) { counts[phase]++; break; }
                }
            }
        }
        return counts;
    }

    /** Cohort = learners whose first activity day is `day0`; value = share active on day0 + horizon. */
    static Map<Integer, Double> retention(List<Event> events, int day0, List<Integer> horizons) {
        Map<String, Integer> firstDay = new TreeMap<>();
        for (Event e : events) firstDay.merge(e.learner, e.day, Math::min);
        Set<String> cohort = new TreeSet<>();
        for (Map.Entry<String, Integer> e : firstDay.entrySet()) if (e.getValue() == day0) cohort.add(e.getKey());
        Map<Integer, Double> out = new TreeMap<>();
        for (int horizon : horizons) {
            Set<String> active = new TreeSet<>();
            for (Event e : events) if (cohort.contains(e.learner) && e.day == day0 + horizon) active.add(e.learner);
            out.put(horizon, cohort.isEmpty() ? 0.0 : (double) active.size() / cohort.size());
        }
        return out;
    }

    static final class Dashboard {
        final Funnel overall; final Map<String, Funnel> segments; final Map<Integer, Double> retention;
        Dashboard(Funnel overall, Map<String, Funnel> segments, Map<Integer, Double> retention) {
            this.overall = overall; this.segments = segments; this.retention = retention;
        }
        String render() {
            StringBuilder out = new StringBuilder("overall" + overall.render());
            for (Map.Entry<String, Funnel> e : segments.entrySet()) out.append(' ').append(e.getKey()).append(e.getValue().render());
            for (Map.Entry<Integer, Double> e : retention.entrySet()) {
                out.append(" d").append(e.getKey()).append('=').append(String.format(Locale.ROOT, "%.3f", e.getValue()));
            }
            return out.toString();
        }
    }

    static Dashboard build(List<Event> events, Map<String, String> segmentOf) {
        Map<String, Funnel> segments = new TreeMap<>();
        for (String segment : new TreeSet<>(segmentOf.values())) {
            Set<String> cohort = new TreeSet<>();
            for (Map.Entry<String, String> e : segmentOf.entrySet()) if (e.getValue().equals(segment)) cohort.add(e.getKey());
            segments.put(segment, funnel(events, cohort));
        }
        return new Dashboard(funnel(events, null), segments, retention(events, 0, List.of(1, 3, 7)));
    }

    static Map<String, String> segmentOf() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("L1", "novice"); m.put("L2", "novice");
        m.put("L3", "expert"); m.put("L4", "expert"); m.put("L5", "expert");
        return m;
    }

    /** Scripted stream: L1 solves on day 0 and returns, L2 stalls, L3 solves late, L4 joins on day 2. */
    static List<Event> stream() {
        return List.of(
                new Event("L1", 0, 0), new Event("L1", 1, 0), new Event("L1", 2, 0),
                new Event("L1", 1, 1), new Event("L1", 2, 3), new Event("L1", 1, 8),
                new Event("L2", 0, 0), new Event("L2", 1, 0), new Event("L2", 0, 1),
                new Event("L3", 0, 0), new Event("L3", 1, 0), new Event("L3", 2, 1),
                new Event("L3", 1, 3), new Event("L3", 2, 7),
                new Event("L4", 0, 2), new Event("L4", 2, 2),
                new Event("L5", 0, 5));
    }

    public static void main(String[] args) {
        List<Event> events = stream();
        Dashboard dash = build(events, segmentOf());
        check(Arrays.equals(dash.overall.counts, funnelOracle(events))
                        && Arrays.equals(dash.overall.counts, new int[]{5, 3, 3}),
                "the funnel must agree with the brute-force oracle: " + dash.overall.render());

        check(Math.abs(dash.overall.rate(0) - 0.6) < 1e-12 && dash.overall.rate(1) == 1.0
                        && Math.abs(dash.overall.drop(0) - 0.4) < 1e-12 && dash.overall.drop(1) == 0.0
                        && Math.abs(dash.overall.drop(0) + dash.overall.rate(0) - 1.0) < 1e-12
                        && Math.abs(dash.overall.rate(0) * dash.overall.rate(1)
                        - (double) dash.overall.counts[2] / dash.overall.counts[0]) < 1e-12,
                "drop-off must be the complement of conversion and conversions must telescope");

        int[] rollup = new int[3];
        for (Funnel f : dash.segments.values()) for (int i = 0; i < 3; i++) rollup[i] += f.counts[i];
        check(Arrays.equals(rollup, dash.overall.counts) && dash.segments.size() == 2
                        && Arrays.equals(dash.segments.get("novice").counts, new int[]{2, 2, 1})
                        && Arrays.equals(dash.segments.get("expert").counts, new int[]{3, 1, 2}),
                "segments must partition the overall funnel: " + dash.segments.get("novice").render()
                        + " / " + dash.segments.get("expert").render());

        Map<Integer, Double> ret = dash.retention;
        check(ret.get(1) == 1.0 && Math.abs(ret.get(3) - 2.0 / 3) < 1e-12 && Math.abs(ret.get(7) - 1.0 / 3) < 1e-12
                        && ret.get(1) >= ret.get(3) && ret.get(3) >= ret.get(7),
                "retention must fall monotonically on the day-0 cohort: " + ret);

        Set<String> day0 = new TreeSet<>();
        for (Event e : events) if (e.day == 0) day0.add(e.learner);
        check(day0.equals(new TreeSet<>(List.of("L1", "L2", "L3"))) && !day0.contains("L4")
                        && dash.overall.counts[2] == 3 && dash.segments.get("novice").drop(1) == 0.5,
                "the cohort must be first-activity-day-0 learners only, so L4 is excluded: " + day0);
        check(dash.segments.get("novice").counts[2] <= dash.segments.get("novice").counts[1]
                        && dash.segments.get("expert").counts[1] <= dash.segments.get("expert").counts[0],
                "no phase may ever reach more learners than a phase above it");

        Dashboard empty = build(List.of(), Map.of());
        check(empty.overall.counts[0] == 0 && empty.overall.rate(0) == 0.0 && empty.overall.drop(0) == 1.0
                        && !Double.isNaN(empty.overall.rate(0))
                        && empty.retention.values().stream().allMatch(v -> v == 0.0),
                "an empty stream must report finite zero rates instead of NaN: " + empty.render());

        check(build(stream(), segmentOf()).render().equals(dash.render())
                        && build(stream(), segmentOf()).render().equals(dash.render()),
                "rebuilding the dashboard twice must be byte-identical");

        System.out.println("events=" + events.size() + " learners=" + segmentOf().size()
                + " funnel=" + dash.overall.render());
        System.out.println("retention=" + ret);
        System.out.println("PASS");
    }
}
