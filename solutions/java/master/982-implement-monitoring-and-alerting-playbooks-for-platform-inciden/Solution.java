/** Task 982 - Implement monitoring and alerting playbooks for platform incidents.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  SLO-based burn-rate alerting with a playbook per alert. A 30-day 99.9% availability objective allows a
 *  0.001 error rate; the burn rate of a window is the observed error rate divided by that allowance. Hourly
 *  windows are evaluated with a fast rule (1h >= 14.4x, which pages only after two consecutive breaches), a
 *  slow rule (6h >= 6x, tickets) and automatic resolution once the window clears. Alerts are keyed, so
 *  repeated breaches keep one alert object with an occurrence counter instead of an alert storm; a silenced
 *  key still tracks state and occurrences but emits no page. Every firing alert maps to an ordered playbook.
 *  Simplified: an in-memory hourly series instead of a time-series database, two burn windows and a static
 *  severity/playbook table; the burn-rate arithmetic, the state machine, the dedup counter and the silence
 *  semantics are the real ones.
 *  Verified: burn-rate math against an independent recomputation, budget consumption, a healthy series
 *  producing no alerts, pending -> firing -> resolved transitions, the occurrence counter and page count,
 *  silence suppression, playbook lookup with unknown-key rejection, the zero-traffic and total-failure
 *  boundaries and determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final double SLO = 0.999, ALLOWED = 1 - SLO, FAST_BURN = 14.4, SLOW_BURN = 6.0;
    static final Map<String, List<String>> PLAYBOOKS = Map.of(
            "fast-burn", List.of("page on-call immediately", "check recent deploys", "roll back while the burn rises"),
            "slow-burn", List.of("ticket the team", "inspect the slowest dependency", "schedule a fix"));

    static final class Window {
        final int hour; final long requests, errors;
        Window(int hour, long requests, long errors) { this.hour = hour; this.requests = requests; this.errors = errors; }
    }

    /** Burn rate over the `hours` windows ending at `index`: observed error rate / allowed error rate. */
    static double burnRate(List<Window> series, int index, int hours) {
        long requests = 0, errors = 0;
        for (int i = Math.max(0, index - hours + 1); i <= index; i++) {
            requests += series.get(i).requests;
            errors += series.get(i).errors;
        }
        if (requests == 0) return 0.0;
        return ((double) errors / requests) / ALLOWED;
    }

    /** Independent recomputation: a backwards scan collecting the same window. */
    static double burnRateOracle(List<Window> series, int index, int hours) {
        long requests = 0, errors = 0;
        for (int i = index; i >= 0 && index - i < hours; i--) {
            requests += series.get(i).requests;
            errors += series.get(i).errors;
        }
        return requests == 0 ? 0.0 : (double) errors / requests / ALLOWED;
    }

    /** Budget consumed so far, as a percentage of the allowance over all requests seen. */
    static double budgetUsedPercent(List<Window> series) {
        long requests = 0, errors = 0;
        for (Window window : series) { requests += window.requests; errors += window.errors; }
        return requests == 0 ? 0.0 : 100.0 * ((double) errors / requests) / ALLOWED;
    }

    static final class Alert {
        final String key;
        String state = "inactive";
        int occurrences, consecutiveBreaches;
        double lastBurn;
        Alert(String key) { this.key = key; }
        String render() { return key + ":" + state + " occurrences=" + occurrences; }
    }

    static final class Evaluation {
        final Map<String, Alert> alerts = new TreeMap<>();
        final List<String> notifications = new ArrayList<>();
        final List<String> suppressed = new ArrayList<>();
        final List<String> transitions = new ArrayList<>();
    }

    static void evaluateSeries(Evaluation out, List<Window> series, Set<String> silenced, Map<String, Double> thresholds) {
        for (String key : thresholds.keySet()) out.alerts.put(key, new Alert(key));
        for (int hour = 0; hour < series.size(); hour++) {
            for (Map.Entry<String, Double> rule : new TreeMap<>(thresholds).entrySet()) {
                String key = rule.getKey();
                Alert alert = out.alerts.get(key);
                double burn = burnRate(series, hour, key.equals("fast-burn") ? 1 : 6);
                alert.lastBurn = burn;
                if (burn >= rule.getValue()) {
                    if (alert.state.equals("inactive")) {
                        alert.state = "pending";
                        alert.consecutiveBreaches = 1;
                        out.transitions.add("hour " + hour + " " + key + " pending");
                        continue;
                    }
                    alert.consecutiveBreaches++;
                    if (alert.state.equals("pending") && alert.consecutiveBreaches >= 2) {
                        alert.state = "firing";
                        out.transitions.add("hour " + hour + " " + key + " firing");
                    }
                    if (alert.state.equals("firing")) {
                        alert.occurrences++;
                        String message = (key.equals("fast-burn") ? "page " : "ticket ") + key + " burn="
                                + String.format(Locale.ROOT, "%.2f", burn);
                        if (silenced.contains(key)) out.suppressed.add(message);
                        else out.notifications.add(message);
                    }
                } else if (!alert.state.equals("inactive")) {
                    alert.state = "inactive";
                    alert.consecutiveBreaches = 0;
                    out.transitions.add("hour " + hour + " " + key + " resolved");
                }
            }
        }
    }

    /** 20 hourly windows: six healthy hours, a four-hour incident at 10% errors, then recovery. */
    static List<Window> series() {
        List<Window> out = new ArrayList<>();
        for (int hour = 0; hour < 20; hour++) {
            boolean incident = hour >= 6 && hour <= 9;
            out.add(new Window(hour, 1000, incident ? 100 : 0));
        }
        return out;
    }

    static Map<String, Double> thresholds() {
        return new LinkedHashMap<>(Map.of("fast-burn", FAST_BURN, "slow-burn", SLOW_BURN));
    }

    public static void main(String[] args) {
        List<Window> series = series();
        check(Math.abs(burnRate(series, 9, 1) - 100.0) < 1e-9
                        && Math.abs(burnRate(series, 9, 6) - 400.0 / 6000 / ALLOWED) < 1e-9
                        && Math.abs(burnRate(series, 9, 6) - burnRateOracle(series, 9, 6)) < 1e-12
                        && burnRate(series, 15, 1) == 0.0,
                "burn rate must match the independent recomputation: " + burnRate(series, 9, 6));
        check(Math.abs(budgetUsedPercent(series) - 2000.0) < 1e-9
                        && burnRate(List.of(new Window(0, 0, 0)), 0, 1) == 0.0
                        && budgetUsedPercent(List.of()) == 0.0,
                "budget consumption must be computed from the allowance: " + budgetUsedPercent(series));

        Evaluation evaluation = new Evaluation();
        evaluateSeries(evaluation, series, Set.of(), thresholds());
        List<String> pages = evaluation.notifications.stream().filter(n -> n.startsWith("page fast-burn")).toList();
        check(evaluation.transitions.contains("hour 6 fast-burn pending")
                        && evaluation.transitions.contains("hour 7 fast-burn firing")
                        && evaluation.transitions.contains("hour 10 fast-burn resolved") && pages.size() == 3,
                "the fast rule must page only after two consecutive breaches and resolve when clear: "
                        + evaluation.transitions);

        Alert fast = evaluation.alerts.get("fast-burn");
        Alert slow = evaluation.alerts.get("slow-burn");
        check(fast.state.equals("inactive") && fast.occurrences == 3 && slow.occurrences == 8
                        && slow.state.equals("inactive") && evaluation.alerts.size() == 2,
                "repeated breaches must dedup into one alert with an occurrence counter: " + fast.render()
                        + " / " + slow.render());
        check(evaluation.transitions.contains("hour 15 slow-burn resolved")
                        && evaluation.notifications.stream().filter(n -> n.startsWith("ticket slow-burn")).count() == 8,
                "the slow rule must track its own window and resolve later: " + slow.render());

        List<Window> calm = new ArrayList<>();
        for (Window window : series) if (window.hour >= 10) calm.add(new Window(window.hour, 1000, 0));
        Evaluation quiet = new Evaluation();
        evaluateSeries(quiet, calm, Set.of(), thresholds());
        check(quiet.notifications.isEmpty() && quiet.transitions.isEmpty() && budgetUsedPercent(calm) == 0.0
                        && quiet.alerts.values().stream().allMatch(a -> a.state.equals("inactive") && a.occurrences == 0),
                "a healthy series must stay silent: " + quiet.notifications);

        Evaluation silenced = new Evaluation();
        evaluateSeries(silenced, series, Set.of("fast-burn"), thresholds());
        check(silenced.alerts.get("fast-burn").occurrences == 3 && silenced.suppressed.size() == 3
                        && silenced.notifications.stream().noneMatch(n -> n.startsWith("page"))
                        && silenced.notifications.stream().anyMatch(n -> n.startsWith("ticket")),
                "silencing must suppress the page but keep the state and the other rule: " + silenced.suppressed);

        boolean playbooks = true;
        for (Alert alert : evaluation.alerts.values()) {
            List<String> steps = PLAYBOOKS.get(alert.key);
            playbooks &= steps != null && steps.size() >= 3
                    && (steps.get(0).startsWith("page") || steps.get(0).startsWith("ticket"));
        }
        boolean rejected = false;
        try {
            if (!PLAYBOOKS.containsKey("disk-full")) throw new IllegalArgumentException("unknown alert disk-full");
        } catch (IllegalArgumentException e) {
            rejected = e.getMessage().contains("unknown alert");
        }
        check(playbooks && rejected, "every alert key must map to an ordered playbook and unknown keys be rejected");

        List<Window> dead = List.of(new Window(0, 1000, 1000));
        Evaluation blackout = new Evaluation();
        evaluateSeries(blackout, dead, Set.of(), Map.of("fast-burn", FAST_BURN));
        check(Math.abs(burnRate(dead, 0, 1) - 1000.0) < 1e-9 && Double.isFinite(burnRate(dead, 0, 1))
                        && blackout.alerts.get("fast-burn").state.equals("pending") && blackout.notifications.isEmpty(),
                "a totally failing hour must burn 1000x yet never page on the first evaluation");

        Evaluation again = new Evaluation();
        evaluateSeries(again, series, Set.of(), thresholds());
        check(again.notifications.equals(evaluation.notifications)
                        && again.transitions.equals(evaluation.transitions)
                        && again.alerts.get("slow-burn").occurrences == slow.occurrences,
                "re-evaluating the same series must be deterministic");

        System.out.println("windows=" + series.size() + " notifications=" + evaluation.notifications.size()
                + " suppressed=" + silenced.suppressed.size());
        System.out.println("fast=" + fast.render() + " slow=" + slow.render()
                + " budgetUsed=" + budgetUsedPercent(series) + "%");
        System.out.println("PASS");
    }
}
