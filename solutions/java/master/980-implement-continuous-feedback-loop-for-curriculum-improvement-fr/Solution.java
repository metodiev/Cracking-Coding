/** Task 980 - Implement continuous feedback loop for curriculum improvement from user telemetry.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  A closed feedback loop over per-problem telemetry (attempts, successes, median seconds). Problems are
 *  classified against the desirable-difficulty band 0.40..0.85: below it too hard, above it too easy, and
 *  with too few attempts unmeasured. Each round proposes at most `budget` changes -- a hint for a too-hard
 *  problem (hints capped) or a shifted study position for either class -- and a proposal is accepted only
 *  when it lowers the objective (squared distance from the 0.65 target rate) and keeps every guardrail: the
 *  topic order survives, no problem is dropped and hint levels stay within the cap. Accepted changes feed a
 *  deterministic projection model (a hint is worth +0.05, moving a problem later +0.02, earlier -0.02) that
 *  becomes the next round's telemetry; rejected proposals are rolled back exactly.
 *  Simplified: an elasticity projection instead of a live A/B test and a greedy accept-only loop instead of
 *  a bandit; the banding, the accept-only monotonicity, the guardrail checks and the budget limit are real.
 *  Verified: classification against thresholds, proposal direction per class, the objective recomputed by
 *  brute force, monotone improvement across three rounds, the budget limit, guardrail preservation (topic
 *  order, hint cap, no deletion), exact rollback of a violating proposal, the empty-curriculum boundary and
 *  determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final double HARD = 0.40, EASY = 0.85, TARGET = 0.65, HINT_GAIN = 0.05, SHIFT = 0.02;
    static final int MIN_ATTEMPTS = 30, MAX_HINTS = 3, BUDGET = 3;
    static final List<String> TOPIC_ORDER = List.of("arrays", "hashing", "graphs", "dp");

    static final class Problem {
        final String id, topic; final int attempts; int hints, position; double rate;
        Problem(String id, String topic, int attempts, double rate, int hints, int position) {
            this.id = id; this.topic = topic; this.attempts = attempts; this.rate = rate;
            this.hints = hints; this.position = position;
        }
    }

    static String classify(Problem p) {
        if (p.attempts < MIN_ATTEMPTS) return "unmeasured";
        if (p.rate < HARD) return "too-hard";
        if (p.rate > EASY) return "too-easy";
        return "calibrated";
    }

    /** Objective: squared distance from the target rate, over measured problems only. */
    static double objective(List<Problem> problems) {
        double total = 0;
        for (Problem p : problems) {
            if (p.attempts < MIN_ATTEMPTS) continue;
            total += (p.rate - TARGET) * (p.rate - TARGET);
        }
        return total;
    }

    /** Independent recomputation: the same sum accumulated backwards through a Math.pow expression. */
    static double objectiveBruteForce(List<Problem> problems) {
        double sum = 0;
        for (int i = problems.size() - 1; i >= 0; i--) {
            Problem p = problems.get(i);
            if (p.attempts < MIN_ATTEMPTS) continue;
            sum += Math.pow(p.rate - TARGET, 2);
        }
        return sum;
    }

    /** Guardrail: every problem of an earlier topic must keep a strictly smaller position. */
    static boolean topicOrderPreserved(List<Problem> problems) {
        for (Problem a : problems) {
            for (Problem b : problems) {
                boolean earlier = TOPIC_ORDER.indexOf(a.topic) < TOPIC_ORDER.indexOf(b.topic);
                if (earlier && a.position >= b.position) return false;
            }
        }
        return true;
    }

    static final class Proposal {
        final String problemId, rationale; final int hintDelta, shift;
        Proposal(String problemId, int hintDelta, int shift, String rationale) {
            this.problemId = problemId; this.hintDelta = hintDelta; this.shift = shift; this.rationale = rationale;
        }
    }

    /** At most `budget` proposals, furthest-from-target measured problems first. */
    static List<Proposal> propose(List<Problem> problems, int budget) {
        List<Problem> ranked = new ArrayList<>(problems);
        ranked.sort(Comparator.comparingDouble((Problem p) -> -Math.abs(p.rate - TARGET)).thenComparing(p -> p.id));
        List<Proposal> out = new ArrayList<>();
        for (Problem p : ranked) {
            if (out.size() == budget) break;
            String kind = classify(p);
            if (kind.equals("too-hard")) out.add(new Proposal(p.id, p.hints < MAX_HINTS ? 1 : 0, 1, kind));
            else if (kind.equals("too-easy")) out.add(new Proposal(p.id, 0, -1, kind));
        }
        return out;
    }

    static Problem find(List<Problem> problems, String id) {
        for (Problem p : problems) if (p.id.equals(id)) return p;
        throw new IllegalArgumentException("unknown problem " + id);
    }

    /** Applies the proposals that lower the objective and keep the guardrails; returns the applied count. */
    static int apply(List<Problem> problems, List<Proposal> proposals, List<String> log) {
        int applied = 0;
        for (Proposal proposal : proposals) {
            Problem target = find(problems, proposal.problemId);
            int hintsBefore = target.hints, positionBefore = target.position;
            double rateBefore = target.rate;
            int hints = Math.min(MAX_HINTS, hintsBefore + proposal.hintDelta);
            int position = Math.max(0, positionBefore + proposal.shift);
            if (hints == hintsBefore && position == positionBefore) continue;
            double previousObjective = objective(problems);
            target.hints = hints;
            target.position = position;
            target.rate = Math.min(0.97, Math.max(0.05,
                    rateBefore + HINT_GAIN * proposal.hintDelta + SHIFT * proposal.shift));
            boolean violation = !topicOrderPreserved(problems);
            boolean worse = objective(problems) >= previousObjective;
            if (violation || worse) {
                target.hints = hintsBefore;
                target.position = positionBefore;
                target.rate = rateBefore;
                log.add("rejected " + proposal.problemId + " " + proposal.rationale
                        + (violation ? " (guardrail)" : " (no gain)"));
                continue;
            }
            applied++;
            log.add("accepted " + proposal.problemId + " " + proposal.rationale + " hints=" + target.hints
                    + " position=" + target.position);
        }
        return applied;
    }

    static List<Problem> curriculum() {
        return new ArrayList<>(List.of(
                new Problem("two-sum", "arrays", 200, 0.91, 0, 1),
                new Problem("sort-colors", "arrays", 150, 0.55, 0, 2),
                new Problem("bfs-maze", "graphs", 90, 0.22, 1, 4),
                new Problem("topo-sort", "graphs", 60, 0.35, 0, 5),
                new Problem("coin-change", "dp", 40, 0.80, 2, 6),
                new Problem("unseen", "dp", 5, 0.10, 0, 7)));
    }

    public static void main(String[] args) {
        List<Problem> problems = curriculum();
        Map<String, String> classes = new TreeMap<>();
        for (Problem p : problems) classes.put(p.id, classify(p));
        check(classes.get("two-sum").equals("too-easy") && classes.get("sort-colors").equals("calibrated")
                        && classes.get("bfs-maze").equals("too-hard") && classes.get("coin-change").equals("calibrated")
                        && classes.get("unseen").equals("unmeasured"),
                "classification must follow the band and the minimum sample size: " + classes);

        double initial = objective(problems);
        check(Math.abs(initial - objectiveBruteForce(problems)) < 1e-12 && Math.abs(initial - 0.375) < 1e-9,
                "the objective must match the brute-force recomputation: " + initial);

        List<Proposal> first = propose(problems, BUDGET);
        check(first.size() == BUDGET
                        && first.stream().allMatch(p -> classify(find(problems, p.problemId)).equals(p.rationale))
                        && first.stream().anyMatch(p -> p.rationale.equals("too-easy"))
                        && first.stream().anyMatch(p -> p.rationale.equals("too-hard")),
                "proposals must target measured, mis-calibrated problems: "
                        + first.stream().map(p -> p.problemId).toList());

        List<String> log = new ArrayList<>();
        List<Double> history = new ArrayList<>();
        double previous = initial;
        boolean monotone = true;
        for (int round = 0; round < 3; round++) {
            int applied = apply(problems, propose(problems, BUDGET), log);
            double now = objective(problems);
            monotone &= now <= previous + 1e-12 && applied <= BUDGET;
            history.add(now);
            previous = now;
        }
        check(monotone && previous < initial && log.size() >= 3 && log.stream().anyMatch(l -> l.startsWith("accepted")),
                "each round must respect the budget and never increase the objective: " + history);

        check(topicOrderPreserved(problems) && problems.size() == 6
                        && problems.stream().allMatch(p -> p.hints <= MAX_HINTS)
                        && problems.stream().anyMatch(p -> p.hints > find(curriculum(), p.id).hints)
                        && Math.abs(find(problems, "two-sum").rate - 0.89) < 1e-12
                        && find(problems, "two-sum").position == 0,
                "guardrails must hold while hints grow and the easy problem cools down: "
                        + problems.stream().map(p -> p.id + ":" + p.hints).toList());

        List<Problem> adversarial = curriculum();
        List<String> guardLog = new ArrayList<>();
        int applied = apply(adversarial, List.of(new Proposal("two-sum", 0, 3, "too-easy")), guardLog);
        check(applied == 0 && guardLog.get(0).contains("guardrail") && find(adversarial, "two-sum").rate == 0.91
                        && find(adversarial, "two-sum").position == 1,
                "a proposal that breaks the topic order must be rolled back exactly: " + guardLog);

        List<Problem> empty = new ArrayList<>();
        check(objective(empty) == 0.0 && objectiveBruteForce(empty) == 0.0 && propose(empty, BUDGET).isEmpty()
                        && apply(empty, List.of(), new ArrayList<>()) == 0,
                "an empty curriculum must produce no proposals and a zero objective");

        List<Problem> fresh = curriculum();
        List<String> logAgain = new ArrayList<>();
        for (int round = 0; round < 3; round++) apply(fresh, propose(fresh, BUDGET), logAgain);
        check(logAgain.equals(log) && Math.abs(objective(fresh) - previous) < 1e-12,
                "replaying the loop on the same telemetry must be deterministic");

        System.out.println("problems=" + problems.size() + " objective=" + initial + " -> " + previous
                + " rejected=" + log.stream().filter(l -> l.startsWith("rejected")).count());
        System.out.println("hints=" + problems.stream().map(p -> p.id + ":" + p.hints).toList());
        System.out.println("PASS");
    }
}
