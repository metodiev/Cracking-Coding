/** Task 992 - Implement reproducible random-seed protocol for benchmarking fairness.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  Two properties make a benchmark fair: every variant must get exactly the same amount of work, and
 *  the randomness each run sees must be reproducible and independent. Seeds are therefore derived -
 *  never drawn - from (master seed, task, variant, round, run index) by mixing the fields with FNV-1a
 *  and then avalanching with splitmix64, so the same protocol on any machine replays the identical
 *  stream while different (task, variant) pairs get unrelated streams. The schedule is a per-round
 *  rotation of the variant list: for round r the i-th slot runs variant (i + r) mod V, which gives
 *  every variant the same number of runs per task (R * K) and spreads each variant evenly over the
 *  starting positions, removing order bias. The protocol records every (task, variant, round, run)
 *  assignment with its seed and drops nothing, so duplicates and omissions are detectable.
 *  Simplified: simulated nodes and synthetic tasks with no real timing; the derivation, the rotation
 *  schedule and the balance audits are the real ones.
 *  Verified: derivation is a pure function (two protocol instances agree, replays match), all seeds are
 *  pairwise distinct, per-task and per-variant counts are uniform, starting positions are balanced,
 *  there are no duplicate assignments, the stream passes a bucket/uniformity sanity check, the
 *  single-variant and zero-round boundaries behave, and an independent nested-loop audit agrees.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static long fnv64(String s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= 0x100000001b3L; }
        return h;
    }

    static long splitmix64(long state) {
        long z = state + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** The protocol: a pure derivation from the experiment description to a 64-bit seed. */
    static final class Protocol {
        final long masterSeed;
        Protocol(long masterSeed) { this.masterSeed = masterSeed; }

        long seedFor(String task, String variant, int round, int run) {
            long mixed = fnv64(task) ^ Long.rotateLeft(fnv64(variant), 17);
            mixed ^= splitmix64(masterSeed + 31L * round + 131L * run);
            return splitmix64(mixed);
        }

        double[] draws(String task, String variant, int round, int run, int count) {
            long state = seedFor(task, variant, round, run);
            double[] out = new double[count];
            for (int i = 0; i < count; i++) {
                state = splitmix64(state);
                out[i] = (state >>> 11) * 0x1.0p-53;
            }
            return out;
        }
    }

    static final class Assignment {
        final String task, variant;
        final int round, run;
        final long seed;
        Assignment(String task, String variant, int round, int run, long seed) {
            this.task = task; this.variant = variant; this.round = round; this.run = run; this.seed = seed;
        }
        String key() { return task + "|" + variant + "|" + round + "|" + run; }
    }

    /** The planned experiment: rotation schedule plus one derived seed per run. */
    static final class Schedule {
        final List<String> tasks, variants;
        final int rounds, reruns;
        final List<Assignment> assignments = new ArrayList<>();
        final Map<String, Integer> positionOf = new LinkedHashMap<>();

        Schedule(Protocol protocol, List<String> tasks, List<String> variants, int rounds, int reruns) {
            this.tasks = tasks; this.variants = variants; this.rounds = rounds; this.reruns = reruns;
            for (int round = 0; round < rounds; round++)
                for (String task : tasks)
                    for (int slot = 0; slot < variants.size(); slot++) {
                        String variant = variants.get((slot + round) % variants.size());
                        positionOf.put(task + "|" + round + "|" + variant, slot);
                        for (int run = 0; run < reruns; run++)
                            assignments.add(new Assignment(task, variant, round, run,
                                    protocol.seedFor(task, variant, round, run)));
                    }
        }

        Map<String, Integer> countsByTaskVariant() {
            Map<String, Integer> counts = new TreeMap<>();
            for (Assignment a : assignments) counts.merge(a.task + "|" + a.variant, 1, Integer::sum);
            return counts;
        }

        Map<String, Integer> countsByPosition() {
            Map<String, Integer> counts = new TreeMap<>();
            for (Assignment a : assignments) {
                int slot = positionOf.get(a.task + "|" + a.round + "|" + a.variant);
                counts.merge(a.variant + "@" + slot, 1, Integer::sum);
            }
            return counts;
        }
    }

    static List<String> tasks() { return List.of("sort-1m", "dijkstra-100k", "hashmap-mixed"); }
    static List<String> variants() { return List.of("alpha", "beta", "gamma", "delta"); }

    public static void main(String[] args) {
        Protocol protocol = new Protocol(20261016L);
        Schedule schedule = new Schedule(protocol, tasks(), variants(), 4, 2);

        // --- derivation is pure: a second protocol instance and a fresh call agree exactly.
        Protocol twin = new Protocol(20261016L);
        check(protocol.seedFor("sort-1m", "alpha", 0, 0) == twin.seedFor("sort-1m", "alpha", 0, 0)
                        && Arrays.equals(protocol.draws("sort-1m", "alpha", 0, 0, 5),
                                twin.draws("sort-1m", "alpha", 0, 0, 5)),
                "the same protocol description must derive the same stream on any machine");
        double[] again = protocol.draws("sort-1m", "alpha", 0, 0, 5);
        check(Arrays.equals(again, protocol.draws("sort-1m", "alpha", 0, 0, 5)),
                "re-deriving a stream must replay it identically: " + Arrays.toString(again));
        check(schedule.assignments.get(0).seed == protocol.seedFor("sort-1m", "alpha", 0, 0)
                        && schedule.assignments.get(0).key().equals("sort-1m|alpha|0|0"),
                "the recorded seed must be the derived seed, not a stored copy");

        // --- a different master seed must give an unrelated protocol (no global state).
        Protocol other = new Protocol(20261017L);
        check(other.seedFor("sort-1m", "alpha", 0, 0) != protocol.seedFor("sort-1m", "alpha", 0, 0)
                        && !Arrays.equals(other.draws("sort-1m", "alpha", 0, 0, 5), again),
                "a different master seed must yield a different stream");

        // --- independence: every (task, variant, round, run) seed must be pairwise distinct.
        Set<Long> seeds = new HashSet<>();
        Set<String> keys = new HashSet<>();
        for (Assignment a : schedule.assignments) { seeds.add(a.seed); keys.add(a.key()); }
        check(seeds.size() == schedule.assignments.size(),
                "all derived seeds must be distinct: " + seeds.size() + " of " + schedule.assignments.size());
        check(keys.size() == schedule.assignments.size(),
                "no assignment may be scheduled twice: " + keys.size() + " of " + schedule.assignments.size());
        check(protocol.seedFor("sort-1m", "alpha", 0, 0) != protocol.seedFor("dijkstra-100k", "alpha", 0, 0)
                        && protocol.seedFor("sort-1m", "alpha", 0, 0) != protocol.seedFor("sort-1m", "beta", 0, 0),
                "changing the task or the variant must change the seed");

        // --- fairness: equal work per (task, variant), equal exposure per starting position.
        int expected = schedule.rounds * schedule.reruns;
        check(schedule.assignments.size() == tasks().size() * variants().size() * schedule.rounds * schedule.reruns,
                "the schedule must contain T*V*R*K runs: " + schedule.assignments.size());
        Map<String, Integer> counts = schedule.countsByTaskVariant();
        check(counts.size() == tasks().size() * variants().size()
                        && counts.values().stream().allMatch(c -> c == expected),
                "every variant must run every task exactly " + expected + " times: " + counts);
        Map<String, Integer> positions = schedule.countsByPosition();
        int perPosition = schedule.rounds / variants().size() * tasks().size() * schedule.reruns;
        check(positions.size() == variants().size() * variants().size()
                        && positions.values().stream().allMatch(c -> c == perPosition),
                "each variant must occupy each starting position equally often: " + positions);

        // --- replay the recorded protocol: re-derived draws must equal the recorded seeds' streams.
        int replayed = 0;
        for (Assignment a : schedule.assignments) {
            double[] stream = protocol.draws(a.task, a.variant, a.round, a.run, 3);
            long rederived = protocol.seedFor(a.task, a.variant, a.round, a.run);
            check(rederived == a.seed && stream.length == 3 && stream[0] >= 0 && stream[0] < 1,
                    "replayed run must match the recorded seed: " + a.key());
            replayed++;
        }
        check(replayed == schedule.assignments.size(), "every assignment must be replayable: " + replayed);

        // --- stream sanity (fairness pre-condition): buckets and mean look uniform across variants.
        int buckets = 10, perBucket = 200;
        Map<String, Integer> histogram = new TreeMap<>();
        double sum = 0;
        int total = 0;
        for (String variant : variants()) {
            Protocol p = new Protocol(20261016L);
            for (int i = 0; i < perBucket; i++) {
                double d = p.draws("sort-1m", variant, 0, i, 1)[0];
                histogram.merge(variant + "#" + (int) (d * buckets), 1, Integer::sum);
                sum += d;
                total++;
            }
        }
        int expectedPerBucket = perBucket / buckets;
        int worst = 0;
        for (String variant : variants())
            for (int b = 0; b < buckets; b++)
                worst = Math.max(worst, Math.abs(histogram.getOrDefault(variant + "#" + b, 0) - expectedPerBucket));
        check(worst <= expectedPerBucket * 3 / 4,
                "the derived stream must be roughly uniform, worst bucket deviation " + worst);
        check(sum / total > 0.45 && sum / total < 0.55,
                "the derived stream mean must be near 0.5: " + sum / total);

        // --- boundaries: a single variant is trivially balanced, zero rounds schedule nothing.
        Schedule single = new Schedule(protocol, tasks(), List.of("solo"), 3, 1);
        check(single.assignments.size() == tasks().size() * 3 && single.countsByPosition().size() == 1
                        && single.countsByPosition().values().iterator().next() == tasks().size() * 3,
                "a single variant must be perfectly balanced: " + single.countsByPosition());
        Schedule noRounds = new Schedule(protocol, tasks(), variants(), 0, 2);
        check(noRounds.assignments.isEmpty() && noRounds.countsByTaskVariant().isEmpty(),
                "zero rounds must schedule nothing at all");

        // --- independent oracle: nested-loop audit of the balance and of the derivation path.
        check(oracleAudit(schedule), "independent nested-loop balance audit must agree");
        check(oracleDistinctSeeds(schedule), "independent pairwise seed audit must agree");

        System.out.println("runs: " + schedule.assignments.size() + " (" + tasks().size() + " tasks x "
                + variants().size() + " variants x " + schedule.rounds + " rounds x " + schedule.reruns + " reruns)");
        System.out.println("work per task/variant: " + expected + ", position exposure: " + perPosition);
        System.out.println("distinct seeds: " + seeds.size() + ", stream mean " + String.format("%.4f", sum / total));
        System.out.println("PASS");
    }

    /** Oracle 1: audit fairness with its own nested loops instead of the schedule's helpers. */
    static boolean oracleAudit(Schedule schedule) {
        for (String task : schedule.tasks)
            for (String variant : schedule.variants) {
                int found = 0;
                for (Assignment a : schedule.assignments)
                    if (a.task.equals(task) && a.variant.equals(variant)) found++;
                if (found != schedule.rounds * schedule.reruns) return false;
            }
        for (String variant : schedule.variants)
            for (int slot = 0; slot < schedule.variants.size(); slot++) {
                int found = 0;
                for (Assignment a : schedule.assignments) {
                    Integer recorded = schedule.positionOf.get(a.task + "|" + a.round + "|" + a.variant);
                    if (recorded != null && recorded == slot && a.variant.equals(variant)) found++;
                }
                if (found != schedule.tasks.size() * schedule.reruns * schedule.rounds / schedule.variants.size()) return false;
            }
        return true;
    }

    /** Oracle 2: recount the seeds with an O(n^2) comparison instead of a HashSet. */
    static boolean oracleDistinctSeeds(Schedule schedule) {
        List<Assignment> all = schedule.assignments;
        for (int i = 0; i < all.size(); i++)
            for (int j = i + 1; j < all.size(); j++)
                if (all.get(i).seed == all.get(j).seed) return false;
        return true;
    }
}
