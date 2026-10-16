/** Task 956 - Implement grading rubric generator that assigns difficulty and expected complexity.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory rubric generator. A task is described by five integer features (input scale, state depth,
 *  number of edge cases, proof obligations, code volume - each 0..3) plus an algorithm class. The
 *  generator builds a weighted score, maps it through fixed thresholds to a difficulty level 1..5, looks
 *  the expected complexity up in a class table, estimates effort in minutes, and emits the two features
 *  with the largest contribution as the rationale.
 *  Simplified: the feature vector is supplied instead of being extracted from a task description, and the
 *  complexity table is a lookup rather than a static analysis result; the scoring, thresholds,
 *  validation and tie-breaking rules are the real ones.
 *  Verified: weights sum to one and every rubric is valid; the score and difficulty are monotone in each
 *  feature; class lookup covers every class; the zero and maximal boundaries sit at difficulty 1 and 5;
 *  scoring is independent of iteration order and reproducible; invalid profiles are rejected; level
 *  score ranges never overlap; an independent dot product reproduces every score and threshold score
 *  lands in the expected level.
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

    static final Map<String, Double> WEIGHTS = new LinkedHashMap<>();

    static {
        WEIGHTS.put("inputScale", 0.25);
        WEIGHTS.put("stateDepth", 0.20);
        WEIGHTS.put("edgeCases", 0.20);
        WEIGHTS.put("proofObligations", 0.20);
        WEIGHTS.put("codeVolume", 0.15);
    }

    static final Map<String, String> COMPLEXITY_BY_CLASS = new LinkedHashMap<>();

    static {
        COMPLEXITY_BY_CLASS.put("constant", "O(1)");
        COMPLEXITY_BY_CLASS.put("logarithmic", "O(log n)");
        COMPLEXITY_BY_CLASS.put("linear", "O(n)");
        COMPLEXITY_BY_CLASS.put("linearithmic", "O(n log n)");
        COMPLEXITY_BY_CLASS.put("quadratic", "O(n^2)");
        COMPLEXITY_BY_CLASS.put("exponential", "O(2^n)");
    }

    record Features(int inputScale, int stateDepth, int edgeCases, int proofObligations, int codeVolume,
                    String algorithmClass) {
        int feature(String name) {
            switch (name) {
                case "inputScale": return inputScale;
                case "stateDepth": return stateDepth;
                case "edgeCases": return edgeCases;
                case "proofObligations": return proofObligations;
                default: return codeVolume;
            }
        }
    }

    record Rubric(int difficulty, String expectedComplexity, double score, int estimatedMinutes,
                  List<String> rationale, String digest) {}

    /** Weighted score in 0..100, rounded to one decimal so comparisons are exact. */
    static double weightedScore(Features f, List<String> order) {
        double score = 0;
        for (String name : order) score += WEIGHTS.get(name) * f.feature(name) / 3.0;
        return Math.round(score * 100 * 10) / 10.0;
    }

    static int difficultyFor(double score) {
        if (score < 20) return 1;
        if (score < 40) return 2;
        if (score < 60) return 3;
        if (score < 80) return 4;
        return 5;
    }

    /** Independently written threshold ladder, used as the oracle in the boundary check. */
    static int oracleDifficulty(double score) {
        if (score >= 80) return 5;
        if (score >= 60) return 4;
        if (score >= 40) return 3;
        if (score >= 20) return 2;
        return 1;
    }

    static List<String> validate(Features f) {
        List<String> errors = new ArrayList<>();
        for (String name : WEIGHTS.keySet()) {
            int value = f.feature(name);
            if (value < 0 || value > 3) errors.add(name + " out of range: " + value);
        }
        if (!COMPLEXITY_BY_CLASS.containsKey(f.algorithmClass())) errors.add("unknown class: " + f.algorithmClass());
        return errors;
    }

    static Rubric rubric(Features f) {
        List<String> errors = validate(f);
        if (!errors.isEmpty()) throw new IllegalArgumentException("invalid features: " + errors);
        List<String> order = new ArrayList<>(WEIGHTS.keySet());
        double score = weightedScore(f, order);
        int difficulty = difficultyFor(score);
        List<String> contributions = new ArrayList<>(WEIGHTS.keySet());
        contributions.sort((a, b) -> {
            double ca = WEIGHTS.get(a) * f.feature(a), cb = WEIGHTS.get(b) * f.feature(b);
            if (ca != cb) return Double.compare(cb, ca);
            return a.compareTo(b);                       // deterministic tie-break
        });
        List<String> rationale = new ArrayList<>();
        for (int i = 0; i < 2; i++) rationale.add(contributions.get(i) + "=" + f.feature(contributions.get(i)));
        long h = fnvText(FNV_OFFSET, "rubric");
        for (String name : order) h = fnvText(h, name + ":" + f.feature(name));
        h = fnvText(h, f.algorithmClass() + ":" + score + ":" + difficulty);
        return new Rubric(difficulty, COMPLEXITY_BY_CLASS.get(f.algorithmClass()), score,
                10 * difficulty + 3 * (f.edgeCases() + f.proofObligations()), rationale, hex(h));
    }

    static double weightSum() {
        double sum = 0;
        for (double w : WEIGHTS.values()) sum += w;
        return sum;
    }

    public static void main(String[] args) {
        Features sample = new Features(2, 1, 2, 1, 2, "linearithmic");
        Rubric sampleRubric = rubric(sample);

        // 1. the sample rubric is internally consistent
        check(Math.abs(weightSum() - 1.0) < 1e-9 && sampleRubric.difficulty() >= 1 && sampleRubric.difficulty() <= 5
                        && sampleRubric.expectedComplexity().equals("O(n log n)") && sampleRubric.estimatedMinutes() > 0
                        && sampleRubric.rationale().size() == 2 && sampleRubric.rationale().get(0).startsWith("inputScale"),
                "weights sum to one and the sample rubric is complete and consistent");

        // 2. monotonicity: raising one feature never lowers the score or the level
        boolean monotone = true;
        List<String> names = new ArrayList<>(WEIGHTS.keySet());
        for (int i = 0; i < 64; i++) {
            long state = splitmix64(900 + i);
            int[] values = new int[5];
            for (int j = 0; j < 5; j++) { state = splitmix64(state + j); values[j] = (int) Math.floorMod(state, 4); }
            Features base = new Features(values[0], values[1], values[2], values[3], values[4], "linear");
            Rubric baseRubric = rubric(base);
            for (int j = 0; j < 5; j++) {
                if (values[j] >= 3) continue;
                int[] raised = values.clone();
                raised[j]++;
                Features higher = new Features(raised[0], raised[1], raised[2], raised[3], raised[4], "linear");
                Rubric higherRubric = rubric(higher);
                monotone &= higherRubric.score() >= baseRubric.score()
                        && higherRubric.difficulty() >= baseRubric.difficulty();
            }
            if (!monotone) break;
        }
        check(monotone, "score and difficulty are monotone in every feature over 64 sampled profiles");

        // 3. every algorithm class maps to an expected complexity
        boolean classesOk = true;
        for (Map.Entry<String, String> entry : COMPLEXITY_BY_CLASS.entrySet()) {
            Rubric r = rubric(new Features(1, 1, 1, 1, 1, entry.getKey()));
            classesOk &= entry.getValue().equals(r.expectedComplexity());
        }
        check(classesOk && COMPLEXITY_BY_CLASS.size() == 6, "all six algorithm classes resolve to a complexity");

        // 4. boundaries: zero profile is level 1, maximal profile is level 5
        Rubric zero = rubric(new Features(0, 0, 0, 0, 0, "constant"));
        Rubric maximal = rubric(new Features(3, 3, 3, 3, 3, "exponential"));
        check(zero.score() == 0.0 && zero.difficulty() == 1 && zero.estimatedMinutes() == 10
                        && maximal.score() == 100.0 && maximal.difficulty() == 5 && maximal.estimatedMinutes() == 68,
                "zero profile scores 0 at level 1, maximal profile scores 100 at level 5");

        // 5. scoring is order independent and reproducible
        List<String> reversed = new ArrayList<>(names);
        Collections.reverse(reversed);
        check(weightedScore(sample, names) == weightedScore(sample, reversed)
                        && rubric(sample).digest().equals(sampleRubric.digest())
                        && rubric(sample).equals(sampleRubric),
                "scoring ignores iteration order and repeated generation is identical");

        // 6. invalid profiles are rejected
        int rejected = 0;
        for (Features bad : List.of(
                new Features(4, 0, 0, 0, 0, "linear"),
                new Features(0, -1, 0, 0, 0, "linear"),
                new Features(1, 1, 1, 1, 1, "quantum")))
            try { rubric(bad); } catch (IllegalArgumentException e) { rejected++; }
        check(rejected == 3 && validate(sample).isEmpty(), "out-of-range features and unknown classes are rejected");

        // 7. level score ranges never overlap and all five levels are reachable
        Map<Integer, double[]> ranges = new TreeMap<>();
        for (int i = 0; i <= 300; i++) {
            Rubric r = rubric(new Features(i % 4, (i / 4) % 4, (i / 16) % 4, (i / 64) % 4, (i / 256) % 4, "linear"));
            double[] range = ranges.computeIfAbsent(r.difficulty(), k -> new double[] { 101, -1 });
            range[0] = Math.min(range[0], r.score());
            range[1] = Math.max(range[1], r.score());
        }
        boolean disjoint = true;
        double previousMax = -1;
        for (double[] range : ranges.values()) { disjoint &= range[0] > previousMax; previousMax = range[1]; }
        check(disjoint && ranges.size() == 5, "the five level score ranges are disjoint and all reachable");

        // 8. independent oracle: dot product and threshold ladder recomputed from scratch
        boolean oracleOk = true;
        for (int i = 0; i < 40; i++) {
            long state = splitmix64(5000 + i);
            int a = (int) Math.floorMod(state, 4), b = (int) Math.floorMod(state >>> 8, 4);
            int c = (int) Math.floorMod(state >>> 16, 4), d = (int) Math.floorMod(state >>> 24, 4);
            int e = (int) Math.floorMod(state >>> 32, 4);
            Features f = new Features(a, b, c, d, e, "quadratic");
            double oracle = 0.25 * a / 3.0 + 0.20 * b / 3.0 + 0.20 * c / 3.0 + 0.20 * d / 3.0 + 0.15 * e / 3.0;
            oracle = Math.round(oracle * 100 * 10) / 10.0;
            Rubric r = rubric(f);
            oracleOk &= Math.abs(r.score() - oracle) < 1e-9 && r.difficulty() == oracleDifficulty(oracle)
                    && oracleDifficulty(19.9) == 1 && oracleDifficulty(20.0) == 2 && oracleDifficulty(80.0) == 5;
        }
        check(oracleOk, "an independently written dot product and threshold ladder reproduce every rubric");

        System.out.println("features per task: " + WEIGHTS.size() + ", classes: " + COMPLEXITY_BY_CLASS.size());
        System.out.println("sample: score=" + sampleRubric.score() + " difficulty=" + sampleRubric.difficulty()
                + " " + sampleRubric.expectedComplexity() + " minutes=" + sampleRubric.estimatedMinutes()
                + " rationale=" + sampleRubric.rationale());
        System.out.println("level score ranges: " + Arrays.deepToString(ranges.values().toArray()));
        System.out.println("rubric digest: " + sampleRubric.digest());
        System.out.println("PASS");
    }
}
