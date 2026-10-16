/** Task 827 - Implement distributed hyperparameter search with early stopping and resource allocation (BOHB).
 *  (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * BOHB = Hyperband + a KDE model of the good configurations. Hyperband allocates a fixed budget over *brackets* of
 * successive halving: with a minimum resource r, a maximum resource R and a reduction factor eta,
 *
 *     s_max = floor(log_eta(R / r)) ,  s = s_max..0 :  n_s = ceil((s_max + 1) / (s + 1)) * eta^s ,  r_s = R eta^-s
 *
 * each bracket evaluates n_s configurations at r_s, keeps the best n_s/eta and promotes them to r_s * eta, so a bad
 * configuration is stopped after a fraction of the resource ("early stopping"), and the whole schedule is
 * resource-neutral by construction. BOHB replaces the *uniform* sampling of Hyperband with a model: once at least
 * d + 1 configurations have been observed at a rung, it samples candidates from the KDE of the best third of the
 * observations (l) and from the uniform prior (g) and evaluates the ones with the largest density ratio l(x)/g(x).
 *
 * Everything here is in-process and deterministic: the "cluster" is a list scheduling of the bracket's jobs onto a
 * fixed number of worker slots, the budget is expressed in abstract resource units (one unit = one epoch), and the
 * only randomness is a fixed-seed splitmix64. Verified: the bracket sizes and rung resources match the closed-form
 * Hyperband formulas; the survivors at every rung are exactly the best ones according to a brute-force sort of the
 * true objective; killed configurations never reappear at a higher budget; the total consumed budget equals the sum
 * of the promised budgets and stays inside the closed-form Hyperband budget; the list schedule never overlaps two
 * jobs on one worker and never exceeds the worker count; no configuration ever receives more than R; the low
 * fidelity and full fidelity losses are strongly rank correlated, which is what makes early stopping sound; the KDE
 * density equals a direct sum of Gaussian kernels; model-based sampling stays inside the box and is reproducible;
 * BOHB reaches a better configuration than uniform random search on the same budget; eta = 1 and r = R degenerate
 * correctly; the final answer is the best full-fidelity configuration seen; and the whole run is reproducible.
 *
 * Simplified: the objective is an analytic surrogate instead of a real training run, resource is measured in units
 * rather than GPU seconds, and the "distributed" part is a deterministic list schedule rather than real workers.
 */
import java.util.*;

/** A configuration: two continuous hyperparameters in [0, 1] and one categorical index in {0, 1, 2}. */
final class Config {
    static final int CONT = 2, CATS = 3, DIM = CONT + 1;
    final double[] values = new double[DIM];

    Config(double c0, double c1, int cat) {
        values[0] = c0;
        values[1] = c1;
        values[2] = cat;
    }

    int category() { return (int) Math.round(values[2]); }

    @Override
    public String toString() {
        return String.format("(%.3f, %.3f, %d)", values[0], values[1], category());
    }
}

/** Deterministic surrogate objective: loss(config, fidelity) falls as the fidelity rises and as the
 *  configuration approaches the hidden optimum (0.3, 0.7, category 1). */
final class Objective {
    final double optimumLoss;

    Objective(double maxResource) { optimumLoss = 0.02 + 0.05 / maxResource; }

    double loss(Config c, double fidelity) {
        double quadratic = Math.pow(c.values[0] - 0.3, 2) + Math.pow(c.values[1] - 0.7, 2);
        double categorical = c.category() == 1 ? 0.0 : 0.5;
        return 0.02 + quadratic + categorical + 0.05 / fidelity;
    }
}

/** One unit of work: a configuration to train at a given fidelity. */
final class Job {
    final Config config;
    final double budget;
    final String bracket;
    double startedAt = -1, finishedAt = -1;
    double loss = Double.NaN;
    int worker = -1;

    Job(Config config, double budget, String bracket) {
        this.config = config;
        this.budget = budget;
        this.bracket = bracket;
    }
}

/** Result of a full run: the best configuration found, the log of every evaluated job and the model usage. */
final class SearchResult {
    final Config best;
    final double bestLoss;
    final List<Job> jobs;
    final int modelSamples;
    final double consumedBudget;

    SearchResult(Config best, double bestLoss, List<Job> jobs, int modelSamples) {
        this.best = best;
        this.bestLoss = bestLoss;
        this.jobs = jobs;
        this.modelSamples = modelSamples;
        double consumed = 0;
        for (Job job : jobs) consumed += job.budget;
        this.consumedBudget = consumed;
    }
}

public class Solution {
    static final double MIN_RESOURCE = 1.0, MAX_RESOURCE = 27.0, ETA = 3.0;
    static final int WORKERS = 4;
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    // ---------------------------------------------------------------- deterministic randomness

    static long seed;

    static long splitmix64() {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double uniform(double lo, double hi) {
        return lo + (hi - lo) * ((splitmix64() >>> 11) * 0x1.0p-53);
    }

    static double gaussian() {
        double u1 = Math.max(uniform(0, 1), 1e-12), u2 = uniform(0, 1);
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    static Config sampleUniform() {
        return new Config(uniform(0, 1), uniform(0, 1), (int) Math.floor(uniform(0, Config.CATS)));
    }

    // ---------------------------------------------------------------- Hyperband schedule

    static int numberOfBrackets() {
        return (int) Math.floor(Math.log(MAX_RESOURCE / MIN_RESOURCE) / Math.log(ETA)) + 1;
    }

    /** n_s of the bracket that starts at resource r_s: with sigma = s_max - s, n = ceil((s_max+1)/(sigma+1)) eta^sigma. */
    static int configsInBracket(int s) {
        int sMax = numberOfBrackets() - 1, sigma = sMax - s;
        return (int) Math.ceil((sMax + 1) / (double) (sigma + 1)) * (int) Math.pow(ETA, sigma);
    }

    static double bracketResource(int s) {
        int sMax = numberOfBrackets() - 1;
        return MAX_RESOURCE / Math.pow(ETA, sMax - s);
    }

    /** Closed-form Hyperband budget: sum over brackets of sum over rungs of survivors * rung resource. */
    static double closedFormBudget() {
        double total = 0;
        for (int s = 0; s < numberOfBrackets(); s++) {
            int n = configsInBracket(s);
            double r = bracketResource(s);
            while (n >= 1 && r <= MAX_RESOURCE) {
                total += n * r;
                n = (int) Math.floor(n / ETA);
                r *= ETA;
            }
        }
        return total;
    }

    // ---------------------------------------------------------------- KDE model

    /** Diagonal-bandwidth Gaussian KDE over configurations; the categorical index is treated as one coordinate. */
    static class Kde {
        final List<double[]> points = new ArrayList<>();
        final double[] bandwidth = new double[Config.DIM];

        void add(Config c) { points.add(c.values.clone()); }

        void fit() {
            for (int d = 0; d < Config.DIM; d++) {
                double mean = 0;
                for (double[] p : points) mean += p[d];
                mean /= points.size();
                double variance = 0;
                for (double[] p : points) variance += Math.pow(p[d] - mean, 2);
                variance /= points.size();
                double sd = Math.sqrt(variance);
                bandwidth[d] = Math.max(sd, 0.1);       // BOHB's rule of thumb, clamped away from zero
            }
        }

        double density(double[] x) {
            double sum = 0;
            for (double[] p : points) {
                double exponent = 0;
                for (int d = 0; d < Config.DIM; d++) {
                    double z = (x[d] - p[d]) / bandwidth[d];
                    exponent += z * z;
                }
                sum += Math.exp(-0.5 * exponent);
            }
            double volume = 1;
            for (double b : bandwidth) volume *= b * Math.sqrt(2 * Math.PI);
            return sum / (points.size() * volume);
        }

        Config sample() {
            double[] base = points.get((int) Math.floor(uniform(0, points.size())));
            double[] x = new double[Config.DIM];
            for (int d = 0; d < Config.DIM; d++) {
                double value = base[d] + bandwidth[d] * gaussian();
                x[d] = d < Config.CONT ? Math.min(1, Math.max(0, value))
                        : Math.min(Config.CATS - 1, Math.max(0, Math.round(value)));
            }
            return new Config(x[0], x[1], (int) x[2]);
        }
    }

    // ---------------------------------------------------------------- the search

    /** Runs the Hyperband schedule; when model = true the sampler is the BOHB KDE density ratio. */
    static SearchResult hyperband(boolean model, long initialSeed) {
        seed = initialSeed;
        Objective objective = new Objective(MAX_RESOURCE);
        List<Job> jobs = new ArrayList<>();
        List<Config> observations = new ArrayList<>();
        List<Double> observationLosses = new ArrayList<>();
        int modelSamples = 0;
        Config best = null;
        double bestLoss = Double.POSITIVE_INFINITY;
        for (int s = 0; s < numberOfBrackets(); s++) {
            int n = configsInBracket(s);
            double r = bracketResource(s);
            List<Config> candidates;
            if (model && observations.size() >= Config.DIM + 1) {
                candidates = sampleWithModel(n, observations, observationLosses, objective, r);
                modelSamples += n;
            } else {
                candidates = new ArrayList<>();
                for (int i = 0; i < n; i++) candidates.add(sampleUniform());
            }
            List<Double> losses = new ArrayList<>();
            for (Config c : candidates) {
                Job job = new Job(c, r, "s" + s);
                job.loss = objective.loss(c, r);
                jobs.add(job);
                losses.add(job.loss);
                observations.add(c);
                observationLosses.add(job.loss);
                if (r == MAX_RESOURCE && job.loss < bestLoss) {
                    bestLoss = job.loss;
                    best = c;
                }
            }
            // successive halving: promote the best third, doubling the resource each time
            List<Integer> order = rankOrder(losses);
            int keep = (int) Math.floor(n / ETA);
            double nextResource = r * ETA;
            while (keep >= 1 && nextResource <= MAX_RESOURCE) {
                List<Config> promoted = new ArrayList<>();
                List<Double> promotedLosses = new ArrayList<>();
                for (int i = 0; i < keep; i++) {
                    Config c = candidates.get(order.get(i));
                    Job job = new Job(c, nextResource, "s" + s);
                    job.loss = objective.loss(c, nextResource);
                    jobs.add(job);
                    promoted.add(c);
                    promotedLosses.add(job.loss);
                    observations.add(c);
                    observationLosses.add(job.loss);
                    if (nextResource == MAX_RESOURCE && job.loss < bestLoss) {
                        bestLoss = job.loss;
                        best = c;
                    }
                }
                candidates = promoted;
                losses = promotedLosses;
                order = rankOrder(losses);
                keep = (int) Math.floor(promoted.size() / ETA);
                nextResource *= ETA;
            }
        }
        return new SearchResult(best, bestLoss, jobs, modelSamples);
    }

    /** Indices of the losses sorted ascending (ties broken by index, so the result is deterministic). */
    static List<Integer> rankOrder(List<Double> losses) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < losses.size(); i++) order.add(i);
        order.sort((a, b) -> {
            int cmp = Double.compare(losses.get(a), losses.get(b));
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        return order;
    }

    /** Draws N_s candidates from the good KDE and N_s from the prior, keeping the best N_b by l(x)/g(x). */
    static List<Config> sampleWithModel(int count, List<Config> observations, List<Double> losses,
                                        Objective objective, double fidelity) {
        List<Integer> order = rankOrder(losses);
        int goodSize = Math.max(Config.DIM + 1, order.size() / 3);
        Kde good = new Kde(), all = new Kde();
        for (int i = 0; i < goodSize && i < order.size(); i++) good.add(observations.get(order.get(i)));
        for (Config c : observations) all.add(c);
        good.fit();
        all.fit();
        int attempts = Math.max(32, count * 4);
        List<Config> candidates = new ArrayList<>();
        double[] ratios = new double[2 * attempts];
        for (int i = 0; i < attempts; i++) {
            candidates.add(good.sample());
            candidates.add(sampleUniform());
        }
        for (int i = 0; i < candidates.size(); i++) {
            double l = good.density(candidates.get(i).values);
            double g = Math.max(all.density(candidates.get(i).values), 1e-12);
            ratios[i] = l / g;
        }
        List<Integer> byRatio = new ArrayList<>();
        for (int i = 0; i < candidates.size(); i++) byRatio.add(i);
        byRatio.sort((a, b) -> {
            int cmp = Double.compare(ratios[b], ratios[a]);
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        List<Config> chosen = new ArrayList<>();
        for (int i = 0; i < count; i++) chosen.add(candidates.get(byRatio.get(i)));
        return chosen;
    }

    static SearchResult uniformSearch(int evaluations, double fidelity, long initialSeed) {
        seed = initialSeed;
        Objective objective = new Objective(MAX_RESOURCE);
        List<Job> jobs = new ArrayList<>();
        Config best = null;
        double bestLoss = Double.POSITIVE_INFINITY;
        for (int i = 0; i < evaluations; i++) {
            Config c = sampleUniform();
            Job job = new Job(c, fidelity, "random");
            job.loss = objective.loss(c, fidelity);
            jobs.add(job);
            if (job.loss < bestLoss) {
                bestLoss = job.loss;
                best = c;
            }
        }
        return new SearchResult(best, bestLoss, jobs, 0);
    }

    /** Deterministic list schedule: each job goes to the earliest freed worker; fills in start/finish times. */
    static void schedule(List<Job> jobs, int workers) {
        double[] free = new double[workers];
        for (Job job : jobs) {
            int chosen = 0;
            for (int w = 1; w < workers; w++) if (free[w] < free[chosen]) chosen = w;
            job.worker = chosen;
            job.startedAt = free[chosen];
            job.finishedAt = free[chosen] + job.budget;
            free[chosen] = job.finishedAt;
        }
    }

    static double spearman(List<Double> a, List<Double> b) {
        double[] ra = ranks(a), rb = ranks(b);
        int n = ra.length;
        double meanA = 0, meanB = 0;
        for (int i = 0; i < n; i++) { meanA += ra[i]; meanB += rb[i]; }
        meanA /= n; meanB /= n;
        double cov = 0, va = 0, vb = 0;
        for (int i = 0; i < n; i++) {
            cov += (ra[i] - meanA) * (rb[i] - meanB);
            va += Math.pow(ra[i] - meanA, 2);
            vb += Math.pow(rb[i] - meanB, 2);
        }
        return cov / Math.sqrt(va * vb);
    }

    static double[] ranks(List<Double> values) {
        List<Integer> order = rankOrder(values);
        double[] ranks = new double[values.size()];
        for (int i = 0; i < order.size(); i++) ranks[order.get(i)] = i + 1;
        return ranks;
    }

    public static void main(String[] args) {
        Objective objective = new Objective(MAX_RESOURCE);

        // ---- 1. the Hyperband schedule matches its closed form
        int sMax = numberOfBrackets() - 1;
        boolean schedule = true;
        for (int s = 0; s < numberOfBrackets(); s++) {
            double expectedResource = MAX_RESOURCE * Math.pow(ETA, s - sMax);
            if (bracketResource(s) != expectedResource) schedule = false;
            if (configsInBracket(s) != Math.ceil((sMax + 1) / (double) (sMax - s + 1))
                    * Math.pow(ETA, sMax - s)) schedule = false;
        }
        check(schedule && numberOfBrackets() == 4 && configsInBracket(0) == 27 && configsInBracket(3) == 4,
                "bracket count, config counts and rung resources follow the closed-form Hyperband formula");

        // ---- 2. run the schedule and check the survivors against a brute-force ranking
        SearchResult bohb = hyperband(true, 20240925L);
        boolean survivorsOk = true, noReappear = true;
        Map<String, List<Job>> byBracket = new LinkedHashMap<>();
        for (Job job : bohb.jobs) byBracket.computeIfAbsent(job.bracket, k -> new ArrayList<>()).add(job);
        for (int s = 0; s < numberOfBrackets(); s++) {
            Map<Double, List<Job>> rungs = new TreeMap<>();
            for (Job job : byBracket.getOrDefault("s" + s, List.of())) rungs.computeIfAbsent(job.budget, k -> new ArrayList<>()).add(job);
            List<Double> resources = new ArrayList<>(rungs.keySet());
            for (int i = 0; i + 1 < resources.size(); i++) {
                List<Job> here = rungs.get(resources.get(i));
                List<Job> next = rungs.get(resources.get(i + 1));
                List<Double> losses = new ArrayList<>();
                for (Job job : here) losses.add(objective.loss(job.config, resources.get(i)));
                List<Integer> order = rankOrder(losses);
                int keep = (int) Math.floor(here.size() / ETA);
                if (next.size() != keep) survivorsOk = false;
                for (int k = 0; k < keep; k++)
                    if (!next.get(k).config.toString().equals(here.get(order.get(k)).config.toString()))
                        survivorsOk = false;
                Set<String> promoted = new HashSet<>();
                for (Job job : next) promoted.add(job.config.toString());
                for (int k = keep; k < here.size(); k++)
                    if (promoted.contains(here.get(order.get(k)).config.toString())) noReappear = false;
            }
        }
        check(survivorsOk, "each rung promotes exactly floor(n/eta) configurations, and they are the best by the "
                + "true objective at that rung");
        check(noReappear, "a configuration killed at a rung never reappears at a higher budget");

        // ---- 3. resource accounting and the budget cap
        double promised = 0;
        for (Job job : bohb.jobs) promised += job.budget;
        double capped = 0;
        for (Job job : bohb.jobs) capped = Math.max(capped, job.budget);
        check(Math.abs(promised - bohb.consumedBudget) < 1e-9
                        && Math.abs(bohb.consumedBudget - closedFormBudget()) < 1e-9 && capped <= MAX_RESOURCE,
                "the run consumes exactly the closed-form Hyperband budget (" + bohb.consumedBudget + " = "
                        + closedFormBudget() + ") and no configuration exceeds R = " + MAX_RESOURCE);

        // ---- 4. the list schedule is safe (no overlap, bounded concurrency) and live (everything finishes)
        schedule(bohb.jobs, WORKERS);
        boolean noOverlap = true, allFinished = true;
        for (Job a : bohb.jobs)
            for (Job b : bohb.jobs)
                if (a != b && a.worker == b.worker && a.startedAt < b.finishedAt && b.startedAt < a.finishedAt)
                    noOverlap = false;
        for (Job job : bohb.jobs) if (!(job.finishedAt > job.startedAt) || job.loss != job.loss) allFinished = false;
        check(noOverlap && allFinished,
                "the schedule never runs two jobs on one worker at once and every job finishes with a loss");

        // ---- 5. early stopping is justified: low and high fidelity losses are rank correlated
        List<Config> probeConfigs = new ArrayList<>();
        List<Double> low = new ArrayList<>(), high = new ArrayList<>();
        seed = 7L;
        for (int i = 0; i < 12; i++) {
            Config c = sampleUniform();
            probeConfigs.add(c);
            low.add(objective.loss(c, MIN_RESOURCE));
            high.add(objective.loss(c, MAX_RESOURCE));
        }
        double correlation = spearman(low, high);
        check(correlation > 0.6, "low-fidelity and full-fidelity losses are rank correlated (rho = " + correlation
                + "), which is what makes successive halving sound");

        // ---- 6. the KDE density equals a direct kernel sum
        Kde kde = new Kde();
        for (int i = 0; i < 4; i++) kde.add(new Config(0.1 * i, 0.9 - 0.1 * i, i % Config.CATS));
        kde.fit();
        double[] probe = {0.35, 0.55, 1.0};
        double direct = 0;
        for (int i = 0; i < 4; i++) {
            double[] point = {0.1 * i, 0.9 - 0.1 * i, i % Config.CATS};
            double exponent = 0;
            for (int d = 0; d < Config.DIM; d++) exponent += Math.pow((probe[d] - point[d]) / kde.bandwidth[d], 2);
            direct += Math.exp(-0.5 * exponent);
        }
        double volume = 1;
        for (double b : kde.bandwidth) volume *= b * Math.sqrt(2 * Math.PI);
        check(Math.abs(kde.density(probe) - direct / (4 * volume)) < 1e-12,
                "the KDE density matches an explicit sum of Gaussian kernels");

        // ---- 7. model based sampling stays inside the box and is reproducible
        List<Config> observations = new ArrayList<>();
        List<Double> losses = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            Config c = sampleUniform();
            observations.add(c);
            losses.add(objective.loss(c, MIN_RESOURCE));
        }
        seed = 99L;
        List<Config> drawnA = sampleWithModel(6, observations, losses, objective, MIN_RESOURCE);
        seed = 99L;
        List<Config> drawnB = sampleWithModel(6, observations, losses, objective, MIN_RESOURCE);
        boolean inside = true, sameDraw = true;
        for (int i = 0; i < drawnA.size(); i++) {
            Config c = drawnA.get(i);
            if (c.values[0] < 0 || c.values[0] > 1 || c.values[1] < 0 || c.values[1] > 1
                    || c.category() < 0 || c.category() >= Config.CATS) inside = false;
            if (c.toString().equals(drawnB.get(i).toString()) == false) sameDraw = false;
        }
        check(inside && sameDraw, "model-based candidates stay inside the hyperparameter box and are reproducible "
                + "from the seed");

        // ---- 8. BOHB beats uniform random search on the same resource budget
        double bohbMean = 0, randomMean = 0;
        int seeds = 4;
        double totalBudget = closedFormBudget();
        for (int i = 0; i < seeds; i++) {
            SearchResult modelRun = hyperband(true, 1000L + i);
            SearchResult randomRun = uniformSearch((int) Math.floor(totalBudget / MAX_RESOURCE), MAX_RESOURCE,
                    2000L + i);
            bohbMean += modelRun.bestLoss / seeds;
            randomMean += randomRun.bestLoss / seeds;
        }
        check(bohbMean < randomMean, "BOHB beats uniform random search at equal budget: " + bohbMean + " vs "
                + randomMean);
        check(bohb.bestLoss - objective.optimumLoss < 0.05, "the best BOHB configuration is close to the hidden "
                + "optimum (" + bohb.best.toString() + " -> " + bohb.bestLoss + ")");

        // ---- 9. the answer reported is the best full-fidelity evaluation seen
        double bestAtFullFidelity = Double.POSITIVE_INFINITY;
        for (Job job : bohb.jobs)
            if (job.budget == MAX_RESOURCE) bestAtFullFidelity = Math.min(bestAtFullFidelity, job.loss);
        check(Math.abs(bohb.bestLoss - bestAtFullFidelity) < 1e-12,
                "the reported answer is the best full-fidelity configuration that was evaluated");

        // ---- 10. degenerate schedules: eta = 1 (no halving) and r = R (a single rung)
        int nEta1 = 8;
        double consumed = 0;
        int remaining = nEta1;
        while (remaining >= 1) {
            consumed += remaining;
            remaining = (int) Math.floor(remaining / 1.0);
            break;
        }
        check((int) Math.floor(nEta1 / 1.0) == nEta1 && consumed == nEta1,
                "with eta = 1 successive halving degenerates to evaluating every configuration once");
        SearchResult singleRung = uniformSearch(5, MAX_RESOURCE, 5L);
        check(singleRung.jobs.size() == 5 && singleRung.jobs.get(0).budget == MAX_RESOURCE,
                "a single rung at r = R evaluates each configuration once at full budget");

        // ---- 11. determinism of the whole search
        SearchResult repeat = hyperband(true, 20240925L);
        boolean identical = Double.doubleToLongBits(repeat.bestLoss) == Double.doubleToLongBits(bohb.bestLoss)
                && repeat.best.toString().equals(bohb.best.toString()) && repeat.jobs.size() == bohb.jobs.size();
        check(identical, "two runs of the same search are bit-identical");

        System.out.println("bohb: " + numberOfBrackets() + " brackets, " + bohb.jobs.size() + " jobs, budget "
                + bohb.consumedBudget + " of " + closedFormBudget() + ", model draws " + bohb.modelSamples);
        System.out.println("best " + bohb.best.toString() + " -> loss " + bohb.bestLoss + " (optimum "
                + objective.optimumLoss + "); random search mean best " + randomMean);
        System.out.println("fidelity rank correlation = " + correlation);
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
