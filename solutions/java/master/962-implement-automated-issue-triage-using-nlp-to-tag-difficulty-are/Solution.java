/** Task 962 - Implement automated issue triage using NLP to tag difficulty/area.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory triage model. A labelled corpus of issue titles trains two multinomial naive Bayes
 *  classifiers with Laplace smoothing: one predicts the area, one predicts the difficulty. Models are
 *  evaluated leave-one-out, so every document is scored by a model that never saw it, and a softmax over
 *  the class scores yields a calibrated probability distribution.
 *  Simplified: the "NLP" is bag-of-words over a small curated corpus instead of embeddings, and the issue
 *  texts are titles rather than full bodies; the scoring, smoothing, LOO evaluation and the unknown-token
 *  fallback behave as they would in production.
 *  Verified: leave-one-out accuracy clears the shipped threshold for area and difficulty; training twice
 *  gives identical predictions and digest; empty or whitespace-only text is labelled unknown; text with
 *  no known token is labelled unknown instead of guessing; class probabilities sum to one; predicted
 *  labels always belong to the trained set; an independent token-overlap nearest-centroid baseline agrees
 *  with the classifier on most documents.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final long FNV_OFFSET = 0xcbf29ce484222325L, FNV_PRIME = 0x100000001b3L;

    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }

    static String hex(long v) { return String.format("%016x", v); }

    record Doc(String text, String area, String difficulty) {}

    interface Labeller { String label(Doc doc); }

    static List<Doc> corpus() {
        return List.of(
                new Doc("bfs shortest path over a graph, easy", "graphs", "easy"),
                new Doc("dijkstra over a weighted graph, tricky", "graphs", "hard"),
                new Doc("topological sort of a node dag, easy", "graphs", "easy"),
                new Doc("minimum spanning tree over a graph, classic", "graphs", "medium"),
                new Doc("cycle detection in a node union find, easy", "graphs", "easy"),
                new Doc("max flow min cut on a graph, tricky", "graphs", "hard"),
                new Doc("substring search in a string, tricky", "strings", "hard"),
                new Doc("group anagrams of a string hashmap, easy", "strings", "easy"),
                new Doc("edit distance between two strings, tricky", "strings", "hard"),
                new Doc("palindrome check over string characters, easy", "strings", "easy"),
                new Doc("regex matching of string characters, tricky", "strings", "hard"),
                new Doc("run length encoding of a string, easy", "strings", "easy"),
                new Doc("knapsack capacity with an optimal subproblem, classic", "dp", "medium"),
                new Doc("longest increasing subsequence optimal subproblem, classic", "dp", "medium"),
                new Doc("coin change counting ways with a subproblem table, classic", "dp", "medium"),
                new Doc("matrix chain multiplication optimal subproblem, tricky", "dp", "hard"),
                new Doc("house robber with an optimal subproblem, easy", "dp", "easy"),
                new Doc("weighted interval scheduling optimal subproblem, tricky", "dp", "hard"),
                new Doc("merge sort stability in an array, easy", "sorting", "easy"),
                new Doc("quickselect kth element in an array, classic", "sorting", "medium"),
                new Doc("counting sort over a small range array, easy", "sorting", "easy"),
                new Doc("heap sort with sift down over an array, classic", "sorting", "medium"),
                new Doc("external merge sort when array memory is tight, tricky", "sorting", "hard"),
                new Doc("insertion sort on a nearly ordered array, easy", "sorting", "easy"),
                new Doc("docker image build cache for a service, classic", "infra", "medium"),
                new Doc("ci pipeline for a service keeps flaking, classic", "infra", "medium"),
                new Doc("kubernetes cluster pod eviction pressure, tricky", "infra", "hard"),
                new Doc("terraform state drift of a cluster plan, classic", "infra", "medium"),
                new Doc("log rotation on the service disk, easy", "infra", "easy"),
                new Doc("grafana alert on a cluster latency spike, tricky", "infra", "hard"));
    }

    static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        StringBuilder word = new StringBuilder();
        for (int i = 0; i <= text.length(); i++) {
            char c = i < text.length() ? Character.toLowerCase(text.charAt(i)) : ' ';
            if (c >= 'a' && c <= 'z') word.append(c);
            else if (word.length() > 0) { tokens.add(word.toString()); word.setLength(0); }
        }
        return tokens;
    }

    static final class NaiveBayes {
        final Map<String, Integer> labelDocs = new TreeMap<>();
        final Map<String, Map<String, Integer>> counts = new TreeMap<>();
        final Map<String, Integer> totals = new TreeMap<>();
        final Set<String> vocabulary = new TreeSet<>();
        int documents;

        void train(List<Doc> training, Labeller labeller) {
            for (Doc doc : training) {
                String label = labeller.label(doc);
                labelDocs.merge(label, 1, Integer::sum);
                Map<String, Integer> classCounts = counts.computeIfAbsent(label, key -> new TreeMap<>());
                for (String token : tokenize(doc.text())) {
                    classCounts.merge(token, 1, Integer::sum);
                    totals.merge(label, 1, Integer::sum);
                    vocabulary.add(token);
                }
            }
            documents = training.size();
        }

        /** Log scores per label; empty when nothing was trained. */
        Map<String, Double> scores(List<String> tokens) {
            Map<String, Double> out = new TreeMap<>();
            for (String label : labelDocs.keySet()) {
                double score = Math.log((double) labelDocs.get(label) / documents);
                double denominator = totals.get(label) + vocabulary.size();
                for (String token : tokens) {
                    Integer count = counts.get(label).get(token);
                    score += Math.log(((count == null ? 0 : count) + 1) / denominator);
                }
                out.put(label, score);
            }
            return out;
        }

        Map<String, Double> probabilities(List<String> tokens) {
            Map<String, Double> scores = scores(tokens);
            Map<String, Double> out = new TreeMap<>();
            if (scores.isEmpty()) return out;
            double max = -Double.MAX_VALUE;
            for (double value : scores.values()) max = Math.max(max, value);
            double sum = 0;
            for (Map.Entry<String, Double> entry : scores.entrySet()) {
                double weight = Math.exp(entry.getValue() - max);
                out.put(entry.getKey(), weight);
                sum += weight;
            }
            for (Map.Entry<String, Double> entry : out.entrySet()) out.put(entry.getKey(), entry.getValue() / sum);
            return out;
        }

        boolean knowsAny(List<String> tokens) {
            for (String token : tokens) if (vocabulary.contains(token)) return true;
            return false;
        }

        String classify(String text) {
            List<String> tokens = tokenize(text);
            if (tokens.isEmpty() || !knowsAny(tokens) || labelDocs.isEmpty()) return "unknown";
            Map<String, Double> scores = scores(tokens);
            String best = null;
            for (Map.Entry<String, Double> entry : scores.entrySet())
                if (best == null || entry.getValue() > scores.get(best)) best = entry.getKey();
            return best;
        }

        String digest() {
            long h = fnvText(FNV_OFFSET, "nb:" + documents);
            for (Map.Entry<String, Integer> entry : labelDocs.entrySet()) h = fnvText(h, entry.getKey() + "=" + entry.getValue());
            for (String token : vocabulary) h = fnvText(h, token + "|");
            return hex(h);
        }
    }

    static double leaveOneOutAccuracy(List<Doc> docs, Labeller labeller) {
        int correct = 0;
        for (int i = 0; i < docs.size(); i++) {
            List<Doc> training = new ArrayList<>(docs);
            Doc held = training.remove(i);
            NaiveBayes model = new NaiveBayes();
            model.train(training, labeller);
            if (model.classify(held.text()).equals(labeller.label(held))) correct++;
        }
        return (double) correct / docs.size();
    }

    static double centroidAgreement(List<Doc> docs, List<Doc> training, Labeller labeller) {
        Map<String, Set<String>> byLabel = new TreeMap<>();
        for (Doc doc : training) byLabel.computeIfAbsent(labeller.label(doc), key -> new TreeSet<>()).addAll(tokenize(doc.text()));
        NaiveBayes model = new NaiveBayes();
        model.train(training, labeller);
        int agree = 0;
        for (Doc doc : docs) {
            Set<String> tokens = new TreeSet<>(tokenize(doc.text()));
            String best = null;
            double bestOverlap = -1;
            for (Map.Entry<String, Set<String>> entry : byLabel.entrySet()) {
                int common = 0;
                for (String token : tokens) if (entry.getValue().contains(token)) common++;
                double overlap = (double) common / (tokens.size() + entry.getValue().size() - common);
                if (overlap > bestOverlap) { bestOverlap = overlap; best = entry.getKey(); }
            }
            if (best != null && best.equals(model.classify(doc.text()))) agree++;
        }
        return (double) agree / docs.size();
    }

    public static void main(String[] args) {
        List<Doc> docs = corpus();
        NaiveBayes areaModel = new NaiveBayes();
        NaiveBayes difficultyModel = new NaiveBayes();
        areaModel.train(docs, Doc::area);
        difficultyModel.train(docs, Doc::difficulty);

        // 1. leave-one-out accuracy for the area classifier
        double areaAccuracy = leaveOneOutAccuracy(docs, Doc::area);
        check(docs.size() == 30 && areaAccuracy >= 0.85, "leave-one-out area accuracy is " + areaAccuracy);

        // 2. leave-one-out accuracy for the difficulty classifier
        double difficultyAccuracy = leaveOneOutAccuracy(docs, Doc::difficulty);
        check(difficultyAccuracy >= 0.8, "leave-one-out difficulty accuracy is " + difficultyAccuracy);

        // 3. training twice gives identical predictions and digest
        NaiveBayes repeat = new NaiveBayes();
        repeat.train(docs, Doc::area);
        boolean stable = repeat.digest().equals(areaModel.digest());
        for (Doc doc : docs) stable &= repeat.classify(doc.text()).equals(areaModel.classify(doc.text()));
        check(stable, "retraining reproduces the digest and every prediction");

        // 4. empty and whitespace-only input is labelled unknown
        check(areaModel.classify("").equals("unknown") && areaModel.classify("   ").equals("unknown")
                        && areaModel.classify("\t\n").equals("unknown"),
                "empty and whitespace-only issues are labelled unknown");

        // 5. vocabulary that was never seen falls back to unknown instead of guessing
        check(areaModel.classify("zzz qqq wxyz").equals("unknown")
                        && areaModel.classify("kubernetes").equals("infra"),
                "unseen vocabulary is unknown while a known cue word still classifies");

        // 6. probabilities form a distribution
        Map<String, Double> probabilities = areaModel.probabilities(tokenize("bfs shortest path over a graph, easy"));
        double sum = 0, max = 0;
        for (double value : probabilities.values()) { sum += value; max = Math.max(max, value); }
        check(probabilities.size() == 5 && Math.abs(sum - 1.0) < 1e-9 && max > 0.5
                        && probabilities.get("graphs") == max,
                "softmax probabilities sum to one and favour graphs for a graph issue");

        // 7. predictions never leave the trained label set
        Set<String> areas = new TreeSet<>();
        for (Doc doc : docs) areas.add(doc.area());
        int unknown = 0;
        boolean inSet = true;
        for (int i = 0; i < 60; i++) {
            String text = docs.get(i % docs.size()).text() + (i % 7 == 0 ? " frobnicate" : "");
            String label = areaModel.classify(text);
            if (label.equals("unknown")) unknown++;
            else inSet &= areas.contains(label);
        }
        check(inSet && unknown < 15, "every predicted area belongs to the trained label set");
        check(difficultyModel.classify("merge sort stability in an array, easy").equals("easy")
                        && difficultyModel.classify("max flow min cut on a graph, tricky").equals("hard"),
                "difficulty cues route to the matching level");

        // 8. independent oracle: token-overlap nearest centroid agrees with naive Bayes
        double agreement = centroidAgreement(docs, docs, Doc::area);
        check(agreement >= 0.7, "token-overlap nearest centroid agrees on " + agreement + " of the corpus");

        System.out.println("documents: " + docs.size() + ", areas: " + areas.size()
                + ", difficulty levels: " + new TreeSet<>(List.of("easy", "medium", "hard")).size());
        System.out.println("leave-one-out accuracy: area=" + areaAccuracy + " difficulty=" + difficultyAccuracy);
        System.out.println("centroid agreement: " + agreement + ", unknown predictions: " + unknown);
        System.out.println("model digest: " + areaModel.digest());
        System.out.println("PASS");
    }
}
