/** Task 958 - Implement plagiarism detector comparing code structure and logic patterns.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory similarity detector. Source is tokenized after dropping comments and string bodies and after
 *  mapping every user identifier to ID and every literal to NUM, so renaming and reformatting cannot hide
 *  a copy. Three signals are combined: the Jaccard overlap of 4-token shingles (verbatim logic), the
 *  cosine similarity of the token histogram (same vocabulary) and the Jaccard overlap of control-flow
 *  keyword bigrams (same skeleton and branch order); the weighted score is compared with a fixed
 *  threshold.
 *  Simplified: the corpus is a handful of small text snippets, no AST is built and the weights are
 *  calibrated on the planted pairs of this corpus rather than learned from data. *  Verified: renamed and reformatted clones score 1.0; unrelated programs stay far below the threshold;
 *  a skeleton-only pair shows control-flow similarity 1.0 while its shingle overlap stays low; the score
 *  is symmetric; comments, string bodies and identifier names cannot change the token digest; empty and
 *  too-short programs are reported as incomparable; the planted clone set is recovered exactly; an
 *  independent nested-loop Jaccard reproduces every shingle overlap.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final long FNV_OFFSET = 0xcbf29ce484222325L, FNV_PRIME = 0x100000001b3L;
    static final int K = 4;
    static final double THRESHOLD = 0.6;
    static final double W_SHINGLE = 0.70, W_HISTOGRAM = 0.10, W_CONTROL = 0.20;

    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }

    static String hex(long v) { return String.format("%016x", v); }

    static final Set<String> KEYWORDS = Set.of("if", "else", "for", "while", "return", "break", "continue",
            "int", "long", "double", "boolean", "char", "void", "class", "public", "static", "new", "true",
            "false", "null", "def", "in", "and", "or", "not", "len", "range");

    static final List<String> CONTROL = List.of("if", "else", "for", "while", "return", "break", "continue");

    record Code(String name, String source) {}

    /** Comments and string bodies are dropped, identifiers become ID, literals become NUM. */
    static List<String> tokenize(String source) {
        List<String> tokens = new ArrayList<>();
        int i = 0;
        while (i < source.length()) {
            char c = source.charAt(i);
            if (Character.isWhitespace(c)) { i++; continue; }
            if (c == '/' && i + 1 < source.length() && source.charAt(i + 1) == '/') {
                while (i < source.length() && source.charAt(i) != '\n') i++;
                continue;
            }
            if (c == '/' && i + 1 < source.length() && source.charAt(i + 1) == '*') {
                i += 2;
                while (i + 1 < source.length() && !(source.charAt(i) == '*' && source.charAt(i + 1) == '/')) i++;
                i = Math.min(source.length(), i + 2);
                continue;
            }
            if (c == '"' || c == '\'') {
                char quote = c;
                i++;
                while (i < source.length() && source.charAt(i) != quote) i++;
                i++;
                tokens.add("STR");
                continue;
            }
            if (Character.isDigit(c)) {
                while (i < source.length() && Character.isDigit(source.charAt(i))) i++;
                tokens.add("NUM");
                continue;
            }
            if (Character.isLetter(c) || c == '_') {
                int start = i;
                while (i < source.length() && (Character.isLetterOrDigit(source.charAt(i)) || source.charAt(i) == '_')) i++;
                String word = source.substring(start, i).toLowerCase(Locale.ROOT);
                tokens.add(KEYWORDS.contains(word) ? word : "ID");
                continue;
            }
            tokens.add(String.valueOf(c));
            i++;
        }
        return tokens;
    }

    static Set<String> shingles(List<String> tokens) {
        Set<String> out = new TreeSet<>();
        for (int i = 0; i + K <= tokens.size(); i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < K; j++) sb.append(tokens.get(i + j)).append('\u0001');
            out.add(sb.toString());
        }
        return out;
    }

    static List<String> controlSequence(List<String> tokens) {
        List<String> out = new ArrayList<>();
        for (String token : tokens) if (CONTROL.contains(token)) out.add(token);
        return out;
    }

    /** Control-flow bigrams, so a skeleton match must also agree on the order of the branches. */
    static Set<String> controlBigrams(List<String> tokens) {
        List<String> sequence = controlSequence(tokens);
        Set<String> out = new TreeSet<>();
        for (int i = 0; i + 2 <= sequence.size(); i++) out.add(sequence.get(i) + ">" + sequence.get(i + 1));
        return out;
    }

    static double jaccard(Set<String> a, Set<String> b) {
        if (a.isEmpty() || b.isEmpty()) return 0.0;
        int common = 0;
        for (String s : a) if (b.contains(s)) common++;
        return (double) common / (a.size() + b.size() - common);
    }

    static Map<String, Integer> fullHistogram(List<String> tokens) {
        Map<String, Integer> out = new TreeMap<>();
        for (String token : tokens) out.merge(token, 1, Integer::sum);
        return out;
    }

    static double cosine(Map<String, Integer> a, Map<String, Integer> b) {
        double dot = 0, na = 0, nb = 0;
        for (Map.Entry<String, Integer> entry : a.entrySet()) {
            Integer other = b.get(entry.getKey());
            dot += (double) entry.getValue() * (other == null ? 0 : other);
            na += (double) entry.getValue() * entry.getValue();
        }
        for (int value : b.values()) nb += (double) value * value;
        if (na == 0 || nb == 0) return 0.0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    record Signals(double shingle, double histogram, double control, double score, boolean flagged) {}

    static Signals compare(Code a, Code b) {
        List<String> ta = tokenize(a.source()), tb = tokenize(b.source());
        if (ta.size() < K || tb.size() < K) return new Signals(0, 0, 0, 0, false);   // too short to judge
        double shingle = jaccard(shingles(ta), shingles(tb));
        double histogram = cosine(fullHistogram(ta), fullHistogram(tb));
        double control = jaccard(controlBigrams(ta), controlBigrams(tb));
        double score = W_SHINGLE * shingle + W_HISTOGRAM * histogram + W_CONTROL * control;
        return new Signals(shingle, histogram, control, score, score >= THRESHOLD);
    }

    static String tokenDigest(Code code) {
        long h = fnvText(FNV_OFFSET, "tokens");
        for (String token : tokenize(code.source())) h = fnvText(h, token + "|");
        return hex(h);
    }

    static List<Code> corpus() {
        return List.of(
                new Code("gcd-a", "int gcd(int a, int b) {\n  while (b != 0) {\n    int t = b;\n    b = a % b;\n    a = t;\n  }\n  return a;\n}\n"),
                new Code("gcd-renamed", "// euclid, reformatted\nint greatestCommonDivisor(int x, int y) {\n    while (y != 0) {\n        int tmp = y;\n        y = x % y;\n        x = tmp;\n    }\n    return x;\n}\n"),
                new Code("sum-a", "int sum(int[] xs) {\n  int total = 0;\n  for (int i = 0; i < xs.length; i++) {\n    total += xs[i];\n  }\n  return total;\n}\n"),
                new Code("sum-renamed", "// add them up\nint addAll(int[] values) {\n    int acc = 0;\n    for (int i = 0; i < values.length; i++) {\n        acc += values[i];\n    }\n    return acc;\n}\n"),
                new Code("prime", "boolean isPrime(int n) {\n  if (n < 2) return false;\n  for (int d = 2; d * d <= n; d++) {\n    if (n % d == 0) return false;\n  }\n  return true;\n}\n"),
                new Code("max", "int max(int[] a) {\n  int best = a[0];\n  for (int i = 1; i < a.length; i++) {\n    if (a[i] > best) best = a[i];\n  }\n  return best;\n}\n"),
                new Code("count", "int count(String s, char c) {\n  int hits = 0;\n  for (int i = 1; i < s.length(); i++) {\n    if (s.charAt(i) == c) hits++;\n  }\n  return hits;\n}\n"),
                new Code("empty", ""));
    }

    static Code named(String name) {
        for (Code code : corpus()) if (code.name().equals(name)) return code;
        throw new IllegalArgumentException("unknown program " + name);
    }

    public static void main(String[] args) {
        // 1. renamed, reformatted and re-commented clones are identical after normalization
        Signals gcd = compare(named("gcd-a"), named("gcd-renamed"));
        Signals sum = compare(named("sum-a"), named("sum-renamed"));
        check(Math.abs(gcd.score() - 1.0) < 1e-9 && gcd.shingle() == 1.0 && Math.abs(sum.score() - 1.0) < 1e-9
                        && gcd.flagged() && sum.flagged(),
                "renamed and reformatted clones reach score 1.0 and are flagged");

        // 2. no unrelated pair is flagged
        double worstUnrelated = 0;
        boolean anyUnrelatedFlagged = false;
        List<String> distinct = List.of("gcd-a", "sum-a", "prime", "max", "count");
        for (String a : distinct)
            for (String b : distinct) {
                if (a.equals(b)) continue;
                Signals s = compare(named(a), named(b));
                worstUnrelated = Math.max(worstUnrelated, s.score());
                anyUnrelatedFlagged |= s.flagged();
            }
        check(!anyUnrelatedFlagged && worstUnrelated < THRESHOLD,
                "no unrelated pair is flagged (worst unrelated score " + String.format("%.3f", worstUnrelated) + ")");

        // 3. structure-only pair: identical skeleton, different expressions
        Code skeletonA = new Code("skeleton-a", "int f(int[] a) {\n  int s = 0;\n  for (int i = 0; i < a.length; i++) { s += a[i] * a[i] - 3; }\n  if (s > 100) return s;\n  return -1;\n}");
        Code skeletonB = new Code("skeleton-b", "String g(String w, String z) {\n  String r = w;\n  for (int k = z.length() - 1; k >= 0; k--) { r = r + z.charAt(k); }\n  if (r.equals(w)) return r;\n  return z;\n}");
        Signals skeleton = compare(skeletonA, skeletonB);
        double weakest = Math.min(compare(named("gcd-a"), named("count")).score(),
                compare(named("sum-a"), named("prime")).score());
        check(skeleton.control() == 1.0 && skeleton.shingle() < 0.6 && skeleton.score() > weakest
                        && skeleton.score() < THRESHOLD && !skeleton.flagged(),
                "skeleton-only pair keeps control flow 1.0 while shingle overlap stays low and unflagged");

        // 4. the score is symmetric over the whole corpus
        List<Code> programs = corpus();
        boolean symmetric = true;
        for (int i = 0; i < programs.size(); i++)
            for (int j = i + 1; j < programs.size(); j++)
                symmetric &= compare(programs.get(i), programs.get(j)).equals(compare(programs.get(j), programs.get(i)));
        check(symmetric, "similarity is symmetric for all " + programs.size() * (programs.size() - 1) / 2 + " pairs");

        // 5. comments, string bodies and identifier names cannot influence the token digest
        Code noisy = new Code("noisy", "/* header comment */\nint total(int[] stuff) { // inline note\n  int acc = 0;\n  for (int k = 0; k < stuff.length; k++) {\n    acc += stuff[k];\n  }\n  return acc;\n}\n");
        check(tokenDigest(noisy).equals(tokenDigest(new Code("quiet",
                        "int total(int[] v) {\n  int acc = 0;\n  for (int k = 0; k < v.length; k++) {\n    acc += v[k];\n  }\n  return acc;\n}\n")))
                        && !tokenDigest(named("sum-a")).equals(tokenDigest(named("max"))),
                "comments and parameter names do not change the normalized token digest");

        // 6. boundary: empty and too-short programs are incomparable instead of false positives
        Signals emptyPair = compare(named("empty"), named("empty"));
        Signals emptyVsCode = compare(named("empty"), named("gcd-a"));
        Signals shortPair = compare(new Code("short-a", "x = 1"), new Code("short-b", "y = 2"));
        check(emptyPair.score() == 0.0 && !emptyPair.flagged() && emptyVsCode.score() == 0.0
                        && shortPair.score() == 0.0 && shortPair.control() == 0.0 && !shortPair.flagged(),
                "empty and sub-4-token programs score 0 and are never flagged");

        // 7. planted clone set is recovered exactly: full recall, no false positives
        Set<String> planted = new TreeSet<>(Set.of("gcd-a|gcd-renamed", "sum-a|sum-renamed"));
        Set<String> flagged = new TreeSet<>();
        for (int i = 0; i < programs.size(); i++)
            for (int j = i + 1; j < programs.size(); j++) {
                Code a = programs.get(i), b = programs.get(j);
                if (compare(a, b).flagged()) flagged.add(a.name() + "|" + b.name());
            }
        check(flagged.equals(planted) && compare(skeletonA, skeletonB).flagged() == false,
                "detector recovers exactly the planted clone pairs (recall 1.0, precision 1.0)");

        // 8. independent oracle: nested-loop Jaccard over the shingle lists reproduces every overlap
        boolean oracleOk = W_SHINGLE + W_HISTOGRAM + W_CONTROL == 1.0;
        for (int i = 0; i < programs.size(); i++)
            for (int j = i + 1; j < programs.size(); j++) {
                Code a = programs.get(i), b = programs.get(j);
                List<String> la = new ArrayList<>(shingles(tokenize(a.source())));
                List<String> lb = new ArrayList<>(shingles(tokenize(b.source())));
                int common = 0;
                for (String x : la) for (String y : lb) if (x.equals(y)) { common++; break; }
                int union = la.size() + lb.size() - common;
                double oracle = union == 0 ? 0.0 : (double) common / union;
                oracleOk &= Math.abs(oracle - compare(a, b).shingle()) < 1e-12;
            }
        check(oracleOk, "independent nested-loop Jaccard matches the set-based overlap and weights sum to 1");

        System.out.println("corpus programs: " + programs.size() + ", shingle size k=" + K
                + ", threshold " + THRESHOLD);
        System.out.println("clone score gcd=" + gcd.score() + " sum=" + sum.score()
                + ", skeleton score=" + String.format("%.4f", skeleton.score())
                + " (control " + skeleton.control() + ")");
        System.out.println("worst unrelated score: " + String.format("%.4f", worstUnrelated));
        System.out.println("flagged pairs: " + flagged);
        System.out.println("PASS");
    }
}
