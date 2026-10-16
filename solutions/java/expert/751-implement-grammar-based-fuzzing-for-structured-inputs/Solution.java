/** Task 751 - Implement grammar-based fuzzing for structured inputs. (Tier: Expert, CTCI Ch 11 - Testing)
 *
 * Three deterministic components (splitmix64 only, never Math.random):
 *  1. Deriver - expands a context-free grammar under a length budget, so every sample is *by
 *     construction* a sentence of the grammar. minLength (a least fixpoint) reserves one character per
 *     symbol, and growability (a second fixpoint) decides which children may absorb the leftover
 *     budget: handing budget to P (which is just "+") would throw it away and collapse the corpus.
 *  2. Parser - exact membership by span dynamic programming over productions of any arity plus a
 *     unit-rule closure per span. An accept always carries a parse-tree witness, and the witness is
 *     re-validated independently of the parser (leaf order, production lhs/arity, child labels), so a
 *     "true" cannot be an accident of the DP table.
 *  3. Walker - the independent oracle: the least-fixpoint language, i.e. every derivable sentence up to
 *     a small length, built by fixpoint rather than by parsing.
 *
 * Simplified: the grammar has no epsilon rules (every symbol consumes at least one character), which is
 * all the span DP needs in order to avoid empty spans; that is the only restriction on its shape.
 *
 * Verified: exhaustive parser/oracle agreement over all 780 strings up to length 4; every fuzzed sample
 * parses, honours the length cap, uses only grammar terminals and yields a valid witness; corpus
 * coverage and determinism for a fixed seed; and the payoff of grammar-based fuzzing - byte mutation
 * drifts out of the language while re-deriving a random subtree stays inside it.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" in the fuzzer. */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

/** A context-free grammar: a single upper-case letter is a nonterminal, anything else a terminal. */
final class Grammar {
    private final List<String> lhs = new ArrayList<>();
    private final List<String[]> rhs = new ArrayList<>();
    private final Map<String, List<Integer>> byLhs = new LinkedHashMap<>();
    private final Set<String> nonterminals = new LinkedHashSet<>();
    private final Set<Character> terminals = new TreeSet<>();

    static boolean isTerminal(String symbol) {
        return symbol.length() != 1 || !Character.isUpperCase(symbol.charAt(0));
    }

    int add(String left, String... right) {
        int id = lhs.size();
        lhs.add(left);
        rhs.add(right);
        nonterminals.add(left);
        for (String symbol : right) {
            if (isTerminal(symbol)) terminals.add(symbol.charAt(0));
            else nonterminals.add(symbol);
        }
        byLhs.computeIfAbsent(left, k -> new ArrayList<>()).add(id);
        return id;
    }

    int size() { return lhs.size(); }
    String lhsOf(int id) { return lhs.get(id); }
    String[] rhsOf(int id) { return rhs.get(id); }
    List<Integer> idsFor(String symbol) { return byLhs.getOrDefault(symbol, List.of()); }
    Set<String> nonterminals() { return nonterminals; }
    Set<Character> terminals() { return terminals; }
    String show(int id) { return lhsOf(id) + " -> " + String.join(" ", rhsOf(id)); }
}

/** Witness tree node: production == -1 marks a terminal leaf. */
final class ParseNode {
    final int production;
    final String symbol;
    final List<ParseNode> children;

    ParseNode(int production, String symbol, List<ParseNode> children) {
        this.production = production;
        this.symbol = symbol;
        this.children = children;
    }

    static ParseNode leaf(String terminal) { return new ParseNode(-1, terminal, List.of()); }
}

/** Exact CFG membership with a witness tree: spans filled bottom-up, unit rules closed per span. */
final class CfgParser {
    private final Grammar grammar;
    private final String start;

    CfgParser(Grammar grammar, String start) { this.grammar = grammar; this.start = start; }

    boolean accepts(String input) { return tree(input) != null; }

    ParseNode tree(String input) {
        int n = input.length();
        if (n == 0) return null;
        int stride = n + 1;
        List<Map<String, ParseNode>> cells = new ArrayList<>(Collections.nCopies(stride * stride, null));
        for (int len = 1; len <= n; len++) {
            for (int i = 0; i + len <= n; i++) {
                int j = i + len;
                Map<String, ParseNode> cell = new LinkedHashMap<>();
                if (len == 1) {
                    for (int id = 0; id < grammar.size(); id++) {
                        String[] r = grammar.rhsOf(id);
                        if (r.length == 1 && Grammar.isTerminal(r[0]) && r[0].charAt(0) == input.charAt(i))
                            cell.put(grammar.lhsOf(id), new ParseNode(id, grammar.lhsOf(id),
                                    List.of(ParseNode.leaf(r[0]))));
                    }
                }
                for (int id = 0; id < grammar.size(); id++) {
                    String[] r = grammar.rhsOf(id);
                    if (r.length < 2 || Arrays.stream(r).anyMatch(Grammar::isTerminal)) continue;
                    List<ParseNode> kids = match(cells, stride, r, 0, i, j);
                    if (kids != null)
                        cell.putIfAbsent(grammar.lhsOf(id), new ParseNode(id, grammar.lhsOf(id), kids));
                }
                boolean changed = true;
                while (changed) {                       // unit productions A -> B, to a fixpoint
                    changed = false;
                    for (int id = 0; id < grammar.size(); id++) {
                        String[] r = grammar.rhsOf(id);
                        if (r.length != 1 || Grammar.isTerminal(r[0])) continue;
                        ParseNode child = cell.get(r[0]);
                        if (child != null && !cell.containsKey(grammar.lhsOf(id))) {
                            cell.put(grammar.lhsOf(id), new ParseNode(id, grammar.lhsOf(id), List.of(child)));
                            changed = true;
                        }
                    }
                }
                cells.set(i * stride + j, cell);
            }
        }
        return cells.get(n).get(start);
    }

    /** rhs[idx..] against input[start, end), trying every split; each symbol eats >= 1 character. */
    private List<ParseNode> match(List<Map<String, ParseNode>> cells, int stride, String[] rhs, int idx,
                                  int start, int end) {
        if (idx == rhs.length) return start == end ? new ArrayList<>() : null;
        if (idx == rhs.length - 1) {
            if (start >= end) return null;
            Map<String, ParseNode> cell = cells.get(start * stride + end);
            ParseNode node = cell == null ? null : cell.get(rhs[idx]);
            return node == null ? null : new ArrayList<>(List.of(node));
        }
        int minRest = rhs.length - idx - 1;
        for (int mid = start + 1; mid + minRest <= end; mid++) {
            Map<String, ParseNode> left = cells.get(start * stride + mid);
            ParseNode head = left == null ? null : left.get(rhs[idx]);
            if (head == null) continue;
            List<ParseNode> tail = match(cells, stride, rhs, idx + 1, mid, end);
            if (tail == null) continue;
            List<ParseNode> out = new ArrayList<>();
            out.add(head);
            out.addAll(tail);
            return out;
        }
        return null;
    }
}

/** Grammar-driven sample generator: expands productions under a length budget. */
final class Deriver {
    private final Grammar grammar;
    private final Prng rng;
    private final int depthCap;
    private final Map<String, Integer> minLength = new HashMap<>();
    private final Set<String> growable = new TreeSet<>();
    final Set<Integer> usedProductions = new TreeSet<>();
    int fixpointRounds;

    Deriver(Grammar grammar, long seed, int depthCap) {
        this.grammar = grammar;
        this.rng = new Prng(seed);
        this.depthCap = depthCap;
        computeMinLengths();
        computeGrowable();
    }

    /** Least fixpoint of "min length of a sentence derived from a symbol". */
    private void computeMinLengths() {
        for (Character terminal : grammar.terminals()) minLength.put(String.valueOf(terminal), 1);
        boolean changed = true;
        while (changed) {
            changed = false;
            fixpointRounds++;
            for (int id = 0; id < grammar.size(); id++) {
                int total = 0;
                for (String symbol : grammar.rhsOf(id)) {
                    Integer m = minLength.get(symbol);
                    if (m == null) { total = Integer.MAX_VALUE; break; }
                    total += m;
                }
                String left = grammar.lhsOf(id);
                Integer current = minLength.get(left);
                if (total != Integer.MAX_VALUE && (current == null || total < current)) {
                    minLength.put(left, total);
                    changed = true;
                }
            }
        }
    }

    /**
     * Symbols whose derivation can exceed their minimal length: a production with two or more
     * right-hand-side symbols always yields at least that many characters, and a unit production
     * inherits the capacity of its child. Only these can absorb leftover budget - a share given to P
     * (which is just "+") is thrown away, which is what kept an earlier version of this generator short.
     */
    private void computeGrowable() {
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int id = 0; id < grammar.size(); id++) {
                String left = grammar.lhsOf(id);
                if (growable.contains(left)) continue;
                String[] right = grammar.rhsOf(id);
                if (right.length >= 2 || (!Grammar.isTerminal(right[0]) && growable.contains(right[0]))) {
                    growable.add(left);
                    changed = true;
                }
            }
        }
    }

    int minLengthOf(String symbol) { return minLength.getOrDefault(symbol, Integer.MAX_VALUE); }

    boolean isGrowable(String symbol) { return growable.contains(symbol); }

    /** A sentence of `start` whose length does not exceed lengthCap. */
    String next(String start, int lengthCap) {
        int floor = minLengthOf(start);
        for (int attempt = 0; attempt < 8; attempt++) {
            int budget = floor + (lengthCap > floor ? rng.nextInt(lengthCap - floor + 1) : 0);
            String sentence = derive(start, budget, depthCap);
            if (sentence != null) return sentence;
        }
        return derive(start, lengthCap, depthCap);
    }

    private String derive(String symbol, int budget, int depth) {
        if (Grammar.isTerminal(symbol)) return budget >= 1 ? symbol : null;
        List<Integer> candidates = new ArrayList<>(grammar.idsFor(symbol));
        candidates.removeIf(id -> minCost(id) > budget);
        while (!candidates.isEmpty()) {
            int pick = candidates.get(0);
            if (depth <= 0) {
                // At the depth limit only the cheapest production is guaranteed to bottom out; minimal
                // derivations terminate because this grammar's unit chains (S->E->T->F->n) end in one.
                for (int id : candidates) if (minCost(id) < minCost(pick)) pick = id;
            } else {
                pick = candidates.get(weightedPick(candidates, budget));
            }
            String sentence = expand(pick, budget, depth);
            if (sentence != null) return sentence;
            candidates.remove((Integer) pick);
        }
        return null;
    }

    /**
     * A production that can absorb leftover budget gets weight 4 + slack (slack = budget left unused by
     * the cheapest candidate); a terminal-only production yields one character whatever the budget, so
     * it keeps weight 1 and short samples stay possible but rare.
     */
    private int weightedPick(List<Integer> candidates, int budget) {
        int cheapest = Integer.MAX_VALUE;
        for (int id : candidates) cheapest = Math.min(cheapest, minCost(id));
        int slack = Math.max(0, budget - cheapest);
        int total = 0;
        for (int id : candidates) total += weight(id, slack);
        int roll = rng.nextInt(total);
        for (int i = 0; i < candidates.size(); i++) {
            roll -= weight(candidates.get(i), slack);
            if (roll < 0) return i;
        }
        return candidates.size() - 1;
    }

    private int weight(int production, int slack) {
        for (String symbol : grammar.rhsOf(production)) if (!Grammar.isTerminal(symbol)) return 4 + slack;
        return 1;
    }

    private String expand(int production, int budget, int depth) {
        String[] right = grammar.rhsOf(production);
        int reserved = 0;
        List<Integer> flexible = new ArrayList<>();
        for (int i = 0; i < right.length; i++) {
            reserved += minLengthOf(right[i]);
            if (!Grammar.isTerminal(right[i]) && growable.contains(right[i])) flexible.add(i);
        }
        int leftover = budget - reserved;
        if (leftover < 0) return null;
        String[] pieces = new String[right.length];
        for (int slot = 0; slot < flexible.size(); slot++) {
            // Random composition of the leftover with the last child absorbing the remainder: giving
            // each child an independent uniform share would discard half the budget at every level.
            int extra = slot == flexible.size() - 1 ? leftover : (leftover > 0 ? rng.nextInt(leftover + 1) : 0);
            leftover -= extra;
            int index = flexible.get(slot);
            pieces[index] = derive(right[index], minLengthOf(right[index]) + extra, Math.max(0, depth - 1));
        }
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < right.length; i++) {
            String piece = Grammar.isTerminal(right[i]) ? right[i] : pieces[i];
            if (piece == null) {
                // A non-growable nonterminal (P, M, L, R) gets no budget to absorb, so expand minimally.
                piece = derive(right[i], minLengthOf(right[i]), Math.max(0, depth - 1));
            }
            if (piece == null) return null;
            out.append(piece);
        }
        usedProductions.add(production);
        return out.toString();
    }

    private int minCost(int production) {
        int total = 0;
        for (String symbol : grammar.rhsOf(production)) total += minLengthOf(symbol);
        return total;
    }
}

public class Solution {

    static Grammar buildGrammar() {
        Grammar g = new Grammar();
        g.add("S", "E");
        g.add("E", "E", "P", "T");
        g.add("E", "T");
        g.add("T", "T", "M", "F");
        g.add("T", "F");
        g.add("F", "n");
        g.add("F", "L", "E", "R");
        g.add("P", "+");
        g.add("M", "*");
        g.add("L", "(");
        g.add("R", ")");
        return g;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    /** Leaves of a witness tree, in order. */
    static void collectLeaves(ParseNode node, StringBuilder out) {
        if (node.production < 0) { out.append(node.symbol); return; }
        for (ParseNode child : node.children) collectLeaves(child, out);
    }

    /**
     * Re-validates a witness tree against the grammar without the parser: internal nodes must use a
     * declared production, match its lhs and arity, and their children must carry the rhs labels.
     */
    static void validateWitness(Grammar grammar, ParseNode node) {
        StringBuilder unused = new StringBuilder();
        validateNode(grammar, node, unused);
    }

    private static void validateNode(Grammar grammar, ParseNode node, StringBuilder sink) {
        if (node.production < 0) {
            check(Grammar.isTerminal(node.symbol), "witness leaf is terminal: '" + node.symbol + "'");
            return;
        }
        check(node.production < grammar.size(), "witness production id in range");
        check(grammar.lhsOf(node.production).equals(node.symbol), "witness label matches production lhs");
        String[] right = grammar.rhsOf(node.production);
        check(right.length == node.children.size(), "witness arity of " + grammar.show(node.production));
        for (int i = 0; i < right.length; i++) {
            check(right[i].equals(node.children.get(i).symbol),
                    "witness child " + i + " of " + grammar.show(node.production) + " is '"
                            + node.children.get(i).symbol + "'");
            validateNode(grammar, node.children.get(i), sink);
        }
    }

    /** Independent oracle: the least-fixpoint language, i.e. all derivable sentences up to maxLen. */
    static Map<String, Set<String>> derivableLanguage(Grammar grammar, int maxLen) {
        Map<String, Set<String>> language = new HashMap<>();
        for (String symbol : grammar.nonterminals()) language.put(symbol, new TreeSet<>());
        for (Character terminal : grammar.terminals())
            language.put(String.valueOf(terminal), new TreeSet<>(List.of(String.valueOf(terminal))));
        boolean changed = true;
        int rounds = 0;
        while (changed) {
            changed = false;
            check(++rounds <= 64, "language fixpoint converges");
            for (int id = 0; id < grammar.size(); id++) {
                Set<String> accumulated = new TreeSet<>(List.of(""));
                for (String symbol : grammar.rhsOf(id)) {
                    Set<String> pieces = new TreeSet<>();
                    for (String prefix : accumulated) for (String piece : language.get(symbol))
                        if (prefix.length() + piece.length() <= maxLen) pieces.add(prefix + piece);
                    accumulated = pieces;
                    if (accumulated.isEmpty()) break;
                }
                if (language.get(grammar.lhsOf(id)).addAll(accumulated)) changed = true;
            }
        }
        return language;
    }

    /** All strings over an alphabet up to a given length, in dictionary order. */
    static List<String> enumerateStrings(Set<Character> alphabet, int maxLen) {
        List<Character> letters = new ArrayList<>(alphabet);
        List<String> out = new ArrayList<>();
        List<String> frontier = new ArrayList<>(List.of(""));
        for (int len = 1; len <= maxLen; len++) {
            List<String> next = new ArrayList<>();
            for (String prefix : frontier) for (Character letter : letters) next.add(prefix + letter);
            out.addAll(next);
            frontier = next;
        }
        return out;
    }

    /** Witness node plus the substring of the sample it covers - the target unit for splicing. */
    static final class Span {
        final ParseNode node;
        final int start;
        final int end;
        Span(ParseNode node, int start, int end) { this.node = node; this.start = start; this.end = end; }
    }

    static void collectSpans(ParseNode node, int start, List<Span> out) {
        out.add(new Span(node, start, start + spanLength(node)));
        if (node.production < 0) return;
        int cursor = start;
        for (ParseNode child : node.children) {
            collectSpans(child, cursor, out);
            cursor += spanLength(child);
        }
    }

    static int spanLength(ParseNode node) {
        if (node.production < 0) return 1;
        int total = 0;
        for (ParseNode child : node.children) total += spanLength(child);
        return total;
    }

    /** The corpus: 400 samples, each parsed, witnessed and validated. */
    static List<String> fuzz(Grammar grammar, CfgParser parser, Deriver deriver, int lengthCap) {
        List<String> corpus = new ArrayList<>();
        for (int i = 0; i < 400; i++) {
            String sample = deriver.next("S", lengthCap);
            check(sample != null && sample.length() <= lengthCap, "sample within the cap");
            for (char c : sample.toCharArray()) {
                check(grammar.terminals().contains(c), "sample uses grammar terminals only: '" + c + "'");
            }
            ParseNode witness = parser.tree(sample);
            check(witness != null, "fuzzed sample is accepted: '" + sample + "'");
            validateWitness(grammar, witness);
            StringBuilder leaves = new StringBuilder();
            collectLeaves(witness, leaves);
            check(leaves.toString().equals(sample), "witness covers the sample exactly: '" + sample + "'");
            corpus.add(sample);
        }
        return corpus;
    }

    public static void main(String[] args) {
        Grammar grammar = buildGrammar();
        CfgParser parser = new CfgParser(grammar, "S");

        // 1. grammar well-formedness: every nonterminal is productive and the alphabet is the intended one
        for (String symbol : grammar.nonterminals())
            check(!grammar.idsFor(symbol).isEmpty(), "nonterminal " + symbol + " has productions");
        check(grammar.terminals().equals(new TreeSet<>(List.of('n', '(', ')', '+', '*'))),
                "terminal alphabet " + grammar.terminals());
        for (int id = 0; id < grammar.size(); id++) {
            check(grammar.rhsOf(id).length >= 1 && grammar.rhsOf(id).length <= 3, "no epsilon: " + grammar.show(id));
            check(!Grammar.isTerminal(grammar.lhsOf(id)), "lhs is a nonterminal: " + grammar.show(id));
        }

        // 2. min length and growability fixpoints
        Deriver deriver = new Deriver(grammar, 0xC0FFEE12345L, 14);
        check(deriver.minLengthOf("S") == 1 && deriver.minLengthOf("E") == 1 && deriver.minLengthOf("F") == 1,
                "minLength(S/E/F) == 1");
        check(deriver.fixpointRounds <= grammar.nonterminals().size() + 2,
                "minLength fixpoint rounds " + deriver.fixpointRounds);
        for (String symbol : new String[]{"S", "E", "T", "F"})
            check(deriver.isGrowable(symbol), symbol + " can absorb extra budget");
        for (String symbol : new String[]{"P", "M", "L", "R"})
            check(!deriver.isGrowable(symbol), symbol + " always yields exactly one character");

        // 3. exhaustive differential test: parser vs the independent language oracle
        Set<String> oracle = derivableLanguage(grammar, 4).get("S");
        List<String> universe = enumerateStrings(grammar.terminals(), 4);
        int mismatches = 0, accepted = 0;
        for (String candidate : universe) {
            boolean byParser = parser.accepts(candidate);
            if (byParser) accepted++;
            if (byParser != oracle.contains(candidate)) {
                mismatches++;
                System.out.println("  mismatch: '" + candidate + "' (" + byParser + " vs oracle)");
            }
        }
        check(universe.size() == 780, "enumerated " + universe.size() + " candidates");
        check(mismatches == 0, "parser and language oracle agree everywhere (mismatches=" + mismatches + ")");
        check(accepted == oracle.size(), "accepted " + accepted + " == oracle size " + oracle.size());
        int shortest = Integer.MAX_VALUE;
        for (String sentence : oracle) shortest = Math.min(shortest, sentence.length());
        check(shortest == deriver.minLengthOf("S"), "oracle's shortest sentence is minLength(S)");

        // 4. the fuzzed corpus
        List<String> corpus = fuzz(grammar, parser, deriver, 30);
        Set<String> distinct = new TreeSet<>(corpus);
        check(distinct.size() >= 200, "corpus has " + distinct.size() + " distinct samples");
        check(deriver.usedProductions.size() == grammar.size(),
                "corpus covers all " + grammar.size() + " productions, got " + deriver.usedProductions.size());
        int longest = 0, shortestSample = Integer.MAX_VALUE;
        for (String sample : corpus) {
            longest = Math.max(longest, sample.length());
            shortestSample = Math.min(shortestSample, sample.length());
        }
        check(longest >= 20 && shortestSample <= 4, "length span [" + shortestSample + ", " + longest + "]");

        // 5. determinism: a fresh deriver with the same seed reproduces the corpus exactly
        Deriver replay = new Deriver(grammar, 0xC0FFEE12345L, 14);
        List<String> replayCorpus = new ArrayList<>();
        for (int i = 0; i < 400; i++) replayCorpus.add(replay.next("S", 30));
        check(replayCorpus.equals(corpus), "same seed reproduces the corpus");

        // 6. boundaries: hand-written accepts and rejects
        for (String good : new String[]{"n", "(n)", "n+n", "n*n*n", "((n))", "n+n*n", "(n+n)*(n+n)"})
            check(parser.accepts(good), "accepts '" + good + "'");
        for (String bad : new String[]{"", "(", ")", "+", "n+", "(n", "n n", "*", "nn", ")n("})
            check(!parser.accepts(bad), "rejects '" + bad + "'");

        // 7. mutation differential: byte havoc vs structure-aware (subtree) havoc
        Prng mutationRng = new Prng(0x5EED5EEDL);
        int byteAccepted = 0;
        for (int i = 0; i < 300; i++) {
            String sample = corpus.get(mutationRng.nextInt(corpus.size()));
            if (parser.accepts(byteMutate(sample, grammar, mutationRng))) byteAccepted++;
            String spliced = splice(grammar, deriver, parser.tree(sample), mutationRng);
            ParseNode witness = parser.tree(spliced);
            check(witness != null, "structure-aware mutant stays in the language: '" + spliced + "'");
            validateWitness(grammar, witness);
        }
        check(byteAccepted < 60, "byte mutation acceptance is low (" + byteAccepted + "/300)");

        System.out.println("grammar: " + grammar.size() + " productions, terminals " + grammar.terminals());
        System.out.println("oracle up to length 4: " + oracle.size() + " sentences from " + universe.size()
                + " candidates tested exhaustively");
        System.out.println("corpus: " + corpus.size() + " samples (" + distinct.size() + " distinct), lengths ["
                + shortestSample + ", " + longest + "], productions covered "
                + deriver.usedProductions.size() + "/" + grammar.size());
        System.out.println("mutants inside the language: byte " + byteAccepted + "/300, subtree splicing 300/300");
        System.out.println("PASS");
    }

    /** Byte-level havoc: replace, insert or delete one character at a pseudo-random position. */
    static String byteMutate(String sample, Grammar grammar, Prng rng) {
        List<Character> letters = new ArrayList<>(grammar.terminals());
        int position = rng.nextInt(sample.length());
        char replacement = letters.get(rng.nextInt(letters.size()));
        int kind = rng.nextInt(3);
        if (kind == 0) return sample.substring(0, position) + replacement + sample.substring(position + 1);
        if (kind == 1) return sample.substring(0, position) + replacement + sample.substring(position);
        return sample.substring(0, position) + sample.substring(position + 1);
    }

    /** Structure-aware havoc: re-derive the subtree at a random node and splice it back in. */
    static String splice(Grammar grammar, Deriver deriver, ParseNode witness, Prng rng) {
        List<Span> spans = new ArrayList<>();
        collectSpans(witness, 0, spans);
        Span pick = spans.get(rng.nextInt(spans.size()));
        String replacement = null;
        for (int attempt = 0; attempt < 16 && replacement == null; attempt++) {
            int budget = Math.max(deriver.minLengthOf(pick.node.symbol),
                    pick.end - pick.start + rng.nextInt(4) - 1);
            replacement = deriver.next(pick.node.symbol, budget);
        }
        check(replacement != null, "re-derivation succeeds for " + pick.node.symbol);
        StringBuilder leaves = new StringBuilder();
        collectLeaves(witness, leaves);
        String sample = leaves.toString();
        return sample.substring(0, pick.start) + replacement + sample.substring(pick.end);
    }
}
