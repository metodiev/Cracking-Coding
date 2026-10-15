/** Task 405 - Implement a minimal DFA for a fixed string set (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Scope: the DFA recognizes exactly the finite set of words it was built from. Only the
 * *prefixes* of the set's words matter: the builder's states are exactly the distinct
 * non-empty prefixes plus the empty root, and any character no prefix uses leads to an
 * explicit dead state. The alphabet is the sorted union of the words' characters; characters
 * outside it are documented to reject.
 *
 * Minimization uses the classic table-filling (Moore) algorithm: mark every pair that differs
 * in acceptance, then repeatedly mark a pair when some character leads to an already marked
 * pair; inseparable pairs are merged with union-find. For {"he", "she", "his"} the three
 * accepting states merge into one class, so the complete minimal DFA has 7 states: {""},
 * {"h"}, {"s"}, {"sh"}, {"hi"}, {"he","she","his"} and one dead state. Note "h" does NOT merge
 * with "sh": "h" still accepts the continuation "is" ("his") while "sh" only accepts "e".
 * Dropping the dead state leaves the 6-state partial DFA that tracks exactly the prefixes.
 *
 * Checks: the enumerated language up to length 4 equals the key set exactly, near-miss strings
 * are rejected, every state is reachable, and the minimized automaton has no pair of
 * equivalent states (product-BFS equivalence test), i.e. it is minimal.
 */
import java.util.*;

class ExactMatchDfa {
    final List<String> states = new ArrayList<>();
    final Map<String, Integer> index = new HashMap<>();
    final List<int[]> trans = new ArrayList<>();
    final List<Boolean> accept = new ArrayList<>();
    final int alphabetSize;
    final String alphabet;
    int dead;

    ExactMatchDfa(String[] words, String alphabet) {
        this.alphabet = alphabet;
        this.alphabetSize = alphabet.length();
        states.add("");
        index.put("", 0);
        trans.add(filled(-1));
        accept.add(false);
        for (String word : words) {
            int node = 0;
            String prefix = "";
            for (int i = 0; i < word.length(); i++) {
                char ch = word.charAt(i);
                String nxt = prefix + ch;
                Integer id = index.get(nxt);
                if (id == null) {
                    id = states.size();
                    index.put(nxt, id);
                    states.add(nxt);
                    trans.add(filled(-1));
                    accept.add(false);
                }
                int col = alphabet.indexOf(ch);
                if (trans.get(node)[col] == -1) trans.get(node)[col] = id;
                node = id;
                prefix = nxt;
            }
            accept.set(node, true);
        }
        dead = states.size();
        states.add("(dead)");
        accept.add(false);
        trans.add(filled(dead));
        for (int[] row : trans)
            for (int c = 0; c < alphabetSize; c++)
                if (row[c] == -1) row[c] = dead;
    }

    private int[] filled(int value) {
        int[] row = new int[alphabetSize];
        Arrays.fill(row, value);
        return row;
    }

    int[][] transArray() {
        return trans.toArray(new int[0][]);
    }

    boolean[] acceptArray() {
        boolean[] out = new boolean[accept.size()];
        for (int i = 0; i < out.length; i++) out[i] = accept.get(i);
        return out;
    }
}

class Minimized {
    final int[] classOf;
    final int[][] trans;
    final boolean[] accept;

    Minimized(int[] classOf, int[][] trans, boolean[] accept) {
        this.classOf = classOf;
        this.trans = trans;
        this.accept = accept;
    }
}

class DfaTools {
    static Minimized minimize(int[][] trans, boolean[] accept, int alphabetSize) {
        int n = trans.length;
        boolean[][] marked = new boolean[n][n];
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++)
                if (accept[p] != accept[q]) marked[p][q] = true;
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (marked[p][q]) continue;
                    for (int c = 0; c < alphabetSize; c++) {
                        int a = trans[p][c], b = trans[q][c];
                        if (a == b) continue;
                        int lo = Math.min(a, b), hi = Math.max(a, b);
                        if (marked[lo][hi]) {
                            marked[p][q] = true;
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++)
                if (!marked[p][q]) {
                    int rp = find(parent, p), rq = find(parent, q);
                    if (rp != rq) parent[Math.max(rp, rq)] = Math.min(rp, rq);
                }
        TreeSet<Integer> repSet = new TreeSet<>();
        for (int i = 0; i < n; i++) repSet.add(find(parent, i));
        List<Integer> reps = new ArrayList<>(repSet);
        Map<Integer, Integer> remap = new HashMap<>();
        for (int i = 0; i < reps.size(); i++) remap.put(reps.get(i), i);
        int[] classOf = new int[n];
        for (int i = 0; i < n; i++) classOf[i] = remap.get(find(parent, i));
        int[][] newTrans = new int[reps.size()][alphabetSize];
        boolean[] newAccept = new boolean[reps.size()];
        for (int r = 0; r < reps.size(); r++) {
            newAccept[r] = accept[reps.get(r)];
            for (int c = 0; c < alphabetSize; c++) newTrans[r][c] = classOf[trans[reps.get(r)][c]];
        }
        return new Minimized(classOf, newTrans, newAccept);
    }

    private static int find(int[] parent, int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    static boolean accepts(int[][] trans, boolean[] accept, String word, String alphabet) {
        int state = 0;
        for (int i = 0; i < word.length(); i++) {
            int col = alphabet.indexOf(word.charAt(i));
            if (col == -1) return false;              // documented: outside the alphabet, reject
            state = trans[state][col];
        }
        return accept[state];
    }

    static Set<Integer> reachable(int[][] trans) {
        Set<Integer> seen = new HashSet<>();
        Deque<Integer> stack = new ArrayDeque<>();
        seen.add(0);
        stack.push(0);
        while (!stack.isEmpty()) {
            int state = stack.pop();
            for (int nxt : trans[state])
                if (seen.add(nxt)) stack.push(nxt);
        }
        return seen;
    }

    static boolean allPairsDistinguishable(int[][] trans, boolean[] accept, int alphabetSize) {
        int n = trans.length;
        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                Set<Long> seen = new HashSet<>();
                Deque<int[]> stack = new ArrayDeque<>();
                seen.add(pair(p, q));
                stack.push(new int[]{p, q});
                boolean equivalent = true;
                while (!stack.isEmpty() && equivalent) {
                    int[] top = stack.pop();
                    int a = top[0], b = top[1];
                    if (accept[a] != accept[b]) {
                        equivalent = false;
                        break;
                    }
                    for (int c = 0; c < alphabetSize; c++) {
                        int na = trans[a][c], nb = trans[b][c];
                        long key = pair(Math.min(na, nb), Math.max(na, nb));
                        if (seen.add(key)) stack.push(new int[]{na, nb});
                    }
                }
                if (equivalent) return false;
            }
        }
        return true;
    }

    private static long pair(int a, int b) {
        return a * 1000003L + b;
    }

    static Set<String> language(int[][] trans, boolean[] accept, String alphabet, int maxLen) {
        Set<String> accepted = new TreeSet<>();
        List<String> frontier = new ArrayList<>();
        frontier.add("");
        for (int depth = 0; depth <= maxLen; depth++) {
            List<String> nextFrontier = new ArrayList<>();
            for (String word : frontier) {
                if (accepts(trans, accept, word, alphabet)) accepted.add(word);
                if (word.length() < maxLen)
                    for (int c = 0; c < alphabet.length(); c++) nextFrontier.add(word + alphabet.charAt(c));
            }
            frontier = nextFrontier;
        }
        return accepted;
    }

    static Set<Integer> deadStates(int[][] trans, boolean[] accept) {
        Set<Integer> dead = new HashSet<>();
        for (int state = 0; state < trans.length; state++) {
            if (accept[state]) continue;
            boolean allSelf = true;
            for (int nxt : trans[state]) {
                if (nxt != state) {
                    allSelf = false;
                    break;
                }
            }
            if (allSelf) dead.add(state);
        }
        return dead;
    }

    static String join(Set<Integer> values) {
        StringBuilder sb = new StringBuilder();
        for (int value : new TreeSet<>(values)) {
            if (sb.length() > 0) sb.append(",");
            sb.append(value);
        }
        return sb.toString();
    }
}

public class Solution {
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    public static void main(String[] args) {
        String alphabet = "ehis";
        String[] words = {"he", "she", "his"};
        ExactMatchDfa builder = new ExactMatchDfa(words, alphabet);
        String[] expectReject = {"", "h", "s", "e", "i", "heh", "hehe", "sh", "shi", "shee", "shes",
                "hiss", "hishe", "hhi", "hh", "sheh", "hise"};
        int[][] trans = builder.transArray();
        boolean[] accept = builder.acceptArray();

        Set<String> expectedPrefixes = new TreeSet<>();
        expectedPrefixes.add("");
        for (String word : words)
            for (int k = 1; k <= word.length(); k++) expectedPrefixes.add(word.substring(0, k));
        check(new TreeSet<>(builder.states.subList(0, builder.states.size() - 1)).equals(expectedPrefixes),
                "prefix states");
        check(builder.states.size() == expectedPrefixes.size() + 1 && builder.dead == builder.states.size() - 1,
                "dead state");
        check(DfaTools.reachable(trans).size() == trans.length, "all reachable");

        for (String word : words) check(DfaTools.accepts(trans, accept, word, alphabet), "accept " + word);
        for (String word : expectReject) check(!DfaTools.accepts(trans, accept, word, alphabet), "reject " + word);
        for (String word : new String[]{"hey", "his!", "she?"})
            check(!DfaTools.accepts(trans, accept, word, alphabet), "outside alphabet " + word);
        check(DfaTools.language(trans, accept, alphabet, 4).equals(new TreeSet<>(Arrays.asList(words))),
                "trie language");

        Minimized minimal = DfaTools.minimize(trans, accept, alphabet.length());
        check(minimal.trans.length == 7, "minimal states " + minimal.trans.length);
        int acceptingClasses = 0;
        for (boolean value : minimal.accept) if (value) acceptingClasses++;
        check(acceptingClasses == 1, "one accepting class");
        check(DfaTools.reachable(minimal.trans).size() == minimal.trans.length, "minimal reachable");
        check(DfaTools.allPairsDistinguishable(minimal.trans, minimal.accept, alphabet.length()), "minimal");

        check(DfaTools.language(minimal.trans, minimal.accept, alphabet, 4).equals(new TreeSet<>(Arrays.asList(words))),
                "minimal language");
        for (String word : words) check(DfaTools.accepts(minimal.trans, minimal.accept, word, alphabet), "min accept " + word);
        for (String word : expectReject) check(!DfaTools.accepts(minimal.trans, minimal.accept, word, alphabet), "min reject " + word);

        int classHe = minimal.classOf[builder.index.get("he")];
        int classShe = minimal.classOf[builder.index.get("she")];
        int classHis = minimal.classOf[builder.index.get("his")];
        check(classHe == classShe && classShe == classHis, "accepting merge");
        check(minimal.classOf[builder.index.get("h")] != minimal.classOf[builder.index.get("sh")], "h vs sh");
        check(minimal.classOf[builder.index.get("sh")] != minimal.classOf[builder.index.get("hi")], "sh vs hi");
        check(minimal.classOf[builder.index.get("h")] != minimal.classOf[builder.index.get("s")], "h vs s");
        check(minimal.classOf[builder.index.get("")] != minimal.classOf[builder.dead], "root vs dead");
        Set<Integer> distinctClasses = new HashSet<>();
        for (int value : minimal.classOf) distinctClasses.add(value);
        check(distinctClasses.size() == 7, "distinct classes");
        check(DfaTools.deadStates(minimal.trans, minimal.accept).size() == 1, "single dead class");
        check(minimal.trans.length - 1 == 6, "partial size");

        // second fixed set: {"ab", "b", "ba"} over {"a","b"} -> 5 states
        String alphabet2 = "ab";
        String[] words2 = {"ab", "b", "ba"};
        ExactMatchDfa builder2 = new ExactMatchDfa(words2, alphabet2);
        int[][] trans2 = builder2.transArray();
        boolean[] accept2 = builder2.acceptArray();
        for (String word : words2) check(DfaTools.accepts(trans2, accept2, word, alphabet2), "set2 accept " + word);
        for (String word : new String[]{"", "a", "bb", "aba", "abb", "bab", "baa"})
            check(!DfaTools.accepts(trans2, accept2, word, alphabet2), "set2 reject " + word);
        check(DfaTools.language(trans2, accept2, alphabet2, 4).equals(new TreeSet<>(Arrays.asList(words2))),
                "set2 language");
        Minimized minimal2 = DfaTools.minimize(trans2, accept2, alphabet2.length());
        check(minimal2.trans.length == 5, "set2 minimal states " + minimal2.trans.length);
        check(DfaTools.allPairsDistinguishable(minimal2.trans, minimal2.accept, alphabet2.length()), "set2 minimal");
        check(DfaTools.language(minimal2.trans, minimal2.accept, alphabet2, 4).equals(new TreeSet<>(Arrays.asList(words2))),
                "set2 minimal language");
        check(minimal2.classOf[builder2.index.get("ab")] == minimal2.classOf[builder2.index.get("ba")], "set2 merge");
        check(minimal2.classOf[builder2.index.get("ab")] != minimal2.classOf[builder2.index.get("b")], "set2 b differs");
        check(minimal2.classOf[builder2.index.get("a")] != minimal2.classOf[builder2.dead], "set2 a viable");
        System.out.println("PASS");
    }
}
