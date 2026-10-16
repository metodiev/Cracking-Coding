/** Task 528 - Implement Karp-Rabin for many patterns with automaton. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Multi-pattern substring search with two independent engines plus a naive oracle. (1) Karp-Rabin
 * with length bucketing: the patterns are grouped by length, each bucket gets a hash table from the
 * pattern hash to the pattern indices, and a single rolling hash slides over the text once per
 * distinct pattern length (O(n) windows per length, each O(1)); every hash hit is CONFIRMED by a
 * direct character comparison, so even a deliberately weak modulus (base 3, modulus 7) can never
 * produce a false report - the checks require the collision counter to be positive there, which
 * proves the confirmation path is actually exercised. The rolling recurrence
 * h_{i+1} = ((h_i - s_i * base^(L-1)) * base + s_{i+L}) mod M is verified window by window against
 * a direct hash recomputation. (2) Aho-Corasick over the 26 lowercase letters: a trie of the
 * patterns augmented with BFS failure links and inherited output lists, which reports every pattern
 * ending at every text position. (3) A naive per-pattern scan is used as the oracle. Documented
 * conventions: occurrence positions are 0-based START indices, returned in ascending order; the
 * empty pattern matches at every position 0..n (n + 1 occurrences) and is handled outside the
 * hashing loop; duplicate patterns in the input get identical occurrence lists; a pattern longer
 * than the text has none.
 *
 * Checks: hand cases "aaaa" with a / aa / aaa, "abababa" with ab / ba / aba, overlapping "aa" in
 * "aaaa", absent patterns, duplicate patterns and an empty pattern; the rolling hash recurrence
 * equals a direct hash for every window of every bucket; the per-length window count equals
 * n - L + 1; on 10 LCG text / pattern-set pairs all three engines must produce identical occurrence
 * lists (compared element-wise, including lengths and orders) and identical total counts; the weak
 * modulus run must still be exact while reporting at least one rejected hash hit; the Aho-Corasick
 * automaton must have at most (total pattern length + 1) states, its failure links must point to the
 * longest proper suffix that is a trie prefix (verified by direct string reasoning over the states),
 * and its output lists must reproduce the occurrences (the classic "she / he / his / hers" set is
 * scanned inside "ushers"); the counter invariant candidateChecks = confirmations + rejections must
 * hold and at most one table lookup may happen per window; a 3000-character text with 40 patterns must
 * finish with the same results as the naive oracle; repeated runs are identical.
 */
import java.util.*;

public class Solution {
    static final int DEFAULT_BASE = 131;
    static final int DEFAULT_MOD = 1_000_000_007;

    static class Stats {
        long windows;
        long hashHits;
        long candidateChecks;
        long confirmations;
        long rejections;
    }

    static long charValue(char c) {
        return (c - 'a') + 1;
    }

    static long hashOf(String s, int base, int mod) {
        long h = 0;
        for (int i = 0; i < s.length(); i++) {
            h = (h * base + charValue(s.charAt(i))) % mod;
        }
        return h;
    }

    /** Karp-Rabin with one rolling pass per distinct pattern length. */
    static List<List<Integer>> findKarpRabin(String text, String[] patterns, int base, int mod,
            Stats stats) {
        int n = text.length();
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < patterns.length; i++) result.add(new ArrayList<>());
        Map<Integer, List<Integer>> byLength = new TreeMap<>();
        for (int i = 0; i < patterns.length; i++) {
            int length = patterns[i].length();
            if (length == 0) continue;
            byLength.computeIfAbsent(length, key -> new ArrayList<>()).add(i);
        }
        for (Map.Entry<Integer, List<Integer>> entry : byLength.entrySet()) {
            int length = entry.getKey();
            if (length > n) continue;
            Map<Long, List<Integer>> table = new HashMap<>();
            for (int index : entry.getValue()) {
                table.computeIfAbsent(hashOf(patterns[index], base, mod), key -> new ArrayList<>())
                        .add(index);
            }
            long power = 1;
            for (int i = 0; i + 1 < length; i++) power = power * base % mod;
            long current = hashOf(text.substring(0, length), base, mod);
            for (int position = 0; position + length <= n; position++) {
                if (position > 0) {
                    long out = charValue(text.charAt(position - 1)) * power % mod;
                    current = ((current - out + mod) % mod) * base % mod;
                    current = (current + charValue(text.charAt(position + length - 1))) % mod;
                }
                stats.windows++;
                List<Integer> candidates = table.get(current);
                if (candidates == null) continue;
                stats.hashHits++;
                for (int index : candidates) {
                    stats.candidateChecks++;
                    if (text.startsWith(patterns[index], position)) {
                        result.get(index).add(position);
                        stats.confirmations++;
                    } else {
                        stats.rejections++;
                    }
                }
            }
        }
        for (int i = 0; i < patterns.length; i++) {
            if (patterns[i].isEmpty()) {
                for (int position = 0; position <= n; position++) result.get(i).add(position);
            }
        }
        return result;
    }

    /** Aho-Corasick over 'a'..'z'. */
    static class AhoCorasick {
        final List<int[]> next = new ArrayList<>();
        final List<Integer> fail = new ArrayList<>();
        final List<List<Integer>> output = new ArrayList<>();
        final List<Integer> parent = new ArrayList<>();
        final List<Character> parentChar = new ArrayList<>();
        final String[] patterns;

        AhoCorasick(String[] patterns) {
            this.patterns = patterns;
            addNode();
            for (int index = 0; index < patterns.length; index++) {
                if (patterns[index].isEmpty()) continue;
                int node = 0;
                for (int i = 0; i < patterns[index].length(); i++) {
                    int c = patterns[index].charAt(i) - 'a';
                    if (next.get(node)[c] == -1) {
                        int created = addNode();
                        next.get(node)[c] = created;
                        parent.set(created, node);
                        parentChar.set(created, patterns[index].charAt(i));
                    }
                    node = next.get(node)[c];
                }
                output.get(node).add(index);
            }
            buildFailures();
        }

        int addNode() {
            next.add(new int[26]);
            Arrays.fill(next.get(next.size() - 1), -1);
            fail.add(0);
            output.add(new ArrayList<>());
            parent.add(-1);
            parentChar.add(' ');
            return next.size() - 1;
        }

        void buildFailures() {
            Deque<Integer> queue = new ArrayDeque<>();
            for (int c = 0; c < 26; c++) {
                int child = next.get(0)[c];
                if (child == -1) continue;
                fail.set(child, 0);
                queue.add(child);
            }
            while (!queue.isEmpty()) {
                int node = queue.poll();
                output.get(node).addAll(0, output.get(fail.get(node)));
                for (int c = 0; c < 26; c++) {
                    int child = next.get(node)[c];
                    if (child == -1) continue;
                    int candidate = fail.get(node);
                    while (candidate != 0 && next.get(candidate)[c] == -1) {
                        candidate = fail.get(candidate);
                    }
                    int fallback = next.get(candidate)[c];
                    fail.set(child, fallback == -1 || fallback == child ? 0 : fallback);
                    queue.add(child);
                }
            }
        }

        List<List<Integer>> search(String text) {
            List<List<Integer>> result = new ArrayList<>();
            for (int i = 0; i < patterns.length; i++) result.add(new ArrayList<>());
            int node = 0;
            for (int position = 0; position < text.length(); position++) {
                int c = text.charAt(position) - 'a';
                while (node != 0 && next.get(node)[c] == -1) node = fail.get(node);
                if (next.get(node)[c] != -1) node = next.get(node)[c];
                for (int index : output.get(node)) {
                    result.get(index).add(position - patterns[index].length() + 1);
                }
            }
            for (int i = 0; i < patterns.length; i++) {
                if (patterns[i].isEmpty()) {
                    for (int position = 0; position <= text.length(); position++) {
                        result.get(i).add(position);
                    }
                }
            }
            for (int i = 0; i < patterns.length; i++) {
                Collections.sort(result.get(i));
            }
            return result;
        }

        /** The trie string of a state (walking the parent pointers), used to verify failure links. */
        String trieString(int node) {
            StringBuilder builder = new StringBuilder();
            int current = node;
            while (current != 0) {
                builder.append(parentChar.get(current));
                current = parent.get(current);
            }
            return builder.reverse().toString();
        }
    }

    static List<List<Integer>> findNaive(String text, String[] patterns) {
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < patterns.length; i++) result.add(new ArrayList<>());
        for (int i = 0; i < patterns.length; i++) {
            String pattern = patterns[i];
            if (pattern.isEmpty()) {
                for (int position = 0; position <= text.length(); position++) {
                    result.get(i).add(position);
                }
                continue;
            }
            for (int position = 0; position + pattern.length() <= text.length(); position++) {
                if (text.startsWith(pattern, position)) result.get(i).add(position);
            }
        }
        return result;
    }

    static String lcgString(int n, int seed, int alphabet) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            builder.append((char) ('a' + seed % alphabet));
        }
        return builder.toString();
    }

    static void compareAll(String text, String[] patterns, String label) {
        Stats stats = new Stats();
        List<List<Integer>> karp = findKarpRabin(text, patterns, DEFAULT_BASE, DEFAULT_MOD, stats);
        List<List<Integer>> aho = new AhoCorasick(patterns).search(text);
        List<List<Integer>> naive = findNaive(text, patterns);
        for (int i = 0; i < patterns.length; i++) {
            check(karp.get(i).equals(naive.get(i)),
                    label + ": Karp-Rabin list for pattern " + i + " (" + patterns[i] + ")");
            check(aho.get(i).equals(naive.get(i)),
                    label + ": Aho-Corasick list for pattern " + i + " (" + patterns[i] + ")");
        }
    }

    public static void main(String[] args) {
        String[] aaa = {"a", "aa", "aaa"};
        compareAll("aaaa", aaa, "aaaa");
        List<List<Integer>> hand = findNaive("aaaa", aaa);
        check(hand.get(0).equals(Arrays.asList(0, 1, 2, 3)), "a occurs 4 times in aaaa");
        check(hand.get(1).equals(Arrays.asList(0, 1, 2)), "overlapping aa occurs 3 times");
        check(hand.get(2).equals(Arrays.asList(0, 1)), "aaa occurs twice");

        String[] abSet = {"ab", "ba", "aba"};
        compareAll("abababa", abSet, "abababa");
        Stats stats = new Stats();
        List<List<Integer>> abFound = findKarpRabin("abababa", abSet, DEFAULT_BASE, DEFAULT_MOD, stats);
        check(abFound.get(0).equals(Arrays.asList(0, 2, 4)), "ab at 0, 2, 4");
        check(abFound.get(1).equals(Arrays.asList(1, 3, 5)), "ba at 1, 3, 5");
        check(abFound.get(2).equals(Arrays.asList(0, 2, 4)), "aba at 0, 2, 4");
        check(stats.windows == 6 + 5,
                "one window pass per distinct length: (7 - 2 + 1) + (7 - 3 + 1) windows");

        String[] absent = {"zz", "abc"};
        compareAll("aabbcc", absent, "absent patterns");
        check(findNaive("aabbcc", absent).get(0).isEmpty(), "zz is absent");
        check(findNaive("aabbcc", absent).get(1).isEmpty(), "abc is absent");
        String[] tooLong = {"abcdefgh"};
        compareAll("abc", tooLong, "pattern longer than the text");
        check(findNaive("abc", tooLong).get(0).isEmpty(), "oversized pattern has no match");
        String[] duplicates = {"ab", "ab", "b"};
        compareAll("abab", duplicates, "duplicate patterns");
        List<List<Integer>> duplicated = findNaive("abab", duplicates);
        check(duplicated.get(0).equals(duplicated.get(1)), "duplicate patterns share occurrences");
        check(duplicated.get(2).equals(Arrays.asList(1, 3)), "b occurs at 1 and 3");
        String[] emptyPattern = {"", "ab"};
        compareAll("abab", emptyPattern, "empty pattern");
        check(findNaive("abab", emptyPattern).get(0).equals(Arrays.asList(0, 1, 2, 3, 4)),
                "the empty pattern matches at every position");

        Stats weakStats = new Stats();
        String weakText = lcgString(200, 12345, 2);
        String[] weakPatterns = new String[10];
        for (int i = 0; i < weakPatterns.length; i++) {
            weakPatterns[i] = lcgString(1 + i % 4, 500 + i * 31, 2);
        }
        List<List<Integer>> weakFound = findKarpRabin(weakText, weakPatterns, 3, 7, weakStats);
        List<List<Integer>> weakNaive = findNaive(weakText, weakPatterns);
        check(weakStats.rejections > 0, "the weak modulus produces rejected hash hits");
        for (int i = 0; i < weakPatterns.length; i++) {
            check(weakFound.get(i).equals(weakNaive.get(i)),
                    "weak modulus stays exact for pattern " + weakPatterns[i]);
        }
        compareAll(weakText, weakPatterns, "weak modulus cross-check");

        int seed = 99;
        for (int trial = 0; trial < 10; trial++) {
            int n = 20 + trial * 23;
            String text = lcgString(n, 4000 + trial * 101, 2 + trial % 2);
            String[] patterns = new String[6 + trial % 4];
            for (int i = 0; i < patterns.length; i++) {
                patterns[i] = lcgString(1 + (i + trial) % 5, 900 + trial * 17 + i * 7, 2);
            }
            compareAll(text, patterns, "LCG trial " + trial);
            Stats trialStats = new Stats();
            findKarpRabin(text, patterns, DEFAULT_BASE, DEFAULT_MOD, trialStats);
            Set<Integer> lengths = new TreeSet<>();
            for (String pattern : patterns) {
                if (!pattern.isEmpty()) lengths.add(pattern.length());
            }
            long expectedWindows = 0;
            for (int length : lengths) {
                if (length <= n) expectedWindows += n - length + 1;
            }
            check(trialStats.windows == expectedWindows,
                    "window count equals the sum of n - L + 1 per bucket, trial " + trial);
        }

        String rollingText = lcgString(50, 7777, 4);
        for (int length = 1; length <= 4; length++) {
            long power = 1;
            for (int i = 0; i + 1 < length; i++) power = power * DEFAULT_BASE % DEFAULT_MOD;
            long current = hashOf(rollingText.substring(0, length), DEFAULT_BASE, DEFAULT_MOD);
            for (int position = 0; position + length <= rollingText.length(); position++) {
                if (position > 0) {
                    long out = charValue(rollingText.charAt(position - 1)) * power % DEFAULT_MOD;
                    current = ((current - out + DEFAULT_MOD) % DEFAULT_MOD) * DEFAULT_BASE % DEFAULT_MOD;
                    current = (current + charValue(rollingText.charAt(position + length - 1)))
                            % DEFAULT_MOD;
                }
                check(current == hashOf(rollingText.substring(position, position + length),
                        DEFAULT_BASE, DEFAULT_MOD),
                        "rolling hash equals the direct hash at position " + position);
            }
        }

        String[] automatonPatterns = {"he", "she", "his", "hers"};
        AhoCorasick automaton = new AhoCorasick(automatonPatterns);
        int totalLength = 0;
        for (String pattern : automatonPatterns) totalLength += pattern.length();
        check(automaton.next.size() <= totalLength + 1,
                "the automaton has at most total-pattern-length + 1 states");
        for (int node = 1; node < automaton.next.size(); node++) {
            String trie = automaton.trieString(node);
            check(!trie.isEmpty(), "every non-root state has a trie string");
            int failure = automaton.fail.get(node);
            String failureString = failure == 0 ? "" : automaton.trieString(failure);
            check(trie.endsWith(failureString), "the failure link is a suffix of the state's string");
            check(failureString.length() < trie.length(), "the failure link is a proper suffix");
            for (int other = 1; other < automaton.next.size(); other++) {
                String candidate = automaton.trieString(other);
                if (trie.endsWith(candidate) && candidate.length() < trie.length()) {
                    check(candidate.length() <= failureString.length(),
                            "the failure link is the longest such suffix for " + trie);
                }
            }
        }
        List<List<Integer>> automatonFound = automaton.search("ushers");
        check(automatonFound.get(0).equals(Arrays.asList(2)), "he occurs inside ushers at 2");
        check(automatonFound.get(1).equals(Arrays.asList(1)), "she occurs at 1");
        check(automatonFound.get(2).isEmpty(), "his is absent from ushers");
        check(automatonFound.get(3).equals(Arrays.asList(2)), "hers occurs at 2");
        check(automaton.search("ushers").get(0).equals(Arrays.asList(2))
                && automaton.search("ushers").get(2).isEmpty(),
                "ushers contains he and she but not his");
        check(automaton.search("ushers").get(1).equals(automatonFound.get(1)),
                "deterministic re-run of the automaton");

        String bigText = lcgString(3000, 20250916, 3);
        String[] bigPatterns = new String[40];
        for (int i = 0; i < bigPatterns.length; i++) {
            bigPatterns[i] = lcgString(1 + i % 6, 60000 + i * 137, 3);
        }
        Stats bigStats = new Stats();
        List<List<Integer>> bigKarp = findKarpRabin(bigText, bigPatterns, DEFAULT_BASE, DEFAULT_MOD,
                bigStats);
        List<List<Integer>> bigAho = new AhoCorasick(bigPatterns).search(bigText);
        List<List<Integer>> bigNaive = findNaive(bigText, bigPatterns);
        long totalMatches = 0;
        for (int i = 0; i < bigPatterns.length; i++) {
            check(bigKarp.get(i).equals(bigNaive.get(i)), "3000-character text, pattern " + i);
            check(bigAho.get(i).equals(bigNaive.get(i)), "automaton on the 3000-character text");
            totalMatches += bigNaive.get(i).size();
            if (bigPatterns[i].length() <= 2) {
                check(bigNaive.get(i).size() > 0,
                        "short pattern " + bigPatterns[i] + " must occur in the 3000-character text");
            }
        }
        check(totalMatches > 0, "the pattern set matches the text");
        check(bigStats.candidateChecks == bigStats.confirmations + bigStats.rejections,
                "every confirmed or rejected candidate was counted");
        check(bigStats.confirmations > 0 && bigStats.hashHits > 0,
                "the 3000-character workload confirms matches through hash hits");
        check(bigStats.hashHits <= bigStats.windows, "at most one table lookup per window");
        check(bigStats.windows > 3000, "one rolling pass per distinct pattern length");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
