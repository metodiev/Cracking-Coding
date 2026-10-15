/** Task 314 - Aho-Corasick multi-pattern matching (outline). (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

class AhoCorasick {
    private final List<Map<Character, Integer>> gotoMap = new ArrayList<>();
    private final List<Integer> fail = new ArrayList<>();
    private final List<Set<String>> output = new ArrayList<>();
    private final List<String> patterns = new ArrayList<>();

    /** Patterns are assumed non-empty. */
    AhoCorasick(List<String> patterns) {
        this.patterns.addAll(patterns);
        gotoMap.add(new HashMap<>());
        fail.add(0);
        output.add(new HashSet<>());
        for (String pattern : patterns) {
            int node = 0;
            for (int i = 0; i < pattern.length(); i++) {
                char ch = pattern.charAt(i);
                Integer next = gotoMap.get(node).get(ch);
                if (next == null) {
                    next = gotoMap.size();
                    gotoMap.get(node).put(ch, next);
                    gotoMap.add(new HashMap<>());
                    fail.add(0);
                    output.add(new HashSet<>());
                }
                node = next;
            }
            output.get(node).add(pattern);
        }
        // BFS over the trie: depth-1 states already fail to the root
        ArrayDeque<Integer> queue = new ArrayDeque<>(gotoMap.get(0).values());
        while (!queue.isEmpty()) {
            int node = queue.poll();
            output.get(node).addAll(output.get(fail.get(node)));
            for (Map.Entry<Character, Integer> entry : gotoMap.get(node).entrySet()) {
                char ch = entry.getKey();
                int child = entry.getValue();
                int f = fail.get(node);
                while (f != 0 && !gotoMap.get(f).containsKey(ch)) f = fail.get(f);
                fail.set(child, gotoMap.get(f).getOrDefault(ch, 0));
                queue.add(child);
            }
        }
    }

    /** pattern -> ascending list of start indices (overlapping matches included). */
    Map<String, List<Integer>> search(String text) {
        Map<String, List<Integer>> found = new LinkedHashMap<>();
        for (String pattern : patterns) found.put(pattern, new ArrayList<>());
        int node = 0;
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            while (node != 0 && !gotoMap.get(node).containsKey(ch)) node = fail.get(node);
            node = gotoMap.get(node).getOrDefault(ch, 0);
            for (String pattern : output.get(node)) found.get(pattern).add(i - pattern.length() + 1);
        }
        return found;
    }
}

public class Solution {
    /** Independent oracle: scan each pattern on its own. */
    static Map<String, List<Integer>> naiveSearchAll(String text, List<String> patterns) {
        Map<String, List<Integer>> found = new LinkedHashMap<>();
        for (String pattern : patterns) {
            List<Integer> hits = new ArrayList<>();
            for (int i = 0; i + pattern.length() <= text.length(); i++) {
                if (text.startsWith(pattern, i)) hits.add(i);
            }
            found.put(pattern, hits);
        }
        return found;
    }

    static List<String> allWords(String alphabet, int maxLength) {
        List<String> words = new ArrayList<>();
        List<String> frontier = new ArrayList<>();
        frontier.add("");
        for (int depth = 0; depth < maxLength; depth++) {
            List<String> next = new ArrayList<>();
            for (String word : frontier)
                for (int i = 0; i < alphabet.length(); i++) next.add(word + alphabet.charAt(i));
            words.addAll(next);
            frontier = next;
        }
        return words;
    }

    static String format(Map<String, List<Integer>> result) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, List<Integer>> entry : result.entrySet()) {
            if (sb.length() > 0) sb.append(';');
            sb.append(entry.getKey()).append('=').append(entry.getValue());
        }
        return sb.toString();
    }

    private static int seed = 3140;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // Classic example: hand-checked against "ushers" (u0 s1 h2 e3 r4 s5)
        List<String> classic = Arrays.asList("he", "she", "his", "hers");
        AhoCorasick automaton = new AhoCorasick(classic);
        check(format(automaton.search("ushers")).equals("he=[2];she=[1];his=[];hers=[2]"), "ushers");
        check(format(automaton.search("ushers")).equals(format(naiveSearchAll("ushers", classic))), "ushers oracle");

        // Overlapping matches are all reported
        check(format(new AhoCorasick(Arrays.asList("aa", "aaa")).search("aaaa")).equals("aa=[0, 1, 2];aaa=[0, 1]"), "overlaps");
        AhoCorasick single = new AhoCorasick(Arrays.asList("a", "ab"));
        check(format(single.search("ab")).equals("a=[0];ab=[0]"), "single");
        check(format(single.search("")).equals("a=[];ab=[]"), "empty text");

        for (String alphabet : new String[]{"ab", "abc"}) {
            List<List<String>> patternSets = new ArrayList<>();
            patternSets.add(allWords(alphabet, 3));
            List<String> repeated = new ArrayList<>();
            for (int k = 1; k <= 3; k++) repeated.add(new String(new char[k]).replace('\0', alphabet.charAt(0)));
            patternSets.add(repeated);
            List<String> twos = new ArrayList<>();
            for (String word : allWords(alphabet, 2)) if (word.length() == 2) twos.add(word);
            patternSets.add(twos);

            List<String> texts = new ArrayList<>(Arrays.asList("", alphabet, alphabet + alphabet + alphabet + alphabet,
                    new String(new char[6]).replace('\0', alphabet.charAt(0))));
            for (int i = 0; i < 20; i++) texts.add(lcgString(18, alphabet));

            for (List<String> patterns : patternSets) {
                AhoCorasick ac = new AhoCorasick(patterns);
                for (String text : texts) {
                    Map<String, List<Integer>> result = ac.search(text);
                    check(format(result).equals(format(naiveSearchAll(text, patterns))),
                            "cross-check " + patterns + " / " + text);
                    for (Map.Entry<String, List<Integer>> entry : result.entrySet()) {
                        for (int start : entry.getValue()) {                    // every report is a real match
                            check(text.startsWith(entry.getKey(), start), "report " + text + " " + entry.getKey());
                        }
                    }
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
