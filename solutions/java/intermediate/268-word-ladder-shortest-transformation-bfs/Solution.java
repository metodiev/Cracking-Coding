/** Task 268 - Word ladder shortest transformation (BFS). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /**
     * BFS over the implicit one-letter-difference graph with a wildcard bucket index
     * (h*t -> [hot, hit, ...]) built once, so all neighbours are found in O(L) instead
     * of scanning the whole list. Returns the sequence LENGTH in words (begin and end
     * included), or 0 when no sequence exists (documented, as in LeetCode 127).
     * end not in the list -> 0; begin == end -> 1; mismatched lengths -> 0; duplicates
     * collapse to one word.
     */
    static int ladderLength(String begin, String end, List<String> wordList) {
        Set<String> words = new HashSet<>(wordList);
        if (!words.contains(end) || begin.length() != end.length()) return 0;
        if (begin.equals(end)) return 1;
        int length = begin.length();
        Map<String, List<String>> buckets = new HashMap<>();
        for (String word : words) {
            if (word.length() != length) continue;
            for (int i = 0; i < length; i++) {
                String pattern = word.substring(0, i) + "*" + word.substring(i + 1);
                buckets.computeIfAbsent(pattern, k -> new ArrayList<>()).add(word);
            }
        }
        Deque<String> queue = new ArrayDeque<>();
        queue.add(begin);
        Set<String> seen = new HashSet<>();
        seen.add(begin);
        int steps = 1;
        while (!queue.isEmpty()) {
            for (int size = queue.size(); size > 0; size--) {
                String word = queue.poll();
                for (int i = 0; i < length; i++) {
                    String pattern = word.substring(0, i) + "*" + word.substring(i + 1);
                    List<String> neighbours = buckets.get(pattern);
                    if (neighbours == null) continue;
                    for (String nxt : neighbours) {
                        if (nxt.equals(end)) return steps + 1;
                        if (seen.add(nxt)) queue.add(nxt);
                    }
                }
            }
            steps++;
        }
        return 0;
    }

    /** Naive BFS that scans every word for a one-letter difference; cross-check only. */
    static int ladderLengthOracle(String begin, String end, List<String> wordList) {
        Set<String> words = new HashSet<>(wordList);
        if (!words.contains(end) || begin.length() != end.length()) return 0;
        if (begin.equals(end)) return 1;
        Deque<String> queue = new ArrayDeque<>();
        Deque<Integer> dists = new ArrayDeque<>();
        Set<String> seen = new HashSet<>();
        queue.add(begin);
        dists.add(1);
        seen.add(begin);
        while (!queue.isEmpty()) {
            String word = queue.poll();
            int dist = dists.poll();
            for (String candidate : words) {
                if (seen.contains(candidate) || candidate.length() != word.length()) continue;
                int diff = 0;
                for (int i = 0; i < word.length() && diff <= 1; i++) if (word.charAt(i) != candidate.charAt(i)) diff++;
                if (diff == 1) {
                    if (candidate.equals(end)) return dist + 1;
                    seen.add(candidate);
                    queue.add(candidate);
                    dists.add(dist + 1);
                }
            }
        }
        return 0;
    }

    public static void main(String[] args) {
        check(ladderLength("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log", "cog")) == 5,
                "hit -> cog = 5");
        check(ladderLength("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log")) == 0, "end not in list");
        check(ladderLength("a", "c", Arrays.asList("a", "b", "c")) == 2, "one-letter words");
        check(ladderLength("hit", "hot", Arrays.asList("hot", "dot", "dog")) == 2, "one step");
        check(ladderLength("hot", "hot", Arrays.asList("hot")) == 1, "same word -> 1");
        check(ladderLength("ab", "abc", Arrays.asList("abc")) == 0, "mismatched lengths");
        check(ladderLength("abc", "ab", Arrays.asList("ab")) == 0, "mismatched lengths reversed");
        check(ladderLength("lost", "cost", Arrays.asList("most", "fost", "cost", "cast")) == 2, "lost -> cost direct");
        check(ladderLength("ab", "cd", Arrays.asList("ac", "bd")) == 0, "unreachable");
        check(ladderLength("hit", "cog", Arrays.asList("hot", "hot", "dot", "dog", "lot", "log", "cog")) == 5,
                "duplicates collapse");
        check(ladderLength("hot", "hot", Arrays.asList("dog")) == 0, "end not in list wins over equality");
        check(ladderLength("aaaa", "aaab", Arrays.asList("aaaa", "aaab")) == 2, "four-letter words");

        Random rng = new Random(268);
        for (int test = 0; test < 300; test++) {        // cross-check vs the naive-scan BFS
            Set<String> pool = new TreeSet<>();
            int poolSize = 1 + rng.nextInt(6);
            for (int i = 0; i < poolSize * 3; i++)
                pool.add("" + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3)));
            List<String> wordList = new ArrayList<>(pool);
            if (rng.nextDouble() < 0.5 && !wordList.isEmpty()) wordList.add(wordList.get(0));
            String begin = "" + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3));
            String end = !wordList.isEmpty() && rng.nextDouble() < 0.5
                    ? wordList.get(rng.nextInt(wordList.size()))
                    : "" + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3));
            check(ladderLength(begin, end, wordList) == ladderLengthOracle(begin, end, wordList),
                    "random agreement begin=" + begin + " end=" + end);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
