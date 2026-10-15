/** Task 391 - Implement adaptive Huffman (outline, conceptual). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class Node {
    int weight;
    int seq;
    String ch;
    Node left, right;

    Node(int weight, int seq, String ch) {
        this.weight = weight;
        this.seq = seq;
        this.ch = ch;
    }
}

public class Solution {
    /** Builds the Huffman tree; ties broken by (weight, creation order). */
    static Node buildTree(Map<String, Integer> freq, List<String> alphabet) {
        List<Node> work = new ArrayList<>();
        int seq = 0;
        for (String ch : alphabet) {
            work.add(new Node(freq.get(ch), seq++, ch));
        }
        while (work.size() > 1) {
            work.sort(Comparator.<Node>comparingInt(n -> n.weight).thenComparingInt(n -> n.seq));
            Node x = work.remove(0);
            Node y = work.remove(0);
            Node parent = new Node(x.weight + y.weight, seq++, null);
            parent.left = x;
            parent.right = y;
            work.add(parent);
        }
        return work.get(0);
    }

    static void collectCodes(Node root, String prefix, Map<String, String> out) {
        if (root.ch != null) {
            out.put(root.ch, prefix);
            return;
        }
        collectCodes(root.left, prefix + "0", out);
        collectCodes(root.right, prefix + "1", out);
    }

    static String adaptiveEncode(String text, String alphabet) {
        Map<String, Integer> freq = new HashMap<>();
        List<String> symbols = new ArrayList<>();
        for (int i = 0; i < alphabet.length(); i++) {
            String ch = alphabet.substring(i, i + 1);
            freq.put(ch, 1);
            symbols.add(ch);
        }
        StringBuilder bits = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            String ch = text.substring(i, i + 1);
            if (!freq.containsKey(ch)) throw new IllegalArgumentException("symbol outside the alphabet: " + ch);
            Map<String, String> codes = new HashMap<>();
            collectCodes(buildTree(freq, symbols), "", codes);
            bits.append(codes.get(ch));
            freq.put(ch, freq.get(ch) + 1);
        }
        return bits.toString();
    }

    static String adaptiveDecode(String bits, String alphabet) {
        Map<String, Integer> freq = new HashMap<>();
        List<String> symbols = new ArrayList<>();
        for (int i = 0; i < alphabet.length(); i++) {
            String ch = alphabet.substring(i, i + 1);
            freq.put(ch, 1);
            symbols.add(ch);
        }
        StringBuilder out = new StringBuilder();
        int i = 0;
        while (i < bits.length()) {
            Node node = buildTree(freq, symbols);
            while (node.ch == null) {
                node = bits.charAt(i) == '0' ? node.left : node.right;
                i++;
            }
            out.append(node.ch);
            freq.put(node.ch, freq.get(node.ch) + 1);
        }
        return out.toString();
    }

    private static int seed = 6060;

    private static List<String> lcgStrings(String alphabet, int length, int count) {
        List<String> strings = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < length; i++) {
                seed = (seed * 75 + 74) % 65537;
                sb.append(alphabet.charAt(seed % alphabet.length()));
            }
            strings.add(sb.toString());
        }
        return strings;
    }

    public static void main(String[] args) {
        check(adaptiveEncode("a", "abc").equals("10"), "first symbol code");
        check(adaptiveEncode("aaabbc", "abc").equals("1001000110"), "hand-derived stream");
        check(adaptiveEncode("aaa", "abc").equals("1001"), "three repeats");
        check(adaptiveEncode("", "abc").isEmpty(), "empty input");
        check(adaptiveDecode("1001000110", "abc").equals("aaabbc"), "decode the hand-derived stream");

        List<String> samples = new ArrayList<>(Arrays.asList("a", "abc", "aaabbc", "abababab", "cccbaaa"));
        samples.addAll(lcgStrings("abc", 24, 40));
        for (String s : samples) {
            check(adaptiveDecode(adaptiveEncode(s, "abc"), "abc").equals(s), "round trip " + s);
        }

        Map<String, Integer> finalFreq = new HashMap<>();
        finalFreq.put("a", 41);
        finalFreq.put("b", 2);
        finalFreq.put("c", 2);
        Map<String, String> codes = new HashMap<>();
        collectCodes(buildTree(finalFreq, Arrays.asList("a", "b", "c")), "", codes);
        check(codes.get("a").equals("1") && codes.get("b").length() == 2 && codes.get("c").length() == 2,
              "frequent symbol gets the short code");

        boolean threw = false;
        try {
            adaptiveEncode("d", "abc");
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "out-of-alphabet symbol rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
