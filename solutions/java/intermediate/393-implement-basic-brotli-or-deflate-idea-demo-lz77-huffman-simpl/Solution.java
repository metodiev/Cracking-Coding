/** Task 393 - Implement a basic DEFLATE/Brotli idea demo (LZ77 + Huffman, simplified). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class HuffNode {
    int weight, seq;
    Integer byteValue;
    HuffNode left, right;
    HuffNode(int weight, int seq, Integer byteValue) {
        this.weight = weight;
        this.seq = seq;
        this.byteValue = byteValue;
    }
}

public class Solution {
    /** Returns tokens: {0, byte, 0} literals and {1, length, distance} matches. */
    static List<int[]> lz77Tokens(String text, int window) {
        int n = text.length();
        List<int[]> tokens = new ArrayList<>();
        int i = 0;
        while (i < n) {
            int bestLen = 0, bestDist = 0;
            for (int dist = 1; dist <= Math.min(window, i); dist++) {
                int length = 0;
                while (i + length < n
                        && text.charAt(i + length) == text.charAt(i - dist + length)
                        && length < 255) {
                    length++;
                }
                if (length > bestLen) {
                    bestLen = length;
                    bestDist = dist;
                }
            }
            if (bestLen >= 3) {
                tokens.add(new int[]{1, bestLen, bestDist});
                i += bestLen;
            } else {
                tokens.add(new int[]{0, text.charAt(i), 0});
                i++;
            }
        }
        return tokens;
    }

    static String lz77Decompress(List<int[]> tokens) {
        List<Integer> out = new ArrayList<>();
        for (int[] token : tokens) {
            if (token[0] == 0) {
                out.add(token[1]);
            } else {
                for (int k = 0; k < token[1]; k++) out.add(out.get(out.size() - token[2]));
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int byteValue : out) sb.append((char) byteValue);
        return sb.toString();
    }

    static List<Integer> tokensToBytes(List<int[]> tokens) {
        List<Integer> stream = new ArrayList<>();
        for (int[] token : tokens) {
            stream.add(token[0]);
            stream.add(token[1]);
            if (token[0] == 1) stream.add(token[2]);
        }
        return stream;
    }

    static List<int[]> bytesToTokens(List<Integer> stream) {
        List<int[]> tokens = new ArrayList<>();
        int i = 0;
        while (i < stream.size()) {
            if (stream.get(i) == 0) {
                tokens.add(new int[]{0, stream.get(i + 1), 0});
                i += 2;
            } else {
                tokens.add(new int[]{1, stream.get(i + 1), stream.get(i + 2)});
                i += 3;
            }
        }
        return tokens;
    }

    static HuffNode buildHuffman(Map<Integer, Integer> freq) {
        List<HuffNode> work = new ArrayList<>();
        int seq = 0;
        List<Integer> keys = new ArrayList<>(freq.keySet());
        Collections.sort(keys);
        for (int byteValue : keys) work.add(new HuffNode(freq.get(byteValue), seq++, byteValue));
        while (work.size() > 1) {
            work.sort(Comparator.<HuffNode>comparingInt(n -> n.weight).thenComparingInt(n -> n.seq));
            HuffNode x = work.remove(0);
            HuffNode y = work.remove(0);
            HuffNode parent = new HuffNode(x.weight + y.weight, seq++, null);
            parent.left = x;
            parent.right = y;
            work.add(parent);
        }
        return work.get(0);
    }

    static void collectCodes(HuffNode root, String prefix, Map<Integer, String> out) {
        if (root.byteValue != null) {
            out.put(root.byteValue, prefix);
            return;
        }
        collectCodes(root.left, prefix + "0", out);
        collectCodes(root.right, prefix + "1", out);
    }

    /** Returns {bits, code-map} for the given data. */
    static Object[] huffmanEncode(List<Integer> data) {
        if (data.isEmpty()) return new Object[]{"", new HashMap<Integer, String>()};
        Map<Integer, Integer> freq = new TreeMap<>();
        for (int byteValue : data) freq.merge(byteValue, 1, Integer::sum);
        Map<Integer, String> codes = new TreeMap<>();
        collectCodes(buildHuffman(freq), "", codes);
        StringBuilder bits = new StringBuilder();
        for (int byteValue : data) bits.append(codes.get(byteValue));
        return new Object[]{bits.toString(), codes};
    }

    @SuppressWarnings("unchecked")
    static List<Integer> huffmanDecode(String bits, Map<Integer, String> codes) {
        Map<String, Integer> reverse = new HashMap<>();
        for (Map.Entry<Integer, String> entry : codes.entrySet()) reverse.put(entry.getValue(), entry.getKey());
        List<Integer> out = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < bits.length(); i++) {
            current.append(bits.charAt(i));
            Integer byteValue = reverse.get(current.toString());
            if (byteValue != null) {
                out.add(byteValue);
                current.setLength(0);
            }
        }
        check(current.length() == 0, "truncated code stream");
        return out;
    }

    static boolean isPrefixFree(Map<Integer, String> codes) {
        List<String> table = new ArrayList<>(codes.values());
        for (int i = 0; i < table.size(); i++) {
            for (int j = 0; j < table.size(); j++) {
                if (i != j && table.get(j).startsWith(table.get(i))) return false;
            }
        }
        return true;
    }

    private static int seed = 3131;

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

    static String joinTokens(List<int[]> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int[] token : tokens) {
            if (sb.length() > 0) sb.append(';');
            sb.append(token[0]).append(',').append(token[1]).append(',').append(token[2]);
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(joinTokens(lz77Tokens("abcabcabc", 8)).equals("0,97,0;0,98,0;0,99,0;1,6,3"), "hand-derived tokens");
        check(joinTokens(lz77Tokens("aaaaaaaa", 8)).equals("0,97,0;1,7,1"), "run of a");
        StringBuilder abc10 = new StringBuilder();
        for (int i = 0; i < 10; i++) abc10.append("abc");
        check(joinTokens(lz77Tokens(abc10.toString(), 8)).equals("0,97,0;0,98,0;0,99,0;1,27,3"), "abc x10");
        check(lz77Tokens("", 8).isEmpty(), "empty input");
        check(joinTokens(lz77Tokens("ab", 8)).equals("0,97,0;0,98,0"), "two literals");

        check(lz77Decompress(Arrays.asList(new int[]{0,97,0}, new int[]{0,98,0}, new int[]{0,99,0}, new int[]{1,6,3})).equals("abcabcabc"), "decompress abc");
        check(lz77Decompress(Arrays.asList(new int[]{0,97,0}, new int[]{1,7,1})).equals("aaaaaaaa"), "decompress a-run");

        check(tokensToBytes(Arrays.asList(new int[]{0,97,0}, new int[]{1,6,3})).toString().equals("[0, 97, 1, 6, 3]"), "serialize");
        check(joinTokens(bytesToTokens(Arrays.asList(0, 97, 1, 6, 3))).equals("0,97,0;1,6,3"), "deserialize");

        List<String> samples = new ArrayList<>(Arrays.asList("abcabcabc", "aaaaaaaa", abc10.toString(), "", "ab", "aabaabaa"));
        samples.addAll(lcgStrings("abc", 18, 30));
        for (String s : samples) {
            check(lz77Decompress(lz77Tokens(s, 8)).equals(s), "lz77 round trip " + s);
            check(joinTokens(bytesToTokens(tokensToBytes(lz77Tokens(s, 8)))).equals(joinTokens(lz77Tokens(s, 8))), "token round trip " + s);
        }

        for (String s : samples) {
            Object[] encoded = huffmanEncode(tokensToBytes(lz77Tokens(s, 8)));
            String bits = (String) encoded[0];
            @SuppressWarnings("unchecked")
            Map<Integer, String> codes = (Map<Integer, String>) encoded[1];
            check(isPrefixFree(codes), "prefix-free " + s);
            int weighted = 0;
            for (int byteValue : tokensToBytes(lz77Tokens(s, 8))) weighted += codes.get(byteValue).length();
            check(bits.length() == weighted, "weighted length " + s);
            check(lz77Decompress(bytesToTokens(huffmanDecode(bits, codes))).equals(s), "pipeline round trip " + s);
        }

        for (String text : new String[]{"abcabcabc", "aaaaaaaa", abc10.toString()}) {
            Object[] encoded = huffmanEncode(tokensToBytes(lz77Tokens(text, 8)));
            check(((String) encoded[0]).length() < 8 * text.length(), "compressed " + text);
        }

        check(joinTokens(lz77Tokens("abcabcabc", 2)).equals(
                "0,97,0;0,98,0;0,99,0;0,97,0;0,98,0;0,99,0;0,97,0;0,98,0;0,99,0"), "window too small for a match");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
