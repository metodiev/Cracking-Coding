/** Task 197 - Implement checksum-based file difference (simplified rsync idea). (Tier: Beginner, CTCI Beyond) */
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

public class Solution {
    public static void main(String[] args) {
        byte[] a = new byte[40];
        for (int i = 0; i < a.length; i++) a[i] = (byte) i;

        check(findBlockMatches(a, a, 8).equals(Map.of(0, 0, 1, 8, 2, 16, 3, 24, 4, 32)),
                "identical files match at their own offsets");

        byte[] changed = a.clone();
        changed[10] = (byte) (changed[10] ^ 0xFF);                 // one byte inside block 1
        check(findBlockMatches(a, changed, 8).equals(Map.of(0, 0, 2, 16, 3, 24, 4, 32)),
                "one changed byte drops that block only");

        byte[] other = new byte[64];
        Arrays.fill(other, (byte) 9);
        check(findBlockMatches(a, other, 8).isEmpty(), "completely different content");

        byte[] shifted = new byte[6 + a.length];
        System.arraycopy("HEADER".getBytes(StandardCharsets.UTF_8), 0, shifted, 0, 6);
        System.arraycopy(a, 0, shifted, 6, a.length);
        check(findBlockMatches(a, shifted, 8).equals(Map.of(0, 6, 1, 14, 2, 22, 3, 30, 4, 38)),
                "shifted match positions");

        check(findBlockMatches("ABABABAB".getBytes(StandardCharsets.UTF_8),
                "ZZABABABAB".getBytes(StandardCharsets.UTF_8), 4).equals(Map.of(0, 2, 1, 2)),
                "duplicate blocks share the first occurrence");
        check(findBlockMatches("abc".getBytes(StandardCharsets.UTF_8),
                "xxabcxx".getBytes(StandardCharsets.UTF_8), 8).equals(Map.of(0, 2)),
                "partial tail block");
        check(findBlockMatches(new byte[0], a, 8).isEmpty(), "empty file A");
        check(weakChecksum("AB".getBytes(StandardCharsets.UTF_8), 0, 2)
                != weakChecksum("BA".getBytes(StandardCharsets.UTF_8), 0, 2), "weak checksum is order-sensitive");
        checkThrows(new byte[]{1}, 0);
        System.out.println("PASS");
    }

    static Map<Integer, Integer> findBlockMatches(byte[] a, byte[] b, int blockSize) {
        if (blockSize <= 0) throw new IllegalArgumentException("block_size must be positive");
        List<byte[]> blocks = new ArrayList<>();
        for (int i = 0; i < a.length; i += blockSize) {
            int length = Math.min(blockSize, a.length - i);
            blocks.add(Arrays.copyOfRange(a, i, i + length));
        }

        Map<Long, List<Integer>> indexByWeak = new HashMap<>();
        for (int index = 0; index < blocks.size(); index++) {
            byte[] block = blocks.get(index);
            indexByWeak.computeIfAbsent(weakChecksum(block, 0, block.length), key -> new ArrayList<>()).add(index);
        }

        Map<Integer, Integer> matches = new TreeMap<>();
        for (int start = 0; start + blockSize <= b.length; start++) {
            List<Integer> candidates = indexByWeak.get(weakChecksum(b, start, blockSize));
            if (candidates == null) continue;
            for (int index : candidates) {
                if (matches.containsKey(index) || blocks.get(index).length != blockSize) continue;
                if (strongHash(b, start, blockSize).equals(strongHash(blocks.get(index), 0, blockSize))) {
                    matches.put(index, start);                     // weak -> strong confirmation
                }
            }
        }

        if (!blocks.isEmpty()) {
            int tailIndex = blocks.size() - 1;
            byte[] tail = blocks.get(tailIndex);
            if (tail.length < blockSize && !matches.containsKey(tailIndex)) {
                int position = indexOf(b, tail);                   // simplified tail lookup
                if (position >= 0) matches.put(tailIndex, position);
            }
        }
        return matches;
    }

    static long weakChecksum(byte[] data, int offset, int length) {
        long a = 0;
        long b = 0;
        for (int i = 0; i < length; i++) {
            int value = data[offset + i] & 0xFF;
            a = (a + value) & 0xFFFF;
            b = (b + (long) (length - i) * value) & 0xFFFF;
        }
        return (a << 16) | b;
    }

    static String strongHash(byte[] data, int offset, int length) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(data, offset, length);
            StringBuilder hex = new StringBuilder();
            for (byte value : digest.digest()) hex.append(String.format("%02x", value));
            return hex.toString();
        } catch (NoSuchAlgorithmException exc) {
            throw new IllegalStateException("SHA-256 unavailable", exc);
        }
    }

    static int indexOf(byte[] haystack, byte[] needle) {
        outer:
        for (int i = 0; i + needle.length <= haystack.length; i++) {
            for (int j = 0; j < needle.length; j++) {
                if (haystack[i + j] != needle[j]) continue outer;
            }
            return i;
        }
        return -1;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static void checkThrows(byte[] data, int blockSize) {
        try {
            findBlockMatches(data, data, blockSize);
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected error for block_size " + blockSize);
    }
}
