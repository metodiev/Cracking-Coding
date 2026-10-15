/** Task 163 - Implement basic encryption/decryption with XOR. (Tier: Beginner, CTCI Ch 5) */
import java.util.*;

public class Solution {
    static byte[] xorBytes(byte[] data, byte[] key) {
        if (key.length == 0) throw new IllegalArgumentException("key must not be empty");
        byte[] out = new byte[data.length];
        for (int i = 0; i < data.length; i++) out[i] = (byte) (data[i] ^ key[i % key.length]);
        return out;
    }

    static String xorText(String text, String key) {
        byte[] cipher = xorBytes(text.getBytes(java.nio.charset.StandardCharsets.UTF_8),
                key.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        StringBuilder hex = new StringBuilder();
        for (byte b : cipher) hex.append(String.format("%02x", b));
        return hex.toString();
    }

    static String decryptText(String hexCipher, String key) {
        byte[] cipher = new byte[hexCipher.length() / 2];
        for (int i = 0; i < cipher.length; i++)
            cipher[i] = (byte) Integer.parseInt(hexCipher.substring(2 * i, 2 * i + 2), 16);
        byte[] plain = xorBytes(cipher, key.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        return new String(plain, java.nio.charset.StandardCharsets.UTF_8);
    }

    public static void main(String[] args) {
        String message = "attack at dawn";
        String key = "s3cret";
        String cipher = xorText(message, key);
        check(!cipher.equals(message), "cipher differs from plaintext");
        check(cipher.matches("[0-9a-f]+"), "hex output");
        check(decryptText(cipher, key).equals(message), "round trip");
        check(xorText("", key).isEmpty() && decryptText("", key).isEmpty(), "empty input");

        byte[] raw = {0, (byte) 0xFF, 0x7F};
        check(Arrays.equals(xorBytes(xorBytes(raw, new byte[]{'k'}), new byte[]{'k'}), raw), "self-inverse");
        check(!decryptText(cipher, "other").equals(message), "wrong key is garbage");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
