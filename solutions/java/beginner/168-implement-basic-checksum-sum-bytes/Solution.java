/** Task 168 - Implement basic checksum (sum bytes). (Tier: Beginner, CTCI Ch 5) */
import java.nio.charset.StandardCharsets;

public class Solution {
    static int checksum8(byte[] data) {
        int sum = 0;
        for (byte b : data) sum += (b & 0xFF);
        return sum % 256;
    }

    static int checksum8(String text) {
        return checksum8(text.getBytes(StandardCharsets.UTF_8));
    }

    static int checksum16(byte[] data) {
        int sum = 0;
        for (byte b : data) sum += (b & 0xFF);
        return sum % 65536;
    }

    static int checksum16(String text) {
        return checksum16(text.getBytes(StandardCharsets.UTF_8));
    }

    public static void main(String[] args) {
        check(checksum8("") == 0, "empty");
        check(checksum8("abc") == 38, "abc mod 256");
        check(checksum16("abc") == 294, "abc 16-bit");
        check(checksum8(new byte[]{(byte) 0xFF, 0x01}) == 0, "wraps");
        check(checksum16(new byte[]{(byte) 0xFF, (byte) 0xFF}) == 510, "16-bit sum");

        byte[] original = new byte[50];
        for (int i = 0; i < 50; i++) original[i] = (byte) i;
        byte[] corrupt = original.clone();
        corrupt[10] ^= 0x01;
        check(checksum8(original) != checksum8(corrupt), "detects single-byte flip");
        byte[] swapped = original.clone();
        byte t = swapped[3]; swapped[3] = swapped[4]; swapped[4] = t;
        check(checksum8(swapped) == checksum8(original), "sum is order-insensitive by design");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
