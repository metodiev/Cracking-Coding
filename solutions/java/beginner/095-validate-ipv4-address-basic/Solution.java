/** Task 095 - Validate IPv4 address (basic). (Tier: Beginner, CTCI Ch 1)

 * Leading zeros are rejected ("1.2.3.04" invalid), except the value "0" itself.
 */
public class Solution {
    static boolean isValidIpv4(String s) {
        String[] parts = s.split("\\.", -1);
        if (parts.length != 4) return false;
        for (String part : parts) {
            if (part.isEmpty() || part.length() > 3) return false;
            for (int i = 0; i < part.length(); i++) {
                char c = part.charAt(i);
                if (c < '0' || c > '9') return false;
            }
            if (part.length() > 1 && part.charAt(0) == '0') return false;
            if (Integer.parseInt(part) > 255) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        check(isValidIpv4("192.168.1.1"), "normal");
        check(isValidIpv4("0.0.0.0"), "zeros");
        check(isValidIpv4("255.255.255.255"), "max");
        check(isValidIpv4("10.0.0.255"), "mixed");
        check(isValidIpv4("1.2.3.0"), "trailing zero");
        check(!isValidIpv4("256.1.1.1"), "part > 255");
        check(!isValidIpv4("1.2.3.256"), "last > 255");
        check(!isValidIpv4("1.2.3"), "three parts");
        check(!isValidIpv4("1.2.3.4.5"), "five parts");
        check(!isValidIpv4(""), "empty");
        check(!isValidIpv4(".1.2.3"), "leading dot");
        check(!isValidIpv4("1.2.3."), "trailing dot");
        check(!isValidIpv4("1..2.3"), "empty part");
        check(!isValidIpv4("1.2.3.a"), "letters");
        check(!isValidIpv4("1.2.3.4x"), "trailing letter");
        check(!isValidIpv4("01.2.3.4"), "leading zero");
        check(!isValidIpv4("1.2.3.00"), "leading zeros last");
        check(!isValidIpv4("1.2.3.1234"), "part too long");
        check(!isValidIpv4("-1.2.3.4"), "negative");
        check(!isValidIpv4("1.2.3.4 "), "trailing space");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
