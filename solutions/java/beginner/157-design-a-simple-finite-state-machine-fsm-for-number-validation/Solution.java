/** Task 157 - Design a simple FSM for number validation. (Tier: Beginner, CTCI Ch 7) */
import java.util.*;

public class Solution {
    static final String START = "START", SIGN = "SIGN", INT = "INT", DOT = "DOT", FRAC = "FRAC";

    /** A valid number must end in one of these states. */
    static final Set<String> ACCEPTING = new HashSet<>(List.of(INT, FRAC));

    /**
     * Data-driven transition table: (state, character class) -> next state.
     * Exponent notation (e.g. "1e5") is intentionally unsupported and rejected.
     */
    static final Map<List<String>, String> TRANSITIONS = new HashMap<>();
    static {
        TRANSITIONS.put(List.of(START, "digit"), INT);
        TRANSITIONS.put(List.of(START, "sign"), SIGN);
        TRANSITIONS.put(List.of(SIGN, "digit"), INT);
        TRANSITIONS.put(List.of(INT, "digit"), INT);
        TRANSITIONS.put(List.of(INT, "dot"), DOT);
        TRANSITIONS.put(List.of(DOT, "digit"), FRAC);
        TRANSITIONS.put(List.of(FRAC, "digit"), FRAC);
    }

    /** Character class the FSM consumes; anything else (spaces, letters) is rejected. */
    static String charClass(char c) {
        if (c >= '0' && c <= '9') return "digit";
        if (c == '+' || c == '-') return "sign";
        if (c == '.') return "dot";
        return null;
    }

    /** True iff `s` is a signed decimal number accepted by the FSM above. */
    static boolean accepts(String s) {
        String state = START;
        for (int i = 0; i < s.length(); i++) {
            String cls = charClass(s.charAt(i));
            if (cls == null) return false;
            state = TRANSITIONS.get(List.of(state, cls));
            if (state == null) return false;
        }
        return ACCEPTING.contains(state);
    }

    public static void main(String[] args) {
        check(accepts("123"), "123");
        check(accepts("+1"), "+1");
        check(accepts("-1.5"), "-1.5");
        check(accepts("0"), "0");
        check(accepts("-0.0"), "-0.0");
        check(accepts("007"), "leading zeros");
        check(!accepts("1."), "1. invalid (DOT not accepting)");
        check(!accepts(".5"), ".5 invalid (needs leading digit)");
        check(!accepts(""), "empty invalid");
        check(!accepts("+"), "+ invalid");
        check(!accepts("-"), "- invalid");
        check(!accepts("."), ". invalid");
        check(!accepts("1.2.3"), "1.2.3 invalid");
        check(!accepts("12a"), "12a invalid");
        check(!accepts(" 12"), "leading whitespace invalid");
        check(!accepts("12 "), "trailing whitespace invalid");
        check(!accepts("+-1"), "+-1 invalid");
        check(!accepts("1e5"), "exponents intentionally unsupported");

        // the transition table is explicit and data-driven
        check(!TRANSITIONS.containsKey(List.of(START, "dot")), "no leading dot transition");
        check(TRANSITIONS.get(List.of(INT, "dot")).equals(DOT), "INT --dot--> DOT");
        check(TRANSITIONS.get(List.of(DOT, "digit")).equals(FRAC), "DOT --digit--> FRAC");
        check(ACCEPTING.equals(new HashSet<>(List.of(INT, FRAC))), "accepting states");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
