/** Task 041 - Check if brackets are balanced (single type). (Tier: Beginner, CTCI Ch 3) */
public class Solution {
    static boolean isBalanced(String s) {
        int depth = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') depth++;
            else if (c == ')') { depth--; if (depth < 0) return false; }
        }
        return depth == 0;
    }

    public static void main(String[] args) {
        check(isBalanced("((()))"), "nested");
        check(isBalanced("()()()"), "sequential");
        check(isBalanced(""), "empty");
        check(!isBalanced("(()"), "unclosed");
        check(!isBalanced(")("), "bad order");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
