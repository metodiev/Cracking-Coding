/** Task 042 - Check balanced parentheses with three bracket types. (Tier: Beginner, CTCI Ch 3) */
public class Solution {
    static boolean isBalanced(String s) {
        java.util.Deque<Character> stack = new java.util.ArrayDeque<>();
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '[' || c == '{') stack.push(c);
            else if (c == ')' || c == ']' || c == '}') {
                if (stack.isEmpty()) return false;
                char open = stack.pop();
                if (c == ')' && open != '(') return false;
                if (c == ']' && open != '[') return false;
                if (c == '}' && open != '{') return false;
            }
        }
        return stack.isEmpty();
    }

    public static void main(String[] args) {
        check(isBalanced("{[()]}"), "nested mixed");
        check(isBalanced("()[]{}"), "sequential");
        check(!isBalanced("{[(])}"), "crossed");
        check(!isBalanced("("), "unclosed");
        check(!isBalanced(")"), "stray close");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
