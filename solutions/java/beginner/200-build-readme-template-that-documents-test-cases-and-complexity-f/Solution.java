/** Task 200 - Build README template that documents test cases and complexity. (Tier: Beginner, CTCI Beyond) */
import java.util.*;

public class Solution {
    public static void main(String[] args) {
        String twoSum = renderSection("Two Sum", "Beginner", "O(n)", "O(n)",
                Arrays.asList(new String[]{"1, 2", "3"}, new String[]{"[]", "0"}));
        check(twoSum.equals(String.join("\n",
                "## Two Sum", "", "**Tier:** Beginner", "",
                "| Complexity | Value |", "| --- | --- |", "| Time | O(n) |", "| Space | O(n) |", "",
                "| Input | Expected |", "| --- | --- |",
                "| `1, 2` | `3` |", "| `[]` | `0` |")), "Two Sum section");

        String escaped = renderSection("Escape | Demo", "Beginner", "O(1)", "O(1)",
                Arrays.asList(new String[]{"a | b", "c\nd"}, new String[]{"x\ny", "z"}));
        check(escaped.equals(String.join("\n",
                "## Escape | Demo", "", "**Tier:** Beginner", "",
                "| Complexity | Value |", "| --- | --- |", "| Time | O(1) |", "| Space | O(1) |", "",
                "| Input | Expected |", "| --- | --- |",
                "| `a \\| b` | `c d` |", "| `x y` | `z` |")), "cell escaping");

        String empty = renderSection("Empty Cases", "Beginner", "O(1)", "O(1)", new ArrayList<>());
        check(empty.equals(String.join("\n",
                "## Empty Cases", "", "**Tier:** Beginner", "",
                "| Complexity | Value |", "| --- | --- |", "| Time | O(1) |", "| Space | O(1) |", "",
                "| Input | Expected |", "| --- | --- |")), "no test cases");

        check(!twoSum.endsWith("\n") && !escaped.endsWith("\n") && !empty.endsWith("\n"),
                "no trailing newline");
        System.out.println("PASS");
    }

    static String renderSection(String name, String tier, String timeComplexity,
                                String spaceComplexity, List<String[]> testCases) {
        List<String> lines = new ArrayList<>(Arrays.asList(
                "## " + name,
                "",
                "**Tier:** " + tier,
                "",
                "| Complexity | Value |",
                "| --- | --- |",
                "| Time | " + cell(timeComplexity) + " |",
                "| Space | " + cell(spaceComplexity) + " |",
                "",
                "| Input | Expected |",
                "| --- | --- |"));
        for (String[] testCase : testCases) {
            lines.add("| `" + cell(testCase[0]) + "` | `" + cell(testCase[1]) + "` |");
        }
        return String.join("\n", lines);
    }

    static String cell(String value) {
        return value.replace("|", "\\|").replace("\n", " ");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
