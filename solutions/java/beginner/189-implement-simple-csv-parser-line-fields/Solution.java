/** Task 189 - Implement simple CSV parser (line -> fields). (Tier: Beginner, CTCI Ch 7) */
import java.util.*;

public class Solution {
    /**
     * RFC-4180 basics for a single line:
     * - ',' separates fields outside quotes;
     * - a field starting with '"' is quoted, and "" inside a quoted field is one literal quote;
     * - only ',' or the end of the line may follow a closing quote — anything else is an error;
     * - an empty line yields one empty field and a trailing comma yields a final empty field.
     *
     * Throws IllegalArgumentException for an unterminated quoted field, junk after a closing
     * quote, or a quote inside an unquoted field.
     */
    static List<String> parseCsvLine(String line) {
        List<String> fields = new ArrayList<>();
        int i = 0;
        while (true) {
            StringBuilder chunk = new StringBuilder();
            if (i < line.length() && line.charAt(i) == '"') {
                i++;
                while (true) {
                    if (i >= line.length()) throw new IllegalArgumentException("unterminated quoted field");
                    if (line.charAt(i) == '"') {
                        if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                            chunk.append('"');
                            i += 2;
                            continue;
                        }
                        i++;                     // closing quote
                        break;
                    }
                    chunk.append(line.charAt(i));
                    i++;
                }
                if (i < line.length() && line.charAt(i) != ',') {
                    throw new IllegalArgumentException("unexpected character after closing quote");
                }
            } else {
                while (i < line.length() && line.charAt(i) != ',') {
                    if (line.charAt(i) == '"') throw new IllegalArgumentException("quote inside unquoted field");
                    chunk.append(line.charAt(i));
                    i++;
                }
            }
            fields.add(chunk.toString());
            if (i >= line.length()) return fields;
            i++;                                 // skip the separating comma
        }
    }

    public static void main(String[] args) {
        check(parseCsvLine("a,b,c").equals(Arrays.asList("a", "b", "c")), "plain fields");
        check(parseCsvLine("a,\"b,c\",d").equals(Arrays.asList("a", "b,c", "d")), "quoted comma");
        check(parseCsvLine("\"a\"\"b\",c").equals(Arrays.asList("a\"b", "c")), "escaped quote");
        check(parseCsvLine("").equals(Arrays.asList("")), "empty line");
        check(parseCsvLine("a,,c").equals(Arrays.asList("a", "", "c")), "empty middle field");
        check(parseCsvLine("a,b,").equals(Arrays.asList("a", "b", "")), "trailing empty field");
        check(parseCsvLine(",,").equals(Arrays.asList("", "", "")), "all empty fields");
        check(parseCsvLine("\"\",x").equals(Arrays.asList("", "x")), "quoted empty field");
        check(parseCsvLine("\"line\"\"break\"").equals(Arrays.asList("line\"break")), "escaped quote only");
        check(parseCsvLine("\"a,b\",\"c,d\"").equals(Arrays.asList("a,b", "c,d")), "two quoted fields");
        check(parseCsvLine(" , ").equals(Arrays.asList(" ", " ")), "spaces are data");
        check(parseCsvLine("\"  spaced  \",tail").equals(Arrays.asList("  spaced  ", "tail")), "quotes keep spaces");
        check(parseCsvLine("single").equals(Arrays.asList("single")), "single field");

        checkThrows(() -> parseCsvLine("a,\"b,c"), "unterminated quote");
        checkThrows(() -> parseCsvLine("\""), "lone quote");
        checkThrows(() -> parseCsvLine("a,\"b\"c,d"), "junk after closing quote");
        checkThrows(() -> parseCsvLine("a\"b"), "quote inside unquoted field");
        checkThrows(() -> parseCsvLine("\"unterminated"), "unterminated at end of line");
        System.out.println("PASS");
    }

    static void checkThrows(Runnable action, String label) {
        try {
            action.run();
        } catch (IllegalArgumentException e) {
            return;
        }
        throw new AssertionError("FAILED: expected IllegalArgumentException: " + label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
