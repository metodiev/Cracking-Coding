/** Task 199 - Deserialize JSON with arrays & objects (basic). (Tier: Beginner, CTCI Ch 7) */
import java.util.*;

public class Solution {
    public static void main(String[] args) {
        // The JDK ships no built-in JSON parser, so the cross-check uses hand-built
        // expected structures (Map/List equality) instead of a stdlib oracle.
        check(parseJson("{}").equals(new LinkedHashMap<String, Object>()), "empty object");
        check(parseJson("[]").equals(new ArrayList<>()), "empty array");
        check(parseJson("null") == null, "null literal");
        check(parseJson("true").equals(Boolean.TRUE), "true literal");
        check(parseJson("false").equals(Boolean.FALSE), "false literal");
        check(parseJson("-42").equals(-42), "negative integer");
        check(parseJson("0").equals(0), "zero");
        check(parseJson("  {  \"x\" : [ 1 , 2 ]  } ").equals(Map.of("x", List.of(1, 2))), "whitespace tolerance");
        check(parseJson("\"a\\\"b\\\\c\\n\"").equals("a\"b\\c\n"), "string escapes");
        check(parseJson("\"\\u0041\"").equals("A"), "unicode escape");
        check(parseJson("\"\\/\"").equals("/"), "solidus escape");
        check(parseJson("[1,[2,[3]]]").equals(Arrays.asList(1, Arrays.asList(2, Arrays.asList(3)))),
                "nested arrays");

        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("a", 1);
        expected.put("b", Arrays.asList(Boolean.TRUE, Boolean.FALSE, null));
        expected.put("c", "hi\n");
        Map<String, Object> empty = new LinkedHashMap<>();
        expected.put("d", empty);
        check(parseJson("{\"a\":1,\"b\":[true,false,null],\"c\":\"hi\\n\",\"d\":{}}").equals(expected),
                "nested document");

        String[] bad = {"{\"a\":1,}", "[1,]", "\"abc", "{\"a\" 1}", "tru", "@", "", "1 2", "01", "1.5",
                        "{\"a\":}", "[", "]", "[1 2]", "\"\\x\"", "\"\\u12\""};
        for (String text : bad) checkThrows(text);
        System.out.println("PASS");
    }

    static Object parseJson(String text) {
        JsonParser parser = new JsonParser(text);
        return parser.parse();
    }

    static class JsonParser {
        private final String text;
        private int pos;

        JsonParser(String text) {
            this.text = text;
        }

        Object parse() {
            Object value = parseValue();
            skipWhitespace();
            if (pos != text.length()) fail("trailing characters after JSON value");
            return value;
        }

        private void fail(String message) {
            throw new JsonParseException(message + " at index " + pos);
        }

        private void skipWhitespace() {
            while (pos < text.length() && " \t\n\r".indexOf(text.charAt(pos)) >= 0) pos++;
        }

        private char peek() {
            if (pos >= text.length()) fail("unexpected end of input");
            return text.charAt(pos);
        }

        private void expect(char ch) {
            if (peek() != ch) fail("expected '" + ch + "'");
            pos++;
        }

        private Object parseValue() {
            skipWhitespace();
            char ch = peek();
            switch (ch) {
                case '{': return parseObject();
                case '[': return parseArray();
                case '"': return parseString();
                case 't': case 'f': return parseLiteral();
                case 'n': return parseLiteral();
                default:
                    if (ch == '-' || Character.isDigit(ch)) return parseNumber();
                    fail("unexpected character '" + ch + "'");
                    return null;
            }
        }

        private Object parseLiteral() {
            for (String[] literal : new String[][] {{"true", "bool"}, {"false", "bool"}, {"null", "null"}}) {
                if (text.startsWith(literal[0], pos)) {
                    pos += literal[0].length();
                    return literal[0].equals("true") ? Boolean.TRUE
                            : literal[0].equals("false") ? Boolean.FALSE : null;
                }
            }
            fail("invalid literal");
            return null;
        }

        private Object parseNumber() {
            int start = pos;
            if (peek() == '-') pos++;
            if (pos >= text.length() || !Character.isDigit(text.charAt(pos))) fail("invalid number");
            if (text.charAt(pos) == '0') {
                pos++;
                if (pos < text.length() && Character.isDigit(text.charAt(pos))) fail("leading zeros are not allowed");
            } else {
                while (pos < text.length() && Character.isDigit(text.charAt(pos))) pos++;
            }
            if (pos < text.length() && ".eE".indexOf(text.charAt(pos)) >= 0) {
                fail("only integer numbers are supported");
            }
            return Integer.parseInt(text.substring(start, pos));
        }

        private String parseString() {
            expect('"');
            StringBuilder out = new StringBuilder();
            while (true) {
                if (pos >= text.length()) fail("unterminated string");
                char ch = text.charAt(pos++);
                if (ch == '"') return out.toString();
                if (ch == '\\') out.append(parseEscape());
                else if (ch < 0x20) fail("unescaped control character in string");
                else out.append(ch);
            }
        }

        private char parseEscape() {
            if (pos >= text.length()) fail("unterminated escape");
            char esc = text.charAt(pos++);
            switch (esc) {
                case '"': return '"';
                case '\\': return '\\';
                case '/': return '/';
                case 'b': return '\b';
                case 'f': return '\f';
                case 'n': return '\n';
                case 'r': return '\r';
                case 't': return '\t';
                case 'u':
                    if (pos + 4 > text.length()) fail("incomplete \\u escape");
                    String digits = text.substring(pos, pos + 4);
                    for (int i = 0; i < 4; i++) {
                        if ("0123456789abcdefABCDEF".indexOf(digits.charAt(i)) < 0) fail("invalid \\u escape");
                    }
                    pos += 4;
                    return (char) Integer.parseInt(digits, 16);
                default:
                    fail("invalid escape '\\" + esc + "'");
                    return 0;
            }
        }

        private Map<String, Object> parseObject() {
            expect('{');
            Map<String, Object> result = new LinkedHashMap<>();
            skipWhitespace();
            if (peek() == '}') {
                pos++;
                return result;
            }
            while (true) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                expect(':');
                result.put(key, parseValue());
                skipWhitespace();
                char ch = peek();
                if (ch == ',') {
                    pos++;
                    skipWhitespace();
                    if (peek() == '}') fail("trailing comma in object");
                    continue;
                }
                if (ch == '}') {
                    pos++;
                    return result;
                }
                fail("expected ',' or '}'");
            }
        }

        private List<Object> parseArray() {
            expect('[');
            List<Object> result = new ArrayList<>();
            skipWhitespace();
            if (peek() == ']') {
                pos++;
                return result;
            }
            while (true) {
                result.add(parseValue());
                skipWhitespace();
                char ch = peek();
                if (ch == ',') {
                    pos++;
                    skipWhitespace();
                    if (peek() == ']') fail("trailing comma in array");
                    continue;
                }
                if (ch == ']') {
                    pos++;
                    return result;
                }
                fail("expected ',' or ']'");
            }
        }
    }

    static class JsonParseException extends IllegalArgumentException {
        JsonParseException(String message) {
            super(message);
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static void checkThrows(String text) {
        try {
            parseJson(text);
        } catch (JsonParseException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected parse error for " + text);
    }
}
