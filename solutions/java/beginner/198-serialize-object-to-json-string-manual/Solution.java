/** Task 198 - Serialize object to JSON string (manual). (Tier: Beginner, CTCI Ch 7) */
import java.util.*;

public class Solution {
    public static void main(String[] args) {
        // The JDK ships no built-in JSON parser, so exact-string assertions are the primary
        // cross-language check here (task 199's parser can round-trip this output).
        check(toJson(null).equals("null"), "null");
        check(toJson(Boolean.TRUE).equals("true"), "true");
        check(toJson(Boolean.FALSE).equals("false"), "false");
        check(toJson(42).equals("42"), "int");
        check(toJson(-7).equals("-7"), "negative int");
        check(toJson("").equals("\"\""), "empty string");
        check(toJson(new ArrayList<>()).equals("[]"), "empty array");
        check(toJson(new LinkedHashMap<String, Object>()).equals("{}"), "empty object");
        check(toJson("a\"b\\c\nd\te\rf").equals("\"a\\\"b\\\\c\\nd\\te\\rf\""), "string escapes");
        check(toJson("x\u0001y").equals("\"x\\u0001y\""), "control character escape");

        Map<String, Object> simple = new LinkedHashMap<>();
        simple.put("b", 1);
        simple.put("a", Arrays.asList(Boolean.TRUE, null, "x"));
        check(toJson(simple).equals("{\"a\":[true,null,\"x\"],\"b\":1}"), "sorted keys, nested array");

        check(toJson(nested()).equals("{\"a\":{\"b\":{\"c\":[1,[2,[3,null]]]}}}"), "deep nesting");

        Map<String, Object> bob = new LinkedHashMap<>();
        bob.put("id", 2);
        bob.put("name", "Bob");
        Map<String, Object> ann = new LinkedHashMap<>();
        ann.put("id", 1);
        ann.put("name", "Ann");
        Map<String, Object> doc = new LinkedHashMap<>();
        doc.put("users", Arrays.asList(bob, ann));
        doc.put("count", 2);
        doc.put("ok", true);
        doc.put("tags", new ArrayList<>());
        check(toJson(doc).equals("{\"count\":2,\"ok\":true,\"tags\":[],\"users\":"
                + "[{\"id\":2,\"name\":\"Bob\"},{\"id\":1,\"name\":\"Ann\"}]}"), "deterministic document");

        checkThrows(() -> toJson(new Object()), "unsupported type");
        checkThrows(() -> toJson(3.5), "floats are not supported");
        Map<Object, Object> badKey = new LinkedHashMap<>();
        badKey.put(1, "x");
        checkThrows(() -> toJson(badKey), "object keys must be strings");
        System.out.println("PASS");
    }

    static Map<String, Object> nested() {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put("c", Arrays.asList(1, Arrays.asList(2, Arrays.asList(3, null))));
        Map<String, Object> middle = new LinkedHashMap<>();
        middle.put("b", inner);
        Map<String, Object> outer = new LinkedHashMap<>();
        outer.put("a", middle);
        return outer;
    }

    static String toJson(Object value) {
        StringBuilder out = new StringBuilder();
        writeValue(out, value);
        return out.toString();
    }

    static void writeValue(StringBuilder out, Object value) {
        if (value == null) {
            out.append("null");
        } else if (value instanceof String) {
            writeString(out, (String) value);
        } else if (value instanceof Boolean) {
            out.append(((Boolean) value) ? "true" : "false");
        } else if (value instanceof Integer || value instanceof Long) {
            out.append(value);
        } else if (value instanceof List<?>) {
            out.append('[');
            List<?> list = (List<?>) value;
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) out.append(',');
                writeValue(out, list.get(i));
            }
            out.append(']');
        } else if (value instanceof Map<?, ?>) {
            Map<?, ?> map = (Map<?, ?>) value;
            List<String> keys = new ArrayList<>();
            for (Object key : map.keySet()) {
                if (!(key instanceof String)) throw new IllegalArgumentException("object keys must be strings");
                keys.add((String) key);
            }
            Collections.sort(keys);
            out.append('{');
            for (int i = 0; i < keys.size(); i++) {
                if (i > 0) out.append(',');
                writeString(out, keys.get(i));
                out.append(':');
                writeValue(out, map.get(keys.get(i)));
            }
            out.append('}');
        } else {
            throw new IllegalArgumentException("unsupported type: " + value.getClass().getName());
        }
    }

    static void writeString(StringBuilder out, String value) {
        out.append('"');
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            switch (ch) {
                case '"': out.append("\\\""); break;
                case '\\': out.append("\\\\"); break;
                case '\n': out.append("\\n"); break;
                case '\t': out.append("\\t"); break;
                case '\r': out.append("\\r"); break;
                case '\b': out.append("\\b"); break;
                case '\f': out.append("\\f"); break;
                default:
                    if (ch < 0x20) out.append(String.format("\\u%04x", (int) ch));
                    else out.append(ch);
            }
        }
        out.append('"');
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static void checkThrows(Runnable action, String label) {
        try {
            action.run();
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected error - " + label);
    }
}
