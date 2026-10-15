// Task 198 - Serialize object to JSON string (manual). (Tier: Beginner, CTCI Ch 7)
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json.Nodes;

public static class Solution
{
    public static void Main()
    {
        Check(ToJson(null) == "null", "null");
        Check(ToJson(true) == "true", "true");
        Check(ToJson(false) == "false", "false");
        Check(ToJson(42) == "42", "int");
        Check(ToJson(-7) == "-7", "negative int");
        Check(ToJson("") == "\"\"", "empty string");
        Check(ToJson(new List<object?>()) == "[]", "empty array");
        Check(ToJson(new Dictionary<string, object?>()) == "{}", "empty object");
        Check(ToJson("a\"b\\c\nd\te\rf") == "\"a\\\"b\\\\c\\nd\\te\\rf\"", "string escapes");
        Check(ToJson("x\u0001y") == "\"x\\u0001y\"", "control character escape");

        var simple = new Dictionary<string, object?>
        {
            ["b"] = 1,
            ["a"] = new List<object?> { true, null, "x" },
        };
        Check(ToJson(simple) == "{\"a\":[true,null,\"x\"],\"b\":1}", "sorted keys, nested array");

        var nested = new Dictionary<string, object?>
        {
            ["a"] = new Dictionary<string, object?>
            {
                ["b"] = new Dictionary<string, object?>
                {
                    ["c"] = new List<object?> { 1, new List<object?> { 2, new List<object?> { 3, null } } },
                },
            },
        };
        Check(ToJson(nested) == "{\"a\":{\"b\":{\"c\":[1,[2,[3,null]]]}}}", "deep nesting");

        var doc = new Dictionary<string, object?>
        {
            ["users"] = new List<object?>
            {
                new Dictionary<string, object?> { ["id"] = 2, ["name"] = "Bob" },
                new Dictionary<string, object?> { ["id"] = 1, ["name"] = "Ann" },
            },
            ["count"] = 2,
            ["ok"] = true,
            ["tags"] = new List<object?>(),
        };
        string json = ToJson(doc);
        Check(json == "{\"count\":2,\"ok\":true,\"tags\":[],\"users\":"
                + "[{\"id\":2,\"name\":\"Bob\"},{\"id\":1,\"name\":\"Ann\"}]}", "deterministic document");

        // Cross-check oracle: System.Text.Json parses the hand-written output.
        var parsed = JsonNode.Parse(json)!;
        Check(parsed["count"]!.GetValue<int>() == 2, "oracle: count");
        Check(parsed["ok"]!.GetValue<bool>(), "oracle: ok");
        Check(parsed["tags"]!.AsArray().Count == 0, "oracle: tags");
        Check(parsed["users"]!.AsArray().Count == 2, "oracle: users");
        Check(parsed["users"]![1]!["name"]!.GetValue<string>() == "Ann", "oracle: nested user");

        CheckThrows(() => ToJson(new object()), "unsupported type");
        CheckThrows(() => ToJson(3.5), "floats are not supported");
        CheckThrows(() => ToJson(new Dictionary<object, object> { [1] = "x" }), "object keys must be strings");
        Console.WriteLine("PASS");
    }

    private static string ToJson(object? value)
    {
        var outBuilder = new StringBuilder();
        WriteValue(outBuilder, value);
        return outBuilder.ToString();
    }

    private static void WriteValue(StringBuilder outBuilder, object? value)
    {
        switch (value)
        {
            case null:
                outBuilder.Append("null");
                break;
            case string text:
                WriteString(outBuilder, text);
                break;
            case bool flag:
                outBuilder.Append(flag ? "true" : "false");
                break;
            case int number:
                outBuilder.Append(number);
                break;
            case long number:
                outBuilder.Append(number);
                break;
            case List<object?> list:
                outBuilder.Append('[');
                for (int i = 0; i < list.Count; i++)
                {
                    if (i > 0) outBuilder.Append(',');
                    WriteValue(outBuilder, list[i]);
                }
                outBuilder.Append(']');
                break;
            case Dictionary<string, object?> map:
                WriteMap(outBuilder, map);
                break;
            default:
                throw new ArgumentException("unsupported type: " + value.GetType().Name);
        }
    }

    private static void WriteMap(StringBuilder outBuilder, Dictionary<string, object?> map)
    {
        var keys = new List<string>(map.Keys);
        foreach (object key in map.Keys)
        {
            if (key is not string) throw new ArgumentException("object keys must be strings");
        }
        keys.Sort(StringComparer.Ordinal);
        outBuilder.Append('{');
        for (int i = 0; i < keys.Count; i++)
        {
            if (i > 0) outBuilder.Append(',');
            WriteString(outBuilder, keys[i]);
            outBuilder.Append(':');
            WriteValue(outBuilder, map[keys[i]]);
        }
        outBuilder.Append('}');
    }

    private static void WriteString(StringBuilder outBuilder, string value)
    {
        outBuilder.Append('"');
        foreach (char ch in value)
        {
            switch (ch)
            {
                case '"': outBuilder.Append("\\\""); break;
                case '\\': outBuilder.Append("\\\\"); break;
                case '\n': outBuilder.Append("\\n"); break;
                case '\t': outBuilder.Append("\\t"); break;
                case '\r': outBuilder.Append("\\r"); break;
                case '\b': outBuilder.Append("\\b"); break;
                case '\f': outBuilder.Append("\\f"); break;
                default:
                    if (ch < 0x20) outBuilder.Append("\\u").Append(((int)ch).ToString("x4"));
                    else outBuilder.Append(ch);
                    break;
            }
        }
        outBuilder.Append('"');
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: expected error - " + label);
    }
}
