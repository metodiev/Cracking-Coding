// Task 199 - Deserialize JSON with arrays & objects (basic). (Tier: Beginner, CTCI Ch 7)
using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Nodes;

public static class Solution
{
    public static void Main()
    {
        Check(ParseJson("{}") is Dictionary<string, object?> empty && empty.Count == 0, "empty object");
        Check(ParseJson("[]") is List<object?> emptyArray && emptyArray.Count == 0, "empty array");
        Check(ParseJson("null") == null, "null literal");
        Check(ParseJson("true") is true, "true literal");
        Check(ParseJson("false") is false, "false literal");
        Check((int)ParseJson("-42")! == -42, "negative integer");
        Check((int)ParseJson("0")! == 0, "zero");
        Check(ParseJson("\"a\\\"b\\\\c\\n\"") as string == "a\"b\\c\n", "string escapes");
        Check(ParseJson("\"\\u0041\"") as string == "A", "unicode escape");
        Check(ParseJson("\"\\/\"") as string == "/", "solidus escape");

        var nested = ParseJson("[1,[2,[3]]]") as List<object?>;
        Check(nested != null && nested.Count == 2 && (int)nested[0]! == 1, "nested arrays outer");
        Check(nested![1] is List<object?> inner && (int)inner[0]! == 2, "nested arrays inner");

        var doc = ParseJson("{\"a\":1,\"b\":[true,false,null],\"c\":\"hi\\n\",\"d\":{}}") as Dictionary<string, object?>;
        Check(doc != null && (int)doc["a"]! == 1, "object value");
        Check(doc!["b"] is List<object?> list && list.Count == 3
              && list[0] is true && list[1] is false && list[2] == null, "array values");
        Check(doc["c"] as string == "hi\n", "decoded newline");
        Check(doc["d"] is Dictionary<string, object?> deep && deep.Count == 0, "empty nested object");

        // Cross-check oracle: System.Text.Json parses the same documents.
        string[] documents = {"{}", "[]", "null", "true",
            "{\"a\":1,\"b\":[true,false,null],\"c\":\"hi\\n\",\"d\":{\"e\":[]}}",
            "{\"nested\":[1,[2,[3]]],\"u\":\"\\u0041\",\"esc\":\"a\\\"b\\\\c\"}",
            "  {  \"x\" : [ 1 , 2 ]  } ", "-42", "0"};
        foreach (string text in documents)
        {
            object? value = ParseJson(text);
            string reserialized = JsonSerializer.Serialize(value);
            Check(JsonNode.DeepEquals(JsonNode.Parse(text)!, JsonNode.Parse(reserialized)!),
                    "stdlib oracle: " + text);
        }

        string[] bad = {"{\"a\":1,}", "[1,]", "\"abc", "{\"a\" 1}", "tru", "@", "", "1 2", "01", "1.5",
                        "{\"a\":}", "[", "]", "[1 2]", "\"\\x\"", "\"\\u12\""};
        foreach (string text in bad) CheckThrows(text);
        Console.WriteLine("PASS");
    }

    private static object? ParseJson(string text)
    {
        return new JsonParser(text).Parse();
    }

    private class JsonParser
    {
        private readonly string text;
        private int pos;

        public JsonParser(string text)
        {
            this.text = text;
        }

        public object? Parse()
        {
            object? value = ParseValue();
            SkipWhitespace();
            if (pos != text.Length) Fail("trailing characters after JSON value");
            return value;
        }

        private void Fail(string message)
        {
            throw new JsonParseException(message + " at index " + pos);
        }

        private void SkipWhitespace()
        {
            while (pos < text.Length && " \t\n\r".IndexOf(text[pos]) >= 0) pos++;
        }

        private char Peek()
        {
            if (pos >= text.Length) Fail("unexpected end of input");
            return text[pos];
        }

        private void Expect(char ch)
        {
            if (Peek() != ch) Fail("expected '" + ch + "'");
            pos++;
        }

        private object? ParseValue()
        {
            SkipWhitespace();
            char ch = Peek();
            switch (ch)
            {
                case '{': return ParseObject();
                case '[': return ParseArray();
                case '"': return ParseString();
                case 't':
                case 'f':
                case 'n': return ParseLiteral();
                default:
                    if (ch == '-' || char.IsDigit(ch)) return ParseNumber();
                    Fail("unexpected character '" + ch + "'");
                    return null;
            }
        }

        private object? ParseLiteral()
        {
            if (text.AsSpan(pos).StartsWith("true")) { pos += 4; return true; }
            if (text.AsSpan(pos).StartsWith("false")) { pos += 5; return false; }
            if (text.AsSpan(pos).StartsWith("null")) { pos += 4; return null; }
            Fail("invalid literal");
            return null;
        }

        private int ParseNumber()
        {
            int start = pos;
            if (Peek() == '-') pos++;
            if (pos >= text.Length || !char.IsDigit(text[pos])) Fail("invalid number");
            if (text[pos] == '0')
            {
                pos++;
                if (pos < text.Length && char.IsDigit(text[pos])) Fail("leading zeros are not allowed");
            }
            else
            {
                while (pos < text.Length && char.IsDigit(text[pos])) pos++;
            }
            if (pos < text.Length && ".eE".IndexOf(text[pos]) >= 0) Fail("only integer numbers are supported");
            return int.Parse(text.Substring(start, pos - start));
        }

        private string ParseString()
        {
            Expect('"');
            var outBuilder = new System.Text.StringBuilder();
            while (true)
            {
                if (pos >= text.Length) Fail("unterminated string");
                char ch = text[pos++];
                if (ch == '"') return outBuilder.ToString();
                if (ch == '\\') outBuilder.Append(ParseEscape());
                else if (ch < 0x20) Fail("unescaped control character in string");
                else outBuilder.Append(ch);
            }
        }

        private char ParseEscape()
        {
            if (pos >= text.Length) Fail("unterminated escape");
            char esc = text[pos++];
            switch (esc)
            {
                case '"': return '"';
                case '\\': return '\\';
                case '/': return '/';
                case 'b': return '\b';
                case 'f': return '\f';
                case 'n': return '\n';
                case 'r': return '\r';
                case 't': return '\t';
                case 'u':
                    if (pos + 4 > text.Length) Fail("incomplete \\u escape");
                    string digits = text.Substring(pos, 4);
                    foreach (char digit in digits)
                    {
                        if ("0123456789abcdefABCDEF".IndexOf(digit) < 0) Fail("invalid \\u escape");
                    }
                    pos += 4;
                    return (char)Convert.ToInt32(digits, 16);
                default:
                    Fail("invalid escape '\\" + esc + "'");
                    return '\0';
            }
        }

        private Dictionary<string, object?> ParseObject()
        {
            Expect('{');
            var result = new Dictionary<string, object?>();
            SkipWhitespace();
            if (Peek() == '}')
            {
                pos++;
                return result;
            }
            while (true)
            {
                SkipWhitespace();
                string key = ParseString();
                SkipWhitespace();
                Expect(':');
                result[key] = ParseValue();
                SkipWhitespace();
                char ch = Peek();
                if (ch == ',')
                {
                    pos++;
                    SkipWhitespace();
                    if (Peek() == '}') Fail("trailing comma in object");
                    continue;
                }
                if (ch == '}')
                {
                    pos++;
                    return result;
                }
                Fail("expected ',' or '}'");
            }
        }

        private List<object?> ParseArray()
        {
            Expect('[');
            var result = new List<object?>();
            SkipWhitespace();
            if (Peek() == ']')
            {
                pos++;
                return result;
            }
            while (true)
            {
                result.Add(ParseValue());
                SkipWhitespace();
                char ch = Peek();
                if (ch == ',')
                {
                    pos++;
                    SkipWhitespace();
                    if (Peek() == ']') Fail("trailing comma in array");
                    continue;
                }
                if (ch == ']')
                {
                    pos++;
                    return result;
                }
                Fail("expected ',' or ']'");
            }
        }
    }

    private class JsonParseException : Exception
    {
        public JsonParseException(string message) : base(message) { }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckThrows(string text)
    {
        try
        {
            ParseJson(text);
        }
        catch (JsonParseException)
        {
            return;
        }
        throw new Exception("FAILED: expected parse error for " + text);
    }
}
