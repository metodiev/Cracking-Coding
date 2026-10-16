// Task 803 - Build a small SQL engine with parser, planner, and executor supporting joins and aggregates. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// A complete miniature SQL engine for the subset
//   SELECT [DISTINCT] expr [AS alias], ... FROM t [alias] [[INNER|LEFT] JOIN t2 [alias] ON expr] ...
//   [WHERE expr] [GROUP BY expr, ...] [HAVING expr] [ORDER BY expr [DESC], ...] [LIMIT n]
// built as three real stages:
//   * parser   - hand-written lexer plus a recursive-descent parser with correct precedence
//                (OR < AND < NOT < comparison < additive < multiplicative) producing a query tree of tagged
//                expression nodes;
//   * planner  - binding (every column must resolve unambiguously in its input schema; aggregates are legal only
//                where SQL allows them), star expansion, construction of the logical plan
//                Scan / Filter / Join / Aggregate / Project / Sort / Limit with generated names group0.. and agg0..
//                for group keys and aggregates, so SELECT, HAVING and ORDER BY bind to the same aggregate, plus a
//                rule-based optimizer that pushes single-side predicates below an inner join and chooses a hash
//                join for equality conditions (nested loops otherwise);
//   * executor - materialized operators: the hash join builds on the right input and probes from the left, the
//                aggregate operator is hash grouping with COUNT/SUM/MIN/MAX/AVG, DISTINCT de-duplicates projected
//                rows, ORDER BY is a stable sort with NULLs first ascending, LIMIT takes the top-N after the sort.
//
// Invariants verified: the optimizer never changes a result (optimized versus unoptimized plan for every query in
// the battery); the executor agrees with the reference evaluator; three-valued logic holds (a comparison with NULL
// never passes a WHERE, AND/OR follow Kleene logic); aggregates ignore NULL while COUNT(*) counts rows; the join
// cardinality of a LEFT join is never below the left input's cardinality (the outer row is padded with NULLs) and
// a predicate on the nullable side is never pushed below the join.
//
// Independent oracle: a naive reference evaluator runs the same parsed query with nested-loop joins over the raw
// tables and sort-based grouping, sharing no join, grouping or projection code with the planner and the executor.
// Several queries are additionally anchored with hand-computed expected rows, and the optimizer is checked by plan
// shape (filter below the join, hash join selected for an equi-join, no pushdown on the nullable side of a LEFT
// join, unoptimized plans keeping the filter above the join).
//
// Documented simplifications: no subqueries, no INSERT/UPDATE/DELETE, no RIGHT/FULL outer joins, integers/text/NULL
// only (so AVG and division truncate toward zero and division by zero is an error), no indexes or cost-based join
// ordering, ORDER BY resolves against the FROM/GROUP BY columns rather than the select aliases, and all data is
// tiny (tens of rows) so every operator materializes its input.
#nullable enable
using System;
using System.Collections.Generic;
using System.Text;

internal sealed class SqlException : Exception
{
    internal SqlException(string message) : base(message) { }
}

internal enum ValKind { Null, Int, Text }

/** One SQL value: NULL, an integer or a text literal. */
internal readonly struct Val
{
    internal readonly ValKind Kind;
    internal readonly long N;
    internal readonly string? S;

    private Val(ValKind kind, long number, string? text) { Kind = kind; N = number; S = text; }

    internal static Val Null => default;

    internal static Val Int(long number) => new Val(ValKind.Int, number, null);

    internal static Val Text(string text) => new Val(ValKind.Text, 0, text);

    internal bool IsNull => Kind == ValKind.Null;

    internal long Number() => Kind == ValKind.Int ? N : throw new SqlException("expected an integer but found " + Show());

    internal string Show() => Kind == ValKind.Null ? "NULL" : Kind == ValKind.Int ? N.ToString() : S!;

    /** Grouping and hash-join key: includes the kind so that 5 and '5' never collide. */
    internal string Key() => (int)Kind + "|" + Show();
}

internal sealed class Schema
{
    internal readonly List<string> Tables = new List<string>();
    internal readonly List<string> Names = new List<string>();

    internal int Width => Names.Count;

    internal void Add(string table, string name)
    {
        Tables.Add(table);
        Names.Add(name);
    }

    internal Schema Concat(Schema other)
    {
        var merged = new Schema();
        for (int i = 0; i < Width; i++) merged.Add(Tables[i], Names[i]);
        for (int i = 0; i < other.Width; i++) merged.Add(other.Tables[i], other.Names[i]);
        return merged;
    }

    /** -1 = unknown, -2 = ambiguous. */
    internal int Find(string? table, string name)
    {
        int found = -1;
        for (int i = 0; i < Names.Count; i++)
        {
            if (!string.Equals(Names[i], name, StringComparison.OrdinalIgnoreCase)) continue;
            if (table != null && !string.Equals(Tables[i], table, StringComparison.OrdinalIgnoreCase)) continue;
            if (found >= 0) return -2;
            found = i;
        }
        return found;
    }

    internal string Describe() => string.Join(",", Names);
}

internal sealed class Table
{
    internal readonly string Name;
    internal readonly Schema Schema = new Schema();
    internal readonly List<Val[]> Rows = new List<Val[]>();

    internal Table(string name, params string[] columns)
    {
        Name = name;
        foreach (string column in columns) Schema.Add(name, column);
    }

    internal void AddRow(params Val[] values) => Rows.Add(values);
}

internal sealed class Catalog
{
    internal readonly SortedDictionary<string, Table> Tables = new SortedDictionary<string, Table>(StringComparer.OrdinalIgnoreCase);

    internal void Add(Table table) => Tables[table.Name] = table;

    internal Table Get(string name) => Tables.TryGetValue(name, out Table? table) ? table : throw new SqlException("unknown table: " + name);
}

internal sealed class Token
{
    internal readonly string Text;
    internal readonly ValKind Kind;
    internal readonly long Number;

    private Token(string text, ValKind kind, long number) { Text = text; Kind = kind; Number = number; }

    internal static Token Identifier(string text) => new Token(text, ValKind.Null, 0);

    internal static Token Operator(string text) => new Token(text, ValKind.Null, 0);

    internal static Token Integer(long value) => new Token(value.ToString(), ValKind.Int, value);

    internal static Token StringLiteral(string value) => new Token(value, ValKind.Text, 0);

    internal bool Is(string keyword) => string.Equals(Text, keyword, StringComparison.OrdinalIgnoreCase);
}

internal static class Lexer
{
    internal static List<Token> Tokenize(string sql)
    {
        var tokens = new List<Token>();
        int i = 0;
        while (i < sql.Length)
        {
            char c = sql[i];
            if (char.IsWhiteSpace(c)) { i++; continue; }
            if (char.IsDigit(c))
            {
                int start = i;
                while (i < sql.Length && char.IsDigit(sql[i])) i++;
                tokens.Add(Token.Integer(long.Parse(sql.Substring(start, i - start))));
                continue;
            }
            if (c == '\'')
            {
                int start = ++i;
                while (i < sql.Length && sql[i] != '\'') i++;
                if (i >= sql.Length) throw new SqlException("unterminated string literal");
                tokens.Add(Token.StringLiteral(sql.Substring(start, i - start)));
                i++;
                continue;
            }
            if (char.IsLetter(c) || c == '_')
            {
                int start = i;
                while (i < sql.Length && (char.IsLetterOrDigit(sql[i]) || sql[i] == '_')) i++;
                tokens.Add(Token.Identifier(sql.Substring(start, i - start)));
                continue;
            }
            string two = i + 1 < sql.Length ? sql.Substring(i, 2) : "";
            if (two == "<=" || two == ">=" || two == "<>")
            {
                tokens.Add(Token.Operator(two));
                i += 2;
                continue;
            }
            if ("=<>+-*/,().".IndexOf(c) >= 0)
            {
                tokens.Add(Token.Operator(c.ToString()));
                i++;
                continue;
            }
            throw new SqlException("unexpected character '" + c + "' in the query");
        }
        return tokens;
    }
}

internal enum ExprKind { Column, Literal, Binary, Not, IsNull, Call }

/** Tagged expression node; one interpreter class keeps the parser, planner and evaluator compact. */
internal sealed class Expr
{
    internal ExprKind Kind;
    internal string Name = "";
    internal string? Qualifier;
    internal Val Literal;
    internal Expr? Left;
    internal Expr? Right;
    internal bool Negated;

    internal static Expr Column(string? qualifier, string name) => new Expr { Kind = ExprKind.Column, Qualifier = qualifier, Name = name };

    internal static Expr Value(Val value) => new Expr { Kind = ExprKind.Literal, Literal = value };

    internal static Expr Binary(string op, Expr left, Expr right) => new Expr { Kind = ExprKind.Binary, Name = op, Left = left, Right = right };

    internal static Expr Negate(Expr operand) => new Expr { Kind = ExprKind.Not, Left = operand };

    internal static Expr Call(string function, Expr? argument) => new Expr { Kind = ExprKind.Call, Name = function.ToUpperInvariant(), Left = argument };

    internal bool IsAggregate => Kind == ExprKind.Call;

    internal bool IsStar => Kind == ExprKind.Column && Name == "*";

    /** Canonical text: the key that maps a group key or aggregate to its generated plan column. */
    internal string Text()
    {
        switch (Kind)
        {
            case ExprKind.Column: return Qualifier == null ? Name : Qualifier + "." + Name;
            case ExprKind.Literal: return Literal.Kind == ValKind.Text ? "'" + Literal.S : Literal.Show();
            case ExprKind.Binary: return "(" + Left!.Text() + " " + Name + " " + Right!.Text() + ")";
            case ExprKind.Not: return "NOT " + Left!.Text();
            case ExprKind.IsNull: return Left!.Text() + (Negated ? " IS NOT NULL" : " IS NULL");
            default: return Name + "(" + (Left == null ? "*" : Left.Text()) + ")";
        }
    }

    internal void CollectColumns(List<Expr> found)
    {
        if (Kind == ExprKind.Column) found.Add(this);
        if (Left != null) Left.CollectColumns(found);
        if (Right != null) Right.CollectColumns(found);
    }

    internal void CollectAggregates(List<Expr> found)
    {
        if (Kind == ExprKind.Call && !Contains(found, Text())) found.Add(this);
        if (Left != null) Left.CollectAggregates(found);
        if (Right != null) Right.CollectAggregates(found);
    }

    private static bool Contains(List<Expr> exprs, string text)
    {
        foreach (Expr expr in exprs) if (expr.Text() == text) return true;
        return false;
    }

    internal Val Eval(Schema schema, Val[] row)
    {
        switch (Kind)
        {
            case ExprKind.Column:
                int index = schema.Find(Qualifier, Name);
                if (index == -1) throw new SqlException("unknown column: " + Text());
                if (index == -2) throw new SqlException("ambiguous column: " + Text());
                return row[index];
            case ExprKind.Literal:
                return Literal;
            case ExprKind.IsNull:
                return Bool(Left!.Eval(schema, row).IsNull != Negated);
            case ExprKind.Not:
                bool? truth = Truth(Left!.Eval(schema, row));
                return truth == null ? Val.Null : Bool(truth == false);
            case ExprKind.Call:
                throw new SqlException("aggregate " + Text() + " cannot be evaluated row by row");
            default:
                return EvaluateBinary(schema, row);
        }
    }

    private Val EvaluateBinary(Schema schema, Val[] row)
    {
        Val left = Left!.Eval(schema, row);
        Val right = Right!.Eval(schema, row);
        return Apply(Name, left, right);
    }

    /** Operator semantics in one place so the reference evaluator can reuse them on already folded operands. */
    internal static Val Apply(string op, Val left, Val right)
    {
        switch (op)
        {
            case "AND": return Kleene(Truth(left), Truth(right), true);
            case "OR": return Kleene(Truth(left), Truth(right), false);
            case "=": return Compare(left, right, 0);
            case "<>": return Compare(left, right, 3);
            case "<": return Compare(left, right, 1);
            case ">": return Compare(left, right, 2);
            case "<=": return Compare(left, right, 4);
            case ">=": return Compare(left, right, 5);
        }
        if (left.IsNull || right.IsNull) return Val.Null;                 // arithmetic propagates NULL
        long a = left.Number();
        long b = right.Number();
        switch (op)
        {
            case "+": return Val.Int(a + b);
            case "-": return Val.Int(a - b);
            case "*": return Val.Int(a * b);
            default:
                if (b == 0) throw new SqlException("division by zero");
                return Val.Int(a / b);
        }
    }

    /** Three-valued logic: only TRUE passes a WHERE clause; NULL means unknown. */
    internal static bool? Truth(Val value)
    {
        if (value.IsNull) return null;
        if (value.Kind == ValKind.Int) return value.N != 0;
        return value.S!.Length > 0;
    }

    private static Val Kleene(bool? left, bool? right, bool isAnd)
    {
        bool? result;
        if (isAnd) result = left == false || right == false ? false : left == null || right == null ? null : true;
        else result = left == true || right == true ? true : left == null || right == null ? null : false;
        return result == null ? Val.Null : Bool(result.Value);
    }

    private static Val Compare(Val left, Val right, int mode)
    {
        if (left.IsNull || right.IsNull) return Val.Null;                 // comparison with NULL is unknown
        int order;
        if (left.Kind == ValKind.Int && right.Kind == ValKind.Int) order = left.N.CompareTo(right.N);
        else if (left.Kind == ValKind.Text && right.Kind == ValKind.Text) order = string.CompareOrdinal(left.S, right.S);
        else throw new SqlException("cannot compare " + left.Show() + " with " + right.Show());
        bool result = mode switch
        {
            0 => order == 0,
            1 => order < 0,
            2 => order > 0,
            3 => order != 0,
            4 => order <= 0,
            _ => order >= 0
        };
        return Bool(result);
    }

    private static Val Bool(bool value) => Val.Int(value ? 1 : 0);
}

internal sealed class JoinItem
{
    internal string Table = "";
    internal string? Alias;
    internal bool Outer;
    internal Expr? On;
}

internal sealed class Query
{
    internal bool Distinct;
    internal List<(Expr Expr, string Alias)> Items = new List<(Expr, string)>();
    internal List<JoinItem> From = new List<JoinItem>();
    internal Expr? Where;
    internal List<Expr> GroupBy = new List<Expr>();
    internal Expr? Having;
    internal List<(Expr Key, bool Desc)> OrderBy = new List<(Expr, bool)>();
    internal int? Limit;
}

internal sealed class Parser
{
    private readonly List<Token> _tokens;
    private int _pos;

    internal Parser(string sql) { _tokens = Lexer.Tokenize(sql); }

    internal Query Parse()
    {
        var query = new Query();
        Expect("SELECT");
        query.Distinct = Match("DISTINCT");
        do
        {
            Expr item = ParseExpr();
            string? alias = null;
            if (Match("AS")) alias = ExpectIdentifier();
            else if (TryAlias(out string? bare)) alias = bare;
            query.Items.Add((item, alias ?? item.Text()));
        }
        while (Match(","));
        Expect("FROM");
        var first = new JoinItem { Table = ExpectIdentifier() };
        if (Match("AS")) first.Alias = ExpectIdentifier();
        else if (TryAlias(out string? firstAlias)) first.Alias = firstAlias;
        query.From.Add(first);
        while (true)
        {
            bool outer = false;
            if (Match("INNER")) Expect("JOIN");
            else if (Match("LEFT")) { outer = true; Match("OUTER"); Expect("JOIN"); }
            else if (!Match("JOIN")) break;
            var item = new JoinItem { Table = ExpectIdentifier(), Outer = outer };
            if (Match("AS")) item.Alias = ExpectIdentifier();
            else if (TryAlias(out string? joinAlias)) item.Alias = joinAlias;
            Expect("ON");
            item.On = ParseExpr();
            query.From.Add(item);
        }
        if (Match("WHERE")) query.Where = ParseExpr();
        if (Match("GROUP"))
        {
            Expect("BY");
            do { query.GroupBy.Add(ParseExpr()); } while (Match(","));
        }
        if (Match("HAVING")) query.Having = ParseExpr();
        if (Match("ORDER"))
        {
            Expect("BY");
            do
            {
                Expr key = ParseExpr();
                bool desc = Match("DESC");
                if (!desc) Match("ASC");
                query.OrderBy.Add((key, desc));
            }
            while (Match(","));
        }
        if (Match("LIMIT"))
        {
            Token token = Next();
            if (token.Kind != ValKind.Int) throw new SqlException("LIMIT expects an integer literal");
            if (token.Number < 0) throw new SqlException("LIMIT must not be negative");
            query.Limit = (int)token.Number;
        }
        if (_pos != _tokens.Count) throw new SqlException("unexpected trailing input near '" + _tokens[_pos].Text + "'");
        return query;
    }

    private Expr ParseExpr() => ParseOr();

    private Expr ParseOr()
    {
        Expr left = ParseAnd();
        while (Match("OR")) left = Expr.Binary("OR", left, ParseAnd());
        return left;
    }

    private Expr ParseAnd()
    {
        Expr left = ParseNot();
        while (Match("AND")) left = Expr.Binary("AND", left, ParseNot());
        return left;
    }

    private Expr ParseNot() => Match("NOT") ? Expr.Negate(ParseNot()) : ParseComparison();

    private Expr ParseComparison()
    {
        Expr left = ParseAdditive();
        foreach (string op in new[] { "=", "<>", "<=", ">=", "<", ">" })
        {
            if (Match(op)) return Expr.Binary(op, left, ParseAdditive());
        }
        return left;
    }

    private Expr ParseAdditive()
    {
        Expr left = ParseTerm();
        while (true)
        {
            if (Match("+")) left = Expr.Binary("+", left, ParseTerm());
            else if (Match("-")) left = Expr.Binary("-", left, ParseTerm());
            else return left;
        }
    }

    private Expr ParseTerm()
    {
        Expr left = ParseUnary();
        while (true)
        {
            if (Match("*")) left = Expr.Binary("*", left, ParseUnary());
            else if (Match("/")) left = Expr.Binary("/", left, ParseUnary());
            else return left;
        }
    }

    private Expr ParseUnary()
    {
        if (Match("("))
        {
            Expr inner = ParseExpr();
            Expect(")");
            return ParsePostfix(inner);
        }
        if (Match("-")) return Expr.Binary("-", Expr.Value(Val.Int(0)), ParseUnary());
        return ParsePostfix(ParsePrimary());
    }

    private Expr ParsePostfix(Expr expr)
    {
        if (!Match("IS")) return expr;
        bool negated = Match("NOT");
        Expect("NULL");
        return new Expr { Kind = ExprKind.IsNull, Left = expr, Negated = negated };
    }

    private Expr ParsePrimary()
    {
        Token token = Next();
        if (token.Kind == ValKind.Int) return Expr.Value(Val.Int(token.Number));
        if (token.Kind == ValKind.Text) return Expr.Value(Val.Text(token.Text));
        if (token.Is("NULL")) return Expr.Value(Val.Null);
        if (token.Text == "*") return Expr.Column(null, "*");              // only legal in the select list
        if (token.Text.Length == 0 || !(char.IsLetter(token.Text[0]) || token.Text[0] == '_'))
            throw new SqlException("unexpected token '" + token.Text + "' in an expression");
        if (Peek("("))
        {
            Next();
            Expr? argument = null;
            if (!Match("*")) argument = ParseExpr();
            Expect(")");
            string name = token.Text.ToUpperInvariant();
            if (name != "COUNT" && name != "SUM" && name != "MIN" && name != "MAX" && name != "AVG")
                throw new SqlException("unknown function: " + name);
            return Expr.Call(name, argument);
        }
        if (Match("."))
        {
            Token column = Next();
            if (column.Kind != ValKind.Null || column.Text.Length == 0 || !(char.IsLetter(column.Text[0]) || column.Text[0] == '_'))
                throw new SqlException("expected a column name after '" + token.Text + ".'");
            return Expr.Column(token.Text, column.Text);
        }
        return Expr.Column(null, token.Text);
    }

    /** A bare alias: any identifier that is not the start of the next clause. */
    private bool TryAlias(out string? alias)
    {
        alias = null;
        if (_pos >= _tokens.Count) return false;
        Token token = _tokens[_pos];
        if (token.Kind != ValKind.Null || token.Text.Length == 0) return false;
        if (!(char.IsLetter(token.Text[0]) || token.Text[0] == '_')) return false;
        foreach (string keyword in new[]
                 { "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "JOIN", "INNER", "LEFT", "OUTER", "ON", "AS", "DESC", "ASC", "AND", "OR" })
        {
            if (token.Is(keyword)) return false;
        }
        _pos++;
        alias = token.Text;
        return true;
    }

    private Token Next()
    {
        if (_pos >= _tokens.Count) throw new SqlException("unexpected end of query");
        return _tokens[_pos++];
    }

    private bool Peek(string keyword) => _pos < _tokens.Count && _tokens[_pos].Is(keyword);

    private bool Match(string keyword)
    {
        if (!Peek(keyword)) return false;
        _pos++;
        return true;
    }

    private void Expect(string keyword)
    {
        if (!Match(keyword))
        {
            throw new SqlException("expected '" + keyword + "' but found '"
                + (_pos < _tokens.Count ? _tokens[_pos].Text : "end of query") + "'");
        }
    }

    private string ExpectIdentifier()
    {
        Token token = Next();
        if (token.Kind != ValKind.Null || token.Text.Length == 0 || !(char.IsLetter(token.Text[0]) || token.Text[0] == '_'))
            throw new SqlException("expected a table or column name but found '" + token.Text + "'");
        return token.Text;
    }
}

internal enum NodeKind { Scan, Filter, Join, Aggregate, Project, Sort, Limit }

/** Tagged plan node: covers the logical plan, the physical join choice and the operator implementation. */
internal sealed class Node
{
    internal NodeKind Kind;
    internal Table? Table;
    internal string Alias = "";
    internal Node? Left;
    internal Node? Right;
    internal Expr? Predicate;
    internal bool OuterJoin;
    internal bool HashJoin;
    internal Expr? HashLeftKey;
    internal Expr? HashRightKey;
    internal List<Expr> GroupKeys = new List<Expr>();
    internal List<Expr> Aggregates = new List<Expr>();
    internal List<(Expr Expr, string Alias)> Items = new List<(Expr, string)>();
    internal bool Distinct;
    internal List<(Expr Key, bool Desc)> SortKeys = new List<(Expr, bool)>();
    internal int Limit;

    internal Schema Schema()
    {
        var schema = new Schema();
        switch (Kind)
        {
            case NodeKind.Scan:
                foreach (string column in Table!.Schema.Names) schema.Add(Alias, column);
                return schema;
            case NodeKind.Filter:
            case NodeKind.Sort:
            case NodeKind.Limit:
                return Left!.Schema();
            case NodeKind.Join:
                return Left!.Schema().Concat(Right!.Schema());
            case NodeKind.Aggregate:
                for (int i = 0; i < GroupKeys.Count; i++) schema.Add("", "group" + i);
                for (int i = 0; i < Aggregates.Count; i++) schema.Add("", "agg" + i);
                return schema;
            default:
                foreach (var item in Items) schema.Add("", item.Alias);
                return schema;
        }
    }

    internal string Describe(int depth)
    {
        string pad = new string(' ', depth * 2);
        switch (Kind)
        {
            case NodeKind.Scan:
                return pad + "Scan(" + Alias + ")";
            case NodeKind.Filter:
                return pad + "Filter(" + Predicate!.Text() + ")\n" + Left!.Describe(depth + 1);
            case NodeKind.Join:
                return pad + "Join(" + (HashJoin ? "hash" : "loop") + (OuterJoin ? ",left" : ",inner")
                    + " on " + Predicate!.Text() + ")\n" + Left!.Describe(depth + 1) + "\n" + Right!.Describe(depth + 1);
            case NodeKind.Aggregate:
                return pad + "Aggregate(keys=" + GroupKeys.Count + ", aggs=" + Texts(Aggregates) + ")\n" + Left!.Describe(depth + 1);
            case NodeKind.Project:
                return pad + "Project(" + Aliases() + (Distinct ? ", distinct" : "") + ")\n" + Left!.Describe(depth + 1);
            case NodeKind.Sort:
                return pad + "Sort(" + SortTexts() + ")\n" + Left!.Describe(depth + 1);
            default:
                return pad + "Limit(" + Limit + ")\n" + Left!.Describe(depth + 1);
        }
    }

    private static string Texts(List<Expr> exprs)
    {
        var texts = new List<string>();
        foreach (Expr expr in exprs) texts.Add(expr.Text());
        return string.Join(",", texts);
    }

    private string Aliases()
    {
        var aliases = new List<string>();
        foreach (var item in Items) aliases.Add(item.Alias);
        return string.Join(",", aliases);
    }

    private string SortTexts()
    {
        var keys = new List<string>();
        foreach (var item in SortKeys) keys.Add(item.Key.Text() + (item.Desc ? " DESC" : ""));
        return string.Join(",", keys);
    }
}

internal sealed class ResultSet
{
    internal readonly Schema Schema;
    internal readonly List<Val[]> Rows;

    internal ResultSet(Schema schema, List<Val[]> rows)
    {
        Schema = schema;
        Rows = rows;
    }

    internal string RenderRows()
    {
        var lines = new List<string>();
        foreach (Val[] row in Rows) lines.Add(Render(row));
        return string.Join("\n", lines);
    }

    /** Order-insensitive comparison form used by the differential tests. */
    internal string Canonical()
    {
        var lines = new List<string>();
        foreach (Val[] row in Rows) lines.Add(Render(row));
        lines.Sort(StringComparer.Ordinal);
        return Schema.Describe() + "::" + string.Join(";", lines);
    }

    private static string Render(Val[] row)
    {
        var cells = new List<string>();
        foreach (Val value in row) cells.Add(value.Show());
        return string.Join("|", cells);
    }
}

internal static class Planner
{
    internal static Node Build(Catalog catalog, Query query, bool optimize)
    {
        Node plan = BuildFrom(catalog, query);
        if (query.Where != null)
        {
            Bind(query.Where, plan.Schema(), false, "WHERE");
            plan = new Node { Kind = NodeKind.Filter, Left = plan, Predicate = query.Where };
        }

        List<(Expr Expr, string Alias)> items = ExpandStar(query.Items, plan.Schema());
        var aggregates = new List<Expr>();
        foreach (var item in items) item.Expr.CollectAggregates(aggregates);
        if (query.Having != null) query.Having.CollectAggregates(aggregates);
        foreach (var key in query.OrderBy) key.Key.CollectAggregates(aggregates);

        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        if (aggregates.Count > 0 || query.GroupBy.Count > 0)
        {
            var aggregate = new Node { Kind = NodeKind.Aggregate, Left = plan };
            Schema input = plan.Schema();
            for (int i = 0; i < query.GroupBy.Count; i++)
            {
                Bind(query.GroupBy[i], input, false, "GROUP BY");
                aggregate.GroupKeys.Add(query.GroupBy[i]);
                map[query.GroupBy[i].Text()] = "group" + i;
            }
            for (int i = 0; i < aggregates.Count; i++)
            {
                Bind(aggregates[i].Left, input, false, "an aggregate argument");
                aggregate.Aggregates.Add(aggregates[i]);
                map[aggregates[i].Text()] = "agg" + i;
            }
            plan = aggregate;
            if (query.Having != null)
            {
                Expr having = Rewrite(query.Having, map);
                ValidateColumns(having, plan.Schema(), true);
                plan = new Node { Kind = NodeKind.Filter, Left = plan, Predicate = having };
            }
        }
        bool grouped = plan.Kind == NodeKind.Aggregate || (plan.Kind == NodeKind.Filter && plan.Left!.Kind == NodeKind.Aggregate);

        // ORDER BY is planned below the projection so that it can also sort by columns that are not selected.
        if (query.OrderBy.Count > 0)
        {
            var sort = new Node { Kind = NodeKind.Sort, Left = plan };
            foreach (var key in query.OrderBy)
            {
                Expr expression = Rewrite(key.Key, map);
                ValidateColumns(expression, plan.Schema(), grouped);
                sort.SortKeys.Add((expression, key.Desc));
            }
            plan = sort;
        }

        var project = new Node { Kind = NodeKind.Project, Left = plan, Distinct = query.Distinct };
        foreach (var item in items)
        {
            Expr expression = Rewrite(item.Expr, map);
            ValidateColumns(expression, plan.Schema(), grouped);
            project.Items.Add((expression, item.Alias));
        }
        plan = project;
        if (query.Limit != null) plan = new Node { Kind = NodeKind.Limit, Left = plan, Limit = query.Limit.Value };
        return optimize ? Optimize(plan) : plan;
    }

    private static Node BuildFrom(Catalog catalog, Query query)
    {
        JoinItem first = query.From[0];
        Node plan = new Node { Kind = NodeKind.Scan, Table = catalog.Get(first.Table), Alias = first.Alias ?? first.Table };
        for (int i = 1; i < query.From.Count; i++)
        {
            JoinItem item = query.From[i];
            Node right = new Node { Kind = NodeKind.Scan, Table = catalog.Get(item.Table), Alias = item.Alias ?? item.Table };
            var join = new Node { Kind = NodeKind.Join, Left = plan, Right = right, OuterJoin = item.Outer, Predicate = item.On };
            Bind(item.On, join.Schema(), false, "ON");
            plan = join;
        }
        return plan;
    }

    /** SELECT * becomes one column reference per input column, keeping the input's qualifiers. */
    private static List<(Expr Expr, string Alias)> ExpandStar(List<(Expr Expr, string Alias)> items, Schema input)
    {
        var expanded = new List<(Expr, string)>();
        foreach (var item in items)
        {
            if (!item.Expr.IsStar)
            {
                expanded.Add(item);
                continue;
            }
            for (int i = 0; i < input.Width; i++)
            {
                string table = input.Tables[i];
                expanded.Add((Expr.Column(table.Length == 0 ? null : table, input.Names[i]), input.Names[i]));
            }
        }
        return expanded;
    }

    /**
     * Above an aggregate the only legal columns are the generated group/aggregate names; anything else means a
     * non-grouped column leaked into SELECT, HAVING or ORDER BY.
     */
    private static void ValidateColumns(Expr expression, Schema schema, bool aboveAggregate)
    {
        var columns = new List<Expr>();
        expression.CollectColumns(columns);
        foreach (Expr column in columns)
        {
            int index = schema.Find(column.Qualifier, column.Name);
            if (index < 0 && aboveAggregate)
                throw new SqlException("column " + column.Text() + " must appear in GROUP BY or inside an aggregate");
            if (index == -1) throw new SqlException("unknown column: " + column.Text());
            if (index == -2) throw new SqlException("ambiguous column: " + column.Text());
        }
    }

    internal static void Bind(Expr? expression, Schema schema, bool allowAggregates, string clause)
    {
        if (expression == null) return;
        if (expression.Kind == ExprKind.Call)
        {
            if (!allowAggregates) throw new SqlException("aggregate " + expression.Text() + " is not allowed in " + clause);
            Bind(expression.Left, schema, false, clause);
            return;
        }
        if (expression.Kind == ExprKind.Column)
        {
            int index = schema.Find(expression.Qualifier, expression.Name);
            if (index == -1) throw new SqlException("unknown column " + expression.Text() + " in " + clause);
            if (index == -2) throw new SqlException("ambiguous column " + expression.Text() + " in " + clause);
            return;
        }
        Bind(expression.Left, schema, allowAggregates, clause);
        Bind(expression.Right, schema, allowAggregates, clause);
    }

    /** Replaces every group key / aggregate by its generated column name. */
    internal static Expr Rewrite(Expr expression, Dictionary<string, string> map)
    {
        if (map.TryGetValue(expression.Text(), out string? generated)) return Expr.Column(null, generated);
        if (expression.Kind == ExprKind.Column || expression.Kind == ExprKind.Literal) return expression;
        var copy = new Expr
        {
            Kind = expression.Kind, Name = expression.Name, Qualifier = expression.Qualifier,
            Literal = expression.Literal, Negated = expression.Negated
        };
        copy.Left = expression.Left == null ? null : Rewrite(expression.Left, map);
        copy.Right = expression.Right == null ? null : Rewrite(expression.Right, map);
        return copy;
    }

    internal static Node Optimize(Node node)
    {
        if (node.Left != null) node.Left = Optimize(node.Left);
        if (node.Right != null) node.Right = Optimize(node.Right);
        if (node.Kind == NodeKind.Filter && node.Left!.Kind == NodeKind.Join)
        {
            Node join = node.Left;
            Schema leftSchema = join.Left!.Schema();
            Schema rightSchema = join.Right!.Schema();
            bool leftOnly = ReferencesOnly(node.Predicate!, leftSchema, rightSchema);
            bool rightOnly = !join.OuterJoin && ReferencesOnly(node.Predicate!, rightSchema, leftSchema);
            if (leftOnly || rightOnly)
            {
                var pushed = new Node { Kind = NodeKind.Filter, Predicate = node.Predicate };
                if (leftOnly) { pushed.Left = join.Left; join.Left = pushed; }
                else { pushed.Left = join.Right; join.Right = pushed; }
                return Optimize(join);                          // the outer filter disappears into the join input
            }
        }
        if (node.Kind == NodeKind.Join) SelectJoinStrategy(node);
        return node;
    }

    /** A predicate may be pushed down only if every column resolves on that side alone. */
    private static bool ReferencesOnly(Expr expression, Schema side, Schema other)
    {
        var columns = new List<Expr>();
        expression.CollectColumns(columns);
        if (columns.Count == 0) return false;
        foreach (Expr column in columns)
        {
            if (side.Find(column.Qualifier, column.Name) < 0) return false;
            if (other.Find(column.Qualifier, column.Name) >= 0) return false;
        }
        return true;
    }

    /** Physical choice: an equi-join between the two sides becomes a hash join, everything else nested loops. */
    private static void SelectJoinStrategy(Node join)
    {
        Expr? condition = join.Predicate;
        if (condition == null || condition.Kind != ExprKind.Binary || condition.Name != "=") return;
        if (condition.Left!.Kind != ExprKind.Column || condition.Right!.Kind != ExprKind.Column) return;
        Schema leftSchema = join.Left!.Schema();
        Schema rightSchema = join.Right!.Schema();
        if (leftSchema.Find(condition.Left.Qualifier, condition.Left.Name) >= 0
            && rightSchema.Find(condition.Right.Qualifier, condition.Right.Name) >= 0)
        {
            join.HashJoin = true;
            join.HashLeftKey = condition.Left;
            join.HashRightKey = condition.Right;
        }
    }
}

internal static class Executor
{
    internal static ResultSet Run(Node node)
    {
        switch (node.Kind)
        {
            case NodeKind.Scan:
            {
                var rows = new List<Val[]>();
                foreach (Val[] row in node.Table!.Rows) rows.Add((Val[])row.Clone());
                return new ResultSet(node.Schema(), rows);
            }
            case NodeKind.Filter:
            {
                ResultSet input = Run(node.Left!);
                var rows = new List<Val[]>();
                foreach (Val[] row in input.Rows)
                {
                    if (Expr.Truth(node.Predicate!.Eval(input.Schema, row)) == true) rows.Add(row);
                }
                return new ResultSet(input.Schema, rows);
            }
            case NodeKind.Join:
                return RunJoin(node);
            case NodeKind.Aggregate:
                return RunAggregate(node);
            case NodeKind.Project:
                return RunProject(node);
            case NodeKind.Sort:
                return RunSort(node);
            default:
            {
                ResultSet input = Run(node.Left!);
                var rows = new List<Val[]>();
                for (int i = 0; i < input.Rows.Count && i < node.Limit; i++) rows.Add(input.Rows[i]);
                return new ResultSet(input.Schema, rows);
            }
        }
    }

    private static ResultSet RunJoin(Node node)
    {
        ResultSet left = Run(node.Left!);
        ResultSet right = Run(node.Right!);
        Schema schema = node.Schema();
        var rows = new List<Val[]>();
        if (node.HashJoin)
        {
            var buckets = new Dictionary<string, List<Val[]>>(StringComparer.Ordinal);
            foreach (Val[] rightRow in right.Rows)
            {
                Val key = node.HashRightKey!.Eval(right.Schema, rightRow);
                if (key.IsNull) continue;                        // NULL never equals NULL
                if (!buckets.TryGetValue(key.Key(), out List<Val[]>? bucket))
                {
                    bucket = new List<Val[]>();
                    buckets[key.Key()] = bucket;
                }
                bucket.Add(rightRow);
            }
            foreach (Val[] leftRow in left.Rows)
            {
                Val key = node.HashLeftKey!.Eval(left.Schema, leftRow);
                bool matched = false;
                if (!key.IsNull && buckets.TryGetValue(key.Key(), out List<Val[]>? bucket))
                {
                    foreach (Val[] rightRow in bucket)
                    {
                        rows.Add(Combine(leftRow, rightRow));
                        matched = true;
                    }
                }
                if (!matched && node.OuterJoin) rows.Add(Combine(leftRow, Nulls(right.Schema.Width)));
            }
        }
        else
        {
            foreach (Val[] leftRow in left.Rows)
            {
                bool matched = false;
                foreach (Val[] rightRow in right.Rows)
                {
                    Val[] candidate = Combine(leftRow, rightRow);
                    if (Expr.Truth(node.Predicate!.Eval(schema, candidate)) == true)
                    {
                        rows.Add(candidate);
                        matched = true;
                    }
                }
                if (!matched && node.OuterJoin) rows.Add(Combine(leftRow, Nulls(right.Schema.Width)));
            }
        }
        return new ResultSet(schema, rows);
    }

    private sealed class Group
    {
        internal Val[] Keys = Array.Empty<Val>();
        internal List<Val[]> Rows = new List<Val[]>();
    }

    private static ResultSet RunAggregate(Node node)
    {
        ResultSet input = Run(node.Left!);
        var groups = new SortedDictionary<string, Group>(StringComparer.Ordinal);
        foreach (Val[] row in input.Rows)
        {
            var keys = new Val[node.GroupKeys.Count];
            var text = new StringBuilder();
            for (int i = 0; i < node.GroupKeys.Count; i++)
            {
                keys[i] = node.GroupKeys[i].Eval(input.Schema, row);
                text.Append(keys[i].Key()).Append('\u0001');
            }
            if (!groups.TryGetValue(text.ToString(), out Group? group))
            {
                group = new Group { Keys = keys };
                groups[text.ToString()] = group;
            }
            group.Rows.Add(row);
        }
        if (node.GroupKeys.Count == 0 && input.Rows.Count == 0)
        {
            // No GROUP BY over an empty input is still exactly one group: COUNT(*) = 0, SUM = NULL.
            groups[string.Empty] = new Group { Keys = Array.Empty<Val>() };
        }
        var rows = new List<Val[]>();
        foreach (Group group in groups.Values)
        {
            var output = new Val[node.GroupKeys.Count + node.Aggregates.Count];
            for (int i = 0; i < group.Keys.Length; i++) output[i] = group.Keys[i];
            for (int i = 0; i < node.Aggregates.Count; i++)
                output[group.Keys.Length + i] = Fold(node.Aggregates[i], input.Schema, group.Rows);
            rows.Add(output);
        }
        return new ResultSet(node.Schema(), rows);
    }

    /** COUNT/SUM/MIN/MAX/AVG over the non-NULL arguments of one group; AVG truncates. */
    internal static Val Fold(Expr aggregate, Schema schema, List<Val[]> rows)
    {
        if (aggregate.Name == "COUNT" && aggregate.Left == null) return Val.Int(rows.Count);
        long count = 0;
        long sum = 0;
        Val min = Val.Null;
        Val max = Val.Null;
        foreach (Val[] row in rows)
        {
            Val value = aggregate.Left!.Eval(schema, row);
            if (value.IsNull) continue;                          // aggregates ignore NULL, COUNT(*) does not
            count++;
            sum += value.Number();
            if (min.IsNull || value.N < min.N) min = value;
            if (max.IsNull || value.N > max.N) max = value;
        }
        switch (aggregate.Name)
        {
            case "COUNT": return Val.Int(count);
            case "MIN": return min;
            case "MAX": return max;
            default: return count == 0 ? Val.Null : Val.Int(aggregate.Name == "SUM" ? sum : sum / count);
        }
    }

    private static ResultSet RunProject(Node node)
    {
        ResultSet input = Run(node.Left!);
        var rows = new List<Val[]>();
        var seen = new SortedSet<string>(StringComparer.Ordinal);
        foreach (Val[] row in input.Rows)
        {
            var output = new Val[node.Items.Count];
            for (int i = 0; i < node.Items.Count; i++) output[i] = node.Items[i].Expr.Eval(input.Schema, row);
            var key = new StringBuilder();
            foreach (Val value in output) key.Append(value.Key()).Append('\u0001');
            if (node.Distinct && !seen.Add(key.ToString())) continue;
            rows.Add(output);
        }
        return new ResultSet(node.Schema(), rows);
    }

    private static ResultSet RunSort(Node node)
    {
        ResultSet input = Run(node.Left!);
        var indexed = new List<(Val[] Row, int Index)>();
        for (int i = 0; i < input.Rows.Count; i++) indexed.Add((input.Rows[i], i));
        indexed.Sort((a, b) =>
        {
            foreach (var item in node.SortKeys)
            {
                int comparison = Compare(item.Key.Eval(input.Schema, a.Row), item.Key.Eval(input.Schema, b.Row));
                if (comparison != 0) return item.Desc ? -comparison : comparison;
            }
            return a.Index.CompareTo(b.Index);                   // stable sort
        });
        var rows = new List<Val[]>();
        foreach (var item in indexed) rows.Add(item.Row);
        return new ResultSet(input.Schema, rows);
    }

    /** ORDER BY: NULL is smallest, so DESC naturally puts NULLs last. */
    internal static int Compare(Val left, Val right)
    {
        if (left.IsNull && right.IsNull) return 0;
        if (left.IsNull) return -1;
        if (right.IsNull) return 1;
        if (left.Kind == ValKind.Int && right.Kind == ValKind.Int) return left.N.CompareTo(right.N);
        return string.CompareOrdinal(left.Show(), right.Show());
    }

    private static Val[] Combine(Val[] left, Val[] right)
    {
        var combined = new Val[left.Length + right.Length];
        Array.Copy(left, combined, left.Length);
        Array.Copy(right, 0, combined, left.Length, right.Length);
        return combined;
    }

    private static Val[] Nulls(int width)
    {
        var nulls = new Val[width];
        for (int i = 0; i < width; i++) nulls[i] = Val.Null;
        return nulls;
    }
}

/** Independent reference engine: nested-loop joins and sort-based grouping, no planner and no hash tables. */
internal static class ReferenceEngine
{
    private sealed class RowWithMap
    {
        internal readonly Schema Schema;
        internal readonly Dictionary<string, Val> Map;
        internal readonly Val[]? SourceRow;

        internal RowWithMap(Schema schema, Dictionary<string, Val> map, Val[]? sourceRow = null)
        {
            Schema = schema;
            Map = map;
            SourceRow = sourceRow;
        }
    }

    internal static ResultSet Evaluate(Catalog catalog, Query query)
    {
        Schema schema = new Schema();
        var rows = new List<Val[]>();
        JoinItem first = query.From[0];
        Table table = catalog.Get(first.Table);
        string alias = first.Alias ?? first.Table;
        for (int i = 0; i < table.Schema.Width; i++) schema.Add(alias, table.Schema.Names[i]);
        foreach (Val[] row in table.Rows) rows.Add((Val[])row.Clone());

        for (int j = 1; j < query.From.Count; j++)
        {
            JoinItem item = query.From[j];
            Table rightTable = catalog.Get(item.Table);
            string rightAlias = item.Alias ?? item.Table;
            Schema rightSchema = new Schema();
            for (int i = 0; i < rightTable.Schema.Width; i++) rightSchema.Add(rightAlias, rightTable.Schema.Names[i]);
            Schema combined = schema.Concat(rightSchema);
            var next = new List<Val[]>();
            foreach (Val[] leftRow in rows)
            {
                bool matched = false;
                foreach (Val[] rightRow in rightTable.Rows)
                {
                    var candidate = new Val[leftRow.Length + rightRow.Length];
                    Array.Copy(leftRow, candidate, leftRow.Length);
                    Array.Copy(rightRow, 0, candidate, leftRow.Length, rightRow.Length);
                    if (Expr.Truth(item.On!.Eval(combined, candidate)) == true)
                    {
                        next.Add(candidate);
                        matched = true;
                    }
                }
                if (!matched && item.Outer)
                {
                    var padded = new Val[leftRow.Length + rightTable.Schema.Width];
                    Array.Copy(leftRow, padded, leftRow.Length);
                    for (int i = leftRow.Length; i < padded.Length; i++) padded[i] = Val.Null;
                    next.Add(padded);
                }
            }
            schema = combined;
            rows = next;
        }

        if (query.Where != null)
        {
            var filtered = new List<Val[]>();
            foreach (Val[] row in rows) if (Expr.Truth(query.Where.Eval(schema, row)) == true) filtered.Add(row);
            rows = filtered;
        }

        var aggregates = new List<Expr>();
        var items = new List<(Expr Expr, string Alias)>();
        foreach (var item in query.Items)
        {
            if (!item.Expr.IsStar)
            {
                items.Add(item);
                continue;
            }
            for (int i = 0; i < schema.Width; i++)
            {
                string owner = schema.Tables[i];
                items.Add((Expr.Column(owner.Length == 0 ? null : owner, schema.Names[i]), schema.Names[i]));
            }
        }
        foreach (var item in items) item.Expr.CollectAggregates(aggregates);
        if (query.Having != null) query.Having.CollectAggregates(aggregates);
        foreach (var key in query.OrderBy) key.Key.CollectAggregates(aggregates);

        var outputRows = new List<RowWithMap>();
        if (aggregates.Count > 0 || query.GroupBy.Count > 0)
        {
            // Sort-based grouping: bucket rows by their rendered key, then fold each bucket with plain loops.
            var buckets = new SortedDictionary<string, List<Val[]>>(StringComparer.Ordinal);
            var keys = new SortedDictionary<string, Val[]>(StringComparer.Ordinal);
            foreach (Val[] row in rows)
            {
                var text = new StringBuilder();
                var values = new Val[query.GroupBy.Count];
                for (int i = 0; i < query.GroupBy.Count; i++)
                {
                    values[i] = query.GroupBy[i].Eval(schema, row);
                    text.Append(values[i].Key()).Append('\u0001');
                }
                if (!buckets.TryGetValue(text.ToString(), out List<Val[]>? bucket))
                {
                    bucket = new List<Val[]>();
                    buckets[text.ToString()] = bucket;
                    keys[text.ToString()] = values;
                }
                bucket.Add(row);
            }
            if (query.GroupBy.Count == 0 && rows.Count == 0) buckets[string.Empty] = new List<Val[]>();
            foreach (var bucket in buckets)
            {
                var map = new Dictionary<string, Val>(StringComparer.Ordinal);
                if (keys.TryGetValue(bucket.Key, out Val[]? groupValues))
                {
                    for (int i = 0; i < query.GroupBy.Count; i++) map[query.GroupBy[i].Text()] = groupValues[i];
                }
                foreach (Expr aggregate in aggregates) map[aggregate.Text()] = Executor.Fold(aggregate, schema, bucket.Value);
                outputRows.Add(new RowWithMap(schema, map));
            }
        }
        else
        {
            foreach (Val[] row in rows) outputRows.Add(new RowWithMap(schema, new Dictionary<string, Val>(StringComparer.Ordinal), row));
        }

        if (query.Having != null)
        {
            var kept = new List<RowWithMap>();
            foreach (RowWithMap row in outputRows)
            {
                if (Expr.Truth(EvaluateMapped(query.Having, row)) == true) kept.Add(row);
            }
            outputRows = kept;
        }

        var projected = new List<(Val[] Row, int Index)>();
        for (int i = 0; i < outputRows.Count; i++)
        {
            var output = new Val[items.Count];
            for (int k = 0; k < items.Count; k++) output[k] = EvaluateMapped(items[k].Expr, outputRows[i]);
            projected.Add((output, i));
        }
        if (query.OrderBy.Count > 0)
        {
            projected.Sort((a, b) =>
            {
                foreach (var key in query.OrderBy)
                {
                    int comparison = Executor.Compare(EvaluateMapped(key.Key, outputRows[a.Item2]), EvaluateMapped(key.Key, outputRows[b.Item2]));
                    if (comparison != 0) return key.Desc ? -comparison : comparison;
                }
                return a.Item2.CompareTo(b.Item2);
            });
        }
        if (query.Distinct)
        {
            var seen = new SortedSet<string>(StringComparer.Ordinal);
            var unique = new List<(Val[], int)>();
            var deduplicated = new List<(Val[], int)>();
            foreach (var item in projected) deduplicated.Add(item);
            foreach (var item in deduplicated)
            {
                var key = new StringBuilder();
                foreach (Val value in item.Item1) key.Append(value.Key()).Append('\u0001');
                if (seen.Add(key.ToString())) unique.Add(item);
            }
            projected = unique;
        }
        if (query.Limit != null && projected.Count > query.Limit.Value)
            projected.RemoveRange(query.Limit.Value, projected.Count - query.Limit.Value);

        var schemaOut = new Schema();
        foreach (var item in items) schemaOut.Add("", item.Alias);
        var resultRows = new List<Val[]>();
        foreach (var item in projected) resultRows.Add(item.Item1);
        return new ResultSet(schemaOut, resultRows);
    }

    /** Folds an expression bottom-up: group keys and aggregates come from the map, plain columns from the row. */
    private static Val EvaluateMapped(Expr expression, RowWithMap row)
    {
        if (row.Map.TryGetValue(expression.Text(), out Val value)) return value;
        switch (expression.Kind)
        {
            case ExprKind.Literal:
                return expression.Literal;
            case ExprKind.Column:
                if (row.SourceRow != null) return expression.Eval(row.Schema, row.SourceRow);
                throw new SqlException("reference evaluator cannot resolve " + expression.Text());
            case ExprKind.Call:
                throw new SqlException("reference evaluator cannot resolve " + expression.Text());
            case ExprKind.Not:
                Val operand = EvaluateMapped(expression.Left!, row);
                return Expr.Negate(Expr.Column(null, "@u")).Eval(UnarySchema, new[] { operand });
            case ExprKind.IsNull:
                Val inner = EvaluateMapped(expression.Left!, row);
                var probe = new Expr { Kind = ExprKind.IsNull, Left = Expr.Column(null, "@u"), Negated = expression.Negated };
                return probe.Eval(UnarySchema, new[] { inner });
            default:
                Val left = EvaluateMapped(expression.Left!, row);
                Val right = EvaluateMapped(expression.Right!, row);
                return Expr.Apply(expression.Name, left, right);
        }
    }

    private static readonly Schema UnarySchema = CreateUnarySchema();

    private static Schema CreateUnarySchema()
    {
        var schema = new Schema();
        schema.Add("", "@u");
        return schema;
    }
}

public static class Solution
{
    public static void Main()
    {
        Catalog catalog = BuildCatalog();

        // 1-5. Hand-computed anchors for the parser, filter, star expansion, ORDER BY/LIMIT and DISTINCT.
        Check(Rows(catalog, "SELECT name FROM users WHERE id = 3") == "cy", "a simple filter returns the expected row");
        Check(Head(catalog, "SELECT name FROM users WHERE id = 3") == "name", "the output column keeps the expression name");
        Check(Rows(catalog, "SELECT * FROM depts ORDER BY id DESC LIMIT 2") == "3|research\n2|sales", "star expansion plus ORDER BY DESC and LIMIT return the top two departments");
        Check(Rows(catalog, "SELECT id + 10 FROM users WHERE name = 'ada'") == "11", "arithmetic in the select list is evaluated");
        Check(Rows(catalog, "SELECT DISTINCT dept_id FROM users ORDER BY dept_id") == "NULL\n1\n2\n3", "DISTINCT removes duplicates and NULL sorts first");

        // 6-8. Joins: an inner join drops the unmatched row, a left join pads it with NULLs.
        Check(Rows(catalog, "SELECT u.name, d.name FROM users u JOIN depts d ON u.dept_id = d.id ORDER BY u.id")
            == "ada|engineering\nbob|engineering\ncy|sales\neve|sales\nfinn|research", "the inner join matches only rows with a department");
        Check(Rows(catalog, "SELECT COUNT(*) FROM users u LEFT JOIN depts d ON u.dept_id = d.id") == "6", "a left join keeps every left row");
        Check(Rows(catalog, "SELECT d.name FROM users u LEFT JOIN depts d ON u.dept_id = d.id WHERE d.id IS NULL") == "NULL",
            "the unmatched left row is padded with NULLs and visible through IS NULL");

        // 9-15. Aggregates with and without GROUP BY, HAVING and NULL handling.
        Check(Rows(catalog, "SELECT COUNT(*), COUNT(dept_id), SUM(dept_id), MIN(dept_id), MAX(dept_id) FROM users") == "6|5|9|1|3",
            "COUNT(*) counts rows while the other aggregates skip NULLs");
        Check(Rows(catalog, "SELECT SUM(amount), COUNT(amount), AVG(amount) FROM sales") == "179|5|35",
            "SUM/COUNT/AVG ignore the NULL amount and AVG truncates");
        Check(Rows(catalog, "SELECT d.name, COUNT(*) FROM users u LEFT JOIN depts d ON u.dept_id = d.id GROUP BY d.name ORDER BY d.name")
            == "NULL|1\nengineering|2\nresearch|1\nsales|2", "GROUP BY produces one row per group including the NULL group");
        Check(Rows(catalog, "SELECT d.name, COUNT(*) FROM users u JOIN depts d ON u.dept_id = d.id GROUP BY d.name HAVING COUNT(*) > 1")
            == "engineering|2\nsales|2", "HAVING filters groups on an aggregate");
        Check(Rows(catalog, "SELECT COUNT(*) FROM users WHERE id > 100") == "0", "a count over an empty input is zero");
        Check(Rows(catalog, "SELECT SUM(amount) FROM sales WHERE amount > 1000") == "NULL", "SUM over an empty input is NULL");
        Check(Rows(catalog, "SELECT SUM(amount), COUNT(*) FROM sales GROUP BY amount > 20 ORDER BY SUM(amount)")
            == "NULL|1\n15|2\n164|3", "GROUP BY accepts an arbitrary expression and the NULL group sorts first");
        Check(Rows(catalog, "SELECT dept_id, COUNT(*) FROM users GROUP BY dept_id ORDER BY dept_id") == "NULL|1\n1|2\n2|2\n3|1",
            "grouping by a nullable column keeps the NULL group");

        // 16-19. NULL semantics and three-valued logic.
        Check(Rows(catalog, "SELECT id FROM users WHERE dept_id = NULL") == "", "a comparison with NULL is unknown and filters everything");
        Check(Rows(catalog, "SELECT id FROM users WHERE dept_id IS NOT NULL ORDER BY id") == "1\n2\n3\n5\n6", "IS NOT NULL selects the rows with a department");
        Check(Rows(catalog, "SELECT COUNT(*) FROM users WHERE dept_id = 1 OR name = 'finn'") == "3", "OR combines two predicates");
        Check(Rows(catalog, "SELECT COUNT(*) FROM users WHERE NOT (dept_id = 1) AND dept_id IS NOT NULL") == "3", "NOT binds tighter than AND and NULL stays unknown");
        Check(Rows(catalog, "SELECT COUNT(*) FROM users WHERE dept_id <> 1") == "3", "a NULL dept_id never satisfies a comparison");

        // 20-21. Differential test 1: the optimizer never changes a result.
        string[] battery =
        {
            "SELECT u.name, d.name FROM users u JOIN depts d ON u.dept_id = d.id WHERE u.id > 1",
            "SELECT u.name FROM users u LEFT JOIN depts d ON u.dept_id = d.id WHERE d.name = 'sales'",
            "SELECT u.name FROM users u LEFT JOIN depts d ON u.dept_id = d.id WHERE d.name IS NULL",
            "SELECT d.name, COUNT(*) FROM users u JOIN depts d ON u.dept_id = d.id GROUP BY d.name HAVING COUNT(*) >= 2 ORDER BY d.name DESC",
            "SELECT s.user_id, SUM(s.amount) FROM sales s WHERE s.amount IS NOT NULL GROUP BY s.user_id ORDER BY SUM(s.amount) DESC",
            "SELECT u.name, COUNT(s.amount), SUM(s.amount) FROM users u LEFT JOIN sales s ON s.user_id = u.id GROUP BY u.name ORDER BY u.name",
            "SELECT DISTINCT d.name FROM depts d JOIN users u ON u.dept_id = d.id WHERE u.dept_id < 3 ORDER BY d.name",
            "SELECT u.name, d.name, s.amount FROM users u JOIN depts d ON u.dept_id = d.id JOIN sales s ON s.user_id = u.id ORDER BY u.id, s.amount",
            "SELECT COUNT(*) FROM users u JOIN depts d ON u.dept_id = d.id WHERE u.name <> 'ada'",
            "SELECT dept_id, COUNT(*) FROM users WHERE dept_id IS NOT NULL GROUP BY dept_id HAVING COUNT(*) > 0 ORDER BY dept_id",
            "SELECT * FROM users WHERE id <= 3 ORDER BY id",
            "SELECT u.name, COUNT(*) FROM users u LEFT JOIN sales s ON s.user_id = u.id GROUP BY u.name HAVING COUNT(*) > 1 ORDER BY u.name"
        };
        foreach (string sql in battery)
        {
            ResultSet optimized = Execute(catalog, sql, true);
            ResultSet plain = Execute(catalog, sql, false);
            Check(optimized.Canonical() == plain.Canonical(), "the optimizer preserves the result of: " + sql);
        }

        // 22-23. Differential test 2: the executor agrees with the independent reference evaluator.
        foreach (string sql in battery)
        {
            ResultSet optimized = Execute(catalog, sql, true);
            ResultSet reference = ReferenceEngine.Evaluate(catalog, new Parser(sql).Parse());
            Check(optimized.Canonical() == reference.Canonical(), "the executor matches the reference evaluator for: " + sql);
        }
        Check(ReferenceEngine.Evaluate(catalog, new Parser("SELECT COUNT(*), SUM(dept_id) FROM users").Parse()).RenderRows() == "6|9",
            "the reference evaluator agrees on a hand-checked aggregate");

        // 24-27. Plan shape: pushdown happens, the hash join is chosen and the outer join is left alone.
        string plainPlan = PlanText(catalog, "SELECT u.name FROM users u JOIN depts d ON u.dept_id = d.id WHERE u.id = 3", false);
        string optimizedPlan = PlanText(catalog, "SELECT u.name FROM users u JOIN depts d ON u.dept_id = d.id WHERE u.id = 3", true);
        Check(plainPlan.Contains("Join(loop,inner on (u.dept_id = d.id))") && plainPlan.Contains("Filter((u.id = 3))"),
            "the unoptimized plan keeps the filter above a nested-loop join");
        Check(optimizedPlan.Contains("Join(hash,inner on (u.dept_id = d.id))"), "an equi-join is planned as a hash join");
        int joinAt = optimizedPlan.IndexOf("Join(", StringComparison.Ordinal);
        int filterAt = optimizedPlan.IndexOf("Filter((u.id = 3))", StringComparison.Ordinal);
        Check(joinAt >= 0 && filterAt > joinAt, "predicate pushdown moved the filter below the join");
        string outerPlan = PlanText(catalog, "SELECT u.name FROM users u LEFT JOIN depts d ON u.dept_id = d.id WHERE d.name IS NULL", true);
        int outerFilterAt = outerPlan.IndexOf("Filter(d.name IS NULL)", StringComparison.Ordinal);
        int outerJoinAt = outerPlan.IndexOf("Join(", StringComparison.Ordinal);
        Check(outerFilterAt >= 0 && outerJoinAt >= 0 && outerFilterAt < outerJoinAt && CountOccurrences(outerPlan, "Filter(") == 1,
            "a predicate on the nullable side of a LEFT join is not pushed down (one filter, above the join)");
        string leftPush = PlanText(catalog, "SELECT u.name FROM users u LEFT JOIN depts d ON u.dept_id = d.id WHERE u.id = 1", true);
        Check(leftPush.IndexOf("Join(", StringComparison.Ordinal) < leftPush.IndexOf("Filter((u.id = 1))", StringComparison.Ordinal),
            "a predicate on the preserved side of a LEFT join is pushed down");

        // 28-30. Aliases, expression projections and joins whose equi-condition is reversed.
        ResultSet aliased = Execute(catalog, "SELECT COUNT(*) AS total, SUM(dept_id) AS sum_dept FROM users", true);
        Check(aliased.Schema.Describe() == "total,sum_dept" && aliased.RenderRows() == "6|9", "AS renames the output columns");
        Check(Rows(catalog, "SELECT name FROM users WHERE id * 2 = 4") == "bob", "arithmetic works inside a predicate");
        Check(Rows(catalog, "SELECT u.name FROM users u JOIN depts d ON d.id = u.dept_id AND d.name = 'research'") == "finn",
            "an equi-join with an extra conjunct still returns the matching rows");

        // 31-33. Rejected input: syntax, binding and aggregate placement.
        Check(Throws(() => new Parser("SELECT FROM users").Parse()), "a missing select list is a syntax error");
        Check(Throws(() => new Parser("SELECT name FROM").Parse()), "a missing table is a syntax error");
        Check(Throws(() => new Parser("SELECT name FROM users WHERE").Parse()), "an incomplete predicate is a syntax error");
        Check(Throws(() => new Parser("SELECT name FROM users WHERE id >").Parse()), "a trailing operator is a syntax error");
        Check(Throws(() => new Parser("SELECT name FROM users LIMIT -1").Parse()), "a negative LIMIT is rejected");
        Check(Throws(() => new Parser("SELECT name FROM users LIMIT 'x'").Parse()), "a non-integer LIMIT is rejected");
        Check(Throws(() => Execute(catalog, "SELECT missing FROM users", true)), "an unknown column is rejected by the planner");
        Check(Throws(() => Execute(catalog, "SELECT name FROM missing", true)), "an unknown table is rejected by the planner");
        Check(Throws(() => Execute(catalog, "SELECT name FROM users WHERE COUNT(*) > 1", true)), "an aggregate in WHERE is rejected");
        Check(Throws(() => Execute(catalog, "SELECT name, COUNT(*) FROM users GROUP BY dept_id", true)),
            "a non-grouped column in the select list is rejected");
        Check(Throws(() => Execute(catalog, "SELECT name FROM users WHERE id = 'ada'", true)), "comparing an integer with text fails at runtime");
        Check(Throws(() => Execute(catalog, "SELECT 1 / 0 FROM users", true)), "division by zero is rejected");
        Check(Throws(() => Execute(catalog, "SELECT COUNT(COUNT(*)) FROM users", true)), "a nested aggregate is rejected");
        Check(Throws(() => Execute(catalog, "SELECT SUM(name) FROM users", true)), "SUM over a text column is rejected");
        Check(Throws(() => Execute(catalog, "SELECT u.name FROM users u JOIN depts d ON u.dept_id = d.id GROUP BY u.name HAVING d.name = 'x'", true)),
            "a non-grouped column in HAVING is rejected");

        // 34-37. Boundary cases: cardinality of outer joins, empty groups, three-way joins and determinism.
        int leftCount = Execute(catalog, "SELECT u.id FROM users u", true).Rows.Count;
        int outerCount = Execute(catalog, "SELECT u.id, d.id FROM users u LEFT JOIN depts d ON u.dept_id = d.id", true).Rows.Count;
        int innerCount = Execute(catalog, "SELECT u.id, d.id FROM users u JOIN depts d ON u.dept_id = d.id", true).Rows.Count;
        Check(outerCount == 6 && outerCount > innerCount && leftCount == 6, "the LEFT join preserves the left cardinality while the inner join drops the unmatched row");
        Check(Rows(catalog, "SELECT u.name, s.amount FROM users u LEFT JOIN sales s ON s.user_id = u.id WHERE u.name = 'finn'") == "finn|NULL",
            "a left join with no match still returns the left row");
        Check(Rows(catalog, "SELECT COUNT(*) FROM users u JOIN depts d ON u.dept_id = d.id JOIN sales s ON s.user_id = u.id WHERE s.amount > 15")
            == "2", "a three-way join with a trailing filter counts the matching combinations");
        Check(Rows(catalog, "SELECT COUNT(*) FROM depts d JOIN users u ON u.dept_id = d.id WHERE d.id = 99") == "0", "an empty join result still aggregates to one row");
        Check(PlanText(catalog, "SELECT u.name FROM users u JOIN depts d ON u.dept_id = d.id", true)
            == PlanText(catalog, "SELECT u.name FROM users u JOIN depts d ON u.dept_id = d.id", true), "planning is deterministic");
        Check(Rows(catalog, "SELECT users.name FROM users JOIN depts ON users.dept_id = depts.id WHERE depts.name = 'research'") == "finn",
            "a join without aliases resolves fully qualified columns");
        Check(Throws(() => Execute(catalog, "SELECT name FROM users JOIN depts ON users.dept_id = depts.id", true)),
            "an unqualified column that exists on both sides is ambiguous");
        Console.WriteLine("  battery: " + battery.Length + " queries x (optimized, unoptimized, reference) all agreed");

        Console.WriteLine("PASS");
    }

    private static Catalog BuildCatalog()
    {
        var catalog = new Catalog();
        var depts = new Table("depts", "id", "name");
        depts.AddRow(Val.Int(1), Val.Text("engineering"));
        depts.AddRow(Val.Int(2), Val.Text("sales"));
        depts.AddRow(Val.Int(3), Val.Text("research"));
        var users = new Table("users", "id", "name", "dept_id");
        users.AddRow(Val.Int(1), Val.Text("ada"), Val.Int(1));
        users.AddRow(Val.Int(2), Val.Text("bob"), Val.Int(1));
        users.AddRow(Val.Int(3), Val.Text("cy"), Val.Int(2));
        users.AddRow(Val.Int(4), Val.Text("dan"), Val.Null);
        users.AddRow(Val.Int(5), Val.Text("eve"), Val.Int(2));
        users.AddRow(Val.Int(6), Val.Text("finn"), Val.Int(3));
        var sales = new Table("sales", "user_id", "amount");
        sales.AddRow(Val.Int(1), Val.Int(10));
        sales.AddRow(Val.Int(1), Val.Int(25));
        sales.AddRow(Val.Int(2), Val.Int(5));
        sales.AddRow(Val.Int(3), Val.Int(40));
        sales.AddRow(Val.Int(3), Val.Null);
        sales.AddRow(Val.Int(9), Val.Int(99));
        catalog.Add(depts);
        catalog.Add(users);
        catalog.Add(sales);
        return catalog;
    }

    private static ResultSet Execute(Catalog catalog, string sql, bool optimize)
    {
        Query query = new Parser(sql).Parse();
        Node plan = Planner.Build(catalog, query, optimize);
        return Executor.Run(plan);
    }

    private static string Rows(Catalog catalog, string sql) => Execute(catalog, sql, true).RenderRows();

    private static string Head(Catalog catalog, string sql) => Execute(catalog, sql, true).Schema.Describe();

    private static string PlanText(Catalog catalog, string sql, bool optimize)
        => Planner.Build(catalog, new Parser(sql).Parse(), optimize).Describe(0);

    private static int CountOccurrences(string text, string needle)
    {
        int count = 0;
        int from = 0;
        while (true)
        {
            int at = text.IndexOf(needle, from, StringComparison.Ordinal);
            if (at < 0) return count;
            count++;
            from = at + needle.Length;
        }
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
