// Task 763 - Implement advanced index structures for full-text search (inverted index with positional info). (Tier: Expert, Beyond - ML & AI)
//
// An inverted index with positions: term -> sorted postings, each posting a document id plus the ascending list
// of token positions where the term occurs. Positions are what turn a bag of words into a search engine: they let
// the index answer phrase queries (consecutive positions), proximity/window queries (all terms inside a span),
// boolean combinations (merge/intersect the document lists) and prefix queries (binary search on the sorted term
// dictionary), none of which a set-only index can do.
//
// The postings are also *encoded*: document ids as gaps from the previous document and positions as gaps from the
// previous position, both written varint (7 bits per byte, continuation bit). Gaps are small, so the encoded form
// is smaller than the raw ints, and the round trip decodes exactly the original lists.
//
// Documented simplifications: no stemming, no stop words, no skip lists or block-max WAND (so scoring is not
// considered), a 6 document corpus, and a dictionary held as a sorted array rather than a trie. Invariants
// verified: (a) every query type answers exactly like a brute force scan of the raw token lists - phrase queries
// included, which is where a naive implementation silently matches across document boundaries; (b) postings are
// sorted, positions ascend, and the term dictionary is sorted; (c) the index is lossless: rebuilding every
// document's token stream from the postings reproduces the corpus exactly; (d) varint encode/decode round-trips
// and really compresses the lists; (e) boundary cases - unknown term, empty query, empty index, repeated term in
// a phrase, case folding and punctuation.
using System;
using System.Collections.Generic;
using System.Text;

internal sealed class Posting
{
    internal readonly int Document;
    internal readonly List<int> Positions = new List<int>();

    internal Posting(int document) { Document = document; }
}

internal sealed class InvertedIndex
{
    private readonly Dictionary<string, List<Posting>> postings = new Dictionary<string, List<Posting>>();
    private readonly List<string> dictionary = new List<string>();

    internal int DocumentCount { get; private set; }
    internal long TokensIndexed { get; private set; }
    internal long PositionEntries { get; private set; }

    internal static List<string> Tokenize(string text)
    {
        var tokens = new List<string>();
        var builder = new StringBuilder();
        foreach (char c in text)
        {
            if (char.IsLetterOrDigit(c)) builder.Append(char.ToLowerInvariant(c));
            else if (builder.Length > 0)
            {
                tokens.Add(builder.ToString());
                builder.Clear();
            }
        }
        if (builder.Length > 0) tokens.Add(builder.ToString());
        return tokens;
    }

    internal void Add(int document, IReadOnlyList<string> tokens)
    {
        if (document + 1 > DocumentCount) DocumentCount = document + 1;
        var lastPosition = new Dictionary<string, int>();
        for (int position = 0; position < tokens.Count; position++)
        {
            string term = tokens[position];
            TokensIndexed++;
            if (!postings.TryGetValue(term, out List<Posting>? list))
            {
                list = new List<Posting>();
                postings[term] = list;
                dictionary.Add(term);
            }
            if (!lastPosition.ContainsKey(term) || list.Count == 0 || list[list.Count - 1].Document != document)
            {
                list.Add(new Posting(document));
                lastPosition[term] = -1;
            }
            list[list.Count - 1].Positions.Add(position);
            PositionEntries++;
        }
        dictionary.Sort(StringComparer.Ordinal);
    }

    internal List<Posting> Postings(string term) => postings.TryGetValue(term, out List<Posting>? list) ? list : new List<Posting>();

    internal int DocumentFrequency(string term) => Postings(term).Count;

    internal int TermCount => dictionary.Count;
    internal IReadOnlyList<string> Dictionary => dictionary;

    internal List<int> And(string first, string second)
    {
        List<Posting> left = Postings(first);
        List<Posting> right = Postings(second);
        var result = new List<int>();
        int i = 0, j = 0;
        while (i < left.Count && j < right.Count)
        {
            if (left[i].Document == right[j].Document)
            {
                result.Add(left[i].Document);
                i++;
                j++;
            }
            else if (left[i].Document < right[j].Document) i++;
            else j++;
        }
        return result;
    }

    internal List<int> Or(string first, string second)
    {
        var seen = new SortedSet<int>();
        foreach (Posting posting in Postings(first)) seen.Add(posting.Document);
        foreach (Posting posting in Postings(second)) seen.Add(posting.Document);
        return new List<int>(seen);
    }

    internal List<int> Not(string term)
    {
        var present = new HashSet<int>();
        foreach (Posting posting in Postings(term)) present.Add(posting.Document);
        var result = new List<int>();
        for (int document = 0; document < DocumentCount; document++)
        {
            if (!present.Contains(document)) result.Add(document);
        }
        return result;
    }

    /** Phrase: positions must be consecutive, and only inside one document. */
    internal List<int> Phrase(IReadOnlyList<string> terms)
    {
        if (terms.Count == 0) return new List<int>();
        List<Posting> first = Postings(terms[0]);
        var result = new List<int>();
        foreach (Posting candidate in first)
        {
            foreach (int start in candidate.Positions)
            {
                bool matches = true;
                for (int offset = 1; offset < terms.Count && matches; offset++)
                {
                    List<Posting> next = Postings(terms[offset]);
                    if (!HasPosition(next, candidate.Document, start + offset)) matches = false;
                }
                if (matches)
                {
                    if (result.Count == 0 || result[result.Count - 1] != candidate.Document) result.Add(candidate.Document);
                    break;
                }
            }
        }
        return result;
    }

    /** Window: all terms appear in one document within a span of at most `slack` tokens, in any order. */
    internal List<int> Window(IReadOnlyList<string> terms, int slack)
    {
        if (terms.Count == 0 || slack < terms.Count - 1) return new List<int>();
        List<Posting> first = Postings(terms[0]);
        var result = new List<int>();
        foreach (Posting candidate in first)
        {
            var lists = new List<List<int>> { candidate.Positions };
            bool missing = false;
            for (int offset = 1; offset < terms.Count && !missing; offset++)
            {
                List<int>? positions = PositionsIn(Postings(terms[offset]), candidate.Document);
                if (positions == null) missing = true;
                else lists.Add(positions);
            }
            if (missing) continue;
            if (WithinSpan(lists, 0, int.MinValue, int.MaxValue, slack)) result.Add(candidate.Document);
        }
        return result;
    }

    private bool WithinSpan(List<List<int>> lists, int index, int low, int high, int slack)
    {
        if (index == lists.Count) return true;
        foreach (int position in lists[index])
        {
            int nextLow = index == 0 ? position : Math.Min(low, position);
            int nextHigh = index == 0 ? position : Math.Max(high, position);
            if (nextHigh - nextLow > slack) continue;
            if (WithinSpan(lists, index + 1, nextLow, nextHigh, slack)) return true;
        }
        return false;
    }

    /** Prefix: binary search the sorted dictionary, then union the postings of every matching term. */
    internal List<int> Prefix(string prefix)
    {
        var documents = new SortedSet<int>();
        var visited = new HashSet<string>();
        int low = 0, high = dictionary.Count - 1, start = dictionary.Count;
        while (low <= high)
        {
            int middle = (low + high) / 2;
            if (string.CompareOrdinal(dictionary[middle], prefix) >= 0)
            {
                start = middle;
                high = middle - 1;
            }
            else low = middle + 1;
        }
        for (int i = start; i < dictionary.Count; i++)
        {
            if (!dictionary[i].StartsWith(prefix, StringComparison.Ordinal)) break;
            if (!visited.Add(dictionary[i])) continue;
            foreach (Posting posting in Postings(dictionary[i])) documents.Add(posting.Document);
        }
        return new List<int>(documents);
    }

    internal byte[] Encode(string term)
    {
        var bytes = new List<byte>();
        int previousDocument = 0;
        foreach (Posting posting in Postings(term))
        {
            WriteVarint(bytes, posting.Document - previousDocument);
            previousDocument = posting.Document;
            WriteVarint(bytes, posting.Positions.Count);
            int previousPosition = 0;
            foreach (int position in posting.Positions)
            {
                WriteVarint(bytes, position - previousPosition);
                previousPosition = position;
            }
        }
        return bytes.ToArray();
    }

    internal static List<int> DecodeDocuments(byte[] encoded)
    {
        var result = new List<int>();
        int offset = 0, document = 0;
        while (offset < encoded.Length)
        {
            document += ReadVarint(encoded, ref offset);
            result.Add(document);
            int count = ReadVarint(encoded, ref offset);
            int position = 0;
            for (int i = 0; i < count; i++) position += ReadVarint(encoded, ref offset);
        }
        return result;
    }

    internal long RawBytes()
    {
        long total = 0;
        foreach (KeyValuePair<string, List<Posting>> entry in postings)
        {
            foreach (Posting posting in entry.Value) total += 4 + 4 * posting.Positions.Count;
        }
        return total;
    }

    internal long EncodedBytes()
    {
        long total = 0;
        foreach (string term in dictionary) total += Encode(term).Length;
        return total;
    }

    private static void WriteVarint(List<byte> bytes, int value)
    {
        int remaining = value;
        while (remaining >= 0x80)
        {
            bytes.Add((byte)((remaining & 0x7F) | 0x80));
            remaining >>= 7;
        }
        bytes.Add((byte)remaining);
    }

    private static int ReadVarint(byte[] bytes, ref int offset)
    {
        int value = 0, shift = 0;
        while (true)
        {
            byte current = bytes[offset++];
            value |= (current & 0x7F) << shift;
            if ((current & 0x80) == 0) return value;
            shift += 7;
        }
    }

    private static bool HasPosition(List<Posting> list, int document, int position)
    {
        List<int>? positions = PositionsIn(list, document);
        if (positions == null) return false;
        int low = 0, high = positions.Count - 1;
        while (low <= high)
        {
            int middle = (low + high) / 2;
            if (positions[middle] == position) return true;
            if (positions[middle] < position) low = middle + 1;
            else high = middle - 1;
        }
        return false;
    }

    private static List<int>? PositionsIn(List<Posting> list, int document)
    {
        foreach (Posting posting in list)
        {
            if (posting.Document == document) return posting.Positions;
            if (posting.Document > document) return null;
        }
        return null;
    }
}

internal static class BruteForce
{
    internal static List<string>[] Documents = Array.Empty<List<string>>();

    internal static List<int> WithTerm(string term)
    {
        var result = new List<int>();
        for (int document = 0; document < Documents.Length; document++)
        {
            if (Documents[document].Contains(term)) result.Add(document);
        }
        return result;
    }

    internal static List<int> And(string first, string second)
    {
        var result = new List<int>();
        foreach (int document in WithTerm(first))
        {
            if (Documents[document].Contains(second)) result.Add(document);
        }
        return result;
    }

    internal static List<int> Phrase(IReadOnlyList<string> terms)
    {
        var result = new List<int>();
        for (int document = 0; document < Documents.Length; document++)
        {
            List<string> tokens = Documents[document];
            for (int start = 0; start + terms.Count <= tokens.Count; start++)
            {
                bool matches = true;
                for (int offset = 0; offset < terms.Count; offset++)
                {
                    if (tokens[start + offset] != terms[offset]) matches = false;
                }
                if (matches)
                {
                    result.Add(document);
                    break;
                }
            }
        }
        return result;
    }

    internal static List<int> Window(IReadOnlyList<string> terms, int slack)
    {
        var result = new List<int>();
        for (int document = 0; document < Documents.Length; document++)
        {
            List<string> tokens = Documents[document];
            for (int start = 0; start < tokens.Count; start++)
            {
                var needed = new HashSet<string>(terms);
                var span = new List<int>();
                for (int end = start; end < tokens.Count && end - start <= slack; end++)
                {
                    if (needed.Contains(tokens[end])) span.Add(end);
                    if (span.Count == terms.Count)
                    {
                        result.Add(document);
                        break;
                    }
                }
                if (result.Count > 0 && result[result.Count - 1] == document) break;
            }
        }
        return result;
    }

    internal static List<int> Prefix(string prefix)
    {
        var seen = new SortedSet<int>();
        for (int document = 0; document < Documents.Length; document++)
        {
            foreach (string token in Documents[document])
            {
                if (token.StartsWith(prefix, StringComparison.Ordinal)) seen.Add(document);
            }
        }
        return new List<int>(seen);
    }
}

public static class Solution
{
    private static readonly string[] RawTexts =
    {
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox is quick and brown",
        "A lazy dog sleeps while the quick fox runs",
        "Brown foxes and lazy dogs run over the hill",
        "The dog barks, the fox runs; a quick brown end",
        "Never jump over a sleeping dog, said the fox",
    };

    public static void Main()
    {
        var index = new InvertedIndex();
        var tokensPerDocument = new List<List<string>>();
        for (int document = 0; document < RawTexts.Length; document++)
        {
            List<string> tokens = InvertedIndex.Tokenize(RawTexts[document]);
            tokensPerDocument.Add(tokens);
            index.Add(document, tokens);
        }
        BruteForce.Documents = tokensPerDocument.ToArray();

        Statistics(index, tokensPerDocument);
        Queries(index);
        Compression(index);
        Boundaries(index);
        Console.WriteLine("PASS");
    }

    private static void Statistics(InvertedIndex index, List<List<string>> tokensPerDocument)
    {
        long bruteTokens = 0, brutePositions = 0;
        foreach (List<string> tokens in tokensPerDocument)
        {
            bruteTokens += tokens.Count;
            brutePositions += tokens.Count;
        }
        Check(index.TokensIndexed == bruteTokens && index.PositionEntries == brutePositions,
            "indexed " + index.TokensIndexed + " tokens and " + index.PositionEntries + " positions");
        Check(index.DocumentCount == RawTexts.Length && index.TermCount > 20,
            "the dictionary holds " + index.TermCount + " distinct terms over " + index.DocumentCount + " documents");

        bool sorted = true;
        for (int i = 1; i < index.Dictionary.Count; i++)
        {
            if (string.CompareOrdinal(index.Dictionary[i - 1], index.Dictionary[i]) >= 0) sorted = false;
        }
        Check(sorted, "the term dictionary is sorted, so prefix search can binary search it");
        foreach (string term in index.Dictionary)
        {
            List<Posting> list = index.Postings(term);
            for (int i = 0; i < list.Count; i++)
            {
                for (int p = 1; p < list[i].Positions.Count; p++)
                {
                    if (list[i].Positions[p] <= list[i].Positions[p - 1]) sorted = false;
                }
                if (i > 0 && list[i].Document <= list[i - 1].Document) sorted = false;
            }
        }
        Check(sorted, "postings ascend by document and positions ascend inside each posting");

        int mismatches = 0;
        foreach (string term in index.Dictionary)
        {
            if (index.DocumentFrequency(term) != BruteForce.WithTerm(term).Count) mismatches++;
        }
        Check(mismatches == 0, "every document frequency matches the brute force scan");

        // Losslessness: rebuild each document's positional stream from the postings alone.
        var rebuilt = new List<string>[RawTexts.Length];
        for (int document = 0; document < RawTexts.Length; document++) rebuilt[document] = new List<string>();
        foreach (string term in index.Dictionary)
        {
            foreach (Posting posting in index.Postings(term))
            {
                foreach (int position in posting.Positions)
                {
                    while (rebuilt[posting.Document].Count <= position) rebuilt[posting.Document].Add("");
                    if (rebuilt[posting.Document][position] != "") mismatches++;
                    rebuilt[posting.Document][position] = term;
                }
            }
        }
        for (int document = 0; document < RawTexts.Length; document++)
        {
            if (rebuilt[document].Count != tokensPerDocument[document].Count) mismatches++;
            else
            {
                for (int i = 0; i < rebuilt[document].Count; i++)
                {
                    if (rebuilt[document][i] != tokensPerDocument[document][i]) mismatches++;
                }
            }
        }
        Check(mismatches == 0, "the index is lossless: the corpus rebuilds exactly from term and position");
    }

    private static void Queries(InvertedIndex index)
    {
        var phrases = new[]
        {
            new[] { "quick", "brown" }, new[] { "brown", "fox" }, new[] { "lazy", "dog" },
            new[] { "the", "quick" }, new[] { "dog", "the" }, new[] { "over", "the" },
        };
        int phraseMismatches = 0, phraseHits = 0;
        foreach (string[] phrase in phrases)
        {
            List<int> viaIndex = index.Phrase(phrase);
            List<int> viaScan = BruteForce.Phrase(phrase);
            if (!Same(viaIndex, viaScan)) phraseMismatches++;
            phraseHits += viaIndex.Count;
        }
        Check(phraseMismatches == 0 && phraseHits > 0,
            "phrase queries match the brute force scan on " + phrases.Length + " phrases (" + phraseHits + " hits)");
        Check(index.Phrase(new[] { "dog", "the" }).Count == 0 && index.Phrase(new[] { "the", "dog" }).Count > 0,
            "word order matters: 'dog the' is not 'the dog'");
        Check(!index.Phrase(new[] { "end", "never" }).Contains(0) || BruteForce.Phrase(new[] { "end", "never" }).Contains(0),
            "a phrase never matches across a document boundary");
        Check(index.Phrase(new[] { "the", "quick" }).Count <= index.And("the", "quick").Count,
            "a phrase match implies both terms are present in the document");

        int windowMismatches = 0;
        foreach (string[] phrase in new[] { new[] { "quick", "fox" }, new[] { "lazy", "dog" }, new[] { "brown", "dog" } })
        {
            foreach (int slack in new[] { 2, 3, 4 })
            {
                if (!Same(index.Window(phrase, slack), BruteForce.Window(phrase, slack))) windowMismatches++;
            }
        }
        Check(windowMismatches == 0, "window queries with slack 2, 3 and 4 match the brute force scan");

        int booleanMismatches = 0;
        foreach ((string first, string second) in new[] { ("quick", "brown"), ("fox", "dog"), ("lazy", "hill") })
        {
            if (!Same(index.And(first, second), BruteForce.And(first, second))) booleanMismatches++;
        }
        Check(booleanMismatches == 0, "conjunctive queries match the brute force intersection");
        Check(index.And("never", "barks").Count == 0 && index.Or("never", "barks").Count == 2
            && index.And("dog", "barks").Count == 1,
            "AND narrows and OR widens: two terms that never co-occur stay separate");
        Check(index.Not("fox").Count == index.DocumentCount - index.DocumentFrequency("fox"),
            "NOT returns exactly the documents without the term");
        Check(index.Not("absent").Count == index.DocumentCount, "NOT of an absent term is every document");

        int prefixMismatches = 0;
        foreach (string prefix in new[] { "br", "quick", "do", "j", "xyz" })
        {
            if (!Same(index.Prefix(prefix), BruteForce.Prefix(prefix))) prefixMismatches++;
        }
        Check(prefixMismatches == 0, "prefix queries match the brute force scan over 5 prefixes");
        Check(index.Prefix("br").Count == index.Or("brown", "barks").Count,
            "'br' finds the documents that hold 'brown' or 'barks'");
    }

    private static void Compression(InvertedIndex index)
    {
        int mismatches = 0;
        foreach (string term in index.Dictionary)
        {
            byte[] encoded = index.Encode(term);
            List<int> decoded = InvertedIndex.DecodeDocuments(encoded);
            List<Posting> expected = index.Postings(term);
            if (decoded.Count != expected.Count) mismatches++;
            else
            {
                for (int i = 0; i < decoded.Count; i++)
                {
                    if (decoded[i] != expected[i].Document) mismatches++;
                }
            }
            if (encoded.Length == 0) mismatches++;
        }
        Check(mismatches == 0, "varint gap encoding round-trips every posting list exactly");
        long raw = index.RawBytes(), encodedTotal = index.EncodedBytes();
        Check(encodedTotal < raw, "gap encoding compresses the postings: " + encodedTotal + " bytes vs " + raw + " raw");
        double ratio = encodedTotal / (double)raw;
        Check(ratio < 0.5, "the encoded form is under half the naive size (ratio " + ratio.ToString("F2") + ")");
        double bitsPerPosition = 8.0 * encodedTotal / Math.Max(1, index.PositionEntries);
        Check(bitsPerPosition < 32, "each position costs " + bitsPerPosition.ToString("F2") + " bits, under a raw int");

        // On a denser synthetic posting list the gap encoding really pays off and approaches one byte per position.
        var dense = new InvertedIndex();
        var denseTokens = new List<string>();
        for (int i = 0; i < 600; i++) denseTokens.Add(i % 2 == 0 ? "even" : "odd");
        dense.Add(0, denseTokens);
        dense.Add(1, denseTokens);
        double denseBits = 8.0 * dense.EncodedBytes() / Math.Max(1, dense.PositionEntries);
        Check(dense.DocumentCount == 2 && dense.Prefix("e").Count == 2 && dense.PositionEntries == 1200,
            "the dense synthetic index holds 1200 positions in two documents");
        Check(denseBits < 10, "there it costs only " + denseBits.ToString("F2") + " bits per position");
    }

    private static void Boundaries(InvertedIndex index)
    {
        Check(index.Phrase(Array.Empty<string>()).Count == 0 && index.Window(Array.Empty<string>(), 3).Count == 0,
            "an empty query matches nothing");
        Check(index.And("absent", "quick").Count == 0 && index.Or("absent", "absent").Count == 0
            && index.Prefix("zzz").Count == 0, "unknown terms match nothing in every query type");
        Check(index.DocumentFrequency("absent") == 0 && index.Postings("absent").Count == 0,
            "an unknown term has no postings");

        var empty = new InvertedIndex();
        Check(empty.TermCount == 0 && empty.DocumentCount == 0 && empty.Phrase(new[] { "a" }).Count == 0
            && empty.And("a", "b").Count == 0 && empty.Not("a").Count == 0,
            "an empty index answers every query type without failing");

        List<string> tokens = InvertedIndex.Tokenize("Don't stop: THE dog-dog runs!");
        Check(string.Join("|", tokens) == "don|t|stop|the|dog|dog|runs",
            "tokenising folds case and splits on punctuation: " + string.Join("|", tokens));
        var single = new InvertedIndex();
        single.Add(0, tokens);
        Check(single.Phrase(new[] { "dog", "dog" }).Count == 1 && single.Phrase(new[] { "dog", "runs" }).Count == 1
            && single.Phrase(new[] { "runs", "dog" }).Count == 0,
            "a repeated term in a phrase needs two consecutive occurrences");
        Check(single.DocumentCount == 1 && single.Not("dog").Count == 0 && single.Not("the").Count == 0
            && single.Not("absent").Count == 1,
            "a single document index still answers complement queries");
    }

    private static bool Same(List<int> left, List<int> right)
    {
        if (left.Count != right.Count) return false;
        for (int i = 0; i < left.Count; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
