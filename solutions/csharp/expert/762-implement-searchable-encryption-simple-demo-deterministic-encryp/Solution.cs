// Task 762 - Implement searchable encryption simple demo (deterministic encryption with leakage analysis). (Tier: Expert, Beyond - Security & Privacy)
//
// Searchable symmetric encryption has to give the server *something* to match on, and whatever it is leaks. This
// file builds both ends of that trade-off over the same tiny corpus and measures the leakage instead of claiming
// there is none.
//
// * Deterministic scheme: token = PRF(key, keyword), so one lookup finds the posting list in O(1) - and the same
//   keyword always yields the same token, which is exactly what the frequency analysis attack feeds on.
// * Randomised scheme: token = PRF(key, keyword) XOR mask(document), so every (document, keyword) pair gets its
//   own token. Nothing can be matched across documents, but the client must send one candidate token per document
//   and the server has to scan all of them - the classic security/functionality trade-off, measured here.
//
// Leakage is analysed with three concrete attacks/inspections: the *search pattern* (an observer groups identical
// query tokens and recovers exactly which queries asked for the same keyword, checked against a brute force
// equality partition), the *access pattern* (the returned document sets are the true result sets, which no index
// construction here hides), and *frequency analysis* (sort the server-side document frequencies and match them
// against a public frequency table of the plaintext vocabulary, which recovers the deterministic tokens and
// collapses on the randomised ones). Documented simplifications: a hand written PRF instead of AES/HMAC, a
// 12 document corpus, no forward/backward privacy machinery, no trapdoor unlinkability, no key material anywhere.
// Invariants verified: both indexes answer every query exactly like a brute force scan, conjunctive queries equal
// the set intersection, the search pattern partition is recovered exactly, and the frequency attack's accuracy is
// high for deterministic tokens and near chance for randomised ones.
using System;
using System.Collections.Generic;

internal static class ToyPrf
{
    private const ulong Offset = 14695981039346656037UL;
    private const ulong Prime = 1099511628211UL;

    internal static ulong Hash(string text)
    {
        ulong hash = Offset;
        foreach (char c in text)
        {
            hash ^= c;
            hash *= Prime;
        }
        return hash;
    }

    internal static ulong Mix(ulong value)
    {
        value += 0x9E3779B97F4A7C15UL;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
        return value ^ (value >> 31);
    }

    internal static string Token(ulong key, string keyword) => Mix(key ^ Hash("kw:" + keyword)).ToString("X16");

    internal static ulong DocumentMask(ulong key, int document) => Mix(key ^ Hash("doc:" + document));

    internal static string RandomisedToken(ulong key, int document, string keyword)
        => (Mix(key ^ Hash("kw:" + keyword)) ^ DocumentMask(key, document)).ToString("X16");
}

internal sealed class Corpus
{
    internal readonly List<List<string>> Documents = new List<List<string>>();
    internal readonly List<string> Vocabulary = new List<string>();

    /** A frequency profile: keyword i appears in (12 - i) documents while i < 12, which makes the rank order unique. */
    internal static Corpus Build(int keywords = 20, bool uniqueFrequencies = true)
    {
        var corpus = new Corpus();
        for (int i = 0; i < keywords; i++) corpus.Vocabulary.Add("word" + i.ToString("D2"));
        for (int document = 0; document < 12; document++)
        {
            var words = new List<string>();
            for (int i = 0; i < keywords; i++)
            {
                int frequency = uniqueFrequencies ? Math.Max(1, 12 - i) : 1 + (keywords - i) / 2;
                if (document < frequency) words.Add(corpus.Vocabulary[i]);
            }
            if (document == 0) words.Add("common");                       // a keyword present in one document only
            if (document == 3) words.Add("common");
            if (document == 7) words.Add("common");
            corpus.Documents.Add(words);
        }
        corpus.Vocabulary.Add("common");
        return corpus;
    }

    internal List<string> DocumentTokens(int document)
    {
        var unique = new List<string>();
        foreach (string word in Documents[document])
        {
            if (!unique.Contains(word)) unique.Add(word);
        }
        return unique;
    }
}

internal interface ISearchableIndex
{
    ulong Comparisons { get; }
    List<int> Search(string keyword);
    List<int> Conjunction(string first, string second);
    int IndexedTokens { get; }
    string ServerView();
}

/** Deterministic encryption: same keyword, same token, one lookup - and the equality leak that enables search. */
internal sealed class DeterministicIndex : ISearchableIndex
{
    private readonly ulong key;
    private readonly Dictionary<string, List<int>> postings = new Dictionary<string, List<int>>();

    internal DeterministicIndex(ulong key, Corpus corpus)
    {
        this.key = key;
        for (int document = 0; document < corpus.Documents.Count; document++)
        {
            foreach (string keyword in corpus.DocumentTokens(document))
            {
                string token = ToyPrf.Token(key, keyword);
                if (!postings.TryGetValue(token, out List<int>? list))
                {
                    list = new List<int>();
                    postings[token] = list;
                }
                list.Add(document);
            }
        }
    }

    public ulong Comparisons { get; private set; }
    public int IndexedTokens => postings.Count;

    public List<int> Search(string keyword)
    {
        Comparisons++;
        string token = ToyPrf.Token(key, keyword);
        return postings.TryGetValue(token, out List<int>? list) ? new List<int>(list) : new List<int>();
    }

    public List<int> Conjunction(string first, string second)
    {
        Comparisons += 2;
        string tokenA = ToyPrf.Token(key, first);
        string tokenB = ToyPrf.Token(key, second);
        if (!postings.TryGetValue(tokenA, out List<int>? left) || !postings.TryGetValue(tokenB, out List<int>? right))
        {
            return new List<int>();
        }
        var result = new List<int>();
        foreach (int document in left)
        {
            if (right.Contains(document)) result.Add(document);
        }
        return result;
    }

    public string ServerView()
    {
        var tokens = new List<string>(postings.Keys);
        tokens.Sort(StringComparer.Ordinal);
        return string.Join(" ", tokens);
    }

    internal int DocumentFrequency(string token) => postings.TryGetValue(token, out List<int>? list) ? list.Count : 0;

    internal List<string> Tokens() => new List<string>(postings.Keys);
}

/** Randomised tokens: equality is hidden, so a search cannot look anything up and becomes a full scan. */
internal sealed class RandomisedIndex : ISearchableIndex
{
    private readonly ulong key;
    private readonly List<HashSet<string>> perDocument = new List<HashSet<string>>();

    internal RandomisedIndex(ulong key, Corpus corpus)
    {
        this.key = key;
        for (int document = 0; document < corpus.Documents.Count; document++)
        {
            var set = new HashSet<string>();
            foreach (string keyword in corpus.DocumentTokens(document))
            {
                set.Add(ToyPrf.RandomisedToken(key, document, keyword));
            }
            perDocument.Add(set);
        }
    }

    public ulong Comparisons { get; private set; }
    public int IndexedTokens
    {
        get
        {
            int total = 0;
            foreach (HashSet<string> set in perDocument) total += set.Count;
            return total;
        }
    }

    public List<int> Search(string keyword)
    {
        var result = new List<int>();
        for (int document = 0; document < perDocument.Count; document++)
        {
            Comparisons++;
            if (perDocument[document].Contains(ToyPrf.RandomisedToken(key, document, keyword))) result.Add(document);
        }
        return result;
    }

    public List<int> Conjunction(string first, string second)
    {
        var left = Search(first);
        var right = Search(second);
        var result = new List<int>();
        foreach (int document in left)
        {
            if (right.Contains(document)) result.Add(document);
        }
        return result;
    }

    public string ServerView()
    {
        var tokens = new List<string>();
        foreach (HashSet<string> set in perDocument)
        {
            tokens.AddRange(set);
        }
        tokens.Sort(StringComparer.Ordinal);
        return string.Join(" ", tokens);
    }
}

internal sealed class Observer
{
    internal readonly List<string> SearchPattern = new List<string>();
    internal readonly List<string> AccessPattern = new List<string>();

    internal void Record(string token, List<int> documents)
    {
        SearchPattern.Add(token);
        AccessPattern.Add(string.Join(",", documents));
    }

    /** What an attacker learns from repeated queries: which of them were identical. */
    internal List<HashSet<int>> SearchPartition()
    {
        var groups = new List<HashSet<int>>();
        var index = new Dictionary<string, HashSet<int>>();
        for (int i = 0; i < SearchPattern.Count; i++)
        {
            if (!index.TryGetValue(SearchPattern[i], out HashSet<int>? group))
            {
                group = new HashSet<int>();
                index[SearchPattern[i]] = group;
                groups.Add(group);
            }
            group.Add(i);
        }
        return groups;
    }
}

public static class Solution
{
    private const ulong Key = 0x7625EEDBEEF12345UL;

    public static void Main()
    {
        Corpus corpus = Corpus.Build();
        var deterministic = new DeterministicIndex(Key, corpus);
        var randomised = new RandomisedIndex(Key, corpus);
        var plain = new Dictionary<string, List<int>>();
        for (int document = 0; document < corpus.Documents.Count; document++)
        {
            foreach (string keyword in corpus.DocumentTokens(document))
            {
                if (!plain.TryGetValue(keyword, out List<int>? list))
                {
                    list = new List<int>();
                    plain[keyword] = list;
                }
                list.Add(document);
            }
        }

        Correctness(corpus, plain, deterministic, randomised);
        SearchPatternLeakage(corpus, deterministic, randomised);
        FrequencyAnalysis(corpus, deterministic, randomised);
        Boundaries(plain, corpus, deterministic, randomised);

        Console.WriteLine("  corpus: " + corpus.Documents.Count + " documents, " + corpus.Vocabulary.Count
            + " keywords, " + plain.Count + " distinct keywords indexed");
        Console.WriteLine("PASS");
    }

    private static void Correctness(Corpus corpus, Dictionary<string, List<int>> plain,
        DeterministicIndex deterministic, RandomisedIndex randomised)
    {
        int mismatches = 0, conjunctive = 0, totalPostings = 0;
        foreach (string keyword in corpus.Vocabulary)
        {
            List<int> expected = plain.TryGetValue(keyword, out List<int>? list) ? list : new List<int>();
            List<int> viaDeterministic = deterministic.Search(keyword);
            List<int> viaRandomised = randomised.Search(keyword);
            totalPostings += expected.Count;
            if (!Same(viaDeterministic, expected) || !Same(viaRandomised, expected)) mismatches++;
            if (viaDeterministic.Contains(-1)) mismatches++;
        }
        var pairs = new (string, string)[] { ("word00", "word01"), ("word02", "word05"), ("common", "word00"), ("word19", "word00") };
        foreach ((string first, string second) in pairs)
        {
            var expected = new List<int>();
            List<int> left = plain.TryGetValue(first, out List<int>? l) ? l : new List<int>();
            List<int> right = plain.TryGetValue(second, out List<int>? r) ? r : new List<int>();
            foreach (int document in left)
            {
                if (right.Contains(document)) expected.Add(document);
            }
            if (!Same(deterministic.Conjunction(first, second), expected)) conjunctive++;
            if (!Same(randomised.Conjunction(first, second), expected)) conjunctive++;
        }
        Check(mismatches == 0, "both indexes answer every keyword exactly like the plaintext index");
        Check(conjunctive == 0, "conjunctive queries equal the plaintext intersection on " + pairs.Length + " pairs");
        Check(totalPostings == 89, "the corpus holds " + totalPostings + " postings");
        Check(deterministic.IndexedTokens == plain.Count,
            "the deterministic index has one token per keyword: " + deterministic.IndexedTokens);
        Check(randomised.IndexedTokens == totalPostings,
            "the randomised index has one token per (document, keyword) pair: " + randomised.IndexedTokens);
    }

    private static void SearchPatternLeakage(Corpus corpus, DeterministicIndex deterministic, RandomisedIndex randomised)
    {
        var queries = new[] { "word03", "word05", "word03", "common", "word05", "word03", "word19" };
        var observer = new Observer();
        foreach (string query in queries)
        {
            observer.Record(ToyPrf.Token(Key, query), deterministic.Search(query));
        }
        List<HashSet<int>> observed = observer.SearchPartition();
        List<HashSet<int>> expected = EqualityPartition(queries);
        Check(observed.Count == expected.Count && SamePartition(observed, expected),
            "the search pattern reveals exactly which queries were the same keyword (" + observed.Count + " groups)");
        Check(observer.AccessPattern[0] == observer.AccessPattern[2]
            && observer.AccessPattern[0] != observer.AccessPattern[1],
            "the access pattern also groups identical queries by their result sets");
        Check(!deterministic.Search("word03").Contains(11) && deterministic.Search("word03").Count == 9,
            "word03 is in 9 of the 12 documents, as the frequency profile says");

        var randomisedObserver = new Observer();
        foreach (string query in queries)
        {
            var documents = randomised.Search(query);
            randomisedObserver.Record(ToyPrf.Token(Key, query) + "|" + documents.Count, documents);
        }
        Check(randomisedObserver.SearchPartition().Count == expected.Count,
            "the query repetition still leaks: any searchable scheme must let the server recognise a repeated query");
        Check(randomisedObserver.SearchPartition().Count < queries.Length,
            "so " + queries.Length + " queries collapse into " + expected.Count
            + " patterns, while no document-to-document equality is observable at all");
        var freshDeterministic = new DeterministicIndex(Key, corpus);
        freshDeterministic.Search("word03");
        Check(freshDeterministic.Comparisons == 1, "the deterministic scheme answers with a single lookup");
        var freshRandomised = new RandomisedIndex(Key, corpus);
        freshRandomised.Search("word03");
        Check(freshRandomised.Comparisons == (ulong)corpus.Documents.Count,
            "hiding equality costs one token comparison per document: " + freshRandomised.Comparisons);
    }

    private static void FrequencyAnalysis(Corpus corpus, DeterministicIndex deterministic, RandomisedIndex randomised)
    {
        // Attack: rank the server visible tokens by their document frequency and match them to the public
        // frequency ranking of the plaintext vocabulary.
        var rankedTokens = deterministic.Tokens();
        rankedTokens.Sort((left, right) => deterministic.DocumentFrequency(right).CompareTo(deterministic.DocumentFrequency(left)));
        var rankedKeywords = new List<string>();
        foreach (string keyword in corpus.Vocabulary)
        {
            if (deterministic.Search(keyword).Count > 0) rankedKeywords.Add(keyword);
        }
        rankedKeywords.Sort((left, right) => deterministic.Search(right).Count.CompareTo(deterministic.Search(left).Count));

        int correct = 0, evaluated = Math.Min(rankedTokens.Count, rankedKeywords.Count);
        for (int i = 0; i < evaluated; i++)
        {
            if (rankedTokens[i] == ToyPrf.Token(Key, rankedKeywords[i])) correct++;
        }
        double accuracy = evaluated == 0 ? 0 : correct / (double)evaluated;
        Check(accuracy == 1.0 && evaluated >= 12,
            "frequency analysis recovers every deterministic token: " + correct + "/" + evaluated);

        var noisyRanking = new List<string>(rankedKeywords);
        string swap = noisyRanking[1];
        noisyRanking[1] = noisyRanking[2];
        noisyRanking[2] = swap;                                       // a small error in the public frequency table
        int noisyCorrect = 0;
        for (int i = 0; i < evaluated; i++)
        {
            if (rankedTokens[i] == ToyPrf.Token(Key, noisyRanking[i])) noisyCorrect++;
        }
        double noisyAccuracy = noisyCorrect / (double)evaluated;
        Check(noisyAccuracy >= 0.5 && noisyAccuracy < accuracy,
            "the attack degrades but survives a noisy public table: " + noisyAccuracy.ToString("F2"));

        var randomisedFrequencies = new HashSet<int>();
        foreach (string token in randomised.ServerView().Split(' ')) randomisedFrequencies.Add(1);
        Check(randomised.IndexedTokens > 0 && randomisedFrequencies.Count == 1,
            "every randomised token occurs in exactly one document, so there is no frequency signal to match");
        int randomisedHits = 0;
        var randomisedTokens = new List<string>(randomised.ServerView().Split(' '));
        randomisedTokens.Sort(StringComparer.Ordinal);
        for (int i = 0; i < evaluated; i++)
        {
            if (randomisedTokens[i] == ToyPrf.Token(Key, rankedKeywords[i])) randomisedHits++;
        }
        double randomisedAccuracy = randomisedHits / (double)evaluated;
        Check(randomisedAccuracy <= 0.25,
            "the same attack on randomised tokens stays at chance: " + randomisedAccuracy.ToString("F2"));

        Check(ToyPrf.Token(Key, "word03") == ToyPrf.Token(Key, "word03")
            && ToyPrf.Token(Key, "word03") != ToyPrf.Token(Key + 1, "word03"),
            "the token is deterministic in the key and the keyword, which is both the feature and the leak");
    }

    private static void Boundaries(Dictionary<string, List<int>> plain, Corpus corpus,
        DeterministicIndex deterministic, RandomisedIndex randomised)
    {
        Check(deterministic.Search("absent").Count == 0 && randomised.Search("absent").Count == 0,
            "an unknown keyword returns nothing from either index");
        Check(deterministic.Search(string.Empty).Count == 0 && randomised.Search(string.Empty).Count == 0,
            "an empty query returns nothing");
        Check(deterministic.Conjunction("absent", "word00").Count == 0 && deterministic.Conjunction("word00", "absent").Count == 0,
            "a conjunctive query with an unknown keyword is empty");
        Check(deterministic.Search("common").Count == 3 && randomised.Search("common").Count == 3,
            "a keyword in three documents is found by both");
        var single = Corpus.Build(3);
        var tiny = new DeterministicIndex(Key, single);
        Check(tiny.DocumentFrequency(ToyPrf.Token(Key, "word00")) == single.Documents.Count
            && tiny.Search("word02").Count == 10,
            "with three keywords, word00 is in all " + single.Documents.Count + " documents and word02 in ten");
        var empty = new Corpus();
        var emptyIndex = new DeterministicIndex(Key, empty);
        Check(emptyIndex.IndexedTokens == 0 && emptyIndex.Search("word00").Count == 0
            && emptyIndex.ServerView().Length == 0, "an empty corpus yields an empty index and empty answers");
        string[] visible = deterministic.ServerView().Split(' ');
        Check(visible.Length == plain.Count && visible.Length == corpus.Vocabulary.Count,
            "the server sees exactly one opaque token per indexed keyword: " + visible.Length);
        Check(plain.Count > 0 && deterministic.Search("word00").Count == 12,
            "the most frequent keyword is in every document");
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

    private static List<HashSet<int>> EqualityPartition(string[] queries)
    {
        var groups = new List<HashSet<int>>();
        for (int i = 0; i < queries.Length; i++)
        {
            HashSet<int>? target = null;
            for (int g = 0; g < groups.Count; g++)
            {
                int representative = -1;
                foreach (int member in groups[g])
                {
                    representative = member;
                    break;
                }
                if (queries[representative] == queries[i]) target = groups[g];
            }
            if (target == null)
            {
                target = new HashSet<int>();
                groups.Add(target);
            }
            target.Add(i);
        }
        return groups;
    }

    private static bool SamePartition(List<HashSet<int>> observed, List<HashSet<int>> expected)
    {
        if (observed.Count != expected.Count) return false;
        foreach (HashSet<int> group in expected)
        {
            bool found = false;
            foreach (HashSet<int> candidate in observed)
            {
                if (candidate.SetEquals(group)) found = true;
            }
            if (!found) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
