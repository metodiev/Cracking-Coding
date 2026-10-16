// Task 990 - Implement reproducible Docker images for canonical environment with JDK/test harness.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// A Dockerfile is parsed into typed instructions, each canonicalised (`ENV`/`LABEL` keys sorted, kinds
// lower-cased) and folded into a layer chain rooted at the resolved manifest digest of the base image:
// layer_i = FNV(layer_{i-1} | canonical_i). Identical inputs therefore produce byte-identical image digests,
// while a declared base pin that disagrees with the resolved digest is refused as drift. A rule scan flags
// the classic reproducibility hazards (unpinned base, `date`, unpinned apt/pip installs, floating `:latest`)
// and a build tripping any rule is not publishable. Layer digests drive cache reuse: the number of identical
// leading layers between two builds is exactly the reusable cache size. Verified: layer-by-layer equality of
// repeated builds, ENV-key order irrelevance against RUN-order significance, every rule finding, structural
// refusals, the chain against independently transcribed canonical texts (oracle), cache-hit counting and the
// FROM-only boundary.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }
    private static string Error(Action action)
    {
        try { action(); return ""; }
        catch (Exception ex) { return ex.Message; }
    }
    private static ulong Fnv(string text, ulong seed)
    {
        unchecked
        {
            ulong h = 14695981039346656037UL ^ seed;
            foreach (byte b in Encoding.UTF8.GetBytes(text)) { h ^= b; h *= 1099511628211UL; }
            return h;
        }
    }
    private static string Hex(ulong value) => value.ToString("x16", CultureInfo.InvariantCulture);

    private sealed class Instruction
    {
        public readonly string Kind, Args;
        public Instruction(string kind, string args) { Kind = kind; Args = args; }
    }

    private static Instruction Parse(string line)
    {
        string trimmed = line.Trim();
        int space = trimmed.IndexOf(' ');
        if (space <= 0) throw new Exception("malformed-instruction:" + trimmed);
        string kind = trimmed.Substring(0, space).ToUpperInvariant();
        if (kind != "FROM" && kind != "RUN" && kind != "ENV" && kind != "LABEL" && kind != "COPY" && kind != "WORKDIR")
            throw new Exception("unknown-instruction:" + kind);
        return new Instruction(kind, trimmed.Substring(space + 1).Trim());
    }

    /** Key-sorted pair list, so `ENV B=2 A=1` and `ENV A=1 B=2` canonicalise to the same text. */
    private static string SortPairs(string args)
    {
        var pairs = new List<string>();
        foreach (string part in args.Split(' ')) if (part.Length > 0) pairs.Add(part);
        pairs.Sort(StringComparer.Ordinal);
        return string.Join(" ", pairs);
    }

    private static string Canonical(Instruction instruction)
    {
        string args = instruction.Args;
        if (instruction.Kind == "ENV" || instruction.Kind == "LABEL") args = SortPairs(args);
        return instruction.Kind.ToLowerInvariant() + " " + args;
    }

    private static bool Pinned(string fromArgs)
    {
        int at = fromArgs.IndexOf("@sha256:");
        if (at <= 0 || fromArgs.Length - at - 8 != 64) return false;
        foreach (char c in fromArgs.Substring(at + 8))
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
        return true;
    }

    private static List<string> Scan(Instruction instruction)
    {
        var findings = new List<string>();
        if (instruction.Kind == "FROM" && !Pinned(instruction.Args))
            findings.Add("unpinned-base:" + instruction.Args);
        if (instruction.Kind == "RUN")
        {
            if (instruction.Args.Contains("date")) findings.Add("non-reproducible:date");
            if (instruction.Args.Contains("apt-get install") && !instruction.Args.Contains("="))
                findings.Add("unpinned-package:apt-get");
            if (instruction.Args.Contains("pip install") && !instruction.Args.Contains("=="))
                findings.Add("unpinned-package:pip");
            if (instruction.Args.Contains(":latest")) findings.Add("floating-tag:latest");
        }
        return findings;
    }

    private sealed class Layer
    {
        public readonly string Canonical;
        public readonly ulong Digest;
        public Layer(string canonical, ulong digest) { Canonical = canonical; Digest = digest; }
    }

    private sealed class Image
    {
        public readonly List<Layer> Layers = new List<Layer>();
        public readonly List<string> Findings = new List<string>();
        public ulong Digest;
        public bool Reproducible => Findings.Count == 0;
    }

    private static Image Build(List<string> dockerfile, string resolvedBase)
    {
        if (dockerfile.Count == 0) throw new Exception("empty-dockerfile");
        var image = new Image();
        ulong head = Fnv(resolvedBase, 0);
        for (int i = 0; i < dockerfile.Count; i++)
        {
            Instruction instruction = Parse(dockerfile[i]);
            if (instruction.Kind == "FROM")
            {
                if (i != 0) throw new Exception("from-not-first");
                if (Pinned(instruction.Args) && instruction.Args.Substring(instruction.Args.IndexOf("@sha256:") + 8) != resolvedBase)
                    throw new Exception("base-digest-mismatch");
            }
            image.Findings.AddRange(Scan(instruction));
            string canonical = Canonical(instruction);
            head = Fnv(Hex(head) + "|" + canonical, 0);
            image.Layers.Add(new Layer(canonical, head));
        }
        image.Digest = head;
        return image;
    }

    /** Cache reuse: how many leading layers two builds share unchanged. */
    private static int CacheHits(Image left, Image right)
    {
        int hits = 0;
        while (hits < left.Layers.Count && hits < right.Layers.Count && left.Layers[hits].Digest == right.Layers[hits].Digest) hits++;
        return hits;
    }

    private static List<string> Dockerfile(params string[] lines) => new List<string>(lines);
    private const string Base = "9e1f4c7a2b3d5e608f1a2b3c4d5e6f70891a2b3c4d5e6f708192a3b4c5d6e7f8";

    public static void Main()
    {
        List<string> canonicalFile = Dockerfile("FROM eclipse-temurin:21-jdk@sha256:" + Base, "RUN apt-get install -y curl=8.5.0",
            "ENV JAVA_HOME=/opt/java LANG=C.UTF-8", "COPY harness /opt/harness", "WORKDIR /opt/harness");
        Image first = Build(canonicalFile, Base);
        Image second = Build(Dockerfile(canonicalFile.ToArray()), Base);
        bool layersAgree = first.Layers.Count == second.Layers.Count;
        for (int i = 0; i < first.Layers.Count && layersAgree; i++)
            if (first.Layers[i].Digest != second.Layers[i].Digest) layersAgree = false;
        Check(first.Reproducible && layersAgree && first.Digest == second.Digest && first.Layers.Count == 5,
              "identical Dockerfiles produce byte-identical layer digests");

        Image envSwapped = Build(Dockerfile("FROM eclipse-temurin:21-jdk@sha256:" + Base, "RUN apt-get install -y curl=8.5.0",
            "ENV LANG=C.UTF-8 JAVA_HOME=/opt/java", "COPY harness /opt/harness", "WORKDIR /opt/harness"), Base);
        Image runSwapped = Build(Dockerfile("FROM eclipse-temurin:21-jdk@sha256:" + Base, "RUN apt-get install -y curl=8.5.0",
            "RUN ./gradlew --offline test", "ENV JAVA_HOME=/opt/java LANG=C.UTF-8", "COPY harness /opt/harness",
            "WORKDIR /opt/harness"), Base);
        Check(envSwapped.Digest == first.Digest && runSwapped.Digest != first.Digest && runSwapped.Layers.Count == 6,
              "ENV key order is irrelevant while RUN order changes the image");

        Image unpinned = Build(Dockerfile("FROM eclipse-temurin:21-jdk:latest", "RUN echo hi"), Base);
        Image hazards = Build(Dockerfile("FROM eclipse-temurin:21-jdk@sha256:" + Base, "RUN date > /build-stamp",
            "RUN apt-get install -y curl", "RUN pip install pytest"), Base);
        Image floating = Build(Dockerfile("FROM base@sha256:" + Base, "RUN yarn add pkg:latest"), Base);
        Check(string.Join(",", unpinned.Findings) == "unpinned-base:eclipse-temurin:21-jdk:latest"
              && string.Join(",", hazards.Findings) == "non-reproducible:date,unpinned-package:apt-get,unpinned-package:pip"
              && string.Join(",", floating.Findings) == "floating-tag:latest" && !floating.Reproducible,
              "unpinned base, date, unpinned installs and a floating tag are all flagged");

        Check(Error(() => Build(new List<string>(), Base)) == "empty-dockerfile"
              && Error(() => Build(Dockerfile("RUN echo hi", "FROM base@sha256:" + Base), Base)) == "from-not-first"
              && Error(() => Build(Dockerfile("EXPOSE 8080"), Base)) == "unknown-instruction:EXPOSE"
              && Error(() => Build(Dockerfile("FROM base@sha256:" + new string('0', 64)), Base)) == "base-digest-mismatch",
              "structural refusals and the base pin drift");

        // Oracle: independently transcribed canonical texts folded the same way must reproduce the digest.
        ulong reference = Fnv(Base, 0);
        foreach (string text in new[] { "from eclipse-temurin:21-jdk@sha256:" + Base, "run apt-get install -y curl=8.5.0",
                                        "env JAVA_HOME=/opt/java LANG=C.UTF-8", "copy harness /opt/harness", "workdir /opt/harness" })
            reference = Fnv(Hex(reference) + "|" + text, 0);
        Image extended = Build(Dockerfile("FROM eclipse-temurin:21-jdk@sha256:" + Base, "RUN apt-get install -y curl=8.5.0",
            "ENV JAVA_HOME=/opt/java LANG=C.UTF-8", "COPY harness /opt/harness", "WORKDIR /opt/harness",
            "RUN ./gradlew --offline test"), Base);
        Image unrelated = Build(Dockerfile("FROM other@sha256:" + Base, "RUN echo different"), Base);
        Check(reference == first.Digest && first.Layers[2].Canonical == "env JAVA_HOME=/opt/java LANG=C.UTF-8"
              && CacheHits(first, extended) == 5 && CacheHits(first, unrelated) == 0
              && extended.Digest != first.Digest && extended.Reproducible,
              "digest chain matches the independent transcription (oracle) and appending a layer keeps the prefix cached");

        Image minimal = Build(Dockerfile("FROM scratch@sha256:" + Base), Base);
        Check(minimal.Layers.Count == 1 && minimal.Reproducible
              && minimal.Digest == Fnv(Hex(Fnv(Base, 0)) + "|from scratch@sha256:" + Base, 0),
              "FROM-only boundary image");

        Console.WriteLine("layers: " + first.Layers.Count + ", cache hits on append: " + CacheHits(first, extended)
                          + ", image digest: " + Hex(first.Digest));
        Console.WriteLine("PASS");
    }
}
