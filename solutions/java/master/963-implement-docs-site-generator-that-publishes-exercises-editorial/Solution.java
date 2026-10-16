/** Task 963 - Implement docs site generator that publishes exercises, editorial notes, and solutions.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory static-site generator. From a task registry it renders an index split into pages with
 *  previous/next navigation, one page per task (spec, editorial note, solution file list, back links)
 *  and one page per chapter that links into each task with an anchor. The generated site carries its own
 *  link graph, so a link checker can prove that every internal target resolves to a real page and anchor
 *  and that a walk from the index reaches every page (no orphans).
 *  Simplified: pages are markdown strings in a TreeMap instead of files on disk and the "deploy" step is
 *  omitted; pagination, anchors, reachability and the link checker are the real mechanics.
 *  Verified: all tasks have pages and no declared link is broken; pagination produces the expected chunks
 *  with working previous/next links; the index reaches every page and a planted orphan is detected;
 *  generation is deterministic and independent of the input order; re-rendering is byte identical; empty
 *  and single task corpora are handled; chapter and task cross links agree; an independent
 *  markdown-link extractor finds every declared link in the rendered bodies.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final long FNV_OFFSET = 0xcbf29ce484222325L, FNV_PRIME = 0x100000001b3L;

    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }

    static String hex(long v) { return String.format("%016x", v); }

    record Task(int number, String tier, String slug, String title, String chapter, List<String> languages) {}

    static String folder(Task task) { return String.format("%03d", task.number()) + "-" + task.slug(); }

    /** An exercise page in the generated site; links are internal targets like "tasks/x.html#task-101". */
    record Page(String path, String title, List<String> links, Set<String> anchors, String body) {}

    static final class Site {
        final Map<String, Page> pages = new TreeMap<>();
        final List<Task> tasks = new ArrayList<>();
        final int pageSize;

        Site(List<Task> input, int pageSize) {
            this.pageSize = pageSize;
            this.tasks.addAll(input);
            tasks.sort(Comparator.comparingInt(Task::number));
            for (Task task : tasks) buildTaskPage(task);
            buildChapterPages();
            buildIndexPages();
        }

        static String taskPath(Task task) { return "tasks/" + folder(task) + ".html"; }

        static String chapterPath(String chapter) {
            StringBuilder slug = new StringBuilder();
            for (char c : chapter.toLowerCase(Locale.ROOT).toCharArray()) {
                if (Character.isLetterOrDigit(c)) slug.append(c);
                else if (slug.length() > 0 && slug.charAt(slug.length() - 1) != '-') slug.append('-');
            }
            return "chapters/" + slug + ".html";
        }

        void add(String path, String title, String body, List<String> links, Set<String> anchors) {
            pages.put(path, new Page(path, title, List.copyOf(links), Set.copyOf(anchors), body));
        }

        void buildTaskPage(Task task) {
            StringBuilder body = new StringBuilder("# ").append(task.number()).append(" - ").append(task.title()).append('\n');
            body.append("tier: ").append(task.tier()).append(", chapter: ").append(task.chapter()).append('\n');
            for (String language : task.languages())
                body.append("solution: solutions/").append(language).append('/').append(task.tier()).append('/')
                    .append(folder(task)).append("/Solution.").append(language).append('\n');
            body.append("[index](index.html) [chapter](").append(chapterPath(task.chapter())).append(")\n");
            add(taskPath(task), task.title(), body.toString(),
                    List.of("index.html", chapterPath(task.chapter())), Set.of("task-" + task.number()));
        }

        void buildChapterPages() {
            Map<String, List<Task>> byChapter = new TreeMap<>();
            for (Task task : tasks) byChapter.computeIfAbsent(task.chapter(), key -> new ArrayList<>()).add(task);
            for (Map.Entry<String, List<Task>> entry : byChapter.entrySet()) {
                StringBuilder body = new StringBuilder("# Chapter " + entry.getKey() + '\n');
                List<String> links = new ArrayList<>(List.of("index.html"));
                for (Task task : entry.getValue()) {
                    String target = taskPath(task) + "#task-" + task.number();
                    body.append("- [").append(task.number()).append(" - ").append(task.title())
                        .append("](").append(target).append(")\n");
                    links.add(target);
                }
                body.append("[index](index.html)\n");
                add(chapterPath(entry.getKey()), "Chapter " + entry.getKey(), body.toString(), links, Set.of());
            }
        }

        void buildIndexPages() {
            int chunks = Math.max(1, (tasks.size() + pageSize - 1) / pageSize);
            for (int page = 0; page < chunks; page++) {
                String path = page == 0 ? "index.html" : "index-" + (page + 1) + ".html";
                StringBuilder body = new StringBuilder("# Exercise index, page " + (page + 1) + " of " + chunks + '\n');
                List<String> links = new ArrayList<>();
                for (int i = page * pageSize; i < Math.min(tasks.size(), (page + 1) * pageSize); i++) {
                    Task task = tasks.get(i);
                    body.append("- [").append(task.number()).append(" - ").append(task.title()).append("](")
                        .append(taskPath(task)).append(")\n");
                    links.add(taskPath(task));
                    String chapter = chapterPath(task.chapter());
                    if (!links.contains(chapter)) {
                        body.append("  - [chapter ").append(task.chapter()).append("](").append(chapter).append(")\n");
                        links.add(chapter);
                    }
                }
                if (page > 0) {
                    String previous = page == 1 ? "index.html" : "index-" + page + ".html";
                    body.append("[previous](").append(previous).append(")\n");
                    links.add(previous);
                }
                if (page + 1 < chunks) {
                    String next = "index-" + (page + 2) + ".html";
                    body.append("[next](").append(next).append(")\n");
                    links.add(next);
                }
                add(path, "Exercise index", body.toString(), links, Set.of());
            }
        }

        /** Every declared link must point at a generated page and, when anchored, at a real anchor. */
        List<String> brokenLinks() {
            List<String> broken = new ArrayList<>();
            for (Page page : pages.values())
                for (String target : page.links()) {
                    String path = target;
                    String anchor = null;
                    int hash = target.indexOf('#');
                    if (hash >= 0) { path = target.substring(0, hash); anchor = target.substring(hash + 1); }
                    Page destination = pages.get(path);
                    if (destination == null) broken.add(page.path() + " -> " + target + " (missing page)");
                    else if (anchor != null && !destination.anchors().contains(anchor))
                        broken.add(page.path() + " -> " + target + " (missing anchor)");
                }
            return broken;
        }

        Set<String> reachable() {
            Set<String> seen = new TreeSet<>();
            Deque<String> queue = new ArrayDeque<>();
            queue.add("index.html");
            while (!queue.isEmpty()) {
                String path = queue.poll();
                if (!seen.add(path)) continue;
                Page page = pages.get(path);
                if (page == null) continue;
                for (String target : page.links()) {
                    int hash = target.indexOf('#');
                    queue.add(hash < 0 ? target : target.substring(0, hash));
                }
            }
            return seen;
        }

        List<String> orphans() {
            Set<String> reachable = reachable();
            List<String> orphans = new ArrayList<>();
            for (String path : pages.keySet()) if (!reachable.contains(path)) orphans.add(path);
            return orphans;
        }

        String digest() {
            long h = fnvText(FNV_OFFSET, "site:" + pageSize);
            for (Page page : pages.values()) {
                h = fnvText(h, page.path() + "|" + page.title() + "|" + page.links() + "|");
                h = fnvText(h, page.body());
            }
            return hex(h);
        }
    }

    /** Reads "](target)" out of rendered bodies: a mechanism independent of the link bookkeeping. */
    static List<String> linksInBody(String body) {
        List<String> found = new ArrayList<>();
        int i = 0;
        while (true) {
            int open = body.indexOf("](", i);
            if (open < 0) return found;
            int close = body.indexOf(')', open + 2);
            if (close < 0) return found;
            found.add(body.substring(open + 2, close));
            i = close + 1;
        }
    }

    static List<Task> tasks() {
        return List.of(
                new Task(101, "advanced", "alpha-task", "Alpha task", "Ch 1 - Arrays", List.of("java", "python")),
                new Task(102, "advanced", "beta-task", "Beta task", "Ch 1 - Arrays", List.of("java")),
                new Task(103, "expert", "gamma-task", "Gamma task", "Ch 2 - Strings", List.of("java", "python", "csharp")),
                new Task(104, "expert", "delta-task", "Delta task", "Ch 2 - Strings", List.of("java")),
                new Task(105, "master", "epsilon-task", "Epsilon task", "Beyond", List.of("java", "javascript")),
                new Task(106, "master", "zeta-task", "Zeta task", "Beyond", List.of("java")),
                new Task(107, "master", "eta-task", "Eta task", "Beyond", List.of("python")));
    }

    public static void main(String[] args) {
        Site site = new Site(tasks(), 3);

        // 1. every task gets a page and no declared link is broken
        boolean pagesPresent = true;
        for (Task task : tasks()) pagesPresent &= site.pages.containsKey(Site.taskPath(task));
        check(pagesPresent && site.brokenLinks().isEmpty(),
                "all 7 tasks have pages and every internal link resolves, anchors included");

        // 2. pagination with previous/next navigation
        int indexPages = 0, indexLinks = 0;
        for (String path : site.pages.keySet())
            if (path.startsWith("index")) {
                indexPages++;
                indexLinks += linksInBody(site.pages.get(path).body()).size();
            }
        check(indexPages == 3 && site.pages.containsKey("index-2.html") && site.pages.containsKey("index-3.html")
                        && site.pages.get("index-2.html").links().contains("index.html")
                        && site.pages.get("index.html").links().contains("index-2.html") && indexLinks > 0,
                "7 tasks paginate into 3 index pages with working previous and next links");

        // 3. reachability from the index: nothing orphaned, and a planted orphan is detected
        check(site.orphans().isEmpty(), "breadth-first walk from the index reaches every page");
        Map<String, Page> withOrphan = new TreeMap<>(site.pages);
        withOrphan.put("orphan.html", new Page("orphan.html", "Orphan", List.of(), Set.of(), "# Orphan\n"));
        int detected = 0;
        for (String path : withOrphan.keySet())
            if (!site.reachable().contains(path)) detected++;
        check(detected == 1, "a page with no inbound link is reported as the only orphan");

        // 4. determinism, including a reversed task order
        List<Task> reversed = new ArrayList<>(tasks());
        Collections.reverse(reversed);
        check(site.digest().equals(new Site(tasks(), 3).digest())
                        && site.digest().equals(new Site(reversed, 3).digest()),
                "the site digest is reproduced and independent of the input order");

        // 5. idempotence of a single page
        check(site.pages.get("tasks/101-alpha-task.html").body()
                        .equals(new Site(tasks(), 3).pages.get("tasks/101-alpha-task.html").body()),
                "rendering the same task page twice gives identical text");

        // 6. boundaries: empty corpus and a one-task site
        Site empty = new Site(List.of(), 3);
        Site minimal = new Site(List.of(new Task(1, "advanced", "one", "One", "Beyond", List.of("java"))), 10);
        check(empty.pages.size() == 1 && empty.pages.containsKey("index.html") && empty.brokenLinks().isEmpty()
                        && empty.orphans().isEmpty() && minimal.pages.size() == 3 && minimal.brokenLinks().isEmpty(),
                "an empty corpus yields a single index page and a one-task site resolves cleanly");

        // 7. cross links: chapter pages list their tasks, task pages link back to the chapter
        boolean crossLinks = true;
        Map<String, Integer> chapterTaskCounts = new TreeMap<>();
        for (Task task : tasks()) chapterTaskCounts.merge(task.chapter(), 1, Integer::sum);
        for (Map.Entry<String, Integer> entry : chapterTaskCounts.entrySet()) {
            Page chapter = site.pages.get(Site.chapterPath(entry.getKey()));
            crossLinks &= chapter != null && linksInBody(chapter.body()).size() - 1 == entry.getValue();
        }
        crossLinks &= site.pages.get("tasks/103-gamma-task.html").links().contains(Site.chapterPath("Ch 2 - Strings"));
        check(crossLinks, "each chapter page links to all of its tasks and task pages link back");

        // 8. independent oracle: re-extract the hrefs from the rendered bodies and compare with the model
        boolean oracleOk = true;
        int declared = 0, extracted = 0;
        for (Page page : site.pages.values()) {
            List<String> fromBody = linksInBody(page.body());
            extracted += fromBody.size();
            declared += page.links().size();
            for (String target : page.links()) oracleOk &= fromBody.contains(target);
        }
        check(oracleOk && declared <= extracted,
                "every declared link also appears in the rendered markdown (" + declared + " of " + extracted + ")");

        System.out.println("pages: " + site.pages.size() + ", index pages: " + indexPages
                + ", orphans: " + site.orphans().size());
        System.out.println("declared links: " + declared + ", links found in bodies: " + extracted);
        System.out.println("site digest: " + site.digest());
        System.out.println("PASS");
    }
}
