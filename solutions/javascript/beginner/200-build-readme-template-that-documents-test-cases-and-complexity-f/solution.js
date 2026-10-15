/** Task 200 - Build README template that documents test cases and complexity. (Tier: Beginner, CTCI Beyond) */
"use strict";

function cell(value) {
  return String(value).replace(/\|/g, "\\|").replace(/\n/g, " ");
}

function renderSection(name, tier, timeComplexity, spaceComplexity, testCases) {
  const lines = [
    `## ${name}`,
    "",
    `**Tier:** ${tier}`,
    "",
    "| Complexity | Value |",
    "| --- | --- |",
    `| Time | ${cell(timeComplexity)} |`,
    `| Space | ${cell(spaceComplexity)} |`,
    "",
    "| Input | Expected |",
    "| --- | --- |",
  ];
  for (const [input, expected] of testCases) {
    lines.push(`| \`${cell(input)}\` | \`${cell(expected)}\` |`);
  }
  return lines.join("\n").replace(/\n+$/, "");
}

module.exports = { renderSection };

if (require.main === module) {
  const assert = require("node:assert");

  const twoSum = renderSection("Two Sum", "Beginner", "O(n)", "O(n)", [["1, 2", "3"], ["[]", "0"]]);
  assert.strictEqual(twoSum, [
    "## Two Sum",
    "",
    "**Tier:** Beginner",
    "",
    "| Complexity | Value |",
    "| --- | --- |",
    "| Time | O(n) |",
    "| Space | O(n) |",
    "",
    "| Input | Expected |",
    "| --- | --- |",
    "| `1, 2` | `3` |",
    "| `[]` | `0` |",
  ].join("\n"));

  const escaped = renderSection("Escape | Demo", "Beginner", "O(1)", "O(1)",
    [["a | b", "c\nd"], ["x\ny", "z"]]);
  assert.strictEqual(escaped, [
    "## Escape | Demo",
    "",
    "**Tier:** Beginner",
    "",
    "| Complexity | Value |",
    "| --- | --- |",
    "| Time | O(1) |",
    "| Space | O(1) |",
    "",
    "| Input | Expected |",
    "| --- | --- |",
    "| `a \\| b` | `c d` |",
    "| `x y` | `z` |",
  ].join("\n"));

  const empty = renderSection("Empty Cases", "Beginner", "O(1)", "O(1)", []);
  assert.strictEqual(empty, [
    "## Empty Cases",
    "",
    "**Tier:** Beginner",
    "",
    "| Complexity | Value |",
    "| --- | --- |",
    "| Time | O(1) |",
    "| Space | O(1) |",
    "",
    "| Input | Expected |",
    "| --- | --- |",
  ].join("\n"));

  for (const section of [twoSum, escaped, empty]) assert.ok(!section.endsWith("\n"));
  console.log("PASS");
}
