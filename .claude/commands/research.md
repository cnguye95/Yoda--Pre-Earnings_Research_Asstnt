---
description: Research a topic and weigh options against this repo. Use when the user asks to research an unfamiliar topic, compare tools/approaches, or wants options weighed before a decision — not for topics already well understood or single-answer factual lookups.
argument-hint: <topic or question>
allowed-tools: Read, Grep, Glob, WebSearch, WebFetch
---

<!--
Purpose: Speed up high-quality learning while coding with Claude Code;
counter shallow, lazy research.
Use when: researching unfamiliar topics before a decision.
-->

Research the following topic for me: $ARGUMENTS

Do this in 5 passes;
1. Define relevant core concepts: what they are, the problem solved, to give the rest grounding.

2. Web-search to map what's out there, THEN read deeper. Run searches to find what the field uses now, whether newly introduced, or long-established, then use WebFetch to open and read the full pages for the most relevant results — do not rely on search snippets alone. Snippets are for finding sources; fetch the sources and read them before characterizing anything. For each product or approach you describe, prefer a detail you confirmed by reading its page over a one-line summary from a search result.

3. Lay out main approaches/variants and what distinguishes them. If the topic genuinely has only one form, the definition is the deliverable and there's nothing more to enumerate.

4. Weigh realistic options against THIS repo: read the relevant code/structure and note where each option fits or fights what's here. Start by reading PLAN.md and README if they exist, then follow imports or folder structure to find code directly relevant to the topic.

5. Output a comparison table, then output a 2-3 item list with the open question for each. Stop here. Do NOT converge on a single recommendation or write an implementation plan. Stop here.



