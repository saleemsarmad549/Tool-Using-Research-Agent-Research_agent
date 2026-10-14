"""
Research agent: answers a question by calling a search tool and citing sources.

Setup:
    pip install anthropic requests
    export ANTHROPIC_API_KEY=...
    export TAVILY_API_KEY=...        # or swap search_web() for any search API

Run:
    python research_agent.py "What are the main causes of the 2008 financial crisis?"
"""
import json
import os
import re
import sys

import anthropic
import requests

MODEL = "claude-sonnet-5-5"
MAX_TURNS = 8          # hard cap on agent loop iterations
RESULTS_PER_SEARCH = 5

SYSTEM_PROMPT = """You are a careful research assistant.

Process:
1. Break the question into what you need to find out, then call `web_search`
   (several times, with different queries, if needed). Refine queries when
   results are thin or off-target.
2. Base every factual claim ONLY on the search results you received.
3. Cite sources inline using their ids, e.g. "Inflation fell in 2024 [2]."
   Use only ids that appeared in search results. Never invent ids or URLs.
4. If sources conflict, say so and cite both. If the results don't answer the
   question, say what is missing instead of guessing.
5. Paraphrase in your own words; do not copy long passages.
6. Finish with a concise answer. Do not include a sources list; it is
   appended automatically."""

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web. Returns a numbered list of results, each with an "
            "id, title, url, and a content snippet. Use short, specific queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    }
]


class SourceRegistry:
    """Assigns a stable id to each unique URL seen, so citations are checkable."""

    def __init__(self):
        self.by_url: dict[str, int] = {}
        self.sources: dict[int, dict] = {}

    def add(self, title: str, url: str) -> int:
        if url not in self.by_url:
            sid = len(self.by_url) + 1
            self.by_url[url] = sid
            self.sources[sid] = {"title": title, "url": url}
        return self.by_url[url]


def search_web(query: str, n: int = RESULTS_PER_SEARCH) -> list[dict]:
    """Swap this function to use a different search provider."""
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": os.environ["TAVILY_API_KEY"],
            "query": query,
            "max_results": n,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in resp.json().get("results", [])
    ]


def run_search_tool(query: str, registry: SourceRegistry) -> str:
    try:
        results = search_web(query)
    except Exception as e:  # return errors to the model so it can adapt
        return f"Search failed: {e}"
    if not results:
        return "No results found. Try a different query."
    lines = []
    for r in results:
        sid = registry.add(r["title"], r["url"])
        lines.append(
            f"[{sid}] {r['title']}\nURL: {r['url']}\n{r['content']}\n"
        )
    return "\n".join(lines)


def format_answer(text: str, registry: SourceRegistry) -> str:
    """Validate citations and append a sources list of only the cited ones."""
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    invalid = cited - registry.sources.keys()
    if invalid:
        text += (
            f"\n\n(Warning: the answer cited unknown source ids "
            f"{sorted(invalid)}; treat those claims as unverified.)"
        )
    valid = sorted(cited & registry.sources.keys())
    if not valid:
        return text + "\n\n(No sources were cited.)"
    sources = "\n".join(
        f"[{i}] {registry.sources[i]['title']} - {registry.sources[i]['url']}"
        for i in valid
    )
    return f"{text}\n\nSources:\n{sources}"


def research(question: str, verbose: bool = True) -> str:
    client = anthropic.Anthropic()
    registry = SourceRegistry()
    messages = [{"role": "user", "content": question}]

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return format_answer(text, registry)

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"[turn {turn + 1}] search: {block.input.get('query')}",
                          file=sys.stderr)
                output = run_search_tool(block.input["query"], registry)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    # Out of turns: force a final answer from what we have, with no more tools.
    messages.append(
        {
            "role": "user",
            "content": "Search budget exhausted. Answer now using only the "
                       "results gathered so far, with citations, and note any gaps.",
        }
    )
    response = client.messages.create(
        model=MODEL, max_tokens=2000, system=SYSTEM_PROMPT, messages=messages,
        tools=TOOLS, tool_choice={"type": "none"},
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return format_answer(text, registry)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Research question: ")
    print(research(q))
