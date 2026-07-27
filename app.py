import os
import operator
import json
import asyncio
import argparse
import sqlite3
import difflib
import time
import re
from datetime import datetime, UTC
import tweepy
from typing import TypedDict, Annotated, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph, START, END

load_dotenv()

print("[DEBUG] Checking LangSmith environment variables...")
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and not os.getenv("LANGCHAIN_API_KEY"):
    raise RuntimeError(
        "LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is not set. "
        "Get a key at https://smith.langchain.com/settings"
    )
os.environ.setdefault("LANGCHAIN_PROJECT", "twitter-agent")

print("[DEBUG] Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=1.0,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

MAX_RETRIES = 1
SIMILARITY_THRESHOLD = 0.6
DB_PATH = "tweet_history.db"


def init_db(path: str = DB_PATH) -> None:
    print(f"[DEBUG] Initializing database at {path}...")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_text TEXT NOT NULL,
            tags TEXT,
            tweet_id TEXT,
            source_url TEXT,
            published_at TEXT NOT NULL
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(tweets)")}
    if "source_url" not in existing_cols:
        print("[DEBUG] Adding source_url column to existing DB...")
        conn.execute("ALTER TABLE tweets ADD COLUMN source_url TEXT")
    conn.commit()
    conn.close()
    print("[DEBUG] Database initialization complete.")


def get_recent_tweets(limit: int = 50, path: str = DB_PATH) -> list[str]:
    print(f"[DEBUG] Fetching up to {limit} recent tweets from DB...")
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT tweet_text FROM tweets ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    print(f"[DEBUG] Retrieved {len(rows)} recent tweets.")
    return [r[0] for r in rows]


def record_tweet(tweet_text: str, tags: list[str], tweet_id: str, source_url: str = None, path: str = DB_PATH) -> None:
    print(f"[DEBUG] Recording published tweet to DB (ID: {tweet_id})...")
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO tweets (tweet_text, tags, tweet_id, source_url, published_at) VALUES (?, ?, ?, ?, ?)",
        (tweet_text, " ".join(tags) if tags else "", tweet_id, source_url, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    conn.close()
    print("[DEBUG] Tweet recorded successfully.")


def most_similar(tweet_text: str, recent: list[str]) -> float:
    print("[DEBUG] Calculating similarity against recent tweets...")
    if not recent:
        print("[DEBUG] No recent tweets to compare against.")
        return 0.0
    score = max(
        difflib.SequenceMatcher(None, tweet_text.lower(), r.lower()).ratio()
        for r in recent
    )
    print(f"[DEBUG] Highest similarity score: {score:.2f}")
    return score


class AgentState(TypedDict):
    research_notes: Annotated[list[dict], operator.add]
    story_suggestions: str
    tweet_draft: str
    critique: str
    source_url: Optional[str]
    tags: list[str]
    retries: Annotated[int, operator.add]
    dedup_note: str
    published_id: Optional[str]


RESEARCH_QUERIES = [
    "trending news India today",
    "trending news USA today",
    "finance markets news today",
    "cricket news today",
    "technology news today",
    "entertainment pop culture news today",
    "trending news politics today",
    "trending news in football"
]


def research_node(state: AgentState) -> dict:
    print("\n[DEBUG] === ENTERING NODE: research ===")
    search = DuckDuckGoSearchAPIWrapper()
    notes: list[dict] = []
    seen_titles = set()
    for i, query in enumerate(RESEARCH_QUERIES):
        print(f"[DEBUG] Executing search query: '{query}'")
        if i > 0:
            time.sleep(3)
        for r in search.results(query, max_results=2):
            title = r.get("title", "").strip()
            snippet = r.get("snippet", "").strip()
            link = r.get("link", "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            notes.append({"title": title, "snippet": snippet, "link": link})
    print(f"[DEBUG] Research complete. Found {len(notes)} unique stories.")
    return {"research_notes": notes}


def get_text(response):
    print("[DEBUG] Extracting text from LLM response...")
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            elif hasattr(part, "text") and part.text:
                parts.append(part.text)
            else:
                parts.append(str(part))
        return "".join(parts).strip()
    return str(content).strip()


def _format_notes(notes: list[dict]) -> str:
    return "\n".join(
        f"{i+1}. {n['title']}: {n['snippet']}" for i, n in enumerate(notes)
    )


def _identify_source_url(tweet_text: str, notes: list[dict]) -> Optional[str]:
    """Helper to match generated tweet to news source without changing user prompt."""
    if not notes or not tweet_text:
        return None
    notes_text = _format_notes(notes)
    prompt = (
        f"Here are news items:\n{notes_text}\n\n"
        f"Which news item (1 to {len(notes)}) is this tweet reacting to?\n"
        f"Tweet: \"{tweet_text}\"\n\n"
        f"Respond with ONLY the single number of the news item. If it does not relate to any, respond with 0."
    )
    try:
        print("[DEBUG] Identifying matching source news link for tweet...")
        response = llm.invoke([HumanMessage(content=prompt)])
        text = get_text(response).strip()
        numbers = re.findall(r'\d+', text)
        if numbers:
            idx = int(numbers[0])
            if 1 <= idx <= len(notes):
                matched_url = notes[idx - 1].get("link") or None
                print(f"[DEBUG] Matched news source URL (Item #{idx}): {matched_url}")
                return matched_url
    except Exception as e:
        print(f"[DEBUG] Could not identify source URL: {e}")
    return None


GENERATE_PROMPT = """You are the voice behind a brand's X account: whip-smart, \
sarcastic, unafraid of a strong opinion, and occasionally just flat-out fed up \
with something dumb happening in the news. Think a columnist who's done \
pretending to be neutral - dry wit, sharp timing, willing to call something \
bad "bad" without hedging.

Based on the research notes below, pick ONE timely story and write a single \
tweet (strictly under 200 chars) with a real point of view on it.

Your opinions should be hot takes, not generic platitudes.

Go for: cutting one-liners, a well-earned "come on" at bad performance or \
decisions, calling out hypocrisy or nonsense directly, dark or dry humor, \
genuine irritation when something deserves it. Be specific and vivid rather \
than generic - name the actual decision, stat, or event you're reacting to.

For eg: If tweet is about criticizing kohli, then you can use keywords like best NRI cricketer or\
calling out a person then you can use your existing knowledge to target that person/event\
positively or negatively as per the tweet requires.

For eg YOU are tweeting in support of abortion, then use keywords that will appeal to feminist\
even though they may offend some conservatives.

Dont care about offending anyone, be unapologetic about your humour and hot takes.
You can use Dark humour, be unapologetic about your humour and hot takes. Use your\
imagination to fullest to maximize interaction by user.
Try to generate humanlike tweets that are not generic, bland, or formulaic.
The Twitter account is based in india
Today's news items (pick ONE to tweet about):
{notes}

{revision_note}

{suggestions_block}

Respond with ONLY the tweet text, nothing else."""


SUGGESTION_PROMPT = """You are the voice behind a brand's X account: whip-smart, \
sarcastic, unafraid of a strong opinion, and occasionally just flat-out fed up \
with something dumb happening in the news. Think a columnist who's done \
pretending to be neutral - dry wit, sharp timing, willing to call something \
bad "bad" without hedging.

Filter stories from the research notes below, pick controversial topics that are likely to generate engagement,\
and propose a specific angle for each. The angle should be like a suggestion for how to frame the story in a punchy, \
opinionated way. Suggesting spicy keywords like manchild or best NRI cricketer \
to criticize kohli or calling out a person then you can use your existing knowledge to target that person/event\
positively or negatively as per the tweet requires. If the story is about abortion, then use keywords that will appeal to feminist even though they may offend some conservatives.

Dont care about offending anyone, be unapologetic about your humour and hot takes.
You can use Dark humour, be unapologetic about your humour and hot takes. Use your\
imagination to fullest to maximize interaction by user.

News items:
{notes}

Respond with a short numbered list: story, then a one-line suggested angle."""


def suggestion_node(state: AgentState) -> dict:
    print("\n[DEBUG] === ENTERING NODE: suggestion ===")
    numbered_notes = _format_notes(state["research_notes"])
    print("[DEBUG] Formatted notes for suggestion prompt. Invoking LLM...")
    time.sleep(2)
    prompt = SUGGESTION_PROMPT.format(notes=numbered_notes)
    response = llm.invoke([HumanMessage(content=prompt)])
    time.sleep(2)
    content = get_text(response)
    print(f"[DEBUG] Story suggestions generated:\n{content.strip()}")
    return {"story_suggestions": content.strip()}


def generate_node(state: AgentState) -> dict:
    print("\n[DEBUG] === ENTERING NODE: generate ===")
    if state.get("critique") or state.get("dedup_note"):
        print("[DEBUG] Previous critique or dedup note detected. Sleeping 2s...")
        time.sleep(2)
        
    revision_note = ""
    if state.get("critique"):
        revision_note = f"Previous critique to address:\n{state['critique']}"
        print(f"[DEBUG] Revision note: {revision_note}")
    if state.get("dedup_note"):
        revision_note += f"\n{state['dedup_note']}"
        print(f"[DEBUG] Dedup note: {state['dedup_note']}")

    notes = state["research_notes"]
    suggestions_block = ""
    if state.get("story_suggestions"):
        suggestions_block = f"Suggested angles (for inspiration only):\n{state['story_suggestions']}"

    prompt = GENERATE_PROMPT.format(
        notes=_format_notes(notes),
        suggestions_block=suggestions_block,
        revision_note=revision_note,
    )
    
    print("[DEBUG] Invoking LLM to generate tweet draft...")
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = get_text(response)
    print(f"[DEBUG] Raw LLM Draft Output:\n{raw}")

    tweet_text = raw
    if isinstance(tweet_text, list):
        tweet_text = " ".join(str(x) for x in tweet_text)

    source_url = _identify_source_url(tweet_text, notes)

    print(f"[DEBUG] Final processed tweet draft: {tweet_text}")
    return {"tweet_draft": str(tweet_text), "source_url": source_url, "dedup_note": ""}


CRITIQUE_PROMPT = """You are an editor for a sarcastic, opinionated brand \
voice on X, reviewing this draft tweet:

"{tweet}"

Push it to be sharper: is the line as punchy and quotable as it could be? \
Is the sarcasm landing, or does it feel flat/generic? Would a specific \
detail, a sharper verb, or a more pointed comparison make it hit harder?

If it's already sharp, specific, engaging and clean on that front, respond with \
exactly: APPROVED
Otherwise, give specific, actionable feedback for revision."""


def critique_node(state: AgentState) -> dict:
    print("\n[DEBUG] === ENTERING NODE: critique ===")
    draft = state.get("tweet_draft", "")

    if isinstance(draft, list):
        print("[DEBUG] Draft was a list, casting to string.")
        draft = " ".join(str(x) for x in draft)

    prompt = CRITIQUE_PROMPT.format(tweet=draft)
    print("[DEBUG] Invoking LLM for critique...")
    time.sleep(2)
    response = llm.invoke([HumanMessage(content=prompt)])
    time.sleep(2)
    critique = get_text(response)
    print(f"[DEBUG] LLM Critique result: {critique}")
    
    final_critique = "" if critique.strip() == "APPROVED" else critique
    print(f"[DEBUG] Returning from critique with retries increment: 1")
    return {
        "critique": final_critique,
        "retries": 1,
    }


def should_revise(state: AgentState) -> str:
    print("\n[DEBUG] === ROUTING: should_revise ===")
    retries = state.get("retries", 0)
    has_critique = bool(state.get("critique"))
    
    print(f"[DEBUG] Current State -> retries: {retries}, MAX_RETRIES: {MAX_RETRIES}, Needs Critique: {has_critique}")
    
    if has_critique and retries < MAX_RETRIES:
        print("[DEBUG] Decision: Routing to 'generate'")
        return "generate"
    
    print("[DEBUG] Decision: Routing to 'tag' (Either APPROVED or max retries reached)")
    return "tag"


TAG_PROMPT = """Suggest 2-4 relevant, non-spammy hashtags for this tweet:

"{tweet}"

Respond with ONLY the hashtags, space-separated, nothing else."""


def tag_node(state: AgentState) -> dict:
    print("\n[DEBUG] === ENTERING NODE: tag ===")
    draft = state.get("tweet_draft", "")
    if isinstance(draft, list):
        draft = " ".join(str(x) for x in draft)

    prompt = TAG_PROMPT.format(tweet=draft)
    print("[DEBUG] Invoking LLM for hashtags...")
    response = llm.invoke([HumanMessage(content=prompt)])
    tags = get_text(response).split()
    print(f"[DEBUG] Generated tags: {tags}")
    return {"tags": tags}


def dedup_check_node(state: AgentState) -> dict:
    print("\n[DEBUG] === ENTERING NODE: dedup_check ===")
    draft = state.get("tweet_draft", "")
    if isinstance(draft, list):
        draft = " ".join(str(x) for x in draft)

    recent = get_recent_tweets(limit=50)
    score = most_similar(draft, recent)
    
    if score >= SIMILARITY_THRESHOLD:
        note = (
            f"Your last draft was too similar (similarity={score:.2f}) "
            "to a tweet already posted recently. Pick a different topic "
            "or a substantially different angle."
        )
        print(f"[DEBUG] Dedup failed! Returning dedup note: {note}")
        return {
            "dedup_note": note,
            "retries": 1,
        }
        
    print("[DEBUG] Dedup passed successfully.")
    return {"dedup_note": ""}


def should_retry_dedup(state: AgentState) -> str:
    print("\n[DEBUG] === ROUTING: should_retry_dedup ===")
    retries = state.get("retries", 0)
    has_dedup = bool(state.get("dedup_note"))
    
    print(f"[DEBUG] Current State -> retries: {retries}, MAX_RETRIES: {MAX_RETRIES}, Dedup Note: {has_dedup}")
    
    if has_dedup and retries < MAX_RETRIES:
        print("[DEBUG] Decision: Routing to 'generate' to fix dedup issue")
        return "generate"
        
    print("[DEBUG] Decision: Routing to 'publish'")
    return "publish"


def publish_node(state: AgentState) -> dict:
    print("\n[DEBUG] === ENTERING NODE: publish ===")
    tweet_text = state.get("tweet_draft", "")
    if isinstance(tweet_text, list):
        tweet_text = " ".join(str(x) for x in tweet_text)

    # 1. Append Tags
    if state.get("tags"):
        tags = state["tags"]
        if isinstance(tags, list):
            tweet_text = f"{tweet_text} {' '.join(tags)}"
        else:
            tweet_text = f"{tweet_text} {tags}"

    # 2. Append URL
    url = state.get("source_url")
    if url:
        tweet_text = f"{tweet_text} {url}"

    # 3. STRICT LENGTH ENFORCEMENT (Preserves URL if present)
    if len(tweet_text) > 280:
        print(f"[DEBUG] Tweet is {len(tweet_text)} chars. Truncating to 280...")
        if url and url in tweet_text:
            text_without_url = tweet_text.replace(f" {url}", "").replace(url, "")
            allowed_len = 280 - 24 - 4
            tweet_text = f"{text_without_url[:allowed_len]}... {url}"
        else:
            tweet_text = tweet_text[:277] + "..."

    print(f"[DEBUG] Final payload targeting X API ({len(tweet_text)} chars):\n{tweet_text}")

    client = tweepy.Client(
        consumer_key=os.getenv("TWITTER_API_KEY"),
        consumer_secret=os.getenv("TWITTER_API_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
    )

    try:
        print("[DEBUG] Dispatching to Tweepy V2 Client...")
        response = client.create_tweet(text=tweet_text)
        published_id = str(response.data["id"])
        print(f"[DEBUG] Tweet published! X API ID: {published_id}")

        record_tweet(
            tweet_text=tweet_text,
            tags=state.get("tags", []),
            tweet_id=published_id,
            source_url=state.get("source_url"),
        )
        return {"published_id": published_id}
    
    except tweepy.errors.Forbidden as e:
        print(f"[DEBUG] 403 Forbidden! Twitter rejected the content.")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[DEBUG] Twitter API Exact Reason: {e.response.text}")
        return {"published_id": None}
        
    except Exception as e:
        print(f"[DEBUG] Failed to post to Twitter. Exception details: {e}")
        return {"published_id": None}


def build_graph():
    print("[DEBUG] Constructing LangGraph architecture...")
    graph = StateGraph(AgentState)

    graph.add_node("research", research_node)
    graph.add_node("suggestion", suggestion_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.add_node("tag", tag_node)
    graph.add_node("dedup_check", dedup_check_node)
    graph.add_node("publish", publish_node)

    graph.add_edge(START, "research")
    graph.add_edge("research", "suggestion")
    graph.add_edge("suggestion", "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique", should_revise, {"generate": "generate", "tag": "tag"}
    )
    graph.add_edge("tag", "dedup_check")
    graph.add_conditional_edges(
        "dedup_check",
        should_retry_dedup,
        {"generate": "generate", "publish": "publish"},
    )
    graph.add_edge("publish", END)

    print("[DEBUG] LangGraph construction complete.")
    return graph


async def run_once():
    print(f"\n[DEBUG] ======= STARTING NEW RUN: {datetime.now(UTC).isoformat()} =======")
    init_db()
    app = build_graph().compile()

    run_config = {
        "run_name": f"twitter-agent-{datetime.now(UTC).strftime('%Y-%m-%d')}",
        "tags": ["twitter-agent", "autonomous-daily-run"],
        "metadata": {"run_date": datetime.now(UTC).isoformat()},
    }

    print("[DEBUG] Invoking app.ainvoke()...")
    result = await app.ainvoke(
        {"retries": 0, "dedup_note": "", "research_notes": []},
        config=run_config,
    )
    print(f"\n[DEBUG] App execution finished. Analyzing results...")

    if not result.get("published_id") and result.get("tweet_draft"):
        print(
            f"[{datetime.now(UTC).isoformat()}] Hit retry limit without a "
            "fully 'clean' draft - publishing the last version anyway."
        )
        try:
            publish_result = publish_node(result)
            result = {**result, **publish_result}
        except Exception as e:
            print(f"[{datetime.now(UTC).isoformat()}] Fallback publish also failed: {e}")

    if result.get("published_id"):
        print(f"[{datetime.now(UTC).isoformat()}] Published tweet ID: {result['published_id']}")
        print(f"Text: {result['tweet_draft']}")
    else:
        print(f"[{datetime.now(UTC).isoformat()}] Did not publish (no usable draft was produced).")

    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        print(f"LangSmith project: {project} (see https://smith.langchain.com)")
    return result


async def run_loop(interval_seconds: int = 24 * 60 * 60):
    print(f"[DEBUG] Starting run_loop with interval {interval_seconds} seconds.")
    while True:
        try:
            await run_once()
        except Exception as e:
            print(f"[{datetime.now(UTC).isoformat()}] Run failed: {e}")
        print(f"[DEBUG] Sleeping for {interval_seconds} seconds...")
        await asyncio.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run continuously, once every 24h")
    args = parser.parse_args()

    if args.loop:
        asyncio.run(run_loop())
    else:
        asyncio.run(run_once())
