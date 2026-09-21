import os
import re
import json
import time
import random
from collections import Counter
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()
# Models are tried in this order. When one is overloaded (503) we move on to the next one
# (with a single model, it just retries that one).
# Override in .env with GEMINI_MODELS="model-a,model-b,model-c".
DEFAULT_MODELS = [
    "gemini-3.5-flash",  # the model that answers reliably on this key
    # backups can be added here or via GEMINI_MODELS in .env, e.g. "gemini-3.1-flash-lite"
]
MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", ",".join(DEFAULT_MODELS)).split(",") if m.strip()]

ATTEMPTS_PER_MODEL = int(os.getenv("AI_MAX_RETRIES", "2"))  # tries on ONE model before moving to the next
RETRYABLE = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "504", "DEADLINE_EXCEEDED")

MAX_POST_CHARS = 600  # long posts are trimmed when analyzing

# ---- idea pipeline settings ----
TREND_SHARE = 0.6              # 60% of ideas remix trending topics, the rest are fresh
TOP_TOPICS = 3                 # how many trending topics feed the "trend remix" ideas
MIN_TOPIC_POSTS = 2            # a topic needs at least this many posts to count as a trend
RECENCY_HALF_LIFE_DAYS = 90    # a post this old counts half as much as a post from today
UNKNOWN_DATE_WEIGHT = 0.5      # weight for posts where no date could be read
KEYWORDS_PER_TOPIC = 4         # top keywords reported for each trending topic
MAX_SOURCE_POSTS = 5           # raw posts sent per trending topic (newest first)
SOURCE_POST_CHARS = 300        # each raw post is trimmed to this length

# main.py only saves topic/draft_copy/cta/image/keywords for ideas, so the post date,
# occasion and idea type are written into the topic, e.g. "[2026-11-05 | Diwali | Trend] Festive Buffet".
# If you later add columns to GeneratedIdea, set this to False.
EMBED_DATE_IN_TOPIC = True

# The scraper keeps the post date inside the content text, e.g. "Sep 12, 2026".
DATE_RE = re.compile(r"\b([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4})\b")


class AIProviderError(Exception):
    """Raised when the AI could not give a usable result. Catch it in the route."""


# ==========================================
# Schemas: the JSON shape we want back
# ==========================================

class PostAnalysis(BaseModel):
    post_number: int = Field(description="The post number exactly as given in the input (1, 2, 3...).")
    topic: str = Field(description="Short category label, e.g. 'Buffet Offer', 'Event Announcement', 'New Menu'.")
    keywords: List[str] = Field(description="2 to 5 marketing keywords from the post.")


class AnalysisResult(BaseModel):
    analysis: List[PostAnalysis]


class IdeaItem(BaseModel):
    idea_type: str = Field(description="Exactly 'trend_remix' or 'fresh'.")
    occasion: str = Field(description="Festival/occasion this idea is tied to, or 'Evergreen' if none.")
    suggested_post_date: str = Field(description="Best date to publish, format YYYY-MM-DD.")
    topic: str = Field(description="Core headline or topic of the post.")
    draft_copy: str = Field(description="Ready-to-publish Google Maps update copy, with emojis.")
    cta_suggested: str = Field(description="Google Maps CTA label, e.g. 'Book', 'Call now', 'Learn more', 'Order online'.")
    image_concept: str = Field(description="Short description of the picture or graphic to use.")
    keywords: List[str] = Field(description="2 to 4 target keywords used in the idea.")


class IdeasResult(BaseModel):
    ideas: List[IdeaItem]


# ==========================================
# One simple AI call with retries
# ==========================================

def _call_ai(prompt: str, schema, temperature: float):
    """
    Sends ONE request at a time and waits for Google's answer before doing anything else.
    If a model answers "busy", it retries that model once after a pause, then moves on to
    the next model in MODELS. Raises AIProviderError only when no model could answer.
    """
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    last_error = None
    for model in MODELS:
        for attempt in range(ATTEMPTS_PER_MODEL):
            started = time.time()
            try:
                print(f"[AI] {model}: sending {schema.__name__} request {attempt + 1}/{ATTEMPTS_PER_MODEL}")
                response = client.models.generate_content(model=model, contents=prompt, config=config)
                result = schema.model_validate_json(response.text)
                print(f"[AI] answered by {model}")
                return result
            except Exception as e:
                last_error = e
                msg = str(e)
                print(f"[AI] {model} answered with an error after {time.time() - started:.1f}s: "
                      f"{' '.join(msg.split())[:200]}")

                if "404" in msg or "NOT_FOUND" in msg:
                    print(f"[AI] {model} is not available for this key, skipping it")
                    break  # next model
                if not any(code in msg for code in RETRYABLE):
                    raise AIProviderError(msg) from e  # real error (auth, bad request...), no point trying others

                if attempt < ATTEMPTS_PER_MODEL - 1:
                    wait = min(5 * 2 ** attempt, 45) + random.uniform(0, 2)
                    print(f"[AI] {model} is busy, retrying it in {wait:.0f}s")
                    time.sleep(wait)
                else:
                    print(f"[AI] {model} is busy, trying the next model...")

    raise AIProviderError(f"No model could answer. Last error: {last_error}")


# ==========================================
# Step 1: analyze ALL posts in ONE call
# ==========================================

def analyze_posts(post_contents: List[str]) -> List[Dict[str, Any]]:
    """
    Sends every post in a single AI call and returns one {"topic", "keywords"} per post,
    in the same order. main.py saves them in the DB.
    Raises AIProviderError if the AI fails (nothing gets saved).
    """
    if not post_contents:
        return []

    joined = "\n\n".join(
        f"POST {i}:\n{text.strip()[:MAX_POST_CHARS]}" for i, text in enumerate(post_contents, start=1)
    )

    prompt = f"""
You are an expert local SEO and Google Maps marketing analyst.
Below are {len(post_contents)} Google Maps updates posted by competitors.

For EVERY post, give a short standard topic label and 2 to 5 marketing keywords.
Use the same label for the same kind of post (for example always 'Buffet Offer', never
'Buffet Offers' or 'Buffet Deal' in the same batch).
Use the post_number exactly as given in the input.

{joined}
"""

    result = _call_ai(prompt, AnalysisResult, temperature=0.2)  # the ONLY AI call for analysis

    by_number = {p.post_number: p for p in result.analysis}
    output = []
    missing = 0
    for n in range(1, len(post_contents) + 1):
        item = by_number.get(n)
        if item:
            output.append({"topic": item.topic, "keywords": item.keywords})
        else:
            missing += 1
            output.append({"topic": None, "keywords": []})  # left unanalyzed, not filled with fake data

    if missing == len(post_contents):
        raise AIProviderError("AI response did not contain any usable analysis.")
    if missing:
        print(f"[AI] {missing} post(s) were missing from the response and stay unanalyzed")
    return output


# ==========================================
# Step 2: idea pipeline (trend remix + fresh)
# ==========================================

def _keyword_list(value: Any) -> List[str]:
    """Keywords may be a list or the comma-separated string stored in the DB."""
    if not value:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [k.strip().lower() for k in value if k and k.strip()]


def _topic_key(topic: str) -> str:
    """'Buffet Offers' and 'buffet offer' count as the same topic."""
    return " ".join(topic.lower().split()).rstrip("s")


def _post_date(content: str) -> Optional[date]:
    m = DATE_RE.search(content or "")
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date()
    except ValueError:
        return None


def _strip_date(content: str) -> str:
    return DATE_RE.sub("", content or "", count=1).strip()


def _recency_weight(post_date: Optional[date], today: date) -> float:
    if post_date is None:
        return UNKNOWN_DATE_WEIGHT
    age_days = max((today - post_date).days, 0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def _find_trends(posts: List[Dict[str, Any]], today: date):
    """
    Plain Python, no AI:
      1. group posts by topic
      2. score each topic = sum of post weights (newer posts weigh more)
      3. for the top topics, find their keywords and pick the newest raw posts
    Returns (trend_groups, topic_counts, keyword_counts).
    """
    items = []
    for p in posts:
        content = p.get("content") or ""
        post_date = _post_date(content)
        items.append({
            "content": content,
            "topic": (p.get("topic") or "").strip(),
            "keywords": _keyword_list(p.get("keywords")),
            "date": post_date,
            "weight": _recency_weight(post_date, today),
        })

    if items and not any(i["date"] for i in items):
        print("[AI] no post dates found in the content, ranking topics by post count only")

    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    labels: Dict[str, Counter] = {}
    keyword_counts: Counter = Counter()
    for i in items:
        for kw in set(i["keywords"]):
            if len(kw) >= 3:
                keyword_counts[kw] += 1
        if not i["topic"]:
            continue
        key = _topic_key(i["topic"])
        by_topic.setdefault(key, []).append(i)
        labels.setdefault(key, Counter())[i["topic"]] += 1

    def label(key: str) -> str:
        return labels[key].most_common(1)[0][0]

    ranked = sorted(by_topic.items(), key=lambda kv: sum(i["weight"] for i in kv[1]), reverse=True)
    top = [kv for kv in ranked if len(kv[1]) >= MIN_TOPIC_POSTS][:TOP_TOPICS]
    if not top and ranked:
        top = ranked[:1]  # tiny data set: still use the single best topic

    groups = []
    for key, topic_items in top:
        kw_scores: Counter = Counter()
        for i in topic_items:
            for kw in set(i["keywords"]):
                if len(kw) >= 3:
                    kw_scores[kw] += i["weight"]

        if any(i["date"] for i in topic_items):
            newest = sorted(topic_items, key=lambda i: i["date"] or date.min, reverse=True)[:MAX_SOURCE_POSTS]
        else:
            newest = list(reversed(topic_items[-MAX_SOURCE_POSTS:]))  # DB order: last = newest

        dated = [i["date"] for i in topic_items if i["date"]]
        groups.append({
            "trending_topic": label(key),
            "posts_in_topic": len(topic_items),
            "trend_score": round(sum(i["weight"] for i in topic_items), 2),
            "latest_post_date": str(max(dated)) if dated else None,
            "top_keywords": [kw for kw, _ in kw_scores.most_common(KEYWORDS_PER_TOPIC)],
            "source_posts_newest_first": [_strip_date(i["content"])[:SOURCE_POST_CHARS] for i in newest],
        })

    topic_counts = {label(key): len(v) for key, v in sorted(by_topic.items(), key=lambda kv: -len(kv[1]))[:10]}
    return groups, topic_counts, dict(keyword_counts.most_common(15))


def _idea_label(idea: Dict[str, Any]) -> str:
    if not EMBED_DATE_IN_TOPIC:
        return idea["topic"]
    tags = [idea["suggested_post_date"]]
    if idea["occasion"] and idea["occasion"].strip().lower() != "evergreen":
        tags.append(idea["occasion"])
    tags.append("Trend" if idea["idea_type"] == "trend_remix" else "Fresh")
    return f"[{' | '.join(tags)}] {idea['topic']}"


def generate_ideas(
    posts: List[Dict[str, Any]],
    count: int = 5,
    avoid_topics: Optional[List[str]] = None,
    business_name: Optional[str] = None,
    festivals: Optional[List[Dict[str, str]]] = None,
    region: str = "India",
    days_ahead: int = 60,
) -> List[Dict[str, Any]]:
    """
    posts: analyzed posts from the DB, e.g.
           [{"content": p.content, "topic": p.topic, "keywords": p.keywords_detected}, ...]

    Pipeline (one AI call):
      1. Rank topics by number of posts, newer posts counting more (plain Python).
      2. ~60% of the ideas: rework the newest raw posts of the trending topics (and their
         top keywords) into NEW, unique posts.
      3. ~40% of the ideas: completely fresh angles competitors are not using.

    festivals: optional [{"name": "Diwali", "date": "2026-11-08"}]; otherwise the AI
               picks major occasions for `region` (double-check those dates).
    Returns ideas sorted by suggested post date. Raises AIProviderError on failure.
    """
    if not posts:
        raise AIProviderError("No posts available to base ideas on.")

    today = date.today()
    end = today + timedelta(days=days_ahead)

    groups, topic_counts, keyword_counts = _find_trends(posts, today)

    if groups:
        n_trend = min(count, max(1, round(count * TREND_SHARE)))
    else:
        n_trend = 0  # nothing analyzed yet, so every idea is fresh
    n_fresh = count - n_trend

    if festivals:
        festival_text = "Use these upcoming festivals/occasions:\n" + "\n".join(
            f"- {f['name']}: {f['date']}" for f in festivals
        )
    else:
        festival_text = (
            f"Use major festivals and occasions in {region} that fall between {today} and {end}. "
            f"Only use dates you are confident about."
        )

    part_a = ""
    if n_trend:
        part_a = f"""
PART A - TREND REMIX: create exactly {n_trend} ideas with idea_type "trend_remix".
Below are the topics competitors post about most. They are ranked by trend_score, which counts
how many posts use the topic and gives newer posts more weight, so the first topic is the hottest right now.
Take a trending topic and rework one of its source posts into a NEW post for our business.
Spread the ideas across the topics and give more ideas to the hottest topic. Use each topic's
top_keywords as the flavor of the idea.
- Keep what works (the kind of offer, the hook, the structure) but rewrite everything in new words.
- Never copy sentences, names, prices or distinctive phrases from the source posts.
- Add a clear unique twist competitors are not using: a fresh hook, a different audience
  (families, office groups, couples...), a signature detail, a time-limited element or a festival tie-in.

TRENDING TOPICS AND THEIR NEWEST POSTS:
{json.dumps(groups, ensure_ascii=False, indent=2)}
"""

    part_b = ""
    if n_fresh:
        part_b = f"""
PART B - FRESH: create exactly {n_fresh} ideas with idea_type "fresh".
These must be completely different from what competitors do. Stay away from these overused topics and keywords:
{json.dumps({"topics": topic_counts, "keywords": keyword_counts}, ensure_ascii=False)}
Think of angles nobody above covers: seasonal moments, behind-the-scenes, community, loyalty,
a specific audience, a new experience.
"""

    avoid_text = ", ".join(avoid_topics) if avoid_topics else "None"

    prompt = f"""
You are a local business marketing strategy engine for Google Maps update posts.

OUR BUSINESS: {business_name or "a local business"}
Today's date: {today}
Planning window: {today} to {end}

{festival_text}

Create {count} post ideas in total, in the parts below. Set idea_type on every idea.
{part_a}{part_b}
RULES FOR ALL IDEAS:
- Give every idea a suggested_post_date inside the planning window (YYYY-MM-DD). Tie ideas to the
  upcoming festivals/occasions where it fits and post a few days BEFORE the occasion.
  Use 'Evergreen' as the occasion for ideas not tied to a date.
- Do NOT reuse these previous topics (ignore any date tags in them): [{avoid_text}].
- Use realistic Google Maps CTA labels and give clear image direction.
- Do not invent facts about our business (prices, timings, dish names, addresses).
  Where a specific detail is needed, write a [PLACEHOLDER] instead.
"""

    result = _call_ai(prompt, IdeasResult, temperature=0.7)
    ideas = [i.model_dump() for i in result.ideas[:count]]

    got_trend = sum(1 for i in ideas if i["idea_type"] == "trend_remix")
    if got_trend != n_trend:
        print(f"[AI] wanted {n_trend} trend + {n_fresh} fresh ideas, got {got_trend} trend + {len(ideas) - got_trend} other")

    ideas.sort(key=lambda i: i["suggested_post_date"])

    return [
        {
            "topic": _idea_label(i),
            "draft_copy": i["draft_copy"],
            "cta_suggested": i["cta_suggested"],
            "image_concept": i["image_concept"],
            "keywords": i["keywords"],
            # extra keys, main.py ignores them today but they're there if you add columns later
            "idea_type": i["idea_type"],
            "occasion": i["occasion"],
            "suggested_post_date": i["suggested_post_date"],
        }
        for i in ideas
    ]