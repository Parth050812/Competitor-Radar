import os
import re
import json
import time
import base64
import random
import hashlib
from collections import Counter
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from io import BytesIO

load_dotenv()

# ---------------------------------------------------------------------------
# Gemini client / model selection
#
# The old version hard-coded one model and treated every 503 as "model busy".
# A 503 can also be a temporary capacity/routing problem even when the model
# is valid for the API key.  We therefore:
#   1. honour explicit GEMINI_MODELS/GEMINI_MODEL settings;
#   2. discover usable generateContent models from the API when no list is set;
#   3. rotate to another model immediately on 503/429/5xx;
#   4. temporarily cool down a model that just failed;
#   5. validate the response before returning it.
# ---------------------------------------------------------------------------

_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=_API_KEY) if _API_KEY else genai.Client()

# ---------------------------------------------------------------------------
# Hugging Face image generation
# ---------------------------------------------------------------------------
# Image generation is intentionally separate from the Gemini text pipeline.
# Each GeneratedIdea calls this function only when the user clicks
# "Generate Image" (or "Regenerate Image").
#
# .env:
#   HF_TOKEN=hf_...
# Optional:
#   HF_IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
#   HF_IMAGE_PROVIDER=auto
# ---------------------------------------------------------------------------

_HF_TOKEN = os.getenv("HF_TOKEN")
HF_IMAGE_MODEL = os.getenv(
    "HF_IMAGE_MODEL",
    "black-forest-labs/FLUX.1-schnell",
)
HF_IMAGE_PROVIDER = os.getenv("HF_IMAGE_PROVIDER", "auto")

image_client = (
    InferenceClient(
        provider=HF_IMAGE_PROVIDER,
        api_key=_HF_TOKEN,
    )
    if _HF_TOKEN
    else None
)
# Optional explicit model order:
# GEMINI_MODELS="gemini-2.5-flash,gemini-2.0-flash"
# or:
# GEMINI_MODEL="gemini-2.5-flash"
#
# If neither is supplied, the code discovers models available to this key.
EXPLICIT_MODELS = [
    m.strip()
    for m in (
        os.getenv("GEMINI_MODELS")
        or os.getenv("GEMINI_MODEL")
        or ""
    ).split(",")
    if m.strip()
]

# Models are cached for this process. A failed model is temporarily skipped.
_MODEL_CACHE: Optional[List[str]] = None
_MODEL_COOLDOWN_UNTIL: Dict[str, float] = {}

MODEL_DISCOVERY_TTL = int(os.getenv("GEMINI_MODEL_CACHE_SECONDS", "900"))
MODEL_COOLDOWN_503 = int(os.getenv("GEMINI_503_COOLDOWN_SECONDS", "30"))
MODEL_COOLDOWN_429 = int(os.getenv("GEMINI_429_COOLDOWN_SECONDS", "60"))

# Number of attempts across the model pool, not repeated attempts on one
# model. This avoids getting stuck on a model returning 503.
MAX_MODEL_ATTEMPTS = max(1, int(os.getenv("AI_MAX_RETRIES", "4")))

RETRYABLE_STATUS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "INTERNAL",
    "DEADLINE_EXCEEDED",
    "SERVICE_UNAVAILABLE",
)

MAX_POST_CHARS = 600


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
    topic: str = Field(description="Short standard category label, e.g. 'Buffet Offer', 'Event Announcement', 'New Menu'.")
    sub_topic: str = Field(description="A specific, useful sub-topic inside the main topic, e.g. 'Year-end family buffet', 'Gift card pricing', 'New seafood dish'.")
    content_type: str = Field(description="Content format/purpose, e.g. 'Offer', 'Announcement', 'New Menu', 'Event', 'Engagement', 'General'.")
    offer_pattern: str = Field(description="Specific offer/promotion mechanic if present, otherwise 'None'.")
    cta: str = Field(description="The call-to-action used or implied by the post, e.g. 'Book', 'Order online', 'Learn more', 'Call now', otherwise 'None'.")
    keywords: List[str] = Field(description="2 to 5 marketing keywords from the post.")


class AnalysisResult(BaseModel):
    analysis: List[PostAnalysis]


class IdeaItem(BaseModel):
    idea_type: str = Field(description="Exactly 'trend_remix' or 'fresh'.")
    source_trend: str = Field(description="For trend_remix, the exact trending topic label supplied in the trend analysis. For fresh, use 'None'.")
    occasion: str = Field(description="Festival/occasion this idea is tied to, or 'Evergreen' if none.")
    suggested_post_date: str = Field(description="Best date to publish, format YYYY-MM-DD.")
    topic: str = Field(description="Core headline or topic of the post.")
    draft_copy: str = Field(description="Ready-to-publish Google Maps update copy, with emojis. Naturally mention the exact business name once when it fits, so the user can paste the copy directly.")
    cta_suggested: str = Field(description="Google Maps CTA label, e.g. 'Book', 'Call now', 'Learn more', 'Order online'.")
    image_concept: str = Field(description="Concrete, self-contained visual direction that MUST match the draft copy: exact hero subject, setting, people/hands if needed, composition, and important visual details. Never suggest an unrelated generic food scene.")
    keywords: List[str] = Field(description="2 to 4 target keywords used in the idea.")
    focus_keywords_used: List[str] = Field(default_factory=list, description="Exact user focus keywords that this idea is built around. If focus keywords are supplied, include at least one relevant exact focus keyword here.")


class IdeasResult(BaseModel):
    ideas: List[IdeaItem]


# ==========================================
# One simple AI call with retries
# ==========================================

def _model_name(model: Any) -> str:
    """Return a model name accepted by generate_content()."""
    name = getattr(model, "name", None) or str(model)
    return name.removeprefix("models/").strip()


def _supports_generate_content(model: Any) -> bool:
    """
    The SDK has changed the exact shape of model metadata across releases, so
    inspect it defensively rather than depending on one SDK version.
    """
    actions = getattr(model, "supported_actions", None)

    if actions:
        try:
            actions = [str(a).lower() for a in actions]
            if not any(
                "generatecontent" in a or "generate_content" in a
                for a in actions
            ):
                return False
        except Exception:
            pass

    name = _model_name(model).lower()

    # Never select embedding / tokenizer / pure image or audio models for text.
    blocked = (
        "embedding",
        "text-embedding",
        "aqa",
        "imagen",
        "veo",
        "tts",
        "speech",
    )
    return not any(part in name for part in blocked)


def _model_priority(name: str) -> tuple:
    """
    Prefer Flash models for this application because these requests are short
    and frequent. Discovery still determines what is actually available.
    """
    n = name.lower()

    if "flash" in n and "lite" in n:
        return (0, n)
    if "flash" in n:
        return (1, n)
    if "pro" in n:
        return (2, n)
    return (3, n)


def _discover_models(force: bool = False) -> List[str]:
    """
    Ask Gemini which models this API key can see.

    This is the important difference from the old implementation: we don't
    assume that a hard-coded model name is usable just because it exists in
    documentation.
    """
    global _MODEL_CACHE

    if EXPLICIT_MODELS:
        return [
            m.removeprefix("models/").strip()
            for m in EXPLICIT_MODELS
            if m.strip()
        ]

    now = time.time()

    if (
        not force
        and _MODEL_CACHE is not None
        and _MODEL_CACHE
        and _MODEL_CACHE[0] != "__DISCOVERY_FAILED__"
    ):
        return list(_MODEL_CACHE)

    try:
        discovered = []

        for model in client.models.list():
            if not _supports_generate_content(model):
                continue

            name = _model_name(model)

            if name and name not in discovered:
                discovered.append(name)

        discovered.sort(key=_model_priority)

        if discovered:
            _MODEL_CACHE = discovered
            print(
                "[AI] discovered models: "
                + ", ".join(discovered[:12])
                + (" ..." if len(discovered) > 12 else "")
            )
            return list(discovered)

        raise RuntimeError("Gemini API returned no usable generateContent models.")

    except Exception as e:
        # If discovery itself is unavailable, use a conservative fallback list.
        # These are only attempted; a 404 simply causes the next candidate.
        print(f"[AI] model discovery failed: {str(e)[:220]}")

        fallback = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
        ]

        _MODEL_CACHE = fallback
        return list(fallback)


def _is_retryable_error(message: str) -> bool:
    upper = message.upper()
    return any(code in upper for code in RETRYABLE_STATUS)


def _status_from_error(message: str) -> Optional[int]:
    match = re.search(r"\b(4\d\d|5\d\d)\b", message)
    return int(match.group(1)) if match else None


def _mark_model_failed(model: str, status: Optional[int]) -> None:
    now = time.time()

    if status == 429 or "RESOURCE_EXHAUSTED" in model.upper():
        _MODEL_COOLDOWN_UNTIL[model] = now + MODEL_COOLDOWN_429
    elif status == 503 or status in (500, 502, 504):
        _MODEL_COOLDOWN_UNTIL[model] = now + MODEL_COOLDOWN_503


def _available_models(models: List[str]) -> List[str]:
    now = time.time()
    return [
        m for m in models
        if _MODEL_COOLDOWN_UNTIL.get(m, 0) <= now
    ]


def _parse_response(response: Any, schema):
    """
    Validate the structured response. Supports both Pydantic's model
    validation and the SDK's parsed response when available.
    """
    parsed = getattr(response, "parsed", None)

    if parsed is not None:
        if isinstance(parsed, schema):
            return parsed

        try:
            return schema.model_validate(parsed)
        except Exception:
            pass

    raw = getattr(response, "text", None)

    if not raw:
        raise ValueError("Gemini returned an empty response.")

    return schema.model_validate_json(raw)


def _call_ai(prompt: str, schema, temperature: float):
    """
    Send one request at a time, but rotate across usable Gemini models.

    Important:
      - 503 does NOT mean the model is nonexistent.
      - 503/429/5xx cause the current model to be cooled down and the next
        available model is tried.
      - 404/NOT_FOUND causes the model to be removed from the current pool.
      - authentication / malformed-request errors are raised immediately.
      - if discovery was successful, we only try models exposed to this key.
    """

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    )

    models = _discover_models()

    if not models:
        raise AIProviderError("No Gemini generateContent models are available.")

    last_error = None
    attempted = set()

    for round_number in range(2):
        candidates = [
            m for m in _available_models(models)
            if m not in attempted
        ]

        if not candidates:
            # All candidates may have been cooled down after transient errors.
            # On the second pass, force a fresh discovery in case Google's
            # available-capacity/model list changed.
            if round_number == 0:
                time.sleep(1)
                continue

            models = _discover_models(force=True)
            candidates = _available_models(models)

        for model in candidates:
            attempted.add(model)

            started = time.time()

            try:
                print(
                    f"[AI] {model}: sending {schema.__name__} request "
                    f"(pool attempt {len(attempted)}/{MAX_MODEL_ATTEMPTS})"
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )

                result = _parse_response(response, schema)

                print(
                    f"[AI] answered by {model} "
                    f"in {time.time() - started:.1f}s"
                )
                return result

            except Exception as e:
                last_error = e
                msg = " ".join(str(e).split())
                status = _status_from_error(msg)

                print(
                    f"[AI] {model} failed after "
                    f"{time.time() - started:.1f}s: {msg[:260]}"
                )

                upper = msg.upper()

                # Model genuinely unavailable for this key/version.
                if (
                    status == 404
                    or "NOT_FOUND" in upper
                    or "MODEL_NOT_FOUND" in upper
                ):
                    print(
                        f"[AI] {model} is not available for this API key; "
                        f"trying another discovered model."
                    )
                    continue

                # Temporary capacity/rate/service errors:
                # DO NOT keep hammering the same model.
                if _is_retryable_error(msg):
                    _mark_model_failed(model, status)

                    if status == 429 or "RESOURCE_EXHAUSTED" in upper:
                        print(
                            f"[AI] {model} hit a rate/quota limit; "
                            f"temporarily skipping it."
                        )
                    else:
                        print(
                            f"[AI] {model} returned a temporary "
                            f"{status or 'service'} error; rotating model."
                        )

                    if len(attempted) >= MAX_MODEL_ATTEMPTS:
                        break

                    continue

                # Validation errors can happen because a model does not support
                # the requested structured-output schema. Try another model.
                if (
                    "RESPONSE_SCHEMA" in upper
                    or "SCHEMA" in upper
                    or "STRUCTURED OUTPUT" in upper
                    or isinstance(e, ValueError)
                ):
                    print(
                        f"[AI] {model} could not satisfy the structured "
                        f"response; trying another model."
                    )
                    continue

                # Authentication, invalid API key, malformed request, etc.
                # Rotating models cannot fix those.
                raise AIProviderError(msg) from e

            if len(attempted) >= MAX_MODEL_ATTEMPTS:
                break

        if len(attempted) >= MAX_MODEL_ATTEMPTS:
            break

    raise AIProviderError(
        "No Gemini model could answer the request. "
        f"Last error: {last_error}"
    )



def check_gemini_models() -> List[str]:
    """
    Optional diagnostic helper. Call this from a shell/test route to see
    exactly which generateContent models the current API key exposes.
    """
    models = _discover_models(force=True)
    print("[AI] usable generateContent models:")
    for model in models:
        print(f"  - {model}")
    return models

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

For EVERY post, return ALL of these fields:
- topic: a short standard category label.
- sub_topic: the specific subject inside that category.
- content_type: what kind of marketing post it is.
- offer_pattern: the exact offer/promotion mechanic if there is one, otherwise 'None'.
- cta: the call-to-action used or implied, otherwise 'None'.
- keywords: 2 to 5 useful marketing keywords.
Use the same topic label for the same kind of post (for example always 'Buffet Offer', never
'Buffet Offers' or 'Buffet Deal' in the same batch). The sub-topic should be specific to the actual post.
Do not invent an offer, CTA, product, or event that is not supported by the post.
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
            output.append({
                "topic": item.topic,
                "sub_topic": item.sub_topic,
                "content_type": item.content_type,
                "offer_pattern": item.offer_pattern,
                "cta": item.cta,
                "keywords": item.keywords,
            })
        else:
            missing += 1
            output.append({
                "topic": None,
                "sub_topic": None,
                "content_type": None,
                "offer_pattern": None,
                "cta": None,
                "keywords": [],
            })  # left unanalyzed, not filled with fake data

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


def _coerce_post_date(value: Any, content: str = "") -> Optional[date]:
    """Prefer the DB's published_date, then fall back to a date embedded in content."""
    raw = str(value or "").strip()
    if raw:
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                pass
    return _post_date(content)


def _strip_date(content: str) -> str:
    return DATE_RE.sub("", content or "", count=1).strip()


def _recency_weight(post_date: Optional[date], today: date) -> float:
    if post_date is None:
        return UNKNOWN_DATE_WEIGHT
    age_days = max((today - post_date).days, 0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def find_trends(posts: List[Dict[str, Any]], today: Optional[date] = None):
    """
    Shared deterministic trend analysis used by both the Spot Trends chart and
    idea generation. The important rule is that topic coverage across rivals is
    the primary trend signal; recency/post volume are supporting signals.

    Each post may contain competitor_id and published_date. If published_date is
    missing, the date embedded in the post text is used as a fallback.

    Returns:
      trend_rows: every analyzed topic with chart-ready coverage metrics
      groups: top topics plus source posts/keywords for the AI idea prompt
      topic_counts: raw post counts
      keyword_counts: overall keyword counts
    """
    today = today or date.today()
    items = []
    for p in posts:
        content = p.get("content") or ""
        post_date = _coerce_post_date(p.get("published_date"), content)
        items.append({
            "content": content,
            "topic": (p.get("topic") or "").strip(),
            "keywords": _keyword_list(p.get("keywords")),
            "date": post_date,
            "weight": _recency_weight(post_date, today),
            "competitor_id": p.get("competitor_id"),
        })

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

    # Coverage is the same metric rendered by Spot Trends. A topic used by
    # 4 of 5 tracked rivals is therefore ahead of one used by 2 of 5, even
    # if the latter has a few more posts. Recency breaks ties.
    competitor_ids = {p.get("competitor_id") for p in items if p.get("competitor_id") is not None}
    total_competitors = max(len(competitor_ids), 1)

    ranked = []
    for key, topic_items in by_topic.items():
        comp_ids = {i["competitor_id"] for i in topic_items if i.get("competitor_id") is not None}
        coverage_pct = round(len(comp_ids) / total_competitors * 100, 1) if comp_ids else 0.0
        recency_score = round(sum(i["weight"] for i in topic_items), 2)
        ranked.append((key, topic_items, len(comp_ids), coverage_pct, recency_score))

    ranked.sort(key=lambda row: (-row[3], -row[4], -len(row[1]), label(row[0]).lower()))

    trend_rows = []
    for key, topic_items, comp_count, coverage_pct, recency_score in ranked:
        dated = [i["date"] for i in topic_items if i["date"]]
        trend_rows.append({
            "topic": label(key),
            "competitors_using": comp_count,
            "total_competitors": total_competitors,
            "occurrence_pct": coverage_pct,
            "post_count": len(topic_items),
            "recency_score": recency_score,
            "latest_post_date": str(max(dated)) if dated else None,
        })

    # Only topics with repeated evidence feed the trend-remix ideas. For tiny
    # datasets, still expose the strongest single topic so the feature remains useful.
    eligible = [row for row in ranked if len(row[1]) >= MIN_TOPIC_POSTS]
    if not eligible and ranked:
        eligible = ranked[:1]
    top = eligible[:TOP_TOPICS]

    groups = []
    for key, topic_items, comp_count, coverage_pct, recency_score in top:
        kw_scores: Counter = Counter()
        for i in topic_items:
            for kw in set(i["keywords"]):
                if len(kw) >= 3:
                    kw_scores[kw] += i["weight"]

        newest = sorted(
            topic_items,
            key=lambda i: i["date"] or date.min,
            reverse=True,
        )[:MAX_SOURCE_POSTS]
        dated = [i["date"] for i in topic_items if i["date"]]
        groups.append({
            "trending_topic": label(key),
            "competitors_using": comp_count,
            "total_competitors": total_competitors,
            "coverage_pct": coverage_pct,
            "posts_in_topic": len(topic_items),
            "trend_score": coverage_pct,
            "recency_score": recency_score,
            "latest_post_date": str(max(dated)) if dated else None,
            "top_keywords": [kw for kw, _ in kw_scores.most_common(KEYWORDS_PER_TOPIC)],
            "source_posts_newest_first": [_strip_date(i["content"])[:SOURCE_POST_CHARS] for i in newest],
        })

    topic_counts = {
        label(key): len(v)
        for key, v in sorted(by_topic.items(), key=lambda kv: -len(kv[1]))[:10]
    }
    return trend_rows, groups, topic_counts, dict(keyword_counts.most_common(15))


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
    focus_keywords: Optional[List[str]] = None,
    festivals: Optional[List[Dict[str, str]]] = None,
    region: str = "India",
    days_ahead: int = 60,
    previous_ideas: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    posts: analyzed posts from the DB, e.g.
           [{"content": p.content, "topic": p.topic, "keywords": p.keywords_detected}, ...]

    Pipeline (one AI call):
      1. Use the same competitor-coverage trend analysis shown by Spot Trends;
         recency and post volume break ties (plain Python).
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

    trend_rows, groups, topic_counts, keyword_counts = find_trends(posts, today)

    focus_keywords = [str(k).strip() for k in (focus_keywords or []) if str(k).strip()]
    focus_text = ", ".join(focus_keywords) if focus_keywords else "None set"

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
These are the EXACT topics shown in the Spot Trends chart. They are ranked primarily by
competitor coverage (competitors_using / total_competitors), with recency and post volume as tie-breakers.
For every trend_remix idea, set source_trend to the exact "trending_topic" label from the data below.
Take that trending topic and rework one of its source posts into a NEW post for our business.
Spread the ideas across the topics and give more ideas to the topics with higher coverage. Use each topic's
top_keywords as the flavor of the idea.
- Keep what works (the kind of offer, the hook, the structure) but rewrite everything in new words.
- Never copy sentences, names, prices or distinctive phrases from the source posts.
- Add a clear unique twist competitors are not using: a fresh hook, a different audience
  (families, office groups, couples...), a signature detail, a time-limited element or a festival tie-in.

SPOT TRENDS DATA (this is the same analysis rendered in the dashboard chart):
{json.dumps(groups, ensure_ascii=False, indent=2)}

ALL CHART TREND ROWS:
{json.dumps(trend_rows, ensure_ascii=False, indent=2)}
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

    all_avoid = list(avoid_topics or [])
    for prev in (previous_ideas or []):
        t = str(prev.get("topic") or "").strip()
        if t and t not in all_avoid:
            all_avoid.append(t)
    avoid_text = ", ".join(all_avoid) if all_avoid else "None"

    prompt = f"""
You are a local business marketing strategy engine for Google Maps update posts.

OUR BUSINESS: {business_name or "a local business"}
OUR FOCUS KEYWORDS: {focus_text}
Today's date: {today}
Planning window: {today} to {end}

{festival_text}

Create {count} post ideas in total, in the parts below. Set idea_type on every idea.
{part_a}{part_b}
RULES FOR ALL IDEAS:
- For trend_remix, source_trend MUST exactly match one of the supplied trending_topic labels.
- For fresh ideas, source_trend MUST be "None".
- Give every idea a suggested_post_date inside the planning window (YYYY-MM-DD). Tie ideas to the
  upcoming festivals/occasions where it fits and post a few days BEFORE the occasion.
  Use 'Evergreen' as the occasion for ideas not tied to a date.
- Do NOT reuse these previous topics (ignore any date tags in them): [{avoid_text}].
- Use realistic Google Maps CTA labels and give clear image direction.
- Do not invent facts about our business (prices, timings, dish names, addresses).
  Where a specific detail is needed, write a [PLACEHOLDER] instead.
- The exact business name is **{business_name or "a local business"}**. Naturally mention that exact name once in each draft_copy when it fits, so the user can copy the post directly. Do not repeat it unnaturally.
- YOUR FOCUS KEYWORDS ARE A REQUIRED INPUT, NOT A SUGGESTION. If one or more focus keywords are supplied, EVERY idea must be built around at least one of them. Do not generate an unrelated idea just because a competitor trend is available.
- The focus keyword controls the subject of the idea. Example: if the focus keyword is "gift card", the idea must actually be about gift cards (not an unrelated dish, buffet, or behind-the-scenes content). If it is "food festival", the idea must actually be about a food-festival/food-event angle.
- For every idea, set focus_keywords_used to the exact focus keyword(s) that drive the idea. Use the exact phrase naturally in the draft_copy and include it in the target keywords.
- Never output a generic idea and merely add the focus keyword as a tag. The post topic, draft copy, CTA and image concept must all support the chosen focus keyword.
- If multiple focus keywords are supplied, distribute ideas across them when possible.
- Competitor topics and keywords are supporting inspiration only. The user's focus keywords determine the subject/angle of the idea.
- image_concept is a hard visual instruction. It MUST describe the same subject and message as draft_copy, and must never be swapped for a generic food festival, buffet, gift-card, or unrelated scene.
- image_concept must describe ONLY what is visible: the product/dish, setting, props, lighting and composition. Do NOT put any text, words, slogans, prices, signage, logos or business names in image_concept (the business name is added to the image separately).
"""

    result = _call_ai(prompt, IdeasResult, temperature=0.7)
    ideas = [i.model_dump() for i in result.ideas[:count]]

    got_trend = sum(1 for i in ideas if i["idea_type"] == "trend_remix")
    if got_trend != n_trend:
        print(f"[AI] wanted {n_trend} trend + {n_fresh} fresh ideas, got {got_trend} trend + {len(ideas) - got_trend} other")

    ideas.sort(key=lambda i: i["suggested_post_date"])

    # Make the generated copy immediately pasteable. The prompt asks the model
    # to mention the business name, but enforce it here as a safety net so a
    # missed instruction never produces copy with the wrong/absent brand.
    brand = (business_name or "").strip()
    if brand:
        for idea in ideas:
            copy = (idea.get("draft_copy") or "").strip()
            if brand.casefold() not in copy.casefold():
                idea["draft_copy"] = f"{brand}: {copy}" if copy else brand

    return [
        {
            "topic": _idea_label(i),
            "draft_copy": i["draft_copy"],
            "source_trend": i.get("source_trend", "None"),
            "focus_keywords_used": [
                fk for fk in focus_keywords
                if any(fk.casefold() in str(k).casefold() or str(k).casefold() in fk.casefold() for k in i.get("keywords", []))
                or fk.casefold() in str(i.get("draft_copy", "")).casefold()
            ],
            "cta_suggested": i["cta_suggested"],
            "image_concept": i["image_concept"],
            "keywords": i["keywords"],
            "focus_keywords_used": i.get("focus_keywords_used", []),
            # extra keys, main.py ignores them today but they're there if you add columns later
            "idea_type": i["idea_type"],
            "occasion": i["occasion"],
            "suggested_post_date": i["suggested_post_date"],
        }
        for i in ideas
    ]

def _image_headline(topic: str, draft_copy: str = "") -> str:
    """
    Turn the idea topic into a short headline suitable for rendering inside
    the generated advertisement. Prefer the topic because it is already the
    AI-generated core headline; fall back to the first sentence of the copy.
    """
    headline = (topic or "").strip()

    # Remove the metadata prefix used by _idea_label():
    # "[2026-11-05 | Diwali | Fresh] Festive Buffet"
    if headline.startswith("[") and "]" in headline:
        headline = headline.split("]", 1)[1].strip()

    # Keep generated image text short enough to render cleanly.
    headline = re.sub(r"\s+", " ", headline)
    headline = headline.strip(" .:-")
    if not headline:
        headline = re.split(r"[.!?\n]", (draft_copy or "").strip(), maxsplit=1)[0].strip()

    return headline[:70]


def _clean_visual_concept(
    concept: str,
    business_name: str = "",
    previous_business_names: Optional[List[str]] = None,
) -> str:
    """
    Reduce the stored image_concept to a purely visual description.

    Old ideas (or ideas written for a previous business) can contain brand names,
    quoted slogans, or sentences about signs/logos/text. Image models happily
    paint all of that into the picture, so strip it before it reaches the model.
    """
    text = " ".join((concept or "").split())

    # 1. Remove the current and any previous business names from the concept.
    names = [business_name, *(previous_business_names or [])]
    for name in sorted({n.strip() for n in names if n and n.strip()}, key=len, reverse=True):
        text = re.sub(re.escape(name), "the venue", text, flags=re.IGNORECASE)

    # 2. Remove quoted strings (slogans, headlines, "text reading ...").
    text = re.sub(r'["\u201c\u201d][^"\u201c\u201d]{1,150}["\u201c\u201d]', "", text)

    # 3. Drop whole sentences that talk about writing/branding things in the image.
    text_words = re.compile(
        r"\b(text|caption|headline|tagline|slogan|logo|sign(?:age|board)?|banner|"
        r"typography|lettering|font|watermark|overlay(?:ed)?|label(?:led)?|"
        r"written|reads|reading|says|poster|menu board|price tag)\b",
        re.IGNORECASE,
    )
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if s.strip() and not text_words.search(s)]

    cleaned = " ".join(kept).strip()
    if not cleaned:  # every sentence mentioned text - keep the quote-free version
        cleaned = text.strip()
    return re.sub(r"\s+", " ", cleaned)


def generate_image(
    image_concept: str,
    topic: str = "",
    draft_copy: str = "",
    cta_suggested: str = "",
    business_name: str = "",
    previous_business_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate exactly one marketing image for one GeneratedIdea using
    Hugging Face Inference Providers.

    The picture is built from the visual brief (image_concept) only. The ONLY
    text allowed in the image is the CURRENT business name, shown once below
    the product. topic / draft_copy / cta_suggested are accepted for backwards
    compatibility but deliberately NOT sent to the image model: any words in
    the prompt tend to get painted into the picture (and the copy may still
    contain an older business name).

    Returns:
        {
            "data": bytes,
            "mime_type": str,
        }
    """
    if not _HF_TOKEN or image_client is None:
        raise AIProviderError(
            "HF_TOKEN is not configured. Add your Hugging Face token to .env."
        )

    if not image_concept or not image_concept.strip():
        raise AIProviderError("This idea does not have an image concept.")

    brand = " ".join((business_name or "").split())
    visual_brief = _clean_visual_concept(image_concept, brand, previous_business_names)

    if brand:
        text_rule = (
            f'TEXT IN THE IMAGE: exactly one line of text, "{brand}", written in clean, '
            f"elegant lettering, centered in the empty space directly BELOW the product. "
            f"Spell it exactly: {brand}. "
            f"There is no other text anywhere: no tagline, slogan, caption, price, "
            f"offer, call-to-action, logo, watermark, numbers or extra letters."
        )
    else:
        text_rule = (
            "TEXT IN THE IMAGE: none. No words, letters, numbers, logos or watermarks anywhere."
        )

    prompt = f"""Professional commercial advertising photograph, photorealistic, premium food and hospitality photography.

SUBJECT (main focus, sharp and centered, fills most of the frame): {visual_brief}

COMPOSITION: The product is the hero. Leave a clean, uncluttered area directly below it for the business name. Natural commercial lighting, realistic shadows and materials, shallow depth of field, believable proportions. Single image, no collage, no panels, no borders, no UI.

{text_rule}
"""

    try:
        print(
            f"[IMAGE AI] generating with Hugging Face "
            f"model={HF_IMAGE_MODEL}, provider={HF_IMAGE_PROVIDER}"
        )

        output_image = image_client.text_to_image(
            prompt=prompt,
            model=HF_IMAGE_MODEL,
        )

        # huggingface_hub returns a PIL Image for text_to_image().
        buffer = BytesIO()
        output_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        if not image_bytes:
            raise AIProviderError("Hugging Face returned an empty image.")

        return {
            "data": image_bytes,
            "mime_type": "image/png",
        }

    except AIProviderError:
        raise

    except Exception as e:
        msg = " ".join(str(e).split())
        print(f"[IMAGE AI] Hugging Face generation failed: {msg[:500]}")
        raise AIProviderError(
            f"Hugging Face image generation failed: {msg[:300]}"
        ) from e


# ==========================================
# Spot Trends: plain-English chart explanation (Hugging Face text model)
# ==========================================
# A text model cannot "see" the chart, so it is given the exact numbers the chart
# is drawn from (find_trends output). That is more reliable than reading pixels.
#
# .env (all optional, HF_TOKEN is already required for images):
#   HF_EXPLAIN_MODELS=Qwen/Qwen2.5-72B-Instruct,meta-llama/Llama-3.1-8B-Instruct
#   HF_EXPLAIN_PROVIDER=auto
# The models are tried in order; if one is not served for your token/provider
# the next one is used.

HF_EXPLAIN_PROVIDER = os.getenv("HF_EXPLAIN_PROVIDER", "auto")
HF_EXPLAIN_MODELS = [
    m.strip()
    for m in os.getenv(
        "HF_EXPLAIN_MODELS",
        "Qwen/Qwen2.5-72B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-7B-Instruct",
    ).split(",")
    if m.strip()
]

explain_client = (
    InferenceClient(provider=HF_EXPLAIN_PROVIDER, api_key=_HF_TOKEN)
    if _HF_TOKEN
    else None
)

# Same trend data -> same explanation. A new analysis changes the numbers, which
# changes the key, which triggers a fresh explanation automatically.
_EXPLAIN_CACHE: Dict[str, Dict[str, Any]] = {}


def _trend_fingerprint(trend_rows: List[Dict[str, Any]]) -> str:
    payload = json.dumps(trend_rows, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def explain_trends(
    trend_rows: List[Dict[str, Any]],
    groups: Optional[List[Dict[str, Any]]] = None,
    business_name: str = "",
    force: bool = False,
) -> Dict[str, Any]:
    """
    Ask a free Hugging Face model to explain the Spot Trends chart.

    Returns {"text": str, "model": str, "cached": bool}.
    Raises AIProviderError if no model could answer.
    """
    if not trend_rows:
        raise AIProviderError("No trend data to explain yet. Analyze posts first.")
    if explain_client is None:
        raise AIProviderError("HF_TOKEN is not configured. Add your Hugging Face token to .env.")

    key = _trend_fingerprint(trend_rows) + "|" + (business_name or "")
    if not force and key in _EXPLAIN_CACHE:
        return {**_EXPLAIN_CACHE[key], "cached": True}

    chart = [
        {
            "topic": r["topic"],
            "rivals_using": f"{r['competitors_using']} of {r['total_competitors']}",
            "coverage_pct": r["occurrence_pct"],
            "posts": r["post_count"],
            "latest_post": r.get("latest_post_date"),
        }
        for r in trend_rows[:12]
    ]
    keywords = {
        g["trending_topic"]: g.get("top_keywords", [])
        for g in (groups or [])
    }

    system = (
        "You are a local-marketing analyst explaining a bar chart to a busy business owner. "
        "Use ONLY the numbers given. Never invent topics, percentages or competitors. "
        "Write plain English, no jargon."
    )
    user = f"""The bar chart "Spot trends" shows, for each topic, the % of tracked rival businesses that posted about it at least once on Google Maps. It is coverage across rivals, NOT share of all posts.
{('Our business: ' + business_name) if business_name else ''}

CHART DATA (highest coverage first):
{json.dumps(chart, ensure_ascii=False, indent=1)}

TOP KEYWORDS PER LEADING TOPIC:
{json.dumps(keywords, ensure_ascii=False)}

Write the explanation in this format (short, about 150 words total):
1. One sentence: what the chart shows overall.
2. "What stands out": 2-3 short bullet points (leading topics, gaps between bars, topics only one rival uses).
3. "What to do": 2 short bullet points on what our business should post or avoid copying.
"""

    last_error: Optional[Exception] = None
    for model in HF_EXPLAIN_MODELS:
        try:
            print(f"[TREND EXPLAIN] asking {model}")
            resp = explain_client.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=450,
                temperature=0.3,
            )
            text = (resp.choices[0].message.content or "").strip()
            if len(text) < 40:
                raise ValueError("model returned a too-short answer")
            result = {"text": text, "model": model}
            _EXPLAIN_CACHE[key] = result
            return {**result, "cached": False}
        except Exception as e:
            last_error = e
            print(f"[TREND EXPLAIN] {model} failed: {' '.join(str(e).split())[:300]}")
            continue

    raise AIProviderError(f"No Hugging Face model could explain the chart. Last error: {last_error}")