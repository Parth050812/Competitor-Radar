import os
import re
import json
import time
import base64
import random
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


def generate_image(
    image_concept: str,
    topic: str = "",
    draft_copy: str = "",
    cta_suggested: str = "",
    business_name: str = "",
) -> Dict[str, Any]:
    """
    Generate exactly one marketing image for one GeneratedIdea using
    Hugging Face Inference Providers.

    The image model receives the actual post context rather than only the
    image_concept, so the visual can match the message being advertised.

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

    headline = _image_headline(topic, draft_copy)

    context_lines = [
        f"POST TOPIC / HEADLINE: {headline}" if headline else "",
        f"POST COPY: {draft_copy.strip()}" if draft_copy else "",
        f"CTA: {cta_suggested.strip()}" if cta_suggested else "",
        f"BUSINESS: {business_name.strip()}" if business_name else "",
        f"IMAGE CONCEPT: {image_concept.strip()}",
    ]
    context = "\n".join(line for line in context_lines if line)

    prompt = f"""
  Create ONE premium, finished advertising photograph for a local business.
  
  This is a PROFESSIONAL COMMERCIAL ADVERTISEMENT, not a flyer, poster,
  social-media template, collage, or generic AI artwork.
  
  POST CONTEXT:
  {context}
  
  BUSINESS / BRAND:
  {business_name}
  
  IMPORTANT CREATIVE DIRECTION:
  The PRODUCT / FOOD / SERVICE must be the absolute hero of the image.
  
  Think like a world-class food advertising photographer and creative director.
  
  The viewer should immediately look at the product first.
  
  COMPOSITION:
  - Make the main product the largest and most visually important element.
  - Product should occupy roughly 60–75% of the visual attention.
  - Put the product prominently in the center or slightly below center.
  - Use an intentional hero-product composition.
  - Show realistic texture, crisp edges, appetizing detail, natural imperfections,
    realistic ingredients, realistic surfaces and believable lighting.
  - Use cinematic commercial photography rather than an "AI art" appearance.
  - Use depth of field carefully: the PRODUCT must remain sharp and detailed.
  - Background can have tasteful depth and atmosphere, but must never compete
    with the product.
  - Avoid excessive blur.
  - Avoid excessive bokeh.
  - Avoid oversaturated colors.
  - Avoid plastic-looking food.
  - Avoid surreal lighting.
  - Avoid floating objects or physically impossible food.
  - Make the scene feel like a real professional advertising photoshoot.
  
  LIGHTING:
  - Premium commercial food photography.
  - Cinematic but believable lighting.
  - Beautiful directional key light on the product.
  - Natural highlights and realistic shadows.
  - Subtle warm atmosphere where appropriate.
  - Rich but realistic colors.
  - Detailed food texture.
  - High dynamic range.
  - Professional editorial color grading.
  - Photorealistic camera rendering.
  - Real lens characteristics.
  - No obvious AI artifacts.
  
  VISUAL STYLE:
  Think:
  premium restaurant campaign,
  high-end food commercial,
  modern brand advertising,
  editorial food photography,
  cinematic product photography.
  
  The final image should look like something a major consumer brand
  could actually publish as a campaign advertisement.
  
  TEXT DESIGN:
  DO NOT place the post title or topic as a giant headline.
  
  DO NOT render:
  "{headline}"
  
  Instead, create ONE short, catchy BRAND TAGLINE inspired by the post context.
  
  The tagline should feel like memorable advertising copy:
  short,
  playful,
  confident,
  rhythmic,
  easy to remember,
  and emotionally connected to the product.
  
  Examples of the STYLE of tagline:
  "Made to Make You Smile"
  "Good Food. Great Moments."
  "Bring Your Appetite."
  "Made Fresh. Made Happy."
  "Gather. Eat. Repeat."
  
  Do NOT copy these examples literally unless they naturally fit.
  
  Create a NEW tagline based on the actual post context.
  
  TYPOGRAPHY:
  - The tagline should use a bold, expressive, funky advertising type style.
  - Think modern brand campaign typography rather than a boring default font.
  - Use playful letterforms, confident weight, tasteful personality and strong
    visual rhythm.
  - The typography should feel intentionally designed by a professional
    graphic designer.
  - Do not use generic Arial/Helvetica-style plain text.
  - Do not use a corporate presentation font.
  - Do not use huge block text covering the product.
  - Keep the tagline relatively small compared with the hero product.
  - Place the tagline ABOVE the hero product, with generous breathing room.
  - Make sure the tagline is clearly readable but subordinate to the product.
  - Never place text directly across the most important part of the food.
  
  BRAND NAME:
  At the bottom of the advertisement, add the business/brand name:
  
  "{business_name}"
  
  Treat this like a premium brand signature.
  
  The brand name should be smaller than the product and visually refined.
  It can sit beneath the product with clean spacing and subtle styling.
  
  TEXT HIERARCHY:
  1. HERO PRODUCT — overwhelmingly dominant
  2. SHORT FUNKY TAGLINE — secondary
  3. BRAND NAME — small signature at the bottom
  
  The advertisement should still look beautiful even if the viewer ignores
  all the text.
  
  LAYOUT:
  - Clean premium composition.
  - Strong visual hierarchy.
  - Generous negative space around typography.
  - No giant title.
  - No paragraph text.
  - No bullet points.
  - No captions.
  - No fake promotional copy.
  - No unnecessary decorative elements.
  - No collage.
  - No multiple panels.
  - No UI.
  - No poster template.
  - No borders.
  
  FACTUAL SAFETY:
  Only use information explicitly supported by the post context.
  Do not invent:
  prices,
  discounts,
  offers,
  addresses,
  phone numbers,
  opening hours,
  awards,
  ingredients,
  claims,
  locations,
  or product names.
  
  Do not invent a logo.
  Do not create fake brand marks.
  
  FINAL QUALITY:
  The final result should look like a REAL photograph from a premium
  commercial advertising campaign, not an AI-generated illustration.
  
  The product must be irresistibly appetizing, physically believable,
  highly detailed and the unmistakable center of attention.
  
  OUTPUT:
  ONE finished advertising image.
  Premium commercial photography.
  Clean composition.
  Photorealistic.
  Cinematic.
  Brand-ready.
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
