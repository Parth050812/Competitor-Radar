from typing import Optional, List, Any
from datetime import datetime
from sqlmodel import Field, SQLModel, Relationship
from pydantic import field_validator,BaseModel


def _parse_dt(value: Any):
    """Shared helper so we don't repeat this validator in every model."""
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    target_business: str  # our own business/profile name
    created_at: datetime = Field(default_factory=datetime.utcnow)

    competitors: List["Competitor"] = Relationship(back_populates="project")
    keywords: List["Keyword"] = Relationship(back_populates="project")
    generated_ideas: List["GeneratedIdea"] = Relationship(back_populates="project")

    @field_validator("created_at", mode="before")
    @classmethod
    def _v(cls, value):
        return _parse_dt(value)


class Keyword(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    text: str

    project: Optional[Project] = Relationship(back_populates="keywords")


class Competitor(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    name: str
    maps_url: Optional[str] = None
    place_id: Optional[str] = None  # Google's internal id, extracted from the confirmed maps_url
    address: Optional[str] = None   # branch address, used so Google opens the right branch of a chain
    last_scraped_at: Optional[datetime] = None
    last_scrape_status: str = "never_run"  # never_run | success | no_updates | captcha_blocked | error

    project: Optional[Project] = Relationship(back_populates="competitors")
    posts: List["ScrapedPost"] = Relationship(back_populates="competitor")
    logs: List["ScrapeLog"] = Relationship(back_populates="competitor")

    @field_validator("last_scraped_at", mode="before")
    @classmethod
    def _v(cls, value):
        return _parse_dt(value)


class ScrapedPost(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    competitor_id: int = Field(foreign_key="competitor.id")
    content: str
    post_url: Optional[str] = None
    published_date: Optional[str] = None  # kept as raw string, Maps rarely gives a clean date
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    title: Optional[str] = None            # post headline, e.g. "Year-End Fest: buy 4 buffets and get 1 free!"
    validity: Optional[str] = None         # e.g. "Valid 5 Dec - 21 Dec"
    post_id: Optional[str] = None          # Google's data-post-id (stable, unique per post)
    cta_text: Optional[str] = None
    topic: Optional[str] = None            # filled in later by AI analysis
    sub_topic: Optional[str] = None
    content_type: Optional[str] = None
    offer_pattern: Optional[str] = None
    analysis_cta: Optional[str] = None
    keywords_detected: Optional[str] = None  # comma-separated, simple is fine for the core version
    content_hash: str = Field(index=True, unique=True)  # unique = duplicate protection at the DB level
    scraped_at: datetime = Field(default_factory=datetime.utcnow)

    competitor: Optional[Competitor] = Relationship(back_populates="posts")

    @field_validator("scraped_at", mode="before")
    @classmethod
    def _v(cls, value):
        return _parse_dt(value)


class ScrapeLog(SQLModel, table=True):
    """One row per scrape run per competitor — section 21 of the spec."""
    id: Optional[int] = Field(default=None, primary_key=True)
    competitor_id: int = Field(foreign_key="competitor.id")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    posts_found: int = 0
    new_posts: int = 0
    duplicates_skipped: int = 0
    status: str = "running"  # running | success | no_updates | captcha_blocked | error
    error_message: Optional[str] = None
    # True when the run stopped early because 3 already-saved posts appeared in a row.
    up_to_date: bool = False

    competitor: Optional[Competitor] = Relationship(back_populates="logs")


class GeneratedIdea(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")

    topic: str
    draft_copy: str
    cta_suggested: str
    image_concept: str
    source_trend: Optional[str] = None
    strategy_reason: Optional[str] = None

    # Generated image stored directly in SQLite
    image_data: Optional[bytes] = Field(default=None)
    image_mime_type: Optional[str] = Field(default=None)
    # Business name used when the stored image was generated.
    # Lets the app detect stale images if the project business name changes.
    image_business_name: Optional[str] = None

    keywords: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    project: Optional[Project] = Relationship(back_populates="generated_ideas")

    @field_validator("created_at", mode="before")
    @classmethod
    def _v(cls, value):
        return _parse_dt(value)

class GeneratedIdeaResponse(BaseModel):
    id: int
    project_id: int
    topic: str
    draft_copy: str
    cta_suggested: str
    image_concept: str
    source_trend: Optional[str] = None
    strategy_reason: Optional[str] = None
    keywords: Optional[str] = None
    created_at: datetime

    image_business_name: Optional[str] = None
    has_image: bool
    image_url: Optional[str] = None