import os
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from sqlmodel import SQLModel, Session, create_engine, select
from pydantic import BaseModel
from fastapi import HTTPException
from models import Project, Competitor, ScrapedPost, GeneratedIdea, Keyword, ScrapeLog
from scraper import run_competitor_scrape, search_place_candidates, CaptchaBlocked
from ai_providers import analyze_posts, generate_ideas, AIProviderError  # simple wrapper module, see below
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sqlite_url = f"sqlite:///{BASE_DIR / 'database.db'}"
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


app = FastAPI(title="Google Maps Competitor Update Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])   



@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "frontend.html")


@app.get("/logo.png")
async def favicon():
    return FileResponse(BASE_DIR / "logo.png", media_type="image/png")


# --- request bodies (simple, no query-param endpoints) ---

class CompetitorIn(BaseModel):
    name: str
    maps_url: Optional[str] = None
    place_id: Optional[str] = None


class KeywordIn(BaseModel):
    text: str


class IdeaRequest(BaseModel):
    count: int = 5


# --- 1. PROJECTS ---

@app.post("/projects/", response_model=Project)
def create_project(project: Project, session: Session = Depends(get_session)):
    project.id = None
    # SQLModel table models don't reliably run field_validator on request bodies,
    # so force this server-side rather than trusting whatever the client sent.
    project.created_at = datetime.utcnow()
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@app.get("/projects/", response_model=List[Project])
def list_projects(session: Session = Depends(get_session)):
    return session.exec(select(Project)).all()


@app.get("/projects/{project_id}/dashboard")
def project_dashboard(project_id: int, session: Session = Depends(get_session)):
    competitors = session.exec(select(Competitor).where(Competitor.project_id == project_id)).all()
    competitor_ids = [c.id for c in competitors]
    posts = session.exec(select(ScrapedPost).where(ScrapedPost.competitor_id.in_(competitor_ids))).all() if competitor_ids else []
    ideas = session.exec(select(GeneratedIdea).where(GeneratedIdea.project_id == project_id)).all()

    topic_counts = {}
    for p in posts:
        if p.topic:
            topic_counts[p.topic] = topic_counts.get(p.topic, 0) + 1

    return {
        "competitor_count": len(competitors),
        "total_posts": len(posts),
        "generated_idea_count": len(ideas),
        "top_topics": sorted(topic_counts.items(), key=lambda x: -x[1])[:5],
        "last_scrape": max((c.last_scraped_at for c in competitors if c.last_scraped_at), default=None),
    }


# --- 2. COMPETITORS (full CRUD, per spec section 5) ---

@app.get("/competitors/search/")
def search_competitors(q: str):
    """
    Step 1 of adding a competitor: search Google Maps and return candidates
    for the user to pick from, instead of guessing off the typed name alone.
    """
    try:
        return search_place_candidates(q)
    except CaptchaBlocked as e:
        raise HTTPException(status_code=503, detail=f"Google showed a CAPTCHA: {e}")


@app.post("/projects/{project_id}/competitors/", response_model=Competitor)
def add_competitor(project_id: int, body: CompetitorIn, session: Session = Depends(get_session)):
    """
    Step 2: save the candidate the user confirmed. Dedup on place_id when we have
    one (more reliable than name), otherwise fall back to name matching.
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if body.place_id:
        existing = session.exec(
            select(Competitor).where(
                Competitor.project_id == project_id,
                Competitor.place_id == body.place_id,
            )
        ).first()
        if existing:
            return existing

    competitor = Competitor(
        project_id=project_id,
        name=body.name,
        maps_url=body.maps_url,
        place_id=body.place_id,
    )
    session.add(competitor)
    session.commit()
    session.refresh(competitor)
    return competitor


@app.get("/projects/{project_id}/competitors/", response_model=List[Competitor])
def list_competitors(project_id: int, session: Session = Depends(get_session)):
    return session.exec(select(Competitor).where(Competitor.project_id == project_id)).all()


@app.put("/competitors/{competitor_id}", response_model=Competitor)
def edit_competitor(competitor_id: int, body: CompetitorIn, session: Session = Depends(get_session)):
    competitor = session.get(Competitor, competitor_id)
    if not competitor:
        raise HTTPException(status_code=404, detail="Competitor not found")
    competitor.name = body.name
    competitor.maps_url = body.maps_url
    session.add(competitor)
    session.commit()
    session.refresh(competitor)
    return competitor


# --- 3. KEYWORDS ---

@app.post("/projects/{project_id}/keywords/", response_model=Keyword)
def add_keyword(project_id: int, body: KeywordIn, session: Session = Depends(get_session)):
    if not session.get(Project, project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    keyword = Keyword(project_id=project_id, text=body.text)
    session.add(keyword)
    session.commit()
    session.refresh(keyword)
    return keyword


@app.get("/projects/{project_id}/keywords/", response_model=List[Keyword])
def list_keywords(project_id: int, session: Session = Depends(get_session)):
    return session.exec(select(Keyword).where(Keyword.project_id == project_id)).all()


# --- 4. SCRAPING (background worker + persistent logs + CAPTCHA handling) ---

def background_scrape_worker(competitor_id: int):
    with Session(engine) as session:
        competitor = session.get(Competitor, competitor_id)
        if not competitor:
            return

        log = ScrapeLog(competitor_id=competitor_id)
        session.add(log)
        session.commit()
        session.refresh(log)

        try:
            scraped_data = run_competitor_scrape(competitor.name, maps_url=competitor.maps_url)

            new_count = 0
            dup_count = 0
            for item in scraped_data:
                exists = session.exec(
                    select(ScrapedPost).where(ScrapedPost.content_hash == item["content_hash"])
                ).first()
                if exists:
                    dup_count += 1
                    continue
                session.add(ScrapedPost(
                    competitor_id=competitor_id,
                    content=item["content"],
                    published_date=item.get("date"),
                    title=item.get("title"),
                    validity=item.get("validity"),
                    post_id=item.get("post_id"),
                    cta_text=item.get("cta_text"),
                    image_url=item.get("image_url"),
                    video_url=item.get("video_url"),
                    post_url=item.get("post_url"),
                    content_hash=item["content_hash"],
                ))
                new_count += 1

            competitor.last_scraped_at = datetime.utcnow()
            competitor.last_scrape_status = "success"
            log.status = "success"
            log.posts_found = len(scraped_data)
            log.new_posts = new_count
            log.duplicates_skipped = dup_count

        except CaptchaBlocked as e:
            competitor.last_scrape_status = "captcha_blocked"
            log.status = "captcha_blocked"
            log.error_message = str(e)

        except Exception as e:
            competitor.last_scrape_status = "error"
            log.status = "error"
            log.error_message = str(e)

        finally:
            log.finished_at = datetime.utcnow()
            session.add(competitor)
            session.add(log)
            session.commit()


@app.post("/competitors/{competitor_id}/scrape/")
def trigger_scrape(competitor_id: int, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    competitor = session.get(Competitor, competitor_id)
    if not competitor:
        raise HTTPException(status_code=404, detail="Competitor not found")

    background_tasks.add_task(background_scrape_worker, competitor_id)
    return {"status": "Scraping task queued", "competitor": competitor.name}


@app.get("/competitors/{competitor_id}/logs/", response_model=List[ScrapeLog])
def get_scrape_logs(competitor_id: int, session: Session = Depends(get_session)):
    return session.exec(
        select(ScrapeLog).where(ScrapeLog.competitor_id == competitor_id).order_by(ScrapeLog.started_at.desc())
    ).all()


# --- 5. POSTS REPOSITORY (with filters, per spec section 10) ---

@app.get("/projects/{project_id}/posts/", response_model=List[ScrapedPost])
def get_project_posts(
    project_id: int,
    competitor_id: Optional[int] = None,
    topic: Optional[str] = None,
    session: Session = Depends(get_session),
):
    statement = select(ScrapedPost).join(Competitor).where(Competitor.project_id == project_id)
    if competitor_id:
        statement = statement.where(ScrapedPost.competitor_id == competitor_id)
    if topic:
        statement = statement.where(ScrapedPost.topic == topic)
    return session.exec(statement).all()


# --- 6. AI ANALYSIS + TRENDS ---

@app.post("/projects/{project_id}/analyze/")
def analyze_project(project_id: int, session: Session = Depends(get_session)):
    competitors = session.exec(select(Competitor).where(Competitor.project_id == project_id)).all()
    competitor_ids = [c.id for c in competitors]
    posts = session.exec(select(ScrapedPost).where(ScrapedPost.competitor_id.in_(competitor_ids))).all() if competitor_ids else []

    if not posts:
        raise HTTPException(status_code=400, detail="No posts to analyze yet — scrape competitors first")
    try:
        results = analyze_posts([p.content for p in posts])
    except AIProviderError:
        raise HTTPException(status_code=503, detail="AI is busy right now, please try again in a minute.")
    for post, result in zip(posts, results):
        post.topic = result.get("topic")
        post.keywords_detected = ", ".join(result.get("keywords", []))
        session.add(post)
    session.commit()

    return {"analyzed_posts": len(posts)}


@app.get("/projects/{project_id}/trends/")
def project_trends(project_id: int, session: Session = Depends(get_session)):
    competitors = session.exec(select(Competitor).where(Competitor.project_id == project_id)).all()
    competitor_ids = [c.id for c in competitors]
    posts = session.exec(select(ScrapedPost).where(ScrapedPost.competitor_id.in_(competitor_ids))).all() if competitor_ids else []

    topic_to_competitors = {}
    for post in posts:
        if not post.topic:
            continue
        topic_to_competitors.setdefault(post.topic, set()).add(post.competitor_id)

    total_competitors = len(competitors) or 1
    return [
        {
            "topic": topic,
            "competitors_using": len(comp_ids),
            "total_competitors": total_competitors,
            "occurrence_pct": round(len(comp_ids) / total_competitors * 100, 1),
        }
        for topic, comp_ids in sorted(topic_to_competitors.items(), key=lambda x: -len(x[1]))
    ]


# --- 7. IDEA GENERATION (with dedup vs. previously generated ideas) ---


@app.post("/projects/{project_id}/generate-ideas/")
def generate_project_ideas(project_id: int, body: IdeaRequest, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    competitors = session.exec(select(Competitor).where(Competitor.project_id == project_id)).all()
    competitor_ids = [c.id for c in competitors]
    posts = session.exec(select(ScrapedPost).where(ScrapedPost.competitor_id.in_(competitor_ids))).all() if competitor_ids else []
    if not posts:
        raise HTTPException(status_code=400, detail="No repository data yet — scrape and analyze first")
    if not any(p.topic for p in posts):
        raise HTTPException(status_code=400, detail="Posts are not analyzed yet — run analyze first")

    previous_ideas = session.exec(select(GeneratedIdea).where(GeneratedIdea.project_id == project_id)).all()
    previous_topics = [idea.topic for idea in previous_ideas]

    try:
        new_ideas = generate_ideas(
            posts=[
                {"content": p.content, "topic": p.topic, "keywords": p.keywords_detected}
                for p in posts
            ],
            count=body.count,
            avoid_topics=previous_topics,
            business_name=project.target_business,
        )
    except AIProviderError as e:
        print(f"[generate-ideas] failed: {e}")
        raise HTTPException(status_code=503, detail="AI is busy right now, please try again in a minute.")

    saved = []
    for idea in new_ideas:
        record = GeneratedIdea(
            project_id=project_id,
            topic=idea["topic"],
            draft_copy=idea["draft_copy"],
            cta_suggested=idea["cta_suggested"],
            image_concept=idea["image_concept"],
            keywords=", ".join(idea.get("keywords", [])),
        )
        session.add(record)
        saved.append(record)
    session.commit()
    for r in saved:
        session.refresh(r)

    return saved

@app.delete("/competitors/{competitor_id}")
def delete_competitor(competitor_id: int, session: Session = Depends(get_session)):
    competitor = session.get(Competitor, competitor_id)
    if not competitor:
        raise HTTPException(status_code=404, detail="Competitor not found")
 
    for post in session.exec(select(ScrapedPost).where(ScrapedPost.competitor_id == competitor_id)).all():
        session.delete(post)
    for log in session.exec(select(ScrapeLog).where(ScrapeLog.competitor_id == competitor_id)).all():
        session.delete(log)
    session.flush()  # children are gone before the rival itself is deleted
 
    session.delete(competitor)
    session.commit()
    return {"status": "deleted"}
 
 
# 2) ADD these two routes anywhere below the generate-ideas route.
 
@app.get("/projects/{project_id}/ideas/", response_model=List[GeneratedIdea])
def list_project_ideas(project_id: int, session: Session = Depends(get_session)):
    return session.exec(
        select(GeneratedIdea).where(GeneratedIdea.project_id == project_id).order_by(GeneratedIdea.id)
    ).all()
 
 
@app.delete("/ideas/{idea_id}")
def delete_idea(idea_id: int, session: Session = Depends(get_session)):
    idea = session.get(GeneratedIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404, detail="Idea not found")
    session.delete(idea)
    session.commit()
    return {"status": "deleted"}