from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import operator
import os
BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    "http://127.0.0.1:8000"
)
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)
BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    "http://127.0.0.1:8000"
)

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str
    audience: str
    tone: str
    requested_mode: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    include_code: bool
    include_images: bool

    final: str


def resolve_router_mode(topic: str, llm_decision: Optional[dict] = None) -> str:
    text = (topic or "").strip().lower()
    if not text:
        return "closed_book"

    recent_keywords = [
        "latest", "breaking", "this week", "recent", "news", "today", "just announced",
        "new release", "launch", "pricing", "policy", "update", "live", "current"
    ]
    if any(keyword in text for keyword in recent_keywords):
        return "open_book"

    if llm_decision:
        if llm_decision.get("needs_research") is True:
            if str(llm_decision.get("mode", "")).lower() == "open_book":
                return "open_book"
            if str(llm_decision.get("mode", "")).lower() == "hybrid":
                return "hybrid"
        if llm_decision.get("needs_research") is False:
            return "closed_book"

    comparison_keywords = ["compare", "vs", "difference between", "versus", "which is better", "adoption"]
    if any(keyword in text for keyword in comparison_keywords):
        return "hybrid"

    teaching_keywords = ["what is", "how does", "how to", "overview", "basics", "introduction", "guide", "explain", "tutorial"]
    if any(keyword in text for keyword in teaching_keywords):
        return "closed_book"

    return "hybrid"


def sanitize_markdown_for_user_settings(markdown: str, allow_code: bool = True, allow_images: bool = True) -> str:
    text = markdown or ""
    if not allow_code:
        text = re.sub(r"```[\s\S]*?```", "\n", text)
        text = re.sub(r"`[^`\n]+`", "", text)
    if not allow_images:
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _valid_source_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    return True


def _source_value(item, key: str, default=""):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _clean_evidence(evidence: list) -> list:
    items = []
    seen_urls = set()
    for item in evidence or []:
        url = _source_value(item, "url")
        if not _valid_source_url(str(url or "")):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        items.append(item)
    return items


# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatOpenRouter(
    model="qwen/qwen3-30b-a3b-instruct-2507",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_retries=2,
)

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
- Add "official", "primary source", or the relevant authority to queries when useful.
"""

def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    requested_mode = state.get("requested_mode", "auto")
    if requested_mode in {"closed_book", "hybrid", "open_book"}:
        resolved_mode = requested_mode
    else:
        resolved_mode = resolve_router_mode(
            state["topic"],
            decision.model_dump() if hasattr(decision, "model_dump") else {
                "needs_research": decision.needs_research,
                "mode": decision.mode,
            },
        )
        if resolved_mode == "closed_book":
            resolved_mode = "hybrid"

    if resolved_mode == "open_book":
        recency_days = 7
    elif resolved_mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": bool(decision.needs_research or resolved_mode in {"hybrid", "open_book"}),
        "mode": resolved_mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": f"{query} official primary source research paper"})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer current official and primary sources: government sites, standards bodies,
  official project documentation, universities, peer-reviewed papers, and original
  company announcements. Use reputable journalism only when no primary source exists.
- Never invent, rewrite, or guess a URL. Copy URLs exactly from the raw results.
- Prefer the newest relevant result for time-sensitive topics.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_clean_evidence(_tavily_search(q, max_results=6)))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    allowed_urls = {item["url"] for item in raw if item.get("url")}
    dedup = {}
    for e in pack.evidence:
        if e.url in allowed_urls:
            dedup[e.url] = e
    evidence = list(dedup.values())
    if not evidence:
        evidence = [EvidenceItem(**item) for item in raw if item.get("url")]

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        dated = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]
        if dated:
            evidence = dated

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
                "audience": state.get("audience", "General technical readers"),
                "tone": state.get("tone", "Clear and practical"),
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".
- Write complete article prose under the heading: at least 2 paragraphs and 120 words.
- Never return a list of headings, an outline, or headings without explanatory text.
- Include source links when a fact or claim is based on the provided evidence.
- Use inline citations like [Source](URL) right next to the sentence or claim that comes from evidence.
- If the evidence does not support a statement, say "Not found in provided sources." rather than inventing it.

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
- If requires_code==false, do not include code blocks or inline code snippets.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in _clean_evidence(payload.get("evidence", []))]
    audience = payload.get("audience") or plan.audience or "General technical readers"
    tone = payload.get("tone") or plan.tone or "Clear and practical"

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    prompt = (
        f"Blog title: {plan.blog_title}\n"
        f"Audience: {audience}\n"
        f"Tone: {tone}\n"
        f"Blog kind: {plan.blog_kind}\n"
        f"Constraints: {plan.constraints}\n"
        f"Topic: {payload['topic']}\n"
        f"Mode: {payload.get('mode')}\n"
        f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
        f"Section title: {task.title}\n"
        f"Goal: {task.goal}\n"
        f"Target words: {task.target_words}\n"
        f"Tags: {task.tags}\n"
        f"requires_research: {task.requires_research}\n"
        f"requires_citations: {task.requires_citations}\n"
        f"requires_code: {task.requires_code}\n"
        f"Audience hint: Write for {audience}. Use the right depth for that audience. If the audience is science-minded, use more technical detail; if the audience is general readers, explain clearly without jargon.\n"
        f"Bullets:{bullets_text}\n\n"
        f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
    )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(content=prompt),
        ]
    ).content.strip()

    prose = re.sub(r"^\s*#{1,6}\s+.*$", "", section_md, flags=re.MULTILINE).strip()
    if len(prose.split()) < 40:
        section_md = llm.invoke(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(content=f"The previous response was only an outline. Rewrite it as a finished section with complete prose under the heading.\n\n{prompt}"),
            ]
        ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    evidence = _clean_evidence(state.get("evidence", []) or [])
    source_links = []
    for item in evidence[:10]:
        url = _source_value(item, "url")
        title = _source_value(item, "title", "Source")
        if url:
            source_links.append(f"- [{title}]({url})")
    sources_section = "\n\n## Sources\n" + "\n".join(source_links) if source_links else ""
    merged_md = f"# {plan.blog_title}\n\n{body}\n{sources_section}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.

Decide whether this technical blog should contain useful technical visuals.

Rules:
- Prefer 1–3 images when they materially improve understanding.
- For programming, AI, software engineering, architecture, workflows, APIs, databases,
  machine learning, or system-design topics, strongly prefer at least 1 technical visual.
- Never create decorative or generic stock-style images.
- Prefer diagrams, architecture diagrams, flowcharts, conceptual technical illustrations,
  or step-by-step visual explanations.
- Insert placeholders exactly as [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If images would not improve understanding, return images=[].
- Every proposed image must include a useful prompt for the image generator.
- Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    if not state.get("include_images", True):
        return {
            "md_with_placeholders": state["merged_md"],
            "image_specs": [],
        }

    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _aigurulab_generate_image_url(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
) -> str:
    """
    Generate an image using AI Guru Lab and return its public image URL.
    """

    api_key = os.getenv("AIGURULAB_API_KEY")

    if not api_key:
        raise RuntimeError("AIGURULAB_API_KEY is not set.")

    url = "https://aigurulab.tech/api/generate-image"

    payload = {
        "width": width,
        "height": height,
        "input": prompt,
        "model": "sdxl",
    }

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"AI Guru Lab image generation failed: "
            f"{response.status_code} - {response.text}"
        )

    data = response.json()

    print("AI GURU LAB RESPONSE:")
    print(data)

    image_url = data.get("image") or data.get("image_url") or data.get("url")
    if not image_url and isinstance(data.get("data"), dict):
        image_url = (
            data["data"].get("image")
            or data["data"].get("image_url")
            or data["data"].get("url")
        )

    if not image_url:
        raise RuntimeError(
            "AI Guru Lab returned no image URL."
        )

    return image_url


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    include_images = bool(state.get("include_images", True))
    include_code = bool(state.get("include_code", True))
    print("IMAGE SPECS:", image_specs)

    if not include_images or not image_specs:
        final_md = sanitize_markdown_for_user_settings(md, allow_code=include_code, allow_images=False)
        filename = f"{_safe_slug(plan.blog_title)}.md"
        (BASE_DIR / filename).write_text(final_md, encoding="utf-8")
        return {"final": final_md}

    images_dir = IMAGES_DIR
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = Path(spec["filename"]).name
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                image_url = _aigurulab_generate_image_url(spec["prompt"])
                image_response = requests.get(image_url, timeout=120)
                image_response.raise_for_status()
                out_path.write_bytes(image_response.content)
                img_md = (
                    f"![{spec['alt']}]({BACKEND_BASE_URL}/images/{filename})\n"
                    f"*{spec['caption']}*"
                )
                md = md.replace(placeholder, img_md)
            except Exception:
                md = md.replace(placeholder, "")
                continue
        else:
            img_md = (
                f"![{spec['alt']}]"
                f"({BACKEND_BASE_URL}/images/{filename})\n"
                f"*{spec['caption']}*"
            )
            md = md.replace(placeholder, img_md)

    final_md = sanitize_markdown_for_user_settings(md, allow_code=include_code, allow_images=True)
    filename = f"{_safe_slug(plan.blog_title)}.md"
    (BASE_DIR / filename).write_text(final_md, encoding="utf-8")
    return {"final": final_md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

graph_app = g.compile()

api = FastAPI(title="AI Blog Writer API")
api.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

frontend_url = os.getenv("FRONTEND_URL", "").rstrip("/")
api.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        frontend_url,
    ] if origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class BlogRequest(BaseModel):
    topic: str
    as_of: str
    audience: Optional[str] = None
    tone: Optional[str] = None
    mode: Optional[str] = None
    blog_kind: Optional[str] = None
    requires_code: Optional[bool] = None
    include_code: Optional[bool] = None
    include_images: Optional[bool] = None

@api.get("/")
def root():
    return {"status": "AI Blog Writer API is running"}


@api.post("/api/generate")
def generate_blog(request: BlogRequest):
    allow_code = bool(request.requires_code if request.requires_code is not None else request.include_code if request.include_code is not None else True)
    allow_images = bool(request.include_images if request.include_images is not None else True)
    requested_mode = request.mode or "auto"

    initial_state = {
        "topic": request.topic,
        "audience": request.audience or "General technical readers",
        "tone": request.tone or "Clear and practical",
        "requested_mode": requested_mode,
        "mode": requested_mode,
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": request.as_of,
        "recency_days": 3650,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "include_code": allow_code,
        "include_images": allow_images,
        "final": "",
    }

    result = graph_app.invoke(initial_state)
    final_md = sanitize_markdown_for_user_settings(
        result.get("final", ""),
        allow_code=allow_code,
        allow_images=allow_images,
    )

    return {
        "content": final_md,
        "final": final_md,
        "mode": result.get("mode", requested_mode),
        "evidence": [e.model_dump() if hasattr(e, "model_dump") else e for e in result.get("evidence", [])],
        "image_specs": result.get("image_specs", []),
        "plan": (
            result["plan"].model_dump() if result.get("plan") is not None and hasattr(result["plan"], "model_dump") else result.get("plan")
        ),
    }