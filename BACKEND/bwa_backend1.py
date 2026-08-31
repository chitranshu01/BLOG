from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import operator
import os
import re
from html import escape as html_escape
from urllib.parse import quote as url_quote
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

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
    needs_entity_verification: bool = False


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Visual planning schema ----
class DiagramNode(BaseModel):
    id: str = Field(..., description="Short unique node id, e.g. query or llm.")
    label: str = Field(..., description="Short, readable English label.")
    description: Optional[str] = None


class DiagramEdge(BaseModel):
    source: str
    target: str
    label: Optional[str] = None


class ChartPoint(BaseModel):
    label: str
    value: float


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Filename hint for the visual.")
    alt: str
    caption: str
    type: Literal["diagram", "illustration", "chart"] = "diagram"
    prompt: str = Field(..., description="Prompt used for AI illustration only.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"
    diagram_nodes: List[DiagramNode] = Field(default_factory=list)
    diagram_edges: List[DiagramEdge] = Field(default_factory=list)
    chart_title: Optional[str] = None
    chart_kind: Optional[Literal["bar", "line"]] = None
    chart_data: List[ChartPoint] = Field(default_factory=list)


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class VisualDecision(BaseModel):
    needs_images: bool
    image_count: int = Field(default=0, ge=0, le=3)
    rationale: str


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    needs_entity_verification: bool
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

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatOpenAI(
    model="qwen/qwen3-30b-a3b-instruct-2507",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    timeout=120,
    max_retries=1,
)

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book: evergreen concepts that do not depend on current facts.
- hybrid: evergreen topic plus current examples, tools, models, or practices.
- open_book: volatile weekly/news/latest/pricing/policy topics.

Entity verification:
- If the topic names or clearly refers to a real-world person, public figure,
  company, product, organization, place, event, or other specific entity,
  set needs_entity_verification=true.
- For a real-world entity, set needs_research=true unless the request is clearly
  fictional or creative.
- When entity verification is needed, include identity queries such as:
  "<topic> official biography/about", "<topic> official profile", and
  "<topic> recent news" as appropriate.
- Never infer or assume a person's profession, employer, achievements, or other
  biographical facts from the topic alone.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- Prefer authoritative and primary sources.
- For open_book weekly roundup, include queries reflecting the last 7 days.

Return a routing decision only."""


def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    queries = list(decision.queries)

    if decision.needs_entity_verification:
        verification_queries = [
            f"{state['topic']} official biography about",
            f"{state['topic']} official profile",
            f"{state['topic']} recent news",
        ]
        queries = list(dict.fromkeys(verification_queries + queries))

    return {
        "needs_research": decision.needs_research or decision.needs_entity_verification,
        "mode": decision.mode,
        "queries": queries,
        "recency_days": recency_days,
        "needs_entity_verification": decision.needs_entity_verification,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        print("TAVILY_API_KEY is missing.")
        return []

    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=max_results)
        
        response = tool.invoke({"query": query})
        print("TAVILY RAW RESPONSE:", response)

        # TavilySearch returns a structured response.
        if isinstance(response, dict):
            results = response.get("results", [])
        else:
            results = response or []

        out: List[dict] = []

        for r in results:
            if not isinstance(r, dict):
                continue

            url = r.get("url") or ""

            if not url:
                continue

            out.append(
                {
                    "title": r.get("title") or "",
                    "url": url,
                    "snippet": (
                        r.get("content")
                        or r.get("snippet")
                        or ""
                    ),
                    "published_at": (
                        r.get("published_date")
                        or r.get("published_at")
                    ),
                    "source": r.get("source"),
                }
            )

        print(f"TAVILY QUERY: {query}")
        print(f"TAVILY RESULTS: {len(out)}")
        

        return out

    except Exception as e:
        print(
            "TAVILY ERROR:",
            type(e).__name__,
            str(e),
        )
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
- Prefer relevant, authoritative, and primary sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null.
- Keep snippets short.
- Deduplicate by URL.
- Never invent a source, title, date, author, quote, or factual detail.
"""


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

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

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

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
                "needs_entity_verification": state.get("needs_entity_verification", False),
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
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

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- Never invent a person, event, organization, company, product, date, statistic,
  role, quote, announcement, or source.
- If needs_entity_verification==true, every factual claim about the named entity must
  be supported by provided Evidence URLs.
- If mode=="open_book", do not introduce any specific current event/company/model/
  funding/policy claim unless supported by provided Evidence URLs.
- For each supported external claim, attach a Markdown link ([Source](URL)).
- If a requested factual point cannot be verified from the provided evidence,
  write "Not verified in the retrieved sources." rather than guessing.
- If requires_citations==true (hybrid tasks), cite Evidence URLs for external claims.

Formatting:
- Use normal Markdown headings and bullets.
- Do not output literal "\\n", "\\#", or "\\*".

Code:
- If requires_code==true, include at least one minimal snippet.
"""
def _append_verified_sources(
    section_md: str,
    evidence: List[EvidenceItem],
) -> str:
    valid = [e for e in evidence if e.url]

    if not valid:
        return section_md

    # Remove any hallucinated [Source](...) links.
    section_md = re.sub(
        r"\[Source\]\([^)]*\)",
        "",
        section_md,
    )

    # Add only real retrieved sources.
    source_block = "\n\n### Sources\n\n"

    for e in valid[:5]:
        title = e.title or e.url
        source_block += f"- [{title}]({e.url})\n"

    return section_md.rstrip() + source_block

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    needs_entity_verification = bool(payload.get("needs_entity_verification", False))

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
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
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()
    if evidence and (
        task.requires_citations
        or payload.get("mode") in {"hybrid", "open_book"}
        or payload.get("needs_entity_verification", False)
    ):
        section_md = _append_verified_sources(
        section_md,
        evidence,
    )


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
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


VISUAL_DECISION_SYSTEM = """You are the visual decision module for an AI blog writer.

Your job in this step is ONLY to decide whether this topic/article needs visuals.
Do NOT create prompts or image specifications yet.

Decision rules:
- Return needs_images=false when visuals would add little value (simple biography,
  short opinion, basic definition, simple news update with no useful visual story).
- Return needs_images=true when a diagram, workflow, architecture, comparison,
  timeline, quantitative chart, or conceptual illustration would materially improve
  understanding.
- Prefer 1 visual for a simple useful case, 2 for substantial technical content,
  and up to 3 only when every visual has a distinct purpose.
- For programming, AI, software engineering, architecture, workflows, APIs,
  databases, machine learning, or system design, seriously consider a diagram.
- Do not create visuals merely because the user asked for a maximum count.

Return strictly VisualDecision.
"""

IMAGE_PLAN_SYSTEM = """You are the visual planning module for a professional technical blog.

A separate decision module has already determined that visuals are useful.
Create only the approved number of visual specifications.

Visual types:
1. diagram
   - For architecture, workflows, pipelines, processes, system design, technical
     concepts, or component relationships.
   - The application will render diagrams as SVG, so supply short English labels
     in diagram_nodes and diagram_edges.
2. chart
   - Only when the article contains real quantitative data.
   - Supply exact chart_data supported by the article/evidence.
   - The application renders the chart programmatically.
3. illustration
   - For conceptual/decorative artwork.
   - AI Guru Lab will generate it.
   - Do NOT put text, labels, letters, numbers, logos, UI elements, captions, or
     paragraphs into the illustration. Use a visual-only composition.

Rules:
- Every visual must materially improve understanding.
- Never invent quantitative chart data.
- Never invent facts that are not in the article/evidence.
- All titles, diagram labels, edge labels, alt text, and captions must be concise,
  grammatically correct ENGLISH.
- Keep diagram node labels short (ideally 1–4 words).
- Insert placeholders exactly as [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- Return strictly GlobalImagePlan.
"""

def _parse_size(size: str) -> tuple[int, int]:
    try:
        width_s, height_s = size.lower().split("x", 1)
        return int(width_s), int(height_s)
    except Exception:
        return 1024, 1024


def _svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;charset=utf-8," + url_quote(svg, safe="")


def _wrap_svg_text(text: str, max_chars: int = 24) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def _render_diagram_svg(
    title: str,
    nodes: list[DiagramNode],
    edges: list[DiagramEdge],
    width: int,
    height: int,
) -> str:
    safe_title = html_escape(title or "Technical Diagram")
    node_list = nodes[:10]
    if not node_list:
        node_list = [DiagramNode(id="topic", label="Technical Concept")]

    cols = min(4, max(1, len(node_list)))
    rows = (len(node_list) + cols - 1) // cols
    node_w = min(280, (width - 140 - (cols - 1) * 36) / cols)
    node_h = 100
    top = 130
    row_gap = max(55, (height - top - 70 - rows * node_h) / max(1, rows - 1))
    x_positions = [70 + i * (node_w + 36) for i in range(cols)]
    positions: dict[str, tuple[float, float]] = {}

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">',
        '<stop offset="0%" stop-color="#0d1424"/>',
        '<stop offset="100%" stop-color="#182236"/>',
        "</linearGradient>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#8ea2c5"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" rx="24" fill="url(#bg)"/>',
        f'<text x="{width/2}" y="52" text-anchor="middle" fill="#f5f7fb" font-family="Arial, sans-serif" font-size="28" font-weight="700">{safe_title}</text>',
    ]

    for i, node in enumerate(node_list):
        row = i // cols
        col = i % cols
        x = x_positions[col]
        y = top + row * (node_h + row_gap)
        positions[node.id] = (x, y)

    for edge in edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        sx, sy = positions[edge.source]
        tx, ty = positions[edge.target]
        x1 = sx + node_w if tx >= sx else sx
        x2 = tx if tx >= sx else tx + node_w
        y1 = sy + node_h / 2
        y2 = ty + node_h / 2
        svg.append(
            f'<path d="M {x1:.1f} {y1:.1f} C {(x1+x2)/2:.1f} {y1:.1f}, {(x1+x2)/2:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}" '
            'fill="none" stroke="#8ea2c5" stroke-width="2.2" marker-end="url(#arrow)"/>'
        )
        if edge.label:
            svg.append(
                f'<text x="{(x1+x2)/2:.1f}" y="{(y1+y2)/2-8:.1f}" text-anchor="middle" fill="#c7d3e8" '
                f'font-family="Arial, sans-serif" font-size="12">{html_escape(edge.label)}</text>'
            )

    for i, node in enumerate(node_list):
        x, y = positions[node.id]
        fill = "#202d46" if i % 2 == 0 else "#243650"
        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_w:.1f}" height="{node_h}" rx="16" '
            f'fill="{fill}" stroke="#7288ae" stroke-width="1.4"/>'
        )
        lines = _wrap_svg_text(node.label, 26)
        base_y = y + 40 - (len(lines) - 1) * 8
        for j, line in enumerate(lines):
            svg.append(
                f'<text x="{x + node_w/2:.1f}" y="{base_y + j*22:.1f}" text-anchor="middle" '
                f'fill="#ffffff" font-family="Arial, sans-serif" font-size="16" font-weight="700">{html_escape(line)}</text>'
            )

        if node.description:
            desc_lines = _wrap_svg_text(node.description, 34)[:2]
            for j, line in enumerate(desc_lines):
                svg.append(
                    f'<text x="{x + node_w/2:.1f}" y="{y + 80 + j*13:.1f}" text-anchor="middle" '
                    f'fill="#aebbd0" font-family="Arial, sans-serif" font-size="10">{html_escape(line)}</text>'
                )

    svg.append("</svg>")
    return "".join(svg)


def _render_chart_svg(
    title: str,
    points: list[ChartPoint],
    kind: str,
    width: int,
    height: int,
) -> str:
    safe_title = html_escape(title or "Data Chart")
    pts = points[:12] or [ChartPoint(label="No data", value=0)]
    max_value = max(abs(float(p.value)) for p in pts) or 1

    left = 90
    right = width - 50
    top = 120
    bottom = height - 90
    chart_w = right - left
    chart_h = bottom - top

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="24" fill="#111827"/>',
        f'<text x="{width/2}" y="48" text-anchor="middle" fill="#f5f7fb" font-family="Arial, sans-serif" font-size="26" font-weight="700">{safe_title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#7183a4" stroke-width="2"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#7183a4" stroke-width="2"/>',
    ]

    if kind == "line":
        step = chart_w / max(1, len(pts) - 1)
        coords = []
        for i, p in enumerate(pts):
            x = left + i * step
            y = bottom - (float(p.value) / max_value) * (chart_h - 20)
            coords.append((x, y))
        path_d = " ".join(("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}" for i, (x, y) in enumerate(coords))
        svg.append(f'<path d="{path_d}" fill="none" stroke="#7aa2ff" stroke-width="4"/>')
        for (x, y), p in zip(coords, pts):
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#7aa2ff"/>')
            svg.append(
                f'<text x="{x:.1f}" y="{bottom+24}" text-anchor="middle" fill="#c7d3e8" '
                f'font-family="Arial, sans-serif" font-size="11">{html_escape(p.label[:18])}</text>'
            )
    else:
        gap = 14
        bar_w = max(20, (chart_w - gap*(len(pts)+1)) / max(1, len(pts)))
        for i, p in enumerate(pts):
            x = left + gap + i * (bar_w + gap)
            bar_h = (float(p.value) / max_value) * (chart_h - 20)
            y = bottom - bar_h
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(0, bar_h):.1f}" rx="8" fill="#7aa2ff"/>'
            )
            svg.append(
                f'<text x="{x + bar_w/2:.1f}" y="{bottom+24}" text-anchor="middle" fill="#c7d3e8" '
                f'font-family="Arial, sans-serif" font-size="11">{html_escape(p.label[:18])}</text>'
            )
            svg.append(
                f'<text x="{x + bar_w/2:.1f}" y="{max(top+14, y-8):.1f}" text-anchor="middle" fill="#f5f7fb" '
                f'font-family="Arial, sans-serif" font-size="11" font-weight="700">{p.value:g}</text>'
            )

    svg.append("</svg>")
    return "".join(svg)


def decide_images(state: State) -> dict:
    decision_maker = llm.with_structured_output(VisualDecision)
    plan = state["plan"]
    assert plan is not None

    decision = decision_maker.invoke(
        [
            SystemMessage(content=VISUAL_DECISION_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Blog kind: {plan.blog_kind}\n\n"
                    f"Draft article:\n{state['merged_md']}"
                )
            ),
        ]
    )

    print("VISUAL DECISION:", decision.model_dump())

    if not decision.needs_images or decision.image_count <= 0:
        return {
            "md_with_placeholders": state["merged_md"],
            "image_specs": [],
        }

    planner = llm.with_structured_output(GlobalImagePlan)
    image_plan = planner.invoke(
        [
            SystemMessage(content=IMAGE_PLAN_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Approved visual count: {min(decision.image_count, 3)}\n\n"
                    f"Article draft:\n{state['merged_md']}\n\n"
                    f"Research evidence:\n"
                    f"{[e.model_dump() for e in state.get('evidence', [])][:20]}"
                )
            ),
        ]
    )

    image_specs = image_plan.images[:min(decision.image_count, 3)]
    for idx, spec in enumerate(image_specs, start=1):
        spec.placeholder = f"[[IMAGE_{idx}]]"

    # If the planner failed to return enough specs, use only what it returned.
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_specs],
    }


def _aigurulab_generate_image_url(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
) -> str:
    """Generate a text-free decorative illustration using AI Guru Lab."""
    api_key = os.getenv("AIGURULAB_API_KEY")

    if not api_key:
        raise RuntimeError("AIGURULAB_API_KEY is not set.")

    url = "https://aigurulab.tech/api/generate-image"
    safe_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: visual-only illustration. Do not render readable text, letters, "
        "numbers, labels, UI elements, logos, captions, or paragraphs. No typography."
    )

    payload = {
        "width": width,
        "height": height,
        "input": safe_prompt,
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
            f"AI Guru Lab image generation failed: {response.status_code} - {response.text}"
        )

    data = response.json()
    print("AI GURU LAB RESPONSE:")
    print(data)

    image_url = data.get("image")
    if not image_url:
        raise RuntimeError("AI Guru Lab returned no image URL.")

    return image_url


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def _normalize_markdown(markdown: str) -> str:
    return (
        markdown
        .replace("\\n", "\n")
        .replace("\\r", "")
        .replace("\\#", "#")
        .replace("\\*", "*")
        .strip()
    )


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    print("IMAGE SPECS:", image_specs)

    if not image_specs:
        md = _normalize_markdown(md)
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    for spec in image_specs:
        placeholder = spec["placeholder"]
        visual_type = spec.get("type", "diagram")

        try:
            width, height = _parse_size(spec.get("size", "1024x1024"))

            if visual_type == "diagram":
                svg = _render_diagram_svg(
                    title=spec.get("caption") or "Technical Diagram",
                    nodes=[DiagramNode(**n) for n in spec.get("diagram_nodes", [])],
                    edges=[DiagramEdge(**e) for e in spec.get("diagram_edges", [])],
                    width=width,
                    height=height,
                )
                image_src = _svg_data_uri(svg)

            elif visual_type == "chart":
                svg = _render_chart_svg(
                    title=spec.get("chart_title") or spec.get("caption") or "Data Chart",
                    points=[ChartPoint(**p) for p in spec.get("chart_data", [])],
                    kind=spec.get("chart_kind") or "bar",
                    width=width,
                    height=height,
                )
                image_src = _svg_data_uri(svg)

            else:
                image_src = _aigurulab_generate_image_url(
                    spec["prompt"],
                    width=width,
                    height=height,
                )

            img_md = (
                f"![{spec['alt']}]({image_src})\n"
                f"*{spec['caption']}*"
            )
            md = md.replace(placeholder, img_md)

        except Exception as e:
            prompt_block = (
                f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
                f"> **Alt:** {spec.get('alt', '')}\n>\n"
                f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                f"> **Error:** {e}\n"
            )
            md = md.replace(placeholder, prompt_block)

    md = _normalize_markdown(md)
    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")

    return {"final": md}


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

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

api.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class BlogRequest(BaseModel):
    topic: str
    as_of: str
@api.get("/")
def root():
    return {"status": "AI Blog Writer API is running"}


@api.post("/api/generate")
def generate_blog(request: BlogRequest):

    initial_state = {
        "topic": request.topic,
        "mode": "closed_book",
        "needs_research": False,
        "needs_entity_verification": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": request.as_of,
        "recency_days": 3650,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    result = graph_app.invoke(initial_state)
    return {
        "content": result["final"],
        "final": result["final"],
        "mode": result.get("mode", ""),
        "needs_research": result.get("needs_research", False),
        "needs_entity_verification": result.get("needs_entity_verification", False),
        "evidence": [
            e.model_dump() if hasattr(e, "model_dump") else e
            for e in result.get("evidence", [])
        ],
        "image_specs": result.get("image_specs", []),
        "sections": len(result.get("sections", [])),
        "plan": (
            result["plan"].model_dump()
            if result.get("plan") is not None and hasattr(result["plan"], "model_dump")
            else result.get("plan")
        ),
    }