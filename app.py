"""Chula Engineering Expert Finder: semantic search over faculty research profiles."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).parent / "data" / "professors.json"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
DEPARTMENT_NAMES = {
    "chemical_engineering": "Chemical Engineering",
    "civil_engineering": "Civil Engineering",
    "computer_engineering": "Computer Engineering",
    "electrical_engineering": "Electrical Engineering",
    "environmental_sustainable_engineering": "Environmental and Sustainable Engineering",
    "industrial_engineering": "Industrial Engineering",
    "international_school_of_engineering": "International School of Engineering",
    "mechanical_engineering": "Mechanical Engineering",
    "metallurgical_engineering": "Metallurgical Engineering",
    "mining_petroleum_engineering": "Mining and Petroleum Engineering",
    "nuclear_engineering": "Nuclear Engineering",
    "survey_engineering": "Survey Engineering",
    "water_resources_engineering": "Water Resources Engineering",
}


def _clean_text(value: Any) -> str:
    """Convert optional dataset values to clean strings without losing Thai text."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_interests(value: Any) -> list[str]:
    """Handle interest lists, strings, and paragraph-like source values."""
    if isinstance(value, list):
        pieces = [_clean_text(item) for item in value]
    elif isinstance(value, (tuple, set)):
        pieces = [_clean_text(item) for item in value]
    else:
        text = _clean_text(value)
        pieces = re.split(r"[\n;|•]+", text) if text else []
    # Keep only source-provided, non-empty interests and remove exact duplicates.
    return list(dict.fromkeys(item.strip(" ,") for item in pieces if item.strip(" ,")))


def load_data(path: Path = DATA_PATH) -> tuple[list[dict[str, Any]], str]:
    """Read JSON or CSV faculty records and return normalized profiles plus file hash."""
    if not path.exists():
        raise FileNotFoundError(f"Faculty dataset was not found: {path}")
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        if path.suffix.lower() == ".json":
            raw_records = json.loads(raw_bytes.decode("utf-8-sig"))
            if isinstance(raw_records, dict):
                raw_records = raw_records.get("professors", raw_records.get("faculty", []))
        elif path.suffix.lower() == ".csv":
            raw_records = pd.read_csv(path, encoding="utf-8-sig").to_dict("records")
        else:
            raise ValueError("Only JSON and CSV faculty datasets are supported.")
    except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read the faculty dataset: {error}") from error
    if not isinstance(raw_records, list):
        raise ValueError("The dataset must contain a list of faculty records.")
    return normalize_faculty_data(raw_records), digest


def normalize_faculty_data(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map the observed Chula dataset fields to one consistent schema."""
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in records:
        if not isinstance(row, dict):
            continue
        name_en = _clean_text(row.get("name_en"))
        name_th = _clean_text(row.get("name_th"))
        faculty_id = _clean_text(row.get("professor_id"))
        # Prefer the stable source ID; use a name identity only when it is absent.
        identity = ("id", faculty_id.casefold()) if faculty_id else (
            "name", (name_en or name_th).casefold()
        )
        if not identity[1] or identity in seen:
            continue
        seen.add(identity)
        department_id = _clean_text(row.get("department_id"))
        normalized.append({
            "professor_id": faculty_id,
            "name_th": name_th,
            "name_en": name_en,
            "department": DEPARTMENT_NAMES.get(
                department_id, department_id.replace("_", " ").title() or "Not listed"
            ),
            "academic_title": _clean_text(row.get("academic_title")),
            "email": _clean_text(row.get("email")),
            "research_interests": _normalize_interests(row.get("research_interests")),
            "research_description": _clean_text(row.get("research_description")),
            "profile_url": _clean_text(row.get("profile_url")),
            "source_url": _clean_text(row.get("source_url")),
        })
    return normalized


def build_search_text(profile: dict[str, Any]) -> str:
    """Build model input only from research information in the faculty dataset."""
    interests = "; ".join(profile["research_interests"])
    description = profile["research_description"]
    return " ".join(part for part in (
        f"Research interests: {interests}" if interests else "",
        f"Research description: {description}" if description else "",
    ) if part)


@st.cache_resource(show_spinner="Loading multilingual embedding model (first run may take a while)…")
def load_embedding_model():
    """Load the multilingual sentence embedding model once per Streamlit session."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="mps")

@st.cache_data(show_spinner="Preparing faculty research profiles…")
def generate_embeddings(profiles: list[dict[str, Any]], dataset_hash: str):
    """Encode profile research text; the dataset hash invalidates cached vectors."""
    model = load_embedding_model()
    usable = [(profile, build_search_text(profile)) for profile in profiles]
    usable = [(profile, text) for profile, text in usable if text]
    if not usable:
        return [], [], []
    documents = [text for _, text in usable]
    # Qwen3 provides task-aware helpers; its document encoder keeps profile
    # text in the same retrieval space as encode_query() below.
    encode_documents = getattr(model, "encode_document", model.encode)
    vectors = encode_documents(
        documents, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False, batch_size=16,
    )
    return [profile for profile, _ in usable], [text for _, text in usable], vectors


def semantic_search(query: str, profiles: list[dict[str, Any]], profile_texts: list[str],
                    vectors, top_k: int) -> list[dict[str, Any]]:
    """Retrieve the most similar research profiles using normalized cosine similarity."""
    if not query.strip() or not profiles:
        return []
    model = load_embedding_model()
    # Use the model's query prompt when supported (Qwen3-Embedding does this).
    encode_query = getattr(model, "encode_query", model.encode)
    query_vector = encode_query(
        [query.strip()], normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    )
    similarities = cosine_similarity(query_vector, vectors)[0]
    order = similarities.argsort()[::-1][:top_k]
    return [{"profile": profiles[index], "text": profile_texts[index],
             "semantic_score": float(similarities[index])} for index in order]


@st.cache_resource(show_spinner="Loading multilingual reranker…")
def load_reranker():
    """Load Qwen's multilingual research-profile reranker; caller handles failures."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(
        RERANKER_MODEL_NAME,
        device="mps",
        prompts={
            "faculty_search": (
                "Given a research topic, rank faculty research profiles by relevance."
            )
        },
        default_prompt_name="faculty_search",
    )


def rerank_candidates(query: str, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool, str]:
    """Rerank query-profile pairs; return semantic order if model setup or inference fails."""
    if not candidates:
        return [], False, "No candidates to rerank."
    try:
        model = load_reranker()
        pairs = [(query, item["text"]) for item in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked = [dict(item, rerank_score=float(score))
                  for item, score in zip(candidates, scores)]
        return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True), True, ""
    except Exception as error:  # Network/model errors should not block semantic search.
        return candidates, False, str(error)


def _safe_url(value: str) -> bool:
    """Accept only absolute HTTP(S) links from the dataset."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def display_expert_card(result: dict[str, Any], rank: int, reranked: bool) -> None:
    """Render a faculty card with source-backed research evidence and contact links."""
    profile = result["profile"]
    name = profile["name_en"] or profile["name_th"] or "Name not listed"
    if profile["name_en"] and profile["name_th"]:
        name += f" · {profile['name_th']}"
    with st.container(border=True):
        st.markdown(f"### {rank}. {name}")
        if profile["academic_title"]:
            st.caption(profile["academic_title"])
        st.write(f"**Department:** {profile['department']}")
        if profile["research_interests"]:
            st.write("**Research interests:** " + ", ".join(profile["research_interests"]))
        if profile["research_description"]:
            st.write("**Research description:** " + profile["research_description"])
        st.caption("Matching evidence comes from the listed research interests and description.")
        details = []
        if profile["email"]:
            st.write(f"**Email:** {profile['email']}")
        if _safe_url(profile["profile_url"]):
            details.append(f"[Faculty profile]({profile['profile_url']})")
        if _safe_url(profile["source_url"]):
            details.append(f"[Research source]({profile['source_url']})")
        if details:
            st.markdown(" · ".join(details))
        metric_label = "Cross-Encoder relevance score" if reranked else "Cosine similarity score"
        metric = result.get("rerank_score") if reranked else result["semantic_score"]
        st.caption(f"{metric_label}: {metric:.4f} (ranking signal, not an expertise probability)")


def render_dashboard(profiles: list[dict[str, Any]]) -> None:
    """Show dataset coverage and an interactive department count chart."""
    total = len(profiles)
    with_interests = sum(bool(p["research_interests"] or p["research_description"]) for p in profiles)
    with_email = sum(bool(p["email"]) for p in profiles)
    departments = sorted({p["department"] for p in profiles if p["department"]})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Faculty profiles", total)
    col2.metric("Departments", len(departments))
    col3.metric("Profiles with research data", with_interests)
    col4.metric("Profiles with email", with_email)
    counts = pd.DataFrame([
        {"Department": department, "Faculty members": sum(p["department"] == department for p in profiles)}
        for department in departments
    ]).sort_values("Faculty members", ascending=True)
    if not counts.empty:
        chart = px.bar(counts, x="Faculty members", y="Department", orientation="h",
                       title="Faculty members by department", color="Faculty members",
                       color_continuous_scale=["#f4c7d5", "#a3164b"])
        chart.update_layout(margin=dict(l=10, r=10, t=55, b=10), coloraxis_showscale=False)
        st.plotly_chart(chart, use_container_width=True)


def main() -> None:
    """Build the Streamlit interface and run search only after explicit submission."""
    st.set_page_config(page_title="Chula Engineering Expert Finder", page_icon="🔎",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
      .stApp { background: #fff; }
      [data-testid="stHeader"] { background: rgba(255,255,255,.92); }
      h1, h2, h3 { color: #8f1745; }
      .hero { padding: 1.25rem 1.5rem; border-radius: 16px; background: linear-gradient(110deg,#fff7fa,#fff); border: 1px solid #f2dce5; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title("Chula Engineering Expert Finder")
    st.subheader("Discover Engineering Experts Through AI-Powered Research Matching")
    st.write("Explore recommendations based on publicly available faculty research profiles. "
             "This is an independent academic project and is not an officially endorsed university product.")
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        profiles, dataset_hash = load_data()
    except (OSError, ValueError) as error:
        st.error(f"Faculty data could not be loaded. {error}")
        st.info("Check that data/professors.json exists and contains valid UTF-8 JSON.")
        return

    with st.expander("Dataset overview", expanded=True):
        render_dashboard(profiles)
    departments = sorted({p["department"] for p in profiles if p["department"]})
    with st.sidebar:
        st.header("Search settings")
        department_filter = st.selectbox("Department", ["All departments", *departments])
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=8)
        require_interests = st.checkbox("Exclude profiles without research-interest data", value=True)
        st.caption("Both English and Thai queries are supported.")

    suggestions = [
        "Computer vision for detecting electronic waste",
        "Machine learning for medical image analysis",
        "Renewable energy and battery optimization",
        "การใช้ AI วิเคราะห์ภาพทางการแพทย์",
        "การจัดการขยะและเศรษฐกิจหมุนเวียน",
    ]
    st.markdown("#### Suggested topics")
    selected_topic = st.selectbox("Choose a starting topic (optional)", [""] + suggestions,
                                  label_visibility="collapsed")
    with st.form("expert_search"):
        query = st.text_area("Describe a research topic or technical problem",
                             value=selected_topic, height=110,
                             placeholder="For example: machine learning for flood prediction")
        submitted = st.form_submit_button("Find relevant faculty", type="primary", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Enter a research topic before searching.")
            return
        filtered = [p for p in profiles if (department_filter == "All departments" or p["department"] == department_filter)
                    and (not require_interests or bool(p["research_interests"] or p["research_description"]))]
        if not filtered:
            st.info("No faculty profiles match the selected filters.")
            return
        try:
            with st.spinner("Encoding faculty research profiles and retrieving candidates…"):
                searchable, texts, vectors = generate_embeddings(profiles, dataset_hash)
                allowed_ids = {p["professor_id"] for p in filtered}
                filtered_profiles = [p for p in searchable if p["professor_id"] in allowed_ids]
                filtered_texts = [text for p, text in zip(searchable, texts) if p["professor_id"] in allowed_ids]
                filtered_vectors = vectors[[i for i, p in enumerate(searchable) if p["professor_id"] in allowed_ids]]
                candidates = semantic_search(query, filtered_profiles, filtered_texts, filtered_vectors,
                                             min(len(filtered_profiles), max(top_k * 3, top_k)))
            if not candidates:
                st.info("No searchable research profiles are available for these filters.")
                return
            with st.spinner("Reranking candidate research profiles…"):
                ranked, reranked, rerank_error = rerank_candidates(query, candidates)
            if reranked:
                ranked = ranked[:top_k]
                st.success(f"Showing {len(ranked)} matches, reranked with a multilingual Cross-Encoder.")
            else:
                ranked = ranked[:top_k]
                st.warning(f"Details: {rerank_error[:240]}")
            for rank, item in enumerate(ranked, start=1):
                display_expert_card(item, rank, reranked)
        except Exception as error:
            st.error("The search models could not be loaded or run. Check network access and installed dependencies, then retry.")
            st.caption(str(error)[:500])

    with st.expander("About the search method and limitations"):
        st.markdown(f"""
        1. You describe a topic in English or Thai.
        2. **Semantic retrieval** encodes the query and faculty research profiles with `{EMBEDDING_MODEL_NAME}`.
        3. Normalized embeddings are compared with cosine similarity to retrieve candidates.
        4. **Reranking** compares each query directly with each candidate profile using `{RERANKER_MODEL_NAME}`.
        5. Results display the source-provided interests and descriptions used as matching evidence.

        Scores are ranking signals, not calibrated expertise probabilities. A research-interest match
        does not guarantee a faculty member's availability or willingness to collaborate. The reranker
        model card specifies a research/evaluation-only license; check its terms before other uses.
        """)


if __name__ == "__main__":
    main()
