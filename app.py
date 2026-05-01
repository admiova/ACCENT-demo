import streamlit as st
import pandas as pd
import os
import re
import json
from pathlib import Path
import pdfplumber
import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "merged_data_clean.csv"

os.environ["ANTHROPIC_API_KEY"] = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
api_key = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-6"

INT1_PATH = BASE_DIR.parent / "int1_processing" / "KORT_1_INT_1_2022_(lr).pdf"
INT1_CACHE_PATH = BASE_DIR / "int1_cache.json"

# ---------------------------------------------------------------------------
# INT1 section index
# ---------------------------------------------------------------------------

SECTIONS = [
    {"code": "C", "page_start": 14, "page_end": 16, "name_en": "Natural Features"},
    {"code": "D", "page_start": 17, "page_end": 19, "name_en": "Cultural Features"},
    {"code": "E", "page_start": 20, "page_end": 22, "name_en": "Landmarks"},
    {"code": "F", "page_start": 23, "page_end": 26, "name_en": "Ports"},
    {"code": "G", "page_start": 27, "page_end": 29, "name_en": "Topographic Terms"},
    {"code": "H", "page_start": 30, "page_end": 32, "name_en": "Tides and Currents"},
    {"code": "I", "page_start": 33, "page_end": 35, "name_en": "Depths"},
    {"code": "J", "page_start": 36, "page_end": 37, "name_en": "Nature of the Seabed"},
    {"code": "K", "page_start": 38, "page_end": 41, "name_en": "Rocks, Wrecks and Obstructions"},
    {"code": "L", "page_start": 42, "page_end": 45, "name_en": "Offshore Installations"},
    {"code": "M", "page_start": 46, "page_end": 51, "name_en": "Tracks and Routes"},
    {"code": "N", "page_start": 52, "page_end": 56, "name_en": "Areas and Limits"},
    {"code": "O", "page_start": 57, "page_end": 58, "name_en": "Hydrographic Terms"},
    {"code": "P", "page_start": 59, "page_end": 66, "name_en": "Lights"},
    {"code": "Q", "page_start": 67, "page_end": 73, "name_en": "Buoys and Beacons"},
    {"code": "R", "page_start": 74, "page_end": 74, "name_en": "Fog Signals"},
    {"code": "S", "page_start": 75, "page_end": 76, "name_en": "Radar, Radio and Satellite Navigation Systems"},
    {"code": "T", "page_start": 77, "page_end": 78, "name_en": "Services"},
    {"code": "U", "page_start": 79, "page_end": 80, "name_en": "Small Craft (Leisure) Facilities"},
]

# ---------------------------------------------------------------------------
# INT1 loading — cached as JSON dict keyed by section code
# ---------------------------------------------------------------------------

@st.cache_data
def load_int1_sections() -> dict:
    """Load INT1 sections from JSON cache if available, otherwise parse PDF."""
    if INT1_CACHE_PATH.exists():
        return json.loads(INT1_CACHE_PATH.read_text(encoding="utf-8"))

    sections_dict = {}
    with pdfplumber.open(INT1_PATH) as pdf:
        for sec in SECTIONS:
            parts = []
            for page_num in range(sec["page_start"], sec["page_end"] + 1):
                page = pdf.pages[page_num - 1]
                text = page.extract_text()
                if text:
                    parts.append(text)
            sections_dict[sec["code"]] = {
                "name_en": sec["name_en"],
                "text": "\n\n".join(parts),
            }

    INT1_CACHE_PATH.write_text(json.dumps(sections_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    return sections_dict

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SECTION_PROMPT = """You are a nautical charting expert.
 
Your task is to identify which INT1 / KORT1 sections are relevant for the Notice to Mariners.
 
Instructions:
- Read the notice carefully and identify the main chart-relevant feature or hazard.
- Focus on what is being charted, not how it is marked or buoyed.
- Buoys, beacons and lights are secondary marking features — only include section Q (Buoys and Beacons) or P (Lights) if the buoy or light IS the main charted feature.
- Include additional sections only if they are clearly relevant for distinct secondary features.
- Return only sections that are clearly relevant. Fewer is better.
- Use only section letters from the list below.
- Choose based on charting representation, not ordinary word meaning.
 
Notice:
{notice_text}
 
Available INT1 / KORT1 sections:
C - Natural Features — coastline, relief, water features, lava, supplementary national symbols
D - Cultural Features — settlements/buildings, roads/railways/airfields, other cultural features
E - Landmarks — landmarks visible from sea, religious buildings, towers, chimneys
F - Ports — protection structures, harbour installations, canals and barrages, transhipment facilities
G - Topographic Terms — general geographical and topographic terminology
H - Tides and Currents — tidal levels, tide tables, tidal streams and currents
I - Depths — soundings, depths in fairways and areas, depth contours
J - Nature of the Seabed — types of seabed, intertidal areas, qualifying terms
K - Rocks, Wrecks and Obstructions — rocks, wrecks and fouls, obstructions and aquaculture
L - Offshore Installations — platforms and moorings, underwater installations, submarine cables, submarine pipes
M - Tracks and Routes — routeing measures, radar surveillance, radio reporting, ferries
N - Areas and Limits — restricted areas, military practice areas, international boundaries
O - Hydrographic Terms — general hydrographic terminology
P - Lights — light structures and major floating lights, light characters, colours of lights and marks
Q - Buoys and Beacons — lateral, cardinal, isolated danger, safe water, special marks
R - Fog Signals — fog signal types and symbols
S - Radar, Radio and Satellite Navigation Systems — radar, radio beacons, GNSS
T - Services — pilotage, vessel traffic services, rescue stations
U - Small Craft (Leisure) Facilities — marinas, yacht berths, facilities for leisure craft

Return exactly in this format:

MAIN_FEATURE: <short phrase>
SECTIONS:
- <section letter>
- <section letter>
REASON:
<short explanation>
"""

CODE_PROMPT = """You are a nautical charting expert.

Your task is to retrieve all plausible INT1 / KORT1 candidate code(s) for the Notice to Mariners.

Instructions:
- Use the Notice to Mariners as the main source of truth.
- Focus on the overall charting meaning of the notice, not only exact wording.
- Use the INT1 section material to identify any code that could reasonably apply.
- Be inclusive rather than strict: if a code is plausibly relevant, include it.
- Exclude a code only if it is clearly incompatible with the notice.
- If multiple distinct features or interpretations are possible, include codes for all of them.
- Include both primary candidates and secondary/alternative candidates if they are plausible.
- Do NOT invent codes.
- If no code is even remotely plausible, return NONE.

Notice:
{notice_text}

Relevant INT1 section material:
{section_text}

Return exactly in this format:

MAIN_FEATURE: <short phrase>
SECONDARY_FEATURES: <short phrase or NONE>
CODES:
- <code1>
- <code2>
REASON:
<short explanation>
"""

GENERATE_PROMPT = """You are an expert in nautical chart corrections.

You are given:
1. A Notice to Mariners
2. Relevant INT1 / KORT1 code(s) identified for the notice

Write the corresponding chart correction.

Rules:
- Use the full notice as the main source of information.
- You MUST select the single most relevant INT1 / KORT1 code for the main charted feature and include it in the correction in the format (INT 1 – X 00) where X is the section letter and 00 is the code number. Use the reasoning provided to guide your selection. Every correction should include code for main feature..
- Use the INT1 / KORT1 code(s) only as supporting chart-symbol guidance, not as a source of factual information.
- Do NOT add or infer new factual information.
- Be concise and action-focused.
- Use numbering format: 1), 2), etc.
- Keep coordinate lines separate from descriptive/action text.
- Do not include explanations or extra commentary.
- Generate exactly ONE chart correction. If the notice references multiple charts, select one chart at random and generate the correction for that chart only.
- Match the style and structure of official chart corrections shown in examples below.

Example 1:

    Notice: 
    NM-137-26
    Denmark. The Liim Fiord. Thisted Bredning. Buoyed measuring equipment missing.
    References
    NM-706-25.
    Details
    56° 55.036'N - 008° 44.353'E, yellow spar buoy with cross topmark Fl(5)Y.20s.
    Charts
    109 (INT 1449), 108 (INT 1448).
    (Havsans 11 February 2026. Published 20 February 2026)

    Chart correction:
    Chart 109 
    Denmark. The Liim Fiord. Thisted Bredning. Buoyed measuring equipment missing.
    Details:
    Reference: 00145-25.
    Delete the buoyage as specified at position 1).
    Delete the text ""Rec. St."" nearby.
    1) 56° 55.036'N - 008° 44.353'E, yellow spar buoy with cross topmark, Fl(5)Y.20s.
    Source: NM-137-26

    Example 2:
    Notice:
    NM-039-26
    Denmark. The Waters South of Zealand. Storstrøm. Havnsø Nakke. Marine farm reported.
    References
    NM-1001-25 - (updated repetition).
    Details
    Until further notice, a marine farm has been reported at the mentioned position 1).
    the marine farm is marked as stated in position 2) - 4)
    1) 54° 57.141'N - 011° 54.001'E
    2) 54° 57.210'N - 011° 53.900'E yellow spar buoy with cross topmark
    3) 54° 57.120'N - 011° 53.860'E yellow spar buoy with cross topmark
    4) 54° 57.150'N - 011° 54.170'E yellow spar buoy with cross topmark
    The marine farm is not visible above the surface.
    Charts
    162, 104.
    (DEMA 16 January 2025. Published 21 January 2026)

    Chart correction:
    Chart 162 
    Denmark. Smålandsfarvandet. Storstrøm. Havnsø Nakke. Marine farm reported.
    Details:
    Reference: 00250-25
    Move the marine farm (INT 1 – K 48.2) and the text ""Buoyed"" from position 1) to position 2).
    1) 54° 57.180'N - 011° 54.077'E
    2) 54° 57.141'N - 011° 54.001'E
    Source: NM-039-26

Now generate the chart correction.

Chart: {chart_id}

Notice:
{notice_text}

INT1 / KORT1 result:
{int1_result}

Output:
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def call_llm(prompt: str, max_tokens: int = 500) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

def parse_sections(section_result: str) -> list:
    sections = re.findall(r"-\s*([A-Z])\b", section_result)
    seen, out = set(), []
    for s in sections:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out[:3]  # cap at 3 sections

def build_section_text(selected: list, int1_sections: dict, max_chars: int = 12000) -> str:
    parts = []
    for code in selected:
        if code in int1_sections:
            sec = int1_sections[code]
            parts.append(f"Section {code} — {sec['name_en']}\n{sec['text']}")
    return "\n\n".join(parts)[:max_chars]

# ---------------------------------------------------------------------------
# Password gate
# ---------------------------------------------------------------------------

def check_password():
    if st.session_state.get("authenticated"):
        return True
    st.markdown("""
    <style>
    .login-box {
        max-width: 380px;
        margin: 6rem auto;
        background: #ffffff;
        border: 1px solid #ddd;
        border-top: 4px solid #e8b400;
        border-radius: 4px;
        padding: 2.5rem 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .login-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e2d4a;
        margin-bottom: 0.3rem;
    }
    .login-sub {
        font-size: 0.82rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    </style>
    <div class="login-box">
      <div class="login-title">⚓ ACCENT</div>
      <div class="login-sub">Chart Correction Generator — enter password to continue</div>
    </div>
    """, unsafe_allow_html=True)
    pwd = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Enter password...")
    if st.button("Continue →", use_container_width=True):
        if pwd == st.secrets.get("APP_PASSWORD", "accent2026"):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not check_password():
    st.stop()

# ---------------------------------------------------------------------------
# Load INT1 sections (cached)
# ---------------------------------------------------------------------------

_loading = st.empty()
with _loading:
    with st.spinner("Loading reference manual..."):
        int1_sections = load_int1_sections()
_loading.empty()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["sufficient_context"] == 1].copy()
    df["label"] = df["nm_id"] + " — " + df["title_ntm"].fillna("")
    return df

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ACCENT", page_icon="⚓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #f5f5f0; color: #1a1a1a; }

.app-header {
    background: #1e2d4a;
    color: #ffffff;
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 3px solid #e8b400;
}
.app-header h1 { font-size: 1.5rem; font-weight: 700; margin: 0; letter-spacing: 0.04em; color: #ffffff; }
.app-header p  { margin: 0.3rem 0 0 0; font-size: 0.85rem; color: #9090b0; }

.section-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #666; margin-bottom: 0.5rem;
}

.result-panel { background: #ffffff; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.07); margin-bottom: 1rem; }
.result-panel-header { padding: 0.7rem 1.2rem; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; border-bottom: 1px solid #eee; }
.result-panel-header.generated { background: #1e2d4a; color: #ffffff; border-color: #e8b400; border-bottom-width: 2px; }
.result-panel-header.gold      { background: #f0f7f0; color: #2d5a2d; border-color: #b8d4b8; }
.result-panel-header.int1      { background: #f0f4ff; color: #1a3a7a; border-color: #b8ccee; }
.result-panel-header.section   { background: #fdf8ee; color: #7a5000; border-color: #e8d48a; }
.result-panel-body {
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; line-height: 1.7; color: #1a1a1a;
    white-space: pre-wrap; word-break: break-word;
    min-height: 160px; background: #fafaf8;
}

/* Timeline */
.timeline-wrapper { position: relative; margin: 1.5rem 0; }
.timeline-track {
    display: flex; align-items: flex-start; gap: 0; position: relative;
}
.timeline-track::before {
    content: ''; position: absolute; top: 20px; left: 20px; right: 20px;
    height: 2px; background: #ddd; z-index: 0;
}
.stage-block { flex: 1; display: flex; flex-direction: column; align-items: center; position: relative; z-index: 1; }
.stage-dot {
    width: 40px; height: 40px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem; margin-bottom: 0.5rem; border: 2px solid;
}
.stage-dot.pending { background: #f5f5f0; border-color: #bbb; color: #999; }
.stage-dot.running { background: #fff8e1; border-color: #e8b400; color: #b8860b; }
.stage-dot.done    { background: #1e2d4a; border-color: #1e2d4a; color: #ffffff; }
.stage-dot.error   { background: #fef2f2; border-color: #dc2626; color: #dc2626; }
.stage-name { font-size: 0.72rem; font-weight: 600; text-align: center; color: #444; text-transform: uppercase; letter-spacing: 0.06em; max-width: 100px; }
.stage-desc { font-size: 0.68rem; text-align: center; color: #888; margin-top: 0.2rem; max-width: 110px; }

/* Button */
.stButton > button {
    background: #1e2d4a !important; color: #ffffff !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 0.9rem !important; border: none !important; border-radius: 3px !important;
    padding: 0.65rem 2rem !important; letter-spacing: 0.05em !important;
    border-bottom: 3px solid #e8b400 !important;
}
.stButton > button:hover { background: #2a3f60 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #1e2d4a; border-right: 2px solid #e8b400; }
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] code { background: #2a3f60 !important; color: #e8b400 !important; padding: 1px 5px; border-radius: 2px; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div { background: #ffffff !important; border: 2px solid #1e2d4a !important; border-radius: 3px !important; color: #1a1a1a !important; }
[data-testid="stSelectbox"] > div > div:hover { border-color: #e8b400 !important; }

hr { border-color: #ddd; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
  <h1>⚓ ACCENT — Chart Correction Generator</h1>
  <p>Automatic generation of nautical chart corrections from Notices to Mariners using a three-stage AI pipeline</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")
    st.markdown("---")
    # st.markdown("**Model**")
    # st.markdown(f"`{MODEL}`")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("Stage 0 — Section retrieval  \nStage 1 — INT1 code lookup  \nStage 2 — Correction generation")
    st.markdown("---")
    if api_key:
        st.markdown("🟢 **API key active**")
    else:
        st.markdown("🔴 **API key missing**")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = load_data()

# ---------------------------------------------------------------------------
# Step 1 — Notice input
# ---------------------------------------------------------------------------

st.markdown('<div class="section-label">Step 1 — Select or enter a Notice to Mariners</div>', unsafe_allow_html=True)

input_mode = st.radio(
    "Input mode",
    options=["Select from dataset", "Enter custom notice"],
    horizontal=True,
    label_visibility="collapsed",
)

if input_mode == "Select from dataset":
    selected_label = st.selectbox(
        "Choose a notice",
        options=df["label"].tolist(),
        index=0,
        label_visibility="collapsed",
    )
    row = df[df["label"] == selected_label].iloc[0]
    notice_text = row["raw_text_ntm"]
    chart_id    = row["chart_id_cc"]
    gold_cc     = row["raw_text_cc"]
    notice_key  = selected_label

else:
    notice_text = st.text_area(
        "Paste notice text here",
        height=200,
        placeholder="Paste the full Notice to Mariners text here...",
        label_visibility="collapsed",
    )
    gold_cc    = None
    chart_id   = "—"
    notice_key = f"custom::{notice_text[:80]}"

# Reset pipeline state when input changes
if st.session_state.get("last_notice") != notice_key:
    for k in ["s0_state", "s1_state", "s2_state", "stage0_output", "stage1_output", "stage2_output", "result_gold_cc"]:
        st.session_state.pop(k, None)
    st.session_state["last_notice"] = notice_key

# ---------------------------------------------------------------------------
# Step 2 — Input documents
# ---------------------------------------------------------------------------

st.markdown('<div class="section-label">Step 2 — Review input</div>', unsafe_allow_html=True)

if input_mode == "Select from dataset":
    col_ntm, col_gold = st.columns(2)
    with col_ntm:
        st.markdown(f"""
<div class="result-panel">
  <div class="result-panel-header int1">Notice to Mariners (input)</div>
  <div class="result-panel-body">{notice_text}</div>
</div>
""", unsafe_allow_html=True)
    with col_gold:
        st.markdown(f"""
<div class="result-panel">
  <div class="result-panel-header gold">Gold Standard Chart Correction</div>
  <div class="result-panel-body">{gold_cc}</div>
</div>
""", unsafe_allow_html=True)
else:
    if notice_text.strip():
        st.markdown(f"""
<div class="result-panel">
  <div class="result-panel-header int1">Notice to Mariners (input)</div>
  <div class="result-panel-body">{notice_text}</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Paste a notice above to continue.")

# ---------------------------------------------------------------------------
# Step 3 — Run pipeline
# ---------------------------------------------------------------------------

st.markdown('<div class="section-label" style="margin-top:1rem">Step 3 — Run pipeline</div>', unsafe_allow_html=True)

s0_state = st.session_state.get("s0_state", "pending")
s1_state = st.session_state.get("s1_state", "pending")
s2_state = st.session_state.get("s2_state", "pending")

def dot(state, label, desc):
    icon = {"done": "✓", "error": "!", "running": "▶"}.get(state, "")
    return f"""
<div class="stage-block">
  <div class="stage-dot {state}">{icon}</div>
  <div class="stage-name">{label}</div>
  <div class="stage-desc">{desc}</div>
</div>"""

st.markdown(f"""
<div class="timeline-wrapper">
  <div class="timeline-track">
    {dot("done", "Input", "Notice selected")}
    {dot(s0_state, "Stage 0", "Section retrieval")}
    {dot(s1_state, "Stage 1", "INT1 code lookup")}
    {dot(s2_state, "Stage 2", "Correction generation")}
    {dot("done" if s2_state == "done" else "pending", "Output", "Ready for review")}
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")
generate = st.button("▶  Generate chart correction", use_container_width=True, disabled=not notice_text.strip())

if generate:
    if not api_key:
        st.error("API key not set. Cannot run the pipeline.")
    else:
        st.session_state["s0_state"] = "running"
        st.session_state["section_text_temp"] = None
        st.rerun()

if st.session_state.get("s0_state") == "running":
    with st.spinner("Stage 0 — Identifying relevant INT1 sections..."):
        try:
            stage0_output = call_llm(SECTION_PROMPT.format(notice_text=notice_text), max_tokens=300)
            selected_sections = parse_sections(stage0_output)
            st.session_state["s0_state"] = "done"
            st.session_state["stage0_output"] = stage0_output
            st.session_state["selected_sections"] = selected_sections
            st.session_state["section_text_temp"] = build_section_text(selected_sections, int1_sections)
            st.session_state["s1_state"] = "running"
        except Exception as e:
            st.session_state["s0_state"] = "error"
            st.error(f"Stage 0 failed: {e}")
            st.stop()
    st.rerun()

if st.session_state.get("s1_state") == "running":
    with st.spinner("Stage 1 — Looking up INT1 / KORT1 codes..."):
        try:
            stage1_output = call_llm(CODE_PROMPT.format(
                notice_text=notice_text,
                section_text=st.session_state.get("section_text_temp", ""),
            ), max_tokens=600)
            st.session_state["s1_state"] = "done"
            st.session_state["stage1_output"] = stage1_output
            st.session_state["s2_state"] = "running"
        except Exception as e:
            st.session_state["s1_state"] = "error"
            st.error(f"Stage 1 failed: {e}")
            st.stop()
    st.rerun()

if st.session_state.get("s2_state") == "running":
    with st.spinner("Stage 2 — Generating chart correction..."):
        try:
            stage2_output = call_llm(GENERATE_PROMPT.format(
                chart_id=chart_id,
                notice_text=notice_text,
                int1_result=st.session_state["stage1_output"],
            ), max_tokens=500)
            st.session_state["s2_state"] = "done"
            st.session_state["stage2_output"] = stage2_output
            st.session_state["result_gold_cc"] = gold_cc
        except Exception as e:
            st.session_state["s2_state"] = "error"
            st.error(f"Stage 2 failed: {e}")
            st.stop()
    st.rerun()

# ---------------------------------------------------------------------------
# Step 4 — Results
# ---------------------------------------------------------------------------

if "stage2_output" in st.session_state:
    st.markdown('<div class="section-label" style="margin-top:1.5rem">Step 4 — Review results</div>', unsafe_allow_html=True)

    gold = st.session_state.get("result_gold_cc")

    if gold:
        col_gen, col_ref = st.columns(2)
        with col_gen:
            st.markdown(f"""
<div class="result-panel">
  <div class="result-panel-header generated">⚙ Model output — Generated correction</div>
  <div class="result-panel-body">{st.session_state['stage2_output']}</div>
</div>
""", unsafe_allow_html=True)
        with col_ref:
            st.markdown(f"""
<div class="result-panel">
  <div class="result-panel-header gold">✓ Gold standard correction</div>
  <div class="result-panel-body">{gold}</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="result-panel">
  <div class="result-panel-header generated">⚙ Model output — Generated correction</div>
  <div class="result-panel-body">{st.session_state['stage2_output']}</div>
</div>
""", unsafe_allow_html=True)

    with st.expander("View intermediate pipeline outputs"):
        selected = st.session_state.get("selected_sections", [])
        sections_label = ", ".join(
            f"{c} ({int1_sections[c]['name_en']})" for c in selected if c in int1_sections
        )
        st.markdown(f"""
<div class="result-panel">
  <div class="result-panel-header section">Stage 0 — Sections retrieved: {sections_label}</div>
  <div class="result-panel-body">{st.session_state.get('stage0_output', '')}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="result-panel">
  <div class="result-panel-header int1">Stage 1 — INT1 codes identified</div>
  <div class="result-panel-body">{st.session_state.get('stage1_output', '')}</div>
</div>
""", unsafe_allow_html=True)