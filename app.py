import base64
import os
import re
import warnings
from pathlib import Path
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Waltonen Quote Estimator",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""<style>
/* ============ GLOBAL BACKGROUND ============ */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
.stApp,
.main,
.block-container {
    background:#f4f6fa !important;
}

[data-testid="stHeader"] { box-shadow:none; border-bottom:1px solid #e4e7eb; }
[data-testid="stToolbar"] button { color:#475569 !important; }

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background:#ffffff !important;
    border-right:1px solid #e4e7eb;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown strong,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color:#1e293b !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#64748b !important; }

/* Sidebar buttons (example queries + new conversation) */
[data-testid="stSidebar"] .stButton > button {
    background:#f8fafc !important;
    color:#334155 !important;
    border:1px solid #e2e8f0 !important;
    border-radius:8px !important;
    text-align:left !important;
    padding:10px 14px !important;
    font-weight:500 !important;
    font-size:.85rem !important;
    line-height:1.4 !important;
    transition:all .15s ease !important;
    box-shadow:0 1px 2px rgba(15,23,42,.04) !important;
    white-space:normal !important;
    height:auto !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background:#eef4ff !important;
    border-color:#3b82f6 !important;
    color:#1e40af !important;
    transform:translateY(-1px);
    box-shadow:0 4px 12px rgba(59,130,246,.15) !important;
}

/* Sidebar dividers */
[data-testid="stSidebar"] hr { border-color:#e4e7eb !important; }

/* ============ CHAT INPUT (bottom bar) ============ */
[data-testid="stChatInput"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div {
    background:#f4f6fa !important;
    border-top:1px solid #e4e7eb;
}
[data-testid="stChatInput"] textarea {
    background:#ffffff !important;
    color:#1e293b !important;
    border:1px solid #d1d9e6 !important;
    border-radius:10px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color:#3b82f6 !important;
    box-shadow:0 0 0 3px rgba(59,130,246,.12) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color:#94a3b8 !important; }
[data-testid="stChatInput"] button { color:#475569 !important; }

/* ============ CHAT MESSAGES ============ */
[data-testid="stChatMessage"] {
    background:#ffffff !important;
    border:1px solid #e4e7eb;
    border-radius:12px;
    padding:16px 18px !important;
    margin-bottom:12px;
    box-shadow:0 1px 3px rgba(15,23,42,.04);
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] div {
    color:#1e293b !important;
}
[data-testid="stChatMessage"] code {
    background:#f1f5f9 !important;
    color:#0f172a !important;
    padding:2px 6px;
    border-radius:4px;
}

/* Chat avatar */
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
    background:#eef2f7 !important;
    border:1px solid #e4e7eb;
}

/* ============ ESTIMATE CARDS ============ */
.metric-card {
    background:#ffffff;
    border:1px solid #e4e7eb;
    border-radius:12px;
    padding:18px 18px;
    text-align:center;
    min-height:108px;
    display:flex; flex-direction:column; justify-content:center;
    transition:all .2s ease;
    word-wrap:break-word;
    overflow-wrap:break-word;
    box-shadow:0 1px 3px rgba(15,23,42,.04);
}
.metric-card:hover {
    border-color:#3b82f6;
    box-shadow:0 8px 24px rgba(59,130,246,.12);
    transform:translateY(-2px);
}
.metric-label {
    color:#64748b;
    font-size:.7rem;
    text-transform:uppercase;
    letter-spacing:1.2px;
    margin-bottom:8px;
    font-weight:600;
}
.metric-value {
    color:#0f172a;
    font-size:1.05rem;
    font-weight:700;
    line-height:1.4;
    word-wrap:break-word;
    overflow-wrap:break-word;
    white-space:normal;
}
.metric-value.lg { font-size:1.35rem; color:#1e293b; }

/* ============ TOTAL BANNER ============ */
.total-banner {
    background: linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%);
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    margin-top: 18px;
    margin-bottom: 8px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
}
.total-banner .lbl {
    color:#bfdbfe;
    font-size:.72rem;
    letter-spacing:2.2px;
    text-transform:uppercase;
    font-weight:600;
}
.total-banner .amt {
    color:#ffffff;
    font-size:2.1rem;
    font-weight:800;
    margin-top:4px;
    letter-spacing:-.5px;
}

/* ============ REFERENCE SECTION ============ */
.ref-section { margin-top:22px; margin-bottom:14px; }
.ref-label {
    color:#64748b;
    font-size:.7rem;
    letter-spacing:1.2px;
    text-transform:uppercase;
    margin-bottom:10px;
    font-weight:600;
}
.ref-badge {
    display:inline-block;
    background:#eef4ff;
    color:#1e40af;
    border:1px solid #bfdbfe;
    padding:4px 10px;
    border-radius:6px;
    font-size:.78rem;
    font-weight:600;
    margin-right:6px;
    font-family:ui-monospace,SFMono-Regular,monospace;
}

.est-header {
    color:#1e40af;
    font-size:.82rem;
    font-weight:700;
    letter-spacing:1px;
    text-transform:uppercase;
    margin-bottom:12px;
}

/* ============ PROJECT FILE ROW ============ */
.proj-file-row {
    display:flex; align-items:center; gap:12px;
    background:#ffffff;
    border:1px solid #e4e7eb;
    border-radius:10px;
    padding:12px 16px;
    margin-bottom:8px;
    box-shadow:0 1px 2px rgba(15,23,42,.03);
}
.proj-file-icon { font-size:1.5rem; }
.proj-file-name { color:#0f172a; font-size:.9rem; font-weight:600; flex:1; }
.proj-file-code {
    color:#0d9488;
    font-size:.78rem;
    font-family:ui-monospace,SFMono-Regular,monospace;
    font-weight:600;
}

/* ============ APP HEADER ============ */
.app-header {
    background:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);
    border:1px solid #e4e7eb;
    border-left:4px solid #f97316;
    border-radius:14px;
    padding:26px 30px;
    margin-bottom:22px;
    box-shadow:0 2px 8px rgba(15,23,42,.04);
}
.app-header h1 {
    color:#0f172a;
    margin:0;
    font-size:1.75rem;
    font-weight:800;
    letter-spacing:-.4px;
}
.app-header p {
    color:#64748b;
    margin:8px 0 0;
    font-size:.92rem;
    line-height:1.5;
}

.built-by {
    color:#94a3b8;
    font-size:.7rem;
    text-align:center;
    margin-top:6px;
    letter-spacing:.3px;
}

/* ============ EXPANDER ============ */
[data-testid="stExpander"] {
    background:#ffffff !important;
    border:1px solid #e4e7eb !important;
    border-radius:10px !important;
    box-shadow:0 1px 3px rgba(15,23,42,.04);
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color:#1e293b !important;
    font-weight:600 !important;
}
[data-testid="stExpander"] summary:hover { background:#f8fafc !important; }

/* ============ DIVIDER ============ */
hr { border-color:#e4e7eb !important; }

/* ============ DOWNLOAD / ICON BUTTONS in main area ============ */
.main .stButton > button,
.main .stDownloadButton > button {
    background:#ffffff !important;
    color:#475569 !important;
    border:1px solid #e2e8f0 !important;
    border-radius:8px !important;
    font-weight:500 !important;
    transition:all .15s ease !important;
}
.main .stButton > button:hover,
.main .stDownloadButton > button:hover {
    background:#eef4ff !important;
    border-color:#3b82f6 !important;
    color:#1e40af !important;
}

/* ============ ERROR / WARNING / INFO BOXES ============ */
[data-testid="stAlert"] {
    border-radius:10px;
    border:1px solid #e4e7eb;
}

/* ============ STATUS / TEXT ============ */
.main p, .main li, .main span:not(.metric-value):not(.metric-label):not(.lbl):not(.amt):not(.ref-badge) {
    color:#1e293b;
}

/* Streamlit st.write rendered text in main area */
.main [data-testid="stMarkdownContainer"] p { color:#1e293b; }

/* Spinner text */
.stSpinner > div { color:#475569 !important; }
</style>""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in {
    "messages": [],
    "prev_response_id": None,
    "_client": None,
    "_agent_name": None,
}.items():
    st.session_state.setdefault(_k, _v)

# ── Azure Blob Storage — project PDF files ─────────────────────────────────────
_APP_DIR = Path(__file__).parent


def _build_container_client():
    """Return a ContainerClient if storage env vars are set, else None."""
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "project-data")
        if not conn_str or conn_str.startswith("DefaultEndpointsProtocol=https;AccountName=..."):
            return None, None
        svc = BlobServiceClient.from_connection_string(conn_str)
        return svc.get_container_client(container), container
    except Exception:
        return None, None


def _index_blobs(container_client) -> dict[str, str]:
    """List blobs and map each RJ### code in the blob name → blob name."""
    index: dict[str, str] = {}
    if container_client is None:
        return index
    try:
        for blob in container_client.list_blobs():
            filename = Path(blob.name).name
            for code in re.findall(r"RJ\d{3,4}", filename, re.IGNORECASE):
                index[code.upper()] = blob.name
    except Exception:
        pass
    return index


def _index_local_files() -> dict[str, Path]:
    """Fallback: scan repo directory for PDFs with RJ### in the filename."""
    index: dict[str, Path] = {}
    for pdf in _APP_DIR.glob("**/*.pdf"):
        for code in re.findall(r"RJ\d{3,4}", pdf.name, re.IGNORECASE):
            index[code.upper()] = pdf
    return index


_CONTAINER_CLIENT, _CONTAINER_NAME = _build_container_client()
_BLOB_INDEX: dict[str, str] = _index_blobs(_CONTAINER_CLIENT)
_LOCAL_INDEX: dict[str, Path] = _index_local_files() 


def _find_related_files(refs: list[str]) -> list[dict]:
    """
    Return file descriptors for each ref.
    Azure blob is used when available; local file is the fallback.
    """
    results = []
    for ref in refs:
        code = ref.upper()
        blob_name = _BLOB_INDEX.get(code)
        local_path = _LOCAL_INDEX.get(code)
        if blob_name:
            results.append({"code": code, "source": "blob",
                            "blob_name": blob_name, "name": Path(blob_name).name})
        elif local_path:
            results.append({"code": code, "source": "local",
                            "path": local_path, "name": local_path.name})
    return results


@st.cache_data(show_spinner=False, ttl=300)
def _download_blob(blob_name: str) -> bytes | None:
    """Download a blob and return its bytes (cached 5 min)."""
    if _CONTAINER_CLIENT is None:
        return None
    try:
        return _CONTAINER_CLIENT.get_blob_client(blob_name).download_blob().readall()
    except Exception:
        return None


def _get_pdf_bytes(item: dict) -> bytes | None:
    """Resolve bytes from blob or local path depending on source."""
    if item["source"] == "blob":
        return _download_blob(item["blob_name"])
    try:
        return item["path"].read_bytes()
    except Exception:
        return None

# ── Client factory ─────────────────────────────────────────────────────────────
def _build_client(endpoint: str):
    credential = DefaultAzureCredential()

    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=credential,
    )
    return project_client.get_openai_client()


# ── Auto-connect from env ──────────────────────────────────────────────────────
def _ensure_ready():
    client = st.session_state["_client"]
    agent  = st.session_state["_agent_name"]

    if client is None:
        ep   = os.getenv("AZURE_AI_ENDPOINT", "")
        key  = os.getenv("AZURE_API_KEY", "")
        name = os.getenv("AZURE_AGENT_NAME", "dev-ktech-demo")
        
        if ep and key and name:
            try:
                client = _build_client(ep)
                st.session_state.update({"_client": client, "_agent_name": name})
                agent = name
            except Exception as e:
                # This catches real connection issues (like a bad API key)
                st.error(f"❌ Connection failed: {e}")
                return None, None
        else:
            return None, None

    return client, agent

# ── Formatting guide ───────────────────────────────────────────────────────────
_GUIDE = (
    "\n\n---\n"
    "RESPONSE FORMAT — always use these exact section headers for estimates:\n\n"
    "- **Timeline**: [X weeks with brief phase breakdown]\n"
    "- **Material Cost**: [amount] USD  *(raw materials & finishes)*\n"
    "- **Labor Cost**: [amount] USD  *(fabrication & installation)*\n"
    "- **Total Estimate**: [amount] USD\n"
    "- **Reference Projects**: [list project codes from the knowledge base, e.g. RJ979, RJ908]\n\n"
    "If original data shows one combined cost, split it using typical industry ratios.\n"
    "---"
)


def _run_query(client, agent_name, user_text, prev_id):
    kwargs = dict(
        input=[{"role": "user", "content": user_text + _GUIDE}],
        extra_body={"agent_reference": {"name": agent_name, "version": "15", "type": "agent_reference"}},
    )
    if prev_id:
        kwargs["previous_response_id"] = prev_id

    response = client.responses.create(**kwargs)

    # Loop: auto-approve any MCP approval requests until the run truly completes
    max_iterations = 5
    while max_iterations > 0:
        approval_requests = [
            item for item in getattr(response, "output", [])
            if getattr(item, "type", "") == "mcp_approval_request"
        ]
        if not approval_requests:
            break

        approval_inputs = [
            {
                "type": "mcp_approval_response",
                "approval_request_id": req.id,
                "approve": True,
            }
            for req in approval_requests
        ]

        response = client.responses.create(
            input=approval_inputs,
            previous_response_id=response.id,
            extra_body={"agent_reference": {"name": agent_name, "version": "9", "type": "agent_reference"}},
        )
        max_iterations -= 1

    # Now extract output_text normally
    r_id = getattr(response, "id", "unknown")
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "total_tokens", "N/A") if usage else "N/A"
    st.write(f"⏳ Response ID: {r_id} · Tokens used: {tokens}")

    text = getattr(response, "output_text", "") or ""
    return text.strip() or "No response received.", r_id

def _parse(text: str) -> dict:
    # st.write(f"🔍 {text}")
    refs = sorted(set(re.findall(r"\bRJ\d{3,4}\b", text, re.IGNORECASE)))

    def timeline():
        # First grab everything after the Timeline label up to the next section
        m = re.search(
            r"\*{0,2}timeline\*{0,2}\**\s*[:\-]\s*(.+?)(?=\n(?:\s*[-*•]?\s*\*{0,2}(?:material|labor|total|reference))|$)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None
        raw = m.group(1).strip("* \t\n")

        # Extract just the duration headline (e.g. "4 weeks", "6-8 weeks", "10 days")
        dur = re.search(
            r"(\d+(?:\s*[-–]\s*\d+)?\s*(?:week|day|month)s?)",
            raw, re.IGNORECASE,
        )
        if dur:
            return dur.group(1).strip()

        # Fallback: first line only, truncated if very long
        first_line = raw.split("\n")[0].split(" - ")[0].strip()
        return (first_line[:50] + "…") if len(first_line) > 50 else first_line

    def cost(*labels):
        for lbl in labels:
            for pat in [
                rf"\*{{0,2}}{re.escape(lbl)}\*{{0,2}}\**\s*[:\-]\s*(?:USD\s*)?\$?([\d,]+(?:\.\d{{1,2}})?)",
                rf"\*{{0,2}}{re.escape(lbl)}\*{{0,2}}\**\s*[:\-]\s*(?:USD\s*)?([\d,]+(?:\.\d{{1,2}})?)",
                rf"\*{{0,2}}{re.escape(lbl)}\*{{0,2}}\**\s*[:\-]\s*([\d,]+(?:\.\d{{1,2}})?)\s*USD",
            ]:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    return float(m.group(1).replace(",", ""))
        return None 

    mat = cost("material cost", "material costs", "materials cost", "materials")
    lab = cost("labor cost", "labour cost", "labor", "labour", "installation cost", "fabrication cost")
    tot = cost("total estimate", "total cost", "grand total", "total")
    if tot is None and mat and lab:
        tot = mat + lab

    tl = timeline()
    return {
        "has_estimate": bool(refs or tl or mat or lab or tot),
        "timeline": tl, "material_cost": mat, "labor_cost": lab,
        "total_cost": tot, "references": refs,
    }


# ── Rendering ──────────────────────────────────────────────────────────────────
def _usd(v):
    return f"$ {v:,.0f}" if v is not None else "—"

def _render_project_files(refs, key_pfx):
    files = _find_related_files(refs)
    if not files:
        return
    with st.expander(f"📁  Related Project Files  ({len(files)})", expanded=True):
        for item in files:
            t_key = f"sh_{key_pfx}_{item['code']}"
            c_info, c_dl, c_prev = st.columns([5, 1, 1])
            with c_info:
                st.markdown(
                    f'<div class="proj-file-row"><span class="proj-file-icon">📄</span>'
                    f'<div><div class="proj-file-name">{item["name"]}</div>'
                    f'<div class="proj-file-code">{item["code"]}</div></div></div>',
                    unsafe_allow_html=True,
                )
            with c_dl:
                b = _get_pdf_bytes(item)
                if b:
                    st.download_button(
                        "⬇️", b, item["name"], "application/pdf",
                        key=f"dl_{key_pfx}_{item['code']}",
                    )
            with c_prev:
                if st.button("👁️", key=f"bt_{key_pfx}_{item['code']}"):
                    st.session_state[t_key] = not st.session_state.get(t_key, False)
                    st.rerun()
            if st.session_state.get(t_key, False):
                b = _get_pdf_bytes(item)
                if b:
                    b64 = base64.b64encode(b).decode()
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}" '
                        f'width="100%" height="680" '
                        f'style="border:1px solid #e4e7eb;border-radius:10px;'
                        f'margin:8px 0 16px"></iframe>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("PDF could not be loaded.")
                    
def _render_card(p: dict, key_pfx: str = "0"):
    st.markdown('<div class="est-header">📋 Estimate Summary</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, lbl, val, large in (
        (c1, "⏱", "Timeline",      p["timeline"] or "—",       False),
        (c2, "🔩", "Material Cost", _usd(p["material_cost"]),   True),
        (c3, "👷", "Labor Cost",    _usd(p["labor_cost"]),      True),
    ):
        with col:
            cls = "metric-value lg" if large else "metric-value"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{icon} {lbl}</div>'
                f'<div class="{cls}">{val}</div></div>',
                unsafe_allow_html=True,
            )

    if p["total_cost"]:
        st.markdown(
            f'<div class="total-banner"><div class="lbl">Total Estimate</div>'
            f'<div class="amt">{_usd(p["total_cost"])}</div></div>',
            unsafe_allow_html=True,
        )

    if p["references"]:
        badges = "".join(f'<span class="ref-badge">📁 {r}</span>' for r in p["references"])
        st.markdown(
            f'<div class="ref-section">'
            f'<div class="ref-label">📚 Reference Evidence from Knowledge Base</div>'
            f'<div>{badges}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")  # breathing room before the expander
        _render_project_files(p["references"], key_pfx=key_pfx)

    st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📐 Waltonen\n**Quote Estimator**")
    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.update({"messages": [], "prev_response_id": None})
        st.rerun()

    st.divider()
    st.markdown("**💡 Try these queries:**")
    _EXAMPLES = [
    "We need a quote for a weld fixture for our body shop line. Running 3 shifts, roughly 200k units/year. We have 3 part variants and need changeover under 90 seconds. FANUC robots already on the line. Need simulation done in Process Simulate and full PPAP support. Tolerance at datums is ±0.5mm.",
    "Looking for a supplier to machine 300 aluminum brackets in 7075-T6. Need 5-axis capability, tolerances around ±0.003 inch, surface finish 125 µin. CMM reports required per part. Parts need anodizing and we're AS9100. Can you split delivery into 2 lots?",
    "We're working on a ground vehicle program and need a simulation model of our hydraulic steering subsystem. Has to run real-time and integrate into our Simulink environment. On-prem only due to ITAR. Need systems engineering docs and validation against test data.",
    "We have an old transmission housing — no drawings, no CAD. Need someone to scan it, build a full parametric model with GD&T, and then look at redesigning it to cut weight by around 8–10%. We'd also want a machined prototype at the end for testing.",
    "Need a turnkey assembly cell. Cycle time target is around 50 seconds. Two robots, vision for part pick, Allen-Bradley controls. End of line needs torque check and a camera inspection. We'll need FAT at your facility and SAT at ours, plus operator training."]
    for _ex in _EXAMPLES:
        if st.button(_ex, key=f"ex_{hash(_ex)}", use_container_width=True):
            st.session_state["_pending"] = _ex
            st.rerun()

    st.divider()
    st.caption("Powered by Azure AI Foundry · GPT-4o · Responses API")
    st.markdown('<div class="built-by">Built by KitelyTech</div>', unsafe_allow_html=True)


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown(
    """<div class="app-header">
    <h1>📐 Waltonen Quote Estimator</h1>
    <p>AI-powered instant manufacturing cost estimates from historical project data ·
    Ask for any manufacturing, signage, or furniture project quote</p>
</div>""",
    unsafe_allow_html=True,
)

# ── Ready check ────────────────────────────────────────────────────────────────
client, agent_name = _ensure_ready()
_ready = client is not None and bool(agent_name)

if not _ready:
    st.error(
        "⚠️ Azure connection not configured. "
        "Set `AZURE_AI_ENDPOINT`, `AZURE_API_KEY`, and `AZURE_AGENT_NAME` in your `.env` file."
    )

# ── Chat history ───────────────────────────────────────────────────────────────
for _i, _msg in enumerate(st.session_state["messages"]):
    _av = "👤" if _msg["role"] == "user" else "🤖"
    with st.chat_message(_msg["role"], avatar=_av):
        if _msg["role"] == "assistant" and _msg.get("parsed", {}).get("has_estimate"):
            _render_card(_msg["parsed"], key_pfx=str(_i))
        st.markdown(_msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
_chat_val = st.chat_input("Describe the project or ask a follow-up question…", disabled=not _ready)
user_input: str | None = st.session_state.pop("_pending", None) or _chat_val

if user_input and _ready:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching knowledge base and computing estimate…"):
            try:
                resp_text, new_resp_id = _run_query(
                    client, agent_name, user_input,
                    st.session_state["prev_response_id"],
                )
                st.session_state["prev_response_id"] = new_resp_id
                parsed = _parse(resp_text)
                msg_key = str(len(st.session_state["messages"]))
                if parsed["has_estimate"]:
                    _render_card(parsed, key_pfx=msg_key)
                st.markdown(resp_text)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": resp_text, "parsed": parsed}
                )
            except Exception as exc:
                err = f"❌ Error: {exc}"
                st.error(err)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": err, "parsed": {"has_estimate": False}}
                )