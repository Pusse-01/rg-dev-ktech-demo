"""
Waltonen Quote Estimator Demo
Built by KitelyTech — Azure AI Foundry Responses API + Streamlit
"""
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
[data-testid="stAppViewContainer"] { background:#0d1117; }
[data-testid="stSidebar"] { background:#161b22 !important; border-right:1px solid #30363d; }
[data-testid="stSidebar"] .stMarkdown p { color:#8b949e; }
[data-testid="stChatMessage"] { background:transparent; }

.metric-card {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
    padding:14px 16px; text-align:center; height:100px;
    display:flex; flex-direction:column; justify-content:center;
    transition:border-color .2s;
}
.metric-card:hover { border-color:#388bfd; }
.metric-label { color:#8b949e; font-size:.68rem; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:6px; }
.metric-value { color:#e6edf3; font-size:1.05rem; font-weight:700; }
.metric-value.lg { font-size:1.3rem; }

.total-banner {
    background:linear-gradient(90deg,#1f6feb,#388bfd); border-radius:10px;
    padding:14px 20px; text-align:center; margin-top:10px;
}
.total-banner .lbl { color:#cae8ff; font-size:.7rem; letter-spacing:2px; text-transform:uppercase; }
.total-banner .amt { color:#fff; font-size:2rem; font-weight:800; margin-top:2px; }

.ref-section { margin-top:12px; }
.ref-label { color:#8b949e; font-size:.68rem; letter-spacing:1.2px; text-transform:uppercase; }
.ref-badge {
    display:inline-block; background:#0f2a1a; color:#56d364;
    border:1px solid #238636; border-radius:6px;
    padding:3px 10px; font-size:.78rem; font-weight:600;
    margin:4px 3px 0; font-family:monospace;
}

.est-header {
    color:#58a6ff; font-size:.8rem; font-weight:600;
    letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;
}

.proj-file-row {
    display:flex; align-items:center; gap:10px;
    background:#161b22; border:1px solid #21262d;
    border-radius:8px; padding:10px 14px; margin-bottom:8px;
}
.proj-file-icon { font-size:1.4rem; }
.proj-file-name { color:#e6edf3; font-size:.88rem; font-weight:600; flex:1; }
.proj-file-code { color:#56d364; font-size:.78rem; font-family:monospace; }

.app-header {
    background:linear-gradient(135deg,#0d1117,#161b22,#1c2128);
    border:1px solid #30363d; border-left:4px solid #f78166;
    border-radius:12px; padding:24px 28px; margin-bottom:20px;
}
.app-header h1 { color:#e6edf3; margin:0; font-size:1.7rem; font-weight:800; }
.app-header p  { color:#8b949e; margin:6px 0 0; font-size:.9rem; }

.built-by { color:#484f58; font-size:.7rem; text-align:center; margin-top:4px; }
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
        container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "project-files")
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
_BLOB_INDEX: dict[str, str] = _index_blobs(_CONTAINER_CLIENT)   # code → blob name
_LOCAL_INDEX: dict[str, Path] = _index_local_files()            # code → local path (fallback)


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
def _build_client(endpoint: str, api_key: str):
    """
    Uses the API Key from your .env. Removed .project_name to fix the AttributeError.
    """
    # Use the API key explicitly
    credential = DefaultAzureCredential()

    project_client = AIProjectClient(
        endpoint=endpoint,
        credential=credential,
    )
    
    # We'll show a generic success message since .project_name isn't available
    # st.success("✅ Successfully connected to Azure AI Foundry")
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
                client = _build_client(ep, key)
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


def _run_query(client, agent_name: str, user_text: str, prev_id: str | None) -> tuple[str, str]:
    kwargs: dict = dict(
        input=[{"role": "user", "content": user_text + _GUIDE}],
        extra_body={"agent_reference": {"name": agent_name, "version": "9", "type": "agent_reference"}},
    )
    if prev_id:
        kwargs["previous_response_id"] = prev_id

    response = client.responses.create(**kwargs)
    # st.write(f"⏳ Response ID: {response} · Tokens used: {getattr(response, 'tokens_used', 'N/A')}")

    # Safely extract usage and response info
    r_id = getattr(response, "id", getattr(response, "response_id", "unknown"))
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "total_tokens", "N/A") if usage else "N/A"
    
    st.write(f"⏳ Response ID: {r_id} · Tokens used: {tokens}")

    text: str = getattr(response, "output_text", "") or ""
    st.write(response)
    
    # # UPDATED PARSING LOGIC: Extract data from memories or tools if output_text is empty
    if not text:
        for item in getattr(response, "output", []):
            # 1. Standard text content
            content = getattr(item, "content", None)
            if content:
                if isinstance(content, list):
                    for c in content:
                        text += getattr(c, "text", "") + "\n\n"
                elif isinstance(content, str):
                    text += content + "\n\n"
            
            # 2. Memory Search calls (Handles the 'ResponseOutputMessage' log structure)
            if getattr(item, "type", "") == "memory_search_call":
                memories = getattr(item, "memories", [])
                for mem in memories:
                    if isinstance(mem, dict) and "content" in mem:
                        text += mem["content"] + "\n\n"
                        
            # 3. Tool arguments (Fallback for 'McpApprovalRequest')
            args = getattr(item, "arguments", "")
            if isinstance(args, str) and args:
                text += args + "\n\n"

    return text.strip() or "No response received.", r_id


# ── Response parsing ───────────────────────────────────────────────────────────
# def _parse(text: str) -> dict:
#     refs = sorted(set(re.findall(r"\bRJ\d{3,4}\b", text, re.IGNORECASE)))

#     def timeline():
#         m = re.search(r"\*{0,2}timeline\*{0,2}\**\s*[:\-]\s*([^\n]{5,120})", text, re.IGNORECASE)
#         return m.group(1).strip("* \t") if m else None

#     def cost(*labels):
#         for lbl in labels:
#             for pat in [
#                 rf"\*{{0,2}}{re.escape(lbl)}\*{{0,2}}\**\s*[:\-]\s*(?:USD\s*)?\$?([\d,]+(?:\.\d{{1,2}})?)",
#                 rf"\*{{0,2}}{re.escape(lbl)}\*{{0,2}}\**\s*[:\-]\s*([\d,]+(?:\.\d{{1,2}})?)\s*USD",
#             ]:
#                 m = re.search(pat, text, re.IGNORECASE)
#                 if m:
#                     return float(m.group(1).replace(",", ""))
#         return None

#     mat = cost("material cost", "material costs", "materials cost", "materials")
#     lab = cost("labor cost", "labour cost", "labor", "labour", "installation cost", "fabrication cost")
#     tot = cost("total estimate", "total cost", "grand total", "total")
#     if tot is None and mat and lab:
#         tot = mat + lab

#     tl = timeline()
#     return {
#         "has_estimate": bool(refs or tl or mat or lab or tot),
#         "timeline": tl, "material_cost": mat, "labor_cost": lab,
#         "total_cost": tot, "references": refs,
#     }
def _parse(text: str) -> dict:
    # st.write(f"🔍 {text}")
    refs = sorted(set(re.findall(r"\bRJ\d{3,4}\b", text, re.IGNORECASE)))

    def timeline():
        m = re.search(r"\*{0,2}timeline\*{0,2}\**\s*[:\-]\s*(.+?)(?=\n(?:material|labor|total|reference)|$)", text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip("* \t\n") if m else None

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
    return f"USD {v:,.0f}" if v is not None else "—"


# def _render_project_files(refs: list[str], key_pfx: str):
#     """Expandable section showing downloadable + previewable PDFs for each ref."""
#     files = _find_related_files(refs)
#     if not files:
#         return

#     with st.expander(f"📁  Related Project Files  ({len(files)} available)", expanded=True):
#         for item in files:
#             # ── file row ──────────────────────────────────────────────────────
#             c_info, c_dl, c_prev = st.columns([5, 1, 1])
#             with c_info:
#                 src_label = "Azure" if item["source"] == "blob" else "Local"
#                 st.markdown(
#                     f'<div class="proj-file-row">'
#                     f'<span class="proj-file-icon">📄</span>'
#                     f'<div>'
#                     f'<div class="proj-file-name">{item["name"]}</div>'
#                     f'<div class="proj-file-code">{item["code"]} · {src_label}</div>'
#                     f'</div></div>',
#                     unsafe_allow_html=True,
#                 )
#             with c_dl:
#                 pdf_bytes = _get_pdf_bytes(item)
#                 if pdf_bytes:
#                     st.download_button(
#                         "⬇️ Download",
#                         data=pdf_bytes,
#                         file_name=item["name"],
#                         mime="application/pdf",
#                         key=f"dl_{key_pfx}_{item['code']}",
#                         use_container_width=True,
#                     )
#             with c_prev:
#                 toggle_key = f"show_{key_pfx}_{item['code']}"
#                 if toggle_key not in st.session_state:
#                     st.session_state[toggle_key] = False
#                 label = "🙈 Hide" if st.session_state[toggle_key] else "👁️ Preview"
#                 if st.button(label, key=f"btn_{key_pfx}_{item['code']}", use_container_width=True):
#                     st.session_state[toggle_key] = not st.session_state[toggle_key]
#                     st.rerun()

#             # ── inline PDF preview ────────────────────────────────────────────
#             if st.session_state.get(toggle_key, False):
#                 preview_bytes = _get_pdf_bytes(item)
#                 if preview_bytes:
#                     b64 = base64.b64encode(preview_bytes).decode()
#                     st.markdown(
#                         f'<iframe src="data:application/pdf;base64,{b64}" '
#                         f'width="100%" height="680" '
#                         f'style="border:1px solid #30363d;border-radius:8px;'
#                         f'margin:8px 0 16px"></iframe>',
#                         unsafe_allow_html=True,
#                     )
#                 else:
#                     st.warning("PDF could not be loaded.")

def _render_project_files(refs, key_pfx):
    files = _find_related_files(refs)
    if not files: return
    with st.expander(f"📁  Related Project Files  ({len(files)})", expanded=True):
        for item in files:
            c_info, c_dl, c_prev = st.columns([5, 1, 1])
            with c_info:
                st.markdown(f'<div class="proj-file-row"><span class="proj-file-icon">📄</span>'
                            f'<div><div class="proj-file-name">{item["name"]}</div>'
                            f'<div class="proj-file-code">{item["code"]}</div></div></div>', unsafe_allow_html=True)
            with c_dl:
                b = _get_pdf_bytes(item)
                if b: st.download_button("⬇️", b, item["name"], "application/pdf", key=f"dl_{key_pfx}_{item['code']}")
            with c_prev:
                t_key = f"sh_{key_pfx}_{item['code']}"
                if st.button("👁️", key=f"bt_{key_pfx}_{item['code']}"):
                    st.session_state[t_key] = not st.session_state.get(t_key, False)
                    st.rerun()
            if st.session_state.get(t_key, False):
                b = _get_pdf_bytes(item)
                if b:
                    b64 = base64.b64encode(b).decode()
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


# # ── Auto-connect from env ──────────────────────────────────────────────────────
# def _ensure_ready():
#     client = st.session_state["_client"]
#     agent  = st.session_state["_agent_name"]

#     if client is None:
#         ep   = os.getenv("AZURE_AI_ENDPOINT", "")
#         key  = os.getenv("AZURE_API_KEY", "")
#         name = os.getenv("AZURE_AGENT_NAME", "dev-ktech-demo")
#         st.info(ep and key and name and "Attempting to connect to Azure AI Foundry project…")
#         if ep and key:
#             try:
#                 client = _build_client(ep, key)
#                 st.session_state.update({"_client": client, "_agent_name": name})
#                 agent = name
#             except Exception:
#                 return None, None

#     return client, agent


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
        # Targets RJ979 (Lake Garden Wardrobe)
        "Quote a built-in MDF wardrobe with PU spray paint and internal LED rods.",
        
        # Targets RJ908 (Charcoal Restaurant)
        "Cost for a black powder-coated aluminum restaurant sign and 4 brushed brass lift button signs.",
        
        # Targets RJ985 (Luxury Teak Dining Set)
        "Estimate for an 8-seater solid teak dining table with upholstered chairs and a ribbed credenza.",
        
        # Targets RJ972 (The Archives Buckets)
        "Manufacturing timeline and cost for 10 oxidized brass buckets with GI rod stands.",
        
        # Targets RJ969 (ITC Mirrors)
        "Quote for 5 LED backlit vanity mirrors and 2 frameless full-length dressing mirrors.",
        
        # Targets RJ973 (Bedheads Rework - *New addition based on your PDFs*)
        "Cost to rework and reupholster a bedhead feature wall with velvet fabric and button tufting.",
        
        # Targets RJ979 + RJ985 (The "What-if" scenario)
        "Estimate a custom built-in wardrobe, but instead of MDF, price it using solid local teak wood based on past projects."
    ]
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
