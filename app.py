"""
Synacal Manufacturing Cost Estimator
Conversational AI estimator using Azure AI Foundry Responses API + Streamlit
"""
import os
import re
import warnings
from datetime import datetime

# Suppress openai v2.x deprecation noise (Responses API is the preferred API anyway)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Synacal Cost Estimator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """<style>
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

.est-header { color:#58a6ff; font-size:.8rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px; }

.app-header {
    background:linear-gradient(135deg,#0d1117,#161b22,#1c2128);
    border:1px solid #30363d; border-left:4px solid #f78166;
    border-radius:12px; padding:24px 28px; margin-bottom:20px;
}
.app-header h1 { color:#e6edf3; margin:0; font-size:1.7rem; font-weight:800; }
.app-header p { color:#8b949e; margin:6px 0 0; font-size:.9rem; }
</style>""",
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in {
    "messages": [],
    "prev_response_id": None,   # chains Responses API turns together
    "_client": None,
    "_agent_name": None,
}.items():
    st.session_state.setdefault(_k, _v)


# ── Client factory (Responses API) ────────────────────────────────────────────
def _build_client(endpoint: str, api_key: str):
    """
    Return an OpenAI client configured for the Azure AI Foundry Responses API.

    Accepted endpoint formats
    ─────────────────────────
    A) "Endpoint (Responses)" copied directly from the agent Playground panel:
       https://resource.services.ai.azure.com/api/projects/name/openai/v1/responses
       → strips /responses, uses remainder as base_url

    B) Full OpenAI v1 path (no /responses suffix):
       https://resource.services.ai.azure.com/api/projects/name/openai/v1

    C) Project endpoint (shorter form):
       https://resource.services.ai.azure.com/api/projects/name
       https://resource.services.ai.azure.com   (appends /api/projects/_project)

    D) Classic Azure OpenAI endpoint:
       https://resource.openai.azure.com
    """
    ep = endpoint.rstrip("/")

    if "openai.azure.com" in ep:
        # ── Classic Azure OpenAI endpoint ──────────────────────────────────
        from openai import AzureOpenAI
        return AzureOpenAI(azure_endpoint=ep, api_key=api_key, api_version="v1")

    # ── Azure AI Foundry / AI Services endpoint ────────────────────────────
    from openai import AzureOpenAI

    # Normalise to the project path if only the base domain was given
    if "services.ai.azure.com" in ep and "/api/projects/" not in ep:
        ep += "/api/projects/_project"

    # Strip /responses suffix to get the base OpenAI v1 URL
    if ep.endswith("/responses"):
        ep = ep[: -len("/responses")]

    # ep should now end with .../openai/v1  OR  .../api/projects/name
    # AzureOpenAI needs the endpoint BEFORE /openai; trim /openai/v1 if present
    if ep.endswith("/openai/v1"):
        azure_ep = ep[: -len("/openai/v1")]
    elif ep.endswith("/openai"):
        azure_ep = ep[: -len("/openai")]
    else:
        azure_ep = ep

    return AzureOpenAI(azure_endpoint=azure_ep, api_key=api_key, api_version="v1")


# ── Formatting guide appended to user messages ────────────────────────────────
_GUIDE = (
    "\n\n---\n"
    "RESPONSE FORMAT — always use these exact section headers for estimates:\n\n"
    "- **Timeline**: [X weeks with brief phase breakdown]\n"
    "- **Material Cost**: USD [amount]  *(raw materials & finishes)*\n"
    "- **Labor Cost**: USD [amount]  *(fabrication & installation)*\n"
    "- **Total Estimate**: USD [amount]\n"
    "- **Reference Projects**: [list project codes from the knowledge base, e.g. RJ979, RJ908]\n\n"
    "If original data shows one combined cost, split it using typical industry ratios.\n"
    "---"
)


def _run_query(
    client,
    agent_name: str,
    user_text: str,
    prev_response_id: str | None,
) -> tuple[str, str]:
    """
    Call the Azure AI Foundry Responses API.
    Returns (response_text, new_response_id).
    Chains turns via previous_response_id for full conversation context.
    """
    kwargs: dict = dict(
        model=agent_name,
        input=[{"role": "user", "content": user_text + _GUIDE}],
    )
    if prev_response_id:
        kwargs["previous_response_id"] = prev_response_id

    response = client.responses.create(**kwargs)

    # Extract text — output_text is the convenience attribute in openai v2.x
    text: str = getattr(response, "output_text", "") or ""
    if not text:
        for item in getattr(response, "output", []):
            for c in getattr(item, "content", []):
                text += getattr(c, "text", "")

    return text.strip() or "No response received.", response.id


# ── Response parsing ───────────────────────────────────────────────────────────
def _parse(text: str) -> dict:
    refs = sorted(set(re.findall(r"\bRJ\d{3,4}\b", text, re.IGNORECASE)))

    def timeline():
        m = re.search(r"\*{0,2}timeline\*{0,2}\**\s*[:\-]\s*([^\n]{5,120})", text, re.IGNORECASE)
        return m.group(1).strip("* \t") if m else None

    def cost(*labels):
        for lbl in labels:
            pats = [
                rf"\*{{0,2}}{re.escape(lbl)}\*{{0,2}}\**\s*[:\-]\s*(?:USD\s*)?\$?([\d,]+(?:\.\d{{1,2}})?)",
                rf"\*{{0,2}}{re.escape(lbl)}\*{{0,2}}\**\s*[:\-]\s*([\d,]+(?:\.\d{{1,2}})?)\s*USD",
            ]
            for p in pats:
                m = re.search(p, text, re.IGNORECASE)
                if m:
                    return float(m.group(1).replace(",", ""))
        return None

    mat = cost("material cost", "material costs", "materials cost", "materials")
    lab = cost("labor cost", "labour cost", "labor", "labour", "installation cost", "fabrication cost")
    tot = cost("total estimate", "total cost", "grand total", "total")

    if tot is None and mat is not None and lab is not None:
        tot = mat + lab

    tl = timeline()
    return {
        "has_estimate": bool(refs or tl or mat or lab or tot),
        "timeline": tl,
        "material_cost": mat,
        "labor_cost": lab,
        "total_cost": tot,
        "references": refs,
    }


# ── Rendering ──────────────────────────────────────────────────────────────────
def _usd(v):
    return f"USD {v:,.0f}" if v is not None else "—"


def _render_card(p: dict):
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
    st.divider()


# ── Auto-connect from env ──────────────────────────────────────────────────────
def _ensure_ready():
    client = st.session_state["_client"]
    agent  = st.session_state["_agent_name"]

    if client is None:
        ep   = os.getenv("AZURE_AI_ENDPOINT", "")
        key  = os.getenv("AZURE_API_KEY", "")
        name = os.getenv("AZURE_AGENT_NAME", "dev-ktech-demo")
        if ep and key:
            try:
                client = _build_client(ep, key)
                st.session_state.update({"_client": client, "_agent_name": name})
                agent = name
            except Exception:
                return None, None

    return client, agent


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Synacal\n**Cost Estimator**")
    st.divider()

    _cfg_open = not bool(os.getenv("AZURE_API_KEY") and os.getenv("AZURE_AI_ENDPOINT"))
    with st.expander("🔑 Azure Configuration", expanded=_cfg_open):
        st.caption(
            "Paste the **Endpoint (Responses)** URL from your agent's "
            "Playground → *(three-dot menu / Publish panel)*"
        )
        _ep = st.text_input(
            "Endpoint (Responses)",
            value=os.getenv("AZURE_AI_ENDPOINT", ""),
            placeholder="https://resource.services.ai.azure.com/api/projects/name/openai/v1/responses",
        )
        _key = st.text_input(
            "API Key",
            value=os.getenv("AZURE_API_KEY", ""),
            type="password",
            help="From AI Foundry Home page → API key field",
        )
        _agent = st.text_input(
            "Agent Name",
            value=os.getenv("AZURE_AGENT_NAME", "dev-ktech-demo"),
            help="Exact name shown in AI Foundry → Build → Agents",
        )
        if st.button("🔗 Connect", type="primary", use_container_width=True):
            with st.spinner("Connecting…"):
                try:
                    _c = _build_client(_ep, _key)
                    st.session_state.update(
                        {"_client": _c, "_agent_name": _agent, "prev_response_id": None}
                    )
                    st.success(f"Connected!  Agent: **{_agent}**")
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.update({"messages": [], "prev_response_id": None})
        st.rerun()

    st.divider()
    st.markdown("**💡 Try these queries:**")
    _EXAMPLES = [
        "Quote a custom MDF wardrobe with LED rails and PU spray paint",
        "Cost for brushed brass identity signage in a restaurant",
        "How much for 4 lift button identity signs in brass?",
        "Teak dining set and credenza — manufacturing cost & timeline?",
        "Oxidized brass decorative buckets for a hotel lobby",
        "LED backlit mirror vanity — cost and lead time?",
        "What if we switch the MDF wardrobe to solid local Teak?",
    ]
    for _ex in _EXAMPLES:
        if st.button(_ex, key=f"ex_{hash(_ex)}", use_container_width=True):
            st.session_state["_pending"] = _ex
            st.rerun()

    st.divider()
    st.caption("Powered by Azure AI Foundry · GPT-4o · Responses API")


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown(
    """<div class="app-header">
    <h1>⚙️ Manufacturing Cost Estimator</h1>
    <p>AI-powered instant estimates from Synacal's historical project knowledge base ·
    Ask for any manufacturing, signage, or furniture project quote</p>
</div>""",
    unsafe_allow_html=True,
)

# ── Ready check ────────────────────────────────────────────────────────────────
client, agent_name = _ensure_ready()
_ready = client is not None and bool(agent_name)

if not _ready:
    st.info(
        "👈 **Configure your Azure credentials** in the sidebar.\n\n"
        "**Where to find your Endpoint (Responses):**\n"
        "1. AI Foundry → Build → Agents → click `dev-ktech-demo`\n"
        "2. Click **Publish** (top-right) → copy **Endpoint (Responses)**\n"
        "   e.g. `https://resource.services.ai.azure.com/api/projects/name/openai/v1/responses`\n\n"
        "**API Key:** AI Foundry Home page → API key field\n\n"
        "Or set `AZURE_AI_ENDPOINT`, `AZURE_API_KEY`, `AZURE_AGENT_NAME` in a `.env` file."
    )

# ── Chat history ───────────────────────────────────────────────────────────────
for _msg in st.session_state["messages"]:
    _av = "👤" if _msg["role"] == "user" else "🤖"
    with st.chat_message(_msg["role"], avatar=_av):
        if _msg["role"] == "assistant" and _msg.get("parsed", {}).get("has_estimate"):
            _render_card(_msg["parsed"])
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
                    client,
                    agent_name,
                    user_input,
                    st.session_state["prev_response_id"],
                )
                st.session_state["prev_response_id"] = new_resp_id
                parsed = _parse(resp_text)
                if parsed["has_estimate"]:
                    _render_card(parsed)
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
