"""
Synacal Manufacturing Cost Estimator
Conversational AI estimator using Azure AI Foundry (agent protocol) + Streamlit
"""
import os
import re
import time
from datetime import datetime

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
    "thread_id": None,
    "_oai": None,          # openai.OpenAI client for the agent
    "_agent_id": None,     # resolved assistant/agent ID for runs
}.items():
    st.session_state.setdefault(_k, _v)

# ── Auth helper ────────────────────────────────────────────────────────────────
class _BearerKey:
    """Wrap a project API key as a TokenCredential so AIProjectClient accepts it."""
    def __init__(self, key: str):
        self._key = key

    def get_token(self, *_scopes, **_kw):
        from azure.core.credentials import AccessToken
        return AccessToken(self._key, int(time.time()) + 86400)


# ── Client factory ─────────────────────────────────────────────────────────────
def _normalise_endpoint(ep: str) -> str:
    """Ensure the endpoint has the /api/projects/_project suffix if needed."""
    ep = ep.rstrip("/")
    if ep.endswith(".services.ai.azure.com"):
        ep += "/api/projects/_project"
    return ep


def _make_oai_client(endpoint: str, api_key: str, agent_name: str):
    """
    Return an openai.OpenAI client pointed at the Foundry agent's OpenAI-compatible
    protocol endpoint.  No actual API calls are made until the first query.
    """
    from azure.ai.projects import AIProjectClient  # local import keeps startup fast

    full_ep = _normalise_endpoint(endpoint)
    project = AIProjectClient(
        endpoint=full_ep,
        credential=_BearerKey(api_key),
        allow_preview=True,
    )
    # Override api_key so the returned OpenAI client uses Bearer auth with the API key
    oai = project.get_openai_client(agent_name=agent_name, api_key=api_key)
    return oai


def _resolve_agent_id(oai_client, agent_name: str) -> str:
    """Try to find the assistant ID by listing assistants; fall back to agent name."""
    try:
        for page in oai_client.beta.assistants.list(limit=100):
            # page may be a SyncPage or list depending on SDK version
            items = page if isinstance(page, list) else getattr(page, "data", [page])
            for a in items:
                if getattr(a, "name", None) == agent_name:
                    return a.id
    except Exception:
        pass
    return agent_name  # Foundry routes by endpoint; name often works too


# ── Formatting guide appended to every user message ───────────────────────────
_GUIDE = (
    "\n\n---\n"
    "RESPONSE FORMAT (use when giving estimates):\n\n"
    "- **Timeline**: [X weeks with phase details]\n"
    "- **Material Cost**: USD [amount]  *(raw materials & finishes)*\n"
    "- **Labor Cost**: USD [amount]  *(fabrication & installation)*\n"
    "- **Total Estimate**: USD [amount]\n"
    "- **Reference Projects**: [Project codes e.g. RJ979, RJ908 from the knowledge base]\n\n"
    "Even when original data has a single cost figure, split it into Material/Labor "
    "using typical industry ratios for the product type.\n"
    "---"
)


def _run_query(oai, agent_id: str, user_text: str, thread_id: str | None) -> tuple[str, str]:
    """Send a message, run the agent, return (response_text, thread_id)."""
    if thread_id is None:
        thread_id = oai.beta.threads.create().id

    oai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_text + _GUIDE,
    )

    run = oai.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=agent_id,
    )

    if run.status == "failed":
        err = getattr(run, "last_error", "unknown error")
        raise RuntimeError(f"Agent run failed: {err}")

    msgs = oai.beta.threads.messages.list(thread_id=thread_id)
    msg_list = getattr(msgs, "data", list(msgs))

    for m in msg_list:
        if m.role == "assistant":
            parts = []
            for c in m.content:
                if hasattr(c, "text") and hasattr(c.text, "value"):
                    parts.append(c.text.value)
            return "\n\n".join(parts) or "No response text.", thread_id

    return "The agent returned no response.", thread_id


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
            f'<div class="ref-section"><div class="ref-label">📚 Reference Evidence from Knowledge Base</div>'
            f'<div>{badges}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()


# ── Auto-connect from env ──────────────────────────────────────────────────────
def _ensure_ready():
    oai = st.session_state["_oai"]
    agent_id = st.session_state["_agent_id"]

    if oai is None:
        ep = os.getenv("AZURE_AI_ENDPOINT", "")
        key = os.getenv("AZURE_API_KEY", "")
        name = os.getenv("AZURE_AGENT_NAME", os.getenv("AZURE_AGENT_ID", "dev-ktech-demo"))
        if ep and key and name:
            try:
                oai = _make_oai_client(ep, key, name)
                st.session_state["_oai"] = oai
                st.session_state["_agent_name"] = name
            except Exception:
                return None, None

    if agent_id is None and oai is not None:
        name = st.session_state.get("_agent_name", os.getenv("AZURE_AGENT_NAME", "dev-ktech-demo"))
        agent_id = _resolve_agent_id(oai, name)
        st.session_state["_agent_id"] = agent_id

    return oai, agent_id


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Synacal\n**Cost Estimator**")
    st.divider()

    _cfg_open = not bool(os.getenv("AZURE_API_KEY") and os.getenv("AZURE_AI_ENDPOINT"))
    with st.expander("🔑 Azure Configuration", expanded=_cfg_open):
        _ep = st.text_input(
            "Project Endpoint",
            value=os.getenv("AZURE_AI_ENDPOINT", ""),
            placeholder="https://xxx.services.ai.azure.com/api/projects/_project",
            help="From AI Foundry Home → Project endpoint. Append /api/projects/_project if not already present.",
        )
        _key = st.text_input(
            "API Key",
            value=os.getenv("AZURE_API_KEY", ""),
            type="password",
            help="From AI Foundry Home → API key",
        )
        _agent = st.text_input(
            "Agent Name",
            value=os.getenv("AZURE_AGENT_NAME", os.getenv("AZURE_AGENT_ID", "dev-ktech-demo")),
            help="Name shown in AI Foundry → Build → Agents",
        )
        if st.button("🔗 Connect", type="primary", use_container_width=True):
            with st.spinner("Connecting…"):
                try:
                    _oai = _make_oai_client(_ep, _key, _agent)
                    _aid = _resolve_agent_id(_oai, _agent)
                    st.session_state.update({"_oai": _oai, "_agent_id": _aid, "_agent_name": _agent})
                    st.success(f"Connected to **{_agent}**")
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.update({"messages": [], "thread_id": None})
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
    st.caption("Powered by Azure AI Foundry · GPT-4o · Azure AI Search")

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
oai, agent_id = _ensure_ready()
_ready = oai is not None and agent_id is not None

if not _ready:
    st.info(
        "👈 **Configure your Azure credentials** in the sidebar to begin.\n\n"
        "You need:\n"
        "- **Project Endpoint** – from AI Foundry Home page (Project endpoint field).\n"
        "  Append `/api/projects/_project` if the URL ends with `.services.ai.azure.com`\n"
        "- **API Key** – from AI Foundry Home page (API key field)\n"
        "- **Agent Name** – `dev-ktech-demo` (or whatever your agent is named)\n\n"
        "Or set `AZURE_AI_ENDPOINT`, `AZURE_API_KEY`, and `AZURE_AGENT_NAME` in a `.env` file."
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
                resp, new_tid = _run_query(oai, agent_id, user_input, st.session_state["thread_id"])
                st.session_state["thread_id"] = new_tid
                parsed = _parse(resp)
                if parsed["has_estimate"]:
                    _render_card(parsed)
                st.markdown(resp)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": resp, "parsed": parsed}
                )
            except Exception as exc:
                err = f"❌ Error: {exc}"
                st.error(err)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": err, "parsed": {"has_estimate": False}}
                )
