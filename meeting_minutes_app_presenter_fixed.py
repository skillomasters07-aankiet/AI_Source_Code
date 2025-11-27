# meeting_minutes_app_presenter_fixed.py
# ===========================================================
# Streamlit Meeting Minutes Generator (Fixed + TXT upload + Robust JSON cleanup)
# - Avoids f-string brace issue by using a safe template.replace()
# - Adds sidebar file uploader for meeting notes (.txt)
# - Adds robust local JSON cleanup + LLM-based cleanup fallback
# - Presenter-ready layout, history, action tracker, AI analysis
# ===========================================================

import streamlit as st
import json
import datetime
import re
import httpx

# ---- ADDED FOR PDF EXPORT (minimal change) ----
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
# -----------------------------------------------

# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="Meeting Minutes — Hackathon Demo",
    page_icon="📋",
    layout="wide",
)

# ---------------------- LLM VARS ----------------------
LLM_API_KEY = "sk-UXpc9PN5563A8E4PxeTBPw"  # set your key in production
LLM_BASE_URL = "https://genailab.tcs.in"
LLM_MODEL = "azure_ai/genailab-maas-DeepSeek-V3-0324"

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# ---------------------- JSON UTILITIES ----------------------

JSON_OBJ_RE = re.compile(r"(\{[\s\S]*\})", re.DOTALL)
JSON_ARR_RE = re.compile(r"(\[[\s\S]*\])", re.DOTALL)

def extract_json_by_brace(text: str):
    text = text.strip()
    if not text:
        return None
    for start_idx, ch in enumerate(text):
        if ch in ("{", "["):
            opener = ch
            closer = "}" if ch == "{" else "]"
            depth = 0
            in_str = False
            esc = False
            for i in range(start_idx, len(text)):
                c = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == opener:
                        depth += 1
                    elif c == closer:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start_idx:i+1])
                            except Exception:
                                return None
    return None

def extract_json(text: str):
    """
    Existing extraction heuristics (tries direct load, markers, regex, brace-matching).
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start_marker = "===JSON_START==="
    end_marker = "===JSON_END==="
    if start_marker in text and end_marker in text:
        try:
            start = text.index(start_marker) + len(start_marker)
            end = text.index(end_marker, start)
            snippet = text[start:end].strip()
            return json.loads(snippet)
        except Exception:
            pass
    for regex in (JSON_OBJ_RE, JSON_ARR_RE):
        m = regex.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return extract_json_by_brace(text)

# ---------------------- DEFAULT MINUTES ----------------------

def default_minutes():
    return {
        "title": "Meeting Minutes",
        "date": datetime.date.today().isoformat(),
        "participants": [],
        "summary": "",
        "decisions": [],
        "action_items": [],
        "open_questions": [],
        "key_takeaways": []
    }

# ---------------------- SAFE PROMPT (NO F-STRING BRACES) ----------------------

def build_prompt(meeting_text: str) -> str:
    # Use a normal string and substitute the meeting text to avoid f-string brace parsing
    template = """
You convert raw meeting notes into structured JSON meeting minutes.

Return ONLY a VALID JSON OBJECT (no commentary) between the exact markers:
===JSON_START===
{
  "title": "string",
  "date": "YYYY-MM-DD",
  "participants": ["name", "..."],
  "summary": "string",
  "decisions": ["..."],
  "action_items": [{"assignee": "name", "due_date": "YYYY-MM-DD", "description": "...", "status": "open"}],
  "open_questions": ["..."],
  "key_takeaways": ["..."]
}
===JSON_END===

Input Notes:
<<<MEETING_NOTES>>>
{MEETING_TEXT}
<<<END_NOTES>>>

Output EXACTLY the JSON object between the markers ===JSON_START=== and ===JSON_END===, nothing else.
"""
    return template.replace("{MEETING_TEXT}", meeting_text or "")

# ------------------ Robust JSON cleanup + LLM-based fixer ------------------

def try_local_cleanup(raw_text: str):
    """
    Try multiple local cleanups to coax json.loads to succeed.
    Returns Python object or None.
    """
    if not raw_text:
        return None

    # 1) If exact markers exist, extract between them first
    start_marker = "===JSON_START==="
    end_marker = "===JSON_END==="
    if start_marker in raw_text and end_marker in raw_text:
        try:
            s = raw_text.index(start_marker) + len(start_marker)
            e = raw_text.index(end_marker, s)
            snippet = raw_text[s:e].strip()
            return json.loads(snippet)
        except Exception:
            pass

    # 2) Try brute-force regex for first {...} or [...]
    obj_re = re.compile(r"(\{[\s\S]*\})", re.DOTALL)
    arr_re = re.compile(r"(\[[\s\S]*\])", re.DOTALL)
    for regex in (obj_re, arr_re):
        m = regex.search(raw_text)
        if m:
            candidate = m.group(1)
            # quick sanitation: remove trailing commas before } or ]
            cand = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                return json.loads(cand)
            except Exception:
                # try simple fixes: convert single quotes to double quotes (risky)
                try:
                    cand2 = cand.replace("'", '"')
                    return json.loads(cand2)
                except Exception:
                    pass

    # 3) Brace-matching fallback
    try:
        return extract_json_by_brace(raw_text)
    except Exception:
        return None


def ask_llm_to_clean(raw_text: str, temperature=0.0):
    """
    Ask the LLM to return ONLY the valid JSON object (no commentary) extracted
    from the raw_text. Returns (parsed_object_or_None, cleaned_raw_text).
    """
    cleanup_prompt = (
        "You were previously asked to produce a JSON meeting minutes object but the response contained\n"
        "extra commentary and/or invalid JSON. Extract the JSON object only and RETURN ONLY that\n"
        "JSON object and nothing else. If you cannot find a JSON object, return an empty JSON object {}.\n\n"
        "Here is the original output (do not invent anything, only extract):\n\n"
        "-----BEGIN ORIGINAL-----\n"
        f"{raw_text}\n"
        "-----END ORIGINAL-----\n\n"
        "Return exactly the JSON object (no code fences or explanation)."
    )

    parsed = None
    cleaned_raw = None

    if ChatOpenAI is None:
        return None, None

    try:
        client = httpx.Client(verify=False)
        llm = ChatOpenAI(base_url=LLM_BASE_URL, model=LLM_MODEL, api_key=LLM_API_KEY, http_client=client)
        resp = llm.invoke(cleanup_prompt, temperature=temperature) if hasattr(llm, 'invoke') else llm.invoke(cleanup_prompt)

        # normalize raw response to string
        if isinstance(resp, str):
            cleaned_raw = resp
        elif isinstance(resp, dict):
            cleaned_raw = resp.get("content") or resp.get("text") or json.dumps(resp, default=str)
        else:
            cleaned_raw = str(resp)

        # Try local extraction on the cleaned output
        parsed = try_local_cleanup(cleaned_raw)
        return parsed, cleaned_raw

    except Exception:
        return None, None

# ---------------------- LLM CALL (patched) ----------------------

def call_llm(prompt: str, temperature=0.2):
    """
    Call the LLM and attempt robust JSON extraction.
    Returns (parsed_python_obj_or_None, raw_text_used_for_debug).
    """
    if ChatOpenAI is None:
        mock = default_minutes()
        mock["summary"] = "Mock summary — LLM client not available in this environment."
        raw = json.dumps(mock, indent=2)
        return mock, raw

    try:
        client = httpx.Client(verify=False)
        llm = ChatOpenAI(
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            http_client=client
        )
        resp = llm.invoke(prompt)

        # Obtain a raw string from resp safely
        raw = None
        if isinstance(resp, str):
            raw = resp
        elif isinstance(resp, dict):
            raw = resp.get("content") or resp.get("text") or json.dumps(resp, default=str)
        else:
            raw = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)

        raw = "" if raw is None else str(raw)
        st.session_state.raw_output = raw

        # 1) Try existing extractor
        parsed = extract_json(raw)

        # 2) Try local cleanup heuristics
        if parsed is None:
            parsed = try_local_cleanup(raw)

        # 3) If still None, ask the LLM to clean its previous output (low-temp)
        if parsed is None:
            parsed_fix, cleaned_raw = ask_llm_to_clean(raw, temperature=0.0)
            if parsed_fix is not None:
                parsed = parsed_fix
                # Prefer cleaned raw for debugging
                if cleaned_raw:
                    raw = cleaned_raw
                    st.session_state.raw_output = raw

        return parsed, raw

    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None, f"LLM error: {e}"

# ---------------------- SESSION STATE ----------------------

if "minutes" not in st.session_state:
    st.session_state.minutes = default_minutes()
if "raw_output" not in st.session_state:
    st.session_state.raw_output = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ---------------------- STYLING ----------------------

st.markdown(
    """
    <style>
    body {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .big-title {font-size:28px; font-weight:700;}
    .muted {color: #6c757d;}
    .card {background: #ffffff; padding: 16px; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06);}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- HEADER ----------------------

header_col1, header_col2 = st.columns([6,1])
with header_col1:
    st.markdown("<div class='big-title'>📋 AI 🤖 Meeting Minutes 🤖</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'> AI Powered — editable minutes, history, dashboards and quick AI analysis.</div>", unsafe_allow_html=True)
with header_col2:
    st.metric(label="Saved meetings", value=len(st.session_state.history))

st.markdown("---")

# ---------------------- SIDEBAR (with TXT upload + Retry) ----------------------

with st.sidebar:
    st.header("Controls")

    # File uploader for .txt meeting notes
    uploaded_file = st.file_uploader("Upload meeting notes (.txt)", type=["txt"])
    if uploaded_file is not None:
        try:
            file_text = uploaded_file.read().decode("utf-8")
        except Exception:
            file_text = uploaded_file.read().decode("latin-1")
        st.session_state._uploaded_notes = file_text
        st.success("Loaded uploaded file into notes preview.")

    # Text area shows uploaded file if present, otherwise manual paste
    meeting_text = st.text_area(
        "Paste raw meeting notes (or upload .txt above):",
        value=st.session_state.get("_uploaded_notes", ""),
        height=200,
        placeholder="Paste messy notes here — bullets, chat logs, etc."
    )

    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.05)
    if st.button("Generate Minutes"):
        prompt = build_prompt(meeting_text)
        parsed, raw = call_llm(prompt, temperature)
        if parsed is None:
            st.error("Could not parse JSON from model output.")
            st.session_state.minutes = default_minutes()
        else:
            minutes = default_minutes()
            # only update keys that exist in parsed dict
            if isinstance(parsed, dict):
                minutes.update(parsed)
            # ensure all keys present
            for k in default_minutes().keys():
                if k not in minutes:
                    minutes[k] = default_minutes()[k]
            st.session_state.minutes = minutes
        st.session_state.raw_output = raw

    st.markdown("---")
    if st.button("Save current meeting to history"):
        m = st.session_state.minutes.copy()
        m["saved_at"] = datetime.datetime.now().isoformat()
        st.session_state.history.insert(0, m)
        st.success("Saved to history")

    st.markdown("---")
    st.header("AI Analysis")
    st.write("Quick sentiment + risk analysis for current minutes")
    if st.button("Analyze current minutes (AI)"):
        analysis_prompt = (
            "Analyze the following meeting minutes for overall sentiment (positive/neutral/negative), "
            "and return JSON with fields: {\"sentiment\":\"...\", \"risk_flags\": [\"...\"], \"recommendations\": [\"...\"]}. "
            "Return ONLY the JSON object.\n\nMinutes:\n" + json.dumps(st.session_state.minutes)
        )
        parsed_a, raw_a = call_llm(analysis_prompt, temperature)
        if parsed_a is None:
            st.error("Could not parse analysis JSON from model output.")
            st.session_state.analysis = {"sentiment": "unknown", "risk_flags": [], "recommendations": []}
        else:
            st.session_state.analysis = parsed_a
        st.session_state.raw_output = raw_a or st.session_state.raw_output

    st.markdown("---")
    st.header("History")
    if len(st.session_state.history) == 0:
        st.info("No saved meetings yet — save current meeting to build history.")
    else:
        sel = st.selectbox(
            "Open saved meeting",
            options=[f"{i+1}. {h.get('title','Meeting')} — {h.get('date','') or h.get('saved_at','')}" for i,h in enumerate(st.session_state.history)]
        )
        idx = int(sel.split(".")[0]) - 1
        if st.button("Load selected meeting"):
            st.session_state.minutes = st.session_state.history[idx].copy()
            st.success("Loaded into editor")
        if st.button("Clear history"):
            st.session_state.history = []
            st.success("Cleared history")

    st.markdown("---")
    
# ---------------------- MAIN: Tabs ----------------------

tabs = st.tabs(["Edit", "One-page Summary", "History & Tracker", "Action Dashboard", "Analytics"])

# --- EDIT TAB ---
with tabs[0]:
    st.subheader("Edit Meeting Minutes")
    minutes = st.session_state.minutes
    left, right = st.columns([2,1])
    with left:
        minutes["title"] = st.text_input("Title", minutes.get("title","Meeting Minutes"))
        try:
            minutes["date"] = st.date_input("Date", datetime.date.fromisoformat(minutes.get("date", datetime.date.today().isoformat()))).isoformat()
        except Exception:
            minutes["date"] = datetime.date.today().isoformat()
        participants = st.text_area("Participants (one per line)", "\n".join(minutes.get('participants',[])))
        minutes["participants"] = [p.strip() for p in participants.splitlines() if p.strip()]
        minutes["summary"] = st.text_area("Executive Summary", minutes.get('summary',''), height=140)

        st.markdown("### Decisions")
        dec = st.text_area("Decisions (one per line)", "\n".join(minutes.get('decisions',[])))
        minutes["decisions"] = [d for d in dec.splitlines() if d.strip()]

        st.markdown("### Open Questions")
        oq = st.text_area("Open Questions (one per line)", "\n".join(minutes.get('open_questions',[])))
        minutes["open_questions"] = [q for q in oq.splitlines() if q.strip()]

    with right:
        st.markdown("### Action Items")
        if st.button("Add Action Item"):
            ai = minutes.get('action_items', [])
            ai.append({"assignee":"","due_date":"","description":"","status":"open"})
            minutes['action_items'] = ai

        # show action items in compact form
        for i, ai in enumerate(minutes.get('action_items', [])):
            with st.expander(f"{i+1}. {ai.get('description','(no description)')}"):
                assignee = st.text_input("Assignee", value=ai.get('assignee',''), key=f"ai_assignee_{i}")
                due = st.text_input("Due (YYYY-MM-DD)", value=ai.get('due_date',''), key=f"ai_due_{i}")
                desc = st.text_area("Description", value=ai.get('description',''), key=f"ai_desc_{i}", height=80)
                status = st.selectbox("Status", options=['open','in_progress','done'], index=['open','in_progress','done'].index(ai.get('status','open')), key=f"ai_status_{i}")
                if st.button("Remove", key=f"ai_rm_{i}"):
                    minutes['action_items'].pop(i)
                else:
                    minutes['action_items'][i] = {"assignee":assignee, "due_date":due, "description":desc, "status":status}

    if st.button("Save Edits"):
        st.session_state.minutes = minutes
        st.success("Current minutes updated")

# --- ONE-PAGE SUMMARY ---
with tabs[1]:
    st.subheader("One-page Summary")
    m = st.session_state.minutes
    st.markdown(f"# {m.get('title')}")
    st.markdown(f"**Date:** {m.get('date')} — **Participants:** {', '.join(m.get('participants',[]))}")
    st.markdown("---")
    st.markdown("## Executive Summary")
    st.write(m.get('summary','_No summary provided_'))
    st.markdown("## Key Takeaways")
    if m.get('key_takeaways'):
        for k in m.get('key_takeaways'):
            st.write(f"- {k}")
    else:
        st.info("No key takeaways")

    st.markdown("---")
    st.markdown("## Action Items")
    for i, ai in enumerate(m.get('action_items', [])):
        st.markdown(f"**{i+1}. {ai.get('description','(no description)')}**")
        st.write(f"Assignee: {ai.get('assignee','')} • Due: {ai.get('due_date','')} • Status: {ai.get('status','open')}")

# --- HISTORY & TRACKER ---
with tabs[2]:
    st.subheader("History & Meeting Browser")
    if not st.session_state.history:
        st.info("No meetings saved to history yet — use the 'Save current meeting to history' button in the sidebar.")
    else:
        for i, h in enumerate(st.session_state.history):
            with st.expander(f"{i+1}. {h.get('title','Meeting')} — {h.get('date','')}"):
                st.markdown(f"**Saved at:** {h.get('saved_at','')}")
                st.markdown("**Summary**")
                st.write(h.get('summary',''))
                st.markdown("**Action Items**")
                for aidx, ai in enumerate(h.get('action_items', [])):
                    st.write(f"- [{ai.get('status','open')}] {ai.get('description','')} — {ai.get('assignee','')} (Due: {ai.get('due_date','')})")
                if st.button(f"Load into Editor", key=f"load_{i}"):
                    st.session_state.minutes = h.copy()
                    st.success("Loaded selected meeting into editor")

# --- ACTION DASHBOARD ---
with tabs[3]:
    st.subheader("Action Items Dashboard")
    # aggregate from history + current
    all_items = []
    def add_items(m):
        for ai in m.get('action_items', []):
            itm = ai.copy()
            itm['meeting_title'] = m.get('title')
            itm['meeting_date'] = m.get('date')
            all_items.append(itm)
    add_items(st.session_state.minutes)
    for h in st.session_state.history:
        add_items(h)

    if not all_items:
        st.info("No action items found.")
    else:
        import pandas as pd
        rows = []
        overdue = 0
        done = 0
        for it in all_items:
            rows.append({
                'description': it.get('description',''),
                'assignee': it.get('assignee',''),
                'due_date': it.get('due_date',''),
                'status': it.get('status','open'),
                'meeting': it.get('meeting_title','')
            })
            if it.get('status') == 'done':
                done += 1
            try:
                if it.get('due_date'):
                    dd = datetime.date.fromisoformat(it.get('due_date'))
                    if dd < datetime.date.today() and it.get('status') != 'done':
                        overdue += 1
            except Exception:
                pass
        df = pd.DataFrame(rows)
        st.dataframe(df)

        cols = st.columns(3)
        cols[0].metric("Total Actions", len(rows))
        cols[1].metric("Overdue", overdue)
        cols[2].metric("Completed", done)

        st.markdown("**Mark an action as done**")
        sel = st.selectbox("Select action to mark done", options=[f"{i+1}. {r['description']} — {r['assignee']}" for i,r in enumerate(rows)])
        idx = int(sel.split('.')[0]) - 1
        if st.button("Mark Done"):
            target = rows[idx]
            updated = False
            # update current minutes
            for ai in st.session_state.minutes.get('action_items', []):
                if ai.get('description') == target['description'] and ai.get('assignee') == target['assignee']:
                    ai['status'] = 'done'
                    updated = True
            # update history
            if not updated:
                for h in st.session_state.history:
                    for ai in h.get('action_items', []):
                        if ai.get('description') == target['description'] and ai.get('assignee') == target['assignee']:
                            ai['status'] = 'done'
                            updated = True
            if updated:
                st.success('Marked as done')
            else:
                st.warning('Could not find the selected action to update')

# --- ANALYTICS ---
with tabs[4]:
    st.subheader("Analytics")
    total_meetings = 1 + len(st.session_state.history) if st.session_state.minutes else len(st.session_state.history)
    total_actions = sum(len(h.get('action_items', [])) for h in st.session_state.history) + len(st.session_state.minutes.get('action_items', []))
    st.write(f"**Total meetings:** {total_meetings}  —  **Total action items:** {total_actions}")

    st.markdown("### Last AI Analysis")
    if st.session_state.analysis is None:
        st.info("No analysis run yet. Use 'Analyze current minutes (AI)' in the sidebar.")
    else:
        a = st.session_state.analysis
        st.write(f"**MOM Sentiments:** {a.get('sentiment','unknown')}")
        st.write("**Risk Flags:**")
        for f in a.get('risk_flags', []):
            st.write(f"- {f}")
        st.write("**Recommendations:**")
        for r in a.get('recommendations', []):
            st.write(f"- {r}")

    st.markdown("---")
    #st.subheader("Raw LLM Output (debug)")
    #st.text_area("Raw LLM Output", st.session_state.get('raw_output',''), height=200)

# ---------------------- EXPORT ----------------------

st.sidebar.markdown("---")
# ---------- REPLACED: JSON download -> PDF download ----------
def minutes_to_pdf_bytes(minutes_obj):
    """
    Render the meeting minutes dict into a simple PDF and return BytesIO buffer.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, minutes_obj.get("title", "Meeting Minutes"))
    y -= 24

    # Date & participants
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Date: {minutes_obj.get('date','')}")
    y -= 14
    participants = ", ".join(minutes_obj.get("participants", []))
    c.drawString(margin, y, f"Participants: {participants}")
    y -= 20

    # Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Executive Summary")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in str(minutes_obj.get("summary", "")).splitlines():
        if y < margin + 80:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
        c.drawString(margin, y, line)
        y -= 12
    y -= 8

    # Decisions
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Decisions")
    y -= 16
    c.setFont("Helvetica", 10)
    for d in minutes_obj.get("decisions", []):
        if y < margin + 40:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
        c.drawString(margin + 8, y, f"• {d}")
        y -= 12
    y -= 8

    # Action Items
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Action Items")
    y -= 16
    c.setFont("Helvetica", 10)
    for ai in minutes_obj.get("action_items", []):
        if y < margin + 60:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
        desc = ai.get("description","(no description)")
        assignee = ai.get("assignee","")
        due = ai.get("due_date","")
        status = ai.get("status","open")
        c.drawString(margin + 8, y, f"• {desc}")
        y -= 12
        c.drawString(margin + 20, y, f"Assignee: {assignee} • Due: {due} • Status: {status}")
        y -= 16
    y -= 8

    # Open Questions
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Open Questions")
    y -= 16
    c.setFont("Helvetica", 10)
    for q in minutes_obj.get("open_questions", []):
        if y < margin + 40:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
        c.drawString(margin + 8, y, f"• {q}")
        y -= 12
    y -= 8

    # Key Takeaways
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Key Takeaways")
    y -= 16
    c.setFont("Helvetica", 10)
    for k in minutes_obj.get("key_takeaways", []):
        if y < margin + 40:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
        c.drawString(margin + 8, y, f"• {k}")
        y -= 12

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin, margin - 10 + 8, f"Generated: {datetime.datetime.now().isoformat()}")
    c.save()
    buffer.seek(0)
    return buffer

try:
    pdf_buffer = minutes_to_pdf_bytes(st.session_state.minutes)
    st.sidebar.download_button(
        label="Download current minutes (PDF)",
        data=pdf_buffer.getvalue(),
        file_name="meeting_minutes.pdf",
        mime="application/pdf"
    )
except Exception as e:
    st.sidebar.error(f"PDF generation failed: {e}")
# ---------------------------------------------------------------
