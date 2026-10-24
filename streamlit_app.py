"""
HR Attendance Comparison – Planned Shifts vs. Actual Clock-ins
==============================================================

The app only DISPLAYS results on screen: tables, an executive summary and draft emails.
It has no export / save / download features.

HOW TO RUN LOCALLY
------------------
1. pip install streamlit pandas            (optionally: anthropic or openai for real AI)
2. streamlit run app.py

HOW TO DEPLOY ON STREAMLIT COMMUNITY CLOUD
------------------------------------------
1. Push this file (app.py) and a requirements.txt to a GitHub repo.
   requirements.txt:
       streamlit
       pandas
       anthropic        # optional, only for real AI
   (Add `openpyxl` ONLY if you want to upload .xlsx files. CSV uploads need nothing extra.)
2. Go to https://share.streamlit.io -> "New app" -> pick the repo and app.py.
3. (Optional, for real AI) In the app's Settings -> Secrets add:
       ANTHROPIC_API_KEY = "sk-ant-..."
   or  OPENAI_API_KEY = "sk-..."
   Without a key the app uses a built-in mock AI that still produces a Hebrew summary.

EXPECTED FILE FORMAT (CSV, or Excel if openpyxl is installed)
-------------------------------------------------------------
Each file needs an "Employee ID" and/or "Name" column. Hebrew/English headers are
recognised (e.g. "מספר עובד", "שם עובד"). An optional "Email" column is shown in drafts.
Other columns (shift time, clock-in time, department...) are kept and displayed.
"""

import html
import io
import os
from datetime import date

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Column handling
# ----------------------------------------------------------------------------
ID_ALIASES = {"employeeid", "empid", "id", "workerid", "מספרעובד", "מסעובד", "מספרזהות", "תז", "מזהה"}
NAME_ALIASES = {"name", "employeename", "fullname", "employee", "שם", "שםעובד", "שםמלא"}
EMAIL_ALIASES = {"email", "mail", "emailaddress", "אימייל", "מייל", 'דוא"ל', "דואל"}


def _norm_header(h) -> str:
    return str(h).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _find_col(df: pd.DataFrame, aliases: set):
    for col in df.columns:
        if _norm_header(col) in aliases:
            return col
    return None


def read_any(uploaded) -> pd.DataFrame:
    """Read an uploaded CSV / Excel file into a DataFrame (all values as text)."""
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(uploaded, dtype=str)
        except ImportError:
            raise ValueError(
                f"Reading '{uploaded.name}' requires the 'openpyxl' package. "
                "Either add `openpyxl` to requirements.txt or upload the file as CSV."
            )
    raw = uploaded.getvalue()
    for enc in ("utf-8-sig", "cp1255", "latin-1"):  # cp1255 = Windows Hebrew
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)), dtype=str)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"Could not read {uploaded.name}")


def prepare(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Rename the ID / Name / Email columns to standard names and clean the values."""
    id_col = _find_col(df, ID_ALIASES)
    name_col = _find_col(df, NAME_ALIASES)
    email_col = _find_col(df, EMAIL_ALIASES)
    if id_col is None and name_col is None:
        raise ValueError(f'"{label}": no "Employee ID" or "Name" column found. Columns: {list(df.columns)}')

    rename = {}
    if id_col:
        rename[id_col] = "Employee ID"
    if name_col:
        rename[name_col] = "Name"
    if email_col:
        rename[email_col] = "Email"
    df = df.rename(columns=rename).copy()

    if "Employee ID" in df:
        df["Employee ID"] = df["Employee ID"].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if "Name" in df:
        df["Name"] = df["Name"].fillna("").astype(str).str.strip()
    return df


def _name_key(s: str) -> str:
    return " ".join(str(s).lower().split())


# ----------------------------------------------------------------------------
# Comparison logic
# ----------------------------------------------------------------------------
def compare(planned: pd.DataFrame, actual: pd.DataFrame):
    """Return (matched, missing, unplanned, info). Matching is by Employee ID when both
    files have it, otherwise falls back to Name."""
    use_id = "Employee ID" in planned and "Employee ID" in actual
    for d in (planned, actual):
        if use_id:
            d["_key"] = d["Employee ID"]
            if "Name" in d:  # rows with an empty ID fall back to name
                empty = d["_key"] == ""
                d.loc[empty, "_key"] = "name:" + d.loc[empty, "Name"].map(_name_key)
        else:
            d["_key"] = d["Name"].map(_name_key)

    dup_p = int(planned["_key"].duplicated().sum())
    dup_a = int(actual["_key"].duplicated().sum())
    planned = planned.drop_duplicates("_key")
    actual = actual.drop_duplicates("_key")

    matched = planned.merge(actual, on="_key", how="inner", suffixes=("", " (Actual)"))
    missing = planned[~planned["_key"].isin(actual["_key"])]
    unplanned = actual[~actual["_key"].isin(planned["_key"])]

    # Warn when the ID matches but the name differs between files
    name_mismatch = 0
    if use_id and "Name" in matched and "Name (Actual)" in matched:
        diff = matched["Name"].map(_name_key) != matched["Name (Actual)"].map(_name_key)
        name_mismatch = int(diff.sum())
        matched = matched.assign(**{"Name Mismatch": diff.map({True: "⚠️", False: ""})})
        matched = matched.drop(columns=[c for c in ["Employee ID (Actual)", "Email (Actual)"] if c in matched])

    clean = lambda d: d.drop(columns=["_key"]).reset_index(drop=True)
    info = {"use_id": use_id, "dup_planned": dup_p, "dup_actual": dup_a, "name_mismatch": name_mismatch}
    return clean(matched), clean(missing), clean(unplanned), info


# ----------------------------------------------------------------------------
# AI layer (real LLM if a key exists, otherwise a mock)
# ----------------------------------------------------------------------------
def _get_secret(name: str):
    try:
        return st.secrets[name]
    except Exception:  # no secrets.toml / key missing
        return os.environ.get(name)


def call_llm(prompt: str):
    """Call Anthropic or OpenAI if a key is configured. Returns text or None."""
    try:
        key = _get_secret("ANTHROPIC_API_KEY")
        if key:
            import anthropic

            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        key = _get_secret("OPENAI_API_KEY")
        if key:
            from openai import OpenAI

            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
    except Exception as e:  # never crash the app because of the AI call
        st.warning(f"AI call failed, using the built-in mock instead: {e}")
    return None


def _names(df: pd.DataFrame, limit=40) -> str:
    if "Name" in df:
        vals = df["Name"].tolist()
    else:
        vals = df.get("Employee ID", pd.Series(dtype=str)).tolist()
    return ", ".join(vals[:limit]) or "—"


def generate_executive_summary(matched, missing, unplanned, report_date: str) -> str:
    """Hebrew 'Managerial Executive Summary' about the discrepancies."""
    total_planned = len(matched) + len(missing)
    pct = round(100 * len(matched) / total_planned) if total_planned else 0

    prompt = f"""אתה אנליסט משאבי אנוש. כתוב סיכום מנהלים קצר בעברית (עד 150 מילים) על פערי נוכחות ליום {report_date}.
נתונים:
- תוכננו {total_planned} עובדים; הגיעו בפועל {len(matched)} ({pct}%).
- לא התייצבו ({len(missing)}): {_names(missing)}
- התייצבו ללא תכנון ({len(unplanned)}): {_names(unplanned)}
כלול: תמונת מצב, הערכת השפעה תפעולית, ו-2-3 המלצות פעולה. אל תמציא עובדים או נתונים שלא קיבלת."""
    text = call_llm(prompt)
    if text:
        return text

    # ---- MOCK AI (placeholder) ----
    lines = [
        f"סיכום מנהלים – {report_date}",
        "",
        f"מתוך {total_planned} עובדים שתוכננו למשמרת, {len(matched)} ({pct}%) התייצבו בפועל.",
    ]
    if missing.empty:
        lines.append("כל העובדים המתוכננים התייצבו – לא נרשמו היעדרויות לא מוסברות.")
    else:
        lines.append(f"{len(missing)} עובדים לא התייצבו ולא רשמו כניסה: {_names(missing, 10)}.")
        if pct < 85:
            lines.append("שיעור ההתייצבות נמוך מהיעד (85%) ועלול לפגוע בכיסוי המשמרת.")
    if not unplanned.empty:
        lines.append(f"{len(unplanned)} עובדים נכנסו ללא שיבוץ מתוכנן: {_names(unplanned, 10)}. ייתכן שמדובר בהחלפות משמרת לא מתועדות.")
    lines += [
        "",
        "המלצות:",
        "1. ליצור קשר עם העובדים החסרים ולתעד את סיבת ההיעדרות.",
        "2. לאשר או לתקן בדיעבד את השיבוץ של העובדים שלא תוכננו.",
        "3. אם התופעה חוזרת – לבחון את תהליך עדכון הסידור והחלפות המשמרות.",
        "",
        "(סיכום זה נוצר על ידי AI מדומה. הגדר מפתח API לקבלת ניתוח מלא.)",
    ]
    return "\n".join(lines)


def build_email(kind: str, row: pd.Series, manager: str, company: str, report_date: str):
    """Return (to, subject, body) – Hebrew draft for a 'missing' or 'unplanned' employee."""
    name = row.get("Name") or "עובד/ת יקר/ה"
    to = row.get("Email", "") or ""
    sign = f"בברכה,\n{manager}\n{company}"
    if kind == "missing":
        subject = f"אי התייצבות למשמרת בתאריך {report_date}"
        body = (
            f"שלום {name},\n\n"
            f"בבדיקת הנוכחות מיום {report_date} עולה כי היית משובץ/ת למשמרת אך לא נרשמה עבורך כניסה.\n"
            "נבקש שתחזור/י אלינו בהקדם עם הסבר להיעדרות (או אישור במידה ויש).\n"
            "אם מדובר בתקלה בשעון הנוכחות או בטעות בדיווח – אנא עדכן/י ונטפל בכך מיידית.\n\n"
            f"{sign}"
        )
    else:
        subject = f"כניסה למשמרת ללא שיבוץ מתוכנן – {report_date}"
        body = (
            f"שלום {name},\n\n"
            f"בבדיקת הנוכחות מיום {report_date} נרשמה כניסה שלך למשמרת אשר לא הופיעה בסידור העבודה המתוכנן.\n"
            "נבקש שתעדכן/י אותנו מי אישר את הגעתך (למשל החלפת משמרת או בקשת מנהל), כדי שנוכל לעדכן את הסידור.\n"
            "לתשומת ליבך: יש לתאם שינויים בסידור מראש מול המנהל הישיר.\n\n"
            f"{sign}"
        )
    return to, subject, body


# ----------------------------------------------------------------------------
# Helpers for the UI
# ----------------------------------------------------------------------------
def rtl_block(text: str):
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f'<div dir="rtl" style="text-align:right;line-height:1.7;padding:1rem;'
        f'border:1px solid rgba(128,128,128,.35);border-radius:8px">{safe}</div>',
        unsafe_allow_html=True,
    )


def sample_data():
    planned = pd.DataFrame({
        "Employee ID": ["101", "102", "103", "104", "105"],
        "Name": ["דנה כהן", "יוסי לוי", "מיכל ברק", "אבי מזרחי", "רונית שרון"],
        "Email": ["dana@example.com", "yossi@example.com", "michal@example.com", "avi@example.com", "ronit@example.com"],
        "Shift": ["08:00-16:00"] * 5,
    })
    actual = pd.DataFrame({
        "Employee ID": ["101", "103", "105", "106"],
        "Name": ["דנה כהן", "מיכל ברק", "רונית שרון", "תומר אלון"],
        "Clock-in": ["07:55", "08:10", "08:02", "09:30"],
    })
    return planned, actual


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="HR Attendance Comparison", page_icon="🗓️", layout="wide")
st.title("🗓️ HR Attendance Comparison")
st.caption("Compare planned shifts with actual clock-ins, get an AI executive summary and draft emails (Hebrew).")

with st.sidebar:
    st.header("Settings")
    report_date = st.date_input("Report date", value=date.today()).strftime("%d/%m/%Y")
    manager = st.text_input("Manager name", value="המנהל/ת")
    company = st.text_input("Company / department", value="מחלקת משאבי אנוש")
    use_sample = st.checkbox("Use sample data (demo)")
    ai_mode = "Real AI (API key found)" if (_get_secret("ANTHROPIC_API_KEY") or _get_secret("OPENAI_API_KEY")) else "Mock AI (no API key)"
    st.info(f"AI mode: **{ai_mode}**")

col1, col2 = st.columns(2)
with col1:
    planned_file = st.file_uploader("📋 Planned Shifts", type=["xlsx", "xls", "csv"], key="planned")
with col2:
    actual_file = st.file_uploader("⏱️ Actual Clock-ins", type=["xlsx", "xls", "csv"], key="actual")

if use_sample:
    planned_raw, actual_raw = sample_data()
elif planned_file and actual_file:
    try:
        planned_raw, actual_raw = read_any(planned_file), read_any(actual_file)
    except Exception as e:
        st.error(f"Could not read files: {e}")
        st.stop()
else:
    st.info("Upload both files (or tick “Use sample data” in the sidebar) to begin.")
    st.stop()

try:
    planned_df = prepare(planned_raw, "Planned Shifts")
    actual_df = prepare(actual_raw, "Actual Clock-ins")
except ValueError as e:
    st.error(str(e))
    st.stop()

matched, missing, unplanned, info = compare(planned_df, actual_df)

if not info["use_id"]:
    st.warning("Employee ID column missing in one of the files – matching by Name only.")
if info["dup_planned"] or info["dup_actual"]:
    st.warning(f"Duplicate employees ignored (planned: {info['dup_planned']}, actual: {info['dup_actual']}).")
if info["name_mismatch"]:
    st.warning(f"{info['name_mismatch']} matched employee(s) have the same ID but a different name – see ⚠️ in the Matched table.")

# KPIs
total_planned = len(matched) + len(missing)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Planned", total_planned)
k2.metric("✅ Matched", len(matched))
k3.metric("❌ Missing", len(missing))
k4.metric("❓ Unplanned", len(unplanned))

tab_tables, tab_ai, tab_mail = st.tabs(["📊 Tables", "🤖 Executive Summary", "✉️ Draft Emails"])

with tab_tables:
    st.subheader("✅ Matched – planned and arrived")
    st.dataframe(matched, use_container_width=True, hide_index=True)
    st.subheader("❌ Missing – planned but did not clock in")
    st.dataframe(missing, use_container_width=True, hide_index=True)
    st.subheader("❓ Unplanned – clocked in but not scheduled")
    st.dataframe(unplanned, use_container_width=True, hide_index=True)

with tab_ai:
    if st.button("Generate executive summary", type="primary"):
        with st.spinner("Analyzing discrepancies..."):
            st.session_state["summary"] = generate_executive_summary(matched, missing, unplanned, report_date)
    if "summary" in st.session_state:
        rtl_block(st.session_state["summary"])

with tab_mail:
    def render_emails(kind: str, df: pd.DataFrame):
        if df.empty:
            st.success("No employees in this category 🎉")
            return
        for i, row in df.iterrows():
            to, subject, body = build_email(kind, row, manager, company, report_date)
            label = row.get("Name") or row.get("Employee ID") or f"#{i + 1}"
            with st.expander(f"{label}" + (f"  –  {to}" if to else "")):
                st.text("To: " + (to or "(no email in file)"))
                st.text("Subject:")
                st.code(subject, language=None)
                st.text("Body:")
                st.code(body, language=None)  # built-in copy icon for the manager

    sub1, sub2 = st.tabs([f"❌ Missing ({len(missing)})", f"❓ Unplanned ({len(unplanned)})"])
    with sub1:
        render_emails("missing", missing)
    with sub2:
        render_emails("unplanned", unplanned)