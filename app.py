from pathlib import Path
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# ----------------------------
# Paths and environment
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_PATH = OUTPUT_DIR / "triage_log.csv"

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

# ----------------------------
# Prompts
# ----------------------------
SYSTEM_PROMPT = """
You are a healthcare safety triage assistant.

Your job is to review a healthcare incident narrative and return valid JSON with:
incident_category
severity
summary
recommended_action

Allowed incident_category values:
medication error
fall
line/tube issue
documentation error
infection concern
communication issue
other

Allowed severity values:
low
medium
high

Respond ONLY with valid JSON.
"""

FOLLOW_UP_PROMPT = """
You are a healthcare safety assistant.

A user provided an incident narrative that may be too vague.

Ask EXACTLY ONE specific follow-up question that is tailored to this incident.
Avoid listing all possible categories unless absolutely necessary.
Focus on the single most important missing detail needed for triage.

Requirements:
- Return only one question
- Keep it short and natural
- Do not explain your reasoning
- Do not provide multiple questions
- Return plain text only
"""

# ----------------------------
# Core functions
# ----------------------------
def rule_based_classifier(text: str) -> str:
    text = text.lower()

    if "dose" in text or "medication" in text:
        return "medication error"
    if "slip" in text or "fell" in text or "fall" in text:
        return "fall"
    if "iv" in text or "catheter" in text or "line" in text or "tube" in text:
        return "line/tube issue"
    if "chart" in text or "document" in text or "documentation" in text:
        return "documentation error"
    if "communication" in text or "handoff" in text:
        return "communication issue"

    return "other"


def llm_classifier(text: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Incident narrative:\n{text}"}
            ],
            temperature=0
        )

        content = response.choices[0].message.content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "incident_category": None,
                "severity": None,
                "summary": None,
                "recommended_action": None,
                "raw_response": content,
                "error": "LLM returned non-JSON output."
            }

    except Exception as e:
        return {
            "incident_category": None,
            "severity": None,
            "summary": None,
            "recommended_action": None,
            "raw_response": None,
            "error": f"LLM classification failed: {str(e)}"
        }


def escalation_checker(category: str, severity: str) -> str:
    high_risk_categories = {"medication error", "fall", "infection concern"}

    if severity == "high":
        return "yes"
    if category in high_risk_categories:
        return "yes"

    return "no"


def needs_follow_up(text: str) -> bool:
    text = text.lower()

    keywords = [
        "dose", "medication",
        "slip", "fell", "fall",
        "iv", "catheter", "line", "tube",
        "chart", "document", "documentation",
        "communication", "handoff",
        "infection", "swelling", "redness"
    ]

    return not any(keyword in text for keyword in keywords)


def generate_follow_up_question(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": FOLLOW_UP_PROMPT},
                {"role": "user", "content": f"Incident narrative:\n{text}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Can you provide one more specific detail about what happened after the treatment?"


def save_to_csv(result: dict) -> bool:
    OUTPUT_DIR.mkdir(exist_ok=True)
    file_exists = LOG_PATH.exists()

    try:
        with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "incident_text",
                    "rule_prediction",
                    "llm_category",
                    "rule_llm_agreement",
                    "severity",
                    "summary",
                    "recommended_action",
                    "escalate",
                    "final_decision",
                    "error"
                ])

            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                result.get("incident_text"),
                result.get("rule_prediction"),
                result.get("llm_category"),
                result.get("rule_llm_agreement"),
                result.get("severity"),
                result.get("summary"),
                result.get("recommended_action"),
                result.get("escalate"),
                result.get("final_decision"),
                result.get("error")
            ])
        return True
    except Exception as e:
        st.warning(f"Could not save result to CSV: {e}")
        return False


def triage_agent(text: str) -> dict:
    rule_prediction = rule_based_classifier(text)
    llm_result = llm_classifier(text)

    llm_category = llm_result.get("incident_category")
    severity = llm_result.get("severity")
    summary = llm_result.get("summary")
    recommended_action = llm_result.get("recommended_action")
    error = llm_result.get("error")

    agreement = (rule_prediction == llm_category) if llm_category is not None else False
    escalate = escalation_checker(llm_category, severity) if llm_category and severity else "no"

    if llm_category is None or severity is None:
        final_decision = "Manual review required"
    elif escalate == "yes":
        final_decision = "Escalate for clinical/safety review"
    else:
        final_decision = "Standard review workflow"

    return {
        "incident_text": text,
        "rule_prediction": rule_prediction,
        "llm_category": llm_category,
        "rule_llm_agreement": agreement,
        "severity": severity,
        "summary": summary,
        "recommended_action": recommended_action,
        "escalate": escalate,
        "final_decision": final_decision,
        "error": error
    }

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Healthcare Incident Triage Agent", layout="wide")
st.title("Healthcare Incident Triage Agent")

tab1, tab2 = st.tabs(["Triage Agent", "Analytics Dashboard"])

with tab1:
    st.subheader("Submit an Incident")
    incident_text = st.text_area("Incident narrative", height=140)

    if "follow_up_question" not in st.session_state:
        st.session_state.follow_up_question = None
    if "original_incident" not in st.session_state:
        st.session_state.original_incident = None

    if st.button("Analyze Incident"):
        if not incident_text.strip():
            st.warning("Please enter an incident narrative.")
        else:
            if needs_follow_up(incident_text):
                st.session_state.follow_up_question = generate_follow_up_question(incident_text)
                st.session_state.original_incident = incident_text
            else:
                result = triage_agent(incident_text)
                save_to_csv(result)
                st.session_state.result = result
                st.session_state.follow_up_question = None

    if st.session_state.follow_up_question:
        st.info(st.session_state.follow_up_question)
        follow_up_response = st.text_input("Additional details")

        if st.button("Submit Additional Details"):
            combined_text = f"{st.session_state.original_incident} {follow_up_response}".strip()
            result = triage_agent(combined_text)
            save_to_csv(result)
            st.session_state.result = result
            st.session_state.follow_up_question = None

    if "result" in st.session_state:
        result = st.session_state.result

        c1, c2, c3 = st.columns(3)
        c1.metric("LLM Category", result["llm_category"] or "N/A")
        c2.metric("Severity", result["severity"] or "N/A")
        c3.metric("Escalate", result["escalate"] or "N/A")

        st.write("**Rule-based category:**", result["rule_prediction"])
        st.write("**Rule vs LLM agreement:**", result["rule_llm_agreement"])
        st.write("**Summary:**", result["summary"])
        st.write("**Recommended action:**", result["recommended_action"])
        st.write("**Final triage decision:**", result["final_decision"])

        if result["error"]:
            st.warning(result["error"])

with tab2:
    st.subheader("Logged Incident Analytics")

    if not LOG_PATH.exists():
        st.info("No triage log found yet. Run the agent first.")
    else:
        df = pd.read_csv(LOG_PATH)

        st.write(f"Total incidents logged: {len(df)}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Incidents", len(df))
        c2.metric("Escalation Rate", f"{(df['escalate'].eq('yes').mean() * 100):.1f}%")
        c3.metric("Disagreement Rate", f"{((df['rule_llm_agreement'] == False).mean() * 100):.1f}%")

        st.write("### Incident Distribution")
        category_counts = df["llm_category"].value_counts(dropna=False)
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        category_counts.plot(kind="bar", ax=ax1)
        ax1.set_ylabel("Count")
        ax1.set_title("Incidents by Category")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig1)

        st.write("### Severity Distribution")
        severity_counts = df["severity"].value_counts(dropna=False)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        severity_counts.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Count")
        ax2.set_title("Incidents by Severity")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig2)

        st.write("### Category x Severity")
        cross_tab = pd.crosstab(df["llm_category"], df["severity"], dropna=False)
        st.dataframe(cross_tab)

        st.write("### Escalation by Category")
        escalation_tab = pd.crosstab(df["llm_category"], df["escalate"], dropna=False)
        st.dataframe(escalation_tab)

        st.write("### Logged Records")
        st.dataframe(df)