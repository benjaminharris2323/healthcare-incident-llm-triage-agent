import os
import json
import csv
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_PATH = OUTPUT_DIR / "triage_log.csv"

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

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
# Tool 1: Rule-based classifier
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

# ----------------------------
# Tool 2: LLM classifier
# ----------------------------
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

# ----------------------------
# Tool 3: Escalation logic
# ----------------------------
def escalation_checker(category: str, severity: str) -> str:
    high_risk_categories = {"medication error", "fall", "infection concern"}

    if severity == "high":
        return "yes"

    if category in high_risk_categories:
        return "yes"

    return "no"

# ----------------------------
# Tool 4: Follow-up question logic
# ----------------------------
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

# ----------------------------
# Tool 5: Save results to CSV
# ----------------------------
def save_to_csv(result: dict):
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

    except Exception as e:
        print(f"\nWarning: Could not save result to CSV. Error: {e}")

# ----------------------------
# Agent logic
# ----------------------------
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
# CLI interface
# ----------------------------
if __name__ == "__main__":
    print("Healthcare Incident Triage Agent")
    print("--------------------------------")

    incident_text = input("Enter incident narrative:\n> ").strip()

    if not incident_text:
        print("No incident text entered. Exiting.")
    else:
        if needs_follow_up(incident_text):
            follow_up_question = generate_follow_up_question(incident_text)
            print("\nFollow-up Question:")
            print("--------------------------------")
            print(follow_up_question)

            follow_up_response = input("\nAdditional details:\n> ").strip()
            if follow_up_response:
                incident_text = incident_text + " " + follow_up_response

        result = triage_agent(incident_text)

        print("\nAgent Output")
        print("--------------------------------")
        print(f"Rule-based category: {result['rule_prediction']}")
        print(f"LLM category: {result['llm_category']}")
        print(f"Rule vs LLM agreement: {result['rule_llm_agreement']}")
        print(f"Severity: {result['severity']}")
        print(f"Summary: {result['summary']}")
        print(f"Recommended action: {result['recommended_action']}")
        print(f"Escalate: {result['escalate']}")
        print(f"Final triage decision: {result['final_decision']}")

        if result["error"]:
            print(f"Error note: {result['error']}")

        save_to_csv(result)
        print(f"\nResult saved to {LOG_PATH}")