import os
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# Paths and environment setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DATA_PATH = PROJECT_ROOT / "data" / "incident_reports_sample.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "incident_predictions.csv"

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

client = OpenAI(api_key=api_key)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully\n")
print(df.head())

# ----------------------------
# Prompt
# ----------------------------
SYSTEM_PROMPT = """
You are a healthcare safety analyst. Your task is to review healthcare incident narratives
and convert them into structured triage fields.

Use only the information provided in the narrative.
Do not diagnose patients.

Return valid JSON with the following fields:
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
"""

def analyze_incident(text: str) -> dict:
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
            "raw_response": content
        }

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
# Run pipeline
# ----------------------------
results = []

for _, row in df.iterrows():
    event_id = row["event_id"]
    event_text = row["event_text"]

    print("\n-----------------------------------")
    print(f"Event ID: {event_id}")
    print(f"Incident: {event_text}")

    rule_prediction = rule_based_classifier(event_text)
    result = analyze_incident(event_text)

    print(f"Rule-Based Prediction: {rule_prediction}")
    print("\nParsed Output:")
    print(result)

    results.append({
        "event_id": event_id,
        "event_text": event_text,
        "rule_prediction": rule_prediction,
        "incident_category": result.get("incident_category"),
        "rule_matches_llm": rule_prediction == result.get("incident_category"),
        "severity": result.get("severity"),
        "summary": result.get("summary"),
        "recommended_action": result.get("recommended_action"),
    })

# ----------------------------
# Save outputs
# ----------------------------
OUTPUT_DIR.mkdir(exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\n-----------------------------------")
print("Pipeline completed successfully.")
print(f"Predictions saved to: {OUTPUT_PATH}")
print("\nFinal output preview:")
print(results_df.head())

# ----------------------------
# Simple evaluation metric
# ----------------------------
accuracy = results_df["rule_matches_llm"].mean()
print(f"\nRule vs LLM agreement: {accuracy:.2%}")