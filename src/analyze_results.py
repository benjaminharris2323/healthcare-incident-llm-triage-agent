from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "outputs" / "triage_log.csv"

if not LOG_PATH.exists():
    raise FileNotFoundError(f"Could not find log file: {LOG_PATH}")

df = pd.read_csv(LOG_PATH)

print("\nHealthcare Incident Triage Analytics")
print("-----------------------------------")
print(f"Total incidents logged: {len(df)}")

print("\nIncident distribution:")
print(df["llm_category"].value_counts(dropna=False))

print("\nSeverity distribution:")
print(df["severity"].value_counts(dropna=False))

print("\nEscalation distribution:")
print(df["escalate"].value_counts(dropna=False))

if "rule_llm_agreement" in df.columns:
    disagreement_rate = (df["rule_llm_agreement"] == False).mean() * 100
    print(f"\nRule vs LLM disagreement rate: {disagreement_rate:.2f}%")

print("\nCategory x Severity:")
print(pd.crosstab(df["llm_category"], df["severity"], dropna=False))

print("\nEscalation by category:")
esc_by_cat = pd.crosstab(df["llm_category"], df["escalate"], dropna=False)
print(esc_by_cat)

if "timestamp" in df.columns:
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        daily_counts = df.groupby(df["timestamp"].dt.date).size()
        print("\nIncidents by day:")
        print(daily_counts)
    except Exception as e:
        print(f"\nCould not parse timestamps: {e}")