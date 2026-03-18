# LLM-Powered Healthcare Incident Triage

A lightweight LLM pipeline that converts healthcare incident narratives into structured safety classifications and triage summaries.

## Overview

Healthcare organizations receive large volumes of safety incident reports written as narrative text. These reports often describe events such as medication errors, patient falls, documentation issues, and communication breakdowns.

Because these narratives are unstructured, categorizing and triaging incidents consistently can be time-consuming and difficult.

This project demonstrates a simple **LLM-powered triage pipeline** that converts healthcare incident narratives into structured safety classifications and summaries.

The system reads incident narratives from a dataset, uses a **large language model (LLM)** to classify each incident, and compares the results to a **rule-based baseline classifier**.

---

## Project Objective

The goal of this project is to build a lightweight pipeline that:

- Loads healthcare incident narratives from a dataset
- Classifies incidents into safety categories
- Generates structured summaries and recommended actions
- Compares LLM predictions with a rule-based baseline

---

## Pipeline Architecture

```
Incident Narratives (CSV)
        │
        ▼
Rule-Based Baseline Classifier
        │
        ▼
LLM Classification (GPT-4o-mini)
        │
        ▼
Structured Incident Triage Output
        │
        ▼
Prediction Dataset + Evaluation
```

---

## Dataset

The project uses a small synthetic dataset of healthcare incident narratives.

Example incidents include:

- Medication administration errors
- Patient falls
- IV line complications
- Documentation mistakes
- Communication delays

Example dataset:

| event_id | event_text |
|--------|-------------|
| 1 | Patient received duplicate dose of medication after shift handoff confusion |
| 2 | Patient slipped while walking to restroom and reported hip pain |

---

## Output Fields

Each incident is converted into structured fields:

| Field | Description |
|------|-------------|
| incident_category | Type of healthcare safety incident |
| severity | Estimated severity level |
| summary | Short summary of the incident |
| recommended_action | Suggested safety response |

Example output:

| event_id | incident_category | severity | summary |
|--------|------------------|---------|---------|
| 1 | medication error | medium | Duplicate medication dose due to shift handoff confusion |
| 2 | fall | medium | Patient slipped while walking to restroom |

All predictions are saved to:

`outputs/incident_predictions.csv`

---

## Baseline vs LLM Comparison

A simple **rule-based classifier** is implemented using keyword matching to provide a baseline comparison.

Example rules:

```python
if "medication" in text or "dose" in text:
    category = "medication error"

if "slip" in text or "fall" in text:
    category = "fall"

if "IV" in text:
    category = "line/tube issue"

if "chart" in text or "documentation" in text:
    category = "documentation error"

The pipeline compares the rule-based prediction to the LLM classification.

### Example Results

| event_id | rule_prediction | incident_category | rule_matches_llm | severity |
|---------|-----------------|------------------|------------------|----------|
| 1 | medication error | medication error | TRUE | medium |
| 2 | fall | fall | TRUE | medium |
| 3 | line/tube issue | line/tube issue | TRUE | medium |
| 4 | medication error | documentation error | FALSE | medium |
| 5 | communication issue | communication issue | TRUE | medium |

Example evaluation metric printed by the pipeline:

Rule vs LLM agreement: 80%

This comparison demonstrates how a large language model can provide more nuanced classification than simple keyword-based rules.

---

## Installation

Clone the repository and install the required dependencies.

```bash
pip install -r requirements.txt


Create a .env file in the project root directory and add your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

---

## Running the Pipeline

Run the incident triage pipeline from the project root directory:

```bash
python src/test_pipeline.py
```

The script will:

1. Load the incident dataset
2. Run the rule-based baseline classifier
3. Generate LLM predictions
4. Compare rule-based and LLM predictions
5. Export results to a CSV file

Example console output:

```
Dataset loaded successfully
Rule vs LLM agreement: 80.00%
Pipeline completed successfully.
Predictions saved to: outputs/incident_predictions.csv
```

---

## Project Structure

```
healthcare-incident-llm-agent/
│
├── data/
│   └── incident_reports_sample.csv
│
├── outputs/
│   └── incident_predictions.csv
│
├── src/
│   └── test_pipeline.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- OpenAI API
- GPT-4o-mini
- python-dotenv

---

## Future Improvements

Possible extensions for this project include:

- Using a larger real-world healthcare incident dataset
- Multi-label incident classification
- Retrieval-Augmented Generation (RAG) using hospital safety guidelines
- Severity risk scoring models
- Visualization dashboard for incident monitoring

---

## Author

Benjamin Harris

