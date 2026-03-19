# AI Healthcare Incident Triage Agent

An AI-powered healthcare incident triage system that combines rule-based logic, large language model (LLM) classification, escalation decision-making, follow-up questioning, persistent logging, analytics, and a Streamlit user interface.

---

## Overview

Healthcare organizations receive large volumes of safety incident reports written as unstructured narrative text. These reports often describe events such as medication errors, patient falls, documentation issues, communication breakdowns, and possible infections.

This project builds an AI triage agent that transforms those narratives into structured outputs and decision workflows.

Unlike a simple LLM pipeline, this system introduces:

- Rule-based baseline classification
- LLM-powered classification using GPT-4o-mini
- Escalation decision logic
- Follow-up question handling for ambiguous cases
- Persistent logging of triage results
- Analytics for logged incident data
- Streamlit UI for interactive use

---

## Key Features

- Hybrid classification (rule-based + LLM)
- Escalation decision logic
- Follow-up question generation for vague inputs
- CSV logging of all triage results
- Interactive Streamlit UI
- Built-in analytics dashboard

---

## Agent Workflow

User Input (Incident Narrative)
            │
            ▼
Rule-Based Classifier
            │
            ▼
LLM Classifier (GPT-4o-mini)
            │
            ▼
Follow-Up Question (if needed)
            │
            ▼
Escalation Logic
            │
            ▼
Final Decision + Structured Output
            │
            ▼
Log Results (CSV) + Display in UI

---

## Example Output

Example Input:
Patient slipped while getting out of bed and reported hip pain.

Example Output:
- Rule-Based Category: fall
- LLM Category: fall
- Severity: medium
- Escalate: yes
- Final Decision: Escalate for clinical/safety review
- Recommended Action: Assess patient for injury and monitor condition

---

## Installation

Clone the repository and install dependencies:

pip install -r requirements.txt

Create a `.env` file in the project root directory:

OPENAI_API_KEY=your_api_key_here

---

## Running the Application

Launch the Streamlit app:

python -m streamlit run app.py

---

## Using the Application

1. Enter a healthcare incident narrative
2. Click "Analyze Incident"
3. The agent will:
   - Run rule-based classification
   - Run LLM classification
   - Apply escalation logic
4. If input is vague:
   - A follow-up question will be asked
5. View results:
   - Category
   - Severity
   - Escalation decision
   - Summary
   - Recommended action
   - Final decision

---

## Analytics Dashboard

Navigate to the "Analytics Dashboard" tab in the UI.

### Metrics

- Total incidents logged
- Escalation rate (%)
- Rule vs LLM disagreement rate (%)

### Visualizations

- Incident distribution by category
- Severity distribution
- Category vs severity breakdown
- Escalation by category

### Data Table

A full table of all logged incidents is displayed within the dashboard.

---

## Logged Data

All outputs are saved to:

outputs/triage_log.csv

Each record includes:

- Timestamp
- Incident text
- Rule-based prediction
- LLM category
- Severity
- Summary
- Recommended action
- Escalation decision
- Final decision

---

## Data

The data/ directory can be used for:

- Sample incident narratives
- Batch testing inputs
- Future dataset expansion

---

## Project Structure

healthcare-incident-llm-triage-agent/
├── app.py
├── data/
│   └── sample_incidents.csv
├── outputs/
│   └── triage_log.csv
├── src/
│   ├── triage_agent.py
│   └── analyze_results.py
├── .gitignore
├── README.md
└── requirements.txt

Note: The .env file is required locally but excluded via .gitignore.

---

## Technologies Used

- Python
- OpenAI API
- GPT-4o-mini
- Streamlit
- Pandas
- Matplotlib
- python-dotenv

---

## Future Improvements

- Add confidence scoring
- Implement Retrieval-Augmented Generation (RAG)
- Improve escalation logic with historical data
- Add authentication and roles
- Deploy as a web application

---

## Author

Benjamin Harris