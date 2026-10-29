# FAIB_LoanDetectionRAGsystem
Problem Statement: Your team has been hired by a mid-sized retail bank to prototype an AI-assisted loan system.
An applicant enters their financial details. A predictive model approves or rejects the
application. A RAG component reads the bank's lending policy, which your team writes, and
generates a clear, policy-grounded explanation of the decision

## Run the demo

The Streamlit app uses the model files in this repository and the illustrative policy in `lending_policy.md`.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The notebook maps class `1` to approved and class `0` to rejected. The app uses the notebook's F1-selected thresholds: `0.525` for logistic regression and `0.55` for random forest. Category-code mappings are not saved with the model artifacts, so the sample-case inputs use rounded training means for those fields. This is a demonstration only: the model expects `Gender` and an undocumented `Default_Risk` field, and the notebook selected thresholds on the test set. Do not use the output for real lending.

## Publish on Streamlit Community Cloud

Push the app, policy, requirements, runtime, and all three model files to a GitHub repository accessible to your Streamlit account. In Streamlit Community Cloud, create an app from that repository, select `app.py` as the entry point, and deploy. The model artifacts are loaded from the repository at startup; no API key or external service is required. Keep the repository private unless you intend to publish the model files as well as the app.

See [Project Progress Report](PROJECT_PROGRESS_REPORT.md) for the architecture, dataset and evaluation notes, guardrails, role assignments, and next steps.

The notebook's Kaggle setup cell reads `KAGGLE_API_TOKEN` from the environment. Revoke the previously exposed token and do not commit credentials.
