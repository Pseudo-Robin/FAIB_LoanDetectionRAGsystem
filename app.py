from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "lending_policy.md"
MODEL_FILES = {
    "Logistic regression": ROOT / "logistic_regression_model.pkl",
    "Random forest": ROOT / "random_forest_model.pkl",
}
SCALER_PATH = ROOT / "scaler.pkl"
MODEL_THRESHOLDS = {
    "Logistic regression": 0.525,
    "Random forest": 0.55,
}
DEMO_OVERRIDES = {
    "Logistic regression": {
        "Approval example": {
            "Age": 22,
            "Dependents": 3,
            "Annual_Income": 113700,
            "Monthly_Expenses": 6200,
            "Credit_Score": 843,
            "Existing_Loans": 3,
            "Total_Existing_Loan_Amount": 26400,
            "Outstanding_Debt": 41600,
            "Loan_Amount_Requested": 42100,
            "Loan_Term": 164,
            "Interest_Rate": 8.4,
            "Transaction_Frequency": 8,
            "Default_Risk": 0,
        },
        "Rejection example": {
            "Age": 30,
            "Dependents": 4,
            "Annual_Income": 24000,
            "Monthly_Expenses": 1200,
            "Credit_Score": 306,
            "Existing_Loans": 4,
            "Total_Existing_Loan_Amount": 85400,
            "Outstanding_Debt": 68100,
            "Loan_Amount_Requested": 7000,
            "Loan_Term": 18,
            "Interest_Rate": 18.2,
            "Transaction_Frequency": 0,
            "Default_Risk": 1,
        },
        "Borderline example": {
            "Age": 53,
            "Dependents": 1,
            "Annual_Income": 106800,
            "Monthly_Expenses": 7400,
            "Credit_Score": 642,
            "Existing_Loans": 0,
            "Total_Existing_Loan_Amount": 72400,
            "Outstanding_Debt": 62400,
            "Loan_Amount_Requested": 14600,
            "Loan_Term": 68,
            "Interest_Rate": 19,
            "Transaction_Frequency": 9,
            "Default_Risk": 1,
        },
    },
    "Random forest": {
        "Approval example": {
            "Age": 47,
            "Dependents": 5,
            "Annual_Income": 74500,
            "Monthly_Expenses": 3400,
            "Credit_Score": 765,
            "Existing_Loans": 4,
            "Total_Existing_Loan_Amount": 26100,
            "Outstanding_Debt": 15500,
            "Loan_Amount_Requested": 26100,
            "Loan_Term": 153,
            "Interest_Rate": 8.1,
            "Transaction_Frequency": 0,
            "Default_Risk": 1,
        },
        "Rejection example": {
            "Age": 58,
            "Dependents": 3,
            "Annual_Income": 42000,
            "Monthly_Expenses": 4800,
            "Credit_Score": 529,
            "Existing_Loans": 5,
            "Total_Existing_Loan_Amount": 16200,
            "Outstanding_Debt": 12000,
            "Loan_Amount_Requested": 11100,
            "Loan_Term": 75,
            "Interest_Rate": 14,
            "Transaction_Frequency": 16,
            "Default_Risk": 0,
        },
        "Borderline example": {
            "Age": 65,
            "Dependents": 2,
            "Annual_Income": 148900,
            "Monthly_Expenses": 2500,
            "Credit_Score": 693,
            "Existing_Loans": 4,
            "Total_Existing_Loan_Amount": 72200,
            "Outstanding_Debt": 59900,
            "Loan_Amount_Requested": 45600,
            "Loan_Term": 217,
            "Interest_Rate": 17.1,
            "Transaction_Frequency": 4,
            "Default_Risk": 1,
        },
    },
}
DEMO_SCENARIOS = [
    "Approval example",
    "Rejection example",
    "Borderline example",
    "Custom inputs",
]

FIELD_GROUPS = {
    "Applicant profile": [
        "Gender",
        "Age",
        "Marital_Status",
        "Dependents",
        "Education",
        "Employment_Status",
        "Occupation_Type",
        "Residential_Status",
        "City/Town",
    ],
    "Financial position": [
        "Annual_Income",
        "Monthly_Expenses",
        "Credit_Score",
        "Existing_Loans",
        "Total_Existing_Loan_Amount",
        "Outstanding_Debt",
        "Loan_History",
    ],
    "Requested facility": [
        "Loan_Amount_Requested",
        "Loan_Term",
        "Loan_Purpose",
        "Interest_Rate",
        "Loan_Type",
        "Co-Applicant",
        "Bank_Account_History",
        "Transaction_Frequency",
        "Default_Risk",
    ],
}

FIELD_LABELS = {
    "Gender": "Gender code",
    "Age": "Age (years)",
    "Marital_Status": "Marital status code",
    "Dependents": "Dependents",
    "Education": "Education code",
    "Employment_Status": "Employment status code",
    "Occupation_Type": "Occupation type code",
    "Residential_Status": "Residential status code",
    "City/Town": "City or town code",
    "Annual_Income": "Annual income",
    "Monthly_Expenses": "Monthly expenses",
    "Credit_Score": "Credit score",
    "Existing_Loans": "Number of existing loans",
    "Total_Existing_Loan_Amount": "Total existing loan amount",
    "Outstanding_Debt": "Outstanding debt",
    "Loan_History": "Loan history code",
    "Loan_Amount_Requested": "Requested loan amount",
    "Loan_Term": "Loan term (months)",
    "Loan_Purpose": "Loan purpose code",
    "Interest_Rate": "Interest rate (%)",
    "Loan_Type": "Loan type code",
    "Co-Applicant": "Co-applicant code",
    "Bank_Account_History": "Bank account history code",
    "Transaction_Frequency": "Monthly transaction frequency",
    "Default_Risk": "Default risk input",
}

CATEGORICAL_FIELDS = {
    "Gender",
    "Marital_Status",
    "Education",
    "Employment_Status",
    "Occupation_Type",
    "Residential_Status",
    "City/Town",
    "Loan_History",
    "Loan_Purpose",
    "Loan_Type",
    "Co-Applicant",
    "Bank_Account_History",
    "Default_Risk",
}
INTEGER_FIELDS = {
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "Loan_Term",
    "Transaction_Frequency",
}

FIELD_HELP = {
    "Gender": "This trained artifact expects a gender code. Its use is a fairness and compliance concern; this demo must not be used for lending decisions.",
    "Default_Risk": "The training definition and provenance of this field were not provided. Do not treat it as a verified risk measure.",
}


@st.cache_resource
def load_assets():
    scaler = joblib.load(SCALER_PATH)
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    sections = []
    for section_id, title, body in re.findall(
        r"(?m)^### (P\d+) - ([^\n]+)\n(.*?)(?=^### P\d+ - |\Z)",
        policy_text,
        re.DOTALL,
    ):
        cleaned_body = body.strip()
        if cleaned_body:
            sections.append(
                {"id": section_id, "title": title.strip(), "body": cleaned_body}
            )
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    policy_matrix = vectorizer.fit_transform(
        [f"{section['title']} {section['body']}" for section in sections]
    )
    return scaler, models, sections, vectorizer, policy_matrix


def retrieve_policy(question, sections, vectorizer, policy_matrix, limit=3):
    query = question.strip()
    if not query:
        return []
    scores = cosine_similarity(vectorizer.transform([query]), policy_matrix).ravel()
    ranked = scores.argsort()[::-1][:limit]
    return [
        (sections[index], float(scores[index]))
        for index in ranked
        if scores[index] > 0
    ]


def policy_answer(question, sections, vectorizer, policy_matrix):
    matches = retrieve_policy(question, sections, vectorizer, policy_matrix)
    if not matches:
        return (
            "The sample policy does not contain a clear match for that question. "
            "Please consult a qualified human reviewer.",
            [],
        )
    citations = [
        f"[{section['id']}] {section['title']}"
        for section, _score in matches
    ]
    answer = (
        "The most relevant sample-policy passages are "
        + ", ".join(citations)
        + ". Review the cited wording below; this demo does not infer rules beyond it."
    )
    return answer, matches


def build_demo_values(model_name, scenario, scaler, features):
    values = dict(zip(features, scaler.mean_))
    for feature in CATEGORICAL_FIELDS | INTEGER_FIELDS:
        values[feature] = float(round(values[feature]))
    values.update(DEMO_OVERRIDES.get(model_name, {}).get(scenario, {}))
    return values


def update_demo_inputs():
    model_name = st.session_state.get("model_name", "Logistic regression")
    scenario = st.session_state.get("demo_scenario", "Approval example")
    st.session_state.pop("last_assessment", None)
    if scenario == "Custom inputs":
        return
    values = build_demo_values(model_name, scenario, scaler, features)
    for feature, value in values.items():
        st.session_state[f"input_{feature}"] = float(value)


def predict_application(values, model_name, scaler, models, features):
    application = pd.DataFrame([values], columns=features)
    model = models[model_name]
    model_input = (
        scaler.transform(application)
        if model_name == "Logistic regression"
        else application
    )
    probabilities = model.predict_proba(model_input)[0]
    approval_index = int(np.flatnonzero(model.classes_ == 1)[0])
    approval_probability = float(probabilities[approval_index])
    threshold = MODEL_THRESHOLDS[model_name]
    predicted_class = 1 if approval_probability >= threshold else 0
    return predicted_class, probabilities, model.classes_, approval_probability, threshold


def build_decision_explanation(
    model_name, predicted_class, approval_probability, threshold, sections, vectorizer, matrix
):
    margin = abs(approval_probability - threshold)
    is_borderline = margin <= 0.03
    decision = "approved" if predicted_class == 1 else "rejected"
    query = (
        "borderline model score manual human review credit history repayment capacity"
        if is_borderline
        else "verified income repayment capacity existing debt credit history loan review"
    )
    matches = retrieve_policy(query, sections, vectorizer, matrix, limit=3)
    policy_by_id = {section["id"]: section for section in sections}
    cited_sections = [policy_by_id["P01"]]
    for section, _score in matches:
        if section not in cited_sections:
            cited_sections.append(section)
    if is_borderline:
        explanation = (
            f"This synthetic case is borderline: the model's approved-class score is "
            f"{approval_probability:.1%}, {margin:.1%} from its {threshold:.1%} cutoff. "
            "Treat this as a manual-review case, not an automatic outcome. The sample "
            "policy says the model is only an initial screen and requires human review."
        )
    else:
        explanation = (
            f"The {model_name} model classifies this synthetic application as {decision} "
            f"(class {predicted_class}); its approved-class score is "
            f"{approval_probability:.1%} against a {threshold:.1%} demonstration cutoff. "
            "That score is not a policy rule or a final lending decision. The sample "
            "policy requires a reviewer to verify the application and document the outcome."
        )
    return explanation, cited_sections, is_borderline


def portfolio_projection(
    true_approved,
    false_approved,
    true_rejected,
    false_rejected,
    contribution_per_good_approval,
    loss_per_false_approval,
    contribution_lost_per_false_rejection,
    review_share,
    review_cost,
    inference_cost,
):
    cohort_size = true_approved + false_approved + true_rejected + false_rejected
    reviewed = round(cohort_size * review_share)
    default_proxy_loss = false_approved * loss_per_false_approval
    missed_contribution = false_rejected * contribution_lost_per_false_rejection
    good_loan_contribution = true_approved * contribution_per_good_approval
    human_review_cost = reviewed * review_cost
    model_inference_cost = cohort_size * inference_cost
    net_value = (
        good_loan_contribution
        - default_proxy_loss
        - missed_contribution
        - human_review_cost
        - model_inference_cost
    )
    return {
        "cohort_size": cohort_size,
        "reviewed": reviewed,
        "good_loan_contribution": good_loan_contribution,
        "false_approval_loss": default_proxy_loss,
        "false_rejection_loss": missed_contribution,
        "human_review_cost": human_review_cost,
        "inference_cost": model_inference_cost,
        "net_value": net_value,
    }


def format_usd(amount):
    if amount < 0:
        return f"-${abs(amount):,.0f}"
    return f"${amount:,.0f}"


st.set_page_config(
    page_title="The Lending Desk",
    page_icon="✿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
    :root {
        --ink: #171717;
        --muted: #6b6b68;
        --paper: #fff;
        --line: #e4e6dc;
        --leaf: #b8e34b;
        --leaf-pale: #f2f8d9;
        --coral: #f4775d;
        --aqua: #55cdb7;
        --sky-pale: #e8f5f2;
    }
    .stApp { background: var(--paper); color: var(--ink); font-family: 'DM Sans', sans-serif; }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 3.1rem !important; }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #171717 !important;
    }
    h1, h2 { font-family: 'Fraunces', Georgia, serif !important; font-weight: 500 !important; }
    h1 { font-size: 2.65rem !important; line-height: 1.05 !important; margin: .25rem 0 .35rem !important; }
    h2 { font-size: 1.85rem !important; }
    h3 { font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important; font-weight: 700 !important; }
    p, label, input, button { color: var(--ink); }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stRadio"] > div { gap: .5rem; flex-wrap: wrap; }
    [data-testid="stRadio"] label { border: 1px solid var(--line); border-radius: 999px; padding: .42rem .95rem; background: #fff; transition: transform .16s ease, border-color .16s ease, background .16s ease; }
    [data-testid="stRadio"] label:hover { border-color: var(--coral); transform: translateY(-2px); }
    [data-testid="stRadio"] label:has(input:checked) { border-color: var(--ink); background: var(--leaf); color: var(--ink); box-shadow: 3px 3px 0 var(--ink); }
    [data-testid="stRadio"] label:has(input:checked) p { color: var(--ink); font-weight: 700; }
    [data-testid="stMetric"] { background: var(--leaf-pale); border: 1px solid #e3edbd; border-radius: 10px 10px 10px 2px; padding: .85rem 1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stForm"] { border: 0; padding: 0; }
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 6px; background: #fff; }
    [data-testid="stAlert"] { border-radius: 6px; }
    [data-testid="stProgressBar"] > div > div { background: var(--aqua); }
    .topline { display:flex; justify-content:space-between; align-items:center; position:relative; isolation:isolate; overflow:hidden; min-height:62px; border:1px solid var(--ink); border-radius:12px 12px 3px 12px; padding:.75rem 1rem; margin:0 0 .45rem; box-shadow:4px 4px 0 var(--ink); background-color:#fbfced !important; background-image:repeating-linear-gradient(135deg, transparent 0 16px, rgba(184,227,75,.28) 16px 21px) !important; animation: arrive .45s ease-out both; }
    .topline::after { content:"✳"; position:absolute; z-index:-1; right:42%; top:-1.25rem; color:rgba(244,119,93,.25); font-size:5.5rem; transform:rotate(12deg); }
    .wordmark { font-weight:700; font-size:.88rem; }
    .wordmark span { display:inline-grid; place-items:center; width:1.65rem; height:1.65rem; color:var(--ink); background:var(--coral); border:1px solid var(--ink); border-radius:50%; margin-right:.35rem; transform:rotate(-9deg); }
    .top-note { color:var(--ink); font-size:.78rem; font-weight:600; background:var(--aqua); border:1px solid var(--ink); border-radius:999px; padding:.3rem .7rem; }
    .brand-side { display:flex; align-items:center; gap:.7rem; }
    .mini-chart { display:flex; align-items:flex-end; gap:3px; height:24px; padding:2px 3px; border-bottom:1px solid var(--ink); transform:rotate(-4deg); }
    .mini-chart i { display:block; width:5px; border:1px solid var(--ink); border-bottom:0; border-radius:3px 3px 0 0; }
    .mini-chart i:nth-child(1) { height:9px; background:var(--coral); }
    .mini-chart i:nth-child(2) { height:15px; background:var(--aqua); }
    .mini-chart i:nth-child(3) { height:20px; background:var(--leaf); }
    .accent-ribbon { display:flex; height:5px; gap:4px; margin:0 4px 1.5rem; }
    .accent-ribbon span { height:100%; border-radius:0 0 4px 4px; }
    .accent-ribbon span:nth-child(1) { width:46%; background:var(--coral); }
    .accent-ribbon span:nth-child(2) { width:22%; background:var(--aqua); }
    .accent-ribbon span:nth-child(3) { width:12%; background:var(--leaf); }
    .kicker { color:#b74e39; font-size:.72rem; font-weight:700; text-transform:uppercase; }
    .lede { color:var(--muted); font-size:.95rem; }
    .risk-note { font-size:.82rem; color:var(--ink); border:1px solid var(--ink); border-left:5px solid var(--coral); border-radius:3px 10px 10px 3px; padding:.65rem .8rem; background:var(--sky-pale); }
    .result-label { color:#b74e39; font-size:.75rem; font-weight:700; text-transform:uppercase; }
    .decision-card { position:relative; overflow:hidden; margin-top:.5rem; padding:1.15rem; border:1.5px solid var(--ink); border-radius:14px 14px 4px 14px; background:#fbfced; box-shadow:4px 4px 0 var(--ink); }
    .decision-card.approved { border-top:6px solid #78b93e; }
    .decision-card.rejected { border-top:6px solid var(--coral); }
    .decision-card.borderline { border-top:6px solid #e5b638; background:#fff9e8; }
    .decision-card::after { content:"✳"; position:absolute; right:-.35rem; top:-1.6rem; color:rgba(244,119,93,.2); font-size:5.2rem; transform:rotate(14deg); pointer-events:none; }
    .decision-top { position:relative; z-index:1; display:flex; justify-content:space-between; align-items:flex-start; gap:.5rem; }
    .decision-title { margin:.25rem 0 .1rem; color:var(--ink); font-family:'Fraunces', Georgia, serif; font-size:1.75rem; line-height:1.05; }
    .decision-badge { border:1px solid var(--ink); border-radius:999px; padding:.25rem .55rem; background:var(--aqua); color:var(--ink); font-size:.62rem; font-weight:700; white-space:nowrap; }
    .borderline .decision-badge { background:#f6d773; }
    .score-row { display:flex; align-items:baseline; gap:.5rem; margin-top:1.1rem; }
    .score-value { color:var(--ink); font-family:'Fraunces', Georgia, serif; font-size:2.6rem; line-height:1; }
    .score-caption { color:var(--muted); font-size:.78rem; }
    .score-track { position:relative; height:12px; margin:.75rem 0 .35rem; border:1px solid var(--ink); border-radius:999px; background:#fff; }
    .score-fill { height:100%; border-radius:inherit; background:var(--aqua); }
    .rejected .score-fill { background:var(--coral); }
    .borderline .score-fill { background:#f0c64c; }
    .score-cutoff { position:absolute; top:-5px; width:3px; height:20px; border:1px solid var(--ink); background:var(--ink); }
    .cutoff-caption { display:flex; justify-content:space-between; color:var(--muted); font-size:.7rem; }
    .decision-explanation { margin-top:.9rem; padding-top:.7rem; border-top:1px dashed #c9c9b8; color:var(--ink); font-size:.86rem; line-height:1.5; }
    .decision-foot { margin-top:.8rem; color:var(--muted); font-size:.7rem; }
    .policy-source { border-left: 4px solid var(--aqua); background:#f7fbfa; border-radius:0 8px 8px 0; padding: .65rem .85rem; margin: .8rem 0 1rem; }
    .tiny { color:var(--muted); font-size:.78rem; }
    .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button { border-radius:999px 999px 999px 4px; background:var(--leaf) !important; border:1px solid var(--ink) !important; color:#111 !important; box-shadow:3px 3px 0 var(--ink); transition:transform .16s ease, box-shadow .16s ease; }
    .stButton button[kind="primary"] *, [data-testid="stFormSubmitButton"] button * { color:#111 !important; }
    .stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover { background:#c9ef63 !important; transform:translate(-1px,-1px); box-shadow:5px 5px 0 var(--ink); }
    div[data-testid="stNumberInput"] input:focus, div[data-testid="stTextInput"] input:focus { border-color:#b74e39 !important; box-shadow:0 0 0 2px rgba(244,119,93,.2) !important; }
    div[data-testid="stSelectbox"] > div > div { border-radius:8px; }
    @keyframes arrive { from { opacity:0; transform:translateY(-5px); } to { opacity:1; transform:translateY(0); } }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; } }
    @media (max-width: 700px) { [data-testid="stMainBlockContainer"] { padding-top:2rem !important; } h1 { font-size:2.1rem !important; } .top-note { display:none; } }
    </style>
    """,
    unsafe_allow_html=True,
)

scaler, models, policy_sections, policy_vectorizer, policy_matrix = load_assets()
features = list(scaler.feature_names_in_)

st.markdown(
    """
    <div class="topline">
      <div class="wordmark"><span>✿</span> THE LENDING DESK</div>
            <div class="brand-side">
                <div class="mini-chart" aria-hidden="true"><i></i><i></i><i></i></div>
                <div class="top-note">A tiny prototype for thoughtful decisions</div>
            </div>
    </div>
        <div class="accent-ribbon" aria-hidden="true"><span></span><span></span><span></span></div>
    """,
    unsafe_allow_html=True,
)
header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown('<div class="kicker">FAIB · Loan review studio</div>', unsafe_allow_html=True)
    st.title("A little more clarity.")
    st.markdown('<div class="lede">Explore a model result, then check it against the policy.</div>', unsafe_allow_html=True)
with header_right:
    st.markdown('<div class="risk-note">Demo data only<br>Never a real lending decision</div>', unsafe_allow_html=True)

main_view = st.radio(
    "Choose a view",
    [
        "Review an application",
        "Ask the policy",
        "Read the policy",
        "Business case",
        "Prompt log",
        "Risks & returns",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_view",
)
st.divider()

if main_view == "Review an application":
    model_col, case_col = st.columns([1, 1])
    model_name = model_col.selectbox(
        "Model",
        list(models),
        key="model_name",
        on_change=update_demo_inputs,
    )
    scenario = case_col.selectbox(
        "Starting point",
        DEMO_SCENARIOS,
        key="demo_scenario",
        on_change=update_demo_inputs,
    )

    initial_values = build_demo_values(model_name, scenario, scaler, features)
    for feature, value in initial_values.items():
        input_key = f"input_{feature}"
        if input_key not in st.session_state:
            st.session_state[input_key] = float(value)

    form_column, outcome_column = st.columns([1.25, 0.75], gap="large")
    with form_column:
        st.markdown("### Applicant details")
        st.caption(
            f"{scenario} · category codes use training means · change any field before assessment"
            if scenario != "Custom inputs"
            else "Enter numeric values using the model's original units and category codes."
        )
        with st.form("application_form"):
            for group_name, group_features in FIELD_GROUPS.items():
                st.markdown(f"#### {group_name}")
                columns = st.columns(2, gap="medium")
                group_index = 0
                for feature in group_features:
                    if feature not in features:
                        continue
                    is_category = feature in CATEGORICAL_FIELDS
                    help_text = FIELD_HELP.get(
                        feature,
                        "Use the same numeric encoding and units used during model training.",
                    )
                    if feature in INTEGER_FIELDS or is_category:
                        step, number_format = 1.0, "%.0f"
                    elif feature == "Interest_Rate":
                        step, number_format = 0.1, "%.1f"
                    else:
                        step, number_format = 100.0, "%.0f"
                    with columns[group_index % 2]:
                        st.number_input(
                            FIELD_LABELS[feature],
                            min_value=0.0,
                            step=step,
                            format=number_format,
                            help=help_text,
                            key=f"input_{feature}",
                        )
                    group_index += 1
            submitted = st.form_submit_button(
                "Review this example  →", type="primary", use_container_width=True
            )

        if submitted:
            values = {
                feature: st.session_state[f"input_{feature}"]
                for feature in features
            }
            (
                predicted_class,
                probabilities,
                classes,
                approval_probability,
                threshold,
            ) = predict_application(values, model_name, scaler, models, features)
            explanation, cited_sections, is_borderline = build_decision_explanation(
                model_name,
                predicted_class,
                approval_probability,
                threshold,
                policy_sections,
                policy_vectorizer,
                policy_matrix,
            )
            st.session_state["last_assessment"] = {
                "model": model_name,
                "scenario": scenario,
                "decision": "Approved" if predicted_class == 1 else "Rejected",
                "approval_probability": approval_probability,
                "threshold": threshold,
                "classes": classes,
                "probabilities": probabilities,
                "explanation": explanation,
                "sections": cited_sections,
                "borderline": is_borderline,
            }

    with outcome_column:
        st.markdown("### Decision note")
        assessment = st.session_state.get("last_assessment")
        if assessment:
            score = assessment["approval_probability"]
            cutoff = assessment["threshold"]
            distance = abs(score - cutoff)
            distance_text = (
                f"{distance * 100:.1f} percentage points "
                f"{'above' if score >= cutoff else 'below'} the demo cutoff."
            )
            card_state = "borderline" if assessment["borderline"] else (
                "approved" if assessment["decision"] == "Approved" else "rejected"
            )
            decision_title = "Near the cutoff" if assessment["borderline"] else assessment["decision"]
            badge = "HUMAN REVIEW" if assessment["borderline"] else "MODEL SUGGESTION"
            if assessment["borderline"]:
                explanation = (
                    f"The model leans {assessment['decision'].lower()}, but the score is "
                    f"only {distance * 100:.1f} points from the cutoff. A person should "
                    "verify the application and policy before any decision."
                )
            else:
                explanation = (
                    f"{distance_text} This demo score is not a final lending decision; "
                    "verify the application and policy before acting."
                )
            st.markdown(
                f'<div class="decision-card {card_state}">'
                f'<div class="decision-top"><div>'
                f'<div class="result-label">{assessment["scenario"]} · {assessment["model"]}</div>'
                f'<div class="decision-title">{decision_title}</div>'
                f'</div><span class="decision-badge">{badge}</span></div>'
                f'<div class="score-row"><span class="score-value">{score:.1%}</span>'
                f'<span class="score-caption">approved-class score</span></div>'
                f'<div class="score-track"><div class="score-fill" style="width:{score * 100:.1f}%"></div>'
                f'<span class="score-cutoff" style="left:calc({cutoff * 100:.1f}% - 2px)"></span></div>'
                f'<div class="cutoff-caption"><span>0%</span><span>Demo cutoff {cutoff:.1%}</span><span>100%</span></div>'
                f'<div class="decision-explanation">{explanation}</div>'
                f'<div class="decision-foot">Class 1 = approved · demonstration only</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("#### Your result will land here")
            st.markdown(
                "Choose a sample case or edit the fields, then run the review. "
                "The output includes a model score and the policy passages used."
            )
            st.markdown(
                '<div class="policy-source"><strong>Small print</strong><br>'
                "The supplied model uses Gender and an undocumented Default_Risk field. "
                "Its outputs are for demonstration only.</div>",
                unsafe_allow_html=True,
            )

    assessment = st.session_state.get("last_assessment")
    if assessment:
        st.markdown("### Policy receipts")
        receipt_columns = st.columns(min(len(assessment["sections"]), 3), gap="medium")
        for column, section in zip(receipt_columns, assessment["sections"]):
            with column:
                st.markdown(
                    f'<div class="policy-source"><strong>[{section["id"]}] '
                    f'{section["title"]}</strong><br>{section["body"]}</div>',
                    unsafe_allow_html=True,
                )

elif main_view == "Ask the policy":
    st.markdown("### The policy, in plain sight")
    st.markdown(
        "Ask a question. The answer stays close to the source and shows its receipts."
    )
    with st.form("policy_question_form"):
        question = st.text_input(
            "Your question",
            placeholder="What does the policy say about repayment capacity?",
            label_visibility="collapsed",
        )
        ask_policy = st.form_submit_button("Find the relevant clauses  →", type="primary")
    if ask_policy:
        answer, matches = policy_answer(
            question, policy_sections, policy_vectorizer, policy_matrix
        )
        st.markdown("### Answer")
        st.write(answer)
        if matches:
            st.markdown("### Retrieved passages")
            for section, _score in matches:
                st.markdown(
                    f'<div class="policy-source"><strong>[{section["id"]}] '
                    f'{section["title"]}</strong><br>{section["body"]}</div>',
                    unsafe_allow_html=True,
                )
    with st.expander("Why retrieve policy instead of relying on general AI knowledge?"):
        st.write(
            "A general-purpose model can sound certain while using stale or unsupported "
            "rules. Retrieval grounds the response in this specific policy and exposes the "
            "exact clauses. This demo uses TF-IDF retrieval plus a response template, not "
            "an external LLM; when no clause matches, it asks for human review."
        )

elif main_view == "Read the policy":
    st.markdown("### Lending policy")
    st.markdown(
        "An illustrative document for this prototype, not an actual lender policy or "
        "compliance-approved standard."
    )
    st.divider()
    st.markdown(POLICY_PATH.read_text(encoding="utf-8"))

elif main_view == "Business case":
    st.markdown("### Defend the decision system")
    st.markdown(
        "This is a screening prototype, not an autonomous lender. Its business case "
        "depends on verified portfolio outcomes, review capacity, and an approved policy."
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Test cohort", "10,400 applications")
    metric_columns[1].metric("Logistic accuracy", "85.56%")
    metric_columns[2].metric("Majority baseline", "65.36%")
    metric_columns[3].metric("RAG model calls", "None")

    st.markdown("#### What it costs")
    cost_columns = st.columns(3, gap="large")
    with cost_columns[0]:
        st.markdown("**Run cost**")
        st.write(
            "Inference is local scikit-learn and policy search is local TF-IDF. "
            "There is no per-token LLM charge. Hosting, storage, monitoring, and "
            "engineering still cost money; Streamlit hosting pricing depends on the selected plan."
        )
    with cost_columns[1]:
        st.markdown("**Review cost**")
        st.write(
            "A human-review queue needs trained staff, time, escalation rules, "
            "and quality assurance. The scenario calculator makes this cost adjustable."
        )
    with cost_columns[2]:
        st.markdown("**Build and upkeep**")
        st.write(
            "Data validation, policy ownership, fairness testing, threshold monitoring, "
            "security, incident response, and retraining are ongoing responsibilities."
        )

    st.markdown("#### How it fails, who owns the response")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Failure": "Incorrect approval",
                    "Business impact": "Credit loss, collections burden, customer harm",
                    "Accountable role": "Lending decision owner; reviewer signs off",
                },
                {
                    "Failure": "Incorrect rejection",
                    "Business impact": "Lost contribution, unfair exclusion, complaint risk",
                    "Accountable role": "Lending owner and compliance; provide review and required reasons",
                },
                {
                    "Failure": "Biased or leaked feature",
                    "Business impact": "Unlawful or discriminatory outcomes; invalid metrics",
                    "Accountable role": "Model risk, data steward, and compliance before release",
                },
                {
                    "Failure": "Wrong or stale policy passage",
                    "Business impact": "Misleading explanation or inconsistent treatment",
                    "Accountable role": "Policy owner; version, approve, and audit the source document",
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Accountability cannot be delegated to a model: the bank's designated lending "
        "owner remains responsible for policy, final decisions, and customer remedies."
    )

elif main_view == "Prompt log":
    st.markdown("### Prompt and change log")
    st.caption("Key request summaries and the implementation decisions they drove.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Source": "Coding assistant",
                    "Key prompt (summary)": "Build a Streamlit loan demo with the supplied models, scaler, and lending-policy RAG.",
                    "What went wrong": "The first version used a generic sidebar-and-tabs layout; a chart dependency also failed in the local Python runtime.",
                    "What changed": "Rebuilt the review desk UI; replaced the chart with native progress indicators and retained policy citations.",
                },
                {
                    "Source": "Coding assistant",
                    "Key prompt (summary)": "Show an approval, rejection, and borderline case; make the UI minimal, white, and readable.",
                    "What went wrong": "Model artifacts alone did not reveal labels or thresholds; category encodings were undocumented.",
                    "What changed": "Read the notebook outputs; mapped class 1 to approved; added model-specific thresholds and synthetic, explicitly coded examples.",
                },
                {
                    "Source": "Coding assistant",
                    "Key prompt (summary)": "Replace the UI completely, then add business, prompt-log, and risk views.",
                    "What went wrong": "The initial primary button inherited black text on a dark background; the local server also needed a restart to show a redesign.",
                    "What changed": "Set explicit green button and black text styles; restarted the app; added the three requested views.",
                },
                {
                    "Source": "In-app LLM",
                    "Key prompt": "None. No generative LLM is configured or called by this prototype.",
                    "What went wrong": "No LLM prompt failure. An early policy-heading parser merged sections and could return oversized retrieval chunks.",
                    "What changed": "Restricted section titles to one line; policy questions now use TF-IDF retrieval, cited source passages, and a no-match human-review fallback.",
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
        height=300,
    )
    st.markdown("#### What the app actually sends")
    st.write(
        "A visitor's policy question is used as a local TF-IDF search query. It is not "
        "sent to an external model, stored as a prompt log, or mixed with applicant details. "
        "The response is a template around the retrieved policy clauses."
    )

elif main_view == "Risks & returns":
    st.markdown("### Risk, loss, and return")
    st.write(
        "The worksheet below turns classification errors into an illustrative portfolio "
        "scenario. Every dollar value is an assumption you can change."
    )
    st.warning(
        "Important limitation: the notebook target is historical loan approval, not observed repayment or default. "
        "A false approval below means an application the historical label marked rejected; "
        "the assumed dollar loss is not measured default loss."
    )

    st.markdown("#### Assumptions per application")
    assumption_columns = st.columns(3, gap="medium")
    with assumption_columns[0]:
        contribution = st.number_input(
            "Contribution from a good approval (USD)",
            min_value=0.0,
            value=1500.0,
            step=100.0,
            key="assumption_contribution",
        )
        false_approval_loss = st.number_input(
            "Loss from an incorrect approval (USD)",
            min_value=0.0,
            value=8000.0,
            step=500.0,
            key="assumption_false_approval",
        )
    with assumption_columns[1]:
        false_rejection_loss = st.number_input(
            "Missed contribution from an incorrect rejection (USD)",
            min_value=0.0,
            value=1200.0,
            step=100.0,
            key="assumption_false_rejection",
        )
        review_share_percent = st.number_input(
            "Applications sent to manual review (%)",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=1.0,
            key="assumption_review_share",
        )
    with assumption_columns[2]:
        review_cost = st.number_input(
            "Cost per manual review (USD)",
            min_value=0.0,
            value=35.0,
            step=5.0,
            key="assumption_review_cost",
        )
        inference_cost = st.number_input(
            "Inference cost per application (USD)",
            min_value=0.0,
            value=0.02,
            step=0.01,
            format="%.2f",
            key="assumption_inference_cost",
        )

    test_counts = {
        "true_approved": 6353,
        "false_approved": 1055,
        "true_rejected": 2548,
        "false_rejected": 444,
    }
    approved_all_counts = {
        "true_approved": 6797,
        "false_approved": 3603,
        "true_rejected": 0,
        "false_rejected": 0,
    }
    assumptions = {
        "contribution_per_good_approval": contribution,
        "loss_per_false_approval": false_approval_loss,
        "contribution_lost_per_false_rejection": false_rejection_loss,
        "review_share": review_share_percent / 100,
        "review_cost": review_cost,
        "inference_cost": inference_cost,
    }
    model_projection = portfolio_projection(**test_counts, **assumptions)
    baseline_projection = portfolio_projection(
        **approved_all_counts,
        **{
            **assumptions,
            "review_share": 0,
            "review_cost": 0,
            "inference_cost": 0,
        },
    )
    profit_delta = model_projection["net_value"] - baseline_projection["net_value"]
    break_even_loss = (
        (6797 - 6353) * contribution
        + 444 * false_rejection_loss
        + model_projection["human_review_cost"]
        + model_projection["inference_cost"]
    ) / (3603 - 1055)

    st.markdown("#### Notebook test-set counts")
    count_columns = st.columns(4)
    count_columns[0].metric("Correct approvals", "6,353")
    count_columns[1].metric("Incorrect approvals", "1,055")
    count_columns[2].metric("Correct rejections", "2,548")
    count_columns[3].metric("Incorrect rejections", "444")

    st.markdown("#### Illustrative cohort comparison")
    result_columns = st.columns(3)
    result_columns[0].metric(
        "Model net contribution",
        format_usd(model_projection["net_value"]),
        help="Assumption-based estimate, not observed portfolio profit.",
    )
    result_columns[1].metric(
        "Approve-all baseline",
        format_usd(baseline_projection["net_value"]),
        help="Assumes all 10,400 applications are approved.",
    )
    result_columns[2].metric(
        "Difference vs baseline",
        format_usd(profit_delta),
        delta_color="normal",
    )
    st.caption(
        "Model net = true approvals × assumed contribution − false approvals × assumed loss "
        "− false rejections × assumed missed contribution − manual reviews × review cost "
        "− applications × inference cost."
    )
    st.markdown("#### Estimated costs for this cohort")
    cost_rows = [
        {
            "Cost item": "Incorrect approvals",
            "Cases": "1,055",
            "Assumed cost per case": format_usd(false_approval_loss),
            "Estimated total": format_usd(model_projection["false_approval_loss"]),
        },
        {
            "Cost item": "Incorrect rejections",
            "Cases": "444",
            "Assumed cost per case": format_usd(false_rejection_loss),
            "Estimated total": format_usd(model_projection["false_rejection_loss"]),
        },
        {
            "Cost item": "Manual review",
            "Cases": f"{model_projection['reviewed']:,}",
            "Assumed cost per case": format_usd(review_cost),
            "Estimated total": format_usd(model_projection["human_review_cost"]),
        },
        {
            "Cost item": "Inference",
            "Cases": f"{model_projection['cohort_size']:,}",
            "Assumed cost per case": format_usd(inference_cost),
            "Estimated total": format_usd(model_projection["inference_cost"]),
        },
    ]
    st.dataframe(
        pd.DataFrame(cost_rows),
        hide_index=True,
        use_container_width=True,
    )
    cost_total = (
        model_projection["false_approval_loss"]
        + model_projection["false_rejection_loss"]
        + model_projection["human_review_cost"]
        + model_projection["inference_cost"]
    )
    summary_columns = st.columns([1, 2], gap="large")
    summary_columns[0].metric("Total estimated costs", format_usd(cost_total))
    summary_columns[1].metric(
        "Break-even false-approval loss",
        format_usd(break_even_loss),
        help="The model's estimated advantage over approving everyone reaches zero at this assumed loss per incorrect approval.",
    )
    st.caption(
        "Below the break-even loss, the approve-all baseline has the higher estimated value; "
        "above it, the model has the higher estimated value under these assumptions."
    )
    with st.expander("Risks to address before going live"):
        st.markdown(
            "- **Bias:** the model uses Gender; test disparate outcomes and remove protected features and proxies unless approved and justified.\n"
            "- **Privacy:** never send applicant financial details to an unapproved service; define access, retention, deletion, and incident response.\n"
            "- **Misuse:** a score is not a lending decision; require human review, documented reasons, and applicable notices.\n"
            "- **Model validity:** `Default_Risk` is undocumented and may leak the target. The threshold was selected on the reported test set.\n"
            "- **RAG:** stale or incomplete policy can produce irrelevant retrieval. Version approved documents and route low-confidence matches to a policy owner.\n"
            "- **Economics:** replace every assumption with observed default severity, margin, review and hosting costs before investment decisions."
        )