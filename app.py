import json

from sklearn.metrics import confusion_matrix
import seaborn as sns

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Transaction Monitoring",
    page_icon="🏦",
    layout="wide"
)

model = joblib.load("models/fraud_detection_model.pkl")

with open("outputs/model_evaluation.json", "r") as file:
    evaluation = json.load(file)

lr = evaluation["Logistic Regression"]
rf = evaluation["Random Forest"]

st.markdown("""
<style>
.stApp {
    background: #101827;
    color: #E7EDF5;
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}

h1 {
    color: #E7EDF5 !important;
    font-size: 34px !important;
}

h2, h3, p, label {
    color: #E7EDF5 !important;
}

[data-testid="stMetric"] {
    background: #1B2638;
    border: 1px solid #34445B;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #AAB8CA !important;
}

[data-testid="stMetricValue"] {
    color: #28B8A5 !important;
}

.stButton > button {
    background: #28B8A5;
    color: #101827;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 9px 20px;
}

.stButton > button:hover {
    background: #38D2BC;
    color: #101827;
}

[data-testid="stFileUploader"] {
    background: #1B2638;
    border: 1px dashed #52647C;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #34445B;
    border-radius: 10px;
}

[data-testid="stSelectbox"] {
    background: #1B2638;
    border-radius: 8px;
}

hr {
    border-color: #34445B;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


st.title("Transaction Monitoring")
st.caption("Financial Operations | Transaction Review")

st.write(
    "Review transaction activity and identify records "
    "that may require further attention."
)

columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

uploaded_file = st.file_uploader(
    "Select transaction file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    missing = [c for c in columns if c not in df.columns]

    if missing:
        st.error(f"Required columns are missing: {missing}")
        st.stop()

    X = df[columns].apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any() or not __import__("numpy").isfinite(X).all().all():
        st.error("The file contains missing or invalid values.")
        st.stop()

    st.subheader("Transaction Preview")
    st.dataframe(df.head(10), use_container_width=True)

    
# Save review results when the button is clicked
if "results" not in st.session_state:
    st.session_state.results = None

if st.button("Review Transactions"):
    predictions = model.predict(X)

    results = df.copy()
    results["Review Status"] = [
        "Needs Review" if p == 1 else "No Concern Detected"
        for p in predictions
    ]

    st.session_state.results = results

# Display results even after Streamlit reruns

if st.session_state.results is not None:

    results = st.session_state.results

    flagged = (results["Review Status"] == "Needs Review").sum()
    clear = (results["Review Status"] == "No Concern Detected").sum()

    st.divider()
    st.subheader("Review Overview")

    a, b, c = st.columns(3)
    a.metric("Total Transactions", f"{len(results):,}")
    b.metric("Needs Review", f"{flagged:,}")
    c.metric("No Concern Detected", f"{clear:,}")

    # This is my Transaction distribution chart section 

    st.subheader("Transaction Distribution")

    total = flagged + clear
    flagged_pct = (flagged / total) * 100
    clear_pct = (clear / total) * 100

    col1, col2 = st.columns(2)

    col1.metric("Needs Review", f"{flagged:,}")
    col1.caption(f"{flagged_pct:.3f}% of total transactions")

    col2.metric("No Concern Detected", f"{clear:,}")
    col2.caption(f"{clear_pct:.3f}% of total transactions")

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#101827")
    ax.set_facecolor("#1B2638")

    bars = ax.barh(
        ["No Concern Detected", "Needs Review"],
        [clear, flagged],
        color=["#28B8A5", "#E5AD55"],
        height=0.45
    )

    ax.set_xlim(0, total * 1.12)
    ax.set_xlabel("Number of Transactions", color="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#52647C")
    ax.spines["left"].set_color("#52647C")

    ax.text(
        clear + total * 0.01,
        0,
        f"{clear:,} ({clear_pct:.2f}%)",
        color="white",
        va="center"
    )

    ax.text(
        total * 0.01,
        1,
        f"{flagged:,} ({flagged_pct:.2f}%)",
        color="white",
        va="center"
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Transaction review table
    st.subheader("Transaction Review Details")

    status = st.selectbox(
        "Filter Transactions",
        ["All", "Needs Review", "No Concern Detected"]
    )

    if status == "All":
        display = results
    else:
        display = results[results["Review Status"] == status]

    view_columns = ["Time", "Amount", "Review Status"]

    st.dataframe(
        display[view_columns],
        use_container_width=True,
        hide_index=True
    )

    # Download report
    report = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Review Report",
        report,
        "transaction_review_report.csv",
        "text/csv"
    )

    st.info(
        "Transactions marked for review require human verification. "
        "A model prediction does not confirm fraud."
    )

else:
    st.info("Upload a CSV file to begin transaction review.")

# This is my Model Performance section #1

st.divider()
st.subheader("Model Performance")
st.caption("Random Forest | Held-out test dataset")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{rf['accuracy']:.2%}")
m2.metric("Precision", f"{rf['precision']:.2%}")
m3.metric("Recall", f"{rf['recall']:.2%}")
m4.metric("F1-Score", f"{rf['f1_score']:.2%}")

st.info(
    "These metrics represent model evaluation on the held-out "
    "test dataset, not the uploaded transaction file."
)

st.subheader("Confusion Matrix")

cm = rf["confusion_matrix"]

fig, ax = plt.subplots(figsize=(6, 3.8))
fig.patch.set_facecolor("#101827")

sns.heatmap(
    cm,
    annot=True,
    fmt=",",
    cmap="YlGnBu",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"],
    ax=ax
)

ax.set_xlabel("Predicted Class", color="white")
ax.set_ylabel("Actual Class", color="white")
ax.tick_params(colors="white")

colorbar = ax.collections[0].colorbar
colorbar.ax.yaxis.set_tick_params(color="white")
plt.setp(colorbar.ax.get_yticklabels(), color="white")

st.pyplot(fig, use_container_width=True)
plt.close(fig)

#This is my Model comparison section #2

st.divider()
st.subheader("Model Comparison")
st.caption("Performance comparison on the held-out test dataset")

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Logistic Regression": [
        lr["accuracy"] * 100,
        lr["precision"] * 100,
        lr["recall"] * 100,
        lr["f1_score"] * 100
    ],
    "Random Forest": [
        rf["accuracy"] * 100,
        rf["precision"] * 100,
        rf["recall"] * 100,
        rf["f1_score"] * 100
    ]
})

st.dataframe(
    comparison.round(2),
    use_container_width=True,
    hide_index=True
)

fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor("#101827")
ax.set_facecolor("#1B2638")

x = range(len(comparison))
width = 0.35

bars1 = ax.bar(
    [i - width/2 for i in x],
    comparison["Logistic Regression"],
    width,
    label="Logistic Regression",
    color="#78BDF2"
)

bars2 = ax.bar(
    [i + width/2 for i in x],
    comparison["Random Forest"],
    width,
    label="Random Forest",
    color="#28B8A5"
)

ax.set_ylabel("Score (%)", color="white")
ax.set_xticks(list(x))
ax.set_xticklabels(comparison["Metric"], color="white")
ax.set_ylim(0, 110)
ax.tick_params(colors="white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(facecolor="#1B2638", labelcolor="white")

ax.bar_label(bars1, fmt="%.1f", padding=3, color="white")
ax.bar_label(bars2, fmt="%.1f", padding=3, color="white")

st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.info(
    "Random Forest achieved higher precision and F1-score, "
    "while Logistic Regression achieved higher fraud recall."
)



st.divider()
st.caption("Transaction Monitoring | Internal Review Dashboard")
