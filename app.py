import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Prediction System")
st.caption("Educational machine-learning demonstration — not a medical diagnosis.")

DATA_PATH = "diabetes_dataset_synthetic.csv"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

@st.cache_resource
def train_model(data):
    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)
    # In this dataset format, zero values for these fields are treated as missing.
    # Do not apply this rule blindly to every real-world dataset.
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        X.loc[X[col] == 0, col] = np.nan
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Confusion matrix": confusion_matrix(y_test, y_pred),
        "Report": classification_report(y_test, y_pred, zero_division=0)
    }
    return model, metrics, X_test, y_test

try:
    data = load_data()
except FileNotFoundError:
    st.error("Dataset file not found. Keep diabetes_dataset_synthetic.csv in the same folder as app.py.")
    st.stop()

st.warning(
    "Dataset disclosure: the included CSV is synthetic data generated for software demonstration. "
    "Its labels are simulated and must not be used for clinical decisions or to claim real-world performance."
)

with st.sidebar:
    st.header("About the project")
    st.write("Classifier: Logistic Regression")
    st.write(f"Rows: {len(data)}")
    st.write("Split: 80% train / 20% test (stratified)")
    st.write("Preprocessing: median imputation + standardization")
    st.markdown("---")
    st.write("The prediction is not a diagnosis. Consult a qualified healthcare professional for medical concerns.")

model, metrics, X_test, y_test = train_model(data)

tab1, tab2, tab3 = st.tabs(["Predict", "Model evaluation", "Dataset"])
with tab1:
    st.subheader("Enter patient attributes")
    c1, c2, c3 = st.columns(3)
    with c1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=1.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.number_input("Blood pressure (mmHg)", min_value=1.0, max_value=200.0, value=72.0, step=1.0)
    with c2:
        skin = st.number_input("Skin thickness (mm)", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
        insulin = st.number_input("Insulin (µU/mL)", min_value=1.0, max_value=1000.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=28.0, step=0.1)
    with c3:
        pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)

    if st.button("Predict", type="primary"):
        sample = pd.DataFrame([{
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree, "Age": age
        }], columns=FEATURES)
        pred = int(model.predict(sample)[0])
        score = float(model.predict_proba(sample)[0, 1])
        st.metric("Model-estimated positive-class probability", f"{score:.1%}")
        if pred == 1:
            st.error("The model classified this input as Outcome = 1.")
        else:
            st.success("The model classified this input as Outcome = 0.")
        st.info("This output reflects a model trained on synthetic data. It is not a personal medical risk estimate or diagnosis.")

with tab2:
    st.subheader("Held-out test-set metrics")
    metric_cols = st.columns(5)
    for col, key in zip(metric_cols, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
        col.metric(key, f"{metrics[key]:.3f}")
    st.caption("These metrics describe performance on a held-out portion of the synthetic dataset only.")
    cm = metrics["Confusion matrix"]
    st.write("Confusion matrix (rows = actual class; columns = predicted class)")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
    st.text("Classification report")
    st.code(metrics["Report"])

with tab3:
    st.subheader("Dataset preview")
    st.dataframe(data.head(20), use_container_width=True)
    st.write("Class counts")
    st.bar_chart(data[TARGET].value_counts().sort_index())
    st.download_button(
        "Download dataset CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="diabetes_dataset_synthetic.csv",
        mime="text/csv"
    )
