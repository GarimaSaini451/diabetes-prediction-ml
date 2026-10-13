# Diabetes Prediction System Using Machine Learning Classifiers

## Project overview
A Streamlit web application that demonstrates binary classification with **one classifier: Logistic Regression**. It accepts eight input features, runs the fitted pipeline, and displays an outcome class and model-estimated positive-class probability.

## Important dataset disclosure
`diabetes_dataset_synthetic.csv` is **synthetic educational data**, generated for this project because it is not a verified clinical dataset. Its labels are simulated. Metrics are only a software demonstration and must not be presented as clinical accuracy or as results on the Pima Indians Diabetes dataset. Do not use the app to make medical decisions.

For a research submission that requires a recognized benchmark, replace the CSV with a properly sourced public dataset (for example, the Pima Indians Diabetes dataset), record the source/license, and rerun the evaluation. Verify column definitions and missing-value conventions before training.

## Features
- Streamlit interface for eight input variables
- Logistic Regression classifier
- Stratified 80/20 train-test split
- Median imputation and feature standardization in a scikit-learn pipeline
- Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and classification report
- Dataset preview, class counts, and CSV download

## Run locally
1. Install Python 3.10 or later.
2. Extract this project folder.
3. Open a terminal in the folder.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the application:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL shown by Streamlit (normally `http://localhost:8501`).

## Input features
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target: `Outcome` (0 or 1).

## Methodology
1. Load the CSV.
2. Separate the input features and target.
3. Treat zero values in selected clinical fields as missing, then impute with the median. This convention is dataset-specific and must be verified when replacing the dataset.
4. Split data into 80% training and 20% test sets using stratification.
5. Fit a pipeline consisting of median imputation, standardization, and Logistic Regression.
6. Evaluate once on the held-out test set.
7. Use the trained model to classify new input rows.

## Metrics
- Accuracy: fraction of all test samples classified correctly.
- Precision: fraction of positive predictions that are truly positive in the test labels.
- Recall (sensitivity): fraction of actual positives detected.
- F1-score: harmonic mean of precision and recall.
- ROC-AUC: ranking/discrimination metric based on predicted probabilities.
- Confusion matrix: counts of true negatives, false positives, false negatives, and true positives.

No metric value is hard-coded; the application computes metrics at runtime from the held-out test split.

## Limitations
- Synthetic data and simulated labels are not medically valid.
- Dataset distribution may not represent any real population.
- Logistic Regression assumes a linear relationship between predictors and log-odds.
- A single random split can vary; rigorous work should use cross-validation and external validation.
- No calibration or clinical validation is provided.
- The app is an educational prototype, not a diagnostic device.

## Files
- `app.py`: Streamlit application and model pipeline
- `diabetes_dataset_synthetic.csv`: synthetic sample dataset
- `requirements.txt`: Python dependencies
- `Project_Report.docx`: project report
- `Project_Presentation.pptx`: presentation
