import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text

CI_MODE = os.getenv("CI", "false").lower() == "true"

# -----------------------------
# 1. Load datasets
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fake_df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "fake.csv"))
true_df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "true.csv"))

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
if CI_MODE:
    print("Running in CI mode: using small dataset")
    df = df.sample(n=2000, random_state=42)
# -----------------------------
# 2. Preprocess
# -----------------------------
df["cleaned_text"] = df["text"].apply(clean_text)

X = df["cleaned_text"]
y = df["label"]

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. MLflow experiment
# -----------------------------
mlflow.set_experiment("Fake-News-Detection")

with mlflow.start_run():

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # 5. Log to MLflow
    # -----------------------------
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("ngram_range", "1-2")
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # -----------------------------
    # 6. Save artifacts locally
    # -----------------------------
    joblib.dump(model, "../models/fake_news_model.pkl")
    joblib.dump(tfidf, "../models/tfidf_vectorizer.pkl")

    mlflow.log_artifact("../models/fake_news_model.pkl")
    mlflow.log_artifact("../models/tfidf_vectorizer.pkl")

print("\n✅ Model, Vectorizer saved and tracked with MLflow successfully")
