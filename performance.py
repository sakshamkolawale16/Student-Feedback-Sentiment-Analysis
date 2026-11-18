import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocess import clean_text

# Load dataset
data = pd.read_csv("feedback.csv")

X = data["feedback"].apply(clean_text)
y = data["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train evaluation model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, average="macro", zero_division=0
)
recall = recall_score(
    y_test, predictions, average="macro", zero_division=0
)
f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)

# Website
st.title("📊 ML Model Performance")

st.write("Evaluation of Student Feedback Sentiment Model")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy*100:.2f}%")
c2.metric("Precision", f"{precision*100:.2f}%")
c3.metric("Recall", f"{recall*100:.2f}%")
c4.metric("F1 Score", f"{f1*100:.2f}%")

st.divider()

# Confusion matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["negative", "neutral", "positive"]
)

st.dataframe(
    pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Neutral", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Neutral", "Predicted Positive"]
    )
)