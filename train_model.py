import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("feedback.csv")

# Divide dataset
from preprocess import clean_text

X = data["feedback"].apply(clean_text)
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create ML model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("MODEL ACCURACY:", accuracy * 100, "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, predictions))

# Train final model using complete dataset
model.fit(X, y)
os.makedirs("models", exist_ok=True)
# Save model
joblib.dump(model, "models/sentiment_model.pkl")

print("\nModel saved successfully!")