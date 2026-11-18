import pandas as pd

# Load student feedback dataset
data = pd.read_csv("feedback.csv")

# Display feedback
print("STUDENT FEEDBACK DATA")
print(data)

# Count sentiments
print("\nSENTIMENT ANALYSIS")
print(data["sentiment"].value_counts())