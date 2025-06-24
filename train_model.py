import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer


# Load dataset
df = pd.read_csv("data/course_dataset.csv")


# Combine skills into one string per row
X = df["skills"]
y = df["course_title"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Load sentence transformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_embedded = embedder.encode(X.tolist())

# Train classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_embedded, y_encoded)

# Save model components
joblib.dump((classifier, label_encoder, embedder), "course_model.pkl")

print("Model trained and saved as 'course_model.pkl'")
