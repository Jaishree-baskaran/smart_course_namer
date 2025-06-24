import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from embeddings_model import get_embedder

# Load the dataset
df = pd.read_csv("course_dataset.csv")

# Drop any missing values
df = df.dropna()

# Separate features and labels
X = df["skills"]
y = df["course_title"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Load the sentence-transformer embedder
embedder = get_embedder()

# Convert all skill descriptions to embeddings
X_embeddings = embedder.encode(X.tolist())

# Train a classifier on top of the embeddings
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_embeddings, y_encoded)

# Save the model components
joblib.dump((classifier, label_encoder, embedder), "course_model.pkl")

print("Course model trained and saved as 'course_model.pkl'")
