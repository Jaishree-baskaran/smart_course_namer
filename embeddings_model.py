import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load dataset
df = pd.read_csv("course_dataset.csv")
df.dropna(inplace=True)

# Step 2: Extract skills and course labels
X = df["skills"].astype(str).tolist()
y = df["course_title"].astype(str).tolist()

# Step 3: Label encode the course titles
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Convert skills into embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = embedder.encode(X)

# Step 5: Train classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_embeddings, y_encoded)

# Step 6: Save model + label encoder + embedder
joblib.dump((classifier, label_encoder, embedder), "course_model.pkl")
print(" Model trained and saved as 'course_model.pkl'")
