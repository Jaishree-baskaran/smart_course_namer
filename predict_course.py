import joblib

# Load model
classifier, label_encoder, embedder = joblib.load("course_model.pkl")

# Get user input
print(">> \nEnter your skills (comma-separated, e.g., python, sql, flask):")
user_input = input("Skills: ").strip().lower()

if not user_input:
    print("⚠️ Please enter at least one skill to get a course recommendation.")
else:
    # Embed and predict
    embedding = embedder.encode([user_input])
    predicted = classifier.predict(embedding)
    predicted_label = label_encoder.inverse_transform(predicted)[0]

    print(f"\n Suggested Course Title: {predicted_label} Course")
