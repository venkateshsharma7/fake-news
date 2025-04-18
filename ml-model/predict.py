import sys
import joblib
import os

# Get paths relative to the current script
model_path = os.path.join(os.path.dirname(__file__), "fake_news_model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

# Load the model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Get input text from command-line argument
text = sys.argv[1]

# Transform input text and make prediction
features = vectorizer.transform([text])
prediction = model.predict(features)

# Output result (IMPORTANT: Only print this and flush)
if prediction[0] == 1:
    print("Real News")
else:
    print("Fake News")

# Flush output for Node.js to receive it immediately
sys.stdout.flush()
