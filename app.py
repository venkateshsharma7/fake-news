from flask import Flask, request, render_template
import joblib
import os
import re
import gdown
import zipfile
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect
import numpy as np
import requests
import difflib

app = Flask(__name__, template_folder='templates')

# ---------------------- API Key ----------------------
import os
NEWSDATA_API_KEY = os.environ.get("NEWSDATA_API_KEY", "")

# ---------------------- Load Models ----------------------
MODEL1_PATH = os.path.join('ml-model', 'fake_news_model.pkl')
VECTORIZER1_PATH = os.path.join('ml-model', 'vectorizer.pkl')
model1 = joblib.load(MODEL1_PATH)
vectorizer1 = joblib.load(VECTORIZER1_PATH)

MODEL2_PATH = os.path.join('ml-model', 'model2.pkl')
EMBEDDER2_DIR = os.path.join('ml-model', 'embedding_model2')
EMBEDDER_ZIP_PATH = os.path.join('ml-model', 'embedding_model2.zip')

# Auto-download embedding model if missing
if not os.path.exists(EMBEDDER2_DIR):
    print("[INFO] embedding_model2 not found. Downloading from Google Drive...")
    gdown.download(
        url="https://drive.google.com/uc?export=download&id=171x5kcV2ST-8uYzn2U78t_aZkbnJeNPg",
        output=EMBEDDER_ZIP_PATH,
        quiet=False
    )
    with zipfile.ZipFile(EMBEDDER_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('ml-model')
    print("[INFO] embedding_model2 downloaded and extracted.")

model2 = joblib.load(MODEL2_PATH)
embedder2 = SentenceTransformer(EMBEDDER2_DIR)

classifier3 = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels3 = ["Real News", "Fake News", "Satire", "Exaggerated", "Opinion", "Serious", "Conspiracy"]

# ---------------------- Utilities ----------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang != 'en':
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated
        return text
    except:
        return text

def loose_match(a, b, threshold=0.4):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def extract_keywords(text, max_keywords=5):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stopwords = {'this', 'that', 'have', 'with', 'from', 'their', 'being', 'about', 'would', 'there', 'which'}
    keywords = [w for w in words if w not in stopwords]
    return ' '.join(keywords[:max_keywords])

def fact_check_with_newsdata(query):
    try:
        query = extract_keywords(query)
        print(f"[QUERY USED FOR FACT CHECK]: {query}")
        url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={query}&language=en"
        res = requests.get(url, timeout=5)
        print(f"NewsData.io API status: {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            articles = data.get('results', [])
            print(f"NewsData.io returned {len(articles)} articles.")
            match_count = sum(1 for article in articles if loose_match(query, article.get('title', '')))
            print(f"Loose matches: {match_count}")
            if match_count >= 3:
                return 0.3, True, match_count
            elif match_count == 2:
                return 0.15, True, match_count
            elif match_count == 1:
                return 0.05, True, match_count
            else:
                return 0, True, match_count
        print("API response not 200.")
        return 0, False, 0
    except Exception as e:
        print(f"NewsData.io ERROR: {e}")
        return 0, False, 0

# ---------------------- Routes ----------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('news')
    if text is None or str(text).strip() == "":
        return render_template('index.html', prediction=None, original_text="", translated_text="", fact_checked=False, match_count=0)

    translated_text = translate_to_english(text)
    cleaned_text = clean_text(translated_text)
    print("\n--- PREDICTION REPORT ---")
    print(f"Original Text: {text}")
    print(f"Translated Text: {translated_text}")

    # TF-IDF
    features = vectorizer1.transform([translated_text])
    prob1 = model1.predict_proba(features)[0][1]

    # Semantic
    embedding = embedder2.encode([cleaned_text])
    prob2 = model2.predict_proba(embedding)[0][1]

    # Tone
    result_raw = classifier3(cleaned_text, candidate_labels=labels3)
    top_label = result_raw['labels'][0]
    adjust = 0
    if top_label == "Serious": adjust = 0.1
    elif top_label == "Satire": adjust = -0.2
    elif top_label in ["Exaggerated", "Conspiracy"]: adjust = -0.1
    elif top_label == "Opinion": adjust = -0.05
    elif top_label == "Real News": adjust = 0.05

    # Score & Fact Check
    model_score = max(0, min(1, (0.4 * prob1 + 0.6 * prob2 + adjust)))
    fact_score, fact_success, match_count = fact_check_with_newsdata(cleaned_text)
    final_score = max(0, min(1, model_score + fact_score))

    print(f"TF-IDF: {prob1:.4f}, Semantic: {prob2:.4f}, Tone: {top_label}, Fact Boost: {fact_score:.2f}, Final Score: {final_score:.4f}")
    print(f"[DEBUG] fact_success = {fact_success}, match_count = {match_count}")

    if final_score >= 0.8:
        result = "✅ Real News"
    elif final_score >= 0.6:
        result = "🟢 Likely Real"
    elif final_score >= 0.4:
        result = "🟠 Likely Fake"
    else:
        result = "❌ Fake News"

    print(f"Model Output: {result}\n")

    return render_template(
        'index.html',
        prediction=result,
        original_text=text,
        translated_text=translated_text,
        fact_checked=fact_success,
        match_count=match_count
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
