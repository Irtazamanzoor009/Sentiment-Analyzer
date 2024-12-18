print("Starting Flask app")

from flask import Flask, request, jsonify
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask_cors import CORS
import os
import nltk

app = Flask(__name__)
CORS(app)

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

negation_words = {"not", "no", "never", "none"}
stop_words = set(stopwords.words('english')) - negation_words

def expand_contractions(text):
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "isn't": "is not",
        "aren't": "are not",
        "doesn't": "does not",
        "didn't": "did not",
        "don't": "do not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded, text)
    return text

def preprocess_text(text):
    text = expand_contractions(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/testpurpose', methods=['GET'])
def testingPurpose():
    return jsonify({'message': 'This is a test-purpose API', 'status': 'success'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data.get('review', '')
        cleaned_review = preprocess_text(review)
        vectorized_review = vectorizer.transform([cleaned_review])
        sentiment = model.predict(vectorized_review)[0]
        return jsonify({'review': review, 'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failure'})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
