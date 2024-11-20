from flask import Flask, request, jsonify
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask_cors import CORS
import os
import nltk

# Set a local directory for NLTK data
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')  # Use the current file's directory
os.makedirs(nltk_data_dir, exist_ok=True)  # Create directory if it doesn't exist
nltk.data.path.append(nltk_data_dir)  # Add it to NLTK's data search path

# Download required NLTK resources
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and vectorizer
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'sentiment_model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Define stop words with negation handling
negation_words = {"not", "no", "never", "none"}
stop_words = set(stopwords.words('english')) - negation_words

# Helper function to expand contractions
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

# Helper function to preprocess text
def preprocess_text(text):
    text = expand_contractions(text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)

# Test endpoint
@app.route('/testpurpose', methods=['GET'])
def testingPurpose():
    return jsonify({'message': 'This is a test-purpose API', 'status': 'success'})

# Predict sentiment endpoint
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

# Run app
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
