#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import nltk
import os

nltk_packages = ['punkt', 'stopwords', 'wordnet']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg=='punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

app = Flask(__name__)

def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return ' '.join([token for token in tokens if token.lower() not in stop_words])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def tokenize_text(text):
    return word_tokenize(text)

def get_bow(texts):
    if isinstance(texts, str):
        texts = [texts]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return {
        'features': vectorizer.get_feature_names_out().tolist(),
        'vectors': X.toarray().tolist()
    }

def get_tfidf(texts):
    if isinstance(texts, str):
        texts = [texts]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return {
        'features': vectorizer.get_feature_names_out().tolist(),
        'vectors': X.toarray().tolist()
    }

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text', '')
    texts = data.get('texts', None)
    action = data.get('action', '').lower()

    if not text and not texts:
        return jsonify({"error": "No text or texts provided"}), 400

    if action == 'lowercase':
        result = to_lowercase(text)
    elif action == 'remove_punctuation':
        result = remove_punctuation(text)
    elif action == 'remove_stopwords':
        result = remove_stopwords(text)
    elif action == 'lemmatize':
        result = lemmatize_text(text)
    elif action == 'tokenize':
        result = tokenize_text(text)
    elif action == 'bow':
        result = get_bow(texts if texts else text)
    elif action == 'tfidf':
        result = get_tfidf(texts if texts else text)
    elif action == 'all':
        processed = to_lowercase(text)
        processed = remove_punctuation(processed)
        processed = remove_stopwords(processed)
        processed = lemmatize_text(processed)
        tokens = tokenize_text(processed)
        result = {
            "processed_text": processed,
            "tokens": tokens
        }
    else:
        return jsonify({"error": "Invalid action"}), 400

    return jsonify({"result": result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


# In[ ]:




