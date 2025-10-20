from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
import string
import nltk
import os

nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

try:
    nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
except NameError:
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

app = Flask(__name__)



def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def stem_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def tokenize_text(text):
    tokens = word_tokenize(text)
    output = ','.join(tokens)
    return output


def get_bow(texts):
    if isinstance(texts, str):
        texts = [texts]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out().tolist()
    vectors = X.toarray().tolist()
    return {"features": features, "vectors": vectors}

def get_tfidf(texts):
    if isinstance(texts, str):
        texts = [texts]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out().tolist()
    vectors = X.toarray().tolist()
    return {"features": features, "vectors": vectors}

def pos_tagging(text):
    filtered_text = remove_stopwords(text)  
    tokens = word_tokenize(filtered_text)   
    return [{"word": word, "pos": pos} for word, pos in tokens]

def named_entity_recognition(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    pos_tags_list = pos_tag(tokens)
    chunks = ne_chunk(pos_tags_list)
    entities = []
    for chunk in chunks:
        if isinstance(chunk, Tree): 
            entity_name = " ".join(c[0] for c in chunk.leaves())
            entity_type = chunk.label()
            entities.append({"entity": entity_name, "type": entity_type})
    return entities

@app.route("/")
def home():
    return "App is running!"

@app.route("/process", methods=["POST"])
def process_text():
    data = request.json
    text = data.get("text", "")
    texts = data.get("texts", None)
    action = data.get("action", "").lower()
    
    if not text and not texts:
        return jsonify({"error": "No text or texts provided"}), 400
    
    result = None

    if action == "lowercase":
        result = to_lowercase(text)
    elif action == "remove_punctuation":
        result = remove_punctuation(text)
    elif action == "remove_stopwords":
        result = remove_stopwords(text)
    elif action == "lemmatize":
        result = lemmatize_text(text)
    elif action == "stem":
        result = stem_text(text)
    elif action == "tokenize":
        result = tokenize_text(text)
    elif action == "pos_tagging":
        result = pos_tagging(text)
    elif action == "ner":
        result = named_entity_recognition(text)
    elif action == "bow":
        input_texts = texts if texts else text
        result = get_bow(input_texts)
    elif action == "tfidf":
        input_texts = texts if texts else text
        result = get_tfidf(input_texts)
    elif action == "vocabulary":
        input_texts = texts if texts else text
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        vectorizer = CountVectorizer()
        vectorizer.fit(input_texts)
        vocab = sorted(vectorizer.get_feature_names_out().tolist())    
        result = {"vocabulary": vocab}
    elif action == "all":
        processed_text = to_lowercase(text)
        processed_text = remove_punctuation(processed_text)
        processed_text = remove_stopwords(processed_text)
        processed_text = lemmatize_text(processed_text)
        processed_text = stem_text(processed_text)
        tokens = tokenize_text(processed_text)
        result = {
            "processed_text": processed_text,
            "tokens": tokens,
        }
    else:
        return jsonify({"error": "Invalid action"}), 400

    return jsonify({"result": result})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)