from flask import Flask, render_template, request, jsonify
import nltk
import string
import random
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

with open('data.txt', 'r', errors='ignore') as f:
    raw_doc = f.read().lower()

sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

lemmer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in stop_words and token not in string.punctuation]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower()))

greet_inputs = ('hello', 'hi', 'whassup', 'how are you?')
greet_responses = ('hi', 'Hey', 'hey there!', 'There there!!')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# Load pre-trained BERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(sentence_tokens)

def response(user_response):
    sentence_tokens.append(user_response)
    user_embedding = model.encode([user_response])
    similarities = cosine_similarity(user_embedding, sentence_embeddings)
    idx = similarities.argsort()[0][-1]
    flat = similarities.flatten()
    flat.sort()
    req_tfidf = flat[-1]
    if req_tfidf == 0:
        robo1_response = "I am sorry. Unable to understand you!"
    else:
        robo1_response = sentence_tokens[idx]
    sentence_tokens.pop(-1)
    return robo1_response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_response = request.form["msg"].lower()
    if user_response not in ['bye', 'thanks', 'thank you']:
        if greet(user_response):
            return jsonify({"response": greet(user_response)})
        else:
            bot_response = response(user_response)
            return jsonify({"response": bot_response})
    else:
        return jsonify({"response": "Goodbye!"})

if __name__ == "__main__":
    app.run(debug=True)
