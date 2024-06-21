from flask import Flask, render_template, request, jsonify
import nltk
import string
import random
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup


app = Flask(__name__)


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#xóa chú thích
def remove_text_comments(text):
    text = re.sub(r'\[[^\]]*\]', '', text) 
    return text

#Lấy data
def get_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = '\n'.join([p.get_text() for p in paragraphs])
            return text
        else:
            print("Không thể lấy nội dung từ URL")
            return None
    except Exception as e:
        print("Đã có lỗi xảy ra:", str(e))
        return None

def save_text_to_file(text, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Đã lưu văn bản vào file {filename}")
    except Exception as e:
        print("Đã có lỗi xảy ra khi lưu văn bản vào file:", str(e))

url = "https://en.wikipedia.org/wiki/ChatGPT"
text = get_text_from_url(url)
if text:
    save_text_to_file(text, 'data.txt')

#chuyển thành chữ thường
with open('data.txt', 'r', errors='ignore') as f:
    raw_doc = f.read().lower()

raw_doc = remove_text_comments(raw_doc)
raw_doc = raw_doc.lower()

# Tách thành các câu và từ
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

# Khởi tạo lemmatizer và tập hợp các từ dừng
lemmer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

#Xóa StopWords
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in stop_words and token not in string.punctuation]

# Hàm chuẩn hóa văn bản
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower()))

# Câu hỏi và câu trả lời có sẵn
greet_inputs = ('hello', 'hi', 'whassup', 'how are you?')
greet_responses = ('hi', 'Hey', 'hey there!', 'There there!!')

# Trả lời dựa trên thiết lập sẵn
def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# Khởi tạo TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=LemNormalize)

# Fit và chuyển đổi câu thành vector TF-IDF
tfidf_sentence_embeddings = tfidf_vectorizer.fit_transform(sentence_tokens)

# Phản hồi với người
def response(user_response):
    sentence_tokens.append(user_response)
    user_tfidf = tfidf_vectorizer.transform([user_response])
    similarities = cosine_similarity(user_tfidf, tfidf_sentence_embeddings)
    
    idx = similarities.argsort()[0][-1]  
    max_similarity = similarities[0][idx]  
    
    if max_similarity == 0:
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
    if user_response.startswith("http"):
        text = get_text_from_url(user_response)
        if text:
            return jsonify({"response": text})
        else:
            return jsonify({"response": "Data error"})
    elif user_response not in ['bye', 'thanks', 'thank you']:
        if greet(user_response):
            return jsonify({"response": greet(user_response)})
        else:
            bot_response = response(user_response)
            return jsonify({"response": bot_response})
    else:
        return jsonify({"response": "Goodbye!"})

# Chạy Flask
if __name__ == "__main__":
    app.run(debug=True)
