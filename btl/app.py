from flask import Flask, render_template, request, jsonify
import nltk
import string
import random
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Tải các gói ngôn ngữ cần thiết của NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Hàm loại bỏ các chú thích từ văn bản
def remove_text_comments(text):
    # Sử dụng biểu thức chính quy để loại bỏ các chú thích
    text = re.sub(r'\[[^\]]*\]', '', text) # Loại bỏ các chú thích nằm trong dấu ngoặc vuông
    return text

# Hàm lấy văn bản từ URL
def get_text_from_url(url):
    try:
        # Lấy nội dung của trang web
        response = requests.get(url)
        # Kiểm tra xem request có thành công hay không
        if response.status_code == 200:
            # Sử dụng BeautifulSoup để phân tích cú pháp HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            # Trích xuất văn bản từ các thẻ <p>
            paragraphs = soup.find_all('p')
            # Kết hợp văn bản từ các đoạn văn
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

# Thay URL bằng địa chỉ của trang web bạn muốn lấy văn bản
url = "https://en.wikipedia.org/wiki/ChatGPT"
text = get_text_from_url(url)
if text:
    save_text_to_file(text, 'data.txt')

# Đọc dữ liệu từ file và chuyển thành chữ thường
with open('data.txt', 'r', errors='ignore') as f:
    raw_doc = f.read().lower()

raw_doc = remove_text_comments(raw_doc)
raw_doc = raw_doc.lower()

# Tách dữ liệu thành các câu và từ
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

# Khởi tạo lemmatizer và tập hợp các từ dừng
lemmer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Hàm lemmatize các token và loại bỏ từ dừng và dấu câu
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in stop_words and token not in string.punctuation]

# Hàm chuẩn hóa văn bản
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower()))

# Các lời chào và phản hồi chào
greet_inputs = ('hello', 'hi', 'whassup', 'how are you?')
greet_responses = ('hi', 'Hey', 'hey there!', 'There there!!')

# Hàm trả lời lời chào
def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# Khởi tạo TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=LemNormalize)

# Fit và chuyển đổi câu thành vector TF-IDF
tfidf_sentence_embeddings = tfidf_vectorizer.fit_transform(sentence_tokens)

# Hàm phản hồi dựa trên văn bản người dùng nhập vào sử dụng TF-IDF
def response(user_response):
    sentence_tokens.append(user_response)
    user_tfidf = tfidf_vectorizer.transform([user_response])
    similarities = cosine_similarity(user_tfidf, tfidf_sentence_embeddings)
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

# Route cho trang chủ
@app.route("/")
def index():
    return render_template("index.html")

# Route để nhận phản hồi từ người dùng
@app.route("/get_response", methods=["POST"])
def get_response():
    user_response = request.form["msg"].lower()
    if user_response.startswith("http"):
        text = get_text_from_url(user_response)
        if text:
            return jsonify({"response": text})
        else:
            return jsonify({"response": "Sorry, unable to retrieve text from the provided URL."})
    elif user_response not in ['bye', 'thanks', 'thank you']:
        if greet(user_response):
            return jsonify({"response": greet(user_response)})
        else:
            bot_response = response(user_response)
            return jsonify({"response": bot_response})
    else:
        return jsonify({"response": "Goodbye!"})

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True)
