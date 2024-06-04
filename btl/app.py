from flask import Flask, render_template, request, jsonify
import nltk
import string
import random
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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

# Tải mô hình BERT đã được huấn luyện sẵn
model = SentenceTransformer('bert-base-nli-mean-tokens')
# Mã hóa các câu thành vector
sentence_embeddings = model.encode(sentence_tokens)

# Hàm phản hồi dựa trên văn bản người dùng nhập vào
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

# Route cho trang chủ
@app.route("/")
def index():
    return render_template("index.html")

# Route để nhận phản hồi từ người dùng
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

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True)
