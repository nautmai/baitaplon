import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Đọc nội dung từ tệp văn bản
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Xử lý văn bản
def process_text(text):
    # Tách thành từ
    words = nltk.word_tokenize(text)

    # Loại bỏ stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

    return ' '.join(stemmed_words)

# Lưu kết quả vào tệp mới
def save_processed_text(processed_text, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(processed_text)

if __name__ == '__main__':
    input_file_path = 'C:\Users\Admin\OneDrive\Documents\bai_tap\xu_ly_ngon_ngu_tu_nhien\baikiemtradiemB\New_folder\rnKrAvN.txt'  # Đường dẫn tới tệp văn bản đầu vào
    output_file_path = 'test.txt'  # Đường dẫn tới tệp văn bản sau khi xử lý

    input_text = read_text_file(input_file_path)
    processed_text = process_text(input_text)
    save_processed_text(processed_text, output_file_path)

    print(f"Đã xử lý và lưu kết quả vào tệp {output_file_path}")