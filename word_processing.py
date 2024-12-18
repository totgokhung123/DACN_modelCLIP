import os
from pyvi import ViTokenizer, ViUtils
from googletrans import Translator
from difflib import SequenceMatcher
import underthesea
from langdetect import detect

class VietnameseTextProcessor:
    def __init__(self, stopwords_path='D:\\code\\projects\\git\\DACN_modelCLIP-3\\vietnamese-stopwords.txt'):
        # Load danh sách stopwords
        if os.path.exists(stopwords_path):
            with open(stopwords_path, 'rb') as f:
                lines = f.readlines()
            self.stop_words = [line.decode('utf-8').strip() for line in lines]
        else:
            self.stop_words = []

        # Khởi tạo trình dịch Google
        self.translator = Translator()

    def detect_language(self, text):
        # Phát hiện ngôn ngữ của văn bản
        return detect(text)

    def translate_to_english(self, text):
        # Dịch văn bản sang tiếng Anh nếu nó là tiếng Việt
        if self.detect_language(text) == 'vi':
            translated = self.translator.translate(text, src='vi', dest='en')
            return translated.text
        return text

    def lowercasing(self, text):
        # Chuyển văn bản về chữ thường
        return text.lower()

    def uppercasing(self, text):
        # Chuyển văn bản về chữ hoa
        return text.upper()

    def remove_stopwords(self, text):
        # Loại bỏ stopwords khỏi văn bản
        text = ViTokenizer.tokenize(text)
        return " ".join([word for word in text.split() if word not in self.stop_words])

    def remove_accents(self, text):
        # Loại bỏ dấu câu trong văn bản
        return ViUtils.remove_accents(text)

    def add_accents(self, text):
        # Thêm dấu câu vào văn bản (không thực sự cần thiết trừ khi có trường hợp cụ thể)
        return ViUtils.add_accents(text)

    def sentence_segment(self, text):
        # Tách câu
        return underthesea.sent_tokenize(text)

    def text_normalization(self, text):
        # Chuẩn hóa văn bản
        return underthesea.text_normalize(text)

    def text_classification(self, text):
        return underthesea.classify(text)

    def sentiment_analysis(self, text):
        return underthesea.sentiment(text)

    def preprocess_and_translate(self, text):
        text = self.lowercasing(text)
        text = self.remove_stopwords(text)
        text = self.text_normalization(text)
        text = self.translate_to_english(text)
        return text

if __name__ == "__main__":
    processor = VietnameseTextProcessor()
    query_text ="con gà đã làm sạch đựng trong chiếc túi nilon màu xanh biển" 
    processed_text = processor.preprocess_and_translate(query_text)
    print("Văn bản đã xử lý và dịch:", processed_text)
