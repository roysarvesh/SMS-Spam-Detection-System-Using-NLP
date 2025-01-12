import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class SMSSpamDetector:
    def __init__(self, vectorizer_path="vectorizer.pkl", model_path="model.pkl"):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = self._load_model(vectorizer_path)
        self.model = self._load_model(model_path)

    def _load_model(self, path):
        try:
            with open(path, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            st.error(f"Model file {path} not found!")
            return None

    def preprocess_text(self, text):
        # More efficient text preprocessing
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        
        cleaned_tokens = [
            self.ps.stem(token) 
            for token in tokens 
            if token.isalnum() and 
               token not in self.stop_words and 
               token not in string.punctuation
        ]
        
        return " ".join(cleaned_tokens)

    def predict_spam(self, input_sms):
        if not input_sms:
            st.warning("Please enter an SMS")
            return None

        transformed_sms = self.preprocess_text(input_sms)
        vector_input = self.vectorizer.transform([transformed_sms])
        result = self.model.predict(vector_input)[0]
        
        return "Spam" if result == 1 else "Not Spam"

def main():
    st.set_page_config(page_title="SMS Spam Detector", page_icon="🚨")
    st.title("SMS Spam Detection Model")
    st.markdown("*Powered by Machine Learning*")

    detector = SMSSpamDetector()

    input_sms = st.text_area("Enter SMS Text", height=150)

    if st.button('Predict Spam'):
        prediction = detector.predict_spam(input_sms)
        if prediction:
            st.header(prediction)

if __name__ == "__main__":
    main()
