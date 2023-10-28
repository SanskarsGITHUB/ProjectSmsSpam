import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from settings import *

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

def read_csv():
    full_path = os.path.join(os.getcwd(), static, file_name)  # Replace 'file_name.csv' with your actual file name
    spam_df = pd.read_csv(full_path, encoding='latin-1')
    return spam_df

def data_modification(spam_df):
    spam_df = spam_df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    spam_df = spam_df.rename(columns={'v1': 'target', 'v2': 'text'})
    encode = LabelEncoder()
    spam_df['target'] = encode.fit_transform(spam_df['target'])
    spam_df = spam_df.drop_duplicates(keep='first')
    return spam_df

def webapp_function():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    st.title("Email/SMS Spam Classifier")

    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

    st.markdown("Created by Sanskar")

if __name__ == '__main__':
    movies_data = read_csv()
    movies_data = data_modification(movies_data)
    movies_data['text'] = movies_data['text'].apply(transform_text)

    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    webapp_function()
