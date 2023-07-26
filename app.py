import streamlit as st   #Streamlit is a free and open-source framework to rapidly build and share beautiful machine learning and data science web apps.
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

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


model = pickle.load(open(r'C:\Users\Pooja Joshi\Desktop\SVM\SpamHam\model.pkl', 'rb'))
tfidf = pickle.load(open(r'C:\Users\Pooja Joshi\Desktop\SVM\SpamHam\vectorizer.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
   # print("Transformed SMS:", transformed_sms)  # Add this line to print the preprocessed text
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    #print("Vectorized Input:", vector_input)  # Add this line to print the vectorized input
    # 3. predict
    result = model.predict(vector_input)[0]
   # print("Prediction Result:", result)  # Add this line to print the prediction result
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")