import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import pickle

# Read the dataset
bbc_text = pd.read_csv("bbc-text.txt")
bbc_text = bbc_text.rename(columns={'text': 'News_Headline'}, inplace=False)
bbc_text.category = bbc_text.category.map({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})

# Split the data
X = bbc_text.News_Headline
y = bbc_text.category
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1)

# Vectorize the text
vector = CountVectorizer(stop_words='english', lowercase=False)
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_test_transformed = vector.transform(X_test)

# Train the model
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)

# Save the model
saved_model=pickle.dumps(naivebayes)
s=pickle.loads(saved_model)
# Define the Streamlit app
st.header('NEWS CATEGORIZER')
input_text = st.text_area("Please enter the text")
input_text_trans = vector.transform([input_text]).toarray()
if st.button("Check"):
    prediction=(str(list(s.predict(input_text_trans))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))
    st.write(prediction)
    